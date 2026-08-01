"""SpecBlock-Shift Draft CUDA Graph Runner.

Captures and replays the GPU-only tree-building region per
``(batch_capacity, persistent_cross_bucket, pend_bucket)`` shape combination.
Sched/IPC + per-kernel launch overhead drops from ~55ms to ~5ms (matching
EAGLE3's V2 + graph-captured draft path).

## Capture region

The captured chain is request-major and fixed-capacity:

  batched block-1 GPU precompute → batched block-1 mega kernel (Triton)
  → fixed ``[Bcap, P]`` block-2 prep (cross / TTT / positions)
  → one flattened block-2 model forward
  → BFS GPU precompute
  → valid-leaf-gated fixed batched BFS scatter
  → per-request tree buffers and GPU size tensors

Replay stays sync-free.  Fixed-width finalization runs after replay without
reading data-dependent tree sizes back to the host.

## Bucket grid

* ``batch_capacity``: smallest configured CUDA Graph batch capacity greater
  than or equal to the active request count.  Active shrink reuses the same
  graph and only exposes the active request-major output rows.
* ``persistent_cross_bucket``: smallest configured bucket through 16384 >=
  ``max(cross_count[i] - K)``.  The current block-0 TTT slots are not part of
  persistent cross history.
* ``pend_bucket``: smallest of (16, 32, 64) covering block-1's structural
  maximum pending-leaf count.  It directly controls captured pending shapes.

Capture is lazy.  A capacity/cross/pend miss triggers ``capture_one`` then
replays; unsupported shapes fail explicitly.

## Production policy

This is the sole SpecBlock draft graph path.  It is always enabled; capture
failure and unsupported shapes fail explicitly instead of falling back to a
second eager or legacy graph implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.speculative.specblock_worker import SpecBlockWorker

logger = logging.getLogger(__name__)


# Bucket constants for the sole draft graph runner.
_CROSS_BUCKETS: Tuple[int, ...] = (
    128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
)
_PEND_BUCKETS: Tuple[int, ...] = (16, 32, 64)


class SpecBlockDraftCudaGraphRunner:
    """CUDA graph runner for SpecBlockWorker's build_tree GPU chain.

    Lifecycle::

        runner = SpecBlockDraftCudaGraphRunner(worker)
        # later, in worker.draft:
        bcap, cb, pb = runner.resolve_buckets(spec_info, active_bs)
        if not runner.can_run(bcap, cb, pb):
            runner.capture_one(bcap, cb, pb, spec_info=spec_info)
        trees = runner.replay(spec_info, active_bs, bcap, cb, pb)

    Capture is lazy — a ``(batch_capacity, cross_bucket, pend_bucket)`` miss
    triggers ``capture_one`` then replays.  Unsupported shapes fail explicitly.
    """

    def __init__(self, worker: "SpecBlockWorker"):
        self.worker = worker
        self.device = worker.device
        self.K = worker.K
        self.num_layers = worker.num_layers
        self.beam_width = worker.beam_width
        self.total_tokens = worker.total_tokens
        self.rank_classes = worker.rank_classes
        self.max_blocks = worker.max_blocks

        sa = worker.server_args
        configured_bs = [
            int(bs) for bs in getattr(sa, "cuda_graph_bs", [1]) if int(bs) > 0
        ]
        configured_max_bs = max(configured_bs, default=1)
        self.capture_bs = []
        capacity = 1
        while capacity < configured_max_bs:
            self.capture_bs.append(capacity)
            capacity *= 2
        self.capture_bs.append(configured_max_bs)
        self.max_bs = configured_max_bs

        self.cross_buckets = _CROSS_BUCKETS
        self.pend_buckets = _PEND_BUCKETS
        self.max_cross = self.cross_buckets[-1]
        self.max_pend = self.pend_buckets[-1]
        self.max_topk = max(self.beam_width, max(worker.rank_slot_topk))
        # Slot 0 uses ``beam_width``; the remaining K-1 slots use the
        # rank-conditioned table.  Bounding every slot by max_topk forced the
        # slim_r2 production tree (actual maximum 29) into a 64-row pending
        # bucket.  Share the exact bound with the captured tree kernel so graph
        # resolution and capture validation cannot drift apart.
        from sglang.srt.speculative.specblock_tree_kernels import (
            max_block1_structural_nodes,
        )

        self.max_block1_nodes = max_block1_structural_nodes(
            self.K, self.beam_width, worker.rank_slot_topk,
        )
        self.pend_bucket = self._resolve_pend(self.max_block1_nodes)
        if self.pend_bucket is None:
            raise RuntimeError(
                "SpecBlock block-1 pending capacity exceeds all CUDA Graph "
                f"buckets: required={self.max_block1_nodes}, "
                f"buckets={self.pend_buckets}."
            )

        # Expanding every structural pending slot wastes most block-2 work:
        # slim_r2 stores at most 29 rows but only needs 10 leaves to produce a
        # 90-node tree even under its minimum branching factor.  Keep quality
        # headroom by expanding the top 16 cumulative-probability leaves while
        # retaining the full 32-row block-1 storage bucket.
        min_rank_topk = min(worker.rank_slot_topk)
        min_block1_nodes = self.K + max(self.beam_width - 1, 0)
        min_nodes_per_leaf = self.K * max(min_rank_topk, 1)
        required_expand = max(
            1,
            (
                max(self.total_tokens - min_block1_nodes, 0)
                + min_nodes_per_leaf
                - 1
            )
            // min_nodes_per_leaf,
        )
        if min_rank_topk > 1:
            self.expand_bucket = min(
                self.pend_bucket, max(required_expand, 16),
            )
        else:
            self.expand_bucket = self.pend_bucket
        self.max_nodes = max(
            self.total_tokens + 200,
            self.max_block1_nodes
            + self.expand_bucket * self.K * self.max_topk
            + 100,
        )

        # Per-(batch_capacity, cross_bucket, pend_bucket) graph + buffers.
        # Each entry stores: graph, scratch (build_tree_gpu's lazy-init
        # tensors — tree_buf / pend_buf / sizes_buf), input buffers,
        # tree_buf reference (output is read post-replay).
        self.graphs: Dict[Tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.input_buffers: Dict[Tuple[int, int, int], Dict[str, torch.Tensor]] = {}
        self.scratch_per_graph: Dict[Tuple[int, int, int], dict] = {}
        self.sizes_gpu_per_graph: Dict[Tuple[int, int, int], torch.Tensor] = {}
        # Captured pool_version per graph entry.  When spec_kv_pool grows,
        # k_buffer/v_buffer tensor pointers are replaced; any captured
        # graph holding stale pointers will hit illegal memory access on
        # replay.  Bumped pool.pool_version signals invalidation.
        self.pool_version_per_graph: Dict[Tuple[int, int, int], int] = {}

        logger.info(
            "[SpecBlockDraftCudaGraphRunner] enabled "
            "(capture_bs=%s, cross_buckets=%s, pend_buckets=%s, "
            "expand_bucket=%d)",
            self.capture_bs, self.cross_buckets, self.pend_buckets,
            self.expand_bucket,
        )

    # ============================================================
    #  Bucket resolution
    # ============================================================

    def _resolve_batch_capacity(self, active_bs: int) -> Optional[int]:
        for capacity in self.capture_bs:
            if capacity >= active_bs:
                return capacity
        return None

    def _resolve_cross(self, max_persistent_count: int) -> Optional[int]:
        for cb in self.cross_buckets:
            if cb >= max_persistent_count:
                return cb
        return None

    def _resolve_pend(self, required_pend: int) -> Optional[int]:
        for pb in self.pend_buckets:
            if pb >= required_pend:
                return pb
        return None

    def resolve_buckets(
        self, spec_info, active_bs: int,
    ) -> Optional[Tuple[int, int, int]]:
        """Resolve the reusable graph capacity for the active request batch."""
        bcap = self._resolve_batch_capacity(active_bs)
        if bcap is None:
            return None
        counts = list(spec_info.cross_count[:active_bs])
        if len(counts) != active_bs:
            return None
        max_persistent = max(
            (max(int(count) - self.K, 0) for count in counts),
            default=0,
        )
        cb = self._resolve_cross(max_persistent)
        if cb is None or self.pend_bucket is None:
            return None
        return (bcap, cb, self.pend_bucket)

    def can_run(self, bcap: int, cb: int, pb: int) -> bool:
        """Return whether an exact, current-pool graph entry exists.

        Pool growth replaces the K/V tensor storage.  Invalidate every stale
        entry before checking the requested key, including when the requested
        key itself is a cache miss.  This prevents a stale graph from surviving
        until a later lookup and clearing graphs captured against the new pool.
        """
        cur_version = self.worker.spec_kv_pool.pool_version
        if self.graphs and (
            set(self.pool_version_per_graph) != set(self.graphs)
            or any(
                version != cur_version
                for version in self.pool_version_per_graph.values()
            )
        ):
            self._invalidate_all()
            return False
        return (bcap, cb, pb) in self.graphs

    def _invalidate_all(self) -> None:
        """Drop every captured graph + its buffers (buffers reference
        the OLD pool tensors, so they're stale and unsafe).  Caller
        triggers a fresh re-capture on the next iter via capture_one.
        """
        if self.graphs:
            logger.info(
                "[SpecBlockDraftCudaGraphRunner] pool grew; dropping "
                "%d captured graphs (will re-capture)", len(self.graphs),
            )
        self.graphs.clear()
        self.input_buffers.clear()
        self.scratch_per_graph.clear()
        self.sizes_gpu_per_graph.clear()
        self.pool_version_per_graph.clear()

    # ============================================================
    #  Capture
    # ============================================================

    def capture_one(
        self, bcap: int, cb: int, pb: int, spec_info=None,
    ) -> None:
        """Capture a graph for one batch/cross/pending capacity tuple.

        Pre-allocates request-major static inputs, runs warmup iterations to
        initialize all scratch, then captures the GPU-only chain.  Real active
        inputs seed warmup; every inactive capacity row is reset to a complete
        dummy request before both warmup and replay.

        Failure raises ``RuntimeError`` (no fallback to eager).
        """
        if bcap not in self.capture_bs or pb != self.pend_bucket:
            raise RuntimeError(
                "Unsupported SpecBlock draft graph capacity: "
                f"batch_capacity={bcap}, cross_bucket={cb}, "
                f"pend_bucket={pb}."
            )

        key = (bcap, cb, pb)
        if key in self.graphs:
            return  # already captured

        worker = self.worker
        device = self.device
        K = self.K
        draft_model = worker.model_runner.model.inner
        H = draft_model.config.hidden_size
        V_d = draft_model.draft_vocab_size
        rc = self.rank_classes
        dtype = next(draft_model.parameters()).dtype

        # ---- Allocate static input buffers ----
        first_layer = draft_model.layers[0].self_attn
        n_kv = first_layer.n_kv_heads
        hd = first_layer.head_dim
        in_buf = {
            "b0_logits": torch.zeros(
                (bcap, K, V_d), dtype=dtype, device=device
            ),
            "b0_rank_logits": torch.zeros(
                (bcap, K, rc), dtype=dtype, device=device
            ),
            "b0_hidden": torch.zeros(
                (bcap, K, H), dtype=dtype, device=device
            ),
            "b0_input_id": torch.zeros(
                (bcap, 1), dtype=torch.int64, device=device
            ),
            "active_mask": torch.zeros(
                (bcap,), dtype=torch.bool, device=device
            ),
            # Block-2 attends only persistent history; the final K live
            # cross slots are block-0's current TTT entries.
            "cross_loc": torch.zeros(
                (bcap, cb), dtype=torch.int64, device=device
            ),
            "cross_count": torch.zeros(
                (bcap,), dtype=torch.int32, device=device
            ),
            "position_id": torch.zeros(
                (bcap,), dtype=torch.int64, device=device
            ),
            "seq_lens": torch.zeros(
                (bcap,), dtype=torch.int64, device=device
            ),
            "ttt_k": [
                torch.zeros(
                    (bcap, n_kv, K, hd), dtype=dtype, device=device
                )
                for _ in range(self.num_layers)
            ],
            "ttt_v": [
                torch.zeros(
                    (bcap, n_kv, K, hd), dtype=dtype, device=device
                )
                for _ in range(self.num_layers)
            ],
        }
        # Cache request-major layer views once.  Replay can then update all
        # per-layer TTT state with two multi-tensor copies instead of issuing
        # one eager copy kernel per request and layer.
        in_buf["ttt_k_rows"] = [
            tensor[b : b + 1]
            for b in range(bcap)
            for tensor in in_buf["ttt_k"]
        ]
        in_buf["ttt_v_rows"] = [
            tensor[b : b + 1]
            for b in range(bcap)
            for tensor in in_buf["ttt_v"]
        ]
        self.input_buffers[key] = in_buf

        # ---- Worker scratch dict (build_tree_gpu lazy-init) ----
        # Pre-init by calling build_tree_gpu once on these static buffers
        # (warmup), so all internal allocations land before capture.
        scratch: dict = {}
        self.scratch_per_graph[key] = scratch

        cross_kv_cache = self._build_static_cross_cache(in_buf, cb)

        # ---- Seed static buffers with real active requests ----
        if spec_info is not None:
            active_bs = int(spec_info.b0_logits.shape[0])
            if active_bs > bcap:
                raise RuntimeError(
                    "SpecBlock active batch exceeds graph capacity during capture: "
                    f"active_bs={active_bs}, batch_capacity={bcap}."
                )
            self._copy_spec_into_buffers(
                in_buf, spec_info, active_bs=active_bs, cb=cb
            )

        # ---- Warmup x2 (capture-eligible state) ----
        for _ in range(2):
            self._run_chain(in_buf, cross_kv_cache, scratch)
        torch.cuda.synchronize()

        # ---- Capture ----
        g = torch.cuda.CUDAGraph()
        # Use a private mempool so capture-only allocations don't leak
        # into the global allocator (relevant when capture fails partway).
        # Pool grows on capture; freed on graph delete.
        with torch.cuda.graph(g):
            self._run_chain(in_buf, cross_kv_cache, scratch)

        self.graphs[key] = g
        # sizes4 / sizes1 / tree_buf live in scratch; the post-replay
        # finalize reads from them.  Stash sizes_gpu for convenience.
        self.sizes_gpu_per_graph[key] = scratch['sizes4']
        # Stamp current pool_version so can_run can detect future grows.
        self.pool_version_per_graph[key] = (
            self.worker.spec_kv_pool.pool_version
        )

        logger.info(
            "[SpecBlockDraftCudaGraphRunner] captured graph for "
            "batch_capacity=%d cross_bucket=%d pend_bucket=%d "
            "(pool_version=%d)", bcap, cb, pb,
            self.pool_version_per_graph[key],
        )

    def _build_static_cross_cache(
        self, in_buf: dict, cb: int,
    ) -> List[List]:
        """Build the static-width paged cache used by captured block-2."""
        spec_kv_pool = self.worker.spec_kv_pool
        return [
            [spec_kv_pool, L, cb, in_buf["cross_loc"], None]
            for L in range(self.num_layers)
        ]

    def _copy_spec_into_buffers(
        self, in_buf: dict, spec_info, active_bs: int, cb: int,
    ) -> None:
        """Materialize active request state and reset every inactive row."""
        bcap = int(in_buf["b0_logits"].shape[0])
        if not 0 < active_bs <= bcap:
            raise RuntimeError(
                "Invalid SpecBlock active batch for graph replay: "
                f"active_bs={active_bs}, batch_capacity={bcap}."
            )
        tensor_bs = int(spec_info.b0_logits.shape[0])
        state_lengths = (
            len(spec_info.cross_loc), len(spec_info.cross_count),
            len(spec_info.cross_position), len(spec_info.ttt_k),
            len(spec_info.ttt_v),
        )
        if tensor_bs != active_bs or any(n != active_bs for n in state_lengths):
            raise RuntimeError(
                "SpecBlock request state is not aligned with the active batch: "
                f"active_bs={active_bs}, tensor_bs={tensor_bs}, "
                f"state_lengths={state_lengths}."
            )

        # Active rows are fully overwritten below.  Reset only capacity padding
        # so B=1 does not launch dozens of redundant zero kernels every replay.
        if active_bs < bcap:
            for name in (
                "b0_logits", "b0_rank_logits", "b0_hidden", "b0_input_id",
                "active_mask", "cross_loc", "cross_count", "position_id",
                "seq_lens",
            ):
                in_buf[name][active_bs:].zero_()
            for tensors in (in_buf["ttt_k"], in_buf["ttt_v"]):
                for tensor in tensors:
                    tensor[active_bs:].zero_()

        in_buf["b0_logits"][:active_bs].copy_(spec_info.b0_logits)
        in_buf["b0_rank_logits"][:active_bs].copy_(spec_info.b0_rank_logits)
        in_buf["b0_hidden"][:active_bs].copy_(spec_info.b0_hidden)
        in_buf["b0_input_id"][:active_bs].copy_(spec_info.b0_input_id)
        in_buf["active_mask"][:active_bs].fill_(True)
        if spec_info.new_seq_lens is None:
            raise RuntimeError("SpecBlock draft graph requires current sequence lengths.")
        in_buf["seq_lens"][:active_bs].copy_(
            spec_info.new_seq_lens[:active_bs].to(torch.int64)
        )

        for b in range(active_bs):
            stored_count = int(spec_info.cross_count[b])
            persistent_count = max(stored_count - self.K, 0)
            if persistent_count > cb:
                raise RuntimeError(
                    "SpecBlock cross history exceeds the resolved graph bucket: "
                    f"request={b}, persistent_count={persistent_count}, "
                    f"bucket_width={cb}."
                )
            cross_loc_actual = spec_info.cross_loc[b]
            if persistent_count > int(cross_loc_actual.numel()):
                raise RuntimeError(
                    "SpecBlock cross metadata exceeds allocated request state: "
                    f"request={b}, persistent_count={persistent_count}, "
                    f"cross_loc_size={cross_loc_actual.numel()}."
                )
            if persistent_count > 0:
                in_buf["cross_loc"][b, :persistent_count].copy_(
                    cross_loc_actual[:persistent_count]
                )
            in_buf["cross_count"][b] = persistent_count
            # block-2 expands from the precomputed block-0 one step earlier.
            in_buf["position_id"][b] = int(spec_info.cross_position[b]) - 1

        num_active_layers = active_bs * self.num_layers
        torch._foreach_copy_(
            in_buf["ttt_k_rows"][:num_active_layers],
            [
                spec_info.ttt_k[b][layer]
                for b in range(active_bs)
                for layer in range(self.num_layers)
            ],
        )
        torch._foreach_copy_(
            in_buf["ttt_v_rows"][:num_active_layers],
            [
                spec_info.ttt_v[b][layer]
                for b in range(active_bs)
                for layer in range(self.num_layers)
            ],
        )

    def _run_chain(
        self,
        in_buf: dict,
        cross_kv_cache: List[List],
        scratch: dict,
    ) -> None:
        """Run build_tree_gpu's GPU-only chain on static buffers.

        This is the function called inside ``with torch.cuda.graph(g):``.
        All input tensors come from ``in_buf`` (caller copies actual data
        in via ``.copy_()`` before each replay); all scratch lives in
        ``scratch`` (lazy-init on first call, reused on subsequent).

        Output: raw tree buffers and fixed-width pruned topology are both
        populated in ``scratch``.  Dedup, top-k refill, parent remapping, and
        topology construction stay inside the same replay graph, so the draft
        stage has no eager kernel tail.
        """
        from sglang.srt.speculative.specblock_tree_kernels import (
            build_tree_gpu_batched_capturable_region,
            finalize_tree_fixed_batched,
        )

        build_tree_gpu_batched_capturable_region(
            draft_model=self.worker.model_runner.model.inner,
            block1_logits_b=in_buf["b0_logits"],
            block1_rank_logits_b=in_buf["b0_rank_logits"],
            block1_hidden_b=in_buf["b0_hidden"],
            block1_ttt_kv_b=list(zip(in_buf["ttt_k"], in_buf["ttt_v"])),
            cross_kv_cache_b=cross_kv_cache,
            cross_valid_count_b=in_buf["cross_count"],
            position_id_b=in_buf["position_id"],
            active_mask_b=in_buf["active_mask"],
            pend_bucket=self.pend_bucket,
            expand_bucket=self.expand_bucket,
            K=self.K,
            max_blocks=self.max_blocks,
            beam_width=self.beam_width,
            total_tokens=self.total_tokens,
            rank_classes=self.rank_classes,
            rank_slot_topk=self.worker.rank_slot_topk,
            rank_to_factor=self.worker.rank_to_factor,
            d2t=getattr(
                self.worker.model_runner.model.inner, "d2t", None,
            ),
            scratch=scratch,
        )

        sizes4 = scratch["sizes4"]
        if self.max_blocks <= 1:
            raw_sizes = sizes4[:, 0]
            raw_capacity = self.max_block1_nodes
        else:
            raw_sizes = (
                sizes4[:, 0]
                + sizes4[:, 3] * self.K
                + scratch["sizes1"][:, 0]
            )
            raw_capacity = self.max_nodes
        finalize_tree_fixed_batched(
            tree_buf_b=scratch["tree_buf"],
            raw_sizes_b=raw_sizes,
            sample_tokens_b=in_buf["b0_input_id"],
            budget=self.total_tokens,
            raw_capacity=raw_capacity,
            scratch=scratch,
            active_bs=in_buf["b0_logits"].shape[0],
            dedup_depth=self.K * self.max_blocks,
            seq_lens_b=in_buf["seq_lens"],
        )

    # ============================================================
    #  Replay
    # ============================================================

    def replay(
        self,
        spec_info,
        active_bs: int,
        bcap: int,
        cb: int,
        pb: int,
    ) -> List[dict]:
        """Replay a capacity graph and return active request-major trees."""
        from sglang.srt.speculative.specblock_tree_kernels import (
            pack_fixed_tree_outputs,
        )

        key = (bcap, cb, pb)
        g = self.graphs[key]
        in_buf = self.input_buffers[key]
        scratch = self.scratch_per_graph[key]

        self._copy_spec_into_buffers(
            in_buf, spec_info, active_bs=active_bs, cb=cb
        )
        g.replay()

        return pack_fixed_tree_outputs(
            sb_batch=scratch,
            batch_capacity=bcap,
            active_bs=active_bs,
            tree_tokens=self.total_tokens + 1,
        )

    # ============================================================
    #  POC (kept for testing)
    # ============================================================

    @staticmethod
    def poc_capture_kernel_chain(
        device: torch.device,
        run_chain_fn,
        warmup_iters: int = 2,
    ) -> torch.cuda.CUDAGraph:
        """Capture a callable's CUDA kernel chain into a graph.

        Self-contained POC for verifying graph capture mechanics.  Used
        by unit tests; not called from production paths.
        """
        for _ in range(warmup_iters):
            torch.cuda.synchronize()
            run_chain_fn()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run_chain_fn()
        return g
