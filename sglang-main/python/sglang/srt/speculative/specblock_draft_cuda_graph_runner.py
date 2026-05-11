"""SpecBlock-Shift Draft CUDA Graph Runner.

Captures and replays the GPU-only region of :func:`build_tree_gpu` per
``(bs, cross_bucket, pend_bucket)`` shape combination.  Sched/IPC + per-
kernel launch overhead drops from ~55ms to ~5ms (matching EAGLE3's V2 +
graph-captured draft path).

## Capture region

The captured chain is :func:`build_tree_gpu` minus :func:`finalize_tree_gpu`:

  block-1 GPU precompute  →  block-1 mega kernel (Triton)
  → block-2 batch prep (cross_loc / ttt_kv expand)
  → forward_with_cache_graph (draft model, position_id Tensor)
  → BFS GPU precompute (rank walk + topk + lse)
  → BFS sizing kernel
  → BFS scatter kernel v2 (sync-free, valid_mask gated)
  → tree_buf populated, sizes_gpu has counts

Sync-free since commits ``78a4751`` + ``df793ec`` + ``a8dc387``.

Finalize (prune + topology + retrieve_index sort) stays eager — it has
data-dependent output shape (n_nodes_final.item()) and a small CPU-side
sort.  Eager finalize on captured tree_buf is ~1ms / iter, dominated by
the Triton ancestor-mask kernel.

## Bucket grid

* ``bs``: from server_args.cuda_graph_bs (typically [1, 2, 3, 4]); only
  bs=1 is captured today (B>1 path needs build_tree_batched_gpu's
  cross_kv padding integrated — TODO).
* ``cross_bucket``: smallest of (128, 256, 512, 1024, 2048) >= max
  per-req cross_count.  Caller resolves via ``can_run``.
* ``pend_bucket``: smallest of (16, 32, 64) >= N_pend.  N_pend depends
  on rank predictions, so bucket-rounded at query time.

Total bs=1 graphs = 5 × 3 = 15.  Lazy capture: a (cross, pend) miss
triggers ``capture_one`` then replays.  No eviction yet (15 graphs at
~0.3 GB each ~ 5 GB, well within mem-fraction headroom).

## Production policy

CUDA graph is ON by default for SpecBlock-Shift.  ``SPECBLOCK_DRAFT_CUDA_GRAPH=0``
disables it for debugging only.  Capture failure raises (no eager fallback).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.speculative.specblock_worker import SpecBlockWorker

logger = logging.getLogger(__name__)


# Bucket constants (matched to existing _GRAPH_CROSS_BUCKETS in tree_builder).
_CROSS_BUCKETS: Tuple[int, ...] = (128, 256, 512, 1024, 2048)
_PEND_BUCKETS: Tuple[int, ...] = (16, 32, 64)


class SpecBlockDraftCudaGraphRunner:
    """CUDA graph runner for SpecBlockWorker's build_tree GPU chain.

    Lifecycle::

        runner = SpecBlockDraftCudaGraphRunner(worker)
        # later, in worker.draft:
        cb, pb = runner.resolve_buckets(spec_info, bs)
        if runner.can_run(bs, cb, pb):
            tree = runner.replay(spec_info, bs, cb, pb)  # B=1 path
        else:
            tree = build_tree_eager(...)

    Capture is lazy — a (bs, cb, pb) miss triggers ``capture_one`` then
    replays.  No eviction (graph count < 20 typically).
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

        # cuda_graph_bs from server args; today only bs=1 is captured.
        sa = worker.server_args
        self.capture_bs: List[int] = list(getattr(sa, "cuda_graph_bs", [1]))
        if 1 not in self.capture_bs:
            self.capture_bs.insert(0, 1)
        self.max_bs = max(self.capture_bs)

        self.cross_buckets = _CROSS_BUCKETS
        self.pend_buckets = _PEND_BUCKETS
        self.max_cross = self.cross_buckets[-1]
        self.max_pend = self.pend_buckets[-1]
        max_topk = max(self.beam_width, self.K * (self.K + 1))
        max_block1_nodes = self.K + self.K * (max_topk - 1) + 1
        max_block2_nodes = max_block1_nodes * self.K * max_topk
        self.max_nodes = max(
            self.total_tokens + 200,
            max_block1_nodes + max_block2_nodes + 100,
        )

        # Per-(bs, cb, pb) graph + buffer cache.
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

        # Production default ON.  Flag is debug-only.
        self.enabled = (
            os.environ.get("SPECBLOCK_DRAFT_CUDA_GRAPH", "1") == "1"
        )
        if self.enabled:
            logger.info(
                "[SpecBlockDraftCudaGraphRunner] enabled "
                "(capture_bs=%s, cross_buckets=%s, pend_buckets=%s)",
                self.capture_bs, self.cross_buckets, self.pend_buckets,
            )

    # ============================================================
    #  Bucket resolution
    # ============================================================

    def _resolve_cross(self, max_cross_count: int) -> Optional[int]:
        for cb in self.cross_buckets:
            if cb >= max_cross_count:
                return cb
        return None  # exceeds largest bucket

    def _resolve_pend(self, n_pend_estimate: int) -> Optional[int]:
        for pb in self.pend_buckets:
            if pb >= n_pend_estimate:
                return pb
        return None

    def resolve_buckets(
        self, spec_info, bs: int,
    ) -> Optional[Tuple[int, int, int]]:
        """Return (bs, cross_bucket, pend_bucket) for the current spec_info,
        or None if no bucket fits.

        ``cross_bucket`` is the smallest >= max(cross_count[i] for i in B).
        ``pend_bucket`` is the smallest >= K * (max_topk).  Static estimate
        since N_pend is data-dependent but bounded by block-1's max
        expansion.
        """
        if bs not in self.capture_bs:
            return None
        max_cross = max(spec_info.cross_count) if spec_info.cross_count else 0
        cb = self._resolve_cross(max_cross)
        if cb is None:
            return None
        # Static N_pend upper bound.  N_pend <= K + K*(max_factor) per
        # block-1's mega kernel layout; round up to nearest pend bucket.
        n_pend_max = self.K + self.K * (max(self.cross_buckets) // self.K + 1)
        pb = self._resolve_pend(min(self.max_pend, n_pend_max))
        if pb is None:
            return None
        return (bs, cb, pb)

    def can_run(self, bs: int, cb: int, pb: int) -> bool:
        """Return True iff a graph already exists AND its captured
        pool_version matches the pool's current version.

        On version mismatch (pool grew → buffer pointers replaced),
        ALL captured graphs are dropped and the call returns False so
        the caller falls back to eager build_tree_batched and a fresh
        capture happens on the next iter.
        """
        if not self.enabled:
            return False
        # Today: only bs=1 capture is wired.
        if bs != 1:
            return False
        key = (bs, cb, pb)
        if key not in self.graphs:
            return False
        cur_version = self.worker.spec_kv_pool.pool_version
        if self.pool_version_per_graph.get(key, -1) != cur_version:
            # Pool grew since capture — drop all stale graphs.
            self._invalidate_all()
            return False
        return True

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
        self, bs: int, cb: int, pb: int, spec_info=None,
    ) -> None:
        """Capture a graph for (bs, cb, pb).

        Pre-allocates static input buffers at this shape, runs warmup
        iterations to initialize build_tree_gpu's lazy scratch, then
        captures the GPU-only chain into a CUDA graph.

        ``spec_info`` (optional) — if provided, its tensors are copied
        into static buffers before warmup.  All-zero static buffers
        sometimes crash the paged-attention triton kernel on degenerate
        inputs (cross_loc all=0, ttt_mask all=False); using a real
        first-iter spec_info avoids that.  Caller passes the same
        spec_info that triggered the capture.

        Failure raises ``RuntimeError`` (no fallback to eager).
        """
        if not self.enabled:
            raise RuntimeError(
                "[capture_one] runner is disabled (SPECBLOCK_DRAFT_CUDA_GRAPH=0)"
            )
        if bs != 1:
            raise NotImplementedError(
                f"[capture_one] only bs=1 is captured today; got bs={bs}.  "
                f"B>1 needs build_tree_batched_gpu cross_kv padding integration."
            )

        key = (bs, cb, pb)
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
            "b0_logits":      torch.zeros((bs, K, V_d), dtype=dtype, device=device),
            "b0_rank_logits": torch.zeros((bs, K, rc), dtype=dtype, device=device),
            "b0_hidden":      torch.zeros((bs, K, H), dtype=dtype, device=device),
            "b0_input_id":    torch.zeros((bs, 1), dtype=torch.int64, device=device),
            "cross_loc":      torch.zeros((cb,), dtype=torch.int64, device=device),
            "cross_count":    torch.tensor([cb], dtype=torch.int32, device=device),
            "position_id":    torch.zeros((1,), dtype=torch.int64, device=device),
            "ttt_k": [
                torch.zeros((1, n_kv, K, hd), dtype=dtype, device=device)
                for _ in range(self.num_layers)
            ],
            "ttt_v": [
                torch.zeros((1, n_kv, K, hd), dtype=dtype, device=device)
                for _ in range(self.num_layers)
            ],
        }
        self.input_buffers[key] = in_buf

        # ---- Worker scratch dict (build_tree_gpu lazy-init) ----
        # Pre-init by calling build_tree_gpu once on these static buffers
        # (warmup), so all internal allocations land before capture.
        scratch: dict = {}
        self.scratch_per_graph[key] = scratch

        cross_kv_cache = self._build_static_cross_cache(in_buf, cb)

        # ---- Seed static buffers with real inputs from spec_info ----
        # (avoids triton attn OOB on degenerate all-zero buffers).
        if spec_info is not None:
            self._copy_spec_into_buffers(in_buf, spec_info, cb)

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
            "bs=%d cross_bucket=%d pend_bucket=%d "
            "(pool_version=%d)", bs, cb, pb,
            self.pool_version_per_graph[key],
        )

    def _build_static_cross_cache(
        self, in_buf: dict, cb: int,
    ) -> List[List]:
        """Build the [paged_meta] cross_kv_cache backed by static input buffers."""
        spec_kv_pool = self.worker.spec_kv_pool
        return [
            [spec_kv_pool, L, cb, in_buf["cross_loc"], None]
            for L in range(self.num_layers)
        ]

    def _copy_spec_into_buffers(
        self, in_buf: dict, spec_info, cb: int,
    ) -> None:
        """Copy a real spec_info into static buffers (for warmup seeding
        + first replay).  Pads cross_loc to ``cb`` with sentinel slot 0.
        """
        in_buf["b0_logits"].copy_(spec_info.b0_logits)
        in_buf["b0_rank_logits"].copy_(spec_info.b0_rank_logits)
        in_buf["b0_hidden"].copy_(spec_info.b0_hidden)
        in_buf["b0_input_id"].copy_(spec_info.b0_input_id)

        cross_loc_actual = spec_info.cross_loc[0]
        c = int(spec_info.cross_count[0])
        in_buf["cross_loc"].zero_()
        if c > 0:
            in_buf["cross_loc"][:c].copy_(cross_loc_actual)
        in_buf["cross_count"].fill_(c)
        in_buf["position_id"].fill_(int(spec_info.cross_position[0]))

        for L in range(self.num_layers):
            in_buf["ttt_k"][L].copy_(spec_info.ttt_k[0][L])
            in_buf["ttt_v"][L].copy_(spec_info.ttt_v[0][L])

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

        Output: ``scratch['tree_buf']`` populated; ``scratch['sizes4']``
        has [n_nodes_b1, n_active, total_alts1, N_pend].

        This is the "capturable region" — sync-free since the BFS path
        switched to ``_bfs_scatter_kernel_v2`` + tree_start_ptr / alt_offset_ptr.
        Finalize (prune + topology + retrieve sort) is run eager outside
        the graph by ``replay``.
        """
        from sglang.srt.speculative.specblock_tree_kernels import (
            build_tree_gpu_capturable_region,
        )

        build_tree_gpu_capturable_region(
            draft_model=self.worker.model_runner.model.inner,
            block1_logits=in_buf["b0_logits"],
            block1_rank_logits=in_buf["b0_rank_logits"],
            block1_hidden=in_buf["b0_hidden"],
            block1_ttt_kv=list(zip(in_buf["ttt_k"], in_buf["ttt_v"])),
            cross_kv_cache=cross_kv_cache,
            position_id_t=in_buf["position_id"],
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

    # ============================================================
    #  Replay
    # ============================================================

    def replay(self, spec_info, bs: int, cb: int, pb: int) -> dict:
        """Replay the graph for (bs, cb, pb), then run eager finalize.

        spec_info is a SpecBlockDraftInput at iter k (B=1 today).
        Caller must ensure ``can_run(bs, cb, pb)`` returned True.

        Returns the standard tree dict (matches build_tree_gpu's shape).
        """
        from sglang.srt.speculative.specblock_tree_kernels import (
            finalize_tree_gpu,
        )

        key = (bs, cb, pb)
        g = self.graphs[key]
        in_buf = self.input_buffers[key]
        scratch = self.scratch_per_graph[key]

        # ---- Copy current iter's inputs into static buffers ----
        self._copy_spec_into_buffers(in_buf, spec_info, cb)

        # ---- Replay graph ----
        g.replay()

        # ---- Eager finalize (data-dependent shape; outside graph) ----
        sizes4 = scratch['sizes4']
        sizes1 = scratch['sizes1']
        tree_buf = scratch['tree_buf']

        # n_nodes_final = sizes4[0] (n_nodes_b1) + N_pend * K + total_alts_2
        n_nodes_b1 = int(sizes4[0].item())
        N_pend = int(sizes4[3].item())
        total_alts_2 = int(sizes1[0].item()) if N_pend > 0 else 0
        n_nodes_final = n_nodes_b1 + N_pend * self.K + total_alts_2

        sample_token = spec_info.b0_input_id[0, -1:]
        return finalize_tree_gpu(
            tree_buf, n_nodes_final, sample_token,
            self.total_tokens, self.K, self.max_blocks,
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
