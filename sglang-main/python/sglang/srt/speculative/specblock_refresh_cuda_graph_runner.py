"""SpecBlock-Shift refresh CUDA Graph runner.

Captures the GPU-only ``update_cache_and_draft_graph_safe`` chain for fixed
``(batch_capacity, cross_bucket, accept_capacity)`` shapes.  Persistent cross
KV is read directly from :class:`SpecBlockKVPool` by the paged Triton attention
kernel.  Replay returns current K/V tensors; the worker performs the only
persistent mutation afterwards with a valid-prefix paged scatter.

Accepted-path lengths use power-of-two *capacity* buckets.  Inputs remain
padded to the selected capacity, while ``n_per_req`` and the worker's
valid-prefix scatter preserve the logical per-request length.

The runner is unconditional.  Unsupported shapes and capture failures raise
instead of switching to an eager refresh implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.speculative.specblock_worker import (
        SpecBlockWorker,
    )

logger = logging.getLogger(__name__)


_CROSS_BUCKETS: Tuple[int, ...] = (
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
)
# Accepted-path width is a capacity, not a logical sequence length.  Reusing
# power-of-two capacities bounds the lazy graph grid to five shapes and, in
# particular, prevents BS=1 from capturing separately for lengths 1..9.
# The model receives the capacity-padded tensors, but n_per_req gates each
# logical prefix and the worker scatters only n_per_req * K KV slots.
_ACCEPT_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16)
_MAX_BATCH_CAPACITY = 16


class SpecBlockRefreshCudaGraphRunner:
    """Capture and replay the batched SpecBlock refresh model forward."""

    def __init__(self, worker: "SpecBlockWorker"):
        self.worker = worker
        self.device = worker.device
        self.K = worker.K
        self.num_layers = worker.num_layers

        configured_bs = [
            int(bs)
            for bs in getattr(worker.server_args, "cuda_graph_bs", [1])
            if int(bs) > 0
        ]
        configured_max = min(
            max(configured_bs, default=1),
            _MAX_BATCH_CAPACITY,
        )
        self.capture_bs: List[int] = []
        capacity = 1
        while capacity < configured_max:
            self.capture_bs.append(capacity)
            capacity *= 2
        self.capture_bs.append(configured_max)

        self.cross_buckets = _CROSS_BUCKETS
        self.accept_buckets = _ACCEPT_BUCKETS
        self.graphs: Dict[
            Tuple[int, int, int], torch.cuda.CUDAGraph
        ] = {}
        self.input_buffers: Dict[
            Tuple[int, int, int], Dict[str, torch.Tensor]
        ] = {}
        self.output_buffers: Dict[Tuple[int, int, int], Dict] = {}
        self.pool_version_per_graph: Dict[Tuple[int, int, int], int] = {}

        # Refresh graphs execute serially on one worker.  Sharing the graph
        # memory pool avoids multiplying private allocation by every lazy
        # batch/cross/accept bucket while preserving stable captured pointers.
        self.graph_pool = torch.cuda.graph_pool_handle()

        logger.info(
            "[SpecBlockRefreshCudaGraphRunner] enabled "
            "(capture_bs=%s, cross_buckets=%s, accept_buckets=%s)",
            self.capture_bs,
            self.cross_buckets,
            self.accept_buckets,
        )

    def _resolve_batch_capacity(self, active_bs: int) -> Optional[int]:
        for capacity in self.capture_bs:
            if capacity >= active_bs:
                return capacity
        return None

    def _resolve_cross(self, max_cross_count: int) -> Optional[int]:
        for bucket in self.cross_buckets:
            if bucket >= max_cross_count:
                return bucket
        return None

    def _resolve_accept(self, max_n: int) -> Optional[int]:
        for bucket in self.accept_buckets:
            if bucket >= max_n:
                return bucket
        return None

    def resolve_buckets(
        self,
        active_bs: int,
        max_cross_count: int,
        max_n: int,
    ) -> Optional[Tuple[int, int, int]]:
        bcap = self._resolve_batch_capacity(active_bs)
        cross_bucket = self._resolve_cross(max_cross_count)
        accept_bucket = self._resolve_accept(max_n)
        if bcap is None or cross_bucket is None or accept_bucket is None:
            return None
        return bcap, cross_bucket, accept_bucket

    def can_run(self, bcap: int, cross_bucket: int, accept_bucket: int) -> bool:
        current_version = self.worker.spec_kv_pool.pool_version
        if self.graphs and (
            set(self.pool_version_per_graph) != set(self.graphs)
            or any(
                version != current_version
                for version in self.pool_version_per_graph.values()
            )
        ):
            self._invalidate_all()
            return False
        return (bcap, cross_bucket, accept_bucket) in self.graphs

    def _invalidate_all(self) -> None:
        if self.graphs:
            logger.info(
                "[SpecBlockRefreshCudaGraphRunner] pool grew; dropping "
                "%d captured refresh graphs",
                len(self.graphs),
            )
        self.graphs.clear()
        self.input_buffers.clear()
        self.output_buffers.clear()
        self.pool_version_per_graph.clear()
        # A replay result can remain live in the worker stack while KV-pool
        # growth invalidates its graph. Reusing that graph's shared mempool for
        # immediate recapture trips CUDACachingAllocator's use_count assertion.
        # Start a new pool generation; the old pool is released naturally once
        # those stale outputs leave scope.
        self.graph_pool = torch.cuda.graph_pool_handle()

    def capture_one(
        self,
        bcap: int,
        cross_bucket: int,
        accept_bucket: int,
    ) -> None:
        if (
            bcap not in self.capture_bs
            or cross_bucket not in self.cross_buckets
            or accept_bucket not in self.accept_buckets
        ):
            raise RuntimeError(
                "Unsupported SpecBlock refresh CUDA Graph bucket: "
                f"batch_capacity={bcap}, cross_bucket={cross_bucket}, "
                f"accept_bucket={accept_bucket}."
            )

        # resolve_buckets() returns canonical capacities, so this key also
        # deduplicates requests whose logical accepted lengths share a bucket
        # (for example, lengths 5--8 all reuse capacity 8).
        key = (bcap, cross_bucket, accept_bucket)
        if key in self.graphs:
            return

        draft_model = self.worker.model_runner.model.inner
        hidden_size = draft_model.config.hidden_size
        hidden3_size = 3 * hidden_size
        dtype = next(draft_model.parameters()).dtype
        rope_max_position = min(
            int(getattr(draft_model.config, "max_position_embeddings", 131072)),
            int(draft_model.layers[0].self_attn.rope.max_pos),
        ) - 1

        in_buf = {
            "hidden": torch.zeros(
                bcap,
                accept_bucket,
                hidden3_size,
                dtype=dtype,
                device=self.device,
            ),
            "tokens": torch.zeros(
                bcap,
                accept_bucket,
                dtype=torch.long,
                device=self.device,
            ),
            "pos_ids": torch.zeros(
                bcap,
                accept_bucket * self.K,
                dtype=torch.long,
                device=self.device,
            ),
            "cross_loc": torch.zeros(
                bcap,
                cross_bucket,
                dtype=torch.long,
                device=self.device,
            ),
            "cross_mask": torch.zeros(
                bcap,
                cross_bucket,
                dtype=torch.bool,
                device=self.device,
            ),
            # Every inactive capacity row remains a complete one-position dummy
            # request, so last-position gathers are always in bounds.
            "n_per_req": torch.ones(
                bcap,
                dtype=torch.long,
                device=self.device,
            ),
        }
        self.input_buffers[key] = in_buf

        cache = [
            [
                self.worker.spec_kv_pool,
                layer_idx,
                cross_bucket,
                in_buf["cross_loc"],
                None,
            ]
            for layer_idx in range(self.num_layers)
        ]

        def run_forward():
            return draft_model.update_cache_and_draft_graph_safe(
                in_buf["hidden"],
                in_buf["tokens"],
                in_buf["pos_ids"],
                cache,
                rope_max_position,
                accept_bucket,
                in_buf["cross_mask"],
                in_buf["n_per_req"],
            )

        with self.worker.spec_kv_pool.capture_mode():
            for _ in range(2):
                run_forward()
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                (
                    logits,
                    rank_logits,
                    draft_hidden,
                    new_cross_kv,
                    new_ttt_kv,
                ) = run_forward()

        self.graphs[key] = graph
        self.output_buffers[key] = {
            "logits": logits,
            "rank_logits": rank_logits,
            "draft_hidden": draft_hidden,
            "new_cross_kv": new_cross_kv,
            "new_ttt_kv": new_ttt_kv,
            "top_indices": draft_model._last_rank_top_indices,
        }
        self.pool_version_per_graph[key] = (
            self.worker.spec_kv_pool.pool_version
        )

        logger.info(
            "[SpecBlockRefreshCudaGraphRunner] captured "
            "batch_capacity=%d cross_bucket=%d accept_bucket=%d "
            "pool_version=%d",
            bcap,
            cross_bucket,
            accept_bucket,
            self.pool_version_per_graph[key],
        )

    def precapture_up_to(self, max_cross_bucket: int) -> None:
        """Capture common refresh shapes before the server accepts traffic."""
        if max_cross_bucket <= 0:
            return
        keys = [
            (bcap, cross_bucket, accept_bucket)
            for bcap in self.capture_bs
            for cross_bucket in self.cross_buckets
            if cross_bucket <= max_cross_bucket
            for accept_bucket in self.accept_buckets
        ]
        for key in keys:
            self.capture_one(*key)
        logger.info(
            "[SpecBlockRefreshCudaGraphRunner] precaptured %d graphs "
            "through cross_bucket=%d",
            len(keys),
            max_cross_bucket,
        )

    def _copy_inputs(
        self,
        in_buf: Dict[str, torch.Tensor],
        active_bs: int,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        pos_ids: torch.Tensor,
        cross_loc: torch.Tensor,
        cross_mask: torch.Tensor,
        n_per_req: torch.Tensor,
    ) -> None:
        bcap = in_buf["hidden"].shape[0]
        accept_bucket = in_buf["hidden"].shape[1]
        cross_bucket = in_buf["cross_loc"].shape[1]
        if not 0 < active_bs <= bcap:
            raise RuntimeError(
                f"Invalid active_bs={active_bs} for refresh capacity {bcap}."
            )
        expected_hidden_shape = (
            active_bs,
            accept_bucket,
            in_buf["hidden"].shape[-1],
        )
        if hidden.shape != expected_hidden_shape:
            raise RuntimeError(
                "Refresh hidden input does not match resolved graph bucket: "
                f"got={tuple(hidden.shape)}, expected="
                f"{expected_hidden_shape}."
            )
        if tokens.shape != (active_bs, accept_bucket):
            raise RuntimeError(
                f"Refresh token shape {tuple(tokens.shape)} is invalid."
            )
        if pos_ids.shape != (active_bs, accept_bucket * self.K):
            raise RuntimeError(
                f"Refresh position shape {tuple(pos_ids.shape)} is invalid."
            )
        if cross_loc.shape != (active_bs, cross_bucket):
            raise RuntimeError(
                f"Refresh cross_loc shape {tuple(cross_loc.shape)} is invalid."
            )
        if cross_mask.shape != cross_loc.shape:
            raise RuntimeError(
                f"Refresh cross_mask shape {tuple(cross_mask.shape)} is invalid."
            )
        if n_per_req.shape != (active_bs,):
            raise RuntimeError(
                f"Refresh n_per_req shape {tuple(n_per_req.shape)} is invalid."
            )
        # Every active row is copied at full graph capacity, including its
        # zero-padded tails.  Clearing those rows first only adds redundant
        # launches and memory traffic; sanitize solely the inactive batch tail
        # that replay still observes through the fixed-capacity graph.
        if active_bs < in_buf["hidden"].shape[0]:
            in_buf["hidden"][active_bs:].zero_()
            in_buf["tokens"][active_bs:].zero_()
            in_buf["pos_ids"][active_bs:].zero_()
            in_buf["cross_loc"][active_bs:].zero_()
            in_buf["cross_mask"][active_bs:].zero_()
            in_buf["n_per_req"][active_bs:].fill_(1)

        in_buf["hidden"][:active_bs].copy_(hidden)
        in_buf["tokens"][:active_bs].copy_(tokens)
        in_buf["pos_ids"][:active_bs].copy_(pos_ids)
        in_buf["cross_loc"][:active_bs].copy_(cross_loc)
        in_buf["cross_mask"][:active_bs].copy_(cross_mask)
        in_buf["n_per_req"][:active_bs].copy_(n_per_req)

    def replay(
        self,
        bcap: int,
        cross_bucket: int,
        accept_bucket: int,
        active_bs: int,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        pos_ids: torch.Tensor,
        cross_loc: torch.Tensor,
        cross_mask: torch.Tensor,
        n_per_req: torch.Tensor,
    ):
        key = (bcap, cross_bucket, accept_bucket)
        if key not in self.graphs:
            raise RuntimeError(
                f"SpecBlock refresh graph {key} was not captured."
            )
        if (
            self.pool_version_per_graph[key]
            != self.worker.spec_kv_pool.pool_version
        ):
            self._invalidate_all()
            raise RuntimeError(
                "SpecBlock KV pool grew before refresh replay; the stale "
                "captured graph was invalidated."
            )

        in_buf = self.input_buffers[key]
        self._copy_inputs(
            in_buf,
            active_bs,
            hidden,
            tokens,
            pos_ids,
            cross_loc,
            cross_mask,
            n_per_req,
        )
        self.graphs[key].replay()

        out = self.output_buffers[key]
        return (
            out["logits"][:active_bs],
            out["rank_logits"][:active_bs],
            out["draft_hidden"][:active_bs],
            [
                (layer_k[:active_bs], layer_v[:active_bs])
                for layer_k, layer_v in out["new_cross_kv"]
            ],
            [
                (layer_k[:active_bs], layer_v[:active_bs])
                for layer_k, layer_v in out["new_ttt_kv"]
            ],
            out["top_indices"][:active_bs],
        )
