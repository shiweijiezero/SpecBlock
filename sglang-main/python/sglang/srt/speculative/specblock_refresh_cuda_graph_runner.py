"""SpecBlock-Shift Refresh CUDA Graph Runner.

Captures and replays :meth:`SpecBlockWorker._refresh_draft_state`'s
batched ``update_cache_and_draft`` chain per ``(bs, cross_bucket,
accept_max)`` shape combination.

## Capture region

  Build padded hidden / tokens / cross_loc / new_indices on static buffers
  → (caller has already alloc'd new paged slots, copies into static)
  → ``draft_model.update_cache_and_draft_graph_safe`` (graph-friendly variant)
  → output: logits, rank_logits, draft_hidden, ttt_kv (read post-replay)

The graph_safe variant exists in
:mod:`sglang.srt.models._specblock_inference` (``forward_batch_graph_safe``
+ ``update_cache_and_draft_graph_safe``), with no internal arange / dynamic
rope slice / in-place cache mutation — capture-friendly.

## Bucket grid

* ``bs``: from server_args.cuda_graph_bs (typically [1, 2, 4, 8]).
* ``cross_bucket``: smallest of (128, 256, 512, 1024, 2048) >= max
  per-req cross_count.
* ``accept_max``: smallest of (2, 4, 8) >= max(accept_length+1).
  accept_length is data-dependent (verify outcome) — bucket-rounded.

Total graphs = 4 × 5 × 3 = 60.  Lazy capture; no eviction in this
session (60 × ~0.3 GB ≈ 18 GB OK at mem-fraction 0.5 with model + pool).

## Production policy

CUDA graph is ON by default.  ``SPECBLOCK_REFRESH_CUDA_GRAPH=0`` disables
for debugging only.  Capture failure raises (no eager fallback).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.speculative.specblock_worker import SpecBlockWorker

logger = logging.getLogger(__name__)


_CROSS_BUCKETS: Tuple[int, ...] = (128, 256, 512, 1024, 2048)
_ACCEPT_BUCKETS: Tuple[int, ...] = (2, 4, 8)


class SpecBlockRefreshCudaGraphRunner:
    """CUDA graph runner for SpecBlockWorker._refresh_draft_state.

    Per (bs, cross_bucket, accept_max) graph captures the batched
    update_cache_and_draft forward.  Pad rows / pad cross slots are
    masked out via per-row cross_mask + n_per_req tensor.

    Lifecycle::

        runner = SpecBlockRefreshCudaGraphRunner(worker)
        # in worker._refresh_draft_state:
        bucket = runner.resolve_buckets(bs, max_cross, max_N)
        if bucket and runner.can_run(*bucket):
            outputs = runner.replay(static_inputs)  # logits/rank/hidden/ttt_kv
        else:
            outputs = update_cache_and_draft(...)  # eager
    """

    def __init__(self, worker: "SpecBlockWorker"):
        self.worker = worker
        self.device = worker.device
        self.K = worker.K
        self.num_layers = worker.num_layers

        sa = worker.server_args
        self.capture_bs: List[int] = list(getattr(sa, "cuda_graph_bs", [1, 2, 4, 8]))
        self.max_bs = max(self.capture_bs)

        self.cross_buckets = _CROSS_BUCKETS
        self.accept_buckets = _ACCEPT_BUCKETS
        self.max_cross = self.cross_buckets[-1]
        self.max_accept = self.accept_buckets[-1]

        # Per-(bs, cb, ab) cache.
        self.graphs: Dict[Tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.input_buffers: Dict[Tuple[int, int, int], Dict] = {}
        self.output_buffers: Dict[Tuple[int, int, int], Dict] = {}
        # See SpecBlockDraftCudaGraphRunner: pool grow replaces buffer
        # pointers; captured graphs hold stale pointers — invalidate on
        # pool_version bump.
        self.pool_version_per_graph: Dict[Tuple[int, int, int], int] = {}

        self.enabled = (
            os.environ.get("SPECBLOCK_REFRESH_CUDA_GRAPH", "1") == "1"
        )
        if self.enabled:
            logger.info(
                "[SpecBlockRefreshCudaGraphRunner] enabled "
                "(capture_bs=%s, cross_buckets=%s, accept_buckets=%s)",
                self.capture_bs, self.cross_buckets, self.accept_buckets,
            )

    # ============================================================
    #  Bucket resolution
    # ============================================================

    def _resolve_cross(self, max_cross_count: int) -> Optional[int]:
        for cb in self.cross_buckets:
            if cb >= max_cross_count:
                return cb
        return None

    def _resolve_accept(self, max_n: int) -> Optional[int]:
        for ab in self.accept_buckets:
            if ab >= max_n:
                return ab
        return None

    def resolve_buckets(
        self, bs: int, max_cross_count: int, max_n: int,
    ) -> Optional[Tuple[int, int, int]]:
        if bs not in self.capture_bs:
            return None
        cb = self._resolve_cross(max_cross_count)
        ab = self._resolve_accept(max_n)
        if cb is None or ab is None:
            return None
        return (bs, cb, ab)

    def can_run(self, bs: int, cb: int, ab: int) -> bool:
        if not self.enabled:
            return False
        key = (bs, cb, ab)
        if key not in self.graphs:
            return False
        cur_version = self.worker.spec_kv_pool.pool_version
        if self.pool_version_per_graph.get(key, -1) != cur_version:
            self._invalidate_all()
            return False
        return True

    def _invalidate_all(self) -> None:
        if self.graphs:
            logger.info(
                "[SpecBlockRefreshCudaGraphRunner] pool grew; dropping "
                "%d captured graphs (will re-capture)", len(self.graphs),
            )
        self.graphs.clear()
        self.input_buffers.clear()
        self.output_buffers.clear()
        self.pool_version_per_graph.clear()

    # ============================================================
    #  Capture / replay
    # ============================================================

    def capture_one(self, bs: int, cb: int, ab: int) -> None:
        """Capture refresh chain for (bs, cb, ab).

        Pre-allocates static input + output buffers; runs warmup; captures
        ``update_cache_and_draft_graph_safe``.  Failure raises.
        """
        if not self.enabled:
            raise RuntimeError("[capture_one] runner disabled")
        key = (bs, cb, ab)
        if key in self.graphs:
            return

        worker = self.worker
        device = self.device
        draft_model = worker.model_runner.model.inner
        K = self.K
        H = draft_model.config.hidden_size
        H3 = 3 * H
        rope_max_position = getattr(
            draft_model.config, "max_position_embeddings", 131072,
        )
        dtype = next(draft_model.parameters()).dtype

        # ---- Static input buffers ----
        in_buf = {
            "hidden_padded":     torch.zeros((bs, ab, H3), dtype=dtype, device=device),
            "tokens_padded":     torch.zeros((bs, ab), dtype=torch.long, device=device),
            "cross_loc_padded":  torch.zeros((bs, cb), dtype=torch.long, device=device),
            "cross_mask":        torch.zeros((bs, cb), dtype=torch.bool, device=device),
            "new_indices_padded": torch.zeros(
                (bs, ab * K), dtype=torch.long, device=device,
            ),
            # pos_ids for graph_safe path: [B, ab*K] precomputed
            "pos_ids":           torch.zeros((bs, ab * K), dtype=torch.long, device=device),
        }
        self.input_buffers[key] = in_buf

        # The graph_safe path's cache layout is paged 5-tuple:
        # [pool, layer_id, count, cross_loc, new_indices]
        spec_kv_pool = worker.spec_kv_pool
        cache = [
            [spec_kv_pool, L, cb, in_buf["cross_loc_padded"], in_buf["new_indices_padded"]]
            for L in range(self.num_layers)
        ]

        # Warmup x2 to settle scratch.
        for _ in range(2):
            _ = draft_model.update_cache_and_draft_graph_safe(
                in_buf["hidden_padded"],
                in_buf["tokens_padded"],
                in_buf["pos_ids"],
                cache,
                rope_max_position,
                ab,
                in_buf["cross_mask"],
            )
        torch.cuda.synchronize()

        # Capture into a graph.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            logits, rank_logits, draft_hidden, new_kv_full = (
                draft_model.update_cache_and_draft_graph_safe(
                    in_buf["hidden_padded"],
                    in_buf["tokens_padded"],
                    in_buf["pos_ids"],
                    cache,
                    rope_max_position,
                    ab,
                    in_buf["cross_mask"],
                )
            )

        # The captured forward writes into in_buf-derived tensors but
        # creates fresh output tensors on the private mempool.  Save
        # those for replay-time read.
        self.output_buffers[key] = {
            "logits": logits,
            "rank_logits": rank_logits,
            "draft_hidden": draft_hidden,
            "new_kv_full": new_kv_full,
        }
        self.graphs[key] = g
        self.pool_version_per_graph[key] = (
            self.worker.spec_kv_pool.pool_version
        )

        logger.info(
            "[SpecBlockRefreshCudaGraphRunner] captured graph for "
            "bs=%d cross_bucket=%d accept_max=%d (pool_version=%d)",
            bs, cb, ab, self.pool_version_per_graph[key],
        )

    def replay(
        self,
        bs: int, cb: int, ab: int,
        hidden_padded: torch.Tensor,
        tokens_padded: torch.Tensor,
        pos_ids: torch.Tensor,
        cross_loc_padded: torch.Tensor,
        cross_mask: torch.Tensor,
        new_indices_padded: torch.Tensor,
    ):
        """Replay the (bs, cb, ab) graph with current inputs.

        Caller ensures input tensor shapes match the captured shapes
        (pad / truncate as needed).  Returns
        ``(logits, rank_logits, draft_hidden, new_kv_full)`` references
        into the captured output buffers — caller must consume / clone
        before next replay call.
        """
        key = (bs, cb, ab)
        g = self.graphs[key]
        in_buf = self.input_buffers[key]

        in_buf["hidden_padded"].copy_(hidden_padded)
        in_buf["tokens_padded"].copy_(tokens_padded)
        in_buf["pos_ids"].copy_(pos_ids)
        in_buf["cross_loc_padded"].copy_(cross_loc_padded)
        in_buf["cross_mask"].copy_(cross_mask)
        in_buf["new_indices_padded"].copy_(new_indices_padded)

        g.replay()

        out = self.output_buffers[key]
        return (
            out["logits"], out["rank_logits"],
            out["draft_hidden"], out["new_kv_full"],
        )
