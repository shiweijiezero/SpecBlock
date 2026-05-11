"""flashinfer-backed paged cross attention for SpecBlock-Shift draft.

The 3WAY Triton kernel's cross loop (the dominant cost in the draft
forward) is replaced with flashinfer's
:class:`BatchPrefillWithPagedKVCacheWrapper`, which is the same kernel
SGLang main path uses for paged decode -- highly tuned, with stride-0
broadcast / fused mask / shared-mem reuse that our Triton paged variant
cannot match per-token.

Flow per draft attention call::

    cross stage  ->  flashinfer (out_cross, lse_cross)
    ttt+curr stage -> Triton _two_part_attn_fwd_kernel (out_other, lse_other)
    merge: O = lse-weighted combine(O_cross, O_other) via online softmax

Cross stage layout (per row of the pend forward, ``tp = bs * n_pend_max``)::

    q[K=4, n_heads, head_dim]   queries (post-RoPE, GQA-expanded)
    pool[pool_size, n_heads, head_dim]   stored K/V at the pool's per-layer
                                         slice (page_size=1, n_heads = n_heads)
    cross_loc[count_i]   int32 indices into the pool

Each row contributes a (qo_indptr_i, paged_kv_indptr_i) pair to
flashinfer's plan; rows with cross_count == 0 fall through to the
"all ttt+curr" merge path (out_cross == 0, lse_cross == -inf).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class SpecBlockFlashInferCross:
    """Worker-local flashinfer wrapper for paged cross attention.

    One wrapper instance is created per worker; it amortises the
    per-call workspace allocation across decode steps and lets us reuse
    flashinfer's plan() metadata layout when only the cross_loc tensor
    changes between iterations.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        workspace_bytes: int = 128 * 1024 * 1024,  # 128 MB
    ):
        try:
            import flashinfer  # noqa: F401
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper
        except ImportError as e:
            raise RuntimeError(
                "[SpecBlockFlashInferCross] flashinfer is required "
                "(install via SGLang's flashinfer; we use 0.5.x APIs)."
            ) from e

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

        self._workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=self.device,
        )
        # NHD layout: paged_kv_cache shape [num_pages, page_size, n_heads, D].
        # We use page_size = 1 (one slot per page, simplest mapping for
        # SpecBlockKVPool's flat indexing).
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD",
        )
        self._page_size = 1

        # Plan-level cache state: filled by ``plan_step``, consumed by
        # ``run_layer`` so the same plan amortises across all draft
        # layers (cross_loc / cross_counts are layer-invariant within a
        # single decode step).  The cache is invalidated on every new
        # plan_step call (which overwrites _plan_K / _plan_tp).
        self._plan_K: int = 0
        self._plan_tp: int = 0
        self._plan_total_kv: int = 0
        self._plan_scale: float = 0.0

        logger.info(
            "SpecBlockFlashInferCross initialized: n_heads=%d, head_dim=%d, "
            "dtype=%s, workspace=%.0f MB",
            n_heads, head_dim, str(dtype), workspace_bytes / (1 << 20),
        )

    # ------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------

    def plan_step(
        self,
        cross_loc_padded: torch.Tensor,    # [tp, max_count] int64 (or int32)
        cross_counts: List[int],           # length tp (or B if qo_len_per_segment is K * n_pend_max)
        K: int,                            # query length per segment (= queries per row)
        scale: float,
        device: torch.device,
        dtype: torch.dtype,
        *,
        qo_len_per_segment: Optional[int] = None,  # override if segments group multiple rows
    ) -> bool:
        """Build & cache the flashinfer plan for a single decode step.

        Cross_loc / cross_counts are layer-invariant within a step, so
        plan once here and reuse via ``run_layer`` for all draft layers.

        ``qo_len_per_segment`` (default = K) lets callers fuse multiple
        contiguous rows that share the same cross_loc into a single
        flashinfer segment.  E.g., the draft block-2 batched forward
        has B requests × n_pend_max pend rows × K queries each, but
        all pend rows within a request share cross_loc — passing
        ``cross_counts`` of length B (per-req) and ``qo_len_per_segment
        = K * n_pend_max`` collapses 120 segments into 4, saving 90%
        of plan() metadata work.

        Returns False when total_kv == 0 (caller should skip the
        flashinfer cross stage and treat lse_cross as -inf).
        """
        tp = len(cross_counts)
        n_heads = self.n_heads
        D = self.head_dim
        if qo_len_per_segment is None:
            qo_len_per_segment = K

        cum_count = 0
        kv_indptr_list = [0]
        for cnt in cross_counts:
            cum_count += cnt
            kv_indptr_list.append(cum_count)
        total_kv = cum_count

        # Cache plan params for run_layer's empty-cross fast path.  The
        # ``_plan_total_q`` is the query count run_layer's output buffer
        # uses; with per-segment fusion it equals tp * qo_len_per_segment
        # (which equals B * n_pend_max * K = original plan's tp * K).
        total_q = tp * qo_len_per_segment
        self._plan_K = K
        self._plan_tp = tp
        self._plan_qo_len_per_segment = qo_len_per_segment
        self._plan_total_q = total_q
        self._plan_total_kv = total_kv
        self._plan_scale = scale
        self._plan_device = device
        self._plan_dtype = dtype

        if total_kv == 0:
            # No plan needed; run_layer returns zero out + -inf lse.
            return False

        # Strip padded slots from each row of cross_loc_padded.
        loc_parts = []
        for i, cnt in enumerate(cross_counts):
            if cnt == 0:
                continue
            row = cross_loc_padded[i, :cnt]
            if row.dtype != torch.int32:
                row = row.to(torch.int32)
            loc_parts.append(row)
        kv_indices = (
            torch.cat(loc_parts) if loc_parts
            else torch.empty(0, dtype=torch.int32, device=device)
        )

        qo_indptr = torch.arange(
            0, (tp + 1) * qo_len_per_segment, step=qo_len_per_segment,
            dtype=torch.int32, device=device,
        )
        kv_indptr = torch.tensor(
            kv_indptr_list, dtype=torch.int32, device=device,
        )
        last_page_len = torch.full(
            (tp,), self._page_size, dtype=torch.int32, device=device,
        )

        self._wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page_len,
            num_qo_heads=n_heads,
            num_kv_heads=n_heads,
            head_dim_qk=D,
            page_size=self._page_size,
            causal=False,
            sm_scale=scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        return True

    def run_layer(
        self,
        q: torch.Tensor,                    # [total_q, n_heads, head_dim]
        pool_k_layer: torch.Tensor,         # [pool_size, n_heads, head_dim]
        pool_v_layer: torch.Tensor,         # [pool_size, n_heads, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the cached plan against a layer's pool slice.

        ``plan_step`` must have been called for the current decode step.
        Returns ``(out, lse)``::

            out: [total_q, n_heads, head_dim]  attention output
            lse: [total_q, n_heads]            log-sum-exp (for merge)

        ``total_q = _plan_tp * _plan_qo_len_per_segment``.  With default
        ``qo_len_per_segment=K`` this equals legacy ``tp * K``; with
        per-req segmentation (qo_len_per_segment = K * n_pend_max) this
        equals ``B * K * n_pend_max`` (same total).
        """
        total_q = self._plan_total_q
        n_heads = self.n_heads
        D = self.head_dim

        if self._plan_total_kv == 0:
            # Empty-cross fast path: out 0, lse -inf.
            out = torch.zeros((total_q, n_heads, D), dtype=q.dtype, device=q.device)
            lse = torch.full(
                (total_q, n_heads), float("-inf"),
                dtype=torch.float32, device=q.device,
            )
            return out, lse

        # unsqueeze(1) is a stride trick (no copy) since pool is
        # contiguous along (slot, n_heads, D).
        pool_k_4d = pool_k_layer.unsqueeze(1)
        pool_v_4d = pool_v_layer.unsqueeze(1)

        out, lse = self._wrapper.run(
            q, (pool_k_4d, pool_v_4d), return_lse=True,
        )
        return out, lse

    # Backwards-compat shim: combined plan + run for tests / single-layer.
    def forward(
        self,
        q: torch.Tensor,
        pool_k_layer: torch.Tensor,
        pool_v_layer: torch.Tensor,
        cross_loc_padded: torch.Tensor,
        cross_counts: List[int],
        K: int,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.plan_step(
            cross_loc_padded, cross_counts, K, scale, q.device, q.dtype,
        )
        return self.run_layer(q, pool_k_layer, pool_v_layer)
