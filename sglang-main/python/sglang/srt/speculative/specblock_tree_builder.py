"""Standalone SpecBlock-Shift BFS tree builder (Phase 1.3 MVP).

Self-contained PyTorch module that constructs a rank-guided draft tree
for greedy speculative decoding.  Ported from
``benchmarks_hf/algorithms/specblock.py`` so it has no dependency on
benchmarks_hf or SGLang internals; the only callable surface is
:func:`build_tree`.

Scope (Phase 1.3 MVP):
    * batch=1 only (single prompt, single beam root)
    * temperature == 0 (greedy verify path; no stochastic sampling)
    * up to ``max_blocks`` BFS depths (typically 2 for SpecBlock-Shift)
    * pure-PyTorch / numpy implementation; Triton/CUDA kernels in the
      reference live behind their own paths and are intentionally NOT
      ported here (a stubbed comment marks the optimisation hook).

Reference line ranges in ``benchmarks_hf/algorithms/specblock.py``:
    * L60-86   ``_walk_rank_slots_compiled``  -> :func:`_walk_rank_slots`
    * L274-355 ``_vectorized_greedy_chain``   -> :func:`_vectorized_greedy_chain`
    * L1110-1158 ``_prune_tree_np``           -> :func:`_prune_tree_np`
    * L2705-3199 ``_build_tree_from_block1`` block-1 expansion
                                              -> :func:`_expand_block1`
    * L3200-3744 ``_build_tree_from_block1`` BFS (block 2+)
                                              -> :func:`_bfs_expand`
    * L357-416  ``build_tree_buffers_triton`` (Triton kernels)
                                              -> :func:`_build_tree_topology`
                                              (numpy fallback)

Expected draft model interface (matches
``SpecBlockShiftInferenceModel.forward_with_cache``)::

    logits, rank_logits, draft_hidden, new_ttt_kv = draft_model.forward_with_cache(
        hidden=...,             # [B, 1, 3H] block 1 / [B, 1, H] block 2+
        input_ids=...,          # [B, 1]
        cache=cross_kv_cache,   # per-layer [k_buf, v_buf, count, max_len]
        position_id=int,
        use_draft_condition=bool,
        ttt_cache=...,          # per-layer (k, v) or None
        ttt_mask=...,           # [B, K]
        update_cross_cache=bool,
        full_kv_mask=...,       # optional
    )

The draft model is also expected to expose ``draft_model.d2t``: a
``[V_draft]`` long tensor mapping draft vocab ids to target vocab offsets
(``target_id = draft_id + d2t[draft_id]``).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
#  Helpers (small, dependency-free)
# ============================================================

# Branching factor table indexed by rank class.  Mirrors specblock.py
# ``RANK_BRANCH_FACTORS`` (L968) and ``_rank_to_factor`` (L613).  rank
# class 0 is "correct, continue", classes 1/2 trigger small/large
# branches, the last class is "give up" (no branch).
_RANK_TO_FACTOR_DEFAULT = (0, 4, 10, 0)

# Per-slot top-k branching count per rank class.
#
# The batched GPU path defaults to the measured Pareto point
# `slim_r2` -> `(2, 4, 6, 4)`.  It preserves the useful alternatives while
# keeping the root-inclusive target verify width small enough for bs>1.
_RANK_SLOT_TOPK_DEFAULT_MAP = {
    'default':    (2, 4, 10, 0),
    'no_giveup':  (2, 4, 10, 4),
    'slim_r2':    (2, 4, 6, 4),
    'slim_r2b':   (2, 3, 5, 3),
}
_RANK_SLOT_TOPK_DEFAULT = _RANK_SLOT_TOPK_DEFAULT_MAP[
    os.environ.get('SLOT_TOPK_MODE', 'slim_r2')
]

# ADAPTIVE_SLOT0: shrink slot-0 beam (block-1 alternatives) based on greedy
# probability p0 = exp(glp_np[0]).  Mirrors specblock.py:3107-3137.
#
#   0 = off (use full beam_width for slot 0; default)
#   1 = conservative: p0>=0.9 -> 3, p0>=0.8 -> 5, p0>=0.6 -> 8
#   2 = aggressive:   p0>=0.95 -> 2, p0>=0.85 -> 4, p0>=0.6 -> 7
#   3 = extreme:      p0>=0.95 -> 1, p0>=0.9 -> 2, p0>=0.8 -> 4, p0>=0.6 -> 7
#   4 = ultra:        p0>=0.9 -> 1, p0>=0.8 -> 3, p0>=0.6 -> 6
#
# All modes fall back to full beam_width when no threshold matches.  Saved
# block-1 slot-0 budget can fund deeper BFS branches (`total_tokens` is
# tree-wide).
_ADAPTIVE_SLOT0 = int(os.environ.get('ADAPTIVE_SLOT0', '0'))

# ADAPTIVE_ALL: per-slot topk shrinkage for slots k>=1 (block-1 inside
# `_expand_block1` and all leaves * slots in `_bfs_expand`) based on
# pk = exp(glp[k]).  Mirrors specblock.py:3144-3155 (block 1) and
# specblock.py:3481-3497 (BFS).
#
#   0 = off (use rank_slot_topk[rank_pred] verbatim; default)
#   1 = conservative: pk>=0.95 -> min(2,base), pk>=0.85 -> min(3,base)
#   2 = aggressive:   pk>=0.9 -> 1, pk>=0.7 -> min(2,base), pk>=0.5 -> min(3,base)
#
# `base` is `rank_slot_topk[rank_pred]` (or per-pair in BFS).  When neither
# threshold matches, the rank-table value is kept.  give_up class with
# rank_slot_topk[give_up]==0 still terminates block-1 early (HF parity).
_ADAPTIVE_ALL = int(os.environ.get('ADAPTIVE_ALL', '0'))


# ============================================================
#  Constant tensor cache
# ============================================================
#
# `torch.tensor(int_val, device=cuda)` triggers a host->device copy that
# is not allowed inside a cuda graph capture region.  We cache device-
# side constants the slot-topk helpers need so they can be looked up by
# value (no fresh host->device traffic during capture).
#
# Caller (worker init / first eager call) pre-warms the cache by calling
# the helpers at least once outside capture; subsequent capture calls
# reuse the cached tensors.

_CONST_T_CACHE: Dict[Tuple[torch.device, torch.dtype, int], torch.Tensor] = {}


def _const_t(
    value: int, device: torch.device, dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Return a cached 0-d device tensor holding ``value``.

    First call for a given ``(device, dtype, value)`` does the
    host->device copy; subsequent calls return the cached tensor.
    Must be called at least once outside any cuda graph capture region
    so the cache is warm before capture starts.
    """
    key = (device, dtype, value)
    t = _CONST_T_CACHE.get(key)
    if t is None:
        t = torch.tensor(value, device=device, dtype=dtype)
        _CONST_T_CACHE[key] = t
    return t


# ============================================================
#  ADAPTIVE slot-topk computation (shared between numpy + GPU paths)
# ============================================================

def compute_slot_topks_block1_gpu(
    glp_t: torch.Tensor,       # [K] block-1 greedy log_probs (gpu, f32)
    rp_t: torch.Tensor,        # [K] block-1 rank predictions (gpu, long)
    rank_slot_topk_t: torch.Tensor,  # [rank_classes] (gpu, long)
    beam_width: int,
    rank_classes: int,
) -> torch.Tensor:
    """GPU equivalent of :func:`compute_slot_topks_block1`.

    Eliminates the host sync that the numpy version forces (caller no longer
    has to ``.cpu().numpy()`` glp / rp).  Same logic — branching is expressed
    via ``torch.where`` chains keyed off the env constants.

    Returns [K] long tensor on the same device.
    """
    K = int(glp_t.shape[0])
    device = glp_t.device
    give_up_class = rank_classes - 1

    # Base topk per slot via rank_slot_topk[rp].
    base_topks = rank_slot_topk_t[rp_t.long()]  # [K]

    # ---- Slot-0 beam width (with optional ADAPTIVE_SLOT0 shrinkage) ----
    if _ADAPTIVE_SLOT0 == 0 or K == 0:
        # Cached device tensor for the constant beam_width — capture-safe
        # (no host→device copy if the cache was warmed up via a prior
        # eager call).  Worker init pre-warms via the first eager
        # build_tree call.
        slot0_t: Optional[torch.Tensor] = _const_t(beam_width, device).unsqueeze(0)
    else:
        p0 = torch.exp(glp_t[:1])  # [1]
        bw_t = _const_t(beam_width, device).unsqueeze(0)  # [1]
        if _ADAPTIVE_SLOT0 == 2:
            v = torch.where(p0 >= 0.95, torch.minimum(_const_t(2, device), bw_t),
                torch.where(p0 >= 0.85, torch.minimum(_const_t(4, device), bw_t),
                torch.where(p0 >= 0.6, torch.minimum(_const_t(7, device), bw_t), bw_t)))
        elif _ADAPTIVE_SLOT0 == 3:
            v = torch.where(p0 >= 0.95, _const_t(1, device),
                torch.where(p0 >= 0.9, torch.minimum(_const_t(2, device), bw_t),
                torch.where(p0 >= 0.8, torch.minimum(_const_t(4, device), bw_t),
                torch.where(p0 >= 0.6, torch.minimum(_const_t(7, device), bw_t), bw_t))))
        elif _ADAPTIVE_SLOT0 == 4:
            v = torch.where(p0 >= 0.9, _const_t(1, device),
                torch.where(p0 >= 0.8, torch.minimum(_const_t(3, device), bw_t),
                torch.where(p0 >= 0.6, torch.minimum(_const_t(6, device), bw_t), bw_t)))
        else:  # mode 1
            v = torch.where(p0 >= 0.9, torch.minimum(_const_t(3, device), bw_t),
                torch.where(p0 >= 0.8, torch.minimum(_const_t(5, device), bw_t),
                torch.where(p0 >= 0.6, torch.minimum(_const_t(8, device), bw_t), bw_t)))
        slot0_t = v  # [1]

    # ---- Slot k>=1: base_topks with optional ADAPTIVE_ALL shrinkage ----
    slot_topks = base_topks.clone()
    if _ADAPTIVE_ALL:
        pk = torch.exp(glp_t)  # [K]
        if _ADAPTIVE_ALL == 2:
            slot_topks = torch.where(
                pk >= 0.9, _const_t(1, device),
                torch.where(
                    pk >= 0.7, _const_t(2, device),
                    torch.where(
                        pk >= 0.5, _const_t(3, device),
                        base_topks,
                    ),
                ),
            )
        else:
            slot_topks = torch.where(
                pk >= 0.95, torch.minimum(_const_t(2, device), base_topks),
                torch.where(
                    pk >= 0.85, torch.minimum(_const_t(3, device), base_topks),
                    base_topks,
                ),
            )

    # Override slot 0 (slot0_t is always a [1] device tensor now).
    slot_topks = slot_topks.clone()
    slot_topks[0] = slot0_t[0]

    # ---- give_up early-stop: zero out slots from first k>=1 where
    # rp == give_up_class and base_topk == 0 (mirrors numpy's `break`).
    # Cumulative mask: any slot at-or-after first trigger gets zeroed.
    if K > 1:
        give_up_trigger = (rp_t == give_up_class) & (base_topks == 0)  # [K]
        # k=0 doesn't trigger break in numpy; mask via arange instead of
        # in-place index assign (capture-safe).
        k_ge_1_mask = torch.arange(K, device=device) > 0  # [K] bool
        give_up_trigger_k_ge_1 = give_up_trigger & k_ge_1_mask
        # at-or-after-first-trigger mask via cumulative OR
        triggered_at_or_after = torch.cummax(give_up_trigger_k_ge_1.long(), dim=0).values.bool()
        slot_topks = torch.where(triggered_at_or_after, torch.zeros_like(slot_topks), slot_topks)

    return slot_topks


def compute_slot_topks_block1_gpu_batched(
    glp_t: torch.Tensor,             # [B, K] block-1 greedy log_probs (gpu, f32)
    rp_t: torch.Tensor,              # [B, K] block-1 rank predictions (gpu, long)
    rank_slot_topk_t: torch.Tensor,  # [rank_classes] (gpu, long)
    beam_width: int,
    rank_classes: int,
) -> torch.Tensor:
    """Batched [B, K] version of :func:`compute_slot_topks_block1_gpu`.

    Identical logic; broadcast operations across the batch dim.  Returns
    [B, K] long on the same device.
    """
    B, K = int(glp_t.shape[0]), int(glp_t.shape[1])
    device = glp_t.device
    give_up_class = rank_classes - 1

    base_topks = rank_slot_topk_t[rp_t.long()]                  # [B, K]

    # ---- Slot-0 beam width (with optional ADAPTIVE_SLOT0 shrinkage) ----
    if _ADAPTIVE_SLOT0 == 0 or K == 0:
        slot0_t: Optional[torch.Tensor] = (
            _const_t(beam_width, device).expand(B, 1)
        )
    else:
        p0 = torch.exp(glp_t[:, :1])                            # [B, 1]
        bw_t = _const_t(beam_width, device).expand(B, 1)        # [B, 1]
        c1 = _const_t(1, device).expand(B, 1)
        c2 = _const_t(2, device).expand(B, 1)
        c3 = _const_t(3, device).expand(B, 1)
        c4 = _const_t(4, device).expand(B, 1)
        c5 = _const_t(5, device).expand(B, 1)
        c6 = _const_t(6, device).expand(B, 1)
        c7 = _const_t(7, device).expand(B, 1)
        c8 = _const_t(8, device).expand(B, 1)
        if _ADAPTIVE_SLOT0 == 2:
            v = torch.where(p0 >= 0.95, torch.minimum(c2, bw_t),
                torch.where(p0 >= 0.85, torch.minimum(c4, bw_t),
                torch.where(p0 >= 0.6,  torch.minimum(c7, bw_t), bw_t)))
        elif _ADAPTIVE_SLOT0 == 3:
            v = torch.where(p0 >= 0.95, c1,
                torch.where(p0 >= 0.9,  torch.minimum(c2, bw_t),
                torch.where(p0 >= 0.8,  torch.minimum(c4, bw_t),
                torch.where(p0 >= 0.6,  torch.minimum(c7, bw_t), bw_t))))
        elif _ADAPTIVE_SLOT0 == 4:
            v = torch.where(p0 >= 0.9, c1,
                torch.where(p0 >= 0.8,  torch.minimum(c3, bw_t),
                torch.where(p0 >= 0.6,  torch.minimum(c6, bw_t), bw_t)))
        else:  # mode 1
            v = torch.where(p0 >= 0.9, torch.minimum(c3, bw_t),
                torch.where(p0 >= 0.8,  torch.minimum(c5, bw_t),
                torch.where(p0 >= 0.6,  torch.minimum(c8, bw_t), bw_t)))
        slot0_t = v                                              # [B, 1]

    # ---- Slot k>=1: base_topks with optional ADAPTIVE_ALL shrinkage ----
    slot_topks = base_topks.clone()
    if _ADAPTIVE_ALL:
        pk = torch.exp(glp_t)                                    # [B, K]
        if _ADAPTIVE_ALL == 2:
            slot_topks = torch.where(
                pk >= 0.9, _const_t(1, device).expand_as(base_topks),
                torch.where(
                    pk >= 0.7, _const_t(2, device).expand_as(base_topks),
                    torch.where(
                        pk >= 0.5, _const_t(3, device).expand_as(base_topks),
                        base_topks,
                    ),
                ),
            )
        else:
            slot_topks = torch.where(
                pk >= 0.95, torch.minimum(_const_t(2, device), base_topks),
                torch.where(
                    pk >= 0.85, torch.minimum(_const_t(3, device), base_topks),
                    base_topks,
                ),
            )

    # Override slot 0 across batch.
    slot_topks = slot_topks.clone()
    slot_topks[:, 0] = slot0_t[:, 0]

    # give_up early-stop along K axis, broadcast per batch.
    if K > 1:
        give_up_trigger = (rp_t == give_up_class) & (base_topks == 0)  # [B, K]
        k_ge_1_mask = (torch.arange(K, device=device) > 0).unsqueeze(0).expand_as(give_up_trigger)
        give_up_trigger_k_ge_1 = give_up_trigger & k_ge_1_mask
        triggered_at_or_after = torch.cummax(
            give_up_trigger_k_ge_1.long(), dim=1
        ).values.bool()
        slot_topks = torch.where(
            triggered_at_or_after, torch.zeros_like(slot_topks), slot_topks,
        )

    return slot_topks


def compute_slot_topks_bfs_gpu(
    glp_all_t: torch.Tensor,   # [N, K] BFS greedy log_probs (gpu, f32)
    rp_all_t: torch.Tensor,    # [N, K] BFS rank predictions (gpu, long)
    rank_slot_topk_t: torch.Tensor,  # [rank_classes] (gpu, long)
    rank_classes: int,         # noqa: ARG001 (kept for signature parity)
) -> torch.Tensor:
    """GPU equivalent of :func:`compute_slot_topks_bfs`.

    Returns [N, K] long tensor on the same device.
    """
    base_arr = rank_slot_topk_t[rp_all_t.long()]  # [N, K]
    if not _ADAPTIVE_ALL:
        return base_arr

    pk = torch.exp(glp_all_t)
    device = glp_all_t.device
    if _ADAPTIVE_ALL == 2:
        out = torch.where(
            pk >= 0.9, _const_t(1, device),
            torch.where(
                pk >= 0.7, torch.minimum(_const_t(2, device), base_arr),
                torch.where(
                    pk >= 0.5, torch.minimum(_const_t(3, device), base_arr),
                    base_arr,
                ),
            ),
        )
    else:
        out = torch.where(
            pk >= 0.95, torch.minimum(_const_t(2, device), base_arr),
            torch.where(
                pk >= 0.85, torch.minimum(_const_t(3, device), base_arr),
                base_arr,
            ),
        )
    return out


def compute_slot_topks_block1(
    glp_np: np.ndarray,        # [K] block-1 greedy log_probs
    rp_np: np.ndarray,         # [K] block-1 rank predictions
    rank_slot_topk,            # tuple [rank_classes]
    beam_width: int,
    rank_classes: int,
) -> np.ndarray:
    """Compute per-slot topk count for block-1, with ADAPTIVE_SLOT0/ALL applied.

    When _ADAPTIVE_SLOT0 == _ADAPTIVE_ALL == 0, output is equivalent to the
    fixed lookup `rank_slot_topk[rp_np[k]]` for k>0 and `beam_width` for k==0
    (with give_up early-stop for rank_slot_topk[give_up]==0).
    """
    K = len(glp_np)
    give_up_class = rank_classes - 1
    slot_topks_np = np.zeros(K, dtype=np.int64)

    _slot0_bw = beam_width
    if _ADAPTIVE_SLOT0:
        p0 = float(np.exp(glp_np[0])) if len(glp_np) > 0 else 0.5
        if _ADAPTIVE_SLOT0 == 2:
            if p0 >= 0.95: _slot0_bw = min(2, beam_width)
            elif p0 >= 0.85: _slot0_bw = min(4, beam_width)
            elif p0 >= 0.6: _slot0_bw = min(7, beam_width)
        elif _ADAPTIVE_SLOT0 == 3:
            if p0 >= 0.95: _slot0_bw = 1
            elif p0 >= 0.9: _slot0_bw = min(2, beam_width)
            elif p0 >= 0.8: _slot0_bw = min(4, beam_width)
            elif p0 >= 0.6: _slot0_bw = min(7, beam_width)
        elif _ADAPTIVE_SLOT0 == 4:
            if p0 >= 0.9: _slot0_bw = 1
            elif p0 >= 0.8: _slot0_bw = min(3, beam_width)
            elif p0 >= 0.6: _slot0_bw = min(6, beam_width)
        else:  # 1 = conservative
            if p0 >= 0.9: _slot0_bw = min(3, beam_width)
            elif p0 >= 0.8: _slot0_bw = min(5, beam_width)
            elif p0 >= 0.6: _slot0_bw = min(8, beam_width)

    for k in range(K):
        rk = int(rp_np[k])
        if k == 0:
            slot_topks_np[k] = _slot0_bw
        else:
            base_topk = rank_slot_topk[rk]
            if _ADAPTIVE_ALL:
                pk = float(np.exp(glp_np[k])) if k < len(glp_np) else 0.5
                if _ADAPTIVE_ALL == 2:
                    if pk >= 0.9: slot_topks_np[k] = 1
                    elif pk >= 0.7: slot_topks_np[k] = 2
                    elif pk >= 0.5: slot_topks_np[k] = 3
                    else: slot_topks_np[k] = base_topk
                else:
                    if pk >= 0.95: slot_topks_np[k] = min(2, base_topk)
                    elif pk >= 0.85: slot_topks_np[k] = min(3, base_topk)
                    else: slot_topks_np[k] = base_topk
                if rk == give_up_class and base_topk == 0:
                    break
                continue
            if rk == give_up_class and base_topk == 0:
                break
            slot_topks_np[k] = base_topk
    return slot_topks_np


def compute_slot_topks_bfs(
    glp_all_np: np.ndarray,    # [N, K] BFS greedy log_probs
    rp_all_np: np.ndarray,     # [N, K] BFS rank predictions
    rank_slot_topk,            # tuple [rank_classes]
    rank_classes: int,         # noqa: ARG001 (kept for signature parity)
) -> np.ndarray:
    """Compute per-(leaf, slot) topk count for BFS, with ADAPTIVE_ALL applied.

    Returns [N, K] i64 array.  Equivalent to `rank_slot_topk[rp_all_np]` when
    ADAPTIVE_ALL == 0.
    """
    rank_slot_topk_arr = np.array(rank_slot_topk, dtype=np.int64)
    slot_topks_per_rank = rank_slot_topk_arr[rp_all_np].copy()
    if _ADAPTIVE_ALL:
        _pk_np = np.exp(glp_all_np)
        base_arr = slot_topks_per_rank
        if _ADAPTIVE_ALL == 2:
            cut_1 = _pk_np >= 0.9
            cut_2 = (_pk_np >= 0.7) & (_pk_np < 0.9)
            cut_3 = (_pk_np >= 0.5) & (_pk_np < 0.7)
            slot_topks_per_rank = np.where(cut_1, 1, slot_topks_per_rank)
            slot_topks_per_rank = np.where(
                cut_2, np.minimum(2, base_arr), slot_topks_per_rank,
            )
            slot_topks_per_rank = np.where(
                cut_3, np.minimum(3, base_arr), slot_topks_per_rank,
            )
        else:
            cut_a = _pk_np >= 0.95
            cut_b = (_pk_np >= 0.85) & (_pk_np < 0.95)
            slot_topks_per_rank = np.where(
                cut_a, np.minimum(2, base_arr), slot_topks_per_rank,
            )
            slot_topks_per_rank = np.where(
                cut_b, np.minimum(3, base_arr), slot_topks_per_rank,
            )
    return slot_topks_per_rank


# Fine-grained tree-builder profile state.  Filled when
# SPECBLOCK_TREE_PROFILE=1.  Drained by the worker (after the existing
# [SBSprof] print) at iter % 50 == 0 and reset.
_BUILDTREE_PROFILE_STATE: Dict[str, float] = {
    "n": 0,
    "block1": 0.0,
    "plan_step": 0.0,
    "b2_fwd": 0.0,
    "b2_bfs": 0.0,
    "finalize": 0.0,
}

def _walk_rank_slots(
    rank_preds: torch.Tensor,        # [N, K]
    rank_classes: int,
    rank_to_factor: torch.Tensor,    # [rank_classes]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch port of ``_walk_rank_slots_compiled`` (specblock.py L60-86).

    For each leaf, find:
      * ``M`` — the first slot index where the rank is *not* class 0 (with
        +1 if not give_up, else stop at this slot).
      * ``branch_factor`` — branching factor implied by the first
        non-class-0 rank.
      * ``give_up`` — whether the chain hit the give-up class first.
    """
    N, K = rank_preds.shape
    give_up_class = rank_classes - 1

    not_class0 = rank_preds != 0
    has_non_class0 = not_class0.any(dim=1)
    first_non_class0 = not_class0.float().argmax(dim=1)

    first_rank = rank_preds.gather(1, first_non_class0.unsqueeze(1).long()).squeeze(1)
    first_is_give_up = first_rank == give_up_class

    M = torch.where(
        ~has_non_class0,
        torch.full_like(first_non_class0, K),
        torch.where(first_is_give_up, first_non_class0, first_non_class0 + 1),
    )
    branch_factors = torch.where(
        has_non_class0 & ~first_is_give_up,
        rank_to_factor[first_rank.clamp(0, rank_classes - 1)],
        torch.zeros(N, device=rank_preds.device, dtype=torch.long),
    )
    give_up = has_non_class0 & first_is_give_up
    return M, branch_factors, give_up


def _map_draft_to_target(
    draft_ids: torch.Tensor,
    d2t: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply the draft->target vocab offset map (specblock.py L921-925)."""
    if d2t is None:
        return draft_ids
    return draft_ids + d2t[draft_ids]


def _ensure_buffers(
    max_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate the six numpy buffers used by the BFS builder.

    Mirrors specblock.py ``_ensure_tree_buffers`` (small helper, no caching
    here since the standalone module is meant for one-shot use)."""
    tn_tokens = np.empty(max_nodes, dtype=np.int64)
    tn_parents = np.empty(max_nodes, dtype=np.int64)
    tn_lps = np.empty(max_nodes, dtype=np.float64)
    tn_ranks = np.empty(max_nodes, dtype=np.int64)
    tn_blocks = np.empty(max_nodes, dtype=np.int64)
    tn_slots = np.empty(max_nodes, dtype=np.int64)
    return tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots


def _vectorized_greedy_chain(
    n_nodes: int,
    N: int,
    K: int,
    pend_node_indices: np.ndarray,
    pend_cum_lps: np.ndarray,
    gt_all_np: np.ndarray,            # [N, K] greedy target ids
    glp_all_np: np.ndarray,           # [N, K] greedy log-probs
    rp_all_np: np.ndarray,            # [N, K] rank predictions
    pend_depth: int,
    tn_tokens: np.ndarray,
    tn_parents: np.ndarray,
    tn_lps: np.ndarray,
    tn_ranks: np.ndarray,
    tn_blocks: np.ndarray,
    tn_slots: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Append N*K greedy-chain nodes (one chain of length K per leaf).

    Port of specblock.py ``_vectorized_greedy_chain`` (L274-355) without
    the M-truncation branch (we leave that simplification for Phase 1.3
    MVP — every BFS leaf gets the full K-slot chain).  Returns the new
    ``n_nodes`` and a [N, K] map from (leaf, slot) to absolute tree index.
    """
    NK = N * K
    ts = n_nodes
    ar_K = np.arange(K, dtype=np.int64)
    ar_NK = np.arange(NK, dtype=np.int64)

    tn_tokens[ts:ts + NK] = gt_all_np.reshape(-1)
    tn_ranks[ts:ts + NK] = rp_all_np.reshape(-1)
    tn_blocks[ts:ts + NK] = pend_depth
    tn_slots[ts:ts + NK] = np.tile(ar_K, N)

    leaf_idx = ar_NK // K
    slot_idx = ar_NK - leaf_idx * K
    pend_parents_np = np.asarray(pend_node_indices, dtype=np.int64)
    tn_parents[ts:ts + NK] = np.where(
        slot_idx == 0,
        pend_parents_np[leaf_idx],
        ts + ar_NK - 1,
    )

    pend_lps_np = np.asarray(pend_cum_lps, dtype=np.float64)
    cum_glp = np.cumsum(glp_all_np.reshape(N, K), axis=1).reshape(-1)
    tn_lps[ts:ts + NK] = cum_glp + pend_lps_np[leaf_idx]

    greedy_map = np.arange(ts, ts + NK).reshape(N, K)
    return ts + NK, greedy_map


def _prune_tree_np(
    n_nodes: int,
    tn_tokens: np.ndarray,
    tn_parents: np.ndarray,
    tn_lps: np.ndarray,
    tn_ranks: np.ndarray,
    tn_blocks: np.ndarray,
    tn_slots: np.ndarray,
    budget: int,
) -> int:
    """Greedy prune to ``budget`` nodes by cumulative log-prob.

    Port of specblock.py ``_prune_tree_np`` (L1110-1158).  Each surviving
    chain keeps every ancestor.  Compacts buffers in-place; returns the
    new ``n_nodes``.
    """
    if n_nodes <= budget:
        return n_nodes

    sorted_indices = np.argsort(-tn_lps[:n_nodes])  # descending
    kept: set = set()
    for idx in sorted_indices:
        if len(kept) >= budget:
            break
        chain: List[int] = []
        cur = int(idx)
        while cur >= 0 and cur not in kept:
            chain.append(cur)
            cur = int(tn_parents[cur])
        if len(kept) + len(chain) <= budget:
            kept.update(chain)

    kept_sorted = sorted(kept)
    old_to_new: Dict[int, int] = {old: new for new, old in enumerate(kept_sorted)}
    new_n = len(kept_sorted)

    new_tokens = tn_tokens[kept_sorted]
    new_parents = np.array(
        [old_to_new.get(int(tn_parents[old]), -1) for old in kept_sorted],
        dtype=np.int64,
    )
    new_lps = tn_lps[kept_sorted]
    new_ranks = tn_ranks[kept_sorted]
    new_blocks = tn_blocks[kept_sorted]
    new_slots = tn_slots[kept_sorted]

    tn_tokens[:new_n] = new_tokens
    tn_parents[:new_n] = new_parents
    tn_lps[:new_n] = new_lps
    tn_ranks[:new_n] = new_ranks
    tn_blocks[:new_n] = new_blocks
    tn_slots[:new_n] = new_slots
    return new_n


# ============================================================
#  Topology (depth, attention mask, retrieve_index)
# ============================================================

def _build_tree_topology(
    n_nodes: int,
    tn_tokens: np.ndarray,
    tn_parents: np.ndarray,
    sample_token: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Numpy-only port of ``build_tree_buffers_triton`` (specblock.py L357-416).

    The reference implementation uses two Triton kernels for parent/depth
    walking and retrieve-index construction.  This module deliberately
    stays kernel-free (TODO: re-introduce Triton kernels behind a flag
    once SGLang integration is wired up).

    Returns (draft_tokens, tree_mask, position_ids, retrieve_indices).
    """
    N = n_nodes
    Np1 = N + 1

    # Parent buffer: root sits at index 0; tree node i (0..N-1) lives at
    # absolute index i+1; root parent is itself.
    parent_np = np.empty(Np1, dtype=np.int64)
    parent_np[0] = 0
    parent_np[1:Np1] = np.where(tn_parents[:N] >= 0, tn_parents[:N] + 1, 0)

    # Depth + ancestor mask via per-node parent walking.
    depth_np = np.zeros(Np1, dtype=np.int64)
    mask_np = np.zeros((Np1, Np1), dtype=np.float32)
    for node in range(Np1):
        cur = node
        d = 0
        # Cap walks at Np1 to avoid pathological loops on bad parent data.
        for _ in range(Np1):
            mask_np[node, cur] = 1.0
            if cur == 0:
                break
            cur = int(parent_np[cur])
            d += 1
        depth_np[node] = d

    # Leaves = nodes that are no one's parent (skip root self-edge).
    is_parent = np.zeros(Np1, dtype=bool)
    if N > 0:
        np.add.at(is_parent, parent_np[1:Np1], True)
    leaves = np.where(~is_parent)[0]
    num_leaves = int(leaves.shape[0])
    max_depth = int(depth_np.max()) if Np1 > 0 else 0

    ri_np = np.full((max(num_leaves, 1), max_depth + 1), -1, dtype=np.int64)
    for li, leaf in enumerate(leaves):
        cur = int(leaf)
        d = int(depth_np[cur])
        for step in range(d + 1):
            ri_np[li, d - step] = cur
            cur = int(parent_np[cur])

    # Sort retrieve_indices to match Eagle/specblock convention (lex order
    # with -1 treated as +inf so shorter paths sort last).
    sentinel = Np1 + 5
    if num_leaves > 0:
        keys = np.where(ri_np >= 0, ri_np, sentinel)
        order = np.lexsort(keys.T[::-1])
        ri_np = ri_np[order]

    # Final tensors.
    token_ids = np.empty(Np1, dtype=np.int64)
    token_ids[0] = int(sample_token.item()) if torch.is_tensor(sample_token) else int(sample_token)
    if N > 0:
        token_ids[1:] = tn_tokens[:N]
    draft_tokens = torch.from_numpy(token_ids).to(device)
    tree_mask = torch.from_numpy(mask_np).to(device)
    position_ids = torch.from_numpy(depth_np).to(device)
    retrieve_indices = torch.from_numpy(ri_np if num_leaves > 0 else ri_np[:0]).to(device)
    return draft_tokens, tree_mask, position_ids, retrieve_indices


# ============================================================
#  Block 1 expansion (specblock.py L2780-3199)
# ============================================================

def _expand_block1(
    block1_logits: torch.Tensor,        # [1, K, V_draft]
    block1_rank_logits: torch.Tensor,   # [1, K, rank_classes]
    block1_hidden: torch.Tensor,        # [1, K, H]
    ttt_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    K: int,
    beam_width: int,
    rank_slot_topk: Tuple[int, ...],
    rank_classes: int,
    rank_to_factor: torch.Tensor,
    d2t: Optional[torch.Tensor],
    tn_tokens: np.ndarray,
    tn_parents: np.ndarray,
    tn_lps: np.ndarray,
    tn_ranks: np.ndarray,
    tn_blocks: np.ndarray,
    tn_slots: np.ndarray,
) -> Tuple[
    int,                                # n_nodes after block 1
    Dict[str, np.ndarray],              # pending leaves -> next BFS depth
    torch.Tensor,                       # batch_ttt_mask [N_pend, K]
    torch.Tensor,                       # pend_hidden  [N_pend, 1, H]
    torch.Tensor,                       # pend_input_ids [N_pend, 1]
]:
    """Block-1 BFS seed: build the K-slot greedy chain, attach
    rank-driven sibling alternatives, and emit a "pending" descriptor for
    the next BFS depth (block 2+).
    """
    device = block1_logits.device
    last_p = block1_logits[0]                          # [K, V]
    rank_preds = block1_rank_logits[0].argmax(dim=-1)  # [K]
    log_probs = F.log_softmax(last_p.float(), dim=-1)  # [K, V]

    M_t, bf_t, gu_t = _walk_rank_slots(
        rank_preds.unsqueeze(0), rank_classes, rank_to_factor,
    )

    greedy_tokens = last_p.argmax(dim=-1)              # [K]
    greedy_target = _map_draft_to_target(greedy_tokens, d2t)
    greedy_lps = log_probs.gather(1, greedy_tokens.unsqueeze(-1)).squeeze(-1)

    max_topk = max(beam_width, max(rank_slot_topk))
    all_top = torch.topk(last_p, max_topk, dim=-1)
    all_top_target = _map_draft_to_target(all_top.indices, d2t)
    all_top_lps = log_probs.gather(1, all_top.indices)

    # Single sync to numpy.
    gt_np = greedy_target.cpu().numpy()
    glp_np = greedy_lps.detach().cpu().numpy()
    rp_np = rank_preds.cpu().numpy()
    att_np = all_top_target.cpu().numpy()
    atl_np = all_top_lps.detach().cpu().numpy()
    give_up = bool(gu_t[0].item())

    # ---- Greedy chain (K nodes, vectorized) ----
    ar_K = np.arange(K, dtype=np.int64)
    tn_tokens[:K] = gt_np
    tn_parents[:K] = ar_K - 1                          # [-1, 0, 1, ..., K-2]
    tn_lps[:K] = np.cumsum(glp_np)
    tn_ranks[:K] = rp_np
    tn_blocks[:K] = 0
    tn_slots[:K] = ar_K
    n_nodes = K
    greedy_cum_lp = float(tn_lps[K - 1])

    # Per-slot topk: rank-table lookup with optional ADAPTIVE_SLOT0/ALL
    # shrinkage (1:1 port of HF specblock.py:3107-3163).  Both default
    # OFF so non-adaptive runs hit the original rank-table path.
    slot_topks_np = compute_slot_topks_block1(
        glp_np, rp_np, rank_slot_topk, beam_width, rank_classes,
    )

    active_slots = np.nonzero(slot_topks_np > 1)[0]
    n_alt_per_slot = slot_topks_np[active_slots] - 1
    total_alts = int(n_alt_per_slot.sum())

    # ---- Bulk-add alternatives ----
    slot_expanded = np.empty(0, dtype=np.int64)
    alt_within = np.empty(0, dtype=np.int64)
    if total_alts > 0:
        s = n_nodes
        slot_expanded = np.repeat(active_slots, n_alt_per_slot)
        alt_within = np.concatenate(
            [np.arange(1, n + 1) for n in n_alt_per_slot],
        )
        parent_nodes = slot_expanded - 1
        parent_lps = np.where(
            slot_expanded > 0, tn_lps[np.maximum(parent_nodes, 0)], 0.0,
        )
        parent_lps[slot_expanded == 0] = 0.0
        tn_tokens[s:s + total_alts] = att_np[slot_expanded, alt_within]
        tn_parents[s:s + total_alts] = parent_nodes
        tn_lps[s:s + total_alts] = parent_lps + atl_np[slot_expanded, alt_within]
        tn_ranks[s:s + total_alts] = rp_np[slot_expanded]
        tn_blocks[s:s + total_alts] = 0
        tn_slots[s:s + total_alts] = slot_expanded
        n_nodes += total_alts

    # ---- Build pending arrays (g(slot) per active + alternatives + g(K-1)) ----
    pend_g_indices = active_slots.copy()
    pend_g_lps = tn_lps[active_slots]
    pend_g_tokens = gt_np[active_slots]
    pend_g_hidden_slots = active_slots.copy()
    pend_g_ttt_valid = active_slots + 1

    if total_alts > 0:
        alt_tree_indices = np.arange(n_nodes - total_alts, n_nodes)
        pend_a_lps = tn_lps[alt_tree_indices]
        pend_a_tokens = att_np[slot_expanded, alt_within]
        pend_a_hidden_slots = slot_expanded
        pend_a_ttt_valid = slot_expanded + 1
    else:
        alt_tree_indices = np.empty(0, dtype=np.int64)
        pend_a_lps = np.empty(0, dtype=np.float64)
        pend_a_tokens = np.empty(0, dtype=np.int64)
        pend_a_hidden_slots = np.empty(0, dtype=np.int64)
        pend_a_ttt_valid = np.empty(0, dtype=np.int64)

    pend_indices_np = np.concatenate([pend_g_indices, alt_tree_indices])
    pend_lps_np = np.concatenate([pend_g_lps, pend_a_lps])
    pend_tokens_np = np.concatenate([pend_g_tokens, pend_a_tokens])
    pend_hidden_slots_np = np.concatenate([pend_g_hidden_slots, pend_a_hidden_slots])
    pend_ttt_valid_np = np.concatenate([pend_g_ttt_valid, pend_a_ttt_valid])

    # Hitchhike g(K-1) when not give_up and not already pending.
    if not give_up and (K - 1) not in pend_indices_np.tolist():
        pend_indices_np = np.append(pend_indices_np, K - 1)
        pend_lps_np = np.append(pend_lps_np, greedy_cum_lp)
        pend_tokens_np = np.append(pend_tokens_np, int(gt_np[K - 1]))
        pend_hidden_slots_np = np.append(pend_hidden_slots_np, K - 1)
        pend_ttt_valid_np = np.append(pend_ttt_valid_np, K)

    pend_dict = {
        "node_indices": pend_indices_np,
        "cum_lps": pend_lps_np,
        "input_tokens": pend_tokens_np,
        "hidden_slots": pend_hidden_slots_np,
        "ttt_valid": pend_ttt_valid_np,
    }

    if pend_indices_np.shape[0] == 0:
        empty_mask = torch.zeros(0, K, dtype=torch.bool, device=device)
        empty_hidden = block1_hidden[:, :0, :]
        empty_ids = torch.empty(0, 1, dtype=torch.long, device=device)
        return n_nodes, pend_dict, empty_mask, empty_hidden, empty_ids

    N_pend = int(pend_indices_np.shape[0])
    pend_hidden_idx_t = torch.from_numpy(pend_hidden_slots_np).to(device)
    pend_input_ids = torch.from_numpy(pend_tokens_np).to(device).unsqueeze(1)
    pend_hidden = block1_hidden[0, pend_hidden_idx_t, :].unsqueeze(1)  # [N_pend,1,H]

    ar_K_np = np.arange(K, dtype=np.int64)
    mask_np = ar_K_np[None, :] < pend_ttt_valid_np[:, None]
    batch_ttt_mask = torch.from_numpy(mask_np).to(device)

    return n_nodes, pend_dict, batch_ttt_mask, pend_hidden, pend_input_ids


# ============================================================
#  BFS expansion (single iter for block 2+)
# ============================================================

def _bfs_expand(
    n_nodes: int,
    pend_dict: Dict[str, np.ndarray],
    block_logits: torch.Tensor,        # [N_pend, K, V_draft]
    block_rank_logits: torch.Tensor,   # [N_pend, K, rank_classes]
    K: int,
    rank_slot_topk: Tuple[int, ...],
    rank_classes: int,
    d2t: Optional[torch.Tensor],
    pend_depth: int,
    tn_tokens: np.ndarray,
    tn_parents: np.ndarray,
    tn_lps: np.ndarray,
    tn_ranks: np.ndarray,
    tn_blocks: np.ndarray,
    tn_slots: np.ndarray,
) -> int:
    """Single BFS expansion step (port of inner loop in specblock.py L3309-3472).

    Greedy-chain every pending leaf (N_pend * K nodes), then attach
    rank-driven sibling alternatives.  Returns the updated ``n_nodes``.
    Pending construction for the *next* BFS depth is intentionally
    omitted: the Phase 1.3 MVP only supports max_blocks <= 2, so block 2
    is the leaf-most expansion.
    """
    N = int(pend_dict["node_indices"].shape[0])
    if N == 0:
        return n_nodes

    pend_node_indices = pend_dict["node_indices"]
    pend_cum_lps = pend_dict["cum_lps"]

    max_factor = max(rank_slot_topk)  # widest topk we ever need

    # Decode per-leaf greedy/topk in one fused-ish pass.
    # NOTE(perf TODO): the reference uses ``_bfs_gpu_ops_fused`` here with
    # ``@torch.compile`` to land argmax+topk+lse in a single CUDA graph.
    # Phase 1.3 keeps this simple — re-introduce when SGLang wires CUDA
    # graph capture for the draft model.
    all_rank_preds = block_rank_logits.argmax(dim=-1)                   # [N, K]
    all_greedy_tokens = block_logits.argmax(dim=-1)                     # [N, K]
    all_greedy_target = _map_draft_to_target(all_greedy_tokens, d2t)    # [N, K]
    log_probs = F.log_softmax(block_logits.float(), dim=-1)             # [N, K, V]
    all_greedy_lps = log_probs.gather(2, all_greedy_tokens.unsqueeze(-1)).squeeze(-1)

    top = torch.topk(block_logits, max_factor, dim=-1)
    top_target_all = _map_draft_to_target(top.indices, d2t)             # [N, K, max_factor]
    top_lps_all = log_probs.gather(2, top.indices)                      # [N, K, max_factor]

    # Bulk transfer to CPU.
    rp_all_np = all_rank_preds.cpu().numpy()
    gt_all_np = all_greedy_target.cpu().numpy()
    glp_all_np = all_greedy_lps.detach().cpu().numpy()
    tt_all_np = top_target_all.cpu().numpy()
    tl_all_np = top_lps_all.detach().cpu().numpy()

    # ---- Greedy chains (N*K nodes) ----
    tree_start = n_nodes
    n_nodes, _greedy_map = _vectorized_greedy_chain(
        n_nodes, N, K, pend_node_indices, pend_cum_lps,
        gt_all_np, glp_all_np, rp_all_np, pend_depth,
        tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots,
    )

    # ---- Multi-slot branching alternatives ----
    give_up_class = rank_classes - 1
    rank_slot_topk_arr = np.array(rank_slot_topk, dtype=np.int64)
    bfs_give_up_has_topk = rank_slot_topk_arr[give_up_class] > 0
    arange_K_np = np.arange(K, dtype=np.int64)

    if bfs_give_up_has_topk:
        stop_at_np = np.full(N, K, dtype=np.int64)
    else:
        give_up_mask_np = rp_all_np == give_up_class
        has_give_up_np = give_up_mask_np.any(axis=1)
        stop_at_np = np.where(has_give_up_np, give_up_mask_np.argmax(axis=1), K)

    slot_in_range = arange_K_np[None, :] < stop_at_np[:, None]
    slot_topks_per_rank = compute_slot_topks_bfs(
        glp_all_np, rp_all_np, rank_slot_topk, rank_classes,
    )

    slot_active = (slot_topks_per_rank > 1) & slot_in_range
    pair_leaves, pair_slots = np.nonzero(slot_active)
    n_pairs = len(pair_leaves)
    if n_pairs == 0:
        return n_nodes

    pair_topks = slot_topks_per_rank[pair_leaves, pair_slots]
    n_alt_per_pair = np.maximum(pair_topks - 1, 0)
    total_alts = int(n_alt_per_pair.sum())
    if total_alts == 0:
        return n_nodes

    s = n_nodes
    pair_exp = np.repeat(np.arange(n_pairs, dtype=np.int64), n_alt_per_pair)
    cum_alts = np.cumsum(n_alt_per_pair)
    prev_cum = np.empty(n_pairs, dtype=np.int64)
    prev_cum[0] = 0
    if n_pairs > 1:
        prev_cum[1:] = cum_alts[:-1]
    flat_idx = np.arange(total_alts, dtype=np.int64)
    alt_within = flat_idx - prev_cum[pair_exp] + 1

    leaf_exp = pair_leaves[pair_exp]
    slot_exp = pair_slots[pair_exp]
    pend_parent_np = np.asarray(pend_node_indices, dtype=np.int64)
    pend_lp_np = np.asarray(pend_cum_lps, dtype=np.float64)

    base_for_leaf = tree_start + leaf_exp * K
    bp_idx = np.where(slot_exp > 0, base_for_leaf + slot_exp - 1, pend_parent_np[leaf_exp])
    bp_lp = np.where(
        slot_exp > 0,
        tn_lps[np.maximum(base_for_leaf + slot_exp - 1, 0)],
        pend_lp_np[leaf_exp],
    )

    tn_tokens[s:s + total_alts] = tt_all_np[leaf_exp, slot_exp, alt_within]
    tn_parents[s:s + total_alts] = bp_idx
    tn_lps[s:s + total_alts] = bp_lp + tl_all_np[leaf_exp, slot_exp, alt_within]
    tn_ranks[s:s + total_alts] = rp_all_np[leaf_exp, slot_exp]
    tn_blocks[s:s + total_alts] = pend_depth
    tn_slots[s:s + total_alts] = slot_exp
    n_nodes += total_alts
    return n_nodes


# ============================================================
#  Public entrypoint
# ============================================================

def build_tree(
    draft_model,
    block1_logits: torch.Tensor,                                   # [1, K, V_draft]
    block1_rank_logits: torch.Tensor,                              # [1, K, rank_classes]
    block1_hidden: torch.Tensor,                                   # [1, K, H]
    block1_ttt_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    initial_input_id: torch.Tensor,                                # [1, 1] last verified token
    cross_kv_cache: List[List],                                    # per-layer [k, v, count, max_len]
    position_id: int,
    K: int = 4,
    max_blocks: int = 2,
    beam_width: int = 10,
    total_tokens: int = 90,
    rank_classes: int = 4,
    rank_to_factor: Optional[Tuple[int, ...]] = None,
    rank_slot_topk: Optional[Tuple[int, ...]] = None,
    d2t: Optional[torch.Tensor] = None,
    scratch: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    """Build a SpecBlock-Shift draft tree via BFS (greedy / batch=1).

    Parameters mirror specblock.py ``_build_tree_from_block1`` (L2705).
    The first decoder block (block 1) must already have been forwarded by
    the caller, with its outputs passed in as
    (``block1_logits``, ``block1_rank_logits``, ``block1_hidden``,
    ``block1_ttt_kv``).  The builder then drives up to
    ``max_blocks - 1`` BFS expansions through ``draft_model.forward_with_cache``.

    Returns a dict with the schema documented in the task spec:
      * ``draft_token``    — [N+1] long tensor (root token + tree nodes)
      * ``parents``        — [N+1] long tensor (root parent = -1)
      * ``depth``          — [N+1] long tensor (root depth = 0)
      * ``cum_log_prob``   — [N+1] float tensor (root cum_lp = 0)
      * ``retrieve_index`` — [num_leaves, max_depth+1] long tensor
      * ``tree_mask``      — [N+1, N+1] float tensor (1.0 = ancestor)
      * ``position_ids``   — alias of ``depth`` (kept for SGLang convenience)
    """
    assert block1_logits.shape[0] == 1, "Phase 1.3 MVP supports batch=1 only."
    if rank_to_factor is None:
        rank_to_factor = _RANK_TO_FACTOR_DEFAULT
    if rank_slot_topk is None:
        rank_slot_topk = _RANK_SLOT_TOPK_DEFAULT
    if d2t is None:
        d2t = getattr(draft_model, "d2t", None)

    device = block1_logits.device

    # Production path: GPU end-to-end via Triton kernels.  CPU smoke-test
    # path (device.type != 'cuda') falls through to numpy reference below.
    if device.type == "cuda":
        from sglang.srt.speculative.specblock_tree_kernels import build_tree_gpu

        if scratch is None:
            scratch = getattr(build_tree, "_default_scratch", None)
            if scratch is None:
                scratch = {}
                build_tree._default_scratch = scratch
        return build_tree_gpu(
            draft_model=draft_model,
            block1_logits=block1_logits,
            block1_rank_logits=block1_rank_logits,
            block1_hidden=block1_hidden,
            block1_ttt_kv=block1_ttt_kv,
            initial_input_id=initial_input_id,
            cross_kv_cache=cross_kv_cache,
            position_id=position_id,
            K=K,
            max_blocks=max_blocks,
            beam_width=beam_width,
            total_tokens=total_tokens,
            rank_classes=rank_classes,
            rank_slot_topk=tuple(rank_slot_topk),
            rank_to_factor=tuple(rank_to_factor),
            d2t=d2t,
            scratch=scratch,
        )
    rank_to_factor_t = torch.tensor(
        rank_to_factor, dtype=torch.long, device=device,
    )

    # Generous upper bound on the max number of nodes we might emit
    # before pruning.  Mirrors specblock.py L2746-2761.
    max_topk = max(beam_width, max(rank_slot_topk))
    max_block1_nodes = K + K * (max_topk - 1) + 1
    max_block2_nodes = max_block1_nodes * K * max_topk
    max_nodes = max(total_tokens + 200, max_block1_nodes + max_block2_nodes + 100)
    tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots = _ensure_buffers(max_nodes)

    sample_token = initial_input_id[:, -1]  # [1]

    # ============================================================
    #  Block 1 expansion
    # ============================================================
    n_nodes, pend_dict, batch_ttt_mask, pend_hidden, pend_input_ids = _expand_block1(
        block1_logits, block1_rank_logits, block1_hidden, block1_ttt_kv,
        K=K,
        beam_width=beam_width,
        rank_slot_topk=rank_slot_topk,
        rank_classes=rank_classes,
        rank_to_factor=rank_to_factor_t,
        d2t=d2t,
        tn_tokens=tn_tokens, tn_parents=tn_parents, tn_lps=tn_lps,
        tn_ranks=tn_ranks, tn_blocks=tn_blocks, tn_slots=tn_slots,
    )

    # ============================================================
    #  BFS loop (block 2 .. max_blocks)
    # ============================================================
    pend_depth = 1
    num_layers = len(block1_ttt_kv)
    batch_ttt_kv = block1_ttt_kv
    while pend_depth < max_blocks and pend_dict["node_indices"].shape[0] > 0:
        N_pend = int(pend_dict["node_indices"].shape[0])

        # Expand paged cross_kv_cache + ttt_kv to N_pend.
        # cross_kv_cache layout (paged): per-layer [kv_pool, layer_id,
        # count, cross_loc, new_cross_loc=None], with cross_loc shared
        # across layers within a request.
        kv_pool = cross_kv_cache[0][0]
        stored_count = int(cross_kv_cache[0][2])
        # Current block-0's final K slots belong to the path-specific TTT
        # region.  Keep only older committed positions in full cross context.
        count = max(stored_count - K, 0)
        cross_loc_1d = cross_kv_cache[0][3]
        if count > 0:
            loc_expanded = (
                cross_loc_1d[:count].unsqueeze(0).expand(N_pend, -1).contiguous()
            )
        else:
            sentinel_device = (
                cross_loc_1d.device if cross_loc_1d is not None
                else block1_logits.device
            )
            loc_expanded = torch.zeros(
                N_pend, 1, dtype=torch.int64, device=sentinel_device,
            )
        batch_cross_cache = [
            [kv_pool, L, count, loc_expanded, None]
            for L in range(num_layers)
        ]

        batched_ttt_kv = [
            (kv[0].expand(N_pend, -1, -1, -1), kv[1].expand(N_pend, -1, -1, -1))
            for kv in batch_ttt_kv
        ]

        # Combined cross+ttt mask for the draft attention (saves per-layer cat).
        if count > 0:
            cross_ones = batch_ttt_mask.new_ones(N_pend, count)
            full_kv_mask = torch.cat([cross_ones, batch_ttt_mask], dim=1)
        else:
            full_kv_mask = batch_ttt_mask

        block_logits, block_rank_logits, block_draft_hidden, _new_ttt_kv = (
            draft_model.forward_with_cache(
                hidden=pend_hidden,
                input_ids=pend_input_ids,
                cache=batch_cross_cache,
                position_id=position_id,
                use_draft_condition=True,
                ttt_cache=batched_ttt_kv,
                ttt_mask=batch_ttt_mask,
                update_cross_cache=False,
                full_kv_mask=full_kv_mask,
            )
        )

        n_nodes = _bfs_expand(
            n_nodes, pend_dict, block_logits, block_rank_logits,
            K=K,
            rank_slot_topk=rank_slot_topk,
            rank_classes=rank_classes,
            d2t=d2t,
            pend_depth=pend_depth,
            tn_tokens=tn_tokens, tn_parents=tn_parents, tn_lps=tn_lps,
            tn_ranks=tn_ranks, tn_blocks=tn_blocks, tn_slots=tn_slots,
        )

        # MVP: only one BFS iter (max_blocks=2).  Higher max_blocks would
        # require constructing the next pending descriptor here; the
        # reference does that via the n_pairs/total_alts/n_hitch trio.
        # See specblock.py L3482-3744.
        pend_dict = {"node_indices": np.empty(0, dtype=np.int64)}
        pend_depth += 1

    # ============================================================
    #  Prune to ``total_tokens`` budget
    # ============================================================
    if n_nodes > total_tokens:
        n_nodes = _prune_tree_np(
            n_nodes, tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots,
            total_tokens,
        )

    # ============================================================
    #  Topology (parents / depth / mask / retrieve_index)
    # ============================================================
    draft_tokens, tree_mask, position_ids, retrieve_indices = _build_tree_topology(
        n_nodes, tn_tokens, tn_parents, sample_token, device,
    )

    # Build the dict expected by the spec.
    Np1 = n_nodes + 1
    parents_full = np.empty(Np1, dtype=np.int64)
    parents_full[0] = -1
    if n_nodes > 0:
        parents_full[1:Np1] = tn_parents[:n_nodes]  # already in [-1..n_nodes-1]
    cum_lp_full = np.empty(Np1, dtype=np.float32)
    cum_lp_full[0] = 0.0
    if n_nodes > 0:
        cum_lp_full[1:Np1] = tn_lps[:n_nodes].astype(np.float32)

    parents_t = torch.from_numpy(parents_full).to(device)
    cum_lp_t = torch.from_numpy(cum_lp_full).to(device)

    return {
        "draft_token": draft_tokens,
        "parents": parents_t,
        "depth": position_ids,
        "cum_log_prob": cum_lp_t,
        "retrieve_index": retrieve_indices,
        "tree_mask": tree_mask,
        "position_ids": position_ids,
    }


# ============================================================
#  Multi-batch entrypoint: build_tree_batched
# ============================================================
#
#  Multi-req batched build_tree.  Used by SGLang worker.draft() in bs>1
#  scenarios.  Internally:
#
#  Phase 1 (per-req, CPU numpy + tiny GPU):
#      Iterate B reqs, each calls _expand_block1 (~tens of microseconds
#      of numpy + 1 small GPU sync).  Per-req tn_* buffers are kept
#      separate so each req's tree state is independent.
#
#  Phase 2 (BATCHED block-2 GPU forward) -- the bottleneck we batch:
#      Stack pend_hidden / ttt_kv / cross_kv across all reqs into
#      [total_pend, ...] tensors (with cross_count padded to per-batch
#      max_cross_count + per-row mask), do ONE forward_with_cache call,
#      then split the output back per-req.
#
#  Phase 3 (per-req, CPU numpy):
#      Per-req _bfs_expand using its own block_logits slice.
#
#  Phase 4 (per-req, CPU numpy):
#      Per-req _prune_tree_np + _build_tree_topology + finalize dict.
#
#  Phases 1/3/4 are inherently per-req (each tree's BFS pending count
#  N_pend is rank-class driven and varies per req).  Only Phase 2 is the
#  GPU bottleneck that benefits from batching.
# ============================================================

def build_tree_batched(
    draft_model,
    block1_logits_b: torch.Tensor,                      # [B, K, V_draft]
    block1_rank_logits_b: torch.Tensor,                 # [B, K, rank_classes]
    block1_hidden_b: torch.Tensor,                      # [B, K, H]
    block1_ttt_kv_b: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    initial_input_id_b: torch.Tensor,                   # [B, 1]
    cross_kv_cache_b: List[List[List]],                 # B × num_layers × [k_buf, v_buf, count]
    cross_position_b: List[int],                        # length B
    K: int = 4,
    max_blocks: int = 2,
    beam_width: int = 10,
    total_tokens: int = 90,
    rank_classes: int = 4,
    rank_to_factor: Optional[Tuple[int, ...]] = None,
    rank_slot_topk: Optional[Tuple[int, ...]] = None,
    d2t: Optional[torch.Tensor] = None,
    need_retrieve_index: bool = True,
    block1_top_indices_b: Optional[torch.Tensor] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Batched version of :func:`build_tree`.

    Returns a list of B tree dicts (same schema as ``build_tree``).
    """
    B = int(block1_logits_b.shape[0])
    if rank_to_factor is None:
        rank_to_factor = _RANK_TO_FACTOR_DEFAULT
    if rank_slot_topk is None:
        rank_slot_topk = _RANK_SLOT_TOPK_DEFAULT
    if d2t is None:
        d2t = getattr(draft_model, "d2t", None)

    device = block1_logits_b.device

    # Production path: GPU end-to-end via Triton kernels (no numpy
    # round-trip).  Includes the B=1 fast path via ``build_tree_gpu``
    # and B>1 batched path via ``build_tree_batched_gpu``.  Both share
    # _alloc_per_req_scratch for buffer caching.
    #
    # CPU smoke-test path (device.type != 'cuda'): falls through to the
    # numpy phase chain below.  This is dev-only — production scheduler
    # always runs on cuda.
    if device.type == "cuda":
        from sglang.srt.speculative.specblock_tree_kernels import (
            build_tree_gpu,
            build_tree_batched_gpu,
        )
        if B == 1:
            scratch = getattr(build_tree_batched, "_default_scratch", None)
            if scratch is None:
                scratch = {}
                build_tree_batched._default_scratch = scratch
            tree = build_tree_gpu(
                draft_model=draft_model,
                block1_logits=block1_logits_b,
                block1_rank_logits=block1_rank_logits_b,
                block1_hidden=block1_hidden_b,
                block1_ttt_kv=block1_ttt_kv_b[0],
                initial_input_id=initial_input_id_b,
                cross_kv_cache=cross_kv_cache_b[0],
                position_id=cross_position_b[0],
                K=K, max_blocks=max_blocks, beam_width=beam_width,
                total_tokens=total_tokens, rank_classes=rank_classes,
                rank_slot_topk=tuple(rank_slot_topk),
                rank_to_factor=tuple(rank_to_factor),
                d2t=d2t, scratch=scratch,
            )
            return [tree]

        scratch_b = getattr(build_tree_batched, "_default_scratch_b", None)
        if scratch_b is None:
            scratch_b = []
            build_tree_batched._default_scratch_b = scratch_b
        return build_tree_batched_gpu(
            draft_model=draft_model,
            block1_logits_b=block1_logits_b,
            block1_rank_logits_b=block1_rank_logits_b,
            block1_hidden_b=block1_hidden_b,
            block1_ttt_kv_b=block1_ttt_kv_b,
            initial_input_id_b=initial_input_id_b,
            cross_kv_cache_b=cross_kv_cache_b,
            cross_position_b=cross_position_b,
            K=K, max_blocks=max_blocks, beam_width=beam_width,
            total_tokens=total_tokens, rank_classes=rank_classes,
            rank_slot_topk=tuple(rank_slot_topk),
            rank_to_factor=tuple(rank_to_factor),
            d2t=d2t, scratch_b=scratch_b,
            need_retrieve_index=need_retrieve_index,
            block1_top_indices_b=block1_top_indices_b,
        )

    # ---- CPU dev path (numpy reference; not used in production) ----
    # Single-req CPU: delegate to build_tree (preserve bit-equivalent behavior).
    if B == 1:
        tree = build_tree(
            draft_model=draft_model,
            block1_logits=block1_logits_b,
            block1_rank_logits=block1_rank_logits_b,
            block1_hidden=block1_hidden_b,
            block1_ttt_kv=block1_ttt_kv_b[0],
            initial_input_id=initial_input_id_b,
            cross_kv_cache=cross_kv_cache_b[0],
            position_id=cross_position_b[0],
            K=K, max_blocks=max_blocks, beam_width=beam_width,
            total_tokens=total_tokens, rank_classes=rank_classes,
            rank_to_factor=rank_to_factor, rank_slot_topk=rank_slot_topk,
            d2t=d2t,
        )
        return [tree]

    rank_to_factor_t = torch.tensor(
        rank_to_factor, dtype=torch.long, device=device,
    )
    max_topk = max(beam_width, max(rank_slot_topk))
    max_block1_nodes = K + K * (max_topk - 1) + 1
    max_block2_nodes = max_block1_nodes * K * max_topk
    max_nodes = max(total_tokens + 200, max_block1_nodes + max_block2_nodes + 100)

    # ---- Optional fine-grained tree-builder profile (env
    # SPECBLOCK_TREE_PROFILE=1).  Phase-level mirrors HF's draft
    # profile (block1 / b2_fwd / b2_bfs / finalize).  Drained / printed
    # by the worker at the same checkpoint as [SBSprof].
    _tp_on = os.environ.get("SPECBLOCK_TREE_PROFILE", "0") == "1"
    if _tp_on:
        import time as _time
        _tp_state = _BUILDTREE_PROFILE_STATE
        _tp_state["n"] += 1
        torch.cuda.synchronize()
        _t_p0 = _time.perf_counter()

    # ============================================================
    # Phase 1: per-req block-1 expand
    # ============================================================
    per_req: List[Dict] = []
    for b in range(B):
        bufs = _ensure_buffers(max_nodes)
        n_nodes_b, pend_dict_b, ttt_mask_b, pend_hidden_b, pend_input_ids_b = (
            _expand_block1(
                block1_logits_b[b:b+1], block1_rank_logits_b[b:b+1],
                block1_hidden_b[b:b+1], block1_ttt_kv_b[b],
                K=K, beam_width=beam_width, rank_slot_topk=rank_slot_topk,
                rank_classes=rank_classes, rank_to_factor=rank_to_factor_t,
                d2t=d2t,
                tn_tokens=bufs[0], tn_parents=bufs[1], tn_lps=bufs[2],
                tn_ranks=bufs[3], tn_blocks=bufs[4], tn_slots=bufs[5],
            )
        )
        per_req.append({
            "bufs": bufs,
            "n_nodes": n_nodes_b,
            "pend_dict": pend_dict_b,
            "ttt_mask": ttt_mask_b,
            "pend_hidden": pend_hidden_b,
            "pend_input_ids": pend_input_ids_b,
            "block_logits": None,        # filled in Phase 2
            "block_rank_logits": None,
        })

    if _tp_on:
        torch.cuda.synchronize()
        _t_p1 = _time.perf_counter()
        _tp_state["block1"] += _t_p1 - _t_p0

    # ============================================================
    # Phase 2: BATCHED block-2 forward across all reqs
    # ============================================================
    if max_blocks >= 2:
        _batched_block2_forward(
            per_req, draft_model, cross_kv_cache_b, block1_ttt_kv_b,
            cross_position_b, K=K, device=device,
        )

    if _tp_on:
        torch.cuda.synchronize()
        _t_p2 = _time.perf_counter()
        _tp_state["b2_fwd"] += _t_p2 - _t_p1

    # ============================================================
    # Phase 3 + 4: per-req BFS expand + prune + topology
    # ============================================================
    trees: List[Dict[str, torch.Tensor]] = []
    for b in range(B):
        st = per_req[b]
        n_nodes = st["n_nodes"]
        bufs = st["bufs"]
        tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots = bufs

        if (
            max_blocks >= 2
            and st["block_logits"] is not None
            and st["pend_dict"]["node_indices"].shape[0] > 0
        ):
            n_nodes = _bfs_expand(
                n_nodes, st["pend_dict"],
                st["block_logits"], st["block_rank_logits"],
                K=K, rank_slot_topk=rank_slot_topk,
                rank_classes=rank_classes, d2t=d2t, pend_depth=1,
                tn_tokens=tn_tokens, tn_parents=tn_parents, tn_lps=tn_lps,
                tn_ranks=tn_ranks, tn_blocks=tn_blocks, tn_slots=tn_slots,
            )

        if n_nodes > total_tokens:
            n_nodes = _prune_tree_np(
                n_nodes, tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots,
                total_tokens,
            )

        sample_token = initial_input_id_b[b, -1].view(1)  # [1]
        draft_tokens, tree_mask, position_ids, retrieve_indices = _build_tree_topology(
            n_nodes, tn_tokens, tn_parents, sample_token, device,
        )

        Np1 = n_nodes + 1
        parents_full = np.empty(Np1, dtype=np.int64)
        parents_full[0] = -1
        if n_nodes > 0:
            parents_full[1:Np1] = tn_parents[:n_nodes]
        cum_lp_full = np.empty(Np1, dtype=np.float32)
        cum_lp_full[0] = 0.0
        if n_nodes > 0:
            cum_lp_full[1:Np1] = tn_lps[:n_nodes].astype(np.float32)
        parents_t = torch.from_numpy(parents_full).to(device)
        cum_lp_t = torch.from_numpy(cum_lp_full).to(device)

        trees.append({
            "draft_token": draft_tokens,
            "parents": parents_t,
            "depth": position_ids,
            "cum_log_prob": cum_lp_t,
            "retrieve_index": retrieve_indices,
            "tree_mask": tree_mask,
            "position_ids": position_ids,
        })

    if _tp_on:
        torch.cuda.synchronize()
        _t_p3 = _time.perf_counter()
        _tp_state["finalize"] += _t_p3 - _t_p2

    return trees


N_PEND_MAX_DEFAULT = 64  # fixed-shape pad budget for build_tree_batched
                         # (Stage B graph: must be >= max(actual N_pend)
                         # across the run to keep cache_key tp stable;
                         # SpecBlock's worst-case N_pend ~ 1 + K*topk = 41)
                          # phase 2 forward.  Stage A (pad to fixed shape) +
                          # Stage B (cuda graph capture) prerequisite.


def _batched_block2_forward(
    per_req: List[Dict],
    draft_model,
    cross_kv_cache_b: List[List[List]],
    block1_ttt_kv_b: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    cross_position_b: List[int],
    K: int,
    device: torch.device,
    n_pend_max: Optional[int] = None,
):
    """Phase 2 of build_tree_batched: batched block-2 GPU forward.

    Stacks B reqs' [N_pend_b, K, H] pend hidden + ttt_kv + cross_kv into
    a single [B * N_pend_max, ...] forward call.  Each req's actual
    pending leaves ``N_pend_b`` (variable) gets padded with zeros + a
    ttt_mask of all-False so its row contributes nothing to softmax.
    cross_kv per-req different counts -> pad to max(cross_count_b) +
    [B * N_pend_max, max_cross_count] valid mask.

    The fixed-shape (B * N_pend_max) batched forward is the prerequisite
    for cuda graph capture (Stage B).  Stage A alone has marginal effect
    -- pad rows waste GPU compute -- so this padding only pays off after
    Stage B replays via cuda graph.

    Mutates ``per_req`` in-place: sets ``block_logits`` and
    ``block_rank_logits`` slices for each req with N_pend > 0.
    """
    B = len(per_req)
    N_pend_per_req = [
        int(p["pend_dict"]["node_indices"].shape[0]) for p in per_req
    ]
    _b2_deep = os.environ.get("SPECBLOCK_DEEP_PROFILE", "0") == "1"
    if _b2_deep:
        import time as _b2_time
        torch.cuda.synchronize()
        _b2_t0 = _b2_time.perf_counter()

    total_pend_real = sum(N_pend_per_req)
    if total_pend_real == 0:
        return

    if n_pend_max is None:
        n_pend_max = max(N_pend_per_req)
    else:
        n_pend_max = max(n_pend_max, max(N_pend_per_req))
    total_pend = B * n_pend_max

    # Stack per-req tensors into [B * n_pend_max, ...] (fixed shape).
    # Each req's [N_pend_b, ...] is padded to [n_pend_max, ...] with zeros
    # for hidden/input_ids and all-False for ttt_mask.  Pad rows produce
    # garbage outputs but never get read (Phase 3 _bfs_expand only iterates
    # the valid first N_pend_b rows of each req's slice).
    template_h = per_req[0]["pend_hidden"]    # [Nb, 1, H]
    template_id = per_req[0]["pend_input_ids"]  # [Nb, 1]
    template_m = per_req[0]["ttt_mask"]        # [Nb, K]
    H_dim = template_h.shape[-1]

    pend_hidden_list, pend_input_ids_list, ttt_mask_list = [], [], []
    for b in range(B):
        N = N_pend_per_req[b]
        if N == n_pend_max:
            ph = per_req[b]["pend_hidden"]
            pid = per_req[b]["pend_input_ids"]
            tm = per_req[b]["ttt_mask"]
        else:
            # Pad to n_pend_max
            ph = template_h.new_zeros(n_pend_max, 1, H_dim)
            pid = template_id.new_zeros(n_pend_max, 1)
            tm = template_m.new_zeros(n_pend_max, K)
            if N > 0:
                ph[:N] = per_req[b]["pend_hidden"]
                pid[:N] = per_req[b]["pend_input_ids"]
                tm[:N] = per_req[b]["ttt_mask"]
        pend_hidden_list.append(ph)
        pend_input_ids_list.append(pid)
        ttt_mask_list.append(tm)
    pend_hidden_b = torch.cat(pend_hidden_list, dim=0)        # [B*Nm, 1, H]
    pend_input_ids_b = torch.cat(pend_input_ids_list, dim=0)  # [B*Nm, 1]
    ttt_mask_b = torch.cat(ttt_mask_list, dim=0)              # [B*Nm, K]

    # Stack ttt_kv per layer.  Each per-req entry is [1, n_kv, K, D];
    # expand to [n_pend_max, n_kv, K, D] before cat (uniform B*Nm).
    num_layers = len(block1_ttt_kv_b[0])
    batched_ttt_kv = []
    for L in range(num_layers):
        k_parts, v_parts = [], []
        for b in range(B):
            k = block1_ttt_kv_b[b][L][0]
            v = block1_ttt_kv_b[b][L][1]
            k_parts.append(k.expand(n_pend_max, -1, -1, -1))
            v_parts.append(v.expand(n_pend_max, -1, -1, -1))
        batched_ttt_kv.append(
            (torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0))
        )

    # Stack cross_kv via paged gather (no dense per-req K/V copy).
    # Each req's paged cache layer has the layout:
    #   [kv_pool, layer_id, count, cross_loc, new_cross_loc=None]
    # All layers within a req share the same cross_loc tensor.
    # Each paged cache ends with the current block-0 K slots.  Those slots
    # must remain path-specific TTT context, not globally visible cross KV.
    cross_counts = [
        max(int(cross_kv_cache_b[b][0][2]) - K, 0) for b in range(B)
    ]
    max_cross_count = max(cross_counts)
    kv_pool = cross_kv_cache_b[0][0][0]
    assert kv_pool is not None and hasattr(kv_pool, "set_kv"), (
        "build_tree_batched expects paged cross_kv_cache_b "
        "(layer[0][0] should be a SpecBlockKVPool); "
        f"got {type(kv_pool).__name__}"
    )

    if max_cross_count > 0:
        # Build [B, max_cross_count] padded indices.  Pad slots are
        # the pool's sentinel slot 0 (always-zero K/V); attention masks
        # them out via full_kv_mask_b's cross_mask portion below.
        loc_per_b = []
        for b in range(B):
            cnt = cross_counts[b]
            cross_loc_b_t = cross_kv_cache_b[b][0][3]  # layer-shared
            real_loc = cross_loc_b_t[:cnt].to(device, dtype=torch.int64)
            if cnt == max_cross_count:
                loc_per_b.append(real_loc)
            else:
                pad = torch.zeros(
                    max_cross_count - cnt, dtype=torch.int64, device=device,
                )
                loc_b_padded = torch.cat([real_loc, pad], dim=0)
                loc_per_b.append(loc_b_padded)
        loc_padded_b = torch.stack(loc_per_b, dim=0)  # [B, max_cross_count]
        # Expand to [B*n_pend_max, max_cross_count] for the pend forward.
        loc_expanded = (
            loc_padded_b.unsqueeze(1)
            .expand(-1, n_pend_max, -1)
            .reshape(B * n_pend_max, max_cross_count)
            .contiguous()
        )
        # Per-row cross_counts after the n_pend_max expand: each req's
        # row is replicated n_pend_max times so its real count repeats.
        # flashinfer (Stage D) needs these to strip pad-region indices.
        per_row_cross_counts: List[int] = []
        for b in range(B):
            per_row_cross_counts.extend([cross_counts[b]] * n_pend_max)
        batched_cross_cache = []
        for L in range(num_layers):
            batched_cross_cache.append([
                kv_pool, L, max_cross_count, loc_expanded, None,
                per_row_cross_counts,
            ])

        # cross_mask: per-req valid bits, padded to [B*n_pend_max, max_cross_count].
        cross_mask_parts = []
        for b in range(B):
            cnt = cross_counts[b]
            m = torch.zeros(
                n_pend_max, max_cross_count,
                dtype=torch.bool, device=device,
            )
            m[:, :cnt] = True
            cross_mask_parts.append(m)
        cross_mask = torch.cat(cross_mask_parts, dim=0)
        full_kv_mask_b = torch.cat([cross_mask, ttt_mask_b], dim=1)
    else:
        # No cross context -- attention will skip the cross read path.
        # Build a single-sentinel paged cache so the [B*n_pend_max, 1]
        # gather still returns a valid tensor view.
        sentinel_loc = torch.zeros(
            B * n_pend_max, 1, dtype=torch.int64, device=device,
        )
        per_row_cross_counts = [0] * (B * n_pend_max)
        batched_cross_cache = [
            [kv_pool, L, 0, sentinel_loc, None, per_row_cross_counts]
            for L in range(num_layers)
        ]
        full_kv_mask_b = ttt_mask_b

    # Position id: per-row [B*n_pend_max] tensor (each req's base repeated
    # n_pend_max times -- pad rows reuse the same base since they're masked).
    pos_per_pend = []
    for b in range(B):
        pos_per_pend.extend([cross_position_b[b]] * n_pend_max)
    pos_id_tensor = torch.tensor(pos_per_pend, dtype=torch.long, device=device)

    if _b2_deep:
        torch.cuda.synchronize()
        _b2_t_pad = _b2_time.perf_counter()

    # ---- Stage D: pre-plan the flashinfer paged cross attention ONCE
    # for this decode step.  Cross_loc / per_row_cross_counts are layer-
    # invariant, so the plan is reused by every layer's run_layer call
    # inside attention.forward.  Per-req segmentation: collapse the
    # B*n_pend_max-row plan to a B-row plan (each req's pend rows share
    # cross_loc), reducing plan() metadata work ~30x at bs=4 n_pend=30.
    if (
        max_cross_count > 0
        and getattr(kv_pool, "flashinfer_cross", None) is not None
        and os.environ.get("SPECBLOCK_FLASHINFER", "0") == "1"
    ):
        # All SpecBlock attention layers share scale = 1/sqrt(head_dim).
        scale = 1.0 / (kv_pool.head_dim ** 0.5)
        # Per-req plan: tp = B segments, each with K * n_pend_max queries.
        kv_pool.flashinfer_cross.plan_step(
            cross_loc_padded=loc_padded_b,           # [B, max_cross]
            cross_counts=cross_counts,               # length B
            K=K,
            scale=scale,
            device=pend_hidden_b.device,
            dtype=pend_hidden_b.dtype,
            qo_len_per_segment=K * n_pend_max,       # group all pend rows per req
        )

    if _b2_deep:
        torch.cuda.synchronize()
        _b2_t_plan = _b2_time.perf_counter()

    # This batched builder is no longer a production worker route.  Keep its
    # direct forward for offline diagnostics only; the worker's sole draft
    # execution path is SpecBlockDraftCudaGraphRunner.
    block_logits_b, block_rank_logits_b, _hidden, _ttt = (
        draft_model.forward_with_cache(
            hidden=pend_hidden_b,
            input_ids=pend_input_ids_b,
            cache=batched_cross_cache,
            position_id=pos_id_tensor,
            use_draft_condition=True,
            ttt_cache=batched_ttt_kv,
            ttt_mask=ttt_mask_b,
            update_cross_cache=False,
            full_kv_mask=full_kv_mask_b,
        )
    )

    if _b2_deep:
        torch.cuda.synchronize()
        _b2_t_fwd = _b2_time.perf_counter()

    # Split outputs back per-req.  Each req owns rows [b*n_pend_max,
    # (b+1)*n_pend_max), but only the first N_pend_b are valid; pad rows
    # are discarded (never read by Phase 3 _bfs_expand).
    for b in range(B):
        N = N_pend_per_req[b]
        if N == 0:
            continue
        start = b * n_pend_max
        per_req[b]["block_logits"] = block_logits_b[start:start + N]
        per_req[b]["block_rank_logits"] = block_rank_logits_b[start:start + N]

    if _b2_deep:
        torch.cuda.synchronize()
        _b2_t_split = _b2_time.perf_counter()
        st = _batched_block2_forward.__dict__.setdefault(
            "_DEEP_STATE",
            {"n": 0, "pad_setup": 0.0, "plan": 0.0, "fwd": 0.0, "split": 0.0},
        )
        st["n"] += 1
        st["pad_setup"] += _b2_t_pad - _b2_t0
        st["plan"] += _b2_t_plan - _b2_t_pad
        st["fwd"] += _b2_t_fwd - _b2_t_plan
        st["split"] += _b2_t_split - _b2_t_fwd
        if st["n"] % 50 == 0:
            n = st["n"]
            import logging as _lg
            _lg.getLogger(__name__).info(
                "[DEEPprof:b2_fwd] n=%d pad_setup=%.2fms plan=%.2fms "
                "fwd=%.2fms split=%.2fms total=%.2fms",
                n,
                st["pad_setup"]/n*1000, st["plan"]/n*1000,
                st["fwd"]/n*1000, st["split"]/n*1000,
                (st["pad_setup"]+st["plan"]+st["fwd"]+st["split"])/n*1000,
            )


# ============================================================
#  Unit test (CPU, dummy model)
# ============================================================

if __name__ == "__main__":
    """Smoke test on CPU with a dummy draft model.

    The dummy model fakes the ``forward_with_cache`` interface by
    sampling fresh random logits each call; we only check shape +
    structural invariants of the returned tree.
    """

    torch.manual_seed(0)
    np.random.seed(0)

    K = 4
    H = 16
    V_draft = 32
    rank_classes = 4
    num_layers = 2
    beam_width = 5
    total_tokens = 30
    max_blocks = 2
    position_id = 0
    device = torch.device("cpu")

    class _DummyDraftModel:
        """Returns random logits + zero hidden + dummy ttt_kv each call."""

        def __init__(self):
            self.d2t = torch.zeros(V_draft, dtype=torch.long)

        def forward_with_cache(self, hidden, input_ids, cache, position_id,
                               use_draft_condition, ttt_cache, ttt_mask,
                               update_cross_cache, full_kv_mask=None):
            B = hidden.shape[0]
            logits = torch.randn(B, K, V_draft)
            rank_logits = torch.randn(B, K, rank_classes)
            draft_hidden = torch.zeros(B, K, H)
            new_ttt_kv = [
                (torch.zeros(B, 1, K, H), torch.zeros(B, 1, K, H))
                for _ in range(num_layers)
            ]
            return logits, rank_logits, draft_hidden, new_ttt_kv

    draft_model = _DummyDraftModel()
    block1_logits = torch.randn(1, K, V_draft)
    block1_rank_logits = torch.randn(1, K, rank_classes)
    block1_hidden = torch.randn(1, K, H)
    block1_ttt_kv = [
        (torch.zeros(1, 1, K, H), torch.zeros(1, 1, K, H))
        for _ in range(num_layers)
    ]
    initial_input_id = torch.tensor([[7]], dtype=torch.long)
    cross_kv_cache = [
        [None, None, 0, 64] for _ in range(num_layers)
    ]

    out = build_tree(
        draft_model=draft_model,
        block1_logits=block1_logits,
        block1_rank_logits=block1_rank_logits,
        block1_hidden=block1_hidden,
        block1_ttt_kv=block1_ttt_kv,
        initial_input_id=initial_input_id,
        cross_kv_cache=cross_kv_cache,
        position_id=position_id,
        K=K,
        max_blocks=max_blocks,
        beam_width=beam_width,
        total_tokens=total_tokens,
        rank_classes=rank_classes,
    )

    Np1 = out["draft_token"].shape[0]
    print(f"[unit-test] tree size N+1 = {Np1} (budget = {total_tokens + 1})")

    # ---- Invariants ----
    assert out["parents"].shape == (Np1,), f"parents shape {out['parents'].shape}"
    assert int(out["parents"][0].item()) == -1, "root parent must be -1"
    assert out["depth"].shape == (Np1,), f"depth shape {out['depth'].shape}"
    assert int(out["depth"][0].item()) == 0, "root depth must be 0"
    assert out["draft_token"].shape == (Np1,), f"draft_token shape {out['draft_token'].shape}"
    assert out["cum_log_prob"].shape == (Np1,), f"cum_log_prob shape {out['cum_log_prob'].shape}"
    assert float(out["cum_log_prob"][0].item()) == 0.0, "root cum_log_prob must be 0"

    ri = out["retrieve_index"]
    assert ri.dim() == 2, f"retrieve_index dim {ri.dim()}"
    assert ri.shape[1] >= 1, "retrieve_index must have at least 1 column"
    # Each retrieve row starts at root (0) since paths are root->leaf order.
    if ri.shape[0] > 0:
        # First column should be root index (0) for all valid paths.
        first_col = ri[:, 0].tolist()
        assert all(v in (-1, 0) for v in first_col), (
            f"retrieve_index[:,0] should be 0 (root) or -1 (padding); got {first_col}"
        )

    # Tree mask sanity: diagonal is all 1 (every node sees itself); root col
    # is always 1 (every node has root as ancestor).
    tm = out["tree_mask"]
    assert tm.shape == (Np1, Np1), f"tree_mask shape {tm.shape}"
    assert torch.all(tm.diagonal() == 1.0), "tree_mask diagonal must be 1.0"
    assert torch.all(tm[:, 0] == 1.0), "tree_mask root column must be 1.0"

    # Total nodes within budget.
    assert Np1 <= total_tokens + 1, f"tree exceeds budget: {Np1} > {total_tokens + 1}"

    # Parents reference valid indices.
    for i in range(1, Np1):
        p = int(out["parents"][i].item())
        assert -1 <= p < Np1 - 1, f"parents[{i}] = {p} out of range"

    print("[unit-test] all invariants PASS")
