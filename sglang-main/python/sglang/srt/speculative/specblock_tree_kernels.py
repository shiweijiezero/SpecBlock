"""GPU/Triton tree-construction kernels for SpecBlock-Shift.

Ports the offline reference (`benchmarks_hf/algorithms/tree_build_triton.py`
+ `tree_finalize_triton.py` + the two topology kernels in
`benchmarks_hf/algorithms/specblock.py` L147-272) into the SGLang
speculative module so the SGLang worker has zero numpy/Python BFS path.

Public surface:
  * MAX_TREE_DEPTH                          — depth bound for parent walks
  * `_bfs_gpu_ops_fused`                    — GPU helper: rank walk + topk + log probs
  * `triton_build_block1`                   — kernel 1 (block-1 mega) wrapper
  * `triton_build_bfs`                      — kernel 2 (BFS sizing + scatter) wrappers
  * `finalize_tree_gpu`                     — prune + topology + retrieve_index
  * `_tree_depth_mask_kernel`,
    `_tree_retrieve_kernel`                  — exposed for `finalize_tree_gpu`

All kernels operate on GPU tensors end-to-end. The single non-GPU path is
the lex-sort of `retrieve_indices` rows (reference does the same), which
is a tiny (<= a few hundred ints) `cpu().tolist()` round-trip.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.utils import is_metax_c500


# =============================================================================
#   Constants
# =============================================================================

MAX_TREE_DEPTH = 20  # K * max_blocks upper bound; ample for K<=8, max_blocks<=2.


# =============================================================================
#   GPU helper: fused rank walk + topk + log-probs (block 2+ inputs)
# =============================================================================

def _bfs_gpu_ops_fused(
    block_logits: torch.Tensor,          # [N, K, V_draft] bf16/fp16/fp32
    block_rank_logits: torch.Tensor,     # [N, K, rank_classes]
    d2t_offsets: Optional[torch.Tensor], # [V_draft] long  (or None)
    max_factor: int,
    rank_classes: int,
    rank_to_factor: torch.Tensor,        # [rank_classes] long
):
    """Super-fused BFS GPU ops: rank walk + argmax + topk + log probs in one pass.

    Mirrors `benchmarks_hf/algorithms/specblock.py::_bfs_gpu_ops_fused`.
    All outputs stay on GPU. logsumexp is computed once per row in fp32
    over the bf16/fp16 logits to keep mem traffic low.
    """
    # === Rank predictions + rank walk ===
    all_rank_preds = block_rank_logits.argmax(dim=-1)  # [N, K]
    N, K = all_rank_preds.shape
    give_up_class = rank_classes - 1

    not_class0 = (all_rank_preds != 0)
    has_non_class0 = not_class0.any(dim=1)
    first_non_class0 = not_class0.float().argmax(dim=1)
    first_rank = all_rank_preds.gather(
        1, first_non_class0.unsqueeze(1).long()
    ).squeeze(1)
    first_is_give_up = first_rank == give_up_class
    M = torch.where(
        ~has_non_class0, K,
        torch.where(first_is_give_up, first_non_class0, first_non_class0 + 1),
    )
    bf = torch.where(
        has_non_class0 & ~first_is_give_up,
        rank_to_factor[first_rank.clamp(0, rank_classes - 1)],
        torch.zeros(N, device=all_rank_preds.device, dtype=torch.long),
    )
    give_up = has_non_class0 & first_is_give_up

    # === Fused topk over [N*K, V] — argmax is topk[:, 0] so we skip a separate argmax.
    V = block_logits.shape[-1]
    flat_logits = block_logits.view(N * K, V)
    top_vals, top_idx = torch.topk(flat_logits, max_factor, dim=-1)  # [N*K, max_factor]
    top_idx_nk = top_idx.view(N, K, max_factor)
    top_vals_nk = top_vals.view(N, K, max_factor)

    all_greedy_tokens = top_idx_nk[..., 0]
    if d2t_offsets is not None:
        all_greedy_target = all_greedy_tokens + d2t_offsets[all_greedy_tokens]
    else:
        all_greedy_target = all_greedy_tokens
    greedy_vals = top_vals_nk[..., 0]

    # === Log probs: lse over bf16 logits, greedy lp = greedy_val - lse.
    lse = torch.logsumexp(block_logits.float(), dim=-1)
    all_greedy_lps = greedy_vals.float() - lse

    if d2t_offsets is not None:
        top_target_flat = top_idx + d2t_offsets[top_idx]
    else:
        top_target_flat = top_idx
    flat_lse = lse.view(N * K, 1)
    top_lps_flat = top_vals.float() - flat_lse

    return (
        all_rank_preds, all_greedy_tokens, all_greedy_target, all_greedy_lps,
        M, bf, give_up,
        top_target_flat.view(N, K, max_factor),
        top_lps_flat.view(N, K, max_factor),
    )


# =============================================================================
#   Block-1 mega kernel  (port of reference tree_build_triton.py)
# =============================================================================

@triton.jit
def _block1_mega_kernel(
    rank_preds_ptr,            # [K] i64
    greedy_target_ptr,         # [K] i64
    greedy_lps_ptr,            # [K] f32
    all_top_target_ptr,        # [K, MAX_TOPK] i64
    all_top_lps_ptr,           # [K, MAX_TOPK] f32
    slot_topks_K_ptr,          # [K] i64 — per-slot topk count, ADAPTIVE-aware

    tree_tokens_ptr,           # [MAX_NODES] i64
    tree_parents_ptr,          # [MAX_NODES] i64
    tree_lps_ptr,              # [MAX_NODES] f32
    tree_ranks_ptr,            # [MAX_NODES] i64
    tree_blocks_ptr,           # [MAX_NODES] i64
    tree_slots_ptr,            # [MAX_NODES] i64
    pend_hidden_slots_ptr,     # [PEND_MAX] i64
    pend_input_ids_ptr,        # [PEND_MAX] i64
    pend_ttt_valid_ptr,        # [PEND_MAX] i64
    pend_node_indices_ptr,     # [PEND_MAX] i64
    pend_cum_lps_ptr,          # [PEND_MAX] f32
    sizes_ptr,                 # [4] i64: {n_nodes_b1, n_active, total_alts1, N_pend}

    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    RANK_CLASSES: tl.constexpr,
    GIVE_UP_CLASS: tl.constexpr,
):
    """Single-program block-1 branching + scatter + pending construction.

    Grid: (1,). K is tiny (=4-8) so full vectorization inside one program.
    Output layout on tree buffer:
      [0 : K]                       — greedy chain (slot 0..K-1)
      [K : K + total_alts1]         — alternatives (flattened by cumsum)
    Output layout on pending arrays:
      [0 : n_active]                         — active slots (slot_topks>1)
      [n_active : n_active + total_alts1]    — alternative entries
      [n_active + total_alts1]               — hitchhike g(K-1)  (if !give_up & !active[K-1])
    """
    k_vec = tl.arange(0, K)                                     # [K]
    rank_preds = tl.load(rank_preds_ptr + k_vec)                # [K] i64
    greedy_tgt = tl.load(greedy_target_ptr + k_vec)             # [K] i64
    greedy_lps = tl.load(greedy_lps_ptr + k_vec)                # [K] f32

    # ADAPTIVE-aware: caller pre-computes slot_topks_K combining
    # slot-0 beam (with optional ADAPTIVE_SLOT0 shrinkage) and slots>0
    # rank-table values (with optional ADAPTIVE_ALL shrinkage).
    slot_topks = tl.load(slot_topks_K_ptr + k_vec)              # [K] i64

    # Give-up flag: (rank_preds != 0).any() & first non-zero rank == GIVE_UP
    non_zero = (rank_preds != 0).to(tl.int32)                   # [K]
    csum_nz = tl.cumsum(non_zero, 0)                            # [K]
    is_first_nz = (csum_nz == 1) & (non_zero == 1)              # [K]
    any_nz = tl.sum(non_zero) > 0
    first_nz_rank = tl.sum(
        tl.where(is_first_nz, rank_preds, tl.zeros([K], dtype=tl.int64))
    )
    give_up = any_nz & (first_nz_rank == GIVE_UP_CLASS)

    n_alt_slot = tl.maximum(slot_topks - 1, 0)                  # [K]
    cum_alt_incl = tl.cumsum(n_alt_slot, 0)
    cum_alt_excl = cum_alt_incl - n_alt_slot
    total_alts1 = tl.sum(n_alt_slot)

    active_mask = slot_topks > 1
    active_mask_i64 = active_mask.to(tl.int64)
    active_rank_incl = tl.cumsum(active_mask_i64, 0)
    active_rank = active_rank_incl - active_mask_i64
    n_active = tl.sum(active_mask_i64)

    # Hitchhike g(K-1) iff !give_up AND slot K-1 not already in pend.
    last_active = tl.sum(
        tl.where(k_vec == (K - 1), active_mask_i64, tl.zeros_like(active_mask_i64))
    )
    hitch_add = ((~give_up) & (last_active == 0)).to(tl.int64)

    # Greedy chain [0..K-1]
    cum_greedy_lps = tl.cumsum(greedy_lps, 0)
    cum_greedy_lps_excl = cum_greedy_lps - greedy_lps
    parent_chain = k_vec.to(tl.int64) - 1

    tl.store(tree_tokens_ptr + k_vec, greedy_tgt)
    tl.store(tree_parents_ptr + k_vec, parent_chain)
    tl.store(tree_lps_ptr + k_vec, cum_greedy_lps)
    tl.store(tree_ranks_ptr + k_vec, rank_preds)
    tl.store(tree_blocks_ptr + k_vec, tl.zeros([K], dtype=tl.int64))
    tl.store(tree_slots_ptr + k_vec, k_vec.to(tl.int64))

    # Alternatives 2D tile
    J_PAD: tl.constexpr = 16
    K_ALT: tl.constexpr = K * J_PAD

    alt_flat = tl.arange(0, K_ALT)
    alt_k = alt_flat // J_PAD
    alt_j_idx = alt_flat - alt_k * J_PAD
    alt_j = alt_j_idx + 1

    match_k = alt_k[:, None] == k_vec[None, :]
    slot_topks_per_alt = tl.sum(
        match_k.to(tl.int64) * slot_topks[None, :], axis=1
    )
    alt_valid = (alt_j_idx < (MAX_TOPK - 1)) & (alt_j < slot_topks_per_alt)
    alt_valid_i64 = alt_valid.to(tl.int64)
    alt_rank_incl = tl.cumsum(alt_valid_i64, 0)
    alt_rank = alt_rank_incl - alt_valid_i64

    parent_lp_alt = tl.sum(
        match_k.to(tl.float32) * cum_greedy_lps_excl[None, :], axis=1
    )
    alt_off = alt_k * MAX_TOPK + alt_j
    alt_top_tgt = tl.load(all_top_target_ptr + alt_off, mask=alt_valid, other=0)
    alt_top_lps = tl.load(all_top_lps_ptr + alt_off, mask=alt_valid, other=0.0)
    alt_lp_node = parent_lp_alt + alt_top_lps

    alt_parent = tl.where(
        alt_k == 0,
        tl.full([K_ALT], -1, dtype=tl.int64),
        alt_k.to(tl.int64) - 1,
    )
    alt_rank_pred = tl.load(rank_preds_ptr + alt_k)

    tree_alt_pos = K + alt_rank
    tl.store(tree_tokens_ptr + tree_alt_pos, alt_top_tgt, mask=alt_valid)
    tl.store(tree_parents_ptr + tree_alt_pos, alt_parent, mask=alt_valid)
    tl.store(tree_lps_ptr + tree_alt_pos, alt_lp_node, mask=alt_valid)
    tl.store(tree_ranks_ptr + tree_alt_pos, alt_rank_pred, mask=alt_valid)
    tl.store(tree_blocks_ptr + tree_alt_pos,
             tl.zeros([K_ALT], dtype=tl.int64), mask=alt_valid)
    tl.store(tree_slots_ptr + tree_alt_pos, alt_k.to(tl.int64), mask=alt_valid)

    pend_alt_pos = n_active + alt_rank
    tl.store(pend_hidden_slots_ptr + pend_alt_pos, alt_k.to(tl.int64), mask=alt_valid)
    tl.store(pend_input_ids_ptr    + pend_alt_pos, alt_top_tgt, mask=alt_valid)
    tl.store(pend_ttt_valid_ptr    + pend_alt_pos, alt_k.to(tl.int64) + 1, mask=alt_valid)
    tl.store(pend_node_indices_ptr + pend_alt_pos, tree_alt_pos.to(tl.int64), mask=alt_valid)
    tl.store(pend_cum_lps_ptr      + pend_alt_pos, alt_lp_node, mask=alt_valid)

    # Active-slot pending entries [0..n_active-1]
    tl.store(pend_hidden_slots_ptr + active_rank, k_vec.to(tl.int64), mask=active_mask)
    tl.store(pend_input_ids_ptr    + active_rank, greedy_tgt,          mask=active_mask)
    tl.store(pend_ttt_valid_ptr    + active_rank, k_vec.to(tl.int64) + 1, mask=active_mask)
    tl.store(pend_node_indices_ptr + active_rank, k_vec.to(tl.int64),  mask=active_mask)
    tl.store(pend_cum_lps_ptr      + active_rank, cum_greedy_lps,      mask=active_mask)

    # Hitchhike g(K-1) at n_active + total_alts1
    hitch_pos = n_active + total_alts1
    hitch_mask = (hitch_add == 1)
    last_greedy_tgt = tl.sum(
        tl.where(k_vec == (K - 1), greedy_tgt, tl.zeros([K], dtype=tl.int64))
    )
    last_cum_lp = tl.sum(
        tl.where(k_vec == (K - 1), cum_greedy_lps, tl.zeros([K], dtype=tl.float32))
    )

    ones1 = tl.arange(0, 1)
    hitch_pos_vec = hitch_pos + ones1
    hitch_mask_vec = (ones1 == 0) & hitch_mask
    val_last_tgt = (ones1 * 0).to(tl.int64) + last_greedy_tgt
    val_last_lp  = (ones1 * 0).to(tl.float32) + last_cum_lp
    val_km1      = (ones1 * 0).to(tl.int64) + (K - 1)
    val_k        = (ones1 * 0).to(tl.int64) + K
    tl.store(pend_hidden_slots_ptr + hitch_pos_vec, val_km1,       mask=hitch_mask_vec)
    tl.store(pend_input_ids_ptr    + hitch_pos_vec, val_last_tgt,  mask=hitch_mask_vec)
    tl.store(pend_ttt_valid_ptr    + hitch_pos_vec, val_k,         mask=hitch_mask_vec)
    tl.store(pend_node_indices_ptr + hitch_pos_vec, val_km1,       mask=hitch_mask_vec)
    tl.store(pend_cum_lps_ptr      + hitch_pos_vec, val_last_lp,   mask=hitch_mask_vec)

    n_nodes_b1 = K + total_alts1
    n_pend = n_active + total_alts1 + hitch_add
    size_idx = tl.arange(0, 4)
    size_vals = tl.where(size_idx == 0, n_nodes_b1,
                tl.where(size_idx == 1, n_active,
                tl.where(size_idx == 2, total_alts1,
                         n_pend)))
    tl.store(sizes_ptr + size_idx, size_vals)


# =============================================================================
#   Block-1 mega kernel — BATCHED (grid=(B,), bidx-routed)
# =============================================================================

@triton.jit
def _block1_mega_kernel_batched(
    rank_preds_ptr,            # [B, K] i64
    greedy_target_ptr,         # [B, K] i64
    greedy_lps_ptr,            # [B, K] f32
    all_top_target_ptr,        # [B, K, MAX_TOPK] i64
    all_top_lps_ptr,           # [B, K, MAX_TOPK] f32
    slot_topks_K_ptr,          # [B, K] i64

    tree_tokens_ptr,           # [B, MAX_NODES] i64
    tree_parents_ptr,
    tree_lps_ptr,              # f32
    tree_ranks_ptr,
    tree_blocks_ptr,
    tree_slots_ptr,
    pend_hidden_slots_ptr,     # [B, PEND_MAX] i64
    pend_input_ids_ptr,
    pend_ttt_valid_ptr,
    pend_node_indices_ptr,
    pend_cum_lps_ptr,           # f32
    sizes_ptr,                  # [B, 4] i64

    STRIDE_NODES,
    STRIDE_PEND,
    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    RANK_CLASSES: tl.constexpr,
    GIVE_UP_CLASS: tl.constexpr,
):
    """grid=(B,) batched version of _block1_mega_kernel.

    Same logic per program; bidx routes per-req inputs/outputs.  STRIDE_NODES
    = max_nodes (output row stride), STRIDE_PEND = pend_max.  All other
    indexing identical to single-req kernel.
    """
    bidx = tl.program_id(0)
    k_vec = tl.arange(0, K)
    bK = bidx * K
    bN = bidx * STRIDE_NODES
    bP = bidx * STRIDE_PEND
    bKM = bidx * K * MAX_TOPK

    rank_preds = tl.load(rank_preds_ptr + bK + k_vec)
    greedy_tgt = tl.load(greedy_target_ptr + bK + k_vec)
    greedy_lps = tl.load(greedy_lps_ptr + bK + k_vec)
    slot_topks = tl.load(slot_topks_K_ptr + bK + k_vec)

    non_zero = (rank_preds != 0).to(tl.int32)
    csum_nz = tl.cumsum(non_zero, 0)
    is_first_nz = (csum_nz == 1) & (non_zero == 1)
    any_nz = tl.sum(non_zero) > 0
    first_nz_rank = tl.sum(
        tl.where(is_first_nz, rank_preds, tl.zeros([K], dtype=tl.int64))
    )
    give_up = any_nz & (first_nz_rank == GIVE_UP_CLASS)

    n_alt_slot = tl.maximum(slot_topks - 1, 0)
    cum_alt_incl = tl.cumsum(n_alt_slot, 0)
    cum_alt_excl = cum_alt_incl - n_alt_slot
    total_alts1 = tl.sum(n_alt_slot)

    active_mask = slot_topks > 1
    active_mask_i64 = active_mask.to(tl.int64)
    active_rank_incl = tl.cumsum(active_mask_i64, 0)
    active_rank = active_rank_incl - active_mask_i64
    n_active = tl.sum(active_mask_i64)

    last_active = tl.sum(
        tl.where(k_vec == (K - 1), active_mask_i64, tl.zeros_like(active_mask_i64))
    )
    hitch_add = ((~give_up) & (last_active == 0)).to(tl.int64)

    cum_greedy_lps = tl.cumsum(greedy_lps, 0)
    cum_greedy_lps_excl = cum_greedy_lps - greedy_lps
    parent_chain = k_vec.to(tl.int64) - 1

    tl.store(tree_tokens_ptr  + bN + k_vec, greedy_tgt)
    tl.store(tree_parents_ptr + bN + k_vec, parent_chain)
    tl.store(tree_lps_ptr     + bN + k_vec, cum_greedy_lps)
    tl.store(tree_ranks_ptr   + bN + k_vec, rank_preds)
    tl.store(tree_blocks_ptr  + bN + k_vec, tl.zeros([K], dtype=tl.int64))
    tl.store(tree_slots_ptr   + bN + k_vec, k_vec.to(tl.int64))

    J_PAD: tl.constexpr = 16
    K_ALT: tl.constexpr = K * J_PAD

    alt_flat = tl.arange(0, K_ALT)
    alt_k = alt_flat // J_PAD
    alt_j_idx = alt_flat - alt_k * J_PAD
    alt_j = alt_j_idx + 1

    match_k = alt_k[:, None] == k_vec[None, :]
    slot_topks_per_alt = tl.sum(
        match_k.to(tl.int64) * slot_topks[None, :], axis=1
    )
    alt_valid = (alt_j_idx < (MAX_TOPK - 1)) & (alt_j < slot_topks_per_alt)
    alt_valid_i64 = alt_valid.to(tl.int64)
    alt_rank_incl = tl.cumsum(alt_valid_i64, 0)
    alt_rank = alt_rank_incl - alt_valid_i64

    parent_lp_alt = tl.sum(
        match_k.to(tl.float32) * cum_greedy_lps_excl[None, :], axis=1
    )
    alt_off = alt_k * MAX_TOPK + alt_j
    alt_top_tgt = tl.load(all_top_target_ptr + bKM + alt_off, mask=alt_valid, other=0)
    alt_top_lps = tl.load(all_top_lps_ptr    + bKM + alt_off, mask=alt_valid, other=0.0)
    alt_lp_node = parent_lp_alt + alt_top_lps

    alt_parent = tl.where(
        alt_k == 0,
        tl.full([K_ALT], -1, dtype=tl.int64),
        alt_k.to(tl.int64) - 1,
    )
    alt_rank_pred = tl.load(rank_preds_ptr + bK + alt_k)

    tree_alt_pos = K + alt_rank
    tl.store(tree_tokens_ptr  + bN + tree_alt_pos, alt_top_tgt, mask=alt_valid)
    tl.store(tree_parents_ptr + bN + tree_alt_pos, alt_parent, mask=alt_valid)
    tl.store(tree_lps_ptr     + bN + tree_alt_pos, alt_lp_node, mask=alt_valid)
    tl.store(tree_ranks_ptr   + bN + tree_alt_pos, alt_rank_pred, mask=alt_valid)
    tl.store(tree_blocks_ptr  + bN + tree_alt_pos,
             tl.zeros([K_ALT], dtype=tl.int64), mask=alt_valid)
    tl.store(tree_slots_ptr   + bN + tree_alt_pos, alt_k.to(tl.int64), mask=alt_valid)

    pend_alt_pos = n_active + alt_rank
    tl.store(pend_hidden_slots_ptr + bP + pend_alt_pos, alt_k.to(tl.int64), mask=alt_valid)
    tl.store(pend_input_ids_ptr    + bP + pend_alt_pos, alt_top_tgt, mask=alt_valid)
    tl.store(pend_ttt_valid_ptr    + bP + pend_alt_pos, alt_k.to(tl.int64) + 1, mask=alt_valid)
    tl.store(pend_node_indices_ptr + bP + pend_alt_pos, tree_alt_pos.to(tl.int64), mask=alt_valid)
    tl.store(pend_cum_lps_ptr      + bP + pend_alt_pos, alt_lp_node, mask=alt_valid)

    tl.store(pend_hidden_slots_ptr + bP + active_rank, k_vec.to(tl.int64), mask=active_mask)
    tl.store(pend_input_ids_ptr    + bP + active_rank, greedy_tgt,          mask=active_mask)
    tl.store(pend_ttt_valid_ptr    + bP + active_rank, k_vec.to(tl.int64) + 1, mask=active_mask)
    tl.store(pend_node_indices_ptr + bP + active_rank, k_vec.to(tl.int64),  mask=active_mask)
    tl.store(pend_cum_lps_ptr      + bP + active_rank, cum_greedy_lps,      mask=active_mask)

    hitch_pos = n_active + total_alts1
    hitch_mask = (hitch_add == 1)
    last_greedy_tgt = tl.sum(
        tl.where(k_vec == (K - 1), greedy_tgt, tl.zeros([K], dtype=tl.int64))
    )
    last_cum_lp = tl.sum(
        tl.where(k_vec == (K - 1), cum_greedy_lps, tl.zeros([K], dtype=tl.float32))
    )

    ones1 = tl.arange(0, 1)
    hitch_pos_vec = hitch_pos + ones1
    hitch_mask_vec = (ones1 == 0) & hitch_mask
    val_last_tgt = (ones1 * 0).to(tl.int64) + last_greedy_tgt
    val_last_lp  = (ones1 * 0).to(tl.float32) + last_cum_lp
    val_km1      = (ones1 * 0).to(tl.int64) + (K - 1)
    val_k        = (ones1 * 0).to(tl.int64) + K
    tl.store(pend_hidden_slots_ptr + bP + hitch_pos_vec, val_km1,      mask=hitch_mask_vec)
    tl.store(pend_input_ids_ptr    + bP + hitch_pos_vec, val_last_tgt, mask=hitch_mask_vec)
    tl.store(pend_ttt_valid_ptr    + bP + hitch_pos_vec, val_k,        mask=hitch_mask_vec)
    tl.store(pend_node_indices_ptr + bP + hitch_pos_vec, val_km1,      mask=hitch_mask_vec)
    tl.store(pend_cum_lps_ptr      + bP + hitch_pos_vec, val_last_lp,  mask=hitch_mask_vec)

    n_nodes_b1 = K + total_alts1
    n_pend = n_active + total_alts1 + hitch_add
    size_idx = tl.arange(0, 4)
    size_vals = tl.where(size_idx == 0, n_nodes_b1,
                tl.where(size_idx == 1, n_active,
                tl.where(size_idx == 2, total_alts1,
                         n_pend)))
    tl.store(sizes_ptr + bidx * 4 + size_idx, size_vals)


# =============================================================================
#   BFS sizing + scatter kernels
# =============================================================================

@triton.jit
def _bfs_sizing_kernel(
    all_rank_preds_ptr,        # [N, K] i64  (kept for parity / future use)
    slot_topks_NK_ptr,         # [N*K] i64 — per-(leaf, slot) topk count, ADAPTIVE-aware

    cum_alt_excl_per_leaf_ptr, # [BLOCK_N] i64
    sizes_ptr,                 # [1] i64

    N,
    K: tl.constexpr,
    RANK_CLASSES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-leaf n_alt prefix-sum (1 program, BLOCK_N tile).

    Scatter kernel reads cum_alt_excl_per_leaf to compute its global write
    offsets, eliminating any need for atomics.
    """
    leaf_vec = tl.arange(0, BLOCK_N)
    slot_vec = tl.arange(0, K)
    leaf_mask = leaf_vec < N
    lk_off = leaf_vec[:, None] * K + slot_vec[None, :]
    lk_mask = leaf_mask[:, None]

    # all_rank_preds_ptr is no longer read once slot_topks_NK bypasses the
    # rank table; the parameter is kept in the kernel signature for parity
    # with the wrapper but not loaded here (saves an unneeded load).
    base_topk_2d = tl.load(slot_topks_NK_ptr + lk_off, mask=lk_mask, other=0)

    is_active_2d = base_topk_2d > 1
    n_alt_pair = tl.where(is_active_2d & lk_mask, base_topk_2d - 1, 0)
    n_alt_per_leaf = tl.sum(n_alt_pair, axis=1)

    cum_incl = tl.cumsum(n_alt_per_leaf, 0)
    cum_excl = cum_incl - n_alt_per_leaf
    total = tl.sum(n_alt_per_leaf)

    tl.store(cum_alt_excl_per_leaf_ptr + leaf_vec, cum_excl, mask=leaf_mask)
    ones1 = tl.arange(0, 1)
    tl.store(sizes_ptr + ones1, (ones1 * 0).to(tl.int64) + total)


@triton.jit
def _bfs_scatter_kernel_v2(
    all_rank_preds_ptr,        # [N, K] i64
    all_greedy_target_ptr,     # [N, K] i64
    all_greedy_lps_ptr,        # [N, K] f32
    top_target_all_ptr,        # [N, K, MAX_TOPK] i64
    top_lps_all_ptr,           # [N, K, MAX_TOPK] f32
    pend_cum_lps_ptr,          # [N] f32
    pend_node_indices_ptr,     # [N] i64
    slot_topks_NK_ptr,         # [N*K] i64
    cum_alt_excl_per_leaf_ptr, # [BLOCK_N] i64

    tree_tokens_ptr, tree_parents_ptr, tree_lps_ptr,
    tree_ranks_ptr, tree_blocks_ptr, tree_slots_ptr,

    valid_mask_ptr,            # [N] bool — True for real, False for pad
    alt_offset_ptr,            # [1] i64 — tree_start + real_N * K (alts start here)
    tree_start_ptr,            # [1] i64 — tree_start (replaces int kernel arg)
    N,
    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    RANK_CLASSES: tl.constexpr,
    GIVE_UP_CLASS: tl.constexpr,
    PEND_DEPTH: tl.constexpr,
    J_PAD: tl.constexpr,
):
    """v2: same as _bfs_scatter_kernel but with GPU-side pad-row gating.

    Differences from v1:
      - ``valid_mask_ptr`` gates ALL stores (chain + alt) per-leaf via the
        scalar mask broadcast through tl.store.
      - ``alt_offset_ptr`` replaces ``tree_start + N*K`` for the alt region
        start, allowing caller to pass real_N*K (GPU int, no host sync) so
        alts pack contiguously after real chains regardless of N (=pad_max).

    When all leaves real (valid_mask=True everywhere), behavior is
    bit-equivalent to v1 with alt_offset_ptr = tree_start + N*K.
    """
    leaf = tl.program_id(0)
    slot_vec = tl.arange(0, K)

    is_valid_scalar = tl.load(valid_mask_ptr + leaf) > 0
    # is_valid_vec broadcasts to [K] for chain stores via mask=is_valid_vec.
    is_valid_vec = is_valid_scalar & (slot_vec < K)  # [K] bool

    lk_off = leaf * K + slot_vec
    rank_vec = tl.load(all_rank_preds_ptr + lk_off)
    greedy_tgt = tl.load(all_greedy_target_ptr + lk_off)
    greedy_lps = tl.load(all_greedy_lps_ptr + lk_off)
    pend_cum_lp = tl.load(pend_cum_lps_ptr + leaf)
    pend_node = tl.load(pend_node_indices_ptr + leaf)

    tree_start_v = tl.load(tree_start_ptr)
    chain_base = (tree_start_v + leaf * K).to(tl.int64)
    chain_pos = chain_base + slot_vec.to(tl.int64)
    cum_greedy_lps = tl.cumsum(greedy_lps, 0)
    cum_greedy_lps_excl = cum_greedy_lps - greedy_lps

    parent_vec = tl.where(slot_vec == 0, pend_node, chain_pos - 1)
    lp_vec = pend_cum_lp + cum_greedy_lps

    tl.store(tree_tokens_ptr  + chain_pos, greedy_tgt,    mask=is_valid_vec)
    tl.store(tree_parents_ptr + chain_pos, parent_vec,    mask=is_valid_vec)
    tl.store(tree_lps_ptr     + chain_pos, lp_vec,        mask=is_valid_vec)
    tl.store(tree_ranks_ptr   + chain_pos, rank_vec,      mask=is_valid_vec)
    tl.store(tree_blocks_ptr  + chain_pos, tl.full([K], PEND_DEPTH, dtype=tl.int64), mask=is_valid_vec)
    tl.store(tree_slots_ptr   + chain_pos, slot_vec.to(tl.int64), mask=is_valid_vec)

    slot_topks = tl.load(slot_topks_NK_ptr + leaf * K + slot_vec)
    is_active = slot_topks > 1
    n_alt_slot = tl.where(is_active, slot_topks - 1, 0)
    cum_alt_slot_incl = tl.cumsum(n_alt_slot, 0)
    cum_alt_slot_excl = cum_alt_slot_incl - n_alt_slot

    leaf_alt_base = tl.load(cum_alt_excl_per_leaf_ptr + leaf)

    J_COUNT: tl.constexpr = MAX_TOPK - 1
    j_vec = tl.arange(0, J_PAD)
    j_val = j_vec + 1
    j_real = j_vec < J_COUNT

    valid_2d = (
        is_active[:, None]
        & j_real[None, :]
        & (j_val[None, :] < slot_topks[:, None])
        & is_valid_scalar  # gate by leaf-valid
    )

    within_leaf_off_2d = cum_alt_slot_excl[:, None] + j_vec[None, :]
    # alt_offset = tree_start + real_N * K (GPU-supplied; replaces tree_start + N*K)
    alt_offset = tl.load(alt_offset_ptr)
    alt_pos_2d = alt_offset + leaf_alt_base + within_leaf_off_2d

    tt_off_2d = leaf * K * MAX_TOPK + slot_vec[:, None] * MAX_TOPK + j_val[None, :]
    tt_2d = tl.load(top_target_all_ptr + tt_off_2d, mask=valid_2d, other=0)
    tlp_2d = tl.load(top_lps_all_ptr + tt_off_2d, mask=valid_2d, other=0.0)

    chain_parent_vec = chain_base + slot_vec.to(tl.int64) - 1
    alt_parent_2d = tl.where(
        slot_vec[:, None] == 0,
        tl.broadcast_to(pend_node.to(tl.int64)[None, None], [K, J_PAD]),
        tl.broadcast_to(chain_parent_vec[:, None], [K, J_PAD]),
    )

    parent_lp_2d = (pend_cum_lp + cum_greedy_lps_excl)[:, None]
    alt_lp_2d = parent_lp_2d + tlp_2d

    alt_rank_2d = tl.broadcast_to(rank_vec[:, None], [K, J_PAD])
    alt_slot_2d = tl.broadcast_to(slot_vec[:, None].to(tl.int64), [K, J_PAD])
    alt_block_2d = tl.full([K, J_PAD], PEND_DEPTH, dtype=tl.int64)

    tl.store(tree_tokens_ptr  + alt_pos_2d, tt_2d,         mask=valid_2d)
    tl.store(tree_parents_ptr + alt_pos_2d, alt_parent_2d, mask=valid_2d)
    tl.store(tree_lps_ptr     + alt_pos_2d, alt_lp_2d,     mask=valid_2d)
    tl.store(tree_ranks_ptr   + alt_pos_2d, alt_rank_2d,   mask=valid_2d)
    tl.store(tree_blocks_ptr  + alt_pos_2d, alt_block_2d,  mask=valid_2d)
    tl.store(tree_slots_ptr   + alt_pos_2d, alt_slot_2d,   mask=valid_2d)


@triton.jit
def _bfs_scatter_kernel_flat(
    all_rank_preds_ptr,        # [sum_N, K] i64 (flat across reqs)
    all_greedy_target_ptr,     # [sum_N, K] i64
    all_greedy_lps_ptr,        # [sum_N, K] f32
    top_target_all_ptr,        # [sum_N, K, MAX_TOPK] i64
    top_lps_all_ptr,           # [sum_N, K, MAX_TOPK] f32
    pend_cum_lps_ptr,          # [B, PEND_MAX] f32 (batched buffer view)
    pend_node_indices_ptr,     # [B, PEND_MAX] i64 (batched buffer view)
    slot_topks_NK_ptr,         # [sum_N*K] i64
    valid_leaf_ptr,            # [sum_N] bool

    bidx_per_leaf_ptr,         # [sum_N] i64 — original req index (0..B-1)
    leaf_local_ptr,            # [sum_N] i64 — leaf index within its req
    tree_start_per_req_ptr,    # [B] i64 — n_nodes_b1 per req (chain start)
    alt_base_per_req_ptr,      # [B] i64 — tree_start + N_b*K (alts start)
    cum_alt_excl_per_leaf_ptr, # [sum_N] i64 — per-leaf cum within its req

    tree_tokens_ptr, tree_parents_ptr, tree_lps_ptr,
    tree_ranks_ptr, tree_blocks_ptr, tree_slots_ptr,

    STRIDE_NODES,              # row stride of tree_*_ptr [B, MAX_NODES]
    STRIDE_PEND,               # row stride of pend_*_ptr [B, PEND_MAX]
    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    RANK_CLASSES: tl.constexpr,
    GIVE_UP_CLASS: tl.constexpr,
    PEND_DEPTH: tl.constexpr,
    J_PAD: tl.constexpr,
):
    """Flat batched BFS scatter: grid=(sum_N,).

    Each program processes one leaf identified by ``pid``.  Per-leaf
    metadata (bidx, leaf_local, tree_start, alt_base, cum_alt_excl)
    routes writes into the bidx-th row of the batched ``[B, MAX_NODES]``
    tree buffer.  Parent values are stored as LOCAL row indices so
    downstream 1D-view consumers see the same parent layout as the
    pre-batched per-req kernel.
    """
    pid = tl.program_id(0)
    slot_vec = tl.arange(0, K)

    leaf_valid = tl.load(valid_leaf_ptr + pid).to(tl.int1)
    bidx = tl.load(bidx_per_leaf_ptr + pid).to(tl.int64)
    leaf_local = tl.load(leaf_local_ptr + pid).to(tl.int64)
    tree_start = tl.load(tree_start_per_req_ptr + bidx).to(tl.int64)
    alt_base = tl.load(alt_base_per_req_ptr + bidx).to(tl.int64)
    leaf_alt_base = tl.load(cum_alt_excl_per_leaf_ptr + pid).to(tl.int64)
    bN = bidx * STRIDE_NODES
    bP = bidx * STRIDE_PEND

    lk_off = pid * K + slot_vec
    rank_vec = tl.load(all_rank_preds_ptr + lk_off)
    greedy_tgt = tl.load(all_greedy_target_ptr + lk_off)
    greedy_lps = tl.load(all_greedy_lps_ptr + lk_off)
    pend_cum_lp = tl.load(pend_cum_lps_ptr + bP + leaf_local)
    pend_node = tl.load(pend_node_indices_ptr + bP + leaf_local)

    chain_base_local = tree_start + leaf_local * K
    chain_pos_local = chain_base_local + slot_vec.to(tl.int64)
    chain_pos_global = bN + chain_pos_local
    cum_greedy_lps = tl.cumsum(greedy_lps, 0)
    cum_greedy_lps_excl = cum_greedy_lps - greedy_lps

    parent_vec = tl.where(slot_vec == 0, pend_node, chain_pos_local - 1)
    lp_vec = pend_cum_lp + cum_greedy_lps

    chain_mask = (slot_vec >= 0) & leaf_valid
    tl.store(tree_tokens_ptr  + chain_pos_global, greedy_tgt, mask=chain_mask)
    tl.store(tree_parents_ptr + chain_pos_global, parent_vec, mask=chain_mask)
    tl.store(tree_lps_ptr     + chain_pos_global, lp_vec, mask=chain_mask)
    tl.store(tree_ranks_ptr   + chain_pos_global, rank_vec, mask=chain_mask)
    tl.store(
        tree_blocks_ptr + chain_pos_global,
        tl.full([K], PEND_DEPTH, dtype=tl.int64),
        mask=chain_mask,
    )
    tl.store(
        tree_slots_ptr + chain_pos_global,
        slot_vec.to(tl.int64),
        mask=chain_mask,
    )

    slot_topks = tl.load(slot_topks_NK_ptr + pid * K + slot_vec)
    is_active = slot_topks > 1
    n_alt_slot = tl.where(is_active, slot_topks - 1, 0)
    cum_alt_slot_incl = tl.cumsum(n_alt_slot, 0)
    cum_alt_slot_excl = cum_alt_slot_incl - n_alt_slot

    J_COUNT: tl.constexpr = MAX_TOPK - 1
    j_vec = tl.arange(0, J_PAD)
    j_val = j_vec + 1
    j_real = j_vec < J_COUNT

    valid_2d = (
        leaf_valid
        & is_active[:, None]
        & j_real[None, :]
        & (j_val[None, :] < slot_topks[:, None])
    )

    within_leaf_off_2d = cum_alt_slot_excl[:, None] + j_vec[None, :]
    alt_pos_local_2d = alt_base + leaf_alt_base + within_leaf_off_2d
    alt_pos_global_2d = bN + alt_pos_local_2d

    tt_off_2d = pid * K * MAX_TOPK + slot_vec[:, None] * MAX_TOPK + j_val[None, :]
    tt_2d = tl.load(top_target_all_ptr + tt_off_2d, mask=valid_2d, other=0)
    tlp_2d = tl.load(top_lps_all_ptr + tt_off_2d, mask=valid_2d, other=0.0)

    chain_parent_vec_local = chain_base_local + slot_vec.to(tl.int64) - 1
    alt_parent_2d = tl.where(
        slot_vec[:, None] == 0,
        tl.broadcast_to(pend_node.to(tl.int64)[None, None], [K, J_PAD]),
        tl.broadcast_to(chain_parent_vec_local[:, None], [K, J_PAD]),
    )

    parent_lp_2d = (pend_cum_lp + cum_greedy_lps_excl)[:, None]
    alt_lp_2d = parent_lp_2d + tlp_2d

    alt_rank_2d = tl.broadcast_to(rank_vec[:, None], [K, J_PAD])
    alt_slot_2d = tl.broadcast_to(slot_vec[:, None].to(tl.int64), [K, J_PAD])
    alt_block_2d = tl.full([K, J_PAD], PEND_DEPTH, dtype=tl.int64)

    tl.store(tree_tokens_ptr  + alt_pos_global_2d, tt_2d,         mask=valid_2d)
    tl.store(tree_parents_ptr + alt_pos_global_2d, alt_parent_2d, mask=valid_2d)
    tl.store(tree_lps_ptr     + alt_pos_global_2d, alt_lp_2d,     mask=valid_2d)
    tl.store(tree_ranks_ptr   + alt_pos_global_2d, alt_rank_2d,   mask=valid_2d)
    tl.store(tree_blocks_ptr  + alt_pos_global_2d, alt_block_2d,  mask=valid_2d)
    tl.store(tree_slots_ptr   + alt_pos_global_2d, alt_slot_2d,   mask=valid_2d)


@triton.jit
def _bfs_scatter_kernel(
    all_rank_preds_ptr,        # [N, K] i64
    all_greedy_target_ptr,     # [N, K] i64
    all_greedy_lps_ptr,        # [N, K] f32
    top_target_all_ptr,        # [N, K, MAX_TOPK] i64
    top_lps_all_ptr,           # [N, K, MAX_TOPK] f32
    pend_cum_lps_ptr,          # [N] f32
    pend_node_indices_ptr,     # [N] i64
    slot_topks_NK_ptr,         # [N*K] i64 — per-(leaf, slot) topk count, ADAPTIVE-aware
    cum_alt_excl_per_leaf_ptr, # [BLOCK_N] i64

    tree_tokens_ptr, tree_parents_ptr, tree_lps_ptr,
    tree_ranks_ptr, tree_blocks_ptr, tree_slots_ptr,

    N, tree_start,
    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    RANK_CLASSES: tl.constexpr,
    GIVE_UP_CLASS: tl.constexpr,
    PEND_DEPTH: tl.constexpr,
    J_PAD: tl.constexpr,
):
    """Grid-parallel BFS scatter, one program per leaf."""
    leaf = tl.program_id(0)
    slot_vec = tl.arange(0, K)

    lk_off = leaf * K + slot_vec
    rank_vec = tl.load(all_rank_preds_ptr + lk_off)
    greedy_tgt = tl.load(all_greedy_target_ptr + lk_off)
    greedy_lps = tl.load(all_greedy_lps_ptr + lk_off)
    pend_cum_lp = tl.load(pend_cum_lps_ptr + leaf)
    pend_node = tl.load(pend_node_indices_ptr + leaf)

    chain_base = (tree_start + leaf * K).to(tl.int64)
    chain_pos = chain_base + slot_vec.to(tl.int64)
    cum_greedy_lps = tl.cumsum(greedy_lps, 0)
    cum_greedy_lps_excl = cum_greedy_lps - greedy_lps

    parent_vec = tl.where(slot_vec == 0, pend_node, chain_pos - 1)
    lp_vec = pend_cum_lp + cum_greedy_lps

    tl.store(tree_tokens_ptr  + chain_pos, greedy_tgt)
    tl.store(tree_parents_ptr + chain_pos, parent_vec)
    tl.store(tree_lps_ptr     + chain_pos, lp_vec)
    tl.store(tree_ranks_ptr   + chain_pos, rank_vec)
    tl.store(tree_blocks_ptr  + chain_pos, tl.full([K], PEND_DEPTH, dtype=tl.int64))
    tl.store(tree_slots_ptr   + chain_pos, slot_vec.to(tl.int64))

    # ADAPTIVE-aware: caller pre-computes slot_topks_NK [N, K] combining
    # rank-table values and any ADAPTIVE_ALL shrinkage.
    slot_topks = tl.load(slot_topks_NK_ptr + leaf * K + slot_vec)  # [K] i64
    is_active = slot_topks > 1
    n_alt_slot = tl.where(is_active, slot_topks - 1, 0)
    cum_alt_slot_incl = tl.cumsum(n_alt_slot, 0)
    cum_alt_slot_excl = cum_alt_slot_incl - n_alt_slot

    leaf_alt_base = tl.load(cum_alt_excl_per_leaf_ptr + leaf)

    J_COUNT: tl.constexpr = MAX_TOPK - 1
    j_vec = tl.arange(0, J_PAD)
    j_val = j_vec + 1
    j_real = j_vec < J_COUNT

    valid_2d = (
        is_active[:, None]
        & j_real[None, :]
        & (j_val[None, :] < slot_topks[:, None])
    )

    within_leaf_off_2d = cum_alt_slot_excl[:, None] + j_vec[None, :]
    alt_pos_2d = tree_start + N * K + leaf_alt_base + within_leaf_off_2d

    tt_off_2d = leaf * K * MAX_TOPK + slot_vec[:, None] * MAX_TOPK + j_val[None, :]
    tt_2d = tl.load(top_target_all_ptr + tt_off_2d, mask=valid_2d, other=0)
    tlp_2d = tl.load(top_lps_all_ptr + tt_off_2d, mask=valid_2d, other=0.0)

    chain_parent_vec = chain_base + slot_vec.to(tl.int64) - 1
    alt_parent_2d = tl.where(
        slot_vec[:, None] == 0,
        tl.broadcast_to(pend_node.to(tl.int64)[None, None], [K, J_PAD]),
        tl.broadcast_to(chain_parent_vec[:, None], [K, J_PAD]),
    )

    parent_lp_2d = (pend_cum_lp + cum_greedy_lps_excl)[:, None]
    alt_lp_2d = parent_lp_2d + tlp_2d

    alt_rank_2d = tl.broadcast_to(rank_vec[:, None], [K, J_PAD])
    alt_slot_2d = tl.broadcast_to(slot_vec[:, None].to(tl.int64), [K, J_PAD])
    alt_block_2d = tl.full([K, J_PAD], PEND_DEPTH, dtype=tl.int64)

    tl.store(tree_tokens_ptr  + alt_pos_2d, tt_2d,         mask=valid_2d)
    tl.store(tree_parents_ptr + alt_pos_2d, alt_parent_2d, mask=valid_2d)
    tl.store(tree_lps_ptr     + alt_pos_2d, alt_lp_2d,     mask=valid_2d)
    tl.store(tree_ranks_ptr   + alt_pos_2d, alt_rank_2d,   mask=valid_2d)
    tl.store(tree_blocks_ptr  + alt_pos_2d, alt_block_2d,  mask=valid_2d)
    tl.store(tree_slots_ptr   + alt_pos_2d, alt_slot_2d,   mask=valid_2d)


# =============================================================================
#   Topology kernels (depth + ancestor mask, leaf retrieve_index)
# =============================================================================

@triton.jit
def _tree_depth_mask_kernel(
    parent_ptr, depth_ptr, mask_ptr,
    Np1, MAX_DEPTH: tl.constexpr,
):
    """Compute depth and ancestor mask for one tree node.

    Each program walks parent chain to root, recording depth and setting
    mask[node, ancestor] = 1.0 for each ancestor (including self).
    """
    pid = tl.program_id(0)

    cur = pid
    depth = 0
    for _ in range(MAX_DEPTH + 1):
        tl.store(mask_ptr + pid * Np1 + cur, 1.0)
        parent = tl.load(parent_ptr + cur)
        is_root = cur == 0
        depth += tl.where(is_root, 0, 1)
        cur = tl.where(is_root, 0, parent)

    tl.store(depth_ptr + pid, depth)


@triton.jit
def _tree_retrieve_kernel(
    parent_ptr, depth_ptr, leaves_ptr, ri_ptr,
    max_depth, Np1, num_leaves,
    MAX_DEPTH: tl.constexpr,
):
    """Build root-to-leaf path for one leaf node.

    Walks from leaf to root, writes at reversed column index so the result
    is root→leaf order.
    """
    lid = tl.program_id(0)
    if lid >= num_leaves:
        return

    leaf = tl.load(leaves_ptr + lid)
    leaf_depth = tl.load(depth_ptr + leaf)

    cur = leaf
    for d in range(MAX_DEPTH + 1):
        col = leaf_depth - d
        valid = (d <= leaf_depth) & (col >= 0)
        tl.store(ri_ptr + lid * (max_depth + 1) + tl.where(valid, col, 0), cur, mask=valid)
        parent = tl.load(parent_ptr + cur)
        cur = tl.where(cur > 0, parent, cur)


# =============================================================================
#   Batched depth+mask / retrieve kernels (S3c)
# =============================================================================

@triton.jit
def _tree_depth_mask_kernel_batched(
    parent_ptr,        # [B, MAX_NP1] i32 (padded with 0 self-root for pad slots)
    depth_ptr,         # [B, MAX_NP1] i32
    mask_ptr,          # [B, MAX_NP1, MAX_NP1] f32 (zero-init'd by caller)
    MAX_NP1,           # row stride for parent/depth
    MASK_STRIDE,       # row stride for mask = MAX_NP1 * MAX_NP1
    MAX_DEPTH: tl.constexpr,
):
    """grid=(B, MAX_NP1) batched depth+ancestor-mask walk.

    Same per-node logic as ``_tree_depth_mask_kernel``; bidx routes the
    per-req row of the [B, MAX_NP1] / [B, MAX_NP1, MAX_NP1] buffers.
    Pad slots (pid >= Np1_b for req b) are stored with parent=0 by caller,
    so their walk terminates after one step writing mask[pad, 0] and
    mask[pad, pad]; these rows are sliced away in the per-req output dict.
    """
    bidx = tl.program_id(0)
    pid = tl.program_id(1)

    bP = bidx * MAX_NP1
    bM = bidx * MASK_STRIDE

    cur = pid
    depth = 0
    for _ in range(MAX_DEPTH + 1):
        tl.store(mask_ptr + bM + pid * MAX_NP1 + cur, 1.0)
        parent = tl.load(parent_ptr + bP + cur)
        is_root = cur == 0
        depth += tl.where(is_root, 0, 1)
        cur = tl.where(is_root, 0, parent)

    tl.store(depth_ptr + bP + pid, depth)


@triton.jit
def _tree_retrieve_kernel_batched(
    parent_ptr,        # [B, MAX_NP1] i32
    depth_ptr,         # [B, MAX_NP1] i32
    leaves_ptr,        # [B, MAX_NP1] i32 (padded with MAX_NP1 sentinel)
    num_leaves_ptr,    # [B] i32 — per-req real leaf count
    ri_ptr,            # [B, MAX_NP1, max_depth+1] i32 (-1 pre-filled)
    max_depth,
    MAX_NP1,
    RI_STRIDE,         # = MAX_NP1 * (max_depth+1)
    MAX_DEPTH: tl.constexpr,
):
    """grid=(B, MAX_NP1) batched retrieve.  Each program at (bidx, lid)
    builds the root->leaf path for the lid-th leaf of req bidx.  Gated by
    ``lid < num_leaves[bidx]`` so over-allocated lid slots no-op.
    """
    bidx = tl.program_id(0)
    lid = tl.program_id(1)

    nl = tl.load(num_leaves_ptr + bidx)
    if lid >= nl:
        return

    bP = bidx * MAX_NP1
    bL = bidx * MAX_NP1
    bRI = bidx * RI_STRIDE

    leaf = tl.load(leaves_ptr + bL + lid)
    leaf_depth = tl.load(depth_ptr + bP + leaf)

    cur = leaf
    for d in range(MAX_DEPTH + 1):
        col = leaf_depth - d
        valid = (d <= leaf_depth) & (col >= 0)
        tl.store(
            ri_ptr + bRI + lid * (max_depth + 1) + tl.where(valid, col, 0),
            cur,
            mask=valid,
        )
        parent = tl.load(parent_ptr + bP + cur)
        cur = tl.where(cur > 0, parent, cur)


# =============================================================================
#   Ancestor closure (for prune)
# =============================================================================

@triton.jit
def _ancestor_closure_kernel(
    parents_ptr,      # [N] int64
    keep_ptr,         # [N] int32
    N,
    MAX_DEPTH: tl.constexpr,
):
    """For each kept node i, mark its parent chain kept."""
    pid = tl.program_id(0)
    if pid >= N:
        return
    mine = tl.load(keep_ptr + pid)
    if mine == 0:
        return
    cur = tl.load(parents_ptr + pid)
    for _ in range(MAX_DEPTH):
        valid = cur >= 0
        if valid:
            prev = tl.load(keep_ptr + cur)
            if prev == 0:
                tl.store(keep_ptr + cur, 1)
            cur = tl.load(parents_ptr + cur)


# =============================================================================
#   Python wrappers
# =============================================================================

def triton_build_block1(
    rank_preds: torch.Tensor,            # [K] i64
    greedy_target: torch.Tensor,         # [K] i64
    greedy_lps: torch.Tensor,            # [K] f32
    all_top_target: torch.Tensor,        # [K, MAX_TOPK] i64
    all_top_lps: torch.Tensor,           # [K, MAX_TOPK] f32
    slot_topks_K: torch.Tensor,          # [K] i64 — caller-provided per-slot topk (ADAPTIVE-aware)
    tree_buf: dict,
    pend_buf: dict,
    sizes_buf: torch.Tensor,             # [4] i64 scratch
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
):
    """Launch block-1 mega-kernel.

    Returns (n_nodes_b1, n_active, total_alts1, N_pend) as host ints (default,
    1 sync), or the GPU sizes_buf [4] tensor when env
    ``SPECBLOCK_NPEND_GPU_ONLY=1`` (caller pads N_pend to buffer capacity).
    """
    _block1_mega_kernel[(1,)](
        rank_preds, greedy_target, greedy_lps,
        all_top_target, all_top_lps,
        slot_topks_K,
        tree_buf['tokens'], tree_buf['parents'], tree_buf['lps'],
        tree_buf['ranks'], tree_buf['blocks'], tree_buf['slots'],
        pend_buf['hidden_slots'], pend_buf['input_ids'],
        pend_buf['ttt_valid'], pend_buf['node_indices'], pend_buf['cum_lps'],
        sizes_buf,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class,
    )
    if os.environ.get("SPECBLOCK_NPEND_GPU_ONLY", "0") == "1":
        return sizes_buf  # [4] gpu int64
    sizes_cpu = sizes_buf.cpu().tolist()
    return sizes_cpu[0], sizes_cpu[1], sizes_cpu[2], sizes_cpu[3]


def triton_build_bfs_v2(
    all_rank_preds: torch.Tensor,
    all_greedy_target: torch.Tensor,
    all_greedy_lps: torch.Tensor,
    top_target_all: torch.Tensor,
    top_lps_all: torch.Tensor,
    pend_cum_lps: torch.Tensor,
    pend_node_indices: torch.Tensor,
    slot_topks_NK: torch.Tensor,
    valid_mask: torch.Tensor,            # [N] bool
    alt_offset: torch.Tensor,            # [1] i64 (gpu)
    tree_start_ptr: torch.Tensor,        # [1] i64 (gpu, replaces tree_start int arg)
    tree_buf: dict,
    sizes_buf: torch.Tensor,
    cum_alt_buf: torch.Tensor,
    N: int,
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
    pend_depth: int,
    block_n: int = 32,
    j_pad: int = 16,
):
    """v2 BFS: pad-aware scatter (valid_mask + alt_offset_ptr).

    Sizing kernel unchanged: caller masks slot_topks_NK[pad rows]=0 so
    cum_alt sums skip pad.  Returns total_alts_2 as a GPU tensor (no
    host sync).  Caller computes n_nodes_final on GPU and syncs once at
    finalize boundary if needed.
    """
    slot_topks_NK_flat = slot_topks_NK.reshape(-1).contiguous()
    _bfs_sizing_kernel[(1,)](
        all_rank_preds, slot_topks_NK_flat,
        cum_alt_buf, sizes_buf,
        N,
        K=K, RANK_CLASSES=rank_classes,
        BLOCK_N=block_n,
    )
    _bfs_scatter_kernel_v2[(N,)](
        all_rank_preds, all_greedy_target, all_greedy_lps,
        top_target_all, top_lps_all,
        pend_cum_lps, pend_node_indices,
        slot_topks_NK_flat, cum_alt_buf,
        tree_buf['tokens'], tree_buf['parents'], tree_buf['lps'],
        tree_buf['ranks'], tree_buf['blocks'], tree_buf['slots'],
        valid_mask, alt_offset, tree_start_ptr,
        N,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class,
        PEND_DEPTH=pend_depth,
        J_PAD=j_pad,
    )
    # Return total_alts_2 as GPU tensor (sizes_buf[0:1]).  Caller may
    # sync at finalize boundary if int needed.
    return sizes_buf[0:1]


def triton_build_bfs_flat_batched(
    all_rank_preds_full: torch.Tensor,    # [sum_N, K] i64
    all_greedy_target_full: torch.Tensor, # [sum_N, K] i64
    all_greedy_lps_full: torch.Tensor,    # [sum_N, K] f32
    top_target_all_full: torch.Tensor,    # [sum_N, K, MAX_TOPK] i64
    top_lps_all_full: torch.Tensor,       # [sum_N, K, MAX_TOPK] f32
    slot_topks_NK_full: torch.Tensor,     # [sum_N, K] i64

    pend_buf_b: dict,                     # batched pend buf: each field [B, PEND_MAX]
    tree_buf_b: dict,                     # batched tree buf: each field [B, MAX_NODES]

    n_nodes_b1_b: torch.Tensor,           # [B] i64 — tree_start per req
    N_per_req_b: torch.Tensor,            # [B] i64 — leaves per req (0 for invalid)
    cum_off_t: torch.Tensor,              # [B+1] i64 — flat offsets per req

    *,
    B: int,
    sum_N: int,
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
    pend_depth: int,
    max_nodes: int,
    pend_max: int,
    j_pad: int = 16,
    device: torch.device = None,
) -> torch.Tensor:
    """Single batched BFS scatter across all reqs.

    Replaces B per-req ``_bfs_sizing_kernel[(1,)]`` + ``_bfs_scatter_kernel[(N,)]``
    launches with a single flat ``_bfs_scatter_kernel_flat[(sum_N,)]`` launch.

    Cum_alt prefix sums are computed pure-torch on GPU (segmented cumsum)
    so no per-req sizing kernel is needed.  Returns ``total_alts_per_req``
    [B] long tensor (caller does one ``.cpu().tolist()`` instead of B).
    """
    if sum_N == 0:
        return torch.zeros(B, dtype=torch.long, device=device)

    if device is None:
        device = all_rank_preds_full.device

    # ---- Per-leaf bidx + leaf_local via cum_off table.
    # bucketize semantics:
    #   right=False: returns smallest i s.t. input <= boundaries[i]
    #   right=True:  returns smallest i s.t. input <  boundaries[i]
    # We want smallest i s.t. global idx < cum_off[i+1] (= boundaries[i])
    # — that's right=True.  (right=False groups boundary value with the
    # previous bucket, so a leaf exactly at cum_off[b+1] would be
    # mis-assigned to req b-1.)
    leaf_arange = torch.arange(sum_N, device=device, dtype=torch.long)
    bidx_per_leaf_t = torch.bucketize(
        leaf_arange, cum_off_t[1:], right=True,
    ).to(torch.long)  # [sum_N]
    leaf_local_t = leaf_arange - cum_off_t[bidx_per_leaf_t]  # [sum_N]

    # ---- Per-req metadata: tree_start, alt_base = tree_start + N_b * K
    alt_base_per_req_b = n_nodes_b1_b + N_per_req_b * K  # [B]

    # ---- Per-leaf cum_alt_excl within its req.
    # n_alt_per_leaf = sum_k max(slot_topks - 1, 0)
    n_alt_slot = (slot_topks_NK_full - 1).clamp(min=0)             # [sum_N, K]
    n_alt_per_leaf = n_alt_slot.sum(dim=1)                          # [sum_N]
    csum_global = n_alt_per_leaf.cumsum(0)                          # [sum_N]
    csum_excl_global = csum_global - n_alt_per_leaf                 # [sum_N]
    # csum at the start (excl) of each leaf's own req:
    start_off_per_leaf = cum_off_t[:B][bidx_per_leaf_t]             # [sum_N]
    start_csum_per_leaf = csum_excl_global.gather(0, start_off_per_leaf)
    cum_alt_excl_per_leaf = csum_excl_global - start_csum_per_leaf  # [sum_N]

    # ---- total_alts per req via scatter_add.
    total_alts_per_req_b = torch.zeros(B, dtype=torch.long, device=device)
    total_alts_per_req_b.scatter_add_(0, bidx_per_leaf_t, n_alt_per_leaf)

    # ---- Single flat scatter kernel launch.
    valid_leaf = torch.ones(sum_N, dtype=torch.bool, device=device)
    _bfs_scatter_kernel_flat[(sum_N,)](
        all_rank_preds_full.to(torch.int64).contiguous(),
        all_greedy_target_full.to(torch.int64).contiguous(),
        all_greedy_lps_full.to(torch.float32).contiguous(),
        top_target_all_full.to(torch.int64).contiguous(),
        top_lps_all_full.to(torch.float32).contiguous(),
        pend_buf_b['cum_lps'],
        pend_buf_b['node_indices'],
        slot_topks_NK_full.reshape(-1).contiguous(),
        valid_leaf,
        bidx_per_leaf_t.contiguous(),
        leaf_local_t.contiguous(),
        n_nodes_b1_b.contiguous(),
        alt_base_per_req_b.contiguous(),
        cum_alt_excl_per_leaf.contiguous(),
        tree_buf_b['tokens'], tree_buf_b['parents'], tree_buf_b['lps'],
        tree_buf_b['ranks'], tree_buf_b['blocks'], tree_buf_b['slots'],
        max_nodes,
        pend_max,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class,
        PEND_DEPTH=pend_depth,
        J_PAD=j_pad,
    )

    return total_alts_per_req_b  # [B] gpu long


def triton_build_bfs_fixed_batched(
    all_rank_preds: torch.Tensor,
    all_greedy_target: torch.Tensor,
    all_greedy_lps: torch.Tensor,
    top_target_all: torch.Tensor,
    top_lps_all: torch.Tensor,
    slot_topks: torch.Tensor,
    valid_leaf_b: torch.Tensor,
    pend_buf_b: dict,
    tree_buf_b: dict,
    sizes4_b: torch.Tensor,
    *,
    B: int,
    pend_bucket: int,
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
    max_nodes: int,
) -> torch.Tensor:
    """Scatter fixed ``[B, P]`` pending rows without compacting or host sync."""
    total_rows = B * pend_bucket
    device = valid_leaf_b.device
    flat_valid = valid_leaf_b.reshape(total_rows)
    bidx = (
        torch.arange(B, device=device, dtype=torch.long)
        .unsqueeze(1)
        .expand(B, pend_bucket)
        .reshape(total_rows)
    )
    leaf_local = (
        torch.arange(pend_bucket, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(B, pend_bucket)
        .reshape(total_rows)
    )

    slot_topks_b = slot_topks.reshape(B, pend_bucket, K)
    n_alt_per_leaf = (slot_topks_b - 1).clamp(min=0).sum(dim=2)
    n_alt_per_leaf = torch.where(
        valid_leaf_b, n_alt_per_leaf, torch.zeros_like(n_alt_per_leaf)
    )
    cum_alt_incl = n_alt_per_leaf.cumsum(dim=1)
    cum_alt_excl = cum_alt_incl - n_alt_per_leaf
    total_alts = cum_alt_incl[:, -1]

    tree_start = sizes4_b[:, 0].contiguous()
    n_pend = sizes4_b[:, 3].contiguous()
    alt_base = (tree_start + n_pend * K).contiguous()

    _bfs_scatter_kernel_flat[(total_rows,)](
        all_rank_preds.to(torch.int64).contiguous(),
        all_greedy_target.to(torch.int64).contiguous(),
        all_greedy_lps.to(torch.float32).contiguous(),
        top_target_all.to(torch.int64).contiguous(),
        top_lps_all.to(torch.float32).contiguous(),
        pend_buf_b["cum_lps"], pend_buf_b["node_indices"],
        slot_topks.reshape(-1).contiguous(), flat_valid.contiguous(),
        bidx.contiguous(), leaf_local.contiguous(),
        tree_start, alt_base, cum_alt_excl.reshape(-1).contiguous(),
        tree_buf_b["tokens"], tree_buf_b["parents"], tree_buf_b["lps"],
        tree_buf_b["ranks"], tree_buf_b["blocks"], tree_buf_b["slots"],
        max_nodes, pend_bucket,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class, PEND_DEPTH=1, J_PAD=16,
    )
    return total_alts


def triton_build_bfs(
    all_rank_preds: torch.Tensor,        # [N, K]
    all_greedy_target: torch.Tensor,     # [N, K]
    all_greedy_lps: torch.Tensor,        # [N, K]
    top_target_all: torch.Tensor,        # [N, K, MAX_TOPK]
    top_lps_all: torch.Tensor,           # [N, K, MAX_TOPK]
    pend_cum_lps: torch.Tensor,          # [N] f32
    pend_node_indices: torch.Tensor,     # [N] i64
    slot_topks_NK: torch.Tensor,         # [N, K] i64 — caller-provided per-(leaf,slot) topk (ADAPTIVE-aware)
    tree_buf: dict,
    sizes_buf: torch.Tensor,             # [1] i64 scratch
    cum_alt_buf: torch.Tensor,           # [BLOCK_N] i64 scratch
    tree_start: int,
    N: int,
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
    pend_depth: int,
    block_n: int = 32,
    j_pad: int = 16,
):
    """Launch BFS kernels: sizing (1 program) + scatter (grid=N). Returns total_alts_2."""
    # Kernels expect a flat [N*K] tensor.  Use reshape (not view) so an
    # accidentally non-contiguous input still works without raising.
    slot_topks_NK_flat = slot_topks_NK.reshape(-1).contiguous()
    _bfs_sizing_kernel[(1,)](
        all_rank_preds, slot_topks_NK_flat,
        cum_alt_buf, sizes_buf,
        N,
        K=K, RANK_CLASSES=rank_classes,
        BLOCK_N=block_n,
    )
    _bfs_scatter_kernel[(N,)](
        all_rank_preds, all_greedy_target, all_greedy_lps,
        top_target_all, top_lps_all,
        pend_cum_lps, pend_node_indices,
        slot_topks_NK_flat, cum_alt_buf,
        tree_buf['tokens'], tree_buf['parents'], tree_buf['lps'],
        tree_buf['ranks'], tree_buf['blocks'], tree_buf['slots'],
        N, tree_start,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class,
        PEND_DEPTH=pend_depth,
        J_PAD=j_pad,
    )
    return int(sizes_buf.cpu().item())


# =============================================================================
#   Prune (top-K + ancestor closure)
# =============================================================================

def gpu_prune_tree_batched(
    tree_buf_b: dict,
    n_nodes_per_req: List[int],
    prune_rows: List[int],
    budget: int,
) -> dict:
    """Prune several trees with one set of batched PyTorch kernels.

    Cumulative log-probability is monotone along every tree edge because each
    child adds a non-positive token log-probability.  Therefore the top-k nodes
    are already ancestor-closed; sorting their original indices reproduces the
    compact order of :func:`gpu_prune_tree` without per-request closure kernels.
    """
    device = tree_buf_b['tokens'].device
    num_pruned = len(prune_rows)
    if num_pruned == 0:
        return {}

    max_n = max(n_nodes_per_req[row] for row in prune_rows)
    all_rows = prune_rows == list(range(len(n_nodes_per_req)))
    row_index = None if all_rows else torch.tensor(
        prune_rows, dtype=torch.long, device=device,
    )
    row_n = torch.tensor(
        [n_nodes_per_req[row] for row in prune_rows],
        dtype=torch.long, device=device,
    )
    arange_n = torch.arange(max_n, device=device).unsqueeze(0)
    valid = arange_n < row_n.unsqueeze(1)

    def select_rows(name: str) -> torch.Tensor:
        src = tree_buf_b[name]
        if all_rows:
            return src[:num_pruned, :max_n]
        return src.index_select(0, row_index)[:, :max_n]

    lps = select_rows('lps')
    scores = torch.where(valid, lps, torch.full_like(lps, float('-inf')))
    top_idx = torch.topk(scores, budget, dim=1, largest=True).indices
    kept_idx = top_idx.sort(dim=1).values

    old_to_new = torch.full(
        (num_pruned, max_n), -1, dtype=torch.long, device=device,
    )
    compact_idx = torch.arange(budget, device=device).unsqueeze(0).expand(
        num_pruned, -1,
    )
    old_to_new.scatter_(1, kept_idx, compact_idx)

    def gather_field(name: str) -> torch.Tensor:
        return torch.gather(select_rows(name), 1, kept_idx)

    old_parents = gather_field('parents')
    parent_new_pos = torch.gather(old_to_new, 1, old_parents.clamp(min=0))
    new_parents = torch.where(
        old_parents >= 0,
        parent_new_pos,
        torch.full_like(old_parents, -1),
    )

    return {
        'tokens': gather_field('tokens'),
        'parents': new_parents,
        'lps': gather_field('lps'),
        'ranks': gather_field('ranks'),
        'blocks': gather_field('blocks'),
        'slots': gather_field('slots'),
    }


@triton.jit
def _dedup_tree_rows_by_depth_kernel(
    tokens_ptr,
    parents_ptr,
    lps_ptr,
    raw_n_ptr,
    compact_packed_buf_ptr,
    combined_winner_buf_ptr,
    node_state_buf_ptr,
    stride_raw: tl.constexpr,
    stride_compact: tl.constexpr,
    N: tl.constexpr,
    COMPACT_N: tl.constexpr,
    MAX_DEDUP_DEPTH: tl.constexpr,
    MAX_WALK_DEPTH: tl.constexpr,
):
    """Merge duplicate semantic paths with compact per-depth sorts."""
    bidx = tl.program_id(0)
    raw_base = bidx * stride_raw
    compact_base = bidx * stride_compact
    n = tl.load(raw_n_ptr + bidx)
    pid = tl.arange(0, N)
    compact_pid = tl.arange(0, COMPACT_N)
    valid = pid < n

    tokens = tl.load(
        tokens_ptr + raw_base + pid,
        mask=valid,
        other=0,
    ).to(tl.int64)
    parents = tl.load(
        parents_ptr + raw_base + pid,
        mask=valid,
        other=-1,
    ).to(tl.int64)

    # Raw nodes are topologically ordered.  Static depth remains valid while
    # same-depth losers are redirected to the winner under the same parent.
    cur = pid.to(tl.int64)
    depths = tl.zeros([N], dtype=tl.int32)
    for _ in range(MAX_WALK_DEPTH):
        cur_parent = tl.load(
            parents_ptr + raw_base + cur,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        has_parent = cur_parent >= 0
        depths += has_parent.to(tl.int32)
        cur = tl.where(has_parent, cur_parent, cur)

    key_shift = tl.cast(1 << 32, tl.int64)
    nplus = tl.cast(N + 1, tl.int64)
    sentinel_packed = tl.cast(1 << 62, tl.int64)
    sentinel_key = tl.cast(-1, tl.int64)
    xor_ffff = tl.cast(0xFFFFFFFF, tl.int64)
    low_mask = tl.cast(0xFFFFFFFF, tl.int64)
    mask_shift = tl.cast(32, tl.int64)

    # Compact each depth before sorting.  The production W90 geometry has at
    # most 170 nodes at any depth, so a 256-lane sort replaces eight 1024-lane
    # sorts without changing winner selection or parent canonicalization.
    for target_d in tl.static_range(1, MAX_DEDUP_DEPTH + 1):
        lps = tl.load(
            lps_ptr + raw_base + pid,
            mask=valid,
            other=float("-inf"),
        )
        is_at_d = (depths == target_d) & valid & (lps != float("-inf"))
        compact_rank = tl.cumsum(is_at_d.to(tl.int32), axis=0) - 1
        n_at_d = tl.sum(is_at_d.to(tl.int32), axis=0)

        tl.store(
            compact_packed_buf_ptr + compact_base + compact_pid,
            tl.full([COMPACT_N], sentinel_packed, tl.int64),
        )
        tl.store(
            combined_winner_buf_ptr + compact_base + compact_pid,
            tl.zeros([COMPACT_N], tl.int64),
        )
        tl.store(
            node_state_buf_ptr + raw_base + pid,
            tl.zeros([N], tl.int64),
        )
        tl.debug_barrier()

        keys = (parents + 1) * key_shift + tokens
        packed = keys * nplus + pid.to(tl.int64)
        tl.store(
            compact_packed_buf_ptr + compact_base + compact_rank,
            packed,
            mask=is_at_d,
        )
        tl.debug_barrier()

        compact_packed = tl.load(
            compact_packed_buf_ptr + compact_base + compact_pid,
        )
        sorted_packed = tl.sort(compact_packed)
        sorted_raw_pid = sorted_packed % nplus
        sorted_keys = sorted_packed // nplus
        sorted_valid = compact_pid < n_at_d

        tl.store(
            compact_packed_buf_ptr + compact_base + compact_pid,
            sorted_keys,
        )
        tl.debug_barrier()
        sorted_keys_prev = tl.load(
            compact_packed_buf_ptr + compact_base + compact_pid - 1,
            mask=(compact_pid > 0) & sorted_valid,
            other=sentinel_key,
        )
        is_first = (sorted_keys != sorted_keys_prev) & sorted_valid
        group_id = (
            tl.cumsum(is_first.to(tl.int32), 0) - 1
        ).to(tl.int64)

        sorted_lps = tl.load(
            lps_ptr + raw_base + sorted_raw_pid,
            mask=sorted_valid,
            other=float("-inf"),
        )
        lps_bits = (
            sorted_lps.to(tl.int32, bitcast=True).to(tl.int64) & low_mask
        )
        encoded_lp = lps_bits ^ xor_ffff
        pid_inv = (sorted_raw_pid ^ xor_ffff) & low_mask
        combined = (encoded_lp << 32) | pid_inv
        tl.atomic_max(
            combined_winner_buf_ptr + compact_base + group_id,
            combined,
            mask=sorted_valid,
        )
        tl.debug_barrier()
        winner_combined = tl.load(
            combined_winner_buf_ptr + compact_base + group_id,
            mask=sorted_valid,
            other=0,
        ).to(tl.int64)
        winner_pid = (winner_combined & low_mask) ^ xor_ffff
        to_mask = sorted_valid & (combined != winner_combined)
        node_state = (
            to_mask.to(tl.int64) << mask_shift
        ) | (winner_pid & low_mask)
        tl.store(
            node_state_buf_ptr + raw_base + sorted_raw_pid,
            node_state,
            mask=sorted_valid,
        )
        tl.store(
            lps_ptr + raw_base + sorted_raw_pid,
            float("-inf"),
            mask=to_mask,
        )
        tl.debug_barrier()

        parents_clamped = tl.maximum(parents, 0)
        parent_state = tl.load(
            node_state_buf_ptr + raw_base + parents_clamped,
            mask=valid,
            other=0,
        ).to(tl.int64)
        parent_is_masked = (parent_state >> mask_shift) != 0
        parent_winner = parent_state & low_mask
        parents = tl.where(
            (parents >= 0) & parent_is_masked,
            parent_winner,
            parents,
        )

    tl.store(parents_ptr + raw_base + pid, parents, mask=valid)


@triton.jit
def _finalize_one_block_fixed_kernel(
    raw_tokens_ptr,
    raw_parents_ptr,
    raw_lps_ptr,
    kept_idx_ptr,
    kept_valid_ptr,
    sample_tokens_ptr,
    seq_lens_ptr,
    out_tokens_ptr,
    out_parents_ptr,
    out_topo_parents_ptr,
    out_lps_ptr,
    out_depth_ptr,
    out_positions_ptr,
    out_mask_ptr,
    out_retrieve_index_ptr,
    out_next_token_ptr,
    out_next_sibling_ptr,
    stride_raw: tl.constexpr,
    stride_kept: tl.constexpr,
    stride_out: tl.constexpr,
    stride_mask: tl.constexpr,
    N: tl.constexpr,
    BUDGET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
):
    """Pack a pruned one-block tree and build topology in one program/request."""
    bidx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    nonroot = offs - 1
    is_root = offs == 0
    in_tree = offs < N
    is_nonroot = (offs > 0) & in_tree

    kept = tl.load(
        kept_idx_ptr + bidx * stride_kept + nonroot,
        mask=is_nonroot,
        other=0,
    ).to(tl.int64)
    kept_valid = tl.load(
        kept_valid_ptr + bidx * stride_kept + nonroot,
        mask=is_nonroot,
        other=0,
    ).to(tl.int1)
    node_valid = is_root | (is_nonroot & kept_valid)

    old_parent = tl.load(
        raw_parents_ptr + bidx * stride_raw + kept,
        mask=is_nonroot & kept_valid,
        other=-1,
    ).to(tl.int64)

    # Map raw parent indices to compact non-root indices.  Top-k cumulative
    # probability is ancestor-closed, so every non-root parent has one match.
    cand = tl.arange(0, BLOCK_N)
    cand_valid = cand < BUDGET
    cand_old = tl.load(
        kept_idx_ptr + bidx * stride_kept + cand,
        mask=cand_valid,
        other=-2,
    ).to(tl.int64)
    cand_keep = tl.load(
        kept_valid_ptr + bidx * stride_kept + cand,
        mask=cand_valid,
        other=0,
    ).to(tl.int1)
    parent_match = (
        old_parent[:, None] == cand_old[None, :]
    ) & cand_keep[None, :] & is_nonroot[:, None] & kept_valid[:, None]
    mapped_parent = tl.max(
        tl.where(parent_match, cand[None, :], -1), axis=1,
    ).to(tl.int64)
    raw_parent_out = tl.where(
        is_root,
        -1,
        tl.where(
            kept_valid,
            tl.where(old_parent >= 0, mapped_parent, -1),
            -2,
        ),
    )

    sample = tl.load(sample_tokens_ptr + bidx)
    raw_token = tl.load(
        raw_tokens_ptr + bidx * stride_raw + kept,
        mask=is_nonroot & kept_valid,
        other=0,
    )
    raw_lp = tl.load(
        raw_lps_ptr + bidx * stride_raw + kept,
        mask=is_nonroot & kept_valid,
        other=float("-inf"),
    )
    tl.store(
        out_tokens_ptr + bidx * stride_out + offs,
        tl.where(is_root, sample, raw_token),
        mask=in_tree,
    )
    tl.store(
        out_parents_ptr + bidx * stride_out + offs,
        raw_parent_out,
        mask=in_tree,
    )
    tl.store(
        out_lps_ptr + bidx * stride_out + offs,
        tl.where(is_root, 0.0, raw_lp),
        mask=in_tree,
    )

    # Depth follows raw parent links directly, avoiding an intermediate
    # old-to-new mapping tensor.
    cur_old = kept
    active = is_nonroot & kept_valid
    depth = tl.zeros((BLOCK_N,), dtype=tl.int64)
    for _ in tl.static_range(0, MAX_DEPTH):
        depth += active.to(tl.int64)
        parent_old = tl.load(
            raw_parents_ptr + bidx * stride_raw + cur_old,
            mask=active,
            other=-1,
        ).to(tl.int64)
        active = active & (parent_old >= 0)
        cur_old = tl.where(active, parent_old, cur_old)
    tl.store(
        out_depth_ptr + bidx * stride_out + offs,
        depth,
        mask=in_tree,
    )

    # Ancestor mask.  Invalid padded nodes attend only to themselves so the
    # target attention row stays finite, but no topology link can reach them.
    mask_offs = tl.arange(0, BLOCK_N * BLOCK_N)
    q = mask_offs // N
    k = mask_offs - q * N
    mask_in = (q < N) & (k < N)
    q_nonroot = q - 1
    q_is_root = q == 0
    q_kept = tl.load(
        kept_idx_ptr + bidx * stride_kept + q_nonroot,
        mask=(q > 0) & (q < N),
        other=0,
    ).to(tl.int64)
    q_valid = tl.load(
        kept_valid_ptr + bidx * stride_kept + q_nonroot,
        mask=(q > 0) & (q < N),
        other=0,
    ).to(tl.int1)
    k_old = tl.load(
        kept_idx_ptr + bidx * stride_kept + (k - 1),
        mask=(k > 0) & (k < N),
        other=-2,
    ).to(tl.int64)
    visible = (q_is_root & (k == 0)) | ((q > 0) & ~q_valid & (k == q))
    visible |= (q > 0) & q_valid & (k == 0)
    cur_q_old = q_kept
    q_active = (q > 0) & q_valid
    for _ in tl.static_range(0, MAX_DEPTH):
        visible |= q_active & (k > 0) & (k_old == cur_q_old)
        parent_old = tl.load(
            raw_parents_ptr + bidx * stride_raw + cur_q_old,
            mask=q_active,
            other=-1,
        ).to(tl.int64)
        q_active = q_active & (parent_old >= 0)
        cur_q_old = tl.where(q_active, parent_old, cur_q_old)
    tl.store(
        out_mask_ptr + bidx * stride_mask + mask_offs,
        visible.to(tl.float32),
        mask=mask_in,
    )

    # First-child and next-sibling links in topology coordinates (root=0).
    topo_parent = tl.where(
        is_nonroot & kept_valid,
        tl.where(raw_parent_out >= 0, raw_parent_out + 1, 0),
        -1,
    )
    seq_len = tl.load(seq_lens_ptr + bidx).to(tl.int64)
    tl.store(
        out_topo_parents_ptr + bidx * stride_out + offs,
        topo_parent,
        mask=in_tree,
    )
    tl.store(
        out_positions_ptr + bidx * stride_out + offs,
        depth + seq_len,
        mask=in_tree,
    )
    tl.store(
        out_retrieve_index_ptr + bidx * stride_out + offs,
        offs + bidx * N,
        mask=in_tree,
    )
    node_idx = offs
    child_match = (
        (node_idx[None, :] > 0)
        & node_valid[None, :]
        & (topo_parent[None, :] == offs[:, None])
    )
    inf = N + 1
    next_token = tl.min(
        tl.where(child_match, node_idx[None, :], inf), axis=1,
    )
    sibling_match = (
        (node_idx[None, :] > offs[:, None])
        & node_valid[None, :]
        & is_nonroot[:, None]
        & kept_valid[:, None]
        & (topo_parent[None, :] == topo_parent[:, None])
    )
    next_sibling = tl.min(
        tl.where(sibling_match, node_idx[None, :], inf), axis=1,
    )
    next_token = tl.where(next_token < inf, next_token, -1)
    next_sibling = tl.where(next_sibling < inf, next_sibling, -1)
    tl.store(
        out_next_token_ptr + bidx * stride_out + offs,
        next_token,
        mask=in_tree,
    )
    tl.store(
        out_next_sibling_ptr + bidx * stride_out + offs,
        next_sibling,
        mask=in_tree,
    )


@triton.jit
def _pack_depth_topology_kernel(
    raw_tokens_ptr,
    raw_parents_ptr,
    raw_lps_ptr,
    kept_idx_ptr,
    kept_valid_ptr,
    sample_tokens_ptr,
    seq_lens_ptr,
    out_tokens_ptr,
    out_parents_ptr,
    out_topo_parents_ptr,
    out_lps_ptr,
    out_depth_ptr,
    out_positions_ptr,
    out_retrieve_index_ptr,
    stride_raw: tl.constexpr,
    stride_kept: tl.constexpr,
    stride_out: tl.constexpr,
    N: tl.constexpr,
    BUDGET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
):
    """Stage 1: pack values and derive topology without an outer product."""
    b = tl.program_id(0)
    node = tl.arange(0, BLOCK_N)
    nonroot = node - 1
    in_tree = node < N
    is_root = node == 0
    is_nonroot = (node > 0) & in_tree

    kept = tl.load(
        kept_idx_ptr + b * stride_kept + nonroot,
        mask=is_nonroot,
        other=0,
    ).to(tl.int64)
    keep = tl.load(
        kept_valid_ptr + b * stride_kept + nonroot,
        mask=is_nonroot,
        other=0,
    ).to(tl.int1)
    valid_nonroot = is_nonroot & keep

    old_parent = tl.load(
        raw_parents_ptr + b * stride_raw + kept,
        mask=valid_nonroot,
        other=-1,
    ).to(tl.int64)

    # Each node carries only one candidate parent id.  This serial candidate
    # search avoids the BLOCK_N x BLOCK_N SSA tensor in the old kernel.
    mapped_parent = tl.full((BLOCK_N,), -1, tl.int64)
    for candidate in tl.static_range(0, BUDGET):
        candidate_old = tl.load(
            kept_idx_ptr + b * stride_kept + candidate
        ).to(tl.int64)
        candidate_valid = tl.load(
            kept_valid_ptr + b * stride_kept + candidate
        ).to(tl.int1)
        mapped_parent = tl.where(
            valid_nonroot
            & (old_parent >= 0)
            & candidate_valid
            & (old_parent == candidate_old),
            candidate,
            mapped_parent,
        )

    raw_parent_out = tl.where(
        is_root,
        -1,
        tl.where(valid_nonroot, tl.where(old_parent >= 0, mapped_parent, -1), -2),
    )
    topo_parent = tl.where(
        valid_nonroot,
        tl.where(raw_parent_out >= 0, raw_parent_out + 1, 0),
        -1,
    )

    sample = tl.load(sample_tokens_ptr + b)
    token = tl.load(
        raw_tokens_ptr + b * stride_raw + kept,
        mask=valid_nonroot,
        other=0,
    )
    lp = tl.load(
        raw_lps_ptr + b * stride_raw + kept,
        mask=valid_nonroot,
        other=float("-inf"),
    )

    cur_old = kept
    active = valid_nonroot
    depth = tl.zeros((BLOCK_N,), dtype=tl.int64)
    for _ in tl.static_range(0, MAX_DEPTH):
        depth += active.to(tl.int64)
        parent_old = tl.load(
            raw_parents_ptr + b * stride_raw + cur_old,
            mask=active,
            other=-1,
        ).to(tl.int64)
        active = active & (parent_old >= 0)
        cur_old = tl.where(active, parent_old, cur_old)

    seq_len = tl.load(seq_lens_ptr + b).to(tl.int64)
    out = b * stride_out + node
    tl.store(out_tokens_ptr + out, tl.where(is_root, sample, token), mask=in_tree)
    tl.store(out_parents_ptr + out, raw_parent_out, mask=in_tree)
    tl.store(out_topo_parents_ptr + out, topo_parent, mask=in_tree)
    tl.store(out_lps_ptr + out, tl.where(is_root, 0.0, lp), mask=in_tree)
    tl.store(out_depth_ptr + out, depth, mask=in_tree)
    tl.store(out_positions_ptr + out, depth + seq_len, mask=in_tree)
    tl.store(out_retrieve_index_ptr + out, node + b * N, mask=in_tree)


@triton.jit
def _ancestor_mask_row_kernel(
    raw_parents_ptr,
    kept_idx_ptr,
    kept_valid_ptr,
    out_mask_ptr,
    stride_raw: tl.constexpr,
    stride_kept: tl.constexpr,
    stride_mask: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
):
    """Stage 2: one program writes one query's N-wide ancestor mask row."""
    pid = tl.program_id(0)
    b = pid // N
    q = pid - b * N
    key = tl.arange(0, BLOCK_N)
    in_tree = key < N

    q_is_root = q == 0
    q_nonroot = q > 0
    q_old = tl.load(
        kept_idx_ptr + b * stride_kept + (q - 1), mask=q_nonroot, other=0
    ).to(tl.int64)
    q_valid = tl.load(
        kept_valid_ptr + b * stride_kept + (q - 1), mask=q_nonroot, other=0
    ).to(tl.int1)

    key_old = tl.load(
        kept_idx_ptr + b * stride_kept + (key - 1),
        mask=(key > 0) & in_tree,
        other=-2,
    ).to(tl.int64)

    visible = (q_is_root & (key == 0)) | ((q_nonroot & ~q_valid) & (key == q))
    visible |= q_nonroot & q_valid & (key == 0)

    current = q_old
    active = q_nonroot & q_valid
    for _ in tl.static_range(0, MAX_DEPTH):
        visible |= active & (key > 0) & in_tree & (key_old == current)
        parent = tl.load(
            raw_parents_ptr + b * stride_raw + current, mask=active, other=-1
        ).to(tl.int64)
        active = active & (parent >= 0)
        current = tl.where(active, parent, current)

    tl.store(
        out_mask_ptr + b * stride_mask + q * N + key,
        visible,
        mask=in_tree,
    )


@triton.jit
def _topology_links_row_kernel(
    topo_parents_ptr,
    kept_valid_ptr,
    out_next_token_ptr,
    out_next_sibling_ptr,
    stride_kept: tl.constexpr,
    stride_out: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Stage 3: one program finds one node's first child and sibling."""
    pid = tl.program_id(0)
    b = pid // N
    q = pid - b * N
    candidate = tl.arange(0, BLOCK_N)
    in_tree = candidate < N
    candidate_nonroot = (candidate > 0) & in_tree
    candidate_valid = tl.load(
        kept_valid_ptr + b * stride_kept + (candidate - 1),
        mask=candidate_nonroot,
        other=0,
    ).to(tl.int1)
    candidate_parent = tl.load(
        topo_parents_ptr + b * stride_out + candidate,
        mask=in_tree,
        other=-1,
    ).to(tl.int64)
    q_parent = tl.load(topo_parents_ptr + b * stride_out + q).to(tl.int64)
    q_valid = (q == 0) | tl.load(
        kept_valid_ptr + b * stride_kept + (q - 1), mask=q > 0, other=0
    ).to(tl.int1)

    inf = N + 1
    child = tl.min(
        tl.where(
            candidate_nonroot & candidate_valid & (candidate_parent == q),
            candidate,
            inf,
        ),
        axis=0,
    )
    sibling = tl.min(
        tl.where(
            candidate_nonroot
            & candidate_valid
            & (candidate > q)
            & (q > 0)
            & q_valid
            & (candidate_parent == q_parent),
            candidate,
            inf,
        ),
        axis=0,
    )
    out = b * stride_out + q
    tl.store(out_next_token_ptr + out, tl.where(child < inf, child, -1))
    tl.store(out_next_sibling_ptr + out, tl.where(sibling < inf, sibling, -1))

def _finalize_one_block_fixed_metax(
    tree_buf_b,
    kept_idx,
    kept_valid,
    sample_tokens_b,
    seq_lens_b,
    out,
    budget,
):
    """Build fixed topology with three low-memory Triton stages on C500."""
    batch_size = kept_idx.shape[0]
    tree_tokens = budget + 1
    block_n = triton.next_power_of_2(tree_tokens)
    _pack_depth_topology_kernel[(batch_size,)](
        tree_buf_b["tokens"],
        tree_buf_b["parents"],
        tree_buf_b["lps"],
        kept_idx,
        kept_valid,
        sample_tokens_b[:, 0],
        seq_lens_b,
        out["tokens"],
        out["parents"],
        out["topo_parents"],
        out["lps"],
        out["depth"],
        out["positions"],
        out["retrieve_index"],
        tree_buf_b["tokens"].shape[1],
        budget,
        tree_tokens,
        N=tree_tokens,
        BUDGET=budget,
        BLOCK_N=block_n,
        MAX_DEPTH=MAX_TREE_DEPTH,
        num_warps=4,
        num_stages=1,
    )
    _ancestor_mask_row_kernel[(batch_size * tree_tokens,)](
        tree_buf_b["parents"],
        kept_idx,
        kept_valid,
        out["mask"],
        tree_buf_b["tokens"].shape[1],
        budget,
        tree_tokens * tree_tokens,
        N=tree_tokens,
        BLOCK_N=block_n,
        MAX_DEPTH=MAX_TREE_DEPTH,
        num_warps=4,
        num_stages=1,
    )
    _topology_links_row_kernel[(batch_size * tree_tokens,)](
        out["topo_parents"],
        kept_valid,
        out["next_token"],
        out["next_sibling"],
        budget,
        tree_tokens,
        N=tree_tokens,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=1,
    )

def finalize_one_block_fixed_batched(
    tree_buf_b: dict,
    raw_sizes_b: torch.Tensor,
    sample_tokens_b: torch.Tensor,
    budget: int,
    raw_capacity: int,
    sb_batch: dict,
    active_bs: Optional[int] = None,
    dedup_depth: int = 4,
    seq_lens_b: Optional[torch.Tensor] = None,
) -> list:
    """Finalize fixed-width trees without host metadata synchronization."""
    device = tree_buf_b["tokens"].device
    B = int(sample_tokens_b.shape[0])
    if active_bs is None:
        active_bs = B
    if not 0 < active_bs <= B:
        raise RuntimeError(
            f"Invalid active batch {active_bs} for tree capacity {B}."
        )
    N = budget + 1
    raw_n = (
        raw_sizes_b[:B, 0] if raw_sizes_b.dim() > 1 else raw_sizes_b[:B]
    ).long()
    if seq_lens_b is None:
        zero_seq_lens = sb_batch.get("zero_seq_lens")
        if zero_seq_lens is None or zero_seq_lens.shape[0] < B:
            zero_seq_lens = torch.zeros(B, dtype=torch.long, device=device)
            sb_batch["zero_seq_lens"] = zero_seq_lens
        seq_lens_b = zero_seq_lens[:B]
    elif seq_lens_b.shape != (B,):
        raise RuntimeError(
            f"Invalid fixed-tree seq_lens shape {tuple(seq_lens_b.shape)} for B={B}."
        )
    if not 0 < dedup_depth <= MAX_TREE_DEPTH:
        raise RuntimeError(f"Invalid SpecBlock dedup depth {dedup_depth}.")

    # The HF production builder merges the same block-1/block-2 collisions
    # before pruning.  Apply the equivalent depth-ordered merge to every
    # request row: the highest-score duplicate survives, its children are
    # redirected to that survivor, and the freed W90 slots are refilled by
    # the next highest-score unique paths.
    dedup_block_n = triton.next_power_of_2(raw_capacity)
    # Exact fixed B1 W90 geometry uses an 89-node non-root budget and a
    # 769-node raw buffer: one block-1 tree plus 16 expanded roots, each
    # contributing at most top-k=10 nodes at a depth. Including block 1, the
    # structural depth width is <= 17 * 10 = 170, rounded to 256 lanes.
    # Other geometries retain their full raw width.
    dedup_compact_n = (
        256
        if budget == 89 and raw_capacity == 769 and dedup_depth == 8
        else dedup_block_n
    )
    dedup_scratch = sb_batch.get("dedup_scratch")
    dedup_shape = (B, dedup_block_n, dedup_compact_n)
    if dedup_scratch is None or dedup_scratch["shape"] != dedup_shape:
        dedup_scratch = {
            "shape": dedup_shape,
            "compact_packed": torch.empty(
                (B, dedup_compact_n), dtype=torch.long, device=device,
            ),
            "combined_winner": torch.empty(
                (B, dedup_compact_n), dtype=torch.long, device=device,
            ),
            "node_state": torch.empty(
                (B, dedup_block_n), dtype=torch.long, device=device,
            ),
        }
        sb_batch["dedup_scratch"] = dedup_scratch
    _dedup_tree_rows_by_depth_kernel[(B,)](
        tree_buf_b["tokens"],
        tree_buf_b["parents"],
        tree_buf_b["lps"],
        raw_n,
        dedup_scratch["compact_packed"],
        dedup_scratch["combined_winner"],
        dedup_scratch["node_state"],
        tree_buf_b["tokens"].shape[1],
        dedup_compact_n,
        N=dedup_block_n,
        COMPACT_N=dedup_compact_n,
        MAX_DEDUP_DEPTH=dedup_depth,
        MAX_WALK_DEPTH=MAX_TREE_DEPTH,
        num_warps=4,
    )

    score_width = max(raw_capacity, budget)
    raw_lps = tree_buf_b["lps"][:B, :raw_capacity]
    if score_width > raw_capacity:
        scores = F.pad(
            raw_lps,
            (0, score_width - raw_capacity),
            value=float("-inf"),
        )
    else:
        scores = raw_lps
    valid_raw = (
        torch.arange(score_width, device=device).unsqueeze(0)
        < raw_n.unsqueeze(1)
    )
    scores = torch.where(
        valid_raw, scores, torch.full_like(scores, float("-inf")),
    )
    kept_idx = torch.topk(scores, budget, dim=1, largest=True).indices
    kept_idx = kept_idx.sort(dim=1).values
    kept_scores = torch.gather(scores, 1, kept_idx)
    kept_valid = torch.isfinite(kept_scores)

    # Cache one output set per active-batch shape.  Active shrink revisits the
    # same shapes across requests/repeats, so a single mutable slot would force
    # allocator traffic whenever B changes.
    fast_outputs = sb_batch.setdefault("fast_fin_outputs", {})
    shape_key = (B, N)
    out = fast_outputs.get(shape_key)
    if out is None:
        out = {
            "tokens": torch.empty(B, N, dtype=torch.long, device=device),
            "parents": torch.empty(B, N, dtype=torch.long, device=device),
            "topo_parents": torch.empty(B, N, dtype=torch.long, device=device),
            "lps": torch.empty(B, N, dtype=torch.float32, device=device),
            "depth": torch.empty(B, N, dtype=torch.long, device=device),
            "positions": torch.empty(B, N, dtype=torch.long, device=device),
            "mask": torch.empty(B, N, N, dtype=torch.bool, device=device),
            "retrieve_index": torch.empty(B, N, dtype=torch.long, device=device),
            "next_token": torch.empty(B, N, dtype=torch.long, device=device),
            "next_sibling": torch.empty(B, N, dtype=torch.long, device=device),
        }
        fast_outputs[shape_key] = out

    if is_metax_c500():
        _finalize_one_block_fixed_metax(
            tree_buf_b,
            kept_idx,
            kept_valid,
            sample_tokens_b,
            seq_lens_b,
            out,
            budget,
        )
    else:
        block_n = triton.next_power_of_2(N)
        _finalize_one_block_fixed_kernel[(B,)](
            tree_buf_b["tokens"], tree_buf_b["parents"], tree_buf_b["lps"],
            kept_idx, kept_valid, sample_tokens_b[:, 0], seq_lens_b,
            out["tokens"], out["parents"], out["topo_parents"],
            out["lps"], out["depth"], out["positions"], out["mask"],
            out["retrieve_index"], out["next_token"], out["next_sibling"],
            tree_buf_b["tokens"].shape[1], budget, N, N * N,
            N=N, BUDGET=budget, BLOCK_N=block_n, MAX_DEPTH=MAX_TREE_DEPTH,
        )

    return pack_fixed_tree_outputs(
        sb_batch=sb_batch,
        batch_capacity=B,
        active_bs=active_bs,
        tree_tokens=N,
    )


def pack_fixed_tree_outputs(
    sb_batch: dict,
    batch_capacity: int,
    active_bs: int,
    tree_tokens: int,
) -> list:
    """Package captured fixed-width output buffers without launching kernels."""
    if not 0 < active_bs <= batch_capacity:
        raise RuntimeError(
            f"Invalid active batch {active_bs} for tree capacity {batch_capacity}."
        )
    out = sb_batch["fast_fin_outputs"][(batch_capacity, tree_tokens)]
    active_out = {name: tensor[:active_bs] for name, tensor in out.items()}
    empty_ri = sb_batch.get("empty_retrieve_index")
    if empty_ri is None:
        empty_ri = torch.empty(
            0,
            MAX_TREE_DEPTH + 1,
            dtype=torch.long,
            device=out["tokens"].device,
        )
        sb_batch["empty_retrieve_index"] = empty_ri

    trees = []
    for b in range(active_bs):
        depth = active_out["depth"][b]
        trees.append({
            "draft_token": active_out["tokens"][b],
            "parents": active_out["parents"][b],
            "depth": depth,
            "cum_log_prob": active_out["lps"][b],
            "retrieve_index": empty_ri,
            "tree_mask": active_out["mask"][b],
            "position_ids": depth,
            "retrive_next_token": active_out["next_token"][b],
            "retrive_next_sibling": active_out["next_sibling"][b],
            # Worker consumes active request-major zero-copy views.
            "_batched_fixed_outputs": active_out,
        })
    return trees


def finalize_tree_fixed_batched(
    tree_buf_b: dict,
    raw_sizes_b: torch.Tensor,
    sample_tokens_b: torch.Tensor,
    budget: int,
    raw_capacity: int,
    scratch: dict,
    active_bs: int,
    dedup_depth: int = 4,
    seq_lens_b: Optional[torch.Tensor] = None,
) -> list:
    """Finalize one- or multi-block capacity rows at a fixed verify width."""
    return finalize_one_block_fixed_batched(
        tree_buf_b=tree_buf_b,
        raw_sizes_b=raw_sizes_b,
        sample_tokens_b=sample_tokens_b,
        budget=budget,
        raw_capacity=raw_capacity,
        sb_batch=scratch,
        active_bs=active_bs,
        dedup_depth=dedup_depth,
        seq_lens_b=seq_lens_b,
    )


def gpu_prune_tree(tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots,
                   n_nodes, budget, max_tree_depth=MAX_TREE_DEPTH):
    """GPU-side prune via top-K seeds + ancestor closure.

    Returns compacted (tokens, parents, lps, ranks, blocks, slots, new_n)
    on GPU; parents are re-indexed to the new compact positions.
    """
    device = tn_tokens.device
    assert n_nodes > budget, "caller must check; this path only runs when prune needed"

    topk_n = min(budget, n_nodes)
    lps_slice = tn_lps[:n_nodes]
    top_idx = torch.topk(lps_slice, topk_n, largest=True).indices

    keep = torch.zeros(n_nodes, dtype=torch.int32, device=device)
    keep[top_idx] = 1

    _ancestor_closure_kernel[(n_nodes,)](
        tn_parents[:n_nodes].contiguous(),
        keep,
        n_nodes,
        MAX_DEPTH=max_tree_depth,
    )

    keep_bool = keep.bool()
    kept_idx = keep_bool.nonzero(as_tuple=True)[0]
    new_n = kept_idx.shape[0]

    old_to_new = torch.full((n_nodes,), -1, dtype=torch.int64, device=device)
    old_to_new[kept_idx] = torch.arange(new_n, device=device, dtype=torch.int64)

    new_tokens = tn_tokens[kept_idx]
    new_lps = tn_lps[kept_idx]
    new_ranks = tn_ranks[kept_idx]
    new_blocks = tn_blocks[kept_idx]
    new_slots = tn_slots[kept_idx]

    old_parents = tn_parents[kept_idx]
    new_parents = torch.where(
        old_parents >= 0,
        old_to_new[old_parents.clamp(min=0)],
        torch.full_like(old_parents, -1),
    )

    return new_tokens, new_parents, new_lps, new_ranks, new_blocks, new_slots, new_n


# =============================================================================
#   Sort retrieve_indices rows lexicographically (small CPU sort)
# =============================================================================

def _sort_retrieve_indices_gpu(ri: torch.Tensor, maxitem: int) -> torch.Tensor:
    """Lex-sort rows of retrieve_indices, treating -1 as +inf.

    ri: [num_leaves, D] int32 on GPU. Returns sorted ri (same shape).

    Pure-GPU lex sort: encode each row as a single int64 hash key
    ``sum(c_d * (maxitem+1)^(D-1-d))`` (treating -1 as ``maxitem``).
    For typical D <= 12 and maxitem <= 200, the hash fits comfortably in
    int64 (200**12 ≈ 4e27, larger than int64 — but we use float64 keys
    instead for the lex order).  ``torch.argsort`` then orders rows.

    Replaces the prior CPU-side ``.cpu().tolist() + sort`` which was
    the last remaining host sync in finalize_tree_gpu.
    """
    if ri.numel() == 0:
        return ri.to(torch.long)
    num_leaves, D = ri.shape
    maxitem_v = int(maxitem)
    # Map -1 → maxitem (so "-1 is +inf" lex semantics).
    ri_int = ri.to(torch.long)
    masked = torch.where(ri_int >= 0, ri_int, torch.full_like(ri_int, maxitem_v))
    # Lex key: row-wise weighted sum.  Use float64 to fit large powers
    # without overflow at D=12.
    weights = torch.tensor(
        [(maxitem_v + 1) ** (D - 1 - d) for d in range(D)],
        dtype=torch.float64, device=ri.device,
    )  # [D]
    keys = (masked.to(torch.float64) * weights.unsqueeze(0)).sum(dim=1)  # [num_leaves]
    order = torch.argsort(keys, stable=True)
    return ri_int[order]


# =============================================================================
#   Finalize: prune + topology + retrieve_index
# =============================================================================

def finalize_tree_gpu(
    tree_buf: dict,
    n_nodes: int,
    sample_token: torch.Tensor,    # [1] long on GPU
    budget: int,
    K: int,
    max_blocks: int,
    max_tree_depth: int = MAX_TREE_DEPTH,
):
    """End-to-end GPU finalize. Returns dict with the SGLang tree fields:

      draft_token      [Np1] long
      parents          [Np1] long  (root parent = -1)
      depth            [Np1] long
      cum_log_prob     [Np1] float32
      retrieve_index   [num_leaves, max_depth+1] long  (sorted, -1 padded)
      tree_mask        [Np1, Np1] float32  (1.0 = ancestor)
      position_ids     alias of depth
    """
    device = tree_buf['tokens'].device
    tn_tokens = tree_buf['tokens']
    tn_parents = tree_buf['parents']
    tn_lps = tree_buf['lps']
    tn_ranks = tree_buf['ranks']
    tn_blocks = tree_buf['blocks']
    tn_slots = tree_buf['slots']

    N = n_nodes
    if N > budget:
        (tn_tokens, tn_parents, tn_lps,
         tn_ranks, tn_blocks, tn_slots, N) = gpu_prune_tree(
            tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots,
            N, budget, max_tree_depth=max_tree_depth,
        )
    else:
        tn_tokens = tn_tokens[:N].contiguous()
        tn_parents = tn_parents[:N].contiguous()
        tn_lps = tn_lps[:N]
        tn_ranks = tn_ranks[:N]
        tn_blocks = tn_blocks[:N]
        tn_slots = tn_slots[:N]

    Np1 = N + 1
    parent_t = torch.empty(Np1, dtype=torch.int32, device=device)
    parent_t[0] = 0
    if N > 0:
        parent_t[1:] = torch.where(
            tn_parents[:N] >= 0,
            (tn_parents[:N] + 1).to(torch.int32),
            torch.zeros(N, dtype=torch.int32, device=device),
        )

    depth_t = torch.zeros(Np1, dtype=torch.int32, device=device)
    mask_flat = torch.zeros(Np1 * Np1, dtype=torch.float32, device=device)

    _tree_depth_mask_kernel[(Np1,)](
        parent_t, depth_t, mask_flat,
        Np1, MAX_DEPTH=max_tree_depth,
    )

    is_parent = torch.zeros(Np1, dtype=torch.bool, device=device)
    if N > 0:
        is_parent.scatter_(0, parent_t[1:].long(), True)
    leaves = (~is_parent).nonzero(as_tuple=True)[0].to(torch.int32)
    num_leaves = int(leaves.shape[0])

    ri = torch.full((max(num_leaves, 1), max_tree_depth + 1), -1,
                    dtype=torch.int32, device=device)
    if num_leaves > 0:
        _tree_retrieve_kernel[(num_leaves,)](
            parent_t, depth_t, leaves, ri,
            max_tree_depth, Np1, num_leaves,
            MAX_DEPTH=max_tree_depth,
        )
        ri_sorted = _sort_retrieve_indices_gpu(ri, maxitem=Np1 + 5)
    else:
        ri_sorted = torch.empty(0, max_tree_depth + 1, dtype=torch.long, device=device)

    draft_tokens = torch.empty(Np1, dtype=torch.long, device=device)
    draft_tokens[0] = sample_token.squeeze()
    if N > 0:
        draft_tokens[1:] = tn_tokens

    tree_mask = mask_flat.reshape(Np1, Np1)
    tree_position_ids = depth_t.long()

    parents_full = torch.full((Np1,), -1, dtype=torch.long, device=device)
    if N > 0:
        parents_full[1:] = tn_parents.long()

    cum_lp_full = torch.zeros(Np1, dtype=torch.float32, device=device)
    if N > 0:
        cum_lp_full[1:] = tn_lps.float()

    return {
        "draft_token": draft_tokens,
        "parents": parents_full,
        "depth": tree_position_ids,
        "cum_log_prob": cum_lp_full,
        "retrieve_index": ri_sorted,
        "tree_mask": tree_mask,
        "position_ids": tree_position_ids,
    }


# =============================================================================
#   Batched finalize (S3c)
# =============================================================================

def _alloc_finalize_scratch(
    sb_batch: dict, B: int, max_Np1: int, max_tree_depth: int, device: torch.device,
):
    """Lazy-init [B, max_Np1] etc. batched finalize buffers in sb_batch."""
    cur_B = sb_batch.get('fin_B', 0)
    cur_NP1 = sb_batch.get('fin_max_Np1', 0)
    cur_D = sb_batch.get('fin_max_depth', 0)
    need = cur_B < B or cur_NP1 < max_Np1 or cur_D < max_tree_depth

    if need or 'fin_parent' not in sb_batch:
        sb_batch['fin_parent'] = torch.zeros(B, max_Np1, dtype=torch.int32, device=device)
        sb_batch['fin_depth'] = torch.zeros(B, max_Np1, dtype=torch.int32, device=device)
        sb_batch['fin_mask'] = torch.zeros(B, max_Np1, max_Np1, dtype=torch.float32, device=device)
        sb_batch['fin_is_parent'] = torch.zeros(B, max_Np1, dtype=torch.bool, device=device)
        sb_batch['fin_leaves'] = torch.zeros(B, max_Np1, dtype=torch.int32, device=device)
        sb_batch['fin_num_leaves'] = torch.zeros(B, dtype=torch.int32, device=device)
        sb_batch['fin_ri'] = torch.full(
            (B, max_Np1, max_tree_depth + 1), -1, dtype=torch.int32, device=device,
        )
        sb_batch['fin_B'] = B
        sb_batch['fin_max_Np1'] = max_Np1
        sb_batch['fin_max_depth'] = max_tree_depth


def _sort_retrieve_indices_gpu_batched(
    ri_b: torch.Tensor, num_leaves_b: torch.Tensor, maxitem: int,
) -> torch.Tensor:
    """Batched lex-sort of retrieve_indices rows.

    ri_b: [B, max_Np1, D] int32.  num_leaves_b: [B] int32 — only first
    num_leaves[b] rows are real; pad rows have all -1 from caller init.
    Returns [B, max_Np1, D] long with each [b, :num_leaves[b]] sorted by
    lex key (treating -1 as +inf).
    """
    B, max_NL, D = ri_b.shape
    if B == 0 or max_NL == 0:
        return ri_b.to(torch.long)
    device = ri_b.device
    ri_l = ri_b.to(torch.long)
    masked = torch.where(ri_l >= 0, ri_l, torch.full_like(ri_l, maxitem))
    weights = torch.tensor(
        [(maxitem + 1) ** (D - 1 - d) for d in range(D)],
        dtype=torch.float64, device=device,
    )                                                                    # [D]
    keys = (masked.to(torch.float64) * weights.view(1, 1, D)).sum(dim=2)  # [B, max_NL]
    # Pad rows (lid >= num_leaves) are full of -1 → key = D * maxitem (huge);
    # they sort last naturally.  Plus we override to ensure they're stable.
    arange_nl = torch.arange(max_NL, device=device).unsqueeze(0).expand(B, -1)
    pad_mask = arange_nl >= num_leaves_b.unsqueeze(1).long()
    keys = torch.where(pad_mask, torch.full_like(keys, float('inf')), keys)
    order = torch.argsort(keys, dim=1, stable=True)                      # [B, max_NL]
    sorted_ri = torch.gather(
        ri_l, dim=1, index=order.unsqueeze(-1).expand(-1, -1, D),
    )
    return sorted_ri


def finalize_tree_gpu_batched(
    tree_buf_b: dict,                       # batched [B, MAX_NODES] views
    n_nodes_per_req_cpu: list,              # length B; post-prune sizes (already on host)
    pruned_tensors_per_req: list,           # length B; dict of new tensors if prune ran, else None
    batched_pruned: Optional[dict],          # [B, budget] tensors when every row pruned
    sample_tokens_b: torch.Tensor,          # [B, 1] long
    budget: int,
    K: int,
    max_blocks: int,
    sb_batch: dict,
    device: torch.device,
    max_tree_depth: int = MAX_TREE_DEPTH,
    need_retrieve_index: bool = True,
) -> list:
    """Batched end-to-end finalize across B reqs.

    Replaces B per-req ``finalize_tree_gpu`` calls (~5 kernel launches each
    + many tensor allocs) with a single batched depth_mask + retrieve +
    sort + per-req output dict construction.  Prune is still run per-req
    upstream when needed (its work depends on per-req N and is rare-path
    for non-saturated trees).

    Returns a list of B tree dicts with the same schema as
    :func:`finalize_tree_gpu`.
    """
    B = len(n_nodes_per_req_cpu)
    max_Np1 = budget + 1

    _alloc_finalize_scratch(sb_batch, B, max_Np1, max_tree_depth, device)
    parent_t_b = sb_batch['fin_parent']        # [B, max_Np1] i32
    depth_t_b = sb_batch['fin_depth']          # [B, max_Np1] i32
    mask_b = sb_batch['fin_mask']              # [B, max_Np1, max_Np1] f32
    is_parent_b = sb_batch['fin_is_parent']    # [B, max_Np1] bool
    leaves_b = sb_batch['fin_leaves']          # [B, max_Np1] i32
    num_leaves_b = sb_batch['fin_num_leaves']  # [B] i32
    ri_b = sb_batch['fin_ri']                  # [B, max_Np1, max_depth+1] i32

    # Zero relevant slices [:B] only (buffers may be over-allocated for B_max).
    parent_t_b[:B].zero_()
    depth_t_b[:B].zero_()
    mask_b[:B].zero_()
    if need_retrieve_index:
        is_parent_b[:B].zero_()
        ri_b[:B].fill_(-1)

    # Fill parent_t_b from post-prune parents.  The common saturated path
    # prunes every row to the same budget, so transform the whole batch in one
    # operation instead of launching B independent torch.where kernels.
    if batched_pruned is not None:
        tn_p_b = batched_pruned['parents'][:B, :budget]
        parent_t_b[:B, 1:budget + 1] = torch.where(
            tn_p_b >= 0,
            (tn_p_b + 1).to(torch.int32),
            torch.zeros_like(tn_p_b, dtype=torch.int32),
        )
    else:
        for b in range(B):
            N_b = n_nodes_per_req_cpu[b]
            if N_b == 0:
                continue
            pruned = pruned_tensors_per_req[b]
            if pruned is not None:
                tn_p = pruned['parents'][:N_b]
            else:
                tn_p = tree_buf_b['parents'][b, :N_b]
            parent_t_b[b, 1:N_b + 1] = torch.where(
                tn_p >= 0,
                (tn_p + 1).to(torch.int32),
                torch.zeros(N_b, dtype=torch.int32, device=device),
            )
            # pad positions [N_b+1, max_Np1) already 0 from zero_() — the
            # kernel walks parent=0 (self-root) for these, finishes fast,
            # and writes are discarded when caller slices [:Np1_b].

    # Launch batched depth_mask kernel.
    # sb_batch buffers are over-allocated to B_max-ever-seen; slice to current B
    # before reshape so view() succeeds and bidx range matches.
    parent_t_view = parent_t_b[:B]
    depth_t_view = depth_t_b[:B]
    mask_view = mask_b[:B].reshape(B, -1)
    is_parent_view = is_parent_b[:B]
    leaves_view = leaves_b[:B]
    num_leaves_view = num_leaves_b[:B]
    ri_view_all = ri_b[:B]
    _tree_depth_mask_kernel_batched[(B, max_Np1)](
        parent_t_view, depth_t_view, mask_view,
        max_Np1, max_Np1 * max_Np1,
        MAX_DEPTH=max_tree_depth,
    )

    if need_retrieve_index:
        # Extract leaves per req via vectorized torch ops.  is_parent[b, p] =
        # True iff some node i in (0, Np1_b) has parent_t[b, i] = p.
        # Cap mask so pad rows don't pollute is_parent.
        arange_np1 = torch.arange(max_Np1, device=device).unsqueeze(0).expand(B, -1)
        np1_per_req_t = torch.tensor(
            [n_nodes_per_req_cpu[b] + 1 for b in range(B)],
            device=device, dtype=torch.long,
        )
        valid_node_mask = arange_np1 < np1_per_req_t.unsqueeze(1)
        valid_for_scatter = valid_node_mask.clone()
        valid_for_scatter[:, 0] = False
        bidx_grid = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_Np1)
        parent_idx_2d = parent_t_view.long()
        sel_b = bidx_grid[valid_for_scatter]
        sel_p = parent_idx_2d[valid_for_scatter]
        if sel_b.numel() > 0:
            is_parent_view[sel_b, sel_p] = True

        is_leaf_b = (~is_parent_view) & valid_node_mask
        leaf_positions = torch.where(
            is_leaf_b, arange_np1, torch.full_like(arange_np1, max_Np1),
        )
        leaves_sorted, _ = leaf_positions.sort(dim=1)
        leaves_view.copy_(leaves_sorted.to(torch.int32))
        num_leaves_view.copy_(is_leaf_b.sum(dim=1).to(torch.int32))

        _tree_retrieve_kernel_batched[(B, max_Np1)](
            parent_t_view, depth_t_view, leaves_view, num_leaves_view, ri_view_all,
            max_tree_depth, max_Np1, max_Np1 * (max_tree_depth + 1),
            MAX_DEPTH=max_tree_depth,
        )
        ri_sorted_b = _sort_retrieve_indices_gpu_batched(
            ri_view_all, num_leaves_view, max_Np1 + 5,
        )
        # Only the generic tree API needs leaf counts on the host.  The SGLang
        # SpecBlock verifier consumes parent links directly and skips this sync.
        num_leaves_cpu = num_leaves_view.cpu().tolist()
    else:
        ri_sorted_b = None
        num_leaves_cpu = None

    # Materialize fresh batched outputs once.  Returning views keeps the same
    # per-request API while avoiding B independent allocations/clones for every
    # field.  These tensors are distinct from reusable sb_batch scratch, so the
    # next draft iteration cannot overwrite data still consumed by verify.
    depth_out_b = depth_t_view.long()
    mask_out_b = mask_b[:B].clone()
    draft_tokens_b = torch.empty(
        B, max_Np1, dtype=torch.long, device=device,
    )
    parents_full_b = torch.full(
        (B, max_Np1), -1, dtype=torch.long, device=device,
    )
    cum_lp_full_b = torch.zeros(
        B, max_Np1, dtype=torch.float32, device=device,
    )
    draft_tokens_b[:, 0] = sample_tokens_b[:, 0]

    if need_retrieve_index:
        ri_out_b = ri_sorted_b.clone()
        empty_ri = None
    else:
        ri_out_b = None
        empty_ri = torch.empty(
            0, max_tree_depth + 1, dtype=torch.long, device=device,
        )

    if batched_pruned is not None:
        draft_tokens_b[:, 1:budget + 1] = batched_pruned['tokens'][:B, :budget]
        parents_full_b[:, 1:budget + 1] = batched_pruned['parents'][:B, :budget]
        cum_lp_full_b[:, 1:budget + 1] = batched_pruned['lps'][:B, :budget]
    else:
        for b in range(B):
            N_b = n_nodes_per_req_cpu[b]
            if N_b == 0:
                continue
            pruned = pruned_tensors_per_req[b]
            if pruned is not None:
                tn_tokens = pruned['tokens'][:N_b]
                tn_parents = pruned['parents'][:N_b]
                tn_lps = pruned['lps'][:N_b]
            else:
                tn_tokens = tree_buf_b['tokens'][b, :N_b]
                tn_parents = tree_buf_b['parents'][b, :N_b]
                tn_lps = tree_buf_b['lps'][b, :N_b]

            draft_tokens_b[b, 1:N_b + 1] = tn_tokens
            parents_full_b[b, 1:N_b + 1] = tn_parents.long()
            cum_lp_full_b[b, 1:N_b + 1] = tn_lps.float()

    trees = []
    for b in range(B):
        Np1_b = n_nodes_per_req_cpu[b] + 1
        depth_per_req = depth_out_b[b, :Np1_b]
        if need_retrieve_index:
            nl = num_leaves_cpu[b]
            ri_per_req = ri_out_b[b, :max(nl, 1), :]
        else:
            ri_per_req = empty_ri
        trees.append({
            "draft_token": draft_tokens_b[b, :Np1_b],
            "parents": parents_full_b[b, :Np1_b],
            "depth": depth_per_req,
            "cum_log_prob": cum_lp_full_b[b, :Np1_b],
            "retrieve_index": ri_per_req,
            "tree_mask": mask_out_b[b, :Np1_b, :Np1_b],
            "position_ids": depth_per_req,
        })

    return trees


# =============================================================================
#   GPU-only verify-side helpers (replace numpy parent walking in worker.verify)
# =============================================================================

def build_retrieve_links_gpu(
    trees: list,        # list[dict] with key "parents" (raw, root=-1) on GPU
    device: torch.device,
    N_full: int,
):
    """Build (retrive_next_token, retrive_next_sibling) for tree verification.

    All-GPU: stacks per-req parents, converts raw → topo (root=0), then uses
    scatter_reduce_amin and a 3D mask over [bs, N, N] (≤ ~130KB at bs=8,
    N=90) to compute the child linked list in vectorised torch ops.

    Returns
    -------
    retrive_next_token : LongTensor [bs, N_full] — first child of each parent
    retrive_next_sibling : LongTensor [bs, N_full] — next sibling of each node
    Both use -1 for "no entry".
    """
    bs = len(trees)
    parents_all = torch.stack(
        [t["parents"][:N_full].long().to(device) for t in trees], dim=0,
    )  # [bs, N]
    topo_parent = torch.where(
        parents_all == -1, torch.zeros_like(parents_all), parents_all + 1,
    )

    j_idx = torch.arange(N_full, device=device).unsqueeze(0).expand(bs, -1)  # [bs, N]
    INF = N_full + 1

    # next_token[ridx, p] = min j > 0 with topo_parent[ridx, j] == p
    flat_idx_tok = (
        torch.arange(bs, device=device).view(-1, 1) * N_full + topo_parent
    )  # [bs, N]
    j_vals_for_tok = torch.where(
        j_idx >= 1, j_idx, torch.full_like(j_idx, INF),
    )
    next_tok_buf = torch.full(
        (bs * N_full,), INF, dtype=torch.long, device=device,
    )
    next_tok_buf.scatter_reduce_(
        0, flat_idx_tok.flatten(), j_vals_for_tok.flatten(),
        reduce='amin', include_self=True,
    )
    next_tok_buf = next_tok_buf.reshape(bs, N_full)
    retrive_next_token = torch.where(
        next_tok_buf == INF, torch.full_like(next_tok_buf, -1), next_tok_buf,
    )

    # next_sibling[ridx, j] = min j' > j with topo_parent[ridx, j'] == topo_parent[ridx, j]
    # 3D match: [bs, N, N]. At bs=8, N=90 → 130KB int64 = trivial.
    p_2d = topo_parent.unsqueeze(2)             # [bs, N, 1]
    pp_2d = topo_parent.unsqueeze(1)            # [bs, 1, N]
    j_2d = j_idx.unsqueeze(2)                   # [bs, N, 1]
    jp_2d = j_idx.unsqueeze(1)                  # [bs, 1, N]

    mask_sib = (jp_2d > j_2d) & (p_2d == pp_2d) & (j_2d >= 1)
    INF_t = torch.full_like(jp_2d, INF)
    candidate_jp = torch.where(mask_sib, jp_2d, INF_t)
    min_jp = candidate_jp.min(dim=2).values     # [bs, N]
    retrive_next_sibling = torch.where(
        min_jp == INF, torch.full_like(min_jp, -1), min_jp,
    )

    return retrive_next_token, retrive_next_sibling


def pad_tree_gpu(tree: dict, N_target: int) -> dict:
    """Pad / truncate tree to exactly N_target nodes — no Python for loop.

    Mirrors `worker._pad_tree` but uses vectorised torch ops:
      * pad tokens with 0
      * pad parents with -1 (-> root in topo)
      * pad depth with 1
      * pad cum_log_prob with -1e9
      * extend tree_mask: padded nodes attend to root (col 0) + self only
    """
    N_cur = int(tree["draft_token"].shape[0])
    if N_cur >= N_target:
        for k in ("draft_token", "depth"):
            tree[k] = tree[k][:N_target]
        if "position_ids" in tree:
            tree["position_ids"] = tree["position_ids"][:N_target]
        tree["parents"] = tree["parents"][:N_target]
        tree["cum_log_prob"] = tree["cum_log_prob"][:N_target]
        tree["tree_mask"] = tree["tree_mask"][:N_target, :N_target]
        return tree

    n_pad = N_target - N_cur
    device = tree["draft_token"].device

    pad_tokens = torch.zeros(n_pad, dtype=tree["draft_token"].dtype, device=device)
    tree["draft_token"] = torch.cat([tree["draft_token"], pad_tokens], dim=0)
    pad_parents = torch.full((n_pad,), -1, dtype=tree["parents"].dtype, device=device)
    tree["parents"] = torch.cat([tree["parents"], pad_parents], dim=0)
    pad_depth = torch.ones(n_pad, dtype=tree["depth"].dtype, device=device)
    tree["depth"] = torch.cat([tree["depth"], pad_depth], dim=0)
    if "position_ids" in tree:
        pos_pad = torch.ones(n_pad, dtype=tree["position_ids"].dtype, device=device)
        tree["position_ids"] = torch.cat([tree["position_ids"], pos_pad], dim=0)
    pad_lp = torch.full(
        (n_pad,), -1e9, dtype=tree["cum_log_prob"].dtype, device=device,
    )
    tree["cum_log_prob"] = torch.cat([tree["cum_log_prob"], pad_lp], dim=0)

    # tree_mask: extend [N_cur, N_cur] → [N_target, N_target]. Padded nodes
    # attend root (col 0) + self only — no Python for loop.
    old_mask = tree["tree_mask"]
    new_mask = torch.zeros(
        (N_target, N_target), dtype=old_mask.dtype, device=device,
    )
    new_mask[:N_cur, :N_cur] = old_mask
    pad_rows = torch.arange(N_cur, N_target, device=device)
    fill_val = torch.ones((), dtype=old_mask.dtype, device=device)
    new_mask[pad_rows, 0] = fill_val
    new_mask[pad_rows, pad_rows] = fill_val
    tree["tree_mask"] = new_mask
    return tree


# =============================================================================
#   Public driver: GPU-only end-to-end build_tree (matches old Python signature)
# =============================================================================

def build_tree_gpu(
    draft_model,
    block1_logits: torch.Tensor,                                # [1, K, V_draft]
    block1_rank_logits: torch.Tensor,                           # [1, K, rank_classes]
    block1_hidden: torch.Tensor,                                # [1, K, H]
    block1_ttt_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    initial_input_id: torch.Tensor,                             # [1, 1]
    cross_kv_cache: List[List],
    position_id: int,
    K: int,
    max_blocks: int,
    beam_width: int,
    total_tokens: int,
    rank_classes: int,
    rank_slot_topk: Tuple[int, ...],
    rank_to_factor: Tuple[int, ...],
    d2t: Optional[torch.Tensor],
    scratch: dict,
):
    """End-to-end GPU tree builder for a single batch row.

    `scratch` is a per-worker dict where we lazy-init persistent GPU buffers
    (tree_buf, pend_buf, sizes4, sizes1, cum_alt_buf, rank_to_factor_t).
    Caller passes in the same dict every iter so we never re-alloc.

    ADAPTIVE_SLOT0/ALL: per-iter we recompute slot_topks_K/_NK from the
    block-1 / BFS rank predictions + greedy log-probs and upload them to
    the device.  These tensors are NOT cached in `scratch` because their
    values are content-dependent (different greedy-prob each iter).

    Returns the standard SGLang tree dict (see finalize_tree_gpu).
    """
    device = block1_logits.device
    give_up_class = rank_classes - 1
    max_topk = max(beam_width, max(rank_slot_topk))

    # --- Lazy-init persistent GPU buffers ---
    max_block1_nodes = K + K * (max_topk - 1) + 1
    max_block2_nodes = max_block1_nodes * K * max_topk
    max_nodes = max(total_tokens + 200, max_block1_nodes + max_block2_nodes + 100)

    tree_buf = scratch.get('tree_buf')
    if tree_buf is None or tree_buf['tokens'].shape[0] < max_nodes:
        tree_buf = {
            'tokens':  torch.empty(max_nodes, dtype=torch.long, device=device),
            'parents': torch.empty(max_nodes, dtype=torch.long, device=device),
            'lps':     torch.empty(max_nodes, dtype=torch.float32, device=device),
            'ranks':   torch.empty(max_nodes, dtype=torch.long, device=device),
            'blocks':  torch.empty(max_nodes, dtype=torch.long, device=device),
            'slots':   torch.empty(max_nodes, dtype=torch.long, device=device),
        }
        scratch['tree_buf'] = tree_buf

    pend_max = max_block1_nodes
    pend_buf = scratch.get('pend_buf')
    if pend_buf is None or pend_buf['hidden_slots'].shape[0] < pend_max:
        pend_buf = {
            'hidden_slots': torch.empty(pend_max, dtype=torch.long, device=device),
            'input_ids':    torch.empty(pend_max, dtype=torch.long, device=device),
            'ttt_valid':    torch.empty(pend_max, dtype=torch.long, device=device),
            'node_indices': torch.empty(pend_max, dtype=torch.long, device=device),
            'cum_lps':      torch.empty(pend_max, dtype=torch.float32, device=device),
        }
        scratch['pend_buf'] = pend_buf

    sizes4 = scratch.get('sizes4')
    if sizes4 is None:
        sizes4 = torch.empty(4, dtype=torch.long, device=device)
        scratch['sizes4'] = sizes4
    sizes1 = scratch.get('sizes1')
    if sizes1 is None:
        sizes1 = torch.empty(1, dtype=torch.long, device=device)
        scratch['sizes1'] = sizes1

    bfs_block_n = scratch.get('bfs_block_n', 32)
    cum_alt_buf = scratch.get('cum_alt_buf')
    if cum_alt_buf is None or cum_alt_buf.shape[0] < bfs_block_n:
        cum_alt_buf = torch.empty(bfs_block_n, dtype=torch.long, device=device)
        scratch['cum_alt_buf'] = cum_alt_buf

    rank_to_factor_t = scratch.get('rank_to_factor_t')
    if rank_to_factor_t is None:
        rank_to_factor_t = torch.tensor(
            rank_to_factor, dtype=torch.long, device=device,
        )
        scratch['rank_to_factor_t'] = rank_to_factor_t

    sample_token = initial_input_id[:, -1]

    # GPU-only path: pad N_pend to pend_max constant, eliminate block-1
    # host sync.  Pad rows masked via valid_mask (per-leaf gate in
    # _bfs_scatter_kernel_v2).  alt_offset = tree_start + real_N*K
    # (GPU int) packs alts contiguously after real chains.
    npend_gpu_only = os.environ.get("SPECBLOCK_NPEND_GPU_ONLY", "0") == "1"

    if npend_gpu_only:
        # Pre-clear pend_buf so pad rows have safe sentinel data.
        pend_buf['hidden_slots'].zero_()
        pend_buf['input_ids'].zero_()
        pend_buf['ttt_valid'].zero_()
        pend_buf['node_indices'].zero_()
        pend_buf['cum_lps'].fill_(float('-inf'))

    # --- Block-1 GPU precompute (no host syncs) ---
    last_p = block1_logits[0]                                       # [K, V_d]
    log_probs = F.log_softmax(last_p.float(), dim=-1)              # [K, V]
    rank_preds_1d = block1_rank_logits[0].argmax(dim=-1)           # [K]
    top_idx_2d = torch.topk(last_p, max_topk, dim=-1).indices      # [K, max_topk]
    if d2t is not None:
        all_top_target_2d = top_idx_2d + d2t[top_idx_2d]
    else:
        all_top_target_2d = top_idx_2d
    all_top_lps_2d = log_probs.gather(1, top_idx_2d)
    greedy_tgt_1d = all_top_target_2d[:, 0].contiguous()
    greedy_lps_1d = all_top_lps_2d[:, 0].contiguous()

    # --- Block-1 ADAPTIVE: GPU-side slot_topks_K [K] (no host sync) ---
    from sglang.srt.speculative.specblock_tree_builder import (
        compute_slot_topks_block1_gpu,
    )

    rank_slot_topk_t_static = scratch.get('rank_slot_topk_t')
    if rank_slot_topk_t_static is None:
        rank_slot_topk_t_static = torch.tensor(
            rank_slot_topk, dtype=torch.long, device=device,
        )
        scratch['rank_slot_topk_t'] = rank_slot_topk_t_static

    slot_topks_K = compute_slot_topks_block1_gpu(
        greedy_lps_1d.to(torch.float32),
        rank_preds_1d.to(torch.long),
        rank_slot_topk_t_static,
        beam_width,
        rank_classes,
    )

    # --- Kernel 1: block-1 mega ---
    block1_out = triton_build_block1(
        rank_preds_1d.to(torch.int64),
        greedy_tgt_1d.to(torch.int64),
        greedy_lps_1d.to(torch.float32),
        all_top_target_2d.to(torch.int64),
        all_top_lps_2d.to(torch.float32),
        slot_topks_K,
        tree_buf, pend_buf, sizes4,
        K, max_topk, rank_classes, give_up_class,
    )

    if npend_gpu_only:
        sizes_gpu = block1_out  # [4] gpu int64
        n_nodes_b1_t = sizes_gpu[0:1]  # [1] gpu
        N_pend_t = sizes_gpu[3:4]       # [1] gpu
        N_pend = pend_max  # PAD: kernels run with full pend_buf extent
        if max_blocks <= 1:
            # max_blocks <= 1 means no BFS — finalize immediately.
            # Sync once for finalize int.
            n_nodes_b1 = int(n_nodes_b1_t.item())
            return finalize_tree_gpu(
                tree_buf, n_nodes_b1, sample_token, total_tokens, K, max_blocks,
            )
        # n_nodes_b1 sync deferred to BFS / finalize.
        n_nodes_b1 = None
    else:
        n_nodes_b1, _n_active, _total_alts1, N_pend = block1_out
        if N_pend == 0 or max_blocks <= 1:
            return finalize_tree_gpu(
                tree_buf, n_nodes_b1, sample_token, total_tokens, K, max_blocks,
            )

    # --- Block-2 batch prep on GPU ---
    pend_slots = pend_buf['hidden_slots'][:N_pend]
    pend_input_ids = pend_buf['input_ids'][:N_pend].unsqueeze(1)
    pend_ttt_valid_t = pend_buf['ttt_valid'][:N_pend]
    pend_node_indices_t = pend_buf['node_indices'][:N_pend]
    pend_cum_lps_t = pend_buf['cum_lps'][:N_pend]

    pend_hidden = block1_hidden[0, pend_slots, :].unsqueeze(1)     # [N_pend, 1, H]
    arange_K_t = torch.arange(K, device=device)
    batch_ttt_mask = arange_K_t.unsqueeze(0) < pend_ttt_valid_t.unsqueeze(1)

    # cross_kv_cache: per-layer cache passed by caller.  Two layouts:
    #   (a) Dense legacy:  [k_buf, v_buf, count, max_len]
    #       - k_buf shape [1, n_heads, max_len, head_dim]; expand to N_pend.
    #   (b) Paged (Stage D production): [pool, layer_id, count, cross_loc, new_cross_loc]
    #       - forward_with_cache reads via _read_cross_kv_paged, no caller
    #         materialization; pass through unchanged.
    from sglang.srt.models._specblock_inference import _is_paged_cache
    is_paged_cross = bool(cross_kv_cache) and _is_paged_cache(cross_kv_cache[0])

    if is_paged_cross:
        stored_cross_count = int(cross_kv_cache[0][2])
        # The final K entries are block-0's current slots.  They are visible
        # only through the path-specific TTT cache below; exposing them again
        # as full cross context leaks future/sibling slots and double-counts KV.
        cross_count = max(stored_cross_count - K, 0)
        # paged 5-tuple: [pool, layer_id, count, cross_loc, new_cross_loc].
        # cross_loc may be 1D [stored_count] (single-req upstream) — retain
        # only persistent history, then expand for every pending row.
        batch_cross_cache = []
        for layer_cache in cross_kv_cache:
            pool, layer_id, _count, cross_loc, new_cross_loc = layer_cache
            cross_loc = cross_loc[:cross_count] if cross_loc is not None else None
            if cross_loc is not None and cross_loc.dim() == 1:
                cross_loc_2d = cross_loc.unsqueeze(0).expand(N_pend, -1).contiguous()
            else:
                cross_loc_2d = cross_loc
            batch_cross_cache.append(
                [pool, layer_id, cross_count, cross_loc_2d, new_cross_loc]
            )
    else:
        stored_cross_count = int(cross_kv_cache[0][2]) if cross_kv_cache else 0
        cross_count = max(stored_cross_count - K, 0)
        batch_cross_cache = []
        for layer_cache in cross_kv_cache:
            k_buf, v_buf = layer_cache[0], layer_cache[1]
            if cross_count > 0 and k_buf is not None:
                k_view = k_buf[:, :, :cross_count, :]
                v_view = v_buf[:, :, :cross_count, :]
                batch_cross_cache.append([
                    k_view.expand(N_pend, -1, -1, -1),
                    v_view.expand(N_pend, -1, -1, -1),
                    cross_count,
                ])
            else:
                batch_cross_cache.append([None, None, 0])

    if cross_count > 0:
        cross_ones = batch_ttt_mask.new_ones(N_pend, cross_count)
        full_kv_mask = torch.cat([cross_ones, batch_ttt_mask], dim=1)
    else:
        full_kv_mask = batch_ttt_mask

    batch_ttt_kv = [
        (kv[0].expand(N_pend, -1, -1, -1), kv[1].expand(N_pend, -1, -1, -1))
        for kv in block1_ttt_kv
    ]

    # --- Block-2 forward ---
    block_logits, block_rank_logits, _block_draft_hidden, _new_ttt_kv = (
        draft_model.forward_with_cache(
            hidden=pend_hidden,
            input_ids=pend_input_ids,
            cache=batch_cross_cache,
            position_id=position_id,
            use_draft_condition=True,
            ttt_cache=batch_ttt_kv,
            ttt_mask=batch_ttt_mask,
            update_cross_cache=False,
            full_kv_mask=full_kv_mask,
        )
    )

    # --- Block-2 GPU precompute ---
    (all_rank_preds_2, _all_greedy_tokens_2, all_greedy_target_2,
     all_greedy_lps_2, _M2, _bf2, _gu2,
     top_target_all_2, top_lps_all_2) = _bfs_gpu_ops_fused(
        block_logits, block_rank_logits, d2t, max_topk,
        rank_classes, rank_to_factor_t,
    )

    # Grow BLOCK_N if N_pend exceeds tile.
    if N_pend > bfs_block_n:
        bfs_block_n = max(64, 1 << (N_pend - 1).bit_length())
        cum_alt_buf = torch.empty(bfs_block_n, dtype=torch.long, device=device)
        scratch['bfs_block_n'] = bfs_block_n
        scratch['cum_alt_buf'] = cum_alt_buf

    # --- BFS ADAPTIVE: GPU-side slot_topks_NK [N_pend, K] (no host sync) ---
    from sglang.srt.speculative.specblock_tree_builder import (
        compute_slot_topks_bfs_gpu,
    )

    slot_topks_NK = compute_slot_topks_bfs_gpu(
        all_greedy_lps_2.to(torch.float32),
        all_rank_preds_2.to(torch.long),
        rank_slot_topk_t_static,
        rank_classes,
    )

    # GPU-only mode: build valid_mask + alt_offset (GPU tensors).
    # slot_topks_NK[pad rows] = 0 makes BFS sizing skip pad in cum_alt.
    # _bfs_scatter_kernel_v2 gates per-leaf stores by valid_mask and
    # places alts at alt_offset (= tree_start + real_N*K) so alts pack
    # immediately after real chains regardless of pad_max kernel arg.
    if npend_gpu_only:
        arange_pend = torch.arange(N_pend, device=device, dtype=torch.long)
        pend_valid_mask = arange_pend < N_pend_t.to(arange_pend.dtype)  # [pend_max]
        slot_topks_NK = slot_topks_NK * pend_valid_mask.unsqueeze(1).long()

    # --- Kernel 2: BFS sizing + scatter ---
    if npend_gpu_only:
        # No host syncs in BFS now.  tree_start_ptr = sizes_gpu[0:1]
        # (= n_nodes_b1).  alt_offset = sizes_gpu[0] + sizes_gpu[3] * K
        # (= tree_start + real_N * K) all on GPU.
        tree_start_ptr = sizes_gpu[0:1].contiguous()  # [1]
        alt_offset_gpu = (
            sizes_gpu[0:1].to(torch.long)
            + sizes_gpu[3:4].to(torch.long) * K
        ).contiguous()
        total_alts_2_t = triton_build_bfs_v2(  # [1] gpu i64
            all_rank_preds_2.to(torch.int64),
            all_greedy_target_2.to(torch.int64),
            all_greedy_lps_2.to(torch.float32),
            top_target_all_2.to(torch.int64),
            top_lps_all_2.to(torch.float32),
            pend_cum_lps_t.to(torch.float32),
            pend_node_indices_t.to(torch.int64),
            slot_topks_NK,
            valid_mask=pend_valid_mask,
            alt_offset=alt_offset_gpu,
            tree_start_ptr=tree_start_ptr,
            tree_buf=tree_buf, sizes_buf=sizes1, cum_alt_buf=cum_alt_buf,
            N=N_pend, K=K, max_topk=max_topk, rank_classes=rank_classes,
            give_up_class=give_up_class,
            pend_depth=1, block_n=bfs_block_n, j_pad=16,
        )
        # n_nodes_final all GPU.  SINGLE sync at finalize boundary —
        # gpu_only path now has 1 sync per build_tree_gpu call (vs 4
        # in legacy + numpy slot_topks path).  Cuda graph capture region
        # [block-1 mega ... BFS scatter] is sync-free and ready to capture.
        n_nodes_final_gpu = (
            sizes_gpu[0:1].to(torch.long)
            + N_pend_t.to(torch.long) * K + total_alts_2_t
        )
        n_nodes_final = int(n_nodes_final_gpu.item())
    else:
        total_alts_2 = triton_build_bfs(
            all_rank_preds_2.to(torch.int64),
            all_greedy_target_2.to(torch.int64),
            all_greedy_lps_2.to(torch.float32),
            top_target_all_2.to(torch.int64),
            top_lps_all_2.to(torch.float32),
            pend_cum_lps_t.to(torch.float32),
            pend_node_indices_t.to(torch.int64),
            slot_topks_NK,
            tree_buf, sizes1, cum_alt_buf,
            tree_start=n_nodes_b1,
            N=N_pend, K=K, max_topk=max_topk, rank_classes=rank_classes,
            give_up_class=give_up_class,
            pend_depth=1, block_n=bfs_block_n, j_pad=16,
        )
        n_nodes_final = n_nodes_b1 + N_pend * K + total_alts_2

    return finalize_tree_gpu(
        tree_buf, n_nodes_final, sample_token, total_tokens, K, max_blocks,
    )


# =============================================================================
#   Public driver: B>1 GPU-only end-to-end build_tree (batched)
# =============================================================================

def _alloc_per_req_scratch(
    sb: dict,
    *,
    max_nodes: int,
    pend_max: int,
    bfs_block_n_default: int,
    rank_slot_topk: Tuple[int, ...],
    rank_to_factor: Tuple[int, ...],
    device: torch.device,
) -> dict:
    """Lazy-init per-req persistent GPU scratch buffers.

    Shared between :func:`build_tree_gpu` (B=1) and
    :func:`build_tree_batched_gpu` (B>1) so per-req scratch dicts have
    the same layout.  Mutates ``sb`` in place; returns it for chaining.
    """
    tree_buf = sb.get('tree_buf')
    if tree_buf is None or tree_buf['tokens'].shape[0] < max_nodes:
        tree_buf = {
            'tokens':  torch.empty(max_nodes, dtype=torch.long, device=device),
            'parents': torch.empty(max_nodes, dtype=torch.long, device=device),
            'lps':     torch.empty(max_nodes, dtype=torch.float32, device=device),
            'ranks':   torch.empty(max_nodes, dtype=torch.long, device=device),
            'blocks':  torch.empty(max_nodes, dtype=torch.long, device=device),
            'slots':   torch.empty(max_nodes, dtype=torch.long, device=device),
        }
        sb['tree_buf'] = tree_buf

    pend_buf = sb.get('pend_buf')
    if pend_buf is None or pend_buf['hidden_slots'].shape[0] < pend_max:
        pend_buf = {
            'hidden_slots': torch.empty(pend_max, dtype=torch.long, device=device),
            'input_ids':    torch.empty(pend_max, dtype=torch.long, device=device),
            'ttt_valid':    torch.empty(pend_max, dtype=torch.long, device=device),
            'node_indices': torch.empty(pend_max, dtype=torch.long, device=device),
            'cum_lps':      torch.empty(pend_max, dtype=torch.float32, device=device),
        }
        sb['pend_buf'] = pend_buf

    if 'sizes4' not in sb:
        sb['sizes4'] = torch.empty(4, dtype=torch.long, device=device)
    if 'sizes1' not in sb:
        sb['sizes1'] = torch.empty(1, dtype=torch.long, device=device)

    if 'bfs_block_n' not in sb:
        sb['bfs_block_n'] = bfs_block_n_default
    cum_alt_buf = sb.get('cum_alt_buf')
    if cum_alt_buf is None or cum_alt_buf.shape[0] < sb['bfs_block_n']:
        sb['cum_alt_buf'] = torch.empty(
            sb['bfs_block_n'], dtype=torch.long, device=device,
        )

    if 'rank_slot_topk_t' not in sb:
        sb['rank_slot_topk_t'] = torch.tensor(
            rank_slot_topk, dtype=torch.long, device=device,
        )
    if 'rank_to_factor_t' not in sb:
        sb['rank_to_factor_t'] = torch.tensor(
            rank_to_factor, dtype=torch.long, device=device,
        )
    return sb


def _alloc_batch_scratch(
    sb_batch: dict,
    *,
    B: int,
    max_nodes: int,
    pend_max: int,
    bfs_block_n_default: int,
    rank_slot_topk: Tuple[int, ...],
    rank_to_factor: Tuple[int, ...],
    device: torch.device,
) -> dict:
    """Lazy-init batched scratch with [B, *] tensors shared across all reqs.

    Replaces the per-req scratch list pattern with a single dict whose
    tree_buf / pend_buf fields are 2D ``[B, max_*]``.  This lets the
    new batched Triton kernels (``_block1_mega_kernel_batched`` etc.)
    run with ``grid=(B,)`` and route per-program via ``bidx`` offsets.

    Buffers grow on shape demand (e.g. when B or max_nodes increases).
    """
    cur_B = sb_batch.get('B', 0)
    cur_MN = sb_batch.get('max_nodes', 0)
    cur_PM = sb_batch.get('pend_max', 0)

    need_realloc_tree = (
        'tree_buf' not in sb_batch or cur_B < B or cur_MN < max_nodes
    )
    if need_realloc_tree:
        sb_batch['tree_buf'] = {
            'tokens':  torch.empty(B, max_nodes, dtype=torch.long, device=device),
            'parents': torch.empty(B, max_nodes, dtype=torch.long, device=device),
            'lps':     torch.empty(B, max_nodes, dtype=torch.float32, device=device),
            'ranks':   torch.empty(B, max_nodes, dtype=torch.long, device=device),
            'blocks':  torch.empty(B, max_nodes, dtype=torch.long, device=device),
            'slots':   torch.empty(B, max_nodes, dtype=torch.long, device=device),
        }
        sb_batch['max_nodes'] = max_nodes

    need_realloc_pend = (
        'pend_buf' not in sb_batch or cur_B < B or cur_PM < pend_max
    )
    if need_realloc_pend:
        sb_batch['pend_buf'] = {
            'hidden_slots': torch.empty(B, pend_max, dtype=torch.long, device=device),
            'input_ids':    torch.empty(B, pend_max, dtype=torch.long, device=device),
            'ttt_valid':    torch.empty(B, pend_max, dtype=torch.long, device=device),
            'node_indices': torch.empty(B, pend_max, dtype=torch.long, device=device),
            'cum_lps':      torch.empty(B, pend_max, dtype=torch.float32, device=device),
        }
        sb_batch['pend_max'] = pend_max

    if 'sizes4' not in sb_batch or cur_B < B:
        sb_batch['sizes4'] = torch.empty(B, 4, dtype=torch.long, device=device)
    if 'sizes1' not in sb_batch or cur_B < B:
        sb_batch['sizes1'] = torch.empty(B, 1, dtype=torch.long, device=device)

    sb_batch['B'] = max(cur_B, B)

    if 'bfs_block_n' not in sb_batch:
        sb_batch['bfs_block_n'] = bfs_block_n_default
    if (
        'cum_alt_buf' not in sb_batch
        or sb_batch['cum_alt_buf'].shape[0] < B
        or sb_batch['cum_alt_buf'].shape[1] < sb_batch['bfs_block_n']
    ):
        sb_batch['cum_alt_buf'] = torch.empty(
            B, sb_batch['bfs_block_n'], dtype=torch.long, device=device,
        )

    if 'rank_slot_topk_t' not in sb_batch:
        sb_batch['rank_slot_topk_t'] = torch.tensor(
            rank_slot_topk, dtype=torch.long, device=device,
        )
    if 'rank_to_factor_t' not in sb_batch:
        sb_batch['rank_to_factor_t'] = torch.tensor(
            rank_to_factor, dtype=torch.long, device=device,
        )
    return sb_batch


def build_tree_batched_gpu(
    draft_model,
    block1_logits_b: torch.Tensor,                      # [B, K, V_draft]
    block1_rank_logits_b: torch.Tensor,                 # [B, K, rank_classes]
    block1_hidden_b: torch.Tensor,                      # [B, K, H]
    block1_ttt_kv_b: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    initial_input_id_b: torch.Tensor,                   # [B, 1]
    cross_kv_cache_b: List[List[List]],                 # B × num_layers × [paged_meta]
    cross_position_b: List[int],                        # length B
    K: int,
    max_blocks: int,
    beam_width: int,
    total_tokens: int,
    rank_classes: int,
    rank_slot_topk: Tuple[int, ...],
    rank_to_factor: Tuple[int, ...],
    d2t: Optional[torch.Tensor],
    scratch_b: List[dict],
    need_retrieve_index: bool = True,
    block1_top_indices_b: Optional[torch.Tensor] = None,
) -> List[dict]:
    """End-to-end GPU tree builder for B>1 reqs.

    Replaces the old numpy ``_expand_block1`` / ``_bfs_expand`` /
    ``_prune_tree_np`` / ``_build_tree_topology`` per-req chain with
    Triton kernel calls.  Phase-2 block-2 forward stays batched across
    all B reqs (the same batched forward used by the numpy path).

    Phase order:
        1. per-req block-1: Triton ``_block1_mega_kernel``
        2. BATCHED block-2 forward across all B reqs (single GPU call)
        3. per-req BFS: Triton ``_bfs_sizing_kernel`` + ``_bfs_scatter_kernel``
        4. per-req finalize: Triton prune (top-K + ancestor) + topology

    Per-req kernel launches at B=4 add ~50us overhead vs single-req
    launch — negligible vs the >40ms phase-2 forward they sit alongside.

    Returns a list of B tree dicts (same schema as :func:`build_tree`).
    """
    from sglang.srt.speculative.specblock_tree_builder import (
        compute_slot_topks_block1_gpu_batched,
        compute_slot_topks_bfs_gpu,
        _batched_block2_forward,
    )

    B = int(block1_logits_b.shape[0])
    device = block1_logits_b.device
    give_up_class = rank_classes - 1
    max_topk = max(beam_width, max(rank_slot_topk))
    max_block1_nodes = K + K * (max_topk - 1) + 1
    max_block2_nodes = max_block1_nodes * K * max_topk
    max_nodes = max(total_tokens + 200, max_block1_nodes + max_block2_nodes + 100)
    pend_max = max_block1_nodes

    # Ensure scratch_b has B entries (lazy-init).
    while len(scratch_b) < B:
        scratch_b.append({})

    # ---- Batched scratch (S3a): allocate [B, *] shared buffers once.
    # ``_block1_mega_kernel_batched`` writes per-req rows in a single
    # grid=(B,) launch; downstream code reads per-req via 1D views. ----
    sb_batch = getattr(build_tree_batched_gpu, '_batch_scratch', None)
    if sb_batch is None:
        sb_batch = {}
        build_tree_batched_gpu._batch_scratch = sb_batch
    _alloc_batch_scratch(
        sb_batch, B=B,
        max_nodes=max_nodes, pend_max=pend_max,
        bfs_block_n_default=32,
        rank_slot_topk=rank_slot_topk, rank_to_factor=rank_to_factor,
        device=device,
    )
    tree_buf_b = sb_batch['tree_buf']     # dict of [B, max_nodes]
    pend_buf_b = sb_batch['pend_buf']     # dict of [B, pend_max]
    sizes4_b = sb_batch['sizes4']         # [B, 4]
    sizes1_b = sb_batch['sizes1']         # [B, 1]
    rank_slot_topk_t = sb_batch['rank_slot_topk_t']
    rank_to_factor_t = sb_batch['rank_to_factor_t']

    _bt_deep = os.environ.get("SPECBLOCK_DEEP_PROFILE", "0") == "1"
    if _bt_deep:
        import time as _bt_time
        torch.cuda.synchronize()
        _bt_t0 = _bt_time.perf_counter()
        _bt_t1_loop = 0.0  # batched mega kernel
        _bt_t1_setup = 0.0  # batched setup (slot_topks etc.)

    # ---- Phase 1 batched setup: do log_softmax / argmax / topk / gather
    # ONCE on [B, K, V] / [B, K, ...] inputs (was B per-req calls with
    # ~8 small kernel launches each = ~32 kernel launches at bs=4). ----
    all_rank_preds_b = block1_rank_logits_b.argmax(dim=-1)                   # [B, K]
    cached_topk = (
        block1_top_indices_b is not None
        and block1_top_indices_b.shape[-1] >= max_topk
    )
    if cached_topk:
        # The rank head already sorted these exact b0 logits.  Reuse its
        # candidate indices; retain F.log_softmax so cumulative scores stay
        # bit-identical to the established tree builder.
        all_top_idx_b = block1_top_indices_b[..., :max_topk]
        all_log_probs_b = F.log_softmax(block1_logits_b.float(), dim=-1)
        all_top_lps_b = all_log_probs_b.gather(2, all_top_idx_b)
    else:
        all_log_probs_b = F.log_softmax(block1_logits_b.float(), dim=-1)
        all_top_idx_b = torch.topk(
            block1_logits_b, max_topk, dim=-1,
        ).indices
        all_top_lps_b = all_log_probs_b.gather(2, all_top_idx_b)
    if d2t is not None:
        all_top_target_b = all_top_idx_b + d2t[all_top_idx_b]
    else:
        all_top_target_b = all_top_idx_b
    all_greedy_tgt_b = all_top_target_b[:, :, 0].contiguous()                # [B, K]
    all_greedy_lps_b = all_top_lps_b[:, :, 0].contiguous().to(torch.float32) # [B, K]
    all_rank_preds_long_b = all_rank_preds_b.to(torch.long)

    # Batched slot_topks_K [B, K] (replaces per-req call inside loop).
    slot_topks_K_b = compute_slot_topks_block1_gpu_batched(
        all_greedy_lps_b, all_rank_preds_long_b,
        rank_slot_topk_t, beam_width, rank_classes,
    )                                                                        # [B, K]

    if _bt_deep:
        torch.cuda.synchronize()
        _bt_p1a = _bt_time.perf_counter()
        _bt_t1_setup = _bt_p1a - _bt_t0

    # ---- Phase 1: SINGLE batched block-1 mega kernel grid=(B,) ----
    _block1_mega_kernel_batched[(B,)](
        all_rank_preds_long_b.contiguous(),
        all_greedy_tgt_b.to(torch.int64).contiguous(),
        all_greedy_lps_b.contiguous(),
        all_top_target_b.to(torch.int64).contiguous(),
        all_top_lps_b.to(torch.float32).contiguous(),
        slot_topks_K_b.contiguous(),
        tree_buf_b['tokens'], tree_buf_b['parents'], tree_buf_b['lps'],
        tree_buf_b['ranks'], tree_buf_b['blocks'], tree_buf_b['slots'],
        pend_buf_b['hidden_slots'], pend_buf_b['input_ids'],
        pend_buf_b['ttt_valid'], pend_buf_b['node_indices'], pend_buf_b['cum_lps'],
        sizes4_b,
        max_nodes,    # STRIDE_NODES
        pend_max,     # STRIDE_PEND
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class,
    )

    if _bt_deep:
        torch.cuda.synchronize()
        _bt_p1b = _bt_time.perf_counter()
        _bt_t1_loop = _bt_p1b - _bt_p1a

    # The production checkpoint expands one draft block.  At fixed verify
    # width, finalize directly from GPU sizes: no sizes4 D2H barrier, Python
    # per-request metadata, generic prune, or separate topology-link pass.
    if max_blocks <= 1:
        trees = finalize_one_block_fixed_batched(
            tree_buf_b=tree_buf_b,
            raw_sizes_b=sizes4_b,
            sample_tokens_b=initial_input_id_b,
            budget=total_tokens,
            raw_capacity=max_block1_nodes,
            sb_batch=sb_batch,
        )
        if _bt_deep:
            torch.cuda.synchronize()
            _bt_fast_end = _bt_time.perf_counter()
            st = build_tree_batched_gpu.__dict__.setdefault(
                "_DEEP_FAST_STATE",
                {"n": 0, "setup": 0.0, "mega": 0.0, "finalize": 0.0},
            )
            st["n"] += 1
            st["setup"] += _bt_t1_setup
            st["mega"] += _bt_t1_loop
            st["finalize"] += _bt_fast_end - _bt_p1b
            if st["n"] % 50 == 0:
                n = st["n"]
                import logging as _lg
                _lg.getLogger(__name__).info(
                    "[DEEPprof:build_tree_bgpu_fast] n=%d setup=%.2f "
                    "mega=%.2f finalize=%.2f total=%.2fms",
                    n, st["setup"] / n * 1000, st["mega"] / n * 1000,
                    st["finalize"] / n * 1000,
                    (st["setup"] + st["mega"] + st["finalize"]) / n * 1000,
                )
        return trees

    # ---- Per-req sb alias from batched buffers (no copy).  Downstream
    # Phase 3+4 still loops per-req using these aliases.  cum_alt_buf
    # stays per-req lazy alloc (grown only when N_pend > bfs_block_n). ----
    sizes4_cpu = sizes4_b.cpu().tolist()  # ONE host sync (was B syncs)

    per_req: List[dict] = []
    arange_K_t = torch.arange(K, device=device)
    for b in range(B):
        sb = scratch_b[b]
        sb['tree_buf'] = {k: tree_buf_b[k][b] for k in tree_buf_b}
        sb['pend_buf'] = {k: pend_buf_b[k][b] for k in pend_buf_b}
        sb['sizes4'] = sizes4_b[b]
        sb['sizes1'] = sizes1_b[b]
        sb['rank_slot_topk_t'] = rank_slot_topk_t
        sb['rank_to_factor_t'] = rank_to_factor_t
        sb['bfs_block_n'] = sb.get('bfs_block_n', 32)
        if 'cum_alt_buf' not in sb or sb['cum_alt_buf'].shape[0] < sb['bfs_block_n']:
            sb['cum_alt_buf'] = torch.empty(
                sb['bfs_block_n'], dtype=torch.long, device=device,
            )

        n_nodes_b1, n_active, total_alts1, N_pend = sizes4_cpu[b]
        pend_buf = sb['pend_buf']

        if N_pend > 0:
            pend_slots = pend_buf['hidden_slots'][:N_pend]
            pend_input_ids_t = pend_buf['input_ids'][:N_pend].unsqueeze(1)
            pend_ttt_valid_t = pend_buf['ttt_valid'][:N_pend]
            pend_hidden_t = block1_hidden_b[b, pend_slots, :].unsqueeze(1)  # [N_pend, 1, H]
            ttt_mask_t = arange_K_t.unsqueeze(0) < pend_ttt_valid_t.unsqueeze(1)
            import numpy as _np
            placeholder_node_indices = _np.empty(N_pend, dtype=_np.int64)
        else:
            pend_hidden_t = block1_hidden_b[:, :0, :]
            pend_input_ids_t = torch.empty(0, 1, dtype=torch.long, device=device)
            ttt_mask_t = torch.zeros(0, K, dtype=torch.bool, device=device)
            import numpy as _np
            placeholder_node_indices = _np.empty(0, dtype=_np.int64)

        per_req.append({
            'sb': sb,
            'n_nodes_b1': int(n_nodes_b1),
            'N_pend': int(N_pend),
            'pend_dict': {'node_indices': placeholder_node_indices},
            'pend_hidden': pend_hidden_t,
            'pend_input_ids': pend_input_ids_t,
            'ttt_mask': ttt_mask_t,
            'block_logits': None,
            'block_rank_logits': None,
        })

    if _bt_deep:
        torch.cuda.synchronize()
        _bt_t_p1 = _bt_time.perf_counter()

    # ---- Phase 2: BATCHED block-2 forward across all B reqs ----
    if max_blocks >= 2:
        _batched_block2_forward(
            per_req, draft_model, cross_kv_cache_b, block1_ttt_kv_b,
            cross_position_b, K=K, device=device,
        )

    if _bt_deep:
        torch.cuda.synchronize()
        _bt_t_p2 = _bt_time.perf_counter()

    # ---- Phase 3 + 4: per-req BFS + finalize ----
    if _bt_deep:
        torch.cuda.synchronize()
        _bt_t34_bfs_ops = 0.0
        _bt_t34_slot_topk = 0.0
        _bt_t34_build_bfs = 0.0
        _bt_t34_finalize = 0.0

    # ---- S2: BFS GPU precompute batched.  Concat valid per-req
    # block_logits / block_rank_logits into ONE call to
    # _bfs_gpu_ops_fused (was B per-req calls = 2.47ms at bs=4). ----
    valid_block_logits: List[torch.Tensor] = []
    valid_block_rank_logits: List[torch.Tensor] = []
    cum_off: List[int] = [0] * (B + 1)
    for b in range(B):
        N_b = per_req[b]['N_pend']
        bl = per_req[b].get('block_logits')
        if max_blocks >= 2 and N_b > 0 and bl is not None:
            valid_block_logits.append(bl)
            valid_block_rank_logits.append(per_req[b]['block_rank_logits'])
            cum_off[b + 1] = cum_off[b] + N_b
        else:
            cum_off[b + 1] = cum_off[b]

    if _bt_deep:
        torch.cuda.synchronize()
        _bt_a = _bt_time.perf_counter()

    if valid_block_logits:
        all_block_logits = (
            valid_block_logits[0] if len(valid_block_logits) == 1
            else torch.cat(valid_block_logits, dim=0)
        )
        all_block_rank_logits = (
            valid_block_rank_logits[0] if len(valid_block_rank_logits) == 1
            else torch.cat(valid_block_rank_logits, dim=0)
        )
        (all_rank_preds_full, _all_gtok_full, all_greedy_target_full,
         all_greedy_lps_full, _M_full, _bf_full, _gu_full,
         top_target_all_full, top_lps_all_full) = _bfs_gpu_ops_fused(
            all_block_logits, all_block_rank_logits, d2t, max_topk,
            rank_classes, rank_to_factor_t,
        )
    else:
        all_rank_preds_full = None  # no req has BFS work

    if _bt_deep:
        torch.cuda.synchronize()
        _bt_b = _bt_time.perf_counter()
        _bt_t34_bfs_ops += _bt_b - _bt_a

    # ---- S3b: SINGLE batched BFS scatter across all reqs ----
    sum_N = int(cum_off[B])
    cum_off_t = torch.tensor(cum_off, dtype=torch.long, device=device)  # [B+1]
    N_per_req_b = cum_off_t[1:] - cum_off_t[:B]                          # [B] contig

    if sum_N > 0 and all_rank_preds_full is not None:
        if _bt_deep:
            torch.cuda.synchronize()
            _bt_b_loop = _bt_time.perf_counter()

        # sizes4_b may be over-allocated ([B_max, 4]) — slice to current B.
        n_nodes_b1_b = sizes4_b[:B, 0].contiguous()                      # [B] contig

        # Batched slot_topks across all sum_N leaves.
        slot_topks_NK_full = compute_slot_topks_bfs_gpu(
            all_greedy_lps_full.to(torch.float32),
            all_rank_preds_full.to(torch.long),
            rank_slot_topk_t,
            rank_classes,
        )

        if _bt_deep:
            torch.cuda.synchronize()
            _bt_c = _bt_time.perf_counter()
            _bt_t34_slot_topk += _bt_c - _bt_b_loop

        total_alts_per_req_b = triton_build_bfs_flat_batched(
            all_rank_preds_full, all_greedy_target_full, all_greedy_lps_full,
            top_target_all_full, top_lps_all_full,
            slot_topks_NK_full,
            pend_buf_b, tree_buf_b,
            n_nodes_b1_b, N_per_req_b, cum_off_t,
            B=B, sum_N=sum_N,
            K=K, max_topk=max_topk, rank_classes=rank_classes,
            give_up_class=give_up_class, pend_depth=1,
            max_nodes=max_nodes, pend_max=pend_max,
            device=device,
        )

        if _bt_deep:
            torch.cuda.synchronize()
            _bt_d = _bt_time.perf_counter()
            _bt_t34_build_bfs += _bt_d - _bt_c

        # ONE host sync replaces B per-req syncs.
        total_alts_cpu = total_alts_per_req_b.cpu().tolist()
    else:
        total_alts_cpu = [0] * B

    # ---- S3c: BATCHED finalize across all reqs.
    # Per-req: compute n_nodes_final + run prune (if N>budget) into a
    # per-req post-prune scratch dict.  Then call finalize_tree_gpu_batched
    # which does depth_mask + retrieve + sort + output dict packing in
    # a single batched pass (was 5 kernel launches × B reqs). ----
    if _bt_deep:
        torch.cuda.synchronize()
        _bt_e = _bt_time.perf_counter()

    raw_n_nodes_per_req: List[int] = []
    sample_tokens_b = initial_input_id_b[:, -1:]  # [B, 1]

    for b in range(B):
        st = per_req[b]
        n_nodes_b1 = st['n_nodes_b1']
        N_b_in_concat = cum_off[b + 1] - cum_off[b]
        if max_blocks >= 2 and N_b_in_concat > 0:
            n_nodes_final = n_nodes_b1 + N_b_in_concat * K + total_alts_cpu[b]
        else:
            n_nodes_final = n_nodes_b1
        raw_n_nodes_per_req.append(n_nodes_final)

    prune_rows = [
        b for b, n_nodes in enumerate(raw_n_nodes_per_req)
        if n_nodes > total_tokens
    ]
    batched_pruned = gpu_prune_tree_batched(
        tree_buf_b, raw_n_nodes_per_req, prune_rows, total_tokens,
    )
    pruned_row_to_index = {row: idx for idx, row in enumerate(prune_rows)}

    n_nodes_per_req_cpu: List[int] = []
    pruned_tensors_per_req: List[Optional[dict]] = []
    for b, n_nodes_final in enumerate(raw_n_nodes_per_req):
        prune_idx = pruned_row_to_index.get(b)
        if prune_idx is None:
            pruned_tensors_per_req.append(None)
            n_nodes_per_req_cpu.append(n_nodes_final)
            continue
        pruned_tensors_per_req.append({
            name: tensor[prune_idx]
            for name, tensor in batched_pruned.items()
        })
        n_nodes_per_req_cpu.append(total_tokens)

    all_rows_pruned = len(prune_rows) == B
    trees = finalize_tree_gpu_batched(
        tree_buf_b, n_nodes_per_req_cpu, pruned_tensors_per_req,
        batched_pruned if all_rows_pruned else None,
        sample_tokens_b, total_tokens, K, max_blocks,
        sb_batch, device, need_retrieve_index=need_retrieve_index,
    )

    if _bt_deep:
        torch.cuda.synchronize()
        _bt_f = _bt_time.perf_counter()
        _bt_t34_finalize += _bt_f - _bt_e
        _bt_t_p34 = _bt_time.perf_counter()
        st = build_tree_batched_gpu.__dict__.setdefault(
            "_DEEP_STATE",
            {
                "n": 0, "phase1": 0.0, "phase2": 0.0, "phase34": 0.0,
                "bfs_ops": 0.0, "slot_topk": 0.0,
                "build_bfs": 0.0, "finalize": 0.0,
            },
        )
        st.setdefault("p1_mega", 0.0)
        st["n"] += 1
        st["phase1"] += _bt_t_p1 - _bt_t0
        st["phase2"] += _bt_t_p2 - _bt_t_p1
        st["phase34"] += _bt_t_p34 - _bt_t_p2
        st["p1_mega"] += _bt_t1_loop
        st["bfs_ops"] += _bt_t34_bfs_ops
        st["slot_topk"] += _bt_t34_slot_topk
        st["build_bfs"] += _bt_t34_build_bfs
        st["finalize"] += _bt_t34_finalize
        if st["n"] % 50 == 0:
            n = st["n"]
            import logging as _lg
            _lg.getLogger(__name__).info(
                "[DEEPprof:build_tree_bgpu] n=%d phase1=%.2f (mega=%.2f setup=%.2f) "
                "phase2=%.2f phase3+4=%.2f (bfs_ops=%.2f slot_topk=%.2f "
                "build_bfs=%.2f finalize=%.2f) total=%.2fms",
                n,
                st["phase1"]/n*1000,
                st["p1_mega"]/n*1000,
                (st["phase1"] - st["p1_mega"])/n*1000,
                st["phase2"]/n*1000,
                st["phase34"]/n*1000,
                st["bfs_ops"]/n*1000, st["slot_topk"]/n*1000,
                st["build_bfs"]/n*1000, st["finalize"]/n*1000,
                (st["phase1"]+st["phase2"]+st["phase34"])/n*1000,
            )

    return trees


# =============================================================================
#   Capturable region: build_tree_gpu's GPU-only chain (block-1 .. BFS scatter)
#
#   Used by SpecBlockDraftCudaGraphRunner.  Differences from build_tree_gpu:
#     - position_id is Tensor [1] (Tensor in, Tensor in attention) so the
#       capture-friendly forward_with_cache_graph is used.
#     - Always npend_gpu_only=True (sync-free).
#     - Skips finalize_tree_gpu (data-dependent shape; runs eager post-replay).
#     - Inputs: STATIC tensors caller copies into before each replay.
#
#   Output: tree_buf is populated in-place; sizes4 / sizes1 contain the
#   counts (n_nodes_b1, n_active, total_alts1, N_pend, total_alts_2) the
#   caller's eager finalize uses.
# =============================================================================

def build_tree_gpu_capturable_region(
    draft_model,
    block1_logits: torch.Tensor,                            # [1, K, V_draft]
    block1_rank_logits: torch.Tensor,                       # [1, K, rank_classes]
    block1_hidden: torch.Tensor,                            # [1, K, H]
    block1_ttt_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    cross_kv_cache: List[List],                             # paged 5-tuple list, static-width cross_loc
    cross_valid_count_t: torch.Tensor,                      # [1] int, live persistent cross prefix
    position_id_t: torch.Tensor,                            # [1] long  (replaces int)
    K: int,
    max_blocks: int,
    beam_width: int,
    total_tokens: int,
    rank_classes: int,
    rank_slot_topk: Tuple[int, ...],
    rank_to_factor: Tuple[int, ...],
    d2t: Optional[torch.Tensor],
    scratch: dict,
) -> None:
    """Run build_tree_gpu's capture-eligible region on static buffers.

    Mutates ``scratch['tree_buf']`` (populated tree); ``scratch['sizes4']``
    holds [n_nodes_b1, n_active, total_alts1, N_pend]; ``scratch['sizes1']``
    holds [total_alts_2].  Caller derives ``n_nodes_final = n_nodes_b1 +
    N_pend * K + total_alts_2`` post-replay (single .item() sync).

    No host syncs inside; safe under ``torch.cuda.graph()`` capture.
    """
    from sglang.srt.speculative.specblock_tree_builder import (
        compute_slot_topks_block1_gpu,
        compute_slot_topks_bfs_gpu,
    )

    device = block1_logits.device
    give_up_class = rank_classes - 1
    max_topk = max(beam_width, max(rank_slot_topk))

    max_block1_nodes = K + K * (max_topk - 1) + 1
    max_block2_nodes = max_block1_nodes * K * max_topk
    max_nodes = max(total_tokens + 200, max_block1_nodes + max_block2_nodes + 100)
    pend_max = max_block1_nodes

    # Lazy-init persistent scratch (skipped on subsequent calls).
    _alloc_per_req_scratch(
        scratch,
        max_nodes=max_nodes, pend_max=pend_max,
        bfs_block_n_default=64,
        rank_slot_topk=rank_slot_topk, rank_to_factor=rank_to_factor,
        device=device,
    )
    tree_buf = scratch['tree_buf']
    pend_buf = scratch['pend_buf']
    sizes4 = scratch['sizes4']
    sizes1 = scratch['sizes1']
    cum_alt_buf = scratch['cum_alt_buf']
    rank_slot_topk_t = scratch['rank_slot_topk_t']
    rank_to_factor_t = scratch['rank_to_factor_t']
    bfs_block_n = scratch['bfs_block_n']

    # ---- Pre-clear pend_buf so pad rows have safe sentinel data ----
    pend_buf['hidden_slots'].zero_()
    pend_buf['input_ids'].zero_()
    pend_buf['ttt_valid'].zero_()
    pend_buf['node_indices'].zero_()
    pend_buf['cum_lps'].fill_(float('-inf'))

    # ---- Block-1 GPU precompute (no host syncs) ----
    last_p = block1_logits[0]                               # [K, V_d]
    log_probs = F.log_softmax(last_p.float(), dim=-1)
    rank_preds_1d = block1_rank_logits[0].argmax(dim=-1)    # [K]
    top_idx_2d = torch.topk(last_p, max_topk, dim=-1).indices  # [K, max_topk]
    if d2t is not None:
        all_top_target_2d = top_idx_2d + d2t[top_idx_2d]
    else:
        all_top_target_2d = top_idx_2d
    all_top_lps_2d = log_probs.gather(1, top_idx_2d)
    greedy_tgt_1d = all_top_target_2d[:, 0].contiguous()
    greedy_lps_1d = all_top_lps_2d[:, 0].contiguous()

    # ADAPTIVE slot_topks_K (GPU)
    slot_topks_K = compute_slot_topks_block1_gpu(
        greedy_lps_1d.to(torch.float32),
        rank_preds_1d.to(torch.long),
        rank_slot_topk_t,
        beam_width, rank_classes,
    )

    # ---- Block-1 mega kernel.  npend_gpu_only path: returns sizes_buf
    # (no host sync); we read sizes4[3] (= N_pend) on GPU only. ----
    _block1_mega_kernel[(1,)](
        rank_preds_1d.to(torch.int64),
        greedy_tgt_1d.to(torch.int64),
        greedy_lps_1d.to(torch.float32),
        all_top_target_2d.to(torch.int64),
        all_top_lps_2d.to(torch.float32),
        slot_topks_K,
        tree_buf['tokens'], tree_buf['parents'], tree_buf['lps'],
        tree_buf['ranks'], tree_buf['blocks'], tree_buf['slots'],
        pend_buf['hidden_slots'], pend_buf['input_ids'],
        pend_buf['ttt_valid'], pend_buf['node_indices'], pend_buf['cum_lps'],
        sizes4,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class,
    )

    if max_blocks <= 1:
        return  # block-1 only; finalize reads sizes4[0] post-replay

    # ---- Block-2 batch prep on GPU ----
    # PAD: kernels run with full pend_max extent; pad rows masked via
    # pend_valid_mask (= arange < N_pend) inside _bfs_scatter_kernel_v2.
    N_pend_capture = pend_max  # static for capture; actual N gated by mask
    pend_slots = pend_buf['hidden_slots'][:N_pend_capture]
    pend_input_ids = pend_buf['input_ids'][:N_pend_capture].unsqueeze(1)
    pend_ttt_valid_t = pend_buf['ttt_valid'][:N_pend_capture]
    pend_hidden = block1_hidden[0, pend_slots, :].unsqueeze(1)  # [pend_max, 1, H]
    arange_K_t = torch.arange(K, device=device)
    batch_ttt_mask = arange_K_t.unsqueeze(0) < pend_ttt_valid_t.unsqueeze(1)

    # cross_kv_cache carries a static bucket width while cross_valid_count_t
    # changes on every replay.  The final K live slots were removed by the
    # graph runner because block-2 sees them through its path-specific TTT
    # cache.  Keep the static width for graph shape and mask sentinel padding
    # out of the attention softmax.
    cross_count_int = int(cross_kv_cache[0][2])
    batch_cross_cache = []
    for layer_cache in cross_kv_cache:
        pool, layer_id, _count, cross_loc, new_cross_loc = layer_cache
        if cross_loc is not None and cross_loc.dim() == 1:
            cross_loc_2d = cross_loc.unsqueeze(0).expand(N_pend_capture, -1).contiguous()
        else:
            cross_loc_2d = cross_loc
        batch_cross_cache.append(
            [pool, layer_id, cross_count_int, cross_loc_2d, new_cross_loc]
        )

    if cross_count_int > 0:
        cross_offsets = torch.arange(
            cross_count_int, device=device, dtype=cross_valid_count_t.dtype
        )
        cross_mask = cross_offsets.unsqueeze(0) < cross_valid_count_t.reshape(1, 1)
        cross_mask = cross_mask.expand(N_pend_capture, -1)
        full_kv_mask = torch.cat([cross_mask, batch_ttt_mask], dim=1)
    else:
        full_kv_mask = batch_ttt_mask

    batch_ttt_kv = [
        (kv[0].expand(N_pend_capture, -1, -1, -1),
         kv[1].expand(N_pend_capture, -1, -1, -1))
        for kv in block1_ttt_kv
    ]

    # ---- Block-2 forward (graph-safe variant) ----
    # pos_ids must be [N_pend, K] (one position per slot per pend row).
    # Each pend row's slot k is at absolute position ``position_id + 1 + k``.
    # The non-graph forward_with_cache (specblock_shift_inference.py:127-142)
    # builds the same shape from int position_id; the graph variant takes
    # pos_ids as input so we precompute it here.
    rope_max_position = getattr(
        draft_model.config, "max_position_embeddings", 131072,
    )
    pos_offsets = torch.arange(K, device=device, dtype=torch.long).unsqueeze(0)  # [1, K]
    pos_ids = (
        position_id_t.to(torch.long).unsqueeze(1) + 1 + pos_offsets
    ).expand(N_pend_capture, K).contiguous()  # [N_pend, K]
    block_logits, block_rank_logits, _block_draft_hidden, _new_ttt_kv = (
        draft_model.forward_with_cache_graph(
            hidden=pend_hidden,
            input_ids=pend_input_ids,
            pos_ids=pos_ids,
            cache=batch_cross_cache,
            rope_max_position=rope_max_position,
            ttt_cache=batch_ttt_kv,
            ttt_mask=batch_ttt_mask,
            full_kv_mask=full_kv_mask,
        )
    )

    # ---- BFS GPU precompute ----
    (all_rank_preds_2, _all_greedy_tokens_2, all_greedy_target_2,
     all_greedy_lps_2, _M2, _bf2, _gu2,
     top_target_all_2, top_lps_all_2) = _bfs_gpu_ops_fused(
        block_logits, block_rank_logits, d2t, max_topk,
        rank_classes, rank_to_factor_t,
    )

    slot_topks_NK = compute_slot_topks_bfs_gpu(
        all_greedy_lps_2.to(torch.float32),
        all_rank_preds_2.to(torch.long),
        rank_slot_topk_t,
        rank_classes,
    )

    # Apply pad-row mask to slot_topks (zero-out pad rows so cum_alt skips them).
    arange_pend = torch.arange(N_pend_capture, device=device, dtype=torch.long)
    pend_valid_mask = arange_pend < sizes4[3:4].to(arange_pend.dtype)  # [pend_max]
    slot_topks_NK = slot_topks_NK * pend_valid_mask.unsqueeze(1).long()

    # alt_offset = sizes4[0] (n_nodes_b1) + sizes4[3] (N_pend) * K (GPU int).
    tree_start_ptr = sizes4[0:1].contiguous()
    alt_offset_gpu = (
        sizes4[0:1].to(torch.long) + sizes4[3:4].to(torch.long) * K
    ).contiguous()

    triton_build_bfs_v2(
        all_rank_preds_2.to(torch.int64),
        all_greedy_target_2.to(torch.int64),
        all_greedy_lps_2.to(torch.float32),
        top_target_all_2.to(torch.int64),
        top_lps_all_2.to(torch.float32),
        pend_buf['cum_lps'][:N_pend_capture].to(torch.float32),
        pend_buf['node_indices'][:N_pend_capture].to(torch.int64),
        slot_topks_NK,
        valid_mask=pend_valid_mask,
        alt_offset=alt_offset_gpu,
        tree_start_ptr=tree_start_ptr,
        tree_buf=tree_buf, sizes_buf=sizes1, cum_alt_buf=cum_alt_buf,
        N=N_pend_capture, K=K, max_topk=max_topk, rank_classes=rank_classes,
        give_up_class=give_up_class,
        pend_depth=1, block_n=bfs_block_n, j_pad=16,
    )
    # On exit: tree_buf populated; sizes4 has [n_nodes_b1, ..., N_pend];
    # sizes1[0] has total_alts_2.


def max_block1_structural_nodes(
    K: int,
    beam_width: int,
    rank_slot_topk: Tuple[int, ...],
) -> int:
    """Return the exact block-1 structural node upper bound.

    Slot 0 uses ``beam_width`` while slots 1..K-1 use the largest
    rank-conditioned top-k.  The final ``+1`` accounts for the root.
    """
    max_rank_topk = max(rank_slot_topk)
    return (
        K
        + max(beam_width - 1, 0)
        + max(K - 1, 0) * max(max_rank_topk - 1, 0)
        + 1
    )


def build_tree_gpu_batched_capturable_region(
    draft_model,
    block1_logits_b: torch.Tensor,
    block1_rank_logits_b: torch.Tensor,
    block1_hidden_b: torch.Tensor,
    block1_ttt_kv_b: List[Tuple[torch.Tensor, torch.Tensor]],
    cross_kv_cache_b: List[List],
    cross_valid_count_b: torch.Tensor,
    position_id_b: torch.Tensor,
    active_mask_b: torch.Tensor,
    pend_bucket: int,
    expand_bucket: int,
    K: int,
    max_blocks: int,
    beam_width: int,
    total_tokens: int,
    rank_classes: int,
    rank_slot_topk: Tuple[int, ...],
    rank_to_factor: Tuple[int, ...],
    d2t: Optional[torch.Tensor],
    scratch: dict,
) -> None:
    """Capture-safe request-level batched SpecBlock tree construction."""
    from sglang.srt.speculative.specblock_tree_builder import (
        compute_slot_topks_bfs_gpu,
        compute_slot_topks_block1_gpu_batched,
    )

    B = int(block1_logits_b.shape[0])
    device = block1_logits_b.device
    give_up_class = rank_classes - 1
    max_topk = max(beam_width, max(rank_slot_topk))
    max_block1_nodes = max_block1_structural_nodes(
        K, beam_width, rank_slot_topk,
    )
    if pend_bucket < max_block1_nodes:
        raise RuntimeError(
            "SpecBlock pending graph bucket is smaller than the structural "
            f"maximum: bucket={pend_bucket}, required={max_block1_nodes}."
        )
    if not 0 < expand_bucket <= pend_bucket:
        raise RuntimeError(
            "SpecBlock expansion bucket must fit the pending storage bucket: "
            f"expand={expand_bucket}, pending={pend_bucket}."
        )
    max_nodes = max(
        total_tokens + 200,
        max_block1_nodes + expand_bucket * K * max_topk + 100,
    )

    _alloc_batch_scratch(
        scratch,
        B=B,
        max_nodes=max_nodes,
        pend_max=pend_bucket,
        bfs_block_n_default=64,
        rank_slot_topk=rank_slot_topk,
        rank_to_factor=rank_to_factor,
        device=device,
    )
    tree_buf = scratch["tree_buf"]
    pend_buf = scratch["pend_buf"]
    sizes4 = scratch["sizes4"][:B]
    sizes1 = scratch["sizes1"][:B]
    rank_slot_topk_t = scratch["rank_slot_topk_t"]
    rank_to_factor_t = scratch["rank_to_factor_t"]

    for tensor in pend_buf.values():
        tensor.zero_()
    pend_buf["cum_lps"].fill_(float("-inf"))
    sizes1.zero_()

    rank_preds = block1_rank_logits_b.argmax(dim=-1)
    log_probs = F.log_softmax(block1_logits_b.float(), dim=-1)
    top_idx = torch.topk(block1_logits_b, max_topk, dim=-1).indices
    all_top_target = top_idx + d2t[top_idx] if d2t is not None else top_idx
    all_top_lps = log_probs.gather(2, top_idx)
    greedy_target = all_top_target[:, :, 0].contiguous()
    greedy_lps = all_top_lps[:, :, 0].contiguous().to(torch.float32)
    slot_topks = compute_slot_topks_block1_gpu_batched(
        greedy_lps,
        rank_preds.to(torch.long),
        rank_slot_topk_t,
        beam_width,
        rank_classes,
    )

    _block1_mega_kernel_batched[(B,)](
        rank_preds.to(torch.int64).contiguous(),
        greedy_target.to(torch.int64).contiguous(),
        greedy_lps.contiguous(),
        all_top_target.to(torch.int64).contiguous(),
        all_top_lps.to(torch.float32).contiguous(),
        slot_topks.contiguous(),
        tree_buf["tokens"], tree_buf["parents"], tree_buf["lps"],
        tree_buf["ranks"], tree_buf["blocks"], tree_buf["slots"],
        pend_buf["hidden_slots"], pend_buf["input_ids"],
        pend_buf["ttt_valid"], pend_buf["node_indices"],
        pend_buf["cum_lps"], sizes4,
        max_nodes, pend_bucket,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class,
    )
    sizes4.mul_(active_mask_b.to(sizes4.dtype).unsqueeze(1))

    if max_blocks <= 1:
        return

    # Expand only the highest-probability pending roots.  Descendant log
    # probability can never exceed its pending root, so this preserves the
    # verifier budget for the most valuable subtrees while halving the fixed
    # block-2 model batch from 32 to 16 leaves in the production geometry.
    selected_idx = torch.topk(
        pend_buf["cum_lps"][:B, :pend_bucket],
        k=expand_bucket,
        dim=1,
    ).indices
    selected_pend = {
        name: torch.gather(value[:B, :pend_bucket], 1, selected_idx)
        for name, value in pend_buf.items()
    }
    selected_count = sizes4[:, 3].clamp(max=expand_bucket)
    sizes4[:, 3].copy_(selected_count)

    pend_slots = selected_pend["hidden_slots"]
    hidden_index = pend_slots.unsqueeze(2).expand(
        B, expand_bucket, block1_hidden_b.shape[2]
    )
    pend_hidden = block1_hidden_b.gather(1, hidden_index).reshape(
        B * expand_bucket, 1, block1_hidden_b.shape[2]
    )
    pend_input_ids = selected_pend["input_ids"].reshape(
        B * expand_bucket, 1
    )
    ttt_valid = selected_pend["ttt_valid"]
    slot_offsets = torch.arange(K, device=device, dtype=torch.long)
    # Keep pending leaves grouped by request.  The grouped paged-attention
    # kernel shares persistent prefix K/V loads across leaves while preserving
    # a distinct TTT-valid mask for every leaf.
    ttt_mask = slot_offsets.reshape(1, 1, K) < ttt_valid.unsqueeze(2)

    cross_width = int(cross_kv_cache_b[0][2])
    batch_cross_cache = []
    for layer_cache in cross_kv_cache_b:
        pool, layer_id, _count, cross_loc, new_cross_loc = layer_cache
        batch_cross_cache.append(
            [pool, layer_id, cross_width, cross_loc, new_cross_loc]
        )

    if cross_width > 0:
        cross_offsets = torch.arange(
            cross_width, device=device, dtype=cross_valid_count_b.dtype
        )
        full_kv_mask = (
            cross_offsets.reshape(1, cross_width)
            < cross_valid_count_b.reshape(B, 1)
        )
    else:
        full_kv_mask = torch.empty(
            (B, 0), dtype=torch.bool, device=device,
        )

    batch_ttt_kv = block1_ttt_kv_b
    pos_ids = (
        position_id_b.to(torch.long).reshape(B, 1, 1)
        + 1
        + slot_offsets.reshape(1, 1, K)
    ).expand(B, expand_bucket, K).reshape(B * expand_bucket, K)
    rope_max_position = getattr(
        draft_model.config, "max_position_embeddings", 131072
    )
    block_logits, block_rank_logits, _hidden, _ttt = (
        draft_model.forward_with_cache_graph(
            hidden=pend_hidden,
            input_ids=pend_input_ids,
            pos_ids=pos_ids,
            cache=batch_cross_cache,
            rope_max_position=rope_max_position,
            ttt_cache=batch_ttt_kv,
            ttt_mask=ttt_mask,
            full_kv_mask=full_kv_mask,
        )
    )

    (
        all_rank_preds_2,
        _all_greedy_tokens_2,
        all_greedy_target_2,
        all_greedy_lps_2,
        _M2,
        _bf2,
        _gu2,
        top_target_all_2,
        top_lps_all_2,
    ) = _bfs_gpu_ops_fused(
        block_logits,
        block_rank_logits,
        d2t,
        max_topk,
        rank_classes,
        rank_to_factor_t,
    )
    slot_topks_2 = compute_slot_topks_bfs_gpu(
        all_greedy_lps_2.to(torch.float32),
        all_rank_preds_2.to(torch.long),
        rank_slot_topk_t,
        rank_classes,
    )
    valid_leaf_b = (
        torch.arange(expand_bucket, device=device, dtype=sizes4.dtype)
        .unsqueeze(0)
        < sizes4[:, 3:4]
    ) & active_mask_b.unsqueeze(1)
    slot_topks_2 = slot_topks_2 * valid_leaf_b.reshape(-1, 1).to(
        slot_topks_2.dtype
    )
    total_alts = triton_build_bfs_fixed_batched(
        all_rank_preds_2,
        all_greedy_target_2,
        all_greedy_lps_2,
        top_target_all_2,
        top_lps_all_2,
        slot_topks_2,
        valid_leaf_b,
        selected_pend,
        tree_buf,
        sizes4,
        B=B,
        pend_bucket=expand_bucket,
        K=K,
        max_topk=max_topk,
        rank_classes=rank_classes,
        give_up_class=give_up_class,
        max_nodes=max_nodes,
    )
    sizes1[:, 0].copy_(total_alts)
