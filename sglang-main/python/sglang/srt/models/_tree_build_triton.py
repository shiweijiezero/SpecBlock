"""Triton mega-kernels for rank-guided tree construction.

Replaces the numpy/Python tree-building path in specblock.py with two single-
program kernels. Design goals:
  - Block-1 branching + scatter + pending construction: 1 kernel launch.
  - BFS branching + scatter: 1 kernel launch.
  - Only 2 host syncs per tree build (1 for N_pend after block-1, 1 for final
    n_nodes after BFS).
  - Deterministic output positions: intra-program prefix-sum, no atomics.

Supported config (matches SpecBlockAlgorithm._tree_gpu_supported):
  - K=4, max_blocks=2, rank_classes=4, give_up_class=3
  - MAX_TOPK=10 (matches eager max(beam_width=10, max(RANK_SLOT_TOPK)))
  - ADAPTIVE_SLOT0 ∈ {0, 1, 2, 3, 4}
  - ADAPTIVE_ALL   ∈ {0, 1, 2}
  - RANK_SLOT_TOPK[give_up_class] > 0 (no_giveup family) → stop_at==K always
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# =============================================================================
#   Block-1 mega kernel
# =============================================================================

@triton.jit
def _block1_mega_kernel(
    # === Inputs (GPU, contiguous) ===
    rank_preds_ptr,            # [K] i64
    greedy_target_ptr,         # [K] i64
    greedy_lps_ptr,            # [K] f32
    all_top_target_ptr,        # [K, MAX_TOPK] i64
    all_top_lps_ptr,           # [K, MAX_TOPK] f32
    rank_slot_topk_table_ptr,  # [RANK_CLASSES] i64

    # === Outputs (pre-allocated) ===
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

    # === Runtime scalar ===
    beam_width,  # i32

    # === Compile-time constants ===
    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    RANK_CLASSES: tl.constexpr,
    GIVE_UP_CLASS: tl.constexpr,
    ADAPTIVE_SLOT0_MODE: tl.constexpr,
    ADAPTIVE_ALL_MODE: tl.constexpr,
):
    """Single-program block-1 branching + scatter + pending construction.

    Grid: (1,). K is tiny (=4) so full vectorization inside one program.
    Output layout on tree buffer:
      [0 : K]                       — greedy chain (slot 0..K-1)
      [K : K + total_alts1]         — alternatives (flattened by cumsum)
    Output layout on pending arrays:
      [0 : n_active]                         — active slots (slot_topks>1)
      [n_active : n_active + total_alts1]    — alternative entries
      [n_active + total_alts1]               — hitchhike g(K-1)  (if !give_up & !active[K-1])
    """
    # -----------------------------------------------------------------
    # Load per-slot inputs
    # -----------------------------------------------------------------
    k_vec = tl.arange(0, K)                                     # [K]
    rank_preds = tl.load(rank_preds_ptr + k_vec)                # [K] i64
    greedy_tgt = tl.load(greedy_target_ptr + k_vec)             # [K] i64
    greedy_lps = tl.load(greedy_lps_ptr + k_vec)                # [K] f32

    # -----------------------------------------------------------------
    # Compute slot_topks[K] via adaptive_slot0 (slot 0) + adaptive_all (slot>0)
    # -----------------------------------------------------------------
    # Base topk by rank: rank_slot_topk_table[rank_preds]
    base_topk = tl.load(rank_slot_topk_table_ptr + rank_preds)  # [K] i64

    # Probabilities from greedy log-probs
    p_vec = tl.exp(greedy_lps)                                  # [K] f32
    # Use [K]-vec all along; scalar p0 is just p_vec[0] — we compute the
    # adaptive_slot0 rule on p_vec (vectorized) and later pick slot-0 entry
    # via `k_vec == 0` mask.  This avoids tricky []-shaped scalar juggling.
    bw_vec = tl.full([K], 0, dtype=tl.int64) + beam_width       # [K] all beam_width
    one_vec = tl.full([K], 1, dtype=tl.int64)
    two_vec = tl.full([K], 2, dtype=tl.int64)
    three_vec = tl.full([K], 3, dtype=tl.int64)
    four_vec = tl.full([K], 4, dtype=tl.int64)
    five_vec = tl.full([K], 5, dtype=tl.int64)
    six_vec = tl.full([K], 6, dtype=tl.int64)
    seven_vec = tl.full([K], 7, dtype=tl.int64)
    eight_vec = tl.full([K], 8, dtype=tl.int64)

    # adaptive_slot0 rule applied per-slot (only slot-0 value will be selected)
    if ADAPTIVE_SLOT0_MODE == 0:
        slot0_rule = bw_vec
    elif ADAPTIVE_SLOT0_MODE == 1:
        slot0_rule = tl.where(p_vec >= 0.9, tl.minimum(bw_vec, three_vec),
                     tl.where(p_vec >= 0.8, tl.minimum(bw_vec, five_vec),
                     tl.where(p_vec >= 0.6, tl.minimum(bw_vec, eight_vec),
                              bw_vec)))
    elif ADAPTIVE_SLOT0_MODE == 2:
        slot0_rule = tl.where(p_vec >= 0.95, tl.minimum(bw_vec, two_vec),
                     tl.where(p_vec >= 0.85, tl.minimum(bw_vec, four_vec),
                     tl.where(p_vec >= 0.6,  tl.minimum(bw_vec, seven_vec),
                              bw_vec)))
    elif ADAPTIVE_SLOT0_MODE == 3:
        slot0_rule = tl.where(p_vec >= 0.95, one_vec,
                     tl.where(p_vec >= 0.9,  tl.minimum(bw_vec, two_vec),
                     tl.where(p_vec >= 0.8,  tl.minimum(bw_vec, four_vec),
                     tl.where(p_vec >= 0.6,  tl.minimum(bw_vec, seven_vec),
                              bw_vec))))
    else:  # 4 = ultra
        slot0_rule = tl.where(p_vec >= 0.9, one_vec,
                     tl.where(p_vec >= 0.8, tl.minimum(bw_vec, three_vec),
                     tl.where(p_vec >= 0.6, tl.minimum(bw_vec, six_vec),
                              bw_vec)))

    # adaptive_all rule applied per-slot (only slot>0 values will be selected)
    if ADAPTIVE_ALL_MODE == 0:
        slot_k_rule = base_topk
    elif ADAPTIVE_ALL_MODE == 1:
        slot_k_rule = tl.where(p_vec >= 0.95, tl.minimum(base_topk, two_vec),
                      tl.where(p_vec >= 0.85, tl.minimum(base_topk, three_vec),
                               base_topk))
    else:  # 2 = aggressive
        slot_k_rule = tl.where(p_vec >= 0.9, one_vec,
                      tl.where(p_vec >= 0.7, tl.minimum(base_topk, two_vec),
                      tl.where(p_vec >= 0.5, tl.minimum(base_topk, three_vec),
                               base_topk)))

    slot_topks = tl.where(k_vec == 0, slot0_rule, slot_k_rule)  # [K] i64

    # -----------------------------------------------------------------
    # Give-up flag: (rank_preds != 0).any() & first non-zero == GIVE_UP
    # -----------------------------------------------------------------
    non_zero = (rank_preds != 0).to(tl.int32)                  # [K]
    # First non-zero index via cumsum==1 trick
    csum_nz = tl.cumsum(non_zero, 0)                           # [K]
    is_first_nz = (csum_nz == 1) & (non_zero == 1)             # [K]
    any_nz = tl.sum(non_zero) > 0                              # scalar bool
    first_nz_rank = tl.sum(tl.where(is_first_nz, rank_preds, tl.zeros([K], dtype=tl.int64)))
    give_up = any_nz & (first_nz_rank == GIVE_UP_CLASS)        # scalar bool

    # -----------------------------------------------------------------
    # Cum sums for alternatives + active slots
    # -----------------------------------------------------------------
    n_alt_slot = tl.maximum(slot_topks - 1, 0)                 # [K]
    # Inclusive cumsum → exclusive
    cum_alt_incl = tl.cumsum(n_alt_slot, 0)                    # [K]
    cum_alt_excl = cum_alt_incl - n_alt_slot                   # [K]
    total_alts1 = tl.sum(n_alt_slot)                           # scalar

    active_mask = slot_topks > 1                               # [K] bool
    active_mask_i64 = active_mask.to(tl.int64)                 # [K]
    active_rank_incl = tl.cumsum(active_mask_i64, 0)           # [K]
    active_rank = active_rank_incl - active_mask_i64           # [K] 0-idx exclusive
    n_active = tl.sum(active_mask_i64)                         # scalar

    # Hitchhike: add g(K-1) iff !give_up AND !active_mask[K-1]
    last_active = tl.sum(tl.where(k_vec == (K - 1), active_mask_i64, tl.zeros_like(active_mask_i64)))
    hitch_add = ((~give_up) & (last_active == 0)).to(tl.int64)  # scalar 0 or 1

    # -----------------------------------------------------------------
    # Write greedy chain [0 .. K-1]
    # -----------------------------------------------------------------
    cum_greedy_lps = tl.cumsum(greedy_lps, 0)                   # [K] f32
    cum_greedy_lps_excl = cum_greedy_lps - greedy_lps          # [K] for parent lookups
    parent_chain = k_vec.to(tl.int64) - 1                       # [-1, 0, 1, ..., K-2]

    tl.store(tree_tokens_ptr + k_vec, greedy_tgt)
    tl.store(tree_parents_ptr + k_vec, parent_chain)
    tl.store(tree_lps_ptr + k_vec, cum_greedy_lps)
    tl.store(tree_ranks_ptr + k_vec, rank_preds)
    tl.store(tree_blocks_ptr + k_vec, tl.zeros([K], dtype=tl.int64))
    tl.store(tree_slots_ptr + k_vec, k_vec.to(tl.int64))

    # -----------------------------------------------------------------
    # Write alternatives  [K .. K + total_alts1 - 1]
    # and alt entries in pend arrays [n_active .. n_active + total_alts1 - 1]
    #
    # Flatten (k, j) space with J_PAD (next pow2 ≥ MAX_TOPK-1) for tl.arange
    # pow2 requirement. K_ALT = K * J_PAD; mask j_idx < MAX_TOPK-1 so padded
    # slots are inert. alt_valid[i] = (valid j) & (alt_j < slot_topks[alt_k]).
    # -----------------------------------------------------------------
    J_PAD: tl.constexpr = 16  # next pow2 ≥ MAX_TOPK-1 = 9
    K_ALT: tl.constexpr = K * J_PAD

    alt_flat = tl.arange(0, K_ALT)                              # [K_ALT]
    alt_k = alt_flat // J_PAD                                   # [K_ALT] slot index 0..K-1
    alt_j_idx = alt_flat - alt_k * J_PAD                        # [K_ALT] 0..J_PAD-1
    alt_j = alt_j_idx + 1                                       # [K_ALT] j ∈ [1..J_PAD]

    # Gather slot_topks[alt_k] via 2D broadcast-match
    match_k = alt_k[:, None] == k_vec[None, :]                  # [K_ALT, K]
    slot_topks_per_alt = tl.sum(
        match_k.to(tl.int64) * slot_topks[None, :], axis=1
    )                                                           # [K_ALT]
    # valid iff (real j index within MAX_TOPK-1) AND (alt_j < slot_topks for this slot)
    alt_valid = (alt_j_idx < (MAX_TOPK - 1)) & (alt_j < slot_topks_per_alt)
    alt_valid_i64 = alt_valid.to(tl.int64)
    alt_rank_incl = tl.cumsum(alt_valid_i64, 0)                 # [K_ALT]
    alt_rank = alt_rank_incl - alt_valid_i64                    # [K_ALT] 0-idx among valid

    # Gather parent_lp = 0 if alt_k==0 else cum_greedy_lps_excl[alt_k]
    match_k_parent = alt_k[:, None] == k_vec[None, :]           # [K_ALT, K]
    parent_lp_alt = tl.sum(
        match_k_parent.to(tl.float32) * cum_greedy_lps_excl[None, :], axis=1
    )                                                           # [K_ALT]
    # Load top_target/top_lps at [alt_k, alt_j] via 2D offsets
    alt_off = alt_k * MAX_TOPK + alt_j                          # [K_ALT]
    alt_top_tgt = tl.load(all_top_target_ptr + alt_off, mask=alt_valid, other=0)
    alt_top_lps = tl.load(all_top_lps_ptr + alt_off, mask=alt_valid, other=0.0)
    alt_lp_node = parent_lp_alt + alt_top_lps                   # [K_ALT]

    # Parent for alternatives: -1 if alt_k==0 else alt_k - 1
    alt_parent = tl.where(alt_k == 0,
                          tl.full([K_ALT], -1, dtype=tl.int64),
                          alt_k.to(tl.int64) - 1)
    alt_rank_pred = tl.load(rank_preds_ptr + alt_k)             # [K_ALT] rank per slot

    # Write tree buf at K + alt_rank
    tree_alt_pos = K + alt_rank                                 # [K_ALT]
    tl.store(tree_tokens_ptr + tree_alt_pos, alt_top_tgt, mask=alt_valid)
    tl.store(tree_parents_ptr + tree_alt_pos, alt_parent, mask=alt_valid)
    tl.store(tree_lps_ptr + tree_alt_pos, alt_lp_node, mask=alt_valid)
    tl.store(tree_ranks_ptr + tree_alt_pos, alt_rank_pred, mask=alt_valid)
    tl.store(tree_blocks_ptr + tree_alt_pos, tl.zeros([K_ALT], dtype=tl.int64), mask=alt_valid)
    tl.store(tree_slots_ptr + tree_alt_pos, alt_k.to(tl.int64), mask=alt_valid)

    # Write pend alt entries at n_active + alt_rank
    pend_alt_pos = n_active + alt_rank                          # [K_ALT]
    tl.store(pend_hidden_slots_ptr + pend_alt_pos, alt_k.to(tl.int64), mask=alt_valid)
    tl.store(pend_input_ids_ptr    + pend_alt_pos, alt_top_tgt, mask=alt_valid)
    tl.store(pend_ttt_valid_ptr    + pend_alt_pos, alt_k.to(tl.int64) + 1, mask=alt_valid)
    tl.store(pend_node_indices_ptr + pend_alt_pos, tree_alt_pos.to(tl.int64), mask=alt_valid)
    tl.store(pend_cum_lps_ptr      + pend_alt_pos, alt_lp_node, mask=alt_valid)

    # -----------------------------------------------------------------
    # Write active-slot pending entries [0 .. n_active-1]
    # For slot k if active_mask[k]: pos = active_rank[k]
    # -----------------------------------------------------------------
    tl.store(pend_hidden_slots_ptr + active_rank, k_vec.to(tl.int64), mask=active_mask)
    tl.store(pend_input_ids_ptr    + active_rank, greedy_tgt,          mask=active_mask)
    tl.store(pend_ttt_valid_ptr    + active_rank, k_vec.to(tl.int64) + 1, mask=active_mask)
    tl.store(pend_node_indices_ptr + active_rank, k_vec.to(tl.int64),  mask=active_mask)
    tl.store(pend_cum_lps_ptr      + active_rank, cum_greedy_lps,      mask=active_mask)

    # -----------------------------------------------------------------
    # Write hitchhike g(K-1) at position n_active + total_alts1 (if needed)
    # -----------------------------------------------------------------
    hitch_pos = n_active + total_alts1                                # scalar
    hitch_mask = (hitch_add == 1)
    last_greedy_tgt = tl.sum(tl.where(k_vec == (K - 1), greedy_tgt,
                                      tl.zeros([K], dtype=tl.int64)))     # scalar i64
    last_cum_lp = tl.sum(tl.where(k_vec == (K - 1), cum_greedy_lps,
                                  tl.zeros([K], dtype=tl.float32)))       # scalar f32

    # [1]-vec store for the single hitchhike entry. hitch_mask (scalar bool)
    # broadcasts into the [1] shape via logical-and with an always-true [1] mask.
    ones1 = tl.arange(0, 1)                                                  # [1]: [0]
    hitch_pos_vec = hitch_pos + ones1                                        # [1] i64
    hitch_mask_vec = (ones1 == 0) & hitch_mask                               # [1] bool
    val_last_tgt  = (ones1 * 0).to(tl.int64) + last_greedy_tgt               # [1] i64
    val_last_lp   = (ones1 * 0).to(tl.float32) + last_cum_lp                 # [1] f32
    val_km1       = (ones1 * 0).to(tl.int64) + (K - 1)                       # [1] i64
    val_k         = (ones1 * 0).to(tl.int64) + K                             # [1] i64
    tl.store(pend_hidden_slots_ptr + hitch_pos_vec, val_km1,       mask=hitch_mask_vec)
    tl.store(pend_input_ids_ptr    + hitch_pos_vec, val_last_tgt,  mask=hitch_mask_vec)
    tl.store(pend_ttt_valid_ptr    + hitch_pos_vec, val_k,         mask=hitch_mask_vec)
    tl.store(pend_node_indices_ptr + hitch_pos_vec, val_km1,       mask=hitch_mask_vec)
    tl.store(pend_cum_lps_ptr      + hitch_pos_vec, val_last_lp,   mask=hitch_mask_vec)

    # -----------------------------------------------------------------
    # Write sizes: {n_nodes_b1, n_active, total_alts1, N_pend}
    # n_nodes_b1 = K + total_alts1
    # N_pend     = n_active + total_alts1 + hitch_add
    # -----------------------------------------------------------------
    n_nodes_b1 = K + total_alts1
    n_pend = n_active + total_alts1 + hitch_add
    size_idx = tl.arange(0, 4)
    size_vals = tl.where(size_idx == 0, n_nodes_b1,
                tl.where(size_idx == 1, n_active,
                tl.where(size_idx == 2, total_alts1,
                         n_pend)))
    tl.store(sizes_ptr + size_idx, size_vals)


# =============================================================================
#   BFS kernels — split into (sizing) + (grid-parallel scatter)
# =============================================================================

@triton.jit
def _bfs_sizing_kernel(
    # === Inputs ===
    all_rank_preds_ptr,        # [N, K] i64
    all_greedy_lps_ptr,        # [N, K] f32
    rank_slot_topk_table_ptr,  # [RANK_CLASSES] i64

    # === Outputs ===
    cum_alt_excl_per_leaf_ptr, # [BLOCK_N] i64 — exclusive prefix sum of n_alt_per_leaf
    sizes_ptr,                 # [1] i64 — total_alts_2

    # === Runtime scalar ===
    N,

    # === Compile-time constants ===
    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    RANK_CLASSES: tl.constexpr,
    ADAPTIVE_ALL_MODE: tl.constexpr,
    BLOCK_N: tl.constexpr,     # upper bound for N (must be pow2, e.g. 32)
):
    """Single-program sizing pass. Computes per-leaf n_alt, prefix-scans to
    produce cum_alt_excl_per_leaf (global offset each leaf's alts start at),
    and writes total_alts_2 scalar. Downstream grid-parallel scatter kernel
    reads cum_alt_excl_per_leaf to place its writes deterministically.

    Tile: [BLOCK_N=32, K=4] = 128 elements per field. Very small.
    """
    leaf_vec = tl.arange(0, BLOCK_N)                                # [BLOCK_N]
    slot_vec = tl.arange(0, K)                                      # [K]
    leaf_mask = leaf_vec < N
    lk_off = leaf_vec[:, None] * K + slot_vec[None, :]
    lk_mask = leaf_mask[:, None]

    rank_2d = tl.load(all_rank_preds_ptr + lk_off, mask=lk_mask, other=0)
    greedy_lps_2d = tl.load(all_greedy_lps_ptr + lk_off, mask=lk_mask, other=0.0)
    base_topk_2d = tl.load(rank_slot_topk_table_ptr + rank_2d, mask=lk_mask, other=0)

    if ADAPTIVE_ALL_MODE == 0:
        slot_topks_2d = base_topk_2d
    elif ADAPTIVE_ALL_MODE == 1:
        pk_2d = tl.exp(greedy_lps_2d)
        slot_topks_2d = tl.where(pk_2d >= 0.95, tl.minimum(base_topk_2d, 2),
                        tl.where(pk_2d >= 0.85, tl.minimum(base_topk_2d, 3),
                                 base_topk_2d))
    else:  # 2
        pk_2d = tl.exp(greedy_lps_2d)
        slot_topks_2d = tl.where(pk_2d >= 0.9, tl.full([BLOCK_N, K], 1, dtype=tl.int64),
                        tl.where(pk_2d >= 0.7, tl.minimum(base_topk_2d, 2),
                        tl.where(pk_2d >= 0.5, tl.minimum(base_topk_2d, 3),
                                 base_topk_2d)))

    is_active_2d = slot_topks_2d > 1
    n_alt_pair = tl.where(is_active_2d & lk_mask, slot_topks_2d - 1, 0)   # [BLOCK_N, K]
    n_alt_per_leaf = tl.sum(n_alt_pair, axis=1)                            # [BLOCK_N]

    cum_incl = tl.cumsum(n_alt_per_leaf, 0)
    cum_excl = cum_incl - n_alt_per_leaf
    total = tl.sum(n_alt_per_leaf)

    tl.store(cum_alt_excl_per_leaf_ptr + leaf_vec, cum_excl, mask=leaf_mask)
    ones1 = tl.arange(0, 1)
    tl.store(sizes_ptr + ones1, (ones1 * 0).to(tl.int64) + total)


@triton.jit
def _bfs_scatter_kernel(
    # === Inputs ===
    all_rank_preds_ptr,        # [N, K] i64
    all_greedy_target_ptr,     # [N, K] i64
    all_greedy_lps_ptr,        # [N, K] f32
    top_target_all_ptr,        # [N, K, MAX_TOPK] i64
    top_lps_all_ptr,           # [N, K, MAX_TOPK] f32
    pend_cum_lps_ptr,          # [N] f32
    pend_node_indices_ptr,     # [N] i64
    rank_slot_topk_table_ptr,  # [RANK_CLASSES] i64
    cum_alt_excl_per_leaf_ptr, # [BLOCK_N] i64 — from sizing kernel

    # === Outputs (tree_buf from tree_start onwards) ===
    tree_tokens_ptr, tree_parents_ptr, tree_lps_ptr,
    tree_ranks_ptr, tree_blocks_ptr, tree_slots_ptr,

    # === Runtime scalars ===
    N, tree_start,

    # === Compile-time constants ===
    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    RANK_CLASSES: tl.constexpr,
    GIVE_UP_CLASS: tl.constexpr,
    ADAPTIVE_ALL_MODE: tl.constexpr,
    PEND_DEPTH: tl.constexpr,
    J_PAD: tl.constexpr,       # next pow2 ≥ MAX_TOPK-1 (e.g. 16)
):
    """Grid-parallel BFS scatter, one program per leaf.

    Grid: (N,). Each program handles one leaf's greedy chain (K nodes)
    and alternatives (up to K * (MAX_TOPK-1) nodes, masked) with a 2D tile
    [K=4, J_PAD=16] = 64 elements per field. Parallelism = N across SMs.

    Reads cum_alt_excl_per_leaf[leaf] from the sizing kernel to place its
    alt writes at the correct global offset. Per-leaf slot_topks are
    recomputed (cheap K=4 vec work); no scratch beyond the prefix-sum.

    Output layout on tree buffer:
      [tree_start + leaf*K : tree_start + (leaf+1)*K]     — greedy chain
      [tree_start + N*K + cum_alt_excl[leaf] : ...]        — alts
    """
    leaf = tl.program_id(0)
    slot_vec = tl.arange(0, K)                                              # [K]

    # -----------------------------------------------------------------
    # Per-leaf loads
    # -----------------------------------------------------------------
    lk_off = leaf * K + slot_vec                                             # [K]
    rank_vec = tl.load(all_rank_preds_ptr + lk_off)                          # [K] i64
    greedy_tgt = tl.load(all_greedy_target_ptr + lk_off)                     # [K] i64
    greedy_lps = tl.load(all_greedy_lps_ptr + lk_off)                        # [K] f32
    pend_cum_lp = tl.load(pend_cum_lps_ptr + leaf)                           # scalar f32
    pend_node = tl.load(pend_node_indices_ptr + leaf)                        # scalar i64

    # -----------------------------------------------------------------
    # Greedy chain writes: tree_start + leaf*K + slot  (K nodes)
    # -----------------------------------------------------------------
    chain_base = (tree_start + leaf * K).to(tl.int64)                        # scalar i64
    chain_pos = chain_base + slot_vec.to(tl.int64)                           # [K] i64

    cum_greedy_lps = tl.cumsum(greedy_lps, 0)                                # [K] f32
    cum_greedy_lps_excl = cum_greedy_lps - greedy_lps                        # [K] (slot 0 → 0)

    parent_vec = tl.where(slot_vec == 0, pend_node, chain_pos - 1)           # [K] i64
    lp_vec = pend_cum_lp + cum_greedy_lps                                    # [K] f32

    tl.store(tree_tokens_ptr  + chain_pos, greedy_tgt)
    tl.store(tree_parents_ptr + chain_pos, parent_vec)
    tl.store(tree_lps_ptr     + chain_pos, lp_vec)
    tl.store(tree_ranks_ptr   + chain_pos, rank_vec)
    tl.store(tree_blocks_ptr  + chain_pos, tl.full([K], PEND_DEPTH, dtype=tl.int64))
    tl.store(tree_slots_ptr   + chain_pos, slot_vec.to(tl.int64))

    # -----------------------------------------------------------------
    # Per-leaf slot_topks (recompute — cheap K=4 vec work).
    # GIVE_UP_HAS_TOPK assumed True (checked in _tree_gpu_supported).
    # -----------------------------------------------------------------
    base_topk = tl.load(rank_slot_topk_table_ptr + rank_vec)                 # [K] i64

    if ADAPTIVE_ALL_MODE == 0:
        slot_topks = base_topk
    elif ADAPTIVE_ALL_MODE == 1:
        pk = tl.exp(greedy_lps)
        slot_topks = tl.where(pk >= 0.95, tl.minimum(base_topk, 2),
                     tl.where(pk >= 0.85, tl.minimum(base_topk, 3),
                              base_topk))
    else:  # 2
        pk = tl.exp(greedy_lps)
        slot_topks = tl.where(pk >= 0.9, tl.full([K], 1, dtype=tl.int64),
                     tl.where(pk >= 0.7, tl.minimum(base_topk, 2),
                     tl.where(pk >= 0.5, tl.minimum(base_topk, 3),
                              base_topk)))

    is_active = slot_topks > 1                                               # [K]
    n_alt_slot = tl.where(is_active, slot_topks - 1, 0)                      # [K]
    cum_alt_slot_incl = tl.cumsum(n_alt_slot, 0)                             # [K]
    cum_alt_slot_excl = cum_alt_slot_incl - n_alt_slot                       # [K]

    # Global leaf offset from sizing kernel.
    leaf_alt_base = tl.load(cum_alt_excl_per_leaf_ptr + leaf)                # scalar i64

    # -----------------------------------------------------------------
    # Alternative writes: 2D tile [K, J_PAD]
    # Pos = tree_start + N*K + leaf_alt_base + cum_alt_slot_excl[slot] + j_idx
    # -----------------------------------------------------------------
    J_COUNT: tl.constexpr = MAX_TOPK - 1
    j_vec = tl.arange(0, J_PAD)                                              # [J_PAD] j-1 index
    j_val = j_vec + 1                                                        # [J_PAD] j ∈ [1..J_PAD]
    j_real = j_vec < J_COUNT                                                 # [J_PAD] valid-j mask

    valid_2d = (
        is_active[:, None]
        & j_real[None, :]
        & (j_val[None, :] < slot_topks[:, None])
    )                                                                        # [K, J_PAD]

    within_leaf_off_2d = cum_alt_slot_excl[:, None] + j_vec[None, :]         # [K, J_PAD]
    alt_pos_2d = tree_start + N * K + leaf_alt_base + within_leaf_off_2d     # [K, J_PAD]

    # Load top_target / top_lps at [leaf, slot, j_val]
    tt_off_2d = leaf * K * MAX_TOPK + slot_vec[:, None] * MAX_TOPK + j_val[None, :]
    tt_2d = tl.load(top_target_all_ptr + tt_off_2d, mask=valid_2d, other=0)
    tlp_2d = tl.load(top_lps_all_ptr + tt_off_2d, mask=valid_2d, other=0.0)

    # Parent: slot 0 → pend_node ; slot>0 → chain_base + slot - 1
    chain_parent_vec = chain_base + slot_vec.to(tl.int64) - 1                # [K] i64
    alt_parent_2d = tl.where(
        slot_vec[:, None] == 0,
        tl.broadcast_to(pend_node.to(tl.int64)[None, None], [K, J_PAD]),
        tl.broadcast_to(chain_parent_vec[:, None], [K, J_PAD]),
    )                                                                        # [K, J_PAD]

    parent_lp_2d = (pend_cum_lp + cum_greedy_lps_excl)[:, None]              # [K, 1] f32
    alt_lp_2d = parent_lp_2d + tlp_2d                                        # [K, J_PAD]

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
#   Python wrappers
# =============================================================================

def triton_build_block1(
    rank_preds: torch.Tensor,          # [K] i64
    greedy_target: torch.Tensor,       # [K] i64
    greedy_lps: torch.Tensor,          # [K] f32
    all_top_target: torch.Tensor,      # [K, MAX_TOPK] i64
    all_top_lps: torch.Tensor,         # [K, MAX_TOPK] f32
    rank_slot_topk_table: torch.Tensor,  # [RANK_CLASSES] i64
    tree_buf: dict,
    pend_buf: dict,
    sizes_buf: torch.Tensor,           # [4] i64 scratch
    beam_width: int,
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
    adaptive_slot0: int,
    adaptive_all: int,
):
    """Launch block-1 mega-kernel. Returns (n_nodes_b1, n_active, total_alts1, N_pend) ints."""
    _block1_mega_kernel[(1,)](
        rank_preds, greedy_target, greedy_lps,
        all_top_target, all_top_lps,
        rank_slot_topk_table,
        tree_buf['tokens'], tree_buf['parents'], tree_buf['lps'],
        tree_buf['ranks'], tree_buf['blocks'], tree_buf['slots'],
        pend_buf['hidden_slots'], pend_buf['input_ids'],
        pend_buf['ttt_valid'], pend_buf['node_indices'], pend_buf['cum_lps'],
        sizes_buf,
        beam_width,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class,
        ADAPTIVE_SLOT0_MODE=adaptive_slot0,
        ADAPTIVE_ALL_MODE=adaptive_all,
    )
    sizes_cpu = sizes_buf.cpu().tolist()  # 1 sync; 4 ints
    return sizes_cpu[0], sizes_cpu[1], sizes_cpu[2], sizes_cpu[3]


def triton_build_bfs(
    all_rank_preds: torch.Tensor,      # [N, K]
    all_greedy_target: torch.Tensor,   # [N, K]
    all_greedy_lps: torch.Tensor,      # [N, K]
    top_target_all: torch.Tensor,      # [N, K, MAX_TOPK]
    top_lps_all: torch.Tensor,         # [N, K, MAX_TOPK]
    pend_cum_lps: torch.Tensor,        # [N] f32
    pend_node_indices: torch.Tensor,   # [N] i64
    rank_slot_topk_table: torch.Tensor,
    tree_buf: dict,
    sizes_buf: torch.Tensor,           # [1] i64 scratch
    cum_alt_buf: torch.Tensor,         # [BLOCK_N] i64 scratch for leaf offsets
    tree_start: int,
    N: int,
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
    adaptive_all: int,
    pend_depth: int,
    block_n: int = 32,
    j_pad: int = 16,
):
    """Launch BFS kernels: sizing (1 program) + scatter (grid=N). Returns total_alts_2 int.

    The sizing kernel computes per-leaf n_alt prefix-sum into cum_alt_buf and
    writes total_alts_2 to sizes_buf. The scatter kernel runs grid=N programs
    in parallel (one SM each on A100), reading cum_alt_buf for the leaf's
    global offset and writing greedy chain + alternatives for that leaf.
    """
    _bfs_sizing_kernel[(1,)](
        all_rank_preds, all_greedy_lps, rank_slot_topk_table,
        cum_alt_buf, sizes_buf,
        N,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        ADAPTIVE_ALL_MODE=adaptive_all,
        BLOCK_N=block_n,
    )
    _bfs_scatter_kernel[(N,)](
        all_rank_preds, all_greedy_target, all_greedy_lps,
        top_target_all, top_lps_all,
        pend_cum_lps, pend_node_indices,
        rank_slot_topk_table, cum_alt_buf,
        tree_buf['tokens'], tree_buf['parents'], tree_buf['lps'],
        tree_buf['ranks'], tree_buf['blocks'], tree_buf['slots'],
        N, tree_start,
        K=K, MAX_TOPK=max_topk, RANK_CLASSES=rank_classes,
        GIVE_UP_CLASS=give_up_class,
        ADAPTIVE_ALL_MODE=adaptive_all,
        PEND_DEPTH=pend_depth,
        J_PAD=j_pad,
    )
    return int(sizes_buf.cpu().item())
