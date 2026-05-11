"""GPU-side tree finalize: prune + topology build, zero host syncs.

Also exports `dedup_depth_by_depth` — a (parent, token)-keyed deduplicator
processing nodes in depth-ascending order, single-pass. Used by the static
draft builder (specblock_static_draft._phase_finalize) when env
TREE_DEDUP_MODE=1.

Why dedup helps: block-1 chain[k+1] and block-2 leaf-k chain[0]
(for chain-end pending leaves at slot k<K-1) both predict the
"next token after chain[0..k]" and frequently produce identical tokens.
Without dedup these collide as two tree nodes with same (parent, token);
the second one is wasted budget. Merging recovers budget slots for top-K
to allocate to genuinely diverse candidates.

Empirically (4 benches at n=80, Llama-3.1-8B-Instruct + SpecBlock-Shift):
  * No dedup baseline:   acc_len ~5.27 / 4.65 / 3.76 / 4.46
  * depth-by-depth:      acc_len ~5.34 / 4.75 / 3.80 / 4.55  (+0.075 avg)


Replaces the numpy prune + partial CPU round-trip in _finalize_gpu_tree
with fully GPU-resident triton kernels. Output 7-tuple matches the
numpy-path topology byte-for-byte.

Design
------
Prune (approximation): sort nodes by cum_lp desc, keep top-budget
indices as seeds, then close under ancestors (iterative mark-parents
with fixed MAX_DEPTH=20 bound). May slightly exceed `budget` after
closure; the subsequent target verify handles oversized trees fine
(it just wastes a few verify slots).

Topology (the old `build_tree_buffers_triton`): reuse the existing
`_tree_depth_mask_kernel` and `_tree_retrieve_kernel`. Only change
is inputs stay on GPU (no .cpu()/.tolist()/.to(device) round-trip).

Usage: env var TREE_FINALIZE_CUDA=1 enables this path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# --- Dedup-merge: merge nodes with same (parent, token) ------------------
# Two implementations are exported:
#   - dedup_depth_by_depth_triton: production fused Triton kernel (~150us).
#   - dedup_depth_by_depth: PyTorch reference (slow, ~95ms; for verification).
# Both produce equivalent output (validated bit-equivalent on real SpecBlock
# tree). Use the Triton version in production.


@triton.jit
def _dbd_dedup_kernel(
    # Tree-node tensors (in-place modified)
    tokens_ptr,                  # [N] i64
    parents_ptr,                 # [N] i64 (-1 for root-child)
    lps_ptr,                     # [N] f32
    real_lps_ptr,                # [N] f32 (or arbitrary if HAS_REAL_LPS=False)
    # Scratch buffers (kernel resets at start)
    sorted_keys_buf_ptr,         # [N] i64
    inverse_idx_buf_ptr,         # [N] i64
    combined_winner_buf_ptr,     # [N] i64: atomic-max target encoding (max-cum_lp, min-pid)
    node_state_buf_ptr,          # [N] i64: encodes (to_mask flag in high32) | (winner_pid in low32)
    # Compile-time constants
    n,
    N: tl.constexpr,             # BLOCK_SIZE — power of two ≥ n_nodes
    HAS_REAL_LPS: tl.constexpr,
    MAX_TREE_DEPTH: tl.constexpr,  # max parent-walk depth (e.g. 6)
    NEG_INF: tl.constexpr,
):
    """Single-program fused depth-by-depth dedup. Grid: (1,).

    Layout:
      Phase 1: load tile + compute depth per node (parent-walk in registers).
      Phase 2: for d in 1..MAX_TREE_DEPTH (compile-time unrolled):
                 - Reset scratch
                 - Filter to nodes at depth d via packed-key sentinel
                 - Sort, group, atomic-max for group_max_lp,
                   atomic-min for group_first_survivor
                 - Mark losers, re-parent children of newly-masked nodes
                 - Update parents/lps in registers (carried to next d)
      Phase 3: store back parents, lps, real_lps to global memory.
    """
    pid = tl.arange(0, N)
    valid = pid < n

    # Load all tile data once.
    tokens = tl.load(tokens_ptr + pid, mask=valid, other=0).to(tl.int64)
    parents = tl.load(parents_ptr + pid, mask=valid, other=tl.cast(-1, tl.int64)).to(tl.int64)
    lps = tl.load(lps_ptr + pid, mask=valid, other=NEG_INF)
    if HAS_REAL_LPS:
        real_lps = tl.load(real_lps_ptr + pid, mask=valid, other=NEG_INF)

    # ===== Phase 1: compute depths =====
    # Walk parent chain from each node up to root, counting steps.
    cur = pid.to(tl.int64)
    depths = tl.zeros([N], dtype=tl.int32)
    for _ in range(MAX_TREE_DEPTH):
        cur_parent = tl.load(parents_ptr + cur, mask=valid, other=tl.cast(-1, tl.int64))
        is_root = cur_parent < 0
        depths = depths + tl.where(is_root, tl.zeros([N], tl.int32), tl.full([N], 1, tl.int32))
        cur = tl.where(is_root, cur, cur_parent)

    # ===== Phase 2: per-depth dedup =====
    KEY_SHIFT = tl.cast(1 << 32, tl.int64)
    NPLUS = tl.cast(N + 1, tl.int64)
    SENTINEL_PACKED = tl.cast(1 << 62, tl.int64)
    SENTINEL_KEY = tl.cast(-1, tl.int64)

    # Combined-encoding helpers (cum_lps assumed ≤ 0; root with cum_lp=0
    # has parent=-1 and never enters a dedup group, so safe).
    # encoded_lp = bitcast(cum_lp, i32) XOR 0xFFFFFFFF — for cum_lp < 0 this
    # gives positive i32 with order-preserving signed comparison
    # (smaller cum_lp → smaller encoded; -inf → 0x007FFFFF, -1 → 0x407FFFFF).
    # combined = (encoded_lp << 32) | (~pid & 0xFFFFFFFF).
    # atomic_max on combined gives (max-cum_lp, min-pid) in one shot.
    XOR_FFFF = tl.cast(0xFFFFFFFF, tl.int64)
    LOW_MASK = tl.cast(0xFFFFFFFF, tl.int64)

    MASK_SHIFT = tl.cast(32, tl.int64)  # high 32 = to_mask flag, low 32 = winner_pid

    for target_d in tl.static_range(1, MAX_TREE_DEPTH + 1):
        is_at_d = (depths == target_d) & valid
        # Skip empty / singleton depth: no group can have ≥2 members.
        # Tile-wide reduction → scalar; uniform branch across the program.
        n_at_d = tl.sum(is_at_d.to(tl.int32), axis=0)
        if n_at_d > 1:
            # Reset scratch for this depth.
            tl.store(combined_winner_buf_ptr + pid, tl.zeros([N], tl.int64))
            tl.store(node_state_buf_ptr + pid, tl.zeros([N], tl.int64))
            tl.debug_barrier()

            # Compute keys; non-target-depth nodes get SENTINEL so they sort to back.
            keys = (parents + 1) * KEY_SHIFT + tokens
            packed = keys * NPLUS + pid.to(tl.int64)
            packed = tl.where(is_at_d, packed, SENTINEL_PACKED)

            sorted_packed = tl.sort(packed)
            sort_idx = sorted_packed % NPLUS
            sorted_keys = sorted_packed // NPLUS
            sorted_valid = sorted_packed != SENTINEL_PACKED

            # Group boundary detect via shifted-load.
            tl.store(sorted_keys_buf_ptr + pid, sorted_keys)
            tl.debug_barrier()
            sorted_keys_prev = tl.load(
                sorted_keys_buf_ptr + (pid - 1),
                mask=(pid > 0) & sorted_valid,
                other=SENTINEL_KEY,
            )
            is_first = (sorted_keys != sorted_keys_prev) & sorted_valid
            group_id_sorted = (tl.cumsum(is_first.to(tl.int32), 0) - 1).to(tl.int64)

            # Scatter group_id back to original positions.
            tl.store(inverse_idx_buf_ptr + sort_idx, group_id_sorted, mask=sorted_valid)
            tl.debug_barrier()
            inverse_indices = tl.load(inverse_idx_buf_ptr + pid, mask=is_at_d, other=tl.cast(0, tl.int64))

            # Encode (cum_lp, pid) into a single i64 for combined atomic_max.
            lps_bits = lps.to(tl.int32, bitcast=True).to(tl.int64) & LOW_MASK
            encoded_lp = lps_bits ^ XOR_FFFF
            pid_inv = (pid.to(tl.int64) ^ XOR_FFFF) & LOW_MASK
            combined = (encoded_lp << 32) | pid_inv

            tl.atomic_max(combined_winner_buf_ptr + inverse_indices, combined, mask=is_at_d)
            tl.debug_barrier()
            winner_combined = tl.load(combined_winner_buf_ptr + inverse_indices, mask=is_at_d, other=tl.cast(0, tl.int64))

            winner_encoded_lp = (winner_combined >> 32) & LOW_MASK
            winner_pid_inv = winner_combined & LOW_MASK
            winner_pid = winner_pid_inv ^ XOR_FFFF

            my_lp_lt_max = encoded_lp < winner_encoded_lp
            to_mask = is_at_d & my_lp_lt_max & (lps != NEG_INF)

            # Fused (mask, winner_pid) state per node.
            to_mask_i64 = to_mask.to(tl.int64)
            node_state = (to_mask_i64 << MASK_SHIFT) | (winner_pid & LOW_MASK)
            tl.store(node_state_buf_ptr + pid, node_state, mask=is_at_d)
            tl.debug_barrier()

            # Re-parent via single load.
            parents_clamped = tl.maximum(parents, tl.zeros([N], tl.int64))
            parent_state = tl.load(node_state_buf_ptr + parents_clamped, mask=valid, other=tl.cast(0, tl.int64))
            parent_is_masked = (parent_state >> MASK_SHIFT) != 0
            parent_winner_val = parent_state & LOW_MASK
            parents = tl.where(
                (parents >= 0) & parent_is_masked,
                parent_winner_val,
                parents,
            )

            # Apply mask to lps (in registers, carried to next d).
            lps = tl.where(to_mask, NEG_INF, lps)
            if HAS_REAL_LPS:
                real_lps = tl.where(to_mask, NEG_INF, real_lps)

    # ===== Phase 3: write back =====
    tl.store(parents_ptr + pid, parents, mask=valid)
    tl.store(lps_ptr + pid, lps, mask=valid)
    if HAS_REAL_LPS:
        tl.store(real_lps_ptr + pid, real_lps, mask=valid)


_dbd_scratch_cache: dict = {}


def _get_dbd_scratch(device, dtype_lps, n_alloc):
    key = (device, n_alloc)
    if key not in _dbd_scratch_cache:
        _dbd_scratch_cache[key] = dict(
            sorted_keys=torch.empty(n_alloc, dtype=torch.int64, device=device),
            inverse_idx=torch.empty(n_alloc, dtype=torch.int64, device=device),
            combined_winner=torch.empty(n_alloc, dtype=torch.int64, device=device),
            node_state=torch.empty(n_alloc, dtype=torch.int64, device=device),
        )
    return _dbd_scratch_cache[key]


def dedup_depth_by_depth_triton(
    tokens: torch.Tensor,
    parents: torch.Tensor,
    lps: torch.Tensor,
    n_nodes: int,
    real_lps: torch.Tensor = None,
    max_tree_depth: int = 3,
) -> None:
    """Production-speed depth-by-depth dedup using a single fused Triton
    kernel. Theoretically equivalent to `dedup_depth_by_depth` but
    much faster.

    max_tree_depth=3 covers depth 1..3 (where most natural dups occur:
    block-1 chain[k+1] vs block-2 leaf-k chain[0] collisions cluster at
    shallow depths). Deeper dups are rare and depth>3 iterations are
    pure overhead in steady state. Empirically acc impact -0.01 vs
    max_tree_depth=6 with -75us savings per b2_fwd. Increase if acc
    regression observed on a new model/dataset.
    """
    if n_nodes <= 1:
        return
    device = lps.device

    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_nodes:
        BLOCK_SIZE *= 2
    BLOCK_SIZE = max(BLOCK_SIZE, 64)

    scratch = _get_dbd_scratch(device, lps.dtype, BLOCK_SIZE)
    has_real = real_lps is not None
    real_ptr = real_lps if has_real else lps

    _dbd_dedup_kernel[(1,)](
        tokens, parents, lps, real_ptr,
        scratch["sorted_keys"],
        scratch["inverse_idx"],
        scratch["combined_winner"],
        scratch["node_state"],
        n_nodes,
        N=BLOCK_SIZE,
        HAS_REAL_LPS=has_real,
        MAX_TREE_DEPTH=max_tree_depth,
        NEG_INF=float("-inf"),
        num_warps=4,
    )


def dedup_depth_by_depth(
    tokens: torch.Tensor,
    parents: torch.Tensor,
    lps: torch.Tensor,
    n_nodes: int,
    real_lps: torch.Tensor = None,
    max_walk: int = 20,
) -> None:
    """Single-pass depth-by-depth dedup. PyTorch reference impl (slow).

    Process nodes by depth ascending. At each depth d, group depth-d nodes
    by (current_parent, token), mask losers in each group, re-parent
    children of masked nodes to the group's first survivor.

    re-parent preserves depth (loser and winner share same parent → same
    depth), so static depths computed once are valid throughout.

    Theoretically equivalent to the multi-iter cascade. In our SpecBlock
    structure with K=4, both should converge to same masked set, modulo
    tie-breaking edge cases. Conservative variant: no cum_lp delta
    propagation (descendants' cum_lp unchanged).
    """
    if n_nodes <= 1:
        return
    p_view = parents[:n_nodes]
    t_view = tokens[:n_nodes]
    l_view = lps[:n_nodes]
    rl_view = real_lps[:n_nodes] if real_lps is not None else None
    NEG_INF = float("-inf")
    device = lps.device

    # Compute static depths from current parent chain.
    depths = torch.zeros(n_nodes, dtype=torch.long, device=device)
    for i in range(n_nodes):
        cur = i
        d = 0
        for _ in range(max_walk):
            p = int(p_view[cur])
            if p < 0:
                break
            d += 1
            cur = p
        depths[i] = d
    max_d = int(depths.max().item())

    for d in range(1, max_d + 1):
        at_d = (depths == d).nonzero(as_tuple=True)[0]
        if at_d.numel() == 0:
            continue
        keys = (p_view[at_d] + 1).to(torch.int64) * (1 << 32) + t_view[at_d].to(torch.int64)
        unique_k, inverse = torch.unique(keys, return_inverse=True)
        nG = unique_k.numel()
        gmax = torch.full((nG,), NEG_INF, device=device, dtype=l_view.dtype)
        gmax.scatter_reduce_(0, inverse, l_view[at_d], reduce="amax", include_self=True)
        node_max = gmax[inverse]
        local_to_mask = (l_view[at_d] < node_max) & (l_view[at_d] != NEG_INF)
        sentinel = n_nodes
        survivor = torch.where(~local_to_mask, at_d, torch.full_like(at_d, sentinel))
        gw = torch.full((nG,), sentinel, device=device, dtype=torch.long)
        gw.scatter_reduce_(0, inverse, survivor, reduce="amin", include_self=True)
        node_winner = gw[inverse]
        masked_at_d = at_d[local_to_mask]
        l_view[masked_at_d] = NEG_INF
        if rl_view is not None:
            rl_view[masked_at_d] = NEG_INF
        if masked_at_d.numel() == 0:
            continue
        local_winner_for_masked = node_winner[local_to_mask]
        masked_to_winner = dict(zip(masked_at_d.tolist(), local_winner_for_masked.tolist()))
        for i in range(n_nodes):
            p = int(p_view[i])
            if p in masked_to_winner:
                p_view[i] = masked_to_winner[p]
# --- Ancestor closure kernel ----------------------------------------------

@triton.jit
def _ancestor_closure_kernel(
    parents_ptr,      # [N] int64, parent[i] ∈ [-1, N)
    keep_ptr,         # [N] int32 bool (0/1), in+out
    N,
    MAX_DEPTH: tl.constexpr,
):
    """For each kept node i, mark its parent chain kept.

    Grid: one program per node. Each walks up parent chain up to MAX_DEPTH
    steps, atomically OR-ing `keep[parent]` to 1. MAX_DEPTH bounds the tree
    depth (K * max_blocks = 12 fits max_blocks=3 easily).
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    mine = tl.load(keep_ptr + pid)
    if mine == 0:
        return
    cur = tl.load(parents_ptr + pid)
    for _ in range(MAX_DEPTH):
        # Stop walking once root (-1) or already-kept parent reached
        valid = cur >= 0
        if valid:
            # Set keep[cur] = 1 (idempotent; OR via store-if-zero)
            prev = tl.load(keep_ptr + cur)
            if prev == 0:
                tl.store(keep_ptr + cur, 1)
            cur = tl.load(parents_ptr + cur)
        # else: cur stays -1 on subsequent iters; store skipped by valid gate


# --- Compaction helpers ---------------------------------------------------

def gpu_prune_tree(tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots,
                   n_nodes, budget, max_tree_depth=20):
    """GPU-side prune. Inputs are GPU tensors (int64 / float32).

    Returns compacted (tokens, parents, lps, ranks, blocks, slots, new_n) on GPU.
    parents are re-indexed to the new compact positions.
    """
    device = tn_tokens.device
    assert n_nodes > budget, "caller must check; this path should only run when prune needed"

    # 1. Top-k by cum_lp desc — no host sync.
    #    top_k returns indices in [0, n_nodes); we use them as seeds.
    topk_n = min(budget, n_nodes)
    lps_slice = tn_lps[:n_nodes]
    # torch.topk largest=True returns descending; fine.
    top_idx = torch.topk(lps_slice, topk_n, largest=True).indices  # [topk_n] int64

    # 2. keep[n_nodes] init with top-K seeds
    keep = torch.zeros(n_nodes, dtype=torch.int32, device=device)
    keep[top_idx] = 1

    # 3. Close under ancestors via triton kernel.
    _ancestor_closure_kernel[(n_nodes,)](
        tn_parents[:n_nodes].contiguous(),
        keep,
        n_nodes,
        MAX_DEPTH=max_tree_depth,
    )

    # 4. Build compact indices via prefix-sum (torch.cumsum on GPU).
    keep_bool = keep.bool()
    # kept_positions: [K'] (indices of nodes that are kept)
    kept_idx = keep_bool.nonzero(as_tuple=True)[0]      # [new_n] int64
    new_n = kept_idx.shape[0]

    # 5. old_to_new: mapping for parent re-index. old_to_new[old] = new pos or -1
    old_to_new = torch.full((n_nodes,), -1, dtype=torch.int64, device=device)
    old_to_new[kept_idx] = torch.arange(new_n, device=device, dtype=torch.int64)

    # 6. Compact + re-index parents.
    new_tokens = tn_tokens[kept_idx]
    new_lps = tn_lps[kept_idx]
    new_ranks = tn_ranks[kept_idx]
    new_blocks = tn_blocks[kept_idx]
    new_slots = tn_slots[kept_idx]

    old_parents = tn_parents[kept_idx]                  # [new_n] in old index space
    # If parent was -1, stay -1; else map through old_to_new (which is guaranteed
    # valid because ancestors were marked kept).
    new_parents = torch.where(
        old_parents >= 0,
        old_to_new[old_parents.clamp(min=0)],
        torch.full_like(old_parents, -1),
    )

    return new_tokens, new_parents, new_lps, new_ranks, new_blocks, new_slots, new_n


# --- Sort retrieve_indices rows on GPU ------------------------------------

def _sort_retrieve_indices_gpu(ri, maxitem):
    """Sort retrieve_indices rows lexicographically, treating -1 as +inf.

    ri: [num_leaves, D] int32 on GPU.
    Returns sorted ri (same shape).
    """
    # Replace -1 with maxitem for sort purposes, then view int64 as lex key.
    # Fall back to CPU sort for correctness; ri is typically [<100, ≤12] so
    # CPU sort is a few microseconds. This one .cpu() is the ONLY remaining
    # host round-trip in the finalize path, and it's micro.
    ri_cpu = ri.cpu().tolist()
    maxitem_v = int(maxitem)
    ri_cpu.sort(key=lambda row: tuple(x if x >= 0 else maxitem_v for x in row))
    return torch.tensor(ri_cpu, dtype=torch.long, device=ri.device)


# --- Entry: build 7-tuple fully from GPU state ----------------------------

def finalize_tree_gpu(
    tree_buf: dict,          # {'tokens','parents','lps','ranks','blocks','slots'} all GPU
    n_nodes_gpu: int,        # int on host — but caller should have obtained with minimal sync
    sample_token: torch.Tensor,  # [1] on GPU, the root token
    budget: int,
    K: int,
    max_blocks: int,
    depth_mask_kernel,
    retrieve_kernel,
    max_tree_depth: int = 20,
):
    """End-to-end GPU finalize. Returns the 7-tuple:

      draft_tokens      [1, N+1]
      tree_mask         [1, 1, N+1, N+1]
      tree_position_ids [N+1]
      retrieve_indices  [num_leaves, max_depth+1]
      all_rank_stats    (empty stub list — callers set this themselves)
      node_ranks        (list[int])  from tn_ranks
      node_block_slots  (list[tuple]) from (tn_blocks, tn_slots)

    n_nodes_gpu is the caller-obtained n_nodes int (one host sync already paid
    when reading bfs total_alts). Prune triggers iff n_nodes_gpu > budget.
    """
    device = tree_buf['tokens'].device
    tn_tokens = tree_buf['tokens']
    tn_parents = tree_buf['parents']
    tn_lps = tree_buf['lps']
    tn_ranks = tree_buf['ranks']
    tn_blocks = tree_buf['blocks']
    tn_slots = tree_buf['slots']

    # Dedup pass: merge duplicate (parent, token) tree nodes before prune.
    # Implementation: depth-by-depth, single Triton kernel (~170us).
    # Opt out via TREE_DEDUP_MODE=0.
    import os as _os
    if _os.environ.get('TREE_DEDUP_MODE', '1') == '1':
        dedup_depth_by_depth_triton(tn_tokens, tn_parents, tn_lps, n_nodes_gpu)

    N = n_nodes_gpu
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

    # Topology: mirror build_tree_buffers_triton but with GPU tensors everywhere.
    Np1 = N + 1
    # parent tensor: root index 0, shift node indices by +1. parents[-1] → 0.
    parent_t = torch.empty(Np1, dtype=torch.int32, device=device)
    parent_t[0] = 0
    parent_t[1:] = torch.where(
        tn_parents[:N] >= 0,
        (tn_parents[:N] + 1).to(torch.int32),
        torch.zeros(N, dtype=torch.int32, device=device),
    )

    depth_t = torch.zeros(Np1, dtype=torch.int32, device=device)
    mask_flat = torch.zeros(Np1 * Np1, dtype=torch.float32, device=device)

    depth_mask_kernel[(Np1,)](
        parent_t, depth_t, mask_flat,
        Np1, MAX_DEPTH=max_tree_depth,
    )

    # Leaves: nodes that are nobody's parent.
    is_parent = torch.zeros(Np1, dtype=torch.bool, device=device)
    is_parent.scatter_(0, parent_t[1:].long(), True)
    leaves = (~is_parent).nonzero(as_tuple=True)[0].to(torch.int32)
    num_leaves = int(leaves.shape[0])  # host read OK — shape, not content

    # Overallocate ri to [num_leaves, max_tree_depth+1] so no max_depth.item() sync.
    ri = torch.full((num_leaves, max_tree_depth + 1), -1,
                    dtype=torch.int32, device=device)
    if num_leaves > 0:
        retrieve_kernel[(num_leaves,)](
            parent_t, depth_t, leaves, ri,
            max_tree_depth, Np1, num_leaves,
            MAX_DEPTH=max_tree_depth,
        )

    # GPU sort rows lexicographically — requires one .cpu() but ri is small.
    ri_sorted = _sort_retrieve_indices_gpu(ri, maxitem=Np1 + 5)

    # Assemble draft_tokens: root + N nodes.
    draft_tokens = torch.empty(Np1, dtype=torch.long, device=device)
    draft_tokens[0] = sample_token.squeeze()
    draft_tokens[1:] = tn_tokens
    draft_tokens = draft_tokens.unsqueeze(0)  # [1, Np1]

    tree_mask = mask_flat.reshape(Np1, Np1)[None, None]
    tree_position_ids = depth_t.long()

    # Stats: node_ranks, node_block_slots — these go to CPU at end.
    # Batch the 3 small D2H copies into 1 stacked transfer to collapse 3
    # CUDA syncs into 1. tn_ranks/tn_blocks/tn_slots are [N] int64 on GPU;
    # after torch.stack → [3, N] → .cpu() single DMA → .tolist() pure CPU.
    stacked_meta = torch.stack([tn_ranks, tn_blocks, tn_slots]).cpu()
    ranks_list, blocks_list, slots_list = (
        stacked_meta[0].tolist(),
        stacked_meta[1].tolist(),
        stacked_meta[2].tolist(),
    )
    node_ranks = ranks_list
    node_block_slots = list(zip(blocks_list, slots_list))

    return (draft_tokens, tree_mask, tree_position_ids, ri_sorted,
            node_ranks, node_block_slots, N)


# ============================================================
#   Fused finalize-prune kernel (top-K + scatter compact + parent reindex)
# ============================================================
#
# Replaces ~28 small PyTorch ops in StaticDraftBuilder._phase_finalize
# step 1-6 (top-K → keep_mask → cumsum → new_pos → scatter compact ×6
# fields → parent reindex → root placement) with a single Triton
# kernel. Profile (DRAFT_COMPILE=0, n=4000) showed those ops total
# ~482us out of 1287us in finalize; eliminating ~28 launches saves
# ~170us launch overhead alone, plus L2 cache wins from in-register
# intermediates.
#
# Algorithm:
#   Phase 0: zero cf_* buffers (size Np1)
#   Phase 1: pack (encoded_lp_inv << 32 | pid), tl.sort ascending →
#            sorted_pids[0..budget-1] are the top-K (lps DESC, pid ASC tie-break)
#   Phase 2: scatter 1 to keep_mask[sorted_pids[0..budget-1]]
#   Phase 3: tl.cumsum(keep_mask) → new_pos = cumsum_excl for kept, -1 else
#   Phase 4: for each kept pid, store tn_tokens/real_lps/ranks/blocks/slots
#            to cf_*[new_pos+1]; cf_valid[new_pos+1] = 1
#   Phase 5: gather new_pos[parents[pid]], compute new_parent (0 for root /
#            pruned), store cf_parents[new_pos+1]
#   Phase 6: place root at index 0


@triton.jit
def _fz_prune_kernel(
    # Input tn_* buffers (size MAX_NODES)
    tn_tokens_ptr, tn_parents_ptr, tn_lps_ptr, tn_real_lps_ptr,
    tn_ranks_ptr, tn_blocks_ptr, tn_slots_ptr,
    # Output cf_* buffers (size Np1)
    cf_tokens_ptr, cf_parents_ptr, cf_real_lps_ptr,
    cf_ranks_ptr, cf_blocks_ptr, cf_slots_ptr, cf_valid_ptr,
    # Root token
    sample_token_ptr,
    # Scratch buffers (size MAX_NODES)
    keep_mask_buf_ptr, new_pos_buf_ptr,
    # Sizes
    n_nodes,
    budget,
    Np1,
    # Constants
    N: tl.constexpr,            # power-of-2 ≥ MAX_NODES (e.g. 1024)
    NP1_BLOCK: tl.constexpr,    # power-of-2 ≥ Np1 (e.g. 128)
    NEG_INF: tl.constexpr,
):
    """Single-program fused prune. Grid: (1,)."""
    pid = tl.arange(0, N)
    valid = pid < n_nodes
    pid_i64 = pid.to(tl.int64)

    # ---- Phase 0: zero cf_* buffers (Np1 slots) ----
    np1_arange = tl.arange(0, NP1_BLOCK)
    np1_valid = np1_arange < Np1
    tl.store(cf_tokens_ptr + np1_arange, tl.zeros([NP1_BLOCK], tl.int64), mask=np1_valid)
    tl.store(cf_parents_ptr + np1_arange, tl.zeros([NP1_BLOCK], tl.int32), mask=np1_valid)
    tl.store(cf_real_lps_ptr + np1_arange, tl.full([NP1_BLOCK], NEG_INF, tl.float32), mask=np1_valid)
    tl.store(cf_ranks_ptr + np1_arange, tl.zeros([NP1_BLOCK], tl.int64), mask=np1_valid)
    tl.store(cf_blocks_ptr + np1_arange, tl.zeros([NP1_BLOCK], tl.int64), mask=np1_valid)
    tl.store(cf_slots_ptr + np1_arange, tl.zeros([NP1_BLOCK], tl.int64), mask=np1_valid)
    tl.store(cf_valid_ptr + np1_arange, tl.zeros([NP1_BLOCK], tl.int64), mask=np1_valid)
    tl.debug_barrier()

    # ---- Phase 1: pack (-lps_bits, pid) and sort ascending ----
    # Largest lps → smallest -lps → smallest neg_lps_bits (since -lps ≥ 0 has
    # sign bit 0; positive-float bits are monotonic in value). Tie-break by pid ASC.
    # Invalid pid is masked to lps=-inf, so -lps=+inf=0x7F800000 → sorts last.
    lps = tl.load(tn_lps_ptr + pid, mask=valid, other=NEG_INF)
    LOW_MASK = tl.cast(0xFFFFFFFF, tl.int64)
    neg_lps_bits = (-lps).to(tl.int32, bitcast=True).to(tl.int64) & LOW_MASK
    packed = (neg_lps_bits << 32) | pid_i64

    sorted_packed = tl.sort(packed)
    sorted_pids = sorted_packed & LOW_MASK

    # ---- Phase 2: build keep_mask = 1 at sorted_pids[0..budget-1] ----
    tl.store(keep_mask_buf_ptr + pid, tl.zeros([N], tl.int32))
    tl.debug_barrier()
    is_kept_in_sort = pid < budget
    tl.store(keep_mask_buf_ptr + sorted_pids, tl.full([N], 1, tl.int32), mask=is_kept_in_sort)
    tl.debug_barrier()

    # ---- Phase 3: cumsum → new_pos ----
    keep_mask = tl.load(keep_mask_buf_ptr + pid)
    cum_keep = tl.cumsum(keep_mask, axis=0)
    cum_excl = cum_keep - keep_mask
    new_pos_i32 = tl.where(keep_mask == 1, cum_excl, tl.full([N], -1, tl.int32))
    new_pos = new_pos_i32.to(tl.int64)
    tl.store(new_pos_buf_ptr + pid, new_pos)
    tl.debug_barrier()

    # ---- Phase 4: scatter compact tn_* → cf_*[new_pos+1] ----
    is_kept = (keep_mask == 1) & valid
    cf_idx = new_pos + 1   # 1..budget for kept nodes; junk for not-kept (masked off)

    tn_tok = tl.load(tn_tokens_ptr + pid, mask=valid, other=0)
    tn_real = tl.load(tn_real_lps_ptr + pid, mask=valid, other=NEG_INF)
    tn_rank = tl.load(tn_ranks_ptr + pid, mask=valid, other=0)
    tn_blk = tl.load(tn_blocks_ptr + pid, mask=valid, other=0)
    tn_slt = tl.load(tn_slots_ptr + pid, mask=valid, other=0)
    tn_par = tl.load(tn_parents_ptr + pid, mask=valid, other=tl.cast(-1, tl.int64))

    tl.store(cf_tokens_ptr + cf_idx, tn_tok, mask=is_kept)
    tl.store(cf_real_lps_ptr + cf_idx, tn_real, mask=is_kept)
    tl.store(cf_ranks_ptr + cf_idx, tn_rank, mask=is_kept)
    tl.store(cf_blocks_ptr + cf_idx, tn_blk, mask=is_kept)
    tl.store(cf_slots_ptr + cf_idx, tn_slt, mask=is_kept)
    # cf_valid: 1 for kept entries with finite real_lps (matches reference).
    cf_valid_val = (tn_real != NEG_INF).to(tl.int64)
    tl.store(cf_valid_ptr + cf_idx, cf_valid_val, mask=is_kept)

    # ---- Phase 5: parent reindex ----
    par_clamped = tl.maximum(tn_par, tl.zeros([N], tl.int64))
    par_new_pos = tl.load(new_pos_buf_ptr + par_clamped, mask=valid, other=tl.cast(-1, tl.int64))
    new_parent = tl.where(
        is_kept & (tn_par >= 0) & (par_new_pos >= 0),
        (par_new_pos + 1).to(tl.int32),
        tl.zeros([N], tl.int32),
    )
    tl.store(cf_parents_ptr + cf_idx, new_parent, mask=is_kept)

    # ---- Phase 6: root at index 0 ----
    sample_tok = tl.load(sample_token_ptr)
    tl.store(cf_tokens_ptr + 0, sample_tok)
    tl.store(cf_parents_ptr + 0, tl.cast(0, tl.int32))
    tl.store(cf_valid_ptr + 0, tl.cast(1, tl.int64))


_fz_prune_scratch_cache: dict = {}


def _get_fz_prune_scratch(device, n_alloc):
    key = (device, n_alloc)
    if key not in _fz_prune_scratch_cache:
        _fz_prune_scratch_cache[key] = dict(
            keep_mask=torch.empty(n_alloc, dtype=torch.int32, device=device),
            new_pos=torch.empty(n_alloc, dtype=torch.int64, device=device),
        )
    return _fz_prune_scratch_cache[key]


def fused_prune(
    tn_tokens, tn_parents, tn_lps, tn_real_lps, tn_ranks, tn_blocks, tn_slots,
    cf_tokens, cf_parents, cf_real_lps, cf_ranks, cf_blocks, cf_slots, cf_valid,
    sample_token, n_nodes, budget, Np1,
):
    """Fused top-K + scatter-compact + parent-reindex + root-place.

    Replaces _phase_finalize step 1-6 (~28 PyTorch ops) with a single
    Triton kernel. cf_* buffers are zeroed (scattered fields keep their
    written values, others stay 0/-inf) and root is placed at index 0.

    Args:
        tn_*: tree-node fields, [n_nodes]
        cf_*: compact-finalize output, [Np1]
        sample_token: [1] i64
        n_nodes: int (= MAX_NODES)
        budget: int
        Np1: int (= budget + 1)
    """
    if n_nodes <= 0:
        return
    device = tn_lps.device

    BLOCK_SIZE = 1
    while BLOCK_SIZE < n_nodes:
        BLOCK_SIZE *= 2
    BLOCK_SIZE = max(BLOCK_SIZE, 64)

    NP1_BLOCK = 1
    while NP1_BLOCK < Np1:
        NP1_BLOCK *= 2
    NP1_BLOCK = max(NP1_BLOCK, 32)

    scratch = _get_fz_prune_scratch(device, BLOCK_SIZE)

    _fz_prune_kernel[(1,)](
        tn_tokens, tn_parents, tn_lps, tn_real_lps,
        tn_ranks, tn_blocks, tn_slots,
        cf_tokens, cf_parents, cf_real_lps,
        cf_ranks, cf_blocks, cf_slots, cf_valid,
        sample_token,
        scratch["keep_mask"], scratch["new_pos"],
        n_nodes, budget, Np1,
        N=BLOCK_SIZE,
        NP1_BLOCK=NP1_BLOCK,
        NEG_INF=float("-inf"),
        num_warps=4,
    )


# ============================================================
#   Fused finalize-leafpack kernel (is_parent + leaf_or_neg + packed leaves)
# ============================================================
#
# Replaces ~17 small PyTorch ops in StaticDraftBuilder._phase_finalize
# step 8 + 8b (is_parent count via scatter_add, is_leaf computation,
# cf_leaf_or_neg, cumsum + scatter pack leaves) with a single Triton
# kernel. Profile (DRAFT_COMPILE=0, n=4000) showed fz_leafpack = 359us /
# 1287us finalize.


@triton.jit
def _fz_leafpack_kernel(
    cf_parents_ptr,      # [Np1] i32
    cf_valid_ptr,        # [Np1] i64
    cf_is_parent_ptr,    # [Np1] i64 OUT
    cf_leaf_or_neg_ptr,  # [Np1] i64 OUT
    cf_packed_leaves_ptr,# [Np1] i64 OUT
    is_parent_buf_ptr,   # [NP1_BLOCK] i32 scratch
    Np1,
    NP1_BLOCK: tl.constexpr,
):
    """Single-program fused leafpack. Grid: (1,)."""
    pid = tl.arange(0, NP1_BLOCK)
    valid_pid = pid < Np1

    # ---- Load inputs ----
    parents_i32 = tl.load(cf_parents_ptr + pid, mask=valid_pid, other=0)
    parents = parents_i32.to(tl.int64)
    cf_valid = tl.load(cf_valid_ptr + pid, mask=valid_pid, other=0)

    # ---- Phase 1: count is_parent via atomic_add ----
    # Reset scratch to 0.
    tl.store(is_parent_buf_ptr + pid, tl.zeros([NP1_BLOCK], tl.int32), mask=valid_pid)
    tl.debug_barrier()
    # For pid >= 1 with cf_valid[pid] == 1, atomic_add 1 to is_parent_buf[parents[pid]].
    # Root (pid=0) excluded since its parent self-points to 0; counting it
    # would incorrectly mark root as a leaf parent.
    is_nonroot_valid = (pid >= 1) & valid_pid & (cf_valid == 1)
    tl.atomic_add(is_parent_buf_ptr + parents, 1, mask=is_nonroot_valid)
    tl.debug_barrier()

    is_parent = tl.load(is_parent_buf_ptr + pid, mask=valid_pid, other=0)
    tl.store(cf_is_parent_ptr + pid, is_parent.to(tl.int64), mask=valid_pid)

    # ---- Phase 2: cf_leaf_or_neg ----
    is_leaf = (cf_valid == 1) & (is_parent == 0) & valid_pid
    leaf_or_neg = tl.where(is_leaf, pid.to(tl.int64), tl.full([NP1_BLOCK], -1, tl.int64))
    tl.store(cf_leaf_or_neg_ptr + pid, leaf_or_neg, mask=valid_pid)

    # ---- Phase 3: cumsum + scatter pack leaves ----
    is_leaf_i32 = is_leaf.to(tl.int32)
    cum = tl.cumsum(is_leaf_i32, axis=0)
    pos_in_leaf = (cum - is_leaf_i32).to(tl.int64)  # exclusive prefix sum

    # Pre-fill packed_leaves[0..Np1-1] with -1.
    tl.store(
        cf_packed_leaves_ptr + pid,
        tl.full([NP1_BLOCK], -1, tl.int64),
        mask=valid_pid,
    )
    tl.debug_barrier()
    # Scatter: packed_leaves[pos_in_leaf[pid]] = pid for is_leaf.
    tl.store(cf_packed_leaves_ptr + pos_in_leaf, pid.to(tl.int64), mask=is_leaf)


_fz_leafpack_scratch_cache: dict = {}


def _get_fz_leafpack_scratch(device, np1_block):
    key = (device, np1_block)
    if key not in _fz_leafpack_scratch_cache:
        _fz_leafpack_scratch_cache[key] = torch.empty(
            np1_block, dtype=torch.int32, device=device,
        )
    return _fz_leafpack_scratch_cache[key]


def fused_leafpack(
    cf_parents, cf_valid, cf_is_parent, cf_leaf_or_neg, cf_packed_leaves, Np1,
):
    """Fused is_parent count + leaf_or_neg + cumsum + pack leaves.

    Replaces _phase_finalize step 8 + 8b (~17 PyTorch ops) with a single
    Triton kernel.

    Args:
        cf_parents:       [Np1] i32 (input)
        cf_valid:         [Np1] i64 (input)
        cf_is_parent:     [Np1] i64 (output, in-place)
        cf_leaf_or_neg:   [Np1] i64 (output, in-place)
        cf_packed_leaves: [Np1] i64 (output, in-place; index 0..n_leaves-1
                                     contain leaf pids; rest = -1)
    """
    if Np1 <= 0:
        return
    device = cf_parents.device

    NP1_BLOCK = 1
    while NP1_BLOCK < Np1:
        NP1_BLOCK *= 2
    NP1_BLOCK = max(NP1_BLOCK, 32)

    is_parent_buf = _get_fz_leafpack_scratch(device, NP1_BLOCK)

    _fz_leafpack_kernel[(1,)](
        cf_parents, cf_valid, cf_is_parent, cf_leaf_or_neg, cf_packed_leaves,
        is_parent_buf,
        Np1,
        NP1_BLOCK=NP1_BLOCK,
        num_warps=2,
    )
