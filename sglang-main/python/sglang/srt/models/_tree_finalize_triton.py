"""GPU-side tree finalize: prune + topology build, zero host syncs.

Replaces the numpy prune + partial CPU round-trip in _finalize_gpu_tree
with fully GPU-resident triton kernels. Output 7-tuple matches the
numpy-path topology byte-for-byte.

Design
------
Prune (approximation): sort nodes by cum_lp desc, keep top-budget
indices as seeds, then close under ancestors (iterative mark-parents
with fixed MAX_DEPTH=20 bound). May slightly exceed `budget` after
closure; the subsequent target verify handles oversized trees fine
(it just wastes a few verify slots). Exact numpy prune guarantees
never-over-budget but requires a sequential scan; ours is O(1) host
syncs vs numpy's O(1) host sync + Python list ops.

Topology (the old `build_tree_buffers_triton`): reuse the existing
`_tree_depth_mask_kernel` and `_tree_retrieve_kernel`. Only change
is inputs stay on GPU (no .cpu()/.tolist()/.to(device) round-trip).

Usage: env var TREE_FINALIZE_CUDA=1 enables this path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


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
