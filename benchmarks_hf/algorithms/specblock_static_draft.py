"""SpecBlock static-shape, sync-free draft tree builder.

Self-contained module that replaces the dispatch + numpy/triton/CUDA matrix
in `_build_tree_from_block1_*` with a single static-shape pipeline:

  block-1 mega   →  block-2 batch prep  →  block-2 forward
  →  block-2 BFS →  finalize (top-K + scatter compact + topology + retrieve)

Design (EAGLE-Pangu-style "padded with validity" + rank-aware bias on prune):

* All slots branch uniformly to MAX_TOPK alternatives (no adaptive_slot0 /
  adaptive_all generation-time pruning). Tree shapes are entirely
  compile-time constants:

      block-1 tree nodes       = K * MAX_TOPK
      pending leaves           = K * MAX_TOPK
      block-2 tree nodes       = N_pend * K * MAX_TOPK
      total tree node capacity = K * MAX_TOPK * (1 + K * MAX_TOPK)

  For K=4, MAX_TOPK=6 → N_pend=24, total ≤ 600.

* Rank-aware semantics enter at prune via *cumulative* rank bias along the
  parent chain:

      cum_bias[node]  = sum_{ancestors a} rank_offset[rank_class[a]]
      effective_lp[node] = real_cum_lp[node] + cum_bias[node]

  `rank_offset` is non-positive (give_up class largest negative). Both
  contributions are non-increasing along the path, so cum_lp monotonicity
  is preserved → top-K(effective_lp, k=total_tokens) alone yields a tree
  closed under parentage (no ancestor-closure pass needed).

* All buffers pre-allocated at compile-time max shape. Zero host syncs on
  the hot path: no .cpu(), no .item(), no nonzero, no dynamic-shape ops.
  Compatible with torch.cuda.graph(...) capture as-is.

* Output 7-tuple is bit-compatible with `_build_tree_from_block1_*`,
  drop-in replacement under env `SPECBLOCK_STATIC=1`.
"""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
#                   Module-level constants
# ============================================================

MAX_TREE_DEPTH = 20  # >= 2 * K. Upper bound for parent-walk depth.


# ============================================================
#                     Triton kernels
# ============================================================

@triton.jit
def _block1_static_kernel(
    # ---- Inputs (GPU) ----
    rank_preds_ptr,          # [K] i64 — per-slot rank class
    top_target_ptr,          # [K, MAX_TOPK] i64 — target vocab ids (greedy at [:, 0])
    top_lps_ptr,             # [K, MAX_TOPK] f32 — log_probs of those targets
    rank_offset_ptr,         # [RANK_CLASSES] f32 — non-positive bias for prune
    # ---- Tree node outputs ----
    tn_tokens_ptr,           # [MAX_NODES] i64
    tn_parents_ptr,          # [MAX_NODES] i64
    tn_lps_ptr,              # [MAX_NODES] f32 — real_cum_lp + cum_bias (used for prune)
    tn_real_lps_ptr,         # [MAX_NODES] f32 — pure cum_lp (for diagnostics)
    tn_ranks_ptr,            # [MAX_NODES] i64
    tn_blocks_ptr,           # [MAX_NODES] i64 (=0 for block-1)
    tn_slots_ptr,            # [MAX_NODES] i64 (slot index k)
    # ---- Pending leaf outputs ----
    pend_hidden_slots_ptr,   # [N_PEND] i64 — slot k of block-1 hidden to use as input
    pend_input_ids_ptr,      # [N_PEND] i64
    pend_ttt_valid_ptr,      # [N_PEND] i64 (= k+1)
    pend_node_indices_ptr,   # [N_PEND] i64 — tree node id of this pending leaf
    pend_cum_lps_ptr,        # [N_PEND] f32 — REAL cum_lp (no bias)
    pend_cum_bias_ptr,       # [N_PEND] f32 — cum_bias (path-cumulative rank offset)
    # ---- Compile-time constants ----
    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
):
    """Block-1 mega kernel — single program, K-serial unrolled.

    Tree-buf layout (positions 0..K*MAX_TOPK):
      * Greedy chain: [0..K), pos k = slot-k greedy
      * Alts:        [K..K*MAX_TOPK), per-slot per-alt:
                     pos = K + k*(MAX_TOPK-1) + (j-1)

    Pending-buf layout (positions 0..K*MAX_TOPK):
      * Chain-end pending: [0..K), pos k references chain[k]
      * Alt pending:       [K..K*MAX_TOPK), same indexing as alts

    Cumulative rank bias is propagated along the chain so monotonicity of
    `tn_lps` along parent links is preserved.
    """
    if tl.program_id(0) != 0:
        return

    # cum_glp_before[k] = sum_{i<k} top_lps[i, 0]
    # cum_bias_before[k] = sum_{i<k} rank_offset[rank_preds[i]]
    cum_glp_before = tl.cast(0.0, tl.float32)
    cum_bias_before = tl.cast(0.0, tl.float32)

    for k in range(K):
        glp_k = tl.load(top_lps_ptr + k * MAX_TOPK + 0)
        gt_k  = tl.load(top_target_ptr + k * MAX_TOPK + 0)
        r_k   = tl.load(rank_preds_ptr + k)
        bias_k = tl.load(rank_offset_ptr + r_k)

        cum_glp_after  = cum_glp_before + glp_k
        cum_bias_after = cum_bias_before + bias_k

        # ---- Chain node at pos k ----
        # parent_chain: k=0 → -1 (orphan / root child); k>0 → k-1
        if k == 0:
            parent_chain = tl.cast(-1, tl.int64)
        else:
            parent_chain = tl.cast(k - 1, tl.int64)

        chain_real_lp = cum_glp_after
        chain_eff_lp  = cum_glp_after + cum_bias_after

        tl.store(tn_tokens_ptr + k, gt_k)
        tl.store(tn_parents_ptr + k, parent_chain)
        tl.store(tn_real_lps_ptr + k, chain_real_lp)
        tl.store(tn_lps_ptr + k, chain_eff_lp)
        tl.store(tn_ranks_ptr + k, r_k)
        tl.store(tn_blocks_ptr + k, tl.cast(0, tl.int64))
        tl.store(tn_slots_ptr + k, tl.cast(k, tl.int64))

        # ---- Chain-end pending entry at pend[k] ----
        tl.store(pend_hidden_slots_ptr + k, tl.cast(k, tl.int64))
        tl.store(pend_input_ids_ptr + k, gt_k)
        tl.store(pend_ttt_valid_ptr + k, tl.cast(k + 1, tl.int64))
        tl.store(pend_node_indices_ptr + k, tl.cast(k, tl.int64))
        tl.store(pend_cum_lps_ptr + k, chain_real_lp)
        tl.store(pend_cum_bias_ptr + k, cum_bias_after)

        # ---- Alts (j = 1..MAX_TOPK-1) ----
        # Alt at slot k branches off chain[k-1] (or root if k=0). It has its
        # own log_prob (top_lps[k, j]) and its own slot-k rank bias.
        # alt_real_lp  = real_cum_lp(parent) + log_prob(alt)
        # alt_cum_bias = cum_bias(parent)    + bias_k
        # The alt parent is the SAME tree node as the chain[k] parent, so
        # parent's cum_bias_before is what we add bias_k to.
        parent_alt = parent_chain
        for j in range(1, MAX_TOPK):
            alt_idx = k * (MAX_TOPK - 1) + (j - 1)
            tn_pos   = K + alt_idx
            pend_pos = K + alt_idx

            top_tok = tl.load(top_target_ptr + k * MAX_TOPK + j)
            top_lp  = tl.load(top_lps_ptr + k * MAX_TOPK + j)

            alt_real_lp  = cum_glp_before + top_lp
            alt_cum_bias = cum_bias_before + bias_k
            alt_eff_lp   = alt_real_lp + alt_cum_bias

            tl.store(tn_tokens_ptr + tn_pos, top_tok)
            tl.store(tn_parents_ptr + tn_pos, parent_alt)
            tl.store(tn_real_lps_ptr + tn_pos, alt_real_lp)
            tl.store(tn_lps_ptr + tn_pos, alt_eff_lp)
            tl.store(tn_ranks_ptr + tn_pos, r_k)
            tl.store(tn_blocks_ptr + tn_pos, tl.cast(0, tl.int64))
            tl.store(tn_slots_ptr + tn_pos, tl.cast(k, tl.int64))

            # Alt pending: same indexing as alt tn pos.
            tl.store(pend_hidden_slots_ptr + pend_pos, tl.cast(k, tl.int64))
            tl.store(pend_input_ids_ptr + pend_pos, top_tok)
            tl.store(pend_ttt_valid_ptr + pend_pos, tl.cast(k + 1, tl.int64))
            tl.store(pend_node_indices_ptr + pend_pos, tl.cast(tn_pos, tl.int64))
            tl.store(pend_cum_lps_ptr + pend_pos, alt_real_lp)
            tl.store(pend_cum_bias_ptr + pend_pos, alt_cum_bias)

        cum_glp_before  = cum_glp_after
        cum_bias_before = cum_bias_after


@triton.jit
def _block2_bfs_static_kernel(
    # ---- Inputs (per pending leaf) ----
    rank_preds_ptr,          # [N_PEND, K] i64
    top_target_ptr,          # [N_PEND, K, MAX_TOPK] i64
    top_lps_ptr,             # [N_PEND, K, MAX_TOPK] f32
    pend_node_indices_ptr,   # [N_PEND] i64
    pend_cum_lps_ptr,        # [N_PEND] f32 — real cum_lp from root to leaf
    pend_cum_bias_ptr,       # [N_PEND] f32 — cum_bias from root to leaf
    rank_offset_ptr,         # [RANK_CLASSES] f32
    # ---- Tree node outputs ----
    tn_tokens_ptr,           # [MAX_NODES] i64
    tn_parents_ptr,          # [MAX_NODES] i64
    tn_lps_ptr,              # [MAX_NODES] f32 (real + cum_bias)
    tn_real_lps_ptr,         # [MAX_NODES] f32 (real only)
    tn_ranks_ptr,            # [MAX_NODES] i64
    tn_blocks_ptr,           # [MAX_NODES] i64 (= PEND_DEPTH for block-2)
    tn_slots_ptr,            # [MAX_NODES] i64
    # ---- Compile-time constants ----
    N_PEND: tl.constexpr,
    K: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    TREE_START: tl.constexpr,  # = N_PEND (block-1 wrote to [0..N_PEND))
    PEND_DEPTH: tl.constexpr,
):
    """Block-2 BFS expansion — one program per pending leaf, K-serial unrolled.

    Layout per leaf l:
      * Greedy chain: tn[TREE_START + l*K + k] for k in [0, K)
      * Alts:        tn[TREE_START + N_PEND*K + l*K*(MAX_TOPK-1) + k*(MAX_TOPK-1) + (j-1)]

    parent of chain[0]/alt[0] = pend_node (block-1 leaf), else chain[k-1].
    """
    leaf = tl.program_id(0)
    if leaf >= N_PEND:
        return

    pend_node = tl.load(pend_node_indices_ptr + leaf)
    pend_lp   = tl.load(pend_cum_lps_ptr + leaf)
    pend_bias = tl.load(pend_cum_bias_ptr + leaf)

    chain_base = TREE_START + leaf * K
    alt_base = TREE_START + N_PEND * K + leaf * K * (MAX_TOPK - 1)

    cum_glp_before = tl.cast(0.0, tl.float32)
    cum_bias_before = tl.cast(0.0, tl.float32)

    for k in range(K):
        glp_k = tl.load(top_lps_ptr + (leaf * K + k) * MAX_TOPK + 0)
        gt_k  = tl.load(top_target_ptr + (leaf * K + k) * MAX_TOPK + 0)
        r_k   = tl.load(rank_preds_ptr + leaf * K + k)
        bias_k = tl.load(rank_offset_ptr + r_k)

        cum_glp_after  = cum_glp_before + glp_k
        cum_bias_after = cum_bias_before + bias_k

        chain_pos = chain_base + k
        # parent for chain[k]: k==0 → pend_node; k>0 → chain_pos - 1
        if k == 0:
            parent_chain = pend_node
        else:
            parent_chain = tl.cast(chain_pos - 1, tl.int64)

        chain_real_lp = pend_lp + cum_glp_after
        chain_eff_lp  = chain_real_lp + (pend_bias + cum_bias_after)

        tl.store(tn_tokens_ptr + chain_pos, gt_k)
        tl.store(tn_parents_ptr + chain_pos, parent_chain)
        tl.store(tn_real_lps_ptr + chain_pos, chain_real_lp)
        tl.store(tn_lps_ptr + chain_pos, chain_eff_lp)
        tl.store(tn_ranks_ptr + chain_pos, r_k)
        tl.store(tn_blocks_ptr + chain_pos, tl.cast(PEND_DEPTH, tl.int64))
        tl.store(tn_slots_ptr + chain_pos, tl.cast(k, tl.int64))

        # ---- Alts (j = 1..MAX_TOPK-1) ----
        parent_alt = parent_chain
        for j in range(1, MAX_TOPK):
            alt_pos = alt_base + k * (MAX_TOPK - 1) + (j - 1)
            top_tok = tl.load(top_target_ptr + (leaf * K + k) * MAX_TOPK + j)
            top_lp  = tl.load(top_lps_ptr + (leaf * K + k) * MAX_TOPK + j)

            alt_real_lp  = pend_lp + cum_glp_before + top_lp
            alt_eff_lp   = alt_real_lp + (pend_bias + cum_bias_before + bias_k)

            tl.store(tn_tokens_ptr + alt_pos, top_tok)
            tl.store(tn_parents_ptr + alt_pos, parent_alt)
            tl.store(tn_real_lps_ptr + alt_pos, alt_real_lp)
            tl.store(tn_lps_ptr + alt_pos, alt_eff_lp)
            tl.store(tn_ranks_ptr + alt_pos, r_k)
            tl.store(tn_blocks_ptr + alt_pos, tl.cast(PEND_DEPTH, tl.int64))
            tl.store(tn_slots_ptr + alt_pos, tl.cast(k, tl.int64))

        cum_glp_before  = cum_glp_after
        cum_bias_before = cum_bias_after


@triton.jit
def _depth_mask_static_kernel(
    parent_ptr,              # [Np1] i32
    valid_ptr,               # [Np1] i64
    depth_ptr,               # [Np1] i32 (output)
    mask_ptr,                # [Np1*Np1] f32 (output)
    Np1,
    MAX_DEPTH: tl.constexpr,
):
    """Compute depth + ancestor mask for one tree node, validity-gated.

    Convention: parent[0]=0 (root self-loop). Other parents in [0, Np1).
    Padded slots (valid=0) get mask[i, i]=1.0 and depth=0 (self-attend only).
    """
    pid = tl.program_id(0)
    if pid >= Np1:
        return

    is_valid = tl.load(valid_ptr + pid) != 0

    if is_valid:
        cur = pid
        depth = 0
        for _ in range(MAX_DEPTH + 1):
            tl.store(mask_ptr + pid * Np1 + cur, 1.0)
            parent = tl.load(parent_ptr + cur)
            is_root = cur == 0
            depth += tl.where(is_root, 0, 1)
            cur = tl.where(is_root, 0, parent)
        tl.store(depth_ptr + pid, depth)
    else:
        tl.store(mask_ptr + pid * Np1 + pid, 1.0)
        tl.store(depth_ptr + pid, 0)


@triton.jit
def _retrieve_static_kernel(
    parent_ptr,              # [Np1] i32
    depth_ptr,               # [Np1] i32
    leaf_or_neg_ptr,         # [Np1] i64 — pos i if leaf else -1
    ri_ptr,                  # [Np1, max_depth+1] i32 — pre-init -1
    max_depth,
    Np1,
    MAX_DEPTH: tl.constexpr,
):
    """Per row: if leaf_or_neg[lid] >= 0, walk parent chain root→leaf and
    write into ri[lid]. Non-leaves stay at pre-init -1.
    """
    lid = tl.program_id(0)
    if lid >= Np1:
        return

    leaf = tl.load(leaf_or_neg_ptr + lid)
    if leaf < 0:
        return

    leaf_depth = tl.load(depth_ptr + leaf)
    cur = leaf
    for d in range(MAX_DEPTH + 1):
        col = leaf_depth - d
        valid = (d <= leaf_depth) & (col >= 0)
        tl.store(
            ri_ptr + lid * (max_depth + 1) + tl.where(valid, col, 0),
            cur,
            mask=valid,
        )
        parent = tl.load(parent_ptr + cur)
        cur = tl.where(cur > 0, parent, cur)


# ============================================================
#                  StaticDraftBuilder class
# ============================================================

class StaticDraftBuilder:
    """End-to-end static-shape, sync-free draft tree builder.

    Configure via env (or constructor args):
      STATIC_MAX_TOPK         (default 6)   per-slot uniform branching factor
      STATIC_RANK_BIAS_LAMBDA (default 0.5) coefficient λ on rank offset

    Output 7-tuple bit-compatible with `_build_tree_from_block1_*`.
    """

    def __init__(
        self,
        draft_model,
        K: int,
        total_tokens: int,
        max_blocks: int,
        rank_classes: int = 4,
        max_topk: Optional[int] = None,
        rank_bias_lambda: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        assert max_blocks == 2, (
            f"StaticDraftBuilder currently supports max_blocks=2; got {max_blocks}"
        )
        self.draft_model = draft_model
        self.K = K
        self.total_tokens = total_tokens
        self.max_blocks = max_blocks
        self.rank_classes = rank_classes

        if max_topk is None:
            max_topk = int(os.environ.get("STATIC_MAX_TOPK", "6"))
        self.MAX_TOPK = max_topk

        if rank_bias_lambda is None:
            # Default 0: pure cum_lp prune (matches baseline dispatch).
            # Empirically λ=0.5 over-penalized rank>0 paths and dropped acc;
            # λ=0 yielded acc_len exactly matching dispatch on 4 benchmarks
            # (humaneval/alpaca/wmt23/mtbench) at n=80.
            rank_bias_lambda = float(os.environ.get("STATIC_RANK_BIAS_LAMBDA", "0.0"))
        self.rank_bias_lambda = rank_bias_lambda

        # ---- Compile-time tree shape constants ----
        self.N_BLOCK1 = K * self.MAX_TOPK
        self.N_PEND = K * self.MAX_TOPK
        self.N_BLOCK2 = self.N_PEND * K * self.MAX_TOPK
        self.MAX_NODES = self.N_BLOCK1 + self.N_BLOCK2

        if device is None:
            device = next(draft_model.parameters()).device
        self.device = device
        self.H = draft_model.config.hidden_size
        self.num_layers = draft_model.num_layers
        self.dtype = next(draft_model.parameters()).dtype

        # ---- Tree node buffers ----
        self.tn_tokens   = torch.zeros(self.MAX_NODES, dtype=torch.long, device=device)
        self.tn_parents  = torch.full((self.MAX_NODES,), -1, dtype=torch.long, device=device)
        self.tn_lps      = torch.full((self.MAX_NODES,), float("-inf"),
                                       dtype=torch.float32, device=device)
        self.tn_real_lps = torch.full((self.MAX_NODES,), float("-inf"),
                                       dtype=torch.float32, device=device)
        self.tn_ranks    = torch.zeros(self.MAX_NODES, dtype=torch.long, device=device)
        self.tn_blocks   = torch.zeros(self.MAX_NODES, dtype=torch.long, device=device)
        self.tn_slots    = torch.zeros(self.MAX_NODES, dtype=torch.long, device=device)

        # ---- Pending leaf buffer ----
        self.pend_hidden_slots = torch.zeros(self.N_PEND, dtype=torch.long, device=device)
        self.pend_input_ids    = torch.zeros(self.N_PEND, dtype=torch.long, device=device)
        self.pend_ttt_valid    = torch.zeros(self.N_PEND, dtype=torch.long, device=device)
        self.pend_node_indices = torch.zeros(self.N_PEND, dtype=torch.long, device=device)
        self.pend_cum_lps      = torch.full((self.N_PEND,), float("-inf"),
                                             dtype=torch.float32, device=device)
        self.pend_cum_bias     = torch.zeros(self.N_PEND, dtype=torch.float32, device=device)

        # ---- Rank-aware bias table: rank_offset[c] = -λ * c (non-positive). ----
        self.rank_offset_t = torch.tensor(
            [-self.rank_bias_lambda * c for c in range(self.rank_classes)],
            dtype=torch.float32, device=device,
        )

        # ---- Block-2 forward batched input buffers (static [N_PEND, ...]) ----
        self.b2_hidden    = torch.zeros(self.N_PEND, 1, self.H, dtype=self.dtype, device=device)
        self.b2_input_ids = torch.zeros(self.N_PEND, 1, dtype=torch.long, device=device)
        self.b2_ttt_mask  = torch.zeros(self.N_PEND, K, dtype=torch.bool, device=device)
        self._arange_K    = torch.arange(K, device=device, dtype=torch.long)

        # ---- Finalize compact buffers (static [Np1, ...]) ----
        Np1 = total_tokens + 1
        self.Np1 = Np1
        self.cf_tokens   = torch.zeros(Np1, dtype=torch.long, device=device)
        self.cf_parents  = torch.zeros(Np1, dtype=torch.int32, device=device)
        self.cf_real_lps = torch.full((Np1,), float("-inf"), dtype=torch.float32, device=device)
        self.cf_ranks    = torch.zeros(Np1, dtype=torch.long, device=device)
        self.cf_blocks   = torch.zeros(Np1, dtype=torch.long, device=device)
        self.cf_slots    = torch.zeros(Np1, dtype=torch.long, device=device)
        self.cf_valid    = torch.zeros(Np1, dtype=torch.long, device=device)
        self.cf_depth    = torch.zeros(Np1, dtype=torch.int32, device=device)
        self.cf_mask     = torch.zeros(Np1 * Np1, dtype=torch.float32, device=device)
        self.cf_ri       = torch.full((Np1, MAX_TREE_DEPTH + 1), -1,
                                       dtype=torch.int32, device=device)
        self.cf_is_parent = torch.zeros(Np1, dtype=torch.long, device=device)
        self.cf_leaf_or_neg = torch.full((Np1,), -1, dtype=torch.long, device=device)
        # Packed leaves: row i = i-th valid leaf's original compact index.
        # Used to feed the retrieve kernel so cf_ri rows are dense at the top
        # (matches the existing num_leaves-shaped retrieve_indices convention,
        # so downstream best_candidate=0 always lands on a real leaf row).
        self.cf_packed_leaves = torch.full((Np1,), -1, dtype=torch.long, device=device)
        self._packed_leaves_tmp = torch.full((Np1 + 1,), -1, dtype=torch.long, device=device)
        self._arange_Np1 = torch.arange(Np1, device=device, dtype=torch.long)

        # Sentinel-aware scatter buffers (Np1+1; last slot = trash).
        self._sc_tokens   = torch.zeros(Np1 + 1, dtype=torch.long, device=device)
        self._sc_parents  = torch.zeros(Np1 + 1, dtype=torch.long, device=device)
        self._sc_real_lps = torch.full((Np1 + 1,), float("-inf"),
                                        dtype=torch.float32, device=device)
        self._sc_ranks    = torch.zeros(Np1 + 1, dtype=torch.long, device=device)
        self._sc_blocks   = torch.zeros(Np1 + 1, dtype=torch.long, device=device)
        self._sc_slots    = torch.zeros(Np1 + 1, dtype=torch.long, device=device)

        # Optional CUDA-graph cache (block-2 forward).
        self._draft_graph_cache = None

        # Diagnostic stub for downstream stat consumers (don't break them).
        self._dummy_rank_stat = [{
            "rank_preds": [0] * K, "M": K, "branch": 0, "parent_block": -1,
        }]
        self._stub_node_ranks: List[int] = [0] * Np1
        self._stub_node_block_slots: List[Tuple[int, int]] = [(0, 0)] * Np1

    # ---------------------------------------------------------------
    #   Public hooks
    # ---------------------------------------------------------------
    def attach_graph_cache(self, graph_cache) -> None:
        self._draft_graph_cache = graph_cache

    # ---------------------------------------------------------------
    #   Per-iter scratch reset
    # ---------------------------------------------------------------
    def _reset_per_iter(self) -> None:
        self.tn_tokens.zero_()
        self.tn_parents.fill_(-1)
        self.tn_lps.fill_(float("-inf"))
        self.tn_real_lps.fill_(float("-inf"))
        self.tn_ranks.zero_()
        self.tn_blocks.zero_()
        self.tn_slots.zero_()

        self.pend_ttt_valid.zero_()
        self.pend_cum_lps.fill_(float("-inf"))
        self.pend_cum_bias.zero_()

        self.cf_tokens.zero_()
        self.cf_parents.zero_()
        self.cf_real_lps.fill_(float("-inf"))
        self.cf_ranks.zero_()
        self.cf_blocks.zero_()
        self.cf_slots.zero_()
        self.cf_valid.zero_()
        self.cf_depth.zero_()
        self.cf_mask.zero_()
        self.cf_ri.fill_(-1)
        self.cf_is_parent.zero_()
        self.cf_leaf_or_neg.fill_(-1)
        self.cf_packed_leaves.fill_(-1)
        self._packed_leaves_tmp.fill_(-1)

    # ---------------------------------------------------------------
    #   Phase 1: block-1 mega-kernel
    # ---------------------------------------------------------------
    def _phase_block1(
        self,
        b0_logits: torch.Tensor,        # [1, K, V_d]
        b0_rank_logits: torch.Tensor,   # [1, K, R]
        d2t: torch.Tensor,              # [V_d]
    ) -> None:
        last_p = b0_logits[0]
        log_probs = F.log_softmax(last_p.float(), dim=-1)
        rank_preds = b0_rank_logits[0].argmax(dim=-1)
        top_idx = torch.topk(last_p, self.MAX_TOPK, dim=-1).indices    # [K, MAX_TOPK]
        top_target = top_idx + d2t[top_idx]                             # [K, MAX_TOPK]
        top_lps = log_probs.gather(1, top_idx)                          # [K, MAX_TOPK]

        _block1_static_kernel[(1,)](
            rank_preds.contiguous(),
            top_target.contiguous(),
            top_lps.contiguous(),
            self.rank_offset_t,
            self.tn_tokens, self.tn_parents,
            self.tn_lps, self.tn_real_lps,
            self.tn_ranks, self.tn_blocks, self.tn_slots,
            self.pend_hidden_slots, self.pend_input_ids,
            self.pend_ttt_valid, self.pend_node_indices,
            self.pend_cum_lps, self.pend_cum_bias,
            K=self.K,
            MAX_TOPK=self.MAX_TOPK,
            num_warps=1,
        )

    # ---------------------------------------------------------------
    #   Phase 2: block-2 batch prep
    # ---------------------------------------------------------------
    def _phase_block2_batch_prep(
        self,
        b0_draft_hidden: torch.Tensor,  # [1, K, H]
    ) -> None:
        gathered = b0_draft_hidden[0].index_select(0, self.pend_hidden_slots)
        self.b2_hidden[:, 0, :].copy_(gathered)
        self.b2_input_ids[:, 0].copy_(self.pend_input_ids)
        torch.lt(
            self._arange_K.unsqueeze(0),
            self.pend_ttt_valid.unsqueeze(1),
            out=self.b2_ttt_mask,
        )

    # ---------------------------------------------------------------
    #   Phase 3: block-2 forward
    # ---------------------------------------------------------------
    def _phase_block2_forward(
        self,
        b0_ttt_kv: list,
        draft_cache: list,
        draft_position: int,
    ):
        N_PEND = self.N_PEND
        K = self.K
        cross_count = draft_cache[0][2]
        effective_cross_count = cross_count - K if cross_count >= K else 0

        # ---- Try graph replay first ----
        if self._draft_graph_cache is not None:
            try:
                cross_slices = []
                for layer_cache in draft_cache:
                    if effective_cross_count > 0 and layer_cache[0] is not None:
                        cross_slices.append((
                            layer_cache[0][:, :, :effective_cross_count, :],
                            layer_cache[1][:, :, :effective_cross_count, :],
                        ))
                    else:
                        cross_slices.append(None)
                return self._draft_graph_cache.run(
                    hidden=self.b2_hidden,
                    input_ids=self.b2_input_ids,
                    cross_cache_slices=cross_slices,
                    effective_cross_count=effective_cross_count,
                    ttt_cache=[b0_ttt_kv[l] for l in range(self.num_layers)],
                    ttt_mask=self.b2_ttt_mask,
                    position_id=draft_position,
                )
            except Exception:
                pass  # eager fallback

        # ---- Eager fallback ----
        if effective_cross_count > 0:
            cross_ones = self.b2_ttt_mask.new_ones(N_PEND, effective_cross_count)
            full_kv_mask = torch.cat([cross_ones, self.b2_ttt_mask], dim=1)
        else:
            full_kv_mask = self.b2_ttt_mask

        batch_cross_cache = []
        for layer_cache in draft_cache:
            if effective_cross_count > 0 and layer_cache[0] is not None:
                k_view = layer_cache[0][:, :, :effective_cross_count, :]
                v_view = layer_cache[1][:, :, :effective_cross_count, :]
                batch_cross_cache.append([
                    k_view.expand(N_PEND, -1, -1, -1),
                    v_view.expand(N_PEND, -1, -1, -1),
                    effective_cross_count,
                ])
            else:
                batch_cross_cache.append([None, None, 0])

        batch_ttt_kv = [
            (b0_ttt_kv[l][0].expand(N_PEND, -1, -1, -1),
             b0_ttt_kv[l][1].expand(N_PEND, -1, -1, -1))
            for l in range(self.num_layers)
        ]

        return self.draft_model.forward_with_cache(
            hidden=self.b2_hidden,
            input_ids=self.b2_input_ids,
            cache=batch_cross_cache,
            position_id=draft_position,
            use_draft_condition=True,
            ttt_cache=batch_ttt_kv,
            ttt_mask=self.b2_ttt_mask,
            update_cross_cache=False,
            full_kv_mask=full_kv_mask,
        )

    # ---------------------------------------------------------------
    #   Phase 4: block-2 BFS mega-kernel
    # ---------------------------------------------------------------
    def _phase_block2_bfs(
        self,
        block_logits: torch.Tensor,       # [N_PEND, K, V_d]
        block_rank_logits: torch.Tensor,  # [N_PEND, K, R]
        d2t: torch.Tensor,
    ) -> None:
        N_PEND = self.N_PEND
        K = self.K
        MAX_TOPK = self.MAX_TOPK

        rank_preds_b2 = block_rank_logits.argmax(dim=-1)

        V_d = block_logits.shape[-1]
        flat_logits = block_logits.view(N_PEND * K, V_d)
        top_vals, top_idx = torch.topk(flat_logits, MAX_TOPK, dim=-1)
        top_target_flat = top_idx + d2t[top_idx]
        lse = torch.logsumexp(block_logits.float(), dim=-1)
        top_lps_flat = top_vals.float() - lse.view(N_PEND * K, 1)
        top_target_b2 = top_target_flat.view(N_PEND, K, MAX_TOPK)
        top_lps_b2    = top_lps_flat.view(N_PEND, K, MAX_TOPK)

        _block2_bfs_static_kernel[(N_PEND,)](
            rank_preds_b2.contiguous(),
            top_target_b2.contiguous(),
            top_lps_b2.contiguous(),
            self.pend_node_indices,
            self.pend_cum_lps,
            self.pend_cum_bias,
            self.rank_offset_t,
            self.tn_tokens, self.tn_parents,
            self.tn_lps, self.tn_real_lps,
            self.tn_ranks, self.tn_blocks, self.tn_slots,
            N_PEND=N_PEND,
            K=K,
            MAX_TOPK=MAX_TOPK,
            TREE_START=self.N_BLOCK1,
            PEND_DEPTH=1,
            num_warps=1,
        )

    # ---------------------------------------------------------------
    #   Phase 5: finalize
    # ---------------------------------------------------------------
    def _phase_finalize(self, sample_token: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        Np1 = self.Np1
        budget = self.total_tokens
        device = self.device

        # Sub-event timing for finalize (env STATIC_DRAFT_PROFILE_FINALIZE=1).
        _fp = os.environ.get("STATIC_DRAFT_PROFILE_FINALIZE", "0") == "1"
        if _fp:
            from . import _specblock_inference_model_base as _sim
            _ev_dedup_s = torch.cuda.Event(enable_timing=True); _ev_dedup_s.record()

        # Debug: dump tree state to disk for B-vs-M comparison.
        # Set env DUMP_TREE_PATH=/path/to/dir to enable.
        _dump_path = os.environ.get("DUMP_TREE_PATH")
        if _dump_path is not None:
            if not hasattr(self, "_dump_iter"):
                self._dump_iter = 0
                os.makedirs(_dump_path, exist_ok=True)
            _it = self._dump_iter
            torch.save({
                "tokens": self.tn_tokens.cpu().clone(),
                "parents": self.tn_parents.cpu().clone(),
                "lps": self.tn_lps.cpu().clone(),
                "real_lps": self.tn_real_lps.cpu().clone(),
            }, os.path.join(_dump_path, f"iter{_it:04d}_pre_dedup.pt"))

        # ---- 0. Dedup-merge pass (depth-by-depth, fused Triton kernel) ----
        # TREE_DEDUP_MODE: 0 = off, 1 = depth-by-depth dedup (default).
        # Empirical: depth-by-depth gives +0.075 acc avg over no-dedup
        # across 4 benches (humaneval/alpaca/wmt23/mtbench at n=80).
        # Triton kernel: ~270us per call (vs 95ms PyTorch reference).
        # Set TREE_DEDUP_PYTORCH=1 to use the slow PyTorch reference instead
        # (e.g. for verification).
        _dedup_mode = int(os.environ.get("TREE_DEDUP_MODE", "1"))
        if _dedup_mode == 1:
            if os.environ.get("TREE_DEDUP_PYTORCH", "0") == "1":
                from .tree_finalize_triton import dedup_depth_by_depth as _dfn
            else:
                from .tree_finalize_triton import dedup_depth_by_depth_triton as _dfn
            _dfn(
                self.tn_tokens, self.tn_parents, self.tn_lps,
                self.MAX_NODES, real_lps=self.tn_real_lps,
            )

        if _dump_path is not None:
            torch.save({
                "tokens": self.tn_tokens.cpu().clone(),
                "parents": self.tn_parents.cpu().clone(),
                "lps": self.tn_lps.cpu().clone(),
                "real_lps": self.tn_real_lps.cpu().clone(),
            }, os.path.join(_dump_path, f"iter{_it:04d}_post_dedup.pt"))

        if _fp:
            _ev_prune_s = torch.cuda.Event(enable_timing=True); _ev_prune_s.record()
            _sim._PROFILE_EVENTS.append(("fz_dedup", _ev_dedup_s, _ev_prune_s))

        # ---- 1-6. Fused prune + scatter compact + parent reindex + root placement ----
        # Single Triton kernel replacing ~28 PyTorch ops. Opt out via env
        # FZ_PRUNE_FUSED=0 to fall back to the original PyTorch path
        # (kept around for debugging / reference).
        if os.environ.get("FZ_PRUNE_FUSED", "1") == "1":
            from .tree_finalize_triton import fused_prune
            sample_tok_1d = sample_token.view(1) if sample_token.dim() == 0 else sample_token
            fused_prune(
                self.tn_tokens, self.tn_parents, self.tn_lps, self.tn_real_lps,
                self.tn_ranks, self.tn_blocks, self.tn_slots,
                self.cf_tokens, self.cf_parents, self.cf_real_lps,
                self.cf_ranks, self.cf_blocks, self.cf_slots, self.cf_valid,
                sample_tok_1d,
                self.MAX_NODES, budget, Np1,
            )
        else:
            # Reference PyTorch path
            top_idx = torch.topk(self.tn_lps, budget, largest=True).indices
            keep_mask = torch.zeros(self.MAX_NODES, dtype=torch.long, device=device)
            keep_mask.scatter_(0, top_idx, 1)
            cumsum_excl = torch.cumsum(keep_mask, dim=0) - keep_mask
            sentinel_pos = budget + 1
            new_pos = torch.where(
                keep_mask == 1,
                cumsum_excl + 1,
                torch.full_like(cumsum_excl, sentinel_pos),
            )
            self._sc_tokens.zero_()
            self._sc_parents.zero_()
            self._sc_real_lps.fill_(float("-inf"))
            self._sc_ranks.zero_()
            self._sc_blocks.zero_()
            self._sc_slots.zero_()
            self._sc_tokens.scatter_(0, new_pos, self.tn_tokens)
            self._sc_real_lps.scatter_(0, new_pos, self.tn_real_lps)
            self._sc_ranks.scatter_(0, new_pos, self.tn_ranks)
            self._sc_blocks.scatter_(0, new_pos, self.tn_blocks)
            self._sc_slots.scatter_(0, new_pos, self.tn_slots)
            old_parents = self.tn_parents
            parents_clamped = old_parents.clamp(min=0)
            gathered_new_parent = new_pos.gather(0, parents_clamped)
            new_parents_for_old = torch.where(
                (old_parents >= 0) & (gathered_new_parent < sentinel_pos),
                gathered_new_parent,
                torch.zeros_like(old_parents),
            )
            self._sc_parents.scatter_(0, new_pos, new_parents_for_old)
            self.cf_tokens.copy_(self._sc_tokens[:Np1])
            self.cf_parents.copy_(self._sc_parents[:Np1].to(torch.int32))
            self.cf_real_lps.copy_(self._sc_real_lps[:Np1])
            self.cf_ranks.copy_(self._sc_ranks[:Np1])
            self.cf_blocks.copy_(self._sc_blocks[:Np1])
            self.cf_slots.copy_(self._sc_slots[:Np1])
            self.cf_valid.copy_((self.cf_real_lps > float("-inf")).long())
            self.cf_tokens[0] = sample_token.squeeze()
            self.cf_parents[0] = 0
            self.cf_valid[0] = 1
            self.cf_ranks[0] = 0
            self.cf_blocks[0] = 0
            self.cf_slots[0] = 0

        if _fp:
            _ev_topo1_s = torch.cuda.Event(enable_timing=True); _ev_topo1_s.record()
            _sim._PROFILE_EVENTS.append(("fz_prune", _ev_prune_s, _ev_topo1_s))

        # ---- 7. Topology kernel ----
        _depth_mask_static_kernel[(Np1,)](
            self.cf_parents, self.cf_valid,
            self.cf_depth, self.cf_mask,
            Np1, MAX_DEPTH=MAX_TREE_DEPTH,
        )

        if _fp:
            _ev_leafpack_s = torch.cuda.Event(enable_timing=True); _ev_leafpack_s.record()
            _sim._PROFILE_EVENTS.append(("fz_topo1", _ev_topo1_s, _ev_leafpack_s))

        # ---- 8 + 8b. Fused is_parent + leaf_or_neg + pack leaves ----
        # Single Triton kernel replacing ~17 PyTorch ops. Opt out via
        # FZ_LEAFPACK_FUSED=0 to fall back to the original PyTorch path.
        if os.environ.get("FZ_LEAFPACK_FUSED", "1") == "1":
            from .tree_finalize_triton import fused_leafpack
            fused_leafpack(
                self.cf_parents, self.cf_valid,
                self.cf_is_parent, self.cf_leaf_or_neg, self.cf_packed_leaves,
                Np1,
            )
        else:
            self.cf_is_parent.zero_()
            nonroot_parents = self.cf_parents[1:].to(torch.long)
            valid_children = self.cf_valid[1:]
            increment_src = torch.where(
                valid_children == 1,
                torch.ones_like(nonroot_parents),
                torch.zeros_like(nonroot_parents),
            )
            self.cf_is_parent.scatter_add_(0, nonroot_parents, increment_src)

            arange_Np1 = self._arange_Np1
            is_leaf = (self.cf_valid == 1) & (self.cf_is_parent == 0)
            self.cf_leaf_or_neg.copy_(
                torch.where(is_leaf, arange_Np1, torch.full_like(arange_Np1, -1)),
            )

            is_leaf_long = is_leaf.long()
            pos_in_leaf_set = torch.cumsum(is_leaf_long, dim=0) - is_leaf_long
            scatter_dst = torch.where(
                is_leaf,
                pos_in_leaf_set,
                torch.full_like(pos_in_leaf_set, Np1),
            )
            self._packed_leaves_tmp.fill_(-1)
            self._packed_leaves_tmp.scatter_(0, scatter_dst, arange_Np1)
            self.cf_packed_leaves.copy_(self._packed_leaves_tmp[:Np1])

        if _fp:
            _ev_topo2_s = torch.cuda.Event(enable_timing=True); _ev_topo2_s.record()
            _sim._PROFILE_EVENTS.append(("fz_leafpack", _ev_leafpack_s, _ev_topo2_s))

        # ---- 9. Retrieve kernel ----
        _retrieve_static_kernel[(Np1,)](
            self.cf_parents, self.cf_depth, self.cf_packed_leaves, self.cf_ri,
            MAX_TREE_DEPTH, Np1, MAX_DEPTH=MAX_TREE_DEPTH,
        )

        if _fp:
            _ev_topo2_e = torch.cuda.Event(enable_timing=True); _ev_topo2_e.record()
            _sim._PROFILE_EVENTS.append(("fz_topo2", _ev_topo2_s, _ev_topo2_e))

        # ---- 10. Outputs ----
        draft_tokens = self.cf_tokens.unsqueeze(0)
        tree_mask = self.cf_mask.view(Np1, Np1).unsqueeze(0).unsqueeze(0)
        tree_position_ids = self.cf_depth.long()
        retrieve_indices = self.cf_ri

        if _dump_path is not None:
            torch.save({
                "cf_tokens": self.cf_tokens.cpu().clone(),
                "cf_parents": self.cf_parents.cpu().clone(),
                "cf_valid": self.cf_valid.cpu().clone(),
                "cf_depth": self.cf_depth.cpu().clone(),
                "cf_real_lps": self.cf_real_lps.cpu().clone(),
                "cf_ri": self.cf_ri.cpu().clone(),
                "cf_is_parent": self.cf_is_parent.cpu().clone(),
                "n_leaves": int((self.cf_leaf_or_neg >= 0).sum().item()),
                "n_valid": int(self.cf_valid.sum().item()),
            }, os.path.join(_dump_path, f"iter{_it:04d}_compact.pt"))
            self._dump_iter += 1

        return draft_tokens, tree_mask, tree_position_ids, retrieve_indices

    # ---------------------------------------------------------------
    #   Public entry
    # ---------------------------------------------------------------
    @torch.no_grad()
    def build_tree(
        self,
        b0_logits: torch.Tensor,
        b0_rank_logits: torch.Tensor,
        b0_draft_hidden: torch.Tensor,
        b0_ttt_kv: list,
        input_id: torch.Tensor,
        draft_cache: list,
        draft_position: int,
        temperature: float = 0.0,
        collect_stats: bool = False,
    ) -> Tuple:
        self._reset_per_iter()

        # Deep profile: drop any pre-build_tree events (block-0 forward etc.)
        _deep_profile = os.environ.get("STATIC_DRAFT_PROFILE_DEEP", "0") == "1"
        if _deep_profile:
            from . import _specblock_inference_model_base as _sim
            _sim._PROFILE_EVENTS.clear()

        d2t = getattr(self.draft_model, "d2t", None)
        if d2t is None:
            d2t = torch.zeros(
                b0_logits.shape[-1], dtype=torch.long, device=self.device,
            )

        # Per-phase CUDA event timing (env STATIC_DRAFT_PROFILE=1).
        # Adds ~6 events per call (~6us); aggregated into self._phase_times_us.
        _profile = getattr(self, "_phase_profile_enabled", None)
        if _profile is None:
            self._phase_profile_enabled = (
                os.environ.get("STATIC_DRAFT_PROFILE", "0") == "1"
            )
            _profile = self._phase_profile_enabled
        if _profile:
            if not hasattr(self, "_phase_times_us"):
                self._phase_times_us = {
                    "block1": 0.0, "b2_prep": 0.0, "b2_forward": 0.0,
                    "b2_bfs": 0.0, "finalize": 0.0, "total": 0.0,
                }
                self._phase_count = 0
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            ev[0].record()
            self._phase_block1(b0_logits, b0_rank_logits, d2t)
            ev[1].record()
            self._phase_block2_batch_prep(b0_draft_hidden)
            ev[2].record()
            block_logits, block_rank_logits, _block_draft_hidden, _new_ttt_kv = \
                self._phase_block2_forward(b0_ttt_kv, draft_cache, draft_position)
            ev[3].record()
            self._phase_block2_bfs(block_logits, block_rank_logits, d2t)
            ev[4].record()
            sample_token = input_id[:, -1]
            draft_tokens, tree_mask, tree_position_ids, retrieve_indices = \
                self._phase_finalize(sample_token)
            ev[5].record()
            torch.cuda.synchronize()
            self._phase_times_us["block1"]    += ev[0].elapsed_time(ev[1]) * 1000
            self._phase_times_us["b2_prep"]   += ev[1].elapsed_time(ev[2]) * 1000
            self._phase_times_us["b2_forward"]+= ev[2].elapsed_time(ev[3]) * 1000
            self._phase_times_us["b2_bfs"]    += ev[3].elapsed_time(ev[4]) * 1000
            self._phase_times_us["finalize"]  += ev[4].elapsed_time(ev[5]) * 1000
            self._phase_times_us["total"]     += ev[0].elapsed_time(ev[5]) * 1000
            self._phase_count += 1
            if self._phase_count % 200 == 0:
                n = self._phase_count
                t = self._phase_times_us
                print(
                    f"  [STATIC_DRAFT_PROFILE n={n}] "
                    f"block1={t['block1']/n:.0f}us "
                    f"b2_prep={t['b2_prep']/n:.0f}us "
                    f"b2_fwd={t['b2_forward']/n:.0f}us "
                    f"b2_bfs={t['b2_bfs']/n:.0f}us "
                    f"finalize={t['finalize']/n:.0f}us "
                    f"total={t['total']/n:.0f}us",
                    flush=True,
                )

        if _deep_profile:
            from . import _specblock_inference_model_base as _sim
            if not hasattr(self, "_deep_times_us"):
                self._deep_times_us = {}
                self._deep_count = 0
                self._deep_skipped = 0
            for label, s, e in _sim._PROFILE_EVENTS:
                try:
                    dt = s.elapsed_time(e) * 1000
                    self._deep_times_us[label] = self._deep_times_us.get(label, 0.0) + dt
                except (RuntimeError, ValueError):
                    self._deep_skipped += 1
            _sim._PROFILE_EVENTS.clear()
            self._deep_count += 1
            if self._deep_count % 200 == 0:
                n = self._deep_count
                t = self._deep_times_us
                layer_keys = ["norm1", "qkv_proj", "rope_repeat", "kv_cat",
                              "attn_compute", "o_proj", "norm2", "mlp"]
                fz_keys = ["fz_dedup", "fz_prune", "fz_topo1",
                           "fz_leafpack", "fz_topo2"]
                msg = " ".join(f"{k}={t[k]/n:.0f}us" for k in layer_keys if k in t)
                tot = sum(t[k] for k in layer_keys if k in t) / n
                print(
                    f"  [DEEP_PROFILE n={n}] {msg} | sum2L={tot:.0f}us",
                    flush=True,
                )
                fz_msg = " ".join(f"{k}={t[k]/n:.0f}us" for k in fz_keys if k in t)
                if fz_msg:
                    fz_tot = sum(t[k] for k in fz_keys if k in t) / n
                    print(
                        f"  [FZ_PROFILE   n={n}] {fz_msg} | sumFZ={fz_tot:.0f}us",
                        flush=True,
                    )
                fwd_keys = ["fwd_input_layer", "fwd_layers_shift",
                            "fwd_final_norm", "fwd_lm_head", "fwd_rank_head"]
                fwd_msg = " ".join(f"{k}={t[k]/n:.0f}us" for k in fwd_keys if k in t)
                if fwd_msg:
                    fwd_tot = sum(t[k] for k in fwd_keys if k in t) / n
                    print(
                        f"  [FWD_PROFILE  n={n}] {fwd_msg} | sumFWD={fwd_tot:.0f}us",
                        flush=True,
                    )
                mlp_keys = ["mlp_gate_up", "mlp_silu_mul", "mlp_down_proj"]
                mlp_msg = " ".join(f"{k}={t[k]/n:.0f}us" for k in mlp_keys if k in t)
                if mlp_msg:
                    print(
                        f"  [MLP_PROFILE  n={n}] {mlp_msg} (per-call summed across 2 layers)",
                        flush=True,
                    )
                rh_keys = ["rh_topk_lse", "rh_features", "rh_matmul"]
                rh_msg = " ".join(f"{k}={t[k]/n:.0f}us" for k in rh_keys if k in t)
                if rh_msg:
                    rh_tot = sum(t[k] for k in rh_keys if k in t) / n
                    print(
                        f"  [RH_PROFILE   n={n}] {rh_msg} | sumRH={rh_tot:.0f}us",
                        flush=True,
                    )
        else:
            self._phase_block1(b0_logits, b0_rank_logits, d2t)
            self._phase_block2_batch_prep(b0_draft_hidden)
            block_logits, block_rank_logits, _block_draft_hidden, _new_ttt_kv = \
                self._phase_block2_forward(b0_ttt_kv, draft_cache, draft_position)
            self._phase_block2_bfs(block_logits, block_rank_logits, d2t)
            sample_token = input_id[:, -1]
            draft_tokens, tree_mask, tree_position_ids, retrieve_indices = \
                self._phase_finalize(sample_token)

        # Stats — return stubs (downstream stat consumers tolerate dummies).
        all_rank_stats: List[dict] = list(self._dummy_rank_stat)
        if collect_stats:
            stacked = torch.stack(
                [self.cf_ranks, self.cf_blocks, self.cf_slots]
            ).cpu()
            ranks_l = stacked[0].tolist()
            blocks_l = stacked[1].tolist()
            slots_l = stacked[2].tolist()
            node_ranks: List[int] = ranks_l
            node_block_slots: List[Tuple[int, int]] = list(zip(blocks_l, slots_l))
        else:
            node_ranks = self._stub_node_ranks
            node_block_slots = self._stub_node_block_slots

        return (
            draft_tokens, tree_mask, tree_position_ids, retrieve_indices,
            all_rank_stats, node_ranks, node_block_slots,
        )
