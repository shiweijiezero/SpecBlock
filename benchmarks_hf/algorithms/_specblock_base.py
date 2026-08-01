"""SpecBlock: Multi-block TTT with Rank-guided Tree Speculative Decoding.

推理流程：
1. Prefill: target forward → KV cache + hidden_3h + first_token; draft prefill
2. Draft tree construction (BFS expansion with batched forward):
   - 每个 block forward 产出 K 个 slot 的 logits，全部 K 个 greedy token 进 tree（零额外开销）
   - Block 1 (batch=1): forward(target_hidden 3H), rank 决定截断点 M 和分支因子
     - rank=0: correct (top-1), continue greedy chain
     - rank=1: small branch (factor=4), g(M-1) 与 top-0 合并, top-1..top-(bf-1) 为替代分支
     - rank=2: large branch (factor=10), 同上
     - rank=3: give up (rank>10), 全 K greedy 保留碰运气, 无 pending (不续 block)
     - g(K-1) 搭便车: 不触发下一个 block, 但若有其他 pending 则加入 batch (K-slot ttt_mask)
   - BFS loop (up to max_blocks):
     - 收集所有 pending leaves → 单次 batched forward, 每个 leaf 一视同仁
     - 每个 leaf 的结果独立处理: 全 K greedy + 合并分支 + hitchhike candidate → 新 pending
     - 不同 pending 的 ttt_mask 可能不同 (M-slot vs K-slot)
   - Post-expansion pruning: 若 tree 超过 total_tokens budget, 按 cum_log_prob 裁剪
     保留最高概率路径, 同时保留所有祖先节点
3. Verify: target model single forward over tree (Eagle3 style tree attention)
4. Accept: 找最长接受路径, 裁剪 KV cache
5. 更新 draft cross-position cache for accepted positions

Key semantic: ttt_cache 物理上存所有 K slots KV，通过 variable ttt_mask 控制有效范围。
cross_cache (draft_cache): all-K-slot KV from all prior decode positions, shared across all branches.
"""

import os
import json
import time
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Iterator, Any

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache

from .base import BaseAlgorithm


# NVTX instrumentation (env-gated, zero cost when off)
_NVTX_ENABLED = os.environ.get('NVTX_PROF', '0') == '1'
if _NVTX_ENABLED:
    try:
        import torch.cuda.nvtx as _nvtx
        def _nv_push(name): _nvtx.range_push(name)
        def _nv_pop(): _nvtx.range_pop()
    except Exception:
        def _nv_push(name): pass
        def _nv_pop(): pass
else:
    def _nv_push(name): pass
    def _nv_pop(): pass


# ============================================================
#    Compiled GPU ops for BFS tree construction
# ============================================================

@torch.compile(dynamic=True)
def _walk_rank_slots_compiled(rank_preds, rank_classes, rank_to_factor):
    """Compiled rank walking: find M, branch_factor, give_up for each leaf."""
    N, K = rank_preds.shape
    give_up_class = rank_classes - 1

    not_class0 = (rank_preds != 0)
    has_non_class0 = not_class0.any(dim=1)
    first_non_class0 = not_class0.float().argmax(dim=1)

    first_rank = rank_preds.gather(1, first_non_class0.unsqueeze(1).long()).squeeze(1)
    first_is_give_up = (first_rank == give_up_class)

    M = torch.where(~has_non_class0, K,
         torch.where(first_is_give_up, first_non_class0, first_non_class0 + 1))

    branch_factors = torch.where(
        has_non_class0 & ~first_is_give_up,
        rank_to_factor[first_rank.clamp(0, rank_classes - 1)],
        torch.zeros(N, device=rank_preds.device, dtype=torch.long))

    give_up = has_non_class0 & first_is_give_up
    return M, branch_factors, give_up


@torch.compile(dynamic=True)
def _bfs_gpu_ops(block_logits, block_rank_logits, d2t_offsets, max_factor):
    """Fused BFS GPU operations: argmax + rank + logsumexp + topk in one compiled graph.

    Args:
        block_logits: [N, K, V_draft] draft logits
        block_rank_logits: [N, K, rank_classes] rank predictions
        d2t_offsets: [V_draft] draft-to-target offset mapping
        max_factor: int, topk branching factor

    Returns:
        all_rank_preds: [N, K]
        all_greedy_tokens: [N, K] draft vocab ids
        all_greedy_target: [N, K] target vocab ids
        all_greedy_lps: [N, K] log probs of greedy tokens
        lse: [N, K] logsumexp for later use
    """
    # Rank predictions
    all_rank_preds = block_rank_logits.argmax(dim=-1)  # [N, K]

    # Greedy tokens + target mapping
    all_greedy_tokens = block_logits.argmax(dim=-1)  # [N, K]
    all_greedy_target = all_greedy_tokens + d2t_offsets[all_greedy_tokens]  # [N, K]

    # Log probs (fused logsumexp + gather)
    lse = torch.logsumexp(block_logits.float(), dim=-1)  # [N, K]
    all_greedy_lps = (
        block_logits.gather(2, all_greedy_tokens.unsqueeze(-1)).squeeze(-1).float() - lse
    )

    return all_rank_preds, all_greedy_tokens, all_greedy_target, all_greedy_lps, lse



@torch.compile(dynamic=True)
def _bfs_branch_ops_all_slots(block_logits, lse, d2t_offsets, max_factor):
    """Topk over ALL [N, K] slots simultaneously for multi-slot branching.

    Args:
        block_logits: [N, K, V_draft]
        lse: [N, K] pre-computed logsumexp
        d2t_offsets: [V_draft]
        max_factor: int

    Returns:
        top_target_all: [N, K, max_factor]
        top_lps_all: [N, K, max_factor]
    """
    N, K, V = block_logits.shape
    flat_logits = block_logits.view(N * K, V)
    top = torch.topk(flat_logits, max_factor, dim=-1)
    top_target = top.indices + d2t_offsets[top.indices]
    flat_lse = lse.view(N * K, 1)
    top_lps = flat_logits.gather(1, top.indices).float() - flat_lse
    return top_target.view(N, K, max_factor), top_lps.view(N, K, max_factor)


@torch.compile(dynamic=True)
def _bfs_gpu_ops_fused(block_logits, block_rank_logits, d2t_offsets, max_factor,
                       rank_classes, rank_to_factor):
    """Super-fused BFS GPU ops: rank walk + argmax + topk + log probs all in one compile.

    Combines _bfs_gpu_ops + _bfs_branch_ops_all_slots + _walk_rank_slots into
    one compiled graph to cut launch/dispatch cost in the BFS hot loop.

    block_logits stays in its original dtype (bf16) for argmax/topk; logsumexp
    is computed in float32 only over the top-k range (small) rather than the
    whole vocabulary, avoiding a 5MB float32 materialization.

    Returns:
        all_rank_preds:    [N, K]
        all_greedy_tokens: [N, K]  (draft ids)
        all_greedy_target: [N, K]  (target ids)
        all_greedy_lps:    [N, K]  (float32)
        M:                 [N]     (long)
        bf:                [N]     (long)
        give_up:           [N]     (bool)
        top_target_all:    [N, K, max_factor]
        top_lps_all:       [N, K, max_factor] (float32)
    """
    # === Rank predictions + rank walk ===
    all_rank_preds = block_rank_logits.argmax(dim=-1)  # [N, K]
    N, K = all_rank_preds.shape
    give_up_class = rank_classes - 1

    not_class0 = (all_rank_preds != 0)
    has_non_class0 = not_class0.any(dim=1)
    first_non_class0 = not_class0.float().argmax(dim=1)
    first_rank = all_rank_preds.gather(1, first_non_class0.unsqueeze(1).long()).squeeze(1)
    first_is_give_up = (first_rank == give_up_class)
    M = torch.where(~has_non_class0, K,
         torch.where(first_is_give_up, first_non_class0, first_non_class0 + 1))
    bf = torch.where(
        has_non_class0 & ~first_is_give_up,
        rank_to_factor[first_rank.clamp(0, rank_classes - 1)],
        torch.zeros(N, device=all_rank_preds.device, dtype=torch.long))
    give_up = has_non_class0 & first_is_give_up

    # === Fused topk over [N*K, V] — argmax is topk[:, 0] so we skip a separate argmax.
    V = block_logits.shape[-1]
    flat_logits = block_logits.view(N * K, V)
    top_vals, top_idx = torch.topk(flat_logits, max_factor, dim=-1)  # [N*K, max_factor]
    top_idx_nk = top_idx.view(N, K, max_factor)
    top_vals_nk = top_vals.view(N, K, max_factor)

    # Greedy tokens = topk[..., 0] (max_factor >= 1 guaranteed in callers).
    all_greedy_tokens = top_idx_nk[..., 0]
    all_greedy_target = all_greedy_tokens + d2t_offsets[all_greedy_tokens]
    greedy_vals = top_vals_nk[..., 0]

    # === Log probs: lse over bf16 logits, greedy lp = greedy_val - lse.
    lse = torch.logsumexp(block_logits.float(), dim=-1)
    all_greedy_lps = greedy_vals.float() - lse

    # === Topk log probs + target mapping ===
    top_target_flat = top_idx + d2t_offsets[top_idx]
    flat_lse = lse.view(N * K, 1)
    top_lps_flat = top_vals.float() - flat_lse

    return (
        all_rank_preds, all_greedy_tokens, all_greedy_target, all_greedy_lps,
        M, bf, give_up,
        top_target_flat.view(N, K, max_factor), top_lps_flat.view(N, K, max_factor),
    )


# ============================================================
#    Triton kernels for tree buffer construction
# ============================================================

MAX_TREE_DEPTH = 20  # K * max_blocks upper bound

@triton.jit
def _tree_depth_mask_kernel(
    parent_ptr, depth_ptr, mask_ptr,
    Np1, MAX_DEPTH: tl.constexpr,
):
    """Compute depth and ancestor mask for one tree node.

    Each program instance handles one node: walks parent chain to root,
    recording depth and setting mask[node, ancestor] = 1.0 for each ancestor.
    """
    pid = tl.program_id(0)

    cur = pid
    depth = 0
    for _ in range(MAX_DEPTH + 1):
        # Mark current node as ancestor in mask
        tl.store(mask_ptr + pid * Np1 + cur, 1.0)
        # Walk to parent (root has parent=0, stays at 0)
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

    Walks from leaf to root, writes at reversed column index
    so the result is root→leaf order.
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


def _vectorized_greedy_chain(n_nodes, N, K, pend_node_indices, pend_cum_lps,
                              gt_all_np, glp_all_np, rp_all_np, pend_depth,
                              tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots,
                              M_np=None, arange_K_np=None, arange_NK_np=None):
    """Add greedy chain nodes to tree buffers.

    Each leaf i produces min(K, M_i) greedy tokens in a chain.
    If M_np is None, all leaves produce K tokens (original behavior).
    arange_K_np and arange_NK_np are optional caller-supplied buffers; when
    present they avoid per-call np.arange allocation.
    Returns (updated n_nodes, greedy_node_map [N, K] with -1 for truncated slots).
    """
    NK = N * K
    ts = n_nodes  # tree_start

    # Reuse caller-provided arange buffers when possible.
    _ar_K = arange_K_np if (arange_K_np is not None and len(arange_K_np) >= K) else np.arange(K, dtype=np.int64)
    _ar_NK = arange_NK_np if (arange_NK_np is not None and len(arange_NK_np) >= NK) else np.arange(NK, dtype=np.int64)
    _ar_K = _ar_K[:K]
    _ar_NK = _ar_NK[:NK]

    # First: write all N*K nodes (vectorized, fast)
    tn_tokens[ts:ts+NK] = gt_all_np.reshape(-1)
    tn_ranks[ts:ts+NK] = rp_all_np.reshape(-1)
    tn_blocks[ts:ts+NK] = pend_depth
    tn_slots[ts:ts+NK] = np.tile(_ar_K, N)

    leaf_idx = _ar_NK // K
    slot_idx = _ar_NK - leaf_idx * K  # cheaper than modulo
    # Accept numpy array or list for pend_node_indices / pend_cum_lps.
    if isinstance(pend_node_indices, np.ndarray):
        pend_parents_np = pend_node_indices
    else:
        pend_parents_np = np.asarray(pend_node_indices, dtype=np.int64)
    tn_parents[ts:ts+NK] = np.where(slot_idx == 0, pend_parents_np[leaf_idx], ts + _ar_NK - 1)

    if isinstance(pend_cum_lps, np.ndarray):
        pend_lps_np = pend_cum_lps
    else:
        pend_lps_np = np.asarray(pend_cum_lps, dtype=np.float64)
    cum_glp = np.cumsum(glp_all_np.reshape(N, K), axis=1).reshape(-1)
    tn_lps[ts:ts+NK] = cum_glp + pend_lps_np[leaf_idx]

    if M_np is None or (M_np >= K).all():
        # No truncation needed
        greedy_map = np.arange(ts, ts + NK).reshape(N, K)
        return ts + NK, greedy_map

    # M-truncation: compact out slots >= M_i
    # Build validity mask
    valid = slot_idx < np.repeat(M_np, K)  # [NK] bool
    valid_indices = np.nonzero(valid)[0]
    n_valid = len(valid_indices)

    if n_valid == NK:
        greedy_map = np.arange(ts, ts + NK).reshape(N, K)
        return ts + NK, greedy_map

    # Compact: move valid nodes to front
    # Build old→new index mapping
    greedy_map = np.full((N, K), -1, dtype=np.int64)
    new_pos = ts
    for vi in valid_indices:
        leaf_i = vi // K
        slot_k = vi % K
        old_pos = ts + vi
        if new_pos != old_pos:
            tn_tokens[new_pos] = tn_tokens[old_pos]
            tn_ranks[new_pos] = tn_ranks[old_pos]
            tn_blocks[new_pos] = tn_blocks[old_pos]
            tn_slots[new_pos] = tn_slots[old_pos]
            tn_lps[new_pos] = tn_lps[old_pos]
        # Fix parent
        if slot_k == 0:
            tn_parents[new_pos] = pend_parents_np[leaf_i]
        else:
            tn_parents[new_pos] = new_pos - 1  # previous valid node in same chain
        greedy_map[leaf_i, slot_k] = new_pos
        new_pos += 1

    return new_pos, greedy_map


def build_tree_buffers_triton(n_nodes, tn_tokens, tn_parents, sample_token, device, K, max_blocks):
    """Build tree_tokens, tree_mask, position_ids, retrieve_indices using Triton kernels.

    Args:
        n_nodes: number of tree nodes (int)
        tn_tokens: numpy array [max_nodes] of token ids
        tn_parents: numpy array [max_nodes] of parent indices (-1 for root children)
        sample_token: root token tensor
        device: torch device
    """
    N = n_nodes
    Np1 = N + 1

    # Build parent tensor: +1 offset (root is index 0, nodes are 1..N)
    parent_np = np.empty(Np1, dtype=np.int32)
    parent_np[0] = 0  # root's parent is self
    parent_np[1:N+1] = np.where(tn_parents[:N] >= 0, tn_parents[:N] + 1, 0)
    parent_t = torch.from_numpy(parent_np).to(device)

    # Allocate output buffers
    depth_t = torch.zeros(Np1, dtype=torch.int32, device=device)
    mask_flat = torch.zeros(Np1 * Np1, dtype=torch.float32, device=device)

    # Kernel 1: depth + mask (one launch, Np1 threads)
    _tree_depth_mask_kernel[(Np1,)](
        parent_t, depth_t, mask_flat,
        Np1, MAX_DEPTH=MAX_TREE_DEPTH,
    )

    # Find leaves via scatter (GPU, no sync)
    is_parent = torch.zeros(Np1, dtype=torch.bool, device=device)
    is_parent.scatter_(0, parent_t[1:].long(), True)
    leaves = (~is_parent).nonzero(as_tuple=True)[0].to(torch.int32)
    num_leaves = leaves.shape[0]

    # Kernel 2: retrieve_indices (one launch, num_leaves threads)
    max_depth = depth_t.max().item()  # 1 CUDA sync
    ri = torch.full((num_leaves, max_depth + 1), -1, dtype=torch.int32, device=device)
    if num_leaves > 0:
        _tree_retrieve_kernel[(num_leaves,)](
            parent_t, depth_t, leaves, ri,
            max_depth, Np1, num_leaves,
            MAX_DEPTH=MAX_TREE_DEPTH,
        )

    # Sort retrieve_indices (CPU, ~60 rows — instant)
    ri_cpu = ri.tolist()
    maxitem = Np1 + 5
    ri_cpu.sort(key=lambda row: tuple(x if x >= 0 else maxitem for x in row))

    # Build final tensors
    token_ids = np.empty(Np1, dtype=np.int64)
    token_ids[0] = sample_token.item()
    token_ids[1:] = tn_tokens[:N]
    tree_tokens = torch.from_numpy(token_ids).to(device)[None]
    tree_mask = mask_flat.reshape(Np1, Np1)[None, None]
    tree_position_ids = depth_t.long()
    retrieve_indices = torch.tensor(ri_cpu, dtype=torch.long, device=device)

    return tree_tokens, tree_mask, tree_position_ids, retrieve_indices


# ============================================================
#              Utility functions
# ============================================================

def get_cache_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, DynamicCache):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].shape[2]


def crop_cache(past_key_values, target_length: int):
    if isinstance(past_key_values, DynamicCache):
        past_key_values.crop(target_length)
        return past_key_values
    return tuple(
        (k[..., :target_length, :], v[..., :target_length, :])
        for k, v in past_key_values
    )


def _clone_static_tree_result(result):
    """Detach StaticDraftBuilder outputs from its persistent reusable buffers."""
    return (
        result[0].clone(),
        result[1].clone(),
        result[2].clone(),
        result[3].clone(),
        list(result[4]),
        list(result[5]),
        list(result[6]),
    )


# ============================================================
#              SpecBlock Algorithm
# ============================================================

class _SpecBlockAlgorithmBase(BaseAlgorithm):
    """SpecBlock: Multi-block TTT with rank-guided tree speculative decoding.

    Config format: "batch_size,max_blocks,total_tokens"
    - max_blocks: maximum TTT blocks (0 = use config.json default)
    - total_tokens: tree budget after pruning (e.g., 60)
    """

    supports_true_batch = False

    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        device: str = "cuda",
        draft_tokens: int = 60,     # Total tree budget (after pruning)
        total_tokens: int = None,   # Alias for draft_tokens (from run_eval config)
        max_blocks: int = None,     # Maximum TTT blocks (None = use config default)
        beam_width: int = 10,       # Max pending leaves per BFS depth (beam search pruning)
        tree_K: int = None,         # Effective K for block 1 tree building (None = use model K)
        tree_K_bfs: int = None,     # Effective K for block 2+ BFS (None = use tree_K)
        diverse_beam: bool = False, # Diversity-aware beam selection
        slot_topk_mode: str = None, # Override RANK_SLOT_TOPK: 'no_giveup', 'generous', 'uniform', 'flat10'
        strict_linear: bool = False, # Explicit no-branching ablation; preserve one full greedy chain
        draft_quantize: str = None, # None, "int8", or "int4"
        **kwargs
    ):
        super().__init__(model_path, draft_model_path, device, **kwargs)
        # Only the two concrete base algorithms own this batch core.  Adaptation
        # subclasses override scalar hooks and must not silently inherit batching.
        self.supports_true_batch = (
            type(self).__name__ == "SpecBlockAlgorithm"
            and type(self).__module__.rsplit(".", 1)[-1] == "specblock"
        )
        self.draft_quantize = draft_quantize
        self.total_tokens = total_tokens if total_tokens is not None else draft_tokens
        self.beam_width = beam_width
        self._strict_linear = bool(strict_linear)
        # Static-shape draft path attrs (filled by load_model when SPECBLOCK_STATIC=1).
        # Always initialize here so the dispatch check in
        # _build_tree_from_block1_dispatch / _build_draft_tree never AttributeError's,
        # regardless of which subclass's load_model() ran.
        self._static_draft = None
        self._use_static_draft = False
        self._tree_K = tree_K       # Applied after model K is loaded
        self._tree_K_bfs = tree_K_bfs  # For BFS iterations (None = same as _tree_K)
        self._diverse_beam = diverse_beam  # Diversity-aware beam selection
        self._coverage_prune = bool(int(os.environ.get('COVERAGE_PRUNE', '0')))  # Coverage-aware post-pruning
        self._adaptive_budget = bool(int(os.environ.get('ADAPTIVE_BUDGET', '0')))  # Adaptive per-slot budget allocation
        # Protect block-1 slot-0 alternatives from beam pruning: guarantee all depth-1
        # leaves get block-2 forward (otherwise they're standalone depth-1 nodes with
        # no continuation, wasting tree budget when they match target).
        self._protect_d1 = bool(int(os.environ.get('PROTECT_D1', '0')))
        # Depth-stratified beam: each depth gets >= N leaves, rest globally.
        # Reserves guaranteed continuation for every depth layer.
        self._strat_beam = int(os.environ.get('STRAT_BEAM', '0'))
        # Override block-1 slot-0 beam count (alternatives). 0 = use self.beam_width.
        # Smaller value saves block-1 budget for deeper BFS; must be <= beam_width.
        self._slot0_beam = int(os.environ.get('SLOT0_BEAM', '0'))
        # Adaptive slot 0 beam based on greedy prob. When enabled, high-confidence
        # slot 0 uses fewer alts (greedy is reliable), low-confidence uses full beam_width.
        # Data: p0>=0.8 greedy matches 95% (need few alts); p0<0.6 only 45% (need hedging).
        # Mode: 1=conservative, 2=aggressive, 3=extreme, 4=ultra
        self._adaptive_slot0 = int(os.environ.get('ADAPTIVE_SLOT0', '0'))
        # Adaptive all slots: per-slot alt count based on slot k greedy prob.
        # Overrides RANK_SLOT_TOPK for slots 1-3 when enabled.
        self._adaptive_all = int(os.environ.get('ADAPTIVE_ALL', '0'))
        # Limit pending leaves per hidden_slot (source slot of block 1).
        # When set, ensures hidden state diversity in beam by capping leaves from same source slot.
        # e.g., HIDDEN_SLOT_CAP=3 means at most 3 pending from slot 0 (the rest cut).
        self._hidden_slot_cap = int(os.environ.get('HIDDEN_SLOT_CAP', '0'))
        # Iteration-level adaptive budget: scale total_tokens per iter based on block 1 difficulty.
        # Easy iter (all high conf) uses small budget, hard iter uses large budget.
        # Values: 0=off, 1=conservative, 2=aggressive, 5=hard-only boost
        self._iter_adapt_budget = int(os.environ.get('ITER_ADAPT_BUDGET', '0'))
        # Per-iter tree profile instrumentation adds ~0.3-0.5 ms of Python overhead
        # per iter (perf_counter_ns + dict updates). Default OFF in production; set
        # PROFILE_TREE=1 to enable for profile_specblock.py.
        self._profile_tree_enabled = os.environ.get('PROFILE_TREE', '0') == '1'
        # Conditional max_blocks: easy iter uses fewer blocks (skip block 3 forward).
        # 0=off (constant max_blocks); 1=p0>0.9 → max_blocks-1; 2=aggressive (p0>0.85)
        self._cond_max_blocks = int(os.environ.get('COND_MAX_BLOCKS', '0'))
        # CUDA graph for BFS draft forward: captures bucketed graphs keyed by
        # (B, cc_bucket, ttt_count) and replays to eliminate Python dispatch overhead.
        # Block 1 forward (update_cross_cache=True) stays eager.
        self._draft_cuda_graph = os.environ.get('DRAFT_CUDA_GRAPH', '0') == '1'
        self._draft_graph_cache = None
        # GPU-native tree builder: eliminates block1+BFS syncs by doing all branching
        # on GPU tensors. Supports mb=2, no_giveup/default/slim_r2 topk,
        # adaptive_slot0 in {0,1,2,3,4}, adaptive_all in {0,1,2}. Falls back to
        # _build_tree_from_block1 for unsupported configs.
        self._tree_gpu = os.environ.get('TREE_GPU', '0') == '1'
        # Per-block CUDA event timing: ~50µs Python+CUDA overhead per record(),
        # 4-6 events per iter → ~200-300µs/iter. Default ON for backward compat;
        # set BFS_EVENTS=0 to skip all .record() calls (loses per-block timing
        # report but speeds up per-iter on the hot path).
        self._bfs_events_enabled = os.environ.get('BFS_EVENTS', '1') == '1'
        # Triton mega-kernel tree builder: collapses Python/numpy tree building
        # into 2 triton kernel launches (block-1 + BFS). Same _tree_gpu_supported
        # config constraints apply. Env var TREE_BUILD_TRITON=1 to enable.
        self._tree_build_triton = os.environ.get('TREE_BUILD_TRITON', '0') == '1'
        # Native CUDA C++ tree builder: same algorithmic path as triton but
        # compiled via torch cpp_extension, trading ~80µs triton launch for
        # ~8µs cudaLaunchKernel. Requires the same _tree_gpu_supported config.
        # TREE_BUILD_CUDA=1 takes precedence over TREE_BUILD_TRITON.
        self._tree_build_cuda = os.environ.get('TREE_BUILD_CUDA', '0') == '1'
        # GPU-side finalize (prune + topology): eliminates ~1 ms of numpy
        # path + double host sync in _finalize_gpu_tree. Prune uses
        # approximation (top-K by cum_lp + ancestor closure) which may slightly
        # exceed budget; target verify tolerates small over-budget trees fine.
        self._tree_finalize_cuda = os.environ.get('TREE_FINALIZE_CUDA', '0') == '1'
        # Fixed-N pending mode: block-1 kernel selects top N_fixed pending
        # by cum_lp, pads rest with dummy entries (ttt_valid=0 masks them
        # in BFS forward attention). Eliminates the N_pend host sync;
        # enables static-shape BFS forward setup → CUDA graph prerequisite.
        # Set to 0 or absent to disable.
        self._tree_fixed_n = int(os.environ.get('TREE_FIXED_N', '0'))
        # Override RANK_SLOT_TOPK for block 2+ only. Format: "a,b,c,d" (comma-separated).
        # Allows cheaper BFS per block (e.g., "2,2,4,2") while keeping block 1 full.
        bfs_topk_str = os.environ.get('BFS_SLOT_TOPK', '')
        if bfs_topk_str:
            self._bfs_slot_topk = [int(x) for x in bfs_topk_str.split(',')]
        else:
            self._bfs_slot_topk = None
        # Override RANK_SLOT_TOPK if mode specified (param or env var)
        _mode = slot_topk_mode or os.environ.get('SLOT_TOPK_MODE', '')
        if _mode and _mode in self._SLOT_TOPK_CONFIGS:
            self.RANK_SLOT_TOPK = self._SLOT_TOPK_CONFIGS[_mode]
            print(f"  RANK_SLOT_TOPK overridden to '{_mode}': {self.RANK_SLOT_TOPK}")
        # Online logit correction: build running statistics of which tokens
        # target prefers vs draft predicts, use as log-bias to shift draft's
        # top-K selection toward target's preferences over the session.
        # Strict spec decode preserved (final accepted token still = target argmax).
        # Env-gated: LOGIT_CORRECT=1 on, LOGIT_CORRECT_LR default 0.1.
        self._logit_correct_on = os.environ.get('LOGIT_CORRECT', '0') == '1'
        self._logit_correct_lr = float(os.environ.get('LOGIT_CORRECT_LR', '0.1'))
        # Stats allocated lazily when target vocab size known
        self._target_want_count = None  # [V] rolling count of target's argmax
        self._draft_want_count = None   # [V] rolling count of draft's top-1

        # Online n-gram cache: session-level accumulation of (prev_token, curr_token) →
        # top next tokens (accepted continuations). Used to inject extra tree siblings
        # at depth 1 when draft's own predictions don't cover repeated patterns.
        # Strict spec decode: final accept still target argmax; n-gram only widens tree
        # coverage for tokens target has historically preferred in similar 2-token context.
        # Env-gated NGRAM_CACHE=1 on, NGRAM_TOPK default 3 extra siblings, NGRAM_MIN_COUNT
        # default 2 (only use if seen ≥2 times).
        self._ngram_cache_on = os.environ.get('NGRAM_CACHE', '0') == '1'
        self._ngram_topk = int(os.environ.get('NGRAM_TOPK', '3'))
        self._ngram_min_count = int(os.environ.get('NGRAM_MIN_COUNT', '2'))
        # (prev_tok, curr_tok) → Counter{next_tok: count}
        self._ngram_table = {}
        # Last 2 accepted tokens (for query); updated in generate loop after each iter
        self._ngram_prev2 = (-1, -1)

        self.target_model = None
        self.draft_model = None

        self.hidden_layer_indices = None
        self._events_pool = None
        self._events_pool_size = 0

        # Load draft config, use as defaults for max_blocks/K
        config_path = os.path.join(draft_model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)
                self.K = config_dict.get("diffspec_draft_token_num", 4)
                self.rank_classes = config_dict.get("rank_classes", 4)
                config_max_blocks = config_dict.get("num_ttt_blocks", 3)
        else:
            self.K = 4
            self.rank_classes = 4
            config_max_blocks = 3

        # Command-line overrides config defaults
        self.max_blocks = max_blocks if max_blocks is not None else config_max_blocks
        if self._strict_linear:
            # A strict path is one root-to-leaf greedy continuation with no siblings.
            # Rank predictions remain available for diagnostics but cannot truncate or
            # widen the path. Disable every adaptive/tree mode that could alter shape.
            self.beam_width = 1
            self.RANK_SLOT_TOPK = [1] * self.rank_classes
            self._bfs_slot_topk = None
            self._adaptive_budget = False
            self._adaptive_slot0 = 0
            self._adaptive_all = 0
            self._iter_adapt_budget = 0
            self._cond_max_blocks = 0
            self._slot0_beam = 0
            self._protect_d1 = False
            self._strat_beam = 0
            self._hidden_slot_cap = 0
            self._diverse_beam = False
            self._coverage_prune = False
            self._tree_fixed_n = 0
            if os.environ.get("SPECBLOCK_DYNAMIC_TREE", "0") == "1":
                raise ValueError("strict_linear is incompatible with SPECBLOCK_DYNAMIC_TREE=1")
            linear_k0 = self._tree_K if self._tree_K is not None and self._tree_K < self.K else self.K
            linear_kb = linear_k0
            if self._tree_K_bfs is not None and self._tree_K_bfs < self.K:
                linear_kb = self._tree_K_bfs
            self._strict_linear_nodes = linear_k0 + max(0, self.max_blocks - 1) * linear_kb
            if self.total_tokens != self._strict_linear_nodes:
                raise ValueError(
                    "strict_linear requires the unpruned chain budget: "
                    f"total_tokens={self.total_tokens}, expected={self._strict_linear_nodes} "
                    f"for max_blocks={self.max_blocks}, tree_K={linear_k0}, "
                    f"tree_K_bfs={linear_kb}"
                )
            print(
                "  Strict linear path enabled: "
                f"{self._strict_linear_nodes} draft nodes, no branching"
            )
        else:
            self._strict_linear_nodes = None
        # Pre-build rank→factor lookup table (moved to GPU after model load)
        self._rank_to_factor = torch.tensor([0, 4, 10, 0], dtype=torch.long)
        # Persistent reusable buffers (lazily initialized on first tree build to
        # avoid per-iter malloc of O(max_nodes) int64/float64 numpy arrays).
        self._tree_buf_cap = 0
        self._tree_buf = None
        # Reusable per-iter arange-K arrays (set after load_model to host/device).
        self._arange_K_np = None
        self._arange_K_t = None
        # Reusable CUDA events for block-forward timing inside the BFS loop.
        # Each (start, end) pair is reused across iterations; events are rewritten
        # on .record() so no stale data.
        self._bfs_events = None
        # Reusable target tree-attention mask buffer (avoids per-iter alloc)
        self._verify_mask_buf = None

    def _batch_tree_budget_for_prompt(self, prompt_len: int) -> int:
        """Return the request-local tree budget used by the batched draft loop."""
        return int(self.total_tokens)

    def _chat_template_kwargs(self) -> Dict[str, Any]:
        """Return model-specific chat-template options for every generation path."""
        return {}

    def generate_conversations(
        self,
        conversations: List[List[Dict[str, str]]],
        max_new_tokens: int,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[Dict]:
        """Run the shared request-level batch core, including for B=1."""
        from .specblock_batch import generate_conversations

        return generate_conversations(
            self,
            conversations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    def reset_after_warmup(self):
        """Remove session statistics learned from unmeasured warmup prompts."""
        self._target_want_count = None
        self._draft_want_count = None
        self._ngram_table.clear()
        self._ngram_prev2 = (-1, -1)

    @torch.inference_mode()
    def generate(
        self,
        samples: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[Dict]:
        """Batch samples by conversation turn without serial target forwards."""
        if self.tokenizer is None:
            self.load_model()
        if not self.supports_true_batch:
            if len(samples) > 1:
                raise NotImplementedError(
                    f"{type(self).__name__} has request-local hooks and does not "
                    "support true request batching"
                )
            return BaseAlgorithm.generate(
                self,
                samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        if not samples:
            self.last_batch_metrics = {
                "wall_time": 0.0,
                "prefill_time": 0.0,
                "draft_time": 0.0,
                "target_time": 0.0,
                "verify_time": 0.0,
                "iterations": 0,
                "active_sizes": [],
                "engine_batch_size": 0,
            }
            return []

        contexts = []
        for sample in samples:
            turns = sample.get("turns")
            if turns:
                contexts.append({
                    "turns": list(turns),
                    "turn_idx": 0,
                    "messages": self.prepare_conversation([]),
                    "outputs": [],
                    "metric_rows": [],
                    "result": None,
                })
            else:
                conversation = self.prepare_conversation(
                    list(sample.get("conversation", []))
                )
                contexts.append({
                    "turns": None,
                    "turn_idx": 0,
                    "messages": conversation,
                    "outputs": None,
                    "metric_rows": None,
                    "result": None,
                })

        batch_rows = []
        max_engine_batch = 0
        while True:
            pending_conversations = []
            pending_contexts = []
            for context in contexts:
                if context["result"] is not None:
                    continue
                turns = context["turns"]
                if turns is None:
                    if context["turn_idx"] > 0:
                        continue
                    pending_conversations.append(context["messages"])
                    pending_contexts.append(context)
                elif context["turn_idx"] < len(turns):
                    context["messages"].append({
                        "role": "user",
                        "content": turns[context["turn_idx"]],
                    })
                    pending_conversations.append(context["messages"].copy())
                    pending_contexts.append(context)

            if not pending_conversations:
                break

            max_engine_batch = max(max_engine_batch, len(pending_conversations))
            responses = self.generate_conversations(
                pending_conversations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            batch_rows.append(dict(self.last_batch_metrics or {}))

            for context, response in zip(pending_contexts, responses):
                if context["turns"] is None:
                    context["result"] = response
                    context["turn_idx"] = 1
                    continue
                output = response["output"]
                context["outputs"].append(output)
                context["metric_rows"].append(response.get("metrics", {}))
                context["messages"].append({"role": "assistant", "content": output})
                context["turn_idx"] += 1
                if context["turn_idx"] >= len(context["turns"]):
                    rows = context["metric_rows"]
                    accept_lengths = [
                        value
                        for row in rows
                        for value in row.get("accept_lengths_raw", [])
                    ]
                    total_tokens = sum(row.get("total_tokens", 0) for row in rows)
                    wall_time = sum(row.get("wall_time", 0.0) for row in rows)
                    prefill_time = sum(row.get("prefill_time", 0.0) for row in rows)
                    draft_time = sum(row.get("draft_time", 0.0) for row in rows)
                    target_time = sum(row.get("target_time", 0.0) for row in rows)
                    verify_time = sum(row.get("verify_time", 0.0) for row in rows)
                    other_time = sum(row.get("other_time", 0.0) for row in rows)
                    context["result"] = {
                        "output": context["outputs"],
                        "metrics": {
                            "total_tokens": total_tokens,
                            "output_token_ids": [
                                token_id
                                for row in rows
                                for token_id in row.get("output_token_ids", [])
                            ],
                            "wall_time": wall_time,
                            "tokens_per_second": total_tokens / wall_time if wall_time > 0 else 0.0,
                            "accept_length": self.compute_accept_length(accept_lengths),
                            "iterations": sum(row.get("iterations", 0) for row in rows),
                            "num_turns": len(context["turns"]),
                            "accept_lengths_raw": accept_lengths,
                            "prefill_time": prefill_time,
                            "draft_time": draft_time,
                            "target_time": target_time,
                            "verify_time": verify_time,
                            "other_time": other_time,
                            "draft_pct": draft_time / wall_time * 100 if wall_time > 0 else 0.0,
                            "target_pct": target_time / wall_time * 100 if wall_time > 0 else 0.0,
                            "verify_pct": verify_time / wall_time * 100 if wall_time > 0 else 0.0,
                        },
                    }

        wall_time = sum(row.get("wall_time", 0.0) for row in batch_rows)
        prefill_time = sum(row.get("prefill_time", 0.0) for row in batch_rows)
        draft_time = sum(row.get("draft_time", 0.0) for row in batch_rows)
        target_time = sum(row.get("target_time", 0.0) for row in batch_rows)
        verify_time = sum(row.get("verify_time", 0.0) for row in batch_rows)
        other_time = sum(row.get("batch_other_time", 0.0) for row in batch_rows)
        decode_rounds = sum(row.get("iterations", 0) for row in batch_rows)
        active_sizes = [
            size for row in batch_rows for size in row.get("active_sizes", [])
        ]
        block2_packed_leaves = sum(
            row.get("block2_packed_leaves", 0) for row in batch_rows
        )
        block2_padded_capacity = sum(
            row.get("block2_padded_capacity", 0) for row in batch_rows
        )
        target_prefill_lm_head_rows = sum(
            row.get("target_prefill_lm_head_rows", 0) for row in batch_rows
        )
        target_prefill_lm_head_capacity = sum(
            row.get("target_prefill_lm_head_capacity", 0) for row in batch_rows
        )
        target_verify_lm_head_rows = sum(
            row.get("target_verify_lm_head_rows", 0) for row in batch_rows
        )
        target_verify_lm_head_capacity = sum(
            row.get("target_verify_lm_head_capacity", 0) for row in batch_rows
        )
        target_trace_names = (
            "target_attention_widths",
            "target_fixed_attention_widths",
            "target_committed_widths",
            "target_tree_starts",
            "target_tree_widths",
            "target_commit_reserves",
        )
        target_traces = {
            name: [
                value
                for row in batch_rows
                for value in row.get(name, [])
            ]
            for name in target_trace_names
        }
        draft_forward_times = {}
        for row in batch_rows:
            for depth, value in row.get("draft_forward_times", {}).items():
                draft_forward_times[depth] = (
                    draft_forward_times.get(depth, 0.0) + value
                )
        self.last_batch_metrics = {
            "wall_time": wall_time,
            "prefill_time": prefill_time,
            "draft_time": draft_time,
            "target_time": target_time,
            "verify_time": verify_time,
            "iterations": decode_rounds,
            "active_sizes": active_sizes,
            **target_traces,
            "target_prefill_lm_head_rows": target_prefill_lm_head_rows,
            "target_prefill_lm_head_capacity": target_prefill_lm_head_capacity,
            "target_verify_lm_head_rows": target_verify_lm_head_rows,
            "target_verify_lm_head_capacity": target_verify_lm_head_capacity,
            "target_lm_head_rows_removed": (
                target_prefill_lm_head_capacity
                + target_verify_lm_head_capacity
                - target_prefill_lm_head_rows
                - target_verify_lm_head_rows
            ),
            "block2_packed_leaves": block2_packed_leaves,
            "block2_padded_capacity": block2_padded_capacity,
            "block2_padding_removed": (
                block2_padded_capacity - block2_packed_leaves
            ),
            "block2_padding_removed_pct": (
                100.0
                * (block2_padded_capacity - block2_packed_leaves)
                / block2_padded_capacity
                if block2_padded_capacity > 0
                else 0.0
            ),
            "draft_forward_times": draft_forward_times,
            "engine_batch_size": max_engine_batch,
            "batch_wall_time": wall_time,
            "batch_prefill_time": prefill_time,
            "batch_draft_time": draft_time,
            "batch_target_time": target_time,
            "batch_verify_time": verify_time,
            "batch_other_time": other_time,
            "batch_decode_rounds": decode_rounds,
            "batch_size": max_engine_batch,
            "turn_batches": len(batch_rows),
        }
        return [context["result"] for context in contexts]

    @staticmethod
    def _assert_target_runtime_environment():
        import transformers

        expected_version = "4.57.6"
        if transformers.__version__ != expected_version:
            raise RuntimeError(
                "HF SpecBlock target batching is validated only with "
                f"transformers=={expected_version}; got {transformers.__version__}"
            )
        if os.environ.get("TARGET_COMPILE", "0") != "0":
            raise RuntimeError(
                "selective target materialization requires TARGET_COMPILE=0; "
                "compiling the CausalLM wrapper would be bypassed"
            )

    @classmethod
    def _assert_target_runtime_contract(cls, target_model):
        cls._assert_target_runtime_environment()
        config = target_model.config
        backbone = getattr(target_model, "model", None)
        layers = getattr(backbone, "layers", None)
        lm_head = getattr(target_model, "lm_head", None)
        if (
            getattr(config, "model_type", None) != "llama"
            or not isinstance(layers, torch.nn.ModuleList)
            or len(layers) != int(config.num_hidden_layers)
            or not callable(getattr(backbone, "forward", None))
            or not callable(getattr(lm_head, "forward", None))
            or int(getattr(lm_head, "out_features", -1))
            != int(config.vocab_size)
        ):
            raise RuntimeError(
                "HF SpecBlock target batching requires the validated "
                "LlamaForCausalLM model/layers/lm_head contract"
            )

    def load_model(self):
        """Load target model and SpecBlock draft model."""
        from .hf_fused_projections import fuse_llama_target_projections

        self._assert_target_runtime_environment()
        print(f"Loading target model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        attn_impl = os.environ.get("TARGET_ATTN_IMPL", "triton_tree")
        if attn_impl == "flashinfer_tree":
            from .target_flashinfer_attn import register as register_flashinfer

            register_flashinfer(name="flashinfer_tree")
        elif attn_impl == "triton_tree":
            from .target_triton_attn import register as register_triton

            register_triton(name="triton_tree")
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device,
            "trust_remote_code": True,
        }
        if attn_impl and attn_impl != "sdpa":
            load_kwargs["attn_implementation"] = attn_impl
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **load_kwargs
        )
        fuse_llama_target_projections(self.target_model)
        self._assert_target_runtime_contract(self.target_model)
        self.target_model.eval()

        num_layers = self.target_model.config.num_hidden_layers
        offset = 1
        self.hidden_layer_indices = [
            1 + offset,
            num_layers // 2 - 1 + offset,
            num_layers - 4 + offset,
        ]
        print(f"  Hidden layer indices: {self.hidden_layer_indices}")

        print(f"Loading SpecBlock draft model from {self.draft_model_path}...")
        from ._specblock_inference_model_base import _SpecBlockInferenceModelBase
        from safetensors.torch import load_file

        config = AutoConfig.from_pretrained(self.draft_model_path, trust_remote_code=True)
        config.diffspec_draft_token_num = self.K

        self.draft_model = _SpecBlockInferenceModelBase(config)

        safetensors_path = os.path.join(self.draft_model_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
            self.draft_model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"No checkpoint at {safetensors_path}")

        # Copy embed_tokens from target
        with torch.no_grad():
            target_embed = self.target_model.model.embed_tokens.weight
            self.draft_model.input_layer.embed_tokens.weight.copy_(target_embed)
        print(f"  Copied embed_tokens ({target_embed.shape})")

        self.draft_model = self.draft_model.to(device=self.device, dtype=torch.bfloat16)
        self.draft_model.eval()
        self.draft_model.prepare_for_inference(quantize=getattr(self, 'draft_quantize', None))

        # Vocab mapping
        vocab_mapping_path = os.path.join(self.draft_model_path, "vocab_mapping.pt")
        draft_vocab = getattr(config, 'draft_vocab_size', config.vocab_size)
        if draft_vocab != config.vocab_size and os.path.exists(vocab_mapping_path):
            self.draft_model.load_vocab_mapping(vocab_mapping_path)
            print(f"  Loaded vocab mapping: draft={draft_vocab}, target={config.vocab_size}")
        elif draft_vocab != config.vocab_size:
            print(f"  Using identity mapping (draft={draft_vocab}, target={config.vocab_size}, "
                  f"vocab_mapping.pt not found)")
        else:
            print(f"  No vocab mapping needed (vocab_size={config.vocab_size})")

        # Move rank lookup table to GPU
        self._rank_to_factor = self._rank_to_factor.to(self.device)

        # Pre-compute arange buffers reused across tree builds
        self._arange_K_np = np.arange(self.K, dtype=np.int64)
        self._arange_K_t = torch.arange(self.K, device=self.device)

        # Warmup: pre-compile CUDA kernels for all possible batch sizes
        self._warmup_draft_kernels()

        self._init_draft_cuda_graph()

        # Static-shape, sync-free draft tree builder (opt-in via env).
        # Replaces _build_tree_from_block1_dispatch when SPECBLOCK_STATIC=1.
        # Uniformly branches MAX_TOPK alts per slot; rank-aware effects live
        # in the prune-time cumulative bias (preserves cum_lp monotonicity).
        self._static_draft = None
        self._use_static_draft = (
            not self._strict_linear
            and os.environ.get("SPECBLOCK_STATIC", "0") == "1"
        )
        if self._use_static_draft:
            try:
                from .specblock_static_draft import StaticDraftBuilder
                self._static_draft = StaticDraftBuilder(
                    draft_model=self.draft_model,
                    K=self.K,
                    total_tokens=self.total_tokens,
                    max_blocks=self.max_blocks,
                    rank_classes=self.rank_classes,
                )
                if self._draft_graph_cache is not None:
                    self._static_draft.attach_graph_cache(self._draft_graph_cache)
                print(
                    f"  [SPECBLOCK_STATIC=1] StaticDraftBuilder enabled: "
                    f"MAX_TOPK={self._static_draft.MAX_TOPK}, "
                    f"N_PEND={self._static_draft.N_PEND}, "
                    f"MAX_NODES={self._static_draft.MAX_NODES}, "
                    f"λ={self._static_draft.rank_bias_lambda}"
                )
            except Exception as _e:
                print(
                    f"  WARNING: failed to init StaticDraftBuilder: "
                    f"{type(_e).__name__}: {_e}"
                )
                self._use_static_draft = False
                self._static_draft = None

        print(f"SpecBlock ready! K={self.K}, max_blocks={self.max_blocks}, "
              f"total_tokens={self.total_tokens}")

    @torch.no_grad()
    def _warmup_draft_kernels(self):
        """Pre-compile CUDA kernels for all batch sizes used in tree building."""
        K = self.K
        H = self.draft_model.config.hidden_size
        H3 = H * 3
        num_layers = self.draft_model.num_layers
        device = self.device
        max_N = self.beam_width + 5  # max pending leaves + margin

        print("  Warming up draft model kernels...", end="", flush=True)

        # Prefill warmup (B=1, short seq)
        dummy_h3 = torch.zeros(1, 4, H3, device=device, dtype=torch.bfloat16)
        dummy_ids = torch.zeros(1, 4, device=device, dtype=torch.long)
        cache = self.draft_model.reset_cache(max_cache_len=64 * K)
        self.draft_model.prefill(dummy_h3, dummy_ids)

        # update_cache_and_draft warmup (B=1, various N)
        for N_update in [2, 3, 4, 5]:
            cache = self.draft_model.reset_cache(max_cache_len=64 * K)
            # Prefill 1 position first
            self.draft_model.prefill(dummy_h3[:, :1, :], dummy_ids[:, :1], chunk_size=1)
            dummy_h3_n = torch.zeros(1, N_update, H3, device=device, dtype=torch.bfloat16)
            dummy_ids_n = torch.zeros(1, N_update, device=device, dtype=torch.long)
            self.draft_model.update_cache_and_draft(dummy_h3_n, dummy_ids_n, cache, 1)

        # forward_with_cache warmup for all batch sizes N=1..max_N
        for N in range(1, max_N + 1):
            cache = self.draft_model.reset_cache(max_cache_len=64 * K)
            # Prefill 1 position to initialize cache (stores K entries in cross cache)
            self.draft_model.prefill(dummy_h3[:, :1, :], dummy_ids[:, :1], chunk_size=1)
            cross_count = K
            cross_cache = []
            for lc in cache:
                if lc[0] is not None:
                    cross_cache.append([
                        lc[0][:, :, :cross_count, :].expand(N, -1, -1, -1),
                        lc[1][:, :, :cross_count, :].expand(N, -1, -1, -1),
                        cross_count,
                    ])
                else:
                    cross_cache.append([None, None, 0])

            dummy_hidden = torch.zeros(N, 1, H, device=device, dtype=torch.bfloat16)
            dummy_input = torch.zeros(N, 1, device=device, dtype=torch.long)

            # TTT cache: per-layer (k, v) with C_ttt = K (one prior block)
            n_kv_heads = self.draft_model.config.num_key_value_heads
            head_dim = H // self.draft_model.config.num_attention_heads
            dummy_ttt = [
                (torch.zeros(N, n_kv_heads, K, head_dim, device=device, dtype=torch.bfloat16),
                 torch.zeros(N, n_kv_heads, K, head_dim, device=device, dtype=torch.bfloat16))
                for _ in range(num_layers)
            ]
            dummy_ttt_mask = torch.ones(N, K, device=device, dtype=torch.bool)

            self.draft_model.forward_with_cache(
                hidden=dummy_hidden, input_ids=dummy_input,
                cache=cross_cache, position_id=1,
                use_draft_condition=True,
                ttt_cache=dummy_ttt, ttt_mask=dummy_ttt_mask,
                update_cross_cache=False,
            )

        torch.cuda.synchronize()
        print(" done")

    @torch.no_grad()
    def _warmup_draft_kernels_extended(self):
        """Extended warmup covering multiple cache sizes.

        Base warmup only exercises cache=64*K. Long-prompt benches (cnn_dm
        gen_len=1024 → cache ~4k slots) re-trigger torch.compile at runtime,
        causing draft=46ms stuck cascade. Pre-run with representative cache
        sizes so compile cache populates before first real iter.
        """
        import time as _t
        # Run base warmup first (small cache)
        self._warmup_draft_kernels()

        K = self.K
        H = self.draft_model.config.hidden_size
        H3 = H * 3
        device = self.device

        print("  Extended warmup (long-cache shapes)...", end="", flush=True)
        _t0 = _t.time()

        # Warmup with representative cache sizes covering long-bench range.
        # cache_count grows in multiples of N*K per iter. Power-of-2-like
        # steps should hit common dynamo guard boundaries.
        cache_sizes = [256, 512, 1024, 2048, 4096]
        for cs in cache_sizes:
            # Prefill dummy to populate cache to ~cs slots
            prefill_len = max(1, cs // K)
            dummy_h3 = torch.zeros(1, prefill_len, H3, device=device, dtype=torch.bfloat16)
            dummy_ids = torch.zeros(1, prefill_len, device=device, dtype=torch.long)
            cache = self.draft_model.reset_cache(max_cache_len=(cs + 1024) * K)
            self.draft_model.prefill(dummy_h3, dummy_ids)

            # Warmup update_cache_and_draft at this cache size with N=1..4
            # (typical accept_length+1 range in decoding)
            for N_update in [1, 2, 3, 4, 5]:
                dummy_h3_n = torch.zeros(1, N_update, H3, device=device, dtype=torch.bfloat16)
                dummy_ids_n = torch.zeros(1, N_update, device=device, dtype=torch.long)
                self.draft_model.update_cache_and_draft(
                    dummy_h3_n, dummy_ids_n, cache, prefill_len,
                )

        torch.cuda.synchronize()
        _dt = _t.time() - _t0
        print(f" done ({_dt:.1f}s, {len(cache_sizes)} cache sizes × 5 N values)")

    def _init_draft_cuda_graph(self):
        """Initialize DraftForwardGraphCache if env DRAFT_CUDA_GRAPH=1.

        Called from load_model after draft model is fully ready. Lazy-captures
        per bucket on first call — no precapture here.
        """
        if not self._draft_cuda_graph:
            return
        try:
            from .draft_cuda_graph import DraftForwardGraphCache
            self._draft_graph_cache = DraftForwardGraphCache(
                self.draft_model,
                max_batch=self.beam_width + 5,
                max_cross_count=2048,
                verbose=True,
            )
            print(f"  DraftForwardGraphCache enabled (max_batch={self.beam_width + 5})")
        except Exception as e:
            print(f"  WARNING: failed to init DraftForwardGraphCache: {e}")
            self._draft_graph_cache = None

    def _ensure_events_pool(self, size: int):
        if self._events_pool is None or self._events_pool_size < size:
            self._events_pool_size = max(size, 512)
            self._events_pool = {
                'prefill_start': torch.cuda.Event(enable_timing=True),
                'prefill_end': torch.cuda.Event(enable_timing=True),
                'draft_start': [torch.cuda.Event(enable_timing=True) for _ in range(self._events_pool_size)],
                'draft_end': [torch.cuda.Event(enable_timing=True) for _ in range(self._events_pool_size)],
                'target_start': [torch.cuda.Event(enable_timing=True) for _ in range(self._events_pool_size)],
                'target_end': [torch.cuda.Event(enable_timing=True) for _ in range(self._events_pool_size)],
            }

    def _ensure_tree_buffers(self, max_nodes: int):
        """Ensure pre-allocated numpy tree buffers exist and are large enough.

        Called once at the top of every tree build; reuses prior allocation to
        skip 6× np.empty() per call. Returns a tuple of ndarrays.
        """
        if self._tree_buf is None or self._tree_buf_cap < max_nodes:
            # Grow 1.5x to avoid frequent reallocations when max_nodes creeps up.
            new_cap = max(max_nodes, int(self._tree_buf_cap * 1.5) + 1)
            self._tree_buf = {
                'tokens':  np.empty(new_cap, dtype=np.int64),
                'parents': np.empty(new_cap, dtype=np.int64),
                'lps':     np.empty(new_cap, dtype=np.float64),
                'ranks':   np.empty(new_cap, dtype=np.int64),
                'blocks':  np.empty(new_cap, dtype=np.int64),
                'slots':   np.empty(new_cap, dtype=np.int64),
            }
            self._tree_buf_cap = new_cap
        b = self._tree_buf
        return b['tokens'], b['parents'], b['lps'], b['ranks'], b['blocks'], b['slots']

    def _get_bfs_event_pair(self, depth: int, idx: int):
        """Return a unique (start, end) cuda event pair for the given (depth, idx).

        CUDA events cannot be reused when timings must be aggregated across
        iterations — each .record() overwrites the prior timestamp. We maintain
        a lazy pool per depth; events are created once and retained for the
        lifetime of the algorithm instance.
        """
        if self._bfs_events is None:
            self._bfs_events = {}
        pool = self._bfs_events.get(depth)
        if pool is None:
            pool = []
            self._bfs_events[depth] = pool
        while len(pool) <= idx:
            pool.append((
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            ))
        return pool[idx]

    def _select_hidden_3h_sources(self, hidden_states):
        return tuple(hidden_states[i] for i in self.hidden_layer_indices)

    def _extract_hidden_3h(self, hidden_states) -> torch.Tensor:
        return torch.cat(self._select_hidden_3h_sources(hidden_states), dim=-1)

    def _needs_vocab_mapping(self) -> bool:
        if not hasattr(self, '_vocab_mapping_needed'):
            draft_vocab = getattr(self.draft_model.config, 'draft_vocab_size',
                                  self.draft_model.config.vocab_size)
            target_vocab = self.draft_model.config.vocab_size
            self._vocab_mapping_needed = (draft_vocab != target_vocab)
        return self._vocab_mapping_needed

    def _map_draft_to_target(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        """Map draft vocab IDs to target vocab IDs."""
        if self._needs_vocab_mapping():
            return draft_token_ids + self.draft_model.d2t[draft_token_ids]
        return draft_token_ids

    # Branching factors for each rank class (class_idx → factor)
    # Class 0: correct, continue; Class 1: small branch (4); Class 2: large branch (10); Class 3: give up
    RANK_BRANCH_FACTORS = {0: 0, 1: 4, 2: 10, 3: 0}
    # Per-slot branching: topk per rank class (includes greedy, so actual alternatives = topk - 1)
    # rank=0 (confident): top-2; rank=1: top-4; rank=2: top-10; rank=3 (give up): 0
    # Note: slot 0 always branches with beam_width regardless of rank (tree root diversity)
    RANK_SLOT_TOPK = [2, 4, 10, 0]

    # Alternative topk configs for experimentation (set via _slot_topk_mode)
    _SLOT_TOPK_CONFIGS = {
        'default':     [2, 4, 10, 0],   # original: give_up=0
        'no_giveup':   [2, 4, 10, 4],   # give_up still gets topk=4 (no chain break)
        'no_giveup2':  [2, 4, 10, 2],   # give_up gets minimal topk=2 (cheapest fix)
        'generous':    [4, 6, 10, 4],    # more alternatives for confident + give_up fallback
        'uniform':     [6, 6, 6, 6],     # same topk regardless of rank
        'flat10':      [10, 10, 10, 10], # maximum alternatives everywhere
        'balanced':    [4, 4, 4, 4],     # balanced across all ranks, used with lower beam_width
        'hidden_div':  [6, 6, 6, 4],     # more alts at deeper slots for hidden diversity (use with beam_width=5)
        'slot0_only':  [1, 1, 1, 1],    # only slot 0 branches (beam_width), slots 1-3 greedy only
        'linear':      [1, 1, 1, 1],    # strict mode also forces slot-0 beam=1 and no give-up break
        'slim_r2':     [2, 4, 6, 4],    # slim rank=2 from 10 to 6 (reduce d3/d4 duplicate budget)
        'slim_r2b':    [2, 3, 5, 3],    # even slimmer on d3/d4
    }

    def _walk_rank_slots_batch(self, rank_preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched rank walking for N leaves. Zero .item() calls."""
        if self._strict_linear:
            batch = rank_preds.shape[0]
            return (
                torch.full((batch,), rank_preds.shape[1], dtype=torch.long, device=rank_preds.device),
                torch.zeros(batch, dtype=torch.long, device=rank_preds.device),
                torch.zeros(batch, dtype=torch.bool, device=rank_preds.device),
            )
        return _walk_rank_slots_compiled(rank_preds, self.rank_classes, self._rank_to_factor)

    def _aggregate_rank_stats(
        self,
        all_iter_rank_stats: list,
        accept_lengths_raw: list,
    ) -> Dict:
        """Aggregate per-iteration rank stats into summary metrics.

        Args:
            all_iter_rank_stats: list of rank_stats per iteration.
                Each is a list of dicts with 'rank_preds', 'M', 'branch' per block.
            accept_lengths_raw: list of accept lengths per iteration.

        Returns:
            Dict with aggregated rank statistics.
        """
        K = self.K
        if not all_iter_rank_stats:
            return {}

        # Collect block-1 stats (always available)
        all_M = []
        all_branch = []
        per_slot_rank_counts = [[0] * self.rank_classes for _ in range(K)]
        rank0_pred_count = 0    # times we predicted rank=0
        rank0_pred_accepted = 0  # times pred=0 was actually accepted (slot within accept_length)

        for iter_idx, iter_stats in enumerate(all_iter_rank_stats):
            if not iter_stats:
                continue
            # Block 1 stats (first entry)
            b1 = iter_stats[0]
            rank_preds = b1['rank_preds']
            all_M.append(b1['M'])
            all_branch.append(b1['branch'])

            # Per-slot rank prediction distribution
            for k in range(min(len(rank_preds), K)):
                r = rank_preds[k]
                if 0 <= r < self.rank_classes:
                    per_slot_rank_counts[k][r] += 1

            # Compare rank predictions with accept outcome
            accept_len = accept_lengths_raw[iter_idx] if iter_idx < len(accept_lengths_raw) else 0
            for k in range(min(len(rank_preds), K)):
                if rank_preds[k] == 0:
                    rank0_pred_count += 1
                    if k < accept_len:
                        rank0_pred_accepted += 1

        # Compute rank0 recall: slots that were accepted and predicted rank=0
        rank0_truth_count = 0
        rank0_truth_predicted = 0
        for iter_idx, iter_stats in enumerate(all_iter_rank_stats):
            if not iter_stats:
                continue
            accept_len = accept_lengths_raw[iter_idx] if iter_idx < len(accept_lengths_raw) else 0
            rank_preds = iter_stats[0]['rank_preds']
            for k in range(min(accept_len, len(rank_preds), K)):
                rank0_truth_count += 1
                if rank_preds[k] == 0:
                    rank0_truth_predicted += 1

        return {
            "mean_M": sum(all_M) / len(all_M) if all_M else 0.0,
            "mean_branch": sum(all_branch) / len(all_branch) if all_branch else 0.0,
            "per_slot_rank_dist": per_slot_rank_counts,
            "rank0_precision": (r0p := rank0_pred_accepted / rank0_pred_count if rank0_pred_count > 0 else 0.0),
            "rank0_recall": (r0r := rank0_truth_predicted / rank0_truth_count if rank0_truth_count > 0 else 0.0),
            "rank0_f1": (2 * r0p * r0r / (r0p + r0r)) if (r0p + r0r) > 0 else 0.0,
            "num_blocks_per_iter": [len(s) for s in all_iter_rank_stats],
        }

    @staticmethod
    def _compute_coverage_stats(
        draft_tokens: torch.Tensor,
        target_choices: torch.Tensor,
        retrieve_indices: torch.Tensor,
        node_block_slots: list,
    ) -> Dict[str, Dict]:
        """Compute per-(block, slot) coverage stats across ALL tree nodes.

        For each tree node, checks whether its draft token matches the target's
        greedy choice at that position given that node's ancestor context.
        Aggregates per (block, slot) — not limited to the best candidate path.

        Args:
            draft_tokens: [1, N+1] all tree tokens (1-indexed, index 0 = input)
            target_choices: [num_paths, max_depth] argmax of target logits (CPU tensor)
            retrieve_indices: [num_paths, max_depth+1] node indices (1-indexed)
            node_block_slots: list of (block_idx, slot_idx) per tree node (0-indexed)

        Returns:
            Dict "b{block}_p{pos}" -> {"correct": int, "total": int, "accuracy": float}
        """
        # Build node_0idx -> (path_idx, depth_in_path) using first path that contains it
        node_to_path_depth = {}
        ri_np = retrieve_indices.cpu().numpy() if retrieve_indices.is_cuda else retrieve_indices.numpy()
        num_paths, max_depth_p1 = ri_np.shape
        for p in range(num_paths):
            for d in range(1, max_depth_p1):
                n1 = int(ri_np[p, d])
                if n1 <= 0:
                    break
                n0 = n1 - 1
                if n0 not in node_to_path_depth:
                    node_to_path_depth[n0] = (p, d)

        tc_np = target_choices.numpy()  # [num_paths, max_depth]
        dt_np = draft_tokens[0].cpu().numpy()  # [N+1]

        stats = {}
        for node_0idx, (block_idx, slot_idx) in enumerate(node_block_slots):
            if node_0idx not in node_to_path_depth:
                continue
            p, d = node_to_path_depth[node_0idx]
            if d - 1 >= tc_np.shape[1]:
                continue

            target_tok = int(tc_np[p, d - 1])
            draft_tok  = int(dt_np[node_0idx + 1])  # 1-indexed

            key = f"b{block_idx}_p{slot_idx + 1}"
            if key not in stats:
                stats[key] = {"correct": 0, "total": 0}
            stats[key]["total"]  += 1
            stats[key]["correct"] += int(draft_tok == target_tok)

        return stats

    @staticmethod
    def _aggregate_block_pos_stats(all_iter_stats: list) -> Dict:
        """Aggregate per-block-pos stats across iterations.

        Args:
            all_iter_stats: list of dicts from _compute_per_block_pos_stats

        Returns:
            Dict mapping "b{block}_p{pos}" -> {
                "correct": int, "total": int, "accuracy": float
            }
        """
        agg = {}
        for iter_stats in all_iter_stats:
            for key, counts in iter_stats.items():
                if key not in agg:
                    agg[key] = {"correct": 0, "total": 0}
                agg[key]["correct"] += counts["correct"]
                agg[key]["total"] += counts["total"]

        for key in agg:
            t = agg[key]["total"]
            c = agg[key]["correct"]
            agg[key]["accuracy"] = c / t if t > 0 else 0.0

        return agg

    def _prune_tree_np(self, n_nodes, tn_tokens, tn_parents, tn_lps,
                       tn_ranks, tn_blocks, tn_slots, budget):
        """Prune tree to budget using numpy arrays. Returns new n_nodes.

        Uses coverage-aware pruning when self._coverage_prune is True:
        penalizes nodes whose (depth, token) pair is already covered,
        preferring unique tokens at each depth.
        """
        if n_nodes <= budget:
            return n_nodes

        if getattr(self, '_coverage_prune', False):
            return self._prune_tree_coverage_np(
                n_nodes, tn_tokens, tn_parents, tn_lps,
                tn_ranks, tn_blocks, tn_slots, budget)

        sorted_indices = np.argsort(-tn_lps[:n_nodes])  # descending by cum_log_prob
        kept = set()
        for idx in sorted_indices:
            if len(kept) >= budget:
                break
            chain = []
            cur = int(idx)
            while cur >= 0 and cur not in kept:
                chain.append(cur)
                cur = int(tn_parents[cur])
            if len(kept) + len(chain) <= budget:
                kept.update(chain)

        kept_sorted = sorted(kept)
        old_to_new = {old: new for new, old in enumerate(kept_sorted)}
        new_n = len(kept_sorted)

        # Compact arrays in-place
        new_tokens = tn_tokens[kept_sorted]
        new_parents = np.array([old_to_new.get(int(tn_parents[old]), -1) for old in kept_sorted], dtype=np.int64)
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

    def _prune_tree_coverage_np(self, n_nodes, tn_tokens, tn_parents, tn_lps,
                                tn_ranks, tn_blocks, tn_slots, budget):
        """Path-level dedup pruning: remove duplicate paths, keep best by cum_lp.

        Two paths are "duplicate" if their token sequences (root to leaf) are identical.
        After removing duplicate paths, fill budget with standard cum_lp pruning.
        This preserves path integrity while eliminating redundant block 2 expansions.
        """
        if n_nodes <= budget:
            return n_nodes

        # Step 1: Find all leaf nodes
        is_parent = set()
        for i in range(n_nodes):
            p = int(tn_parents[i])
            if p >= 0:
                is_parent.add(p)
        leaves = [i for i in range(n_nodes) if i not in is_parent]

        # Step 2: For each leaf, extract root-to-leaf token sequence
        leaf_paths = {}  # leaf_idx -> (token_tuple, cum_lp, chain_set)
        for leaf in leaves:
            chain = []
            tokens = []
            cur = leaf
            while cur >= 0:
                chain.append(cur)
                tokens.append(int(tn_tokens[cur]))
                cur = int(tn_parents[cur])
            tokens.reverse()
            chain_set = set(chain)
            leaf_paths[leaf] = (tuple(tokens), float(tn_lps[leaf]), chain_set)

        # Step 3: Group by token sequence, keep only best cum_lp per group
        from collections import defaultdict
        path_groups = defaultdict(list)
        for leaf, (tok_seq, lp, chain) in leaf_paths.items():
            path_groups[tok_seq].append((lp, leaf, chain))

        # Sort each group by cum_lp, mark duplicates for removal
        dup_leaves = set()
        n_deduped = 0
        for tok_seq, group in path_groups.items():
            if len(group) <= 1:
                continue
            group.sort(key=lambda x: -x[0])  # best cum_lp first
            for _, leaf, _ in group[1:]:  # remove all but best
                dup_leaves.add(leaf)
                n_deduped += 1

        # Step 4: Standard cum_lp pruning, but skip duplicate leaves
        sorted_indices = np.argsort(-tn_lps[:n_nodes])
        kept = set()
        for idx in sorted_indices:
            if len(kept) >= budget:
                break
            idx = int(idx)
            # Skip if this is a duplicate leaf (or only reachable from dup leaves)
            if idx in dup_leaves:
                continue
            chain = []
            cur = idx
            while cur >= 0 and cur not in kept:
                chain.append(cur)
                cur = int(tn_parents[cur])
            if len(kept) + len(chain) <= budget:
                kept.update(chain)

        # Fill remaining with any remaining nodes (including from dup paths if budget allows)
        if len(kept) < budget:
            for idx in sorted_indices:
                if len(kept) >= budget:
                    break
                idx = int(idx)
                if idx in kept:
                    continue
                chain = []
                cur = idx
                while cur >= 0 and cur not in kept:
                    chain.append(cur)
                    cur = int(tn_parents[cur])
                if len(kept) + len(chain) <= budget:
                    kept.update(chain)

        kept_sorted = sorted(kept)
        old_to_new = {old: new for new, old in enumerate(kept_sorted)}
        new_n = len(kept_sorted)

        new_tokens = tn_tokens[kept_sorted]
        new_parents = np.array([old_to_new.get(int(tn_parents[old]), -1) for old in kept_sorted], dtype=np.int64)
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

    def _prune_tree(self, tree_nodes: list, node_ranks: list,
                    node_block_slots: list, budget: int) -> Tuple[list, list, list]:
        """Prune tree to budget nodes, keeping best paths by cum_log_prob.

        tree_nodes: list of (token_id, parent_idx, cum_log_prob)
        node_ranks: list of rank predictions per node (parallel to tree_nodes)
        node_block_slots: list of (block_idx, slot_idx) per node
        When removing nodes, all ancestors of kept nodes must also be kept.

        Strategy: sort nodes by cum_log_prob descending, greedily add nodes and
        their ancestors until budget is reached. Then re-index.
        """
        n = len(tree_nodes)
        if n <= budget:
            return tree_nodes, node_ranks, node_block_slots

        # Sort indices by cum_log_prob (best first)
        sorted_indices = sorted(range(n), key=lambda i: tree_nodes[i][2], reverse=True)

        kept = set()
        for idx in sorted_indices:
            if len(kept) >= budget:
                break
            # Trace ancestors
            chain = []
            cur = idx
            while cur >= 0 and cur not in kept:
                chain.append(cur)
                cur = tree_nodes[cur][1]  # parent_idx
            # Add chain if it fits within remaining budget
            if len(kept) + len(chain) <= budget:
                kept.update(chain)

        # Build re-indexed tree
        kept_sorted = sorted(kept)
        old_to_new = {old: new for new, old in enumerate(kept_sorted)}
        pruned = []
        pruned_ranks = []
        pruned_block_slots = []
        for old_idx in kept_sorted:
            tok, parent, clp = tree_nodes[old_idx]
            new_parent = old_to_new[parent] if parent >= 0 and parent in old_to_new else -1
            pruned.append((tok, new_parent, clp))
            pruned_ranks.append(node_ranks[old_idx])
            pruned_block_slots.append(node_block_slots[old_idx])
        return pruned, pruned_ranks, pruned_block_slots

    # ============================================================
    #    Tree Construction via Multi-block TTT + Rank
    # ============================================================

    @torch.no_grad()
    def _build_draft_tree(
        self,
        hidden_3h: torch.Tensor,       # [1, 1, 3H]
        input_id: torch.Tensor,         # [1, 1]
        draft_cache: list,
        draft_position: int,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list, list, list]:
        """Build rank-guided draft tree. Calls block 1 forward then BFS expansion."""
        # Record per-block timing for block 0 (first forward). Skip entirely
        # when BFS_EVENTS=0 to remove ~100µs of Python+CUDA dispatch per iter.
        if self._bfs_events_enabled:
            _b0_idx = len(self._block_forward_events[0])
            blk_start, blk_end = self._get_bfs_event_pair(0, _b0_idx)
            blk_start.record()
        else:
            blk_start = blk_end = None

        # Graph path: replay captures ~30 CUDA launches as one replay, cutting
        # CPU dispatch overhead (~2.7 → 0.1 ms). Cache scatter + count bump is
        # done outside the graph by run_block1 itself.
        # Gated by its own env var so we can A/B independently of BFS graph.
        graph_out = None
        if (self._draft_graph_cache is not None
                and os.environ.get('DRAFT_CUDA_GRAPH_B1', '0') == '1'):
            try:
                graph_out = self._draft_graph_cache.run_block1(
                    hidden_3h=hidden_3h,
                    input_ids=input_id,
                    draft_cache=draft_cache,
                    position_id=draft_position,
                )
            except Exception as _e:
                import traceback
                print(f"[DRAFT_CUDA_GRAPH_B1] fallback: {type(_e).__name__}: {_e}")
                traceback.print_exc()
                graph_out = None

        if graph_out is not None:
            logits, rank_logits, draft_hidden, ttt_kv = graph_out
        else:
            logits, rank_logits, draft_hidden, ttt_kv = self.draft_model.forward_with_cache(
                hidden=hidden_3h,
                input_ids=input_id,
                cache=draft_cache,
                position_id=draft_position,
                use_draft_condition=False,
                ttt_cache=None,
                ttt_mask=None,
                update_cross_cache=True,
            )

        if self._bfs_events_enabled:
            blk_end.record()
            self._block_forward_events[0].append((blk_start, blk_end))

        # Dispatch: STATIC > CUDA fixed-N > CUDA > triton mega-kernel > TREE_GPU > numpy path.
        if not hasattr(self, '_path_logged'):
            print(f'[PATH] static={self._use_static_draft} gpu_supported={self._tree_gpu_supported()} build_triton={self._tree_build_triton} build_cuda={self._tree_build_cuda} tree_gpu={self._tree_gpu}', flush=True)
            self._path_logged = True
        return self._build_tree_from_block1_dispatch(
            logits,
            rank_logits,
            draft_hidden,
            ttt_kv,
            input_id,
            draft_cache,
            draft_position,
            temperature=temperature,
        )

    def _tree_gpu_supported(self) -> bool:
        """GPU-native builder only covers the production config path."""
        return (
            self.max_blocks == 2
            and self._iter_adapt_budget == 0
            and self._cond_max_blocks == 0
            and not self._diverse_beam
            and not self._protect_d1
            and self._strat_beam == 0
            and self._hidden_slot_cap == 0
            and not self._adaptive_budget
            and not self._coverage_prune
            and self._slot0_beam == 0
            and self._bfs_slot_topk is None
            and self._tree_K is None
            and self._tree_K_bfs is None
            # Only implement adaptive_slot0 modes 0-4 and adaptive_all 0-2
            and self._adaptive_slot0 in (0, 1, 2, 3, 4)
            and self._adaptive_all in (0, 1, 2)
            # No-give_up-break behaviour: require give_up class topk > 0
            and self.RANK_SLOT_TOPK[self.rank_classes - 1] > 0
        )

    @torch.no_grad()
    def _build_tree_from_block1_gpu(
        self,
        logits: torch.Tensor,          # [1, K, draft_vocab]
        rank_logits: torch.Tensor,     # [1, K, rank_classes]
        draft_hidden: torch.Tensor,    # [1, K, H]
        ttt_kv: list,
        input_id: torch.Tensor,        # [1, 1]
        draft_cache: list,
        draft_position: int,
        temperature: float = 0.0,
    ):
        """GPU-native tree builder for mb=2.

        Keeps tree nodes on GPU through both block-1 branching and the single
        BFS iter; one bulk GPU→CPU transfer at the end feeds the existing
        numpy prune + triton topology. Targets ~1 sync for sizing + 1 sync
        at the final transfer.
        """
        K = self.K
        device = logits.device
        num_layers = len(ttt_kv)
        beam_width = self.beam_width
        rank_classes = self.rank_classes
        give_up_class = rank_classes - 1
        adaptive_slot0 = self._adaptive_slot0
        adaptive_all = self._adaptive_all

        # Cache per-(K, beam_width, device) constants.
        cache_key = (K, beam_width, device)
        consts = getattr(self, '_tree_gpu_consts', None)
        if consts is None or consts.get('key') != cache_key:
            max_topk = max(beam_width, max(self.RANK_SLOT_TOPK))
            consts = {
                'key': cache_key,
                'max_topk': max_topk,
                'arange_K': torch.arange(K, device=device),
                'rank_slot_topk_t': torch.tensor(
                    self.RANK_SLOT_TOPK, dtype=torch.long, device=device,
                ),
                'two_t': torch.tensor(2, dtype=torch.long, device=device),
                'three_t': torch.tensor(3, dtype=torch.long, device=device),
                'bw_t': torch.tensor(beam_width, dtype=torch.long, device=device),
                'one_t': torch.tensor(1, dtype=torch.long, device=device),
                'min2_bw_t': torch.tensor(min(2, beam_width), dtype=torch.long, device=device),
                'min3_bw_t': torch.tensor(min(3, beam_width), dtype=torch.long, device=device),
                'min4_bw_t': torch.tensor(min(4, beam_width), dtype=torch.long, device=device),
                'min5_bw_t': torch.tensor(min(5, beam_width), dtype=torch.long, device=device),
                'min6_bw_t': torch.tensor(min(6, beam_width), dtype=torch.long, device=device),
                'min7_bw_t': torch.tensor(min(7, beam_width), dtype=torch.long, device=device),
                'min8_bw_t': torch.tensor(min(8, beam_width), dtype=torch.long, device=device),
            }
            self._tree_gpu_consts = consts
        arange_K = consts['arange_K']
        rank_slot_topk_t = consts['rank_slot_topk_t']
        max_topk = consts['max_topk']
        two_t = consts['two_t']
        three_t = consts['three_t']

        # Pre-allocated GPU tree buffer. 6 fields stored in a [6, max_nodes] tensor
        # so we can do one cpu() transfer at the end.
        # Block 1 adds up to K + K*(max_topk-1) nodes.
        # Block 2 (single BFS iter) adds up to N_pend_max*K + N_pend_max*K*(max_topk-1).
        # Upper-bound N_pend_max = K + K*(max_topk-1) + 1 (block-1 pending ceiling).
        max_block1_nodes = K + K * (max_topk - 1) + 1
        max_block2_nodes = max_block1_nodes * K * max_topk
        max_nodes = max(self.total_tokens + 200, max_block1_nodes + max_block2_nodes + 100)
        buf = getattr(self, '_tree_gpu_buf', None)
        if buf is None or buf['tokens'].shape[0] < max_nodes:
            buf = {
                'tokens': torch.empty(max_nodes, dtype=torch.long, device=device),
                'parents': torch.empty(max_nodes, dtype=torch.long, device=device),
                'lps': torch.empty(max_nodes, dtype=torch.float32, device=device),
                'ranks': torch.empty(max_nodes, dtype=torch.long, device=device),
                'blocks': torch.empty(max_nodes, dtype=torch.long, device=device),
                'slots': torch.empty(max_nodes, dtype=torch.long, device=device),
            }
            self._tree_gpu_buf = buf

        sample_token = input_id[:, -1]  # [1]

        # ============================================================
        #   Block 1 GPU branching (no sync)
        # ============================================================
        last_p = logits[0]  # [K, V_d]
        log_probs = F.log_softmax(last_p.float(), dim=-1)  # [K, V]
        rank_preds = rank_logits[0].argmax(dim=-1)  # [K]

        # give_up flag: follows walk_rank_slots semantics — True when
        # first non-rank0 prediction is rank=give_up_class.
        _, _, gu_t = self._walk_rank_slots_batch(rank_preds.unsqueeze(0))  # [1], [1], [1]

        all_top_idx = torch.topk(last_p, max_topk, dim=-1).indices  # [K, max_topk]
        all_top_target = self._map_draft_to_target(all_top_idx)  # [K, max_topk]
        all_top_lps = log_probs.gather(1, all_top_idx)  # [K, max_topk]
        greedy_target = all_top_target[:, 0]  # [K]
        greedy_lps = all_top_lps[:, 0]  # [K]

        # Per-slot topk count (adaptive_slot0 + adaptive_all), GPU-native.
        p_t = greedy_lps.exp()  # [K]
        base_topk = rank_slot_topk_t[rank_preds]  # [K] class-based table lookup
        # F axis novelty: when rank_head is online-adapted (adapt weights shift
        # toward the observed query distribution), rank_preds = argmax over
        # softmax becomes more accurate per-sample → class-based tree branching
        # becomes effectively query-adaptive. No new formula/params/hparams;
        # the existing class semantics (rank 0 confident → narrow, rank 2
        # uncertain → broad) is preserved.

        # Slot 0: adaptive_slot0 dispatch.
        p0 = p_t[0:1]  # [1]
        bw_t = consts['bw_t']
        if adaptive_slot0 == 0:
            slot0_bw = bw_t.unsqueeze(0)  # [1]
        elif adaptive_slot0 == 1:
            slot0_bw = torch.where(p0 >= 0.9, consts['min3_bw_t'].expand_as(p0),
                        torch.where(p0 >= 0.8, consts['min5_bw_t'].expand_as(p0),
                         torch.where(p0 >= 0.6, consts['min8_bw_t'].expand_as(p0),
                          bw_t.expand_as(p0))))
        elif adaptive_slot0 == 2:
            slot0_bw = torch.where(p0 >= 0.95, consts['min2_bw_t'].expand_as(p0),
                        torch.where(p0 >= 0.85, consts['min4_bw_t'].expand_as(p0),
                         torch.where(p0 >= 0.6, consts['min7_bw_t'].expand_as(p0),
                          bw_t.expand_as(p0))))
        elif adaptive_slot0 == 3:
            slot0_bw = torch.where(p0 >= 0.95, consts['one_t'].expand_as(p0),
                        torch.where(p0 >= 0.9, consts['min2_bw_t'].expand_as(p0),
                         torch.where(p0 >= 0.8, consts['min4_bw_t'].expand_as(p0),
                          torch.where(p0 >= 0.6, consts['min7_bw_t'].expand_as(p0),
                           bw_t.expand_as(p0)))))
        else:  # 4: ultra
            slot0_bw = torch.where(p0 >= 0.9, consts['one_t'].expand_as(p0),
                        torch.where(p0 >= 0.8, consts['min3_bw_t'].expand_as(p0),
                         torch.where(p0 >= 0.6, consts['min6_bw_t'].expand_as(p0),
                          bw_t.expand_as(p0))))

        # Slot k>0: adaptive_all branching.
        if adaptive_all == 0:
            slot_k_topk = base_topk
        elif adaptive_all == 1:
            slot_k_topk = torch.where(p_t >= 0.95, torch.minimum(base_topk, two_t),
                           torch.where(p_t >= 0.85, torch.minimum(base_topk, three_t),
                            base_topk))
        else:  # 2: aggressive
            one_t = consts['one_t']
            slot_k_topk = torch.where(p_t >= 0.9, one_t.expand_as(base_topk),
                           torch.where(p_t >= 0.7, torch.minimum(base_topk, two_t),
                            torch.where(p_t >= 0.5, torch.minimum(base_topk, three_t),
                             base_topk)))

        slot_topks = torch.where(arange_K == 0, slot0_bw.expand(K), slot_k_topk)  # [K]
        # Give-up-class topk>0 is required by _tree_gpu_supported → no truncation.

        n_alt_per_slot = (slot_topks - 1).clamp(min=0)  # [K]
        cum_alt_incl = n_alt_per_slot.cumsum(0)  # [K]
        cum_alt_excl = cum_alt_incl - n_alt_per_slot  # [K]

        # Pending plan: active slots (slot_topks>1), alternatives, hitchhike g(K-1).
        active_mask = slot_topks > 1  # [K]
        n_active_t = active_mask.sum()  # scalar
        total_alts1_t = cum_alt_incl[K - 1] if K > 0 else torch.zeros((), dtype=torch.long, device=device)
        # hitchhike: add g(K-1) iff not give_up AND (K-1) not active.
        hitch_add_t = ((~gu_t[0]) & (~active_mask[K - 1])).long()
        N_pend_t = n_active_t + total_alts1_t + hitch_add_t

        # Single sync: fetch sizes needed to shape subsequent kernels.
        size_pack = torch.stack([total_alts1_t, n_active_t, hitch_add_t, N_pend_t]).cpu()
        total_alts1 = int(size_pack[0])
        n_active = int(size_pack[1])
        hitch_add = int(size_pack[2])
        N_pend = int(size_pack[3])

        # ---- Write block 1 greedy chain (K nodes) ----
        buf['tokens'][0:K] = greedy_target
        buf['parents'][0:K] = arange_K - 1  # [-1, 0, 1, ..., K-2]
        buf['lps'][0:K] = greedy_lps.cumsum(0)
        buf['ranks'][0:K] = rank_preds
        buf['blocks'][0:K] = 0
        buf['slots'][0:K] = arange_K
        n_nodes = K

        # ---- Write block 1 alternatives ----
        slot_expanded1 = None  # closed over if total_alts1>0 for pending build
        alt_within1 = None
        if total_alts1 > 0:
            slot_expanded1 = arange_K.repeat_interleave(n_alt_per_slot)  # [total_alts1]
            flat_idx1 = torch.arange(total_alts1, device=device)
            alt_within1 = flat_idx1 - cum_alt_excl[slot_expanded1] + 1
            parents_alts = torch.where(
                slot_expanded1 > 0, slot_expanded1 - 1,
                torch.full_like(slot_expanded1, -1),
            )
            parent_cum_lp = torch.where(
                slot_expanded1 > 0,
                buf['lps'][(slot_expanded1 - 1).clamp(min=0)],
                torch.zeros_like(slot_expanded1, dtype=torch.float32),
            )
            alt_lp = all_top_lps[slot_expanded1, alt_within1]
            buf['tokens'][K:K + total_alts1] = all_top_target[slot_expanded1, alt_within1]
            buf['parents'][K:K + total_alts1] = parents_alts
            buf['lps'][K:K + total_alts1] = parent_cum_lp + alt_lp
            buf['ranks'][K:K + total_alts1] = rank_preds[slot_expanded1]
            buf['blocks'][K:K + total_alts1] = 0
            buf['slots'][K:K + total_alts1] = slot_expanded1
            n_nodes += total_alts1

        # ============================================================
        #   Block 1 pending construction (GPU)
        # ============================================================
        if N_pend == 0:
            # Degenerate: nothing to BFS. Fall through to prune/triton with block 1 only.
            return self._finalize_gpu_tree(
                n_nodes, buf, sample_token, device, K, [], draft_position,
            )

        active_slot_idx = active_mask.nonzero(as_tuple=True)[0]  # [n_active]

        pend_hidden_slots = torch.empty(N_pend, dtype=torch.long, device=device)
        pend_input_ids_t = torch.empty(N_pend, dtype=torch.long, device=device)
        pend_ttt_valid = torch.empty(N_pend, dtype=torch.long, device=device)
        pend_node_indices_t = torch.empty(N_pend, dtype=torch.long, device=device)
        pend_cum_lps_t = torch.empty(N_pend, dtype=torch.float32, device=device)

        # Active slots (g(slot) pending entries)
        pend_hidden_slots[:n_active] = active_slot_idx
        pend_input_ids_t[:n_active] = greedy_target[active_slot_idx]
        pend_ttt_valid[:n_active] = active_slot_idx + 1
        pend_node_indices_t[:n_active] = active_slot_idx
        pend_cum_lps_t[:n_active] = buf['lps'][active_slot_idx]

        # Alternatives
        if total_alts1 > 0:
            alt_start = n_active
            alt_end = alt_start + total_alts1
            alt_tree_idx = torch.arange(K, K + total_alts1, device=device)
            pend_hidden_slots[alt_start:alt_end] = slot_expanded1
            pend_input_ids_t[alt_start:alt_end] = all_top_target[slot_expanded1, alt_within1]
            pend_ttt_valid[alt_start:alt_end] = slot_expanded1 + 1
            pend_node_indices_t[alt_start:alt_end] = alt_tree_idx
            pend_cum_lps_t[alt_start:alt_end] = buf['lps'][alt_tree_idx]

        # Hitchhike g(K-1)
        if hitch_add:
            pos = n_active + total_alts1
            pend_hidden_slots[pos] = K - 1
            pend_input_ids_t[pos] = greedy_target[K - 1]
            pend_ttt_valid[pos] = K
            pend_node_indices_t[pos] = K - 1
            pend_cum_lps_t[pos] = buf['lps'][K - 1]

        # Build pend_hidden, batch_ttt_mask, batch_ttt_kv
        pend_hidden = draft_hidden[0, pend_hidden_slots, :].unsqueeze(1)  # [N_pend, 1, H]
        pend_input_ids = pend_input_ids_t.unsqueeze(1)  # [N_pend, 1]
        batch_ttt_mask = arange_K.unsqueeze(0) < pend_ttt_valid.unsqueeze(1)  # [N_pend, K]
        batch_ttt_kv = [
            (ttt_kv[l][0].expand(N_pend, -1, -1, -1),
             ttt_kv[l][1].expand(N_pend, -1, -1, -1))
            for l in range(num_layers)
        ]

        # ============================================================
        #   Block 2 (BFS) forward
        # ============================================================
        _model_K = self.K
        effective_cross_count = draft_cache[0][2] - _model_K if draft_cache[0][2] >= _model_K else 0
        batch_cross_cache = []
        cross_cache_slices = []
        for layer_cache in draft_cache:
            if effective_cross_count > 0 and layer_cache[0] is not None:
                k_view = layer_cache[0][:, :, :effective_cross_count, :]
                v_view = layer_cache[1][:, :, :effective_cross_count, :]
                batch_cross_cache.append([
                    k_view.expand(N_pend, -1, -1, -1),
                    v_view.expand(N_pend, -1, -1, -1),
                    effective_cross_count,
                ])
                cross_cache_slices.append((k_view, v_view))
            else:
                batch_cross_cache.append([None, None, 0])
                cross_cache_slices.append(None)

        if effective_cross_count > 0:
            _cross_ones = batch_ttt_mask.new_ones(N_pend, effective_cross_count)
            _full_kv_mask = torch.cat([_cross_ones, batch_ttt_mask], dim=1)
        else:
            _full_kv_mask = batch_ttt_mask

        _bfs_idx = len(self._block_forward_events[1])
        blk_start, blk_end = self._get_bfs_event_pair(1, _bfs_idx)
        blk_start.record()

        _graph_outputs = None
        if self._draft_graph_cache is not None:
            try:
                _graph_outputs = self._draft_graph_cache.run(
                    hidden=pend_hidden,
                    input_ids=pend_input_ids,
                    cross_cache_slices=cross_cache_slices,
                    effective_cross_count=effective_cross_count,
                    ttt_cache=batch_ttt_kv,
                    ttt_mask=batch_ttt_mask,
                    position_id=draft_position,
                )
            except Exception:
                _graph_outputs = None

        if _graph_outputs is not None:
            block_logits, block_rank_logits, block_draft_hidden, new_ttt_kv_batch = _graph_outputs
        else:
            block_logits, block_rank_logits, block_draft_hidden, new_ttt_kv_batch = \
                self.draft_model.forward_with_cache(
                    hidden=pend_hidden,
                    input_ids=pend_input_ids,
                    cache=batch_cross_cache,
                    position_id=draft_position,
                    use_draft_condition=True,
                    ttt_cache=batch_ttt_kv,
                    ttt_mask=batch_ttt_mask,
                    update_cross_cache=False,
                    full_kv_mask=_full_kv_mask,
                )

        blk_end.record()
        self._block_forward_events[1].append((blk_start, blk_end))

        # ============================================================
        #   BFS GPU branching (single iter, no pending needed)
        # ============================================================
        d2t_offsets = self.draft_model.d2t
        (all_rank_preds, all_greedy_tokens, all_greedy_target, all_greedy_lps,
         M_all, bf_all, _gu_bfs, top_target_all, top_lps_all) = _bfs_gpu_ops_fused(
            block_logits, block_rank_logits, d2t_offsets, 10,
            rank_classes, self._rank_to_factor,
        )
        # Shapes: [N_pend, K], [N_pend, K, 10]

        # Vectorized greedy chain: N_pend * K nodes
        tree_start = n_nodes
        NK = N_pend * K
        buf['tokens'][tree_start:tree_start + NK] = all_greedy_target.reshape(-1)
        buf['ranks'][tree_start:tree_start + NK] = all_rank_preds.reshape(-1)
        buf['blocks'][tree_start:tree_start + NK] = 1
        slot_expand = arange_K.unsqueeze(0).expand(N_pend, -1).reshape(-1)
        buf['slots'][tree_start:tree_start + NK] = slot_expand

        # Parents: slot 0 → pend_node_indices[leaf]; slot k>0 → tree_start + leaf*K + k - 1
        leaf_idx = (torch.arange(NK, device=device) // K)  # [NK]
        slot_idx_in_chain = torch.arange(NK, device=device) - leaf_idx * K  # [NK]
        local_base = tree_start + torch.arange(NK, device=device)  # [NK] absolute
        chain_parents = torch.where(
            slot_idx_in_chain == 0,
            pend_node_indices_t[leaf_idx],
            local_base - 1,
        )
        buf['parents'][tree_start:tree_start + NK] = chain_parents

        cum_greedy_lps = all_greedy_lps.cumsum(dim=1)  # [N_pend, K]
        buf['lps'][tree_start:tree_start + NK] = (
            pend_cum_lps_t.unsqueeze(1) + cum_greedy_lps
        ).reshape(-1)
        n_nodes += NK

        # Multi-slot branching alternatives (adaptive_all same rule)
        if adaptive_all == 0:
            bfs_slot_topks = rank_slot_topk_t[all_rank_preds]  # [N_pend, K]
        elif adaptive_all == 1:
            _pk_bfs = all_greedy_lps.exp()
            base_bfs = rank_slot_topk_t[all_rank_preds]
            bfs_slot_topks = torch.where(_pk_bfs >= 0.95, torch.minimum(base_bfs, two_t),
                              torch.where(_pk_bfs >= 0.85, torch.minimum(base_bfs, three_t),
                               base_bfs))
        else:  # 2
            _pk_bfs = all_greedy_lps.exp()
            base_bfs = rank_slot_topk_t[all_rank_preds]
            one_t = consts['one_t']
            bfs_slot_topks = torch.where(_pk_bfs >= 0.9, one_t.expand_as(base_bfs),
                              torch.where(_pk_bfs >= 0.7, torch.minimum(base_bfs, two_t),
                               torch.where(_pk_bfs >= 0.5, torch.minimum(base_bfs, three_t),
                                base_bfs)))

        # No give_up truncation (no_giveup/give_up-has-topk).
        n_alt_pair = (bfs_slot_topks - 1).clamp(min=0)  # [N_pend, K]
        n_alt_flat = n_alt_pair.reshape(-1)  # [N_pend*K]
        cum_alt_bfs = n_alt_flat.cumsum(0)
        cum_alt_bfs_excl = cum_alt_bfs - n_alt_flat
        total_alts2_t = cum_alt_bfs[-1]
        total_alts2 = int(total_alts2_t.item())  # sync 2

        if total_alts2 > 0:
            pair_expanded = torch.arange(N_pend * K, device=device).repeat_interleave(n_alt_flat)
            flat_idx2 = torch.arange(total_alts2, device=device)
            alt_within2 = flat_idx2 - cum_alt_bfs_excl[pair_expanded] + 1

            leaf_expanded = pair_expanded // K
            slot_expanded2 = pair_expanded - leaf_expanded * K

            base_for_leaf = tree_start + leaf_expanded * K
            parents_alts2 = torch.where(
                slot_expanded2 > 0,
                base_for_leaf + slot_expanded2 - 1,
                pend_node_indices_t[leaf_expanded],
            )
            parent_lps_alts2 = torch.where(
                slot_expanded2 > 0,
                buf['lps'][(base_for_leaf + slot_expanded2 - 1).clamp(min=0)],
                pend_cum_lps_t[leaf_expanded],
            )
            alt_lp_bfs = top_lps_all[leaf_expanded, slot_expanded2, alt_within2]

            s = n_nodes
            buf['tokens'][s:s + total_alts2] = top_target_all[leaf_expanded, slot_expanded2, alt_within2]
            buf['parents'][s:s + total_alts2] = parents_alts2
            buf['lps'][s:s + total_alts2] = parent_lps_alts2 + alt_lp_bfs
            buf['ranks'][s:s + total_alts2] = all_rank_preds[leaf_expanded, slot_expanded2]
            buf['blocks'][s:s + total_alts2] = 1
            buf['slots'][s:s + total_alts2] = slot_expanded2
            n_nodes += total_alts2

        # Stub rank_stats (downstream only reads for stats/coverage)
        all_rank_stats = [
            {'rank_preds': [0] * K, 'M': K, 'branch': 0, 'parent_block': -1}
        ]
        for _i in range(N_pend):
            all_rank_stats.append(
                {'rank_preds': [0] * K, 'M': K, 'branch': 0, 'parent_block': 0}
            )

        return self._finalize_gpu_tree(
            n_nodes, buf, sample_token, device, K, all_rank_stats, draft_position,
        )

    def _finalize_gpu_tree(self, n_nodes, buf, sample_token, device, K,
                           all_rank_stats, draft_position, tree_budget=None):
        """Final GPU→CPU transfer + numpy prune + triton topology.

        Single-sync transfer: both the int64 stack and float32 lps copy into
        pinned host buffers via non_blocking copies, then one explicit
        synchronize waits on both. Saves ~1 sync per iter (~200-500µs) vs
        the naive double .cpu() pattern.
        """
        # GPU-resident path: prune + topology all on GPU. Skips the pinned
        # transfer + numpy prune entirely.
        tree_budget = self.total_tokens if tree_budget is None else int(tree_budget)
        if getattr(self, '_tree_finalize_cuda', False):
            return self._finalize_gpu_tree_v2(
                n_nodes, buf, sample_token, device, K,
                all_rank_stats, draft_position, tree_budget=tree_budget,
            )
        max_nodes_np = max(tree_budget + 500, n_nodes)
        tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots = \
            self._ensure_tree_buffers(max_nodes_np)

        # Lazy-allocate pinned host buffers for non-blocking transfers.
        pin = getattr(self, '_finalize_pin_buf', None)
        if pin is None or pin['int'].shape[1] < max_nodes_np:
            pin = {
                'int': torch.empty(5, max_nodes_np, dtype=torch.long, pin_memory=True),
                'lps': torch.empty(max_nodes_np, dtype=torch.float32, pin_memory=True),
            }
            self._finalize_pin_buf = pin

        # Stage GPU stack into a contiguous tensor, then non-blocking copy
        # to the pre-pinned host buffer; same for lps. One sync at the end.
        stacked_gpu = torch.stack([
            buf['tokens'][:n_nodes],
            buf['parents'][:n_nodes],
            buf['ranks'][:n_nodes],
            buf['blocks'][:n_nodes],
            buf['slots'][:n_nodes],
        ])  # [5, n_nodes] i64 GPU
        pin['int'][:, :n_nodes].copy_(stacked_gpu, non_blocking=True)
        pin['lps'][:n_nodes].copy_(buf['lps'][:n_nodes], non_blocking=True)
        torch.cuda.synchronize()  # one sync for both transfers

        stacked = pin['int'][:, :n_nodes].numpy()      # zero-copy view of pinned buf
        lps_cpu = pin['lps'][:n_nodes].numpy()         # zero-copy view of pinned buf

        tn_tokens[:n_nodes] = stacked[0]
        tn_parents[:n_nodes] = stacked[1]
        tn_ranks[:n_nodes] = stacked[2]
        tn_blocks[:n_nodes] = stacked[3]
        tn_slots[:n_nodes] = stacked[4]
        tn_lps[:n_nodes] = lps_cpu.astype(np.float64)

        # Prune if over budget
        if n_nodes > tree_budget:
            n_nodes = self._prune_tree_np(
                n_nodes, tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots,
                tree_budget,
            )

        tree_tokens, tree_mask, tree_position_ids, retrieve_indices = \
            build_tree_buffers_triton(
                n_nodes, tn_tokens, tn_parents, sample_token, device, K, self.max_blocks,
            )

        node_ranks = tn_ranks[:n_nodes].tolist()
        node_block_slots = list(
            zip(tn_blocks[:n_nodes].tolist(), tn_slots[:n_nodes].tolist())
        )

        return (tree_tokens, tree_mask, tree_position_ids, retrieve_indices,
                all_rank_stats, node_ranks, node_block_slots)

    def _finalize_gpu_tree_v2(self, n_nodes, buf, sample_token, device, K,
                              all_rank_stats, draft_position, tree_budget=None):
        """GPU-resident finalize: prune + topology, ~0 numpy time.

        Drop-in replacement for _finalize_gpu_tree under TREE_FINALIZE_CUDA=1.
        Returns the same 7-tuple (draft_tokens, tree_mask, tree_position_ids,
        retrieve_indices, all_rank_stats, node_ranks, node_block_slots).
        """
        from .tree_finalize_triton import finalize_tree_gpu

        tree_budget = self.total_tokens if tree_budget is None else int(tree_budget)
        (draft_tokens, tree_mask, tree_position_ids, retrieve_indices,
         node_ranks, node_block_slots, new_n) = finalize_tree_gpu(
            buf, n_nodes, sample_token, tree_budget, K, self.max_blocks,
            _tree_depth_mask_kernel, _tree_retrieve_kernel,
            max_tree_depth=MAX_TREE_DEPTH,
        )
        return (draft_tokens, tree_mask, tree_position_ids, retrieve_indices,
                all_rank_stats, node_ranks, node_block_slots)

    # ============================================================
    #    TREE_BUILD_TRITON=1 path — mega-kernel tree building
    # ============================================================
    @torch.no_grad()
    def _build_tree_from_block1_triton(
        self,
        logits: torch.Tensor,          # [1, K, draft_vocab]
        rank_logits: torch.Tensor,     # [1, K, rank_classes]
        draft_hidden: torch.Tensor,    # [1, K, H]
        ttt_kv: list,
        input_id: torch.Tensor,        # [1, 1]
        draft_cache: list,
        draft_position: int,
        temperature: float = 0.0,
    ):
        """Mega-kernel tree builder.

        Two triton kernel launches total:
          1) block1_mega_kernel: per-slot branching + greedy chain + alts + pending
          2) bfs_mega_kernel:   per-leaf greedy chain + alts (single BFS iter)
        Two host syncs total: one after each kernel for size readback.
        """
        from .tree_build_triton import triton_build_block1, triton_build_bfs

        K = self.K
        device = logits.device
        num_layers = len(ttt_kv)
        beam_width = self.beam_width
        rank_classes = self.rank_classes
        give_up_class = rank_classes - 1
        adaptive_slot0 = self._adaptive_slot0
        adaptive_all = self._adaptive_all
        max_topk = max(beam_width, max(self.RANK_SLOT_TOPK))

        # -----------------------------------------------------------------
        # Lazy-init GPU buffers (shared with TREE_GPU path + pend scratch)
        # -----------------------------------------------------------------
        max_block1_nodes = K + K * (max_topk - 1) + 1
        max_block2_nodes = max_block1_nodes * K * max_topk
        max_nodes = max(self.total_tokens + 200, max_block1_nodes + max_block2_nodes + 100)
        buf = getattr(self, '_tree_gpu_buf', None)
        if buf is None or buf['tokens'].shape[0] < max_nodes:
            buf = {
                'tokens':  torch.empty(max_nodes, dtype=torch.long, device=device),
                'parents': torch.empty(max_nodes, dtype=torch.long, device=device),
                'lps':     torch.empty(max_nodes, dtype=torch.float32, device=device),
                'ranks':   torch.empty(max_nodes, dtype=torch.long, device=device),
                'blocks':  torch.empty(max_nodes, dtype=torch.long, device=device),
                'slots':   torch.empty(max_nodes, dtype=torch.long, device=device),
            }
            self._tree_gpu_buf = buf
        pend_max = max_block1_nodes
        pend_buf = getattr(self, '_triton_pend_buf', None)
        if pend_buf is None or pend_buf['hidden_slots'].shape[0] < pend_max:
            pend_buf = {
                'hidden_slots': torch.empty(pend_max, dtype=torch.long, device=device),
                'input_ids':    torch.empty(pend_max, dtype=torch.long, device=device),
                'ttt_valid':    torch.empty(pend_max, dtype=torch.long, device=device),
                'node_indices': torch.empty(pend_max, dtype=torch.long, device=device),
                'cum_lps':      torch.empty(pend_max, dtype=torch.float32, device=device),
            }
            self._triton_pend_buf = pend_buf
        sizes4 = getattr(self, '_triton_sizes4', None)
        if sizes4 is None:
            sizes4 = torch.empty(4, dtype=torch.long, device=device)
            self._triton_sizes4 = sizes4
        sizes1 = getattr(self, '_triton_sizes1', None)
        if sizes1 is None:
            sizes1 = torch.empty(1, dtype=torch.long, device=device)
            self._triton_sizes1 = sizes1
        # Scratch for BFS sizing kernel to prefix-sum per-leaf n_alt for scatter.
        bfs_block_n = 32  # BLOCK_N constant used below
        cum_alt_buf = getattr(self, '_triton_cum_alt_buf', None)
        if cum_alt_buf is None or cum_alt_buf.shape[0] < bfs_block_n:
            cum_alt_buf = torch.empty(bfs_block_n, dtype=torch.long, device=device)
            self._triton_cum_alt_buf = cum_alt_buf
        # Persistent rank_slot_topk device tensor
        rank_slot_topk_table = getattr(self, '_triton_rank_slot_topk_t', None)
        if rank_slot_topk_table is None:
            rank_slot_topk_table = torch.tensor(
                self.RANK_SLOT_TOPK, dtype=torch.long, device=device,
            )
            self._triton_rank_slot_topk_t = rank_slot_topk_table

        sample_token = input_id[:, -1]

        # =================================================================
        #   Block-1 GPU precompute — match numpy baseline bit-for-bit
        #   (log_softmax path, not logsumexp-fused). Prune ordering depends
        #   on tree_lps at sub-bit scale; divergence here shows up as acc
        #   drift in mtbench even when tree topology is identical.
        # =================================================================
        # Outputs of these ops are C-contiguous (squeeze/argmax/gather slot 0
        # of last-dim contiguous source); skip redundant .contiguous() calls.
        _nv_push("b1_gpu_precompute")
        d2t_offsets = self.draft_model.d2t
        last_p = logits[0]                                              # [K, V_d]
        log_probs = F.log_softmax(last_p.float(), dim=-1)               # [K, V]
        rank_preds_1d = rank_logits[0].argmax(dim=-1)                   # [K]
        top_idx_2d = torch.topk(last_p, max_topk, dim=-1).indices       # [K, max_topk]
        all_top_target_2d = self._map_draft_to_target(top_idx_2d)
        all_top_lps_2d = log_probs.gather(1, top_idx_2d)                # [K, max_topk]
        greedy_tgt_1d = all_top_target_2d[:, 0].contiguous()            # [:, 0] is strided
        greedy_lps_1d = all_top_lps_2d[:, 0].contiguous()               # [:, 0] is strided
        _nv_pop()

        # =================================================================
        #   KERNEL 1 launch + sync for sizes
        # =================================================================
        _nv_push("b1_kernel+sync")
        n_nodes_b1, n_active, total_alts1, N_pend = triton_build_block1(
            rank_preds_1d, greedy_tgt_1d, greedy_lps_1d,
            all_top_target_2d, all_top_lps_2d,
            rank_slot_topk_table, buf, pend_buf, sizes4,
            beam_width, K, max_topk, rank_classes, give_up_class,
            adaptive_slot0, adaptive_all,
        )
        _nv_pop()
        all_rank_stats_stub = [{'rank_preds': [0] * K, 'M': K,
                                'branch': 0, 'parent_block': -1}]

        if N_pend == 0:
            # Degenerate — no BFS iter. Finalize block-1 tree only.
            return self._finalize_gpu_tree(
                n_nodes_b1, buf, sample_token, device, K,
                all_rank_stats_stub, draft_position,
            )

        # =================================================================
        #   Host: build pend_hidden / pend_input_ids / batch_ttt_mask /
        #   batch_ttt_kv / batch_cross_cache (all GPU ops, no extra syncs)
        # =================================================================
        _nv_push("b2_batch_prep")
        pend_slots = pend_buf['hidden_slots'][:N_pend]
        pend_input_ids = pend_buf['input_ids'][:N_pend].unsqueeze(1)   # [N_pend, 1]
        pend_ttt_valid_t = pend_buf['ttt_valid'][:N_pend]
        pend_node_indices_t = pend_buf['node_indices'][:N_pend]
        pend_cum_lps_t = pend_buf['cum_lps'][:N_pend]

        pend_hidden = draft_hidden[0, pend_slots, :].unsqueeze(1)       # [N_pend, 1, H]
        arange_K_t = self._arange_K_t
        batch_ttt_mask = arange_K_t.unsqueeze(0) < pend_ttt_valid_t.unsqueeze(1)

        _model_K = self.K
        effective_cross_count = draft_cache[0][2] - _model_K if draft_cache[0][2] >= _model_K else 0
        batch_cross_cache = []
        for layer_cache in draft_cache:
            if effective_cross_count > 0 and layer_cache[0] is not None:
                k_view = layer_cache[0][:, :, :effective_cross_count, :]
                v_view = layer_cache[1][:, :, :effective_cross_count, :]
                batch_cross_cache.append([
                    k_view.expand(N_pend, -1, -1, -1),
                    v_view.expand(N_pend, -1, -1, -1),
                    effective_cross_count,
                ])
            else:
                batch_cross_cache.append([None, None, 0])
        if effective_cross_count > 0:
            _cross_ones = batch_ttt_mask.new_ones(N_pend, effective_cross_count)
            _full_kv_mask = torch.cat([_cross_ones, batch_ttt_mask], dim=1)
        else:
            _full_kv_mask = batch_ttt_mask
        batch_ttt_kv = [
            (ttt_kv[l][0].expand(N_pend, -1, -1, -1),
             ttt_kv[l][1].expand(N_pend, -1, -1, -1))
            for l in range(num_layers)
        ]
        _nv_pop()

        # =================================================================
        #   Block-2 forward (try CUDA graph BFS first, fallback to eager)
        # =================================================================
        _nv_push("b2_forward")
        if self._bfs_events_enabled:
            _bfs_idx = len(self._block_forward_events[1])
            blk_start, blk_end = self._get_bfs_event_pair(1, _bfs_idx)
            blk_start.record()

        _graph_outputs = None
        if self._draft_graph_cache is not None:
            try:
                # Per-layer pre-expand (k_view, v_view) for graph; graph.run
                # handles expand/copy into bucket-sized static buffers.
                cross_cache_slices_for_graph = []
                for layer_cache in draft_cache:
                    if effective_cross_count > 0 and layer_cache[0] is not None:
                        cross_cache_slices_for_graph.append(
                            (layer_cache[0][:, :, :effective_cross_count, :],
                             layer_cache[1][:, :, :effective_cross_count, :])
                        )
                    else:
                        cross_cache_slices_for_graph.append(None)
                _graph_outputs = self._draft_graph_cache.run(
                    hidden=pend_hidden,
                    input_ids=pend_input_ids,
                    cross_cache_slices=cross_cache_slices_for_graph,
                    effective_cross_count=effective_cross_count,
                    ttt_cache=batch_ttt_kv,
                    ttt_mask=batch_ttt_mask,
                    position_id=draft_position,
                )
            except Exception:
                _graph_outputs = None

        if _graph_outputs is not None:
            block_logits, block_rank_logits, _block_draft_hidden, _new_ttt_kv_batch = _graph_outputs
        else:
            block_logits, block_rank_logits, _block_draft_hidden, _new_ttt_kv_batch = \
                self.draft_model.forward_with_cache(
                    hidden=pend_hidden,
                    input_ids=pend_input_ids,
                    cache=batch_cross_cache,
                    position_id=draft_position,
                    use_draft_condition=True,
                    ttt_cache=batch_ttt_kv,
                    ttt_mask=batch_ttt_mask,
                    update_cross_cache=False,
                    full_kv_mask=_full_kv_mask,
                )
        if self._bfs_events_enabled:
            blk_end.record()
            self._block_forward_events[1].append((blk_start, blk_end))
        _nv_pop()

        # Online adapt (2026-04-25): cache block-2 BFS forward outputs so the
        # hook layer can extract per-(leaf, slot) draft logits/hidden when a
        # tree-node reject lands in block_idx >= 1. Zero-overhead when adapt
        # is not enabled (getattr default False); spec-block-shift inference
        # never sets this attr → unaffected.
        # pend_node_indices_t is the tree-node id of each pending leaf (the
        # block-0/1 region node from which BFS expanded). hooks use it to
        # invert the (block-2 tree node → parent leaf id → leaf_idx → block_logits row).
        if getattr(self, '_adapt_collect_block2', False):
            self._adapt_block2_logits = block_logits.detach()
            self._adapt_block2_hidden = _block_draft_hidden.detach()
            self._adapt_block2_pend_node_indices = pend_node_indices_t.detach().clone()

        # =================================================================
        #   Block-2 GPU precompute via _bfs_gpu_ops_fused
        # =================================================================
        _nv_push("b2_gpu_precompute")
        (all_rank_preds_2, _all_greedy_tokens_2, all_greedy_target_2,
         all_greedy_lps_2, _M2, _bf2, _gu2,
         top_target_all_2, top_lps_all_2) = _bfs_gpu_ops_fused(
            block_logits, block_rank_logits, d2t_offsets, max_topk,
            rank_classes, self._rank_to_factor,
        )
        _nv_pop()

        # =================================================================
        #   KERNEL 2 launch + sync for total_alts_2
        # =================================================================
        _nv_push("b2_kernel+sync")
        # BLOCK_N=32 is the sizing kernel tile upper bound (match _triton_cum_alt_buf).
        # N_pend typically ≤ 15 (= beam_width+5 warmup bound). If exceeds 32,
        # grow buffer + recompile sizing kernel once.
        if N_pend > bfs_block_n:
            bfs_block_n = max(64, 1 << (N_pend - 1).bit_length())
            cum_alt_buf = torch.empty(bfs_block_n, dtype=torch.long, device=device)
            self._triton_cum_alt_buf = cum_alt_buf
        # _bfs_gpu_ops_fused returns C-contiguous tensors (.view at the end);
        # skip .contiguous() Python dispatch overhead (~6 calls × ~50µs).
        total_alts_2 = triton_build_bfs(
            all_rank_preds_2,
            all_greedy_target_2,
            all_greedy_lps_2,
            top_target_all_2,
            top_lps_all_2,
            pend_cum_lps_t, pend_node_indices_t,
            rank_slot_topk_table,
            buf, sizes1, cum_alt_buf,
            tree_start=n_nodes_b1,
            N=N_pend, K=K, max_topk=max_topk, rank_classes=rank_classes,
            give_up_class=give_up_class, adaptive_all=adaptive_all,
            pend_depth=1, block_n=bfs_block_n, j_pad=16,
        )
        n_nodes_final = n_nodes_b1 + N_pend * K + total_alts_2
        _nv_pop()

        # Stub per-leaf rank_stats (downstream stats consumer only)
        all_rank_stats = list(all_rank_stats_stub)
        for _i in range(N_pend):
            all_rank_stats.append(
                {'rank_preds': [0] * K, 'M': K, 'branch': 0, 'parent_block': 0}
            )

        _nv_push("finalize")
        _result = self._finalize_gpu_tree(
            n_nodes_final, buf, sample_token, device, K,
            all_rank_stats, draft_position,
        )
        _nv_pop()
        return _result

    def _get_batched_triton_scratch(
        self,
        row: int,
        device: torch.device,
        tree_budget: int,
        max_topk: int,
    ):
        """Return persistent request-owned scratch for the batched Triton tree path."""
        K = self.K
        max_block1_nodes = K + K * (max_topk - 1) + 1
        max_block2_nodes = max_block1_nodes * K * max_topk
        max_nodes = max(
            int(tree_budget) + 200,
            max_block1_nodes + max_block2_nodes + 100,
        )
        pool = getattr(self, "_batched_triton_scratch", None)
        if pool is None:
            pool = []
            self._batched_triton_scratch = pool
        while len(pool) <= row:
            pool.append(None)
        scratch = pool[row]
        if (
            scratch is None
            or scratch["device"] != device
            or scratch["buf"]["tokens"].shape[0] < max_nodes
            or scratch["pend_buf"]["hidden_slots"].shape[0] < max_block1_nodes
        ):
            scratch = {
                "device": device,
                "buf": {
                    "tokens": torch.empty(max_nodes, dtype=torch.long, device=device),
                    "parents": torch.empty(max_nodes, dtype=torch.long, device=device),
                    "lps": torch.empty(max_nodes, dtype=torch.float32, device=device),
                    "ranks": torch.empty(max_nodes, dtype=torch.long, device=device),
                    "blocks": torch.empty(max_nodes, dtype=torch.long, device=device),
                    "slots": torch.empty(max_nodes, dtype=torch.long, device=device),
                },
                "pend_buf": {
                    "hidden_slots": torch.empty(max_block1_nodes, dtype=torch.long, device=device),
                    "input_ids": torch.empty(max_block1_nodes, dtype=torch.long, device=device),
                    "ttt_valid": torch.empty(max_block1_nodes, dtype=torch.long, device=device),
                    "node_indices": torch.empty(max_block1_nodes, dtype=torch.long, device=device),
                    "cum_lps": torch.empty(max_block1_nodes, dtype=torch.float32, device=device),
                },
                "sizes4": torch.empty(4, dtype=torch.long, device=device),
                "sizes1": torch.empty(1, dtype=torch.long, device=device),
                "cum_alt_buf": torch.empty(32, dtype=torch.long, device=device),
            }
            pool[row] = scratch
        return scratch

    def _prepare_tree_from_block1_triton_batched(
        self,
        logits,
        rank_logits,
        draft_hidden,
        ttt_kv,
        input_id,
        draft_position: int,
        tree_budget: int,
        scratch_row: int,
    ):
        """Prepare one request through block-1 while retaining request-owned scratch."""
        from .tree_build_triton import triton_launch_block1

        K = self.K
        device = logits.device
        rank_classes = self.rank_classes
        give_up_class = rank_classes - 1
        max_topk = max(self.beam_width, max(self.RANK_SLOT_TOPK))
        scratch = self._get_batched_triton_scratch(
            scratch_row, device, tree_budget, max_topk,
        )
        buf = scratch["buf"]
        pend_buf = scratch["pend_buf"]
        rank_slot_topk_table = getattr(self, "_triton_rank_slot_topk_t", None)
        if rank_slot_topk_table is None or rank_slot_topk_table.device != device:
            rank_slot_topk_table = torch.tensor(
                self.RANK_SLOT_TOPK, dtype=torch.long, device=device,
            )
            self._triton_rank_slot_topk_t = rank_slot_topk_table

        last_p = logits[0]
        log_probs = F.log_softmax(last_p.float(), dim=-1)
        rank_preds = rank_logits[0].argmax(dim=-1)
        top_idx = torch.topk(last_p, max_topk, dim=-1).indices
        top_target = self._map_draft_to_target(top_idx)
        top_lps = log_probs.gather(1, top_idx)
        greedy_target = top_target[:, 0].contiguous()
        greedy_lps = top_lps[:, 0].contiguous()
        triton_launch_block1(
            rank_preds,
            greedy_target,
            greedy_lps,
            top_target,
            top_lps,
            rank_slot_topk_table,
            buf,
            pend_buf,
            scratch["sizes4"],
            self.beam_width,
            K,
            max_topk,
            rank_classes,
            give_up_class,
            self._adaptive_slot0,
            self._adaptive_all,
        )
        return {
            "row": scratch_row,
            "tree_budget": int(tree_budget),
            "draft_position": int(draft_position),
            "sample_token": input_id[:, -1],
            "buf": buf,
            "pend_buf": pend_buf,
            "sizes4": scratch["sizes4"],
            "sizes1": scratch["sizes1"],
            "cum_alt_buf": scratch["cum_alt_buf"],
            "rank_stats": [{
                "rank_preds": [0] * K,
                "M": K,
                "branch": 0,
                "parent_block": -1,
            }],
            "rank_slot_topk": rank_slot_topk_table,
            "max_topk": max_topk,
            "ttt_kv": ttt_kv,
            "draft_hidden": draft_hidden,
        }

    def _consume_tree_block1_triton_batched(self, context, sizes):
        """Consume one request after the batch-wide block-1 size readback."""
        if len(sizes) != 4:
            raise ValueError("block-1 size metadata must contain four values")
        n_nodes_b1, _n_active, _total_alts1, n_pending = map(int, sizes)
        context["n_nodes_b1"] = n_nodes_b1
        context["n_pending"] = n_pending
        if n_pending == 0:
            context["result"] = self._finalize_gpu_tree(
                n_nodes_b1,
                context["buf"],
                context["sample_token"],
                context["sample_token"].device,
                self.K,
                context["rank_stats"],
                context["draft_position"],
                tree_budget=context["tree_budget"],
            )
            context.pop("draft_hidden", None)
            return context

        pend_buf = context["pend_buf"]
        pend_slots = pend_buf["hidden_slots"][:n_pending]
        draft_hidden = context.pop("draft_hidden")
        context.update({
            "pend_hidden": draft_hidden[0, pend_slots, :].unsqueeze(1),
            "pend_input_ids": pend_buf["input_ids"][:n_pending].unsqueeze(1),
            "ttt_mask": (
                self._arange_K_t.unsqueeze(0)
                < pend_buf["ttt_valid"][:n_pending].unsqueeze(1)
            ),
            "pend_node_indices": pend_buf["node_indices"][:n_pending],
            "pend_cum_lps": pend_buf["cum_lps"][:n_pending],
        })
        return context

    def _launch_tree_block2_triton_batched(
        self,
        context,
        rank_preds,
        greedy_target,
        greedy_lps,
        top_target,
        top_lps,
    ):
        """Launch one request's BFS kernels without reading size metadata."""
        from .tree_build_triton import triton_launch_bfs

        K = self.K
        n_pending = context["n_pending"]
        cum_alt_buf = context["cum_alt_buf"]
        block_n = int(cum_alt_buf.shape[0])
        if n_pending > block_n:
            block_n = max(64, 1 << (n_pending - 1).bit_length())
            cum_alt_buf = torch.empty(
                block_n, dtype=torch.long, device=rank_preds.device,
            )
            context["cum_alt_buf"] = cum_alt_buf
            self._batched_triton_scratch[context["row"]]["cum_alt_buf"] = cum_alt_buf
        triton_launch_bfs(
            rank_preds,
            greedy_target,
            greedy_lps,
            top_target,
            top_lps,
            context["pend_cum_lps"],
            context["pend_node_indices"],
            context["rank_slot_topk"],
            context["buf"],
            context["sizes1"],
            cum_alt_buf,
            tree_start=context["n_nodes_b1"],
            N=n_pending,
            K=K,
            max_topk=context["max_topk"],
            rank_classes=self.rank_classes,
            give_up_class=self.rank_classes - 1,
            adaptive_all=self._adaptive_all,
            pend_depth=1,
            block_n=block_n,
            j_pad=16,
        )

    def _finalize_tree_block2_triton_batched(self, context, total_alts):
        """Finalize one request after the batch-wide BFS size readback."""
        K = self.K
        n_pending = context["n_pending"]
        n_nodes = context["n_nodes_b1"] + n_pending * K + int(total_alts)
        rank_stats = list(context["rank_stats"])
        rank_stats.extend({
            "rank_preds": [0] * K,
            "M": K,
            "branch": 0,
            "parent_block": 0,
        } for _ in range(n_pending))
        return self._finalize_gpu_tree(
            n_nodes,
            context["buf"],
            context["sample_token"],
            context["sample_token"].device,
            K,
            rank_stats,
            context["draft_position"],
            tree_budget=context["tree_budget"],
        )

    @torch.no_grad()
    def _build_trees_from_block1_batched(
        self,
        logits,
        rank_logits,
        draft_hidden,
        ttt_kv,
        input_ids,
        draft_cache,
        draft_positions,
        tree_budgets,
        temperature: float = 0.0,
    ):
        """Build request-local trees around one global block-2 draft forward."""
        requests = logits.shape[0]
        if (
            rank_logits.shape[0] != requests
            or draft_hidden.shape[0] != requests
            or input_ids.shape != (requests, 1)
            or len(draft_positions) != requests
            or len(tree_budgets) != requests
        ):
            raise ValueError("batched block-1 inputs must have one row per request")
        if len(ttt_kv) != self.draft_model.num_layers:
            raise ValueError("batched block-1 TTT cache must have one entry per layer")
        if any(keys.shape[0] != requests for keys, _values in ttt_kv):
            raise ValueError("batched block-1 TTT cache rows must match requests")
        if self._strict_linear:
            linear_rank_logits = torch.full_like(
                rank_logits, torch.finfo(rank_logits.dtype).min
            )
            linear_rank_logits[..., 0] = 0
            rank_logits = linear_rank_logits

        if draft_cache.use_compiled_b1:
            if requests != 1:
                raise RuntimeError("original-engine B1 specialization requires one active request")
            original_budget = self.total_tokens
            self.total_tokens = int(tree_budgets[0])
            try:
                result = self._build_tree_from_block1_dispatch(
                    logits,
                    rank_logits,
                    draft_hidden,
                    ttt_kv,
                    input_ids,
                    draft_cache.row_view(0),
                    int(draft_positions[0]),
                    temperature=temperature,
                )
            finally:
                self.total_tokens = original_budget
            return [result]

        if self.max_blocks != 2 or not self._tree_gpu_supported():
            raise RuntimeError("batched SpecBlock trees require the supported two-block GPU config")
        if not self._tree_build_triton:
            raise RuntimeError("batched SpecBlock trees require the canonical Triton builder")
        if self._use_static_draft or self._tree_build_cuda or self._tree_fixed_n > 0:
            raise RuntimeError("static/CUDA scalar tree builders are unsupported for batched SpecBlock")
        if getattr(self, "_adapt_collect_block2", False):
            raise RuntimeError("online block-2 adaptation is unsupported in batched SpecBlock")

        launched_contexts = []
        results = [None] * requests
        for row in range(requests):
            launched_contexts.append(
                self._prepare_tree_from_block1_triton_batched(
                    logits[row:row + 1],
                    rank_logits[row:row + 1],
                    draft_hidden[row:row + 1],
                    [
                        (keys[row:row + 1], values[row:row + 1])
                        for keys, values in ttt_kv
                    ],
                    input_ids[row:row + 1],
                    int(draft_positions[row]),
                    int(tree_budgets[row]),
                    row,
                )
            )

        block1_sizes = torch.stack([
            context["sizes4"] for context in launched_contexts
        ]).cpu().tolist()
        self._batched_block1_size_readbacks = (
            getattr(self, "_batched_block1_size_readbacks", 0) + 1
        )
        contexts = []
        for context, sizes in zip(launched_contexts, block1_sizes):
            context = self._consume_tree_block1_triton_batched(context, sizes)
            if "result" in context:
                results[context["row"]] = context["result"]
            else:
                contexts.append(context)

        if contexts:
            K = self.K
            lengths_host = draft_cache.lengths_host
            if len(lengths_host) != requests or any(length < K for length in lengths_host):
                raise ValueError("shared draft cache lengths must include the current block-1 slots")
            context_rows = [int(context["row"]) for context in contexts]
            pending_counts = [int(context["n_pending"]) for context in contexts]
            if any(count <= 0 for count in pending_counts):
                raise ValueError("packed block-2 contexts must contain pending leaves")
            if any(lengths_host[row] <= K for row in context_rows):
                raise ValueError("packed block-2 attention requires non-empty cross cache rows")
            effective_lengths = draft_cache.lengths - K
            effective_cache = [
                [layer[0], layer[1], effective_lengths, layer[3], None]
                for layer in draft_cache.layers
            ]
            max_cross_count = max(
                int(lengths_host[row]) - K for row in context_rows
            )

            pending_hidden = torch.cat([
                context["pend_hidden"] for context in contexts
            ], dim=0).contiguous()
            pending_input_ids = torch.cat([
                context["pend_input_ids"] for context in contexts
            ], dim=0).contiguous()
            pending_ttt_mask = torch.cat([
                context["ttt_mask"] for context in contexts
            ], dim=0).contiguous()
            total_pending = sum(pending_counts)
            if pending_hidden.shape[0] != total_pending:
                raise RuntimeError("packed block-2 leaf count is inconsistent")
            self._batched_block2_packed_leaves = (
                getattr(self, "_batched_block2_packed_leaves", 0)
                + total_pending
            )
            self._batched_block2_padded_capacity = (
                getattr(self, "_batched_block2_padded_capacity", 0)
                + requests * max(pending_counts)
            )
            owner_rows = torch.tensor(
                context_rows,
                dtype=torch.long,
                device=logits.device,
            )
            owner_counts = torch.tensor(
                pending_counts,
                dtype=torch.long,
                device=logits.device,
            )
            leaf_owner = torch.repeat_interleave(
                owner_rows,
                owner_counts,
                output_size=total_pending,
            ).contiguous()
            base_positions = torch.as_tensor(
                draft_positions,
                dtype=torch.long,
                device=logits.device,
            )
            pos_ids = (
                base_positions.index_select(0, leaf_owner)[:, None]
                + 1
                + self._arange_K_t[None, :]
            )
            max_position = max(
                int(draft_positions[row]) + K for row in context_rows
            )
            request_ttt = ttt_kv

            if self._bfs_events_enabled:
                event_index = len(self._block_forward_events[1])
                block_start, block_end = self._get_bfs_event_pair(1, event_index)
                block_start.record()
            block_logits, block_rank_logits, _block_hidden, _block_ttt = \
                self.draft_model.forward_block2_ragged(
                    hidden=pending_hidden,
                    input_ids=pending_input_ids,
                    cache=effective_cache,
                    pos_ids=pos_ids,
                    max_position=max_position,
                    ttt_cache=request_ttt,
                    ttt_mask=pending_ttt_mask,
                    leaf_owner=leaf_owner,
                    max_cross_count=max_cross_count,
                )
            self._batched_block2_forward_calls = (
                getattr(self, "_batched_block2_forward_calls", 0) + 1
            )
            if self._bfs_events_enabled:
                block_end.record()
                self._block_forward_events[1].append((block_start, block_end))

            max_topk = max(self.beam_width, max(self.RANK_SLOT_TOPK))
            (rank_preds, _greedy_tokens, greedy_target, greedy_lps,
             _M, _branch_factors, _give_up, top_target, top_lps) = _bfs_gpu_ops_fused(
                block_logits,
                block_rank_logits,
                self.draft_model.d2t,
                max_topk,
                self.rank_classes,
                self._rank_to_factor,
            )
            offset = 0
            for context in contexts:
                end = offset + context["n_pending"]
                self._launch_tree_block2_triton_batched(
                    context,
                    rank_preds[offset:end],
                    greedy_target[offset:end],
                    greedy_lps[offset:end],
                    top_target[offset:end],
                    top_lps[offset:end],
                )
                offset = end

            bfs_sizes = torch.stack([
                context["sizes1"] for context in contexts
            ]).cpu().tolist()
            self._batched_bfs_size_readbacks = (
                getattr(self, "_batched_bfs_size_readbacks", 0) + 1
            )
            for context, total_alts in zip(contexts, bfs_sizes):
                results[context["row"]] = (
                    self._finalize_tree_block2_triton_batched(
                        context, total_alts[0]
                    )
                )

        if any(result is None for result in results):
            raise RuntimeError("batched SpecBlock tree builder did not finalize every request")
        if self._strict_linear:
            return [self._validate_strict_linear_tree(result) for result in results]
        return results

    # ============================================================
    #    TREE_BUILD_CUDA=1 path — native C++/CUDA mega-kernels
    # ============================================================
    @torch.no_grad()
    def _build_tree_from_block1_cuda(
        self,
        logits: torch.Tensor,
        rank_logits: torch.Tensor,
        draft_hidden: torch.Tensor,
        ttt_kv: list,
        input_id: torch.Tensor,
        draft_cache: list,
        draft_position: int,
        temperature: float = 0.0,
    ):
        """Native CUDA tree builder.

        Byte-identical to _build_tree_from_block1_triton; only difference is
        the kernel dispatcher (torch cpp_extension vs triton). The triton
        launcher carries ~80 µs of Python wrapper + JIT cache lookup overhead
        per call; a hand-written CUDA kernel goes straight through
        cudaLaunchKernel (~5-10 µs). Two host syncs remain (block-1 sizes
        readback, BFS total_alts readback) — same as triton.
        """
        from .tree_build_cuda_loader import cuda_build_block1, cuda_build_bfs

        K = self.K
        device = logits.device
        num_layers = len(ttt_kv)
        beam_width = self.beam_width
        rank_classes = self.rank_classes
        give_up_class = rank_classes - 1
        adaptive_slot0 = self._adaptive_slot0
        adaptive_all = self._adaptive_all
        max_topk = max(beam_width, max(self.RANK_SLOT_TOPK))

        # Lazy-init GPU buffers (share with triton path if already allocated).
        max_block1_nodes = K + K * (max_topk - 1) + 1
        max_block2_nodes = max_block1_nodes * K * max_topk
        max_nodes = max(self.total_tokens + 200, max_block1_nodes + max_block2_nodes + 100)
        buf = getattr(self, '_tree_gpu_buf', None)
        if buf is None or buf['tokens'].shape[0] < max_nodes:
            buf = {
                'tokens':  torch.empty(max_nodes, dtype=torch.long, device=device),
                'parents': torch.empty(max_nodes, dtype=torch.long, device=device),
                'lps':     torch.empty(max_nodes, dtype=torch.float32, device=device),
                'ranks':   torch.empty(max_nodes, dtype=torch.long, device=device),
                'blocks':  torch.empty(max_nodes, dtype=torch.long, device=device),
                'slots':   torch.empty(max_nodes, dtype=torch.long, device=device),
            }
            self._tree_gpu_buf = buf
        pend_max = max_block1_nodes
        pend_buf = getattr(self, '_triton_pend_buf', None)
        if pend_buf is None or pend_buf['hidden_slots'].shape[0] < pend_max:
            pend_buf = {
                'hidden_slots': torch.empty(pend_max, dtype=torch.long, device=device),
                'input_ids':    torch.empty(pend_max, dtype=torch.long, device=device),
                'ttt_valid':    torch.empty(pend_max, dtype=torch.long, device=device),
                'node_indices': torch.empty(pend_max, dtype=torch.long, device=device),
                'cum_lps':      torch.empty(pend_max, dtype=torch.float32, device=device),
            }
            self._triton_pend_buf = pend_buf
        sizes4 = getattr(self, '_triton_sizes4', None)
        if sizes4 is None:
            sizes4 = torch.empty(4, dtype=torch.long, device=device)
            self._triton_sizes4 = sizes4
        sizes1 = getattr(self, '_triton_sizes1', None)
        if sizes1 is None:
            sizes1 = torch.empty(1, dtype=torch.long, device=device)
            self._triton_sizes1 = sizes1
        rank_slot_topk_table = getattr(self, '_triton_rank_slot_topk_t', None)
        if rank_slot_topk_table is None:
            rank_slot_topk_table = torch.tensor(
                self.RANK_SLOT_TOPK, dtype=torch.long, device=device,
            )
            self._triton_rank_slot_topk_t = rank_slot_topk_table

        sample_token = input_id[:, -1]

        # Block-1 GPU precompute (same as triton path).
        d2t_offsets = self.draft_model.d2t
        last_p = logits[0]
        log_probs = F.log_softmax(last_p.float(), dim=-1)
        rank_preds_1d = rank_logits[0].argmax(dim=-1)
        top_idx_2d = torch.topk(last_p, max_topk, dim=-1).indices
        all_top_target_2d = self._map_draft_to_target(top_idx_2d)
        all_top_lps_2d = log_probs.gather(1, top_idx_2d)
        greedy_tgt_1d = all_top_target_2d[:, 0].contiguous()
        greedy_lps_1d = all_top_lps_2d[:, 0].contiguous()

        # ==== KERNEL 1: block-1 mega-kernel + 4-int sync ====
        n_nodes_b1, n_active, total_alts1, N_pend = cuda_build_block1(
            rank_preds_1d, greedy_tgt_1d, greedy_lps_1d,
            all_top_target_2d, all_top_lps_2d,
            rank_slot_topk_table, buf, pend_buf, sizes4,
            beam_width, K, max_topk, rank_classes, give_up_class,
            adaptive_slot0, adaptive_all,
        )
        all_rank_stats_stub = [{'rank_preds': [0] * K, 'M': K,
                                'branch': 0, 'parent_block': -1}]

        if N_pend == 0:
            return self._finalize_gpu_tree(
                n_nodes_b1, buf, sample_token, device, K,
                all_rank_stats_stub, draft_position,
            )

        # Host-side pending batch setup (all GPU ops, no extra sync).
        pend_slots = pend_buf['hidden_slots'][:N_pend]
        pend_input_ids = pend_buf['input_ids'][:N_pend].unsqueeze(1)
        pend_ttt_valid_t = pend_buf['ttt_valid'][:N_pend]
        pend_node_indices_t = pend_buf['node_indices'][:N_pend]
        pend_cum_lps_t = pend_buf['cum_lps'][:N_pend]

        pend_hidden = draft_hidden[0, pend_slots, :].unsqueeze(1)
        arange_K_t = self._arange_K_t
        batch_ttt_mask = arange_K_t.unsqueeze(0) < pend_ttt_valid_t.unsqueeze(1)

        _model_K = self.K
        effective_cross_count = draft_cache[0][2] - _model_K if draft_cache[0][2] >= _model_K else 0
        batch_cross_cache = []
        for layer_cache in draft_cache:
            if effective_cross_count > 0 and layer_cache[0] is not None:
                k_view = layer_cache[0][:, :, :effective_cross_count, :]
                v_view = layer_cache[1][:, :, :effective_cross_count, :]
                batch_cross_cache.append([
                    k_view.expand(N_pend, -1, -1, -1),
                    v_view.expand(N_pend, -1, -1, -1),
                    effective_cross_count,
                ])
            else:
                batch_cross_cache.append([None, None, 0])
        if effective_cross_count > 0:
            _cross_ones = batch_ttt_mask.new_ones(N_pend, effective_cross_count)
            _full_kv_mask = torch.cat([_cross_ones, batch_ttt_mask], dim=1)
        else:
            _full_kv_mask = batch_ttt_mask
        batch_ttt_kv = [
            (ttt_kv[l][0].expand(N_pend, -1, -1, -1),
             ttt_kv[l][1].expand(N_pend, -1, -1, -1))
            for l in range(num_layers)
        ]

        # Block-2 forward (BFS iter 1).
        if self._bfs_events_enabled:
            _bfs_idx = len(self._block_forward_events[1])
            blk_start, blk_end = self._get_bfs_event_pair(1, _bfs_idx)
            blk_start.record()
        block_logits, block_rank_logits, _block_draft_hidden, _new_ttt_kv_batch = \
            self.draft_model.forward_with_cache(
                hidden=pend_hidden,
                input_ids=pend_input_ids,
                cache=batch_cross_cache,
                position_id=draft_position,
                use_draft_condition=True,
                ttt_cache=batch_ttt_kv,
                ttt_mask=batch_ttt_mask,
                update_cross_cache=False,
                full_kv_mask=_full_kv_mask,
            )
        if self._bfs_events_enabled:
            blk_end.record()
            self._block_forward_events[1].append((blk_start, blk_end))

        # Block-2 GPU precompute (same _bfs_gpu_ops_fused).
        (all_rank_preds_2, _all_greedy_tokens_2, all_greedy_target_2,
         all_greedy_lps_2, _M2, _bf2, _gu2,
         top_target_all_2, top_lps_all_2) = _bfs_gpu_ops_fused(
            block_logits, block_rank_logits, d2t_offsets, max_topk,
            rank_classes, self._rank_to_factor,
        )

        # ==== KERNEL 2: BFS mega-kernel + 1-int sync ====
        total_alts_2 = cuda_build_bfs(
            all_rank_preds_2,
            all_greedy_target_2,
            all_greedy_lps_2,
            top_target_all_2,
            top_lps_all_2,
            pend_cum_lps_t, pend_node_indices_t,
            rank_slot_topk_table,
            buf, sizes1,
            tree_start=n_nodes_b1,
            N=N_pend, K=K, max_topk=max_topk, rank_classes=rank_classes,
            give_up_class=give_up_class, adaptive_all=adaptive_all,
            pend_depth=1,
        )
        n_nodes_final = n_nodes_b1 + N_pend * K + total_alts_2

        all_rank_stats = list(all_rank_stats_stub)
        for _i in range(N_pend):
            all_rank_stats.append(
                {'rank_preds': [0] * K, 'M': K, 'branch': 0, 'parent_block': 0}
            )

        return self._finalize_gpu_tree(
            n_nodes_final, buf, sample_token, device, K,
            all_rank_stats, draft_position,
        )

    # ============================================================
    #    TREE_FIXED_N=K path — zero host sync for N_pend
    # ============================================================
    @torch.no_grad()
    def _build_tree_from_block1_cuda_fixed(
        self,
        logits: torch.Tensor,
        rank_logits: torch.Tensor,
        draft_hidden: torch.Tensor,
        ttt_kv: list,
        input_id: torch.Tensor,
        draft_cache: list,
        draft_position: int,
        temperature: float = 0.0,
    ):
        """Fixed-N variant: block-1 kernel emits exactly N_fixed pending
        entries (top-N by cum_lp, padded with dummy).

        Savings over _build_tree_from_block1_cuda:
          - no host sync on sizes[3] → N_pend is Python constant.
          - BFS forward input shapes are static at [N_fixed, 1, H].
          - enables CUDA graph capture for the BFS forward path
            (prerequisite for the cc-bucketed graph trial in Day 3).

        Caveats:
          - Dummies carry ttt_valid=0 so batch_ttt_mask masks them. Their
            BFS forward output is garbage with very-negative cum_lp; GPU
            prune naturally drops their descendants. No per-dummy cleanup
            needed in the BFS kernel.
          - The block-1 kernel's sizes[0] (n_nodes_b1) and sizes[2]
            (total_alts1) still need a host read for the BFS tree_start.
            That's 1 sync instead of the prior 2 syncs for this driver.
        """
        from .tree_build_cuda_loader import (
            cuda_build_block1_fixed_n,
            cuda_build_bfs,
            tree_bfs_required_capacity,
        )

        N_fixed = self._tree_fixed_n
        K = self.K
        device = logits.device
        num_layers = len(ttt_kv)
        beam_width = self.beam_width
        rank_classes = self.rank_classes
        give_up_class = rank_classes - 1
        adaptive_slot0 = self._adaptive_slot0
        adaptive_all = self._adaptive_all
        max_topk = max(beam_width, max(self.RANK_SLOT_TOPK))

        # Fixed-N sends every padded leaf through BFS, so block-2 capacity must
        # scale with N_fixed rather than the variable block-1 node ceiling.
        max_block1_nodes = K + K * (max_topk - 1) + 1
        required_nodes = tree_bfs_required_capacity(
            max_block1_nodes,
            N_fixed,
            K,
            max_topk,
        )
        max_nodes = max(self.total_tokens + 200, required_nodes + 100)
        buf = getattr(self, '_tree_gpu_buf', None)
        if buf is None or buf['tokens'].shape[0] < max_nodes:
            buf = {
                'tokens':  torch.empty(max_nodes, dtype=torch.long, device=device),
                'parents': torch.empty(max_nodes, dtype=torch.long, device=device),
                'lps':     torch.empty(max_nodes, dtype=torch.float32, device=device),
                'ranks':   torch.empty(max_nodes, dtype=torch.long, device=device),
                'blocks':  torch.empty(max_nodes, dtype=torch.long, device=device),
                'slots':   torch.empty(max_nodes, dtype=torch.long, device=device),
            }
            self._tree_gpu_buf = buf
        # Pending buf sized to N_fixed.
        pend_buf = getattr(self, '_fixed_pend_buf', None)
        if pend_buf is None or pend_buf['hidden_slots'].shape[0] < N_fixed:
            pend_buf = {
                'hidden_slots': torch.empty(N_fixed, dtype=torch.long, device=device),
                'input_ids':    torch.empty(N_fixed, dtype=torch.long, device=device),
                'ttt_valid':    torch.empty(N_fixed, dtype=torch.long, device=device),
                'node_indices': torch.empty(N_fixed, dtype=torch.long, device=device),
                'cum_lps':      torch.empty(N_fixed, dtype=torch.float32, device=device),
            }
            self._fixed_pend_buf = pend_buf
        sizes4 = getattr(self, '_triton_sizes4', None)
        if sizes4 is None:
            sizes4 = torch.empty(4, dtype=torch.long, device=device)
            self._triton_sizes4 = sizes4
        sizes1 = getattr(self, '_triton_sizes1', None)
        if sizes1 is None:
            sizes1 = torch.empty(1, dtype=torch.long, device=device)
            self._triton_sizes1 = sizes1
        rank_slot_topk_table = getattr(self, '_triton_rank_slot_topk_t', None)
        if rank_slot_topk_table is None:
            rank_slot_topk_table = torch.tensor(
                self.RANK_SLOT_TOPK, dtype=torch.long, device=device,
            )
            self._triton_rank_slot_topk_t = rank_slot_topk_table

        sample_token = input_id[:, -1]

        # Block-1 GPU precompute (same as non-fixed path).
        d2t_offsets = self.draft_model.d2t
        last_p = logits[0]
        log_probs = F.log_softmax(last_p.float(), dim=-1)
        rank_preds_1d = rank_logits[0].argmax(dim=-1)
        top_idx_2d = torch.topk(last_p, max_topk, dim=-1).indices
        all_top_target_2d = self._map_draft_to_target(top_idx_2d)
        all_top_lps_2d = log_probs.gather(1, top_idx_2d)
        greedy_tgt_1d = all_top_target_2d[:, 0].contiguous()
        greedy_lps_1d = all_top_lps_2d[:, 0].contiguous()

        # ==== KERNEL 1 (fixed-N): no sync on N_pend — N_pend = N_fixed constant. ====
        cuda_build_block1_fixed_n(
            rank_preds_1d, greedy_tgt_1d, greedy_lps_1d,
            all_top_target_2d, all_top_lps_2d,
            rank_slot_topk_table, buf, pend_buf, sizes4,
            beam_width, K, max_topk, rank_classes, give_up_class,
            adaptive_slot0, adaptive_all,
            N_fixed,
        )

        # Block-1 sizes still needed for BFS tree_start. 1 sync here.
        # TODO: eliminate by passing tree_start as GPU scalar to BFS kernel.
        sizes_cpu = sizes4.cpu().tolist()
        n_nodes_b1 = sizes_cpu[0]

        all_rank_stats_stub = [{'rank_preds': [0] * K, 'M': K,
                                'branch': 0, 'parent_block': -1}]

        # Host-side pend batch setup — ALL static shape [N_fixed, ...].
        pend_slots = pend_buf['hidden_slots']                     # [N_fixed]
        pend_input_ids = pend_buf['input_ids'].unsqueeze(1)       # [N_fixed, 1]
        pend_ttt_valid_t = pend_buf['ttt_valid']                  # [N_fixed]
        pend_node_indices_t = pend_buf['node_indices']            # [N_fixed]
        pend_cum_lps_t = pend_buf['cum_lps']                      # [N_fixed]

        pend_hidden = draft_hidden[0, pend_slots, :].unsqueeze(1)  # [N_fixed, 1, H]
        arange_K_t = self._arange_K_t
        # Dummies have ttt_valid=0 → batch_ttt_mask row all False.
        batch_ttt_mask = arange_K_t.unsqueeze(0) < pend_ttt_valid_t.unsqueeze(1)

        _model_K = self.K
        effective_cross_count = draft_cache[0][2] - _model_K if draft_cache[0][2] >= _model_K else 0
        batch_cross_cache = []
        for layer_cache in draft_cache:
            if effective_cross_count > 0 and layer_cache[0] is not None:
                k_view = layer_cache[0][:, :, :effective_cross_count, :]
                v_view = layer_cache[1][:, :, :effective_cross_count, :]
                batch_cross_cache.append([
                    k_view.expand(N_fixed, -1, -1, -1),
                    v_view.expand(N_fixed, -1, -1, -1),
                    effective_cross_count,
                ])
            else:
                batch_cross_cache.append([None, None, 0])
        if effective_cross_count > 0:
            _cross_ones = batch_ttt_mask.new_ones(N_fixed, effective_cross_count)
            _full_kv_mask = torch.cat([_cross_ones, batch_ttt_mask], dim=1)
        else:
            _full_kv_mask = batch_ttt_mask
        batch_ttt_kv = [
            (ttt_kv[l][0].expand(N_fixed, -1, -1, -1),
             ttt_kv[l][1].expand(N_fixed, -1, -1, -1))
            for l in range(num_layers)
        ]

        # BFS forward (batch = N_fixed, static shape).
        if self._bfs_events_enabled:
            _bfs_idx = len(self._block_forward_events[1])
            blk_start, blk_end = self._get_bfs_event_pair(1, _bfs_idx)
            blk_start.record()
        _graph_out = None
        if self._draft_graph_cache is not None:
            # Build cross_cache_slices as [1, heads, cc, D] views
            # (graph cache's run() expects pre-sliced views, one per layer).
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
                _graph_out = self._draft_graph_cache.run(
                    hidden=pend_hidden,
                    input_ids=pend_input_ids,
                    cross_cache_slices=cross_slices,
                    effective_cross_count=effective_cross_count,
                    ttt_cache=[ttt_kv[l] for l in range(num_layers)],
                    ttt_mask=batch_ttt_mask,
                    position_id=draft_position,
                )
            except Exception as _ge:
                _graph_out = None
                # Log once for diagnostics, don't spam.
                if not getattr(self, '_graph_fallback_logged', False):
                    print(f"[DRAFT_CUDA_GRAPH] BFS fallback: {type(_ge).__name__}: {_ge}")
                    self._graph_fallback_logged = True

        if _graph_out is not None:
            block_logits, block_rank_logits, _block_draft_hidden, _new_ttt_kv_batch = _graph_out
        else:
            block_logits, block_rank_logits, _block_draft_hidden, _new_ttt_kv_batch = \
                self.draft_model.forward_with_cache(
                    hidden=pend_hidden,
                    input_ids=pend_input_ids,
                    cache=batch_cross_cache,
                    position_id=draft_position,
                    use_draft_condition=True,
                    ttt_cache=batch_ttt_kv,
                    ttt_mask=batch_ttt_mask,
                    update_cross_cache=False,
                    full_kv_mask=_full_kv_mask,
                )
        if self._bfs_events_enabled:
            blk_end.record()
            self._block_forward_events[1].append((blk_start, blk_end))

        (all_rank_preds_2, _all_greedy_tokens_2, all_greedy_target_2,
         all_greedy_lps_2, _M2, _bf2, _gu2,
         top_target_all_2, top_lps_all_2) = _bfs_gpu_ops_fused(
            block_logits, block_rank_logits, d2t_offsets, max_topk,
            rank_classes, self._rank_to_factor,
        )

        # ==== KERNEL 2: BFS scatter. N = N_fixed (static). ====
        total_alts_2 = cuda_build_bfs(
            all_rank_preds_2,
            all_greedy_target_2,
            all_greedy_lps_2,
            top_target_all_2,
            top_lps_all_2,
            pend_cum_lps_t, pend_node_indices_t,
            rank_slot_topk_table,
            buf, sizes1,
            tree_start=n_nodes_b1,
            N=N_fixed, K=K, max_topk=max_topk, rank_classes=rank_classes,
            give_up_class=give_up_class, adaptive_all=adaptive_all,
            pend_depth=1,
        )
        n_nodes_final = n_nodes_b1 + N_fixed * K + total_alts_2
        tree_capacity = int(buf['tokens'].numel())
        if n_nodes_final > tree_capacity:
            raise RuntimeError(
                "fixed-N CUDA tree output exceeds buffer capacity before finalize: "
                f"nodes={n_nodes_final}, capacity={tree_capacity}"
            )

        all_rank_stats = list(all_rank_stats_stub)
        for _i in range(N_fixed):
            all_rank_stats.append(
                {'rank_preds': [0] * K, 'M': K, 'branch': 0, 'parent_block': 0}
            )

        return self._finalize_gpu_tree(
            n_nodes_final, buf, sample_token, device, K,
            all_rank_stats, draft_position,
        )

    def _validate_strict_linear_tree(self, result):
        """Fail fast if a strict-linear builder emits siblings or a truncated path."""
        draft_tokens, tree_mask, tree_position_ids, retrieve_indices = result[:4]
        expected_width = self._strict_linear_nodes + 1  # verified root + draft path
        width = int(draft_tokens.shape[-1])
        expected_path = torch.arange(width, device=retrieve_indices.device)
        expected_mask = torch.tril(torch.ones(
            (1, 1, width, width), dtype=tree_mask.dtype, device=tree_mask.device
        ))
        retrieve_shape_ok = (
            retrieve_indices.ndim == 2
            and retrieve_indices.shape[0] == 1
            and retrieve_indices.shape[1] >= width
        )
        retrieve_path_ok = False
        if retrieve_shape_ok:
            retrieve_row = retrieve_indices[0]
            retrieve_path_ok = (
                torch.equal(retrieve_row[:width], expected_path)
                and bool(torch.all(retrieve_row[width:] == -1).item())
            )
        if (
            width != expected_width
            or not retrieve_path_ok
            or tuple(tree_mask.shape) != (1, 1, width, width)
            or not torch.equal(tree_mask, expected_mask)
            or not torch.equal(
                tree_position_ids,
                torch.arange(width, device=tree_position_ids.device),
            )
        ):
            raise RuntimeError(
                "strict_linear tree invariant failed: "
                f"draft_width={width}, expected_width={expected_width}, "
                f"retrieve_shape={tuple(retrieve_indices.shape)}, "
                f"tree_mask_shape={tuple(tree_mask.shape)}, "
                f"positions={tree_position_ids.tolist()}, "
                f"retrieve={retrieve_indices.tolist()}"
            )
        return result

    @torch.no_grad()
    def _build_tree_from_block1_dispatch(self, *args, **kwargs):
        """Dispatch to GPU-native tree builder when config supports it.

        Highest priority: SPECBLOCK_STATIC=1 → StaticDraftBuilder
          (static-shape, sync-free, graph-friendly; uniform MAX_TOPK
           branching with rank-aware prune via cumulative bias).
        Fast path: triton / cuda / tree_gpu specialized kernels.
        Fallback: default numpy path for unsupported configs.
        """
        if self._strict_linear:
            # CUDA/Triton block-1 kernels use the first non-zero rank to decide
            # whether the endpoint may continue. A strict path ignores this rank
            # policy, so present an all-class-0 view while keeping logits intact.
            linear_rank_logits = torch.full_like(args[1], torch.finfo(args[1].dtype).min)
            linear_rank_logits[..., 0] = 0
            args = (args[0], linear_rank_logits, *args[2:])

        if self._use_static_draft and self._static_draft is not None:
            # StaticDraftBuilder returns views into persistent buffers.  Clone
            # before another request can overwrite them in a batched draft loop.
            result = _clone_static_tree_result(self._static_draft.build_tree(
                b0_logits=args[0],
                b0_rank_logits=args[1],
                b0_draft_hidden=args[2],
                b0_ttt_kv=args[3],
                input_id=args[4],
                draft_cache=args[5],
                draft_position=args[6],
                temperature=kwargs.get(
                    "temperature",
                    args[7] if len(args) > 7 else 0.0,
                ),
            ))
        elif (self._tree_build_cuda and self._tree_fixed_n > 0
                and self._tree_gpu_supported()):
            result = self._build_tree_from_block1_cuda_fixed(*args, **kwargs)
        elif self._tree_build_cuda and self._tree_gpu_supported():
            result = self._build_tree_from_block1_cuda(*args, **kwargs)
        elif self._tree_build_triton and self._tree_gpu_supported():
            result = self._build_tree_from_block1_triton(*args, **kwargs)
        elif self._tree_gpu and self._tree_gpu_supported():
            result = self._build_tree_from_block1_gpu(*args, **kwargs)
        else:
            result = self._build_tree_from_block1(*args, **kwargs)

        if self._strict_linear:
            return self._validate_strict_linear_tree(result)
        return result

    def _build_tree_from_block1(
        self,
        logits: torch.Tensor,          # [1, K, draft_vocab]
        rank_logits: torch.Tensor,     # [1, K, rank_classes]
        draft_hidden: torch.Tensor,    # [1, K, H]
        ttt_kv: list,
        input_id: torch.Tensor,        # [1, 1]
        draft_cache: list,
        draft_position: int,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list, list]:
        """Build rank-guided draft tree from pre-computed block 1 results.

        BFS expansion + post-expansion pruning + tree buffer construction.
        Optimized: batch TTT cache as tensors (no per-leaf pad/stack/cat),
        bulk .tolist() instead of per-leaf .item().

        Returns:
            draft_tokens: [1, N+1] tree tokens (first is root/sample_token)
            tree_mask: [1, 1, N+1, N+1] tree attention mask
            tree_position_ids: [N+1] depth of each node
            retrieve_indices: [num_leaves, max_depth+1] indices for each root-to-leaf path
            rank_stats: list of rank_info dicts per block
            node_ranks: list of rank predictions per tree node (parallel to tree_nodes)
            node_block_slots: list of (block_idx, slot_idx) per tree node
        """
        K = self.K
        device = logits.device
        num_layers = len(ttt_kv)

        # Effective tree_K: use fewer slots for tree building (model outputs full K)
        if self._tree_K is not None and self._tree_K < K:
            _tk = self._tree_K
            logits = logits[:, :_tk, :].contiguous()
            rank_logits = rank_logits[:, :_tk, :].contiguous()
            draft_hidden = draft_hidden[:, :_tk, :].contiguous()
            ttt_kv = [(k[:, :, :_tk, :].contiguous(), v[:, :, :_tk, :].contiguous()) for k, v in ttt_kv]
            K = _tk

        sample_token = input_id[:, -1]  # [1]

        # Pre-allocated numpy tree buffers (C-level ops, no Python tuple overhead)
        # Buffer must hold all nodes before post-expansion pruning.
        # Each block can produce up to beam_width * K * max_factor nodes;
        # max_factor=10 is the topk constant defined in the BFS loop below.
        # Block 1 seeds ~(beam_width + K*max_factor) pending leaves, which all enter block 2
        # before adaptive beam prunes. Block 2 can add (pending) * K * max_factor nodes.
        # max_factor=10 is the topk constant used in the BFS loop below.
        # Allow iter-adaptive budget to scale up to 4x tokens and 3x beam_width
        if self._iter_adapt_budget > 0:
            _max_beam = self.beam_width * 3
            _max_tokens = self.total_tokens * 4
        else:
            _max_beam = self.beam_width
            _max_tokens = self.total_tokens
        _max_pending_b1 = _max_beam + self.K * 10
        max_nodes = _max_tokens + _max_pending_b1 * self.K * 10 + self.K * 10 * self.max_blocks + 150
        tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots = \
            self._ensure_tree_buffers(max_nodes)
        n_nodes = 0

        # Profile tree building phases (skipped in production — saves ~0.3-0.5 ms/iter
        # of Python overhead from perf_counter_ns + dict updates). Enable with PROFILE_TREE=1.
        _profile_on = self._profile_tree_enabled
        if _profile_on:
            import time as _time
            _tree_profile = {}
            _t0 = _time.perf_counter_ns()
        else:
            _tree_profile = None
            _t0 = 0

        # ===== Block 1: batch compute (N=1) =====
        if _profile_on:
            _tb1_start = _time.perf_counter_ns()
        last_p = logits[0]  # [K, draft_vocab]
        rank_preds = rank_logits[0].argmax(dim=-1)  # [K]
        log_probs = F.log_softmax(last_p.float(), dim=-1)  # [K, V]

        M_t, bf_t, gu_t = self._walk_rank_slots_batch(rank_preds.unsqueeze(0))

        greedy_tokens = last_p.argmax(dim=-1)  # [K]
        greedy_target = self._map_draft_to_target(greedy_tokens)  # [K]
        greedy_lps = log_probs.gather(1, greedy_tokens.unsqueeze(-1)).squeeze(-1)  # [K]

        # Merge block1's two syncs into one: topk over all K slots computed
        # now so results are in-flight with greedy/rank transfers. Downstream
        # Python branching (adaptive_slot0/all + numpy scatter) uses all these
        # CPU arrays so we gate on a single sync.
        rank_slot_topk = self.RANK_SLOT_TOPK
        max_topk = max(self.beam_width, max(rank_slot_topk))
        all_top = torch.topk(last_p, max_topk, dim=-1)
        all_top_target = self._map_draft_to_target(all_top.indices)
        all_top_lps = log_probs.gather(1, all_top.indices)

        # Batch GPU→CPU (non_blocking, 1 sync)
        if _profile_on:
            _tb1_gpu = _time.perf_counter_ns()
            _tree_profile['b1_gpu'] = _tree_profile.get('b1_gpu', 0) + (_tb1_gpu - _tb1_start) / 1e6
        _gt_cpu = greedy_target.to('cpu', non_blocking=True)
        _glp_cpu = greedy_lps.to('cpu', non_blocking=True)
        _rp_cpu = rank_preds.to('cpu', non_blocking=True)
        _M_cpu = M_t.to('cpu', non_blocking=True)
        _bf_cpu = bf_t.to('cpu', non_blocking=True)
        _gu_cpu = gu_t.to('cpu', non_blocking=True)
        _att_cpu = all_top_target.to('cpu', non_blocking=True)
        _atl_cpu = all_top_lps.to('cpu', non_blocking=True)
        torch.cuda.synchronize()
        gt_np = _gt_cpu.numpy()          # [K] numpy
        glp_np = _glp_cpu.numpy()        # [K] numpy
        rp_np = _rp_cpu.numpy()          # [K] numpy
        att_np = _att_cpu.numpy()        # [K, max_topk]
        atl_np = _atl_cpu.numpy()        # [K, max_topk]
        M = int(_M_cpu[0])
        branch_factor = int(_bf_cpu[0])

        # Optional n-gram cache injection: replace some slot-0 alternatives with
        # session-learned top next tokens for (prev_tok, root_tok) context. Greedy
        # top-1 unchanged (tree root preserved). Strict spec decode: n-gram only
        # widens tree coverage; final accept still requires target argmax match.
        if self._ngram_cache_on and self._ngram_topk > 0 and max_topk > 2:
            _ctx = self._ngram_prev2
            _tbl = self._ngram_table.get(_ctx)
            if _tbl is not None and len(_tbl) > 0:
                # Get top-K candidates (by count, filter >= min_count)
                _cands = sorted(
                    ((tok, cnt) for tok, cnt in _tbl.items()
                     if cnt >= self._ngram_min_count),
                    key=lambda x: -x[1],
                )[:self._ngram_topk]
                if _cands:
                    # Current slot-0 tokens (target ids)
                    _slot0_targets = att_np[0].tolist()
                    _slot0_lps = atl_np[0].tolist()
                    # Keep top-1 (index 0), replace tail alternatives with n-gram tokens
                    # that aren't already in the list
                    _existing = set(_slot0_targets)
                    _new_toks = [t for t, _ in _cands if t not in _existing]
                    if _new_toks:
                        # Replace last N positions with n-gram tokens (give them
                        # a modest log-prob = min(existing) - 0.1 to keep downstream
                        # score comparisons sane)
                        _min_lp = min(_slot0_lps[1:]) if len(_slot0_lps) > 1 else -5.0
                        _nt_count = min(len(_new_toks), max_topk - 1)
                        for i in range(_nt_count):
                            att_np[0][max_topk - 1 - i] = _new_toks[i]
                            atl_np[0][max_topk - 1 - i] = _min_lp - 0.1

        # Optional online logit correction: shift slot 0 alternatives (index 1..K-1)
        # toward tokens target has historically preferred. Greedy (index 0) stays unchanged.
        # Build running bias log((target_want+1) / (draft_want+1)) from verify stats.
        if self._logit_correct_on and self._target_want_count is not None and max_topk > 2:
            _lr = self._logit_correct_lr
            # Stats in target vocab; att_np[slot, k] is target-vocab token id
            _att_t = all_top_target  # [K, max_topk] on GPU
            _tw_lookup = self._target_want_count.gather(0, _att_t.flatten()).reshape(_att_t.shape)
            _dw_lookup = self._draft_want_count.gather(0, _att_t.flatten()).reshape(_att_t.shape)
            _corr = torch.log((_tw_lookup + 1.0) / (_dw_lookup + 1.0)) * _lr
            _adjusted = all_top_lps + _corr  # [K, max_topk]
            # Re-rank slot 0 alternatives only (keep greedy top-1 unchanged)
            _adj0 = _adjusted[0].cpu().numpy()
            _order = np.argsort(-_adj0)  # descending
            # Force index 0 to stay as original top-1
            if _order[0] != 0:
                _order = np.concatenate([[0], _order[_order != 0]])
            # Rearrange att_np[0] and atl_np[0]
            att_np[0] = att_np[0][_order]
            atl_np[0] = atl_np[0][_order]

        # Optional MMR re-rank on slot 0 siblings (diverse draft candidates).
        # Diag (mtbench:3 → 397 iter) shows draft's slot-0 top-K often cluster
        # in one semantic cluster (e.g. "culture/cultural/history/natural"),
        # missing target top-20 (which spans "adventure/trip/journey/..").
        # 93% of misses have overlap(target_top20, draft_siblings) = 0.
        #
        # MMR selects diverse alternatives from slot-0 top-K: keep greedy[0] as
        # tree root (unchanged), re-select alternatives [1..K-1] by:
        #   score_i = (1-λ)·log_prob_i − λ·max_cos_sim(i, already_selected)
        # Uses target's embedding matrix (shape [vocab, D]) for similarity.
        # Enable via MMR_SIBLINGS=1, λ controlled by MMR_LAMBDA (default 0.5).
        if os.environ.get('MMR_SIBLINGS', '0') == '1' and max_topk > 2:
            _mmr_lam = float(os.environ.get('MMR_LAMBDA', '0.5'))
            # Compute slot-0 token similarity via target embedding on GPU
            _slot0_t = all_top_target[0]  # [max_topk] target token ids (GPU)
            _emb_m = self.target_model.model.embed_tokens.weight
            _emb = F.embedding(_slot0_t, _emb_m).float()  # [max_topk, D]
            _emb_norm = F.normalize(_emb, dim=-1)
            _sim = (_emb_norm @ _emb_norm.T).cpu().numpy()  # [max_topk, max_topk]

            _rel = atl_np[0]  # log probs (higher = better)
            _sel = [0]  # greedy top-1 always first (tree root)
            _rem = list(range(1, max_topk))
            for _ in range(max_topk - 1):
                _best_score = -1e30
                _best_i = _rem[0]
                for _i in _rem:
                    _max_sim = max(_sim[_i, _j] for _j in _sel)
                    _score = (1.0 - _mmr_lam) * _rel[_i] - _mmr_lam * _max_sim
                    if _score > _best_score:
                        _best_score = _score
                        _best_i = _i
                _sel.append(_best_i)
                _rem.remove(_best_i)
            _order = np.array(_sel, dtype=np.int64)
            att_np[0] = att_np[0][_order]
            atl_np[0] = atl_np[0][_order]

        # Iteration-level adaptive budget: scale total_tokens AND beam_width per-iter.
        # Easy iter = small tree (cheap), Hard iter = big tree (more hedge).
        # This redistributes compute: easy iters contribute to hard iters.
        _orig_total_tokens = self.total_tokens
        _orig_beam_width = self.beam_width
        if self._iter_adapt_budget > 0:
            # Use slot 0 prob as primary difficulty signal (correlates with accept outcome)
            # Also combine with min slot prob (any hard slot = hard iter)
            p0 = float(np.exp(glp_np[0])) if len(glp_np) > 0 else 0.5
            min_p = float(np.exp(glp_np.min())) if len(glp_np) > 0 else 0.5
            avg_p = float(np.exp(glp_np).mean())

            if self._iter_adapt_budget == 1:  # conservative
                if p0 > 0.9 and avg_p > 0.7:
                    _budget_scale, _beam_scale = 0.6, 1.0  # easy: fewer tokens
                elif p0 > 0.7:
                    _budget_scale, _beam_scale = 1.0, 1.0  # normal
                else:
                    _budget_scale, _beam_scale = 2.0, 2.0  # hard: 2x everything
            elif self._iter_adapt_budget == 2:  # aggressive
                if p0 > 0.9 and avg_p > 0.7:
                    _budget_scale, _beam_scale = 0.5, 0.7
                elif p0 > 0.7:
                    _budget_scale, _beam_scale = 1.0, 1.0
                else:
                    _budget_scale, _beam_scale = 3.0, 2.5
            elif self._iter_adapt_budget == 3:  # extreme
                if p0 > 0.9:
                    _budget_scale, _beam_scale = 0.4, 0.5
                elif p0 > 0.7:
                    _budget_scale, _beam_scale = 0.8, 0.8
                else:
                    _budget_scale, _beam_scale = 4.0, 3.0
            elif self._iter_adapt_budget == 4:  # balanced: minor easy cut, big hard boost
                if p0 > 0.9 and avg_p > 0.75:
                    _budget_scale, _beam_scale = 0.7, 0.8  # small easy cut
                elif p0 > 0.7:
                    _budget_scale, _beam_scale = 1.1, 1.1  # normal slight boost
                else:
                    _budget_scale, _beam_scale = 3.0, 2.5  # hard 3x
            else:  # 5 = hard-only boost: easy/normal unchanged, only hard gets more budget
                # Hypothesis: mode 1-4 lose because cutting easy-iter budget hurts more
                # than boosting hard-iter helps. Leave easy alone, only boost hard.
                if p0 > 0.6:
                    _budget_scale, _beam_scale = 1.0, 1.0  # easy/normal: unchanged
                else:
                    _budget_scale, _beam_scale = 2.5, 2.0  # hard: 2.5x budget

            self.total_tokens = max(30, int(_orig_total_tokens * _budget_scale))
            # Cap beam_width by max_topk: beam_width controls how many alternatives
            # per slot are considered; all_top only has max_topk columns, indexing
            # beyond crashes with IndexError.
            _new_beam = max(3, int(_orig_beam_width * _beam_scale))
            self.beam_width = min(_new_beam, max_topk)

        # Conditional max_blocks: easy iter skips last block forward (saves ~2.8 ms/iter).
        # Applied independently of iter_adapt_budget; both can be combined.
        iter_max_blocks = self.max_blocks
        if self._cond_max_blocks > 0 and self.max_blocks > 1:
            _p0 = float(np.exp(glp_np[0])) if len(glp_np) > 0 else 0.5
            _avg_p = float(np.exp(glp_np).mean())
            if self._cond_max_blocks == 1:
                # Conservative: only very-easy iters skip last block
                if _p0 > 0.9 and _avg_p > 0.7:
                    iter_max_blocks = self.max_blocks - 1
            elif self._cond_max_blocks == 2:
                # Aggressive: easier threshold
                if _p0 > 0.85:
                    iter_max_blocks = self.max_blocks - 1
            elif self._cond_max_blocks == 3:
                # Extreme: cut to 1 for very easy, 2 for easy
                if _p0 > 0.95:
                    iter_max_blocks = max(1, self.max_blocks - 2)
                elif _p0 > 0.85:
                    iter_max_blocks = self.max_blocks - 1
        give_up = bool(_gu_cpu[0])

        all_rank_stats = [{'rank_preds': rp_np.tolist(), 'M': M, 'branch': branch_factor, 'parent_block': -1}]

        if _profile_on:
            _tb1_tolist = _time.perf_counter_ns()
            _tree_profile['b1_tolist'] = _tree_profile.get('b1_tolist', 0) + (_tb1_tolist - _tb1_gpu) / 1e6

        # Pending state
        pend_node_indices = []
        pend_cum_lps = []
        pend_source_blocks = []

        # Greedy chain: K nodes (numpy vectorized)
        # Cached arange buffer (len == self.K; slice if _tree_K < self.K)
        _ar_K = self._arange_K_np[:K] if self._arange_K_np is not None else np.arange(K, dtype=np.int64)
        tn_tokens[:K] = gt_np
        tn_parents[:K] = _ar_K - 1  # [-1, 0, 1, ..., K-2]
        tn_lps[:K] = np.cumsum(glp_np)
        tn_ranks[:K] = rp_np
        tn_blocks[:K] = 0
        tn_slots[:K] = _ar_K
        n_nodes = K
        greedy_cum_lp = float(tn_lps[K - 1])

        # ===== Multi-slot branching: branch at every slot based on rank =====
        # (topk + transfer hoisted above, merged with first block1 sync)

        # Per-slot topk: adaptive budget allocation based on greedy probability
        slot_topks_np = np.zeros(K, dtype=np.int64)
        give_up_class = self.rank_classes - 1

        if getattr(self, '_adaptive_budget', False):
            # Adaptive: allocate from shared pool based on uncertainty (1 - greedy_prob)
            alt_budget = self.beam_width + sum(rank_slot_topk) - len(rank_slot_topk)  # total alternatives pool
            uncertainties = np.zeros(K, dtype=np.float64)
            for k in range(K):
                uncertainties[k] = max(0.01, 1.0 - float(F.softmax(last_p[k].float(), dim=-1).max().item()))
            # Allocate proportional to uncertainty, minimum 1 per slot
            raw_alloc = uncertainties / uncertainties.sum() * alt_budget
            alloc = np.maximum(np.round(raw_alloc).astype(np.int64), 1)
            # Clamp to max_topk - 1 (alternatives, not counting greedy)
            alloc = np.minimum(alloc, max_topk - 1)
            # Adjust to match budget
            while alloc.sum() > alt_budget:
                alloc[alloc.argmax()] -= 1
            while alloc.sum() < alt_budget and alloc.min() < max_topk - 1:
                idx = alloc.argmin()
                if alloc[idx] < max_topk - 1:
                    alloc[idx] += 1
            for k in range(K):
                slot_topks_np[k] = int(alloc[k]) + 1  # +1 for greedy
        else:
            # Optional override for block-1 slot-0 beam width (separate from self.beam_width
            # which controls BFS adaptive beam).
            _slot0_bw = self._slot0_beam if self._slot0_beam > 0 else self.beam_width
            # Adaptive slot 0 beam based on greedy prob
            if self._adaptive_slot0:
                # slot 0 greedy prob = exp(glp_np[0]) since glp_np is log_prob of greedy token
                p0 = float(np.exp(glp_np[0])) if len(glp_np) > 0 else 0.5
                # Mode controlled by ADAPTIVE_SLOT0: 1=conservative, 2=aggressive, 3=extreme, 4=ultra
                _mode_idx = self._adaptive_slot0  # int, set from env
                if _mode_idx == 2:
                    if p0 >= 0.95: _slot0_bw = min(2, self.beam_width)
                    elif p0 >= 0.85: _slot0_bw = min(4, self.beam_width)
                    elif p0 >= 0.6: _slot0_bw = min(7, self.beam_width)
                    else: _slot0_bw = self.beam_width
                elif _mode_idx == 3:
                    if p0 >= 0.95: _slot0_bw = 1   # just greedy
                    elif p0 >= 0.9: _slot0_bw = min(2, self.beam_width)
                    elif p0 >= 0.8: _slot0_bw = min(4, self.beam_width)
                    elif p0 >= 0.6: _slot0_bw = min(7, self.beam_width)
                    else: _slot0_bw = self.beam_width
                elif _mode_idx == 4:
                    # ultra: even more aggressive on high-conf cases
                    if p0 >= 0.9: _slot0_bw = 1     # just greedy
                    elif p0 >= 0.8: _slot0_bw = min(3, self.beam_width)
                    elif p0 >= 0.6: _slot0_bw = min(6, self.beam_width)
                    else: _slot0_bw = self.beam_width
                else:  # 1 (default conservative)
                    if p0 >= 0.9: _slot0_bw = min(3, self.beam_width)
                    elif p0 >= 0.8: _slot0_bw = min(5, self.beam_width)
                    elif p0 >= 0.6: _slot0_bw = min(8, self.beam_width)
                    else: _slot0_bw = self.beam_width
            for k in range(K):
                rk = int(rp_np[k])
                if k == 0:
                    slot_topks_np[k] = _slot0_bw
                else:
                    # Adaptive all slots: per-slot alt count based on slot k greedy prob
                    if self._adaptive_all:
                        pk = float(np.exp(glp_np[k])) if k < len(glp_np) else 0.5
                        base_topk = rank_slot_topk[rk]
                        if self._adaptive_all == 2:  # aggressive
                            if pk >= 0.9: slot_topks_np[k] = 1
                            elif pk >= 0.7: slot_topks_np[k] = 2
                            elif pk >= 0.5: slot_topks_np[k] = 3
                            else: slot_topks_np[k] = base_topk
                        else:  # 1 = conservative (default)
                            if pk >= 0.95: slot_topks_np[k] = min(2, base_topk)
                            elif pk >= 0.85: slot_topks_np[k] = min(3, base_topk)
                            else: slot_topks_np[k] = base_topk
                        # Still handle give_up break
                        if rk == give_up_class and base_topk == 0:
                            break
                        continue
                    topk_for_rank = rank_slot_topk[rk]
                    if rk == give_up_class and topk_for_rank == 0:
                        break  # true give_up: stop processing further slots
                    slot_topks_np[k] = topk_for_rank

        active_slots = np.nonzero(slot_topks_np > 1)[0]  # slots that branch
        # Compute per-slot alternatives count
        n_alt_per_slot = slot_topks_np[active_slots] - 1  # [n_active]
        total_alts = int(n_alt_per_slot.sum())

        # Parent chain lps (greedy chain already written at tree indices 0..K-1)
        # Parent of slot k: k-1 (previous greedy node), or -1 for slot 0
        # Parent cum_lp: tn_lps[k-1] for k>0, 0.0 for k=0

        # Bulk-add alternatives to tree buffers
        if total_alts > 0:
            s = n_nodes
            slot_expanded = np.repeat(active_slots, n_alt_per_slot)  # [total_alts]
            alt_within = np.concatenate([np.arange(1, n + 1) for n in n_alt_per_slot])  # 1..topk-1 per slot

            parent_nodes = slot_expanded - 1  # -1 for slot 0
            parent_lps = np.where(slot_expanded > 0, tn_lps[np.maximum(parent_nodes, 0)], 0.0)
            parent_lps[slot_expanded == 0] = 0.0  # force 0 for slot 0

            tn_tokens[s:s+total_alts] = att_np[slot_expanded, alt_within]
            tn_parents[s:s+total_alts] = parent_nodes
            tn_lps[s:s+total_alts] = parent_lps + atl_np[slot_expanded, alt_within]
            tn_ranks[s:s+total_alts] = rp_np[slot_expanded]
            tn_blocks[s:s+total_alts] = 0
            tn_slots[s:s+total_alts] = slot_expanded
            n_nodes += total_alts

        # Build pending arrays (g(k) for each active slot + alternatives + g(K-1))
        # g(k) pending entries
        pend_g_indices = active_slots.copy()  # tree indices of g(k) are 0..K-1
        pend_g_lps = tn_lps[active_slots]
        pend_g_tokens = gt_np[active_slots]
        pend_g_hidden_slots = active_slots.copy()
        pend_g_ttt_valid = active_slots + 1

        # Alternative pending entries
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

        # Concatenate
        pend_indices_np = np.concatenate([pend_g_indices, alt_tree_indices])
        pend_lps_np = np.concatenate([pend_g_lps, pend_a_lps])
        pend_tokens_np = np.concatenate([pend_g_tokens, pend_a_tokens])
        pend_hidden_slots_np = np.concatenate([pend_g_hidden_slots, pend_a_hidden_slots])
        pend_ttt_valid_np = np.concatenate([pend_g_ttt_valid, pend_a_ttt_valid])

        # Add greedy chain end g(K-1) if not already pending
        if not give_up and (K - 1) not in pend_indices_np:
            pend_indices_np = np.append(pend_indices_np, K - 1)
            pend_lps_np = np.append(pend_lps_np, greedy_cum_lp)
            pend_tokens_np = np.append(pend_tokens_np, int(gt_np[K - 1]))
            pend_hidden_slots_np = np.append(pend_hidden_slots_np, K - 1)
            pend_ttt_valid_np = np.append(pend_ttt_valid_np, K)

        # Keep pend state as numpy arrays (skip .tolist() round-trip).
        # Downstream code reads them via vectorized np ops; a single list
        # conversion happens only at the point of CPU→GPU transfer (below).
        pend_node_indices = pend_indices_np
        pend_cum_lps = pend_lps_np
        pend_source_blocks_np = np.zeros(pend_indices_np.shape[0], dtype=np.int64)
        pend_source_blocks = pend_source_blocks_np
        pend_input_ids_np = pend_tokens_np
        pend_hidden_slots = pend_hidden_slots_np
        pend_ttt_valid = pend_ttt_valid_np

        # Build GPU batch tensors from numpy (single CPU→GPU transfer for indices)
        if pend_indices_np.shape[0] > 0:
            N_pend = pend_indices_np.shape[0]
            # Pack slot indices + input ids into one numpy buffer, single transfer
            combined_np = np.empty(N_pend * 2, dtype=np.int64)
            combined_np[:N_pend]   = pend_hidden_slots_np
            combined_np[N_pend:]   = pend_tokens_np
            combined_t = torch.from_numpy(combined_np).to(device, non_blocking=True)
            pend_hidden_idx = combined_t[:N_pend]
            pend_input_ids = combined_t[N_pend:].unsqueeze(1)
            pend_hidden = draft_hidden[0, pend_hidden_idx, :].unsqueeze(1)  # [N_pend, 1, H]

            # TTT mask: vectorized construction via arange < valid_count broadcasting
            _ar_K_np = self._arange_K_np[:K] if self._arange_K_np is not None else np.arange(K, dtype=np.int64)
            mask_np = (_ar_K_np[None, :] < pend_ttt_valid_np[:, None])  # [N_pend, K]
            batch_ttt_mask = torch.from_numpy(mask_np).to(device, non_blocking=True)

            batch_ttt_kv = [
                (ttt_kv[l][0].expand(N_pend, -1, -1, -1),
                 ttt_kv[l][1].expand(N_pend, -1, -1, -1))
                for l in range(num_layers)
            ]

        if _profile_on:
            _tb1_end = _time.perf_counter_ns()
            _tree_profile['b1_pyloop'] = _tree_profile.get('b1_pyloop', 0) + (_tb1_end - _tb1_tolist) / 1e6
            _tree_profile['block1_init'] = _tree_profile.get('block1_init', 0) + (_tb1_end - _t0) / 1e6

        # ===== BFS expansion: iterate by depth up to max_blocks =====
        pend_depth = 1
        max_factor = 10  # branching topk factor (constant)

        # Switch K for BFS if tree_K_bfs is specified
        block1_K = K  # save for reference
        if self._tree_K_bfs is not None and self._tree_K_bfs < self.K:
            K = self._tree_K_bfs

        # Reuse cached arange tensor when possible (K equals the full self.K)
        if self._arange_K_t is not None and K == self.K:
            arange_K = self._arange_K_t
        else:
            arange_K = torch.arange(K, device=device)

        # --- Hoist invariants out of BFS loop (computed once) ---
        _bfs_give_up_class = self.rank_classes - 1
        _bfs_active_slot_topk = self._bfs_slot_topk if self._bfs_slot_topk is not None else self.RANK_SLOT_TOPK
        _bfs_rank_slot_topk_arr = np.array(_bfs_active_slot_topk, dtype=np.int64)
        _bfs_give_up_has_topk = _bfs_rank_slot_topk_arr[_bfs_give_up_class] > 0
        _bfs_arange_K_np = self._arange_K_np[:K] if self._arange_K_np is not None else np.arange(K, dtype=np.int64)
        _bfs_K_unchanged = (K == self.K)  # skip contiguous() if unchanged

        # Pre-compute cross cache slices once (invariant across BFS depths)
        # Exclude current position's K slots (added by block 1 forward with update_cross_cache=True)
        _model_K = self.K  # always use model's native K for cross cache
        effective_cross_count = draft_cache[0][2] - _model_K if draft_cache[0][2] >= _model_K else 0
        cross_cache_slices = []
        for layer_cache in draft_cache:
            if effective_cross_count > 0 and layer_cache[0] is not None:
                cross_cache_slices.append((
                    layer_cache[0][:, :, :effective_cross_count, :],
                    layer_cache[1][:, :, :effective_cross_count, :],
                ))
            else:
                cross_cache_slices.append(None)

        while len(pend_node_indices) > 0:
            if pend_depth >= iter_max_blocks:
                break

            N = len(pend_node_indices)

            # --- Cross cache expansion (expand pre-computed slices) ---
            batch_cross_cache = []
            for sl in cross_cache_slices:
                if sl is not None:
                    batch_cross_cache.append([
                        sl[0].expand(N, -1, -1, -1),
                        sl[1].expand(N, -1, -1, -1),
                        effective_cross_count,
                    ])
                else:
                    batch_cross_cache.append([None, None, 0])

            _bfs_idx = len(self._block_forward_events[pend_depth])
            blk_start, blk_end = self._get_bfs_event_pair(pend_depth, _bfs_idx)
            blk_start.record()

            _graph_outputs = None
            if self._draft_graph_cache is not None:
                try:
                    _graph_outputs = self._draft_graph_cache.run(
                        hidden=pend_hidden,
                        input_ids=pend_input_ids,
                        cross_cache_slices=cross_cache_slices,
                        effective_cross_count=effective_cross_count,
                        ttt_cache=batch_ttt_kv,
                        ttt_mask=batch_ttt_mask,
                        position_id=draft_position,
                    )
                except Exception:
                    # Out-of-bucket or other graph failure: fallback to eager.
                    _graph_outputs = None

            if _graph_outputs is not None:
                block_logits, block_rank_logits, block_draft_hidden, new_ttt_kv_batch = _graph_outputs
            else:
                # Eager path: build combined cross+ttt mask once, share across layers.
                if effective_cross_count > 0:
                    _N_cur = batch_ttt_mask.shape[0]
                    _cross_ones = batch_ttt_mask.new_ones(_N_cur, effective_cross_count)
                    _full_kv_mask = torch.cat([_cross_ones, batch_ttt_mask], dim=1)
                else:
                    _full_kv_mask = batch_ttt_mask

                block_logits, block_rank_logits, block_draft_hidden, new_ttt_kv_batch = \
                    self.draft_model.forward_with_cache(
                        hidden=pend_hidden,
                        input_ids=pend_input_ids,
                        cache=batch_cross_cache,
                        position_id=draft_position,
                        use_draft_condition=True,
                        ttt_cache=batch_ttt_kv,
                        ttt_mask=batch_ttt_mask,
                        update_cross_cache=False,
                        full_kv_mask=_full_kv_mask,
                    )
            # Slice to effective BFS K if set (skip entirely when K unchanged)
            if not _bfs_K_unchanged:
                block_logits = block_logits[:, :K, :].contiguous()
                block_rank_logits = block_rank_logits[:, :K, :].contiguous()
                block_draft_hidden = block_draft_hidden[:, :K, :].contiguous()
                new_ttt_kv_batch = [(k[:, :, :K, :].contiguous(), v[:, :, :K, :].contiguous()) for k, v in new_ttt_kv_batch]

            blk_end.record()
            self._block_forward_events[pend_depth].append((blk_start, blk_end))

            # --- Batch compute results (super-fused single compile dispatch) ---
            if _profile_on:
                _tb0 = _time.perf_counter_ns()
            d2t_offsets = self.draft_model.d2t  # [V_draft] offset mapping
            (all_rank_preds, all_greedy_tokens, all_greedy_target, all_greedy_lps,
             M_all, bf_all, _gu, top_target_all, top_lps_all) = _bfs_gpu_ops_fused(
                block_logits, block_rank_logits, d2t_offsets, max_factor,
                self.rank_classes, self._rank_to_factor,
            )

            # === FULL PATH: all blocks — branching + hitchhike + pending ===
            # (last block pending is discarded at next loop iteration via pend_depth >= max_blocks)

            # --- Batch TTT cache update (2*L ops, not N*2*L) ---
            new_block_masks = (
                arange_K.unsqueeze(0) < M_all.unsqueeze(1)
            )  # [N, K]
            combined_kv = [
                (torch.cat([batch_ttt_kv[l][0], new_ttt_kv_batch[l][0]], dim=2),
                 torch.cat([batch_ttt_kv[l][1], new_ttt_kv_batch[l][1]], dim=2))
                for l in range(num_layers)
            ]
            combined_mask = torch.cat([batch_ttt_mask, new_block_masks], dim=1)

            # --- Bulk GPU→CPU (non_blocking, 1 sync) ---
            if _profile_on:
                _tb1 = _time.perf_counter_ns()
                _tree_profile[f'bfs{pend_depth}_gpu_ops'] = _tree_profile.get(f'bfs{pend_depth}_gpu_ops', 0) + (_tb1 - _tb0) / 1e6
            _M_cpu = M_all.to('cpu', non_blocking=True)
            _bf_cpu = bf_all.to('cpu', non_blocking=True)
            _gt_cpu = all_greedy_target.to('cpu', non_blocking=True)
            _glp_cpu = all_greedy_lps.to('cpu', non_blocking=True)
            _rp_cpu = all_rank_preds.to('cpu', non_blocking=True)
            _tt_cpu = top_target_all.to('cpu', non_blocking=True)
            _tl_cpu = top_lps_all.to('cpu', non_blocking=True)
            torch.cuda.synchronize()
            M_np = _M_cpu.numpy()
            bf_np = _bf_cpu.numpy()
            gt_all_np = _gt_cpu.numpy()
            glp_all_np = _glp_cpu.numpy()
            rp_all_np = _rp_cpu.numpy()
            tt_all_np = _tt_cpu.numpy()
            tl_all_np = _tl_cpu.numpy()

            if _profile_on:
                _tb2 = _time.perf_counter_ns()
                _tree_profile[f'bfs{pend_depth}_tolist'] = _tree_profile.get(f'bfs{pend_depth}_tolist', 0) + (_tb2 - _tb1) / 1e6

            # --- Vectorized greedy chains: N*K nodes (or M-truncated) ---
            tree_start = n_nodes
            NK = N * K
            # Lazily cache larger arange-NK buffer on self; grows as N grows.
            _ar_NK = getattr(self, '_arange_NK_np', None)
            if _ar_NK is None or len(_ar_NK) < NK:
                _ar_NK = np.arange(max(NK, 64), dtype=np.int64)
                self._arange_NK_np = _ar_NK
            n_nodes, _greedy_map = _vectorized_greedy_chain(
                n_nodes, N, K, pend_node_indices, pend_cum_lps,
                gt_all_np, glp_all_np, rp_all_np, pend_depth,
                tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots,
                arange_K_np=self._arange_K_np, arange_NK_np=_ar_NK,
            )

            # --- Multi-slot branching + pending (numpy vectorized) ---
            base_rank_stats = len(all_rank_stats)

            for i in range(N):
                all_rank_stats.append({
                    'rank_preds': rp_all_np[i].tolist(),
                    'M': int(M_np[i]),
                    'branch': int(bf_np[i]),
                    'parent_block': pend_source_blocks[i],
                })

            # Stay in numpy if pend state is already ndarray (fast path).
            if isinstance(pend_node_indices, np.ndarray):
                pend_parent_np = pend_node_indices
            else:
                pend_parent_np = np.asarray(pend_node_indices, dtype=np.int64)
            if isinstance(pend_cum_lps, np.ndarray):
                pend_lp_np = pend_cum_lps
            else:
                pend_lp_np = np.asarray(pend_cum_lps, dtype=np.float64)

            # --- Find all active (leaf, slot) branch pairs ---
            # Use hoisted invariants (computed once before BFS loop)
            give_up_class_val = _bfs_give_up_class
            rank_slot_topk_arr = _bfs_rank_slot_topk_arr

            if _bfs_give_up_has_topk:
                # give_up still gets candidates → no stop_at truncation
                stop_at_np = np.full(N, K, dtype=np.int64)
            else:
                # Original: stop at first give_up slot
                give_up_mask_np  = (rp_all_np == give_up_class_val)              # [N, K]
                has_give_up_np   = give_up_mask_np.any(axis=1)                   # [N]
                stop_at_np       = np.where(has_give_up_np,
                                            give_up_mask_np.argmax(axis=1), K)   # [N]

            slot_in_range = _bfs_arange_K_np[None, :] < stop_at_np[:, None]      # [N, K]
            # Active: has topk > 1 for its rank AND within range
            slot_topks_per_rank = rank_slot_topk_arr[rp_all_np].copy()      # [N, K]

            # BFS-level adaptive: cut alts for high-confidence slots in block 2+
            # Same logic as block 1 ADAPTIVE_ALL but applied to all pending×slot pairs.
            if self._adaptive_all:
                # glp_all_np shape [N, K]: greedy log probs per pending leaf per slot
                _pk_np = np.exp(glp_all_np)  # [N, K] probabilities
                base_arr = slot_topks_per_rank  # existing rank-based counts
                if self._adaptive_all == 2:  # aggressive
                    # p>=0.9 → 1 (no alt), 0.7-0.9 → 2, 0.5-0.7 → 3, <0.5 → base
                    cut_1 = (_pk_np >= 0.9)
                    cut_2 = (_pk_np >= 0.7) & (_pk_np < 0.9)
                    cut_3 = (_pk_np >= 0.5) & (_pk_np < 0.7)
                    slot_topks_per_rank = np.where(cut_1, 1, slot_topks_per_rank)
                    slot_topks_per_rank = np.where(cut_2, np.minimum(2, base_arr), slot_topks_per_rank)
                    slot_topks_per_rank = np.where(cut_3, np.minimum(3, base_arr), slot_topks_per_rank)
                else:  # 1 = conservative
                    cut_a = (_pk_np >= 0.95)
                    cut_b = (_pk_np >= 0.85) & (_pk_np < 0.95)
                    slot_topks_per_rank = np.where(cut_a, np.minimum(2, base_arr), slot_topks_per_rank)
                    slot_topks_per_rank = np.where(cut_b, np.minimum(3, base_arr), slot_topks_per_rank)

            slot_active   = (slot_topks_per_rank > 1) & slot_in_range        # [N, K]
            pair_leaves, pair_slots = np.nonzero(slot_active)                # [n_pairs]
            n_pairs = len(pair_leaves)

            pair_ranks     = rp_all_np[pair_leaves, pair_slots]              # [n_pairs]
            pair_topks     = slot_topks_per_rank[pair_leaves, pair_slots]    # [n_pairs]  (was rank-based, now possibly adaptive-cut)
            n_alt_per_pair = np.maximum(pair_topks - 1, 0)                   # [n_pairs]
            total_alts     = int(n_alt_per_pair.sum())

            # --- Add alternatives to tree buffer ---
            if total_alts > 0:
                s = n_nodes
                pair_exp = np.repeat(np.arange(n_pairs, dtype=np.int64), n_alt_per_pair)
                # alt_within[i] = (i - prev_cum[pair_of_i]) + 1; vectorized (avoids
                # per-pair Python loop + np.arange/concat alloc overhead).
                cum_alts = np.cumsum(n_alt_per_pair)
                prev_cum = np.empty(n_pairs, dtype=np.int64)
                prev_cum[0] = 0
                if n_pairs > 1:
                    prev_cum[1:] = cum_alts[:-1]
                flat_idx = np.arange(total_alts, dtype=np.int64)
                alt_within = flat_idx - prev_cum[pair_exp] + 1

                leaf_exp = pair_leaves[pair_exp]
                slot_exp = pair_slots[pair_exp]

                base_for_leaf = tree_start + leaf_exp * K
                bp_idx = np.where(slot_exp > 0,
                                  base_for_leaf + slot_exp - 1,
                                  pend_parent_np[leaf_exp])
                bp_lp = np.where(slot_exp > 0,
                                 tn_lps[np.maximum(base_for_leaf + slot_exp - 1, 0)],
                                 pend_lp_np[leaf_exp])

                tn_tokens[s:s+total_alts]  = tt_all_np[leaf_exp, slot_exp, alt_within]
                tn_parents[s:s+total_alts] = bp_idx
                tn_lps[s:s+total_alts]     = bp_lp + tl_all_np[leaf_exp, slot_exp, alt_within]
                tn_ranks[s:s+total_alts]   = rp_all_np[leaf_exp, slot_exp]
                tn_blocks[s:s+total_alts]  = pend_depth
                tn_slots[s:s+total_alts]   = slot_exp
                n_nodes += total_alts

            # --- Hitchhike g(K-1): leaves with stop_at==K and K-1 not already a branch slot ---
            last_active_per_leaf = np.full(N, -1, dtype=np.int64)
            if n_pairs > 0:
                np.maximum.at(last_active_per_leaf, pair_leaves, pair_slots)
            hitch_needed = (stop_at_np == K) & (last_active_per_leaf != K - 1)
            hitch_leaves = np.nonzero(hitch_needed)[0]
            n_hitch      = len(hitch_leaves)

            # --- Build pending arrays ---
            # [g(slot_k) per pair] + [alternatives] + [hitchhike g(K-1)]
            # Skip entire pending construction + beam prune + next batch transfer
            # on the last BFS iter — next iter's while-check would break anyway.
            _is_last_bfs_iter = (pend_depth + 1 >= iter_max_blocks)
            n_total_pend = 0 if _is_last_bfs_iter else (n_pairs + total_alts + n_hitch)

            if n_total_pend > 0:
                next_node_indices_np  = np.empty(n_total_pend, dtype=np.int64)
                next_cum_lps_np       = np.empty(n_total_pend, dtype=np.float64)
                next_parent_batch_np  = np.empty(n_total_pend, dtype=np.int64)
                next_hidden_si_np     = np.empty(n_total_pend, dtype=np.int64)
                next_input_tokens_np  = np.empty(n_total_pend, dtype=np.int64)
                next_ttt_valid_np     = np.empty(n_total_pend, dtype=np.int64)

                off = 0
                # g(slot_k) for each active pair
                if n_pairs > 0:
                    g_slot_nodes = tree_start + pair_leaves * K + pair_slots
                    next_node_indices_np[off:off+n_pairs]  = g_slot_nodes
                    next_cum_lps_np[off:off+n_pairs]       = tn_lps[g_slot_nodes]
                    next_parent_batch_np[off:off+n_pairs]  = pair_leaves
                    next_hidden_si_np[off:off+n_pairs]     = pair_slots
                    next_input_tokens_np[off:off+n_pairs]  = gt_all_np[pair_leaves, pair_slots]
                    next_ttt_valid_np[off:off+n_pairs]     = pair_slots + 1
                    off += n_pairs

                # Alternatives
                if total_alts > 0:
                    alt_tree_idx = np.arange(n_nodes - total_alts, n_nodes)
                    next_node_indices_np[off:off+total_alts]  = alt_tree_idx
                    next_cum_lps_np[off:off+total_alts]       = tn_lps[alt_tree_idx]
                    next_parent_batch_np[off:off+total_alts]  = leaf_exp
                    next_hidden_si_np[off:off+total_alts]     = slot_exp
                    next_input_tokens_np[off:off+total_alts]  = tt_all_np[leaf_exp, slot_exp, alt_within]
                    next_ttt_valid_np[off:off+total_alts]     = slot_exp + 1
                    off += total_alts

                # Hitchhike g(K-1)
                if n_hitch > 0:
                    hitch_nodes = tree_start + hitch_leaves * K + K - 1
                    next_node_indices_np[off:off+n_hitch]  = hitch_nodes
                    next_cum_lps_np[off:off+n_hitch]       = tn_lps[hitch_nodes]
                    next_parent_batch_np[off:off+n_hitch]  = hitch_leaves
                    next_hidden_si_np[off:off+n_hitch]     = K - 1
                    next_input_tokens_np[off:off+n_hitch]  = gt_all_np[hitch_leaves, K - 1]
                    next_ttt_valid_np[off:off+n_hitch]     = K
                    off += n_hitch

                next_hidden_bi_np      = next_parent_batch_np
                next_source_blocks_np  = base_rank_stats + next_parent_batch_np

                # Keep everything as numpy — avoid N*8 .tolist() conversions per BFS iter.
                next_node_indices  = next_node_indices_np
                next_cum_lps       = next_cum_lps_np
                next_parent_batch  = next_parent_batch_np
                next_hidden_bi     = next_hidden_bi_np
                next_hidden_si     = next_hidden_si_np
                next_input_tokens  = next_input_tokens_np
                next_ttt_valid     = next_ttt_valid_np
                next_source_blocks = next_source_blocks_np
            else:
                _empty_i = np.empty(0, dtype=np.int64)
                _empty_f = np.empty(0, dtype=np.float64)
                next_node_indices  = _empty_i
                next_cum_lps       = _empty_f
                next_parent_batch  = _empty_i
                next_hidden_bi     = _empty_i
                next_hidden_si     = _empty_i
                next_input_tokens  = _empty_i
                next_ttt_valid     = _empty_i
                next_source_blocks = _empty_i

            if _profile_on:
                _tb3 = _time.perf_counter_ns()
                _tree_profile[f'bfs{pend_depth}_pyloop'] = _tree_profile.get(f'bfs{pend_depth}_pyloop', 0) + (_tb3 - _tb2) / 1e6

            # --- Beam pruning (numpy argpartition, O(N)) ---
            # Adaptive beam: limit pending leaves based on remaining budget to avoid
            # computing nodes that will be pruned anyway (saves GPU forward compute).
            remaining_budget = self.total_tokens - n_nodes
            adaptive_beam = min(self.beam_width, max(1, remaining_budget // K))
            if len(next_node_indices) > adaptive_beam:
                # next_cum_lps / next_node_indices are numpy (see fast path above).
                next_cum_lps_np = np.asarray(next_cum_lps)
                # Compute each pending leaf's depth in tree (for stratification/protection)
                nni_arr = np.asarray(next_node_indices, dtype=np.int64)
                pend_depths_arr = None
                protected_idx = None
                if self._protect_d1 and pend_depth == 1:
                    d1_mask = (tn_blocks[nni_arr] == 0) & (tn_slots[nni_arr] == 0)
                    protected_idx = np.nonzero(d1_mask)[0]
                if self._strat_beam > 0 and pend_depth == 1:
                    # Compute depth of each pending leaf in the tree by walking parent chain.
                    # For block-1 pending, depth = tn_slots + 1 (since tn_blocks=0).
                    pend_depths_arr = tn_slots[nni_arr].astype(np.int64) + 1
                if self._diverse_beam and len(next_input_tokens) > 0:
                    # Diversity-aware: ensure at least one representative per unique predicted token
                    next_input_np = np.asarray(next_input_tokens)
                    unique_tokens = np.unique(next_input_np)
                    keep_set = set()
                    # Best candidate per unique token
                    for tok in unique_tokens:
                        tok_mask = (next_input_np == tok)
                        tok_indices = np.nonzero(tok_mask)[0]
                        best = tok_indices[np.argmax(next_cum_lps_np[tok_indices])]
                        keep_set.add(int(best))
                    # Fill remaining with best by cum_lp
                    if len(keep_set) < adaptive_beam:
                        remaining = np.array([i for i in range(len(next_node_indices)) if i not in keep_set])
                        if len(remaining) > 0:
                            n_fill = adaptive_beam - len(keep_set)
                            if n_fill > len(remaining):
                                n_fill = len(remaining)
                            fill = remaining[np.argpartition(-next_cum_lps_np[remaining], min(n_fill, len(remaining)-1))[:n_fill]]
                            keep_set.update(fill.tolist())
                    elif len(keep_set) > adaptive_beam:
                        # More unique tokens than beam - keep best by cum_lp
                        keep_arr = np.array(list(keep_set))
                        keep_arr = keep_arr[np.argpartition(-next_cum_lps_np[keep_arr], adaptive_beam)[:adaptive_beam]]
                        keep_set = set(keep_arr.tolist())
                    keep = np.array(sorted(keep_set))
                elif self._hidden_slot_cap > 0 and pend_depth == 1:
                    # Hidden-slot cap: limit pending per source slot (block 1 hidden index).
                    # Forces hidden diversity in block 2 forward (different slots → different
                    # hidden states → more diverse predictions).
                    nni_arr = np.asarray(next_node_indices, dtype=np.int64)
                    source_slots = tn_slots[nni_arr]  # block 1 source slot
                    cap = self._hidden_slot_cap
                    keep_set = set()
                    for s in np.unique(source_slots):
                        s_idx = np.nonzero(source_slots == s)[0]
                        if len(s_idx) <= cap:
                            keep_set.update(s_idx.tolist())
                        else:
                            best = s_idx[np.argpartition(-next_cum_lps_np[s_idx], cap)[:cap]]
                            keep_set.update(best.tolist())
                    # Fill remaining with global top cum_lp
                    if len(keep_set) < adaptive_beam:
                        remaining = np.array([i for i in range(len(next_node_indices)) if i not in keep_set], dtype=np.int64)
                        n_fill = adaptive_beam - len(keep_set)
                        if n_fill > 0 and len(remaining) > 0:
                            if n_fill >= len(remaining):
                                keep_set.update(remaining.tolist())
                            else:
                                fill = remaining[np.argpartition(-next_cum_lps_np[remaining], n_fill)[:n_fill]]
                                keep_set.update(fill.tolist())
                    if len(keep_set) > adaptive_beam:
                        keep_arr = np.array(list(keep_set))
                        keep_arr = keep_arr[np.argpartition(-next_cum_lps_np[keep_arr], adaptive_beam)[:adaptive_beam]]
                        keep_set = set(keep_arr.tolist())
                    keep = np.array(sorted(keep_set))
                elif self._strat_beam > 0 and pend_depths_arr is not None:
                    # Depth-stratified beam: reserve self._strat_beam slots per depth,
                    # fill rest globally by cum_lp. Ensures all depth layers get continuation.
                    per_depth = int(self._strat_beam)
                    keep_set = set()
                    unique_depths = np.unique(pend_depths_arr)
                    for d in unique_depths:
                        d_mask = (pend_depths_arr == d)
                        d_indices = np.nonzero(d_mask)[0]
                        if len(d_indices) == 0: continue
                        n_keep = min(per_depth, len(d_indices))
                        d_scores = next_cum_lps_np[d_indices]
                        if n_keep >= len(d_indices):
                            best_d = d_indices.tolist()
                        else:
                            best_d = d_indices[np.argpartition(-d_scores, n_keep)[:n_keep]].tolist()
                        keep_set.update(best_d)
                    # Fill remaining beam globally by cum_lp
                    if len(keep_set) < adaptive_beam:
                        remaining = np.array([i for i in range(len(next_node_indices)) if i not in keep_set], dtype=np.int64)
                        n_fill = adaptive_beam - len(keep_set)
                        if n_fill > 0 and len(remaining) > 0:
                            if n_fill >= len(remaining):
                                keep_set.update(remaining.tolist())
                            else:
                                fill = remaining[np.argpartition(-next_cum_lps_np[remaining], n_fill)[:n_fill]]
                                keep_set.update(fill.tolist())
                    # Truncate if too many
                    if len(keep_set) > adaptive_beam:
                        keep_arr = np.array(list(keep_set))
                        keep_arr = keep_arr[np.argpartition(-next_cum_lps_np[keep_arr], adaptive_beam)[:adaptive_beam]]
                        keep_set = set(keep_arr.tolist())
                    keep = np.array(sorted(keep_set))
                elif protected_idx is not None and len(protected_idx) > 0:
                    # Reserve up to N slots for best d1 leaves (by cum_lp among d1 leaves),
                    # then fill the rest globally by cum_lp. Without reservation, d1 leaves
                    # lose to greedy-chain leaves every time.
                    n_d1_reserve = int(os.environ.get('PROTECT_D1_N', '3'))
                    n_d1_reserve = min(n_d1_reserve, len(protected_idx), adaptive_beam // 2)
                    if n_d1_reserve > 0:
                        prot_scores = next_cum_lps_np[protected_idx]
                        if n_d1_reserve >= len(protected_idx):
                            best_prot = protected_idx.tolist()
                        else:
                            best_prot = protected_idx[np.argpartition(-prot_scores, n_d1_reserve)[:n_d1_reserve]].tolist()
                        keep_set = set(best_prot)
                    else:
                        keep_set = set()
                    # Fill remaining globally by cum_lp
                    remaining = np.array([i for i in range(len(next_node_indices)) if i not in keep_set], dtype=np.int64)
                    n_fill = adaptive_beam - len(keep_set)
                    if n_fill > 0 and len(remaining) > 0:
                        if n_fill >= len(remaining):
                            keep_set.update(remaining.tolist())
                        else:
                            fill = remaining[np.argpartition(-next_cum_lps_np[remaining], n_fill)[:n_fill]]
                            keep_set.update(fill.tolist())
                    keep = np.array(sorted(keep_set))
                else:
                    keep = np.argpartition(-next_cum_lps_np, adaptive_beam)[:adaptive_beam]
                # Numpy fancy-index the 8 parallel arrays in one go.
                next_node_indices = next_node_indices[keep]
                next_cum_lps = next_cum_lps[keep]
                next_source_blocks = next_source_blocks[keep]
                next_parent_batch = next_parent_batch[keep]
                next_hidden_bi = next_hidden_bi[keep]
                next_hidden_si = next_hidden_si[keep]
                next_input_tokens = next_input_tokens[keep]
                next_ttt_valid    = next_ttt_valid[keep]


            # --- Construct next pending batch (single CPU→GPU transfer) ---
            if len(next_node_indices) > 0:
                N_pend = len(next_node_indices)
                # Pack all 5 int64 arrays into one numpy buffer, single transfer
                combined_np = np.empty(N_pend * 5, dtype=np.int64)
                combined_np[0:N_pend]           = next_parent_batch
                combined_np[N_pend:2*N_pend]    = next_hidden_bi
                combined_np[2*N_pend:3*N_pend]  = next_hidden_si
                combined_np[3*N_pend:4*N_pend]  = next_input_tokens
                combined_np[4*N_pend:5*N_pend]  = next_ttt_valid
                combined_t = torch.from_numpy(combined_np).to(device, non_blocking=True)
                parent_batch_t   = combined_t[0:N_pend]
                hidden_bi_t      = combined_t[N_pend:2*N_pend]
                hidden_si_t      = combined_t[2*N_pend:3*N_pend]
                pend_input_ids   = combined_t[3*N_pend:4*N_pend].unsqueeze(1)
                pend_ttt_valid_t = combined_t[4*N_pend:5*N_pend]

                batch_ttt_kv = [
                    (combined_kv[l][0][parent_batch_t], combined_kv[l][1][parent_batch_t])
                    for l in range(num_layers)
                ]
                batch_ttt_mask = combined_mask[parent_batch_t]

                # Per-pending ttt_valid: override new-block portion of mask
                # Each pending entry only sees slots 0..ttt_valid-1 from the new block
                # Reuse cached arange_K (already matches BFS K from above).
                batch_ttt_mask[:, -K:] = arange_K.unsqueeze(0) < pend_ttt_valid_t.unsqueeze(1)

                pend_hidden = block_draft_hidden[hidden_bi_t, hidden_si_t].unsqueeze(1)

                pend_node_indices = next_node_indices
                pend_cum_lps = next_cum_lps
                pend_source_blocks = next_source_blocks
            else:
                pend_node_indices = np.empty(0, dtype=np.int64)

            if _profile_on:
                _tb4 = _time.perf_counter_ns()
                _tree_profile[f'bfs{pend_depth}_next_batch'] = _tree_profile.get(f'bfs{pend_depth}_next_batch', 0) + (_tb4 - _tb3) / 1e6
            pend_depth += 1

        if _profile_on:
            _bfs_end = _time.perf_counter_ns()
            _tree_profile['bfs_total'] = (_bfs_end - _t0) / 1e6 - _tree_profile['block1_init']

        # ===== Post-expansion pruning =====
        budget = self.total_tokens
        if n_nodes > budget:
            n_nodes = self._prune_tree_np(
                n_nodes, tn_tokens, tn_parents, tn_lps, tn_ranks, tn_blocks, tn_slots, budget
            )

        # ===== Convert tree buffers (Triton kernels) =====
        tree_tokens, tree_mask, tree_position_ids, retrieve_indices = \
            build_tree_buffers_triton(n_nodes, tn_tokens, tn_parents, sample_token, device, K, self.max_blocks)

        if _profile_on:
            _tree_profile['post_bfs'] = _tree_profile.get('post_bfs', 0) + (_time.perf_counter_ns() - _bfs_end) / 1e6
            _tree_profile['total'] = _tree_profile.get('total', 0) + (_time.perf_counter_ns() - _t0) / 1e6
            self._last_tree_profile = _tree_profile
            if not hasattr(self, '_tp_accum'):
                self._tp_accum = {}
                self._tp_count = 0
            for _k, _v in _tree_profile.items():
                self._tp_accum[_k] = self._tp_accum.get(_k, 0) + _v
            self._tp_count += 1
            if self._tp_count % 100 == 0:
                print(f'[PROF] after {self._tp_count} iters (avg ms/iter):')
                for _k, _v in sorted(self._tp_accum.items(), key=lambda x: -x[1]):
                    print(f'  {_k:<30s}: {_v/self._tp_count:.3f} ms')

        # Convert numpy tree metadata to Python lists for UI/stats
        node_ranks = tn_ranks[:n_nodes].tolist()
        node_block_slots = list(zip(tn_blocks[:n_nodes].tolist(), tn_slots[:n_nodes].tolist()))

        # Restore original total_tokens and beam_width if iter-adaptive was used
        if self._iter_adapt_budget > 0:
            self.total_tokens = _orig_total_tokens
            self.beam_width = _orig_beam_width

        return tree_tokens, tree_mask, tree_position_ids, retrieve_indices, all_rank_stats, node_ranks, node_block_slots

    # ============================================================
    #    Tree Verification (Eagle3 style)
    # ============================================================

    def _tree_verify(
        self,
        draft_tokens: torch.Tensor,       # [1, N+1]
        tree_mask: torch.Tensor,           # [1, 1, N+1, N+1]
        tree_position_ids: torch.Tensor,   # [N+1]
        retrieve_indices: torch.Tensor,    # [num_paths, max_depth+1]
        input_ids: torch.Tensor,           # [1, seq_len]
        past_key_values,
        temperature: float,
    ):
        """Verify tree with target model (single forward).

        Returns:
            accept_length: int
            accepted_tokens: list[int]
            next_token: [1,1] tensor
            past_key_values: updated/cropped
            new_hidden_3h: for next iteration's hidden
            sample_p: probability for bonus token
        """
        position_ids = tree_position_ids + input_ids.shape[1]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        # Build 4D attention mask for tree attention (preallocated buffer reuse)
        prefix_len = get_cache_len(past_key_values)
        N_plus_1 = draft_tokens.shape[1]
        device = draft_tokens.device
        total_len = prefix_len + N_plus_1
        min_val = float("-inf")

        # Preallocate / grow mask buffer once and reuse across iterations
        if (self._verify_mask_buf is None
            or self._verify_mask_buf.shape[2] < N_plus_1
            or self._verify_mask_buf.shape[3] < total_len):
            new_n = max(N_plus_1, 256)
            new_t = max(total_len, prefix_len + 256, 4096)
            self._verify_mask_buf = torch.full(
                (1, 1, new_n, new_t), min_val,
                device=device, dtype=torch.bfloat16,
            )

        full_attn_mask = self._verify_mask_buf[:, :, :N_plus_1, :total_len]
        full_attn_mask.fill_(min_val)
        full_attn_mask[..., :prefix_len] = 0.0
        full_attn_mask[..., prefix_len:][tree_mask.bool()] = 0.0

        # Target forward with tree attention.
        # Env-gated SDP backend selection: default (auto-select by PyTorch based on mask
        # type — math for custom bias) is slowest. Force mem_efficient_attention for bias.
        import os as _os
        _sdp_backend = _os.environ.get('TARGET_SDP_BACKEND', '').lower()
        if _sdp_backend in ('efficient', 'mem_efficient', 'memory_efficient'):
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                target_outputs = self.target_model(
                    draft_tokens, past_key_values=past_key_values,
                    position_ids=position_ids, attention_mask=full_attn_mask,
                    output_hidden_states=True, use_cache=True,
                )
        elif _sdp_backend == 'flash':
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                target_outputs = self.target_model(
                    draft_tokens, past_key_values=past_key_values,
                    position_ids=position_ids, attention_mask=full_attn_mask,
                    output_hidden_states=True, use_cache=True,
                )
        else:
            target_outputs = self.target_model(
                draft_tokens,
                past_key_values=past_key_values,
                position_ids=position_ids,
                attention_mask=full_attn_mask,
                output_hidden_states=True,
                use_cache=True,
            )

        tree_logits = target_outputs.logits  # [1, N+1, vocab]
        past_key_values = target_outputs.past_key_values
        new_hidden_states = target_outputs.hidden_states

        # Extract logits for each path
        logits = tree_logits[0, retrieve_indices]  # [num_paths, max_depth+1, vocab]

        # Build candidates for each path
        padding = torch.tensor([-1], dtype=torch.long, device=draft_tokens.device)
        draft_tokens_ext = torch.cat([draft_tokens[0], padding], dim=0)
        candidates = draft_tokens_ext[retrieve_indices]  # [num_paths, max_depth+1]

        # Greedy verification
        if temperature <= 1e-5:
            target_argmax = torch.argmax(logits[:, :-1], dim=-1)  # [num_paths, max_depth] — reused below
            # Optional: relaxed top-K acceptance. Accept draft token if it's in
            # target's top-K at that position (not just argmax). Trades generation
            # fidelity for acc_len gain. ACCEPT_TOPK=1 = strict (default), >1 = relaxed.
            _accept_topk = int(os.environ.get('ACCEPT_TOPK', '1'))
            if _accept_topk > 1:
                target_topk_idx = torch.topk(logits[:, :-1], _accept_topk, dim=-1).indices  # [paths, depth, K]
                cand_t = candidates[:, 1:].to(logits.device)  # [paths, depth]
                # cand_t in any of top-K?
                posterior_mask = (cand_t.unsqueeze(-1) == target_topk_idx).any(dim=-1).int()
            else:
                posterior_mask = (
                    candidates[:, 1:].to(logits.device) == target_argmax
                ).int()
            candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            sample_p = logits[best_candidate, accept_length]

            # Online logit correction: update running stats of target vs draft preferences.
            # For each (path, depth) position: target_argmax[p,d] is what target wanted;
            # candidates[p, d+1] is what draft proposed. Aggregate counts to adjust
            # draft's future logits toward target's preferred tokens.
            if self._logit_correct_on:
                _V = tree_logits.shape[-1]
                if self._target_want_count is None:
                    dev = tree_logits.device
                    self._target_want_count = torch.zeros(_V, device=dev, dtype=torch.float32)
                    self._draft_want_count = torch.zeros(_V, device=dev, dtype=torch.float32)
                # target argmax token counts
                tw = target_argmax.flatten()  # [num_paths*depth]
                dw = candidates[:, 1:].flatten().to(tw.device)
                # Ignore padding (-1) from candidates
                valid_mask = dw >= 0
                tw_valid = tw[valid_mask]
                dw_valid = dw[valid_mask]
                self._target_want_count.scatter_add_(
                    0, tw_valid, torch.ones_like(tw_valid, dtype=torch.float32),
                )
                self._draft_want_count.scatter_add_(
                    0, dw_valid, torch.ones_like(dw_valid, dtype=torch.float32),
                )
        else:
            # Speculative sampling: walk all candidate paths (EAGLE-style)
            accept_length = 1
            accept_cand = candidates[0][:1]
            best_candidate = 0
            adjustflag = False
            for i in range(1, candidates.shape[1]):
                if i != accept_length:
                    break
                adjustflag = False
                # Find all paths whose prefix matches the accepted sequence so far
                is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
                fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
                gt_logits = logits[fi, i - 1]
                gtp = torch.softmax(gt_logits / temperature, dim=0)
                candidates_set = []
                for j in range(candidates.shape[0]):
                    if is_eq[j]:
                        x = candidates[j, i]
                        xi = x.item()
                        if xi in candidates_set or xi == -1:
                            continue
                        candidates_set.append(xi)
                        r = random.random()
                        px = gtp[xi]
                        qx = 1.0
                        acp = px / qx
                        if r <= acp:
                            accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                            accept_length += 1
                            best_candidate = j
                            break
                        else:
                            gtp[xi] = 0
                            gtp_sum = gtp.sum()
                            if gtp_sum > 0:
                                gtp = gtp / gtp_sum
                            adjustflag = True
            if adjustflag and accept_length != candidates.shape[1]:
                sample_p = gtp
            else:
                gt_logits = logits[best_candidate, accept_length - 1]
                sample_p = torch.softmax(gt_logits / temperature, dim=0)
            accept_length = accept_length - 1
            best_candidate = torch.tensor(best_candidate, dtype=torch.long, device=candidates.device)

        accept_length_val = accept_length.item() if isinstance(accept_length, torch.Tensor) else accept_length

        # Extract accepted token indices from tree (all on GPU, no .item() calls)
        path_indices = retrieve_indices[best_candidate, 1:accept_length_val + 1]  # [accept_length]
        path_indices = path_indices[path_indices >= 0]
        accepted_token_ids = draft_tokens[0, path_indices]  # [accept_length] tensor on GPU

        # Sample next token
        if temperature > 0:
            # In sampling mode, sample_p is already a probability distribution
            if sample_p.sum() > 0:
                next_token = torch.multinomial(sample_p.unsqueeze(0), num_samples=1)
            else:
                # Fallback: re-derive from logits when all probs zeroed out
                gt_logits = logits[best_candidate, accept_length_val]
                probs = F.softmax(gt_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs.unsqueeze(0), num_samples=1)
        else:
            next_token = torch.argmax(sample_p).unsqueeze(0).unsqueeze(0)

        # Crop KV cache: keep prefix + accepted path entries
        prev_input_len = input_ids.shape[1]
        select_indices = retrieve_indices[best_candidate, :accept_length_val + 1]
        select_indices = select_indices[select_indices >= 0]
        select_indices = select_indices + prev_input_len

        # Reorder KV cache
        if isinstance(past_key_values, DynamicCache):
            for layer in past_key_values.layers:
                k, v = layer.keys, layer.values
                tgt_k = k[..., select_indices, :]
                tgt_v = v[..., select_indices, :]
                k[..., prev_input_len:prev_input_len + tgt_k.shape[-2], :] = tgt_k
                v[..., prev_input_len:prev_input_len + tgt_v.shape[-2], :] = tgt_v
            past_key_values.crop(prev_input_len + len(select_indices))
        else:
            new_kv = []
            for k, v in past_key_values:
                tgt_k = k[..., select_indices, :]
                tgt_v = v[..., select_indices, :]
                new_kv.append((
                    torch.cat([k[..., :prev_input_len, :], tgt_k], dim=-2),
                    torch.cat([v[..., :prev_input_len, :], tgt_v], dim=-2),
                ))
            past_key_values = tuple(new_kv)

        # Lazy hidden state extraction (all on GPU, no .item())
        if accept_length_val > 0:
            last_accepted_idx = retrieve_indices[best_candidate, accept_length_val]
            accepted_indices = retrieve_indices[best_candidate, :accept_length_val]
            accepted_indices = accepted_indices[accepted_indices >= 0]
            select_hidden_indices = torch.cat([
                accepted_indices,
                last_accepted_idx.unsqueeze(0),
            ])
            lazy_hidden_3h = torch.cat(
                [new_hidden_states[i][:, select_hidden_indices, :]
                 for i in self.hidden_layer_indices],
                dim=-1,
            )  # [1, accept_length+1, 3H]
            last_hidden_3h = lazy_hidden_3h[:, -1:, :]
        else:
            last_accepted_idx = retrieve_indices[best_candidate, 0]
            last_hidden_3h = torch.cat(
                [new_hidden_states[i][:, last_accepted_idx:last_accepted_idx + 1, :]
                 for i in self.hidden_layer_indices],
                dim=-1,
            )  # [1, 1, 3H]
            lazy_hidden_3h = None

        # Non-blocking CPU transfer of target choices for coverage stats (no hot-path overhead)
        if temperature <= 1e-5:
            target_choices_cpu = target_argmax.to('cpu', non_blocking=True)  # [num_paths, max_depth]
        else:
            target_choices_cpu = torch.argmax(logits[:, :-1], dim=-1).to('cpu', non_blocking=True)

        # Phase 4D v6: lazy top-K. Keep GPU ref to raw tree_logits[0]; hook
        # computes top-K on the single first_reject_node row (saves ~N_nodes×
        # cost of eager compute). Zero overhead when hooks=None.
        # Phase 4D v8: additionally skip storage when hooks signal not to collect
        # (warmup or max_events reached) — removes even the .detach() call.
        _hooks = getattr(self, '_adapt_hooks', None)
        if _hooks is not None and _hooks.should_collect_signals():
            self._adapt_tree_logits_gpu = tree_logits[0].detach()  # [N+1, V] GPU ref
        else:
            self._adapt_tree_logits_gpu = None

        return (
            accept_length_val,
            accepted_token_ids,
            next_token,
            past_key_values,
            last_hidden_3h,
            lazy_hidden_3h,
            best_candidate,
            retrieve_indices,
            target_choices_cpu,
        )

    # ============================================================
    #    Tree Path Extraction for UI Visualization
    # ============================================================

    def _extract_tree_paths(
        self,
        draft_tokens: torch.Tensor,       # [1, N+1]
        retrieve_indices: torch.Tensor,    # [num_leaves, max_depth+1]
        best_candidate: int,
        accept_length: int,
        node_ranks: Optional[list] = None,
        node_block_slots: Optional[list] = None,
    ) -> List[Dict]:
        """Extract all candidate paths from tree for visualization.

        Returns:
            List of path dicts with path_idx, tokens, texts, is_selected, ranks, block_slots
        """
        all_paths = []
        num_leaves = retrieve_indices.shape[0]

        for lid in range(num_leaves):
            path_indices = retrieve_indices[lid]
            path_token_ids = []
            path_ranks = []
            path_block_slots = []
            for idx in path_indices:
                idx_val = idx.item()
                if idx_val < 0:
                    break
                path_token_ids.append(draft_tokens[0, idx_val].item())
                # idx_val=0 is root (no rank), idx_val>0 maps to node_ranks[idx_val-1]
                if node_ranks is not None and idx_val > 0:
                    path_ranks.append(node_ranks[idx_val - 1])
                else:
                    path_ranks.append(None)
                if node_block_slots is not None and idx_val > 0:
                    path_block_slots.append(node_block_slots[idx_val - 1])
                else:
                    path_block_slots.append(None)

            is_selected = (lid == best_candidate.item()
                           if isinstance(best_candidate, torch.Tensor)
                           else lid == best_candidate)

            all_paths.append({
                "path_idx": lid,
                "tokens": path_token_ids,
                "texts": [self.tokenizer.decode([t]) for t in path_token_ids],
                "ranks": path_ranks,
                "block_slots": path_block_slots,
                "is_selected": is_selected,
            })

        return all_paths

    # ============================================================
    #    Full Inference Loop
    # ============================================================

    def _generate_once(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict:
        """Generate B=1 through the same request-level batch core."""
        return self.generate_conversations(
            [conversation],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )[0]

    def _generate_once_streaming(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ):
        """Streaming version of _generate_once for UI visualization.

        Yields per-iteration data with tree_info for colored output and tree display.
        """
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_ids = inputs.input_ids
        input_length = input_ids.shape[1]

        max_iterations = max_new_tokens + 10
        self._ensure_events_pool(max_iterations)

        iterations = 0
        draft_events = 0
        accept_lengths_raw = []
        all_iter_rank_stats = []
        all_iter_block_pos_stats = []
        self._block_forward_events = defaultdict(list)

        with torch.no_grad():
            # === Prefill ===
            _wall_start_event = torch.cuda.Event(enable_timing=True)
            _wall_end_event = torch.cuda.Event(enable_timing=True)
            _wall_start_event.record()
            _wall_start_time = time.perf_counter()  # Python wall clock for real-time yield
            self._events_pool['prefill_start'].record()

            target_outputs = self.target_model(
                input_ids,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = target_outputs.past_key_values
            all_hidden_states = target_outputs.hidden_states
            prefill_logits = target_outputs.logits

            hidden_3h = self._extract_hidden_3h(all_hidden_states)

            if temperature > 0:
                probs = F.softmax(prefill_logits[:, -1, :] / temperature, dim=-1)
                first_token = torch.multinomial(probs, num_samples=1)
            else:
                first_token = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)

            last_hidden_3h = hidden_3h[:, -1:, :]

            # Check EOS
            if first_token[0, 0] == self.tokenizer.eos_token_id:
                torch.cuda.synchronize()
                input_ids = torch.cat([input_ids, first_token], dim=1)
                output_ids = input_ids[0, input_length:]
                yield {
                    "final": True,
                    "output": self.tokenizer.decode(output_ids, skip_special_tokens=True),
                    "metrics": {
                        "total_tokens": len(output_ids),
                        "wall_time": 0, "tokens_per_second": 0,
                        "accept_length": 0, "iterations": 0,
                    }
                }
                return

            current_token = first_token

            # Merged draft prefill + first draft (saves one forward pass)
            shifted_input_ids = torch.cat([input_ids[:, 1:], first_token], dim=1)
            draft_cache, draft_position, b0_logits, b0_rank_logits, b0_draft_hidden, b0_ttt_kv = \
                self.draft_model.prefill_and_draft(
                    hidden_3h, shifted_input_ids, last_hidden_3h, current_token,
                )

            self._events_pool['prefill_end'].record()

            # === First Draft tree building (no separate forward needed) ===
            self._events_pool['draft_start'][0].record()

            draft_tokens, tree_mask, tree_position_ids, retrieve_indices, iter_rank_stats, node_ranks, node_block_slots = \
                self._build_tree_from_block1_dispatch(
                    b0_logits, b0_rank_logits, b0_draft_hidden, b0_ttt_kv,
                    current_token, draft_cache, draft_position - 1,
                    temperature=temperature,
                )

            self._events_pool['draft_end'][0].record()
            draft_events = 1

            # === Decode Loop ===
            _iter_raw_data = []
            eos_token_id = self.tokenizer.eos_token_id
            _tokens_generated = 0
            _yield_overhead = 0.0

            while (input_ids.shape[1] - input_length) < max_new_tokens:
                # 1. Verify Phase
                self._events_pool['target_start'][iterations].record()

                (
                    accept_length, accepted_token_ids, next_token,
                    past_key_values, last_hidden_3h, lazy_hidden_3h,
                    best_candidate, ret_indices, target_choices_cpu,
                ) = self._tree_verify(
                    draft_tokens, tree_mask, tree_position_ids, retrieve_indices,
                    input_ids, past_key_values, temperature,
                )

                self._events_pool['target_end'][iterations].record()

                # 2. Update Phase
                if accept_length > 0:
                    input_ids = torch.cat([input_ids, accepted_token_ids.unsqueeze(0), next_token], dim=1)
                else:
                    input_ids = torch.cat([input_ids, next_token], dim=1)

                # EOS check on GPU, no sync
                if accept_length > 0:
                    all_tokens = torch.cat([accepted_token_ids, next_token[0]])
                else:
                    all_tokens = next_token[0]
                eos_flag = (all_tokens == eos_token_id).any()

                # Store raw data for deferred UI processing (non_blocking tensors resolve after post-gen sync)
                draft_tokens_cpu_ui = draft_tokens.to('cpu', non_blocking=True)
                _iter_raw_data.append((
                    draft_tokens, retrieve_indices, best_candidate, accept_length,
                    accepted_token_ids, next_token, current_token,
                    node_ranks, node_block_slots, iter_rank_stats,
                    draft_tokens_cpu_ui, target_choices_cpu,
                ))
                accept_lengths_raw.append(accept_length)
                all_iter_rank_stats.append(iter_rank_stats)
                iterations += 1

                # 3. Draft Phase
                self._events_pool['draft_start'][draft_events].record()

                if accept_length == 0:
                    self.draft_model.pop_cache(draft_cache)
                    draft_tokens, tree_mask, tree_position_ids, retrieve_indices, iter_rank_stats, node_ranks, node_block_slots = \
                        self._build_draft_tree(
                            last_hidden_3h, next_token,
                            draft_cache, draft_position,
                            temperature=temperature,
                        )
                else:
                    batch_h = lazy_hidden_3h

                    if accept_length == 1:
                        batch_tok = torch.cat([next_token, next_token], dim=1)
                    else:
                        batch_tok = torch.cat([
                            accepted_token_ids[1:].unsqueeze(0),
                            next_token, next_token
                        ], dim=1)

                    blk_start = torch.cuda.Event(enable_timing=True)
                    blk_end = torch.cuda.Event(enable_timing=True)
                    blk_start.record()

                    _upd_out = None
                    if (self._draft_graph_cache is not None
                            and os.environ.get('DRAFT_CUDA_GRAPH_UPD', '0') == '1'):
                        try:
                            _upd_out = self._draft_graph_cache.run_update_cache(
                                hidden_3h=batch_h,
                                input_ids=batch_tok,
                                draft_cache=draft_cache,
                                start_position=draft_position + 1,
                            )
                        except Exception:
                            _upd_out = None

                    if _upd_out is not None:
                        logits, rank_logits, draft_hidden, ttt_kv, draft_position = _upd_out
                    else:
                        logits, rank_logits, draft_hidden, ttt_kv, draft_position = \
                            self.draft_model.update_cache_and_draft(
                                batch_h, batch_tok, draft_cache, draft_position + 1
                            )

                    blk_end.record()
                    self._block_forward_events[0].append((blk_start, blk_end))

                    draft_tokens, tree_mask, tree_position_ids, retrieve_indices, iter_rank_stats, node_ranks, node_block_slots = \
                        self._build_tree_from_block1_dispatch(
                            logits, rank_logits, draft_hidden, ttt_kv,
                            next_token, draft_cache, draft_position - 1,
                            temperature=temperature,
                        )

                self._events_pool['draft_end'][draft_events].record()
                draft_events += 1

                prev_token = current_token
                current_token = next_token

                # Check EOS after draft (overlapped) — this .item() syncs CUDA
                is_eos = eos_flag.item()

                # Real-time yield: prepare data, then measure only yield/UI wait time
                root_id = prev_token[0, 0].item()
                bonus_id = next_token[0, 0].item()
                if accept_length > 0:
                    new_tok_ids = [root_id] + accepted_token_ids.tolist() + [bonus_id]
                else:
                    new_tok_ids = [root_id, bonus_id]
                _tokens_generated += len(new_tok_ids) - 1
                _t_before_yield = time.perf_counter()
                yield {
                    "iteration": iterations - 1,
                    "new_tokens": new_tok_ids,
                    "new_text": self.tokenizer.decode(new_tok_ids),
                    "accepted_count": accept_length,
                    "bonus_token": bonus_id,
                    "bonus_text": self.tokenizer.decode([bonus_id]),
                    "current_metrics": {
                        "accept_length": self.compute_accept_length(accept_lengths_raw),
                        "tokens_so_far": _tokens_generated,
                        "elapsed_time": _t_before_yield - _wall_start_time - _yield_overhead,
                    }
                }

                # Accumulate yield + UI overhead (time spent outside inference)
                _yield_overhead += time.perf_counter() - _t_before_yield

                if is_eos:
                    break

        # === Post-generation: compute timing, then yield UI data ===
        _wall_end_event.record()
        torch.cuda.synchronize()

        wall_time = _wall_start_event.elapsed_time(_wall_end_event) / 1000.0 - _yield_overhead

        # Build detailed iteration data (deferred processing, after generation)
        iteration_details = []
        for iter_idx, raw in enumerate(_iter_raw_data):
            (d_tokens, r_indices, b_candidate, a_length,
             a_token_ids, n_token, c_token,
             n_ranks, n_block_slots, i_rank_stats,
             d_tokens_cpu, t_choices_cpu) = raw

            draft_token_ids = d_tokens[0].tolist()
            best_cand_val = (b_candidate.item()
                             if isinstance(b_candidate, torch.Tensor)
                             else b_candidate)
            bonus_token = n_token[0, 0].item()
            bonus_text = self.tokenizer.decode([bonus_token])
            root_token = c_token[0, 0].item()
            accepted_list = a_token_ids.tolist() if a_length > 0 else []
            new_tokens = [root_token] + accepted_list + [bonus_token]

            all_paths = self._extract_tree_paths(
                d_tokens, r_indices, b_candidate, a_length,
                node_ranks=n_ranks, node_block_slots=n_block_slots,
            )
            selected_path = all_paths[best_cand_val] if best_cand_val < len(all_paths) else None

            rejected_token = None
            rejected_text = None
            if selected_path and a_length + 1 < len(selected_path["tokens"]):
                rejected_token = selected_path["tokens"][a_length + 1]
                rejected_text = self.tokenizer.decode([rejected_token])

            accepted_texts = []
            if selected_path:
                for i in range(1, a_length + 1):
                    if i < len(selected_path["texts"]):
                        accepted_texts.append(selected_path["texts"][i])

            iter_bp_stats = self._compute_coverage_stats(
                d_tokens_cpu, t_choices_cpu, r_indices, n_block_slots,
            )
            all_iter_block_pos_stats.append(iter_bp_stats)

            iteration_details.append({
                "iteration": iter_idx,
                "new_tokens": new_tokens,
                "new_text": self.tokenizer.decode(new_tokens),
                "draft_tokens": draft_token_ids,
                "draft_text": [self.tokenizer.decode([t]) for t in draft_token_ids],
                "accepted_count": a_length,
                "rejected_token": rejected_token,
                "rejected_text": rejected_text,
                "bonus_token": bonus_token,
                "bonus_text": bonus_text,
                "tree_info": {
                    "all_paths": all_paths,
                    "selected_path_idx": best_cand_val,
                    "accepted_tokens": accepted_list,
                    "accepted_texts": accepted_texts,
                },
                "rank_info": {"num_blocks": len(i_rank_stats)} if i_rank_stats else {},
                "block_pos_stats": iter_bp_stats,
            })

        # Final timing
        prefill_time = self._events_pool['prefill_start'].elapsed_time(
            self._events_pool['prefill_end']
        ) / 1000.0

        draft_time = sum(
            self._events_pool['draft_start'][i].elapsed_time(self._events_pool['draft_end'][i])
            for i in range(draft_events)
        ) / 1000.0 if draft_events > 0 else 0.0

        target_time = sum(
            self._events_pool['target_start'][i].elapsed_time(self._events_pool['target_end'][i])
            for i in range(iterations)
        ) / 1000.0 if iterations > 0 else 0.0

        # Compute per-block forward times
        draft_forward_times = {}
        for depth, events in self._block_forward_events.items():
            draft_forward_times[depth] = sum(
                s.elapsed_time(e) for s, e in events
            ) / 1000.0

        output_ids = input_ids[0, input_length:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        num_tokens = len(output_ids)
        other_time = max(0, wall_time - prefill_time - draft_time - target_time)

        rank_stats = self._aggregate_rank_stats(all_iter_rank_stats, accept_lengths_raw)
        block_pos_stats = self._aggregate_block_pos_stats(all_iter_block_pos_stats)

        yield {
            "final": True,
            "output": output_text,
            "iteration_details": iteration_details,
            "metrics": {
                "total_tokens": num_tokens,
                "wall_time": wall_time,
                "tokens_per_second": num_tokens / wall_time if wall_time > 0 else 0,
                "accept_length": self.compute_accept_length(accept_lengths_raw),
                "iterations": iterations,
                "accept_lengths_raw": accept_lengths_raw,
                "prefill_time": prefill_time,
                "draft_time": draft_time,
                "target_time": target_time,
                "other_time": other_time,
                "draft_pct": draft_time / wall_time * 100 if wall_time > 0 else 0,
                "target_pct": target_time / wall_time * 100 if wall_time > 0 else 0,
                "draft_forward_times": draft_forward_times,
                "tree_profile": getattr(self, '_last_tree_profile', {}),
                "rank_stats": rank_stats,
                "block_pos_stats": block_pos_stats,
            }
        }

    def cleanup(self):
        if self.target_model is not None:
            del self.target_model
            self.target_model = None
        if self.draft_model is not None:
            del self.draft_model
            self.draft_model = None
        self._events_pool = None
        super().cleanup()
