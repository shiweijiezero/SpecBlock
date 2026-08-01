"""JIT-compiled CUDA tree builder loader.

Wraps the C++/CUDA kernels in tree_build_cuda.cu via torch.utils.cpp_extension.load.
API matches tree_build_triton.py so specblock.py can dispatch by env var.

First call compiles (~10-30 s); the compiled .so is cached at
TORCH_EXTENSIONS_DIR (set to NFS on the cluster).
"""
from __future__ import annotations

import os
import threading

import torch

_MOD = None
_MOD_LOCK = threading.Lock()
_TREE_FIELDS = ("tokens", "parents", "lps", "ranks", "blocks", "slots")


def _tree_buffer_capacity(tree_buf: dict) -> int:
    capacities = {int(tree_buf[name].numel()) for name in _TREE_FIELDS}
    if len(capacities) != 1:
        raise ValueError("CUDA tree buffers must have identical capacities")
    return capacities.pop()


def tree_bfs_required_capacity(
    tree_start: int,
    leaves: int,
    K: int,
    max_topk: int,
) -> int:
    values = (tree_start, leaves, K, max_topk)
    if tree_start < 0 or leaves < 0 or K <= 0 or max_topk <= 0:
        raise ValueError(f"invalid CUDA BFS dimensions: {values}")
    return tree_start + leaves * K * max_topk


def _get_module():
    global _MOD
    if _MOD is not None:
        return _MOD
    with _MOD_LOCK:
        if _MOD is not None:
            return _MOD
        from torch.utils.cpp_extension import load as _load

        src_dir = os.path.dirname(os.path.abspath(__file__))
        cu_src = os.path.join(src_dir, 'tree_build_cuda.cu')
        _MOD = _load(
            name='specblock_tree_build_cuda',
            sources=[cu_src],
            verbose=bool(int(os.environ.get('TREE_BUILD_CUDA_VERBOSE', '0'))),
            extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
            extra_cflags=['-O3'],
        )
        return _MOD


def cuda_build_block1(
    rank_preds: torch.Tensor,          # [K] i64
    greedy_target: torch.Tensor,       # [K] i64
    greedy_lps: torch.Tensor,          # [K] f32
    all_top_target: torch.Tensor,      # [K, MAX_TOPK] i64
    all_top_lps: torch.Tensor,         # [K, MAX_TOPK] f32
    rank_slot_topk_table: torch.Tensor,  # [RANK_CLASSES] i64
    tree_buf: dict,
    pend_buf: dict,
    sizes_buf: torch.Tensor,           # [4] i64
    beam_width: int,
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
    adaptive_slot0: int,
    adaptive_all: int,
):
    mod = _get_module()
    mod.build_tree_block1_cuda(
        rank_preds, greedy_target, greedy_lps,
        all_top_target, all_top_lps,
        rank_slot_topk_table,
        tree_buf['tokens'], tree_buf['parents'], tree_buf['lps'],
        tree_buf['ranks'], tree_buf['blocks'], tree_buf['slots'],
        pend_buf['hidden_slots'], pend_buf['input_ids'],
        pend_buf['ttt_valid'], pend_buf['node_indices'], pend_buf['cum_lps'],
        sizes_buf,
        int(beam_width), int(K), int(max_topk),
        int(give_up_class),
        int(adaptive_slot0), int(adaptive_all),
    )
    sizes_cpu = sizes_buf.cpu().tolist()  # 1 host sync
    return sizes_cpu[0], sizes_cpu[1], sizes_cpu[2], sizes_cpu[3]


def cuda_build_block1_fixed_n(
    rank_preds: torch.Tensor,
    greedy_target: torch.Tensor,
    greedy_lps: torch.Tensor,
    all_top_target: torch.Tensor,
    all_top_lps: torch.Tensor,
    rank_slot_topk_table: torch.Tensor,
    tree_buf: dict,
    pend_buf: dict,
    sizes_buf: torch.Tensor,
    beam_width: int,
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
    adaptive_slot0: int,
    adaptive_all: int,
    fixed_n: int,
):
    """Launch the fixed-N variant.  N_pend is compile-time constant (fixed_n).

    sizes_buf is still written (n_nodes_b1, n_active, total_alts1, N_real)
    but N_real is informational — caller treats N_pend = fixed_n always.
    No host sync is required on sizes_buf[3] for the pipeline to proceed.
    """
    if fixed_n <= 0:
        raise ValueError("fixed-N CUDA tree builder requires fixed_n > 0")
    required_block1 = K + K * (max_topk - 1) + 1
    if _tree_buffer_capacity(tree_buf) < required_block1:
        raise ValueError("CUDA tree buffer is too small for block-1 output")
    if any(int(pend_buf[name].numel()) < fixed_n for name in (
        "hidden_slots", "input_ids", "ttt_valid", "node_indices", "cum_lps"
    )):
        raise ValueError("fixed-N pending buffers are smaller than fixed_n")
    mod = _get_module()
    mod.build_tree_block1_fixed_n_cuda(
        rank_preds, greedy_target, greedy_lps,
        all_top_target, all_top_lps,
        rank_slot_topk_table,
        tree_buf['tokens'], tree_buf['parents'], tree_buf['lps'],
        tree_buf['ranks'], tree_buf['blocks'], tree_buf['slots'],
        pend_buf['hidden_slots'], pend_buf['input_ids'],
        pend_buf['ttt_valid'], pend_buf['node_indices'], pend_buf['cum_lps'],
        sizes_buf,
        int(beam_width), int(K), int(max_topk),
        int(give_up_class),
        int(adaptive_slot0), int(adaptive_all),
        int(fixed_n),
    )


def cuda_build_bfs(
    all_rank_preds: torch.Tensor,      # [N, K] i64
    all_greedy_target: torch.Tensor,   # [N, K] i64
    all_greedy_lps: torch.Tensor,      # [N, K] f32
    top_target_all: torch.Tensor,      # [N, K, MAX_TOPK] i64
    top_lps_all: torch.Tensor,         # [N, K, MAX_TOPK] f32
    pend_cum_lps: torch.Tensor,        # [N] f32
    pend_node_indices: torch.Tensor,   # [N] i64
    rank_slot_topk_table: torch.Tensor,  # [RANK_CLASSES] i64
    tree_buf: dict,
    total_alts_buf: torch.Tensor,      # [1] i64
    tree_start: int,
    N: int,
    K: int,
    max_topk: int,
    rank_classes: int,
    give_up_class: int,
    adaptive_all: int,
    pend_depth: int,
):
    required_capacity = tree_bfs_required_capacity(
        tree_start,
        N,
        K,
        max_topk,
    )
    if _tree_buffer_capacity(tree_buf) < required_capacity:
        raise ValueError(
            "CUDA tree buffer is too small for worst-case BFS output: "
            f"capacity={_tree_buffer_capacity(tree_buf)}, "
            f"required={required_capacity}"
        )
    mod = _get_module()
    mod.build_tree_bfs_cuda(
        all_rank_preds, all_greedy_target, all_greedy_lps,
        top_target_all, top_lps_all,
        pend_cum_lps, pend_node_indices,
        rank_slot_topk_table,
        tree_buf['tokens'], tree_buf['parents'], tree_buf['lps'],
        tree_buf['ranks'], tree_buf['blocks'], tree_buf['slots'],
        total_alts_buf,
        int(N), int(tree_start), int(K), int(max_topk),
        int(give_up_class),
        int(adaptive_all),
        int(pend_depth),
    )
    return int(total_alts_buf.cpu().item())  # 1 host sync


def warmup():
    """Force JIT compile if not done. Safe to call multiple times."""
    _get_module()
