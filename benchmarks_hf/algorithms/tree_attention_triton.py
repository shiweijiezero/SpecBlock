"""Shape-invariant triton attention for SpecBlock draft forward.

Flash-attention-2 style online-softmax kernel. Unlike cuBLAS matmul-based
attention (whose tile pick depends on K dim), this kernel always reduces
over the full D dimension in one go per tile, producing bit-identical
output across different cross_count values. Used to replace the
Q@K^T + softmax + attn@V chain in SpecBlockAttentionWithCache.forward_batch
(and forward, for block 1) without bucket-padding drift.

Supports:
  - Batch of B sequences
  - n_heads attention heads (KV assumed pre-GQA-repeated to n_heads)
  - Cross region [0, cross_count): fully visible
  - Current region [cross_count, cross_count+M): tree-mask derived from
    position/slot indices (kv_pos < q_pos, or == q_pos with kv_slot <= q_slot)

Limitations (first cut):
  - Inputs assumed contiguous in last two dims
  - Head dim D must be constexpr (hardcoded 128)
  - No TTT support yet — caller should flatten TTT into the cross/current
    regions (or call the block-1 variant once added)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .runtime_capabilities import RUNTIME_CAPABILITIES

_DUMMY_MASK = None


def _query_block_m(width: int, *, cap: int | None = None) -> int:
    block_m = max(
        RUNTIME_CAPABILITIES.minimum_triton_query_tile,
        triton.next_power_of_2(width),
    )
    return min(cap, block_m) if cap is not None else block_m


@triton.jit
def _tree_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    cross_count,  # int32 scalar
    M, C,         # int32 scalars (runtime)
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_c, stride_k_d,
    stride_v_b, stride_v_h, stride_v_c, stride_v_d,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)   # query block along M
    pid_b = tl.program_id(1)   # batch
    pid_h = tl.program_id(2)   # head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, D)                           # [D]

    # --- Load Q block ---
    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q_ptrs = q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d
    q_mask = offs_m[:, None] < M
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)       # [BLOCK_M, D]

    # --- q-side position/slot for tree mask ---
    q_pos = offs_m // K_SLOTS                          # [BLOCK_M]
    q_slot = offs_m - q_pos * K_SLOTS                  # [BLOCK_M]

    # --- Online softmax accumulators ---
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # --- Iterate over KV blocks ---
    k_base = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    v_base = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h

    for n_start in range(0, C, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)       # [BLOCK_N]
        n_in_bounds = offs_n < C                        # [BLOCK_N]

        # Load K, V block
        k_ptrs = k_base + offs_n[:, None] * stride_k_c + offs_d[None, :] * stride_k_d
        v_ptrs = v_base + offs_n[:, None] * stride_v_c + offs_d[None, :] * stride_v_d
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)  # [BLOCK_N, D]
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)  # [BLOCK_N, D]

        # --- Compute Q @ K^T (dot reduces over D in a single pass for fixed BLOCK_D) ---
        qk = tl.dot(q, tl.trans(k)) * scale             # [BLOCK_M, BLOCK_N]

        # --- Tree mask ---
        # cross region (kv_idx < cross_count): always visible
        is_cross = offs_n[None, :] < cross_count        # [1, BLOCK_N]
        # current region: tree causal
        curr_kv = offs_n - cross_count                  # signed; valid only when >= 0
        curr_kv_pos = curr_kv // K_SLOTS
        curr_kv_slot = curr_kv - curr_kv_pos * K_SLOTS
        curr_visible = (curr_kv_pos[None, :] < q_pos[:, None]) | (
            (curr_kv_pos[None, :] == q_pos[:, None]) &
            (curr_kv_slot[None, :] <= q_slot[:, None])
        )
        mask = (is_cross | curr_visible) & n_in_bounds[None, :]
        qk = tl.where(mask, qk, -float('inf'))

        # --- Online softmax update ---
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    # --- Normalize ---
    acc = acc / l_i[:, None]

    # --- Store ---
    o_base = Out_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    o_ptrs = o_base + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=q_mask)


def tree_attention(q, k_full, v_full, cross_count, k_slots, scale):
    """Compute tree-masked attention.

    Args:
        q: [B, H, M, D] query (post-RoPE, post-GQA-repeat to H heads)
        k_full: [B, H, C, D] keys (cross cache followed by current K)
        v_full: [B, H, C, D] values
        cross_count: int (C_cross), number of cross-region positions
                      (current region starts at kv_idx = cross_count)
        k_slots: int, K slots per position (for tree mask)
        scale: float, attention scale factor (1/sqrt(D))

    Returns:
        out: [B, H, M, D] attention output
    """
    B, H, M, D = q.shape
    C = k_full.shape[2]
    assert v_full.shape == k_full.shape
    assert q.dtype == k_full.dtype == v_full.dtype
    assert D == 128, f"kernel hardcoded for D=128, got {D}"

    # Use the smallest power-of-two query tile, capped for large-tree tiling.
    BLOCK_M = _query_block_m(M, cap=32)
    BLOCK_N = RUNTIME_CAPABILITIES.attention_block_n

    out = torch.empty_like(q)

    grid = (triton.cdiv(M, BLOCK_M), B, H)
    _tree_attn_fwd_kernel[grid](
        q, k_full, v_full, out,
        cross_count, M, C,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
        v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale=scale,
        K_SLOTS=k_slots,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
    )
    return out


@triton.jit
def _ragged_tree_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Lengths_ptr, ValidSlots_ptr, Out_ptr,
    M, C, n_kv_groups,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_c, stride_k_d,
    stride_v_b, stride_v_h, stride_v_c, stride_v_d,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Tree attention with a distinct cross length and valid query width per row."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_kh = pid_h // n_kv_groups

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    cross_count = tl.load(Lengths_ptr + pid_b)
    valid_slots = tl.load(ValidSlots_ptr + pid_b)
    total_valid = cross_count + valid_slots
    query_valid = offs_m < valid_slots

    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
        mask=(offs_m[:, None] < M),
        other=0.0,
    )
    q_pos = offs_m // K_SLOTS
    q_slot = offs_m - q_pos * K_SLOTS

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    k_base = K_ptr + pid_b * stride_k_b + pid_kh * stride_k_h
    v_base = V_ptr + pid_b * stride_v_b + pid_kh * stride_v_h
    for n_start in range(0, C, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < C
        row_kv_valid = n_in_bounds & (offs_n < total_valid)
        k = tl.load(
            k_base + offs_n[:, None] * stride_k_c + offs_d[None, :] * stride_k_d,
            mask=row_kv_valid[:, None],
            other=0.0,
        )
        v = tl.load(
            v_base + offs_n[:, None] * stride_v_c + offs_d[None, :] * stride_v_d,
            mask=row_kv_valid[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale

        is_cross = offs_n[None, :] < cross_count
        current_index = offs_n - cross_count
        current_pos = current_index // K_SLOTS
        current_slot = current_index - current_pos * K_SLOTS
        current_visible = (
            (current_pos[None, :] < q_pos[:, None])
            | (
                (current_pos[None, :] == q_pos[:, None])
                & (current_slot[None, :] <= q_slot[:, None])
            )
        )
        normal_visible = (
            (is_cross | current_visible)
            & (offs_n[None, :] < total_valid)
            & query_valid[:, None]
        )
        # Padded query rows are discarded, but keep their softmax finite so
        # NaNs cannot flow through later residual/MLP layers.
        padding_sentinel = (~query_valid[:, None]) & (offs_n[None, :] == 0)
        visible = (normal_visible | padding_sentinel) & n_in_bounds[None, :]
        qk = tl.where(visible, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    o_base = Out_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    tl.store(
        o_base + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d,
        acc.to(Out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M),
    )


def ragged_tree_attention(
    q,
    cache_k,
    cache_v,
    lengths,
    valid_slots,
    max_total,
    k_slots,
    n_kv_groups,
    scale,
):
    """Attend to row-wise ``[cross | current]`` GQA KV in a shared cache."""
    if q.ndim != 4 or cache_k.ndim != 4 or cache_v.ndim != 4:
        raise ValueError("ragged draft attention tensors must be rank 4")
    B, H, M, D = q.shape
    n_kv_groups = int(n_kv_groups)
    if n_kv_groups <= 0 or H % n_kv_groups != 0:
        raise ValueError("query heads must be divisible by n_kv_groups")
    if cache_k.shape != cache_v.shape:
        raise ValueError("ragged draft cache key/value shapes must match")
    if cache_k.shape[0] != B or cache_k.shape[1] * int(n_kv_groups) != H:
        raise ValueError("ragged draft cache batch/head shapes do not match query")
    if cache_k.shape[3] != D:
        raise ValueError("ragged draft cache head dimension does not match query")
    if lengths.shape != valid_slots.shape or lengths.numel() != B:
        raise ValueError("ragged draft metadata must have one entry per batch row")
    if not lengths.is_contiguous() or not valid_slots.is_contiguous():
        raise ValueError("ragged draft metadata must be contiguous")
    if any(tensor.device != q.device for tensor in (cache_k, cache_v, lengths, valid_slots)):
        raise ValueError("ragged draft attention tensors must share one device")
    if q.dtype != cache_k.dtype or q.dtype != cache_v.dtype:
        raise TypeError("ragged draft attention Q/K/V dtypes must match")
    if lengths.dtype != torch.long or valid_slots.dtype != torch.long:
        raise TypeError("ragged draft metadata must use torch.long")
    if D != 128:
        raise ValueError(f"kernel hardcoded for D=128, got {D}")
    if q.device.type != "cuda":
        raise ValueError("ragged draft attention requires CUDA tensors")
    if max_total <= 0 or max_total > cache_k.shape[2]:
        raise ValueError("ragged draft attention width is out of range")
    block_m = _query_block_m(M, cap=32)
    block_n = RUNTIME_CAPABILITIES.attention_block_n
    out = torch.empty_like(q)
    grid = (triton.cdiv(M, block_m), B, H)
    _ragged_tree_attn_fwd_kernel[grid](
        q, cache_k, cache_v, lengths, valid_slots, out,
        M, int(max_total), int(n_kv_groups),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cache_k.stride(0), cache_k.stride(1), cache_k.stride(2), cache_k.stride(3),
        cache_v.stride(0), cache_v.stride(1), cache_v.stride(2), cache_v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale=scale,
        K_SLOTS=k_slots,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D=D,
    )
    return out


# ============================================================
#   Single-position forward with TTT + causal mask
# ============================================================


@triton.jit
def _single_pos_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, TTT_mask_ptr, Out_ptr,
    cross_count,        # int32
    ttt_count,          # int32
    curr_start,         # int32 = cross_count + ttt_count
    C,                  # int32 = total KV length
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_c, stride_k_d,
    stride_v_b, stride_v_h, stride_v_c, stride_v_d,
    stride_ttt_b, stride_ttt_c,    # ttt_mask stride
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,         # M = K_SLOTS for single-position path
    BLOCK_M: tl.constexpr,         # smallest power-of-two query tile
    HAS_TTT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Attention for a single prediction position (M = K_SLOTS queries).

    Mask regions along KV:
      [0, cross_count)             — cross cache, all visible
      [cross_count, curr_start)    — ttt cache, masked by TTT_mask
      [curr_start, C)              — current K, causal within the K slots

    Each program handles all M queries for one (batch, head). BLOCK_M is the
    smallest power-of-two tile covering the query width.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    m_in_bounds = offs_m < K_SLOTS
    offs_d = tl.arange(0, D)

    # Load Q; non-power-of-two tail rows are masked.
    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
        mask=m_in_bounds[:, None], other=0.0,
    )

    # Accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    k_base = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    v_base = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    if HAS_TTT:
        ttt_base = TTT_mask_ptr + pid_b * stride_ttt_b

    for n_start in range(0, C, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < C

        k = tl.load(
            k_base + offs_n[:, None] * stride_k_c + offs_d[None, :] * stride_k_d,
            mask=n_in_bounds[:, None], other=0.0,
        )
        v = tl.load(
            v_base + offs_n[:, None] * stride_v_c + offs_d[None, :] * stride_v_d,
            mask=n_in_bounds[:, None], other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * scale

        # --- Build mask per kv index ---
        is_cross = offs_n < cross_count                          # [BLOCK_N]
        is_ttt = (offs_n >= cross_count) & (offs_n < curr_start)
        is_curr = offs_n >= curr_start
        curr_slot = offs_n - curr_start                          # valid for is_curr

        # TTT mask: lookup per-position visibility
        if HAS_TTT:
            # Load ttt_mask[0:ttt_count] — values for in-ttt kv_idx
            ttt_local = offs_n - cross_count                     # valid only for is_ttt
            # Clamp to [0, ttt_count) for safe load; mask out invalid with is_ttt
            ttt_local_clamped = tl.maximum(tl.minimum(ttt_local, ttt_count - 1), 0)
            ttt_vis = tl.load(
                ttt_base + ttt_local_clamped * stride_ttt_c,
                mask=is_ttt & n_in_bounds, other=0,
            ).to(tl.int1)
        else:
            ttt_vis = tl.zeros([BLOCK_N], dtype=tl.int1)

        # Causal within current K: kv_slot <= q_slot
        causal_vis = (curr_slot[None, :] <= offs_m[:, None])     # [M, BLOCK_N]

        # Combine
        visible = (
            is_cross[None, :] |
            (is_ttt[None, :] & ttt_vis[None, :]) |
            (is_curr[None, :] & causal_vis)
        ) & n_in_bounds[None, :]
        qk = tl.where(visible, qk, -float('inf'))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    acc = acc / l_i[:, None]

    o_base = Out_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    tl.store(
        o_base + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d,
        acc.to(Out_ptr.dtype.element_ty),
        mask=m_in_bounds[:, None],
    )


def single_pos_attention(q, k_full, v_full, cross_count, ttt_count, ttt_mask,
                         k_slots, scale):
    """Single-position attention (M = K slots) with cross + ttt + causal mask.

    Args:
        q: [B, H, K, D]
        k_full: [B, H, C, D] = cat([cross_cache, ttt_k, current_k]) post-GQA
        v_full: [B, H, C, D]
        cross_count: int
        ttt_count: int (0 if no ttt)
        ttt_mask: [B, ttt_count] bool tensor (may be None if ttt_count == 0)
        k_slots: int (= K, must equal q.shape[2])
        scale: float
    """
    B, H, M, D = q.shape
    C = k_full.shape[2]
    assert M == k_slots
    assert D == 128

    has_ttt = (ttt_count > 0) and (ttt_mask is not None)
    curr_start = cross_count + ttt_count

    out = torch.empty_like(q)

    if has_ttt:
        ttt_mask_bool = ttt_mask if ttt_mask.dtype == torch.bool else ttt_mask.bool()
        stride_ttt_b = ttt_mask_bool.stride(0)
        stride_ttt_c = ttt_mask_bool.stride(1)
    else:
        # Pass a dummy pointer — HAS_TTT=False short-circuits the load.
        global _DUMMY_MASK
        if _DUMMY_MASK is None or _DUMMY_MASK.device != q.device:
            _DUMMY_MASK = torch.empty(1, device=q.device, dtype=torch.bool)
        ttt_mask_bool = _DUMMY_MASK
        stride_ttt_b = 0
        stride_ttt_c = 0

    BLOCK_M = _query_block_m(M)
    BLOCK_N = RUNTIME_CAPABILITIES.attention_block_n
    grid = (B, H)
    _single_pos_attn_fwd_kernel[grid](
        q, k_full, v_full, ttt_mask_bool, out,
        cross_count, ttt_count, curr_start, C,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_full.stride(0), k_full.stride(1), k_full.stride(2), k_full.stride(3),
        v_full.stride(0), v_full.stride(1), v_full.stride(2), v_full.stride(3),
        stride_ttt_b, stride_ttt_c,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale=scale,
        K_SLOTS=k_slots,
        BLOCK_M=BLOCK_M,
        HAS_TTT=has_ttt,
        BLOCK_N=BLOCK_N,
        D=D,
    )
    return out


# ============================================================
#   3-way single-pos attention: cross + ttt + curr from 3 buffers
# ============================================================
#
# Bypasses the [cross_expand_to_B, repeat_kv(ttt), curr] -> torch.cat chain in
# SpecBlockAttentionWithCache.forward. Profile (DBD humaneval n=80, n=4000):
#   kv_cat = 4507us / b2_fwd 6382us = 70.6%
# Root cause: cross_cache is physically [1, H, T, D] but expanded to
# [B, H, T, D] via expand() before cat; cat materializes the broadcast into
# B physical copies.
#
# This kernel reads cross/ttt/curr directly via their original strides:
#   * cross_k stride[0] is 0 (broadcast view) -> all batches read same buffer
#   * ttt_k is GQA-non-expanded [B, KH, T_ttt, D]; kernel uses pid_h //
#     n_kv_groups to index KV head (no repeat_kv needed)
#   * cross_k may be either Hq-expanded legacy storage or Hkv-native ragged storage
#   * curr_k is GQA-native [B, KH, K, D], indexed like ttt_k
#
# 3 inner loops (cross / ttt / curr) carry online-softmax state (m_i, l_i,
# acc) across boundaries, mathematically identical to the all-cat single-loop
# variant.


@triton.jit
def _three_part_attn_fwd_kernel(
    Q_ptr,
    CROSS_K_ptr, CROSS_V_ptr,
    TTT_K_ptr, TTT_V_ptr,
    CURR_K_ptr, CURR_V_ptr,
    TTT_MASK_ptr, Out_ptr,
    cross_count,                # int32 (runtime)
    ttt_count,                  # int32
    n_kv_groups,                # int32 (= n_heads / n_kv_heads)
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_ck_b, stride_ck_h, stride_ck_t, stride_ck_d,
    stride_cv_b, stride_cv_h, stride_cv_t, stride_cv_d,
    stride_tk_b, stride_tk_h, stride_tk_t, stride_tk_d,
    stride_tv_b, stride_tv_h, stride_tv_t, stride_tv_d,
    stride_qk_b, stride_qk_h, stride_qk_t, stride_qk_d,
    stride_qv_b, stride_qv_h, stride_qv_t, stride_qv_d,
    stride_mb, stride_mc,        # ttt mask stride
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,
    BLOCK_M: tl.constexpr,         # smallest power-of-two query tile
    CROSS_GQA: tl.constexpr,
    HAS_TTT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """3-buffer attention. Grid: (B, H).

    Region layout (along KV axis):
      cross [0, cross_count)        - all visible, Hq-expanded or Hkv-native
      ttt   [0, ttt_count)          - per-mask visible, Hkv-native
      curr  [0, K_SLOTS)            - causal within slots, Hkv-native

    Online softmax state carried across all 3 loops.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // n_kv_groups

    offs_m = tl.arange(0, BLOCK_M)
    m_in_bounds = offs_m < K_SLOTS
    offs_d = tl.arange(0, D)

    # Load Q; non-power-of-two tail rows are masked.
    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
        mask=m_in_bounds[:, None], other=0.0,
    )

    # Online softmax state (carried across 3 region loops)
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # ---- Loop 1: cross region (all visible) ----
    # cross_k may be batch-broadcast and either Hq-expanded or Hkv-native.
    cross_head = pid_kh if CROSS_GQA else pid_h
    ck_base = CROSS_K_ptr + pid_b * stride_ck_b + cross_head * stride_ck_h
    cv_base = CROSS_V_ptr + pid_b * stride_cv_b + cross_head * stride_cv_h
    for n_start in range(0, cross_count, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < cross_count
        k = tl.load(
            ck_base + offs_n[:, None] * stride_ck_t + offs_d[None, :] * stride_ck_d,
            mask=n_in_bounds[:, None], other=0.0,
        )
        v = tl.load(
            cv_base + offs_n[:, None] * stride_cv_t + offs_d[None, :] * stride_cv_d,
            mask=n_in_bounds[:, None], other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(n_in_bounds[None, :], qk, -float('inf'))
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    # ---- Loop 2: ttt region (GQA-indexed; per-mask visible) ----
    if HAS_TTT:
        tk_base = TTT_K_ptr + pid_b * stride_tk_b + pid_kh * stride_tk_h
        tv_base = TTT_V_ptr + pid_b * stride_tv_b + pid_kh * stride_tv_h
        mask_base = TTT_MASK_ptr + pid_b * stride_mb
        for n_start in range(0, ttt_count, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            n_in_bounds = offs_n < ttt_count
            k = tl.load(
                tk_base + offs_n[:, None] * stride_tk_t + offs_d[None, :] * stride_tk_d,
                mask=n_in_bounds[:, None], other=0.0,
            )
            v = tl.load(
                tv_base + offs_n[:, None] * stride_tv_t + offs_d[None, :] * stride_tv_d,
                mask=n_in_bounds[:, None], other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * scale
            ttt_vis = tl.load(
                mask_base + offs_n * stride_mc,
                mask=n_in_bounds, other=0,
            ).to(tl.int1)
            qk = tl.where(ttt_vis[None, :] & n_in_bounds[None, :], qk, -float('inf'))
            m_ij = tl.max(qk, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_i_new

    # ---- Loop 3: curr region (causal within slots) ----
    # CURR_K/V is GQA-non-expanded [B, KH, K, D] (same as ttt); kernel does GQA index.
    qk_base = CURR_K_ptr + pid_b * stride_qk_b + pid_kh * stride_qk_h
    qv_base = CURR_V_ptr + pid_b * stride_qv_b + pid_kh * stride_qv_h
    for n_start in range(0, K_SLOTS, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < K_SLOTS
        k = tl.load(
            qk_base + offs_n[:, None] * stride_qk_t + offs_d[None, :] * stride_qk_d,
            mask=n_in_bounds[:, None], other=0.0,
        )
        v = tl.load(
            qv_base + offs_n[:, None] * stride_qv_t + offs_d[None, :] * stride_qv_d,
            mask=n_in_bounds[:, None], other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        causal_vis = (offs_n[None, :] <= offs_m[:, None]) & n_in_bounds[None, :]
        qk = tl.where(causal_vis, qk, -float('inf'))
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    # Normalize + write
    acc = acc / l_i[:, None]
    o_base = Out_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    tl.store(
        o_base + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d,
        acc.to(Out_ptr.dtype.element_ty),
        mask=m_in_bounds[:, None],
    )


def three_part_attention(
    q: torch.Tensor,                       # [B, H, K, D]
    cross_k: torch.Tensor,                 # [B, H|KH, T_cross, D] (stride[0] may be 0)
    cross_v: torch.Tensor,
    ttt_k: torch.Tensor,                   # [B, KH, T_ttt, D] (GQA non-expanded), or None
    ttt_v: torch.Tensor,
    curr_k: torch.Tensor,                  # [B, KH, K, D] (GQA non-expanded)
    curr_v: torch.Tensor,
    cross_count: int,
    ttt_count: int,
    ttt_mask: torch.Tensor,                # [B, ttt_count] bool, or None
    k_slots: int,
    n_kv_groups: int,
    scale: float,
) -> torch.Tensor:
    """3-buffer attention without cat. Returns out [B, H, K, D].

    cross_k may be a `.expand(B, ...)` view with stride[0]==0 and may
    store either Hq-expanded or Hkv-native heads. TTT/current KV stay
    GQA-native and are indexed via ``pid_h // n_kv_groups``.
    """
    B, H, M, D = q.shape
    n_kv_groups = int(n_kv_groups)
    if n_kv_groups <= 0 or H % n_kv_groups != 0:
        raise ValueError("query heads must be divisible by n_kv_groups")
    if M != k_slots or D != 128:
        raise ValueError("three-part attention expects K query slots with D=128")
    if cross_k.shape != cross_v.shape or curr_k.shape != curr_v.shape:
        raise ValueError("three-part attention key/value shapes must match")
    if cross_k.ndim != 4 or curr_k.ndim != 4:
        raise ValueError("three-part attention KV tensors must be rank 4")
    if cross_k.shape[0] != B or curr_k.shape[0] != B:
        raise ValueError("three-part attention batch dimensions must match query")
    if cross_k.shape[3] != D or curr_k.shape[3] != D:
        raise ValueError("three-part attention head dimensions must match query")
    kv_heads = H // int(n_kv_groups)
    cross_gqa = cross_k.shape[1] == kv_heads
    if not cross_gqa and cross_k.shape[1] != H:
        raise ValueError("cross cache must use either Hq or Hkv heads")
    if curr_k.shape[1] != kv_heads:
        raise ValueError("current KV must use Hkv heads")
    if not 0 <= int(cross_count) <= cross_k.shape[2]:
        raise ValueError("cross_count exceeds cross-cache width")
    if curr_k.shape[2] < k_slots:
        raise ValueError("current KV width is smaller than K slots")
    if any(tensor.device != q.device for tensor in (cross_k, cross_v, curr_k, curr_v)):
        raise ValueError("three-part attention tensors must share one device")
    if any(tensor.dtype != q.dtype for tensor in (cross_k, cross_v, curr_k, curr_v)):
        raise TypeError("three-part attention Q/K/V dtypes must match")

    has_ttt = (ttt_count > 0) and (ttt_mask is not None) and (ttt_k is not None)
    if has_ttt:
        if ttt_v is None or ttt_k.shape != ttt_v.shape:
            raise ValueError("TTT key/value shapes must match")
        if ttt_k.ndim != 4 or ttt_k.shape[0] != B or ttt_k.shape[1] != kv_heads:
            raise ValueError("TTT KV must use matching batch and Hkv heads")
        if not 0 <= int(ttt_count) <= ttt_k.shape[2]:
            raise ValueError("ttt_count exceeds TTT-cache width")
        if any(tensor.device != q.device for tensor in (ttt_k, ttt_v)):
            raise ValueError("TTT KV must share the query device")
        if any(tensor.dtype != q.dtype for tensor in (ttt_k, ttt_v)):
            raise TypeError("TTT KV dtype must match query")
        if ttt_k.shape[3] != D:
            raise ValueError("TTT KV head dimension must match query")

    out = torch.empty_like(q)

    if has_ttt:
        ttt_mask_bool = ttt_mask if ttt_mask.dtype == torch.bool else ttt_mask.bool()
        stride_mb = ttt_mask_bool.stride(0)
        stride_mc = ttt_mask_bool.stride(1)
    else:
        global _DUMMY_MASK
        if _DUMMY_MASK is None or _DUMMY_MASK.device != q.device:
            _DUMMY_MASK = torch.empty(1, device=q.device, dtype=torch.bool)
        ttt_mask_bool = _DUMMY_MASK
        stride_mb = 0
        stride_mc = 0

    # Dummy ttt buffers when HAS_TTT=False (kernel branch off via constexpr).
    if not has_ttt or ttt_k is None:
        ttt_k_arg = curr_k
        ttt_v_arg = curr_v
        tk_strides = (curr_k.stride(0), curr_k.stride(1), curr_k.stride(2), curr_k.stride(3))
        tv_strides = (curr_v.stride(0), curr_v.stride(1), curr_v.stride(2), curr_v.stride(3))
    else:
        ttt_k_arg = ttt_k
        ttt_v_arg = ttt_v
        tk_strides = (ttt_k.stride(0), ttt_k.stride(1), ttt_k.stride(2), ttt_k.stride(3))
        tv_strides = (ttt_v.stride(0), ttt_v.stride(1), ttt_v.stride(2), ttt_v.stride(3))

    BLOCK_M = _query_block_m(M)
    BLOCK_N = RUNTIME_CAPABILITIES.three_part_attention_block_n
    launch_kwargs = RUNTIME_CAPABILITIES.three_part_attention_launch_kwargs()
    grid = (B, H)
    _three_part_attn_fwd_kernel[grid](
        q,
        cross_k, cross_v,
        ttt_k_arg, ttt_v_arg,
        curr_k, curr_v,
        ttt_mask_bool, out,
        cross_count, ttt_count, n_kv_groups,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cross_k.stride(0), cross_k.stride(1), cross_k.stride(2), cross_k.stride(3),
        cross_v.stride(0), cross_v.stride(1), cross_v.stride(2), cross_v.stride(3),
        tk_strides[0], tk_strides[1], tk_strides[2], tk_strides[3],
        tv_strides[0], tv_strides[1], tv_strides[2], tv_strides[3],
        curr_k.stride(0), curr_k.stride(1), curr_k.stride(2), curr_k.stride(3),
        curr_v.stride(0), curr_v.stride(1), curr_v.stride(2), curr_v.stride(3),
        stride_mb, stride_mc,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale=scale,
        K_SLOTS=k_slots,
        BLOCK_M=BLOCK_M,
        CROSS_GQA=cross_gqa,
        HAS_TTT=has_ttt,
        BLOCK_N=BLOCK_N,
        D=D,
        **launch_kwargs,
    )
    return out


@triton.jit
def _ragged_three_part_attn_fwd_kernel(
    Q_ptr,
    CROSS_K_ptr, CROSS_V_ptr,
    TTT_K_ptr, TTT_V_ptr,
    CURR_K_ptr, CURR_V_ptr,
    CROSS_LENGTHS_ptr, LEAF_OWNER_ptr,
    TTT_MASK_ptr, Out_ptr,
    max_cross_count,
    num_requests,
    ttt_count,
    n_kv_groups,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_ck_b, stride_ck_h, stride_ck_t, stride_ck_d,
    stride_cv_b, stride_cv_h, stride_cv_t, stride_cv_d,
    stride_tk_b, stride_tk_h, stride_tk_t, stride_tk_d,
    stride_tv_b, stride_tv_h, stride_tv_t, stride_tv_d,
    stride_qk_b, stride_qk_h, stride_qk_t, stride_qk_d,
    stride_qv_b, stride_qv_h, stride_qv_t, stride_qv_d,
    stride_mb, stride_mc,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Owner-indexed 3-buffer attention for heterogeneous pending leaves."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // n_kv_groups
    owner_raw = tl.load(LEAF_OWNER_ptr + pid_b)
    owner = tl.maximum(tl.minimum(owner_raw, num_requests - 1), 0)
    cross_count = tl.load(CROSS_LENGTHS_ptr + owner)

    offs_m = tl.arange(0, BLOCK_M)
    m_in_bounds = offs_m < K_SLOTS
    offs_d = tl.arange(0, D)
    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
        mask=m_in_bounds[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    ck_base = CROSS_K_ptr + owner * stride_ck_b + pid_kh * stride_ck_h
    cv_base = CROSS_V_ptr + owner * stride_cv_b + pid_kh * stride_cv_h
    for n_start in range(0, max_cross_count, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < cross_count
        k = tl.load(
            ck_base + offs_n[:, None] * stride_ck_t + offs_d[None, :] * stride_ck_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        v = tl.load(
            cv_base + offs_n[:, None] * stride_cv_t + offs_d[None, :] * stride_cv_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(n_in_bounds[None, :], qk, -float("inf"))
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    tk_base = TTT_K_ptr + owner * stride_tk_b + pid_kh * stride_tk_h
    tv_base = TTT_V_ptr + owner * stride_tv_b + pid_kh * stride_tv_h
    mask_base = TTT_MASK_ptr + pid_b * stride_mb
    for n_start in range(0, ttt_count, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < ttt_count
        k = tl.load(
            tk_base + offs_n[:, None] * stride_tk_t + offs_d[None, :] * stride_tk_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        v = tl.load(
            tv_base + offs_n[:, None] * stride_tv_t + offs_d[None, :] * stride_tv_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        ttt_vis = tl.load(
            mask_base + offs_n * stride_mc,
            mask=n_in_bounds,
            other=0,
        ).to(tl.int1)
        qk = tl.where(ttt_vis[None, :] & n_in_bounds[None, :], qk, -float("inf"))
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    qk_base = CURR_K_ptr + pid_b * stride_qk_b + pid_kh * stride_qk_h
    qv_base = CURR_V_ptr + pid_b * stride_qv_b + pid_kh * stride_qv_h
    for n_start in range(0, K_SLOTS, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < K_SLOTS
        k = tl.load(
            qk_base + offs_n[:, None] * stride_qk_t + offs_d[None, :] * stride_qk_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        v = tl.load(
            qv_base + offs_n[:, None] * stride_qv_t + offs_d[None, :] * stride_qv_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        causal_vis = (offs_n[None, :] <= offs_m[:, None]) & n_in_bounds[None, :]
        qk = tl.where(causal_vis, qk, -float("inf"))
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    o_base = Out_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    tl.store(
        o_base + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d,
        acc.to(Out_ptr.dtype.element_ty),
        mask=m_in_bounds[:, None],
    )


def ragged_three_part_attention(
    q: torch.Tensor,
    cross_k: torch.Tensor,
    cross_v: torch.Tensor,
    cross_lengths: torch.Tensor,
    leaf_owner: torch.Tensor,
    ttt_k: torch.Tensor,
    ttt_v: torch.Tensor,
    ttt_mask: torch.Tensor,
    curr_k: torch.Tensor,
    curr_v: torch.Tensor,
    max_cross_count: int,
    k_slots: int,
    n_kv_groups: int,
    scale: float,
) -> torch.Tensor:
    """Attend flat leaves to request-owned ``cross -> TTT -> current`` KV."""
    if q.ndim != 4 or cross_k.ndim != 4 or ttt_k.ndim != 4 or curr_k.ndim != 4:
        raise ValueError("ragged three-part attention tensors must be rank 4")
    leaves, heads, query_slots, head_dim = q.shape
    n_kv_groups = int(n_kv_groups)
    if n_kv_groups <= 0 or heads % n_kv_groups != 0:
        raise ValueError("query heads must be divisible by n_kv_groups")
    kv_heads = heads // n_kv_groups
    if query_slots != int(k_slots) or head_dim != 128:
        raise ValueError("ragged three-part attention expects K query slots with D=128")
    if cross_k.shape != cross_v.shape or ttt_k.shape != ttt_v.shape:
        raise ValueError("request-owned key/value shapes must match")
    if curr_k.shape != curr_v.shape:
        raise ValueError("current key/value shapes must match")
    requests = cross_k.shape[0]
    if cross_k.shape[1] != kv_heads or ttt_k.shape[1] != kv_heads:
        raise ValueError("cross and TTT KV must use Hkv heads")
    if cross_k.shape[3] != head_dim:
        raise ValueError("cross KV head dimension must match query")
    if ttt_k.shape[0] != requests or ttt_k.shape[3] != head_dim:
        raise ValueError("TTT KV must have one row per request")
    if curr_k.shape[:2] != (leaves, kv_heads) or curr_k.shape[3] != head_dim:
        raise ValueError("current KV must have one Hkv row per leaf")
    if curr_k.shape[2] < int(k_slots):
        raise ValueError("current KV width is smaller than K slots")
    if ttt_mask.shape != (leaves, ttt_k.shape[2]):
        raise ValueError("TTT mask must have one row per leaf and one column per TTT slot")
    if cross_lengths.shape != (requests,) or leaf_owner.shape != (leaves,):
        raise ValueError("ragged block-2 metadata shapes do not match requests/leaves")
    if cross_lengths.dtype != torch.long or leaf_owner.dtype != torch.long:
        raise TypeError("ragged block-2 metadata must use torch.long")
    if not cross_lengths.is_contiguous() or not leaf_owner.is_contiguous():
        raise ValueError("ragged block-2 metadata must be contiguous")
    tensors = (
        q, cross_k, cross_v, ttt_k, ttt_v, ttt_mask,
        curr_k, curr_v, cross_lengths, leaf_owner,
    )
    if q.device.type != "cuda" or any(tensor.device != q.device for tensor in tensors):
        raise ValueError("ragged three-part tensors must share one CUDA device")
    if any(tensor.dtype != q.dtype for tensor in (cross_k, cross_v, ttt_k, ttt_v, curr_k, curr_v)):
        raise TypeError("ragged three-part Q/K/V dtypes must match")
    if not 0 < int(max_cross_count) <= cross_k.shape[2]:
        raise ValueError("max_cross_count is outside cross-cache capacity")

    ttt_mask_bool = ttt_mask if ttt_mask.dtype == torch.bool else ttt_mask.bool()
    out = torch.empty_like(q)
    block_m = _query_block_m(query_slots)
    block_n = RUNTIME_CAPABILITIES.attention_block_n
    _ragged_three_part_attn_fwd_kernel[(leaves, heads)](
        q,
        cross_k, cross_v,
        ttt_k, ttt_v,
        curr_k, curr_v,
        cross_lengths, leaf_owner,
        ttt_mask_bool, out,
        int(max_cross_count), requests, ttt_k.shape[2], n_kv_groups,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cross_k.stride(0), cross_k.stride(1), cross_k.stride(2), cross_k.stride(3),
        cross_v.stride(0), cross_v.stride(1), cross_v.stride(2), cross_v.stride(3),
        ttt_k.stride(0), ttt_k.stride(1), ttt_k.stride(2), ttt_k.stride(3),
        ttt_v.stride(0), ttt_v.stride(1), ttt_v.stride(2), ttt_v.stride(3),
        curr_k.stride(0), curr_k.stride(1), curr_k.stride(2), curr_k.stride(3),
        curr_v.stride(0), curr_v.stride(1), curr_v.stride(2), curr_v.stride(3),
        ttt_mask_bool.stride(0), ttt_mask_bool.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale=scale,
        K_SLOTS=int(k_slots),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D=head_dim,
    )
    return out


# ============================================================
#   GQA-native tree attention (no repeat_kv)
# ============================================================
#
# Draft forward calls `tree_attention` with k/v already repeated to Hq heads
# (`repeat_kv(k, n_kv_groups)`). That costs:
#   - 2 extra kernel launches per layer (k and v repeat)
#   - 4× memory bandwidth for K/V cache writes and reads
#
# The kernel below indexes K/V via pid_hkv = pid_h // GROUP_SIZE, so callers
# can pass [B, Hkv, C, D] K/V directly. Math is identical (each Q head sees
# the same KV head its repeat_kv'd copy would have pointed at), but saves the
# repeat launches and lets the kernel stream smaller KV tiles.


@triton.jit
def _tree_attn_gqa_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    cross_count,
    M, C,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_c, stride_k_d,
    stride_v_b, stride_v_h, stride_v_c, stride_v_d,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,   # Hq / Hkv
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_hkv = pid_h // GROUP_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q_ptrs = q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d
    q_mask = offs_m[:, None] < M
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    q_pos = offs_m // K_SLOTS
    q_slot = offs_m - q_pos * K_SLOTS

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    k_base = K_ptr + pid_b * stride_k_b + pid_hkv * stride_k_h
    v_base = V_ptr + pid_b * stride_v_b + pid_hkv * stride_v_h

    for n_start in range(0, C, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < C

        k_ptrs = k_base + offs_n[:, None] * stride_k_c + offs_d[None, :] * stride_k_d
        v_ptrs = v_base + offs_n[:, None] * stride_v_c + offs_d[None, :] * stride_v_d
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale

        is_cross = offs_n[None, :] < cross_count
        curr_kv = offs_n - cross_count
        curr_kv_pos = curr_kv // K_SLOTS
        curr_kv_slot = curr_kv - curr_kv_pos * K_SLOTS
        curr_visible = (curr_kv_pos[None, :] < q_pos[:, None]) | (
            (curr_kv_pos[None, :] == q_pos[:, None]) &
            (curr_kv_slot[None, :] <= q_slot[:, None])
        )
        mask = (is_cross | curr_visible) & n_in_bounds[None, :]
        qk = tl.where(mask, qk, -float('inf'))

        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    acc = acc / l_i[:, None]

    o_base = Out_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    o_ptrs = o_base + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=q_mask)


def tree_attention_gqa(q, k, v, cross_count, k_slots, scale):
    """GQA variant of tree_attention: k/v are [B, Hkv, C, D] (not pre-repeated).

    Saves two repeat_kv launches + 4× memory traffic in draft forward.

    Args:
        q: [B, Hq, M, D] query (post-RoPE)
        k: [B, Hkv, C, D] keys (NOT repeated to Hq heads)
        v: [B, Hkv, C, D] values (NOT repeated to Hq heads)
        cross_count: int, cross-region length
        k_slots: int, K slots per position (tree mask param)
        scale: float

    Returns:
        out: [B, Hq, M, D]
    """
    B, Hq, M, D = q.shape
    _, Hkv, C, _ = k.shape
    assert v.shape == k.shape
    assert q.dtype == k.dtype == v.dtype
    assert D == 128, f"kernel hardcoded for D=128, got {D}"
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
    group_size = Hq // Hkv

    BLOCK_M = _query_block_m(M, cap=32)
    BLOCK_N = RUNTIME_CAPABILITIES.attention_block_n

    out = torch.empty_like(q)

    grid = (triton.cdiv(M, BLOCK_M), B, Hq)
    _tree_attn_gqa_fwd_kernel[grid](
        q, k, v, out,
        cross_count, M, C,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale=scale,
        K_SLOTS=k_slots,
        GROUP_SIZE=group_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
    )
    return out
