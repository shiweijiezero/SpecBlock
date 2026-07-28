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

_DUMMY_MASK = None
_ATTENTION_BLOCK_N = 32 if "metax" in torch.__version__.lower() else 64


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

    # Block sizes — BLOCK_M tuned for small-M decoding
    BLOCK_M = 16 if M <= 16 else 32
    BLOCK_N = _ATTENTION_BLOCK_N

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
    BLOCK_M: tl.constexpr,         # padded to >=16 for tl.dot portability
    HAS_TTT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Attention for a single prediction position (M = K_SLOTS queries).

    Mask regions along KV:
      [0, cross_count)             — cross cache, all visible
      [cross_count, curr_start)    — ttt cache, masked by TTT_mask
      [curr_start, C)              — current K, causal within the K slots

    Each program handles all M queries for one (batch, head). The physical
    query tile is padded to BLOCK_M because some Triton backends require every
    non-batch tl.dot dimension to be at least 16.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    m_in_bounds = offs_m < K_SLOTS
    offs_d = tl.arange(0, D)

    # Load Q; padded rows participate in the tile but are not stored.
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

    BLOCK_M = 16 if M <= 16 else 32
    BLOCK_N = _ATTENTION_BLOCK_N
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
#   * curr_k is post-repeat_kv [B, H, K, D] (caller still expands current
#     for cross_cache write coherency; could remove later)
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
    BLOCK_M: tl.constexpr,         # padded to >=16 for tl.dot portability
    HAS_TTT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """3-buffer attention. Grid: (B, H).

    Region layout (along KV axis):
      cross [0, cross_count)        - all visible, GQA-expanded source
      ttt   [0, ttt_count)          - per-mask visible, GQA-non-expanded
      curr  [0, K_SLOTS)            - causal within slots, GQA-expanded

    Online softmax state carried across all 3 loops.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // n_kv_groups

    offs_m = tl.arange(0, BLOCK_M)
    m_in_bounds = offs_m < K_SLOTS
    offs_d = tl.arange(0, D)

    # Load Q; padded rows participate in the tile but are not stored.
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
    # cross_k stride_ck_b may be 0 (broadcast view) -> identical pointer for all batches.
    ck_base = CROSS_K_ptr + pid_b * stride_ck_b + pid_h * stride_ck_h
    cv_base = CROSS_V_ptr + pid_b * stride_cv_b + pid_h * stride_cv_h
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
    cross_k: torch.Tensor,                 # [B, H, T_cross, D] (stride[0] may be 0)
    cross_v: torch.Tensor,
    ttt_k: torch.Tensor,                   # [B, KH, T_ttt, D] (GQA non-expanded), or None
    ttt_v: torch.Tensor,
    curr_k: torch.Tensor,                  # [B, H, K, D] (post repeat_kv)
    curr_v: torch.Tensor,
    cross_count: int,
    ttt_count: int,
    ttt_mask: torch.Tensor,                # [B, ttt_count] bool, or None
    k_slots: int,
    n_kv_groups: int,
    scale: float,
) -> torch.Tensor:
    """3-buffer attention without cat. Returns out [B, H, K, D].

    cross_k may be a `.expand(B, ...)` view with stride[0]==0; the kernel
    reads broadcast-correctly via stride.
    ttt_k is GQA-non-expanded; kernel does GQA index via pid_h // n_kv_groups.
    """
    B, H, M, D = q.shape
    assert M == k_slots
    assert D == 128

    has_ttt = (ttt_count > 0) and (ttt_mask is not None) and (ttt_k is not None)

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

    BLOCK_M = 16 if M <= 16 else 32
    BLOCK_N = _ATTENTION_BLOCK_N
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
        HAS_TTT=has_ttt,
        BLOCK_N=BLOCK_N,
        D=D,
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

    BLOCK_M = 16 if M <= 16 else 32
    BLOCK_N = _ATTENTION_BLOCK_N

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
