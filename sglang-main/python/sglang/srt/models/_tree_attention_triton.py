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

from typing import Optional

import torch
import triton
import triton.language as tl

_DUMMY_MASK = None
_DUMMY_MASK_CROSS = None


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
    BLOCK_N = 64

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
def _tree_attn_paged_fwd_kernel(
    Q_ptr,
    POOL_K_ptr,
    POOL_V_ptr,
    CROSS_LOC_ptr,
    CROSS_MASK_ptr,
    CURR_K_ptr,
    CURR_V_ptr,
    Out_ptr,
    cross_count,
    M,
    n_kv_groups,
    stride_q_b,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_pk_slot,
    stride_pk_h,
    stride_pk_d,
    stride_pv_slot,
    stride_pv_h,
    stride_pv_d,
    stride_loc_b,
    stride_loc_t,
    stride_cm_b,
    stride_cm_t,
    stride_ck_b,
    stride_ck_h,
    stride_ck_m,
    stride_ck_d,
    stride_cv_b,
    stride_cv_h,
    stride_cv_m,
    stride_cv_d,
    stride_o_b,
    stride_o_h,
    stride_o_m,
    stride_o_d,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,
    HAS_CROSS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Paged cross + current tree attention for batched refresh."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_kh = pid_h // n_kv_groups

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_mask = offs_m[:, None] < M

    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q = tl.load(
        q_base
        + offs_m[:, None] * stride_q_m
        + offs_d[None, :] * stride_q_d,
        mask=q_mask,
        other=0.0,
    )

    q_pos = offs_m // K_SLOTS
    q_slot = offs_m - q_pos * K_SLOTS
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Persistent cross history is GQA-expanded in the paged pool.
    loc_base = CROSS_LOC_ptr + pid_b * stride_loc_b
    cm_base = CROSS_MASK_ptr + pid_b * stride_cm_b
    for n_start in range(0, cross_count, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < cross_count
        idx = tl.load(
            loc_base + offs_n * stride_loc_t,
            mask=n_in_bounds,
            other=0,
        ).to(tl.int64)
        k = tl.load(
            POOL_K_ptr
            + idx[:, None] * stride_pk_slot
            + pid_h * stride_pk_h
            + offs_d[None, :] * stride_pk_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        v = tl.load(
            POOL_V_ptr
            + idx[:, None] * stride_pv_slot
            + pid_h * stride_pv_h
            + offs_d[None, :] * stride_pv_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        if HAS_CROSS_MASK:
            cross_vis = tl.load(
                cm_base + offs_n * stride_cm_t,
                mask=n_in_bounds,
                other=0,
            ).to(tl.int1)
            visible = cross_vis[None, :] & n_in_bounds[None, :]
        else:
            visible = n_in_bounds[None, :]
        qk = tl.where(visible, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        has_scores = m_i_new != -float("inf")
        alpha = tl.where(has_scores, tl.exp(m_i - m_i_new), 1.0)
        p = tl.where(
            has_scores[:, None],
            tl.exp(qk - m_i_new[:, None]),
            0.0,
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    # Current accepted positions remain non-expanded GQA tensors.  The
    # position/slot mask prevents padded future positions from influencing
    # any real per-request output gathered after replay.
    ck_base = CURR_K_ptr + pid_b * stride_ck_b + pid_kh * stride_ck_h
    cv_base = CURR_V_ptr + pid_b * stride_cv_b + pid_kh * stride_cv_h
    for n_start in range(0, M, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < M
        k = tl.load(
            ck_base
            + offs_n[:, None] * stride_ck_m
            + offs_d[None, :] * stride_ck_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        v = tl.load(
            cv_base
            + offs_n[:, None] * stride_cv_m
            + offs_d[None, :] * stride_cv_d,
            mask=n_in_bounds[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        kv_pos = offs_n // K_SLOTS
        kv_slot = offs_n - kv_pos * K_SLOTS
        curr_visible = (kv_pos[None, :] < q_pos[:, None]) | (
            (kv_pos[None, :] == q_pos[:, None])
            & (kv_slot[None, :] <= q_slot[:, None])
        )
        qk = tl.where(
            curr_visible & n_in_bounds[None, :],
            qk,
            -float("inf"),
        )

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
        o_base
        + offs_m[:, None] * stride_o_m
        + offs_d[None, :] * stride_o_d,
        acc.to(Out_ptr.dtype.element_ty),
        mask=q_mask,
    )


def tree_attention_paged(
    q: torch.Tensor,
    pool_k_layer: torch.Tensor,
    pool_v_layer: torch.Tensor,
    cross_loc: torch.Tensor,
    cross_mask: Optional[torch.Tensor],
    curr_k: torch.Tensor,
    curr_v: torch.Tensor,
    cross_count: int,
    k_slots: int,
    n_kv_groups: int,
    scale: float,
) -> torch.Tensor:
    """Tree attention over paged cross history and non-expanded current KV."""
    B, H, M, D = q.shape
    assert curr_k.shape[0] == B and curr_k.shape[2] == M
    assert curr_v.shape == curr_k.shape
    assert D == 128, f"kernel hardcoded for D=128, got {D}"
    assert pool_k_layer.dim() == 3 and pool_k_layer.shape[1] == H
    assert pool_v_layer.shape == pool_k_layer.shape
    assert cross_loc.shape == (B, cross_count)
    assert cross_loc.dtype == torch.int64

    has_cross_mask = cross_mask is not None
    if has_cross_mask:
        cross_mask_bool = (
            cross_mask if cross_mask.dtype == torch.bool else cross_mask.bool()
        )
        assert cross_mask_bool.shape == cross_loc.shape
        stride_cm_b = cross_mask_bool.stride(0)
        stride_cm_t = cross_mask_bool.stride(1)
    else:
        global _DUMMY_MASK_CROSS
        if _DUMMY_MASK_CROSS is None or _DUMMY_MASK_CROSS.device != q.device:
            _DUMMY_MASK_CROSS = torch.empty(
                1, device=q.device, dtype=torch.bool,
            )
        cross_mask_bool = _DUMMY_MASK_CROSS
        stride_cm_b = 0
        stride_cm_t = 0

    block_m = 16 if M <= 16 else 32
    block_n = 64
    out = torch.empty_like(q)
    grid = (triton.cdiv(M, block_m), B, H)
    _tree_attn_paged_fwd_kernel[grid](
        q,
        pool_k_layer,
        pool_v_layer,
        cross_loc,
        cross_mask_bool,
        curr_k,
        curr_v,
        out,
        cross_count,
        M,
        n_kv_groups,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        pool_k_layer.stride(0),
        pool_k_layer.stride(1),
        pool_k_layer.stride(2),
        pool_v_layer.stride(0),
        pool_v_layer.stride(1),
        pool_v_layer.stride(2),
        cross_loc.stride(0),
        cross_loc.stride(1),
        stride_cm_b,
        stride_cm_t,
        curr_k.stride(0),
        curr_k.stride(1),
        curr_k.stride(2),
        curr_k.stride(3),
        curr_v.stride(0),
        curr_v.stride(1),
        curr_v.stride(2),
        curr_v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        scale=scale,
        K_SLOTS=k_slots,
        HAS_CROSS_MASK=has_cross_mask,
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
    HAS_TTT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Attention for a single prediction position (M = K_SLOTS queries).

    Mask regions along KV:
      [0, cross_count)             — cross cache, all visible
      [cross_count, curr_start)    — ttt cache, masked by TTT_mask
      [curr_start, C)              — current K, causal within the K slots

    Because M = K_SLOTS (small, e.g. 4), each program handles all M queries
    for one (batch, head). Grid is (B, H).
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = tl.arange(0, K_SLOTS)
    offs_d = tl.arange(0, D)

    # Load Q
    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
    )

    # Accumulators
    m_i = tl.full([K_SLOTS], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([K_SLOTS], dtype=tl.float32)
    acc = tl.zeros([K_SLOTS, D], dtype=tl.float32)

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

    BLOCK_N = 64
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

    offs_m = tl.arange(0, K_SLOTS)
    offs_d = tl.arange(0, D)

    # Load Q
    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
    )

    # Online softmax state (carried across 3 region loops)
    m_i = tl.full([K_SLOTS], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([K_SLOTS], dtype=tl.float32)
    acc = tl.zeros([K_SLOTS, D], dtype=tl.float32)

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

    BLOCK_N = 64
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
        HAS_TTT=has_ttt,
        BLOCK_N=BLOCK_N,
        D=D,
    )
    return out


# ============================================================
#   3-way single-pos attention with paged cross K/V (Stage B+ kernel)
# ============================================================
#
# Stage B graph capture is OOM with the dense `three_part_attention`
# wrapper because PyTorch fancy-index gather (`pool[indices]`) materialises
# a [tp, n_heads, count, D] cross_k tensor inside the cuda graph private
# pool -- 5-10 GB of graph workspace per attention op, multiplied by
# layers + warmup = 40+ GB OOM.
#
# This paged variant replaces the dense cross_k/v inputs with a
# `(pool_k_layer, pool_v_layer, cross_loc)` triple.  cross_loc is the
# `[tp, max_count]` int64 tensor of slot indices into the per-layer pool.
# Inner cross loop does an indirect load:
#
#     idx[BLOCK_N] = cross_loc[pid_b, n_start..n_start+BLOCK_N]
#     k[BLOCK_N, D] = pool_k_layer[idx, pid_h, :]
#
# The fancy-index intermediate is gone -- graph private pool only holds
# q/k/v/acc/out tiles (KB / op).  Pool buffers are caller-allocated and
# stable across captures (worker.spec_kv_pool).  cross_mask is per-req
# valid bits (pad indices = 0 sentinel slot, masked here so 0-K/V values
# do not contribute to softmax).


@triton.jit
def _three_part_attn_paged_fwd_kernel(
    Q_ptr,
    POOL_K_ptr, POOL_V_ptr,
    CROSS_LOC_ptr,
    CROSS_MASK_ptr,
    TTT_K_ptr, TTT_V_ptr,
    CURR_K_ptr, CURR_V_ptr,
    TTT_MASK_ptr, Out_ptr,
    cross_count,                 # int32 (= max_count over batch; mask handles per-req short)
    ttt_count,                   # int32
    n_kv_groups,                 # int32 (= n_heads / n_kv_heads)
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_pk_slot, stride_pk_h, stride_pk_d,        # pool_k_layer: [pool_size, n_heads, D]
    stride_pv_slot, stride_pv_h, stride_pv_d,
    stride_loc_b, stride_loc_t,                       # cross_loc: [tp, max_count]
    stride_cm_b, stride_cm_t,                         # cross_mask: [tp, max_count] (0=invalid)
    stride_tk_b, stride_tk_h, stride_tk_t, stride_tk_d,
    stride_tv_b, stride_tv_h, stride_tv_t, stride_tv_d,
    stride_qk_b, stride_qk_h, stride_qk_t, stride_qk_d,
    stride_qv_b, stride_qv_h, stride_qv_t, stride_qv_d,
    stride_mb, stride_mc,         # ttt mask stride
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,
    HAS_TTT: tl.constexpr,
    HAS_CROSS_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """3-buffer paged attention. Grid: (B, H).

    Cross K/V are gathered via `cross_loc` indices into the per-layer pool;
    no dense [tp, H, T, D] intermediate is materialized.  Online softmax
    state is carried across the 3 region loops, identical math to the
    `_three_part_attn_fwd_kernel` dense path.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // n_kv_groups

    offs_m = tl.arange(0, K_SLOTS)
    offs_d = tl.arange(0, D)

    # Load Q
    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
    )

    # Online softmax state (carried across 3 region loops)
    m_i = tl.full([K_SLOTS], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([K_SLOTS], dtype=tl.float32)
    acc = tl.zeros([K_SLOTS, D], dtype=tl.float32)

    # ---- Loop 1: cross region (paged gather; per-req mask) ----
    loc_base = CROSS_LOC_ptr + pid_b * stride_loc_b
    cm_base = CROSS_MASK_ptr + pid_b * stride_cm_b
    for n_start in range(0, cross_count, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_in_bounds = offs_n < cross_count
        # Load slot indices for this block.  Pad slots (mask out) read
        # cross_loc=0 (the pool's zero-sentinel slot); their gathered
        # K/V are zeros and we mask them out below.
        idx = tl.load(
            loc_base + offs_n * stride_loc_t,
            mask=n_in_bounds, other=0,
        ).to(tl.int64)
        # Gather K/V from pool: shape [BLOCK_N, D].  Indirect load -- no
        # intermediate dense buffer, fits in graph private pool.
        k = tl.load(
            POOL_K_ptr
            + idx[:, None] * stride_pk_slot
            + pid_h * stride_pk_h
            + offs_d[None, :] * stride_pk_d,
            mask=n_in_bounds[:, None], other=0.0,
        )
        v = tl.load(
            POOL_V_ptr
            + idx[:, None] * stride_pv_slot
            + pid_h * stride_pv_h
            + offs_d[None, :] * stride_pv_d,
            mask=n_in_bounds[:, None], other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        # Apply cross visibility mask.  When HAS_CROSS_MASK=False (single-
        # req or all-valid path), only n_in_bounds is enforced.
        if HAS_CROSS_MASK:
            cross_vis = tl.load(
                cm_base + offs_n * stride_cm_t,
                mask=n_in_bounds, other=0,
            ).to(tl.int1)
            qk = tl.where(cross_vis[None, :] & n_in_bounds[None, :], qk, -float('inf'))
        else:
            qk = tl.where(n_in_bounds[None, :], qk, -float('inf'))
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        # A bucket may contain no live cross entries for this replay.  Keep
        # the online-softmax state unchanged instead of evaluating
        # exp(-inf - -inf), which would poison later TTT/current regions.
        has_scores = m_i_new != -float('inf')
        alpha = tl.where(has_scores, tl.exp(m_i - m_i_new), 1.0)
        p = tl.where(
            has_scores[:, None], tl.exp(qk - m_i_new[:, None]), 0.0
        )
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
            # Static pending buckets include pad rows whose TTT mask is
            # entirely false.  Preserve empty state until the current causal
            # region contributes the first visible key.
            has_scores = m_i_new != -float('inf')
            alpha = tl.where(has_scores, tl.exp(m_i - m_i_new), 1.0)
            p = tl.where(
                has_scores[:, None], tl.exp(qk - m_i_new[:, None]), 0.0
            )
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_i_new

    # ---- Loop 3: curr region (causal within slots; GQA non-expanded) ----
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
    )


def three_part_attention_paged(
    q: torch.Tensor,                        # [B, H, K, D]
    pool_k_layer: torch.Tensor,             # [pool_size, n_heads, D] (one layer slice)
    pool_v_layer: torch.Tensor,             # [pool_size, n_heads, D]
    cross_loc: torch.Tensor,                # [B, max_count] int64 (paged indices; 0 = sentinel)
    cross_mask: Optional[torch.Tensor],     # [B, max_count] bool (None = all valid)
    ttt_k: Optional[torch.Tensor],          # [B, KH, T_ttt, D] (GQA non-expanded), or None
    ttt_v: Optional[torch.Tensor],
    curr_k: torch.Tensor,                   # [B, KH, K, D] (GQA non-expanded)
    curr_v: torch.Tensor,
    cross_count: int,                       # = cross_loc.shape[1] (max over batch)
    ttt_count: int,
    ttt_mask: Optional[torch.Tensor],       # [B, ttt_count] bool, or None
    k_slots: int,
    n_kv_groups: int,
    scale: float,
) -> torch.Tensor:
    """3-buffer paged attention. Returns out [B, H, K, D].

    Replaces ``three_part_attention``'s dense ``cross_k``/``cross_v``
    inputs with paged ``(pool_k_layer, pool_v_layer, cross_loc)``.  No
    dense cross_k tensor is allocated; the kernel's inner cross loop
    does an indirect load via ``cross_loc[pid_b, ...]`` straight into
    the per-layer pool.  Designed for cuda graph capture: the only
    graph-private intermediates are the per-block q/k/v tiles (BLOCK_N
    × D), so private pool stays in KB-class.
    """
    B, H, M, D = q.shape
    assert M == k_slots
    assert D == 128
    assert pool_k_layer.dim() == 3 and pool_k_layer.shape[1] == H, (
        f"pool_k_layer expected [pool_size, {H}, {D}]; got {tuple(pool_k_layer.shape)}"
    )
    assert cross_loc.dim() == 2 and cross_loc.shape[0] == B, (
        f"cross_loc expected [{B}, max_count]; got {tuple(cross_loc.shape)}"
    )
    assert cross_loc.dtype == torch.int64, "cross_loc must be int64"

    has_ttt = (ttt_count > 0) and (ttt_mask is not None) and (ttt_k is not None)
    has_cross_mask = cross_mask is not None

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

    if has_cross_mask:
        cross_mask_bool = cross_mask if cross_mask.dtype == torch.bool else cross_mask.bool()
        stride_cm_b = cross_mask_bool.stride(0)
        stride_cm_t = cross_mask_bool.stride(1)
    else:
        global _DUMMY_MASK_CROSS
        if _DUMMY_MASK_CROSS is None or _DUMMY_MASK_CROSS.device != q.device:
            _DUMMY_MASK_CROSS = torch.empty(1, device=q.device, dtype=torch.bool)
        cross_mask_bool = _DUMMY_MASK_CROSS
        stride_cm_b = 0
        stride_cm_t = 0

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

    BLOCK_N = 64
    grid = (B, H)
    _three_part_attn_paged_fwd_kernel[grid](
        q,
        pool_k_layer, pool_v_layer,
        cross_loc,
        cross_mask_bool,
        ttt_k_arg, ttt_v_arg,
        curr_k, curr_v,
        ttt_mask_bool, out,
        cross_count, ttt_count, n_kv_groups,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        pool_k_layer.stride(0), pool_k_layer.stride(1), pool_k_layer.stride(2),
        pool_v_layer.stride(0), pool_v_layer.stride(1), pool_v_layer.stride(2),
        cross_loc.stride(0), cross_loc.stride(1),
        stride_cm_b, stride_cm_t,
        tk_strides[0], tk_strides[1], tk_strides[2], tk_strides[3],
        tv_strides[0], tv_strides[1], tv_strides[2], tv_strides[3],
        curr_k.stride(0), curr_k.stride(1), curr_k.stride(2), curr_k.stride(3),
        curr_v.stride(0), curr_v.stride(1), curr_v.stride(2), curr_v.stride(3),
        stride_mb, stride_mc,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale=scale,
        K_SLOTS=k_slots,
        HAS_TTT=has_ttt,
        HAS_CROSS_MASK=has_cross_mask,
        BLOCK_N=BLOCK_N,
        D=D,
    )
    return out


@triton.jit
def _grouped_three_part_attn_paged_fwd_kernel(
    Q_ptr,
    POOL_K_ptr, POOL_V_ptr,
    CROSS_LOC_ptr, CROSS_MASK_ptr,
    TTT_K_ptr, TTT_V_ptr,
    CURR_K_ptr, CURR_V_ptr,
    TTT_MASK_ptr, Out_ptr,
    cross_count,
    leaves_per_request,
    n_kv_groups,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_pk_slot, stride_pk_h, stride_pk_d,
    stride_pv_slot, stride_pv_h, stride_pv_d,
    stride_loc_b, stride_loc_t,
    stride_cm_b, stride_cm_t,
    stride_tk_b, stride_tk_h, stride_tk_t, stride_tk_d,
    stride_tv_b, stride_tv_h, stride_tv_t, stride_tv_d,
    stride_ck_b, stride_ck_h, stride_ck_t, stride_ck_d,
    stride_cv_b, stride_cv_h, stride_cv_t, stride_cv_d,
    stride_mb, stride_ml, stride_mt,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,
    HAS_CROSS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Grouped pending-leaf attention with request-shared cross history.

    ``Q/CURR`` keep the flattened ``[B * P, H, K, D]`` layout used by the
    draft model, but one program processes multiple leaves from the same
    request.  The persistent prefix tile is therefore loaded once for up to
    ``BLOCK_M / K`` leaves instead of once per leaf.  TTT visibility and
    current-slot causality remain leaf-local.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_kh = pid_h // n_kv_groups

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    leaf = offs_m // K_SLOTS
    q_slot = offs_m - leaf * K_SLOTS
    q_valid = leaf < leaves_per_request
    flat_b = pid_b * leaves_per_request + leaf

    q = tl.load(
        Q_ptr
        + flat_b[:, None] * stride_q_b
        + pid_h * stride_q_h
        + q_slot[:, None] * stride_q_m
        + offs_d[None, :] * stride_q_d,
        mask=q_valid[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Persistent prefix: shared by every pending leaf of this request.
    loc_base = CROSS_LOC_ptr + pid_b * stride_loc_b
    cm_base = CROSS_MASK_ptr + pid_b * stride_cm_b
    for n_start in range(0, cross_count, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_valid = offs_n < cross_count
        idx = tl.load(
            loc_base + offs_n * stride_loc_t,
            mask=n_valid,
            other=0,
        ).to(tl.int64)
        k = tl.load(
            POOL_K_ptr
            + idx[:, None] * stride_pk_slot
            + pid_h * stride_pk_h
            + offs_d[None, :] * stride_pk_d,
            mask=n_valid[:, None],
            other=0.0,
        )
        v = tl.load(
            POOL_V_ptr
            + idx[:, None] * stride_pv_slot
            + pid_h * stride_pv_h
            + offs_d[None, :] * stride_pv_d,
            mask=n_valid[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        if HAS_CROSS_MASK:
            cross_vis = tl.load(
                cm_base + offs_n * stride_cm_t,
                mask=n_valid,
                other=0,
            ).to(tl.int1)
            visible = q_valid[:, None] & cross_vis[None, :] & n_valid[None, :]
        else:
            visible = q_valid[:, None] & n_valid[None, :]
        qk = tl.where(visible, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        has_scores = m_i_new != -float("inf")
        alpha = tl.where(has_scores, tl.exp(m_i - m_i_new), 1.0)
        p = tl.where(
            has_scores[:, None], tl.exp(qk - m_i_new[:, None]), 0.0,
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    # Previous-block TTT K/V are shared per request, while each leaf carries
    # its own valid-slot mask.
    # Tensor-core dot requires the reduction dimension to be at least 16;
    # pad the small K-slot TTT region logically without reading past storage.
    ttt_slots = tl.arange(0, 16)
    ttt_slot_valid = ttt_slots < K_SLOTS
    tk_base = TTT_K_ptr + pid_b * stride_tk_b + pid_kh * stride_tk_h
    tv_base = TTT_V_ptr + pid_b * stride_tv_b + pid_kh * stride_tv_h
    tk = tl.load(
        tk_base
        + ttt_slots[:, None] * stride_tk_t
        + offs_d[None, :] * stride_tk_d,
        mask=ttt_slot_valid[:, None],
        other=0.0,
    )
    tv = tl.load(
        tv_base
        + ttt_slots[:, None] * stride_tv_t
        + offs_d[None, :] * stride_tv_d,
        mask=ttt_slot_valid[:, None],
        other=0.0,
    )
    qk = tl.dot(q, tl.trans(tk)) * scale
    ttt_vis = tl.load(
        TTT_MASK_ptr
        + pid_b * stride_mb
        + leaf[:, None] * stride_ml
        + ttt_slots[None, :] * stride_mt,
        mask=q_valid[:, None] & ttt_slot_valid[None, :],
        other=0,
    ).to(tl.int1)
    qk = tl.where(
        q_valid[:, None] & ttt_slot_valid[None, :] & ttt_vis,
        qk,
        -float("inf"),
    )
    m_ij = tl.max(qk, axis=1)
    m_i_new = tl.maximum(m_i, m_ij)
    has_scores = m_i_new != -float("inf")
    alpha = tl.where(has_scores, tl.exp(m_i - m_i_new), 1.0)
    p = tl.where(
        has_scores[:, None], tl.exp(qk - m_i_new[:, None]), 0.0,
    )
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    acc += tl.dot(p.to(tv.dtype), tv)
    m_i = m_i_new

    # Current K/V differ by leaf.  Process the four slot keys one at a time so
    # the kernel does not materialize a BLOCK_M x K x D register tensor.
    for curr_slot in range(0, K_SLOTS):
        ck = tl.load(
            CURR_K_ptr
            + flat_b[:, None] * stride_ck_b
            + pid_kh * stride_ck_h
            + curr_slot * stride_ck_t
            + offs_d[None, :] * stride_ck_d,
            mask=q_valid[:, None],
            other=0.0,
        )
        cv = tl.load(
            CURR_V_ptr
            + flat_b[:, None] * stride_cv_b
            + pid_kh * stride_cv_h
            + curr_slot * stride_cv_t
            + offs_d[None, :] * stride_cv_d,
            mask=q_valid[:, None],
            other=0.0,
        )
        qk_curr = tl.sum(q.to(tl.float32) * ck.to(tl.float32), axis=1) * scale
        visible = q_valid & (curr_slot <= q_slot)
        qk_curr = tl.where(visible, qk_curr, -float("inf"))
        m_i_new = tl.maximum(m_i, qk_curr)
        alpha = tl.exp(m_i - m_i_new)
        p_curr = tl.exp(qk_curr - m_i_new)
        l_i = l_i * alpha + p_curr
        acc = acc * alpha[:, None] + p_curr[:, None] * cv
        m_i = m_i_new

    out = acc / l_i[:, None]
    tl.store(
        Out_ptr
        + flat_b[:, None] * stride_o_b
        + pid_h * stride_o_h
        + q_slot[:, None] * stride_o_m
        + offs_d[None, :] * stride_o_d,
        out.to(Out_ptr.dtype.element_ty),
        mask=q_valid[:, None],
    )


def grouped_three_part_attention_paged(
    q: torch.Tensor,
    pool_k_layer: torch.Tensor,
    pool_v_layer: torch.Tensor,
    cross_loc: torch.Tensor,
    cross_mask: Optional[torch.Tensor],
    ttt_k: torch.Tensor,
    ttt_v: torch.Tensor,
    curr_k: torch.Tensor,
    curr_v: torch.Tensor,
    ttt_mask: torch.Tensor,
    leaves_per_request: int,
    cross_count: int,
    k_slots: int,
    n_kv_groups: int,
    scale: float,
) -> torch.Tensor:
    """Paged draft attention grouped by request across pending leaves."""
    flat_batch, n_heads, query_slots, head_dim = q.shape
    request_batch = int(cross_loc.shape[0])
    assert flat_batch == request_batch * leaves_per_request
    assert query_slots == k_slots and head_dim == 128
    assert curr_k.shape[0] == flat_batch and curr_k.shape[2] == k_slots
    assert curr_v.shape == curr_k.shape
    assert ttt_k.shape[0] == request_batch and ttt_k.shape[2] == k_slots
    assert ttt_v.shape == ttt_k.shape
    assert ttt_mask.shape == (request_batch, leaves_per_request, k_slots)
    assert cross_loc.shape == (request_batch, cross_count)
    assert cross_loc.dtype == torch.int64

    if cross_mask is not None:
        cross_mask_bool = (
            cross_mask if cross_mask.dtype == torch.bool else cross_mask.bool()
        )
        assert cross_mask_bool.shape == cross_loc.shape
        stride_cm_b, stride_cm_t = cross_mask_bool.stride()
        has_cross_mask = True
    else:
        global _DUMMY_MASK_CROSS
        if _DUMMY_MASK_CROSS is None or _DUMMY_MASK_CROSS.device != q.device:
            _DUMMY_MASK_CROSS = torch.empty(1, device=q.device, dtype=torch.bool)
        cross_mask_bool = _DUMMY_MASK_CROSS
        stride_cm_b = stride_cm_t = 0
        has_cross_mask = False

    ttt_mask_bool = ttt_mask if ttt_mask.dtype == torch.bool else ttt_mask.bool()
    out = torch.empty_like(q)
    block_m = 16
    block_n = 64
    grid = (triton.cdiv(leaves_per_request * k_slots, block_m), request_batch, n_heads)
    _grouped_three_part_attn_paged_fwd_kernel[grid](
        q,
        pool_k_layer, pool_v_layer,
        cross_loc, cross_mask_bool,
        ttt_k, ttt_v,
        curr_k, curr_v,
        ttt_mask_bool, out,
        cross_count, leaves_per_request, n_kv_groups,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        pool_k_layer.stride(0), pool_k_layer.stride(1), pool_k_layer.stride(2),
        pool_v_layer.stride(0), pool_v_layer.stride(1), pool_v_layer.stride(2),
        cross_loc.stride(0), cross_loc.stride(1),
        stride_cm_b, stride_cm_t,
        ttt_k.stride(0), ttt_k.stride(1), ttt_k.stride(2), ttt_k.stride(3),
        ttt_v.stride(0), ttt_v.stride(1), ttt_v.stride(2), ttt_v.stride(3),
        curr_k.stride(0), curr_k.stride(1), curr_k.stride(2), curr_k.stride(3),
        curr_v.stride(0), curr_v.stride(1), curr_v.stride(2), curr_v.stride(3),
        ttt_mask_bool.stride(0), ttt_mask_bool.stride(1), ttt_mask_bool.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale=scale,
        K_SLOTS=k_slots,
        HAS_CROSS_MASK=has_cross_mask,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D=head_dim,
    )
    return out


# ============================================================
#   2-way attention (ttt + curr) with explicit LSE output
# ============================================================
#
# Stage D companion to flashinfer's paged cross attention.  The cross
# region is computed by flashinfer (returning out + lse); this kernel
# computes the ttt + curr regions of the SpecBlock-Shift draft attention
# and returns its own (out, lse) so the caller can perform an
# online-softmax merge:
#
#     m_total = max(lse_cross, lse_other)
#     w_cross = exp(lse_cross - m_total)
#     w_other = exp(lse_other - m_total)
#     out_total = (out_cross * w_cross + out_other * w_other)
#                 / (w_cross + w_other)
#
# The kernel structure mirrors `_three_part_attn_paged_fwd_kernel`
# minus the cross loop, plus an Lse_ptr write at the end.


@triton.jit
def _two_part_attn_fwd_kernel(
    Q_ptr,
    TTT_K_ptr, TTT_V_ptr,
    CURR_K_ptr, CURR_V_ptr,
    TTT_MASK_ptr,
    Out_ptr, Lse_ptr,
    ttt_count,                 # int32
    n_kv_groups,                # int32 (= n_heads / n_kv_heads)
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_tk_b, stride_tk_h, stride_tk_t, stride_tk_d,
    stride_tv_b, stride_tv_h, stride_tv_t, stride_tv_d,
    stride_qk_b, stride_qk_h, stride_qk_t, stride_qk_d,
    stride_qv_b, stride_qv_h, stride_qv_t, stride_qv_d,
    stride_mb, stride_mc,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_m,
    scale: tl.constexpr,
    K_SLOTS: tl.constexpr,
    HAS_TTT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """ttt + curr regions only.  Returns (out, lse) for online merge.

    Online softmax state (m_i, l_i, acc) is initialised to the empty
    value (-inf, 0, 0).  After processing both regions, ``out`` =
    acc / l_i and ``lse`` = m_i + log(l_i).  When neither region is
    present (HAS_TTT=False and K_SLOTS=0 -- not used in practice),
    ``lse`` would be -inf; callers must guard.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // n_kv_groups

    offs_m = tl.arange(0, K_SLOTS)
    offs_d = tl.arange(0, D)

    # Load Q
    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q = tl.load(
        q_base + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
    )

    # Online softmax state
    m_i = tl.full([K_SLOTS], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([K_SLOTS], dtype=tl.float32)
    acc = tl.zeros([K_SLOTS, D], dtype=tl.float32)

    # ---- Loop 1: ttt region (GQA-indexed; per-mask visible) ----
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

    # ---- Loop 2: curr region (causal within K_SLOTS; GQA non-expanded) ----
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

    # Normalize + write out
    acc = acc / l_i[:, None]
    o_base = Out_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    tl.store(
        o_base + offs_m[:, None] * stride_o_m + offs_d[None, :] * stride_o_d,
        acc.to(Out_ptr.dtype.element_ty),
    )

    # Write LSE = m_i + log(l_i).  When l_i == 0 (no valid keys, e.g.
    # K_SLOTS=0 and ttt_count=0), lse = -inf which the merge handles.
    lse = tl.where(l_i > 0, m_i + tl.log(l_i), -float('inf'))
    lse_base = Lse_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h
    tl.store(
        lse_base + offs_m * stride_lse_m,
        lse,
    )


def two_part_attention(
    q: torch.Tensor,                       # [B, H, K, D]
    ttt_k: Optional[torch.Tensor],         # [B, KH, T_ttt, D] (GQA non-expanded)
    ttt_v: Optional[torch.Tensor],
    curr_k: torch.Tensor,                  # [B, KH, K, D] (GQA non-expanded)
    curr_v: torch.Tensor,
    ttt_count: int,
    ttt_mask: Optional[torch.Tensor],      # [B, ttt_count] bool
    k_slots: int,
    n_kv_groups: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ttt + curr attention.  Returns (out [B, H, K, D], lse [B, H, K]).

    Skips the cross region entirely; cross's contribution is merged
    later via flashinfer's paged output + lse.
    """
    B, H, M, D = q.shape
    assert M == k_slots
    assert D == 128

    has_ttt = (ttt_count > 0) and (ttt_mask is not None) and (ttt_k is not None)

    out = torch.empty_like(q)
    lse = torch.empty(B, H, k_slots, dtype=torch.float32, device=q.device)

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

    BLOCK_N = 64
    grid = (B, H)
    _two_part_attn_fwd_kernel[grid](
        q,
        ttt_k_arg, ttt_v_arg,
        curr_k, curr_v,
        ttt_mask_bool,
        out, lse,
        ttt_count, n_kv_groups,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        tk_strides[0], tk_strides[1], tk_strides[2], tk_strides[3],
        tv_strides[0], tv_strides[1], tv_strides[2], tv_strides[3],
        curr_k.stride(0), curr_k.stride(1), curr_k.stride(2), curr_k.stride(3),
        curr_v.stride(0), curr_v.stride(1), curr_v.stride(2), curr_v.stride(3),
        stride_mb, stride_mc,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        scale=scale,
        K_SLOTS=k_slots,
        HAS_TTT=has_ttt,
        BLOCK_N=BLOCK_N,
        D=D,
    )
    return out, lse


def merge_attn_outputs(
    out_a: torch.Tensor,    # [..., n_heads, K, D]
    lse_a: torch.Tensor,    # [..., n_heads, K]
    out_b: torch.Tensor,    # [..., n_heads, K, D]
    lse_b: torch.Tensor,    # [..., n_heads, K]
) -> torch.Tensor:
    """Online-softmax merge of two attention partials.

    Given two non-overlapping KV partitions A and B with their own
    softmax outputs (out_a, out_b) and log-sum-exp values (lse_a,
    lse_b), the merged attention output is::

        m = max(lse_a, lse_b)
        w_a = exp(lse_a - m)
        w_b = exp(lse_b - m)
        out = (out_a * w_a + out_b * w_b) / (w_a + w_b)

    -inf in lse_a or lse_b correctly drops that side from the merge.
    """
    m = torch.maximum(lse_a, lse_b)
    # Replace pure -inf rows in m with 0 to avoid 0/0 (no contribution
    # from either side; out stays as the existing 0 sum).  In practice
    # at least one side is finite in production; this guard is defensive.
    finite_m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w_a = torch.exp(lse_a - finite_m)
    w_b = torch.exp(lse_b - finite_m)
    w_sum = w_a + w_b
    # When both sides are -inf, w_sum is 0; clamp to 1 for safe division.
    w_sum_safe = torch.where(w_sum > 0, w_sum, torch.ones_like(w_sum))
    w_a = w_a / w_sum_safe
    w_b = w_b / w_sum_safe
    return (out_a.to(torch.float32) * w_a.unsqueeze(-1)
            + out_b.to(torch.float32) * w_b.unsqueeze(-1)).to(out_a.dtype)


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
    BLOCK_N = 64

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
