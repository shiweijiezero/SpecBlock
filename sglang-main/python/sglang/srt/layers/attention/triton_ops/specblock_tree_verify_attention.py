"""Paged request-grouped attention for SpecBlock target verification."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _specblock_tree_verify_fwd_kernel(
    Q_ptr,
    CURR_K_ptr,
    CURR_V_ptr,
    POOL_K_ptr,
    POOL_V_ptr,
    KV_INDPTR_ptr,
    KV_INDICES_ptr,
    CUSTOM_MASK_ptr,
    MASK_INDPTR_ptr,
    Out_ptr,
    n_kv_groups,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_ck_t,
    stride_ck_h,
    stride_ck_d,
    stride_cv_t,
    stride_cv_h,
    stride_cv_d,
    stride_pk_s,
    stride_pk_h,
    stride_pk_d,
    stride_pv_s,
    stride_pv_h,
    stride_pv_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    scale: tl.constexpr,
    TREE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Attend to an implicit all-visible prefix plus the explicit tree mask."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_kh = pid_h // n_kv_groups

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_valid = offs_m < TREE_N
    q_token = pid_b * TREE_N + offs_m
    q = tl.load(
        Q_ptr
        + q_token[:, None] * stride_q_t
        + pid_h * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=q_valid[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    prefix_begin = tl.load(KV_INDPTR_ptr + pid_b).to(tl.int64)
    prefix_end = tl.load(KV_INDPTR_ptr + pid_b + 1).to(tl.int64)
    prefix_len = prefix_end - prefix_begin

    # Persistent prefix: every tree query sees the complete request prefix.
    for n_start in range(0, prefix_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_valid = offs_n < prefix_len
        slots = tl.load(
            KV_INDICES_ptr + prefix_begin + offs_n,
            mask=n_valid,
            other=0,
        ).to(tl.int64)
        k = tl.load(
            POOL_K_ptr
            + slots[:, None] * stride_pk_s
            + pid_kh * stride_pk_h
            + offs_d[None, :] * stride_pk_d,
            mask=n_valid[:, None],
            other=0.0,
        )
        v = tl.load(
            POOL_V_ptr
            + slots[:, None] * stride_pv_s
            + pid_kh * stride_pv_h
            + offs_d[None, :] * stride_pv_d,
            mask=n_valid[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(q_valid[:, None] & n_valid[None, :], qk, -float("inf"))
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Current verifier tree: visibility is the root-to-node ancestor mask.
    mask_begin = tl.load(MASK_INDPTR_ptr + pid_b).to(tl.int64)
    curr_base = pid_b * TREE_N
    for n_start in range(0, TREE_N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_valid = offs_n < TREE_N
        curr_token = curr_base + offs_n
        k = tl.load(
            CURR_K_ptr
            + curr_token[:, None] * stride_ck_t
            + pid_kh * stride_ck_h
            + offs_d[None, :] * stride_ck_d,
            mask=n_valid[:, None],
            other=0.0,
        )
        v = tl.load(
            CURR_V_ptr
            + curr_token[:, None] * stride_cv_t
            + pid_kh * stride_cv_h
            + offs_d[None, :] * stride_cv_d,
            mask=n_valid[:, None],
            other=0.0,
        )
        visible = tl.load(
            CUSTOM_MASK_ptr
            + mask_begin
            + offs_m[:, None] * TREE_N
            + offs_n[None, :],
            mask=q_valid[:, None] & n_valid[None, :],
            other=0,
        ).to(tl.int1)
        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(visible, qk, -float("inf"))
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        Out_ptr
        + q_token[:, None] * stride_o_t
        + pid_h * stride_o_h
        + offs_d[None, :] * stride_o_d,
        out.to(Out_ptr.dtype.element_ty),
        mask=q_valid[:, None],
    )


def specblock_tree_verify_attention_fwd(
    q: torch.Tensor,
    curr_k: torch.Tensor,
    curr_v: torch.Tensor,
    out: torch.Tensor,
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    custom_mask: torch.Tensor,
    mask_indptr: torch.Tensor,
    tree_tokens: int,
    scale: float,
) -> None:
    """Run fixed-width SpecBlock target attention without materializing prefix masks."""
    if q.ndim != 3 or curr_k.ndim != 3 or curr_v.ndim != 3:
        raise RuntimeError("SpecBlock tree verifier expects flattened [T, H, D] tensors.")
    total_tokens, num_q_heads, head_dim = q.shape
    if total_tokens % tree_tokens != 0:
        raise RuntimeError(
            f"Verifier token count {total_tokens} is not divisible by tree width {tree_tokens}."
        )
    batch_size = total_tokens // tree_tokens
    num_kv_heads = curr_k.shape[1]
    if head_dim != 128 or curr_k.shape[-1] != head_dim or curr_v.shape[-1] != head_dim:
        raise RuntimeError("SpecBlock target verifier currently requires head_dim=128.")
    if num_q_heads % num_kv_heads != 0:
        raise RuntimeError("SpecBlock target verifier requires integral GQA groups.")
    if kv_indptr.numel() < batch_size + 1 or mask_indptr.numel() < batch_size + 1:
        raise RuntimeError("SpecBlock target verifier metadata is smaller than the active batch.")

    block_m = 32
    block_n = 128
    grid = (triton.cdiv(tree_tokens, block_m), batch_size, num_q_heads)
    _specblock_tree_verify_fwd_kernel[grid](
        q,
        curr_k,
        curr_v,
        pool_k,
        pool_v,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        out,
        num_q_heads // num_kv_heads,
        q.stride(0), q.stride(1), q.stride(2),
        curr_k.stride(0), curr_k.stride(1), curr_k.stride(2),
        curr_v.stride(0), curr_v.stride(1), curr_v.stride(2),
        pool_k.stride(0), pool_k.stride(1), pool_k.stride(2),
        pool_v.stride(0), pool_v.stride(1), pool_v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        scale=float(scale),
        TREE_N=tree_tokens,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D=head_dim,
        num_warps=4,
        num_stages=2,
    )
