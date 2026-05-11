"""Custom Triton tree attention kernel for HF Llama target forward.

Motivation
----------
Our tree-verify pass calls target attention with a custom additive-bias mask
[B, 1, Q, KV] (0 or -inf). HF SDPA detects the additive mask and silently
falls back to the `math` backend — which is ~2× slower than flash on
A100 for Q~100, KV~500. flashinfer's plan() overhead makes it even worse
at this small size (see target_flashinfer_attn.py, 13× slower).

This kernel is a fused flash-attention-2 style online-softmax Triton kernel
that:
  - Handles GQA natively (n_heads=32, n_kv_heads=8) — no K/V repeat, no
    4× memory bloat, each Q head maps to its own KV head inside the kernel.
  - Accepts additive bias mask directly — no need for bool conversion or
    bit-packing. Zero plan-time overhead, zero caching machinery.
  - Falls back to SDPA for prefill (large Q) where standard backend wins.

Target shapes (Llama 3.1-8B tree verify, B=1):
    Q: [1, 32, ~60-100, 128]
    K, V: [1, 8, ~400-600, 128]    (not pre-repeated)
    mask: [1, 1, Q, KV] additive bias (bf16, 0 or -inf)

Usage:
    from benchmarks_hf.algorithms.target_triton_attn import register
    register()  # once at process start
    model = AutoModelForCausalLM.from_pretrained(
        ..., attn_implementation="triton_tree",
    )
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


_DUMMY_BIAS: Optional[torch.Tensor] = None


@triton.jit
def _target_tree_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Bias_ptr, Out_ptr,
    Q_LEN, KV_LEN,                         # int32 runtime
    stride_q_b, stride_q_h, stride_q_q, stride_q_d,
    stride_k_b, stride_k_h, stride_k_c, stride_k_d,
    stride_v_b, stride_v_h, stride_v_c, stride_v_d,
    stride_m_b, stride_m_q, stride_m_k,    # bias stride (treated as [B, Q, KV])
    stride_o_b, stride_o_h, stride_o_q, stride_o_d,
    scale,                                 # float32 runtime (supports dynamic scaling)
    GROUP_SIZE: tl.constexpr,              # H_q / H_kv
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """Flash-attention-2 forward, GQA-aware, additive-bias mask, no causal assumption."""
    pid_m = tl.program_id(0)   # tile along Q
    pid_b = tl.program_id(1)   # batch
    pid_h = tl.program_id(2)   # Q head

    pid_hkv = pid_h // GROUP_SIZE  # map Q head -> KV head

    offs_q = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, D)                           # [D]

    # --- Load Q tile ---
    q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q_ptrs = q_base + offs_q[:, None] * stride_q_q + offs_d[None, :] * stride_q_d
    q_in_bounds = offs_q[:, None] < Q_LEN
    q = tl.load(q_ptrs, mask=q_in_bounds, other=0.0)   # [BLOCK_M, D]

    # --- Online softmax accumulators ---
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    k_base = K_ptr + pid_b * stride_k_b + pid_hkv * stride_k_h
    v_base = V_ptr + pid_b * stride_v_b + pid_hkv * stride_v_h
    if HAS_BIAS:
        bias_base = Bias_ptr + pid_b * stride_m_b

    for n_start in range(0, KV_LEN, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)       # [BLOCK_N]
        n_in_bounds = offs_n < KV_LEN                   # [BLOCK_N]

        # Load K, V tiles
        k_ptrs = k_base + offs_n[:, None] * stride_k_c + offs_d[None, :] * stride_k_d
        v_ptrs = v_base + offs_n[:, None] * stride_v_c + offs_d[None, :] * stride_v_d
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)   # [BLOCK_N, D]
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)   # [BLOCK_N, D]

        # QK (fp32 accumulate)
        qk = tl.dot(q, tl.trans(k)) * scale             # [BLOCK_M, BLOCK_N]

        if HAS_BIAS:
            # Load [BLOCK_M, BLOCK_N] of additive bias (bf16/fp16)
            bias_ptrs = (
                bias_base
                + offs_q[:, None] * stride_m_q
                + offs_n[None, :] * stride_m_k
            )
            bias_load_mask = q_in_bounds & n_in_bounds[None, :]
            bias = tl.load(
                bias_ptrs, mask=bias_load_mask, other=-float('inf'),
            ).to(tl.float32)
            qk = qk + bias
        else:
            # No mask: only oob guard (for ragged KV tail)
            qk = tl.where(n_in_bounds[None, :], qk, -float('inf'))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_i_new

    # Normalize; guard l_i==0 (all-masked rows → acc stays 0)
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    # Store
    o_base = Out_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    o_ptrs = o_base + offs_q[:, None] * stride_o_q + offs_d[None, :] * stride_o_d
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=q_in_bounds)


def tree_attention_target(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    additive_mask: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """GQA-aware tree attention for Llama target.

    Args:
        q: [B, Hq, Q, D]
        k: [B, Hkv, KV, D]
        v: [B, Hkv, KV, D]
        additive_mask: [B, 1, Q, KV] bf16/fp16 additive bias (0 or -inf), or None.
        scale: 1/sqrt(D) typically.

    Returns:
        out: [B, Hq, Q, D] same dtype as q.
    """
    B, Hq, Q, D = q.shape
    _, Hkv, KV, _ = k.shape
    assert D == 128, f"kernel hardcoded for D=128, got {D}"
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
    assert v.shape == k.shape

    group_size = Hq // Hkv
    # Block tuning for A100 / tree-verify shapes.
    # For small Q (<=16, single-pos decode), BLOCK_M=16 with modest parallelism.
    # For tree verify (Q~60-120), BLOCK_M=32/BLOCK_N=64 with 4 warps is a solid
    # default. Larger BLOCK_M (64) would cut scheduling overhead but our grid
    # along Q already has few tiles; BLOCK_M=32 keeps more SMs busy.
    if Q <= 16:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 16, 64, 4, 2
    elif Q <= 64:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 32, 64, 4, 2
    else:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 32, 64, 4, 3

    out = torch.empty_like(q)

    has_bias = additive_mask is not None
    if has_bias:
        # Squeeze [B, 1, Q, KV] -> [B, Q, KV] (view, no copy if contiguous layout allows)
        if additive_mask.dim() == 4:
            bias = additive_mask.squeeze(1)
        else:
            bias = additive_mask
        # Broadcast scalar-dim-0 (e.g., [1, Q, KV]) is fine since stride_m_b=0 reads same row
        stride_m_b = bias.stride(0) if bias.shape[0] > 1 else 0
        stride_m_q = bias.stride(-2)
        stride_m_k = bias.stride(-1)
    else:
        global _DUMMY_BIAS
        if _DUMMY_BIAS is None or _DUMMY_BIAS.device != q.device:
            _DUMMY_BIAS = torch.empty(1, device=q.device, dtype=q.dtype)
        bias = _DUMMY_BIAS
        stride_m_b = 0
        stride_m_q = 0
        stride_m_k = 0

    grid = (triton.cdiv(Q, BLOCK_M), B, Hq)
    _target_tree_attn_fwd_kernel[grid](
        q, k, v, bias, out,
        Q, KV,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        stride_m_b, stride_m_q, stride_m_k,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale=float(scale),
        GROUP_SIZE=group_size,
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


# Heuristic: use triton kernel only for tree-verify (small Q). For prefill (large Q)
# HF's SDPA causal fast path beats anything we can JIT in triton.
_TRITON_Q_THRESHOLD = int(os.environ.get('TRITON_TREE_Q_MAX', '200'))


def _sdpa_fallback(module, q, k, v, mask, dropout, scaling, **kwargs):
    """Fall back to HF's sdpa_attention_forward."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(
        module, q, k, v, mask, dropout=dropout, scaling=scaling, **kwargs,
    )


def triton_tree_attention_forward(
    module,
    query_states: torch.Tensor,          # [B, Hq, Q, D]
    key_states: torch.Tensor,            # [B, Hkv, KV, D]
    value_states: torch.Tensor,          # [B, Hkv, KV, D]
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """HF ALL_ATTENTION_FUNCTIONS entry. Matches sdpa_attention_forward signature."""
    B, Hq, Q, D = query_states.shape

    # Prefill / decode paths: SDPA wins.
    if Q > _TRITON_Q_THRESHOLD or B != 1:
        return _sdpa_fallback(
            module, query_states, key_states, value_states,
            attention_mask, dropout, scaling, **kwargs,
        )

    if scaling is None:
        scaling = 1.0 / (float(D) ** 0.5)

    # attention_mask may be None (full-visible decode) or additive bias tensor.
    # tree_attention_target returns [B, Hq, Q, D]; HF sdpa_attention_forward
    # does a final `attn_output.transpose(1, 2).contiguous()` before returning
    # (so caller's `attn_output.reshape(B, Q, -1)` flattens head * head_dim).
    # Match that layout here — otherwise reshape interprets Hq as Q and outputs
    # garbage logits (observed: acc_len drops from ~5 to ~1).
    out = tree_attention_target(
        query_states, key_states, value_states, attention_mask, float(scaling),
    )
    return out.transpose(1, 2).contiguous(), None


def register(name: str = "triton_tree"):
    """Register the triton attention under HF's ALL_ATTENTION_FUNCTIONS.

    Call once at process start, then load model with ``attn_implementation=name``.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[name] = triton_tree_attention_forward
