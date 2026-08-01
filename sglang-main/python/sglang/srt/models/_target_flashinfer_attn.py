"""[EXPERIMENTAL — not winning] Flashinfer tree-attention integration for HF Llama target forward.

Empirical measurement on humaneval:3 single-GPU:
- Default SDPA: ~5 sec/sample
- Flashinfer (this module): ~65 sec/sample (13× slower)

Root cause: flashinfer's `plan()` + ragged-prefill launch overhead exceeds
the math-backend SDPA cost for our tree-verify shapes (Q ≈ 100, KV ≈ 500,
n_heads=32, head_dim=128). Flashinfer shines on large batch × long sequence,
not 100-node tree verify.

Kept for future exploration: if tree gets larger (budget 500+) or a future
flashinfer version reduces small-size overhead.

Usage:
    from benchmarks_hf.algorithms.target_flashinfer_attn import register

Replaces the default SDPA attention (which falls to math backend when a
custom additive-bias mask is present — our tree verify case) with
flashinfer's `BatchPrefillWithRaggedKVCacheWrapper` + `custom_mask`. On
humaneval:82 tree verify the default path spends ~30 ms / iter on the
target forward; flashinfer's flash-like custom-mask kernel is expected
to cut that to ~20 ms (~35% reduction).

Usage:
    from benchmarks_hf.algorithms.target_flashinfer_attn import register

    register()  # call once at process start

    model = AutoModelForCausalLM.from_pretrained(
        ..., attn_implementation="flashinfer_tree",
    )

The registered attention kernel intercepts HF's standard attention
interface call signature:
    (module, q, k, v, attention_mask, dropout, scaling, **kwargs)
        -> (attn_output, attn_weights=None)

Tensor shapes (HF standard, B=1 in our use case):
    q: [B, n_heads, seq_q, head_dim]
    k, v: [B, n_kv_heads, seq_kv, head_dim]   (seq_kv = prefix + seq_q after KV cache update)
    attention_mask: [B, 1, seq_q, seq_kv]  additive bias (0 or -inf), or None
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch


_WORKSPACE_BUF: Optional[torch.Tensor] = None
_WRAPPER = None
_WRAPPER_DEVICE = None

# Cache the last plan's key so we can skip re-plan across 32 layers of the same
# forward pass (all layers see same Q/KV shape + same tree mask).
_LAST_PLAN_KEY: Optional[tuple] = None
_CACHED_PACKED_MASK: Optional[torch.Tensor] = None
_CACHED_MASK_SRC_ID: Optional[int] = None


def _get_wrapper(device: torch.device):
    """Lazy init of the flashinfer prefill wrapper with a 128MB workspace."""
    global _WORKSPACE_BUF, _WRAPPER, _WRAPPER_DEVICE
    if _WRAPPER is None or _WRAPPER_DEVICE != device:
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
        _WORKSPACE_BUF = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        _WRAPPER = BatchPrefillWithRaggedKVCacheWrapper(
            _WORKSPACE_BUF, kv_layout="NHD",
        )
        _WRAPPER_DEVICE = device
    return _WRAPPER


def reset_plan_cache():
    """Clear the plan cache. Call between iters if mask/shape changes drastically."""
    global _LAST_PLAN_KEY
    _LAST_PLAN_KEY = None


def _additive_bias_to_bool_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """HF passes attention_mask as additive bias [B,1,Q,K] with 0 / -inf.
    Convert to bool mask [B*Q*K] where True = keep (0 bias), False = drop (-inf).
    Flashinfer `custom_mask` expects 1D flat over batch × q × k.
    """
    # `-inf` marks masked positions; use `isfinite` for robustness to both
    # `-inf` and very-large-negative sentinel values.
    keep = torch.isfinite(attention_mask)
    return keep


def _sdpa_fallback(module, q, k, v, mask, dropout, scaling, **kwargs):
    """Fall back to HF's sdpa_attention_forward when flashinfer isn't optimal."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(
        module, q, k, v, mask, dropout=dropout, scaling=scaling, **kwargs,
    )


# Heuristic: use flashinfer only for "tree verify" shape (small Q, Q < KV).
# Prefill has Q = full prompt (usually > 100), tree verify has Q ≈ 60-100.
# For prefill we stay on SDPA (HF's causal fast path).
_FLASHINFER_Q_THRESHOLD = int(os.environ.get('FLASHINFER_Q_MAX', '200'))


def flashinfer_tree_attention_forward(
    module,
    query_states: torch.Tensor,          # [B, n_heads, Q, D]
    key_states: torch.Tensor,            # [B, n_kv_heads, KV, D]
    value_states: torch.Tensor,          # [B, n_kv_heads, KV, D]
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """HF attention interface — runs flashinfer ragged-prefill with custom mask
    only when the call looks like a tree-verify (small Q + custom mask). Falls
    back to SDPA otherwise (prefill, single-token decode).
    """
    B, n_heads, Q, D = query_states.shape
    _, n_kv_heads, KV, _ = key_states.shape

    # Fall back to SDPA for prefill / decode paths where flashinfer's custom-mask
    # plan overhead outweighs its kernel speed.
    if Q > _FLASHINFER_Q_THRESHOLD or attention_mask is None or B != 1:
        return _sdpa_fallback(
            module, query_states, key_states, value_states,
            attention_mask, dropout, scaling, **kwargs,
        )

    q_flat = query_states[0].transpose(0, 1).contiguous()          # [Q, H, D]
    k_flat = key_states[0].transpose(0, 1).contiguous()            # [KV, Hkv, D]
    v_flat = value_states[0].transpose(0, 1).contiguous()          # [KV, Hkv, D]

    device = q_flat.device
    wrapper = _get_wrapper(device)

    qo_indptr = torch.tensor([0, Q], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, KV], dtype=torch.int32, device=device)

    # Custom mask: pre-pack bool -> uint8 bits (saves flashinfer's plan-time
    # packbits overhead). Cache packed mask by input data_ptr so across layers
    # we reuse the same packed buffer.
    global _LAST_PLAN_KEY, _CACHED_PACKED_MASK, _CACHED_MASK_SRC_ID
    if attention_mask is not None:
        src_id = attention_mask.data_ptr()
        if _CACHED_MASK_SRC_ID != src_id:
            from flashinfer.quantization import packbits
            bool_mask = _additive_bias_to_bool_mask(attention_mask[0, 0])  # [Q, KV]
            _CACHED_PACKED_MASK = packbits(bool_mask.reshape(-1).contiguous(), bitorder="little")
            _CACHED_MASK_SRC_ID = src_id
        packed_mask = _CACHED_PACKED_MASK
    else:
        packed_mask = None

    # Plan call: expensive — cache across all layers of the same forward pass
    # since they share (Q, KV, mask) invariants. Only re-plan when key changes.
    mask_id = 0 if packed_mask is None else packed_mask.data_ptr()
    plan_key = (Q, KV, n_heads, n_kv_heads, D, mask_id, q_flat.dtype)
    if _LAST_PLAN_KEY != plan_key:
        wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim_qk=D,
            packed_custom_mask=packed_mask,
            causal=(packed_mask is None),
            q_data_type=q_flat.dtype,
            kv_data_type=k_flat.dtype,
            sm_scale=float(scaling) if scaling is not None else None,
        )
        _LAST_PLAN_KEY = plan_key

    out_flat = wrapper.run(q_flat, k_flat, v_flat)          # [Q, H, D]
    out = out_flat.transpose(0, 1).unsqueeze(0).contiguous()  # [1, H, Q, D]

    # Transpose back to HF attention output format: [B, Q, H, D] is what the
    # caller reshapes with `.reshape(*input_shape, -1)`. Looking at Llama
    # attention forward: `attn_output = attn_output.reshape(*input_shape, -1)`
    # where input_shape = (B, Q) and H*D = hidden — so the expected output
    # shape before reshape is [B, Q, H, D] or equivalently [B, H, Q, D] with
    # a transpose. HF eager/sdpa returns [B, H, Q, D] then reshapes. We match.
    return out, None


def register(name: str = "flashinfer_tree"):
    """Register the flashinfer attention under HF's ALL_ATTENTION_FUNCTIONS.

    Call this once at process start before model loading. Then load with
    ``attn_implementation=name``.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[name] = flashinfer_tree_attention_forward
