"""Ragged draft-cache writes and attention for request batching."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _append_ragged_kv_kernel(
    cache_k_ptr,
    cache_v_ptr,
    current_k_ptr,
    current_v_ptr,
    lengths_ptr,
    valid_slots_ptr,
    stride_ck_b,
    stride_ck_h,
    stride_ck_s,
    stride_ck_d,
    stride_cv_b,
    stride_cv_h,
    stride_cv_s,
    stride_cv_d,
    stride_k_b,
    stride_k_h,
    stride_k_s,
    stride_k_d,
    stride_v_b,
    stride_v_h,
    stride_v_s,
    stride_v_d,
    sequence_capacity,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    slot = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    offsets = tl.arange(0, BLOCK_D)
    length = tl.load(lengths_ptr + batch)
    valid_slots = tl.load(valid_slots_ptr + batch)
    destination = length + slot
    mask = (
        (slot < valid_slots)
        & (destination >= 0)
        & (destination < sequence_capacity)
        & (offsets < HEAD_DIM)
    )
    current_k = tl.load(
        current_k_ptr
        + batch * stride_k_b
        + head * stride_k_h
        + slot * stride_k_s
        + offsets * stride_k_d,
        mask=mask,
        other=0.0,
    )
    current_v = tl.load(
        current_v_ptr
        + batch * stride_v_b
        + head * stride_v_h
        + slot * stride_v_s
        + offsets * stride_v_d,
        mask=mask,
        other=0.0,
    )
    tl.store(
        cache_k_ptr
        + batch * stride_ck_b
        + head * stride_ck_h
        + destination * stride_ck_s
        + offsets * stride_ck_d,
        current_k,
        mask=mask,
    )
    tl.store(
        cache_v_ptr
        + batch * stride_cv_b
        + head * stride_cv_h
        + destination * stride_cv_s
        + offsets * stride_cv_d,
        current_v,
        mask=mask,
    )


def assert_ragged_kv_metadata(
    lengths: torch.Tensor,
    valid_slots: torch.Tensor,
    current_width: int,
    sequence_capacity: int,
) -> None:
    """Fail-stop once before all draft layers consume shared ragged metadata."""
    metadata_valid = (
        (lengths >= 0)
        & (valid_slots >= 0)
        & (valid_slots <= int(current_width))
        & (lengths + valid_slots <= int(sequence_capacity))
    )
    torch._assert_async(
        torch.all(metadata_valid),
        "ragged draft KV metadata is outside current/cache bounds",
    )


def append_ragged_kv_(
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    current_keys: torch.Tensor,
    current_values: torch.Tensor,
    lengths: torch.Tensor,
    valid_slots: torch.Tensor,
) -> None:
    """Append each row's valid current KV at its own logical tail."""
    if cache_keys.shape != cache_values.shape:
        raise ValueError("draft cache key/value shapes must match")
    if current_keys.shape != current_values.shape:
        raise ValueError("current draft key/value shapes must match")
    if cache_keys.ndim != 4 or current_keys.ndim != 4:
        raise ValueError("draft KV tensors must be rank 4")
    if cache_keys.shape[:2] != current_keys.shape[:2]:
        raise ValueError("draft cache and current KV batch/head shapes must match")
    if cache_keys.shape[3] != current_keys.shape[3]:
        raise ValueError("draft cache and current KV head dimensions must match")
    if lengths.ndim != 1 or valid_slots.ndim != 1:
        raise ValueError("ragged draft metadata must be one-dimensional")
    if lengths.shape != valid_slots.shape or lengths.numel() != cache_keys.shape[0]:
        raise ValueError("ragged draft metadata must have one entry per batch row")
    if lengths.dtype != torch.long or valid_slots.dtype != torch.long:
        raise TypeError("ragged draft metadata must use torch.long")
    if not lengths.is_contiguous() or not valid_slots.is_contiguous():
        raise ValueError("ragged draft metadata must be contiguous")
    tensors = (
        cache_keys,
        cache_values,
        current_keys,
        current_values,
        lengths,
        valid_slots,
    )
    if cache_keys.device.type != "cuda" or any(
        tensor.device != cache_keys.device for tensor in tensors
    ):
        raise ValueError("ragged draft KV tensors must share one CUDA device")
    if cache_keys.dtype != cache_values.dtype:
        raise TypeError("draft cache key/value dtypes must match")
    if current_keys.dtype != current_values.dtype or current_keys.dtype != cache_keys.dtype:
        raise TypeError("current and cached draft KV dtypes must match")

    head_dim = int(cache_keys.shape[3])
    block_d = triton.next_power_of_2(head_dim)
    _append_ragged_kv_kernel[
        (current_keys.shape[2], cache_keys.shape[0], cache_keys.shape[1])
    ](
        cache_keys,
        cache_values,
        current_keys,
        current_values,
        lengths,
        valid_slots,
        cache_keys.stride(0),
        cache_keys.stride(1),
        cache_keys.stride(2),
        cache_keys.stride(3),
        cache_values.stride(0),
        cache_values.stride(1),
        cache_values.stride(2),
        cache_values.stride(3),
        current_keys.stride(0),
        current_keys.stride(1),
        current_keys.stride(2),
        current_keys.stride(3),
        current_values.stride(0),
        current_values.stride(1),
        current_values.stride(2),
        current_values.stride(3),
        cache_keys.shape[2],
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
    )
