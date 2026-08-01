"""使用 Triton 批量提交 accepted target KV。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_selected_kv_kernel(
    key_ptr,
    value_ptr,
    metadata_ptr,
    source_ptr,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    batch_size,
    sequence_length,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    copy_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row = tl.load(metadata_ptr + copy_idx * 2)
    destination = tl.load(metadata_ptr + copy_idx * 2 + 1)
    source = tl.load(source_ptr + copy_idx)
    offsets = tl.arange(0, BLOCK_D)
    valid_copy = (
        (row >= 0)
        & (row < batch_size)
        & (source >= 0)
        & (source < sequence_length)
        & (destination >= 0)
        & (destination < sequence_length)
    )
    mask = valid_copy & (offsets < HEAD_DIM)

    key_source = (
        key_ptr
        + row * stride_kb
        + head_idx * stride_kh
        + source * stride_ks
        + offsets * stride_kd
    )
    key_destination = (
        key_ptr
        + row * stride_kb
        + head_idx * stride_kh
        + destination * stride_ks
        + offsets * stride_kd
    )
    value_source = (
        value_ptr
        + row * stride_vb
        + head_idx * stride_vh
        + source * stride_vs
        + offsets * stride_vd
    )
    value_destination = (
        value_ptr
        + row * stride_vb
        + head_idx * stride_vh
        + destination * stride_vs
        + offsets * stride_vd
    )

    keys = tl.load(key_source, mask=mask)
    values = tl.load(value_source, mask=mask)
    tl.store(key_destination, keys, mask=mask)
    tl.store(value_destination, values, mask=mask)


def copy_selected_kv_(
    keys: torch.Tensor,
    values: torch.Tensor,
    metadata: torch.Tensor,
    sources: torch.Tensor,
) -> None:
    """把多个 batch row 的 staging KV 直接复制到各自 logical tail。"""
    if keys.dim() != 4 or values.shape != keys.shape:
        raise ValueError("keys and values must share [B,H,S,D] shape")
    if metadata.dim() != 2 or metadata.shape[1] != 2:
        raise ValueError("metadata must have [copy_count,2] shape")
    if sources.dim() != 1 or sources.shape[0] != metadata.shape[0]:
        raise ValueError("sources must match metadata copy count")
    if not keys.is_cuda or values.device != keys.device:
        raise ValueError("keys and values must share one CUDA device")
    if metadata.device != keys.device or sources.device != keys.device:
        raise ValueError("copy metadata must be on the KV device")
    if metadata.dtype != torch.long or sources.dtype != torch.long:
        raise TypeError("copy metadata must use torch.long")
    if not metadata.is_contiguous() or not sources.is_contiguous():
        raise ValueError("copy metadata must be contiguous")

    copy_count = int(sources.numel())
    if copy_count == 0:
        return
    head_dim = int(keys.shape[-1])
    block_d = triton.next_power_of_2(head_dim)
    _copy_selected_kv_kernel[(copy_count, keys.shape[1])](
        keys,
        values,
        metadata,
        sources,
        *keys.stride(),
        *values.stride(),
        keys.shape[0],
        keys.shape[2],
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
