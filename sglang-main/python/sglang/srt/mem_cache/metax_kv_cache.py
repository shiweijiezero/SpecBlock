from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    stride_k_row,
    stride_v_row,
    stride_k_cache_row,
    stride_v_cache_row,
    row_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < row_dim
    cache_row = tl.load(indices_ptr + row).to(tl.int64)
    k = tl.load(k_ptr + row * stride_k_row + offsets, mask=valid)
    v = tl.load(v_ptr + row * stride_v_row + offsets, mask=valid)
    tl.store(
        k_cache_ptr + cache_row * stride_k_cache_row + offsets,
        k,
        mask=valid,
    )
    tl.store(
        v_cache_ptr + cache_row * stride_v_cache_row + offsets,
        v,
        mask=valid,
    )


def store_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,
) -> None:
    """Store K and V rows together without materializing split QKV views."""
    rows = k.numel() // row_dim
    block = 256
    _store_kv_kernel[(rows, triton.cdiv(row_dim, block))](
        k,
        v,
        k_cache,
        v_cache,
        indices,
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        v_cache.stride(0),
        row_dim=row_dim,
        BLOCK=block,
        num_warps=1,
        num_stages=1,
    )
