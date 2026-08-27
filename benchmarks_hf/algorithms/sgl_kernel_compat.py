"""Load optional SGLang-compatible fused operators."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SGLKernelOps:
    rmsnorm: Callable
    silu_and_mul: Callable
    apply_rope: Callable
    fused_add_rmsnorm: Callable


def _load_native_ops() -> SGLKernelOps | None:
    try:
        from sgl_kernel import apply_rope_with_cos_sin_cache_inplace
        from sgl_kernel import fused_add_rmsnorm
        from sgl_kernel import rmsnorm
        from sgl_kernel import silu_and_mul
    except (ImportError, OSError, RuntimeError):
        return None
    return SGLKernelOps(
        rmsnorm=rmsnorm,
        silu_and_mul=silu_and_mul,
        apply_rope=apply_rope_with_cos_sin_cache_inplace,
        fused_add_rmsnorm=fused_add_rmsnorm,
    )


def _compat_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.rms_norm(x, (weight.shape[0],), weight, eps)


def _compat_silu_and_mul(input: torch.Tensor, out: torch.Tensor) -> None:
    torch.ops.sgl_kernel.silu_and_mul(out, input)


def _compat_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    torch.ops.sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps, False)


def _compat_apply_rope(
    *,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    query_heads = query.view(query.shape[0], -1, head_size)
    key_heads = key.view(key.shape[0], -1, head_size)
    torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache(
        query_heads,
        key_heads,
        query_heads,
        key_heads,
        cos_sin_cache,
        positions,
        not is_neox,
        False,
        None,
        None,
        None,
        None,
    )


def load_sgl_kernel_ops() -> SGLKernelOps | None:
    library_path = os.environ.get("SPECBLOCK_SGL_KERNEL_LIBRARY")
    if library_path:
        try:
            torch.ops.load_library(library_path)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"failed to load SPECBLOCK_SGL_KERNEL_LIBRARY={library_path!r}"
            ) from error
        return SGLKernelOps(
            rmsnorm=_compat_rmsnorm,
            silu_and_mul=_compat_silu_and_mul,
            apply_rope=_compat_apply_rope,
            fused_add_rmsnorm=_compat_fused_add_rmsnorm,
        )

    return _load_native_ops()
