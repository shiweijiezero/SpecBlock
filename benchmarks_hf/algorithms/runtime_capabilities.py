"""Runtime-specific execution policies for SpecBlock inference."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeCapabilities:
    minimum_triton_query_tile: int = 1
    attention_block_n: int = 64
    three_part_attention_block_n: int = 64
    three_part_attention_num_warps: int | None = None
    three_part_attention_num_stages: int | None = None
    draft_compile_default: str = "2"
    draft_compile_forward_default: str = "1"
    draft_compile_update_default: str = "1"
    draft_compile_ragged_default: str = "1"
    internal_prewarm_default: str = "1"
    needs_ragged_condition_fallback: bool = False
    use_shared_target_kv_storage: bool = False
    use_native_tree_builder: bool = False
    fuse_target_mlp_residual: bool = False
    enable_target_projection_fusion: bool = False
    target_gate_up_fuse_min_rows: int = 129

    def three_part_attention_launch_kwargs(self) -> dict[str, int]:
        kwargs = {}
        if self.three_part_attention_num_warps is not None:
            kwargs["num_warps"] = self.three_part_attention_num_warps
        if self.three_part_attention_num_stages is not None:
            kwargs["num_stages"] = self.three_part_attention_num_stages
        return kwargs


def _detect_runtime_capabilities() -> RuntimeCapabilities:
    platform = os.environ.get("SPECBLOCK_PLATFORM", "auto").lower()
    if platform not in {"auto", "cuda", "metax"}:
        raise ValueError(
            "SPECBLOCK_PLATFORM must be one of: auto, cuda, metax"
        )
    is_metax = platform == "metax" or (
        platform == "auto" and "metax" in torch.__version__.lower()
    )
    if is_metax:
        return RuntimeCapabilities(
            minimum_triton_query_tile=16,
            attention_block_n=32,
            three_part_attention_block_n=32,
            three_part_attention_num_warps=1,
            three_part_attention_num_stages=1,
            draft_compile_default="2",
            draft_compile_forward_default="0",
            draft_compile_update_default="1",
            draft_compile_ragged_default="0",
            internal_prewarm_default="1",
            needs_ragged_condition_fallback=True,
            use_shared_target_kv_storage=True,
            use_native_tree_builder=True,
            fuse_target_mlp_residual=True,
            enable_target_projection_fusion=True,
            target_gate_up_fuse_min_rows=80,
        )
    return RuntimeCapabilities()


RUNTIME_CAPABILITIES = _detect_runtime_capabilities()
