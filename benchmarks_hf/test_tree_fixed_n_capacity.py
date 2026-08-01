#!/usr/bin/env python3
"""CPU gate for fixed-N CUDA tree buffer capacity checks."""

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parent / "algorithms" / "tree_build_cuda_loader.py"
SPEC = importlib.util.spec_from_file_location("tree_build_cuda_loader_direct", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def main():
    required = MODULE.tree_bfs_required_capacity(
        tree_start=41,
        leaves=90,
        K=4,
        max_topk=10,
    )
    assert required == 3641

    source = (Path(__file__).parent / "algorithms" / "specblock.py").read_text(
        encoding="utf-8"
    )
    assert "tree_bfs_required_capacity(" in source
    assert "fixed-N CUDA tree output exceeds buffer capacity before finalize" in source
    cuda_source = (
        Path(__file__).parent / "algorithms" / "tree_build_cuda.cu"
    ).read_text(encoding="utf-8")
    assert "c10::cuda::CUDAGuard" in cuda_source
    assert "pos < tree_capacity" in cuda_source
    assert "tree_tokens.numel()" in cuda_source

    capacity = 1781
    tree_buf = {
        name: torch.empty(capacity, dtype=(
            torch.float32 if name == "lps" else torch.long
        ))
        for name in MODULE._TREE_FIELDS
    }
    empty_long = torch.empty(0, dtype=torch.long)
    empty_float = torch.empty(0, dtype=torch.float32)
    total_alts = torch.empty(1, dtype=torch.long)
    try:
        MODULE.cuda_build_bfs(
            empty_long,
            empty_long,
            empty_float,
            empty_long,
            empty_float,
            empty_float,
            empty_long,
            empty_long,
            tree_buf,
            total_alts,
            tree_start=41,
            N=90,
            K=4,
            max_topk=10,
            rank_classes=5,
            give_up_class=4,
            adaptive_all=1,
            pend_depth=1,
        )
    except ValueError as error:
        assert "capacity=1781" in str(error)
        assert "required=3641" in str(error)
    else:
        raise AssertionError("unsafe fixed-N BFS capacity was not rejected")

    print("[PASS] fixed-N CUDA tree capacity is sized and guarded")


if __name__ == "__main__":
    main()
