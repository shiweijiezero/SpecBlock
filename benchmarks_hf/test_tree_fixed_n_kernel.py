#!/usr/bin/env python3
"""GPU canary test for worst-case fixed-N CUDA BFS writes."""

import math
import os
import sys

import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

from algorithms.tree_build_cuda_loader import (  # noqa: E402
    cuda_build_bfs,
    tree_bfs_required_capacity,
)


def main():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA is unavailable")
        return

    device = torch.device("cuda")
    N = 90
    K = 4
    max_topk = 10
    tree_start = 41
    capacity = tree_bfs_required_capacity(tree_start, N, K, max_topk)
    guard = 32
    int_sentinel = -777777777
    float_sentinel = -12345.5

    backing = {}
    tree_buf = {}
    for name in ("tokens", "parents", "ranks", "blocks", "slots"):
        storage = torch.full(
            (capacity + guard,), int_sentinel, dtype=torch.long, device=device
        )
        backing[name] = storage
        tree_buf[name] = storage[:capacity]
    lps_storage = torch.full(
        (capacity + guard,), float_sentinel, dtype=torch.float32, device=device
    )
    backing["lps"] = lps_storage
    tree_buf["lps"] = lps_storage[:capacity]

    rank_preds = torch.full((N, K), 2, dtype=torch.long, device=device)
    greedy_target = torch.arange(N * K, dtype=torch.long, device=device).reshape(N, K)
    greedy_lps = torch.full(
        (N, K), math.log(0.1), dtype=torch.float32, device=device
    )
    top_target = torch.arange(
        N * K * max_topk, dtype=torch.long, device=device
    ).reshape(N, K, max_topk)
    top_lps = torch.zeros((N, K, max_topk), dtype=torch.float32, device=device)
    pend_lps = torch.zeros(N, dtype=torch.float32, device=device)
    pend_nodes = torch.zeros(N, dtype=torch.long, device=device)
    rank_slot_topk = torch.tensor([2, 4, 6, 4, 1], dtype=torch.long, device=device)
    total_alts = torch.empty(1, dtype=torch.long, device=device)

    actual_alts = cuda_build_bfs(
        rank_preds,
        greedy_target,
        greedy_lps,
        top_target,
        top_lps,
        pend_lps,
        pend_nodes,
        rank_slot_topk,
        tree_buf,
        total_alts,
        tree_start=tree_start,
        N=N,
        K=K,
        max_topk=max_topk,
        rank_classes=5,
        give_up_class=4,
        adaptive_all=1,
        pend_depth=1,
    )
    torch.cuda.synchronize()

    expected_alts = N * K * (6 - 1)
    assert actual_alts == expected_alts
    last_written = tree_start + N * K + expected_alts - 1
    assert last_written < capacity
    assert int(tree_buf["tokens"][last_written]) != int_sentinel
    for name, storage in backing.items():
        expected = float_sentinel if name == "lps" else int_sentinel
        assert torch.all(storage[capacity:] == expected), name

    print(
        "[PASS] fixed-N CUDA BFS worst case stayed within capacity: "
        f"alts={actual_alts}, last_written={last_written}, capacity={capacity}"
    )


if __name__ == "__main__":
    main()
