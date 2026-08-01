#!/usr/bin/env python3
"""CPU smoke gate for the independent HF batch-capacity planner."""

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "capacity.py"
SPEC = importlib.util.spec_from_file_location("hf_capacity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def main():
    # Llama-3.1-8B BF16: 32 layers × K/V × 8 GQA heads × 128 dimensions.
    class LlamaConfig:
        num_hidden_layers = 32
        num_key_value_heads = 8
        num_attention_heads = 32
        hidden_size = 4096
        head_dim = 128

    geometry = MODULE.KVGeometry.from_config(LlamaConfig())
    assert geometry.bytes_per_batch_token == 131072

    # A B32 mixed-length wave exposes right-padding, yet never imports a model or
    # touches generation/sampling code.  The six formal datasets use the same gate.
    six_dataset_waves = {
        "humaneval": [256] * 32,
        "math500": [512] * 31 + [2048],
        "alpaca": [384] * 32,
        "nq_open": [768] * 30 + [1024, 3072],
        "mtbench": [1024] * 32,
        "wmt23": [640] * 32,
    }
    for dataset, lengths in six_dataset_waves.items():
        report = MODULE.report_wave(
            geometry, lengths, max_new_tokens=512,
            eagle_tree_width=61, specblock_tree_width=91,
        )
        assert report["batch_size"] == 32, dataset
        assert report["max_prompt_tokens"] == max(lengths), dataset
        assert report["eagle3_current_peak_gib"] > report["eagle3_after_release_prefill_gib"], dataset
        assert report["specblock_current_peak_gib"] > 0.0, dataset
        gate = MODULE.capacity_gate(report, available_gib=80.0, reserve_gib=2.0)
        assert gate["pass"], dataset

    # 32 × 1024 target tokens is exactly 4 GiB of Llama-3.1-8B KV payload.
    assert MODULE._estimate("check", geometry, 32, 1024).gib == 4.0
    print("[PASS] independent B32 KV/workspace capacity smoke")


if __name__ == "__main__":
    main()
