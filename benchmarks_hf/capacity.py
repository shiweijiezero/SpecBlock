#!/usr/bin/env python3
"""Read-only capacity planning for HuggingFace batch decoding.

This module deliberately does not import or mutate an inference algorithm.  It
profiles the *same* chat-template prompts that the HF algorithms consume and
computes conservative KV/workspace payload estimates before an evaluation is
launched.  It is therefore safe to run as a preflight gate for B32.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


GiB = 1024 ** 3
MiB = 1024 ** 2


@dataclass(frozen=True)
class KVGeometry:
    """Per-token target KV geometry; bytes include K and V over all layers."""

    layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int

    @property
    def bytes_per_batch_token(self) -> int:
        return (
            self.layers * 2 * self.kv_heads * self.head_dim
            * self.bytes_per_element
        )

    @classmethod
    def from_config(cls, config, dtype: str = "bf16") -> "KVGeometry":
        dtype_bytes = {"fp16": 2, "bf16": 2, "fp32": 4}
        if dtype not in dtype_bytes:
            raise ValueError(f"dtype must be one of {sorted(dtype_bytes)}, got {dtype}")
        hidden_size = int(config.hidden_size)
        heads = int(config.num_attention_heads)
        layers = getattr(config, "num_hidden_layers", None)
        if layers is None:
            layers = getattr(config, "num_layers", None)
        if layers is None:
            raise ValueError("model config has neither num_hidden_layers nor num_layers")
        return cls(
            layers=int(layers),
            kv_heads=int(getattr(config, "num_key_value_heads", heads)),
            head_dim=int(getattr(config, "head_dim", hidden_size // heads)),
            bytes_per_element=dtype_bytes[dtype],
        )


@dataclass(frozen=True)
class MemoryEstimate:
    name: str
    bytes: int

    @property
    def gib(self) -> float:
        return self.bytes / GiB


def _estimate(name: str, geometry: KVGeometry, rows: int, width: int) -> MemoryEstimate:
    return MemoryEstimate(name, geometry.bytes_per_batch_token * int(rows) * int(width))


def _tensor_bytes(shape: Sequence[int], bytes_per_element: int) -> int:
    result = bytes_per_element
    for value in shape:
        result *= int(value)
    return result


def eagle3_peak_estimates(
    geometry: KVGeometry,
    prompt_lengths: Sequence[int],
    tree_width: int = 61,
) -> list[MemoryEstimate]:
    """Peak payloads in the current EAGLE3 batch implementation.

    ``eagle3_batch.py`` retains the padded prefill allocation after splitting it
    into per-request contiguous cache copies.  Verification adds another padded
    target allocation.  The three entries can coexist, hence this intentionally
    reports their sum instead of treating them as mutually exclusive.
    """
    if not prompt_lengths or min(prompt_lengths) <= 0:
        raise ValueError("prompt_lengths must be non-empty positive integers")
    batch = len(prompt_lengths)
    width = max(prompt_lengths)
    # Prefix grows during generation; at the first decode it is the prompt.
    return [
        _estimate("eagle3_prefill_padded_kv", geometry, batch, width),
        _estimate("eagle3_split_live_kv", geometry, 1, sum(prompt_lengths)),
        _estimate("eagle3_verify_padded_kv", geometry, batch, width + tree_width),
        MemoryEstimate(
            "eagle3_tree_workspace",
            # tree IDs + position IDs + dense bool tree mask; small versus KV.
            _tensor_bytes((batch, tree_width), 8) * 2
            + _tensor_bytes((batch, 1, tree_width, tree_width), 1),
        ),
    ]


def specblock_peak_estimates(
    geometry: KVGeometry,
    prompt_lengths: Sequence[int],
    max_new_tokens: int,
    max_tree_width: int = 91,
    mask_bytes_per_element: int = 2,
) -> list[MemoryEstimate]:
    """Peak target-side payloads in the current dense SpecBlock batch cache."""
    if not prompt_lengths or min(prompt_lengths) <= 0:
        raise ValueError("prompt_lengths must be non-empty positive integers")
    if max_new_tokens < 0 or max_tree_width <= 0:
        raise ValueError("max_new_tokens must be non-negative and tree width positive")
    batch = len(prompt_lengths)
    prompt_width = max(prompt_lengths)
    # _DenseTargetCache: prompt + completion budget + two tree-width regions.
    cache_width = prompt_width + int(max_new_tokens) + 2 * int(max_tree_width)
    return [
        _estimate("specblock_dense_target_kv", geometry, batch, cache_width),
        MemoryEstimate(
            "specblock_verify_additive_mask",
            _tensor_bytes((batch, 1, max_tree_width, prompt_width + max_tree_width), mask_bytes_per_element),
        ),
        MemoryEstimate(
            "specblock_tree_ids_positions",
            _tensor_bytes((batch, max_tree_width), 8) * 2,
        ),
    ]


def total_gib(estimates: Iterable[MemoryEstimate]) -> float:
    return sum(item.bytes for item in estimates) / GiB


def report_wave(
    geometry: KVGeometry,
    prompt_lengths: Sequence[int],
    max_new_tokens: int,
    eagle_tree_width: int,
    specblock_tree_width: int,
) -> Mapping[str, object]:
    eagle = eagle3_peak_estimates(geometry, prompt_lengths, eagle_tree_width)
    specblock = specblock_peak_estimates(
        geometry, prompt_lengths, max_new_tokens, specblock_tree_width
    )
    return {
        "batch_size": len(prompt_lengths),
        "max_prompt_tokens": max(prompt_lengths),
        "mean_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
        "prompt_padding_tokens": len(prompt_lengths) * max(prompt_lengths) - sum(prompt_lengths),
        "kv_bytes_per_batch_token": geometry.bytes_per_batch_token,
        "eagle3_current_peak_gib": total_gib(eagle),
        "eagle3_after_release_prefill_gib": total_gib(eagle[1:]),
        "specblock_current_peak_gib": total_gib(specblock),
        "eagle3_components": [asdict(item) | {"gib": item.gib} for item in eagle],
        "specblock_components": [asdict(item) | {"gib": item.gib} for item in specblock],
    }


def capacity_gate(report: Mapping[str, object], available_gib: float, reserve_gib: float = 2.0) -> Mapping[str, object]:
    """Gate only the explicitly estimated tensors, never claim total model fit."""
    usable = float(available_gib) - float(reserve_gib)
    if usable <= 0:
        raise ValueError("available_gib must exceed reserve_gib")
    largest = max(float(report["eagle3_current_peak_gib"]), float(report["specblock_current_peak_gib"]))
    return {
        "estimated_tensor_peak_gib": largest,
        "available_after_reserve_gib": usable,
        "pass": largest <= usable,
        "note": "KV/workspace only; weights, activations, allocator fragmentation, and draft KV are excluded.",
    }


def _prompt_lengths(
    model_path: str, dataset: str, count: int | None, max_new_tokens: int
) -> list[int]:
    # Kept in the CLI path so importing this module remains CPU-only and has no
    # transformers/datasets requirement for unit tests and launch preflight code.
    from transformers import AutoTokenizer
    from benchmark_datasets import load_benchmark_dataset
    from utils.prompt import detect_model_family, prepare_conversation

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    family = detect_model_family(model_path)
    samples = load_benchmark_dataset(dataset, num_samples=count)
    lengths = []
    for sample in samples:
        conversation = sample.get("conversation")
        if conversation is not None:
            conversations = [list(conversation)]
        else:
            # MT-Bench-style later turns include an unknown generated answer. Its
            # exact tokenization is unknowable before decoding; reserve the full
            # completion budget, while tokenizing all known chat-template text.
            messages = []
            conversations = []
            for turn_idx, turn in enumerate(sample.get("turns", [])):
                messages.append({"role": "user", "content": turn})
                conversations.append(list(messages))
                if turn_idx + 1 < len(sample["turns"]):
                    messages.append({"role": "assistant", "content": ""})
        if not conversations:
            raise ValueError("sample has neither conversation nor turns")
        for turn_idx, raw_conversation in enumerate(conversations):
            prepared = prepare_conversation(raw_conversation, family)
            prompt = tokenizer.apply_chat_template(
                prepared, tokenize=False, add_generation_prompt=True
            )
            known = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            lengths.append(known + turn_idx * int(max_new_tokens))
    if not lengths:
        raise RuntimeError(f"{dataset} produced no samples")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="HF B32 KV/workspace capacity preflight")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eagle-tree-width", type=int, default=61)
    parser.add_argument("--specblock-tree-width", type=int, default=91)
    parser.add_argument("--available-gib", type=float, default=None)
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    from transformers import AutoConfig
    geometry = KVGeometry.from_config(
        AutoConfig.from_pretrained(args.model_path, trust_remote_code=True), args.dtype
    )
    output = {"geometry": asdict(geometry), "datasets": {}}
    for dataset in args.datasets:
        lengths = _prompt_lengths(
            args.model_path, dataset, args.max_samples, args.max_new_tokens
        )
        # Use the worst observed contiguous B-size wave. This matches the runner's
        # order-preserving batching and catches a pathological padding tail.
        waves = [lengths[i:i + args.batch_size] for i in range(0, len(lengths), args.batch_size)]
        reports = [report_wave(geometry, wave, args.max_new_tokens, args.eagle_tree_width, args.specblock_tree_width) for wave in waves]
        worst = max(reports, key=lambda item: max(item["eagle3_current_peak_gib"], item["specblock_current_peak_gib"]))
        dataset_report = {
            "samples": len(lengths),
            "max_prompt_tokens": max(lengths),
            "mean_prompt_tokens": sum(lengths) / len(lengths),
            "worst_contiguous_wave": worst,
        }
        if args.available_gib is not None:
            dataset_report["capacity_gate"] = capacity_gate(worst, args.available_gib, args.reserve_gib)
        output["datasets"][dataset] = dataset_report
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
