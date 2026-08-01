#!/usr/bin/env python3
"""Logic gate for request-level HF EAGLE3 target batching."""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from algorithms.eagle3 import EAGLE3Algorithm
from verify_specblock_batch import PROMPTS


def _conversation(prompt):
    return [{"role": "user", "content": prompt}]


def _assert_batch_logic(algorithm, responses, batch_size, max_new_tokens):
    if len(responses) != batch_size:
        raise AssertionError(
            f"expected {batch_size} responses, got {len(responses)}"
        )

    metrics = algorithm.last_batch_metrics or {}
    rounds = int(metrics.get("iterations", -1))
    active_sizes = list(metrics.get("active_sizes", []))
    gpu_calls = int(metrics.get("acceptance_gpu_calls", -1))
    readbacks = int(metrics.get("acceptance_readbacks", -1))

    if rounds <= 0 or len(active_sizes) != rounds:
        raise AssertionError(
            f"invalid decode rounds: rounds={rounds}, active_sizes={active_sizes}"
        )
    if active_sizes[0] != batch_size:
        raise AssertionError(
            f"active batch must start at {batch_size}, got {active_sizes[0]}"
        )
    if any(size <= 0 or size > batch_size for size in active_sizes):
        raise AssertionError(f"active batch sizes are invalid: {active_sizes}")
    if any(lhs < rhs for lhs, rhs in zip(active_sizes, active_sizes[1:])):
        raise AssertionError(f"active batch grew during decode: {active_sizes}")
    if gpu_calls != rounds or readbacks != rounds:
        raise AssertionError(
            "acceptance must use one GPU call and one metadata readback per round; "
            f"rounds={rounds}, gpu_calls={gpu_calls}, readbacks={readbacks}"
        )

    vocab_size = int(algorithm.model.config.vocab_size)
    for request_idx, response in enumerate(responses):
        row = response.get("metrics", {})
        token_ids = row.get("output_token_ids")
        total_tokens = int(row.get("total_tokens", -1))
        accept_lengths = row.get("accept_lengths_raw")
        request_rounds = int(row.get("iterations", -1))

        if not isinstance(token_ids, list) or total_tokens != len(token_ids):
            raise AssertionError(
                f"request {request_idx} token accounting is invalid"
            )
        if total_tokens <= 0 or total_tokens > max_new_tokens:
            raise AssertionError(
                f"request {request_idx} emitted {total_tokens} tokens"
            )
        if any(
            not isinstance(token_id, int)
            or token_id < 0
            or token_id >= vocab_size
            for token_id in token_ids
        ):
            raise AssertionError(
                f"request {request_idx} emitted an invalid token ID"
            )
        if (
            not isinstance(accept_lengths, list)
            or request_rounds != len(accept_lengths)
            or request_rounds <= 0
            or any(
                not isinstance(length, int) or length < 0
                for length in accept_lengths
            )
        ):
            raise AssertionError(
                f"request {request_idx} acceptance accounting is invalid"
            )
        for name in (
            "wall_time",
            "prefill_time",
            "draft_time",
            "target_time",
            "verify_time",
            "other_time",
        ):
            value = float(row.get(name, -1.0))
            if not math.isfinite(value) or value < 0.0:
                raise AssertionError(
                    f"request {request_idx} has invalid {name}={value}"
                )

    return rounds, active_sizes, gpu_calls, readbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--draft-tokens", type=int, choices=(60, 90), required=True)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32]
    )
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=[0.0, 1.0],
        help="Run greedy and true-sampling batch gates without serial fallback.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    max_batch = max(args.batch_sizes)
    prompt_end = args.prompt_offset + max_batch
    if args.prompt_offset < 0 or prompt_end > len(PROMPTS):
        raise ValueError(
            f"prompt slice [{args.prompt_offset}:{prompt_end}] exceeds "
            f"the {len(PROMPTS)} test prompts"
        )
    prompts = PROMPTS[args.prompt_offset:prompt_end]

    algorithm = EAGLE3Algorithm(
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        device=args.device,
        draft_tokens=args.draft_tokens,
        depth=args.depth,
        topk=args.topk,
    )
    algorithm.load_model()

    for temperature in args.temperatures:
        if temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        for batch_size in args.batch_sizes:
            samples = [
                {
                    "id": f"eagle-logic-{args.prompt_offset + idx}",
                    "conversation": _conversation(prompt),
                }
                for idx, prompt in enumerate(prompts[:batch_size])
            ]
            responses = algorithm.generate(
                samples,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
            )
            rounds, active_sizes, gpu_calls, readbacks = _assert_batch_logic(
                algorithm,
                responses,
                batch_size,
                args.max_new_tokens,
            )
            print(
                f"draft_tokens={args.draft_tokens} temperature={temperature:g} "
                f"B={batch_size}: logic_ok=True rounds={rounds} "
                f"active_sizes={active_sizes} acceptance_gpu_calls={gpu_calls} "
                f"acceptance_readbacks={readbacks}",
                flush=True,
            )

    print(
        f"[PASS] EAGLE3-{args.draft_tokens} request-batch logic is valid",
        flush=True,
    )


if __name__ == "__main__":
    main()
