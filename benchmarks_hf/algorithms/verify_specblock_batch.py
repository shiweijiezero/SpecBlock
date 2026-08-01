#!/usr/bin/env python3
"""Greedy exactness gate for request-level HF SpecBlock batching."""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from algorithms.specblock import SpecBlockAlgorithm


PROMPTS = [
    "Reply with exactly one word: yes.",
    "What is 2 + 2? Give only the number.",
    "Write a Python function that returns the square of an integer.",
    "Explain why the sky appears blue in two concise sentences.",
    "Translate 'The library closes at six' into French.",
    "List the first five prime numbers separated by commas.",
    "If Alice has 12 apples and gives Bob 5, how many remain?",
    "Summarize the idea of speculative decoding in one paragraph.",
    "Complete the sequence: 1, 1, 2, 3, 5, 8,",
    "Give a short example of a JSON object with a name and an age.",
    "What is the capital of Japan? Answer briefly.",
    "Explain the difference between a process and a thread.",
    "Implement binary search in Python and state its time complexity.",
    "A train travels 180 km in 3 hours. Compute its average speed and show the equation.",
    "Write a polite two-sentence email declining a meeting because of a scheduling conflict.",
    (
        "Read the following context and answer the final question. Context: "
        "Speculative decoding uses a smaller draft model to propose multiple future tokens. "
        "A larger target model verifies those proposals in parallel, preserving the target "
        "distribution when acceptance is implemented correctly. Tree-based methods include "
        "multiple alternative continuations so that the target can accept a matching branch. "
        "Question: What property must a lossless speculative decoder preserve?"
    ),
]


def _conversation(prompt):
    return [{"role": "user", "content": prompt}]


def _target_reference(algorithm, prompt, max_new_tokens):
    conversation = algorithm.prepare_conversation(_conversation(prompt))
    text = algorithm.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = algorithm.tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).to(algorithm.device)
    with torch.inference_mode():
        output = algorithm.target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=algorithm.tokenizer.pad_token_id,
            eos_token_id=algorithm.tokenizer.eos_token_id,
        )
    return output[0, inputs.input_ids.shape[1]:].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument(
        "--prompt-count",
        type=int,
        help="Number of prompts to cover; must be divisible by every batch size.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strict-linear", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("SPECBLOCK_INTERNAL_PREWARM", "0")
    algorithm = SpecBlockAlgorithm(
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        device=args.device,
        max_blocks=2,
        beam_width=1 if args.strict_linear else 10,
        total_tokens=8 if args.strict_linear else 90,
        strict_linear=args.strict_linear,
    )
    algorithm.load_model()

    max_batch = max(args.batch_sizes)
    prompt_count = args.prompt_count or max_batch
    prompt_end = args.prompt_offset + prompt_count
    if args.prompt_offset < 0 or prompt_count <= 0 or prompt_end > len(PROMPTS):
        raise ValueError(
            f"prompt slice [{args.prompt_offset}:{prompt_end}] exceeds "
            f"the {len(PROMPTS)} test prompts"
        )
    for batch_size in args.batch_sizes:
        if prompt_count % batch_size:
            raise ValueError(
                f"prompt count {prompt_count} must be divisible by B={batch_size}"
            )
    selected_prompts = PROMPTS[args.prompt_offset:prompt_end]

    print("Computing per-request target-only references...", flush=True)
    references = [
        _target_reference(algorithm, prompt, args.max_new_tokens)
        for prompt in selected_prompts
    ]

    for batch_size in args.batch_sizes:
        mismatches = []
        total_rounds = 0
        for batch_start in range(0, prompt_count, batch_size):
            batch_prompts = selected_prompts[batch_start:batch_start + batch_size]
            samples = [
                {
                    "id": f"exact-{args.prompt_offset + batch_start + idx}",
                    "conversation": _conversation(prompt),
                }
                for idx, prompt in enumerate(batch_prompts)
            ]
            responses = algorithm.generate(
                samples,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            if len(responses) != batch_size:
                raise AssertionError(
                    f"B={batch_size}: expected {batch_size} responses, got {len(responses)}"
                )

            for idx, response in enumerate(responses):
                prompt_idx = batch_start + idx
                actual = response["metrics"].get("output_token_ids")
                if actual is None:
                    raise AssertionError(
                        "SpecBlock response is missing metrics.output_token_ids"
                    )
                if actual != references[prompt_idx]:
                    first_diff = next(
                        (
                            pos
                            for pos, (lhs, rhs) in enumerate(
                                zip(actual, references[prompt_idx])
                            )
                            if lhs != rhs
                        ),
                        min(len(actual), len(references[prompt_idx])),
                    )
                    mismatches.append((
                        prompt_idx,
                        first_diff,
                        references[prompt_idx][first_diff:first_diff + 8],
                        actual[first_diff:first_diff + 8],
                        response["metrics"].get("accept_lengths_raw"),
                    ))

            if args.verbose:
                for idx, response in enumerate(responses):
                    print(
                        f"  request={args.prompt_offset + batch_start + idx} "
                        f"accept_lengths={response['metrics'].get('accept_lengths_raw')} "
                        f"tokens={response['metrics'].get('output_token_ids')}",
                        flush=True,
                    )

            metrics = algorithm.last_batch_metrics or {}
            total_rounds += int(metrics.get("iterations") or 0)

        print(
            f"B={batch_size}: exact={not mismatches} prompts={prompt_count} "
            f"rounds={total_rounds}",
            flush=True,
        )
        if mismatches:
            for idx, pos, expected, actual, accept_lengths in mismatches:
                print(
                    f"  request={idx} first_diff={pos} expected={expected} actual={actual} "
                    f"accept_lengths={accept_lengths}",
                    flush=True,
                )
            raise AssertionError(f"B={batch_size}: {len(mismatches)} exact-token mismatches")

    mode = "strict-linear SpecBlock" if args.strict_linear else "SpecBlock"
    print(f"[PASS] {mode} matches target-only token IDs for all requested batch sizes")


if __name__ == "__main__":
    main()
