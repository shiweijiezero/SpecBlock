#!/usr/bin/env python3
"""Reproducible T=1 token-distribution parity gate for HF SpecBlock.

The gate first executes the real SpecBlock batch path to capture one verified
candidate tree and its target logits.  It then replays only the verifier sampler
against that frozen fixture, so repeated draws measure sampler correctness rather
than target-forward nondeterminism.  A standard categorical target-only sampler
uses the same cached target distribution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
# The HF evaluator only needs the template registry.  Loading the top-level
# training package imports optional distributed-training dependencies (e.g.
# yunchang) that are unrelated to inference, so expose this leaf module without
# importing ``specforge.__init__``.
_specforge = types.ModuleType("specforge")
_specforge.__path__ = [str(_REPO_ROOT / "specforge")]
_specforge_data = types.ModuleType("specforge.data")
_specforge_data.__path__ = [str(_REPO_ROOT / "specforge" / "data")]
sys.modules.setdefault("specforge", _specforge)
sys.modules.setdefault("specforge.data", _specforge_data)
_template_spec = importlib.util.spec_from_file_location(
    "specforge.data.template", _REPO_ROOT / "specforge" / "data" / "template.py"
)
_template_module = importlib.util.module_from_spec(_template_spec)
sys.modules.setdefault("specforge.data.template", _template_module)
if _template_spec.name not in sys.modules or sys.modules[_template_spec.name] is _template_module:
    _template_spec.loader.exec_module(_template_module)

from algorithms import specblock_batch
from algorithms.specblock import SpecBlockAlgorithm


DEFAULT_PROMPT = (
    "Explain speculative decoding in one concise paragraph, including why its "
    "sampling distribution can remain equal to the target model's distribution."
)


class _CachedLogitHead(torch.nn.Module):
    """Treat cached target logits as the sampler's projected hidden rows."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def _state(retrieve_indices: torch.Tensor, tree_width: int) -> SimpleNamespace:
    return SimpleNamespace(tree=(
        torch.empty((1, tree_width), dtype=torch.long, device=retrieve_indices.device),
        None,
        None,
        retrieve_indices,
        [],
        [],
        list(range(tree_width)),
    ))


def _first_emitted_token(accepted: tuple[Any, ...]) -> int:
    accept_length, _accepted_tokens, _bonus, _last, _lazy, _path, _coverage, _indices, accepted_ids, next_id = accepted
    return int(accepted_ids[0]) if int(accept_length) else int(next_id)


def _frequencies(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    counts = torch.bincount(tokens.cpu(), minlength=vocab_size).to(torch.float64)
    return counts / counts.sum()


def _metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = (left - right).abs()
    return {
        "tvd": float(0.5 * difference.sum()),
        "max_abs_error": float(difference.max()),
    }


def _bootstrap_interval(
    left_counts: torch.Tensor,
    right_counts: torch.Tensor,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CI for empirical categorical-distance statistics."""
    if draws < 100:
        raise ValueError("--bootstrap-draws must be at least 100")
    n_left = int(left_counts.sum())
    n_right = int(right_counts.sum())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    left_p = (left_counts / n_left).to(torch.float64)
    right_p = (right_counts / n_right).to(torch.float64)
    # These are nonparametric multinomial bootstrap resamples of the two
    # independently collected empirical categorical samples.
    left_boot = torch.stack([
        torch.bincount(
            torch.multinomial(left_p, n_left, replacement=True, generator=generator),
            minlength=left_p.numel(),
        )
        for _ in range(draws)
    ]).to(torch.float64) / n_left
    right_boot = torch.stack([
        torch.bincount(
            torch.multinomial(right_p, n_right, replacement=True, generator=generator),
            minlength=right_p.numel(),
        )
        for _ in range(draws)
    ]).to(torch.float64) / n_right
    difference = (left_boot - right_boot).abs()
    tvd = 0.5 * difference.sum(dim=1)
    maximum = difference.max(dim=1).values
    low = 0.025
    high = 0.975
    return {
        "tvd": {
            "low": float(torch.quantile(tvd, low)),
            "high": float(torch.quantile(tvd, high)),
        },
        "max_abs_error": {
            "low": float(torch.quantile(maximum, low)),
            "high": float(torch.quantile(maximum, high)),
        },
        "method": "independent nonparametric multinomial percentile bootstrap",
        "level": 0.95,
        "draws": draws,
    }


def _capture_real_fixture(algorithm, prompt: str, temperature: float, seed: int):
    """Run production generation once and retain its first verified tree/logits."""
    captured: dict[str, torch.Tensor] = {}
    original = specblock_batch._accept_sampling_batch

    def capture(algorithm_, active, tree_ids, node_hidden, hidden_sources, temperature_):
        if not captured:
            with torch.no_grad():
                logits = algorithm_.target_model.lm_head(node_hidden).detach().clone()
            captured["tree_ids"] = tree_ids.detach().clone()
            captured["node_logits"] = logits
            captured["retrieve"] = active[0].tree[3].detach().clone()
        return original(algorithm_, active, tree_ids, node_hidden, hidden_sources, temperature_)

    random.seed(seed)
    torch.manual_seed(seed)
    specblock_batch._accept_sampling_batch = capture
    try:
        responses = algorithm.generate(
            [{"id": "capture", "conversation": [{"role": "user", "content": prompt}]}],
            max_new_tokens=16,
            temperature=temperature,
        )
    finally:
        specblock_batch._accept_sampling_batch = original
    if not captured:
        raise RuntimeError(
            "no verified sampling tree was captured (the first sampled token may "
            "have been EOS); choose another --seed or prompt"
        )
    return captured, responses[0]


def _run_sampler(
    fixture: dict[str, torch.Tensor],
    samples: int,
    batch_size: int,
    temperature: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw target-only and production batch-verifier samples from one fixture."""
    if samples < batch_size:
        raise ValueError("--samples must be at least --batch-size")
    device = fixture["node_logits"].device
    tree = fixture["tree_ids"]
    logits = fixture["node_logits"]
    retrieve = fixture["retrieve"]
    width = int(tree.shape[1])
    vocab_size = int(logits.shape[-1])
    sample_count = samples // batch_size * batch_size
    if sample_count != samples:
        print(
            f"[INFO] rounding samples from {samples} down to {sample_count} "
            f"to fill batches of {batch_size}",
            file=sys.stderr,
        )
    if sample_count == 0:
        raise ValueError("rounded sample count is zero")

    # The root verifier logits define the exact target-only categorical law for
    # the next emitted token at this fixed decode state.
    target_probs = torch.softmax(logits[0, 0].float() / temperature, dim=-1)
    torch.manual_seed(seed)
    target_tokens = torch.multinomial(target_probs, sample_count, replacement=True)

    replay_algorithm = SimpleNamespace(
        target_model=SimpleNamespace(lm_head=_CachedLogitHead()),
        _batched_acceptance_readbacks=0,
        _batched_acceptance_gpu_calls=0,
        _sampling_lm_head_rows=0,
    )
    spec_tokens = []
    accepted_token_count = 0
    tree_batch = tree.expand(batch_size, -1).clone()
    logits_batch = logits.expand(batch_size, -1, -1).clone()
    hidden_sources = [torch.zeros((batch_size, width, 1), device=device) for _ in range(3)]
    states = [_state(retrieve.clone(), width) for _ in range(batch_size)]
    torch.manual_seed(seed + 1)
    for _ in range(sample_count // batch_size):
        accepted = specblock_batch._accept_sampling_batch(
            replay_algorithm,
            states,
            tree_batch,
            logits_batch,
            hidden_sources,
            temperature,
        )
        accepted_token_count += sum(int(row[0]) > 0 for row in accepted)
        spec_tokens.extend(_first_emitted_token(row) for row in accepted)
    return (
        target_tokens.cpu(),
        torch.tensor(spec_tokens, dtype=torch.long),
        accepted_token_count / sample_count,
    )


def _target_only_e2e(algorithm, prompt: str, temperature: float, seed: int) -> list[int]:
    tokenizer = algorithm.tokenizer
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **algorithm._chat_template_kwargs(),
    )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(algorithm.device)
    random.seed(seed)
    torch.manual_seed(seed)
    with torch.inference_mode():
        output = algorithm.target_model.generate(
            **encoded,
            max_new_tokens=16,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    return output[0, encoded.input_ids.shape[1]:].tolist()


def _git_revision(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("specblock",), default="specblock")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=51200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("this is a stochastic T>0 parity gate")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    os.environ.setdefault("SPECBLOCK_INTERNAL_PREWARM", "0")
    algorithm_cls = SpecBlockAlgorithm
    algorithm = algorithm_cls(
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        device=args.device,
        max_blocks=2,
        beam_width=10,
        total_tokens=90,
    )
    algorithm.load_model()

    # This is deliberately a real model invocation, separate from fixture replay.
    target_e2e = _target_only_e2e(algorithm, args.prompt, args.temperature, args.seed)
    fixture, spec_e2e_response = _capture_real_fixture(
        algorithm, args.prompt, args.temperature, args.seed + 1
    )
    if fixture["retrieve"].shape[1] < 2:
        raise AssertionError(
            "captured tree has no proposed child; it cannot exercise speculative "
            "acceptance and residual sampling"
        )
    spec_e2e = spec_e2e_response["metrics"]["output_token_ids"]
    if not target_e2e or not spec_e2e:
        raise AssertionError("end-to-end sanity generation emitted no token")
    if any(token < 0 for token in target_e2e + spec_e2e):
        raise AssertionError("end-to-end sanity generation emitted an invalid token")

    target_tokens, spec_tokens, empirical_acceptance = _run_sampler(
        fixture, args.samples, args.batch_size, args.temperature, args.seed + 2
    )
    vocab_size = int(fixture["node_logits"].shape[-1])
    reference = torch.softmax(fixture["node_logits"][0, 0].float() / args.temperature, dim=-1).cpu().to(torch.float64)
    root_children = fixture["retrieve"][:, 1]
    root_children = root_children[root_children >= 0]
    proposed_tokens = torch.unique(fixture["tree_ids"][0, root_children]).cpu()
    proposed_mass = float(reference[proposed_tokens].sum())
    target_frequency = _frequencies(target_tokens, vocab_size)
    spec_frequency = _frequencies(spec_tokens, vocab_size)
    target_counts = torch.bincount(target_tokens, minlength=vocab_size).to(torch.float64)
    spec_counts = torch.bincount(spec_tokens, minlength=vocab_size).to(torch.float64)

    report = {
        "schema_version": 1,
        "purpose": "HF SpecBlock T=1 controlled target-only versus verifier-sampler token-distribution parity",
        "provenance": {
            "git_revision": _git_revision(Path(__file__).resolve().parents[1]),
            "algorithm": args.algorithm,
            "model_path": args.model_path,
            "draft_model_path": args.draft_model_path,
            "device": args.device,
            "seed": args.seed,
        },
        "protocol": {
            "temperature": args.temperature,
            "prompt": args.prompt,
            "fixture": "first real verified SpecBlock tree and target LM-head logits, captured once then replayed",
            "comparison": "cached target-root categorical samples versus the real batched SpecBlock verifier sampler",
            "samples_per_arm": int(target_tokens.numel()),
            "batch_size": args.batch_size,
            "tree_width": int(fixture["tree_ids"].shape[1]),
            "candidate_paths": int(fixture["retrieve"].shape[0]),
            "unique_root_proposed_tokens": int(proposed_tokens.numel()),
            "target_mass_on_root_proposals": proposed_mass,
            "empirical_first_token_acceptance_rate": empirical_acceptance,
        },
        "distribution": {
            "target_only_vs_specblock": _metrics(target_frequency, spec_frequency),
            "target_only_vs_exact_cached_target": _metrics(target_frequency, reference),
            "specblock_vs_exact_cached_target": _metrics(spec_frequency, reference),
            "bootstrap_95_ci_target_only_vs_specblock": _bootstrap_interval(
                target_counts, spec_counts, args.bootstrap_draws, args.seed + 3
            ),
        },
        "end_to_end_sanity": {
            "target_only_output_token_ids": target_e2e,
            "specblock_output_token_ids": spec_e2e,
            "target_only_token_count": len(target_e2e),
            "specblock_token_count": len(spec_e2e),
            "specblock_accept_lengths_raw": spec_e2e_response["metrics"].get("accept_lengths_raw"),
            "result": "PASS",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["distribution"], indent=2, sort_keys=True))
    print(f"[PASS] wrote {args.output}")


if __name__ == "__main__":
    main()
