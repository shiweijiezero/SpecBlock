"""CPU synthetic tests for BaselineAlgorithm request-level batching."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


_ALGORITHMS_DIR = Path(__file__).resolve().parent
_PACKAGE = "_baseline_batch_testpkg"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(_PACKAGE)
package.__path__ = [str(_ALGORITHMS_DIR)]
sys.modules[_PACKAGE] = package

base_module = types.ModuleType(f"{_PACKAGE}.base")


class _BaseAlgorithm:
    def __init__(self, model_path, draft_model_path=None, device="cuda", **kwargs):
        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.device = device
        self.tokenizer = None
        self.last_batch_metrics = None

    @staticmethod
    def prepare_conversation(conversation):
        return list(conversation)


base_module.BaseAlgorithm = _BaseAlgorithm
sys.modules[f"{_PACKAGE}.base"] = base_module

transformers_module = types.ModuleType("transformers")
transformers_module.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
transformers_module.AutoTokenizer = type("AutoTokenizer", (), {})
sys.modules.setdefault("transformers", transformers_module)

baseline_module = _load_module(
    f"{_PACKAGE}.baseline", _ALGORITHMS_DIR / "baseline.py"
)
BaselineAlgorithm = baseline_module.BaselineAlgorithm


class _BatchEncoding(dict):
    def __init__(self, input_ids, attention_mask):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)

    @property
    def input_ids(self):
        return self["input_ids"]

    @property
    def attention_mask(self):
        return self["attention_mask"]

    def to(self, device):
        self["input_ids"] = self.input_ids.to(device)
        self["attention_mask"] = self.attention_mask.to(device)
        return self


class _FakeTokenizer:
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token_id = 0
    pad_token = "<pad>"

    def __init__(self):
        self.padding_side = "right"
        self.calls = []

    def apply_chat_template(self, conversation, **kwargs):
        return "|".join(
            f"{message['role']}:{message['content']}" for message in conversation
        )

    def __call__(self, prompts, **kwargs):
        self.calls.append({
            "prompts": list(prompts),
            "padding_side": self.padding_side,
            "kwargs": dict(kwargs),
        })
        rows = []
        for prompt in prompts:
            numbers = [int(value) for value in prompt.replace("|", " ").replace(":", " ").split() if value.isdigit()]
            rows.append(numbers or [1])
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            row_tensor = torch.tensor(row, dtype=torch.long)
            if self.padding_side == "left":
                input_ids[index, width - len(row):] = row_tensor
                attention_mask[index, width - len(row):] = 1
            else:
                input_ids[index, :len(row)] = row_tensor
                attention_mask[index, :len(row)] = 1
        return _BatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, token_ids, skip_special_tokens=True):
        values = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        return " ".join(
            str(value)
            for value in values
            if not skip_special_tokens or value != self.eos_token_id
        )


class _FakeTargetModel:
    def __init__(self):
        self.call_batch_sizes = []
        self.inputs = []

    def generate(self, input_ids, attention_mask, **kwargs):
        self.call_batch_sizes.append(int(input_ids.shape[0]))
        self.inputs.append((input_ids.clone(), attention_mask.clone(), dict(kwargs)))
        if not bool(torch.all(attention_mask[:, -1])):
            raise AssertionError("decoder-only batched generation must be left padded")

        batch_size, prompt_width = input_ids.shape
        outputs = input_ids.new_zeros((batch_size, prompt_width + 3))
        outputs[:, :prompt_width] = input_ids
        for row_idx, last_token in enumerate(input_ids[:, -1].tolist()):
            if last_token % 2 == 0:
                outputs[row_idx, prompt_width:] = torch.tensor([last_token + 10, 0, 0])
            else:
                outputs[row_idx, prompt_width:] = torch.tensor([last_token + 10, last_token + 11, 0])
        return outputs


def _algorithm():
    algorithm = BaselineAlgorithm("synthetic", device="cpu")
    algorithm.tokenizer = _FakeTokenizer()
    algorithm.model = _FakeTargetModel()
    return algorithm


class BaselineBatchTest(unittest.TestCase):
    def test_true_batch_pads_left_and_matches_serial_outputs(self):
        samples = [
            {"conversation": [{"role": "user", "content": "1 2"}]},
            {"conversation": [{"role": "user", "content": "3 4 5"}]},
        ]
        batched = _algorithm()
        batch_results = batched.generate(samples, max_new_tokens=3)

        self.assertTrue(BaselineAlgorithm.supports_true_batch)
        self.assertEqual(batched.model.call_batch_sizes, [2])
        self.assertEqual(batched.tokenizer.calls[0]["padding_side"], "left")
        self.assertTrue(batched.tokenizer.calls[0]["kwargs"]["padding"])
        self.assertEqual(batched.tokenizer.padding_side, "right")
        self.assertEqual(
            batched.model.inputs[0][0].tolist(), [[0, 1, 2], [3, 4, 5]]
        )
        self.assertEqual(
            [result["metrics"]["output_token_ids"] for result in batch_results],
            [[12, 0], [15, 16, 0]],
        )
        self.assertEqual([result["output"] for result in batch_results], ["12", "15 16"])
        self.assertEqual(batched.last_batch_metrics["active_sizes"], [2, 2, 1])
        self.assertEqual(batched.last_batch_metrics["iterations"], 3)
        self.assertEqual(batched.last_batch_metrics["engine_batch_size"], 2)
        self.assertEqual(batched.last_batch_metrics["prefill_time"], 0.0)
        self.assertEqual(
            batched.last_batch_metrics["target_time"],
            batched.last_batch_metrics["wall_time"],
        )

        serial_outputs = []
        for sample in samples:
            serial = _algorithm()
            serial_outputs.append(serial.generate([sample], max_new_tokens=3)[0]["output"])
        self.assertEqual([result["output"] for result in batch_results], serial_outputs)

    def test_multiturn_turn_waves_shrink_active_request_batch(self):
        samples = [
            {"turns": ["1 2", "3 4"]},
            {"turns": ["5 6"]},
        ]
        algorithm = _algorithm()
        results = algorithm.generate(samples, max_new_tokens=3)

        self.assertEqual(algorithm.model.call_batch_sizes, [2, 1])
        self.assertEqual(algorithm.last_batch_metrics["engine_batch_size"], 2)
        self.assertEqual(algorithm.last_batch_metrics["turn_batches"], 2)
        self.assertEqual(algorithm.last_batch_metrics["active_sizes"], [2, 2, 1, 1])
        self.assertEqual(results[0]["output"], ["12", "14"])
        self.assertEqual(results[1]["output"], ["16"])
        self.assertEqual(results[0]["metrics"]["num_turns"], 2)
        self.assertIn("assistant:12", algorithm.tokenizer.calls[1]["prompts"][0])

    def test_b1_metrics_keep_scalar_decode_semantics(self):
        algorithm = _algorithm()
        result = algorithm.generate(
            [{"conversation": [{"role": "user", "content": "1 2"}]}],
            max_new_tokens=3,
        )[0]

        self.assertEqual(algorithm.model.call_batch_sizes, [1])
        self.assertEqual(result["metrics"]["iterations"], 2)
        self.assertEqual(algorithm.last_batch_metrics["iterations"], 2)
        self.assertEqual(algorithm.last_batch_metrics["active_sizes"], [1, 1])
        self.assertEqual(algorithm.last_batch_metrics["engine_batch_size"], 1)

    def test_run_eval_dispatches_bigger_request_batches(self):
        class _RunnerAlgorithm:
            supports_true_batch = True

            def __init__(self):
                self.batch_sizes = []
                self.last_batch_metrics = None

            def generate(self, samples, **kwargs):
                self.batch_sizes.append(len(samples))
                self.last_batch_metrics = {
                    "wall_time": 1.0,
                    "prefill_time": 0.0,
                    "draft_time": 0.0,
                    "target_time": 1.0,
                    "verify_time": 0.0,
                    "iterations": 1,
                    "active_sizes": [len(samples)],
                    "engine_batch_size": len(samples),
                }
                return [
                    {"output": "ok", "metrics": {"total_tokens": 1}}
                    for _ in samples
                ]

            def cleanup(self):
                return None

        algorithm = _RunnerAlgorithm()
        dataset_module = types.ModuleType("benchmark_datasets")
        dataset_module.load_benchmark_dataset = lambda name, **kwargs: [
            {"id": f"sample-{index}", "conversation": [{"role": "user", "content": str(index)}]}
            for index in range(3)
        ]
        algorithms_module = types.ModuleType("algorithms")
        algorithms_module.get_algorithm = lambda *args, **kwargs: algorithm

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            sys.modules,
            {"benchmark_datasets": dataset_module, "algorithms": algorithms_module},
        ):
            run_eval = _load_module(
                "_baseline_batch_run_eval", _ALGORITHMS_DIR.parent / "run_eval.py"
            )
            output = Path(tmp_dir) / "result.jsonl"
            argv = [
                "run_eval.py",
                "--algorithm", "baseline",
                "--model-path", "synthetic",
                "--benchmark-list", "synthetic:3",
                "--config-list", "2,1,1,1",
                "--max-new-tokens", "3",
                "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                run_eval.main()

            self.assertEqual(algorithm.batch_sizes, [2, 2, 1])
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(summary["true_batch"])
            self.assertEqual(summary["measured_samples"], 3)
            self.assertEqual(summary["active_batch_sizes"], [2, 1])


if __name__ == "__main__":
    unittest.main()
