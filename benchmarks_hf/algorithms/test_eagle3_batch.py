"""CPU correctness tests for EAGLE3 request-batch orchestration.

The test modules are loaded under an isolated synthetic package so this file can
run on CPU-only developer machines without importing the full SpecForge training
stack or optional Triton dependencies.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


_ALGORITHMS_DIR = Path(__file__).resolve().parent
_PACKAGE = "_eagle3_batch_testpkg"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType(_PACKAGE)
package.__path__ = [str(_ALGORITHMS_DIR)]
sys.modules[_PACKAGE] = package

official_package = types.ModuleType(f"{_PACKAGE}.eagle_official")
official_package.__path__ = [str(_ALGORITHMS_DIR / "eagle_official")]
sys.modules[f"{_PACKAGE}.eagle_official"] = official_package
_load_module(
    f"{_PACKAGE}.eagle_official.kv_cache",
    _ALGORITHMS_DIR / "eagle_official" / "kv_cache.py",
)

base_module = types.ModuleType(f"{_PACKAGE}.base")


class _BaseAlgorithm:
    def __init__(self, model_path, draft_model_path=None, device="cuda", **kwargs):
        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.device = device
        self.model_family = "other"
        self.tokenizer = None
        self.last_batch_metrics = None

    @staticmethod
    def compute_accept_length(values):
        return sum(values) / len(values) + 1 if values else 0.0

    @staticmethod
    def compute_cumulative_position_accuracy(values, max_depth=None):
        return {"accuracies": [], "accuracies_pct": [], "formatted": ""}

    def cleanup(self):
        return None


base_module.BaseAlgorithm = _BaseAlgorithm
sys.modules[f"{_PACKAGE}.base"] = base_module

ea_model_module = types.ModuleType(f"{_PACKAGE}.eagle_official.ea_model")
ea_model_module.EaModel = type("EaModel", (), {})
sys.modules[f"{_PACKAGE}.eagle_official.ea_model"] = ea_model_module

utils_module = types.ModuleType(f"{_PACKAGE}.eagle_official.utils")
def _identity_logits_processor(*, temperature=0.0, **kwargs):
    return lambda _input_ids, logits: logits
utils_module.prepare_logits_processor = _identity_logits_processor
sys.modules[f"{_PACKAGE}.eagle_official.utils"] = utils_module

eagle3_module = _load_module(f"{_PACKAGE}.eagle3", _ALGORITHMS_DIR / "eagle3.py")
eagle3_batch_module = _load_module(
    f"{_PACKAGE}.eagle3_batch", _ALGORITHMS_DIR / "eagle3_batch.py"
)
EAGLE3Algorithm = eagle3_module.EAGLE3Algorithm
generate_conversations = eagle3_batch_module.generate_conversations


class _BatchEncoding(SimpleNamespace):
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, conversation, **kwargs):
        return conversation[-1]["content"]

    def __call__(self, prompts, **kwargs):
        rows = [[int(part) for part in prompt.split()] for prompt in prompts]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for idx, row in enumerate(rows):
            input_ids[idx, :len(row)] = torch.tensor(row)
            attention_mask[idx, :len(row)] = 1
        return _BatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, token_ids, skip_special_tokens=True):
        values = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        return " ".join(
            str(value)
            for value in values
            if not skip_special_tokens or value != self.eos_token_id
        )

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id


class _FakeDraftLayer:
    def __init__(self):
        self.total_tokens = 1
        self.stable_kv = None
        self.lm_head = SimpleNamespace(weight=torch.zeros(1, 1))
        self.fc = SimpleNamespace(in_features=3)

    def topK_genrate(self, hidden_states, input_ids, head, logits_processor):
        root = input_ids[:, -1:]
        child = torch.full_like(root, 3)
        draft_tokens = torch.cat((root, child), dim=1)
        retrieve = torch.tensor([[0, 1]], dtype=torch.long)
        tree_mask = torch.tensor(
            [[[[True, False], [True, True]]]], dtype=torch.bool
        )
        tree_positions = torch.tensor([0, 1], dtype=torch.long)
        self.stable_kv = ((torch.zeros(1, 1, input_ids.shape[1] - 1, 1),) * 2,)
        return draft_tokens, retrieve, tree_mask, tree_positions


class _FakeTargetModel:
    def __init__(self):
        projection = SimpleNamespace(weight=torch.zeros(1, 1))
        attention = SimpleNamespace(q_proj=projection)
        layer = SimpleNamespace(self_attn=attention)
        inner_model = SimpleNamespace(layers=[layer], tree_mask=None)
        self.base_model = SimpleNamespace(
            model=inner_model,
            lm_head=object(),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.config = SimpleNamespace(
            hidden_size=1,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        self.ea_layer = _FakeDraftLayer()
        self.call_batch_sizes = []

    def __call__(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        position_ids,
        output_orig,
    ):
        self.call_batch_sizes.append(int(input_ids.shape[0]))
        batch_size, query_len = input_ids.shape
        for layer_kv in past_key_values:
            keys = input_ids.to(torch.float32).view(batch_size, 1, query_len, 1)
            values = keys + 100
            layer_kv[0].cat(keys)
            layer_kv[1].cat(values)

        logits = torch.zeros(batch_size, query_len, 8)
        if self.base_model.model.tree_mask is None:
            logits[..., 2] = 100.0  # verified/root token is EOS
        else:
            logits[..., 4] = 100.0  # reject child token 3
        hidden = torch.arange(
            batch_size * query_len, dtype=torch.float32
        ).reshape(batch_size, query_len, 1)
        outputs = SimpleNamespace(hidden_states=(hidden, hidden + 10, hidden + 20))
        return outputs, logits, hidden


class _FakeAlgorithm:
    def __init__(self):
        self.model = _FakeTargetModel()
        self.tokenizer = _FakeTokenizer()
        self.device = "cpu"
        self.model_family = "other"
        self.depth = 7
        self.last_batch_metrics = None

    def _chat_template_kwargs(self):
        return {}

    compute_accept_length = staticmethod(_BaseAlgorithm.compute_accept_length)
    compute_cumulative_position_accuracy = staticmethod(
        _BaseAlgorithm.compute_cumulative_position_accuracy
    )


class Eagle3BatchTest(unittest.TestCase):
    def test_target_prefill_and_verify_are_single_batch_forwards(self):
        conversations = [
            [{"role": "user", "content": "5 6"}],
            [{"role": "user", "content": "7 8 9"}],
        ]
        batched = _FakeAlgorithm()
        batch_results = generate_conversations(
            batched, conversations, max_new_tokens=8, temperature=0.0
        )

        self.assertEqual(batched.model.call_batch_sizes, [2, 2])
        self.assertEqual(batched.last_batch_metrics["active_sizes"], [2])
        self.assertEqual(batched.last_batch_metrics["acceptance_gpu_calls"], 1)
        self.assertEqual(batched.last_batch_metrics["acceptance_readbacks"], 1)
        self.assertEqual(
            [result["metrics"]["output_token_ids"] for result in batch_results],
            [[2], [2]],
        )

        serial_ids = []
        for conversation in conversations:
            serial = _FakeAlgorithm()
            result = generate_conversations(
                serial, [conversation], max_new_tokens=8, temperature=0.0
            )[0]
            serial_ids.append(result["metrics"]["output_token_ids"])
        self.assertEqual(
            [result["metrics"]["output_token_ids"] for result in batch_results],
            serial_ids,
        )

    def test_temperature_one_b32_uses_batched_prefill_and_verify(self):
        conversations = [
            [{"role": "user", "content": f"{idx + 3} {idx + 4}"}]
            for idx in range(32)
        ]
        batched = _FakeAlgorithm()
        results = generate_conversations(
            batched, conversations, max_new_tokens=4, temperature=1.0
        )

        self.assertEqual(len(results), 32)
        self.assertEqual(batched.model.call_batch_sizes, [32, 32])
        self.assertEqual(batched.last_batch_metrics["active_sizes"], [32])
        self.assertEqual(batched.last_batch_metrics["acceptance_gpu_calls"], 1)
        self.assertEqual(batched.last_batch_metrics["acceptance_readbacks"], 1)
        self.assertTrue(all(result["metrics"]["total_tokens"] == 1 for result in results))

    def test_sampling_accepts_only_target_drawn_tree_children(self):
        tree_input_ids = torch.tensor([[1, 3, 4], [1, 3, 4]])
        retrieve = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        states = [
            SimpleNamespace(tree=(None, retrieve, None, None)),
            SimpleNamespace(tree=(None, retrieve, None, None)),
        ]
        logits = torch.zeros(2, 3, 8)
        logits[0, 0, 3] = 100.0  # Child 3 is in the tree: accept it.
        logits[0, 1, 7] = 100.0  # Then draw the next root from child 3.
        logits[1, 0, 6] = 100.0  # Token 6 is absent: reject and emit it.
        identity = lambda _input_ids, values: values

        accept_lengths, selected_nodes, selected_tokens, next_tokens = (
            eagle3_batch_module._sample_tree_paths(
                states, tree_input_ids, logits, identity
            )
        )

        self.assertEqual(accept_lengths.tolist(), [1, 0])
        self.assertEqual(selected_nodes[:, 0].tolist(), [0, 0])
        self.assertEqual(selected_tokens[:, 0].tolist(), [1, 1])
        self.assertEqual(selected_tokens[0, :2].tolist(), [1, 3])
        self.assertEqual(next_tokens[:, 0].tolist(), [7, 6])

    def test_acceptance_uses_one_batched_metadata_readback(self):
        selector_source = inspect.getsource(eagle3_batch_module._select_tree_paths)
        acceptance_source = inspect.getsource(
            eagle3_batch_module._accept_active_batch
        )
        self.assertIn(
            "target_pred = torch.argmax(tree_logits, dim=-1)",
            selector_source,
        )
        self.assertNotIn(".item(", selector_source + acceptance_source)
        self.assertEqual(acceptance_source.count(".cpu()"), 1)

    def test_selector_matches_scalar_official_semantics(self):
        tree_input_ids = torch.tensor([
            [5, 6, 7, 8, 9],
            [1, 2, 2, 4, 0],
        ])
        retrieves = [
            torch.tensor([
                [0, 1, 2, -1],
                [0, 3, 4, -1],
                [0, 1, 4, -1],
            ]),
            torch.tensor([
                [0, 1, -1],
                [0, 2, 3],
            ]),
        ]
        states = [
            SimpleNamespace(tree=(None, retrieve, None, None))
            for retrieve in retrieves
        ]
        predictions = [
            [6, 7, 4, 3, 3],
            [2, 7, 9, 3, 3],
        ]
        tree_logits = torch.zeros(2, 5, 16)
        for batch_idx, row in enumerate(predictions):
            for node_idx, token_id in enumerate(row):
                tree_logits[batch_idx, node_idx, token_id] = 100.0

        (
            accept_lengths,
            selected_nodes,
            selected_tokens,
            next_tokens,
        ) = eagle3_batch_module._select_tree_paths(
            states,
            tree_input_ids,
            tree_logits,
        )

        for batch_idx, retrieve in enumerate(retrieves):
            padding = torch.full((1,), -1, dtype=torch.long)
            candidates = torch.cat((tree_input_ids[batch_idx], padding))[retrieve]
            path_logits = tree_logits[batch_idx][retrieve]
            target_tokens = torch.argmax(path_logits[:, :-1], dim=-1)
            posterior = (candidates[:, 1:] == target_tokens).int()
            lengths = torch.cumprod(posterior, dim=1).sum(dim=1)
            accept_length = int(lengths.max())
            best_leaf = 0 if accept_length == 0 else int(torch.argmax(lengths))
            selected_count = accept_length + 1

            self.assertEqual(int(accept_lengths[batch_idx]), accept_length)
            self.assertEqual(
                selected_nodes[batch_idx, :selected_count].tolist(),
                retrieve[best_leaf, :selected_count].tolist(),
            )
            self.assertEqual(
                selected_tokens[batch_idx, :selected_count].tolist(),
                candidates[best_leaf, :selected_count].tolist(),
            )
            expected_next = int(
                torch.argmax(path_logits[best_leaf, accept_length])
            )
            self.assertEqual(int(next_tokens[batch_idx, 0]), expected_next)

    def test_sampling_uses_batch_decoder_not_base_serial_fallback(self):
        source = (_ALGORITHMS_DIR / "eagle3.py").read_text(encoding="utf-8")
        self.assertNotIn("sampling supports batch_size=1 only", source)
        self.assertNotIn("return BaseAlgorithm.generate(\n                self,", source)
        batch_source = (_ALGORITHMS_DIR / "eagle3_batch.py").read_text(encoding="utf-8")
        self.assertIn("def _sample_tree_paths(", batch_source)
        self.assertIn("torch.multinomial(", batch_source)

    def test_native_runtime_does_not_override_aten_mm(self):
        source = (_ALGORITHMS_DIR / "eagle3.py").read_text(encoding="utf-8")
        self.assertNotIn("enable_hf_batch_invariant_ops", source)

    def test_public_draft_budget_excludes_verified_root(self):
        for draft_tokens in (60, 90):
            with self.subTest(draft_tokens=draft_tokens):
                algorithm = EAGLE3Algorithm.__new__(EAGLE3Algorithm)
                algorithm.model_path = "target"
                algorithm.draft_model_path = "draft"
                algorithm.device = "cpu"
                algorithm.draft_tokens = draft_tokens
                algorithm.depth = 7
                algorithm.topk = 10

                fake_tokenizer = _FakeTokenizer()
                fake_model = SimpleNamespace(
                    eval=lambda: None,
                    get_tokenizer=lambda: fake_tokenizer,
                )
                with patch.object(
                    eagle3_module.EaModel,
                    "from_pretrained",
                    return_value=fake_model,
                    create=True,
                ) as loader:
                    algorithm.load_model()

                kwargs = loader.call_args.kwargs
                self.assertEqual(kwargs["total_token"], draft_tokens + 1)
                self.assertEqual(kwargs["depth"], 7)
                self.assertEqual(kwargs["top_k"], 10)

    def test_canonical_and_tuned_candidate_capacity(self):
        capacity = 10 + 7 * 10 * 10
        self.assertEqual(capacity, 710)
        self.assertGreaterEqual(capacity, 60)
        self.assertGreaterEqual(capacity, 90)
        self.assertGreaterEqual(capacity, 611)

    def test_sampling_output_respects_token_budget(self):
        wrapper_source = (_ALGORITHMS_DIR / "eagle3.py").read_text(
            encoding="utf-8"
        )
        official_source = (
            _ALGORITHMS_DIR / "eagle_official" / "ea_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "output_ids[0, input_length:input_length + max_new_tokens]",
            wrapper_source,
        )
        self.assertNotIn("if new_token > max_new_tokens:", official_source)
        self.assertIn("if new_token >= max_new_tokens:", official_source)

    def test_batch_sampling_reports_verification_time(self):
        batch_source = (_ALGORITHMS_DIR / "eagle3_batch.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('timer.stop("verify", verify_start)', batch_source)
        self.assertIn('"verify_time": verify_time', batch_source)
        self.assertIn('"verify_pct": verify_time / batch_wall_time', batch_source)

    def test_depth_matches_official_generation_rounds(self):
        source = (_ALGORITHMS_DIR / "eagle_official" / "cnets.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("for i in range(depth):", source)

    def test_run_eval_dispatches_real_batches(self):
        source = (_ALGORITHMS_DIR.parent / "run_eval.py").read_text(encoding="utf-8")
        self.assertIn("for batch_start in range(0, len(data), batch_size):", source)
        self.assertIn('not getattr(algorithm, "supports_true_batch", False)', source)
        self.assertIn("responses = algorithm.generate(\n                        batch_samples,", source)
        self.assertIn('"eagle3":                     "1,7,10,60"', source)


if __name__ == "__main__":
    unittest.main()
