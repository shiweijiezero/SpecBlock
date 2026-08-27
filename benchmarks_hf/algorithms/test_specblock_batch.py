"""CPU tests for SpecBlock request-batch sampling acceptance."""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import types
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch


_ALGORITHMS_DIR = Path(__file__).resolve().parent
_PACKAGE = "_specblock_batch_testpkg"
package = types.ModuleType(_PACKAGE)
package.__path__ = [str(_ALGORITHMS_DIR)]
sys.modules[_PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    f"{_PACKAGE}.specblock_batch", _ALGORITHMS_DIR / "specblock_batch.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class _LookupHead(torch.nn.Module):
    """Return a fixture logit row indexed by hidden[..., 0]."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, hidden):
        return self.logits[hidden[..., 0].long()]


def _state(paths, width, device="cpu"):
    return SimpleNamespace(tree=(
        torch.zeros((1, width), dtype=torch.long, device=device), None, None,
        torch.tensor(paths, dtype=torch.long, device=device), None, None, list(range(width)),
    ))


def _algorithm(logits):
    return SimpleNamespace(
        target_model=SimpleNamespace(lm_head=_LookupHead(logits)),
        _batched_acceptance_readbacks=0,
        _batched_acceptance_gpu_calls=0,
    )


class SpecBlockBatchSamplingTest(unittest.TestCase):
    def test_deterministic_ragged_fixture_preserves_paths_and_bonus(self):
        # Row zero accepts proposed 11 then samples its bonus from node 1.  Row
        # one rejects proposed 21 and samples its root residual bonus (22).
        tree_input_ids = torch.tensor([[10, 11, 12], [20, 21, 0]])
        states = [_state([[0, 1], [0, 2]], 3), _state([[0, 1]], 2)]
        logits = torch.full((6, 32), -100.0)
        logits[0, 11] = 100.0
        logits[1, 13] = 100.0
        logits[3, 22] = 100.0
        node_hidden = torch.tensor([[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]])
        hidden_sources = [
            torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2) + offset
            for offset in (0, 100, 200)
        ]
        algorithm = _algorithm(logits)

        torch.manual_seed(0)
        accepted = module._accept_sampling_batch(
            algorithm, states, tree_input_ids, node_hidden, hidden_sources, 1.0
        )

        self.assertEqual(len(accepted), 2)
        self.assertEqual(accepted[0][0], 1)
        self.assertEqual(accepted[0][1].tolist(), [11])
        self.assertEqual(accepted[0][2].tolist(), [[13]])
        self.assertEqual(accepted[0][7].tolist(), [0, 1])
        self.assertEqual(accepted[0][8], [11])
        self.assertEqual(accepted[1][0], 0)
        self.assertEqual(accepted[1][1].tolist(), [])
        self.assertEqual(accepted[1][2].tolist(), [[22]])
        self.assertEqual(accepted[1][7].tolist(), [0])
        self.assertEqual(accepted[1][8], [])
        self.assertEqual(algorithm._batched_acceptance_gpu_calls, 1)
        self.assertEqual(algorithm._batched_acceptance_readbacks, 1)

    def test_residual_sampling_has_target_distribution(self):
        # A single proposed child has p=0.25.  Acceptance plus residual bonus
        # must leave the unconditional continuation equal to the target p.  On
        # a server this is the CUDA B32 all-reject gate for failed_probs.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab, rounds = 4, 300
        logits = torch.log(torch.tensor([[0.10, 0.25, 0.30, 0.35]], device=device))
        algorithm = _algorithm(logits)
        # Repeating the child across retrieve rows exercises ordered sibling
        # deduplication and ensures a duplicate cannot restore rejected mass.
        states = [_state([[0, 1], [0, 1]], 2, device=device) for _ in range(32)]
        tree_ids = torch.tensor([[0, 1]], device=device).expand(32, -1).clone()
        node_hidden = torch.zeros((32, 2, 1), device=device)
        hidden_sources = [torch.zeros((32, 2, 1), device=device) for _ in range(3)]
        next_counts = torch.zeros(vocab)
        accept_count = 0
        torch.manual_seed(1234)
        for _ in range(rounds):
            accepted = module._accept_sampling_batch(
                algorithm, states, tree_ids, node_hidden, hidden_sources, 1.0
            )
            accept_count += sum(row[0] for row in accepted)
            next_counts.scatter_add_(
                0,
                torch.tensor([row[8][0] if row[0] else row[9] for row in accepted]),
                torch.ones(32),
            )
        observed = next_counts / next_counts.sum()
        expected = torch.tensor([0.10, 0.25, 0.30, 0.35])
        self.assertTrue(torch.allclose(observed, expected, atol=0.025), (observed, expected))
        self.assertAlmostEqual(accept_count / (rounds * 32), 0.25, delta=0.025)

    def test_deepest_rejection_then_accept_uses_fresh_bonus_logits(self):
        # Literal scalar walk: reject sibling 1, conditionally accept sibling 2,
        # then (at the deepest leaf) sample a fresh bonus at node 2 rather than
        # carrying root's residual distribution.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = torch.full((3, 8), -100.0, device=device)
        logits[0, 1], logits[0, 2] = 0.0, 0.0
        logits[2, 4] = 100.0
        algorithm = _algorithm(logits)
        states = [_state([[0, 1], [0, 2]], 3, device=device) for _ in range(32)]
        hidden = torch.tensor([[[0.0], [1.0], [2.0]]], device=device).expand(32, -1, -1).clone()
        uniforms = torch.tensor([[0.9, 0.1]], device=device).expand(32, -1).clone()
        with mock.patch.object(module.torch, "rand", return_value=uniforms):
            accepted = module._accept_sampling_batch(
                algorithm, states, torch.tensor([[0, 1, 2]], device=device).expand(32, -1).clone(), hidden,
                [hidden, hidden, hidden], 1.0,
            )
        self.assertTrue(all(row[0] == 1 and row[8] == [2] and row[9] == 4 for row in accepted))

    def test_ragged_failed_row_uses_global_to_local_probability_mapping(self):
        # At depth two only global rows 1 and 2 are eligible.  Row 2 rejects
        # its child, so the residual path must index its compact F=2 logits via
        # the global-row mapping rather than probabilities[global_row=2].
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = torch.full((4, 8), -100.0, device=device)
        logits[0, 1] = 100.0
        logits[1, 2] = 100.0
        logits[2, 2], logits[2, 3] = 0.0, torch.log(torch.tensor(3.0, device=device))
        logits[3, 4] = 100.0
        algorithm = _algorithm(logits)
        states = [
            _state([[0, 1]], 2, device=device),
            _state([[0, 1, 2]], 3, device=device),
            _state([[0, 1, 2]], 3, device=device),
        ]
        tree_ids = torch.tensor([[0, 1, 0], [0, 1, 2], [0, 1, 2]], device=device)
        hidden = torch.tensor([[[0.0], [1.0], [3.0]], [[0.0], [1.0], [3.0]], [[0.0], [2.0], [3.0]]], device=device)
        uniforms = [
            torch.zeros((3, 1), device=device),
            torch.tensor([[0.0], [0.0], [0.9]], device=device),
        ]
        with mock.patch.object(module.torch, "rand", side_effect=uniforms):
            accepted = module._accept_sampling_batch(
                algorithm, states, tree_ids, hidden, [hidden, hidden, hidden], 1.0
            )
        self.assertEqual(accepted[2][0], 1)
        self.assertEqual(accepted[2][8], [1])
        self.assertEqual(accepted[2][9], 3)

    def test_ragged_leaf_depths_are_independent(self):
        logits = torch.full((4, 8), -100.0)
        logits[0, 1], logits[1, 2], logits[2, 3] = 100.0, 100.0, 100.0
        algorithm = _algorithm(logits)
        states = [_state([[0, 1]], 2), _state([[0, 1, 2]], 3)]
        tree_ids = torch.tensor([[0, 1, 0], [0, 1, 2]])
        hidden = torch.tensor([[[0.0], [1.0], [3.0]], [[0.0], [1.0], [2.0]]])
        accepted = module._accept_sampling_batch(
            algorithm, states, tree_ids, hidden, [hidden, hidden, hidden], 1.0
        )
        self.assertEqual([row[0] for row in accepted], [1, 2])
        self.assertEqual([row[9] for row in accepted], [2, 3])

    def test_padded_retrieve_sentinel_never_becomes_a_vocab_index(self):
        # A ragged path can carry -1 past its leaf while another row/path fixes
        # the padded batch width.  It must be a non-proposal, not token 0.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = torch.full((2, 8), -100.0, device=device)
        logits[0, 1], logits[1, 2] = 100.0, 100.0
        algorithm = _algorithm(logits)
        states = [_state([[0, 1, -1], [0, -1, -1]], 2, device=device) for _ in range(32)]
        hidden = torch.tensor([[[0.0], [1.0]]], device=device).expand(32, -1, -1).clone()
        accepted = module._accept_sampling_batch(
            algorithm, states, torch.tensor([[0, 1]], device=device).expand(32, -1).clone(), hidden,
            [hidden, hidden, hidden], 1.0,
        )
        self.assertTrue(all(row[0] == 1 and row[8] == [1] and row[9] == 2 for row in accepted))

    def test_debug_gate_reports_out_of_range_retrieve(self):
        logits = torch.zeros((2, 8))
        algorithm = _algorithm(logits)
        state = _state([[0, 2]], 2)
        hidden = torch.tensor([[[0.0], [1.0]]])
        with mock.patch.dict(os.environ, {"SPECBLOCK_SAMPLING_DEBUG": "1"}):
            with self.assertRaisesRegex(RuntimeError, "out-of-range retrieve index"):
                module._accept_sampling_batch(
                    algorithm, [state], torch.tensor([[0, 1]]), hidden,
                    [hidden, hidden, hidden], 1.0,
                )

    def test_oov_candidate_is_excluded_not_clamped_or_indexed(self):
        # Positive OOV IDs are neither token-0 aliases nor vocabulary indices;
        # absent debug tracing they are simply not tree proposals.
        logits = torch.full((2, 8), -100.0)
        logits[0, 3] = 100.0
        algorithm = _algorithm(logits)
        state = _state([[0, 1]], 2)
        hidden = torch.tensor([[[0.0], [1.0]]])
        accepted = module._accept_sampling_batch(
            algorithm, [state], torch.tensor([[0, 8]]), hidden,
            [hidden, hidden, hidden], 1.0,
        )
        self.assertEqual(accepted[0][0], 0)
        self.assertEqual(accepted[0][9], 3)

    def test_debug_gate_reports_out_of_vocab_tree_token(self):
        logits = torch.zeros((2, 8))
        algorithm = _algorithm(logits)
        state = _state([[0, 1]], 2)
        hidden = torch.tensor([[[0.0], [1.0]]])
        with mock.patch.dict(os.environ, {"SPECBLOCK_SAMPLING_DEBUG": "1"}):
            with self.assertRaisesRegex(RuntimeError, "out-of-vocabulary token"):
                module._accept_sampling_batch(
                    algorithm, [state], torch.tensor([[0, 9]]), hidden,
                    [hidden, hidden, hidden], 1.0,
                )

    def test_initial_condition_fallback_handles_nonzero_chunk_offset(self):
        hidden_sources = [
            torch.arange(8, dtype=torch.float32).reshape(1, 4, 2) + offset
            for offset in (0, 10, 20)
        ]
        weight = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        algorithm = SimpleNamespace(
            draft_model=SimpleNamespace(
                input_layer=SimpleNamespace(
                    condition_proj=SimpleNamespace(weight=weight)
                )
            )
        )
        source_rows = torch.tensor([0])
        source_lengths = torch.tensor([4])
        starts = torch.tensor([2])
        lengths = torch.tensor([3])
        capabilities = SimpleNamespace(needs_ragged_condition_fallback=True)

        with mock.patch.object(module, "RUNTIME_CAPABILITIES", capabilities):
            actual = module._project_initial_ragged_condition(
                algorithm,
                hidden_sources,
                source_rows,
                source_lengths,
                starts,
                lengths,
                max_positions=3,
            )

        positions = torch.tensor([2, 3, 3])
        condition_input = torch.cat(
            [source[0, positions] for source in hidden_sources],
            dim=-1,
        ).unsqueeze(0)
        expected = torch.nn.functional.linear(condition_input, weight)
        self.assertTrue(torch.equal(actual, expected))

    def test_b16_b32_kv_compaction_preserves_surviving_rows(self):
        for batch_size in (16, 32):
            cache = module._DenseTargetCache(1, batch_size, 2, 4, 3)
            cache.prepare_prefill()
            prompt = torch.arange(batch_size * 2, dtype=torch.float32).reshape(batch_size, 1, 2, 1)
            cache.layers[0].update(prompt, prompt + 1000, {"cache_position": torch.arange(2)})
            cache.prepare_tree(3, [2] * batch_size, commit_reserve=2)
            staged = torch.arange(batch_size * 3, dtype=torch.float32).reshape(batch_size, 1, 3, 1) + 100
            cache.layers[0].update(staged, staged + 1000, {"cache_position": torch.arange(3)})
            # Model the committed root plus one accepted tree node, then shrink
            # non-contiguous active rows exactly as the decode loop does.
            layer = cache.layers[0]
            layer.keys[:, :, 2:4, :].copy_(layer.keys[:, :, cache.tree_start:cache.tree_start + 2, :])
            layer.values[:, :, 2:4, :].copy_(layer.values[:, :, cache.tree_start:cache.tree_start + 2, :])
            cache._committed_width = 4
            surviving = [1, batch_size - 1]
            expected_keys = layer.keys[surviving, :, :4, :].clone()
            expected_values = layer.values[surviving, :, :4, :].clone()
            cache.compact_rows(surviving)
            self.assertEqual(cache.active_batch_size, 2)
            self.assertTrue(torch.equal(layer.keys[:2, :, :4, :], expected_keys))
            self.assertTrue(torch.equal(layer.values[:2, :, :4, :], expected_values))

    def test_b32_gate_stays_depth_batched(self):
        # This runs the identical B32 acceptance path on CUDA when available;
        # CPU remains a deterministic fallback for local unit-test runners.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 32
        logits = torch.full((2, 8), -100.0, device=device)
        logits[0, 1] = 100.0
        logits[1, 2] = 100.0
        algorithm = _algorithm(logits)
        states = [_state([[0, 1]], 2, device=device) for _ in range(batch_size)]
        tree_ids = torch.tensor([[0, 1]], device=device).expand(batch_size, -1).clone()
        node_hidden = torch.tensor([[[0.0], [1.0]]], device=device).expand(batch_size, -1, -1).clone()
        hidden_sources = [torch.zeros((batch_size, 2, 1), device=device) for _ in range(3)]

        accepted = module._accept_sampling_batch(
            algorithm, states, tree_ids, node_hidden, hidden_sources, 1.0
        )
        self.assertEqual(len(accepted), batch_size)
        self.assertTrue(all(row[0] == 1 and row[9] == 2 for row in accepted))
        self.assertEqual(algorithm._batched_acceptance_gpu_calls, 1)
        self.assertNotIn("_accept_single_sampling(", inspect.getsource(module._accept_sampling_batch))


if __name__ == "__main__":
    unittest.main()
