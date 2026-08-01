"""CPU invariants for the explicit SpecBlock strict-linear ablation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from benchmarks_hf.algorithms.specblock import SpecBlockAlgorithm
from benchmarks_hf.algorithms.specblock import SpecBlockAlgorithm


class SpecBlockStrictLinearTest(unittest.TestCase):
    def _algorithm(self, *, total_tokens: int = 8) -> SpecBlockAlgorithm:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        Path(tmp.name, "config.json").write_text(
            json.dumps({
                "diffspec_draft_token_num": 4,
                "rank_classes": 4,
                "num_ttt_blocks": 2,
            })
        )
        with mock.patch.dict(
            os.environ,
            {"SPECBLOCK_DYNAMIC_TREE": "0", "TREE_FIXED_N": "0"},
            clear=False,
        ):
            return SpecBlockAlgorithm(
                "meta-llama/Llama-3.1-8B-Instruct",
                tmp.name,
                device="cpu",
                total_tokens=total_tokens,
                max_blocks=2,
                beam_width=10,
                strict_linear=True,
            )

    def test_configuration_forces_one_unpruned_path(self):
        algorithm = self._algorithm()
        self.assertEqual(algorithm.beam_width, 1)
        self.assertEqual(algorithm.RANK_SLOT_TOPK, [1, 1, 1, 1])
        self.assertIsNone(algorithm._bfs_slot_topk)
        self.assertEqual(algorithm._strict_linear_nodes, 8)
        self.assertEqual(algorithm.total_tokens, 8)
        self.assertEqual(algorithm._adaptive_slot0, 0)
        self.assertEqual(algorithm._adaptive_all, 0)
        self.assertEqual(algorithm._iter_adapt_budget, 0)
        self.assertEqual(algorithm._cond_max_blocks, 0)
        self.assertEqual(algorithm._slot0_beam, 0)
        self.assertTrue(algorithm._tree_gpu_supported())

    def test_rank_give_up_cannot_truncate_linear_path(self):
        algorithm = self._algorithm()
        rank_preds = torch.tensor([[0, 3, 2, 1], [3, 3, 3, 3]])
        lengths, branches, give_up = algorithm._walk_rank_slots_batch(rank_preds)
        self.assertEqual(lengths.tolist(), [4, 4])
        self.assertEqual(branches.tolist(), [0, 0])
        self.assertEqual(give_up.tolist(), [False, False])

    def test_shift_pareto_defaults_cannot_restore_branching(self):
        with tempfile.TemporaryDirectory() as draft_dir:
            Path(draft_dir, "config.json").write_text(
                json.dumps({
                    "diffspec_draft_token_num": 4,
                    "rank_classes": 4,
                    "num_ttt_blocks": 2,
                })
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                algorithm = SpecBlockAlgorithm(
                    "meta-llama/Llama-3.1-8B-Instruct",
                    draft_dir,
                    device="cpu",
                    total_tokens=8,
                    max_blocks=2,
                    strict_linear=True,
                )
        self.assertEqual(algorithm.beam_width, 1)
        self.assertEqual(algorithm.RANK_SLOT_TOPK, [1, 1, 1, 1])
        self.assertEqual(algorithm._adaptive_slot0, 0)
        self.assertEqual(algorithm._adaptive_all, 0)
        self.assertTrue(algorithm._tree_gpu_supported())

    def test_dispatch_neutralizes_block1_rank_and_checks_shape(self):
        algorithm = self._algorithm()
        captured = {}

        def build(*args, **kwargs):
            captured["rank_logits"] = args[1].clone()
            width = 9
            return (
                torch.zeros((1, width), dtype=torch.long),
                torch.tril(torch.ones((1, 1, width, width), dtype=torch.float32)),
                torch.arange(width),
                torch.arange(width).unsqueeze(0),
                [],
                [],
                [],
            )

        algorithm._build_tree_from_block1 = build
        logits = torch.zeros((1, 4, 16))
        rank_logits = torch.zeros((1, 4, 4))
        rank_logits[..., 3] = 10
        result = algorithm._build_tree_from_block1_dispatch(
            logits,
            rank_logits,
            torch.zeros((1, 4, 8)),
            [],
            torch.zeros((1, 1), dtype=torch.long),
            [],
            0,
        )

        self.assertEqual(result[0].shape[-1], 9)
        self.assertEqual(captured["rank_logits"].argmax(dim=-1).tolist(), [[0, 0, 0, 0]])

    def test_budget_must_equal_full_linear_chain(self):
        with self.assertRaisesRegex(ValueError, "unpruned chain budget"):
            self._algorithm(total_tokens=7)

    def test_dynamic_tree_budget_is_rejected(self):
        with mock.patch.dict(os.environ, {"SPECBLOCK_DYNAMIC_TREE": "1"}, clear=False):
            with tempfile.TemporaryDirectory() as draft_dir:
                Path(draft_dir, "config.json").write_text(
                    json.dumps({
                        "diffspec_draft_token_num": 4,
                        "rank_classes": 4,
                        "num_ttt_blocks": 2,
                    })
                )
                with self.assertRaisesRegex(ValueError, "SPECBLOCK_DYNAMIC_TREE"):
                    SpecBlockAlgorithm(
                        "meta-llama/Llama-3.1-8B-Instruct",
                        draft_dir,
                        device="cpu",
                        total_tokens=8,
                        max_blocks=2,
                        strict_linear=True,
                    )

    def test_validator_accepts_padded_retrieve_capacity(self):
        algorithm = self._algorithm()
        width = 9
        result = (
            torch.zeros((1, width), dtype=torch.long),
            torch.tril(torch.ones((1, 1, width, width))),
            torch.arange(width),
            torch.cat((
                torch.arange(width),
                torch.full((12,), -1, dtype=torch.long),
            )).unsqueeze(0),
        )
        self.assertIs(algorithm._validate_strict_linear_tree(result), result)

    def test_validator_rejects_nonpadding_after_linear_path(self):
        algorithm = self._algorithm()
        width = 9
        result = (
            torch.zeros((1, width), dtype=torch.long),
            torch.tril(torch.ones((1, 1, width, width))),
            torch.arange(width),
            torch.cat((torch.arange(width), torch.tensor([4]))).unsqueeze(0),
        )
        with self.assertRaisesRegex(RuntimeError, "tree invariant failed"):
            algorithm._validate_strict_linear_tree(result)

    def test_validator_rejects_siblings(self):
        algorithm = self._algorithm()
        width = 9
        result = (
            torch.zeros((1, width), dtype=torch.long),
            torch.tril(torch.ones((1, 1, width, width))),
            torch.arange(width),
            torch.tensor([[0, 1, 2, 3, 4], [0, 5, 6, 7, 8]]),
        )
        with self.assertRaisesRegex(RuntimeError, "tree invariant failed"):
            algorithm._validate_strict_linear_tree(result)

    def test_validator_rejects_noncausal_mask(self):
        algorithm = self._algorithm()
        width = 9
        result = (
            torch.zeros((1, width), dtype=torch.long),
            torch.ones((1, 1, width, width)),
            torch.arange(width),
            torch.arange(width).unsqueeze(0),
        )
        with self.assertRaisesRegex(RuntimeError, "tree invariant failed"):
            algorithm._validate_strict_linear_tree(result)


if __name__ == "__main__":
    unittest.main()
