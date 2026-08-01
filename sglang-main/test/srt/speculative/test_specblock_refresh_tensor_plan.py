import unittest

import torch

from sglang.srt.speculative.specblock_refresh_tensor_plan import (
    build_specblock_refresh_tensor_plan,
)


class TestSpecBlockRefreshTensorPlan(unittest.TestCase):
    def _inputs(self, device="cpu"):
        # Three request-major accepted chains, with draft accept lengths
        # [0, 1, 2] and therefore bonus-inclusive lengths [1, 2, 3].
        tokens = torch.tensor([10, 20, 21, 30, 31, 32], device=device)
        hidden = torch.arange(6, dtype=torch.float32, device=device).reshape(6, 1)
        accept_lengths = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        cross_positions = torch.tensor([5, 7, 11], dtype=torch.int64, device=device)
        existing_loc = torch.tensor(
            [[1, 2, 0, 0, 0], [3, 0, 0, 0, 0], [4, 5, 6, 0, 0]],
            dtype=torch.int64,
            device=device,
        )
        existing_counts = torch.tensor([2, 1, 3], dtype=torch.int64, device=device)
        allocated_loc = torch.tensor(
            [[100, 101, 0, 0, 0, 0], [200, 201, 202, 203, 0, 0], [300, 301, 302, 303, 304, 305]],
            dtype=torch.int64,
            device=device,
        )
        return (
            tokens,
            hidden,
            accept_lengths,
            cross_positions,
            existing_loc,
            existing_counts,
            allocated_loc,
        )

    def _build(self, device="cpu", output_cross_capacity=9):
        return build_specblock_refresh_tensor_plan(
            *self._inputs(device), K=2, output_cross_capacity=output_cross_capacity
        )

    def test_ragged_chains_token_shift_positions_and_cross_append(self):
        plan = self._build()

        self.assertTrue(torch.equal(plan.n_per_req, torch.tensor([1, 2, 3])))
        self.assertTrue(
            torch.equal(
                plan.chain_mask,
                torch.tensor([[True, False, False], [True, True, False], [True, True, True]]),
            )
        )
        self.assertTrue(
            torch.equal(plan.hidden[..., 0], torch.tensor([[0, 0, 0], [1, 2, 0], [3, 4, 5.0]]))
        )
        # The final ID is repeated: N=1 -> [last], N=2 -> [last,last],
        # N=3 -> [id_1,last,last].
        self.assertTrue(
            torch.equal(plan.tokens, torch.tensor([[10, 0, 0], [21, 21, 0], [31, 32, 32]]))
        )
        self.assertTrue(torch.equal(plan.start_positions, torch.tensor([5, 8, 12])))
        self.assertTrue(
            torch.equal(
                plan.pos_ids,
                torch.tensor([[6, 7, 7, 8, 8, 9], [9, 10, 10, 11, 11, 12], [13, 14, 14, 15, 15, 16]]),
            )
        )
        self.assertTrue(torch.equal(plan.new_cross_valid_slots, torch.tensor([2, 4, 6])))
        self.assertTrue(torch.equal(plan.cross_counts, torch.tensor([4, 5, 9])))
        self.assertFalse(plan.accept_overflow.any())
        self.assertFalse(plan.cross_overflow.any())
        self.assertTrue(
            torch.equal(
                plan.cross_loc,
                torch.tensor(
                    [[1, 2, 100, 101, 0, 0, 0, 0, 0], [3, 200, 201, 202, 203, 0, 0, 0, 0], [4, 5, 6, 300, 301, 302, 303, 304, 305]]
                ),
            )
        )

    def test_input_only_plan_matches_previous_per_request_reference(self):
        (
            flat_tokens,
            flat_hidden,
            accept_lengths,
            cross_positions,
            existing_loc,
            existing_counts,
            allocated_loc,
        ) = self._inputs()
        plan = build_specblock_refresh_tensor_plan(
            flat_tokens,
            flat_hidden,
            accept_lengths,
            cross_positions,
            existing_loc,
            existing_counts,
            allocated_loc,
            K=2,
        )

        # Literal CPU reference to the former _refresh_draft_state loop.
        lengths = [int(x) + 1 for x in accept_lengths.tolist()]
        capacity = allocated_loc.shape[1] // 2
        expected_hidden = torch.zeros((len(lengths), capacity, 1))
        expected_tokens = torch.zeros((len(lengths), capacity), dtype=torch.long)
        expected_pos = torch.empty((len(lengths), capacity * 2), dtype=torch.long)
        offset = 0
        for row, n in enumerate(lengths):
            chain_tokens = flat_tokens[offset : offset + n]
            expected_hidden[row, :n] = flat_hidden[offset : offset + n]
            if n == 1:
                expected_tokens[row, :n] = chain_tokens[-1:]
            elif n == 2:
                expected_tokens[row, :n] = torch.cat([chain_tokens[-1:], chain_tokens[-1:]])
            else:
                expected_tokens[row, :n] = torch.cat(
                    [chain_tokens[1:-1], chain_tokens[-1:], chain_tokens[-1:]]
                )
            start = int(cross_positions[row]) + int(accept_lengths[row] > 0)
            expected_pos[row] = torch.tensor(
                [start + 1 + token_step + slot for token_step in range(capacity) for slot in range(2)]
            )
            offset += n

        self.assertTrue(torch.equal(plan.hidden, expected_hidden))
        self.assertTrue(torch.equal(plan.tokens, expected_tokens))
        self.assertTrue(torch.equal(plan.pos_ids, expected_pos))
        self.assertTrue(torch.equal(plan.n_per_req, torch.tensor(lengths)))
        self.assertIsNone(plan.cross_loc)
        self.assertIsNone(plan.cross_overflow)

    def test_overflow_is_gpu_boolean_signal_not_silent_truncation(self):
        plan = self._build(output_cross_capacity=5)
        self.assertTrue(torch.equal(plan.cross_overflow, torch.tensor([False, False, True])))
        self.assertTrue(torch.equal(plan.cross_counts, torch.tensor([4, 5, 9])))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_cuda_input_plan_matches_cpu_reference(self):
        cpu_plan = build_specblock_refresh_tensor_plan(
            *self._inputs("cpu"), K=2
        )
        cuda_plan = build_specblock_refresh_tensor_plan(
            *self._inputs("cuda"), K=2
        )
        for name in (
            "hidden",
            "tokens",
            "pos_ids",
            "n_per_req",
            "chain_mask",
            "accept_overflow",
            "start_positions",
            "cross_counts",
            "new_cross_valid_slots",
        ):
            self.assertTrue(
                torch.equal(getattr(cuda_plan, name).cpu(), getattr(cpu_plan, name)),
                name,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_inputs_and_outputs_remain_on_device(self):
        plan = self._build("cuda")
        for tensor in (
            plan.hidden,
            plan.tokens,
            plan.pos_ids,
            plan.n_per_req,
            plan.chain_mask,
            plan.accept_overflow,
            plan.start_positions,
            plan.cross_loc,
            plan.cross_counts,
            plan.new_cross_valid_slots,
            plan.cross_overflow,
        ):
            self.assertEqual(tensor.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
