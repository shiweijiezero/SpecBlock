import unittest

import torch

from sglang.srt.speculative.specblock_refresh_cuda_graph_runner import (
    _ACCEPT_BUCKETS,
    SpecBlockRefreshCudaGraphRunner,
)


class TestSpecBlockRefreshCudaGraphRunner(unittest.TestCase):
    def test_copy_inputs_updates_active_rows_and_clears_padding(self):
        device = torch.device("cpu")
        runner = SpecBlockRefreshCudaGraphRunner.__new__(
            SpecBlockRefreshCudaGraphRunner
        )
        runner.K = 2

        bcap = 2
        active_bs = 1
        # Logical N=3 is carried in a capacity-4 graph. The final column
        # must remain inert: n_per_req gates the forward and the worker's
        # valid-prefix scatter uses N * K rather than the capacity.
        accept_bucket = 4
        logical_n = 3
        cross_bucket = 4
        hidden_size = 6
        in_buf = {
            "hidden": torch.ones(
                (bcap, accept_bucket, hidden_size), device=device
            ),
            "tokens": torch.ones(
                (bcap, accept_bucket), dtype=torch.long, device=device
            ),
            "pos_ids": torch.ones(
                (bcap, accept_bucket * runner.K),
                dtype=torch.long,
                device=device,
            ),
            "cross_loc": torch.ones(
                (bcap, cross_bucket), dtype=torch.long, device=device
            ),
            "cross_mask": torch.ones(
                (bcap, cross_bucket), dtype=torch.bool, device=device
            ),
            "n_per_req": torch.full(
                (bcap,), 7, dtype=torch.long, device=device
            ),
        }
        hidden = torch.randn(
            (active_bs, accept_bucket, hidden_size), device=device
        )
        tokens = torch.randint(
            0, 100, (active_bs, accept_bucket), device=device
        )
        pos_ids = torch.randint(
            0,
            100,
            (active_bs, accept_bucket * runner.K),
            device=device,
        )
        cross_loc = torch.randint(
            0, 100, (active_bs, cross_bucket), device=device
        )
        hidden[:, logical_n:].zero_()
        tokens[:, logical_n:].zero_()
        pos_ids[:, logical_n * runner.K:].zero_()
        cross_mask = torch.tensor(
            [[True, True, False, False]], device=device
        )
        n_per_req = torch.tensor([logical_n], dtype=torch.long, device=device)

        runner._copy_inputs(
            in_buf,
            active_bs,
            hidden,
            tokens,
            pos_ids,
            cross_loc,
            cross_mask,
            n_per_req,
        )
        self.assertTrue(torch.equal(in_buf["hidden"][:active_bs], hidden))
        self.assertTrue(torch.equal(in_buf["tokens"][:active_bs], tokens))
        self.assertTrue(torch.equal(in_buf["pos_ids"][:active_bs], pos_ids))
        self.assertTrue(torch.equal(in_buf["cross_loc"][:active_bs], cross_loc))
        self.assertTrue(
            torch.equal(in_buf["cross_mask"][:active_bs], cross_mask)
        )
        self.assertTrue(
            torch.equal(in_buf["n_per_req"][:active_bs], n_per_req)
        )
        self.assertEqual(
            torch.count_nonzero(in_buf["hidden"][0, logical_n:]).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(in_buf["tokens"][0, logical_n:]).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(
                in_buf["pos_ids"][0, logical_n * runner.K:]
            ).item(),
            0,
        )
        self.assertFalse(in_buf["cross_mask"][0, logical_n:].any().item())

        self.assertEqual(torch.count_nonzero(in_buf["hidden"][1]).item(), 0)
        self.assertEqual(torch.count_nonzero(in_buf["tokens"][1]).item(), 0)
        self.assertEqual(torch.count_nonzero(in_buf["pos_ids"][1]).item(), 0)
        self.assertEqual(torch.count_nonzero(in_buf["cross_loc"][1]).item(), 0)
        self.assertEqual(torch.count_nonzero(in_buf["cross_mask"][1]).item(), 0)
        self.assertEqual(in_buf["n_per_req"][1].item(), 1)

    def test_copy_inputs_full_batch_overwrites_without_prefill_clear(self):
        runner = SpecBlockRefreshCudaGraphRunner.__new__(
            SpecBlockRefreshCudaGraphRunner
        )
        runner.K = 2
        bcap = active_bs = 2
        accept_bucket = 2
        cross_bucket = 3
        hidden_size = 4
        in_buf = {
            "hidden": torch.full(
                (bcap, accept_bucket, hidden_size), 9.0
            ),
            "tokens": torch.full(
                (bcap, accept_bucket), 9, dtype=torch.long
            ),
            "pos_ids": torch.full(
                (bcap, accept_bucket * runner.K), 9, dtype=torch.long
            ),
            "cross_loc": torch.full(
                (bcap, cross_bucket), 9, dtype=torch.long
            ),
            "cross_mask": torch.ones(
                (bcap, cross_bucket), dtype=torch.bool
            ),
            "n_per_req": torch.full((bcap,), 9, dtype=torch.long),
        }
        hidden = torch.randn(bcap, accept_bucket, hidden_size)
        tokens = torch.tensor([[1, 0], [2, 3]])
        pos_ids = torch.tensor([[4, 5, 0, 0], [6, 7, 8, 9]])
        cross_loc = torch.tensor([[10, 0, 0], [11, 12, 0]])
        cross_mask = torch.tensor(
            [[True, False, False], [True, True, False]]
        )
        n_per_req = torch.tensor([1, 2])

        runner._copy_inputs(
            in_buf,
            active_bs,
            hidden,
            tokens,
            pos_ids,
            cross_loc,
            cross_mask,
            n_per_req,
        )

        self.assertTrue(torch.equal(in_buf["hidden"], hidden))
        self.assertTrue(torch.equal(in_buf["tokens"], tokens))
        self.assertTrue(torch.equal(in_buf["pos_ids"], pos_ids))
        self.assertTrue(torch.equal(in_buf["cross_loc"], cross_loc))
        self.assertTrue(torch.equal(in_buf["cross_mask"], cross_mask))
        self.assertTrue(torch.equal(in_buf["n_per_req"], n_per_req))

    def test_precapture_covers_only_requested_cross_buckets(self):
        runner = SpecBlockRefreshCudaGraphRunner.__new__(
            SpecBlockRefreshCudaGraphRunner
        )
        runner.capture_bs = [1]
        runner.cross_buckets = (32, 64, 128)
        runner.accept_buckets = (1, 2, 4)
        captured = []
        runner.capture_one = lambda *key: captured.append(key)

        runner.precapture_up_to(64)

        self.assertEqual(
            captured,
            [
                (1, 32, 1), (1, 32, 2), (1, 32, 4),
                (1, 64, 1), (1, 64, 2), (1, 64, 4),
            ],
        )

    def test_accept_lengths_share_power_of_two_capture_capacity(self):
        runner = SpecBlockRefreshCudaGraphRunner.__new__(
            SpecBlockRefreshCudaGraphRunner
        )
        runner.capture_bs = [1, 2, 4, 8, 16]
        runner.cross_buckets = (32, 64, 128)
        runner.accept_buckets = _ACCEPT_BUCKETS
        self.assertEqual(_ACCEPT_BUCKETS, (1, 2, 4, 8, 16))

        expected_capacity = {
            1: 1,
            2: 2,
            3: 4,
            4: 4,
            5: 8,
            6: 8,
            7: 8,
            8: 8,
            9: 16,
        }
        keys = {
            runner.resolve_buckets(1, 32, logical_n)
            for logical_n in expected_capacity
        }
        self.assertEqual(
            keys,
            {(1, 32, capacity) for capacity in set(expected_capacity.values())},
        )
        for logical_n, capacity in expected_capacity.items():
            self.assertEqual(
                runner.resolve_buckets(1, 32, logical_n),
                (1, 32, capacity),
            )
        self.assertIsNone(runner.resolve_buckets(1, 32, 17))

        # A capture indexed by capacity 8 is reusable for every logical length
        # in 5..8; no separate graph key can be introduced for those lengths.
        key_for_five = runner.resolve_buckets(1, 32, 5)
        self.assertEqual(key_for_five, runner.resolve_buckets(1, 32, 8))

        # capture_one exits before accessing the worker when its canonical key
        # already exists. This models a second lazy request in the same bucket.
        runner.graphs = {key_for_five: object()}
        runner.capture_one(*key_for_five)
        self.assertEqual(set(runner.graphs), {key_for_five})

if __name__ == "__main__":
    unittest.main()
