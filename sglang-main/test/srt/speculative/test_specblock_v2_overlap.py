import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.speculative.specblock_info import SpecBlockVerifyInput
from sglang.srt.speculative.specblock_worker import SpecBlockWorker
from sglang.srt.speculative.specblock_worker_v2 import (
    SpecBlockDraftWorker,
    SpecBlockWorkerV2,
    _pack_verified_ids_for_scheduler,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend


class _TargetWorker:
    def __init__(self):
        self.model_runner = types.SimpleNamespace(
            attn_backend=types.SimpleNamespace(num_draft_tokens=0),
        )
        self.calls = []
        self.result = None

    def forward_batch_generation(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.result is not None:
            return self.result
        return types.SimpleNamespace(
            logits_output=types.SimpleNamespace(hidden_states=torch.empty(0)),
            can_run_cuda_graph=True,
        )


class _Grammar:
    def __init__(self):
        self.accepted = []

    def accept_token(self, token_id):
        self.accepted.append(token_id)


class _Request:
    def __init__(self, stop_token=None, grammar=None):
        self.stop_token = stop_token
        self.grammar = grammar
        self.output_ids = []
        self.spec_verify_ct = 0
        self.spec_accepted_tokens = 0
        self._finished = False

    def check_finished(self):
        self._finished = self.output_ids[-1] == self.stop_token

    def finished(self):
        return self._finished


class TestSpecBlockV1CompactFill(unittest.TestCase):
    def test_compact_paths_preserve_stop_grammar_and_compaction(self):
        """V1 transfers accepted chains, not the tree-sized prediction buffer."""
        info = SpecBlockVerifyInput.__new__(SpecBlockVerifyInput)
        info.device = torch.device("cpu")
        # Accepted tree paths are intentionally non-contiguous and include a
        # tail after the first request's stop token.
        info.accept_index = torch.tensor(
            [[0, 5, 10, -1], [11, 12, 14, 18]], dtype=torch.int32
        )
        info.accept_length = torch.tensor([2, 3], dtype=torch.int32)
        info.predict = torch.tensor(
            [
                100,
                0,
                0,
                0,
                0,
                999,
                0,
                0,
                0,
                0,
                101,
                201,
                202,
                0,
                203,
                0,
                0,
                0,
                204,
                0,
            ],
            dtype=torch.int32,
        )
        grammar_0, grammar_1 = _Grammar(), _Grammar()
        req_0 = _Request(stop_token=999, grammar=grammar_0)
        req_1 = _Request(grammar=grammar_1)
        batch = types.SimpleNamespace(reqs=[req_0, req_1])
        logits = types.SimpleNamespace(
            next_token_logits=torch.arange(20, dtype=torch.float32).unsqueeze(1),
            hidden_states=torch.arange(40, dtype=torch.float32).reshape(20, 2),
        )

        accept_lens_cpu = info._fill_requests(batch, logits)

        # The stop token remains accepted, later tree nodes are discarded, and
        # grammar receives only non-terminal accepted tokens.
        self.assertEqual(req_0.output_ids, [100, 999])
        self.assertEqual(req_1.output_ids, [201, 202, 203, 204])
        self.assertEqual(grammar_0.accepted, [100])
        self.assertEqual(grammar_1.accepted, [201, 202, 203, 204])
        self.assertEqual([req_0.spec_verify_ct, req_1.spec_verify_ct], [1, 1])
        # Preserve V1's pre-truncation target-accept accounting.
        self.assertEqual(
            [req_0.spec_accepted_tokens, req_1.spec_accepted_tokens], [2, 3]
        )

        # Returned lengths are bonus-inclusive and match every downstream
        # compacted GPU tensor, including logits and captured hidden states.
        self.assertTrue(torch.equal(accept_lens_cpu, torch.tensor([2, 4])))
        expected_indices = torch.tensor(
            [0, 5, 11, 12, 14, 18], dtype=torch.int32
        )
        self.assertTrue(torch.equal(info.accept_index, expected_indices))
        self.assertTrue(
            torch.equal(
                info.verified_id,
                torch.tensor([100, 999, 201, 202, 203, 204]),
            )
        )
        self.assertTrue(
            torch.equal(logits.next_token_logits[:, 0], expected_indices.float())
        )
        expected_hidden = torch.arange(40, dtype=torch.float32).reshape(20, 2)[
            expected_indices
        ]
        self.assertTrue(torch.equal(logits.hidden_states, expected_hidden))

    def test_v1_fill_does_not_materialize_raw_tree_tensors(self):
        import inspect

        source = inspect.getsource(SpecBlockVerifyInput._fill_requests)
        self.assertNotIn("self.accept_index.tolist()", source)
        self.assertNotIn("self.predict.tolist()", source)


class TestSpecBlockV2Overlap(unittest.TestCase):
    def test_scheduler_pack_uses_filtered_paths_not_raw_tree_prefix(self):
        # Raw tree predictions can look like [3, EOS, junk, 4, 5, 18, ...]
        # while accept_index selects the non-contiguous path [0, 3, 4, 5].
        verified_ids = torch.tensor([3, 4, 5, 18, 11, 12, 18])
        accept_lens = torch.tensor([4, 3])

        packed = _pack_verified_ids_for_scheduler(
            verified_ids,
            accept_lens,
            stride=6,
        )

        self.assertTrue(
            torch.equal(
                packed,
                torch.tensor([3, 4, 5, 18, 0, 0, 11, 12, 18, 0, 0, 0]),
            )
        )

    def test_decode_refresh_forwards_gpu_accept_lengths(self):
        worker = SpecBlockDraftWorker.__new__(SpecBlockDraftWorker)
        calls = []

        def refresh(*args, **kwargs):
            calls.append((args, kwargs))

        worker._v1 = types.SimpleNamespace(_refresh_draft_state=refresh)
        worker._outer_worker_v2 = None
        accept_lengths_gpu = torch.tensor([2, 4], dtype=torch.int32)
        verify_info = types.SimpleNamespace(accept_length=accept_lengths_gpu)
        batch = types.SimpleNamespace(spec_info=verify_info)
        result = types.SimpleNamespace(
            logits_output=object(),
            next_token_ids=torch.tensor([10, 11, 12]),
            accept_length_per_req_cpu=[2, 4],
            next_draft_input=None,
        )

        worker._draft_extend_for_decode(batch, result)

        self.assertEqual(len(calls), 1)
        _, kwargs = calls[0]
        self.assertIs(kwargs["accept_lengths_gpu"], accept_lengths_gpu)
        self.assertIs(result.next_draft_input, verify_info)

    def test_prepared_forward_reuses_plan_stream_batch(self):
        """V2 must not rebuild target metadata on the main stream."""
        worker = SpecBlockWorker.__new__(SpecBlockWorker)
        target_worker = _TargetWorker()
        worker._target_worker = target_worker
        worker.page_size = 1

        spec_info = SpecBlockVerifyInput.__new__(SpecBlockVerifyInput)
        spec_info.draft_token_num = 7
        accepted_lens = torch.tensor([3], dtype=torch.int32)
        spec_info.verify = lambda *args, **kwargs: (
            args[1],
            torch.tensor([11, 12, 13], dtype=torch.int32),
            accepted_lens,
        )

        batch = types.SimpleNamespace(
            spec_info=spec_info,
            return_hidden_states=True,
            forward_mode=ForwardMode.DECODE,
            get_model_worker_batch=lambda **_: self.fail(
                "prepared V2 verification must not rebuild ModelWorkerBatch"
            ),
        )
        prepared = object()

        _, verified_ids, num_accepted, accept_lengths, can_run_graph = (
            worker._verify_and_accept(
                batch,
                skip_prepare=True,
                skip_free_cache=True,
                prepared_forward_batch=prepared,
                prepared_can_run_cuda_graph=True,
            )
        )

        self.assertTrue(torch.equal(verified_ids, torch.tensor([11, 12, 13])))
        self.assertEqual(num_accepted, 2)
        self.assertEqual(accept_lengths, [2])
        self.assertTrue(can_run_graph)
        self.assertEqual(len(target_worker.calls), 1)
        args, kwargs = target_worker.calls[0]
        self.assertEqual(args, ())
        self.assertIs(kwargs["forward_batch"], prepared)
        self.assertIsNone(kwargs["model_worker_batch"])
        self.assertTrue(kwargs["is_verify"])
        self.assertTrue(kwargs["skip_attn_backend_init"])

    def test_plan_preparation_sets_dynamic_width_before_metadata(self):
        """The plan stage builds one target ForwardBatch at tree width."""
        worker = SpecBlockWorkerV2.__new__(SpecBlockWorkerV2)
        attn_backend = types.SimpleNamespace(num_draft_tokens=0)
        graph_runner = types.SimpleNamespace(
            can_run=lambda forward_batch: False,
        )
        worker._target_worker = types.SimpleNamespace(
            model_runner=types.SimpleNamespace(
                attn_backend=attn_backend,
                graph_runner=graph_runner,
            )
        )

        batch = types.SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            capture_hidden_mode=None,
            return_hidden_states=True,
        )
        spec_info = types.SimpleNamespace(draft_token_num=13)
        forward_batch = object()

        with patch(
            "sglang.srt.speculative.specblock_worker_v2.ForwardBatch.init_new",
            return_value=forward_batch,
        ):
            prepared, can_run_graph = worker._prepare_verify_forward_batch(
                batch, spec_info
            )

        self.assertIs(prepared, forward_batch)
        self.assertFalse(can_run_graph)
        self.assertEqual(attn_backend.num_draft_tokens, 13)
        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
        self.assertEqual(batch.capture_hidden_mode, CaptureHiddenMode.FULL)
        self.assertFalse(batch.return_hidden_states)

    def test_idle_zero_row_and_padded_batches_bypass_draft_tree(self):
        """Scheduler padding must never enter the positive-BS draft graph."""
        for name, seq_lens in (
            ("zero-row", torch.empty((0,), dtype=torch.int32)),
            ("padded", torch.zeros((2,), dtype=torch.int32)),
        ):
            with self.subTest(name=name):
                worker = SpecBlockWorkerV2.__new__(SpecBlockWorkerV2)
                target_worker = _TargetWorker()
                worker._target_worker = target_worker
                sentinel = types.SimpleNamespace(source=name)
                target_worker.result = sentinel
                batch = types.SimpleNamespace(
                    forward_mode=ForwardMode.IDLE,
                    is_extend_in_batch=False,
                    spec_info=None,
                    seq_lens=seq_lens,
                )

                result = worker.forward_batch_generation(batch)

                self.assertIs(result, sentinel)
                self.assertIsNone(batch.spec_info)
                self.assertEqual(len(target_worker.calls), 1)
                args, kwargs = target_worker.calls[0]
                self.assertEqual(args, (batch,))
                self.assertEqual(kwargs, {})

    def test_graph_mask_repair_clears_stale_capacity_rows(self):
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.cuda_graph_custom_mask = torch.full((20,), 9, dtype=torch.uint8)
        backend.mask_indptr = torch.full((4,), -1, dtype=torch.int64)
        backend.use_specblock_tree_verify = True
        backend.num_draft_tokens = 2
        spec_info = types.SimpleNamespace(
            custom_mask=torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        )

        backend.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs=3
        )

        self.assertTrue(
            torch.equal(
                backend.cuda_graph_custom_mask,
                torch.tensor([1, 2, 3, 4] + [0] * 16, dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                backend.mask_indptr,
                torch.tensor([0, 4, 8, 12], dtype=torch.int64),
            )
        )


if __name__ == "__main__":
    unittest.main()
