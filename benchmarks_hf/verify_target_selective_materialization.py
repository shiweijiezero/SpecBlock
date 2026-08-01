#!/usr/bin/env python3
"""CPU gate for selective target hidden capture and LM-head projection."""

import importlib.util
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path(__file__).parent / "algorithms" / "specblock_batch.py"
SPEC = importlib.util.spec_from_file_location("specblock_batch_direct", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PROJECTION_PATH = Path(__file__).parent / "algorithms" / "hf_fused_projections.py"
PROJECTION_SPEC = importlib.util.spec_from_file_location(
    "hf_fused_projections_direct", PROJECTION_PATH
)
PROJECTION_MODULE = importlib.util.module_from_spec(PROJECTION_SPEC)
sys.modules[PROJECTION_SPEC.name] = PROJECTION_MODULE
PROJECTION_SPEC.loader.exec_module(PROJECTION_MODULE)


class _Layer(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)
        self.fail = False

    def forward(self, hidden, **_kwargs):
        if self.fail:
            raise RuntimeError("synthetic target failure")
        return (hidden + self.offset,)


class _Backbone(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            _Layer(index + 1) for index in range(num_layers)
        )
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
        **_kwargs,
    ):
        assert not output_attentions
        hidden = self.embed_tokens(input_ids)
        all_hidden = (hidden,) if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(hidden)[0]
            if output_hidden_states:
                all_hidden += (hidden,)
        hidden = self.norm(hidden)
        if output_hidden_states:
            all_hidden = all_hidden[:-1] + (hidden,)
        assert return_dict
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=all_hidden,
        )


class _Target(torch.nn.Module):
    def __init__(self, vocab_size=23, hidden_size=8, num_layers=7):
        super().__init__()
        self.model = _Backbone(vocab_size, hidden_size, num_layers)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, output_hidden_states=False):
        outputs = self.model(
            input_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=False,
            return_dict=True,
        )
        return SimpleNamespace(
            logits=self.lm_head(outputs.last_hidden_state),
            hidden_states=outputs.hidden_states,
        )


def _hook_count(target):
    return sum(len(layer._forward_hooks) for layer in target.model.layers)


def _reference_sampling(tree_input_ids, retrieve_indices, node_logits, temperature):
    candidate_source = torch.cat((
        tree_input_ids[0],
        tree_input_ids.new_full((1,), -1),
    ))
    candidate_indices = torch.where(
        retrieve_indices >= 0,
        retrieve_indices,
        retrieve_indices.new_full((), tree_input_ids.shape[1]),
    )
    candidates = candidate_source[candidate_indices]
    logits_source = torch.cat((
        node_logits,
        node_logits.new_zeros((1, node_logits.shape[-1])),
    ))
    logits = logits_source[candidate_indices]

    accepted_width = 1
    accepted_prefix = candidates[0, :1]
    best_candidate = 0
    adjusted = False
    sample_p = None
    for depth in range(1, candidates.shape[1]):
        if depth != accepted_width:
            break
        matching = (
            candidates[:, :accepted_width] == accepted_prefix
        ).all(dim=1)
        first_match = torch.nonzero(matching, as_tuple=True)[0][0]
        sample_p = torch.softmax(
            logits[first_match, depth - 1].float() / temperature,
            dim=0,
        )
        seen_tokens = set()
        adjusted = False
        for path_idx in range(candidates.shape[0]):
            if not bool(matching[path_idx]):
                continue
            token = int(candidates[path_idx, depth])
            if token < 0 or token in seen_tokens:
                continue
            seen_tokens.add(token)
            if random.random() <= float(sample_p[token]):
                accepted_prefix = torch.cat((
                    accepted_prefix,
                    candidates[path_idx, depth:depth + 1],
                ))
                accepted_width += 1
                best_candidate = path_idx
                break
            sample_p[token] = 0
            probability_sum = sample_p.sum()
            if float(probability_sum) > 0:
                sample_p = sample_p / probability_sum
            adjusted = True

    accept_length = accepted_width - 1
    if not adjusted or accepted_width == candidates.shape[1]:
        sample_p = torch.softmax(
            logits[best_candidate, accept_length].float() / temperature,
            dim=0,
        )
    if sample_p is None or float(sample_p.sum()) <= 0:
        sample_p = torch.softmax(
            logits[best_candidate, accept_length].float() / temperature,
            dim=0,
        )
    next_token = torch.multinomial(sample_p.unsqueeze(0), num_samples=1)
    selected = retrieve_indices[best_candidate, :accept_length + 1]
    return (
        accept_length,
        best_candidate,
        selected.tolist(),
        accepted_prefix[1:].tolist(),
        int(next_token[0, 0]),
    )


def main():
    specblock_source = (
        Path(__file__).parent / "algorithms" / "specblock.py"
    ).read_text(encoding="utf-8")
    assert "enable_hf_batch_invariant_ops" not in specblock_source
    projection_source = PROJECTION_PATH.read_text(encoding="utf-8")
    assert "matmul_persistent_split2" not in projection_source

    torch.manual_seed(0)
    packed_weights = tuple(torch.randn(size, 8) for size in (5, 7, 3))
    packed_projection = PROJECTION_MODULE._PackedProjection(
        packed_weights,
        fuse_min_rows=1,
    )
    packed_inputs = torch.randn(4, 6, 8)
    for index, weight in enumerate(packed_weights):
        actual = packed_projection.project(index, packed_inputs)
        expected = torch.nn.functional.linear(packed_inputs, weight)
        torch.testing.assert_close(actual, expected)

    cache_layer = MODULE._DenseTargetCacheLayer(max_cache_len=4)
    cache_layer.lazy_initialization(torch.zeros(1, 2, 2, 4))
    cache_layer.keys[:, :, :2, :].fill_(1)
    cache_layer.values[:, :, :2, :].fill_(2)
    cache_layer.resize(max_cache_len=8, preserve_width=2)
    assert torch.all(cache_layer.keys[:, :, :2, :] == 1)
    assert torch.all(cache_layer.values[:, :, :2, :] == 2)
    assert torch.count_nonzero(cache_layer.keys[:, :, 2:, :]) == 0
    assert torch.count_nonzero(cache_layer.values[:, :, 2:, :]) == 0

    cache_layer.attention_width = 6
    cache_layer.committed_width = 2
    cache_layer.write_start = 4
    cache_layer.keys[:, :, 2:4, :].fill_(torch.finfo(torch.float32).max)
    cache_layer.values[:, :, 2:4, :].fill_(torch.finfo(torch.float32).max)
    staged_keys = torch.randn(1, 2, 2, 4)
    staged_values = torch.randn(1, 2, 2, 4)
    visible_keys, visible_values = cache_layer.update(
        staged_keys,
        staged_values,
        {"cache_position": torch.arange(2)},
    )
    assert torch.count_nonzero(visible_keys[:, :, 2:4, :]) == 0
    assert torch.count_nonzero(visible_values[:, :, 2:4, :]) == 0
    assert torch.equal(visible_keys[:, :, 4:6, :], staged_keys)
    assert torch.equal(visible_values[:, :, 4:6, :], staged_values)

    batch_cache_layer = MODULE._DenseTargetCacheLayer(max_cache_len=8)
    batch_cache_layer.lazy_initialization(torch.zeros(2, 2, 2, 4))
    batch_cache_layer.attention_width = 8
    batch_cache_layer.committed_width = 4
    batch_cache_layer.write_start = 6
    batch_cache_layer.gap_start = 2
    positions = torch.arange(2, 6)
    row_prefixes = torch.tensor([2, 4])
    batch_cache_layer.gap_mask = (
        positions.unsqueeze(0) >= row_prefixes.unsqueeze(1)
    )
    extreme = torch.finfo(torch.float32).max
    batch_cache_layer.keys[:, :, 2:6, :].fill_(extreme)
    batch_cache_layer.values[:, :, 2:6, :].fill_(extreme)
    batch_cache_layer.keys[1, :, 2:4, :].fill_(3)
    batch_cache_layer.values[1, :, 2:4, :].fill_(4)
    batch_staged_keys = torch.randn(2, 2, 2, 4)
    batch_staged_values = torch.randn(2, 2, 2, 4)
    batch_visible_keys, batch_visible_values = batch_cache_layer.update(
        batch_staged_keys,
        batch_staged_values,
        {"cache_position": torch.arange(2)},
    )
    assert torch.count_nonzero(batch_visible_keys[0, :, 2:6, :]) == 0
    assert torch.count_nonzero(batch_visible_values[0, :, 2:6, :]) == 0
    assert torch.all(batch_visible_keys[1, :, 2:4, :] == 3)
    assert torch.all(batch_visible_values[1, :, 2:4, :] == 4)
    assert torch.count_nonzero(batch_visible_keys[1, :, 4:6, :]) == 0
    assert torch.count_nonzero(batch_visible_values[1, :, 4:6, :]) == 0
    assert torch.equal(batch_visible_keys[:, :, 6:8, :], batch_staged_keys)
    assert torch.equal(batch_visible_values[:, :, 6:8, :], batch_staged_values)

    target_cache = MODULE._DenseTargetCache(
        num_layers=1,
        max_batch_size=1,
        prompt_width=16,
        max_new_tokens=64,
        max_tree_width=91,
    )
    assert target_cache.capacity == 16 + 64 + 2 * 91

    target = _Target()
    algorithm = SimpleNamespace(
        target_model=target,
        hidden_layer_indices=[2, 4, 6],
    )
    input_ids = torch.randint(0, 23, (3, 7))

    full = target(input_ids, output_hidden_states=True)
    last_hidden, captured = MODULE._run_target_backbone(
        algorithm,
        input_ids=input_ids,
        use_cache=True,
    )
    assert _hook_count(target) == 0
    assert torch.equal(last_hidden, target.model(input_ids).last_hidden_state)
    for actual, index in zip(captured, algorithm.hidden_layer_indices):
        assert torch.equal(actual, full.hidden_states[index])

    prompt_lengths = [7, 5, 3]
    rows = torch.arange(len(prompt_lengths))
    ends = torch.tensor(prompt_lengths) - 1
    selective_first = target.lm_head(last_hidden[rows, ends]).argmax(dim=-1)
    full_first = full.logits[rows, ends].argmax(dim=-1)
    assert torch.equal(selective_first, full_first)

    tree_widths = [7, 4, 2]
    full_tree_logits = target.lm_head(last_hidden)
    valid = torch.arange(last_hidden.shape[1])[None, :] < torch.tensor(
        tree_widths
    )[:, None]
    for accept_topk in (1, 2, 3):
        os.environ["ACCEPT_TOPK"] = str(accept_topk)
        node_argmax, node_topk, sampling_logits, projected_rows = (
            MODULE._project_tree_decisions(
                algorithm, last_hidden, tree_widths
            )
        )
        assert sampling_logits is None
        assert projected_rows == sum(tree_widths)
        assert torch.equal(
            node_argmax[valid], full_tree_logits.argmax(dim=-1)[valid]
        )
        assert torch.equal(node_argmax[~valid], torch.zeros_like(node_argmax[~valid]))
        if accept_topk == 1:
            assert node_topk is None
        else:
            expected_topk = torch.topk(
                full_tree_logits, accept_topk, dim=-1
            ).indices
            assert torch.equal(node_topk[valid], expected_topk[valid])
            assert torch.equal(
                node_topk[~valid], torch.zeros_like(node_topk[~valid])
            )

    _, _, sampling_logits, projected_rows = MODULE._project_tree_decisions(
        algorithm,
        last_hidden[:1],
        [last_hidden.shape[1]],
        return_sampling_logits=True,
    )
    assert projected_rows == last_hidden.shape[1]
    torch.testing.assert_close(
        sampling_logits,
        full_tree_logits[0],
        rtol=1e-6,
        atol=1e-6,
    )

    target.model.layers[3].fail = True
    try:
        MODULE._run_target_backbone(algorithm, input_ids=input_ids)
    except RuntimeError as error:
        assert str(error) == "synthetic target failure"
    else:
        raise AssertionError("synthetic target failure was not propagated")
    assert _hook_count(target) == 0

    tree_input_ids = torch.tensor([[10, 20, 30]])
    retrieve_indices = torch.tensor([
        [0, 1, -1, -1],
        [0, 2, 2, -1],
    ])
    state = SimpleNamespace(
        tree=(tree_input_ids, None, None, retrieve_indices, [], [], None)
    )
    node_logits = torch.full((3, 64), float("-inf"))
    node_logits[0, 20] = 0
    node_logits[1, 40] = 0
    node_logits[2, 41] = 0
    hidden_sources = tuple(
        torch.arange(6, dtype=torch.float32).reshape(1, 3, 2) + offset
        for offset in (0, 10, 20)
    )
    sampling_algorithm = SimpleNamespace(
        target_model=SimpleNamespace(lm_head=torch.nn.Identity())
    )
    original_random = MODULE.random.random
    MODULE.random.random = lambda: 0.0
    try:
        sampled = MODULE._accept_single_sampling(
            sampling_algorithm,
            [state],
            tree_input_ids,
            node_logits.unsqueeze(0),
            hidden_sources,
            temperature=1.0,
        )[0]
    finally:
        MODULE.random.random = original_random
    assert sampled[0] == 1
    assert sampled[1].tolist() == [20]
    assert sampled[2].tolist() == [[40]]
    assert sampled[7].tolist() == [0, 1]
    assert sampled[8] == [20]
    assert sampled[9] == 40
    assert sampled[6] is None
    assert sampling_algorithm._sampling_lm_head_rows == 2

    parity_tree_ids = torch.tensor([[10, 20, 21, 30, 31, 40, 41, 42]])
    parity_retrieve = torch.tensor([
        [0, 1, 3, 5],
        [0, 2, 4, 6],
        [0, 2, 4, 7],
        [0, 2, -1, -1],
    ])
    parity_state = SimpleNamespace(
        tree=(parity_tree_ids, None, None, parity_retrieve, [], [], None)
    )
    generator = torch.Generator().manual_seed(2026)
    parity_logits = torch.randn(8, 64, generator=generator)
    parity_hidden = tuple(
        torch.randn(1, 8, 3, generator=generator) + offset
        for offset in (0, 10, 20)
    )
    for seed in range(32):
        random.seed(seed)
        torch.manual_seed(seed)
        expected = _reference_sampling(
            parity_tree_ids,
            parity_retrieve,
            parity_logits,
            temperature=1.0,
        )
        random.seed(seed)
        torch.manual_seed(seed)
        actual = MODULE._accept_single_sampling(
            SimpleNamespace(
                target_model=SimpleNamespace(lm_head=torch.nn.Identity())
            ),
            [parity_state],
            parity_tree_ids,
            parity_logits.unsqueeze(0),
            parity_hidden,
            temperature=1.0,
        )[0]
        observed = (
            actual[0],
            actual[5],
            actual[7].tolist(),
            actual[8],
            actual[9],
        )
        assert observed == expected, (seed, observed, expected)

    print("[PASS] selective target projection and B1 sampling acceptance")


if __name__ == "__main__":
    main()
