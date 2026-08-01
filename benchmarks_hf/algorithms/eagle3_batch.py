"""Request-level true batching for the official HuggingFace EAGLE3 decoder.

The target model executes exactly one padded prefill forward for the request
batch and one tree-verification forward for the active requests in every decode
round.  EAGLE's dynamic draft-tree construction remains request-local because
the official draft layer owns a scalar ``stable_kv`` cache; the cache is swapped
per request without serializing target forwards.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import torch

from .eagle_official.kv_cache import KVCache
from .eagle_official.utils import prepare_logits_processor


TargetCache = List[Tuple[torch.Tensor, torch.Tensor]]
TreeResult = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class _RequestState:
    conversation: List[Dict[str, str]]
    prompt_ids: torch.Tensor
    prompt_len: int
    input_ids: torch.Tensor
    target_cache: TargetCache | None = None
    draft_cache: Any = None
    tree: TreeResult | None = None
    iterations: int = 0
    generated_tokens: int = 0
    accept_lengths_raw: List[int] = field(default_factory=list)
    finished: bool = False


class _CudaPhaseTimer:
    """Collect CUDA event pairs and synchronize only after generation."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.events: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def start(self):
        if not self.enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, name: str, start) -> None:
        if not self.enabled:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events[name].append((start, end))

    def elapsed_seconds(self, name: str) -> float:
        if not self.enabled:
            return 0.0
        return sum(start.elapsed_time(end) for start, end in self.events.get(name, ())) / 1000.0


def _target_layers(model) -> Sequence[Any]:
    return model.base_model.model.layers


def _allocate_target_kv(
    model,
    batch_size: int,
    capacity: int,
    initial_length: int,
):
    """Allocate the official preallocated-KV interface for a real batch."""
    config = model.config
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    past_key_values = []
    for layer in _target_layers(model):
        device = layer.self_attn.q_proj.weight.device
        layer_kv = []
        for _ in range(2):
            data = torch.zeros(
                batch_size,
                config.num_key_value_heads,
                capacity,
                head_dim,
                dtype=model.base_model.dtype,
                device=device,
            )
            current_length = torch.tensor(initial_length, dtype=torch.long, device="cpu")
            layer_kv.append(KVCache(data, current_length))
        past_key_values.append(layer_kv)
    return past_key_values


def _split_prefill_cache(past_key_values, prompt_lengths: Sequence[int]) -> List[TargetCache]:
    result: List[TargetCache] = [[] for _ in prompt_lengths]
    for layer_kv in past_key_values:
        keys = layer_kv[0].data
        values = layer_kv[1].data
        for batch_idx, length in enumerate(prompt_lengths):
            # ``contiguous()`` may return an alias for the longest row.  These
            # state-owned caches must be independent before the padded prefill
            # allocation is released below.
            result[batch_idx].append((
                keys[batch_idx:batch_idx + 1, :, :length].clone(),
                values[batch_idx:batch_idx + 1, :, :length].clone(),
            ))
    return result


def _pack_target_caches(model, states: Sequence[_RequestState], tree_width: int):
    prefix_lengths = [int(state.input_ids.shape[1]) for state in states]
    prefix_width = max(prefix_lengths)
    packed = _allocate_target_kv(
        model,
        batch_size=len(states),
        capacity=prefix_width + tree_width,
        initial_length=prefix_width,
    )
    for layer_idx, layer_kv in enumerate(packed):
        for batch_idx, (state, length) in enumerate(zip(states, prefix_lengths)):
            keys, values = state.target_cache[layer_idx]
            layer_kv[0].data[batch_idx:batch_idx + 1, :, :length].copy_(keys)
            layer_kv[1].data[batch_idx:batch_idx + 1, :, :length].copy_(values)
    return packed, prefix_lengths, prefix_width


def _compact_target_cache(
    past_key_values,
    batch_idx: int,
    logical_prefix_len: int,
    physical_prefix_width: int,
    selected_tree_indices: torch.Tensor,
) -> TargetCache:
    compacted: TargetCache = []
    for layer_kv in past_key_values:
        keys = layer_kv[0].data[batch_idx:batch_idx + 1]
        values = layer_kv[1].data[batch_idx:batch_idx + 1]
        indices = selected_tree_indices.to(keys.device) + physical_prefix_width
        compacted.append((
            torch.cat(
                (keys[:, :, :logical_prefix_len], keys.index_select(2, indices)),
                dim=2,
            ).contiguous(),
            torch.cat(
                (values[:, :, :logical_prefix_len], values.index_select(2, indices)),
                dim=2,
            ).contiguous(),
        ))
    return compacted


def _hidden_3h(model, outputs) -> torch.Tensor:
    hidden_states = tuple(outputs.hidden_states)
    if len(hidden_states) != 3:
        raise RuntimeError(
            "EAGLE3 target must expose exactly three auxiliary hidden states; "
            f"got {len(hidden_states)}"
        )
    ea_device = model.ea_layer.lm_head.weight.device
    hidden_states = tuple(hidden.to(ea_device) for hidden in hidden_states)
    hidden_3h = torch.cat(hidden_states, dim=-1)
    expected = int(model.ea_layer.fc.in_features)
    if hidden_3h.shape[-1] != expected:
        raise RuntimeError(
            f"EAGLE3 hidden-state width mismatch: got {hidden_3h.shape[-1]}, "
            f"draft projection expects {expected}"
        )
    return hidden_3h


def _clone_tree(result: TreeResult, device: torch.device) -> TreeResult:
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = result
    return (
        draft_tokens.detach().clone().to(device),
        retrieve_indices.detach().clone().to(device),
        tree_mask.detach().clone().to(device=device, dtype=torch.bool),
        tree_position_ids.detach().clone().to(device),
    )


def _build_tree(
    algorithm,
    state: _RequestState,
    hidden_states: torch.Tensor,
    next_token: torch.Tensor,
    logits_processor=None,
) -> None:
    """Advance one request's official EAGLE draft cache and build its tree."""
    ea_layer = algorithm.model.ea_layer
    ea_layer.stable_kv = state.draft_cache
    draft_input_ids = torch.cat((state.input_ids, next_token), dim=1)
    result = ea_layer.topK_genrate(
        hidden_states,
        draft_input_ids,
        algorithm.model.base_model.lm_head,
        logits_processor=logits_processor,
    )
    state.draft_cache = ea_layer.stable_kv
    state.tree = _clone_tree(result, state.input_ids.device)


def _pack_trees(states: Sequence[_RequestState], pad_token_id: int):
    widths = [int(state.tree[0].shape[1]) for state in states]
    tree_width = max(widths)
    device = states[0].input_ids.device
    input_ids = torch.full(
        (len(states), tree_width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.zeros(
        (len(states), tree_width), dtype=torch.long, device=device
    )
    tree_mask = torch.zeros(
        (len(states), 1, tree_width, tree_width),
        dtype=torch.bool,
        device=device,
    )
    for batch_idx, (state, width) in enumerate(zip(states, widths)):
        draft_tokens, _retrieve, request_mask, request_positions = state.tree
        prefix_len = state.input_ids.shape[1]
        input_ids[batch_idx, :width] = draft_tokens[0]
        position_ids[batch_idx, :width] = request_positions[:width] + prefix_len
        tree_mask[batch_idx, :, :width, :width] = request_mask[:, :, :width, :width]
        for query_idx in range(width, tree_width):
            tree_mask[batch_idx, 0, query_idx, query_idx] = True
            position_ids[batch_idx, query_idx] = prefix_len
    return input_ids, position_ids, tree_mask, widths, tree_width


def _tree_attention_mask(
    prefix_lengths: Sequence[int],
    prefix_width: int,
    tree_width: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(
        (len(prefix_lengths), prefix_width + tree_width),
        dtype=torch.bool,
        device=device,
    )
    for batch_idx, length in enumerate(prefix_lengths):
        mask[batch_idx, :length] = True
        mask[batch_idx, prefix_width:] = True
    return mask


def _stop_token_ids(algorithm) -> set[int]:
    result: set[int] = set()
    eos = getattr(algorithm.tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, (list, tuple, set)):
            result.update(int(token) for token in eos)
        else:
            result.add(int(eos))
    if algorithm.model_family == "llama3":
        eot = algorithm.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None:
            result.add(int(eot))
    return result


def _pack_retrieve_indices(
    states: Sequence[_RequestState],
    device: torch.device,
):
    """Pack request-local leaf paths without copying vocabulary logits."""
    path_counts = [int(state.tree[1].shape[0]) for state in states]
    path_widths = [int(state.tree[1].shape[1]) for state in states]
    max_paths = max(path_counts)
    max_path_width = max(path_widths)
    retrieve = torch.full(
        (len(states), max_paths, max_path_width),
        -1,
        dtype=torch.long,
        device=device,
    )
    leaf_valid = torch.zeros(
        (len(states), max_paths),
        dtype=torch.bool,
        device=device,
    )
    for batch_idx, (state, path_count, path_width) in enumerate(
        zip(states, path_counts, path_widths)
    ):
        retrieve[batch_idx, :path_count, :path_width] = state.tree[1].to(device)
        leaf_valid[batch_idx, :path_count] = True
    return retrieve, leaf_valid


def _select_tree_paths(
    states: Sequence[_RequestState],
    tree_input_ids: torch.Tensor,
    tree_logits: torch.Tensor,
):
    """Select the official greedy leaf paths after one batched target argmax."""
    batch_size, tree_width = tree_input_ids.shape
    if tree_logits.shape[:2] != (batch_size, tree_width):
        raise ValueError("EAGLE3 target logits must match the packed tree")

    retrieve, leaf_valid = _pack_retrieve_indices(states, tree_input_ids.device)
    retrieve_valid = retrieve >= 0
    flat_retrieve = retrieve.clamp_min(0).reshape(batch_size, -1)

    target_pred = torch.argmax(tree_logits, dim=-1)
    path_pred = target_pred.gather(1, flat_retrieve).reshape_as(retrieve)
    path_tokens = tree_input_ids.gather(1, flat_retrieve).reshape_as(retrieve)
    path_tokens.masked_fill_(~retrieve_valid, -1)

    matches = (
        path_tokens[..., 1:].eq(path_pred[..., :-1])
        & retrieve_valid[..., 1:]
        & retrieve_valid[..., :-1]
    )
    path_lengths = matches.to(torch.int32).cumprod(dim=-1).sum(dim=-1)
    path_lengths.masked_fill_(~leaf_valid, -1)
    accept_lengths, best_leaves = path_lengths.max(dim=1)

    batch_rows = torch.arange(batch_size, device=tree_input_ids.device)
    selected_nodes = retrieve[batch_rows, best_leaves]
    selected_tokens = path_tokens[batch_rows, best_leaves]
    bonus_nodes = selected_nodes.gather(1, accept_lengths[:, None].to(torch.long))
    next_tokens = target_pred.gather(1, bonus_nodes)
    return accept_lengths, selected_nodes, selected_tokens, next_tokens


def _sample_tree_paths(
    states: Sequence[_RequestState],
    tree_input_ids: torch.Tensor,
    tree_logits: torch.Tensor,
    logits_processor,
):
    """Batched counterpart of official ``evaluate_posterior`` for T > 0.

    EAGLE's tree proposals are deterministic top-k candidates.  The official
    sampler traverses candidates in retrieve order, accepting a proposed token
    with its target probability and removing rejected candidates from the
    residual target distribution before drawing the next root.  This routine
    preserves that exact request-local decision order while operating every
    proposal-position and target draw as a batch tensor operation.  It never
    launches a per-request target forward.
    """
    batch_size, tree_width = tree_input_ids.shape
    retrieve, leaf_valid = _pack_retrieve_indices(states, tree_input_ids.device)
    retrieve_valid = retrieve >= 0
    safe_retrieve = retrieve.clamp_min(0)
    path_tokens = tree_input_ids.gather(
        1, safe_retrieve.reshape(batch_size, -1)
    ).reshape_as(retrieve)
    path_tokens.masked_fill_(~retrieve_valid, -1)

    # The official scalar rejection sampler is distributionally equivalent to
    # drawing once from the target at each prefix, then accepting precisely if
    # the draw occurs in the draft tree.  This formulation is exact, avoids a
    # B×V residual buffer per leaf, and vectorizes the whole active request set.
    selected_leaves = torch.zeros(
        batch_size, dtype=torch.long, device=tree_input_ids.device
    )
    accept_lengths = torch.zeros_like(selected_leaves)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=tree_input_ids.device)
    next_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=tree_input_ids.device)
    max_path_width = retrieve.shape[-1]
    batch_rows = torch.arange(batch_size, device=tree_input_ids.device)

    for position in range(1, max_path_width):
        selected_prefix = path_tokens[batch_rows, selected_leaves, :position]
        prefix_matches = (
            leaf_valid
            & retrieve_valid[:, :, :position].all(dim=-1)
            & path_tokens[:, :, :position].eq(selected_prefix[:, None, :]).all(dim=-1)
        )
        has_child = (
            prefix_matches & retrieve_valid[:, :, position]
        ).any(dim=1) & ~stopped
        # ``fi`` in the official loop is the first leaf sharing the accepted
        # prefix. Its tree-logit at position - 1 defines p(x | prefix).
        first_leaf = prefix_matches.to(torch.int64).argmax(dim=1)
        logit_nodes = safe_retrieve[batch_rows, first_leaf, position - 1]
        sampled = torch.multinomial(
            torch.softmax(logits_processor(None, tree_logits[batch_rows, logit_nodes]), dim=-1),
            1,
        )
        child_matches = (
            prefix_matches
            & retrieve_valid[:, :, position]
            & path_tokens[:, :, position].eq(sampled)
        )
        accepted = has_child & child_matches.any(dim=1)
        selected_leaves = torch.where(
            accepted, child_matches.to(torch.int64).argmax(dim=1), selected_leaves
        )
        accept_lengths += accepted.to(torch.long)
        rejected = has_child & ~accepted
        next_tokens = torch.where(rejected[:, None], sampled, next_tokens)
        stopped |= rejected

    selected_nodes = retrieve[batch_rows, selected_leaves]
    selected_tokens = path_tokens[batch_rows, selected_leaves]
    bonus_nodes = selected_nodes.gather(1, accept_lengths[:, None])
    bonus_probs = torch.softmax(
        logits_processor(None, tree_logits[batch_rows, bonus_nodes.squeeze(1)]), dim=-1
    )
    next_tokens = torch.where(stopped[:, None], next_tokens, torch.multinomial(bonus_probs, 1))
    return accept_lengths, selected_nodes, selected_tokens, next_tokens


def _accept_active_batch(
    algorithm,
    states: Sequence[_RequestState],
    tree_input_ids: torch.Tensor,
    tree_logits: torch.Tensor,
    hidden_3h: torch.Tensor,
    past_key_values,
    prefix_lengths: Sequence[int],
    physical_prefix_width: int,
    logits_processor=None,
):
    """Accept all active trees with one GPU decision and one metadata readback."""
    batch_size, tree_width = tree_input_ids.shape
    if hidden_3h.shape[:2] != (batch_size, tree_width):
        raise ValueError("EAGLE3 target hidden states must match the packed tree")

    (
        accept_lengths,
        selected_nodes,
        selected_tokens,
        next_tokens,
    ) = (
        _select_tree_paths(states, tree_input_ids, tree_logits)
        if logits_processor is None
        else _sample_tree_paths(states, tree_input_ids, tree_logits, logits_processor)
    )
    safe_selected_nodes = selected_nodes.clamp_min(0)
    selected_hidden = hidden_3h.gather(
        1,
        safe_selected_nodes[:, :, None].expand(-1, -1, hidden_3h.shape[-1]),
    )

    host_acceptance = torch.cat(
        (accept_lengths[:, None].to(torch.long), selected_tokens),
        dim=1,
    ).cpu()
    algorithm._batched_acceptance_gpu_calls = (
        getattr(algorithm, "_batched_acceptance_gpu_calls", 0) + 1
    )
    algorithm._batched_acceptance_readbacks = (
        getattr(algorithm, "_batched_acceptance_readbacks", 0) + 1
    )

    accepted_rows = []
    for batch_idx, state in enumerate(states):
        accept_length = int(host_acceptance[batch_idx, 0])
        selected_count = accept_length + 1
        selected_tree_indices = selected_nodes[batch_idx, :selected_count]
        state.target_cache = _compact_target_cache(
            past_key_values,
            batch_idx,
            prefix_lengths[batch_idx],
            physical_prefix_width,
            selected_tree_indices,
        )
        accepted_rows.append((
            accept_length,
            selected_tokens[batch_idx, :selected_count],
            host_acceptance[batch_idx, 1:1 + selected_count].tolist(),
            next_tokens[batch_idx:batch_idx + 1],
            selected_hidden[batch_idx:batch_idx + 1, :selected_count],
        ))
    return accepted_rows


def _empty_results(algorithm, conversations: Sequence[List[Dict[str, str]]]):
    results = []
    for _ in conversations:
        results.append({
            "output": "",
            "metrics": {
                "total_tokens": 0,
                "output_token_ids": [],
                "wall_time": 0.0,
                "tokens_per_second": 0.0,
                "accept_length": 0.0,
                "iterations": 0,
                "accept_lengths_raw": [],
                "prefill_time": 0.0,
                "draft_time": 0.0,
                "target_time": 0.0,
                "verify_time": 0.0,
                "other_time": 0.0,
            },
        })
    algorithm.last_batch_metrics = {
        "wall_time": 0.0,
        "prefill_time": 0.0,
        "draft_time": 0.0,
        "target_time": 0.0,
        "verify_time": 0.0,
        "iterations": 0,
        "active_sizes": [],
        "engine_batch_size": len(conversations),
        "acceptance_gpu_calls": 0,
        "acceptance_readbacks": 0,
    }
    return results


@torch.inference_mode()
def generate_conversations(
    algorithm,
    conversations: Sequence[List[Dict[str, str]]],
    max_new_tokens: int,
    temperature: float = 0.0,
    **kwargs,
) -> List[Dict]:
    """Generate a request batch with one target forward per active round."""
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    logits_processor = (
        prepare_logits_processor(temperature=temperature)
        if temperature > 1e-5
        else None
    )
    if not conversations:
        return _empty_results(algorithm, conversations)
    if max_new_tokens <= 0:
        return _empty_results(algorithm, conversations)

    acceptance_gpu_calls_before = getattr(
        algorithm, "_batched_acceptance_gpu_calls", 0
    )
    acceptance_readbacks_before = getattr(
        algorithm, "_batched_acceptance_readbacks", 0
    )
    cuda_timing = torch.cuda.is_available() and str(algorithm.device).startswith("cuda")
    if cuda_timing:
        torch.cuda.synchronize()
    wall_start = time.perf_counter()
    timer = _CudaPhaseTimer(cuda_timing)

    prompts = [
        algorithm.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            **algorithm._chat_template_kwargs(),
        )
        for conversation in conversations
    ]
    tokenizer = algorithm.tokenizer
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(algorithm.model.base_model.device)
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask.bool()
    prompt_lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    position_ids.masked_fill_(~attention_mask, 0)

    states: List[_RequestState] = []
    for batch_idx, (conversation, length) in enumerate(zip(conversations, prompt_lengths)):
        prompt_ids = input_ids[batch_idx:batch_idx + 1, :length].clone()
        state = _RequestState(
            conversation=list(conversation),
            prompt_ids=prompt_ids,
            prompt_len=length,
            input_ids=prompt_ids.clone(),
        )
        states.append(state)

    prefill_cache = _allocate_target_kv(
        algorithm.model,
        batch_size=len(states),
        capacity=input_ids.shape[1],
        initial_length=0,
    )
    algorithm.model.base_model.model.tree_mask = None
    prefill_start = timer.start()
    target_outputs, target_logits, _ = algorithm.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=prefill_cache,
        position_ids=position_ids,
        output_orig=True,
    )
    prefill_hidden = _hidden_3h(algorithm.model, target_outputs)
    split_caches = _split_prefill_cache(prefill_cache, prompt_lengths)
    timer.stop("prefill", prefill_start)

    draft_start = timer.start()
    batch_rows = torch.arange(len(states), device=target_logits.device)
    final_prompt_indices = torch.tensor(
        [state.prompt_len - 1 for state in states],
        dtype=torch.long,
        device=target_logits.device,
    )
    first_logits = target_logits[batch_rows, final_prompt_indices]
    if logits_processor is None:
        first_tokens = torch.argmax(first_logits, dim=-1, keepdim=True)
    else:
        first_tokens = torch.multinomial(
            torch.softmax(logits_processor(None, first_logits), dim=-1), 1
        )
    for batch_idx, state in enumerate(states):
        state.target_cache = split_caches[batch_idx]
        request_hidden = prefill_hidden[batch_idx:batch_idx + 1, :state.prompt_len]
        _build_tree(
            algorithm,
            state,
            request_hidden,
            first_tokens[batch_idx:batch_idx + 1],
            logits_processor,
        )
    # The loop-local view otherwise pins the full padded prefill hidden tensor.
    del request_hidden
    timer.stop("draft", draft_start)

    # Each request now owns its compact prompt cache and cloned tree.  None of
    # the padded prefill tensors participate in a later round.  Drop these
    # Python references before verification starts so the CUDA caching allocator
    # can reuse their storage; queued kernels retain their own tensor references,
    # so this introduces neither a synchronization nor a sampling change.
    del target_outputs, target_logits, prefill_hidden, prefill_cache, split_caches
    del input_ids, attention_mask, position_ids, encoded

    stop_ids = _stop_token_ids(algorithm)
    max_logical_length = 8192 - int(algorithm.model.ea_layer.total_tokens) - 10
    active_sizes: List[int] = []

    while True:
        active = [state for state in states if not state.finished]
        if not active:
            break
        active_sizes.append(len(active))

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id or 0
        tree_input_ids, tree_position_ids, tree_mask, widths, tree_width = _pack_trees(
            active, int(pad_token_id)
        )
        packed_cache, prefix_lengths, prefix_width = _pack_target_caches(
            algorithm.model, active, tree_width
        )
        verify_attention_mask = _tree_attention_mask(
            prefix_lengths,
            prefix_width,
            tree_width,
            tree_input_ids.device,
        )

        target_start = timer.start()
        target_model = algorithm.model.base_model.model
        target_model.tree_mask = tree_mask
        target_model.tree_prefix_lengths = torch.tensor(
            prefix_lengths,
            dtype=torch.int32,
            device=tree_input_ids.device,
        )
        target_model.tree_physical_prefix_width = prefix_width
        try:
            verify_outputs, verify_logits, _ = algorithm.model(
                input_ids=tree_input_ids,
                attention_mask=verify_attention_mask,
                past_key_values=packed_cache,
                position_ids=tree_position_ids,
                output_orig=True,
            )
        finally:
            target_model.tree_mask = None
            target_model.tree_prefix_lengths = None
            target_model.tree_physical_prefix_width = None
        verify_hidden = _hidden_3h(algorithm.model, verify_outputs)
        timer.stop("target", target_start)

        verify_start = timer.start()
        accepted_rows = _accept_active_batch(
            algorithm,
            active,
            tree_input_ids,
            verify_logits,
            verify_hidden,
            packed_cache,
            prefix_lengths,
            prefix_width,
            logits_processor=logits_processor,
        )
        timer.stop("verify", verify_start)

        next_rows = []
        for state, accepted in zip(active, accepted_rows):
            (
                accept_length,
                accepted_tokens,
                accepted_token_ids,
                next_token,
                selected_hidden,
            ) = accepted
            remaining = max_new_tokens - state.generated_tokens
            new_ids = accepted_token_ids[:remaining]
            stop_offset = next(
                (
                    idx
                    for idx, token in enumerate(new_ids)
                    if int(token) in stop_ids
                ),
                None,
            )
            if stop_offset is not None:
                new_ids = new_ids[:stop_offset + 1]

            emitted = accepted_tokens[:len(new_ids)].reshape(1, -1)
            state.input_ids = torch.cat(
                (state.input_ids, emitted.to(state.input_ids.device)),
                dim=1,
            )
            state.generated_tokens += len(new_ids)
            state.accept_lengths_raw.append(
                min(accept_length, max(len(new_ids) - 1, 0))
            )
            state.iterations += 1

            state.finished = (
                stop_offset is not None
                or state.generated_tokens >= max_new_tokens
                or state.input_ids.shape[1] > max_logical_length
            )
            if not state.finished:
                next_rows.append((state, selected_hidden, next_token))

        if next_rows:
            draft_start = timer.start()
            for state, selected_hidden, next_token in next_rows:
                _build_tree(
                    algorithm,
                    state,
                    selected_hidden,
                    next_token.to(state.input_ids.device),
                    logits_processor,
                )
            timer.stop("draft", draft_start)

    if cuda_timing:
        torch.cuda.synchronize()
    batch_wall_time = time.perf_counter() - wall_start
    prefill_time = timer.elapsed_seconds("prefill")
    draft_time = timer.elapsed_seconds("draft")
    target_time = timer.elapsed_seconds("target")
    verify_time = timer.elapsed_seconds("verify")
    other_time = max(
        0.0, batch_wall_time - prefill_time - draft_time - target_time - verify_time
    )

    algorithm.last_batch_metrics = {
        "wall_time": batch_wall_time,
        "prefill_time": prefill_time,
        "draft_time": draft_time,
        "target_time": target_time,
        "verify_time": verify_time,
        "iterations": len(active_sizes),
        "active_sizes": active_sizes,
        "engine_batch_size": len(states),
        "batch_wall_time": batch_wall_time,
        "batch_prefill_time": prefill_time,
        "batch_draft_time": draft_time,
        "batch_target_time": target_time,
        "batch_verify_time": verify_time,
        "batch_other_time": other_time,
        "batch_decode_rounds": len(active_sizes),
        "batch_size": len(states),
        "acceptance_gpu_calls": (
            getattr(algorithm, "_batched_acceptance_gpu_calls", 0)
            - acceptance_gpu_calls_before
        ),
        "acceptance_readbacks": (
            getattr(algorithm, "_batched_acceptance_readbacks", 0)
            - acceptance_readbacks_before
        ),
    }

    results = []
    for state in states:
        output_ids = state.input_ids[0, state.prompt_len:].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        num_tokens = len(output_ids)
        pos_acc = algorithm.compute_cumulative_position_accuracy(
            state.accept_lengths_raw, max_depth=algorithm.depth
        )
        results.append({
            "output": output_text,
            "metrics": {
                "total_tokens": num_tokens,
                "output_token_ids": output_ids,
                "wall_time": batch_wall_time,
                "tokens_per_second": num_tokens / batch_wall_time if batch_wall_time > 0 else 0.0,
                "accept_length": algorithm.compute_accept_length(state.accept_lengths_raw),
                "iterations": state.iterations,
                "accept_lengths_raw": list(state.accept_lengths_raw),
                "position_accuracy": pos_acc["accuracies"],
                "position_accuracy_pct": pos_acc["accuracies_pct"],
                "position_accuracy_formatted": pos_acc["formatted"],
                "prefill_time": prefill_time,
                "draft_time": draft_time,
                "target_time": target_time,
                "verify_time": verify_time,
                "other_time": other_time,
                "draft_pct": draft_time / batch_wall_time * 100 if batch_wall_time > 0 else 0.0,
                "target_pct": target_time / batch_wall_time * 100 if batch_wall_time > 0 else 0.0,
                "verify_pct": verify_time / batch_wall_time * 100 if batch_wall_time > 0 else 0.0,
            },
        })
    return results
