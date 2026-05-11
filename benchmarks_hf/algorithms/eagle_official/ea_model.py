import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
# Qwen3 requires LossKwargs which was removed in transformers 4.54+
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

        # Pre-allocated CUDA events pool
        self._events_pool = None
        self._events_pool_size = 0

    def _ensure_events_pool(self, size: int):
        """Ensure we have enough pre-allocated CUDA events."""
        if self._events_pool is None or self._events_pool_size < size:
            self._events_pool_size = max(size, 512)
            self._events_pool = {
                'prefill_start': torch.cuda.Event(enable_timing=True),
                'prefill_end': torch.cuda.Event(enable_timing=True),
                'target_start': [torch.cuda.Event(enable_timing=True) for _ in range(self._events_pool_size)],
                'target_end': [torch.cuda.Event(enable_timing=True) for _ in range(self._events_pool_size)],
                'draft_start': [torch.cuda.Event(enable_timing=True) for _ in range(self._events_pool_size)],
                'draft_end': [torch.cuda.Event(enable_timing=True) for _ in range(self._events_pool_size)],
            }

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        # Check if existing cache is large enough, reallocate if needed
        need_realloc = not hasattr(self, "past_key_values")
        if hasattr(self, "past_key_values_data") and len(self.past_key_values_data) > 0:
            current_cache_size = self.past_key_values_data[0].shape[3]
            if max_length > current_cache_size:
                need_realloc = True

        if not need_realloc:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def eagenerate_with_cuda_events(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            is_llama3=False,
    ):
        """EAGLE generate with CUDA events timing (no sync overhead during generation).

        Uses CUDA events for non-blocking timing that doesn't interrupt GPU computation.

        Returns:
            input_ids: Generated token ids
            new_token: Total number of new tokens generated
            idx: Number of iterations
            timing: Dict with timing breakdown (in seconds)
            accept_lengths: List of accept lengths per iteration
        """
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize CUDA events pool
        max_iterations = max_new_tokens + 10
        self._ensure_events_pool(max_iterations)

        accept_lengths = []

        # Initialize the past key and value states
        need_realloc = not hasattr(self, "past_key_values")
        if hasattr(self, "past_key_values_data") and len(self.past_key_values_data) > 0:
            current_cache_size = self.past_key_values_data[0].shape[3]
            if max_length > current_cache_size:
                need_realloc = True

        if not need_realloc:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        # Prefill timing
        self._events_pool['prefill_start'].record()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        self._events_pool['prefill_end'].record()

        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        iterations = 0
        all_depth_forward_events = []  # Collect per-depth events across all iterations

        for idx in range(max_length):
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            # Target model forward timing
            self._events_pool['target_start'][iterations].record()

            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            self._events_pool['target_end'][iterations].record()

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # Convert tensor to int if needed
            if isinstance(accept_length, torch.Tensor):
                accept_lengths.append(accept_length.item())
            else:
                accept_lengths.append(int(accept_length))

            # Draft model forward timing (inside update_inference_inputs)
            self._events_pool['draft_start'][iterations].record()

            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            self._events_pool['draft_end'][iterations].record()

            # Collect per-depth events from this iteration's topK_genrate
            if hasattr(self.ea_layer, '_depth_forward_events'):
                all_depth_forward_events.extend(self.ea_layer._depth_forward_events)

            iterations += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        # Single sync at end to compute all event timings
        torch.cuda.synchronize()

        # Compute timing from CUDA events
        timing = {
            "prefill_time": self._events_pool['prefill_start'].elapsed_time(self._events_pool['prefill_end']) / 1000.0,
            "target_time": sum(
                self._events_pool['target_start'][i].elapsed_time(self._events_pool['target_end'][i])
                for i in range(iterations)
            ) / 1000.0 if iterations > 0 else 0.0,
            "draft_time": sum(
                self._events_pool['draft_start'][i].elapsed_time(self._events_pool['draft_end'][i])
                for i in range(iterations)
            ) / 1000.0 if iterations > 0 else 0.0,
        }

        # Collect per-depth draft forward times from all iterations
        if all_depth_forward_events:
            from collections import defaultdict
            depth_times = defaultdict(float)
            for depth, s, e in all_depth_forward_events:
                depth_times[depth] += s.elapsed_time(e) / 1000.0
            timing["draft_forward_times"] = dict(depth_times)

        return input_ids, new_token, iterations, timing, accept_lengths

    def eagenerate_streaming(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            is_llama3=False,
    ):
        """EAGLE generate with streaming output for UI visualization.

        Yields iteration-by-iteration results with accepted tokens info.
        """
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        accept_lengths = []

        # Initialize KV cache
        need_realloc = not hasattr(self, "past_key_values")
        if hasattr(self, "past_key_values_data") and len(self.past_key_values_data) > 0:
            current_cache_size = self.past_key_values_data[0].shape[3]
            if max_length > current_cache_size:
                need_realloc = True

        if not need_realloc:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        # Prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )

        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        iterations = 0
        start_time = time.time()
        all_depth_forward_events = []  # Collect per-depth events across all iterations

        # Initialize CUDA events for timing
        max_iterations = max_new_tokens + 10
        self._ensure_events_pool(max_iterations)

        for idx in range(max_length):
            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            # Save draft tokens for visualization
            draft_token_ids = draft_tokens[0, 1:].tolist()  # Exclude first token

            # Target model forward timing
            self._events_pool['target_start'][iterations].record()

            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            self._events_pool['target_end'][iterations].record()

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            acc_len = accept_length.item() if isinstance(accept_length, torch.Tensor) else int(accept_length)
            accept_lengths.append(acc_len)

            # Reconstruct tree paths from candidates for visualization
            best_cand_idx = best_candidate.item() if isinstance(best_candidate, torch.Tensor) else int(best_candidate)
            all_paths = []
            for path_idx in range(candidates.shape[0]):
                path_tokens = candidates[path_idx].tolist()
                # Trim -1 padding
                path_tokens = [t for t in path_tokens if t >= 0]
                all_paths.append({
                    "path_idx": path_idx,
                    "tokens": path_tokens,
                    "is_selected": (path_idx == best_cand_idx),
                })

            # Identify rejected token (first draft token that failed verification on selected path)
            selected_path_tokens = [t for t in candidates[best_cand_idx].tolist() if t >= 0]
            rejected_token_id = None
            if acc_len + 1 < len(selected_path_tokens):
                rejected_token_id = selected_path_tokens[acc_len + 1]

            # Get accepted tokens before update
            prev_len = input_ids.shape[1]

            # Draft model forward timing (inside update_inference_inputs)
            self._events_pool['draft_start'][iterations].record()

            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            self._events_pool['draft_end'][iterations].record()

            # Collect per-depth events from this iteration's topK_genrate
            if hasattr(self.ea_layer, '_depth_forward_events'):
                all_depth_forward_events.extend(self.ea_layer._depth_forward_events)

            # Get newly accepted tokens and bonus
            new_token_ids = input_ids[0, prev_len:].tolist()
            bonus_token_id = sample_token[0, 0].item()
            # Append bonus to new_tokens so UI can render it
            new_token_ids.append(bonus_token_id)

            iterations += 1

            # Yield iteration result with tree info, bonus, and rejected token
            yield {
                "iteration": iterations - 1,
                "new_tokens": new_token_ids,
                "draft_tokens": draft_token_ids,
                "accepted_count": acc_len,
                "bonus_token": bonus_token_id,
                "rejected_token": rejected_token_id,
                "tree_info": {
                    "all_paths": all_paths,
                    "selected_path_idx": best_cand_idx,
                    "accepted_tokens": new_token_ids[:acc_len + 1],  # root + accepted drafts
                },
                "current_metrics": {
                    "accept_length": sum(accept_lengths) / len(accept_lengths) + 1,
                    "tokens_so_far": new_token,
                    "elapsed_time": time.time() - start_time,
                }
            }

            # Check stopping conditions
            if is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
                break
            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

        # Sync and compute CUDA event timings
        torch.cuda.synchronize()
        target_time = sum(
            self._events_pool['target_start'][i].elapsed_time(self._events_pool['target_end'][i])
            for i in range(iterations)
        ) / 1000.0 if iterations > 0 else 0.0
        draft_time = sum(
            self._events_pool['draft_start'][i].elapsed_time(self._events_pool['draft_end'][i])
            for i in range(iterations)
        ) / 1000.0 if iterations > 0 else 0.0

        # Collect per-depth draft forward times from all iterations
        draft_forward_times = None
        if all_depth_forward_events:
            from collections import defaultdict
            depth_times = defaultdict(float)
            for depth, s, e in all_depth_forward_events:
                depth_times[depth] += s.elapsed_time(e) / 1000.0
            draft_forward_times = dict(depth_times)

        # Final yield
        yield {
            "final": True,
            "input_ids": input_ids,
            "new_token": new_token,
            "iterations": iterations,
            "accept_lengths": accept_lengths,
            "elapsed_time": time.time() - start_time,
            "target_time": target_time,
            "draft_time": draft_time,
            "draft_forward_times": draft_forward_times,
        }
