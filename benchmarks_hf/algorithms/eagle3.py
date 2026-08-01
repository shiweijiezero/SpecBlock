"""EAGLE3 speculative decoding algorithm."""

import time
from typing import List, Dict, Iterator, Any
import torch

from .base import BaseAlgorithm
from .eagle_official.ea_model import EaModel


class EAGLE3Algorithm(BaseAlgorithm):
    """EAGLE3 speculative decoding with request-level target batching."""

    supports_true_batch = True

    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        device: str = "cuda",
        draft_tokens: int = 60,
        topk: int = 10,
        depth: int = 7,
        **kwargs
    ):
        super().__init__(model_path, draft_model_path, device, **kwargs)
        self.draft_tokens = draft_tokens
        self.topk = topk
        self.depth = depth
        self.model = None

    def load_model(self):
        """Load the official EAGLE3 draft and native target runtime."""
        if self.draft_tokens <= 0 or self.depth <= 0 or self.topk <= 0:
            raise ValueError(
                "EAGLE3 requires positive draft_tokens, depth, and topk; got "
                f"{self.draft_tokens}, {self.depth}, {self.topk}"
            )
        candidate_capacity = self.topk + self.depth * self.topk * self.topk
        if self.draft_tokens > candidate_capacity:
            raise ValueError(
                f"draft_tokens={self.draft_tokens} exceeds the {candidate_capacity} "
                f"candidates produced by depth={self.depth}, topk={self.topk}"
            )

        print(f"Loading EAGLE3: {self.model_path}")
        print(f"  Draft: {self.draft_model_path}")
        print(
            f"  Config: draft_tokens={self.draft_tokens}, "
            f"verify_nodes={self.draft_tokens + 1}, depth={self.depth}, topk={self.topk}"
        )

        # The legacy official constructor names this total_token and subtracts
        # one internally.  Public SpecForge configs count draft descendants and
        # explicitly exclude the verified root, hence the +1 adapter.
        self.model = EaModel.from_pretrained(
            use_eagle3=True,
            base_model_path=self.model_path,
            ea_model_path=self.draft_model_path,
            total_token=self.draft_tokens + 1,
            depth=self.depth,
            top_k=self.topk,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=self.device,
        )
        self.model.eval()
        self.tokenizer = self.model.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def update_tree_config(self, draft_tokens: int = None, depth: int = None, topk: int = None):
        """Dynamically update tree generation parameters without reloading model.

        Args:
            draft_tokens: New total tokens for tree
            depth: New tree depth
            topk: New top-k value
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model first.")

        ea_layer = self.model.ea_layer

        if draft_tokens is not None:
            if draft_tokens <= 0:
                raise ValueError("draft_tokens must be positive")
            self.draft_tokens = draft_tokens
            # Public draft_tokens excludes the verified root.
            ea_layer.total_tokens = draft_tokens

        if depth is not None:
            if depth <= 0:
                raise ValueError("depth must be positive")
            self.depth = depth
            ea_layer.depth = depth

        if topk is not None:
            if topk <= 0:
                raise ValueError("topk must be positive")
            self.topk = topk
            ea_layer.top_k = topk
            ea_layer.init_tree()

        candidate_capacity = self.topk + self.depth * self.topk * self.topk
        if self.draft_tokens > candidate_capacity:
            raise ValueError(
                f"draft_tokens={self.draft_tokens} exceeds candidate capacity "
                f"{candidate_capacity} for depth={self.depth}, topk={self.topk}"
            )
        print(
            f"Updated tree config: draft_tokens={self.draft_tokens}, "
            f"verify_nodes={self.draft_tokens + 1}, depth={self.depth}, topk={self.topk}"
        )

    def _chat_template_kwargs(self) -> Dict[str, Any]:
        import os

        if os.environ.get("QWEN_NO_THINK", "0") == "1":
            return {"enable_thinking": False}
        return {}

    def generate_conversations(
        self,
        conversations: List[List[Dict[str, str]]],
        max_new_tokens: int,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[Dict]:
        """Run one true target batch for one conversation-turn wave."""
        from .eagle3_batch import generate_conversations

        return generate_conversations(
            self,
            conversations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        samples: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[Dict]:
        """Batch requests by causal conversation turn, never by serial samples."""
        if self.tokenizer is None:
            self.load_model()
        if not samples:
            self.last_batch_metrics = {
                "wall_time": 0.0,
                "prefill_time": 0.0,
                "draft_time": 0.0,
                "target_time": 0.0,
                "verify_time": 0.0,
                "iterations": 0,
                "active_sizes": [],
                "engine_batch_size": 0,
                "acceptance_gpu_calls": 0,
                "acceptance_readbacks": 0,
            }
            return []

        contexts = []
        for sample in samples:
            turns = sample.get("turns")
            if turns:
                contexts.append({
                    "turns": list(turns),
                    "turn_idx": 0,
                    "messages": self.prepare_conversation([]),
                    "outputs": [],
                    "metric_rows": [],
                    "result": None,
                })
            else:
                contexts.append({
                    "turns": None,
                    "turn_idx": 0,
                    "messages": self.prepare_conversation(
                        list(sample.get("conversation", []))
                    ),
                    "outputs": None,
                    "metric_rows": None,
                    "result": None,
                })

        batch_rows = []
        max_engine_batch = 0
        while True:
            pending_conversations = []
            pending_contexts = []
            for context in contexts:
                if context["result"] is not None:
                    continue
                turns = context["turns"]
                if turns is None:
                    if context["turn_idx"] == 0:
                        pending_conversations.append(context["messages"])
                        pending_contexts.append(context)
                elif context["turn_idx"] < len(turns):
                    context["messages"].append({
                        "role": "user",
                        "content": turns[context["turn_idx"]],
                    })
                    pending_conversations.append(context["messages"].copy())
                    pending_contexts.append(context)

            if not pending_conversations:
                break

            max_engine_batch = max(max_engine_batch, len(pending_conversations))
            responses = self.generate_conversations(
                pending_conversations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
            if len(responses) != len(pending_conversations):
                raise RuntimeError(
                    f"EAGLE3 batch returned {len(responses)} responses for "
                    f"{len(pending_conversations)} requests"
                )
            batch_rows.append(dict(self.last_batch_metrics or {}))

            for context, response in zip(pending_contexts, responses):
                if context["turns"] is None:
                    context["result"] = response
                    context["turn_idx"] = 1
                    continue

                output = response["output"]
                context["outputs"].append(output)
                context["metric_rows"].append(response.get("metrics", {}))
                context["messages"].append({"role": "assistant", "content": output})
                context["turn_idx"] += 1
                if context["turn_idx"] >= len(context["turns"]):
                    rows = context["metric_rows"]
                    accept_lengths = [
                        value
                        for row in rows
                        for value in row.get("accept_lengths_raw", [])
                    ]
                    total_tokens = sum(row.get("total_tokens", 0) for row in rows)
                    wall_time = sum(row.get("wall_time", 0.0) for row in rows)
                    prefill_time = sum(row.get("prefill_time", 0.0) for row in rows)
                    draft_time = sum(row.get("draft_time", 0.0) for row in rows)
                    target_time = sum(row.get("target_time", 0.0) for row in rows)
                    verify_time = sum(row.get("verify_time", 0.0) for row in rows)
                    other_time = sum(row.get("other_time", 0.0) for row in rows)
                    pos_acc = self.compute_cumulative_position_accuracy(
                        accept_lengths, max_depth=self.depth
                    )
                    context["result"] = {
                        "output": context["outputs"],
                        "metrics": {
                            "total_tokens": total_tokens,
                            "output_token_ids": [
                                token_id
                                for row in rows
                                for token_id in row.get("output_token_ids", [])
                            ],
                            "wall_time": wall_time,
                            "tokens_per_second": total_tokens / wall_time if wall_time > 0 else 0.0,
                            "accept_length": self.compute_accept_length(accept_lengths),
                            "iterations": sum(row.get("iterations", 0) for row in rows),
                            "num_turns": len(context["turns"]),
                            "accept_lengths_raw": accept_lengths,
                            "position_accuracy": pos_acc["accuracies"],
                            "position_accuracy_pct": pos_acc["accuracies_pct"],
                            "position_accuracy_formatted": pos_acc["formatted"],
                            "prefill_time": prefill_time,
                            "draft_time": draft_time,
                            "target_time": target_time,
                            "verify_time": verify_time,
                            "other_time": other_time,
                            "draft_pct": draft_time / wall_time * 100 if wall_time > 0 else 0.0,
                            "target_pct": target_time / wall_time * 100 if wall_time > 0 else 0.0,
                            "verify_pct": verify_time / wall_time * 100 if wall_time > 0 else 0.0,
                        },
                    }

        wall_time = sum(row.get("wall_time", 0.0) for row in batch_rows)
        prefill_time = sum(row.get("prefill_time", 0.0) for row in batch_rows)
        draft_time = sum(row.get("draft_time", 0.0) for row in batch_rows)
        target_time = sum(row.get("target_time", 0.0) for row in batch_rows)
        verify_time = sum(row.get("verify_time", 0.0) for row in batch_rows)
        active_sizes = [
            size for row in batch_rows for size in row.get("active_sizes", [])
        ]
        decode_rounds = sum(row.get("iterations", 0) for row in batch_rows)
        acceptance_gpu_calls = sum(
            row.get("acceptance_gpu_calls", 0) for row in batch_rows
        )
        acceptance_readbacks = sum(
            row.get("acceptance_readbacks", 0) for row in batch_rows
        )
        self.last_batch_metrics = {
            "wall_time": wall_time,
            "prefill_time": prefill_time,
            "draft_time": draft_time,
            "target_time": target_time,
            "verify_time": verify_time,
            "iterations": decode_rounds,
            "active_sizes": active_sizes,
            "engine_batch_size": max_engine_batch,
            "batch_wall_time": wall_time,
            "batch_prefill_time": prefill_time,
            "batch_draft_time": draft_time,
            "batch_target_time": target_time,
            "batch_verify_time": verify_time,
            "batch_decode_rounds": decode_rounds,
            "batch_size": max_engine_batch,
            "turn_batches": len(batch_rows),
            "acceptance_gpu_calls": acceptance_gpu_calls,
            "acceptance_readbacks": acceptance_readbacks,
        }
        return [context["result"] for context in contexts]

    def _generate_once(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict:
        """Generate single response."""
        is_llama3 = self.model_family == "llama3"

        _ct_kwargs = self._chat_template_kwargs()
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **_ct_kwargs
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.model.base_model.device)

        input_ids = inputs.input_ids
        input_length = input_ids.shape[1]

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        output_ids, new_token, iterations, timing, accept_lengths = self.model.eagenerate_with_cuda_events(
            input_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=8192,
            is_llama3=is_llama3,
        )
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time

        generated_ids = output_ids[0, input_length:input_length + max_new_tokens]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)

        # Compute timing percentages
        total_gpu_time = (
            timing["target_time"]
            + timing["verify_time"]
            + timing["draft_time"]
        )

        # Compute cumulative position accuracy
        pos_acc = self.compute_cumulative_position_accuracy(accept_lengths, max_depth=self.depth)

        return {
            "output": output_text,
            "metrics": {
                "total_tokens": num_tokens,
                "output_token_ids": generated_ids.tolist(),
                "wall_time": elapsed_time,
                "tokens_per_second": num_tokens / elapsed_time if elapsed_time > 0 else 0,
                "accept_length": self.compute_accept_length(accept_lengths),
                "iterations": iterations,
                "accept_lengths_raw": accept_lengths,
                # Cumulative position accuracy
                "position_accuracy": pos_acc['accuracies'],
                "position_accuracy_pct": pos_acc['accuracies_pct'],
                "position_accuracy_formatted": pos_acc['formatted'],
                # Timing breakdown (GPU time via CUDA events)
                "prefill_time": timing["prefill_time"],
                "target_time": timing["target_time"],
                "verify_time": timing["verify_time"],
                "draft_time": timing["draft_time"],
                "target_pct": timing["target_time"] / total_gpu_time * 100 if total_gpu_time > 0 else 0,
                "verify_pct": timing["verify_time"] / total_gpu_time * 100 if total_gpu_time > 0 else 0,
                "draft_pct": timing["draft_time"] / total_gpu_time * 100 if total_gpu_time > 0 else 0,
                # Per-depth draft forward times
                "draft_forward_times": timing.get("draft_forward_times"),
            }
        }

    def _generate_once_streaming(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Streaming version using EaModel's eagenerate_streaming."""
        is_llama3 = self.model_family == "llama3"

        _ct_kwargs = self._chat_template_kwargs()
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **_ct_kwargs
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.model.base_model.device)

        input_ids = inputs.input_ids
        input_length = input_ids.shape[1]

        for result in self.model.eagenerate_streaming(
            input_ids,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=8192,
            is_llama3=is_llama3,
        ):
            if result.get("final"):
                # Final result
                output_ids = result["input_ids"]
                generated_ids = output_ids[0, input_length:]
                output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                num_tokens = len(generated_ids)
                accept_lengths = result["accept_lengths"]
                elapsed = result.get("elapsed_time", 0)

                # Extract timing data
                target_time = result.get("target_time", 0)
                draft_time = result.get("draft_time", 0)
                total_gpu_time = target_time + draft_time

                yield {
                    "final": True,
                    "output": output_text,
                    "metrics": {
                        "total_tokens": num_tokens,
                        "wall_time": elapsed,
                        "tokens_per_second": num_tokens / elapsed if elapsed > 0 else 0,
                        "accept_length": self.compute_accept_length(accept_lengths),
                        "iterations": result["iterations"],
                        "accept_lengths_raw": accept_lengths,
                        # Timing breakdown
                        "target_time": target_time,
                        "draft_time": draft_time,
                        "target_pct": target_time / total_gpu_time * 100 if total_gpu_time > 0 else 0,
                        "draft_pct": draft_time / total_gpu_time * 100 if total_gpu_time > 0 else 0,
                        # Per-depth draft forward times
                        "draft_forward_times": result.get("draft_forward_times"),
                    }
                }
            else:
                # Iteration result
                new_tokens = result["new_tokens"]
                tree_info = result.get("tree_info", {})

                # Extract rejected token info (top-level, fallback to tree_info)
                rejected_token_id = result.get("rejected_token")
                if rejected_token_id is None:
                    rejected_token_id = tree_info.get("rejected_token")
                rejected_text = self.tokenizer.decode([rejected_token_id]) if rejected_token_id is not None else None

                # Extract bonus token info (top-level, fallback to tree_info)
                bonus_token_id = result.get("bonus_token")
                if bonus_token_id is None:
                    bonus_token_id = tree_info.get("bonus_token")
                bonus_text = self.tokenizer.decode([bonus_token_id]) if bonus_token_id is not None else None

                # Decode tree paths for visualization
                decoded_paths = []
                for path_info in tree_info.get("all_paths", []):
                    decoded_paths.append({
                        "path_idx": path_info["path_idx"],
                        "tokens": path_info["tokens"],
                        "texts": [self.tokenizer.decode([t]) for t in path_info["tokens"]],
                        "is_selected": path_info["is_selected"],
                    })

                yield {
                    "iteration": result["iteration"],
                    "new_tokens": new_tokens,
                    "new_text": self.tokenizer.decode(new_tokens),
                    "draft_tokens": result["draft_tokens"],
                    "draft_text": [self.tokenizer.decode([t]) for t in result["draft_tokens"] if t >= 0],
                    "accepted_count": result["accepted_count"],
                    "rejected_token": rejected_token_id,
                    "rejected_text": rejected_text,
                    "bonus_token": bonus_token_id,
                    "bonus_text": bonus_text,
                    # Tree visualization data
                    "tree_info": {
                        "all_paths": decoded_paths,
                        "selected_path_idx": tree_info.get("selected_path_idx"),
                        "accepted_tokens": tree_info.get("accepted_tokens", []),
                        "accepted_texts": [self.tokenizer.decode([t]) for t in tree_info.get("accepted_tokens", [])],
                    },
                    "current_metrics": result["current_metrics"],
                }

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        super().cleanup()
