"""EAGLE3 speculative decoding algorithm."""

import time
from typing import List, Dict, Iterator, Any
import torch

from .base import BaseAlgorithm
from .eagle_official.ea_model import EaModel


class EAGLE3Algorithm(BaseAlgorithm):
    """EAGLE3 speculative decoding."""

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
        """Load EaModel."""
        print(f"Loading EAGLE3: {self.model_path}")
        print(f"  Draft: {self.draft_model_path}")
        print(f"  Config: tokens={self.draft_tokens}, depth={self.depth}, topk={self.topk}")

        self.model = EaModel.from_pretrained(
            use_eagle3=True,
            base_model_path=self.model_path,
            ea_model_path=self.draft_model_path,
            total_token=self.draft_tokens,
            depth=self.depth,
            top_k=self.topk,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=self.device,
        )
        self.model.eval()
        self.tokenizer = self.model.get_tokenizer()

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
            self.draft_tokens = draft_tokens
            ea_layer.total_tokens = draft_tokens - 1

        if depth is not None:
            self.depth = depth
            ea_layer.depth = depth

        if topk is not None:
            self.topk = topk
            ea_layer.top_k = topk
            # Re-initialize tree mask tensors for new topk
            device = ea_layer.embed_tokens.weight.device
            ea_layer.tree_mask_init = torch.eye(topk, device=device)[None, None]
            ea_layer.position_ids = torch.zeros(topk, device=device, dtype=torch.long)

        print(f"Updated tree config: tokens={self.draft_tokens}, depth={self.depth}, topk={self.topk}")

    def _generate_once(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict:
        """Generate single response."""
        is_llama3 = self.model_family == "llama3"

        import os as _os
        _ct_kwargs = {"enable_thinking": False} if _os.environ.get("QWEN_NO_THINK", "0") == "1" else {}
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

        generated_ids = output_ids[0, input_length:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)

        # Compute timing percentages
        total_gpu_time = timing["target_time"] + timing["draft_time"]

        # Compute cumulative position accuracy
        pos_acc = self.compute_cumulative_position_accuracy(accept_lengths, max_depth=self.depth)

        return {
            "output": output_text,
            "metrics": {
                "total_tokens": num_tokens,
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
                "draft_time": timing["draft_time"],
                "target_pct": timing["target_time"] / total_gpu_time * 100 if total_gpu_time > 0 else 0,
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

        import os as _os
        _ct_kwargs = {"enable_thinking": False} if _os.environ.get("QWEN_NO_THINK", "0") == "1" else {}
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
