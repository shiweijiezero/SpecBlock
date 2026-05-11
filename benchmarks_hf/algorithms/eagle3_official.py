"""EAGLE3 Official Implementation Wrapper.

Uses the official EAGLE implementation from eagle_official/ directory.
"""

import time
from typing import List, Dict
import torch

from .base import BaseAlgorithm
from .eagle_official.ea_model import EaModel


class EAGLE3OfficialAlgorithm(BaseAlgorithm):
    """EAGLE3 using official implementation."""

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
        """Initialize EAGLE3 Official.

        Args:
            model_path: Target model path
            draft_model_path: EAGLE3 draft model path
            device: Device
            draft_tokens: Total tree nodes (default: 60)
            topk: Top-k for tree expansion (default: 10)
            depth: Tree depth (default: 7)
        """
        super().__init__(model_path, draft_model_path, device, **kwargs)
        self.draft_tokens = draft_tokens
        self.topk = topk
        self.depth = depth
        self.model = None

    def load_model(self):
        """Load EaModel."""
        print(f"Loading EAGLE3 Official model...")
        print(f"  Target: {self.model_path}")
        print(f"  Draft: {self.draft_model_path}")
        print(f"  Config: total_token={self.draft_tokens}, depth={self.depth}, top_k={self.topk}")

        self.model = EaModel.from_pretrained(
            use_eagle3=True,
            base_model_path=self.model_path,
            ea_model_path=self.draft_model_path,
            total_token=self.draft_tokens,
            depth=self.depth,
            top_k=self.topk,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()
        self.tokenizer = self.model.get_tokenizer()
        print("EAGLE3 Official model loaded!")

    def generate(
        self,
        conversations: List[List[Dict[str, str]]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> List[Dict]:
        """Generate with EAGLE3 Official."""
        if self.model is None:
            self.load_model()

        results = []

        # Check if target model is llama3
        is_llama3 = "llama-3" in self.model_path.lower() or "llama3" in self.model_path.lower()

        for conversation in conversations:
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.model.base_model.device)

            input_ids = inputs.input_ids
            input_length = input_ids.shape[1]

            start_time = time.time()

            # Use eagenerate_with_timing for detailed metrics
            output_ids, new_token, iterations, timing, accept_lengths = self.model.eagenerate_with_timing(
                input_ids,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_length=input_length + max_new_tokens + 100,
                is_llama3=is_llama3,
            )

            elapsed_time = time.time() - start_time
            generated_ids = output_ids[0, input_length:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            num_tokens = len(generated_ids)
            tokens_per_second = num_tokens / elapsed_time if elapsed_time > 0 else 0

            # Calculate accept length stats
            avg_accept_length = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0
            # Accept rate: accepted tokens / (iterations * draft_tokens)
            total_accepted = sum(accept_lengths)
            total_drafted = iterations * self.draft_tokens
            accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0

            # Time breakdown
            total_stage_time = timing["draft_time"] + timing["target_time"] + timing["verify_time"]
            draft_pct = (timing["draft_time"] / total_stage_time * 100) if total_stage_time > 0 else 0
            target_pct = (timing["target_time"] / total_stage_time * 100) if total_stage_time > 0 else 0
            verify_pct = (timing["verify_time"] / total_stage_time * 100) if total_stage_time > 0 else 0

            results.append({
                "output": output_text,
                "metrics": {
                    "total_tokens": num_tokens,
                    "wall_time": elapsed_time,
                    "tokens_per_second": tokens_per_second,
                    "accept_rate": accept_rate,
                    "accept_length": avg_accept_length,
                    "iterations": iterations,
                    "draft_tokens": self.draft_tokens,
                    # Time breakdown
                    "prefill_time": timing["prefill_time"],
                    "draft_time": timing["draft_time"],
                    "target_time": timing["target_time"],
                    "verify_time": timing["verify_time"],
                    "draft_pct": draft_pct,
                    "target_pct": target_pct,
                    "verify_pct": verify_pct,
                }
            })

        return results

    def cleanup(self):
        """Clean up."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
