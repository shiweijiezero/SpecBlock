"""Baseline algorithm (standard autoregressive decoding)."""

import os
import time
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseAlgorithm


def _chat_template_kwargs() -> dict:
    return {"enable_thinking": False} if os.environ.get("QWEN_NO_THINK", "0") == "1" else {}


class BaselineAlgorithm(BaseAlgorithm):
    """Baseline autoregressive decoding."""

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        super().__init__(model_path, None, device, **kwargs)
        self.model = None

    def load_model(self):
        print(f"Loading model: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        attention_backend = os.environ.get("BASELINE_ATTN_IMPL", "eager")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation=attention_backend,
        )
        self.model.eval()

    def _generate_once(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict:
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            **_chat_template_kwargs(),
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        input_length = inputs.input_ids.shape[1]

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time

        generated_ids = outputs[0][input_length:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        num_tokens = len(generated_ids)

        return {
            "output": output_text,
            "metrics": {
                "total_tokens": num_tokens,
                "wall_time": elapsed_time,
                "tokens_per_second": num_tokens / elapsed_time if elapsed_time > 0 else 0,
                "draft_time": 0,
                "target_time": elapsed_time,
                "iterations": num_tokens,
                "draft_pct": 0,
                "target_pct": 100.0,
            }
        }
