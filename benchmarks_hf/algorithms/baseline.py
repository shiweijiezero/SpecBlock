"""Baseline algorithm (standard autoregressive decoding)."""

import os
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseAlgorithm


def _chat_template_kwargs() -> dict:
    return {"enable_thinking": False} if os.environ.get("QWEN_NO_THINK", "0") == "1" else {}


class BaselineAlgorithm(BaseAlgorithm):
    """Standard autoregressive decoding with request-level HF batching."""

    supports_true_batch = True

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

    def _synchronize(self) -> None:
        """Synchronize CUDA timing without making the CPU path CUDA-dependent."""
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def _trim_generated_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Remove generate()'s right padding while retaining a generated EOS."""
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            return generated_ids
        eos_positions = (generated_ids == eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel():
            return generated_ids[: int(eos_positions[0, 0]) + 1]
        return generated_ids

    def generate_conversations(
        self,
        conversations: List[List[Dict[str, str]]],
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> List[Dict]:
        """Generate one causal conversation wave in one padded HF call."""
        if not conversations:
            self.last_batch_metrics = {
                "wall_time": 0.0,
                "prefill_time": 0.0,
                "draft_time": 0.0,
                "target_time": 0.0,
                "verify_time": 0.0,
                "iterations": 0,
                "active_sizes": [],
                "engine_batch_size": 0,
            }
            return []

        prompts = [
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                **_chat_template_kwargs(),
            )
            for conversation in conversations
        ]
        # Decoder-only generation reads logits from the final sequence position.
        # Left-pad only while forming this batch, then restore shared tokenizer
        # state for callers outside this generation request.
        original_padding_side = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = original_padding_side
        input_width = inputs.input_ids.shape[1]

        self._synchronize()
        start_time = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self._synchronize()
        elapsed_time = time.perf_counter() - start_time

        results = []
        for row in outputs:
            # `generate` returns the padded input prefix for every row.  Slice at
            # the shared padded width, never at each row's unpadded prompt length.
            generated_ids = self._trim_generated_ids(row[input_width:])
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            num_tokens = len(generated_ids)
            results.append({
                "output": output_text,
                "metrics": {
                    "total_tokens": num_tokens,
                    "output_token_ids": generated_ids.tolist(),
                    "wall_time": elapsed_time,
                    "tokens_per_second": num_tokens / elapsed_time if elapsed_time > 0 else 0.0,
                    "prefill_time": 0.0,
                    "draft_time": 0.0,
                    # HF generate owns both prefill and decode; expose it as target
                    # time rather than inventing an unmeasurable split.
                    "target_time": elapsed_time,
                    "verify_time": 0.0,
                    "iterations": num_tokens,
                    "draft_pct": 0.0,
                    "target_pct": 100.0,
                    "verify_pct": 0.0,
                },
            })

        # One HF generate invocation runs one decoding round for each request
        # that still has an output token.  Preserve that active-shrink view in
        # batch metrics even though Transformers owns the internal loop.
        max_generated_tokens = max(
            (result["metrics"]["iterations"] for result in results), default=0
        )
        active_sizes = [
            sum(result["metrics"]["iterations"] > step for result in results)
            for step in range(max_generated_tokens)
        ]
        self.last_batch_metrics = {
            "wall_time": elapsed_time,
            "prefill_time": 0.0,
            "draft_time": 0.0,
            "target_time": elapsed_time,
            "verify_time": 0.0,
            "iterations": len(active_sizes),
            "active_sizes": active_sizes,
            "engine_batch_size": len(conversations),
            "batch_wall_time": elapsed_time,
            "batch_prefill_time": 0.0,
            "batch_draft_time": 0.0,
            "batch_target_time": elapsed_time,
            "batch_verify_time": 0.0,
            "batch_decode_rounds": len(active_sizes),
            "batch_size": len(conversations),
        }
        return results

    @torch.inference_mode()
    def generate(
        self,
        samples: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[Dict]:
        """Batch requests by causal turn, shrinking each subsequent turn wave."""
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
                if context["turns"] is None:
                    if context["turn_idx"] == 0:
                        pending_conversations.append(context["messages"])
                        pending_contexts.append(context)
                elif context["turn_idx"] < len(context["turns"]):
                    context["messages"].append({
                        "role": "user",
                        "content": context["turns"][context["turn_idx"]],
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
                    f"Baseline batch returned {len(responses)} responses for "
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
                context["metric_rows"].append(response["metrics"])
                context["messages"].append({"role": "assistant", "content": output})
                context["turn_idx"] += 1
                if context["turn_idx"] >= len(context["turns"]):
                    rows = context["metric_rows"]
                    total_tokens = sum(row.get("total_tokens", 0) for row in rows)
                    wall_time = sum(row.get("wall_time", 0.0) for row in rows)
                    target_time = sum(row.get("target_time", 0.0) for row in rows)
                    total_iterations = sum(row.get("iterations", 0) for row in rows)
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
                            "tokens_per_second": (
                                total_tokens / wall_time if wall_time > 0 else 0.0
                            ),
                            "iterations": total_iterations,
                            "num_turns": len(context["turns"]),
                            "prefill_time": 0.0,
                            "draft_time": 0.0,
                            "target_time": target_time,
                            "verify_time": 0.0,
                            "draft_pct": 0.0,
                            "target_pct": 100.0,
                            "verify_pct": 0.0,
                        },
                    }

        self.last_batch_metrics = {
            "wall_time": sum(row.get("wall_time", 0.0) for row in batch_rows),
            "prefill_time": 0.0,
            "draft_time": 0.0,
            "target_time": sum(row.get("target_time", 0.0) for row in batch_rows),
            "verify_time": 0.0,
            "iterations": sum(row.get("iterations", 0) for row in batch_rows),
            "active_sizes": [
                size for row in batch_rows for size in row.get("active_sizes", [])
            ],
            "engine_batch_size": max_engine_batch,
            "batch_wall_time": sum(row.get("wall_time", 0.0) for row in batch_rows),
            "batch_prefill_time": 0.0,
            "batch_draft_time": 0.0,
            "batch_target_time": sum(
                row.get("target_time", 0.0) for row in batch_rows
            ),
            "batch_verify_time": 0.0,
            "batch_decode_rounds": sum(row.get("iterations", 0) for row in batch_rows),
            "batch_size": max_engine_batch,
            "turn_batches": len(batch_rows),
        }
        return [context["result"] for context in contexts]

    def _generate_once(
        self,
        conversation: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Dict:
        """Generate a single response through the same true-batch core."""
        return self.generate_conversations(
            [conversation],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )[0]
