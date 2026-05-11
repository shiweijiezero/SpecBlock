"""
Offline data generation script using SGLang Engine - OPTIMIZED VERSION.

Key optimizations:
1. No temperature grouping - use per-request sampling params
2. Continuous Refill: maintain fixed-size active pool, refill as conversations complete
3. Write completed conversations incrementally
4. GPU utilization stays high throughout the entire generation process

Usage:
CUDA_VISIBLE_DEVICES=0 python scripts/offline_generate_data.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --raw-data-file cache/dataset/train_dataset.jsonl \
    --output-dir cache/offline-generated/llama-3.1-8b-instruct \
    --shard-id 0 \
    --num-shards 8 \
    --batch-size 500
"""

import argparse
import json
import os
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
from tqdm import tqdm

import sglang as sgl
from transformers import AutoTokenizer


# Default Llama3 style system prompt (aligned with official EAGLE-3)
# Note: "while being safe.  Your" has TWO spaces after the period (official format)
SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--raw-data-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=500, help="Active pool size (conversations processed in parallel)")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Write completed convs every N")
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument(
        "--enable-thinking", action="store_true",
        help="Pass enable_thinking=True to tokenizer.apply_chat_template (Qwen3 thinking mode). "
             "Default False = instruct mode (Qwen3 inserts empty <think></think>). "
             "Non-Qwen3 tokenizers ignore this kwarg.",
    )
    return parser.parse_args()


def get_random_temperature() -> float:
    choices = [0.0, 0.3, 0.5, 0.7, 1.0]
    weights = [4, 1, 1, 1, 3]
    return random.choices(choices, weights=weights)[0]


@dataclass
class ConversationState:
    """Track state of a multi-turn conversation generation."""
    conversation_id: str
    input_conversations: List[Dict[str, str]]
    temperature: float
    current_messages: List[Dict[str, str]] = field(default_factory=list)
    current_turn: int = 0
    completed: bool = False
    error: Optional[str] = None

    def get_next_user_turn(self) -> Optional[str]:
        """Get the next user message to respond to, or None if done."""
        while self.current_turn < len(self.input_conversations):
            conv = self.input_conversations[self.current_turn]
            if conv["role"] == "user":
                return conv["content"]
            self.current_turn += 1
        return None

    def add_assistant_response(self, response: str):
        self.current_messages.append({"role": "assistant", "content": response})
        self.current_turn += 1

    def add_user_message(self, content: str):
        self.current_messages.append({"role": "user", "content": content})


def build_vicuna_prompt(messages: List[Dict[str, str]]) -> str:
    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"USER: {msg['content']} "
        elif msg["role"] == "assistant":
            prompt += f"ASSISTANT: {msg['content']}</s> "
    prompt += "ASSISTANT:"
    return prompt


def build_prompt_from_messages(
    messages: List[Dict[str, str]],
    tokenizer,
    model_path: str = "",
    enable_thinking: bool = False,
) -> str:
    if "vicuna" in model_path.lower():
        return build_vicuna_prompt(messages)
    # enable_thinking is a Qwen3-specific kwarg; other tokenizer chat templates
    # don't reference it in their Jinja so it's harmlessly ignored.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def load_data_for_shard(raw_data_file: str, shard_id: int, num_shards: int) -> List[Dict]:
    all_data = []
    with open(raw_data_file, 'r') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

    shard_data = [item for i, item in enumerate(all_data) if i % num_shards == shard_id]
    print(f"Shard {shard_id}/{num_shards}: {len(shard_data)} records (total: {len(all_data)})", file=sys.stderr)
    return shard_data


def create_state_from_item(item: Dict) -> ConversationState:
    """Create a ConversationState from a data item."""
    return ConversationState(
        conversation_id=str(item["id"]),
        input_conversations=item["conversations"],
        temperature=get_random_temperature(),
    )


def process_multi_turn_continuous_refill(
    engine: sgl.Engine,
    tokenizer,
    pending_queue: Deque[Dict],
    batch_size: int,
    max_tokens: int,
    system_prompt: str,
    model_path: str,
    f_out,
    f_err,
    checkpoint_interval: int,
    total_conversations: int,
    enable_thinking: bool = False,
) -> tuple:
    """
    Process conversations with Continuous Refill strategy.

    Maintains a fixed-size active pool. When conversations complete,
    immediately refill from pending_queue to keep GPU utilization high.

    Returns (success_count, error_count)
    """
    success_count = 0
    error_count = 0
    completed_buffer = []

    # Initialize active pool
    active_pool: List[ConversationState] = []
    while len(active_pool) < batch_size and pending_queue:
        active_pool.append(create_state_from_item(pending_queue.popleft()))

    round_num = 0
    pbar = tqdm(total=total_conversations, desc="Conversations", file=sys.stderr, leave=False)

    while active_pool:
        round_num += 1

        # Collect prompts for this round
        batch_states = []
        batch_prompts = []
        batch_sampling_params = []
        newly_completed = []

        for state in active_pool:
            user_content = state.get_next_user_turn()
            if user_content is None:
                state.completed = True
                newly_completed.append(state)
                continue

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(state.current_messages)
            messages.append({"role": "user", "content": user_content})

            state.add_user_message(user_content)

            try:
                prompt = build_prompt_from_messages(messages, tokenizer, model_path, enable_thinking=enable_thinking)
                batch_states.append(state)
                batch_prompts.append(prompt)
                batch_sampling_params.append({
                    "temperature": state.temperature,
                    "max_new_tokens": max_tokens,
                    "top_p": 0.95 if state.temperature > 0 else 1.0,
                })
            except Exception as e:
                state.error = f"Prompt building failed: {str(e)}"
                newly_completed.append(state)

        # Add newly completed to buffer
        completed_buffer.extend(newly_completed)

        if not batch_prompts:
            # All remaining conversations are done, exit
            break

        # Generate
        try:
            outputs = engine.generate(batch_prompts, batch_sampling_params)
            if isinstance(outputs, dict):
                outputs = [outputs]

            for state, output in zip(batch_states, outputs):
                try:
                    response_text = output["text"]
                    state.add_assistant_response(response_text)
                except Exception as e:
                    state.error = f"Response extraction failed: {str(e)}"
                    completed_buffer.append(state)

        except Exception as e:
            for state in batch_states:
                state.error = f"Batch generation failed: {str(e)}"
                completed_buffer.append(state)

        # Remove completed/errored states from active pool
        active_pool = [s for s in active_pool if not s.completed and s.error is None]

        # === Continuous Refill: refill active pool from pending queue ===
        refilled = 0
        while len(active_pool) < batch_size and pending_queue:
            active_pool.append(create_state_from_item(pending_queue.popleft()))
            refilled += 1

        # Write completed conversations to disk (checkpoint)
        if len(completed_buffer) >= checkpoint_interval:
            for state in completed_buffer:
                if state.error is not None:
                    error_dict = {
                        "conversation_id": state.conversation_id,
                        "conversations": state.current_messages,
                        "error": state.error,
                        "input_conversations": state.input_conversations,
                    }
                    f_err.write(json.dumps(error_dict) + "\n")
                    error_count += 1
                else:
                    output_dict = {
                        "conversation_id": state.conversation_id,
                        "conversations": state.current_messages,
                    }
                    f_out.write(json.dumps(output_dict) + "\n")
                    success_count += 1

            f_out.flush()
            f_err.flush()
            pbar.update(len(completed_buffer))
            completed_buffer = []

        # Log progress
        pending_count = len(pending_queue)
        tqdm.write(
            f"  Round {round_num}: {len(batch_prompts)} prompts, "
            f"active={len(active_pool)}, pending={pending_count}, "
            f"refilled={refilled}",
            file=sys.stderr
        )

    # Write remaining completed conversations
    for state in completed_buffer:
        if state.error is not None:
            error_dict = {
                "conversation_id": state.conversation_id,
                "conversations": state.current_messages,
                "error": state.error,
                "input_conversations": state.input_conversations,
            }
            f_err.write(json.dumps(error_dict) + "\n")
            error_count += 1
        else:
            output_dict = {
                "conversation_id": state.conversation_id,
                "conversations": state.current_messages,
            }
            f_out.write(json.dumps(output_dict) + "\n")
            success_count += 1

    f_out.flush()
    f_err.flush()
    pbar.update(len(completed_buffer))
    pbar.close()

    return success_count, error_count


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data for this shard
    shard_data = load_data_for_shard(args.raw_data_file, args.shard_id, args.num_shards)

    if not shard_data:
        print(f"No data for shard {args.shard_id}", file=sys.stderr)
        return

    # Check for existing output files (checkpoint resume)
    output_file = os.path.join(args.output_dir, f"shard_{args.shard_id}.jsonl")
    error_file = os.path.join(args.output_dir, f"error_{args.shard_id}.jsonl")

    existing_ids = set()
    for fpath in [output_file, error_file]:
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        existing_ids.add(item["conversation_id"])
                    except:
                        pass

    if existing_ids:
        shard_data = [item for item in shard_data if str(item["id"]) not in existing_ids]
        print(f"Resuming: {len(existing_ids)} done, {len(shard_data)} remaining", file=sys.stderr)

    if not shard_data:
        print(f"Shard {args.shard_id} complete", file=sys.stderr)
        return

    # Initialize engine and tokenizer
    print(f"Loading model: {args.model_path}", file=sys.stderr)
    engine = sgl.Engine(model_path=args.model_path, mem_fraction_static=args.mem_fraction)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    total = len(shard_data)
    batch_size = args.batch_size

    print(f"Processing {total} conversations with Continuous Refill (batch_size={batch_size})", file=sys.stderr)

    # Shuffle for randomization
    random.shuffle(shard_data)

    # Create pending queue
    pending_queue = deque(shard_data)

    f_out = open(output_file, 'a')
    f_err = open(error_file, 'a')

    try:
        # Process with Continuous Refill - single pass, no chunks
        total_success, total_error = process_multi_turn_continuous_refill(
            engine=engine,
            tokenizer=tokenizer,
            pending_queue=pending_queue,
            batch_size=batch_size,
            max_tokens=args.max_tokens,
            system_prompt=args.system_prompt,
            model_path=args.model_path,
            f_out=f_out,
            f_err=f_err,
            checkpoint_interval=args.checkpoint_interval,
            total_conversations=total,
            enable_thinking=args.enable_thinking,
        )

    finally:
        f_out.close()
        f_err.close()

    print(f"\nShard {args.shard_id} complete:", file=sys.stderr)
    print(f"  Success: {total_success}", file=sys.stderr)
    print(f"  Errors: {total_error}", file=sys.stderr)
    print(f"  Output: {output_file}", file=sys.stderr)

    engine.shutdown()


if __name__ == "__main__":
    main()
