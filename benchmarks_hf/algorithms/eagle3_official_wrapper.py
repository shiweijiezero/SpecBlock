"""
EAGLE3 Official Wrapper - 直接包装官方EAGLE代码，零修改
"""
import sys
import os
import torch
import time
from typing import List, Tuple, Optional

# 添加官方EAGLE到path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EAGLE_PATH = os.path.join(PROJECT_ROOT, '参考目录', 'EAGLE-main')
if EAGLE_PATH not in sys.path:
    sys.path.insert(0, EAGLE_PATH)

# 直接导入官方EAGLE
from eagle.model.ea_model import EaModel


class EAGLE3OfficialWrapper:
    """直接包装官方EAGLE EaModel，零修改"""

    def __init__(
        self,
        model_path: str,
        draft_model_path: str,
        spec_steps: int = 5,
        draft_tokens: int = 60,
        topk: int = 10,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.draft_model_path = draft_model_path
        self.spec_steps = spec_steps
        self.draft_tokens = draft_tokens
        self.topk = topk
        self.device = device

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载官方EAGLE模型"""
        print(f"Loading official EAGLE model...")
        print(f"  Base model: {self.model_path}")
        print(f"  EAGLE model: {self.draft_model_path}")
        print(f"  Config: total_token={self.draft_tokens}, depth={self.spec_steps}, top_k={self.topk}")

        self.model = EaModel.from_pretrained(
            base_model_path=self.model_path,
            ea_model_path=self.draft_model_path,
            total_token=self.draft_tokens,
            depth=self.spec_steps,
            top_k=self.topk,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device,
            use_eagle3=True,
        )
        self.model.eval()
        self.tokenizer = self.model.get_tokenizer()
        print("Official EAGLE model loaded!")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """生成文本，直接调用官方eagenerate"""

        if self.model is None:
            self.load_model()

        input_ids = input_ids.to(self.model.base_model.device)
        input_len = input_ids.shape[1]

        with torch.inference_mode():
            output_ids, new_token, idx = self.model.eagenerate(
                input_ids,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                log=True,
                is_llama3=True,
            )

        # 计算metrics
        num_generated = len(output_ids[0]) - input_len
        iterations = idx + 1
        accept_length = new_token / iterations if iterations > 0 else 0

        metrics = {
            "num_generated": num_generated,
            "iterations": iterations,
            "new_token": new_token,
            "accept_length": accept_length,
        }

        return output_ids, metrics

    def generate_with_timing(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        """带timing的生成"""

        if self.model is None:
            self.load_model()

        input_ids = input_ids.to(self.model.base_model.device)
        input_len = input_ids.shape[1]

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.inference_mode():
            output_ids, new_token, idx = self.model.eagenerate(
                input_ids,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                log=True,
                is_llama3=True,
            )

        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        num_generated = len(output_ids[0]) - input_len
        iterations = idx + 1
        accept_length = new_token / iterations if iterations > 0 else 0
        throughput = num_generated / elapsed if elapsed > 0 else 0

        metrics = {
            "num_generated": num_generated,
            "elapsed": elapsed,
            "throughput": throughput,
            "iterations": iterations,
            "new_token": new_token,
            "accept_length": accept_length,
        }

        return output_ids, metrics


def test_official_wrapper():
    """测试官方wrapper"""

    # Config
    MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
    DRAFT_PATH = "/path/to/specblock/model/Llama-3.1-8B-Instruct/eagle3_1epoch/epoch_0_step_35000"

    # Initialize
    algo = EAGLE3OfficialWrapper(
        model_path=MODEL_PATH,
        draft_model_path=DRAFT_PATH,
        spec_steps=5,
        draft_tokens=60,
        topk=10,
    )
    algo.load_model()

    # Test prompt (使用和官方测试完全相同的格式)
    SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    USER_PROMPT = "Write a detailed explanation of how neural networks work, including the concepts of forward propagation, backpropagation, and gradient descent."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
    prompt = algo.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = algo.tokenizer([prompt], add_special_tokens=False, return_tensors="pt").input_ids
    print(f"\nInput length: {input_ids.shape[1]} tokens")

    # Warmup
    print("\n" + "="*60)
    print("WARMUP (3 runs)")
    print("="*60)
    for i in range(3):
        _, metrics = algo.generate(input_ids, max_new_tokens=200, temperature=0.0)
        print(f"  Warmup {i+1}: {metrics['num_generated']} tokens, accept_len={metrics['accept_length']:.2f}")

    # Benchmark
    print("\n" + "="*60)
    print("BENCHMARK")
    print("="*60)
    _, metrics = algo.generate_with_timing(input_ids, max_new_tokens=500, temperature=0.0)

    print(f"Generated: {metrics['num_generated']} tokens")
    print(f"Time: {metrics['elapsed']:.2f}s")
    print(f"Throughput: {metrics['throughput']:.2f} tok/s")
    print(f"Accept length: {metrics['accept_length']:.2f}")
    print(f"Speedup vs baseline (28.51): {metrics['throughput']/28.51:.2f}x")


if __name__ == "__main__":
    test_official_wrapper()
