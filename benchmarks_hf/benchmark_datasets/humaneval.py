"""HumanEval dataset loader."""

from typing import List, Dict
from datasets import load_dataset


def load_humaneval(num_samples: int = 164) -> List[Dict]:
    """Load HumanEval code generation dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of conversation dictionaries
    """

    dataset = load_dataset("openai/openai_humaneval")["test"]
    prompts = [q["prompt"] for q in dataset][:num_samples]

    result = []
    for i, prompt in enumerate(prompts):
        result.append({
            "id": i,
            "category": "code",
            "conversation": [{"role": "user", "content": prompt}],
        })

    return result
