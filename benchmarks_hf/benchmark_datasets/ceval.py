"""C-Eval dataset loader."""

from typing import List, Dict
from datasets import load_dataset


def load_ceval(num_samples: int = 200) -> List[Dict]:
    """Load C-Eval Chinese evaluation dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of conversation dictionaries
    """
    dataset = load_dataset("zhaode/ceval")["train"]
    prompts = [q["instruction"] for q in dataset][:num_samples]

    result = []
    for i, prompt in enumerate(prompts):
        result.append({
            "id": i,
            "category": "chinese",
            "conversation": [{"role": "user", "content": prompt}],
        })

    return result
