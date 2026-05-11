"""MATH-500 dataset loader."""

from typing import List, Dict
from datasets import load_dataset


def load_math500(num_samples: int = 500) -> List[Dict]:
    """Load MATH-500 dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of conversation dictionaries
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    prompts = [q["problem"] for q in dataset][:num_samples]

    result = []
    for i, prompt in enumerate(prompts):
        result.append({
            "id": i,
            "category": "math",
            "conversation": [{"role": "user", "content": prompt}],
        })

    return result
