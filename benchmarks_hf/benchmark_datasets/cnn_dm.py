"""CNN/DailyMail dataset loader."""

from typing import List, Dict
from datasets import load_dataset


def load_cnn_dm(num_samples: int = 200) -> List[Dict]:
    """Load CNN/DailyMail summarization dataset.

    Args:
        num_samples: Number of samples to load

    Returns:
        List of conversation dictionaries
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")["test"]

    result = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        prompt = f"Summarize the following article:\n\n{sample['article']}"
        result.append({
            "id": i,
            "category": "summarization",
            "conversation": [{"role": "user", "content": prompt}],
        })

    return result
