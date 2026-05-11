"""MTBench dataset loader."""

from typing import List, Dict
from utils.common import download_and_cache_file, read_jsonl


def load_mtbench(
    num_samples: int = 80,
    split_category: bool = False,
) -> List[Dict]:
    """Load MTBench conversations with multi-turn support.

    MTBench has 2 turns per question. The evaluation:
    1. Generate response for turn 1
    2. Add response to conversation history
    3. Add turn 2 question
    4. Generate response for turn 2
    """
    url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
    download_and_cache_file(url, filename="mtbench.jsonl")

    questions = list(read_jsonl("mtbench.jsonl"))[:num_samples]

    result = []
    for q in questions:
        result.append({
            "id": q["question_id"],
            "category": q.get("category", "general"),
            "turns": q["turns"],
        })

    return result
