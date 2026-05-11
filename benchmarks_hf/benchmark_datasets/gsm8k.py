"""GSM8K dataset loader."""

from typing import List, Dict
from utils.common import download_and_cache_file, read_jsonl


def load_gsm8k(num_samples: int = 200, num_shots: int = 5) -> List[Dict]:
    """Load GSM8K math reasoning dataset.

    Args:
        num_samples: Number of samples to load
        num_shots: Number of few-shot examples to include

    Returns:
        List of conversation dictionaries
    """
    def get_one_example(questions, i, include_answer):
        ret = "Question: " + questions[i]["question"] + "\nAnswer:"
        if include_answer:
            ret += " " + questions[i]["answer"]
        return ret

    def get_few_shot_examples(questions, k):
        ret = ""
        for i in range(k):
            ret += get_one_example(questions, i, True) + "\n\n"
        return ret

    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    download_and_cache_file(url, filename="gsm8k.jsonl")

    questions = list(read_jsonl("gsm8k.jsonl"))[:num_samples]
    few_shot_examples = get_few_shot_examples(questions, num_shots)

    result = []
    for i in range(len(questions)):
        conversation = [
            {
                "role": "user",
                "content": few_shot_examples + get_one_example(questions, i, False),
            }
        ]
        result.append({
            "id": i,
            "category": "math",
            "conversation": conversation,
        })

    return result
