"""Mixed domain dataset: gsm8k (math) + humaneval (code) + wmt23 (translation),
shuffled to simulate realistic mixed production query distribution.

Usage: --benchmark-list mixed_mct:N  (N = total samples, split 1:1:1 among 3 domains)
"""
import random
from typing import List, Dict


def load_mixed_mct(num_samples: int = 600, seed: int = 42) -> List[Dict]:
    from .gsm8k import load_gsm8k
    from .humaneval import load_humaneval
    from .wmt23 import load_wmt23

    per_domain = num_samples // 3
    gsm = load_gsm8k(num_samples=per_domain)
    human = load_humaneval(num_samples=per_domain)
    wmt = load_wmt23(num_samples=per_domain)

    # tag domain in metadata for per-domain analysis later
    for x in gsm:
        x["domain"] = "gsm8k"
    for x in human:
        x["domain"] = "humaneval"
    for x in wmt:
        x["domain"] = "wmt23"

    mixed = gsm + human + wmt
    rng = random.Random(seed)
    rng.shuffle(mixed)

    for i, x in enumerate(mixed):
        x["id"] = i
        x["category"] = f"mixed_{x['domain']}"

    return mixed
