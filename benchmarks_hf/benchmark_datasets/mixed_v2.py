"""Mixed-domain dataset for adapt eval: math500 + wmt23 + alpaca + nq_open.

All four tasks are >=200 samples (matches the current cross-domain protocol).
N samples are split 1:1:1:1, then shuffled.

Usage: --benchmark-list mixed_v2:1000  (250 each domain)
       --benchmark-list mixed_v2:2000  (500 each domain)
"""
import random
from typing import List, Dict


def load_mixed_v2(num_samples: int = 1000, seed: int = 42) -> List[Dict]:
    from .math500 import load_math500
    from .wmt23 import load_wmt23
    from .alpaca import load_alpaca
    from .nq_open import load_nq_open

    per = num_samples // 4
    parts = []
    parts.append(("math500", load_math500(num_samples=per)))
    parts.append(("wmt23", load_wmt23(num_samples=per)))
    parts.append(("alpaca", load_alpaca(num_samples=per)))
    parts.append(("nq_open", load_nq_open(num_samples=per)))

    mixed = []
    for dom, items in parts:
        for x in items:
            x["domain"] = dom
            mixed.append(x)
    rng = random.Random(seed)
    rng.shuffle(mixed)
    for i, x in enumerate(mixed):
        x["id"] = i
        x["category"] = f"mix_{x['domain']}"
    return mixed
