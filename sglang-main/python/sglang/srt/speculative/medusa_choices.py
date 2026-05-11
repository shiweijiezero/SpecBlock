"""
Medusa tree structure configurations.

These are predefined tree structures used during inference to construct
candidate token sequences. Each choice represents a path in the tree:
- (0,) means select head 0's top-1 candidate
- (0, 0) means select head 0's top-1, then from that position select head 0's top-1 again
- (1,) means select head 0's top-2 candidate (second best)
- (0, 1) means select head 0's top-1, then from that position select head 0's top-2

The tree structure allows exploring multiple candidate paths simultaneously
using tree attention, significantly improving the acceptance rate compared
to linear sequence prediction.
"""

# Linear 4 candidates with topk=1 (true linear sequence)
# Each head predicts one token at different positions
# This requires only topk=1 (each head contributes its top-1)
mc_linear_4_topk1 = [
    [0],        # Head 0, top-1 → position i+1
    [0, 0],     # Head 1, top-1 → position i+2
    [0, 0, 0],  # Head 2, top-1 → position i+3
    [0, 0, 0, 0],  # Head 3, top-1 → position i+4
]

# Linear 4 candidates (simplest, matches num_heads)
# This is for testing/debugging or low-latency scenarios
# NOTE: This requires topk >= 4 (samples from head 0's top-4)
# Equivalent to EAGLE3's "4,3,1,4" configuration
mc_linear_4 = [
    [0],  # Head 0, top-1
    [1],  # Head 0, top-2
    [2],  # Head 0, top-3
    [3],  # Head 0, top-4
]

# 63 candidates tree structure (most common, balanced)
mc_sim_7b_63 = [
    [0],
    [0, 0],
    [1],
    [0, 1],
    [2],
    [0, 0, 0],
    [1, 0],
    [0, 2],
    [3],
    [0, 3],
    [4],
    [0, 4],
    [2, 0],
    [0, 5],
    [0, 0, 1],
    [5],
    [0, 6],
    [6],
    [0, 7],
    [0, 1, 0],
    [1, 1],
    [7],
    [0, 8],
    [0, 0, 2],
    [3, 0],
    [0, 9],
    [8],
    [9],
    [1, 0, 0],
    [0, 2, 0],
    [1, 2],
    [0, 0, 3],
    [4, 0],
    [2, 1],
    [0, 0, 4],
    [0, 0, 5],
    [0, 0, 0, 0],
    [0, 1, 1],
    [0, 0, 6],
    [0, 3, 0],
    [5, 0],
    [1, 3],
    [0, 0, 7],
    [0, 0, 8],
    [0, 0, 9],
    [6, 0],
    [0, 4, 0],
    [1, 4],
    [7, 0],
    [0, 1, 2],
    [2, 0, 0],
    [3, 1],
    [2, 2],
    [8, 0],
    [0, 5, 0],
    [1, 5],
    [1, 0, 1],
    [0, 2, 1],
    [9, 0],
    [0, 6, 0],
    [0, 0, 0, 1],
    [1, 6],
    [0, 7, 0],
]

# 60 candidates tree structure (matches EAGLE3's default)
mc_sim_7b_60 = [
    [0],
    [0, 0],
    [1],
    [0, 1],
    [2],
    [0, 0, 0],
    [1, 0],
    [0, 2],
    [3],
    [0, 3],
    [4],
    [0, 4],
    [2, 0],
    [0, 5],
    [0, 0, 1],
    [5],
    [0, 6],
    [6],
    [0, 7],
    [0, 1, 0],
    [1, 1],
    [7],
    [0, 8],
    [0, 0, 2],
    [3, 0],
    [0, 9],
    [8],
    [9],
    [1, 0, 0],
    [0, 2, 0],
    [1, 2],
    [0, 0, 3],
    [4, 0],
    [2, 1],
    [0, 0, 4],
    [0, 0, 5],
    [0, 0, 0, 0],
    [0, 1, 1],
    [0, 0, 6],
    [0, 3, 0],
    [5, 0],
    [1, 3],
    [0, 0, 7],
    [0, 0, 8],
    [0, 0, 9],
    [6, 0],
    [0, 4, 0],
    [1, 4],
    [7, 0],
    [0, 1, 2],
    [2, 0, 0],
    [3, 1],
    [2, 2],
    [8, 0],
    [0, 5, 0],
    [1, 5],
    [1, 0, 1],
    [0, 2, 1],
    [9, 0],
    [0, 6, 0],
]

# Shorter tree structures for faster inference (lower latency, potentially lower accept rate)
mc_sim_7b_38 = [
    [0],
    [0, 0],
    [1],
    [0, 1],
    [2],
    [0, 0, 0],
    [1, 0],
    [0, 2],
    [3],
    [0, 3],
    [4],
    [0, 4],
    [2, 0],
    [0, 5],
    [0, 0, 1],
    [5],
    [0, 6],
    [6],
    [0, 7],
    [0, 1, 0],
    [1, 1],
    [7],
    [0, 8],
    [0, 0, 2],
    [3, 0],
    [0, 9],
    [8],
    [9],
    [1, 0, 0],
    [0, 2, 0],
    [1, 2],
    [0, 0, 3],
    [4, 0],
    [2, 1],
    [0, 0, 4],
    [0, 0, 5],
    [0, 0, 0, 0],
    [0, 1, 1],
]

# Larger tree structure for potentially higher accept rate (higher latency)
mc_sim_7b_96 = mc_sim_7b_63 + [
    [0, 8, 0],
    [0, 0, 0, 2],
    [3, 0, 0],
    [2, 0, 1],
    [0, 1, 4],
    [4, 1],
    [1, 7],
    [0, 0, 1, 0],
    [0, 3, 1],
    [1, 0, 2],
    [2, 3],
    [3, 2],
    [0, 9, 0],
    [0, 2, 2],
    [1, 8],
    [5, 1],
    [0, 1, 5],
    [0, 0, 0, 3],
    [7, 1],
    [2, 4],
    [0, 4, 1],
    [4, 2],
    [0, 0, 10],
    [6, 1],
    [3, 3],
    [1, 9],
    [8, 1],
    [0, 5, 1],
    [9, 1],
    [2, 5],
    [5, 2],
    [1, 0, 3],
    [0, 1, 6],
]

# Configuration mapping
MEDUSA_CHOICES_CONFIGS = {
    "mc_linear_4_topk1": mc_linear_4_topk1,  # 4 candidates, true linear (topk=1)
    "mc_linear_4": mc_linear_4,    # 4 candidates, linear (requires topk>=4)
    "mc_sim_7b_38": mc_sim_7b_38,  # 38 candidates
    "mc_sim_7b_60": mc_sim_7b_60,  # 60 candidates (matches EAGLE3's "4,7,10,60", default)
    "mc_sim_7b_63": mc_sim_7b_63,  # 63 candidates
    "mc_sim_7b_96": mc_sim_7b_96,  # 96 candidates
}


def get_medusa_choices(config_name: str = "mc_sim_7b_60"):
    """
    Get Medusa choices configuration by name.

    Args:
        config_name: Name of the configuration. Options:
            - "mc_linear_4": 4 candidates, linear (matches EAGLE3's "4,3,1,4", for testing)
            - "mc_sim_7b_38": 38 candidates (faster, lower accept rate)
            - "mc_sim_7b_60": 60 candidates (matches EAGLE3's "4,7,10,60", recommended, default)
            - "mc_sim_7b_63": 63 candidates (slightly more nodes)
            - "mc_sim_7b_96": 96 candidates (slower, higher accept rate)

    Returns:
        List of tree choices

    Example usage:
        # Simple configuration (for testing, like EAGLE's "4,3,1,4")
        config_list=("4,4,1,4")
        --medusa-choices mc_linear_4 --medusa-topk 1

        # Full configuration (for best performance, like EAGLE's "4,7,10,60")
        config_list=("4,4,10,60")
        --medusa-choices mc_sim_7b_60 --medusa-topk 10
    """
    if config_name not in MEDUSA_CHOICES_CONFIGS:
        raise ValueError(
            f"Unknown medusa_choices config: {config_name}. "
            f"Available: {list(MEDUSA_CHOICES_CONFIGS.keys())}"
        )
    return MEDUSA_CHOICES_CONFIGS[config_name]
