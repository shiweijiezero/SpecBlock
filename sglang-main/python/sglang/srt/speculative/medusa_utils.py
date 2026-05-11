"""
Utility functions for Medusa tree-based speculative decoding.

This module contains functions to convert Medusa choices into tree structures
that can be used with EAGLE's tree building and verification kernels.
"""

import torch
from typing import List, Tuple

from sglang.srt.speculative.medusa_choices import get_medusa_choices


def build_medusa_parent_and_scores(
    medusa_choices: List[List[int]],
    bs: int,
    topk: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Medusa choices into parent_list and top_scores_index for tree building.

    This function transforms the pre-defined Medusa tree structure (medusa_choices)
    into the format required by build_tree_kernel_efficient(), which is originally
    designed for EAGLE's dynamic tree construction.

    EAGLE3's format (from select_top_k_tokens and organize_draft_results):
    - parent_list: Global indices into the flattened token array
      - Step 0: [-1, 0, 1, ..., topk-1] where -1 is root
      - Step i: topk_cs_index + (topk^2 * (i-1) + topk)
      - Final: torch.cat(parents_list[:-1], dim=1) - excludes last step
    - top_scores_index: Indices used to select which tokens to keep from all candidates

    For Medusa's predefined tree, we simulate this format:
    - parent_list: Parent node index in the sorted choice list (0 = root)
    - top_scores_index: Global index = depth_offset + topk_idx
      where depth_offset = topk * (depth - 1) for depth >= 1

    Args:
        medusa_choices: List of tree paths, e.g., [[0], [0,0], [1], [0,1], ...]
        bs: Batch size
        topk: Top-k candidates per head
        device: Torch device

    Returns:
        parent_list: Tensor of shape (bs, num_choices) containing parent indices.
        top_scores_index: Tensor of shape (bs, num_choices) containing global indices.
    """
    # Sort choices by (depth, indices) to get canonical ordering
    sorted_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    num_choices = len(sorted_choices)

    # Build mapping from choice to index in sorted list
    # Index in sorted list + 1 because index 0 is reserved for root (verified_id)
    choice_to_idx = {tuple(choice): i + 1 for i, choice in enumerate(sorted_choices)}

    # Build parent_list: parent node index for each choice (non-root nodes only)
    # For Medusa's flat tree (all nodes at depth 1), all parents are root (index 0)
    # CRITICAL: Kernel expects parent_list shape (bs, topk*(depth-1)+1) according to eagle_utils.cu:28
    # For depth=1: (bs, topk*0+1) = (bs, 1) with all zeros (root parent)
    max_depth = max(len(choice) for choice in sorted_choices)

    if max_depth == 1:
        # All nodes are at depth 1 (direct children of root)
        # MUST pass (bs, 1) with zeros, NOT (bs, 0)!
        parent_list = torch.zeros(bs, 1, dtype=torch.int64, device=device)
    else:
        # Multi-level tree - build parent indices normally
        parent_indices = []
        for choice in sorted_choices:
            if len(choice) == 1:
                # First level: parent is root (verified_id at index 0)
                parent_idx = 0
            else:
                # Subsequent levels: parent is the prefix
                parent_choice = tuple(choice[:-1])
                parent_idx = choice_to_idx[parent_choice]
            parent_indices.append(parent_idx)

        # Convert to tensor - shape (1, num_choices) then expand to (bs, num_choices)
        parent_list = torch.tensor(
            parent_indices, dtype=torch.int64, device=device
        ).unsqueeze(0).expand(bs, -1).contiguous()

    # Build top_scores_index: indices used to "select" nodes from candidates
    # In EAGLE3, this is torch.topk(scores).indices followed by sort
    # For Medusa's predefined tree, we simply use sequential indices [0, 1, 2, ...]
    # since all nodes are predetermined and no dynamic selection is needed
    # CRITICAL: Must have same length as parent_list (num_choices)
    top_scores_index = torch.arange(
        num_choices, dtype=torch.int64, device=device
    ).unsqueeze(0).expand(bs, -1).contiguous()

    return parent_list, top_scores_index


def generate_medusa_draft_tokens_from_topk(
    medusa_logits: torch.Tensor,
    medusa_choices: List[List[int]],
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate draft tokens based on Medusa choices and topk logits.

    Args:
        medusa_logits: Tensor of shape (num_heads, bs, vocab_size)
            Logits from each Medusa head
        medusa_choices: List of tree paths
        topk: Top-k candidates per head
        device: Torch device

    Returns:
        draft_tokens: Tensor of shape (bs, num_choices) containing
            the selected token IDs for each tree node
    """
    num_heads, bs, vocab_size = medusa_logits.shape
    sorted_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    num_choices = len(sorted_choices)

    # Get topk candidates from all heads
    # Shape: (num_heads, bs, topk)
    topk_tokens = torch.topk(medusa_logits, topk, dim=-1).indices

    # Select tokens based on choices
    draft_tokens = torch.zeros((bs, num_choices), dtype=torch.int64, device=device)

    for choice_idx, choice in enumerate(sorted_choices):
        # For each choice, traverse the path and get final token
        # Simplified: use the first head and the last topk index
        # (This is a simplification; full implementation needs to track parent tokens)

        # For first level: directly index into topk_tokens
        if len(choice) == 1:
            head_idx = 0  # All first-level choices from head 0
            topk_idx = choice[0]
            tokens = topk_tokens[head_idx, :, topk_idx]
            draft_tokens[:, choice_idx] = tokens
        else:
            # For deeper levels: would need to track parent's token
            # For now, use simplified approach with head cycling
            head_idx = (len(choice) - 1) % num_heads
            topk_idx = choice[-1]
            tokens = topk_tokens[head_idx, :, topk_idx]
            draft_tokens[:, choice_idx] = tokens

    return draft_tokens


def get_num_draft_tokens_from_choices(choices_name: str) -> int:
    """
    Get the number of draft tokens (tree nodes) from choices configuration name.

    Args:
        choices_name: Name of the choices configuration
            (e.g., "mc_sim_7b_63", "mc_sim_7b_38", "mc_sim_7b_96")

    Returns:
        Number of draft tokens (tree nodes)
    """
    choices = get_medusa_choices(choices_name)
    return len(choices)


def generate_medusa_tree_mask(
    medusa_choices: List[List[int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Generate static tree attention mask for Medusa based on predefined choices.

    This replaces the dynamic tree building used by EAGLE3. Since Medusa's tree
    structure is predetermined, we can compute the mask once during initialization.

    Args:
        medusa_choices: List of tree paths, e.g., [[0], [0,0], [1], [0,1], ...]
        device: Torch device

    Returns:
        tree_mask: Boolean tensor of shape (num_nodes, num_nodes) where
                   tree_mask[i, j] = True means node i can attend to node j
    """
    # Sort choices by (depth, indices) to get canonical ordering
    sorted_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    num_nodes = len(sorted_choices) + 1  # +1 for root (verified_id)

    # Build mapping from choice to index in sorted list
    # Index 0 is reserved for root (verified_id)
    choice_to_idx = {tuple(choice): i + 1 for i, choice in enumerate(sorted_choices)}

    # Create attention mask: each node can attend to itself and root
    tree_mask = torch.eye(num_nodes, num_nodes, dtype=torch.bool, device=device)
    tree_mask[:, 0] = True  # All nodes can attend to root

    # Add attention to ancestor nodes
    for i, choice in enumerate(sorted_choices):
        node_idx = i + 1  # +1 because index 0 is root
        if len(choice) == 1:
            # Depth 1 nodes only attend to root (already set)
            continue

        # For deeper nodes, add attention to all ancestors
        for depth in range(1, len(choice)):
            ancestor_choice = tuple(choice[:depth])
            ancestor_idx = choice_to_idx[ancestor_choice]
            tree_mask[node_idx, ancestor_idx] = True

    return tree_mask


if __name__ == "__main__":
    # Test the conversion
    import sys

    medusa_choices = [[0], [0, 0], [1], [0, 1], [2]]
    bs = 2
    topk = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test tree mask generation
    tree_mask = generate_medusa_tree_mask(medusa_choices, device)
    print("Medusa choices:", medusa_choices)
    print(f"\nTree mask shape: {tree_mask.shape}")
    print("Tree mask:")
    print(tree_mask.int())

    # Test old functions
    parent_list, top_scores_index = build_medusa_parent_and_scores(
        medusa_choices, bs, topk, device
    )

    print(f"\nParent list (2D tensor, shape={parent_list.shape}):")
    print(parent_list)
    print(f"\nTop scores index (shape={top_scores_index.shape}):")
    print(top_scores_index)

    # Verify correctness
    sorted_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    print(f"\nSorted choices: {sorted_choices}")
    print("\nExpected interpretation:")
    for i, choice in enumerate(sorted_choices):
        print(f"  Choice {i}: {choice} → topk_index={choice[-1]}")
