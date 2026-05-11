"""Prompt utilities for benchmark evaluation.

Provides system prompts and conversation preparation functions
to ensure consistency with training data format.

Uses the same templates as training (specforge/data/template.py).
"""

from typing import List, Dict

# Import from training templates to ensure consistency
from specforge.data.template import TEMPLATE_REGISTRY


def detect_model_family(model_path: str) -> str:
    """Detect model family from model path.

    Args:
        model_path: Path to the model (HuggingFace name or local path)

    Returns:
        Template name that matches the model family
    """
    model_path_lower = model_path.lower()
    if "llama-3" in model_path_lower or "llama3" in model_path_lower:
        return "llama3"
    elif "qwen3" in model_path_lower:
        return "qwen"  # qwen3 uses same system prompt as qwen
    elif "qwen" in model_path_lower:
        return "qwen"
    elif "vicuna" in model_path_lower:
        # Vicuna doesn't have a registered template, use default
        return "default"
    else:
        return "default"


def get_system_prompt(model_family: str) -> str:
    """Get system prompt for a model family from TEMPLATE_REGISTRY.

    Args:
        model_family: Template name (e.g., "llama3", "qwen")

    Returns:
        System prompt string from the registered template
    """
    try:
        template = TEMPLATE_REGISTRY.get(model_family)
        return template.system_prompt or ""
    except (KeyError, AttributeError):
        # Fallback for unregistered templates
        return "You are a helpful assistant."


def prepare_conversation(
    conversation: List[Dict[str, str]],
    model_family: str,
) -> List[Dict[str, str]]:
    """Prepare conversation by adding system prompt if not present.

    This ensures consistency with training data format.

    Args:
        conversation: List of message dicts with "role" and "content"
        model_family: Model family name for selecting system prompt

    Returns:
        Prepared conversation with system prompt
    """
    # If conversation already has system message, use it as-is
    if conversation and conversation[0]["role"] == "system":
        return conversation

    # Add system prompt based on model family
    system_prompt = get_system_prompt(model_family)
    return [{"role": "system", "content": system_prompt}] + conversation
