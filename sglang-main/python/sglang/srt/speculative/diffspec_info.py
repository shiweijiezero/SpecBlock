"""
DiffSpec Speculative Decoding Info Classes.

Extends EAGLE's info classes to support multi-condition tokens.
"""

from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)


@dataclass
class DiffSpecDraftInput(EagleDraftInput):
    """
    DiffSpec Draft Input with multi-condition token support.

    Extends EagleDraftInput with:
    - multi_conditions: M historical condition tokens (batch_size, M, H)
    - condition_history: Ring buffer of past M hidden states per request
    """

    # Multi-condition tokens: (batch_size, M, H)
    # M = num_condition_tokens (default 3)
    multi_conditions: torch.Tensor = None

    # Number of condition tokens
    num_condition_tokens: int = 3

    @classmethod
    def create_idle_input(
        cls,
        device: str,
        hidden_size: int,
        dtype: torch.dtype,
        topk: int,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.LAST,
        num_condition_tokens: int = 3,
    ) -> "DiffSpecDraftInput":
        """Create idle input for empty batch."""
        return cls(
            topk_p=torch.empty((0, topk), dtype=dtype, device=device),
            topk_index=torch.empty((0, topk), dtype=torch.int64, device=device),
            hidden_states=torch.empty((0, hidden_size), dtype=dtype, device=device),
            verified_id=torch.empty((0,), dtype=torch.int64, device=device),
            capture_hidden_mode=capture_hidden_mode,
            multi_conditions=torch.empty(
                (0, num_condition_tokens, hidden_size), dtype=dtype, device=device
            ),
            num_condition_tokens=num_condition_tokens,
        )

    def build_multi_conditions(
        self,
        hidden_states: torch.Tensor,
        fc_layer: torch.nn.Linear,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build multi-condition tokens from hidden states.

        For the first step or when history is not available,
        replicates the current projected hidden state M times.

        For subsequent steps, uses the ring buffer of historical states.

        Args:
            hidden_states: (batch_size, 3*H) - concatenated 3-layer hidden states
            fc_layer: Linear layer for 3*H -> H projection
            history: Optional (batch_size, M, H) historical conditions

        Returns:
            multi_conditions: (batch_size, M, H)
        """
        batch_size = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1] // 3  # Assuming 3*H input

        # Project to H
        projected = fc_layer(hidden_states)  # (batch_size, H)

        if history is not None and history.shape[0] == batch_size:
            # Shift history and add new state
            # history[:, :-1] = history[:, 1:]  # Shift left
            # history[:, -1] = projected  # Add new state
            new_history = torch.cat([
                history[:, 1:, :],  # Drop oldest
                projected.unsqueeze(1),  # Add newest
            ], dim=1)
            return new_history
        else:
            # First step: replicate current state M times
            return projected.unsqueeze(1).expand(-1, self.num_condition_tokens, -1).clone()


@dataclass
class DiffSpecVerifyInput(EagleVerifyInput):
    """
    DiffSpec Verify Input - extends EAGLE's verify input.

    Currently identical to EagleVerifyInput as we use the same
    tree verification mechanism.
    """
    pass


@dataclass
class DiffSpecVerifyOutput(EagleVerifyOutput):
    """
    DiffSpec Verify Output - extends EAGLE's verify output.

    Adds multi-condition state for next iteration.
    """

    # Updated multi-conditions after verification
    multi_conditions: torch.Tensor = None
