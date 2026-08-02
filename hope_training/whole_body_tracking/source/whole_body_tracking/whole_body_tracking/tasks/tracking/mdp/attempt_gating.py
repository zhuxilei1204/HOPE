"""Pure tensor gates for attempt-conditioned strike recovery."""

from __future__ import annotations

import torch


def attempt_conditioned_phase_gate(
    current_phase: torch.Tensor,
    early_next_phase: torch.Tensor,
    current_targeted_attempt: torch.Tensor,
    previous_targeted_attempt: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """Select the attempt latch belonging to each recovery phase.

    Post-strike/hold rewards belong to the current swing. Early pre-strike
    readiness belongs to the previous swing because swing state is reset when
    the next command is sampled.
    """
    gate = current_phase | early_next_phase
    if not enabled:
        return gate
    return (current_phase & current_targeted_attempt) | (
        early_next_phase & previous_targeted_attempt
    )
