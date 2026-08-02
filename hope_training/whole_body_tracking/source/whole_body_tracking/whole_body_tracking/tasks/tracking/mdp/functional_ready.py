"""Pure helpers for sustained, multiplicative READY-state scoring."""

from __future__ import annotations

import torch


def weighted_geometric_mean(
    components: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return a weighted geometric mean over the last dimension."""
    if components.shape[-1] != weights.numel():
        raise ValueError(
            f"component count ({components.shape[-1]}) does not match weights ({weights.numel()})"
        )
    normalized = weights.to(device=components.device, dtype=components.dtype)
    normalized = normalized / normalized.sum().clamp_min(float(eps))
    bounded = components.clamp(min=float(eps), max=1.0)
    return torch.exp(torch.sum(torch.log(bounded) * normalized, dim=-1))


def update_consecutive_steps(
    previous_steps: torch.Tensor,
    ready_now: torch.Tensor,
) -> torch.Tensor:
    """Increment consecutive READY steps and reset immediately on any failed frame."""
    return torch.where(
        ready_now,
        previous_steps + 1,
        torch.zeros_like(previous_steps),
    )


def smooth_overflow_penalty(
    overflow_rms: torch.Tensor,
    free_margin: float = 0.05,
    delta: float = 0.25,
    maximum: float = 4.0,
) -> torch.Tensor:
    """Robust penalty for impossible raw actions without suppressing feasible compensation."""
    excess = (overflow_rms - float(free_margin)).clamp_min(0.0)
    delta = max(float(delta), 1.0e-6)
    penalty = torch.sqrt(torch.square(excess) + delta**2) - delta
    return penalty.clamp_max(float(maximum))
