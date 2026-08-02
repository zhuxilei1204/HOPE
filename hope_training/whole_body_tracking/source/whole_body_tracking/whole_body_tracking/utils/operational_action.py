"""Pure-Torch operational-action costs shared by training and offline audits."""

from __future__ import annotations

import torch


def bounded_squared_excess(
    values: torch.Tensor,
    scale: float,
    maximum: float,
) -> torch.Tensor:
    """Convert a non-negative normalized excess to a smooth bounded cost."""
    divisor = max(float(scale), 1.0e-6)
    return torch.square(values.clamp_min(0.0) / divisor).clamp_max(
        float(maximum)
    )


def robust_log_excess(
    values: torch.Tensor,
    scale: float,
    maximum: float,
) -> torch.Tensor:
    """Return a non-saturating log cost over positive normalized excess.

    Compared with the bounded squared cost this retains ordering across the
    large acceleration excess produced by stochastic policy exploration.
    ``maximum`` remains as a final guard against unbounded reward variance.
    """
    divisor = max(float(scale), 1.0e-6)
    return torch.log1p(values.clamp_min(0.0) / divisor).clamp_max(
        float(maximum)
    )


def grouped_joint_cost(
    per_joint_cost: torch.Tensor,
    *,
    waist_scale: float,
    other_upper_scale: float,
    right_arm_scale: float,
    leg_scale: float,
) -> torch.Tensor:
    """Aggregate canonical A3 joint costs without diluting small body groups."""
    if per_joint_cost.ndim != 2 or per_joint_cost.shape[1] != 31:
        raise ValueError(
            "grouped_joint_cost expects a [batch, 31] canonical A3 tensor"
        )

    def _mean(start: int, end: int) -> torch.Tensor:
        return per_joint_cost[:, start:end].mean(dim=-1)

    return (
        float(waist_scale) * _mean(0, 3)
        + float(other_upper_scale) * _mean(3, 12)
        + float(right_arm_scale) * _mean(12, 19)
        + float(leg_scale) * _mean(19, 31)
    )
