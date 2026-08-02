"""Pure tensor helpers for output-side actuator feasibility."""

from __future__ import annotations

import torch


def _smooth_excess(
    values: torch.Tensor,
    free_margin: float,
    delta: float,
    maximum: float,
) -> torch.Tensor:
    excess = (values - float(free_margin)).clamp_min(0.0)
    delta = max(float(delta), 1.0e-6)
    return (torch.sqrt(torch.square(excess) + delta**2) - delta).clamp_max(
        float(maximum)
    )


def actuator_envelope_components(
    requested_torque: torch.Tensor,
    joint_velocity: torch.Tensor,
    rated_torque: torch.Tensor,
    peak_torque: torch.Tensor,
    rated_speed: torch.Tensor,
    peak_speed: torch.Tensor,
    free_margin: float = 0.05,
    delta: float = 0.25,
    maximum: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rated-load, torque-speed-corner, and peak-envelope penalties."""
    eps = torch.finfo(requested_torque.dtype).eps
    torque_ratio = requested_torque.abs() / rated_torque.clamp_min(eps)
    speed_ratio = joint_velocity.abs() / rated_speed.clamp_min(eps)
    peak_torque_ratio = requested_torque.abs() / peak_torque.clamp_min(eps)
    peak_speed_ratio = joint_velocity.abs() / peak_speed.clamp_min(eps)

    torque_excess = _smooth_excess(
        torque_ratio, 1.0 + float(free_margin), delta, maximum
    )
    speed_excess = _smooth_excess(
        speed_ratio, 1.0 + float(free_margin), delta, maximum
    )
    rated_load = 0.5 * (torque_excess + speed_excess)
    torque_speed_corner = torch.sqrt(
        torch.clamp(torque_excess * speed_excess, min=0.0)
    )
    peak_envelope = 0.5 * (
        _smooth_excess(peak_torque_ratio, 1.0, delta, maximum)
        + _smooth_excess(peak_speed_ratio, 1.0, delta, maximum)
    )
    return rated_load, torque_speed_corner, peak_envelope
