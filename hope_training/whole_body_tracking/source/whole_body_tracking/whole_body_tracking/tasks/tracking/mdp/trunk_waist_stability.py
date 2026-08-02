"""Pure scoring helpers for one-sided trunk and waist stability."""

from __future__ import annotations

import torch


def one_sided_trunk_waist_score(
    torso_gravity_x: torch.Tensor,
    waist_pitch: torch.Tensor,
    waist_default: torch.Tensor,
    *,
    torso_backlean_tolerance: float,
    torso_backlean_std: float,
    waist_backfold_tolerance: float,
    waist_backfold_std: float,
    torso_pitch_rate: torch.Tensor | None = None,
    waist_pitch_rate: torch.Tensor | None = None,
    torso_pitch_rate_std: float = 1.0,
    waist_pitch_rate_std: float = 1.0,
    rate_weight: float = 0.0,
) -> torch.Tensor:
    """Score only backward pose violations, optionally damping recovery rates.

    Forward trunk lean and forward waist motion remain unpenalized. Rate
    regularization is symmetric and is intended only for recovery/hold phases.
    """
    torso_violation = (
        -torso_gravity_x - float(torso_backlean_tolerance)
    ).clamp_min(0.0)
    waist_violation = (
        waist_default - waist_pitch - float(waist_backfold_tolerance)
    ).clamp_min(0.0)

    energy = torch.square(
        torso_violation / max(float(torso_backlean_std), 1.0e-6)
    )
    energy = energy + torch.square(
        waist_violation / max(float(waist_backfold_std), 1.0e-6)
    )

    rate_scale = max(float(rate_weight), 0.0)
    if rate_scale > 0.0:
        if torso_pitch_rate is None or waist_pitch_rate is None:
            raise ValueError("rate tensors are required when rate_weight is positive")
        rate_energy = torch.square(
            torso_pitch_rate / max(float(torso_pitch_rate_std), 1.0e-6)
        )
        rate_energy = rate_energy + torch.square(
            waist_pitch_rate / max(float(waist_pitch_rate_std), 1.0e-6)
        )
        energy = energy + rate_scale * rate_energy

    return torch.exp(-energy)
