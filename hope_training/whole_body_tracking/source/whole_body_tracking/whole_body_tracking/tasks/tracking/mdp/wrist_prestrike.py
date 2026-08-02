"""Pure phase schedules for wrist preparation rewards."""

from __future__ import annotations

import torch


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def asymmetric_wrist_release_scale(
    true_time_to_strike: torch.Tensor,
    *,
    pre_release_start_s: float,
    strike_half_window_s: float,
    post_release_end_s: float,
    strike_scale: float,
    post_strike_full_release_s: float | None = None,
) -> torch.Tensor:
    """Return a smooth wrist-prior scale through prepare, strike, and recovery."""
    gate = planner_task_space_crossfade_gate(
        true_time_to_strike,
        pre_start_s=pre_release_start_s,
        pre_full_s=strike_half_window_s,
        post_full_s=(
            strike_half_window_s
            if post_strike_full_release_s is None
            else post_strike_full_release_s
        ),
        post_end_s=post_release_end_s,
    )
    minimum = float(strike_scale)
    if not 0.0 <= minimum <= 1.0:
        raise ValueError(f"strike_scale must be in [0, 1], got {minimum}")
    return 1.0 - (1.0 - minimum) * gate


def planner_task_space_crossfade_gate(
    true_time_to_strike: torch.Tensor,
    *,
    pre_start_s: float,
    pre_full_s: float,
    post_full_s: float,
    post_end_s: float,
) -> torch.Tensor:
    """Smoothly transfer authority to planner task space around impact.

    The gate is zero before ``pre_start_s``, reaches one at ``pre_full_s``,
    stays one through ``-post_full_s``, and returns to zero at
    ``-post_end_s``. All time arguments are positive durations.
    """
    pre_start = float(pre_start_s)
    pre_full = float(pre_full_s)
    post_full = float(post_full_s)
    post_end = float(post_end_s)
    if pre_full < 0.0:
        raise ValueError(f"pre_full_s must be non-negative, got {pre_full}")
    if post_full < 0.0:
        raise ValueError(f"post_full_s must be non-negative, got {post_full}")
    if pre_start <= pre_full:
        raise ValueError(
            "pre_start_s must be greater than pre_full_s, "
            f"got {pre_start} <= {pre_full}"
        )
    if post_end <= post_full:
        raise ValueError(
            "post_end_s must be greater than post_full_s, "
            f"got {post_end} <= {post_full}"
        )

    tts = true_time_to_strike
    gate = torch.zeros_like(tts)

    pre = (tts < pre_start) & (tts > pre_full)
    pre_progress = (pre_start - tts) / (pre_start - pre_full)
    gate = torch.where(pre, _smoothstep01(pre_progress), gate)

    full = (tts <= pre_full) & (tts >= -post_full)
    gate = torch.where(full, torch.ones_like(gate), gate)

    post = (tts < -post_full) & (tts > -post_end)
    post_progress = ((-tts) - post_full) / (post_end - post_full)
    gate = torch.where(post, 1.0 - _smoothstep01(post_progress), gate)
    return gate


def prestrike_alignment_ramp(
    true_time_to_strike: torch.Tensor,
    *,
    start_s: float,
    full_s: float,
) -> torch.Tensor:
    """Ramp from zero at ``start_s`` to one at ``full_s`` before impact."""
    start = float(start_s)
    full = float(full_s)
    if full < 0.0:
        raise ValueError(f"full_s must be non-negative, got {full}")
    if start <= full:
        raise ValueError(f"start_s must be greater than full_s, got {start} <= {full}")
    progress = (start - true_time_to_strike) / (start - full)
    active = (true_time_to_strike <= start) & (true_time_to_strike > full)
    return _smoothstep01(progress) * active.to(true_time_to_strike.dtype)
