"""Pure helpers for bounded READY regions and directional recovery."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def bounded_error(
    value: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Distance outside a closed interval, zero inside the interval."""
    if float(lower) > float(upper):
        raise ValueError(f"lower bound {lower} exceeds upper bound {upper}")
    return (float(lower) - value).clamp_min(0.0) + (
        value - float(upper)
    ).clamp_min(0.0)


def bounded_gaussian_score(
    value: torch.Tensor,
    lower: float,
    upper: float,
    std: float,
) -> torch.Tensor:
    """Unit score in a target interval with a Gaussian falloff outside it."""
    error = bounded_error(value, lower, upper)
    return torch.exp(-torch.square(error / max(float(std), 1.0e-6)))


def joint_deadband_score(
    joint_pos: torch.Tensor,
    target: torch.Tensor,
    tolerance: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Mean-square joint score with a per-joint unpenalized deadband."""
    if joint_pos.shape != target.shape or joint_pos.shape != tolerance.shape:
        raise ValueError(
            "joint_pos, target, and tolerance must have identical shapes; "
            f"got {joint_pos.shape}, {target.shape}, and {tolerance.shape}"
        )
    error = (torch.abs(joint_pos - target) - tolerance).clamp_min(0.0)
    return torch.exp(
        -torch.mean(torch.square(error), dim=-1) / max(float(std), 1.0e-6) ** 2
    )


def directional_error_progress(
    previous_error: torch.Tensor,
    current_error: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """Normalized signed progress toward a target, clipped to ``[-1, 1]``."""
    scale = max(float(clip), 1.0e-6)
    return torch.clamp((previous_error - current_error) / scale, -1.0, 1.0)


def directional_score_progress(
    previous_score: torch.Tensor,
    current_score: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """Normalized signed progress toward a larger score, clipped to ``[-1, 1]``."""
    scale = max(float(clip), 1.0e-6)
    return torch.clamp((current_score - previous_score) / scale, -1.0, 1.0)


def active_score_progress(
    previous_score: torch.Tensor,
    current_score: torch.Tensor,
    active: torch.Tensor,
    previous_active: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    """Signed score progress with zero payoff on entry and outside the phase."""
    shapes = {
        tuple(value.shape)
        for value in (previous_score, current_score, active, previous_active)
    }
    if len(shapes) != 1:
        raise ValueError("READY potential tensors must have identical shapes")
    progress = directional_score_progress(previous_score, current_score, clip)
    return torch.where(
        active.bool() & previous_active.bool(),
        progress,
        torch.zeros_like(progress),
    )


def validate_survival_milestones(
    milestone_steps: Sequence[int],
    milestone_values: Sequence[float],
) -> None:
    """Validate a strictly increasing one-shot survival reward ladder."""
    if not milestone_steps:
        raise ValueError("READY survival milestones must not be empty")
    if len(milestone_steps) != len(milestone_values):
        raise ValueError("READY survival milestone steps and values must have equal length")
    steps = tuple(int(value) for value in milestone_steps)
    values = tuple(float(value) for value in milestone_values)
    if any(value < 1 for value in steps):
        raise ValueError("READY survival milestone steps must be positive")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("READY survival milestone steps must be strictly increasing")
    if any(value <= 0.0 for value in values):
        raise ValueError("READY survival milestone values must be positive")


def update_survival_milestones(
    consecutive_steps: torch.Tensor,
    next_milestone_index: torch.Tensor,
    active: torch.Tensor,
    safe_now: torch.Tensor,
    milestone_steps: torch.Tensor,
    milestone_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance non-farmable READY survival milestones for each environment.

    Unsafe frames reset the consecutive counter but do not let an active hold
    earn an earlier milestone twice. Leaving the READY hold resets the ladder.
    """
    shapes = {
        tuple(value.shape)
        for value in (consecutive_steps, next_milestone_index, active, safe_now)
    }
    if len(shapes) != 1:
        raise ValueError("READY survival state tensors must have identical shapes")
    if milestone_steps.ndim != 1 or milestone_values.ndim != 1:
        raise ValueError("READY survival milestone tensors must be one-dimensional")
    if milestone_steps.numel() != milestone_values.numel():
        raise ValueError("READY survival milestone tensors must have equal length")

    active = active.bool()
    safe_active = active & safe_now.bool()
    updated_steps = torch.where(
        safe_active,
        consecutive_steps + 1,
        torch.zeros_like(consecutive_steps),
    )
    updated_index = torch.where(
        active,
        next_milestone_index,
        torch.zeros_like(next_milestone_index),
    )
    event_value = torch.zeros_like(consecutive_steps, dtype=milestone_values.dtype)

    for index in range(int(milestone_steps.numel())):
        reached = (
            safe_active
            & (updated_index == index)
            & (updated_steps >= milestone_steps[index])
        )
        event_value = torch.where(reached, milestone_values[index], event_value)
        updated_index = torch.where(reached, updated_index + 1, updated_index)

    return updated_steps, updated_index, event_value


def strike_health_floor_progress(
    source: str,
    ability_level: float,
    targeted_attempt_ema: float,
    contact_ema: float,
    targeted_attempt_threshold: float,
    contact_threshold: float,
) -> float:
    """Return the capability progress used to tighten strike-health shaping.

    ``ability`` preserves the historical behavior. The event-based alternatives
    decouple discovery shaping from the global curriculum, whose safety gate can
    otherwise keep a permissive health floor active indefinitely.
    """
    if source == "ability":
        value = float(ability_level)
    elif source == "targeted_attempt":
        threshold = float(targeted_attempt_threshold)
        if threshold <= 0.0:
            raise ValueError("targeted-attempt health-floor threshold must be positive")
        value = float(targeted_attempt_ema) / threshold
    elif source == "contact":
        threshold = float(contact_threshold)
        if threshold <= 0.0:
            raise ValueError("contact health-floor threshold must be positive")
        value = float(contact_ema) / threshold
    else:
        raise ValueError(
            "impact health-floor progress source must be 'ability', "
            f"'targeted_attempt', or 'contact', got {source!r}"
        )
    return min(max(value, 0.0), 1.0)


def motion_lifecycle_fractions(
    enabled: bool,
    targeted_attempt_ema: float,
    targeted_attempt_low: float,
    targeted_attempt_high: float,
    prestrike_start: float,
    prestrike_end: float,
    recovery_start: float,
    recovery_end: float,
) -> tuple[float, float, float, float]:
    """Interpolate seeded-reset phase probabilities from measured capability."""
    low = float(targeted_attempt_low)
    high = float(targeted_attempt_high)
    if bool(enabled) and high <= low:
        raise ValueError(
            "motion lifecycle targeted-attempt bounds must satisfy low < high"
        )
    values = tuple(
        float(value)
        for value in (
            prestrike_start,
            prestrike_end,
            recovery_start,
            recovery_end,
        )
    )
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("motion lifecycle fractions must lie in [0, 1]")
    for prestrike, recovery in (
        (values[0], values[2]),
        (values[1], values[3]),
    ):
        if prestrike + recovery > 1.0 + 1.0e-6:
            raise ValueError(
                "motion lifecycle prestrike and recovery fractions must sum to <= 1"
            )
    progress = 0.0
    if bool(enabled):
        progress = min(
            max((float(targeted_attempt_ema) - low) / (high - low), 0.0),
            1.0,
        )
    prestrike = values[0] + progress * (values[1] - values[0])
    recovery = values[2] + progress * (values[3] - values[2])
    return prestrike, recovery, 1.0 - prestrike - recovery, progress


def ready_curriculum_stage_value(
    schedule: Sequence[float | int],
    level: int,
) -> float:
    """Return one ability-stage value with explicit bounds checking."""
    if not schedule:
        raise ValueError("READY curriculum schedule must not be empty")
    if int(level) < 0 or int(level) >= len(schedule):
        raise ValueError(
            f"READY curriculum level {level} is outside [0, {len(schedule) - 1}]"
        )
    return float(schedule[int(level)])


def ready_curriculum_should_advance(
    level: int,
    stage_count: int,
    success_ema: float,
    resolved_events: int,
    success_thresholds: Sequence[float],
    minimum_resolved_events: Sequence[int],
) -> bool:
    """Return whether measured READY ability unlocks the next stricter stage."""
    transitions = int(stage_count) - 1
    if transitions < 0:
        raise ValueError("READY curriculum must contain at least one stage")
    if len(success_thresholds) != transitions:
        raise ValueError(
            "READY success thresholds must contain one value per stage transition"
        )
    if len(minimum_resolved_events) != transitions:
        raise ValueError(
            "READY minimum event counts must contain one value per stage transition"
        )
    if int(level) < 0 or int(level) >= int(stage_count):
        raise ValueError(
            f"READY curriculum level {level} is outside [0, {stage_count - 1}]"
        )
    if int(level) >= transitions:
        return False
    return (
        float(success_ema) >= float(success_thresholds[int(level)])
        and int(resolved_events) >= int(minimum_resolved_events[int(level)])
    )


def ready_curriculum_guards_satisfied(
    shadow_success_ema: float,
    shadow_success_threshold: float,
    targeted_attempt_ema: float,
    minimum_targeted_attempt_ema: float,
    return_success_ema: float,
    minimum_return_success_ema: float,
    completed_swings: int,
    minimum_completed_swings: int,
) -> bool:
    """Check next-stage reachability and retained hitting ability."""
    return (
        float(shadow_success_ema) >= float(shadow_success_threshold)
        and float(targeted_attempt_ema) >= float(minimum_targeted_attempt_ema)
        and float(return_success_ema) >= float(minimum_return_success_ema)
        and int(completed_swings) >= int(minimum_completed_swings)
    )
