"""Pure contracts for the ability-driven single-cycle reset curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


@dataclass(frozen=True)
class SingleCycleAbility:
    targeted_attempt: float
    contact: float
    recovery: float
    safety: float
    cycle_ready: float
    resolved_events: int


class SingleCycleResolution(IntEnum):
    NONE = 0
    SAFE = 1
    DEADLINE = 2


def _validated_schedule(
    probabilities: Sequence[float],
    targeted_attempt_thresholds: Sequence[float],
    contact_thresholds: Sequence[float],
    recovery_thresholds: Sequence[float],
    safety_thresholds: Sequence[float],
    cycle_ready_thresholds: Sequence[float],
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    probs = tuple(float(value) for value in probabilities)
    if not probs:
        raise ValueError("single-cycle probability schedule must not be empty")
    if any(value < 0.0 or value > 1.0 for value in probs):
        raise ValueError(f"single-cycle probabilities must be in [0, 1], got {probs}")
    if any(right > left for left, right in zip(probs, probs[1:])):
        raise ValueError(f"single-cycle probabilities must be non-increasing, got {probs}")

    schedules = tuple(
        tuple(float(value) for value in values)
        for values in (
            targeted_attempt_thresholds,
            contact_thresholds,
            recovery_thresholds,
            safety_thresholds,
            cycle_ready_thresholds,
        )
    )
    expected = len(probs) - 1
    if any(len(values) != expected for values in schedules):
        lengths = tuple(len(values) for values in schedules)
        raise ValueError(
            "each single-cycle ability threshold schedule must have "
            f"len(probabilities)-1={expected} entries, got {lengths}"
        )
    if any(any(right < left for left, right in zip(values, values[1:])) for values in schedules):
        raise ValueError("single-cycle ability thresholds must be non-decreasing")
    return probs, schedules


def ability_driven_single_cycle_probability(
    ability: SingleCycleAbility,
    *,
    probabilities: Sequence[float],
    targeted_attempt_thresholds: Sequence[float],
    contact_thresholds: Sequence[float],
    recovery_thresholds: Sequence[float],
    safety_thresholds: Sequence[float],
    cycle_ready_thresholds: Sequence[float],
    min_resolved_events: int,
    min_continuous_fraction: float,
) -> tuple[float, int]:
    """Return reset probability and attained gate level from continuous-pool ability."""
    probs, schedules = _validated_schedule(
        probabilities,
        targeted_attempt_thresholds,
        contact_thresholds,
        recovery_thresholds,
        safety_thresholds,
        cycle_ready_thresholds,
    )
    minimum_continuous = float(min_continuous_fraction)
    if not 0.0 <= minimum_continuous <= 1.0:
        raise ValueError(
            f"min_continuous_fraction must be in [0, 1], got {minimum_continuous}"
        )

    level = 0
    if int(ability.resolved_events) >= max(int(min_resolved_events), 0):
        values = (
            float(ability.targeted_attempt),
            float(ability.contact),
            float(ability.recovery),
            float(ability.safety),
            float(ability.cycle_ready),
        )
        for index in range(len(probs) - 1):
            if all(value >= thresholds[index] for value, thresholds in zip(values, schedules)):
                level = index + 1
            else:
                break

    max_probability = 1.0 - minimum_continuous
    return min(probs[level], max_probability), level


def ability_driven_single_cycle_deadline(
    level: int,
    *,
    deadlines_by_level: Sequence[int],
    fallback_deadline_steps: int,
    min_recovery_steps: int,
    expected_levels: int | None = None,
) -> int:
    """Return a non-decreasing recovery window for the attained ability level."""
    deadlines = tuple(int(value) for value in deadlines_by_level)
    if not deadlines:
        deadlines = (int(fallback_deadline_steps),)
    if expected_levels is not None and len(deadlines) not in (1, int(expected_levels)):
        raise ValueError(
            "single-cycle deadline schedule must contain one fallback value or "
            f"one value per probability level ({expected_levels}), got {len(deadlines)}"
        )
    if any(value <= int(min_recovery_steps) for value in deadlines):
        raise ValueError(
            "each single-cycle deadline must exceed min_recovery_steps, "
            f"got deadlines={deadlines}, min={min_recovery_steps}"
        )
    if any(right < left for left, right in zip(deadlines, deadlines[1:])):
        raise ValueError(
            f"single-cycle deadlines must be non-decreasing, got {deadlines}"
        )
    index = min(max(int(level), 0), len(deadlines) - 1)
    return deadlines[index]


def resolve_single_cycle_recovery(
    *,
    active: bool,
    swing_completed: bool,
    settlement_resolved: bool,
    ready_consecutive_steps: int,
    recovery_elapsed_steps: int,
    required_ready_steps: int,
    min_recovery_steps: int,
    deadline_steps: int,
) -> SingleCycleResolution:
    """Resolve only after reward settlement; deadline is a clean but failed timeout."""
    required_ready = max(int(required_ready_steps), 1)
    minimum_recovery = max(int(min_recovery_steps), 0)
    deadline = int(deadline_steps)
    if deadline <= minimum_recovery:
        raise ValueError(
            f"deadline_steps ({deadline}) must exceed min_recovery_steps ({minimum_recovery})"
        )
    if not (active and swing_completed and settlement_resolved):
        return SingleCycleResolution.NONE
    elapsed = int(recovery_elapsed_steps)
    if elapsed >= minimum_recovery and int(ready_consecutive_steps) >= required_ready:
        return SingleCycleResolution.SAFE
    if elapsed >= deadline:
        return SingleCycleResolution.DEADLINE
    return SingleCycleResolution.NONE
