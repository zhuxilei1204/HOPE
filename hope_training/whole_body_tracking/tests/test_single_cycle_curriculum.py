from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "single_cycle_curriculum.py"
)
SPEC = importlib.util.spec_from_file_location("single_cycle_curriculum", MODULE)
assert SPEC and SPEC.loader
single_cycle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = single_cycle
SPEC.loader.exec_module(single_cycle)


SCHEDULE = {
    "probabilities": (0.65, 0.45, 0.25, 0.10, 0.0),
    "targeted_attempt_thresholds": (0.01, 0.03, 0.06, 0.10),
    "contact_thresholds": (0.005, 0.015, 0.03, 0.05),
    "recovery_thresholds": (0.35, 0.48, 0.60, 0.68),
    "safety_thresholds": (0.90, 0.94, 0.96, 0.98),
    "cycle_ready_thresholds": (0.0, 0.01, 0.03, 0.08),
    "min_resolved_events": 4096,
    "min_continuous_fraction": 0.40,
}


def _ability(**overrides):
    values = {
        "targeted_attempt": 0.20,
        "contact": 0.20,
        "recovery": 0.90,
        "safety": 0.995,
        "cycle_ready": 0.20,
        "resolved_events": 8192,
    }
    values.update(overrides)
    return single_cycle.SingleCycleAbility(**values)


def test_probability_decreases_only_after_all_continuous_gates_pass() -> None:
    probability, level = single_cycle.ability_driven_single_cycle_probability(
        _ability(contact=0.02, recovery=0.50, safety=0.95, cycle_ready=0.02),
        **SCHEDULE,
    )
    assert (probability, level) == (0.25, 2)

    probability, level = single_cycle.ability_driven_single_cycle_probability(
        _ability(contact=0.20, recovery=0.47),
        **SCHEDULE,
    )
    assert (probability, level) == (0.45, 1)


def test_probability_keeps_a_continuous_pool_and_waits_for_evidence() -> None:
    probability, level = single_cycle.ability_driven_single_cycle_probability(
        _ability(resolved_events=100),
        **SCHEDULE,
    )
    assert (probability, level) == (0.60, 0)


def test_invalid_probability_and_threshold_schedules_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-increasing"):
        single_cycle.ability_driven_single_cycle_probability(
            _ability(),
            **{**SCHEDULE, "probabilities": (0.4, 0.6)},
        )
    with pytest.raises(ValueError, match=r"len\(probabilities\)-1"):
        single_cycle.ability_driven_single_cycle_probability(
            _ability(),
            **{**SCHEDULE, "contact_thresholds": (0.1,)},
        )


def test_recovery_waits_for_settlement_and_ready() -> None:
    common = {
        "active": True,
        "swing_completed": True,
        "ready_consecutive_steps": 6,
        "recovery_elapsed_steps": 20,
        "required_ready_steps": 5,
        "min_recovery_steps": 10,
        "deadline_steps": 100,
    }
    assert (
        single_cycle.resolve_single_cycle_recovery(
            settlement_resolved=False,
            **common,
        )
        == single_cycle.SingleCycleResolution.NONE
    )
    assert (
        single_cycle.resolve_single_cycle_recovery(
            settlement_resolved=True,
            **common,
        )
        == single_cycle.SingleCycleResolution.SAFE
    )


def test_recovery_deadline_is_not_a_hard_failure() -> None:
    resolution = single_cycle.resolve_single_cycle_recovery(
        active=True,
        swing_completed=True,
        settlement_resolved=True,
        ready_consecutive_steps=0,
        recovery_elapsed_steps=100,
        required_ready_steps=5,
        min_recovery_steps=10,
        deadline_steps=100,
    )
    assert resolution == single_cycle.SingleCycleResolution.DEADLINE


def test_recovery_deadline_expands_with_ability() -> None:
    kwargs = {
        "deadlines_by_level": (15, 25, 40, 70, 100),
        "fallback_deadline_steps": 100,
        "min_recovery_steps": 10,
        "expected_levels": 5,
    }
    assert single_cycle.ability_driven_single_cycle_deadline(0, **kwargs) == 15
    assert single_cycle.ability_driven_single_cycle_deadline(2, **kwargs) == 40
    assert single_cycle.ability_driven_single_cycle_deadline(9, **kwargs) == 100

    with pytest.raises(ValueError, match="non-decreasing"):
        single_cycle.ability_driven_single_cycle_deadline(
            0,
            **{**kwargs, "deadlines_by_level": (15, 25, 20, 70, 100)},
        )
    with pytest.raises(ValueError, match="one value per probability level"):
        single_cycle.ability_driven_single_cycle_deadline(
            0,
            **{**kwargs, "deadlines_by_level": (15, 25)},
        )
