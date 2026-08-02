from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(__file__))
_PATH = os.path.join(
    _ROOT,
    "source",
    "whole_body_tracking",
    "whole_body_tracking",
    "tasks",
    "tracking",
    "mdp",
    "ready_recovery.py",
)
_SPEC = importlib.util.spec_from_file_location("hope_ready_recovery", _PATH)
ready_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ready_module
_SPEC.loader.exec_module(ready_module)

bounded_error = ready_module.bounded_error
bounded_gaussian_score = ready_module.bounded_gaussian_score
active_score_progress = ready_module.active_score_progress
directional_error_progress = ready_module.directional_error_progress
directional_score_progress = ready_module.directional_score_progress
joint_deadband_score = ready_module.joint_deadband_score
motion_lifecycle_fractions = ready_module.motion_lifecycle_fractions
ready_curriculum_should_advance = ready_module.ready_curriculum_should_advance
ready_curriculum_guards_satisfied = (
    ready_module.ready_curriculum_guards_satisfied
)
ready_curriculum_stage_value = ready_module.ready_curriculum_stage_value
strike_health_floor_progress = ready_module.strike_health_floor_progress
update_survival_milestones = ready_module.update_survival_milestones
validate_survival_milestones = ready_module.validate_survival_milestones


def test_bounded_score_has_no_preferred_point_inside_ready_region() -> None:
    value = torch.tensor([-0.02, 0.03, 0.12])
    score = bounded_gaussian_score(value, -0.02, 0.12, 0.05)
    assert torch.allclose(score, torch.ones_like(score))


def test_bounded_error_penalizes_forward_and_backward_overshoot() -> None:
    value = torch.tensor([-0.12, 0.02, 0.22])
    error = bounded_error(value, -0.02, 0.12)
    assert torch.allclose(error, torch.tensor([0.10, 0.0, 0.10]))


def test_directional_progress_rewards_only_error_reduction() -> None:
    progress = directional_error_progress(
        torch.tensor([0.10, 0.05, 0.02]),
        torch.tensor([0.05, 0.08, 0.02]),
        clip=0.05,
    )
    assert torch.allclose(progress, torch.tensor([1.0, -0.6, 0.0]))


def test_directional_score_progress_rewards_only_score_improvement() -> None:
    progress = directional_score_progress(
        torch.tensor([0.20, 0.60, 0.50]),
        torch.tensor([0.25, 0.57, 0.50]),
        clip=0.05,
    )
    assert torch.allclose(progress, torch.tensor([1.0, -0.6, 0.0]))


def test_active_score_progress_never_pays_on_ready_entry() -> None:
    progress = active_score_progress(
        previous_score=torch.tensor([0.0, 0.4, 0.7, 0.2]),
        current_score=torch.tensor([0.8, 0.5, 0.6, 0.9]),
        active=torch.tensor([True, True, True, False]),
        previous_active=torch.tensor([False, True, True, True]),
        clip=0.1,
    )
    assert torch.allclose(progress, torch.tensor([0.0, 1.0, -1.0, 0.0]))


def test_survival_milestones_pay_once_and_reset_only_after_leaving_ready() -> None:
    steps = torch.tensor([49, 74, 20, 90])
    next_index = torch.tensor([0, 1, 1, 2])
    active = torch.tensor([True, True, True, False])
    safe = torch.tensor([True, True, False, True])
    updated_steps, updated_index, event = update_survival_milestones(
        steps,
        next_index,
        active,
        safe,
        torch.tensor([50, 75, 100]),
        torch.tensor([0.1, 0.2, 0.4]),
    )
    assert torch.equal(updated_steps, torch.tensor([50, 75, 0, 0]))
    assert torch.equal(updated_index, torch.tensor([1, 2, 1, 0]))
    assert torch.allclose(event, torch.tensor([0.1, 0.2, 0.0, 0.0]))


def test_survival_milestone_contract_rejects_farmable_schedules() -> None:
    validate_survival_milestones((50, 75, 100), (0.1, 0.2, 0.4))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_survival_milestones((50, 50), (0.1, 0.2))
    with pytest.raises(ValueError, match="equal length"):
        validate_survival_milestones((50, 75), (0.1,))


def test_joint_anchor_uses_a_deadband_instead_of_one_exact_pose() -> None:
    target = torch.tensor([[0.5, -0.5]])
    tolerance = torch.tensor([[0.1, 0.2]])
    inside = joint_deadband_score(
        torch.tensor([[0.55, -0.65]]), target, tolerance, std=0.2
    )
    outside = joint_deadband_score(
        torch.tensor([[0.9, -0.9]]), target, tolerance, std=0.2
    )
    assert inside.item() == 1.0
    assert 0.0 < outside.item() < inside.item()


def test_ready_curriculum_requires_both_ability_and_resolved_samples() -> None:
    thresholds = (0.30, 0.40, 0.50)
    minimum_events = (1024, 2048, 4096)
    assert not ready_curriculum_should_advance(
        0, 4, 0.35, 1000, thresholds, minimum_events
    )
    assert not ready_curriculum_should_advance(
        0, 4, 0.25, 2048, thresholds, minimum_events
    )
    assert ready_curriculum_should_advance(
        0, 4, 0.35, 2048, thresholds, minimum_events
    )


def test_ready_curriculum_final_stage_never_advances() -> None:
    assert not ready_curriculum_should_advance(
        3,
        4,
        1.0,
        100000,
        (0.30, 0.40, 0.50),
        (1024, 2048, 4096),
    )


def test_ready_curriculum_selects_the_active_stage_value() -> None:
    assert ready_curriculum_stage_value((1.0, 0.82, 0.68, 0.55), 2) == 0.68


def test_ready_curriculum_guards_require_shadow_and_retained_hitting() -> None:
    args = {
        "shadow_success_ema": 0.18,
        "shadow_success_threshold": 0.15,
        "targeted_attempt_ema": 0.62,
        "minimum_targeted_attempt_ema": 0.50,
        "return_success_ema": 0.48,
        "minimum_return_success_ema": 0.40,
        "completed_swings": 12000,
        "minimum_completed_swings": 8192,
    }
    assert ready_curriculum_guards_satisfied(**args)

    args["shadow_success_ema"] = 0.02
    assert not ready_curriculum_guards_satisfied(**args)
    args["shadow_success_ema"] = 0.18

    args["return_success_ema"] = 0.30
    assert not ready_curriculum_guards_satisfied(**args)
    args["return_success_ema"] = 0.48

    args["completed_swings"] = 4096
    assert not ready_curriculum_guards_satisfied(**args)


def test_strike_health_floor_can_follow_targeted_attempt_capability() -> None:
    assert strike_health_floor_progress(
        "targeted_attempt", 0.0, 0.025, 0.0, 0.05, 0.01
    ) == 0.5
    assert strike_health_floor_progress(
        "targeted_attempt", 0.0, 0.08, 0.0, 0.05, 0.01
    ) == 1.0


def test_strike_health_floor_legacy_ability_source_is_unchanged() -> None:
    assert strike_health_floor_progress(
        "ability", 0.35, 0.9, 0.9, 0.05, 0.01
    ) == 0.35


def test_strike_health_floor_validates_event_thresholds_and_source() -> None:
    with pytest.raises(ValueError, match="targeted-attempt"):
        strike_health_floor_progress(
            "targeted_attempt", 0.0, 0.0, 0.0, 0.0, 0.01
        )
    with pytest.raises(ValueError, match="progress source"):
        strike_health_floor_progress(
            "unknown", 0.0, 0.0, 0.0, 0.05, 0.01
        )


def test_motion_lifecycle_fractions_interpolate_from_measured_ability() -> None:
    assert motion_lifecycle_fractions(
        True, 0.01, 0.01, 0.05, 0.60, 0.50, 0.0, 0.15
    ) == pytest.approx((0.60, 0.0, 0.40, 0.0))
    assert motion_lifecycle_fractions(
        True, 0.03, 0.01, 0.05, 0.60, 0.50, 0.0, 0.15
    ) == pytest.approx((0.55, 0.075, 0.375, 0.5))
    assert motion_lifecycle_fractions(
        True, 0.08, 0.01, 0.05, 0.60, 0.50, 0.0, 0.15
    ) == pytest.approx((0.50, 0.15, 0.35, 1.0))


def test_motion_lifecycle_fractions_preserve_fixed_profile_when_disabled() -> None:
    assert motion_lifecycle_fractions(
        False, 1.0, 0.01, 0.05, 0.45, 0.20, 0.35, 0.70
    ) == pytest.approx((0.45, 0.35, 0.20, 0.0))


def test_motion_lifecycle_fractions_reject_invalid_probability_contract() -> None:
    with pytest.raises(ValueError, match="low < high"):
        motion_lifecycle_fractions(
            True, 0.0, 0.05, 0.05, 0.6, 0.5, 0.0, 0.15
        )
    with pytest.raises(ValueError, match="sum to <= 1"):
        motion_lifecycle_fractions(
            True, 0.0, 0.01, 0.05, 0.8, 0.8, 0.3, 0.3
        )
