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
    "closed_loop_v2.py",
)
_SPEC = importlib.util.spec_from_file_location("hope_closed_loop_v2", _PATH)
closed_loop = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = closed_loop
_SPEC.loader.exec_module(closed_loop)


def _bool(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


def test_deploy_ready_hold_forces_no_ball_and_default_stand_resets() -> None:
    selected = closed_loop.deploy_ready_hold_mask(
        in_hold=_bool([True, True, True, True, False]),
        sampled_hold=_bool([False, False, True, False, True]),
        stand_episode=_bool([True, False, False, False, True]),
        default_stand_reset=_bool([False, True, False, False, False]),
        force_stand_episode=True,
        force_default_stand_reset=True,
    )
    assert selected.tolist() == [True, True, True, False, False]


def test_deploy_ready_hold_preserves_legacy_independent_sampling() -> None:
    selected = closed_loop.deploy_ready_hold_mask(
        in_hold=_bool([True, True, False]),
        sampled_hold=_bool([False, True, True]),
        stand_episode=_bool([True, False, True]),
        default_stand_reset=_bool([True, False, True]),
        force_stand_episode=False,
        force_default_stand_reset=False,
    )
    assert selected.tolist() == [False, True, False]


def test_deploy_ready_hold_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        closed_loop.deploy_ready_hold_mask(
            in_hold=_bool([True]),
            sampled_hold=_bool([True, False]),
            stand_episode=_bool([False]),
            default_stand_reset=_bool([False]),
            force_stand_episode=True,
            force_default_stand_reset=True,
        )


def test_face_center_quality_has_plateau_smooth_margin_and_zero_rim() -> None:
    radial = torch.tensor([0.0, 0.061, 0.071, 0.081, 0.10])
    quality = closed_loop.face_center_quality(
        radial,
        inner_radius=0.061,
        outer_radius=0.081,
    )
    assert quality.tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0])


def test_face_center_quality_rejects_invalid_radii() -> None:
    with pytest.raises(ValueError, match="face radii"):
        closed_loop.face_center_quality(
            torch.tensor([0.0]),
            inner_radius=0.081,
            outer_radius=0.061,
        )


def test_face_contact_regions_partition_analytic_contact_radius() -> None:
    center, rim, analytic_outer = closed_loop.face_contact_region_masks(
        torch.tensor([0.0, 0.061, 0.070, 0.081, 0.090, 0.095, 0.12]),
        inner_radius=0.061,
        outer_radius=0.081,
        contact_radius=0.095,
    )
    assert center.tolist() == [True, True, False, False, False, False, False]
    assert rim.tolist() == [False, False, True, True, False, False, False]
    assert analytic_outer.tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert not torch.any(center & rim)
    assert not torch.any(center & analytic_outer)
    assert not torch.any(rim & analytic_outer)


def test_face_contact_regions_reject_invalid_contact_radius() -> None:
    with pytest.raises(ValueError, match="face/contact radii"):
        closed_loop.face_contact_region_masks(
            torch.tensor([0.0]),
            inner_radius=0.061,
            outer_radius=0.081,
            contact_radius=0.081,
        )


def test_wire_compatible_velocity_preserves_speed_and_uses_normal() -> None:
    velocity = torch.tensor([[3.0, 4.0, 0.0], [0.0, 2.0, 0.0]])
    normal = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, -4.0]])
    coupled = closed_loop.wire_compatible_velocity(velocity, normal)
    assert torch.norm(coupled, dim=-1).tolist() == pytest.approx([5.0, 2.0])
    torch.testing.assert_close(
        coupled,
        torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, -2.0]]),
    )


def test_wire_compatible_velocity_falls_back_to_velocity_direction() -> None:
    velocity = torch.tensor([[0.0, 3.0, 4.0]])
    normal = torch.zeros_like(velocity)
    coupled = closed_loop.wire_compatible_velocity(velocity, normal)
    torch.testing.assert_close(coupled, velocity)


def test_targeted_attempt_opens_recovery_after_a_miss() -> None:
    event = closed_loop.recovery_trigger_event(
        "targeted_attempt",
        strike_fired=_bool([True, True, False]),
        targeted_attempt=_bool([True, False, True]),
        ball_contact=_bool([False, False, True]),
    )
    assert event.tolist() == [True, False, False]


def test_legacy_contact_trigger_is_unchanged() -> None:
    event = closed_loop.recovery_trigger_event(
        "contact",
        strike_fired=_bool([True, True]),
        targeted_attempt=_bool([True, False]),
        ball_contact=_bool([False, True]),
    )
    assert event.tolist() == [False, True]


def test_invalid_recovery_trigger_is_rejected() -> None:
    with pytest.raises(ValueError, match="recovery trigger"):
        closed_loop.recovery_trigger_event(
            "anything",
            strike_fired=_bool([True]),
            targeted_attempt=_bool([True]),
            ball_contact=_bool([False]),
        )


def test_recovery_phase_indices_use_ordered_time_windows() -> None:
    phases = closed_loop.recovery_phase_indices(
        torch.tensor([0, 4, 5, 14, 15, 29, 30, 70]),
        step_dt=0.02,
        boundaries_s=(0.10, 0.30, 0.60),
    )
    assert phases.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_recovery_phase_indices_reject_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        closed_loop.recovery_phase_indices(
            torch.tensor([0, 1]),
            step_dt=0.02,
            boundaries_s=(0.10, 0.10, 0.60),
        )


def test_recovery_outcome_bucket_includes_targeted_miss() -> None:
    buckets = closed_loop.recovery_outcome_bucket(
        torch.tensor([-5, -1, 0, 1, 2, 8])
    )
    assert buckets.tolist() == [0, 0, 1, 2, 3, 3]


def test_durable_ready_resolves_only_at_deadline() -> None:
    success, fail = closed_loop.durable_ready_resolution_events(
        pending=_bool([True, True, True, False]),
        elapsed_steps=torch.tensor([59, 60, 60, 60]),
        deadline_steps=60,
        consecutive_ready_steps=torch.tensor([15, 15, 14, 15]),
        required_consecutive_steps=15,
    )
    assert success.tolist() == [False, True, False, False]
    assert fail.tolist() == [False, False, True, False]


def test_terminal_quality_window_has_fixed_duration() -> None:
    active = closed_loop.terminal_quality_window_mask(
        pending=_bool([True, True, True, True, False]),
        elapsed_steps=torch.tensor([40, 41, 54, 55, 50]),
        deadline_steps=55,
        window_steps=15,
    )
    assert active.tolist() == [False, True, True, True, False]


def test_terminal_quality_window_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="within the deadline"):
        closed_loop.terminal_quality_window_mask(
            pending=_bool([True]),
            elapsed_steps=torch.tensor([1]),
            deadline_steps=10,
            window_steps=11,
        )


def test_safety_conditioned_settlement_separates_quality_and_safety() -> None:
    safe, unsafe = closed_loop.safety_conditioned_terminal_events(
        settlement_event=_bool([True, True, True, False]),
        safety_violation_latch=_bool([False, False, True, True]),
    )
    quality = torch.tensor([1.0, 0.4, 1.0, 1.0])
    tier = torch.tensor([2, 1, 2, 2])
    outcome = closed_loop.safety_conditioned_outcome_value(
        safe_settlement_event=safe,
        terminal_quality=quality,
        outcome_tier=tier,
        tier_multipliers=torch.tensor([0.5, 1.0, 1.5]),
    )
    cycle = closed_loop.safety_conditioned_cycle_value(
        safe_settlement_event=safe,
        terminal_quality=quality,
        outcome_tier=tier,
    )
    assert safe.tolist() == [True, True, False, False]
    assert unsafe.tolist() == [False, False, True, False]
    assert outcome.tolist() == pytest.approx([1.5, 0.4, 0.0, 0.0])
    assert cycle.tolist() == pytest.approx([1.0, 0.4, 0.0, 0.0])


def test_operational_settlement_has_three_exclusive_classes() -> None:
    operational, catastrophic, incomplete = (
        closed_loop.operational_terminal_events(
            settlement_event=_bool([True, True, True, False]),
            catastrophic_violation_latch=_bool(
                [False, True, False, True]
            ),
            operational_ready=_bool([True, True, False, True]),
        )
    )
    assert operational.tolist() == [True, False, False, False]
    assert catastrophic.tolist() == [False, True, False, False]
    assert incomplete.tolist() == [False, False, True, False]
    assert (
        operational.int() + catastrophic.int() + incomplete.int()
    ).tolist() == [1, 1, 1, 0]


def test_safe_miss_cannot_receive_outcome_or_cycle_value() -> None:
    safe = _bool([True])
    tier = torch.tensor([-1])
    outcome = closed_loop.safety_conditioned_outcome_value(
        safe_settlement_event=safe,
        terminal_quality=torch.tensor([1.0]),
        outcome_tier=tier,
        tier_multipliers=torch.tensor([0.5, 1.0, 1.5]),
    )
    cycle = closed_loop.safety_conditioned_cycle_value(
        safe_settlement_event=safe,
        terminal_quality=torch.tensor([1.0]),
        outcome_tier=tier,
    )
    assert outcome.item() == 0.0
    assert cycle.item() == 0.0


def test_health_floor_preserves_exploration_and_full_credit() -> None:
    value = closed_loop.health_floor_multiplier(
        torch.tensor([0.0, 0.5, 1.0]), 0.25
    )
    assert value.tolist() == pytest.approx([0.25, 0.625, 1.0])


def test_relative_point_velocity_removes_rigid_torso_motion() -> None:
    relative = closed_loop.rigid_body_relative_point_velocity(
        point_position=torch.tensor([[1.0, 0.0, 0.0]]),
        point_velocity=torch.tensor([[1.5, 4.0, 0.0]]),
        body_position=torch.zeros(1, 3),
        body_linear_velocity=torch.tensor([[1.0, 2.0, 0.0]]),
        body_angular_velocity=torch.tensor([[0.0, 0.0, 2.0]]),
    )
    torch.testing.assert_close(relative, torch.tensor([[0.5, 0.0, 0.0]]))


def test_signed_velocity_progress_rejects_idle_reverse_and_lateral_motion() -> None:
    target = torch.tensor([[2.0, 0.0, 0.0]])
    aligned = closed_loop.signed_directional_velocity_progress(target, target)
    half_speed = closed_loop.signed_directional_velocity_progress(
        torch.tensor([[1.0, 0.0, 0.0]]), target
    )
    stationary = closed_loop.signed_directional_velocity_progress(
        torch.zeros_like(target), target
    )
    reverse = closed_loop.signed_directional_velocity_progress(-target, target)
    lateral = closed_loop.signed_directional_velocity_progress(
        torch.tensor([[0.0, 2.0, 0.0]]), target
    )

    assert aligned.item() > half_speed.item() > stationary.item()
    assert stationary.item() == pytest.approx(0.0)
    assert reverse.item() < 0.0
    assert lateral.item() < 0.0


def test_planner_velocity_alignment_requires_speed_and_direction() -> None:
    target = torch.tensor([[2.0, 0.0, 0.0]])
    aligned = closed_loop.planner_velocity_alignment_score(target, target)
    too_slow = closed_loop.planner_velocity_alignment_score(
        torch.tensor([[1.0, 0.0, 0.0]]), target
    )
    wrong_way = closed_loop.planner_velocity_alignment_score(
        torch.tensor([[-2.0, 0.0, 0.0]]), target
    )
    stationary = closed_loop.planner_velocity_alignment_score(
        torch.zeros_like(target), target
    )

    assert aligned.item() == pytest.approx(1.0)
    assert aligned.item() > too_slow.item() > wrong_way.item()
    assert stationary.item() < 0.01


def test_recovered_planner_velocity_requires_controlled_recovery() -> None:
    common = {
        "speed_ratio": torch.tensor([1.0]),
        "direction_error_rad": torch.tensor([0.0]),
        "position_error": torch.tensor([0.0]),
        "impact_health_score": torch.tensor([1.0]),
    }
    controlled = closed_loop.recovered_planner_velocity_settlement_score(
        **common,
        recovery_peak_base_ang_vel=torch.tensor([0.2]),
    )
    at_budget = closed_loop.recovered_planner_velocity_settlement_score(
        **common,
        recovery_peak_base_ang_vel=torch.tensor([0.8]),
    )
    unstable = closed_loop.recovered_planner_velocity_settlement_score(
        **common,
        recovery_peak_base_ang_vel=torch.tensor([1.6]),
    )
    unhealthy = closed_loop.recovered_planner_velocity_settlement_score(
        **{**common, "impact_health_score": torch.tensor([0.0])},
        recovery_peak_base_ang_vel=torch.tensor([0.2]),
    )

    assert controlled.item() == pytest.approx(at_budget.item())
    assert controlled.item() > unstable.item()
    assert unstable.item() < 0.30
    assert 0.0 < unhealthy.item() < controlled.item()

    components = (
        closed_loop.recovered_planner_velocity_settlement_components(
            **common,
            recovery_peak_base_ang_vel=torch.tensor([0.2]),
        )
    )
    product = torch.stack(components[:-1]).prod(dim=0)
    torch.testing.assert_close(components[-1], product)


def test_recovered_planner_command_requires_normal_alignment() -> None:
    common = {
        "speed_ratio": torch.tensor([1.0]),
        "direction_error_rad": torch.tensor([0.0]),
        "position_error": torch.tensor([0.0]),
        "impact_health_score": torch.tensor([1.0]),
        "recovery_peak_base_ang_vel": torch.tensor([0.2]),
    }
    aligned = closed_loop.recovered_planner_command_settlement_score(
        **common,
        normal_error_rad=torch.tensor([0.0]),
    )
    misaligned = closed_loop.recovered_planner_command_settlement_score(
        **common,
        normal_error_rad=torch.tensor([0.6]),
    )
    assert aligned.item() == pytest.approx(1.0)
    assert 0.0 < misaligned.item() < 0.10


def test_safe_command_cycle_requires_contact_and_safe_recovery() -> None:
    value = closed_loop.safe_command_cycle_value(
        safe_settlement_event=_bool([True, True, False, True]),
        terminal_quality=torch.tensor([0.8, 0.8, 0.8, 0.5]),
        command_score=torch.tensor([0.5, 0.5, 0.5, 1.0]),
        outcome_tier=torch.tensor([-1, 0, 2, 2]),
        require_contact=True,
    )
    assert value.tolist() == pytest.approx([0.0, 0.4, 0.0, 0.5])


def test_recovery_peak_excess_increment_is_bounded_and_not_time_based() -> None:
    budget = torch.tensor([0.8, 0.8, 0.8, 0.8])
    previous = torch.tensor([0.2, 0.8, 1.4, 2.2])
    current = torch.tensor([0.8, 1.4, 1.4, 3.0])
    increment = closed_loop.recovery_peak_excess_increment(
        previous,
        current,
        budget,
        excess_std=0.6,
        max_potential=4.0,
    )
    assert increment.tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])

    first = closed_loop.recovery_peak_excess_increment(
        torch.tensor([0.8]),
        torch.tensor([1.1]),
        torch.tensor([0.8]),
    )
    second = closed_loop.recovery_peak_excess_increment(
        torch.tensor([1.1]),
        torch.tensor([1.4]),
        torch.tensor([0.8]),
    )
    full = closed_loop.recovery_peak_excess_increment(
        torch.tensor([0.8]),
        torch.tensor([1.4]),
        torch.tensor([0.8]),
    )
    torch.testing.assert_close(first + second, full)


def test_recovery_safety_envelope_reports_tail_components() -> None:
    violations = closed_loop.recovery_safety_envelope_violations(
        tilt=torch.tensor([0.1, 0.4, 0.1]),
        abs_pitch=torch.tensor([0.1, 0.1, 0.3]),
        com_x=torch.tensor([0.1, 0.1, 0.2]),
        com_y=torch.tensor([0.1, 0.1, 0.1]),
        waist_overflow=torch.tensor([0.2, 1.2, 0.2]),
        leg_overflow=torch.tensor([0.2, 0.2, 1.4]),
        base_ang_vel=torch.tensor([0.2, 0.2, 2.2]),
        max_tilt=0.35,
        max_abs_pitch=0.25,
        max_com_x=0.16,
        max_com_y=0.18,
        max_waist_overflow=1.0,
        max_leg_overflow=1.2,
        max_base_ang_vel=2.0,
    )
    (
        any_bad,
        tilt_bad,
        pitch_bad,
        com_bad,
        waist_bad,
        leg_bad,
        base_ang_bad,
        support_bad,
    ) = violations
    assert any_bad.tolist() == [False, True, True]
    assert tilt_bad.tolist() == [False, True, False]
    assert pitch_bad.tolist() == [False, False, True]
    assert com_bad.tolist() == [False, False, True]
    assert waist_bad.tolist() == [False, True, False]
    assert leg_bad.tolist() == [False, False, True]
    assert base_ang_bad.tolist() == [False, False, True]
    assert support_bad.tolist() == [False, True, True]


def test_outcome_tier_multiplier_does_not_reward_a_miss() -> None:
    value = closed_loop.outcome_tier_multiplier(
        torch.tensor([-1, 0, 1, 2]),
        torch.tensor([0.5, 1.0, 1.5]),
    )
    assert value.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.5])


def test_achieved_outcome_tier_keeps_highest_physical_result() -> None:
    tier = closed_loop.achieved_outcome_tier(
        contact=_bool([False, True, True, True]),
        net_cross=_bool([False, False, True, True]),
        opponent_bounce=_bool([False, False, False, True]),
    )
    assert tier.tolist() == [-1, 0, 1, 2]


def test_achieved_outcome_tier_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        closed_loop.achieved_outcome_tier(
            contact=_bool([True]),
            net_cross=_bool([True, False]),
            opponent_bounce=_bool([False]),
        )


def test_lifecycle_hold_gate_completion_wins_at_deadline() -> None:
    keep, released, timeout = closed_loop.lifecycle_hold_gate(
        active=_bool([True, True, True, False]),
        complete=_bool([False, True, False, True]),
        elapsed_steps=torch.tensor([4, 5, 5, 8]),
        deadline_steps=5,
    )
    assert keep.tolist() == [True, False, False, False]
    assert released.tolist() == [False, True, False, False]
    assert timeout.tolist() == [False, False, True, False]


def test_station_relocation_terminal_reset_is_a_safety_failure() -> None:
    outcomes = closed_loop.station_relocation_resolution(
        rehearsal=_bool([True, True, True, False]),
        hold_seen=_bool([True, True, True, True]),
        arrived=_bool([True, True, False, True]),
        settled=_bool([True, False, False, True]),
        contact=_bool([True, False, False, True]),
        unsafe=_bool([False, False, True, False]),
        valid=_bool([True, False, True, True]),
        terminal_reset=True,
    )
    resolved, arrival, settled, contact, safety = outcomes
    assert resolved.tolist() == [True, False, True, False]
    assert arrival.tolist() == [True, False, False, False]
    assert settled.tolist() == [True, False, False, False]
    assert contact.tolist() == [True, False, False, False]
    assert safety.tolist() == [False, False, False, False]


def test_station_relocation_normal_wrap_uses_measured_safety() -> None:
    outcomes = closed_loop.station_relocation_resolution(
        rehearsal=_bool([True, True]),
        hold_seen=_bool([True, True]),
        arrived=_bool([True, True]),
        settled=_bool([True, True]),
        contact=_bool([True, True]),
        unsafe=_bool([False, True]),
        valid=_bool([True, True]),
        terminal_reset=False,
    )
    assert outcomes[-1].tolist() == [True, False]


def test_curriculum_range_interpolates_and_clamps_level() -> None:
    assert closed_loop.interpolate_curriculum_range(
        (0.9, 1.4), (0.8, 2.2), -1.0
    ) == pytest.approx((0.9, 1.4))
    assert closed_loop.interpolate_curriculum_range(
        (0.9, 1.4), (0.8, 2.2), 0.5
    ) == pytest.approx((0.85, 1.8))
    assert closed_loop.interpolate_curriculum_range(
        (0.9, 1.4), (0.8, 2.2), 2.0
    ) == pytest.approx((0.8, 2.2))


def test_closed_cycle_requires_every_condition() -> None:
    event = closed_loop.closed_cycle_success_event(
        recovery_success=_bool([True, True, True, True]),
        outcome_tier=torch.tensor([1, 0, 2, 2]),
        healthy_net_cross=_bool([True, True, False, True]),
        unsafe=_bool([False, False, False, True]),
    )
    assert event.tolist() == [True, False, False, False]


def test_lifecycle_priority_is_deterministic() -> None:
    phase = closed_loop.lifecycle_phase_ids(
        no_command=_bool([True, False, False, False, False, False]),
        time_to_strike=torch.tensor([1.0, 0.8, 0.1, 0.0, -0.1, -0.5]),
        strike_window=_bool([False, False, False, True, False, False]),
        recovery_pending=_bool([False, False, False, True, False, True]),
        recovery_success=_bool([False, False, False, False, False, False]),
    )
    assert phase.tolist() == [
        int(closed_loop.ClosedLoopPhase.READY_NO_COMMAND),
        int(closed_loop.ClosedLoopPhase.COMMAND_ACQUIRE),
        int(closed_loop.ClosedLoopPhase.PRE_STRIKE),
        int(closed_loop.ClosedLoopPhase.STRIKE),
        int(closed_loop.ClosedLoopPhase.FOLLOW_THROUGH),
        int(closed_loop.ClosedLoopPhase.RECOVERY),
    ]
    one_hot = closed_loop.one_hot_lifecycle(phase)
    assert one_hot.shape == (6, 7)
    assert torch.all(one_hot.sum(dim=-1) == 1)
