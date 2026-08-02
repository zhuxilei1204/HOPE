from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(__file__))
_PATH = os.path.join(
    _ROOT,
    "source",
    "whole_body_tracking",
    "whole_body_tracking",
    "tasks",
    "tracking",
    "mdp",
    "table_workspace.py",
)
_SPEC = importlib.util.spec_from_file_location("hope_table_workspace", _PATH)
workspace = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = workspace
_SPEC.loader.exec_module(workspace)

interpolate_bounds = workspace.interpolate_bounds
hysteretic_curriculum_transition = workspace.hysteretic_curriculum_transition
event_gated_scalar_curriculum_transition = (
    workspace.event_gated_scalar_curriculum_transition
)
motion_seed_blend = workspace.motion_seed_blend
ramped_curriculum_threshold = workspace.ramped_curriculum_threshold
table_side_lateral_bounds = workspace.table_side_lateral_bounds
validate_table_workspace = workspace.validate_table_workspace
windowed_curriculum_level = workspace.windowed_curriculum_level


def test_side_ranges_cover_regulation_table_width() -> None:
    width = 1.525
    margin = 0.02
    overlap = 0.12
    forehand = table_side_lateral_bounds(width, margin, overlap, 1.0)
    backhand = table_side_lateral_bounds(width, margin, overlap, -1.0)

    assert math.isclose(forehand[0], -(width / 2.0 - margin))
    assert math.isclose(backhand[1], width / 2.0 - margin)
    assert forehand[1] == overlap
    assert backhand[0] == -overlap
    assert forehand[1] >= backhand[0]


def test_bounds_expand_from_core_to_full() -> None:
    core = (-0.55, -0.20)
    full = (-0.7425, 0.12)
    assert interpolate_bounds(core, full, 0.0) == core
    assert interpolate_bounds(core, full, 1.0) == full
    halfway = interpolate_bounds(core, full, 0.5)
    assert halfway == pytest.approx((-0.64625, -0.04))


def test_workspace_validation_rejects_core_outside_table() -> None:
    with pytest.raises(ValueError, match="contained"):
        validate_table_workspace(
            1.525,
            0.02,
            0.12,
            (-0.80, -0.20),
            (-0.15, 0.18),
        )


def test_motion_seed_fades_from_motion_to_independent_workspace() -> None:
    assert motion_seed_blend(0.0, 0.85, 0.25) == pytest.approx(0.85)
    assert motion_seed_blend(0.125, 0.85, 0.25) == pytest.approx(0.425)
    assert motion_seed_blend(0.25, 0.85, 0.25) == 0.0
    assert motion_seed_blend(1.0, 0.85, 0.25) == 0.0


def test_motion_seed_rejects_enabled_zero_length_schedule() -> None:
    with pytest.raises(ValueError, match="end_level"):
        motion_seed_blend(0.0, 0.85, 0.0)


def test_difficulty_windows_unlock_independently() -> None:
    level = 0.60
    assert windowed_curriculum_level(level, 0.10, 1.0) == pytest.approx(5.0 / 9.0)
    assert windowed_curriculum_level(level, 0.50, 1.0) == pytest.approx(0.20)
    assert windowed_curriculum_level(level, 0.60, 1.0) == 0.0
    assert windowed_curriculum_level(level, 0.75, 1.0) == 0.0


def test_physical_outcome_thresholds_ramp_without_step_jump() -> None:
    assert ramped_curriculum_threshold(0.55, 0.55, 0.90, 0.25) == 0.0
    assert ramped_curriculum_threshold(0.60, 0.55, 0.90, 0.25) == pytest.approx(
        0.0357142857
    )
    assert ramped_curriculum_threshold(0.90, 0.55, 0.90, 0.25) == 0.25


def test_curriculum_window_validation_rejects_overlap_or_reverse() -> None:
    with pytest.raises(ValueError, match="curriculum window"):
        windowed_curriculum_level(0.5, 0.6, 0.6)
    with pytest.raises(ValueError, match="curriculum window"):
        windowed_curriculum_level(0.5, 0.8, 0.4)


def test_relocation_curriculum_requires_persistent_advance_checks() -> None:
    level, advance, regress, changed = hysteretic_curriculum_transition(
        level=0,
        max_level=3,
        good=True,
        bad=False,
        resolved_events=20_000,
        min_resolved_events=12_288,
        advance_streak=5,
        regress_streak=0,
        required_advance_checks=6,
        required_regress_checks=4,
    )

    assert (level, advance, regress, changed) == (1, 0, 0, True)


def test_relocation_curriculum_rolls_back_on_arrival_or_settle_collapse() -> None:
    state = (2, 0, 0, False)
    for regress_streak in range(3):
        state = hysteretic_curriculum_transition(
            level=2,
            max_level=4,
            good=False,
            bad=True,
            resolved_events=40_000,
            min_resolved_events=12_288,
            advance_streak=0,
            regress_streak=regress_streak,
            required_advance_checks=6,
            required_regress_checks=4,
        )
        assert state == (2, 0, regress_streak + 1, False)

    state = hysteretic_curriculum_transition(
        level=2,
        max_level=4,
        good=False,
        bad=True,
        resolved_events=40_000,
        min_resolved_events=12_288,
        advance_streak=0,
        regress_streak=3,
        required_advance_checks=6,
        required_regress_checks=4,
    )
    assert state == (1, 0, 0, True)


def test_relocation_curriculum_does_not_regress_level_zero() -> None:
    state = hysteretic_curriculum_transition(
        level=0,
        max_level=4,
        good=False,
        bad=True,
        resolved_events=40_000,
        min_resolved_events=12_288,
        advance_streak=0,
        regress_streak=9,
        required_advance_checks=6,
        required_regress_checks=4,
    )
    assert state == (0, 0, 0, False)


def test_scalar_curriculum_waits_for_disjoint_event_batch() -> None:
    state = event_gated_scalar_curriculum_transition(
        level=0.2,
        good=True,
        bad=False,
        resolved_events=63,
        min_resolved_events=64,
        advance_streak=0,
        regress_streak=0,
        required_advance_checks=2,
        required_regress_checks=1,
        advance_rate=0.1,
        regress_rate=0.2,
    )
    assert state == (0.2, 0, 0, False, False)

    first_check = event_gated_scalar_curriculum_transition(
        level=0.2,
        good=True,
        bad=False,
        resolved_events=64,
        min_resolved_events=64,
        advance_streak=0,
        regress_streak=0,
        required_advance_checks=2,
        required_regress_checks=1,
        advance_rate=0.1,
        regress_rate=0.2,
    )
    assert first_check == (0.2, 1, 0, False, True)

    second_check = event_gated_scalar_curriculum_transition(
        level=first_check[0],
        good=True,
        bad=False,
        resolved_events=64,
        min_resolved_events=64,
        advance_streak=first_check[1],
        regress_streak=first_check[2],
        required_advance_checks=2,
        required_regress_checks=1,
        advance_rate=0.1,
        regress_rate=0.2,
    )
    assert second_check[0] == pytest.approx(0.3)
    assert second_check[1:] == (0, 0, True, True)


def test_scalar_curriculum_safety_regression_has_priority() -> None:
    state = event_gated_scalar_curriculum_transition(
        level=0.5,
        good=True,
        bad=True,
        resolved_events=128,
        min_resolved_events=64,
        advance_streak=2,
        regress_streak=0,
        required_advance_checks=2,
        required_regress_checks=1,
        advance_rate=0.05,
        regress_rate=0.15,
    )
    assert state[0] == pytest.approx(0.35)
    assert state[1:] == (0, 0, True, True)
