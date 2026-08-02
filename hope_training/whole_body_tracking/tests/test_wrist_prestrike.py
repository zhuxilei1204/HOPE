from __future__ import annotations

import importlib.util
import os
import sys

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
    "wrist_prestrike.py",
)
_SPEC = importlib.util.spec_from_file_location("hope_wrist_prestrike", _PATH)
schedule = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = schedule
_SPEC.loader.exec_module(schedule)


def test_asymmetric_release_has_expected_boundaries() -> None:
    tts = torch.tensor([0.50, 0.35, 0.245, 0.14, 0.0, -0.14, -0.22, -0.30, -0.50])
    scale = schedule.asymmetric_wrist_release_scale(
        tts,
        pre_release_start_s=0.35,
        strike_half_window_s=0.14,
        post_release_end_s=0.30,
        strike_scale=0.18,
    )
    assert torch.allclose(scale[[0, 1, 7, 8]], torch.ones(4), atol=1.0e-6)
    assert torch.allclose(scale[[3, 4, 5]], torch.full((3,), 0.18), atol=1.0e-6)
    assert 0.18 < scale[2].item() < 1.0
    assert 0.18 < scale[6].item() < 1.0


def test_prestrike_alignment_is_zero_outside_prepare_window() -> None:
    tts = torch.tensor([0.50, 0.35, 0.235, 0.121, 0.12, 0.0, -0.10])
    ramp = schedule.prestrike_alignment_ramp(tts, start_s=0.35, full_s=0.12)
    assert ramp[0].item() == 0.0
    assert ramp[1].item() == 0.0
    assert 0.0 < ramp[2].item() < 1.0
    assert 0.99 < ramp[3].item() <= 1.0
    assert torch.all(ramp[4:] == 0.0)


def test_asymmetric_crossfade_supports_shorter_post_impact_full_window() -> None:
    tts = torch.tensor([0.50, 0.40, 0.275, 0.15, 0.0, -0.08, -0.15, -0.22, -0.30])
    gate = schedule.planner_task_space_crossfade_gate(
        tts,
        pre_start_s=0.40,
        pre_full_s=0.15,
        post_full_s=0.08,
        post_end_s=0.22,
    )
    assert torch.allclose(gate[[0, 1, 7, 8]], torch.zeros(4), atol=1.0e-6)
    assert torch.allclose(gate[[3, 4, 5]], torch.ones(3), atol=1.0e-6)
    assert 0.0 < gate[2].item() < 1.0
    assert 0.0 < gate[6].item() < 1.0

    release = schedule.asymmetric_wrist_release_scale(
        tts,
        pre_release_start_s=0.40,
        strike_half_window_s=0.15,
        post_strike_full_release_s=0.08,
        post_release_end_s=0.22,
        strike_scale=0.05,
    )
    assert torch.allclose(release, 1.0 - 0.95 * gate, atol=1.0e-6)


def test_invalid_release_schedule_fails() -> None:
    try:
        schedule.asymmetric_wrist_release_scale(
            torch.tensor([0.0]),
            pre_release_start_s=0.10,
            strike_half_window_s=0.14,
            post_release_end_s=0.30,
            strike_scale=0.18,
        )
    except ValueError as exc:
        assert "pre_start_s" in str(exc)
    else:
        raise AssertionError("invalid wrist release schedule should fail")
