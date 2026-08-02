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
    "trunk_waist_stability.py",
)
_SPEC = importlib.util.spec_from_file_location("hope_trunk_waist_stability", _PATH)
stability_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stability_module
_SPEC.loader.exec_module(stability_module)

one_sided_trunk_waist_score = stability_module.one_sided_trunk_waist_score


def test_forward_pose_is_not_penalized() -> None:
    score = one_sided_trunk_waist_score(
        torch.tensor([0.20]),
        torch.tensor([0.30]),
        torch.tensor([0.0]),
        torso_backlean_tolerance=0.05,
        torso_backlean_std=0.10,
        waist_backfold_tolerance=0.10,
        waist_backfold_std=0.10,
    )
    assert torch.allclose(score, torch.ones_like(score))


def test_backward_torso_and_waist_reduce_score() -> None:
    score = one_sided_trunk_waist_score(
        torch.tensor([-0.15]),
        torch.tensor([-0.20]),
        torch.tensor([0.0]),
        torso_backlean_tolerance=0.05,
        torso_backlean_std=0.10,
        waist_backfold_tolerance=0.10,
        waist_backfold_std=0.10,
    )
    assert 0.0 < score.item() < 0.20


def test_recovery_rate_penalty_is_optional() -> None:
    pose_only = one_sided_trunk_waist_score(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torso_backlean_tolerance=0.05,
        torso_backlean_std=0.10,
        waist_backfold_tolerance=0.10,
        waist_backfold_std=0.10,
    )
    damped = one_sided_trunk_waist_score(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torso_backlean_tolerance=0.05,
        torso_backlean_std=0.10,
        waist_backfold_tolerance=0.10,
        waist_backfold_std=0.10,
        torso_pitch_rate=torch.tensor([1.0]),
        waist_pitch_rate=torch.tensor([1.0]),
        torso_pitch_rate_std=1.0,
        waist_pitch_rate_std=1.0,
        rate_weight=0.25,
    )
    assert pose_only.item() == 1.0
    assert damped.item() < pose_only.item()


def test_positive_rate_weight_requires_rate_tensors() -> None:
    try:
        one_sided_trunk_waist_score(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torso_backlean_tolerance=0.05,
            torso_backlean_std=0.10,
            waist_backfold_tolerance=0.10,
            waist_backfold_std=0.10,
            rate_weight=0.25,
        )
    except ValueError as exc:
        assert "rate tensors" in str(exc)
    else:
        raise AssertionError("missing rate tensors should fail")
