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
    "functional_ready.py",
)
_SPEC = importlib.util.spec_from_file_location("hope_functional_ready", _PATH)
functional_ready = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = functional_ready
_SPEC.loader.exec_module(functional_ready)


def test_geometric_ready_cannot_hide_a_failed_component() -> None:
    components = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.01, 1.0, 1.0],
        ]
    )
    weights = torch.ones(5)

    score = functional_ready.weighted_geometric_mean(components, weights)

    assert torch.isclose(score[0], torch.tensor(1.0))
    assert score[1] < 0.45


def test_consecutive_ready_resets_on_single_bad_frame() -> None:
    steps = torch.tensor([14, 14, 0], dtype=torch.long)
    ready = torch.tensor([True, False, True])

    updated = functional_ready.update_consecutive_steps(steps, ready)

    assert updated.tolist() == [15, 0, 1]


def test_overflow_penalty_has_free_region_and_cap() -> None:
    overflow = torch.tensor([0.0, 0.05, 0.30, 100.0])

    penalty = functional_ready.smooth_overflow_penalty(
        overflow,
        free_margin=0.05,
        delta=0.25,
        maximum=4.0,
    )

    assert penalty[0] == 0.0
    assert penalty[1] == 0.0
    assert 0.0 < penalty[2] < 0.25
    assert penalty[3] == 4.0
