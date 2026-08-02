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
    "attempt_gating.py",
)
_SPEC = importlib.util.spec_from_file_location("hope_attempt_gating", _PATH)
gating = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gating
_SPEC.loader.exec_module(gating)


def test_attempt_gate_uses_current_attempt_after_strike() -> None:
    current_phase = torch.tensor([True, True, False, False])
    early_next = torch.tensor([False, False, True, True])
    current_attempt = torch.tensor([True, False, False, False])
    previous_attempt = torch.tensor([False, False, True, False])

    result = gating.attempt_conditioned_phase_gate(
        current_phase,
        early_next,
        current_attempt,
        previous_attempt,
        True,
    )

    assert result.tolist() == [True, False, True, False]


def test_attempt_gate_preserves_legacy_behavior_when_disabled() -> None:
    current_phase = torch.tensor([True, False])
    early_next = torch.tensor([False, True])
    no_attempt = torch.zeros(2, dtype=torch.bool)

    result = gating.attempt_conditioned_phase_gate(
        current_phase,
        early_next,
        no_attempt,
        no_attempt,
        False,
    )

    assert result.tolist() == [True, True]
