from __future__ import annotations

import importlib.util
import os
import sys

import torch
from torch import nn

_ROOT = os.path.dirname(os.path.dirname(__file__))
_PATH = os.path.join(
    _ROOT,
    "source",
    "whole_body_tracking",
    "whole_body_tracking",
    "utils",
    "actor_anchor.py",
)
_SPEC = importlib.util.spec_from_file_location("hope_actor_anchor", _PATH)
anchor_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = anchor_module
_SPEC.loader.exec_module(anchor_module)

ActorParameterAnchor = anchor_module.ActorParameterAnchor


def test_actor_anchor_adds_reference_gradient() -> None:
    actor = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        actor.weight.copy_(torch.tensor([[2.0, -1.0]]))
    reference = {"weight": torch.tensor([[1.0, 1.0]])}
    anchor = ActorParameterAnchor(actor, reference, coefficient=0.5)

    actor.weight.sum().backward()

    expected = torch.ones_like(actor.weight) + 0.5 * (
        actor.weight.detach() - reference["weight"]
    )
    assert torch.allclose(actor.weight.grad, expected)
    metrics = anchor.metrics()
    assert metrics["actor_anchor_rms"] > 0.0
    assert metrics["actor_anchor_relative_l2"] > 0.0
    anchor.close()


def test_actor_anchor_rejects_missing_reference_parameter() -> None:
    actor = nn.Linear(2, 1)
    try:
        ActorParameterAnchor(actor, {"weight": actor.weight.detach()}, coefficient=1.0)
    except ValueError as exc:
        assert "missing parameters" in str(exc)
    else:
        raise AssertionError("missing bias reference should fail")
