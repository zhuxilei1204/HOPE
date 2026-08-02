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


def test_actor_anchor_can_exempt_appended_input_columns() -> None:
    actor = nn.Sequential(nn.Linear(4, 2, bias=False), nn.Tanh(), nn.Linear(2, 1))
    reference = {
        name: torch.zeros_like(parameter)
        for name, parameter in actor.named_parameters()
    }
    with torch.no_grad():
        actor[0].weight.fill_(1.0)
        actor[2].weight.fill_(1.0)
        actor[2].bias.fill_(1.0)

    anchor = ActorParameterAnchor(
        actor,
        reference,
        coefficient=0.5,
        first_layer_input_exempt_start=2,
    )
    sum(parameter.sum() for parameter in actor.parameters()).backward()

    assert torch.allclose(actor[0].weight.grad[:, :2], torch.full((2, 2), 1.5))
    assert torch.allclose(actor[0].weight.grad[:, 2:], torch.ones((2, 2)))
    assert torch.allclose(actor[2].weight.grad, torch.full((1, 2), 1.5))
    assert torch.allclose(actor[2].bias.grad, torch.full((1,), 1.5))
    metrics = anchor.metrics()
    assert metrics["actor_anchor_rms"] > 0.0
    assert metrics["actor_anchor_exempt_rms"] > 0.0
    anchor.close()


def test_actor_anchor_can_weakly_anchor_appended_input_columns() -> None:
    actor = nn.Sequential(nn.Linear(4, 2, bias=False), nn.Tanh(), nn.Linear(2, 1))
    reference = {
        name: torch.zeros_like(parameter)
        for name, parameter in actor.named_parameters()
    }
    with torch.no_grad():
        actor[0].weight.fill_(1.0)

    anchor = ActorParameterAnchor(
        actor,
        reference,
        coefficient=0.5,
        first_layer_input_exempt_start=2,
        exempt_coefficient=0.1,
    )
    sum(parameter.sum() for parameter in actor.parameters()).backward()

    assert torch.allclose(actor[0].weight.grad[:, :2], torch.full((2, 2), 1.5))
    assert torch.allclose(actor[0].weight.grad[:, 2:], torch.full((2, 2), 1.1))
    anchor.close()


def test_actor_anchor_can_exempt_bounded_input_slice() -> None:
    actor = nn.Sequential(nn.Linear(6, 2, bias=False), nn.Tanh(), nn.Linear(2, 1))
    reference = {
        name: torch.zeros_like(parameter)
        for name, parameter in actor.named_parameters()
    }
    with torch.no_grad():
        actor[0].weight.fill_(1.0)

    anchor = ActorParameterAnchor(
        actor,
        reference,
        coefficient=0.5,
        first_layer_input_exempt_start=2,
        first_layer_input_exempt_end=4,
        exempt_coefficient=0.1,
    )
    sum(parameter.sum() for parameter in actor.parameters()).backward()

    assert torch.allclose(actor[0].weight.grad[:, :2], torch.full((2, 2), 1.5))
    assert torch.allclose(actor[0].weight.grad[:, 2:4], torch.full((2, 2), 1.1))
    assert torch.allclose(actor[0].weight.grad[:, 4:], torch.full((2, 2), 1.5))
    anchor.close()


def test_actor_anchor_rejects_exemption_end_without_start() -> None:
    actor = nn.Linear(4, 1)
    reference = {
        name: torch.zeros_like(parameter)
        for name, parameter in actor.named_parameters()
    }
    try:
        ActorParameterAnchor(
            actor,
            reference,
            coefficient=1.0,
            first_layer_input_exempt_end=3,
        )
    except ValueError as exc:
        assert "end requires an exemption start" in str(exc)
    else:
        raise AssertionError("an exemption end without start should fail")


def test_actor_anchor_rejects_negative_exempt_coefficient() -> None:
    actor = nn.Linear(2, 1)
    reference = {
        name: torch.zeros_like(parameter)
        for name, parameter in actor.named_parameters()
    }
    try:
        ActorParameterAnchor(
            actor,
            reference,
            coefficient=1.0,
            exempt_coefficient=-0.1,
        )
    except ValueError as exc:
        assert "exempt coefficient" in str(exc)
    else:
        raise AssertionError("negative exempt coefficient should fail")
