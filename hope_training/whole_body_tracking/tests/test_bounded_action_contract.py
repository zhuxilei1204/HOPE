from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/utils/bounded_action_contract.py"
)
SPEC = importlib.util.spec_from_file_location("bounded_action_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _codec() -> MODULE.BoundedJointActionCodec:
    return MODULE.BoundedJointActionCodec(
        operational_lower=torch.tensor([-1.0, -0.5, -0.25]),
        operational_upper=torch.tensor([1.0, 0.5, 0.75]),
        mechanical_lower=torch.tensor([-1.2, -0.8, -0.5]),
        mechanical_upper=torch.tensor([1.2, 0.8, 1.0]),
        default_q=torch.tensor([0.0, 0.0, 0.25]),
        action_scale=torch.tensor([0.5, 0.25, 0.5]),
        passive_mask=torch.tensor([False, True, False]),
    )


def test_decode_latent_is_bounded_and_never_hits_mechanical_clip() -> None:
    codec = _codec()
    latent = torch.tensor([[100.0, -100.0, 0.0]])
    bounded, q_des, emergency = codec.decode_latent(latent)
    assert torch.all(bounded <= 1.0)
    assert torch.all(bounded >= -1.0)
    torch.testing.assert_close(q_des, torch.tensor([[1.0, 0.0, 0.25]]))
    assert not torch.any(emergency)


def test_joint_target_round_trip_and_effective_feedback() -> None:
    codec = _codec()
    q_target = torch.tensor([[0.5, 0.0, 0.5]])
    encoded = codec.encode_joint_target(q_target)
    assert encoded.feasible.item()
    q_round_trip, emergency = codec.decode_bounded(encoded.bounded_action)
    torch.testing.assert_close(q_round_trip, q_target)
    assert not torch.any(emergency)
    torch.testing.assert_close(
        codec.effective_feedback(q_round_trip),
        torch.tensor([[1.0, 0.0, 0.5]]),
    )


def test_default_target_provides_safe_actor_output_bias() -> None:
    codec = _codec()
    encoded = codec.encode_joint_target(codec.default_q.unsqueeze(0))
    q_des, emergency = codec.decode_bounded(
        torch.tanh(encoded.latent)
    )
    torch.testing.assert_close(q_des, codec.default_q.unsqueeze(0))
    assert not torch.any(emergency)


def test_distillation_gate_rejects_out_of_range_target() -> None:
    codec = _codec()
    encoded = codec.encode_joint_target(torch.tensor([[1.1, 0.0, 0.25]]))
    assert not encoded.feasible.item()
    assert encoded.bounded_action[0, 0] > 1.0


def test_squashed_gaussian_log_prob_matches_change_of_variables() -> None:
    latent = torch.tensor([[0.3, -0.7]], dtype=torch.float64)
    mean = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
    std = torch.tensor([[0.8, 0.6]], dtype=torch.float64)
    action = torch.tanh(latent)
    actual = MODULE.squashed_gaussian_log_prob(action, mean, std)
    normal = torch.distributions.Normal(mean, std)
    expected = (
        normal.log_prob(latent) - torch.log1p(-action.square())
    ).sum(dim=-1)
    torch.testing.assert_close(actual, expected, atol=1.0e-10, rtol=1.0e-10)


def test_invalid_operational_range_is_rejected() -> None:
    try:
        MODULE.BoundedJointActionCodec(
            operational_lower=torch.tensor([-1.2]),
            operational_upper=torch.tensor([0.5]),
            mechanical_lower=torch.tensor([-1.2]),
            mechanical_upper=torch.tensor([1.0]),
            default_q=torch.tensor([0.0]),
            action_scale=torch.tensor([0.5]),
        )
    except ValueError as error:
        assert "strictly inside" in str(error)
    else:
        raise AssertionError("invalid operational range was accepted")
