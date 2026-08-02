from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load(
    "a3_actuator_contract",
    "source/whole_body_tracking/whole_body_tracking/robots/agibot_a3_actuator_contract.py",
)
ENVELOPE = _load(
    "actuator_envelope",
    "source/whole_body_tracking/whole_body_tracking/utils/actuator_envelope.py",
)


def _canonical_joint_order() -> tuple[str, ...]:
    import yaml

    path = ROOT.parent / "config/joint_order_agibot_a3.yaml"
    with path.open("r", encoding="utf-8") as stream:
        return tuple(yaml.safe_load(stream)["joint_order"])


def test_contract_covers_canonical_policy_order_once() -> None:
    canonical_joint_order = _canonical_joint_order()
    CONTRACT.validate_actuator_contract(canonical_joint_order)
    assert set(CONTRACT.A3_SERIAL_JOINT_NAMES).isdisjoint(
        CONTRACT.A3_PARALLEL_JOINT_NAMES
    )
    assert set(CONTRACT.A3_SERIAL_JOINT_NAMES) | set(
        CONTRACT.A3_PARALLEL_JOINT_NAMES
    ) == set(
        canonical_joint_order
    )
    assert len(CONTRACT.A3_ACTUATOR_SPECS) == 31


def test_supplied_output_side_values_are_transcribed() -> None:
    knee = CONTRACT.A3_ACTUATOR_SPECS["left_knee_joint"]
    distal = CONTRACT.A3_ACTUATOR_SPECS["right_wrist_yaw_joint"]
    ankle = CONTRACT.A3_ACTUATOR_SPECS["right_ankle_pitch_joint"]
    assert knee.armature == pytest.approx(0.1203404)
    assert knee.rated_torque == pytest.approx(70.0)
    assert knee.peak_torque == pytest.approx(320.0)
    assert knee.rated_speed == pytest.approx(140.0 * 2.0 * math.pi / 60.0)
    assert distal.rated_torque == pytest.approx(2.0)
    assert distal.peak_torque == pytest.approx(6.0)
    assert ankle.parallel
    assert ankle.peak_torque == pytest.approx(118.2)


def test_actuator_values_preserve_requested_order() -> None:
    values = CONTRACT.actuator_values(
        ("right_wrist_yaw_joint", "left_knee_joint"), "rated_torque"
    )
    assert values == pytest.approx((2.0, 70.0))
    with pytest.raises(KeyError):
        CONTRACT.actuator_values(("not_a_joint",), "rated_torque")


def test_envelope_separates_rated_corner_and_peak_violations() -> None:
    rated_torque = torch.tensor([[10.0]])
    peak_torque = torch.tensor([[30.0]])
    rated_speed = torch.tensor([[5.0]])
    peak_speed = torch.tensor([[10.0]])

    below = ENVELOPE.actuator_envelope_components(
        torch.tensor([[9.0]]),
        torch.tensor([[4.0]]),
        rated_torque,
        peak_torque,
        rated_speed,
        peak_speed,
    )
    assert all(component.item() == pytest.approx(0.0) for component in below)

    torque_only = ENVELOPE.actuator_envelope_components(
        torch.tensor([[15.0]]),
        torch.tensor([[4.0]]),
        rated_torque,
        peak_torque,
        rated_speed,
        peak_speed,
    )
    assert torque_only[0].item() > 0.0
    assert torque_only[1].item() == pytest.approx(0.0)
    assert torque_only[2].item() == pytest.approx(0.0)

    forbidden_corner = ENVELOPE.actuator_envelope_components(
        torch.tensor([[15.0]]),
        torch.tensor([[7.0]]),
        rated_torque,
        peak_torque,
        rated_speed,
        peak_speed,
    )
    assert forbidden_corner[1].item() > 0.0

    above_peak = ENVELOPE.actuator_envelope_components(
        torch.tensor([[35.0]]),
        torch.tensor([[11.0]]),
        rated_torque,
        peak_torque,
        rated_speed,
        peak_speed,
    )
    assert above_peak[2].item() > 0.0
