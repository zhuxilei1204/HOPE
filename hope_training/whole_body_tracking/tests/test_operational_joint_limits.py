"""Unit tests for the N1 provisional operational-limit estimator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/derive_operational_joint_limits.py"
SPEC = importlib.util.spec_from_file_location("operational_limit_estimator", SCRIPT)
assert SPEC and SPEC.loader
estimator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = estimator
SPEC.loader.exec_module(estimator)


def _summary(
    *,
    q_low: list[float],
    q_high: list[float],
    tracking: list[float],
    velocity: list[float],
    lower_overshoot: list[float] | None = None,
    upper_overshoot: list[float] | None = None,
    negative_velocity: list[float] | None = None,
    positive_velocity: list[float] | None = None,
) -> dict:
    lower_overshoot = lower_overshoot or tracking
    upper_overshoot = upper_overshoot or tracking
    negative_velocity = negative_velocity or velocity
    positive_velocity = positive_velocity or velocity
    return {
        "q_p001": np.asarray(q_low),
        "q_p999": np.asarray(q_high),
        "tracking_abs_p99_safe": np.asarray(tracking),
        "dq_abs_p99_safe": np.asarray(velocity),
        "lower_overshoot_p99_safe": np.asarray(lower_overshoot),
        "upper_overshoot_p99_safe": np.asarray(upper_overshoot),
        "negative_velocity_p99_safe": np.asarray(negative_velocity),
        "positive_velocity_p99_safe": np.asarray(positive_velocity),
    }


def test_margin_formula_and_observed_envelope():
    result = estimator.derive_bounds(
        hard_lower=np.array([-1.0, -2.0]),
        hard_upper=np.array([1.0, 2.0]),
        default_q=np.array([0.0, 0.0]),
        motion_lower=np.array([-0.3, -0.5]),
        motion_upper=np.array([0.4, 0.6]),
        summaries={
            "sim": _summary(
                q_low=[-0.2, -0.6],
                q_high=[0.5, 0.4],
                tracking=[0.03, 0.04],
                velocity=[1.0, 2.0],
                lower_overshoot=[0.01, 0.02],
                upper_overshoot=[0.02, 0.01],
                negative_velocity=[1.0, 2.0],
                positive_velocity=[0.5, 1.0],
            ),
            "real": _summary(
                q_low=[-0.4, -0.4],
                q_high=[0.3, 0.8],
                tracking=[0.05, 0.02],
                velocity=[1.5, 1.0],
                lower_overshoot=[0.02, 0.01],
                upper_overshoot=[0.03, 0.02],
                negative_velocity=[1.0, 0.5],
                positive_velocity=[1.5, 1.0],
            ),
        },
        fixed_margin_rad=0.02,
        assumed_delay_s=0.02,
    )
    np.testing.assert_allclose(result["provisional_lower_margin"], [0.06, 0.08])
    np.testing.assert_allclose(result["provisional_upper_margin"], [0.08, 0.06])
    np.testing.assert_allclose(result["provisional_lower"], [-0.94, -1.92])
    np.testing.assert_allclose(result["provisional_upper"], [0.92, 1.94])
    np.testing.assert_allclose(result["observed_lower"], [-0.4, -0.6])
    np.testing.assert_allclose(result["observed_upper"], [0.5, 0.8])
    assert np.all(result["motion_inside"])
    assert np.all(result["observed_inside"])


def test_infeasible_dynamic_margin_is_reported_not_relaxed():
    result = estimator.derive_bounds(
        hard_lower=np.array([-0.2]),
        hard_upper=np.array([0.2]),
        default_q=np.array([0.0]),
        motion_lower=np.array([-0.1]),
        motion_upper=np.array([0.1]),
        summaries={
            "trace": _summary(
                q_low=[-0.1],
                q_high=[0.1],
                tracking=[0.18],
                velocity=[3.0],
            )
        },
        fixed_margin_rad=0.03,
        assumed_delay_s=0.02,
    )
    assert not result["range_valid"][0]
    assert not result["motion_inside"][0]


def test_clamp_inference_detects_only_hard_bound_commands():
    q_des = np.array([[-1.0, 0.0], [0.2, 2.0], [0.0, 0.0]])
    clamp = estimator.infer_hard_clamp(
        q_des,
        np.array([-1.0, -2.0]),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_array_equal(
        clamp,
        np.array([[True, False], [False, True], [False, False]]),
    )


@pytest.mark.parametrize(
    "path",
    [
        estimator.DEFAULT_P1_DIR / "ready_raw.csv",
        estimator.DEFAULT_P1_DIR / "continuous_raw_joint.csv",
    ],
)
def test_both_existing_sim_trace_schemas_are_supported(path: Path):
    names, _adapter = estimator.load_adapter(
        estimator.DEFAULT_ADAPTER,
        estimator.DEFAULT_JOINT_ORDER,
    )
    trace = estimator.load_sim_trace(path, names)
    assert trace["q"].shape[1] == 31
    assert trace["q"].shape == trace["dq"].shape == trace["q_des"].shape
