from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/audit_real_contact_residuals.py"
)
SPEC = importlib.util.spec_from_file_location("audit_real_contact_residuals", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _linear_rows(velocity: np.ndarray, times: np.ndarray) -> list[dict[str, str]]:
    origin = np.asarray([0.2, -1.0, 0.4])
    rows = []
    for time_s in times:
        point = origin + velocity * time_s
        rows.append(
            {
                "time_ns": str(int(round(time_s * 1.0e9))),
                "ball_x": str(point[0]),
                "ball_y": str(point[1]),
                "ball_z": str(point[2]),
            }
        )
    return rows


def test_fit_velocity_recovers_linear_motion() -> None:
    expected = np.asarray([2.0, -0.5, 1.25])
    rows = _linear_rows(expected, np.linspace(-0.1, -0.02, 9))
    velocity, rms_m, count = MODULE.fit_velocity(
        rows, MODULE.BALL_FIELDS, event_time_ns=0
    )
    assert count == 9
    np.testing.assert_allclose(velocity, expected, atol=1.0e-12)
    assert rms_m < 1.0e-12


def test_tangent_basis_is_orthonormal() -> None:
    normal = np.asarray([0.7, -0.6, 0.3])
    normal /= np.linalg.norm(normal)
    first, second = MODULE.tangent_basis(normal)
    np.testing.assert_allclose(np.linalg.norm(first), 1.0)
    np.testing.assert_allclose(np.linalg.norm(second), 1.0)
    np.testing.assert_allclose(first @ normal, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(second @ normal, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(first @ second, 0.0, atol=1.0e-12)


def test_physical_contact_gate_rejects_geometry_without_velocity_reversal() -> None:
    common = {
        "distance_m": 0.05,
        "ball_height_m": 0.4,
        "delta_velocity_mps": 4.0,
        "incoming_relative_normal_mps": -1.0,
        "incoming_fit_rms_m": 0.003,
        "outgoing_fit_rms_m": 0.003,
    }
    assert MODULE.passes_physical_contact_gate(
        **common, outgoing_relative_normal_mps=0.5
    )
    assert not MODULE.passes_physical_contact_gate(
        **common, outgoing_relative_normal_mps=-0.2
    )


def test_calibration_requires_multiple_sessions_and_holdout() -> None:
    seven = [{"physical_contact_weak_label": True} for _ in range(7)]
    result = MODULE.calibration_readiness(seven, session_count=1)
    assert not result["calibration_ready"]
    assert not result["session_group_holdout_possible"]

    thirty = [{"physical_contact_weak_label": True} for _ in range(30)]
    result = MODULE.calibration_readiness(thirty, session_count=3)
    assert result["calibration_ready"]
    assert result["session_group_holdout_possible"]
