from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_recorded_policy_session.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_recorded_policy_session", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reconstruct_active_task_ignores_transient_packet_task_ids() -> None:
    packet_task = np.asarray([0, 20, 0, 21, 0, 0, 0])
    phase = np.asarray([0, 1, 1, 2, 2, 3, 0])
    accepted = np.asarray([False, True, False, False, False, False, False])
    reasons = np.asarray(
        [
            "no_new_command",
            "accepted_initial",
            "no_new_command",
            "phase_blocked",
            "no_new_command",
            "no_new_command",
            "no_new_command",
        ]
    )

    actual = MODULE._reconstruct_active_task_id(
        packet_task, phase, accepted, reasons
    )

    np.testing.assert_array_equal(actual, [-1, 20, 20, 20, 20, 20, -1])


def test_reconstruct_active_task_rejects_unanchored_lifecycle() -> None:
    with pytest.raises(ValueError, match="cannot reconstruct lifecycle"):
        MODULE._reconstruct_active_task_id(
            np.asarray([0, 0]),
            np.asarray([0, 1]),
            np.asarray([False, False]),
            np.asarray(["no_new_command", "no_new_command"]),
        )


def test_velocity_fit_recovers_linear_track() -> None:
    time_ns = np.arange(-80_000_000, -9_999_999, 10_000_000, dtype=np.int64)
    expected_velocity = np.asarray([-2.0, 0.4, 1.1])
    position = (
        np.asarray([0.2, -1.0, 0.3])
        + time_ns[:, None] * 1.0e-9 * expected_velocity
    )

    actual = MODULE._fit_track_velocity(
        time_ns, position, event_ns=0, begin_ms=-80.0, end_ms=-15.0
    )

    assert actual is not None
    np.testing.assert_allclose(actual, expected_velocity, atol=1.0e-12)


def test_net_crossing_after_first_bounce_can_be_rejected_by_caller() -> None:
    time_ns = np.asarray([0, 10, 20, 30], dtype=np.int64) * 10_000_000
    position = np.asarray(
        [
            [0.2, -1.0, 0.2],
            [0.8, -1.0, 0.01],
            [1.2, -1.0, 0.1],
            [1.5, -1.0, 0.1],
        ]
    )

    bounce = MODULE._first_downward_surface_crossing(
        time_ns, position, event_ns=0, z_surface=0.02
    )
    net = MODULE._first_forward_plane_crossing(
        time_ns, position, event_ns=0, x_plane=1.37
    )

    assert bounce is not None
    assert net is not None
    assert bounce[0] < net[0]
