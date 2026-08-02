from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import numpy as np


_ROOT = os.path.dirname(os.path.dirname(__file__))
_PATH = os.path.join(_ROOT, "scripts", "isaac_physical_eval.py")
_SPEC = importlib.util.spec_from_file_location(
    "hope_isaac_physical_eval", _PATH
)
physical_eval = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = physical_eval
_SPEC.loader.exec_module(physical_eval)


def test_physical_recovery_uses_actual_contact_relative_phases() -> None:
    record = physical_eval._new_physical_recovery_record(10.0)
    low = np.zeros(7, dtype=np.float64)
    high_ang = low.copy()
    high_ang[-1] = 1.3

    physical_eval._observe_physical_recovery(
        record, timestamp=10.00, values=low
    )
    physical_eval._observe_physical_recovery(
        record, timestamp=10.10, values=high_ang
    )
    physical_eval._observe_physical_recovery(
        record, timestamp=10.30, values=low
    )
    physical_eval._observe_physical_recovery(
        record, timestamp=10.60, values=low
    )

    assert record["phase_seen"].tolist() == [True, True, True, True]
    assert record["histogram_latches"][1, 6].tolist() == [
        True,
        True,
        False,
        False,
    ]


def test_physical_recovery_summary_conditions_ready_on_outcome() -> None:
    contact = physical_eval._new_physical_recovery_record(0.0)
    contact["outcome_bucket"] = 1
    contact["ready_success"] = False
    contact["phase_seen"][0] = True

    bounce = physical_eval._new_physical_recovery_record(1.0)
    bounce["outcome_bucket"] = 3
    bounce["ready_success"] = True
    bounce["phase_seen"][:2] = True

    summary = physical_eval._summarize_physical_recoveries(
        [contact, bounce]
    )
    assert summary["trigger"] == "actual_physx_contact"
    assert summary["attempts"] == 2
    assert (
        summary["outcome_resolution"]["contact"]["ready_success_rate"]
        == pytest.approx(0.0)
    )
    assert (
        summary["outcome_resolution"]["opponent_bounce"][
            "ready_success_rate"
        ]
        == pytest.approx(1.0)
    )
    assert (
        summary["phase_histograms"]["brake_100_300ms"][
            "attempts_reaching_phase"
        ]
        == 1
    )
