from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_closed_loop_v3_phase_c7.py"
SPEC = importlib.util.spec_from_file_location("closed_loop_v3_phase_c7", SCRIPT)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_module
SPEC.loader.exec_module(audit_module)


def test_c7_ab_changes_only_outcome_credit_assignment() -> None:
    result = audit_module.audit(ROOT)
    assert result["passed"] is True
    assert set(result["ab_differences"]) == audit_module.ALLOWED_AB_DIFFERENCES


def test_c7_contract_separates_route_speed_and_planner_noise() -> None:
    result = audit_module.audit(ROOT)
    checks = result["checks"]
    assert checks["separate_route_curriculum"] is True
    assert checks["separate_speed_curriculum"] is True
    assert checks["fixed_planner_noise"] is True
    assert checks["actual_outcome_settlement"] is True
