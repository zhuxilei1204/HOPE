from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_closed_loop_v3_scratch.py"
SPEC = importlib.util.spec_from_file_location("closed_loop_v3_scratch_audit", SCRIPT)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_module
SPEC.loader.exec_module(audit_module)


def test_contract_reward_whitelist_has_no_dense_ready_payoff() -> None:
    assert "alive" not in audit_module.EXPECTED_NONZERO_REWARDS
    assert "no_command_ready_stability" not in audit_module.EXPECTED_NONZERO_REWARDS
    assert "no_command_instability" in audit_module.EXPECTED_NONZERO_REWARDS
    assert "active_ready_sustained_bonus" in audit_module.EXPECTED_NONZERO_REWARDS
    assert (
        "post_contact_directional_recovery"
        in audit_module.EXPECTED_NONZERO_REWARDS
    )
    assert "post_contact_ready_region" not in audit_module.EXPECTED_NONZERO_REWARDS
    assert "cycle_v2_ready_success_bonus" in audit_module.EXPECTED_NONZERO_REWARDS
    assert "prestrike_station_progress" in audit_module.EXPECTED_NONZERO_REWARDS
    assert (
        "near_impact_planner_velocity_progress"
        in audit_module.EXPECTED_NONZERO_REWARDS
    )
    assert "planner_velocity_band" not in audit_module.EXPECTED_NONZERO_REWARDS


def test_v5_adds_only_non_farmable_ready_potential() -> None:
    added = (
        audit_module.EXPECTED_NONZERO_REWARDS_READY_POTENTIAL_V5
        - audit_module.EXPECTED_NONZERO_REWARDS
    )
    assert added == {"no_command_ready_progress"}
    assert (
        "no_command_ready_stability"
        not in audit_module.EXPECTED_NONZERO_REWARDS_READY_POTENTIAL_V5
    )
    assert (
        "no_command_ready_balance"
        not in audit_module.EXPECTED_NONZERO_REWARDS_READY_POTENTIAL_V5
    )
