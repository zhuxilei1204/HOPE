from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_stage1_command_tracking.py"
SPEC = importlib.util.spec_from_file_location("stage1_command_tracking_audit", SCRIPT)
assert SPEC and SPEC.loader
audit_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_module
SPEC.loader.exec_module(audit_module)


def test_reward_contract_excludes_physical_outcomes() -> None:
    contract = audit_module.load_yaml(audit_module.CONTRACT_PATH)
    required = set(contract["required_nonzero_rewards"])
    forbidden = set(contract["forbidden_reward_terms"])
    assert not (required & forbidden)
    assert "planner_racket_task_space_crossfade" in required
    assert "phase_lower_body_motion_prior" not in required


def test_reward_replay_preserves_exploration_without_idle_optimum() -> None:
    contract = audit_module.load_yaml(audit_module.CONTRACT_PATH)
    replay = audit_module.synthetic_reward_replay(contract)
    ratio = replay["healthy_track"] / replay["healthy_idle"]
    assert ratio >= contract["synthetic_replay"]["required_healthy_track_over_idle_ratio"]
    assert replay["unsafe_track"] > replay["unsafe_idle"]
    assert replay["healthy_track"] > replay["unsafe_track"]
