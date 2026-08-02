"""Pure-Python tests for the frozen observable-contact and action contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_closed_loop_contracts.py"
SPEC = importlib.util.spec_from_file_location("closed_loop_contract_validator", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_contracts_are_structurally_valid_but_not_train_ready():
    result = validator.validate_contracts()
    assert result["schema_valid"], result["errors"]
    assert not result["train_ready"]
    assert result["promotion_decision"] == "BLOCKED"
    assert "operational joint limits are not calibrated" in result["blockers"]
    assert result["checks"]["provisional_operational_audit_exists"]
    assert result["checks"]["migration_teacher_artifacts_frozen"]
    assert result["checks"]["bounded_codec_isolated_and_tested"]
    assert result["checks"]["migration_semantics"]
    assert result["checks"]["task_lifecycle_is_complete"]
    assert result["checks"]["task_reward_is_closed_loop"]
    assert result["checks"]["task_motion_is_prior_only"]
    assert result["checks"]["task_curriculum_is_capability_driven"]
    assert result["checks"]["task_first_motion_exists"]
    assert result["checks"]["experiment_initialization_ab_is_controlled"]
    assert result["checks"]["experiment_budgets_are_gated"]
    assert result["checks"]["experiment_evaluation_is_cross_sim_closed_loop"]
    assert result["checks"]["experiment_execution_is_blocked"]
    assert len(result["hashes"]["contract_set_sha256"]) == 64


def test_spin_is_not_an_observation_or_target():
    contact = _load(validator.DEFAULT_CONTACT)
    observed = {
        field["name"] for field in contact["observed_ball_state"]["fields"]
    }
    assert "angular_velocity" not in observed
    assert contact["latent_spin"]["actor_input"] == "forbidden"
    assert contact["latent_spin"]["planner_input"] == "forbidden"
    assert contact["latent_spin"]["deterministic_reward_target"] == "forbidden"
    assert (
        contact["contact_model"]["tangential_component"]["type"]
        == "conditional_residual_distribution"
    )


def test_legacy_policy_cannot_silently_switch_feedback_contract():
    action = _load(validator.DEFAULT_ACTION)
    legacy = action["legacy_contract"]
    target = action["target_contract"]
    assert legacy["last_action_feedback"] == "raw_action"
    assert legacy["runtime_switch_to_target_contract_allowed"] is False
    assert target["previous_action_observation"]["mode"] == "effective"
    assert target["mechanical_clamp"]["role"] == "emergency_only"


def test_operational_limits_require_measurement_before_training():
    action = _load(validator.DEFAULT_ACTION)
    limits = action["operational_limits"]
    assert limits["status"] == "pending_calibration"
    assert limits["data_file"] is None
    assert limits["training_enabled"] is False
    assert len(limits["required_measurements"]) >= 5


def test_unified_task_is_isolated_until_dependencies_are_ready():
    task = _load(validator.DEFAULT_TASK)
    assert task["implementation"]["active_task_wired"] is False
    assert task["physical_truth"]["ball_route_independent_of_motion"] is True
    assert task["reward"]["accounting"]["ready_positive_reward_per_step"] == "forbidden"
    assert task["curriculum"]["iteration_or_wall_clock_driver"] == "forbidden"


def test_experiment_matrix_cannot_start_long_training():
    matrix = _load(validator.DEFAULT_MATRIX)
    assert matrix["status"] == "blocked_preflight_only"
    assert matrix["long_training_authorized"] is False
    assert matrix["execution"]["no_gpu_job_started"] is True
    long_stage = next(
        stage
        for stage in matrix["training_budgets"]["stages"]
        if stage["id"] == "T5_LONG"
    )
    assert long_stage["enabled"] is False
