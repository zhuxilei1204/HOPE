#!/usr/bin/env python3
"""Validate the frozen contact and target action contracts without Isaac."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTACT = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/cfg/contracts/hope_observable_contact_v1.yaml"
)
DEFAULT_ACTION = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/cfg/contracts/hope_action_execution_v2.yaml"
)
DEFAULT_MIGRATION = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/cfg/contracts/hope_policy_migration_v1.yaml"
)
DEFAULT_TASK = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/cfg/contracts/hope_training_task_v1.yaml"
)
DEFAULT_MATRIX = (
    REPO_ROOT
    / "hope_training/whole_body_tracking/cfg/experiments/hope_closed_loop_v3_matrix.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return document


def _repo_path(value: str) -> Path:
    return REPO_ROOT / value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(
    condition: bool,
    message: str,
    errors: list[str],
) -> None:
    if not condition:
        errors.append(message)


def _per_joint(
    value: Any,
    names: tuple[str, ...],
    field: str,
) -> np.ndarray:
    if isinstance(value, dict):
        missing = [name for name in names if name not in value]
        if missing:
            raise ValueError(f"{field} is missing joints: {missing}")
        return np.asarray([float(value[name]) for name in names], dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (len(names),):
        raise ValueError(f"{field} must have {len(names)} values")
    return array


def validate_contracts(
    contact_path: Path = DEFAULT_CONTACT,
    action_path: Path = DEFAULT_ACTION,
    migration_path: Path = DEFAULT_MIGRATION,
    task_path: Path = DEFAULT_TASK,
    matrix_path: Path = DEFAULT_MATRIX,
) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []
    checks: dict[str, bool] = {}

    contact = _load_yaml(contact_path)
    action = _load_yaml(action_path)
    migration = _load_yaml(migration_path)
    task = _load_yaml(task_path)
    matrix = _load_yaml(matrix_path)

    checks["contract_ids"] = (
        contact.get("contract_id") == "hope_observable_contact_v1"
        and action.get("contract_id") == "hope_action_execution_v2"
        and migration.get("contract_id") == "hope_policy_migration_v1"
        and task.get("contract_id") == "hope_training_task_v1"
    )
    _require(checks["contract_ids"], "unexpected contract_id", errors)

    observed = {
        field["name"] for field in contact["observed_ball_state"]["fields"]
    }
    prohibited = set(contact["observed_ball_state"]["prohibited_fields"])
    spin_fields = {"orientation", "angular_velocity", "spin_axis", "spin_rate"}
    latent = contact["latent_spin"]
    checks["spin_is_latent"] = (
        "position" in observed
        and "linear_velocity" in observed
        and not (observed & spin_fields)
        and spin_fields.issubset(prohibited)
        and latent["observable"] is False
        and latent["planner_input"] == "forbidden"
        and latent["actor_input"] == "forbidden"
        and latent["deterministic_reward_target"] == "forbidden"
    )
    _require(
        checks["spin_is_latent"],
        "spin must be latent and absent from actor/planner observations",
        errors,
    )

    tangent = contact["contact_model"]["tangential_component"]
    checks["tangential_uncertainty"] = (
        tangent["type"] == "conditional_residual_distribution"
        and tangent["angular_velocity_required"] is False
    )
    _require(
        checks["tangential_uncertainty"],
        "tangential contact must use an observable conditional distribution",
        errors,
    )
    if tangent["calibration_status"] != "calibrated":
        blockers.append("contact tangential residual distribution is not calibrated")

    contact_source = _repo_path(contact["contact_model"]["source_of_truth"])
    checks["contact_source_exists"] = contact_source.is_file()
    _require(
        checks["contact_source_exists"],
        f"missing contact source of truth: {contact_source}",
        errors,
    )

    order_path = _repo_path(action["canonical_joint_order_path"])
    adapter_path = _repo_path(action["shared_legacy_adapter_path"])
    checks["action_sources_exist"] = order_path.is_file() and adapter_path.is_file()
    _require(
        checks["action_sources_exist"],
        "joint order or legacy action adapter is missing",
        errors,
    )

    if checks["action_sources_exist"]:
        order_doc = _load_yaml(order_path)
        adapter_doc = _load_yaml(adapter_path)
        joint_names = tuple(str(name) for name in order_doc["joint_order"])
        checks["joint_order_is_canonical_31"] = (
            len(joint_names) == 31 and len(set(joint_names)) == 31
        )
        _require(
            checks["joint_order_is_canonical_31"],
            "canonical joint order must contain 31 unique joints",
            errors,
        )
        if checks["joint_order_is_canonical_31"]:
            try:
                default_q = _per_joint(
                    adapter_doc["default_q"], joint_names, "default_q"
                )
                action_scale = _per_joint(
                    adapter_doc["action_scale"], joint_names, "action_scale"
                )
                lower = _per_joint(
                    adapter_doc["joint_position_clamp"]["lower"],
                    joint_names,
                    "joint_position_clamp.lower",
                )
                upper = _per_joint(
                    adapter_doc["joint_position_clamp"]["upper"],
                    joint_names,
                    "joint_position_clamp.upper",
                )
                checks["legacy_adapter_constants_valid"] = bool(
                    np.all(action_scale > 0.0)
                    and np.all(lower < upper)
                    and np.all(default_q >= lower)
                    and np.all(default_q <= upper)
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"invalid legacy action adapter: {exc}")
                checks["legacy_adapter_constants_valid"] = False
            _require(
                checks["legacy_adapter_constants_valid"],
                "legacy adapter constants are inconsistent",
                errors,
            )

    legacy = action["legacy_contract"]
    target = action["target_contract"]
    checks["legacy_is_explicitly_frozen"] = (
        legacy["compatibility_only"] is True
        and legacy["last_action_feedback"] == "raw_action"
        and legacy["runtime_switch_to_target_contract_allowed"] is False
    )
    _require(
        checks["legacy_is_explicitly_frozen"],
        "legacy raw policies must not silently switch contracts",
        errors,
    )

    checks["target_action_semantics"] = (
        target["policy_distribution"]["type"] == "tanh_squashed_gaussian"
        and target["policy_distribution"]["log_probability_correction_required"] is True
        and target["previous_action_observation"]["mode"] == "effective"
        and target["mechanical_clamp"]["role"] == "emergency_only"
        and target["mechanical_clamp"]["allowed_as_normal_policy_nonlinearity"] is False
    )
    _require(
        checks["target_action_semantics"],
        "target action contract must be bounded/effective with emergency-only hard clamp",
        errors,
    )

    limits = action["operational_limits"]
    provisional_path = _repo_path(limits["provisional_data_file"])
    checks["provisional_operational_audit_exists"] = provisional_path.is_file()
    _require(
        checks["provisional_operational_audit_exists"],
        f"missing provisional operational-limit audit: {provisional_path}",
        errors,
    )
    if (
        limits["status"] != "calibrated"
        or not limits["training_enabled"]
        or not limits["data_file"]
    ):
        blockers.append("operational joint limits are not calibrated")
    if target["implementation_status"] != "implemented":
        blockers.append("bounded/effective actor contract is not implemented")

    teacher = migration["teacher"]
    teacher_onnx = Path(teacher["onnx_path"])
    teacher_checkpoint = Path(teacher["checkpoint_path"])
    checks["migration_teacher_artifacts_frozen"] = (
        teacher_onnx.is_file()
        and teacher_checkpoint.is_file()
        and _sha256(teacher_onnx) == teacher["onnx_sha256"]
        and _sha256(teacher_checkpoint) == teacher["checkpoint_sha256"]
    )
    _require(
        checks["migration_teacher_artifacts_frozen"],
        "migration teacher artifact is missing or its hash changed",
        errors,
    )
    implementation = migration["implementation"]
    codec_path = _repo_path(implementation["codec_module"])
    codec_tests_path = _repo_path(implementation["codec_tests"])
    checks["bounded_codec_isolated_and_tested"] = (
        codec_path.is_file()
        and codec_tests_path.is_file()
        and implementation["active_task_integration"] == "disabled"
        and implementation["ppo_policy_distribution_integration"] == "pending"
    )
    _require(
        checks["bounded_codec_isolated_and_tested"],
        "bounded codec must remain isolated until operational limits are calibrated",
        errors,
    )
    checks["migration_semantics"] = (
        migration["student"]["policy_output"] == "bounded_action"
        and migration["student"]["previous_action"].endswith(
            "effective_feedback"
        )
        and migration["student"]["safe_initialization"][
            "first_deterministic_q_des"
        ]
        == "default_q"
        and migration["student"]["safe_initialization"][
            "generic_zero_actor_output_helper_allowed"
        ]
        is False
        and migration["distillation"]["target"] == "q_des_final"
        and migration["distillation"]["raw_action_target"] == "forbidden"
        and migration["distillation"][
            "sample_projection_to_operational_range"
        ]
        == "forbidden"
    )
    _require(
        checks["migration_semantics"],
        "migration must distill feasible q_des without raw-action projection",
        errors,
    )

    lifecycle = task["lifecycle"]
    expected_states = [
        "READY_NO_COMMAND",
        "COMMAND_ACQUIRE",
        "PRE_STRIKE_RELOCATE",
        "PRE_STRIKE_SETTLE",
        "STRIKE",
        "FOLLOW_THROUGH",
        "RECOVERY",
        "NEXT_READY",
    ]
    checks["task_lifecycle_is_complete"] = (
        lifecycle["states"] == expected_states
        and task["episode"]["bad_posture_carries_into_next_ball"] is True
        and {"miss", "recovery_timeout"}.issubset(
            set(task["episode"]["must_not_reset_on"])
        )
        and "miss_still_opens_recovery" in lifecycle["invariants"]
    )
    _require(
        checks["task_lifecycle_is_complete"],
        "task lifecycle must carry misses and bad posture through recovery",
        errors,
    )
    reward = task["reward"]
    checks["task_reward_is_closed_loop"] = (
        reward["accounting"]["ready_positive_reward_per_step"] == "forbidden"
        and reward["accounting"]["event_rewards_are_one_shot"] is True
        and reward["recovery"]["miss_triggers_recovery"] is True
        and reward["cycle"]["one_shot"] is True
        and task["hard_safety"]["never_relaxed_by_curriculum"] is True
    )
    _require(
        checks["task_reward_is_closed_loop"],
        "task reward must be one-shot, health-gated, and recovery-complete",
        errors,
    )
    checks["task_motion_is_prior_only"] = (
        task["physical_truth"]["ball_route_independent_of_motion"] is True
        and task["motion_prior"]["defines_ball_route"] is False
        and task["motion_prior"]["defines_planner_command"] is False
        and task["motion_prior"]["defines_station"] is False
        and task["planner_command_distribution"]["station"][
            "motion_contact_frame_defines_station"
        ]
        is False
    )
    _require(
        checks["task_motion_is_prior_only"],
        "motion must not define ball route, planner command, or station",
        errors,
    )
    checks["task_curriculum_is_capability_driven"] = (
        task["curriculum"]["driver"] == "capability_ema"
        and task["curriculum"]["iteration_or_wall_clock_driver"] == "forbidden"
        and task["curriculum"]["rollback"]["enabled"] is True
        and task["curriculum"]["promotion"]["hysteresis_required"] is True
    )
    _require(
        checks["task_curriculum_is_capability_driven"],
        "curriculum must use capability EMA, hysteresis, and rollback",
        errors,
    )
    first_motion = _repo_path(
        task["motion_prior"]["first_stage"]["source_clip"]
    )
    checks["task_first_motion_exists"] = first_motion.is_file()
    _require(
        checks["task_first_motion_exists"],
        f"missing first-stage motion: {first_motion}",
        errors,
    )
    if (
        task["implementation"]["active_task_wired"] is not True
        or task["status"] != "train_ready"
    ):
        blockers.append("unified training task is not wired or train ready")

    branches = matrix["initialization_ab"]["branches"]
    branch_ids = [branch["id"] for branch in branches]
    checks["experiment_initialization_ab_is_controlled"] = (
        len(branches) == 2
        and len(set(branch_ids)) == 2
        and matrix["initialization_ab"]["causal_variable"] == "initialization"
        and matrix["initialization_ab"][
            "all_other_task_action_reward_motion_and_optimizer_settings_equal"
        ]
        is True
        and branches[1]["actor"]["first_deterministic_q_des"] == "default_q"
        and "generic_zero_output_bias" in branches[1]["forbidden"]
    )
    _require(
        checks["experiment_initialization_ab_is_controlled"],
        "experiment A/B must differ only by safe initialization",
        errors,
    )
    stages = matrix["training_budgets"]["stages"]
    stage_ids = [stage["id"] for stage in stages]
    long_stage = next(
        (stage for stage in stages if stage["id"] == "T5_LONG"), None
    )
    checks["experiment_budgets_are_gated"] = (
        len(stage_ids) == len(set(stage_ids))
        and matrix["common_training_contract"]["curriculum_driver"]
        == "capability_ema"
        and matrix["common_training_contract"][
            "curriculum_iteration_schedule"
        ]
        == "forbidden"
        and long_stage is not None
        and long_stage["enabled"] is False
        and matrix["long_training_authorized"] is False
    )
    _require(
        checks["experiment_budgets_are_gated"],
        "long training must be disabled and capability-gated",
        errors,
    )
    suite_ids = {
        suite["id"] for suite in matrix["fixed_evaluation"]["suites"]
    }
    required_suites = {
        "E_READY_NO_COMMAND",
        "E_ISAAC_PHYSICAL_CORE",
        "E_MUJOCO_FIXED_COMMAND",
        "E_MUJOCO_PLANNER_REPLAY",
        "E_PLANNER_EXECUTION",
        "E_CONTINUOUS",
    }
    checks["experiment_evaluation_is_cross_sim_closed_loop"] = (
        required_suites.issubset(suite_ids)
        and matrix["fixed_evaluation"][
            "common_command_and_seed_manifest_required"
        ]
        is True
    )
    _require(
        checks["experiment_evaluation_is_cross_sim_closed_loop"],
        "experiment matrix lacks a common-manifest cross-simulator evaluation",
        errors,
    )
    checks["experiment_execution_is_blocked"] = (
        matrix["status"] == "blocked_preflight_only"
        and matrix["execution"]["launcher_created"] is False
        and matrix["execution"]["no_gpu_job_started"] is True
    )
    _require(
        checks["experiment_execution_is_blocked"],
        "blocked experiment matrix must not claim a launcher or GPU job",
        errors,
    )

    contract_set_hash = hashlib.sha256()
    contract_set_hash.update(contact_path.read_bytes())
    contract_set_hash.update(b"\0")
    contract_set_hash.update(action_path.read_bytes())
    contract_set_hash.update(b"\0")
    contract_set_hash.update(migration_path.read_bytes())
    contract_set_hash.update(b"\0")
    contract_set_hash.update(task_path.read_bytes())
    contract_set_hash.update(b"\0")
    contract_set_hash.update(matrix_path.read_bytes())
    hashes = {
        "contact_contract_sha256": _sha256(contact_path),
        "action_contract_sha256": _sha256(action_path),
        "migration_contract_sha256": _sha256(migration_path),
        "task_contract_sha256": _sha256(task_path),
        "experiment_matrix_sha256": _sha256(matrix_path),
        "contract_set_sha256": contract_set_hash.hexdigest(),
    }
    if contact_source.is_file():
        hashes["ball_physics_sha256"] = _sha256(contact_source)
    if order_path.is_file():
        hashes["joint_order_sha256"] = _sha256(order_path)
    if adapter_path.is_file():
        hashes["legacy_action_adapter_sha256"] = _sha256(adapter_path)

    result = {
        "contact_contract": str(contact_path),
        "action_contract": str(action_path),
        "migration_contract": str(migration_path),
        "task_contract": str(task_path),
        "experiment_matrix": str(matrix_path),
        "schema_valid": not errors,
        "train_ready": not errors and not blockers,
        "checks": checks,
        "hashes": hashes,
        "errors": errors,
        "blockers": blockers,
        "decision": "PASS" if not errors else "FAIL",
        "promotion_decision": "PASS" if not errors and not blockers else "BLOCKED",
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--contact-contract", type=Path, default=DEFAULT_CONTACT)
    parser.add_argument("--action-contract", type=Path, default=DEFAULT_ACTION)
    parser.add_argument("--migration-contract", type=Path, default=DEFAULT_MIGRATION)
    parser.add_argument("--task-contract", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--experiment-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-train-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_contracts(
        args.contact_contract,
        args.action_contract,
        args.migration_contract,
        args.task_contract,
        args.experiment_matrix,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if not result["schema_valid"]:
        return 1
    if args.require_train_ready and not result["train_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
