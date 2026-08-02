#!/usr/bin/env python3
"""Audit the C7 immediate-vs-escrow phase-curriculum experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


TASK_A = "HOPEPingPongClosedLoopV3ScratchPhaseImmediateC7AMultiSkill114"
TASK_B = "HOPEPingPongClosedLoopV3ScratchPhaseEscrowC7BMultiSkill114"
ALLOWED_AB_DIFFERENCES = {
    "rewards.health_gated_soft_ball_contact.weight",
    "rewards.exact_impact_planner_task_space_alignment.weight",
    "rewards.physical_outcome.weight",
    "rewards.cycle_v2_ready_success_bonus.weight",
    "rewards.cycle_v2_streak_bonus.weight",
}


def _compose_task(root: Path, task_name: str) -> dict:
    with initialize_config_dir(config_dir=str(root / "cfg"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                f"task={task_name}",
                "algo=ppo_closed_loop_v3_scratch",
            ],
        )
    return OmegaConf.to_container(cfg.task, resolve=True)


def _contract_checks(task: dict) -> dict[str, bool]:
    overrides = task["overrides"]
    return {
        "normal114_contract": task["actor_obs_contract"]
        == "hope_pingpong_normal114",
        "five_motion_manifest": "supplemental_lower_body_contact_aware"
        in task["motion_manifest"],
        "route_independent_of_motion": float(
            overrides["commands.racket_target.ability_curriculum_start_racket_pos_scale"][0]
        )
        == 1.0,
        "separate_route_curriculum": overrides[
            "commands.racket_target.table_workspace_level_source"
        ]
        == "station_relocation",
        "separate_speed_curriculum": overrides[
            "commands.racket_target.one_bounce_speed_curriculum_enabled"
        ]
        is True,
        "fixed_planner_noise": overrides[
            "commands.racket_target.planner_perturb_curriculum_source"
        ]
        == "fixed",
        "relocation_hold_gate": overrides[
            "commands.racket_target.station_relocation_apply_to_swing_holds"
        ]
        is True,
        "recovery_hold_gate": overrides[
            "commands.racket_target.lifecycle_recovery_hold_gate_enabled"
        ]
        is True,
        "actual_outcome_settlement": overrides[
            "commands.racket_target.cycle_v2_settlement_tier_mode"
        ]
        == "achieved",
        "ability_not_iteration": int(
            overrides["commands.racket_target.ability_curriculum_min_resolved_events"]
        )
        >= 4096,
        "bounded_relocation_deadlines": len(
            overrides[
                "commands.racket_target.station_relocation_hold_deadline_steps_by_level"
            ]
        )
        == len(
            overrides["commands.racket_target.station_relocation_abs_y_ranges"]
        ),
        "nonfarmable_relocation_progress": overrides[
            "rewards.prestrike_station_progress.params"
        ]["include_no_command_ready"]
        is True,
    }


def _run_checks(run_dir: Path, task_name: str) -> dict[str, bool]:
    with (run_dir / "params" / "env.yaml").open("r", encoding="utf-8") as stream:
        env = yaml.unsafe_load(stream)
    command = env["commands"]["racket_target"]
    policy = env["observations"]["policy"]
    rewards = env["rewards"]
    expected_cycle_weight = 3.0 if task_name == TASK_A else 16.0
    expected_contact_weight = 2.8 if task_name == TASK_A else 0.8
    return {
        "resolved_normal114": policy["racket_target_normal_w"] is not None
        and policy["stability_feedback"] is None,
        "resolved_route_curriculum": command["table_workspace_level_source"]
        == "station_relocation",
        "resolved_speed_curriculum": command["one_bounce_speed_curriculum_enabled"]
        is True,
        "resolved_relocation_gate": command[
            "station_relocation_apply_to_swing_holds"
        ]
        is True,
        "resolved_recovery_gate": command["lifecycle_recovery_hold_gate_enabled"]
        is True,
        "resolved_actual_tier": command["cycle_v2_settlement_tier_mode"]
        == "achieved",
        "resolved_relocation_progress": rewards["prestrike_station_progress"][
            "params"
        ]["include_no_command_ready"]
        is True,
        "resolved_contact_weight": float(
            rewards["health_gated_soft_ball_contact"]["weight"]
        )
        == expected_contact_weight,
        "resolved_cycle_weight": float(
            rewards["cycle_v2_ready_success_bonus"]["weight"]
        )
        == expected_cycle_weight,
    }


def audit(root: Path, run_dir: Path | None = None, task_name: str | None = None) -> dict:
    task_a = _compose_task(root, TASK_A)
    task_b = _compose_task(root, TASK_B)
    overrides_a = task_a["overrides"]
    overrides_b = task_b["overrides"]
    differing = {
        key
        for key in set(overrides_a) | set(overrides_b)
        if overrides_a.get(key) != overrides_b.get(key)
    }
    result = {
        "checks": _contract_checks(task_a),
        "ab_differences": sorted(differing),
        "ab_isolated": differing == ALLOWED_AB_DIFFERENCES,
    }
    if run_dir is not None:
        if task_name not in (TASK_A, TASK_B):
            raise ValueError("--task is required with --run-dir and must name C7A or C7B")
        result["run_checks"] = _run_checks(run_dir, task_name)
    checks = list(result["checks"].values())
    checks.append(bool(result["ab_isolated"]))
    checks.extend(result.get("run_checks", {}).values())
    result["passed"] = all(checks)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--task", choices=(TASK_A, TASK_B))
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.run_dir, args.task)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
