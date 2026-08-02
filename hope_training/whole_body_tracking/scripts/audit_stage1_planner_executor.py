#!/usr/bin/env python3
"""Audit a resolved Stage1 PlannerExecutor run against its frozen contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "cfg/contracts/hope_stage1_planner_executor_v1.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.unsafe_load(stream)


def reward_weight(rewards: dict, name: str) -> float:
    term = rewards.get(name)
    return 0.0 if term is None else float(term.get("weight", 0.0))


def audit(run_dir: Path) -> dict:
    contract = load_yaml(CONTRACT_PATH)
    env = load_yaml(run_dir / "params/env.yaml")
    agent = load_yaml(run_dir / "params/agent.yaml")
    motion = env["commands"]["motion"]
    command = env["commands"]["racket_target"]
    policy = env["observations"]["policy"]
    rewards = env["rewards"]
    terminations = env["terminations"]
    nonzero = {
        name
        for name, term in rewards.items()
        if term is not None and abs(float(term.get("weight", 0.0))) > 0.0
    }
    required = set(contract["required_nonzero_rewards"])
    forbidden = set(contract["forbidden_reward_terms"])

    exact_params = rewards[
        "exact_impact_planner_task_space_alignment"
    ]["params"]
    cycle_params = rewards["safe_recovered_planner_command"]["params"]
    failure_params = rewards["command_cycle_failure"]["params"]
    checks = {
        "control_contract": int(env["decimation"]) == 4
        and abs(float(env["sim"]["dt"]) - 0.005) < 1.0e-12
        and float(env["episode_length_s"]) == 10.0,
        "normal114": policy["racket_target_normal_w"] is not None
        and policy["stability_feedback"] is None,
        "two_balanced_motions": isinstance(motion["motion_file"], list)
        and len(motion["motion_file"]) == 2
        and tuple(motion["clip_sampling_weights"]) == (0.5, 0.5),
        "continuous_carried_state": motion["wrap_teleport"] is False,
        "no_rigid_ball_command": "physical_ball_shadow"
        not in env["commands"],
        "fixed_planner_plane": command["strike_position_mode"]
        == "table_workspace"
        and command["planner_hit_plane_mode"] == "fixed_x_hit"
        and abs(float(command["planner_hit_plane_x"]) - 0.20)
        < 1.0e-12,
        "coherent_clean_command": command["racket_velocity_mode"]
        == "impact_inverse_landing"
        and command["incoming_trajectory_mode"] == "one_bounce"
        and command["planner_command_mode"] == "v4_wire_compatible"
        and float(command["planner_perturb_fixed_scale"]) == 0.0,
        "execution_driven_curriculum": command[
            "ability_curriculum_enabled"
        ]
        is True
        and float(command["ability_curriculum_net_threshold"]) == 0.0
        and float(command["ability_curriculum_success_threshold"]) == 0.0
        and command["ability_curriculum_require_side_contact"] is True,
        "targeted_attempt_recovery": command[
            "post_contact_ready_enabled"
        ]
        is True
        and command["post_contact_ready_trigger"] == "targeted_attempt",
        "impulse_accounting": exact_params["impulse"] is True
        and cycle_params["impulse"] is True
        and failure_params["impulse"] is True
        and rewards["targeted_contact_miss"]["params"]["impulse"]
        is True
        and rewards["safe_strike_inactivity"]["params"]["impulse"]
        is True
        and rewards["recovery_peak_ang_vel_excess"]["params"][
            "impulse"
        ]
        is True,
        "reachable_command_bootstrap": reward_weight(
            rewards, "racket_position"
        )
        > 0.0
        and reward_weight(rewards, "racket_velocity") > 0.0
        and reward_weight(rewards, "blade_direction") > 0.0
        and abs(reward_weight(rewards, "termination_penalty") + 12.0)
        < 1.0e-12,
        "recovery_gradient_without_arm_suppression": reward_weight(
            rewards, "healthy_trunk_support"
        )
        > 0.0
        and reward_weight(rewards, "post_strike_base_ang_vel") > 0.0
        and reward_weight(rewards, "recovery_peak_ang_vel_excess") < 0.0
        and reward_weight(rewards, "table_no_touch") < 0.0
        and rewards["table_no_touch"]["params"]["ability_scaled"] is True
        and rewards["post_strike_base_ang_vel"]["params"][
            "ability_scaled_std"
        ]
        is True
        and rewards["post_strike_base_ang_vel"]["params"]["include_hold"]
        is False
        and rewards["recovery_peak_ang_vel_excess"]["params"][
            "ability_scaled"
        ]
        is True,
        "exact_reward_whitelist": nonzero == required,
        "physical_outcomes_forbidden": not (nonzero & forbidden),
        "hard_safety": terminations["base_too_low"] is not None
        and terminations["base_tilted"] is not None
        and terminations["table_touch"] is not None
        and terminations["persistent_action_overflow"] is not None
        and terminations["anchor_pos"] is None
        and terminations["anchor_ori"] is None
        and terminations["ee_body_pos"] is None,
        "scratch_ppo": abs(float(agent["policy"]["init_noise_std"]) - 0.45)
        < 1.0e-12
        and abs(float(agent["algorithm"]["learning_rate"]) - 0.0005)
        < 1.0e-12,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "nonzero_rewards": sorted(nonzero),
        "missing_rewards": sorted(required - nonzero),
        "unexpected_rewards": sorted(nonzero - required),
        "forbidden_nonzero": sorted(nonzero & forbidden),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    rendered = json.dumps(result, ensure_ascii=True, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
