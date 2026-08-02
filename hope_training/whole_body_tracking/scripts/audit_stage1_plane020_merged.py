#!/usr/bin/env python3
"""Audit the resolved Plane020 merged foundation and its reward budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "cfg/contracts/hope_stage1_plane020_merged_v1.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.unsafe_load(stream)


def reward_weight(rewards: dict, name: str) -> float:
    term = rewards.get(name)
    return 0.0 if term is None else float(term.get("weight", 0.0))


def reward_budget(contract: dict) -> dict[str, float]:
    cfg = contract["reward_budget"]
    active = (
        float(cfg["task_window_steps"]) * float(cfg["task_window_weight"])
        + float(cfg["prestrike_progress_steps"]) * float(cfg["progress_weight"])
        + float(cfg["event_weight"])
        + float(cfg["strike_balance_weight"])
        * float(cfg["task_window_steps"])
    )
    idle = (
        float(cfg["persistent_ready_weight"])
        * (
            float(cfg["task_window_steps"])
            + float(cfg["prestrike_progress_steps"])
            + float(cfg["recovery_steps"])
        )
    )
    return {
        "active": active,
        "idle": idle,
        "active_over_idle": active / max(idle, 1.0e-9),
    }


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
    budget = reward_budget(contract)
    checks = {
        "control_contract": int(env["decimation"]) == 4
        and abs(float(env["sim"]["dt"]) - 0.005) < 1.0e-12
        and float(env["episode_length_s"]) == 10.0,
        "normal114": policy["racket_target_normal_w"] is not None
        and policy["stability_feedback"] is None,
        "two_balanced_static_motions": isinstance(motion["motion_file"], list)
        and len(motion["motion_file"]) == 2
        and tuple(motion["clip_sampling_weights"]) == (0.5, 0.5),
        "continuous_carried_state": motion["wrap_teleport"] is False,
        "bounded_ready_exposure": float(motion["stand_episode_prob"]) <= 0.02
        and float(command["deploy_ready_hold_prob"]) <= 0.05,
        "fixed_plane_table_workspace": command["strike_position_mode"]
        == "table_workspace"
        and command["planner_hit_plane_mode"] == "fixed_x_hit"
        and abs(float(command["planner_hit_plane_x"]) - 0.20) < 1.0e-12,
        "motion_independent_targets": float(
            command["table_workspace_motion_seed_blend_start"]
        )
        == 0.0,
        "explicit_dynamic_station": command["station_mode"]
        == "dynamic_from_motion"
        and abs(float(command["dynamic_station_xy_clip"][1][1])) >= 0.40,
        "ability_gated_workspace": command["ability_curriculum_enabled"] is True
        and int(command["ability_curriculum_min_resolved_events"]) == 4096
        and int(command["ability_curriculum_required_advance_checks"]) == 2,
        "complex_lifecycle_disabled": command["station_relocation_enabled"]
        is False
        and command["healthy_three_stage_enabled"] is False
        and command["cycle_v2_enabled"] is False
        and command["single_cycle_curriculum_enabled"] is False
        and command["post_contact_ready_enabled"] is False,
        "no_lower_body_motion_prior": reward_weight(
            rewards, "phase_lower_body_motion_prior"
        )
        == 0.0,
        "exact_reward_whitelist": nonzero == required,
        "forbidden_rewards_absent": not (nonzero & forbidden),
        "hard_safety_only": terminations["base_too_low"] is not None
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
        "reward_budget_rejects_idle": budget["active_over_idle"]
        >= float(contract["reward_budget"]["required_active_over_idle_ratio"]),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "nonzero_rewards": sorted(nonzero),
        "missing_rewards": sorted(required - nonzero),
        "unexpected_rewards": sorted(nonzero - required),
        "reward_budget": budget,
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
