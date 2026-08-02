#!/usr/bin/env python3
"""Audit the resolved Stage-1 task and replay its reward budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "cfg" / "contracts" / "hope_stage1_command_tracking_v1.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.unsafe_load(stream)


def synthetic_reward_replay(contract: dict) -> dict[str, float]:
    cfg = contract["synthetic_replay"]
    command = float(cfg["command_weight"])
    motion = float(cfg["motion_weight"])
    health = float(cfg["health_weight"])
    ready = float(cfg["ready_weight"])
    floor = float(cfg["command_health_floor"])
    return {
        "healthy_track": command + 0.8 * motion + health + 0.4 * ready,
        "unsafe_track": floor * command + 0.8 * motion + 0.1 * health,
        "healthy_idle": 0.05 * command + 0.8 * motion + health + ready,
        "unsafe_idle": 0.05 * floor * command + 0.3 * motion + 0.1 * health,
    }


def reward_weight(rewards: dict, name: str) -> float:
    term = rewards.get(name)
    return 0.0 if term is None else float(term.get("weight", 0.0))


def audit(run_dir: Path) -> dict:
    contract = load_yaml(CONTRACT_PATH)
    env = load_yaml(run_dir / "params" / "env.yaml")
    agent = load_yaml(run_dir / "params" / "agent.yaml")
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
    replay = synthetic_reward_replay(contract)
    ratio = replay["healthy_track"] / max(replay["healthy_idle"], 1.0e-9)
    checks = {
        "control_contract": int(env["decimation"]) == 4
        and abs(float(env["sim"]["dt"]) - 0.005) < 1.0e-12
        and float(env["episode_length_s"]) == 10.0,
        "normal114": policy["racket_target_normal_w"] is not None
        and policy["stability_feedback"] is None,
        "two_static_motions": isinstance(motion["motion_file"], list)
        and len(motion["motion_file"]) == 2
        and tuple(motion["clip_sampling_weights"]) == (0.5, 0.5),
        "continuous_no_rsi": motion["wrap_teleport"] is False
        and motion["motion_start_warmup_enabled"] is True
        and float(motion["motion_start_warmup_start_prob"]) == 0.75
        and float(motion["motion_start_warmup_min_prob"]) == 0.75
        and motion["motion_start_warmup_prestrike_enabled"] is False,
        "bounded_ready_exposure": float(motion["stand_episode_prob"]) == 0.08
        and float(command["deploy_ready_hold_prob"]) == 0.20,
        "command_contract": command["racket_velocity_mode"] == "range"
        and command["planner_command_mode"] == "v4_wire_compatible"
        and command["strike_position_mode"] == "motion_box"
        and command["station_mode"] == "dynamic_from_motion",
        "complex_lifecycle_disabled": command["station_relocation_enabled"] is False
        and command["healthy_three_stage_enabled"] is False
        and command["cycle_v2_enabled"] is False
        and command["single_cycle_curriculum_enabled"] is False
        and command["post_contact_ready_enabled"] is False,
        "no_lower_body_motion_prior": reward_weight(
            rewards, "phase_lower_body_motion_prior"
        )
        == 0.0,
        "exact_reward_whitelist": nonzero == required,
        "no_physical_reward": not (nonzero & forbidden),
        "hard_safety_only": terminations["base_too_low"] is not None
        and terminations["base_tilted"] is not None
        and terminations["table_touch"] is not None
        and terminations["persistent_action_overflow"] is not None
        and terminations["anchor_pos"] is None
        and terminations["anchor_ori"] is None
        and terminations["ee_body_pos"] is None,
        "scratch_ppo": abs(float(agent["policy"]["init_noise_std"]) - 0.60) < 1.0e-12
        and abs(float(agent["algorithm"]["learning_rate"]) - 0.0005) < 1.0e-12,
        "reward_replay_track_beats_idle": ratio
        >= float(contract["synthetic_replay"]["required_healthy_track_over_idle_ratio"]),
        "reward_replay_unsafe_has_gradient": replay["unsafe_track"] > replay["unsafe_idle"],
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "nonzero_rewards": sorted(nonzero),
        "missing_rewards": sorted(required - nonzero),
        "unexpected_rewards": sorted(nonzero - required),
        "synthetic_replay": replay,
        "healthy_track_over_idle_ratio": ratio,
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
