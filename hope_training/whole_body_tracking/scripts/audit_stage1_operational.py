#!/usr/bin/env python3
"""Audit the resolved Stage-1 v2 operational-control contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "cfg/contracts/hope_stage1_operational_v2.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.unsafe_load(stream)


def reward_weight(rewards: dict, name: str) -> float:
    term = rewards.get(name)
    return 0.0 if term is None else float(term.get("weight", 0.0))


def _ordered_map(values: dict, names: list[str]) -> np.ndarray:
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"joint map is missing names: {missing}")
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def _motion_envelope(motion_files: list[str], count: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(count, np.inf, dtype=np.float64)
    upper = np.full(count, -np.inf, dtype=np.float64)
    for value in motion_files:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        with np.load(path, allow_pickle=False) as data:
            q = np.asarray(data["joint_pos"], dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != count:
            raise ValueError(f"{path} has invalid joint_pos shape {q.shape}")
        lower = np.minimum(lower, q.min(axis=0))
        upper = np.maximum(upper, q.max(axis=0))
    return lower, upper


def audit(run_dir: Path) -> dict:
    contract = load_yaml(CONTRACT_PATH)
    env = load_yaml(run_dir / "params/env.yaml")
    agent = load_yaml(run_dir / "params/agent.yaml")
    motion = env["commands"]["motion"]
    command = env["commands"]["racket_target"]
    action = env["actions"]["joint_pos"]
    rewards = env["rewards"]
    policy = env["observations"]["policy"]
    terminations = env["terminations"]

    names = list(action["joint_names"])
    hard = action["position_clamp"]
    hard_lower = _ordered_map(
        {name: bounds[0] for name, bounds in hard.items()}, names
    )
    hard_upper = _ordered_map(
        {name: bounds[1] for name, bounds in hard.items()}, names
    )
    margin = _ordered_map(action["operational_margin_fraction"], names)
    operational_lower = hard_lower + margin * (hard_upper - hard_lower)
    operational_upper = hard_upper - margin * (hard_upper - hard_lower)
    default_q = _ordered_map(env["scene"]["robot"]["init_state"]["joint_pos"], names)
    motion_lower, motion_upper = _motion_envelope(motion["motion_file"], len(names))

    nonzero = {
        name
        for name, term in rewards.items()
        if term is not None and abs(float(term.get("weight", 0.0))) > 0.0
    }
    required = set(contract["required_nonzero_rewards"])
    forbidden = set(contract["forbidden_reward_terms"])
    planner_params = rewards["planner_racket_task_space_crossfade"]["params"]
    checks = {
        "control_contract": int(env["decimation"]) == 4
        and abs(float(env["sim"]["dt"]) - 0.005) < 1.0e-12
        and float(env["episode_length_s"]) == 10.0,
        "normal114": policy["racket_target_normal_w"] is not None
        and policy["stability_feedback"] is None,
        "two_static_motions": isinstance(motion["motion_file"], list)
        and len(motion["motion_file"]) == 2,
        "continuous_no_rsi": motion["wrap_teleport"] is False
        and motion["motion_start_warmup_enabled"] is True
        and float(motion["motion_start_warmup_start_prob"]) == 0.75
        and float(motion["motion_start_warmup_min_prob"]) == 0.75,
        "simple_command_distribution": command["strike_position_mode"] == "motion_box"
        and command["planner_command_mode"] == "v4_wire_compatible"
        and command["station_relocation_enabled"] is False
        and command["cycle_v2_enabled"] is False,
        "operational_maps_complete": len(names) == 31
        and len(action["operational_margin_fraction"]) == 31
        and len(action["q_des_velocity_limit"]) == 31
        and len(action["q_des_acceleration_limit"]) == 31,
        "operational_range_valid": bool(
            np.all(operational_lower < operational_upper)
        ),
        "default_inside_operational": bool(
            np.all(default_q >= operational_lower)
            and np.all(default_q <= operational_upper)
        ),
        "motion_inside_operational": bool(
            np.all(motion_lower >= operational_lower)
            and np.all(motion_upper <= operational_upper)
        ),
        "positive_dynamic_limits": bool(
            np.all(_ordered_map(action["q_des_velocity_limit"], names) > 0.0)
            and np.all(
                _ordered_map(action["q_des_acceleration_limit"], names) > 0.0
            )
        ),
        "command_feasibility_gate": planner_params[
            "action_feasibility_metric"
        ]
        == "action_operational_feasibility_score"
        and abs(float(planner_params["action_feasibility_floor"]) - 0.35)
        < 1.0e-12,
        "exact_reward_whitelist": nonzero == required,
        "no_physical_reward": not (nonzero & forbidden),
        "hard_safety_retained": terminations["base_too_low"] is not None
        and terminations["base_tilted"] is not None
        and terminations["table_touch"] is not None
        and terminations["persistent_action_overflow"] is not None,
        "scratch_ppo": abs(float(agent["policy"]["init_noise_std"]) - 0.60)
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
        "minimum_motion_operational_clearance_rad": float(
            np.min(
                np.minimum(
                    motion_lower - operational_lower,
                    operational_upper - motion_upper,
                )
            )
        ),
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

