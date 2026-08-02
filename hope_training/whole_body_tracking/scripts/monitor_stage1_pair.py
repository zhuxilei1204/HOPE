#!/usr/bin/env python3
"""Record Stage-1 milestone health from the two tmux console logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time


ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITERATION = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
VALUE = re.compile(r"^\s*([^:]+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*$")
MILESTONES = (250, 500, 750, 1000, 1500)
FIELDS = {
    "mean_reward": "Mean reward",
    "mean_episode_length": "Mean episode length",
    "planner_crossfade_reward": "Episode_Reward/planner_racket_task_space_crossfade",
    "velocity_progress_reward": "Episode_Reward/near_impact_planner_velocity_progress",
    "joint_limit_reward": "Episode_Reward/joint_limit",
    "overflow_reward": "Episode_Reward/phase_action_overflow",
    "operational_margin_reward": "Episode_Reward/operational_joint_margin",
    "joint_target_slew_reward": "Episode_Reward/joint_target_slew",
    "racket_pos_error_m": "Metrics/racket_target/impact_planner_pos_error_m",
    "racket_vel_error_mps": "Metrics/racket_target/impact_planner_vel_error_mps",
    "racket_speed_ratio": "Metrics/racket_target/impact_planner_speed_ratio",
    "racket_vel_angle_deg": "Metrics/racket_target/impact_planner_vel_angle_deg",
    "racket_normal_error_deg": "Metrics/racket_target/impact_normal_error_deg",
    "impact_health": "Metrics/racket_target/impact_health_score",
    "recovery_ready": "Metrics/racket_target/recovery_ready_score",
    "action_clamp_fraction": "Metrics/racket_target/action_clamp_fraction",
    "waist_clamp_fraction": "Metrics/racket_target/waist_action_clamp_fraction",
    "right_arm_clamp_fraction": "Metrics/racket_target/right_arm_action_clamp_fraction",
    "leg_clamp_fraction": "Metrics/racket_target/leg_action_clamp_fraction",
    "waist_overflow_rms": "Metrics/racket_target/waist_action_overflow_rms",
    "right_arm_overflow_rms": "Metrics/racket_target/right_arm_action_overflow_rms",
    "leg_overflow_rms": "Metrics/racket_target/leg_action_overflow_rms",
    "operational_margin_fraction": "Metrics/racket_target/operational_margin_fraction",
    "waist_operational_margin_fraction": "Metrics/racket_target/waist_operational_margin_fraction",
    "right_arm_operational_margin_fraction": "Metrics/racket_target/right_arm_operational_margin_fraction",
    "leg_operational_margin_fraction": "Metrics/racket_target/leg_operational_margin_fraction",
    "q_des_velocity_violation_fraction": "Metrics/racket_target/q_des_velocity_violation_fraction",
    "q_des_acceleration_violation_fraction": "Metrics/racket_target/q_des_acceleration_violation_fraction",
    "action_operational_feasibility_score": "Metrics/racket_target/action_operational_feasibility_score",
    "waist_action_delta_rms": "Metrics/racket_target/waist_action_delta_rms",
    "leg_action_delta_rms": "Metrics/racket_target/leg_action_delta_rms",
    "torso_angular_velocity": "Metrics/racket_target/impact_health_torso_ang_vel",
    "severe_overflow": "Metrics/racket_target/actuator_overflow_severe",
    "waist_torque_clip": "Metrics/racket_target/waist_torque_clip_fraction",
    "right_arm_torque_clip": "Metrics/racket_target/right_arm_torque_clip_fraction",
    "leg_torque_clip": "Metrics/racket_target/leg_torque_clip_fraction",
    "base_low_termination": "Episode_Termination/base_too_low",
    "tilt_termination": "Episode_Termination/base_tilted",
    "table_termination": "Episode_Termination/table_touch",
    "overflow_termination": "Episode_Termination/persistent_action_overflow",
}


def read_tail(path: Path, byte_count: int = 2_000_000) -> str:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(size - byte_count, 0))
        return stream.read().decode("utf-8", errors="replace")


def latest_complete_iteration(path: Path) -> dict | None:
    text = ANSI.sub("", read_tail(path))
    matches = list(ITERATION.finditer(text))
    if not matches:
        return None
    for index in range(len(matches) - 1, -1, -1):
        start = matches[index].start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]
        if "Total timesteps:" not in block:
            continue
        values: dict[str, float] = {}
        for line in block.splitlines():
            match = VALUE.match(line)
            if match:
                values[match.group(1).strip()] = float(match.group(2))
        current, maximum = (int(value) for value in matches[index].groups())
        result = {"iteration": current, "max_iterations": maximum}
        result.update(
            {
                output: values.get(console_name)
                for output, console_name in FIELDS.items()
            }
        )
        return result
    return None


def classify(metrics: dict) -> list[str]:
    warnings: list[str] = []
    iteration = int(metrics["iteration"])
    if (metrics.get("severe_overflow") or 0.0) > 1.0e-4 or (
        metrics.get("overflow_termination") or 0.0
    ) > 0.0:
        warnings.append("severe_or_persistent_overflow")
    if iteration >= 250:
        if (metrics.get("action_clamp_fraction") or 0.0) > 0.03:
            warnings.append("overall_action_clamp_gt_3pct")
        if (metrics.get("waist_clamp_fraction") or 0.0) > 0.08:
            warnings.append("waist_clamp_gt_8pct")
        if (metrics.get("leg_clamp_fraction") or 0.0) > 0.05:
            warnings.append("leg_clamp_gt_5pct")
        if (metrics.get("operational_margin_fraction") or 0.0) > 0.08:
            warnings.append("operational_margin_outside_gt_8pct")
        if (metrics.get("waist_operational_margin_fraction") or 0.0) > 0.15:
            warnings.append("waist_operational_outside_gt_15pct")
        if (metrics.get("leg_operational_margin_fraction") or 0.0) > 0.10:
            warnings.append("leg_operational_outside_gt_10pct")
        if (metrics.get("q_des_velocity_violation_fraction") or 0.0) > 0.08:
            warnings.append("q_des_velocity_violation_gt_8pct")
        if (metrics.get("q_des_acceleration_violation_fraction") or 0.0) > 0.15:
            warnings.append("q_des_acceleration_violation_gt_15pct")
        if (metrics.get("action_operational_feasibility_score") or 1.0) < 0.80:
            warnings.append("action_operational_feasibility_below_0p8")
        stable = (metrics.get("mean_episode_length") or 0.0) >= 400.0
        weak_command = (
            (metrics.get("racket_speed_ratio") or 0.0) < 0.45
            and (metrics.get("racket_pos_error_m") or 999.0) > 0.40
        )
        if stable and weak_command:
            warnings.append("stable_but_command_tracking_weak")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=CONSOLE_LOG",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    runs = {}
    for value in args.run:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --run value: {value!r}")
        runs[name] = Path(path)

    emitted = {name: set() for name in runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        all_finished = True
        with args.output.open("a", encoding="utf-8") as stream:
            for name, path in runs.items():
                metrics = latest_complete_iteration(path)
                if metrics is None:
                    all_finished = False
                    continue
                if metrics["iteration"] < metrics["max_iterations"] - 1:
                    all_finished = False
                for milestone in MILESTONES:
                    threshold = min(
                        milestone, int(metrics["max_iterations"]) - 1
                    )
                    if metrics["iteration"] < threshold or milestone in emitted[name]:
                        continue
                    record = {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "run": name,
                        "milestone": milestone,
                        "observed_iteration": metrics["iteration"],
                        "metrics": metrics,
                        "warnings": classify(metrics),
                    }
                    rendered = json.dumps(record, ensure_ascii=True)
                    print(rendered, flush=True)
                    stream.write(rendered + "\n")
                    stream.flush()
                    emitted[name].add(milestone)
        if all_finished:
            return
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    main()
