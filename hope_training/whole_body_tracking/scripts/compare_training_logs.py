#!/usr/bin/env python3
"""Compare aligned trailing windows from HOPE rsl_rl text logs."""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")
VALUE_RE = re.compile(r"^\s*([^:]+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")

DEFAULT_METRICS = (
    "Metrics/racket_target/return_success",
    "Metrics/motion/error_body_pos_lower",
    "Metrics/motion/error_body_rot_lower",
    "Metrics/motion/error_body_lin_vel_lower",
    "Metrics/motion/error_body_ang_vel_lower",
    "Metrics/motion/error_joint_pos_legs",
    "Metrics/motion/error_joint_vel_legs",
    "Metrics/racket_target/ability_contact_ema",
    "Metrics/racket_target/ability_net_ema",
    "Metrics/racket_target/ability_success_ema",
    "Metrics/racket_target/recovery_ready_score",
    "Metrics/racket_target/recovery_base_lin_vel",
    "Metrics/racket_target/recovery_base_ang_vel",
    "Metrics/racket_target/recovery_station_error",
    "Metrics/racket_target/recovery_feet_contact_frac",
    "Metrics/racket_target/recovery_racket_speed",
    "Metrics/racket_target/base_pitch_like_signed",
    "Metrics/racket_target/base_backward_velocity",
    "Metrics/racket_target/action_abs_max",
    "Metrics/racket_target/applied_action_abs_max",
    "Metrics/racket_target/effective_action_abs_max",
    "Metrics/racket_target/action_clamp_fraction",
    "Metrics/racket_target/waist_action_clamp_fraction",
    "Metrics/racket_target/right_arm_action_clamp_fraction",
    "Metrics/racket_target/leg_action_clamp_fraction",
    "Metrics/racket_target/waist_action_overflow_rms",
    "Metrics/racket_target/right_arm_action_overflow_rms",
    "Metrics/racket_target/leg_action_overflow_rms",
    "Metrics/racket_target/waist_q_tracking_error_rms",
    "Metrics/racket_target/right_arm_q_tracking_error_rms",
    "Metrics/racket_target/leg_q_tracking_error_rms",
    "Metrics/racket_target/waist_torque_clip_fraction",
    "Metrics/racket_target/right_arm_torque_clip_fraction",
    "Metrics/racket_target/leg_torque_clip_fraction",
    "Episode_Termination/base_too_low",
    "Episode_Termination/base_tilted",
)


def _parse(path: pathlib.Path) -> list[dict]:
    blocks = []
    current = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = ITER_RE.search(line)
        if match:
            if current is not None:
                blocks.append(current)
            current = {"iteration": int(match.group(1))}
            continue
        if current is None:
            continue
        match = VALUE_RE.match(line)
        if match:
            current[match.group(1).strip()] = float(match.group(2))
    if current is not None:
        blocks.append(current)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    parsed = [(pathlib.Path(value), _parse(pathlib.Path(value))) for value in args.logs]
    common_end = min(
        (blocks[-1]["iteration"] for _path, blocks in parsed if blocks),
        default=0,
    )
    common_start = common_end - args.window + 1
    lines = [
        f"# Training log comparison (aligned iterations {common_start}..{common_end})",
        "",
        "| metric | " + " | ".join(path.stem for path, _ in parsed) + " |",
        "|---|" + "---:|" * len(parsed),
    ]
    for metric in DEFAULT_METRICS:
        values = []
        for _path, blocks in parsed:
            tail = [
                block
                for block in blocks
                if common_start <= block["iteration"] <= common_end
            ]
            observed = [block[metric] for block in tail if metric in block]
            values.append("" if not observed else f"{statistics.fmean(observed):.5f}")
        lines.append(f"| {metric} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "| run | first iteration | last iteration | parsed blocks |",
            "|---|---:|---:|---:|",
        ]
    )
    for path, blocks in parsed:
        first = blocks[0]["iteration"] if blocks else "-"
        last = blocks[-1]["iteration"] if blocks else "-"
        lines.append(f"| {path.stem} | {first} | {last} | {len(blocks)} |")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.markdown_out:
        target = pathlib.Path(args.markdown_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
