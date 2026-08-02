#!/usr/bin/env python3
"""Summarize action realizability, clamp, torque, and jitter from a joint trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path")
    parser.add_argument(
        "--scenario",
        choices=("ready", "fake-cycle", "full-task"),
        default="full-task",
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument(
        "--persistent-ticks",
        type=int,
        default=5,
        help="Consecutive overflow ticks treated as persistent feedback pressure.",
    )
    return parser.parse_args()


def _matrix(
    rows: list[dict[str, str]],
    joint_names: list[str],
    prefix: str,
) -> np.ndarray:
    return np.asarray(
        [
            [float(row[f"{prefix}__{joint}"]) for joint in joint_names]
            for row in rows
        ],
        dtype=np.float64,
    )


def _longest_true_runs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    longest = np.zeros(values.shape[1], dtype=np.int64)
    current = np.zeros(values.shape[1], dtype=np.int64)
    for row in values:
        current = np.where(row, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, q)) if values.size else 0.0


def _finite_difference_rms(
    values: np.ndarray,
    consecutive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.diff(values, axis=0)
    first = first[consecutive]
    first_rms = (
        np.sqrt(np.mean(np.square(first), axis=0))
        if first.size
        else np.zeros(values.shape[1], dtype=np.float64)
    )
    second_mask = consecutive[1:] & consecutive[:-1]
    second = np.diff(values, n=2, axis=0)
    second = second[second_mask]
    second_rms = (
        np.sqrt(np.mean(np.square(second), axis=0))
        if second.size
        else np.zeros(values.shape[1], dtype=np.float64)
    )
    return first_rms, second_rms


def _band_power(
    values: np.ndarray,
    sample_hz: float,
    low_hz: float = 5.0,
    high_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] < 16 or sample_hz <= 0.0:
        zeros = np.zeros(values.shape[1], dtype=np.float64)
        return zeros, zeros
    x = values - np.mean(values, axis=0, keepdims=True)
    time = np.arange(x.shape[0], dtype=np.float64)
    design = np.stack((time, np.ones_like(time)), axis=1)
    trend = design @ np.linalg.lstsq(design, x, rcond=None)[0]
    x = (x - trend) * np.hanning(x.shape[0])[:, None]
    spectrum = np.fft.rfft(x, axis=0)
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(x.shape[0], d=1.0 / sample_hz)
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    positive = frequencies > 0.0
    absolute = np.sum(power[band], axis=0) / max(x.shape[0], 1)
    total = np.sum(power[positive], axis=0)
    fraction = np.divide(
        np.sum(power[band], axis=0),
        total,
        out=np.zeros_like(total),
        where=total > 1.0e-12,
    )
    return absolute, fraction


def _group_summary(
    indices: list[int],
    *,
    clamp: np.ndarray,
    overflow: np.ndarray,
    torque_clip: np.ndarray,
    first_rms: np.ndarray,
    second_rms: np.ndarray,
    band_power: np.ndarray,
    band_fraction: np.ndarray,
) -> dict[str, float]:
    return {
        "position_clamp_element_rate": float(np.mean(clamp[:, indices])),
        "overflow_abs_p95": _percentile(
            np.abs(overflow[:, indices]), 0.95
        ),
        "overflow_abs_p99": _percentile(
            np.abs(overflow[:, indices]), 0.99
        ),
        "torque_clip_element_rate": float(
            np.mean(torque_clip[:, indices])
        ),
        "q_des_delta_rms": float(
            math.sqrt(np.mean(np.square(first_rms[indices])))
        ),
        "q_des_second_delta_rms": float(
            math.sqrt(np.mean(np.square(second_rms[indices])))
        ),
        "q_des_5_20hz_power": float(np.mean(band_power[indices])),
        "q_des_5_20hz_power_fraction": float(
            np.mean(band_fraction[indices])
        ),
    }


def analyze(
    rows: list[dict[str, str]],
    joint_names: list[str],
    *,
    source_csv: pathlib.Path,
    scenario: str,
    persistent_ticks: int,
) -> dict:
    raw = _matrix(rows, joint_names, "raw")
    effective = _matrix(rows, joint_names, "effective")
    feedback = _matrix(rows, joint_names, "feedback")
    overflow = _matrix(rows, joint_names, "overflow")
    clamp = _matrix(rows, joint_names, "position_clamped") > 0.5
    q_des = _matrix(rows, joint_names, "q_des")
    torque_clip = _matrix(rows, joint_names, "torque_clipped") > 0.5

    sequence = np.asarray(
        [
            int(float(row.get("sequence_step", index)))
            for index, row in enumerate(rows)
        ],
        dtype=np.int64,
    )
    time_s = np.asarray(
        [float(row.get("sim_time_s", row.get("time_s", index))) for index, row in enumerate(rows)],
        dtype=np.float64,
    )
    time_delta = np.diff(time_s)
    sequence_delta = np.diff(sequence)
    positive_dt = time_delta[time_delta > 1.0e-9]
    dt = float(np.median(positive_dt)) if positive_dt.size else 0.02
    consecutive = (sequence_delta == 1) & (
        np.abs(time_delta - dt) <= max(1.0e-6, 0.25 * dt)
    )
    first_rms, second_rms = _finite_difference_rms(q_des, consecutive)
    band_power, band_fraction = _band_power(q_des, 1.0 / dt)

    overflow_active = np.abs(overflow) > 1.0e-9
    overflow_longest = _longest_true_runs(overflow_active)
    torque_longest = _longest_true_runs(torque_clip)
    per_joint = []
    for index, joint in enumerate(joint_names):
        per_joint.append(
            {
                "joint": joint,
                "position_clamp_rate": float(np.mean(clamp[:, index])),
                "overflow_abs_p95": _percentile(
                    np.abs(overflow[:, index]), 0.95
                ),
                "overflow_abs_p99": _percentile(
                    np.abs(overflow[:, index]), 0.99
                ),
                "overflow_longest_run_ticks": int(
                    overflow_longest[index]
                ),
                "raw_abs_p99": _percentile(
                    np.abs(raw[:, index]), 0.99
                ),
                "effective_abs_p99": _percentile(
                    np.abs(effective[:, index]), 0.99
                ),
                "torque_clip_rate": float(
                    np.mean(torque_clip[:, index])
                ),
                "torque_clip_longest_run_ticks": int(
                    torque_longest[index]
                ),
                "q_des_delta_rms": float(first_rms[index]),
                "q_des_second_delta_rms": float(second_rms[index]),
                "q_des_5_20hz_power": float(band_power[index]),
                "q_des_5_20hz_power_fraction": float(
                    band_fraction[index]
                ),
            }
        )
    per_joint.sort(
        key=lambda item: (
            item["position_clamp_rate"],
            item["overflow_abs_p99"],
        ),
        reverse=True,
    )

    groups = {
        "waist": list(range(0, 3)),
        "left_arm": list(range(3, 10)),
        "head": list(range(10, 12)),
        "right_arm": list(range(12, 19)),
        "legs": list(range(19, 31)),
    }
    group_summary = {
        name: _group_summary(
            indices,
            clamp=clamp,
            overflow=overflow,
            torque_clip=torque_clip,
            first_rms=first_rms,
            second_rms=second_rms,
            band_power=band_power,
            band_fraction=band_fraction,
        )
        for name, indices in groups.items()
    }

    phases = np.asarray(
        [row.get("phase", "unknown") for row in rows], dtype=object
    )
    phase_summary = {}
    strike_torque_longest = 0
    for phase in sorted(set(phases.tolist())):
        mask = phases == phase
        phase_clamp = clamp[mask]
        phase_torque = torque_clip[mask]
        phase_overflow = overflow[mask]
        phase_summary[str(phase)] = {
            "frames": int(np.sum(mask)),
            "position_clamp_element_rate": float(
                np.mean(phase_clamp)
            ),
            "position_clamp_any_joint_frame_rate": float(
                np.mean(np.any(phase_clamp, axis=1))
            ),
            "overflow_abs_p99": _percentile(
                np.abs(phase_overflow), 0.99
            ),
            "torque_clip_element_rate": float(
                np.mean(phase_torque)
            ),
        }
        if "strike" in str(phase).lower() or "swing" in str(phase).lower():
            strike_torque_longest = max(
                strike_torque_longest,
                int(np.max(_longest_true_runs(phase_torque))),
            )

    fallen = np.asarray(
        [
            float(row.get("fallen", row.get("fallen_low", 0.0))) > 0.5
            or float(row.get("fallen_tilt", 0.0)) > 0.5
            for row in rows
        ],
        dtype=bool,
    )
    max_joint_clamp = float(np.max(np.mean(clamp, axis=0)))
    global_summary = {
        "rows": len(rows),
        "duration_s": float(time_s[-1] - time_s[0] + dt),
        "control_hz": float(1.0 / dt),
        "fell": bool(np.any(fallen)),
        "first_fall_row": (
            int(np.argmax(fallen)) if np.any(fallen) else None
        ),
        "position_clamp_element_rate": float(np.mean(clamp)),
        "position_clamp_any_joint_frame_rate": float(
            np.mean(np.any(clamp, axis=1))
        ),
        "max_single_joint_position_clamp_rate": max_joint_clamp,
        "persistent_overflow_joint_count": int(
            np.sum(overflow_longest >= persistent_ticks)
        ),
        "max_overflow_run_ticks": int(np.max(overflow_longest)),
        "overflow_abs_p95": _percentile(np.abs(overflow), 0.95),
        "overflow_abs_p99": _percentile(np.abs(overflow), 0.99),
        "torque_clip_element_rate": float(np.mean(torque_clip)),
        "torque_clip_any_joint_frame_rate": float(
            np.mean(np.any(torque_clip, axis=1))
        ),
        "max_torque_clip_run_ticks": int(np.max(torque_longest)),
        "strike_max_torque_clip_run_ticks": strike_torque_longest,
        "raw_effective_abs_gap_p99": _percentile(
            np.abs(raw - effective), 0.99
        ),
        "feedback_matches_raw_max_error": float(
            np.max(np.abs(feedback - raw))
        ),
        "feedback_matches_effective_max_error": float(
            np.max(np.abs(feedback - effective))
        ),
        "base_ang_vel_norm_p95": _percentile(
            np.asarray(
                [
                    float(row.get("base_ang_vel_norm", 0.0))
                    for row in rows
                ]
            ),
            0.95,
        ),
    }
    gate_checks = {
        "no_fall": not global_summary["fell"],
        "no_persistent_overflow": (
            global_summary["persistent_overflow_joint_count"] == 0
        ),
    }
    if scenario == "ready":
        gate_checks.update(
            {
                "position_clamp_any_frame_lt_1pct": (
                    global_summary[
                        "position_clamp_any_joint_frame_rate"
                    ]
                    < 0.01
                ),
                "torque_clip_elements_lt_1pct": (
                    global_summary["torque_clip_element_rate"] < 0.01
                ),
            }
        )
    else:
        gate_checks.update(
            {
                "position_clamp_elements_lt_0p5pct": (
                    global_summary["position_clamp_element_rate"] < 0.005
                ),
                "every_joint_clamp_lt_5pct": (
                    global_summary[
                        "max_single_joint_position_clamp_rate"
                    ]
                    < 0.05
                ),
                "strike_torque_clip_not_over_3_ticks": (
                    global_summary[
                        "strike_max_torque_clip_run_ticks"
                    ]
                    <= 3
                ),
            }
        )
    return {
        "source_csv": str(source_csv.resolve()),
        "scenario": scenario,
        "persistent_ticks": persistent_ticks,
        "global": global_summary,
        "groups": group_summary,
        "phases": phase_summary,
        "per_joint": per_joint,
        "gate": {
            "passed": bool(all(gate_checks.values())),
            "checks": gate_checks,
        },
    }


def _markdown(result: dict) -> str:
    global_summary = result["global"]
    lines = [
        f"# Action feasibility: {pathlib.Path(result['source_csv']).name}",
        "",
        f"Scenario: `{result['scenario']}`; gate: `{'PASS' if result['gate']['passed'] else 'FAIL'}`.",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for name, value in global_summary.items():
        lines.append(f"| {name} | {value} |")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| check | pass |",
            "|---|---:|",
        ]
    )
    for name, passed in result["gate"]["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(
        [
            "",
            "## Highest clamp joints",
            "",
            "| joint | clamp | overflow p99 | overflow run | torque clip | qdes d1 RMS | qdes 5-20Hz fraction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["per_joint"][:12]:
        lines.append(
            f"| {row['joint']} | {row['position_clamp_rate']:.4f} | "
            f"{row['overflow_abs_p99']:.4f} | "
            f"{row['overflow_longest_run_ticks']} | "
            f"{row['torque_clip_rate']:.4f} | "
            f"{row['q_des_delta_rms']:.5f} | "
            f"{row['q_des_5_20hz_power_fraction']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source = pathlib.Path(args.csv_path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        joint_names = [
            field.removeprefix("raw__")
            for field in (reader.fieldnames or ())
            if field.startswith("raw__")
        ]
        if not joint_names:
            raise ValueError(f"{source} has no raw__<joint> columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source} contains no data rows")
    result = analyze(
        rows,
        joint_names,
        source_csv=source,
        scenario=args.scenario,
        persistent_ticks=args.persistent_ticks,
    )
    text = _markdown(result)
    print(text, end="")
    if args.json_out:
        target = pathlib.Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_out:
        target = pathlib.Path(args.markdown_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
