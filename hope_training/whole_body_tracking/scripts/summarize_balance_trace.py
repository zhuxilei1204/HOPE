"""Summarize event-aligned whole-body balance diagnostics from MuJoCo eval traces."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np


EVENT_METRICS = (
    "base_pitch_deg",
    "torso_pitch_deg",
    "base_lin_vel_x",
    "base_ang_vel_y",
    "com_support_margin",
    "left_foot_speed_xy",
    "right_foot_speed_xy",
    "waist_action_delta_rms",
    "right_arm_action_delta_rms",
    "leg_action_delta_rms",
    "waist_joint_vel_rms",
    "right_arm_joint_vel_rms",
    "leg_joint_vel_rms",
)


def _float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def _bool(row: dict, key: str) -> bool:
    value = _float(row, key)
    return bool(np.isfinite(value) and value > 0.5)


def _stats(values) -> dict:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def _trial_anchor(rows: list[dict]) -> int:
    for idx, row in enumerate(rows):
        if _bool(row, "contacted") and (idx == 0 or not _bool(rows[idx - 1], "contacted")):
            return idx
    finite_tts = [(_float(row, "time_to_strike"), idx) for idx, row in enumerate(rows)]
    finite_tts = [(abs(value), idx) for value, idx in finite_tts if np.isfinite(value)]
    return min(finite_tts)[1] if finite_tts else int(np.argmin([_float(r, "ball_racket_distance") for r in rows]))


def _trial_summary(rows: list[dict], pre_steps: int, post_steps: int) -> tuple[dict, list[tuple[int, dict]]]:
    anchor = _trial_anchor(rows)
    start = max(0, anchor - pre_steps)
    end = min(len(rows), anchor + post_steps + 1)
    window = rows[start:end]

    def values(key: str) -> np.ndarray:
        return np.asarray([_float(row, key) for row in window], dtype=np.float64)

    pitch = values("base_pitch_deg")
    torso_pitch = values("torso_pitch_deg")
    ang_x = values("base_ang_vel_x")
    ang_y = values("base_ang_vel_y")
    support = values("com_support_margin")
    backward = np.maximum(-values("base_lin_vel_x"), 0.0)
    finite_support = support[np.isfinite(support)]
    first = rows[0]
    last = rows[-1]
    summary = {
        "trial": int(float(first["trial"])),
        "side": first.get("side", ""),
        "anchor_tick": int(float(rows[anchor]["tick"])),
        "anchor_kind": "contact" if _bool(rows[anchor], "contacted") else "strike_clock",
        "contact": any(_bool(row, "contacted") for row in rows),
        "net_clear": any(_bool(row, "net_clear") for row in rows),
        "fallen": any(_bool(row, "fallen") for row in rows),
        "peak_abs_base_pitch_deg": float(np.nanmax(np.abs(pitch))),
        "peak_abs_torso_pitch_deg": float(np.nanmax(np.abs(torso_pitch))),
        "peak_base_ang_vel_xy": float(np.nanmax(np.sqrt(np.square(ang_x) + np.square(ang_y)))),
        "peak_backward_velocity": float(np.nanmax(backward)),
        "min_com_support_margin": (
            float(np.min(finite_support)) if finite_support.size else float("nan")
        ),
        "negative_support_fraction": (
            float(np.mean(finite_support < 0.0)) if finite_support.size else float("nan")
        ),
        "mean_leg_action_delta_rms": float(np.nanmean(values("leg_action_delta_rms"))),
        "peak_leg_action_delta_rms": float(np.nanmax(values("leg_action_delta_rms"))),
        "mean_leg_joint_vel_rms": float(np.nanmean(values("leg_joint_vel_rms"))),
        "peak_right_arm_joint_vel_rms": float(np.nanmax(values("right_arm_joint_vel_rms"))),
        "target_shoulder_distance_at_anchor": _float(rows[anchor], "target_shoulder_distance"),
        "target_rel_base_z_at_anchor": _float(rows[anchor], "target_rel_base_z"),
        "final_base_z": _float(last, "base_z"),
    }
    aligned = [(idx - anchor, row) for idx, row in enumerate(rows[start:end], start=start)]
    return summary, aligned


def _aggregate(trials: list[dict]) -> dict:
    keys = [
        "peak_abs_base_pitch_deg",
        "peak_abs_torso_pitch_deg",
        "peak_base_ang_vel_xy",
        "peak_backward_velocity",
        "min_com_support_margin",
        "negative_support_fraction",
        "mean_leg_action_delta_rms",
        "peak_leg_action_delta_rms",
        "mean_leg_joint_vel_rms",
        "peak_right_arm_joint_vel_rms",
        "target_shoulder_distance_at_anchor",
        "target_rel_base_z_at_anchor",
    ]

    def bucket(rows):
        return {key: _stats(row[key] for row in rows) for key in keys}

    return {
        "all": bucket(trials),
        "contact": bucket([row for row in trials if row["contact"]]),
        "no_contact": bucket([row for row in trials if not row["contact"]]),
        "fallen": bucket([row for row in trials if row["fallen"]]),
        "not_fallen": bucket([row for row in trials if not row["fallen"]]),
    }


def _write_event_csv(path: pathlib.Path, aligned_rows: list[list[tuple[int, dict]]], dt: float) -> None:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for trial_rows in aligned_rows:
        for relative_tick, row in trial_rows:
            buckets[relative_tick].append(row)
    fields = ["relative_tick", "relative_time_s", "samples"] + [f"mean_{name}" for name in EVENT_METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for relative_tick in sorted(buckets):
            rows = buckets[relative_tick]
            out = {
                "relative_tick": relative_tick,
                "relative_time_s": relative_tick * dt,
                "samples": len(rows),
            }
            for metric in EVENT_METRICS:
                vals = np.asarray([_float(row, metric) for row in rows], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                out[f"mean_{metric}"] = "" if vals.size == 0 else float(np.mean(vals))
            writer.writerow(out)


def _write_markdown(path: pathlib.Path, result: dict) -> None:
    metrics = [
        ("Base pitch p95 (deg)", "peak_abs_base_pitch_deg"),
        ("Torso pitch p95 (deg)", "peak_abs_torso_pitch_deg"),
        ("Base ang vel p95", "peak_base_ang_vel_xy"),
        ("Backward vel p95", "peak_backward_velocity"),
        ("Min COM margin median (m)", "min_com_support_margin"),
        ("Negative support mean", "negative_support_fraction"),
        ("Leg action delta p95", "peak_leg_action_delta_rms"),
        ("Leg joint vel p95", "mean_leg_joint_vel_rms"),
    ]
    lines = [
        "# Balance Trace Summary",
        "",
        f"- Trials: `{result['trial_count']}`",
        f"- Contact rate: `{result['contact_rate']:.3f}`",
        f"- Fall rate: `{result['fall_rate']:.3f}`",
        "",
        "| metric | all | fallen | not fallen |",
        "|---|---:|---:|---:|",
    ]
    for label, key in metrics:
        column = "median" if "median" in label.lower() else ("mean" if "mean" in label.lower() else "p95")
        values = []
        for bucket in ("all", "fallen", "not_fallen"):
            value = result["aggregate"][bucket][key][column]
            values.append("n/a" if value is None else f"{value:.4f}")
        lines.append(f"| {label} | {' | '.join(values)} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_csv")
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--pre-seconds", type=float, default=0.5)
    parser.add_argument("--post-seconds", type=float, default=1.0)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    parser.add_argument("--event-csv", required=True)
    args = parser.parse_args()

    grouped: dict[int, list[dict]] = defaultdict(list)
    with open(args.trace_csv, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            grouped[int(float(row["trial"]))].append(row)
    pre_steps = int(round(args.pre_seconds / args.control_dt))
    post_steps = int(round(args.post_seconds / args.control_dt))
    summaries = []
    aligned_rows = []
    for trial in sorted(grouped):
        summary, aligned = _trial_summary(grouped[trial], pre_steps, post_steps)
        summaries.append(summary)
        aligned_rows.append(aligned)

    result = {
        "trace_csv": str(pathlib.Path(args.trace_csv).resolve()),
        "trial_count": len(summaries),
        "contact_rate": float(np.mean([row["contact"] for row in summaries])) if summaries else 0.0,
        "fall_rate": float(np.mean([row["fallen"] for row in summaries])) if summaries else 0.0,
        "aggregate": _aggregate(summaries),
        "trials": summaries,
    }
    json_path = pathlib.Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(pathlib.Path(args.markdown_out), result)
    _write_event_csv(pathlib.Path(args.event_csv), aligned_rows, args.control_dt)
    print(json.dumps({"trial_count": len(summaries), "contact_rate": result["contact_rate"], "fall_rate": result["fall_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
