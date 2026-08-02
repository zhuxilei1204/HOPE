#!/usr/bin/env python3
"""Summarize per-joint tracking and PD behavior from a policy I/O capture."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--max-lag-ticks",
        type=int,
        default=10,
        help="Largest q_des -> q lag considered, in 50 Hz control ticks.",
    )
    return parser.parse_args()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _p95_abs(values: np.ndarray) -> float:
    return float(np.percentile(np.abs(values), 95.0)) if values.size else float("nan")


def _best_tracking_lag(
    q_des: np.ndarray, q: np.ndarray, max_lag_ticks: int
) -> tuple[int, float]:
    best = (0, float("inf"))
    for lag in range(max_lag_ticks + 1):
        if lag == 0:
            desired, measured = q_des, q
        else:
            desired, measured = q_des[:-lag], q[lag:]
        if desired.size < 2:
            continue
        rmse = _rms(measured - desired)
        if rmse < best[1]:
            best = (lag, rmse)
    return best


def main() -> int:
    args = parse_args()
    metadata = json.loads(pathlib.Path(args.metadata_json).read_text(encoding="utf-8"))
    joint_names = list(metadata["policy_io_contract"]["joint_order"])
    control_hz = float(metadata["policy_io_contract"]["control_hz"])
    prefixes = (
        "command",
        "q",
        "qd",
        "q_des",
        "tracking_error",
        "torque_requested",
        "torque_applied",
        "torque_clipped",
        "position_clamped",
    )
    columns = [f"{prefix}__{name}" for prefix in prefixes for name in joint_names]
    samples: list[dict] = []
    with pathlib.Path(args.csv).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"segment", *columns}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"capture is missing columns: {missing[:8]}")
        for row in reader:
            samples.append(row)

    segment_names = ["all"] + sorted({row["segment"] for row in samples})
    output_rows = []
    for segment in segment_names:
        selected = samples if segment == "all" else [
            row for row in samples if row["segment"] == segment
        ]
        if not selected:
            continue
        for joint in joint_names:
            arrays = {
                prefix: np.asarray(
                    [float(row[f"{prefix}__{joint}"]) for row in selected],
                    dtype=np.float64,
                )
                for prefix in prefixes
            }
            lag_ticks, lag_rmse = _best_tracking_lag(
                arrays["q_des"], arrays["q"], args.max_lag_ticks
            )
            q_des_rate = (
                np.diff(arrays["q_des"]) * control_hz
                if arrays["q_des"].size > 1
                else np.asarray([], dtype=np.float64)
            )
            output_rows.append(
                {
                    "segment": segment,
                    "joint": joint,
                    "samples": len(selected),
                    "q_des_std_rad": float(np.std(arrays["q_des"])),
                    "q_des_rate_rms_rad_s": _rms(q_des_rate),
                    "tracking_rmse_rad": _rms(arrays["tracking_error"]),
                    "tracking_p95_abs_rad": _p95_abs(arrays["tracking_error"]),
                    "tracking_max_abs_rad": float(
                        np.max(np.abs(arrays["tracking_error"]))
                    ),
                    "best_tracking_lag_ticks": lag_ticks,
                    "best_tracking_lag_ms": 1000.0 * lag_ticks / control_hz,
                    "tracking_rmse_at_best_lag_rad": lag_rmse,
                    "joint_velocity_rms_rad_s": _rms(arrays["qd"]),
                    "joint_velocity_p95_abs_rad_s": _p95_abs(arrays["qd"]),
                    "torque_requested_rms": _rms(arrays["torque_requested"]),
                    "torque_requested_p95_abs": _p95_abs(
                        arrays["torque_requested"]
                    ),
                    "torque_requested_max_abs": float(
                        np.max(np.abs(arrays["torque_requested"]))
                    ),
                    "torque_applied_rms": _rms(arrays["torque_applied"]),
                    "torque_clip_fraction": float(
                        np.mean(arrays["torque_clipped"] > 0.5)
                    ),
                    "position_clip_fraction": float(
                        np.mean(arrays["position_clamped"] > 0.5)
                    ),
                    "command_action_rms": _rms(arrays["command"]),
                }
            )

    out_path = pathlib.Path(args.out_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
