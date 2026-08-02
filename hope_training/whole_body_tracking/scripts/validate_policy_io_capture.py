#!/usr/bin/env python3
"""Validate and compact a HOPE policy I/O + PD diagnostic capture.

The input CSV is produced by ``mujoco_eval_onnx.py --joint-action-diag-csv``.
Validation checks the four boundaries needed for sim-to-real diagnosis:

1. saved observation -> ONNX raw action;
2. command action -> unclamped/clamped q_des;
3. q/q_des/qd/Kp/Kd -> requested torque;
4. requested torque -> actuator-limited torque.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Policy I/O diagnostic CSV.")
    parser.add_argument("--metadata-json", required=True, help="Detailed MuJoCo eval JSON.")
    parser.add_argument("--onnx", default=None, help="Override ONNX path from metadata.")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--npz-out", default=None, help="Optional compressed numeric dataset.")
    return parser.parse_args()


def _load_numeric_columns(csv_path: pathlib.Path, columns: list[str]) -> dict[str, np.ndarray]:
    values = {name: [] for name in columns}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = sorted(set(columns) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"capture is missing required columns: {missing[:8]}")
        for row in reader:
            for name in columns:
                value = row[name]
                values[name].append(np.nan if value == "" else float(value))
    return {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}


def _matrix(columns: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.column_stack([columns[name] for name in names])


def _error(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def main() -> int:
    args = parse_args()
    csv_path = pathlib.Path(args.csv).expanduser().resolve()
    metadata_path = pathlib.Path(args.metadata_json).expanduser().resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = metadata["policy_io_contract"]
    joint_names = list(contract["joint_order"])
    observation_fields = list(contract["observation_fields"])

    obs_cols = [f"obs__{name}" for name in observation_fields]
    raw_cols = [f"raw_policy__{name}" for name in joint_names]
    command_cols = [f"command__{name}" for name in joint_names]
    q_cols = [f"q__{name}" for name in joint_names]
    qd_cols = [f"qd__{name}" for name in joint_names]
    q_des_cols = [f"q_des__{name}" for name in joint_names]
    q_des_unclipped_cols = [f"q_des_unclipped__{name}" for name in joint_names]
    torque_p_cols = [f"torque_p__{name}" for name in joint_names]
    torque_d_cols = [f"torque_d__{name}" for name in joint_names]
    torque_requested_cols = [f"torque_requested__{name}" for name in joint_names]
    torque_applied_cols = [f"torque_applied__{name}" for name in joint_names]
    torque_lo_cols = [f"torque_lower_limit__{name}" for name in joint_names]
    torque_hi_cols = [f"torque_upper_limit__{name}" for name in joint_names]
    scalar_cols = [
        "sequence_step",
        "sim_time_s",
        "trial",
        "tick",
        "command_valid",
        "fallen",
    ]
    required = (
        scalar_cols
        + obs_cols
        + raw_cols
        + command_cols
        + q_cols
        + qd_cols
        + q_des_cols
        + q_des_unclipped_cols
        + torque_p_cols
        + torque_d_cols
        + torque_requested_cols
        + torque_applied_cols
        + torque_lo_cols
        + torque_hi_cols
    )
    cols = _load_numeric_columns(csv_path, required)

    observation = _matrix(cols, obs_cols).astype(np.float32)
    raw_action = _matrix(cols, raw_cols)
    command_action = _matrix(cols, command_cols)
    q = _matrix(cols, q_cols)
    qd = _matrix(cols, qd_cols)
    q_des = _matrix(cols, q_des_cols)
    q_des_unclipped = _matrix(cols, q_des_unclipped_cols)
    torque_p = _matrix(cols, torque_p_cols)
    torque_d = _matrix(cols, torque_d_cols)
    torque_requested = _matrix(cols, torque_requested_cols)
    torque_applied = _matrix(cols, torque_applied_cols)
    torque_lo = _matrix(cols, torque_lo_cols)
    torque_hi = _matrix(cols, torque_hi_cols)

    onnx_path = pathlib.Path(args.onnx or contract["onnx"]).expanduser().resolve()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    replay_action = session.run(None, {input_name: observation})[0]

    default_q = np.asarray(contract["default_q"], dtype=np.float64)
    action_scale = np.asarray(contract["action_scale"], dtype=np.float64)
    clamp_lo = np.asarray(contract["joint_position_clamp_lower"], dtype=np.float64)
    clamp_hi = np.asarray(contract["joint_position_clamp_upper"], dtype=np.float64)
    kp = np.asarray(contract["kp"], dtype=np.float64)
    kd = np.asarray(contract["kd"], dtype=np.float64)

    expected_unclipped = default_q + command_action * action_scale
    # Passive neck targets are explicitly held at default after adapter decoding.
    expected_unclipped[:, 3:5] = default_q[3:5]
    expected_q_des = np.clip(expected_unclipped, clamp_lo, clamp_hi)
    expected_q_des[:, 3:5] = default_q[3:5]
    expected_torque_p = kp * (q_des - q)
    expected_torque_d = -kd * qd
    expected_torque_requested = expected_torque_p + expected_torque_d
    expected_torque_applied = np.clip(expected_torque_requested, torque_lo, torque_hi)

    time_delta = np.diff(cols["sim_time_s"])
    sequence_delta = np.diff(cols["sequence_step"])
    report = {
        "capture_csv": str(csv_path),
        "metadata_json": str(metadata_path),
        "onnx": str(onnx_path),
        "samples": int(observation.shape[0]),
        "observation_dim": int(observation.shape[1]),
        "action_dim": int(raw_action.shape[1]),
        "control_dt_s": {
            "median": float(np.median(time_delta)) if time_delta.size else None,
            "min": float(np.min(time_delta)) if time_delta.size else None,
            "max": float(np.max(time_delta)) if time_delta.size else None,
        },
        "sequence_contiguous": bool(
            sequence_delta.size == 0 or np.all(sequence_delta == 1.0)
        ),
        "onnx_replay_raw_action_error": _error(raw_action, replay_action),
        "adapter_unclipped_q_des_error": _error(q_des_unclipped, expected_unclipped),
        "adapter_clamped_q_des_error": _error(q_des, expected_q_des),
        "pd_p_component_error": _error(torque_p, expected_torque_p),
        "pd_d_component_error": _error(torque_d, expected_torque_d),
        "pd_requested_torque_error": _error(
            torque_requested, expected_torque_requested
        ),
        "actuator_clipped_torque_error": _error(
            torque_applied, expected_torque_applied
        ),
        "command_valid_fraction": float(np.mean(cols["command_valid"])),
        "fallen_sample_fraction": float(np.mean(cols["fallen"])),
    }

    if args.npz_out:
        npz_path = pathlib.Path(args.npz_out).expanduser().resolve()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            sequence_step=cols["sequence_step"].astype(np.int64),
            sim_time_s=cols["sim_time_s"],
            trial=cols["trial"].astype(np.int64),
            tick=cols["tick"].astype(np.int64),
            command_valid=cols["command_valid"].astype(np.int8),
            fallen=cols["fallen"].astype(np.int8),
            observation=observation,
            raw_action=raw_action.astype(np.float32),
            command_action=command_action.astype(np.float32),
            q=q.astype(np.float32),
            qd=qd.astype(np.float32),
            q_des=q_des.astype(np.float32),
            torque_p=torque_p.astype(np.float32),
            torque_d=torque_d.astype(np.float32),
            torque_requested=torque_requested.astype(np.float32),
            torque_applied=torque_applied.astype(np.float32),
            observation_fields=np.asarray(observation_fields),
            joint_names=np.asarray(joint_names),
        )
        report["npz_out"] = str(npz_path)

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.report_json:
        report_path = pathlib.Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
