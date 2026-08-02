#!/usr/bin/env python3
"""Derive provisional A3 operational joint bounds from frozen rollout evidence.

This is an offline audit. It does not modify the shared action adapter or enable
training. Missing zero-offset and braking measurements remain explicit blockers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, TextIO
import zipfile

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ADAPTER = (
    REPO_ROOT / "a3_deploy/a3_deploy_example/config/action_adapter.yaml"
)
DEFAULT_JOINT_ORDER = REPO_ROOT / "hope_training/config/joint_order_agibot_a3.yaml"
DEFAULT_P1_DIR = REPO_ROOT / "analysis/closed_loop_v2_20260731/p1_b19495_action_gate"
DEFAULT_SIM_SOURCES = (
    ("b19495_ready_raw", DEFAULT_P1_DIR / "ready_raw.csv"),
    ("b19495_fake_cycle_raw", DEFAULT_P1_DIR / "fake_raw.csv"),
    ("b19495_continuous_raw", DEFAULT_P1_DIR / "continuous_raw_joint.csv"),
)
DEFAULT_REAL_ZIP = Path("/mnt/ssd/zxl/1310数据包.zip")
DEFAULT_REAL_MEMBER = "20260730_131050_864023110/raw_mdu/policy_trace.csv"
DEFAULT_MOTION_DIR = Path(
    "/mnt/ssd/zxl/HOPE_deploy_models_20260730/"
    "03_hope_Bdeploy_model19495_rollback/motions"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _per_joint(spec: Any, names: tuple[str, ...], field: str) -> np.ndarray:
    if isinstance(spec, dict):
        missing = [name for name in names if name not in spec]
        if missing:
            raise ValueError(f"{field} is missing joints: {missing}")
        return np.asarray([float(spec[name]) for name in names], dtype=np.float64)
    values = np.asarray(spec, dtype=np.float64).reshape(-1)
    if values.shape != (len(names),):
        raise ValueError(f"{field} must contain {len(names)} values")
    return values


def load_adapter(
    adapter_path: Path,
    joint_order_path: Path,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    order_doc = yaml.safe_load(joint_order_path.read_text(encoding="utf-8"))
    adapter = yaml.safe_load(adapter_path.read_text(encoding="utf-8"))
    names = tuple(str(name) for name in order_doc["joint_order"])
    if len(names) != 31 or len(set(names)) != 31:
        raise ValueError("canonical joint order must contain 31 unique joints")
    clamp = adapter["joint_position_clamp"]
    values = {
        "default_q": _per_joint(adapter["default_q"], names, "default_q"),
        "action_scale": _per_joint(
            adapter["action_scale"], names, "action_scale"
        ),
        "hard_lower": _per_joint(clamp["lower"], names, "clamp.lower"),
        "hard_upper": _per_joint(clamp["upper"], names, "clamp.upper"),
    }
    return names, values


def _column_indices(
    header: list[str],
    fields: list[str],
    source: str,
) -> np.ndarray:
    index = {name: i for i, name in enumerate(header)}
    missing = [name for name in fields if name not in index]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing[:4]}")
    return np.asarray([index[name] for name in fields], dtype=np.int64)


def _read_trace(
    stream: TextIO,
    *,
    source: str,
    fields: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    reader = csv.reader(stream)
    header = next(reader)
    indices = {
        key: _column_indices(header, names, source) for key, names in fields.items()
    }
    rows: dict[str, list[np.ndarray]] = {key: [] for key in fields}
    for row in reader:
        if not row:
            continue
        try:
            for key, selected in indices.items():
                rows[key].append(
                    np.asarray([float(row[i]) for i in selected], dtype=np.float64)
                )
        except (IndexError, ValueError):
            continue
    result = {
        key: np.stack(values, axis=0)
        if values
        else np.empty((0, len(fields[key])), dtype=np.float64)
        for key, values in rows.items()
    }
    if not result["q"].size:
        raise ValueError(f"{source} contains no finite trace rows")
    finite = np.ones(result["q"].shape[0], dtype=bool)
    for values in result.values():
        finite &= np.all(np.isfinite(values), axis=1)
    return {key: values[finite] for key, values in result.items()}


def load_sim_trace(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    if f"q_{names[0]}" in header:
        fields = {
            "q": [f"q_{name}" for name in names],
            "dq": [f"qd_{name}" for name in names],
            "q_des": [f"qdes_{name}" for name in names],
            "clamp": [f"position_clamped__{name}" for name in names],
            "torque": [f"torque_applied__{name}" for name in names],
        }
    elif f"q__{names[0]}" in header:
        fields = {
            "q": [f"q__{name}" for name in names],
            "dq": [f"obs__joint_vel__{name}" for name in names],
            "q_des": [f"q_des__{name}" for name in names],
            "clamp": [f"position_clamped__{name}" for name in names],
            "torque": [f"torque_applied__{name}" for name in names],
        }
    else:
        raise ValueError(f"{path} uses an unsupported simulator trace schema")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return _read_trace(stream, source=str(path), fields=fields)


def load_real_trace(
    zip_path: Path,
    member: str,
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    count = len(names)
    fields = {
        "q": [f"q{i}" for i in range(count)],
        "dq": [f"dq{i}" for i in range(count)],
        "q_des": [f"q_des{i}" for i in range(count)],
        "torque": [f"tau{i}" for i in range(count)],
    }
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream:
                return _read_trace(
                    stream,
                    source=f"{zip_path}!{member}",
                    fields=fields,
                )


def infer_hard_clamp(
    q_des: np.ndarray,
    hard_lower: np.ndarray,
    hard_upper: np.ndarray,
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    return np.isclose(q_des, hard_lower[None, :], atol=tolerance, rtol=0.0) | np.isclose(
        q_des, hard_upper[None, :], atol=tolerance, rtol=0.0
    )


def _column_quantile(
    values: np.ndarray,
    quantile: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    output = np.full(values.shape[1], np.nan, dtype=np.float64)
    for index in range(values.shape[1]):
        selected = values[:, index]
        if mask is not None:
            selected = selected[mask[:, index]]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            output[index] = float(np.quantile(selected, quantile))
    return output


def summarize_trace(
    trace: dict[str, np.ndarray],
    hard_lower: np.ndarray,
    hard_upper: np.ndarray,
) -> dict[str, np.ndarray | int]:
    clamp = trace.get("clamp")
    if clamp is None:
        clamp = infer_hard_clamp(trace["q_des"], hard_lower, hard_upper)
    else:
        clamp = clamp > 0.5
    safe = ~clamp
    tracking = np.abs(trace["q_des"] - trace["q"])
    upper_overshoot = np.where(
        trace["dq"] > 0.0,
        np.maximum(trace["q"] - trace["q_des"], 0.0),
        0.0,
    )
    lower_overshoot = np.where(
        trace["dq"] < 0.0,
        np.maximum(trace["q_des"] - trace["q"], 0.0),
        0.0,
    )
    positive_velocity = np.maximum(trace["dq"], 0.0)
    negative_velocity = np.maximum(-trace["dq"], 0.0)
    return {
        "rows": int(trace["q"].shape[0]),
        "q_p001": _column_quantile(trace["q"], 0.001),
        "q_p999": _column_quantile(trace["q"], 0.999),
        "q_des_p001": _column_quantile(trace["q_des"], 0.001),
        "q_des_p999": _column_quantile(trace["q_des"], 0.999),
        "dq_abs_p99_safe": _column_quantile(np.abs(trace["dq"]), 0.99, safe),
        "tracking_abs_p99_safe": _column_quantile(tracking, 0.99, safe),
        "upper_overshoot_p99_safe": _column_quantile(
            upper_overshoot, 0.99, safe
        ),
        "lower_overshoot_p99_safe": _column_quantile(
            lower_overshoot, 0.99, safe
        ),
        "positive_velocity_p99_safe": _column_quantile(
            positive_velocity, 0.99, safe
        ),
        "negative_velocity_p99_safe": _column_quantile(
            negative_velocity, 0.99, safe
        ),
        "hard_clamp_rate": np.mean(clamp, axis=0),
        "torque_abs_p99": _column_quantile(np.abs(trace["torque"]), 0.99),
    }


def load_motion_envelope(
    motion_dir: Path,
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    lower = np.full(len(names), np.inf, dtype=np.float64)
    upper = np.full(len(names), -np.inf, dtype=np.float64)
    files: list[dict[str, Any]] = []
    for path in sorted(motion_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "joint_pos" not in data:
                continue
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != len(names):
            raise ValueError(f"{path} joint_pos has shape {joint_pos.shape}")
        lower = np.minimum(lower, np.min(joint_pos, axis=0))
        upper = np.maximum(upper, np.max(joint_pos, axis=0))
        files.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "frames": int(joint_pos.shape[0]),
            }
        )
    if not files:
        raise ValueError(f"no canonical 31-DoF motion npz files in {motion_dir}")
    return lower, upper, files


def derive_bounds(
    *,
    hard_lower: np.ndarray,
    hard_upper: np.ndarray,
    default_q: np.ndarray,
    motion_lower: np.ndarray,
    motion_upper: np.ndarray,
    summaries: dict[str, dict[str, np.ndarray | int]],
    fixed_margin_rad: float,
    assumed_delay_s: float,
) -> dict[str, np.ndarray]:
    tracking_abs = np.nanmax(
        np.stack(
            [
                np.asarray(summary["tracking_abs_p99_safe"])
                for summary in summaries.values()
            ]
        ),
        axis=0,
    )
    velocity_abs = np.nanmax(
        np.stack(
            [
                np.asarray(summary["dq_abs_p99_safe"])
                for summary in summaries.values()
            ]
        ),
        axis=0,
    )
    upper_overshoot = np.nanmax(
        np.stack(
            [
                np.asarray(summary["upper_overshoot_p99_safe"])
                for summary in summaries.values()
            ]
        ),
        axis=0,
    )
    lower_overshoot = np.nanmax(
        np.stack(
            [
                np.asarray(summary["lower_overshoot_p99_safe"])
                for summary in summaries.values()
            ]
        ),
        axis=0,
    )
    positive_velocity = np.nanmax(
        np.stack(
            [
                np.asarray(summary["positive_velocity_p99_safe"])
                for summary in summaries.values()
            ]
        ),
        axis=0,
    )
    negative_velocity = np.nanmax(
        np.stack(
            [
                np.asarray(summary["negative_velocity_p99_safe"])
                for summary in summaries.values()
            ]
        ),
        axis=0,
    )
    tracking_abs = np.nan_to_num(tracking_abs, nan=0.0)
    velocity_abs = np.nan_to_num(velocity_abs, nan=0.0)
    upper_overshoot = np.nan_to_num(upper_overshoot, nan=0.0)
    lower_overshoot = np.nan_to_num(lower_overshoot, nan=0.0)
    positive_velocity = np.nan_to_num(positive_velocity, nan=0.0)
    negative_velocity = np.nan_to_num(negative_velocity, nan=0.0)
    lower_margin = (
        fixed_margin_rad + lower_overshoot + negative_velocity * assumed_delay_s
    )
    upper_margin = (
        fixed_margin_rad + upper_overshoot + positive_velocity * assumed_delay_s
    )
    lower = hard_lower + lower_margin
    upper = hard_upper - upper_margin

    q_lows = [default_q, motion_lower]
    q_highs = [default_q, motion_upper]
    for summary in summaries.values():
        q_lows.append(np.asarray(summary["q_p001"]))
        q_highs.append(np.asarray(summary["q_p999"]))
    observed_lower = np.nanmin(np.stack(q_lows), axis=0)
    observed_upper = np.nanmax(np.stack(q_highs), axis=0)

    return {
        "safe_tracking_abs_p99": tracking_abs,
        "safe_velocity_abs_p99": velocity_abs,
        "lower_overshoot_p99": lower_overshoot,
        "upper_overshoot_p99": upper_overshoot,
        "negative_velocity_p99": negative_velocity,
        "positive_velocity_p99": positive_velocity,
        "provisional_lower_margin": lower_margin,
        "provisional_upper_margin": upper_margin,
        "provisional_margin": np.maximum(lower_margin, upper_margin),
        "provisional_lower": lower,
        "provisional_upper": upper,
        "observed_lower": observed_lower,
        "observed_upper": observed_upper,
        "range_valid": lower < upper,
        "default_inside": (default_q >= lower) & (default_q <= upper),
        "motion_inside": (motion_lower >= lower) & (motion_upper <= upper),
        "observed_inside": (observed_lower >= lower) & (observed_upper <= upper),
    }


def _float_map(names: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values)}


def _bool_map(names: tuple[str, ...], values: np.ndarray) -> dict[str, bool]:
    return {name: bool(value) for name, value in zip(names, values)}


def write_outputs(
    output_dir: Path,
    *,
    names: tuple[str, ...],
    adapter_path: Path,
    joint_order_path: Path,
    hard: dict[str, np.ndarray],
    motion_lower: np.ndarray,
    motion_upper: np.ndarray,
    motion_files: list[dict[str, Any]],
    summaries: dict[str, dict[str, np.ndarray | int]],
    derived: dict[str, np.ndarray],
    sources: dict[str, str],
    fixed_margin_rad: float,
    assumed_delay_s: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_names = list(summaries)
    ik_probe_lower = hard["hard_lower"] + float(fixed_margin_rad)
    ik_probe_upper = hard["hard_upper"] - float(fixed_margin_rad)
    default_inside_ik_probe = (
        (hard["default_q"] >= ik_probe_lower)
        & (hard["default_q"] <= ik_probe_upper)
    )
    motion_inside_ik_probe = (
        (motion_lower >= ik_probe_lower) & (motion_upper <= ik_probe_upper)
    )
    tracking_stack = np.stack(
        [
            np.asarray(summaries[source]["tracking_abs_p99_safe"])
            for source in source_names
        ]
    )
    velocity_stack = np.stack(
        [
            np.asarray(summaries[source]["dq_abs_p99_safe"])
            for source in source_names
        ]
    )
    tracking_rank = np.nanargmax(
        np.where(np.isfinite(tracking_stack), tracking_stack, -np.inf),
        axis=0,
    )
    velocity_rank = np.nanargmax(
        np.where(np.isfinite(velocity_stack), velocity_stack, -np.inf),
        axis=0,
    )
    lower_margin_source_stack = np.stack(
        [
            np.asarray(summaries[source]["lower_overshoot_p99_safe"])
            + np.asarray(summaries[source]["negative_velocity_p99_safe"])
            * assumed_delay_s
            for source in source_names
        ]
    )
    upper_margin_source_stack = np.stack(
        [
            np.asarray(summaries[source]["upper_overshoot_p99_safe"])
            + np.asarray(summaries[source]["positive_velocity_p99_safe"])
            * assumed_delay_s
            for source in source_names
        ]
    )
    lower_margin_rank = np.nanargmax(
        np.where(
            np.isfinite(lower_margin_source_stack),
            lower_margin_source_stack,
            -np.inf,
        ),
        axis=0,
    )
    upper_margin_rank = np.nanargmax(
        np.where(
            np.isfinite(upper_margin_source_stack),
            upper_margin_source_stack,
            -np.inf,
        ),
        axis=0,
    )
    candidate = {
        "schema_version": 1,
        "contract_id": "hope_operational_limits_provisional_n1",
        "status": "pending_calibration_do_not_train",
        "joint_order": list(names),
        "assumptions": {
            "fixed_safety_margin_rad": float(fixed_margin_rad),
            "total_command_delay_s": float(assumed_delay_s),
            "zero_offset_uncertainty_rad": None,
            "braking_distance_rad": None,
            "tracking_statistic": (
                "directional p99 actual-over-command overshoot on "
                "non-hard-clamped frames"
            ),
            "real_trace_policy": "xh1_r2c_model23993_balanced_not_b19495",
        },
        "mechanical_lower": _float_map(names, hard["hard_lower"]),
        "mechanical_upper": _float_map(names, hard["hard_upper"]),
        "provisional_lower_margin": _float_map(
            names, derived["provisional_lower_margin"]
        ),
        "provisional_upper_margin": _float_map(
            names, derived["provisional_upper_margin"]
        ),
        "provisional_lower": _float_map(names, derived["provisional_lower"]),
        "provisional_upper": _float_map(names, derived["provisional_upper"]),
        "ik_probe_lower": _float_map(names, ik_probe_lower),
        "ik_probe_upper": _float_map(names, ik_probe_upper),
        "observed_capability_lower": _float_map(names, derived["observed_lower"]),
        "observed_capability_upper": _float_map(names, derived["observed_upper"]),
        "checks": {
            "range_valid": _bool_map(names, derived["range_valid"]),
            "default_inside": _bool_map(names, derived["default_inside"]),
            "motion_inside": _bool_map(names, derived["motion_inside"]),
            "observed_inside": _bool_map(names, derived["observed_inside"]),
            "default_inside_ik_probe": _bool_map(
                names, default_inside_ik_probe
            ),
            "motion_inside_ik_probe": _bool_map(names, motion_inside_ik_probe),
        },
        "blockers": [
            "encoder zero-offset uncertainty is missing",
            "joint braking-distance measurement is missing",
            "real trace is from model23993, not B19495",
            "planner IK and table-collision audit against these bounds is pending",
        ],
        "ik_probe_note": (
            "The ik_probe bounds only remove the fixed two-degree mechanical-edge "
            "band. They intentionally ignore uncalibrated tracking/braking margins "
            "and may be used for read-only reachability audit only."
        ),
    }
    candidate_path = output_dir / "operational_limits_provisional.yaml"
    candidate_path.write_text(
        yaml.safe_dump(candidate, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        source_clamp = {
            source: float(np.asarray(summary["hard_clamp_rate"])[index])
            for source, summary in summaries.items()
        }
        rows.append(
            {
                "joint": name,
                "mechanical_lower": float(hard["hard_lower"][index]),
                "mechanical_upper": float(hard["hard_upper"][index]),
                "default_q": float(hard["default_q"][index]),
                "motion_lower": float(motion_lower[index]),
                "motion_upper": float(motion_upper[index]),
                "tracking_abs_p99_safe": float(
                    derived["safe_tracking_abs_p99"][index]
                ),
                "tracking_reference_source": source_names[
                    int(tracking_rank[index])
                ],
                "velocity_abs_p99_safe": float(
                    derived["safe_velocity_abs_p99"][index]
                ),
                "velocity_reference_source": source_names[
                    int(velocity_rank[index])
                ],
                "lower_overshoot_p99": float(
                    derived["lower_overshoot_p99"][index]
                ),
                "upper_overshoot_p99": float(
                    derived["upper_overshoot_p99"][index]
                ),
                "provisional_lower_margin": float(
                    derived["provisional_lower_margin"][index]
                ),
                "provisional_lower_margin_source": source_names[
                    int(lower_margin_rank[index])
                ],
                "provisional_upper_margin": float(
                    derived["provisional_upper_margin"][index]
                ),
                "provisional_upper_margin_source": source_names[
                    int(upper_margin_rank[index])
                ],
                "provisional_margin": float(derived["provisional_margin"][index]),
                "provisional_lower": float(derived["provisional_lower"][index]),
                "provisional_upper": float(derived["provisional_upper"][index]),
                "observed_lower": float(derived["observed_lower"][index]),
                "observed_upper": float(derived["observed_upper"][index]),
                "range_valid": bool(derived["range_valid"][index]),
                "default_inside": bool(derived["default_inside"][index]),
                "motion_inside": bool(derived["motion_inside"][index]),
                "observed_inside": bool(derived["observed_inside"][index]),
                "default_inside_ik_probe": bool(
                    default_inside_ik_probe[index]
                ),
                "motion_inside_ik_probe": bool(
                    motion_inside_ik_probe[index]
                ),
                "max_source_clamp_rate": max(source_clamp.values()),
                "max_source_clamp_name": max(source_clamp, key=source_clamp.get),
            }
        )
    with (output_dir / "per_joint_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_rows: list[dict[str, Any]] = []
    for source, summary in summaries.items():
        for index, name in enumerate(names):
            source_rows.append(
                {
                    "source": source,
                    "joint": name,
                    "rows": int(summary["rows"]),
                    "q_p001": float(np.asarray(summary["q_p001"])[index]),
                    "q_p999": float(np.asarray(summary["q_p999"])[index]),
                    "q_des_p001": float(
                        np.asarray(summary["q_des_p001"])[index]
                    ),
                    "q_des_p999": float(
                        np.asarray(summary["q_des_p999"])[index]
                    ),
                    "tracking_abs_p99_safe": float(
                        np.asarray(summary["tracking_abs_p99_safe"])[index]
                    ),
                    "lower_overshoot_p99_safe": float(
                        np.asarray(summary["lower_overshoot_p99_safe"])[index]
                    ),
                    "upper_overshoot_p99_safe": float(
                        np.asarray(summary["upper_overshoot_p99_safe"])[index]
                    ),
                    "dq_abs_p99_safe": float(
                        np.asarray(summary["dq_abs_p99_safe"])[index]
                    ),
                    "negative_velocity_p99_safe": float(
                        np.asarray(summary["negative_velocity_p99_safe"])[index]
                    ),
                    "positive_velocity_p99_safe": float(
                        np.asarray(summary["positive_velocity_p99_safe"])[index]
                    ),
                    "hard_clamp_rate": float(
                        np.asarray(summary["hard_clamp_rate"])[index]
                    ),
                    "torque_abs_p99": float(
                        np.asarray(summary["torque_abs_p99"])[index]
                    ),
                }
            )
    with (output_dir / "per_source_joint_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    risky = sorted(
        rows,
        key=lambda row: (
            not row["range_valid"],
            not row["observed_inside"],
            not row["motion_inside"],
            row["max_source_clamp_rate"],
            row["provisional_margin"],
        ),
        reverse=True,
    )
    summary_json = {
        "status": "ANALYSIS_ONLY_NOT_CALIBRATED",
        "train_ready": False,
        "joint_count": len(names),
        "invalid_range_joint_count": int(np.sum(~derived["range_valid"])),
        "default_conflict_joint_count": int(np.sum(~derived["default_inside"])),
        "motion_conflict_joint_count": int(np.sum(~derived["motion_inside"])),
        "observed_conflict_joint_count": int(np.sum(~derived["observed_inside"])),
        "ik_probe_default_conflict_joint_count": int(
            np.sum(~default_inside_ik_probe)
        ),
        "ik_probe_motion_conflict_joint_count": int(
            np.sum(~motion_inside_ik_probe)
        ),
        "top_risk_joints": risky[:12],
        "sources": sources,
        "motion_files": motion_files,
        "assumptions": candidate["assumptions"],
        "blockers": candidate["blockers"],
        "hashes": {
            "adapter_sha256": _sha256(adapter_path),
            "joint_order_sha256": _sha256(joint_order_path),
            "candidate_sha256": _sha256(candidate_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# N1 provisional operational-limit audit",
        "",
        "Status: `ANALYSIS ONLY / NOT CALIBRATED / DO NOT TRAIN`",
        "",
        "The real trace is from `xh1_r2c_model23993_balanced`, not B19495. "
        "It contributes end-to-end tracking evidence only.",
        "",
        "## Assumptions",
        "",
        f"- fixed safety margin: `{fixed_margin_rad:.6f} rad`",
        f"- assumed command delay: `{assumed_delay_s:.3f} s`",
        "- zero-offset uncertainty: missing",
        "- braking distance: missing",
        "",
        "## Conflict counts",
        "",
        f"- invalid provisional range: `{summary_json['invalid_range_joint_count']}`",
        f"- default-ready conflicts: `{summary_json['default_conflict_joint_count']}`",
        f"- four-motion conflicts: `{summary_json['motion_conflict_joint_count']}`",
        f"- observed-envelope conflicts: `{summary_json['observed_conflict_joint_count']}`",
        "",
        "## Highest-risk joints",
        "",
        "| joint | max clamp | source | margin rad | motion in | observed in |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in risky[:12]:
        lines.append(
            f"| {row['joint']} | {row['max_source_clamp_rate']:.3f} | "
            f"{row['max_source_clamp_name']} | {row['provisional_margin']:.3f} | "
            f"{int(row['motion_inside'])} | {int(row['observed_inside'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These bounds are a diagnostic safety-inner envelope, not a deploy limit. "
            "A conflict means the observed behavior or motion requires more range than "
            "the provisional dynamic margin permits. It must be resolved through "
            "measurement, IK/station/workspace changes, or action-contract migration; "
            "the mechanical limit must not be silently relaxed.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("source label cannot be empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--joint-order", type=Path, default=DEFAULT_JOINT_ORDER)
    parser.add_argument(
        "--sim-csv",
        action="append",
        type=parse_source,
        help="Repeatable LABEL=PATH. Defaults to the three B19495 raw P1 traces.",
    )
    parser.add_argument("--real-zip", type=Path, default=DEFAULT_REAL_ZIP)
    parser.add_argument("--real-member", default=DEFAULT_REAL_MEMBER)
    parser.add_argument("--motion-dir", type=Path, default=DEFAULT_MOTION_DIR)
    parser.add_argument("--fixed-margin-rad", type=float, default=np.deg2rad(2.0))
    parser.add_argument("--assumed-delay-s", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names, hard = load_adapter(args.adapter, args.joint_order)
    sim_sources = args.sim_csv or list(DEFAULT_SIM_SOURCES)
    summaries: dict[str, dict[str, np.ndarray | int]] = {}
    sources: dict[str, str] = {}
    for label, path in sim_sources:
        print(f"[N1] reading {label}: {path}", flush=True)
        summaries[label] = summarize_trace(
            load_sim_trace(path, names),
            hard["hard_lower"],
            hard["hard_upper"],
        )
        sources[label] = str(path)

    print(f"[N1] reading real trace: {args.real_zip}!{args.real_member}", flush=True)
    real_trace = load_real_trace(args.real_zip, args.real_member, names)
    summaries["real_model23993"] = summarize_trace(
        real_trace,
        hard["hard_lower"],
        hard["hard_upper"],
    )
    sources["real_model23993"] = f"{args.real_zip}!{args.real_member}"

    motion_lower, motion_upper, motion_files = load_motion_envelope(
        args.motion_dir, names
    )
    derived = derive_bounds(
        hard_lower=hard["hard_lower"],
        hard_upper=hard["hard_upper"],
        default_q=hard["default_q"],
        motion_lower=motion_lower,
        motion_upper=motion_upper,
        summaries=summaries,
        fixed_margin_rad=args.fixed_margin_rad,
        assumed_delay_s=args.assumed_delay_s,
    )
    write_outputs(
        args.output_dir,
        names=names,
        adapter_path=args.adapter,
        joint_order_path=args.joint_order,
        hard=hard,
        motion_lower=motion_lower,
        motion_upper=motion_upper,
        motion_files=motion_files,
        summaries=summaries,
        derived=derived,
        sources=sources,
        fixed_margin_rad=args.fixed_margin_rad,
        assumed_delay_s=args.assumed_delay_s,
    )
    print(
        json.dumps(
            {
                "status": "ANALYSIS_ONLY_NOT_CALIBRATED",
                "output_dir": str(args.output_dir),
                "invalid_range_joint_count": int(np.sum(~derived["range_valid"])),
                "motion_conflict_joint_count": int(np.sum(~derived["motion_inside"])),
                "observed_conflict_joint_count": int(
                    np.sum(~derived["observed_inside"])
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
