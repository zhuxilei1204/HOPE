#!/usr/bin/env python3
"""Audit full A3 strike feasibility with a locked waist and lateral station sweep.

The older reachability audit proves that a planner target can be reached when the
waist and right arm are optimized together.  This diagnostic separates three
questions:

1. Can the right arm place and orient the racket while the waist stays at ready?
2. Can the wrist alone finish the normal correction after shoulder/elbow position IK?
3. Does a small lateral station shift improve pose, joint-margin, and velocity feasibility?

The script is read-only.  It uses recorded planner commands and the A3 MuJoCo
kinematics; it does not load or alter a policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
import yaml
from scipy.optimize import least_squares, lsq_linear

import audit_planner_reachability as base


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLANNER_CSV = (
    ROOT
    / "analysis/tablewidth_cycle_v3_20260728/long_planner_alignment/model_32692/contact_diag.csv"
)

WAIST = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
SHOULDER_ELBOW = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)
WRIST = (
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
RIGHT_ARM = SHOULDER_ELBOW + WRIST
FULL_CHAIN = WAIST + RIGHT_ARM

VELOCITY_LIMIT = {
    "waist_yaw_joint": 12.0,
    "waist_roll_joint": 22.7,
    "waist_pitch_joint": 9.2,
    "right_shoulder_pitch_joint": 13.6,
    "right_shoulder_roll_joint": 13.6,
    "right_shoulder_yaw_joint": 15.7,
    "right_elbow_joint": 15.7,
    "right_wrist_roll_joint": 15.7,
    "right_wrist_pitch_joint": 12.7,
    "right_wrist_yaw_joint": 12.7,
}


@dataclass
class PlannerSample:
    trial: int
    side: float
    clip: base.Clip
    position: np.ndarray
    normal: np.ndarray
    velocity: np.ndarray
    station: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", type=Path, default=base.DEFAULT_XML)
    parser.add_argument("--manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=base.DEFAULT_ADAPTER)
    parser.add_argument("--joint-order", type=Path, default=base.DEFAULT_JOINT_ORDER)
    parser.add_argument("--planner-csv", type=Path, default=DEFAULT_PLANNER_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--joint-limits-yaml",
        type=Path,
        default=None,
        help="Optional read-only joint-limit override produced by the N1 audit.",
    )
    parser.add_argument(
        "--joint-limits-prefix",
        default="ik_probe",
        help="YAML field prefix; reads <prefix>_lower and <prefix>_upper.",
    )
    parser.add_argument("--station-y-min", type=float, default=-0.30)
    parser.add_argument("--station-y-max", type=float, default=0.30)
    parser.add_argument("--station-y-step", type=float, default=0.03)
    parser.add_argument("--station-clip-x", nargs=2, type=float, default=(-0.25, 0.25))
    parser.add_argument("--station-clip-y", nargs=2, type=float, default=(-0.90, 0.90))
    parser.add_argument("--position-threshold", type=float, default=0.035)
    parser.add_argument("--normal-threshold-deg", type=float, default=20.0)
    parser.add_argument("--velocity-error-threshold", type=float, default=0.25)
    parser.add_argument("--normal-rate-threshold", type=float, default=1.0)
    return parser.parse_args()


def finite_vector(row: dict[str, str], prefix: str) -> np.ndarray | None:
    vector = np.array([base.f(row.get(f"{prefix}_{axis}")) for axis in "xyz"], dtype=np.float64)
    return vector if np.all(np.isfinite(vector)) else None


def load_samples(
    path: Path,
    clips: list[base.Clip],
    kin: base.A3Kinematics,
    station_clip_x: tuple[float, float],
    station_clip_y: tuple[float, float],
) -> list[PlannerSample]:
    samples: list[PlannerSample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            position = finite_vector(row, "target_pos")
            normal = finite_vector(row, "target_normal")
            velocity = finite_vector(row, "target_vel")
            if position is None or normal is None or velocity is None:
                continue
            side = 1.0 if row.get("side") == "forehand" else -1.0
            clip = base.nearest_clip(position, side, clips)
            station = base.dynamic_station(
                position,
                clip.racket_offset_xy,
                kin.default_root[:2].copy(),
                station_clip_x,
                station_clip_y,
            )
            samples.append(
                PlannerSample(
                    trial=int(row.get("trial") or -1),
                    side=side,
                    clip=clip,
                    position=position,
                    normal=base.unit(normal),
                    velocity=velocity,
                    station=station,
                )
            )
    if not samples:
        raise ValueError(f"No finite planner position/normal/velocity commands in {path}")
    return samples


def joint_bounds(
    kin: base.A3Kinematics, names: Iterable[str]
) -> tuple[np.ndarray, np.ndarray]:
    lower: list[float] = []
    upper: list[float] = []
    override = getattr(kin, "joint_bound_override", {})
    for name in names:
        joint_id = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = kin.model.jnt_range[joint_id]
        if name in override:
            lo = max(float(lo), float(override[name][0]))
            hi = min(float(hi), float(override[name][1]))
            if lo >= hi:
                raise ValueError(
                    f"joint-limit override is empty for {name}: [{lo}, {hi}]"
                )
        lower.append(float(lo))
        upper.append(float(hi))
    return np.asarray(lower), np.asarray(upper)


def solve_pose(
    kin: base.A3Kinematics,
    variable_names: tuple[str, ...],
    target_pos: np.ndarray,
    target_normal: np.ndarray | None,
    normal_sign: float,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    support_q: dict[str, float],
    seed_q: dict[str, float],
) -> dict:
    lo, hi = joint_bounds(kin, variable_names)
    q0 = np.array([seed_q.get(name, support_q[name]) for name in variable_names])
    q0 = np.clip(q0, lo + 1.0e-6, hi - 1.0e-6)
    desired_normal = None if target_normal is None else base.unit(target_normal)

    def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = dict(support_q)
        q.update(dict(zip(variable_names, values)))
        kin.set_pose(root_pos, root_quat, q)
        pos, rotation = kin.racket_pose()
        return pos, rotation[:, 1] * float(normal_sign)

    def residual(values: np.ndarray) -> np.ndarray:
        pos, actual_normal = evaluate(values)
        terms = [(pos - target_pos) / 0.025]
        if desired_normal is not None:
            terms.append(np.cross(actual_normal, desired_normal) / 0.20)
        terms.append((values - q0) * 0.035)
        span = np.maximum(hi - lo, 1.0e-6)
        margin = np.minimum((values - lo) / span, (hi - values) / span)
        terms.append(np.maximum(0.0, 0.05 - margin) * 10.0)
        return np.concatenate(terms)

    default_seed = np.array([kin.default_joint[name] for name in variable_names])
    best: dict | None = None
    for seed in (q0, np.clip(default_seed, lo + 1.0e-6, hi - 1.0e-6)):
        result = least_squares(
            residual,
            seed,
            bounds=(lo, hi),
            max_nfev=180,
            ftol=1.0e-8,
            xtol=1.0e-8,
            gtol=1.0e-8,
        )
        pos, actual_normal = evaluate(result.x)
        pos_error = float(np.linalg.norm(pos - target_pos))
        normal_error = (
            0.0
            if desired_normal is None
            else math.degrees(
                math.acos(float(np.clip(np.dot(actual_normal, desired_normal), -1.0, 1.0)))
            )
        )
        score = pos_error + 0.002 * normal_error
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "values": result.x.copy(),
                "position_error_m": pos_error,
                "normal_error_deg": normal_error,
                "solver_success": bool(result.success),
            }
    assert best is not None
    q = dict(support_q)
    q.update(dict(zip(variable_names, best["values"])))
    kin.set_pose(root_pos, root_quat, q)
    best["q"] = q
    best.update(kin.posture_diagnostics(q))
    return best


def velocity_diagnostics(
    kin: base.A3Kinematics,
    variable_names: tuple[str, ...],
    target_velocity: np.ndarray,
    racket_normal: np.ndarray,
) -> dict[str, float | bool]:
    jac_pos = np.zeros((3, kin.model.nv), dtype=np.float64)
    jac_rot = np.zeros((3, kin.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(kin.model, kin.data, jac_pos, jac_rot, kin.site_id)
    dof_indices = []
    for name in variable_names:
        joint_id = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_indices.append(int(kin.model.jnt_dofadr[joint_id]))
    jp = jac_pos[:, dof_indices]
    jr = jac_rot[:, dof_indices]

    normal = base.unit(racket_normal)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, helper))) > 0.90:
        helper = np.array([1.0, 0.0, 0.0])
    tangent_1 = base.unit(np.cross(normal, helper))
    tangent_2 = base.unit(np.cross(normal, tangent_1))

    limits = np.array([VELOCITY_LIMIT[name] for name in variable_names])

    # Planner commands specify an impact normal, not a target angular velocity.
    # Linear-only feasibility is therefore the primary contract.
    linear_solution = lsq_linear(
        jp, target_velocity, bounds=(-limits, limits), lsmr_tol="auto"
    )
    linear_qdot = linear_solution.x
    linear_velocity = jp @ linear_qdot
    linear_omega = jr @ linear_qdot
    linear_normal_rate = float(
        np.linalg.norm(
            linear_omega - normal * float(np.dot(normal, linear_omega))
        )
    )

    # Separately report the stricter case that keeps the face normal nearly
    # stationary while allowing free spin about it.
    angular_weight = 0.20
    matrix = np.vstack(
        (
            jp,
            angular_weight * (tangent_1 @ jr)[None, :],
            angular_weight * (tangent_2 @ jr)[None, :],
        )
    )
    rhs = np.concatenate((target_velocity, np.zeros(2)))
    stable_solution = lsq_linear(
        matrix, rhs, bounds=(-limits, limits), lsmr_tol="auto"
    )
    stable_qdot = stable_solution.x
    stable_velocity = jp @ stable_qdot
    stable_omega = jr @ stable_qdot
    stable_normal_rate = float(
        np.linalg.norm(
            stable_omega - normal * float(np.dot(normal, stable_omega))
        )
    )
    singular_values = np.linalg.svd(jp, compute_uv=False)
    return {
        "velocity_error_mps": float(
            np.linalg.norm(linear_velocity - target_velocity)
        ),
        "normal_rate_rps": linear_normal_rate,
        "max_velocity_limit_ratio": float(
            np.max(np.abs(linear_qdot) / limits)
        ),
        "face_stable_velocity_error_mps": float(
            np.linalg.norm(stable_velocity - target_velocity)
        ),
        "face_stable_normal_rate_rps": stable_normal_rate,
        "face_stable_max_velocity_limit_ratio": float(
            np.max(np.abs(stable_qdot) / limits)
        ),
        "linear_jacobian_sigma_min": float(singular_values[-1]),
        "linear_jacobian_condition": float(
            singular_values[0] / max(singular_values[-1], 1.0e-9)
        ),
        "velocity_solver_success": bool(linear_solution.success),
        "face_stable_velocity_solver_success": bool(stable_solution.success),
    }


def normalized_margin(
    kin: base.A3Kinematics, q: dict[str, float], names: Iterable[str]
) -> tuple[float, str]:
    minimum = math.inf
    limiting = ""
    for name in names:
        lower, upper = joint_bounds(kin, (name,))
        lo, hi = float(lower[0]), float(upper[0])
        margin = min((q[name] - lo) / (hi - lo), (hi - q[name]) / (hi - lo))
        if margin < minimum:
            minimum = float(margin)
            limiting = name
    return minimum, limiting


def mode_result(
    kin: base.A3Kinematics,
    sample: PlannerSample,
    mode: str,
    root_pos: np.ndarray,
    default_q: dict[str, float],
    motion_q: dict[str, float],
    args: argparse.Namespace,
) -> dict:
    support = dict(default_q)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0])
    if mode == "arm_locked_waist":
        variables = RIGHT_ARM
        result = solve_pose(
            kin,
            variables,
            sample.position,
            sample.normal,
            sample.side,
            root_pos,
            root_quat,
            support,
            motion_q,
        )
    elif mode == "full_waist_arm":
        variables = FULL_CHAIN
        result = solve_pose(
            kin,
            variables,
            sample.position,
            sample.normal,
            sample.side,
            root_pos,
            root_quat,
            support,
            motion_q,
        )
    elif mode == "wrist_after_position":
        position_result = solve_pose(
            kin,
            SHOULDER_ELBOW,
            sample.position,
            None,
            sample.side,
            root_pos,
            root_quat,
            support,
            motion_q,
        )
        support = position_result["q"]
        variables = WRIST
        result = solve_pose(
            kin,
            variables,
            sample.position,
            sample.normal,
            sample.side,
            root_pos,
            root_quat,
            support,
            motion_q,
        )
        result["position_stage_error_m"] = position_result["position_error_m"]
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    _, rotation = kin.racket_pose()
    actual_normal = rotation[:, 1] * sample.side
    # The staged mode isolates whether wrist rotation can finish the pose, but
    # shoulder/elbow remain available to generate the impact linear velocity.
    velocity_variables = RIGHT_ARM if mode == "wrist_after_position" else variables
    result.update(
        velocity_diagnostics(kin, velocity_variables, sample.velocity, actual_normal)
    )
    wrist_margin, wrist_joint = normalized_margin(kin, result["q"], WRIST)
    arm_margin, arm_joint = normalized_margin(kin, result["q"], RIGHT_ARM)
    waist_delta = np.array([result["q"][name] - default_q[name] for name in WAIST])
    wrist_delta = np.array([result["q"][name] - motion_q[name] for name in WRIST])
    result.update(
        {
            "wrist_margin_fraction": wrist_margin,
            "wrist_limiting_joint": wrist_joint,
            "arm_margin_fraction": arm_margin,
            "arm_limiting_joint": arm_joint,
            "waist_delta_rms_rad": float(np.sqrt(np.mean(np.square(waist_delta)))),
            "waist_yaw_delta_rad": float(
                result["q"]["waist_yaw_joint"] - default_q["waist_yaw_joint"]
            ),
            "waist_roll_delta_rad": float(
                result["q"]["waist_roll_joint"] - default_q["waist_roll_joint"]
            ),
            "waist_pitch_delta_rad": float(
                result["q"]["waist_pitch_joint"] - default_q["waist_pitch_joint"]
            ),
            "wrist_delta_from_motion_rms_rad": float(
                np.sqrt(np.mean(np.square(wrist_delta)))
            ),
        }
    )
    result["pose_feasible"] = bool(
        result["position_error_m"] <= args.position_threshold
        and result["normal_error_deg"] <= args.normal_threshold_deg
        and result["wrist_margin_fraction"] >= 0.05
    )
    result["velocity_feasible"] = bool(
        result["velocity_error_mps"] <= args.velocity_error_threshold
        and result["max_velocity_limit_ratio"] <= 1.0 + 1.0e-6
    )
    result["full_feasible"] = bool(
        result["pose_feasible"]
        and result["velocity_feasible"]
        and result["com_support_margin_m"] >= 0.0
    )
    return result


def output_row(
    sample: PlannerSample,
    mode: str,
    station_y_offset: float,
    result: dict,
) -> dict:
    return {
        "trial": sample.trial,
        "side": "forehand" if sample.side >= 0.0 else "backhand",
        "clip_name": sample.clip.name,
        "mode": mode,
        "station_y_offset_m": station_y_offset,
        "target_x": float(sample.position[0]),
        "target_y": float(sample.position[1]),
        "target_z": float(sample.position[2]),
        "station_x": float(sample.station[0]),
        "station_y": float(sample.station[1] + station_y_offset),
        "target_rel_station_y": float(
            sample.position[1] - sample.station[1] - station_y_offset
        ),
        "target_speed_mps": float(np.linalg.norm(sample.velocity)),
        **{
            key: value
            for key, value in result.items()
            if key not in {"q", "values", "score"}
        },
    }


def aggregate(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for row in rows:
        key = (row["side"], row["mode"], float(row["station_y_offset_m"]))
        grouped.setdefault(key, []).append(row)

    curves: dict[str, dict] = {}
    recommendations: dict[str, dict] = {}
    for (side, mode, offset), values in sorted(grouped.items()):
        key = f"{side}|{mode}|{offset:+.3f}"

        def values_of(name: str) -> np.ndarray:
            return np.asarray([float(value[name]) for value in values], dtype=np.float64)

        curves[key] = {
            "count": len(values),
            "pose_feasible_rate": float(np.mean(values_of("pose_feasible"))),
            "velocity_feasible_rate": float(np.mean(values_of("velocity_feasible"))),
            "full_feasible_rate": float(np.mean(values_of("full_feasible"))),
            "position_error_p95_m": float(
                np.percentile(values_of("position_error_m"), 95)
            ),
            "normal_error_p95_deg": float(
                np.percentile(values_of("normal_error_deg"), 95)
            ),
            "velocity_error_p95_mps": float(
                np.percentile(values_of("velocity_error_mps"), 95)
            ),
            "wrist_margin_p05": float(
                np.percentile(values_of("wrist_margin_fraction"), 5)
            ),
            "arm_margin_p05": float(
                np.percentile(values_of("arm_margin_fraction"), 5)
            ),
            "com_margin_p05_m": float(
                np.percentile(values_of("com_support_margin_m"), 5)
            ),
            "max_velocity_ratio_p95": float(
                np.percentile(values_of("max_velocity_limit_ratio"), 95)
            ),
        }

    for side in ("forehand", "backhand"):
        for mode in ("arm_locked_waist", "wrist_after_position", "full_waist_arm"):
            candidates = [
                (key, value)
                for key, value in curves.items()
                if key.startswith(f"{side}|{mode}|")
            ]
            if not candidates:
                continue
            best_key, best = max(
                candidates,
                key=lambda item: (
                    item[1]["full_feasible_rate"],
                    item[1]["pose_feasible_rate"],
                    item[1]["wrist_margin_p05"],
                    -abs(float(item[0].rsplit("|", 1)[1])),
                ),
            )
            recommendations[f"{side}|{mode}"] = {
                "best_station_y_offset_m": float(best_key.rsplit("|", 1)[1]),
                **best,
            }
    return {"curves": curves, "recommendations": recommendations}


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in known:
                known.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joint_names = list(yaml.safe_load(args.joint_order.read_text())["joint_order"])
    adapter = yaml.safe_load(args.adapter.read_text())
    default_q = {name: float(value) for name, value in adapter["default_q"].items()}
    kin = base.A3Kinematics(args.robot_xml, joint_names, default_q)
    if args.joint_limits_yaml is not None:
        limit_doc = yaml.safe_load(args.joint_limits_yaml.read_text(encoding="utf-8"))
        lower_key = f"{args.joint_limits_prefix}_lower"
        upper_key = f"{args.joint_limits_prefix}_upper"
        if lower_key not in limit_doc or upper_key not in limit_doc:
            raise ValueError(
                f"{args.joint_limits_yaml} must define {lower_key} and {upper_key}"
            )
        lower_doc = limit_doc[lower_key]
        upper_doc = limit_doc[upper_key]
        kin.joint_bound_override = {
            name: (float(lower_doc[name]), float(upper_doc[name]))
            for name in joint_names
        }
        # Reject a limit set that excludes the deterministic reset pose.
        for name in joint_names:
            lo, hi = kin.joint_bound_override[name]
            if not lo <= default_q[name] <= hi:
                raise ValueError(
                    f"joint-limit override excludes default_q for {name}: "
                    f"{default_q[name]} not in [{lo}, {hi}]"
                )
    clips = base.load_clips(args.manifest, joint_names)
    for clip in clips:
        motion_q = base.clip_q(clip, joint_names)
        kin.set_pose(clip.root_pos, clip.root_quat, motion_q)
        _, rotation = kin.racket_pose()
        clip.motion_normal = base.unit(rotation[:, 1] * clip.side)

    samples = load_samples(
        args.planner_csv,
        clips,
        kin,
        tuple(args.station_clip_x),
        tuple(args.station_clip_y),
    )
    offsets = np.arange(
        args.station_y_min,
        args.station_y_max + 0.5 * args.station_y_step,
        args.station_y_step,
    )
    rows: list[dict] = []
    modes = ("arm_locked_waist", "wrist_after_position", "full_waist_arm")
    total = len(samples) * len(offsets)
    for index, sample in enumerate(samples):
        motion_q = base.clip_q(sample.clip, joint_names)
        for offset in offsets:
            root_pos = np.array(
                [
                    sample.station[0],
                    sample.station[1] + float(offset),
                    kin.default_root[2],
                ]
            )
            for mode in modes:
                result = mode_result(
                    kin,
                    sample,
                    mode,
                    root_pos,
                    default_q,
                    motion_q,
                    args,
                )
                rows.append(output_row(sample, mode, float(offset), result))
        print(
            f"[strike-feasibility] solved {index + 1}/{len(samples)} planner commands "
            f"({(index + 1) * len(offsets)}/{total} station cases)",
            flush=True,
        )

    write_csv(args.output_dir / "strike_feasibility_rows.csv", rows)
    aggregated = aggregate(rows)
    summary = {
        "inputs": {
            "robot_xml": str(args.robot_xml.resolve()),
            "manifest": str(args.manifest.resolve()),
            "planner_csv": str(args.planner_csv.resolve()),
            "planner_commands": len(samples),
            "station_y_offsets_m": [float(value) for value in offsets],
            "joint_limits_yaml": (
                None
                if args.joint_limits_yaml is None
                else str(args.joint_limits_yaml.resolve())
            ),
            "joint_limits_prefix": args.joint_limits_prefix,
        },
        "thresholds": {
            "position_error_m": args.position_threshold,
            "normal_error_deg": args.normal_threshold_deg,
            "velocity_error_mps": args.velocity_error_threshold,
            "normal_rate_rps": args.normal_rate_threshold,
            "minimum_wrist_margin_fraction": 0.05,
            "minimum_com_support_margin_m": 0.0,
        },
        **aggregated,
    }
    (args.output_dir / "strike_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["recommendations"], indent=2))


if __name__ == "__main__":
    main()
