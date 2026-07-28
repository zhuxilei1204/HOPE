#!/usr/bin/env python3
"""Audit support transfer, foot slip, balance, braking, and recovery in motion clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = (
    ROOT
    / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
)
DEFAULT_MANIFEST = (
    ROOT
    / "hope_training/motions/user_four_motion_manual_hits_a3fk_yaw_leg_stabilized_20260724/manifest.tsv"
)
DEFAULT_ANNOTATIONS = (
    ROOT
    / "analysis/user_four_motion_20260724/motion_annotation_videos_20260724/"
    "recovery_phase_annotations.tsv"
)
DEFAULT_JOINT_ORDER = ROOT / "hope_training/config/joint_order_agibot_a3.yaml"

LEG_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
RIGHT_ARM_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
BODY_NAMES = (
    "pelvis_link",
    "left_hip_roll_Link",
    "left_knee_Link",
    "left_ankle_roll_Link",
    "right_hip_roll_Link",
    "right_knee_Link",
    "right_ankle_roll_Link",
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--joint-order", type=Path, default=DEFAULT_JOINT_ORDER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-height-m", type=float, default=0.030)
    parser.add_argument(
        "--contact-mode",
        choices=("height", "height_velocity", "stored"),
        default="height",
        help=(
            "height_velocity treats a low but moving shuffle foot as swing; "
            "stored uses contact labels embedded by contact-aware retargeting."
        ),
    )
    parser.add_argument("--contact-speed-m-s", type=float, default=0.05)
    return parser.parse_args()


def gradient(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values.astype(np.float64), dt, axis=0, edge_order=1)


def quat_rpy_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)


def signed_polygon_margin(point: np.ndarray, points: np.ndarray) -> float:
    if len(points) < 3:
        return math.nan
    polygon = points[ConvexHull(points).vertices]
    center = polygon.mean(axis=0)
    margins = []
    for idx, p0 in enumerate(polygon):
        p1 = polygon[(idx + 1) % len(polygon)]
        edge = p1 - p0
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        if float(np.dot(normal, center - p0)) < 0.0:
            normal *= -1.0
        margins.append(float(np.dot(normal, point - p0)))
    return min(margins)


def p(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if values.size else math.nan


def norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=-1)


def load_annotations(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["name"]: row for row in csv.DictReader(handle, delimiter="\t")}


def support_corners(
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    contact: bool,
) -> list[np.ndarray]:
    if not contact:
        return []
    q = np.asarray(quat_wxyz, dtype=np.float64)
    rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    corners = []
    for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        corners.append(
            position[:2] + sx * 0.105 * rotation[:2, 0] + sy * 0.050 * rotation[:2, 1]
        )
    return corners


class ComForwardKinematics:
    def __init__(self, xml_path: Path, joint_names: list[str]) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.base_qadr = int(
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint")
            ]
        )
        self.qadr = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.qadr.append(int(self.model.jnt_qposadr[jid]))
        self.mass_ids = np.array(
            [
                bid
                for bid in range(1, self.model.nbody)
                if float(self.model.body_mass[bid]) > 0.0
            ],
            dtype=np.int32,
        )
        self.body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in BODY_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.body_ids < 0):
            missing = [
                name for name, body_id in zip(BODY_NAMES, self.body_ids) if body_id < 0
            ]
            raise ValueError(f"robot model is missing tracked bodies: {missing}")

    def body_trajectory(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        body_pos = np.empty((len(root_pos), len(self.body_ids), 3), dtype=np.float64)
        body_quat = np.empty((len(root_pos), len(self.body_ids), 4), dtype=np.float64)
        for frame in range(len(root_pos)):
            self.data.qpos[self.base_qadr : self.base_qadr + 3] = root_pos[frame]
            self.data.qpos[self.base_qadr + 3 : self.base_qadr + 7] = root_quat[frame]
            self.data.qpos[self.qadr] = joint_pos[frame]
            mujoco.mj_forward(self.model, self.data)
            body_pos[frame] = self.data.xpos[self.body_ids]
            body_quat[frame] = self.data.xquat[self.body_ids]
        return body_pos, body_quat

    def com_trajectory(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
    ) -> np.ndarray:
        output = np.empty_like(root_pos, dtype=np.float64)
        masses = self.model.body_mass[self.mass_ids]
        for frame in range(len(root_pos)):
            self.data.qpos[self.base_qadr : self.base_qadr + 3] = root_pos[frame]
            self.data.qpos[self.base_qadr + 3 : self.base_qadr + 7] = root_quat[frame]
            self.data.qpos[self.qadr] = joint_pos[frame]
            mujoco.mj_forward(self.model, self.data)
            output[frame] = (
                np.sum(self.data.xipos[self.mass_ids] * masses[:, None], axis=0) / masses.sum()
            )
        return output


def angular_velocity_from_quat(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference world angular velocity for body-to-world quaternions."""
    quat_xyzw = np.asarray(quat_wxyz)[..., [1, 2, 3, 0]]
    output = np.zeros((*quat_wxyz.shape[:-1], 3), dtype=np.float64)
    for body in range(quat_wxyz.shape[1]):
        current = Rotation.from_quat(quat_xyzw[:, body])
        if len(current) > 2:
            output[1:-1, body] = (
                current[2:] * current[:-2].inv()
            ).as_rotvec() / (2.0 * dt)
            output[0, body] = (current[1] * current[0].inv()).as_rotvec() / dt
            output[-1, body] = (current[-1] * current[-2].inv()).as_rotvec() / dt
        elif len(current) == 2:
            omega = (current[1] * current[0].inv()).as_rotvec() / dt
            output[:, body] = omega
    return output


def load_clip_arrays(
    path: Path,
    fk: ComForwardKinematics,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load a canonical NPZ or raw GMR pickle into the common audit representation."""
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            data = pickle.load(handle)
        fps = float(data["fps"])
        joint_pos = np.asarray(data["dof_pos"], dtype=np.float64)
        root_pos = np.asarray(data["root_pos"], dtype=np.float64)
        # GMR stores root rotation as xyzw; MuJoCo and the canonical NPZ use wxyz.
        root_quat = np.asarray(data["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]
        body_pos, body_quat = fk.body_trajectory(root_pos, root_quat, joint_pos)
        body_ang = angular_velocity_from_quat(body_quat, 1.0 / fps)
        return fps, joint_pos, body_pos, body_quat, body_ang, None

    data = np.load(path)
    stored_contact = None
    if "left_foot_contact" in data and "right_foot_contact" in data:
        stored_contact = np.stack(
            (
                np.asarray(data["left_foot_contact"], dtype=bool),
                np.asarray(data["right_foot_contact"], dtype=bool),
            ),
            axis=1,
        )
    return (
        float(data["fps"]),
        np.asarray(data["joint_pos"], dtype=np.float64),
        np.asarray(data["body_pos_w"], dtype=np.float64),
        np.asarray(data["body_quat_w"], dtype=np.float64),
        np.asarray(data["body_ang_vel_w"], dtype=np.float64),
        stored_contact,
    )


def audit_clip(
    row: dict,
    annotations: dict[str, dict],
    joint_names: list[str],
    fk: ComForwardKinematics,
    contact_height: float,
    contact_mode: str,
    contact_speed: float,
) -> tuple[dict, list[dict]]:
    path = Path(row["output"])
    fps, joint_pos, body_pos, body_quat, body_ang, stored_contact = load_clip_arrays(path, fk)
    dt = 1.0 / fps
    joint_vel = gradient(joint_pos, dt)
    joint_acc = gradient(joint_vel, dt)
    body_vel = gradient(body_pos, dt)
    root_pos = body_pos[:, 0]
    root_quat = body_quat[:, 0]
    com = fk.com_trajectory(root_pos, root_quat, joint_pos)

    index = {name: idx for idx, name in enumerate(joint_names)}
    leg_idx = np.array([index[name] for name in LEG_NAMES])
    waist_idx = np.array([index[name] for name in WAIST_NAMES])
    arm_idx = np.array([index[name] for name in RIGHT_ARM_NAMES])
    left_pos, right_pos = body_pos[:, 3], body_pos[:, 6]
    left_quat, right_quat = body_quat[:, 3], body_quat[:, 6]
    left_speed = norm(body_vel[:, 3, :2])
    right_speed = norm(body_vel[:, 6, :2])
    floor_ref = float(min(np.percentile(left_pos[:20, 2], 10), np.percentile(right_pos[:20, 2], 10)))
    if contact_mode == "stored":
        if stored_contact is None:
            raise ValueError(f"{path}: contact-mode=stored but NPZ has no contact labels")
        left_contact = stored_contact[:, 0].copy()
        right_contact = stored_contact[:, 1].copy()
    else:
        left_contact = left_pos[:, 2] <= floor_ref + contact_height
        right_contact = right_pos[:, 2] <= floor_ref + contact_height
    if contact_mode == "height_velocity":
        left_contact &= left_speed <= contact_speed
        right_contact &= right_speed <= contact_speed
        no_contact = ~(left_contact | right_contact)
        left_near = left_pos[:, 2] <= floor_ref + contact_height
        right_near = right_pos[:, 2] <= floor_ref + contact_height
        choose_left = no_contact & left_near & (
            ~right_near | (left_speed <= right_speed)
        )
        choose_right = no_contact & right_near & ~choose_left
        left_contact |= choose_left
        right_contact |= choose_right

    support_margin = np.full(len(root_pos), np.nan)
    for frame in range(len(root_pos)):
        corners = support_corners(left_pos[frame], left_quat[frame], bool(left_contact[frame]))
        corners += support_corners(right_pos[frame], right_quat[frame], bool(right_contact[frame]))
        if len(corners) >= 4:
            support_margin[frame] = signed_polygon_margin(com[frame, :2], np.asarray(corners))

    strike = int(row.get("strike_frame") or 60)
    annotation = annotations.get(row["name"], {})
    settle = int(
        annotation.get("settle_start_motion_frame")
        or annotation.get("mapped_motion_frame")
        or min(strike + 20, len(root_pos) - 1)
    )
    settle = int(np.clip(settle, strike + 1, len(root_pos) - 1))
    final_start = max(settle, len(root_pos) - int(round(0.50 * fps)))
    strike_slice = slice(max(0, strike - 10), min(len(root_pos), strike + 11))
    recovery_slice = slice(settle, len(root_pos))
    final_slice = slice(final_start, len(root_pos))

    grounded_speed = np.concatenate((left_speed[left_contact], right_speed[right_contact]))
    leg_range = np.ptp(joint_pos[:, leg_idx], axis=0)
    root_rpy = quat_rpy_deg(root_quat)
    torso_rpy = quat_rpy_deg(body_quat[:, 7])
    root_planar_speed = norm(body_vel[:, 0, :2])
    root_ang_speed = norm(body_ang[:, 0])
    ready_root = root_pos[:20].mean(axis=0)
    final_root = root_pos[final_slice].mean(axis=0)
    ready_q = joint_pos[:20].mean(axis=0)
    final_q = joint_pos[final_slice].mean(axis=0)
    final_joint_delta = final_q - ready_q
    ready_wrist = body_pos[:20, 13].mean(axis=0)
    final_wrist = body_pos[final_slice, 13].mean(axis=0)

    summary = {
        "name": row["name"],
        "source_video": row.get("source_video"),
        "frames": len(root_pos),
        "fps": fps,
        "strike_frame": strike,
        "settle_start_frame_annotated": settle,
        "root_xy_span_m": float(np.linalg.norm(np.ptp(root_pos[:, :2], axis=0))),
        "root_height_span_m": float(np.ptp(root_pos[:, 2])),
        "root_pitch_abs_p95_deg": p(np.abs(root_rpy[:, 1]), 95),
        "torso_pitch_abs_p95_deg": p(np.abs(torso_rpy[:, 1]), 95),
        "leg_joint_range_rms_rad": float(np.sqrt(np.mean(leg_range**2))),
        "leg_joint_speed_p95_rad_s": p(norm(joint_vel[:, leg_idx]), 95),
        "leg_joint_acc_p95_rad_s2": p(norm(joint_acc[:, leg_idx]), 95),
        "right_arm_speed_p95_rad_s": p(norm(joint_vel[:, arm_idx]), 95),
        "strike_leg_speed_p95_rad_s": p(norm(joint_vel[strike_slice, leg_idx]), 95),
        "strike_arm_speed_p95_rad_s": p(norm(joint_vel[strike_slice, arm_idx]), 95),
        "strike_leg_to_arm_speed_ratio": float(
            p(norm(joint_vel[strike_slice, leg_idx]), 95)
            / max(p(norm(joint_vel[strike_slice, arm_idx]), 95), 1.0e-6)
        ),
        "double_support_fraction": float(np.mean(left_contact & right_contact)),
        "single_support_fraction": float(np.mean(left_contact ^ right_contact)),
        "flight_fraction": float(np.mean(~left_contact & ~right_contact)),
        "left_contact_fraction": float(np.mean(left_contact)),
        "right_contact_fraction": float(np.mean(right_contact)),
        "foot_height_gap_p50_m": p(np.abs(left_pos[:, 2] - right_pos[:, 2]), 50),
        "foot_height_gap_p95_m": p(np.abs(left_pos[:, 2] - right_pos[:, 2]), 95),
        "grounded_foot_speed_p50_m_s": p(grounded_speed, 50),
        "grounded_foot_speed_p95_m_s": p(grounded_speed, 95),
        "grounded_foot_speed_max_m_s": p(grounded_speed, 100),
        "left_grounded_speed_p95_m_s": p(left_speed[left_contact], 95),
        "right_grounded_speed_p95_m_s": p(right_speed[right_contact], 95),
        "com_support_margin_p05_m": p(support_margin, 5),
        "com_support_margin_at_strike_m": float(support_margin[strike]),
        "com_outside_support_fraction": float(np.nanmean(support_margin < 0.0)),
        "recovery_root_speed_start_m_s": float(root_planar_speed[settle]),
        "recovery_root_speed_final_p95_m_s": p(root_planar_speed[final_slice], 95),
        "recovery_root_ang_speed_start_rad_s": float(root_ang_speed[settle]),
        "recovery_root_ang_speed_final_p95_rad_s": p(root_ang_speed[final_slice], 95),
        "recovery_right_arm_speed_final_p95_rad_s": p(
            norm(joint_vel[final_slice, arm_idx]), 95
        ),
        "recovery_com_outside_fraction": float(
            np.nanmean(support_margin[recovery_slice] < 0.0)
        ),
        "final_root_xy_from_ready_m": float(np.linalg.norm(final_root[:2] - ready_root[:2])),
        "final_joint_rms_from_ready_rad": float(np.sqrt(np.mean(final_joint_delta**2))),
        "final_leg_rms_from_ready_rad": float(
            np.sqrt(np.mean(final_joint_delta[leg_idx] ** 2))
        ),
        "final_waist_rms_from_ready_rad": float(
            np.sqrt(np.mean(final_joint_delta[waist_idx] ** 2))
        ),
        "final_right_arm_rms_from_ready_rad": float(
            np.sqrt(np.mean(final_joint_delta[arm_idx] ** 2))
        ),
        "final_right_wrist_from_ready_m": float(np.linalg.norm(final_wrist - ready_wrist)),
        "final_root_drift_m": float(
            np.linalg.norm(root_pos[final_slice][-1, :2] - root_pos[final_slice][0, :2])
        ),
        "foot_slip_pass_0p10": bool(p(grounded_speed, 95) < 0.10),
        "foot_slip_pass_0p05": bool(p(grounded_speed, 95) < 0.05),
        "has_support_transfer": bool(np.mean(left_contact ^ right_contact) >= 0.08),
        "has_dynamic_leg_compensation": bool(
            p(norm(joint_vel[strike_slice, leg_idx]), 95) >= 0.35
        ),
        "settles_by_end": bool(
            p(root_planar_speed[final_slice], 95) < 0.08
            and p(root_ang_speed[final_slice], 95) < 0.40
        ),
    }
    frames: list[dict] = []
    for frame in range(len(root_pos)):
        frames.append(
            {
                "name": row["name"],
                "frame": frame,
                "time_s": frame / fps,
                "phase": (
                    "pre_strike"
                    if frame < strike
                    else "strike"
                    if frame == strike
                    else "follow_through"
                    if frame < settle
                    else "settle_recovery"
                ),
                "root_x": root_pos[frame, 0],
                "root_y": root_pos[frame, 1],
                "root_z": root_pos[frame, 2],
                "root_pitch_deg": root_rpy[frame, 1],
                "torso_pitch_deg": torso_rpy[frame, 1],
                "root_speed_xy": root_planar_speed[frame],
                "root_ang_speed": root_ang_speed[frame],
                "left_contact": int(left_contact[frame]),
                "right_contact": int(right_contact[frame]),
                "left_foot_z": left_pos[frame, 2],
                "right_foot_z": right_pos[frame, 2],
                "left_foot_speed_xy": left_speed[frame],
                "right_foot_speed_xy": right_speed[frame],
                "com_x": com[frame, 0],
                "com_y": com[frame, 1],
                "com_support_margin": support_margin[frame],
                "leg_joint_speed_norm": norm(joint_vel[frame, leg_idx]),
                "waist_joint_speed_norm": norm(joint_vel[frame, waist_idx]),
                "right_arm_joint_speed_norm": norm(joint_vel[frame, arm_idx]),
            }
        )
    return summary, frames


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joint_names = list(yaml.safe_load(args.joint_order.read_text())["joint_order"])
    annotations = load_annotations(args.annotations)
    fk = ComForwardKinematics(args.robot_xml, joint_names)
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    summaries = []
    frames = []
    for row in manifest_rows:
        summary, clip_frames = audit_clip(
            row,
            annotations,
            joint_names,
            fk,
            args.contact_height_m,
            args.contact_mode,
            args.contact_speed_m_s,
        )
        summaries.append(summary)
        frames.extend(clip_frames)
        print(
            f"[motion-audit] {summary['name']}: slip_p95="
            f"{summary['grounded_foot_speed_p95_m_s']:.3f} m/s, "
            f"COM_p05={summary['com_support_margin_p05_m']:.3f} m"
        )
    write_csv(args.output_dir / "motion_lower_body_summary.csv", summaries)
    write_csv(args.output_dir / "motion_lower_body_frames.csv", frames)
    output = {
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "annotations": str(args.annotations.resolve()),
            "robot_xml": str(args.robot_xml.resolve()),
            "contact_height_m": args.contact_height_m,
            "contact_mode": args.contact_mode,
            "contact_speed_m_s": args.contact_speed_m_s,
        },
        "acceptance": {
            "support_foot_speed_pass_m_s": 0.05,
            "support_foot_speed_review_m_s": 0.10,
            "com_support_margin_safe": ">= 0 m",
            "dynamic_leg_speed_strike_p95_min_rad_s": 0.35,
        },
        "clips": summaries,
    }
    (args.output_dir / "motion_lower_body_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
