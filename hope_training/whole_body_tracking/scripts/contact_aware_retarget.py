#!/usr/bin/env python3
"""Convert raw A3 GMR clips into contact-aware canonical 50 Hz motion NPZ files.

The correction deliberately leaves the waist, head, and both arms untouched. It
only changes root translation and the twelve leg joints in order to:

* infer planted-foot phases from foot height, velocity, and COM support;
* lock planted foot sites to the ground;
* lift the swing foot instead of letting it scrape through the floor;
* move the COM back into the active support polygon when kinematically feasible;
* solve the resulting pelvis/hip/knee/ankle task with MuJoCo Jacobian IK; and
* regenerate all canonical tracked-body FK and velocity arrays at 50 Hz.

Input rows use the ``segments.tsv`` schema produced by
``split_gmr_motion_segments.py``. GMR root quaternions are xyzw; canonical NPZ
quaternions are wxyz.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = (
    ROOT
    / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
DEFAULT_MANIFEST = (
    ROOT
    / "data/motion_pipeline/supplemental_lower_body_20260728/metadata/segments.tsv"
)
DEFAULT_OUTPUT = (
    ROOT / "hope_training/motions/supplemental_lower_body_contact_aware_20260728"
)
DEFAULT_JOINT_ORDER = ROOT / "hope_training/config/joint_order_agibot_a3.yaml"

TRACKED_BODIES = (
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


@dataclass
class ClipInput:
    fps: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos: np.ndarray


@dataclass
class KinematicTrajectory:
    body_pos: np.ndarray
    body_quat: np.ndarray
    foot_pos: np.ndarray
    foot_rot: np.ndarray
    com: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint-order", type=Path, default=DEFAULT_JOINT_ORDER)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--ready-frames-s", type=float, default=0.40)
    parser.add_argument("--pre-strike-s", type=float, default=2.0)
    parser.add_argument("--post-strike-s", type=float, default=2.5)
    parser.add_argument("--ground-z", type=float, default=0.001)
    parser.add_argument("--contact-height-m", type=float, default=0.040)
    parser.add_argument("--contact-speed-m-s", type=float, default=0.10)
    parser.add_argument("--swing-clearance-m", type=float, default=0.030)
    parser.add_argument("--support-margin-m", type=float, default=0.005)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument(
        "--correction-smoothing-frames",
        type=int,
        default=11,
        help="Odd Savitzky-Golay window applied only to root/leg IK corrections.",
    )
    parser.add_argument(
        "--adaptive-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate several correction windows and keep the best gated result.",
    )
    parser.add_argument("--only", nargs="*", default=None, help="Optional clip names.")
    return parser.parse_args()


def load_joint_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        names = list((yaml.safe_load(handle) or {}).get("joint_order") or [])
    if len(names) != 31:
        raise ValueError(f"{path}: expected 31 canonical joints, got {len(names)}")
    return names


def load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def load_gmr(path: Path) -> ClipInput:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_xyzw = np.asarray(data["root_rot"], dtype=np.float64)
    joint_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path}: root_pos must be [F,3], got {root_pos.shape}")
    if root_xyzw.shape != (len(root_pos), 4):
        raise ValueError(f"{path}: root_rot must be [F,4], got {root_xyzw.shape}")
    if joint_pos.shape != (len(root_pos), 31):
        raise ValueError(f"{path}: dof_pos must be [F,31], got {joint_pos.shape}")
    root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
    root_wxyz /= np.maximum(np.linalg.norm(root_wxyz, axis=1, keepdims=True), 1.0e-12)
    return ClipInput(float(data["fps"]), root_pos, root_wxyz, joint_pos)


def crop_around_strike(
    clip: ClipInput,
    strike_frame: int,
    pre_strike_s: float,
    post_strike_s: float,
) -> tuple[ClipInput, int, int, int]:
    before = int(round(pre_strike_s * clip.fps))
    after = int(round(post_strike_s * clip.fps))
    start = max(0, strike_frame - before)
    end = min(len(clip.root_pos), strike_frame + after + 1)
    cropped = ClipInput(
        clip.fps,
        clip.root_pos[start:end].copy(),
        clip.root_quat_wxyz[start:end].copy(),
        clip.joint_pos[start:end].copy(),
    )
    return cropped, strike_frame - start, start, end


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def quat_yaw_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def circular_mean(values: np.ndarray) -> float:
    return float(np.arctan2(np.sin(values).mean(), np.cos(values).mean()))


def rotate_xy(values: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.empty_like(values)
    result[:, 0] = c * values[:, 0] - s * values[:, 1]
    result[:, 1] = s * values[:, 0] + c * values[:, 1]
    return result


def resample_clip(clip: ClipInput, output_fps: float) -> ClipInput:
    frames = len(clip.root_pos)
    if frames < 2:
        raise ValueError("motion clip needs at least two frames")
    source_t = np.arange(frames, dtype=np.float64) / clip.fps
    duration = source_t[-1]
    output_frames = int(round(duration * output_fps)) + 1
    target_t = np.arange(output_frames, dtype=np.float64) / output_fps
    target_t[-1] = duration

    root_pos = PchipInterpolator(source_t, clip.root_pos, axis=0)(target_t)
    joint_pos = PchipInterpolator(source_t, clip.joint_pos, axis=0)(target_t)
    source_rot = Rotation.from_quat(clip.root_quat_wxyz[:, [1, 2, 3, 0]])
    root_xyzw = Slerp(source_t, source_rot)(target_t).as_quat()
    root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
    return ClipInput(output_fps, root_pos, root_wxyz, joint_pos)


def smooth_root_and_legs(clip: ClipInput, leg_indices: np.ndarray) -> None:
    """Suppress monocular one-frame jitter without filtering the strike arm."""
    window = min(9, len(clip.root_pos) if len(clip.root_pos) % 2 else len(clip.root_pos) - 1)
    if window < 5:
        return
    clip.root_pos[:] = savgol_filter(
        clip.root_pos, window_length=window, polyorder=3, axis=0, mode="interp"
    )
    clip.joint_pos[:, leg_indices] = savgol_filter(
        clip.joint_pos[:, leg_indices],
        window_length=window,
        polyorder=3,
        axis=0,
        mode="interp",
    )


def gradient(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=np.float64), dt, axis=0, edge_order=1)


def angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    quat_xyzw = np.asarray(quat_wxyz)[..., [1, 2, 3, 0]]
    output = np.zeros((*quat_wxyz.shape[:-1], 3), dtype=np.float64)
    for body in range(quat_wxyz.shape[1]):
        rotations = Rotation.from_quat(quat_xyzw[:, body])
        if len(rotations) > 2:
            output[1:-1, body] = (
                rotations[2:] * rotations[:-2].inv()
            ).as_rotvec() / (2.0 * dt)
            output[0, body] = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
            output[-1, body] = (
                rotations[-1] * rotations[-2].inv()
            ).as_rotvec() / dt
        else:
            omega = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
            output[:, body] = omega
    return output


def runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    output: list[tuple[int, int, bool]] = []
    start = 0
    for idx in range(1, len(mask) + 1):
        if idx == len(mask) or bool(mask[idx]) != bool(mask[start]):
            output.append((start, idx, bool(mask[start])))
            start = idx
    return output


def fill_short_false_gaps(mask: np.ndarray, maximum: int) -> np.ndarray:
    result = mask.copy()
    for start, end, value in runs(result):
        if not value and start > 0 and end < len(result) and end - start <= maximum:
            result[start:end] = True
    return result


def remove_short_true_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = mask.copy()
    for start, end, value in runs(result):
        if value and end - start < minimum:
            result[start:end] = False
    return result


def foot_corners(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    corners = []
    for sx, sy in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)):
        corners.append(
            position[:2]
            + sx * 0.105 * rotation[:2, 0]
            + sy * 0.050 * rotation[:2, 1]
        )
    return np.asarray(corners)


def convex_halfspaces(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    output = []
    for idx, point in enumerate(polygon):
        nxt = polygon[(idx + 1) % len(polygon)]
        edge = nxt - point
        inward = np.array([-edge[1], edge[0]], dtype=np.float64)
        inward /= max(float(np.linalg.norm(inward)), 1.0e-12)
        output.append((point, inward))
    return output


def signed_margin(point: np.ndarray, points: np.ndarray) -> float:
    return min(
        float(np.dot(normal, point - edge_point))
        for edge_point, normal in convex_halfspaces(points)
    )


def project_to_polygon(
    point: np.ndarray,
    points: np.ndarray,
    margin: float,
    iterations: int = 24,
) -> np.ndarray:
    projected = np.asarray(point, dtype=np.float64).copy()
    halfspaces = convex_halfspaces(points)
    for _ in range(iterations):
        changed = False
        for edge_point, normal in halfspaces:
            value = float(np.dot(normal, projected - edge_point))
            if value < margin:
                projected += (margin - value) * normal
                changed = True
        if not changed:
            break
    return projected


class A3Kinematics:
    def __init__(self, xml_path: Path, joint_names: list[str]) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.joint_names = joint_names

        root_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint"
        )
        self.root_qadr = int(self.model.jnt_qposadr[root_jid])
        self.root_dadr = int(self.model.jnt_dofadr[root_jid])
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in joint_names
            ],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            missing = [
                name for name, jid in zip(joint_names, self.joint_ids) if jid < 0
            ]
            raise ValueError(f"model is missing joints: {missing}")
        self.joint_qadr = self.model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.joint_dadr = self.model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.leg_canonical = np.array(
            [joint_names.index(name) for name in LEG_NAMES], dtype=np.int32
        )
        self.leg_qadr = self.joint_qadr[self.leg_canonical]
        self.leg_dadr = self.joint_dadr[self.leg_canonical]
        self.variable_dofs = np.concatenate(
            (np.arange(self.root_dadr, self.root_dadr + 3), self.leg_dadr)
        )

        self.tracked_body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in TRACKED_BODIES
            ],
            dtype=np.int32,
        )
        self.foot_site_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
            ],
            dtype=np.int32,
        )
        self.pelvis_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link"
        )
        if self.model.nkey:
            self.default_qpos = self.model.key_qpos[0].copy()
        else:
            self.default_qpos = self.model.qpos0.copy()
        default_quat = self.default_qpos[self.root_qadr + 3 : self.root_qadr + 7]
        self.default_root_xy = self.default_qpos[self.root_qadr : self.root_qadr + 2].copy()
        self.default_root_yaw = float(quat_yaw_wxyz(default_quat))

        self.leg_lower = np.empty(len(self.leg_qadr), dtype=np.float64)
        self.leg_upper = np.empty(len(self.leg_qadr), dtype=np.float64)
        for idx, jid in enumerate(self.joint_ids[self.leg_canonical]):
            if self.model.jnt_limited[jid]:
                self.leg_lower[idx], self.leg_upper[idx] = self.model.jnt_range[jid]
            else:
                self.leg_lower[idx], self.leg_upper[idx] = -np.inf, np.inf
        self.leg_safety_margin = np.minimum(
            0.020, 0.10 * (self.leg_upper - self.leg_lower)
        )

    def set_state(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
    ) -> None:
        self.data.qpos[:] = self.default_qpos
        self.data.qpos[self.root_qadr : self.root_qadr + 3] = root_pos
        self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7] = root_quat
        self.data.qpos[self.joint_qadr] = joint_pos
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        foot_pos = self.data.site_xpos[self.foot_site_ids].copy()
        foot_rot = self.data.site_xmat[self.foot_site_ids].reshape(2, 3, 3).copy()
        com = self.data.subtree_com[self.pelvis_body_id].copy()
        return foot_pos, foot_rot, com, self.data.qpos[self.joint_qadr].copy()

    def trajectory(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
    ) -> KinematicTrajectory:
        frames = len(root_pos)
        body_pos = np.empty((frames, len(TRACKED_BODIES), 3), dtype=np.float64)
        body_quat = np.empty((frames, len(TRACKED_BODIES), 4), dtype=np.float64)
        foot_pos = np.empty((frames, 2, 3), dtype=np.float64)
        foot_rot = np.empty((frames, 2, 3, 3), dtype=np.float64)
        com = np.empty((frames, 3), dtype=np.float64)
        for frame in range(frames):
            self.set_state(root_pos[frame], root_quat[frame], joint_pos[frame])
            body_pos[frame] = self.data.xpos[self.tracked_body_ids]
            body_quat[frame] = self.data.xquat[self.tracked_body_ids]
            foot_pos[frame] = self.data.site_xpos[self.foot_site_ids]
            foot_rot[frame] = self.data.site_xmat[self.foot_site_ids].reshape(2, 3, 3)
            com[frame] = self.data.subtree_com[self.pelvis_body_id]
        return KinematicTrajectory(body_pos, body_quat, foot_pos, foot_rot, com)

    def site_jacobian(self, foot: int) -> tuple[np.ndarray, np.ndarray]:
        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jac_pos,
            jac_rot,
            int(self.foot_site_ids[foot]),
        )
        return jac_pos[:, self.variable_dofs], jac_rot[:, self.variable_dofs]

    def com_jacobian(self) -> np.ndarray:
        jac = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSubtreeCom(
            self.model, self.data, jac, int(self.pelvis_body_id)
        )
        return jac[:, self.variable_dofs]


def canonicalize_root(clip: ClipInput, model: A3Kinematics, ready_seconds: float) -> None:
    ready_frames = max(1, min(len(clip.root_pos), int(round(ready_seconds * clip.fps))))
    source_xy = clip.root_pos[:ready_frames, :2].mean(axis=0)
    source_yaw = circular_mean(quat_yaw_wxyz(clip.root_quat_wxyz[:ready_frames]))
    yaw_delta = model.default_root_yaw - source_yaw
    clip.root_pos[:, :2] = (
        model.default_root_xy
        + rotate_xy(clip.root_pos[:, :2] - source_xy[None, :], yaw_delta)
    )
    clip.root_quat_wxyz[:] = quat_mul_wxyz(
        yaw_quat_wxyz(yaw_delta), clip.root_quat_wxyz
    )
    clip.root_quat_wxyz[:] /= np.maximum(
        np.linalg.norm(clip.root_quat_wxyz, axis=1, keepdims=True), 1.0e-12
    )


def infer_contacts(
    trajectory: KinematicTrajectory,
    fps: float,
    ground_z: float,
    height_threshold: float,
    speed_threshold: float,
) -> np.ndarray:
    foot_pos = trajectory.foot_pos
    foot_speed = np.linalg.norm(gradient(foot_pos[:, :, :2], 1.0 / fps), axis=-1)
    near = foot_pos[:, :, 2] <= ground_z + height_threshold
    contact = near & (foot_speed <= speed_threshold)

    for foot in range(2):
        contact[:, foot] = fill_short_false_gaps(contact[:, foot], maximum=3)
        contact[:, foot] = remove_short_true_runs(contact[:, foot], minimum=3)

    # Prefer a physically plausible support assignment over a height-only label.
    for frame in range(len(contact)):
        scores = (
            foot_speed[frame] / max(speed_threshold, 1.0e-6)
            + np.maximum(foot_pos[frame, :, 2] - ground_z, 0.0)
            / max(height_threshold, 1.0e-6)
        )
        if not contact[frame].any():
            contact[frame, int(np.argmin(scores))] = True

        if contact[frame].sum() == 1:
            stance = int(np.argmax(contact[frame]))
            swing = 1 - stance
            stance_points = foot_corners(
                foot_pos[frame, stance], trajectory.foot_rot[frame, stance]
            )
            swing_points = foot_corners(
                foot_pos[frame, swing], trajectory.foot_rot[frame, swing]
            )
            stance_margin = signed_margin(trajectory.com[frame, :2], stance_points)
            swing_margin = signed_margin(trajectory.com[frame, :2], swing_points)
            both_margin = signed_margin(
                trajectory.com[frame, :2],
                np.concatenate((stance_points, swing_points), axis=0),
            )
            if swing_margin > stance_margin + 0.015 and near[frame, swing]:
                contact[frame] = False
                contact[frame, swing] = True
            elif (
                stance_margin < -0.015
                and both_margin >= -0.010
                and near[frame, swing]
                and foot_speed[frame, swing] <= 2.0 * speed_threshold
            ):
                contact[frame, swing] = True

        if contact[frame].sum() == 2:
            moving = int(np.argmax(foot_speed[frame]))
            planted = 1 - moving
            planted_margin = signed_margin(
                trajectory.com[frame, :2],
                foot_corners(
                    foot_pos[frame, planted], trajectory.foot_rot[frame, planted]
                ),
            )
            if (
                foot_speed[frame, moving] > 1.5 * speed_threshold
                and planted_margin >= -0.010
            ):
                contact[frame, moving] = False

    for foot in range(2):
        contact[:, foot] = fill_short_false_gaps(contact[:, foot], maximum=2)
        contact[:, foot] = remove_short_true_runs(contact[:, foot], minimum=3)
    for frame in range(len(contact)):
        if not contact[frame].any():
            speeds = foot_speed[frame] + 5.0 * np.maximum(
                foot_pos[frame, :, 2] - ground_z, 0.0
            )
            contact[frame, int(np.argmin(speeds))] = True
    return contact


def build_foot_targets(
    raw: KinematicTrajectory,
    contact: np.ndarray,
    ground_z: float,
    swing_clearance: float,
) -> tuple[np.ndarray, np.ndarray]:
    targets = raw.foot_pos.copy()
    target_rot = raw.foot_rot.copy()
    for foot in range(2):
        for start, end, is_contact in runs(contact[:, foot]):
            if is_contact:
                anchor_end = min(end, start + 4)
                anchor_xy = np.median(raw.foot_pos[start:anchor_end, foot, :2], axis=0)
                yaw = circular_mean(
                    Rotation.from_matrix(
                        raw.foot_rot[start:anchor_end, foot]
                    ).as_euler("xyz")[:, 2]
                )
                rotation = Rotation.from_euler("z", yaw).as_matrix()
                targets[start:end, foot, :2] = anchor_xy
                targets[start:end, foot, 2] = ground_z
                target_rot[start:end, foot] = rotation
            else:
                length = end - start
                if length:
                    phase = (np.arange(length, dtype=np.float64) + 1.0) / (length + 1.0)
                    clearance = ground_z + swing_clearance * np.sin(math.pi * phase)
                    targets[start:end, foot, 2] = np.maximum(
                        targets[start:end, foot, 2], clearance
                    )
    return targets, target_rot


def support_points(
    contact_row: np.ndarray,
    foot_target: np.ndarray,
    foot_rotation: np.ndarray,
) -> np.ndarray:
    points = [
        foot_corners(foot_target[foot], foot_rotation[foot])
        for foot in range(2)
        if contact_row[foot]
    ]
    return np.concatenate(points, axis=0)


def solve_frame(
    model: A3Kinematics,
    raw_root: np.ndarray,
    root_quat: np.ndarray,
    raw_joint: np.ndarray,
    initial_root: np.ndarray,
    initial_joint: np.ndarray,
    previous_root: np.ndarray | None,
    previous_joint: np.ndarray | None,
    contact: np.ndarray,
    foot_target: np.ndarray,
    foot_rotation: np.ndarray,
    support_margin: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    root = initial_root.copy()
    joint = initial_joint.copy()
    support = support_points(contact, foot_target, foot_rotation)

    for _ in range(max_iterations):
        model.set_state(root, root_quat, joint)
        current_foot, current_rot, current_com, _ = model.snapshot()
        rows = []
        errors = []

        for foot in range(2):
            jac_pos, jac_rot = model.site_jacobian(foot)
            position_error = foot_target[foot] - current_foot[foot]
            if contact[foot]:
                scale = np.array([0.004, 0.004, 0.002])
                rows.append(jac_pos / scale[:, None])
                errors.append(position_error / scale)
                rotation_error = Rotation.from_matrix(
                    foot_rotation[foot] @ current_rot[foot].T
                ).as_rotvec()
                rows.append(jac_rot[:2] / 0.10)
                errors.append(rotation_error[:2] / 0.10)
            else:
                scale = np.array([0.060, 0.060, 0.015])
                rows.append(jac_pos / scale[:, None])
                errors.append(position_error / scale)

        projected_com = project_to_polygon(
            current_com[:2], support, margin=support_margin
        )
        com_error = projected_com - current_com[:2]
        if np.linalg.norm(com_error) > 1.0e-5:
            jac_com = model.com_jacobian()[:2]
            rows.append(jac_com / 0.020)
            errors.append(com_error / 0.020)

        identity = np.eye(len(model.variable_dofs))
        current_variables = np.concatenate((root, joint[model.leg_canonical]))
        raw_variables = np.concatenate((raw_root, raw_joint[model.leg_canonical]))
        regularization_scale = np.concatenate(
            (np.array([0.20, 0.20, 0.12]), np.full(12, 0.40))
        )
        rows.append(identity / regularization_scale[:, None])
        errors.append((raw_variables - current_variables) / regularization_scale)

        if previous_root is not None and previous_joint is not None:
            # ``initial`` carries the previous frame's IK correction on top of
            # the current raw motion. Regularizing to it smooths only the
            # correction, instead of damping the intended leg trajectory.
            correction_target = np.concatenate(
                (initial_root, initial_joint[model.leg_canonical])
            )
            temporal_scale = np.concatenate(
                (np.array([0.20, 0.20, 0.12]), np.full(12, 0.35))
            )
            rows.append(0.35 * identity / temporal_scale[:, None])
            errors.append(
                0.35 * (correction_target - current_variables) / temporal_scale
            )

        matrix = np.concatenate(rows, axis=0)
        target = np.concatenate(errors, axis=0)
        damping = 2.0e-3
        delta = np.linalg.solve(
            matrix.T @ matrix + damping * np.eye(matrix.shape[1]),
            matrix.T @ target,
        )
        delta[:3] = np.clip(delta[:3], -0.020, 0.020)
        delta[3:] = np.clip(delta[3:], -0.080, 0.080)
        root += delta[:3]
        joint[model.leg_canonical] += delta[3:]
        joint[model.leg_canonical] = np.clip(
            joint[model.leg_canonical],
            model.leg_lower + model.leg_safety_margin,
            model.leg_upper - model.leg_safety_margin,
        )
        # A3's pelvis/foot proportions differ substantially from SMPL. A
        # side-step may require up to roughly one foot spacing of horizontal
        # pelvis correction to put COM over the planted foot.
        root = np.clip(root, raw_root - [0.24, 0.24, 0.15], raw_root + [0.24, 0.24, 0.15])
        if float(np.linalg.norm(delta)) < 2.0e-5:
            break

    return root, joint


def correct_clip(
    model: A3Kinematics,
    clip: ClipInput,
    ground_z: float,
    contact_height: float,
    contact_speed: float,
    swing_clearance: float,
    support_margin: float,
    max_iterations: int,
    correction_smoothing_frames: int,
    contact_override: np.ndarray | None = None,
) -> tuple[ClipInput, KinematicTrajectory, np.ndarray, dict[str, Any]]:
    clip = ClipInput(
        clip.fps,
        clip.root_pos.copy(),
        clip.root_quat_wxyz.copy(),
        clip.joint_pos.copy(),
    )
    raw_fk = model.trajectory(clip.root_pos, clip.root_quat_wxyz, clip.joint_pos)
    low_foot_z = float(np.percentile(raw_fk.foot_pos[:, :, 2], 5))
    clip.root_pos[:, 2] += ground_z - low_foot_z
    raw_fk = model.trajectory(clip.root_pos, clip.root_quat_wxyz, clip.joint_pos)

    if contact_override is None:
        contact = infer_contacts(
            raw_fk,
            clip.fps,
            ground_z,
            contact_height,
            contact_speed,
        )
    else:
        contact = np.asarray(contact_override, dtype=bool).copy()
        if contact.shape != (len(clip.root_pos), 2):
            raise ValueError(
                "contact_override must have shape "
                f"{(len(clip.root_pos), 2)}, got {contact.shape}"
            )
        if np.any(contact.sum(axis=1) == 0):
            raise ValueError("contact_override contains unsupported frames")
    foot_target, foot_rotation = build_foot_targets(
        raw_fk, contact, ground_z, swing_clearance
    )

    corrected_root = np.empty_like(clip.root_pos)
    corrected_joint = clip.joint_pos.copy()
    for frame in range(len(clip.root_pos)):
        if frame:
            initial_root = clip.root_pos[frame] + (
                corrected_root[frame - 1] - clip.root_pos[frame - 1]
            )
            initial_joint = clip.joint_pos[frame].copy()
            initial_joint[model.leg_canonical] += (
                corrected_joint[frame - 1, model.leg_canonical]
                - clip.joint_pos[frame - 1, model.leg_canonical]
            )
            previous_root = corrected_root[frame - 1]
            previous_joint = corrected_joint[frame - 1]
        else:
            initial_root = clip.root_pos[frame].copy()
            initial_joint = clip.joint_pos[frame].copy()
            previous_root = None
            previous_joint = None
        corrected_root[frame], corrected_joint[frame] = solve_frame(
            model,
            clip.root_pos[frame],
            clip.root_quat_wxyz[frame],
            clip.joint_pos[frame],
            initial_root,
            initial_joint,
            previous_root,
            previous_joint,
            contact[frame],
            foot_target[frame],
            foot_rotation[frame],
            support_margin,
            max_iterations,
        )

    # Jacobian IK is solved frame-by-frame, so contact-set changes can otherwise
    # introduce sharp branch changes. Filter only the IK correction, preserving
    # the original temporal motion and the unmodified upper body.
    correction_window = min(
        correction_smoothing_frames,
        len(corrected_root) if len(corrected_root) % 2 else len(corrected_root) - 1,
    )
    if correction_window % 2 == 0:
        correction_window -= 1
    if correction_window >= 5:
        root_correction = savgol_filter(
            corrected_root - clip.root_pos,
            correction_window,
            3,
            axis=0,
            mode="interp",
        )
        leg_correction = savgol_filter(
            corrected_joint[:, model.leg_canonical]
            - clip.joint_pos[:, model.leg_canonical],
            correction_window,
            3,
            axis=0,
            mode="interp",
        )
        corrected_root = clip.root_pos + root_correction
        corrected_joint[:, model.leg_canonical] = np.clip(
            clip.joint_pos[:, model.leg_canonical] + leg_correction,
            model.leg_lower + model.leg_safety_margin,
            model.leg_upper - model.leg_safety_margin,
        )

    corrected = ClipInput(
        clip.fps, corrected_root, clip.root_quat_wxyz.copy(), corrected_joint
    )
    trajectory = model.trajectory(
        corrected.root_pos, corrected.root_quat_wxyz, corrected.joint_pos
    )
    dt = 1.0 / corrected.fps
    foot_speed = np.linalg.norm(gradient(trajectory.foot_pos[:, :, :2], dt), axis=-1)
    contact_height_error = np.abs(trajectory.foot_pos[:, :, 2] - ground_z)
    stable_contact = contact.copy()
    for foot in range(2):
        for start, end, value in runs(contact[:, foot]):
            if value:
                stable_contact[start : min(end, start + 2), foot] = False
                stable_contact[max(start, end - 1) : end, foot] = False
    if not stable_contact.any():
        stable_contact = contact

    support_margins = []
    contact_errors = []
    for frame in range(len(contact)):
        points = support_points(
            contact[frame], foot_target[frame], foot_rotation[frame]
        )
        support_margins.append(
            signed_margin(trajectory.com[frame, :2], points)
        )
        contact_errors.append(
            float(
                np.max(
                    np.linalg.norm(
                        trajectory.foot_pos[frame, contact[frame]]
                        - foot_target[frame, contact[frame]],
                        axis=1,
                    )
                )
            )
        )
    support_margins_array = np.asarray(support_margins)
    leg_correction = (
        corrected.joint_pos[:, model.leg_canonical]
        - clip.joint_pos[:, model.leg_canonical]
    )
    leg_velocity = gradient(
        corrected.joint_pos[:, model.leg_canonical], dt
    )
    leg_acceleration = gradient(leg_velocity, dt)
    summary = {
        "frames": len(corrected.root_pos),
        "fps": corrected.fps,
        "left_contact_fraction": float(contact[:, 0].mean()),
        "right_contact_fraction": float(contact[:, 1].mean()),
        "double_support_fraction": float(np.all(contact, axis=1).mean()),
        "single_support_fraction": float((contact.sum(axis=1) == 1).mean()),
        "contact_foot_speed_p95_m_s": float(
            np.percentile(foot_speed[stable_contact], 95)
        ),
        "contact_foot_height_error_p95_m": float(
            np.percentile(contact_height_error[stable_contact], 95)
        ),
        "com_support_margin_p05_m": float(np.percentile(support_margins_array, 5)),
        "com_outside_support_fraction": float((support_margins_array < 0.0).mean()),
        "ik_contact_error_p95_m": float(np.percentile(contact_errors, 95)),
        "root_correction_p95_m": float(
            np.percentile(
                np.linalg.norm(corrected.root_pos - clip.root_pos, axis=1), 95
            )
        ),
        "leg_correction_rms_p95_rad": float(
            np.percentile(
                np.sqrt(np.mean(np.square(leg_correction), axis=1)), 95
            )
        ),
        "joint_limit_min_margin_rad": float(
            np.min(
                np.minimum(
                    corrected.joint_pos[:, model.leg_canonical] - model.leg_lower,
                    model.leg_upper - corrected.joint_pos[:, model.leg_canonical],
                )
            )
        ),
        "upper_body_max_abs_change_rad": float(
            np.max(
                np.abs(
                    corrected.joint_pos[:, :19]
                    - clip.joint_pos[:, :19]
                )
            )
        ),
        "leg_speed_norm_p95_rad_s": float(
            np.percentile(np.linalg.norm(leg_velocity, axis=1), 95)
        ),
        "leg_acceleration_norm_p95_rad_s2": float(
            np.percentile(np.linalg.norm(leg_acceleration, axis=1), 95)
        ),
    }
    return corrected, trajectory, contact, summary


def provisional_tier(name: str) -> str:
    if "static" in name or "short_translate" in name:
        return "tier1"
    if "long_translate_hit2" in name:
        return "tier2"
    return "quarantine"


def acceptance(summary: dict[str, Any], tier: str) -> tuple[str, list[str]]:
    reasons = []
    if summary["contact_foot_speed_p95_m_s"] > 0.060:
        reasons.append("contact_foot_speed")
    if summary["contact_foot_height_error_p95_m"] > 0.008:
        reasons.append("contact_height")
    if summary["com_outside_support_fraction"] > 0.10:
        reasons.append("com_support")
    if summary["ik_contact_error_p95_m"] > 0.012:
        reasons.append("ik_contact")
    if summary["root_correction_p95_m"] > 0.24:
        reasons.append("root_correction")
    if summary["leg_correction_rms_p95_rad"] > 0.45:
        reasons.append("leg_correction")
    if summary["joint_limit_min_margin_rad"] < 0.015:
        reasons.append("joint_limit")
    if summary["upper_body_max_abs_change_rad"] > 1.0e-8:
        reasons.append("upper_body_changed")
    if summary["leg_speed_norm_p95_rad_s"] > 5.0:
        reasons.append("leg_speed")
    if summary["leg_acceleration_norm_p95_rad_s2"] > 70.0:
        reasons.append("leg_acceleration")
    if tier == "quarantine":
        reasons.append("source_recovery_not_settled")
    return ("pass" if not reasons else "review"), reasons


def quality_score(summary: dict[str, Any], tier: str) -> float:
    status, reasons = acceptance(summary, tier)
    ratios = (
        summary["contact_foot_speed_p95_m_s"] / 0.060,
        summary["contact_foot_height_error_p95_m"] / 0.008,
        summary["com_outside_support_fraction"] / 0.10,
        summary["ik_contact_error_p95_m"] / 0.012,
        summary["root_correction_p95_m"] / 0.24,
        summary["leg_correction_rms_p95_rad"] / 0.45,
        summary["leg_speed_norm_p95_rad_s"] / 5.0,
        summary["leg_acceleration_norm_p95_rad_s2"] / 70.0,
    )
    return (0.0 if status == "pass" else 100.0 * len(reasons)) + float(sum(ratios))


def write_npz(
    path: Path,
    clip: ClipInput,
    trajectory: KinematicTrajectory,
    contact: np.ndarray,
    strike_frame: int,
) -> None:
    dt = 1.0 / clip.fps
    np.savez(
        path,
        fps=np.float32(clip.fps),
        joint_pos=clip.joint_pos.astype(np.float32),
        joint_vel=gradient(clip.joint_pos, dt).astype(np.float32),
        body_pos_w=trajectory.body_pos.astype(np.float32),
        body_quat_w=trajectory.body_quat.astype(np.float32),
        body_lin_vel_w=gradient(trajectory.body_pos, dt).astype(np.float32),
        body_ang_vel_w=angular_velocity(trajectory.body_quat, dt).astype(np.float32),
        left_foot_contact=contact[:, 0].astype(np.uint8),
        right_foot_contact=contact[:, 1].astype(np.uint8),
        strike_frame=np.int32(strike_frame),
    )


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, rows = load_manifest(args.manifest)
    if args.only:
        requested = set(args.only)
        rows = [row for row in rows if row["name"] in requested]
        missing = requested - {row["name"] for row in rows}
        if missing:
            raise ValueError(f"unknown --only clips: {sorted(missing)}")
    if not rows:
        raise RuntimeError("no input motion rows")

    joint_names = load_joint_names(args.joint_order)
    model = A3Kinematics(args.robot_xml, joint_names)
    output_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        source = Path(row["output"]).resolve()
        source_clip = load_gmr(source)
        source_strike_frame = int(row["strike_frame"])
        cropped_source, cropped_strike_frame, crop_start, crop_end = crop_around_strike(
            source_clip,
            source_strike_frame,
            args.pre_strike_s,
            args.post_strike_s,
        )
        clip = resample_clip(cropped_source, args.output_fps)
        smooth_root_and_legs(clip, model.leg_canonical)
        canonicalize_root(clip, model, args.ready_frames_s)
        tier = provisional_tier(row["name"])
        smoothing_windows = [args.correction_smoothing_frames]
        if args.adaptive_smoothing:
            smoothing_windows.extend((7, 9, 15))
        candidates = []
        for smoothing_window in sorted(set(smoothing_windows)):
            result = correct_clip(
                model,
                clip,
                args.ground_z,
                args.contact_height_m,
                args.contact_speed_m_s,
                args.swing_clearance_m,
                args.support_margin_m,
                args.max_iterations,
                smoothing_window,
            )
            candidates.append(
                (quality_score(result[3], tier), smoothing_window, result)
            )
        _, selected_smoothing_window, selected = min(
            candidates, key=lambda item: item[0]
        )
        corrected, trajectory, contact, summary = selected
        strike_time = cropped_strike_frame / cropped_source.fps
        strike_frame = int(
            np.clip(round(strike_time * corrected.fps), 0, len(corrected.root_pos) - 1)
        )
        strike_phase = (
            strike_frame / (len(corrected.root_pos) - 1)
            if len(corrected.root_pos) > 1
            else 0.0
        )
        destination = args.output_dir / f"{row['name']}_contact_aware.npz"
        write_npz(destination, corrected, trajectory, contact, strike_frame)

        status, reasons = acceptance(summary, tier)
        sidecar = {
            "name": row["name"],
            "source_motion": str(source),
            "source_video": row.get("source_video"),
            "source_crop_frames": [crop_start, crop_end],
            "fps": corrected.fps,
            "frame_count": len(corrected.root_pos),
            "strike_frame": strike_frame,
            "strike_phase": strike_phase,
            "swing_side": int(row["swing_side"]),
            "motion_type": row["motion_type"],
            "train_tier": tier,
            "acceptance": status,
            "review_reasons": reasons,
            "retarget": {
                "method": "a3_contact_aware_root_leg_jacobian_ik_v1",
                "ground_site": ["left_foot", "right_foot"],
                "ground_z": args.ground_z,
                "output_fps": args.output_fps,
                "pre_strike_s": args.pre_strike_s,
                "post_strike_s": args.post_strike_s,
                "correction_smoothing_frames": selected_smoothing_window,
                "upper_body_preserved": True,
            },
            "metrics": summary,
        }
        with destination.with_suffix(".yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(sidecar, handle, sort_keys=False, allow_unicode=False)

        output_row: dict[str, Any] = dict(row)
        output_row.update(
            {
                "source": str(source),
                "output": str(destination.resolve()),
                "frames": len(corrected.root_pos),
                "fps": corrected.fps,
                "strike_frame": strike_frame,
                "strike_phase": f"{strike_phase:.9f}",
                "train_tier": tier,
                "acceptance": status,
                "review_reasons": ",".join(reasons),
                "source_crop_start": crop_start,
                "source_crop_end_exclusive": crop_end,
            }
        )
        output_row.update(
            {
                key: f"{value:.9f}"
                for key, value in summary.items()
                if key not in ("frames", "fps")
            }
        )
        output_rows.append(output_row)
        summaries.append({"name": row["name"], "tier": tier, "status": status, **summary})
        print(
            f"{row['name']}: {status} tier={tier} "
            f"slip_p95={summary['contact_foot_speed_p95_m_s']:.4f} "
            f"com_out={summary['com_outside_support_fraction']:.3f} "
            f"ik_p95={summary['ik_contact_error_p95_m']:.4f}"
        )

    fields = list(output_rows[0])
    manifest = args.output_dir / "manifest.tsv"
    write_tsv(manifest, fields, output_rows)
    ready_rows = [
        row
        for row in output_rows
        if row["acceptance"] == "pass" and row["train_tier"] in ("tier1", "tier2")
    ]
    write_tsv(args.output_dir / "manifest_train_ready.tsv", fields, ready_rows)
    with (args.output_dir / "retarget_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"manifest={manifest}")
    print(f"train_ready={len(ready_rows)}/{len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
