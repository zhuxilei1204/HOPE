#!/usr/bin/env python3
"""Audit planner/station targets against A3 kinematic and support constraints.

This is a read-only offline diagnostic.  It does not load a policy or alter a
training task.  Two support postures are tested:

* ``ready_upright``: the shared deploy-ready stand, with the pelvis placed at
  the dynamic station and only waist/right-arm joints available to IK.
* ``motion_strike``: the selected motion's strike-frame root/lower-body pose,
  translated to the dynamic station, with only waist/right-arm joints available
  to IK.

The first answers whether the target inherently requires a lean/leg-limit
workaround.  The second answers whether the target is compatible with the
motion prior that is active at strike time.
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
from scipy.optimize import least_squares
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
DEFAULT_ADAPTER = ROOT / "a3_deploy/a3_deploy_example/config/action_adapter.yaml"
DEFAULT_JOINT_ORDER = ROOT / "hope_training/config/joint_order_agibot_a3.yaml"
DEFAULT_PLANNER_CSV = (
    ROOT
    / "analysis/B17996_explicit_recovery_v1_20260726/"
    "eval_model_22494_mujoco_xhit_compare_20260726/mujoco_xhit0p20/"
    "model_22494_realplanner_xhit_0p20_continuous40_contact_diag.csv"
)

RIGHT_CHAIN = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LEG_JOINTS = (
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
TABLE_AUDIT_BODIES = (
    "torso_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--joint-order", type=Path, default=DEFAULT_JOINT_ORDER)
    parser.add_argument("--planner-csv", type=Path, default=DEFAULT_PLANNER_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-clip", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--table-near-x", type=float, default=0.50)
    parser.add_argument("--table-height", type=float, default=0.76)
    parser.add_argument("--table-length", type=float, default=2.74)
    parser.add_argument("--table-width", type=float, default=1.525)
    parser.add_argument("--station-clip-x", nargs=2, type=float, default=(-0.25, 0.25))
    parser.add_argument("--station-clip-y", nargs=2, type=float, default=(-0.44, 0.44))
    return parser.parse_args()


def f(value: str | float | None, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1.0e-12)


def quat_to_rpy_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)


def signed_polygon_margin(point: np.ndarray, polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return math.nan
    hull = polygon[ConvexHull(polygon).vertices]
    center = hull.mean(axis=0)
    margins: list[float] = []
    for idx, p0 in enumerate(hull):
        p1 = hull[(idx + 1) % len(hull)]
        edge = p1 - p0
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        if float(np.dot(normal, center - p0)) < 0.0:
            normal *= -1.0
        margins.append(float(np.dot(normal, point - p0)))
    return min(margins)


@dataclass
class Clip:
    index: int
    name: str
    path: Path
    side: float
    strike_frame: int
    ranges: np.ndarray
    racket_offset_xy: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray
    motion_normal: np.ndarray | None = None


@dataclass
class Target:
    source: str
    clip: Clip
    sample_id: int
    position: np.ndarray
    normal: np.ndarray
    station: np.ndarray
    planner_trial: int | None = None


class A3Kinematics:
    def __init__(
        self,
        xml_path: Path,
        canonical_names: list[str],
        default_q_by_name: dict[str, float],
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.base_qadr = int(
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint")
            ]
        )
        self.joint_qadr: dict[str, int] = {}
        for name in canonical_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo model is missing canonical joint {name!r}")
            self.joint_qadr[name] = int(self.model.jnt_qposadr[joint_id])
        for name, value in default_q_by_name.items():
            self.data.qpos[self.joint_qadr[name]] = float(value)
        mujoco.mj_forward(self.model, self.data)
        self.default_qpos = self.data.qpos.copy()
        self.default_root = self.default_qpos[self.base_qadr : self.base_qadr + 7].copy()
        self.default_joint = {
            name: float(self.default_qpos[address]) for name, address in self.joint_qadr.items()
        }
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_racket"
        )
        self.body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in TABLE_AUDIT_BODIES
        }
        self.left_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_Link"
        )
        self.right_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_Link"
        )
        excluded = {
            bid
            for bid in range(self.model.nbody)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "")
            in {"world"}
        }
        self.mass_ids = np.array(
            [
                bid
                for bid in range(1, self.model.nbody)
                if bid not in excluded and float(self.model.body_mass[bid]) > 0.0
            ],
            dtype=np.int32,
        )
        self.chain_names = list(RIGHT_CHAIN)
        self.chain_qadr = np.array([self.joint_qadr[name] for name in self.chain_names])
        self.chain_lo = np.empty(len(self.chain_names))
        self.chain_hi = np.empty(len(self.chain_names))
        for idx, name in enumerate(self.chain_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.chain_lo[idx], self.chain_hi[idx] = self.model.jnt_range[jid]

    def set_pose(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        q_by_name: dict[str, float],
    ) -> None:
        self.data.qpos[:] = self.default_qpos
        self.data.qpos[self.base_qadr : self.base_qadr + 3] = root_pos
        self.data.qpos[self.base_qadr + 3 : self.base_qadr + 7] = unit(root_quat)
        for name, value in q_by_name.items():
            self.data.qpos[self.joint_qadr[name]] = float(value)
        mujoco.mj_forward(self.model, self.data)

    def racket_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def solve_chain(
        self,
        target_pos: np.ndarray,
        target_normal: np.ndarray,
        normal_sign: float,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        support_q: dict[str, float],
        seed_q: dict[str, float],
    ) -> dict[str, float | np.ndarray | bool]:
        q0 = np.array([seed_q.get(name, self.default_joint[name]) for name in self.chain_names])
        q0 = np.clip(q0, self.chain_lo + 1.0e-6, self.chain_hi - 1.0e-6)
        normal = unit(target_normal)

        def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q = dict(support_q)
            q.update(dict(zip(self.chain_names, values)))
            self.set_pose(root_pos, root_quat, q)
            pos, rot = self.racket_pose()
            # Training uses local +Y and the target normal already contains the
            # forehand/backhand face sign.
            actual_normal = rot[:, 1] * float(normal_sign)
            return pos, actual_normal

        def residual(values: np.ndarray) -> np.ndarray:
            pos, actual_normal = evaluate(values)
            position = (pos - target_pos) / 0.025
            normal_cross = np.cross(actual_normal, normal) / 0.20
            regularization = (values - q0) * 0.035
            span = np.maximum(self.chain_hi - self.chain_lo, 1.0e-6)
            lower_margin = (values - self.chain_lo) / span
            upper_margin = (self.chain_hi - values) / span
            normalized_margin = np.minimum(lower_margin, upper_margin)
            # The chain is redundant.  Prefer an equally accurate solution with
            # at least 5% hard-limit margin instead of letting least-squares land
            # on a waist/wrist bound for numerical convenience.
            limit_barrier = np.maximum(0.0, 0.05 - normalized_margin) * 10.0
            return np.concatenate((position, normal_cross, regularization, limit_barrier))

        seeds = [
            q0,
            np.array([self.default_joint[name] for name in self.chain_names]),
        ]
        best = None
        for seed in seeds:
            result = least_squares(
                residual,
                np.clip(seed, self.chain_lo + 1.0e-6, self.chain_hi - 1.0e-6),
                bounds=(self.chain_lo, self.chain_hi),
                max_nfev=180,
                ftol=1.0e-8,
                xtol=1.0e-8,
                gtol=1.0e-8,
            )
            pos, actual_normal = evaluate(result.x)
            pos_error = float(np.linalg.norm(pos - target_pos))
            normal_angle = math.degrees(
                math.acos(float(np.clip(np.dot(actual_normal, normal), -1.0, 1.0)))
            )
            score = pos_error + 0.002 * normal_angle
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "q": result.x.copy(),
                    "position_error_m": pos_error,
                    "normal_error_deg": normal_angle,
                    "solver_success": bool(result.success),
                }
        assert best is not None
        q = dict(support_q)
        q.update(dict(zip(self.chain_names, best["q"])))
        self.set_pose(root_pos, root_quat, q)
        diagnostics = self.posture_diagnostics(q)
        best.update(diagnostics)
        best["position_reachable"] = bool(best["position_error_m"] <= 0.035)
        best["pose_reachable"] = bool(
            best["position_error_m"] <= 0.035 and best["normal_error_deg"] <= 20.0
        )
        return best

    def posture_diagnostics(self, q_by_name: dict[str, float]) -> dict[str, float | str]:
        masses = self.model.body_mass[self.mass_ids]
        com = np.sum(self.data.xipos[self.mass_ids] * masses[:, None], axis=0) / masses.sum()
        corners: list[np.ndarray] = []
        for body_id in (self.left_foot_id, self.right_foot_id):
            center = self.data.xpos[body_id, :2]
            rotation = self.data.xmat[body_id].reshape(3, 3)
            for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                corners.append(
                    center
                    + sx * 0.105 * rotation[:2, 0]
                    + sy * 0.050 * rotation[:2, 1]
                )
        support_margin = signed_polygon_margin(com[:2], np.asarray(corners))
        def group_margin(names: Iterable[str]) -> tuple[float, str]:
            min_margin = 1.0
            min_joint = ""
            for name in names:
                value = q_by_name[name]
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0 or not self.model.jnt_limited[jid]:
                    continue
                lo, hi = self.model.jnt_range[jid]
                margin = min((value - lo) / (hi - lo), (hi - value) / (hi - lo))
                if margin < min_margin:
                    min_margin = float(margin)
                    min_joint = name
            return min_margin, min_joint

        min_margin, min_joint = group_margin(q_by_name)
        chain_margin, chain_joint = group_margin(RIGHT_CHAIN)
        leg_margin, leg_joint = group_margin(LEG_JOINTS)
        ankle_margin, ankle_joint = group_margin(
            name for name in LEG_JOINTS if "ankle" in name
        )
        table_collision = False
        min_table_clearance = math.inf
        for body_id in self.body_ids.values():
            point = self.data.xpos[body_id]
            in_xy = (
                0.46 <= point[0] <= 0.50 + 2.74 + 0.04
                and abs(point[1]) <= 1.525 / 2.0 + 0.04
            )
            if in_xy:
                clearance = float(point[2] - 0.76)
                min_table_clearance = min(min_table_clearance, clearance)
                table_collision |= -0.08 <= clearance <= 0.04
        if not math.isfinite(min_table_clearance):
            min_table_clearance = math.nan
        return {
            "com_support_margin_m": support_margin,
            "min_joint_margin_fraction": min_margin,
            "min_margin_joint": min_joint,
            "min_right_chain_margin_fraction": chain_margin,
            "min_right_chain_margin_joint": chain_joint,
            "min_leg_margin_fraction": leg_margin,
            "min_leg_margin_joint": leg_joint,
            "min_ankle_margin_fraction": ankle_margin,
            "min_ankle_margin_joint": ankle_joint,
            "table_zone_collision": bool(table_collision),
            "min_upper_body_table_clearance_m": min_table_clearance,
        }


def load_clips(
    manifest_path: Path,
    joint_names: list[str],
) -> list[Clip]:
    clips: list[Clip] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            path = Path(row["output"]).resolve()
            data = np.load(path)
            strike = int(row.get("strike_frame") or 60)
            ranges = np.array(
                [
                    [f(row[f"racket_pos_{axis}_lo"]), f(row[f"racket_pos_{axis}_hi"])]
                    for axis in "xyz"
                ],
                dtype=np.float64,
            )
            q = np.asarray(data["joint_pos"][strike], dtype=np.float64)
            clips.append(
                Clip(
                    index=idx,
                    name=row["name"],
                    path=path,
                    side=f(row.get("swing_side"), 1.0),
                    strike_frame=strike,
                    ranges=ranges,
                    racket_offset_xy=np.array(
                        [f(row["motion_racket_offset_x"]), f(row["motion_racket_offset_y"])],
                        dtype=np.float64,
                    ),
                    root_pos=np.asarray(data["body_pos_w"][strike, 0], dtype=np.float64),
                    root_quat=np.asarray(data["body_quat_w"][strike, 0], dtype=np.float64),
                    joint_pos=q,
                )
            )
    if not clips:
        raise ValueError(f"No clips in {manifest_path}")
    return clips


def clip_q(clip: Clip, names: list[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, clip.joint_pos)}


def dynamic_station(
    position: np.ndarray,
    offset_xy: np.ndarray,
    fixed_station: np.ndarray,
    clip_x: tuple[float, float],
    clip_y: tuple[float, float],
) -> np.ndarray:
    rel = position[:2] - offset_xy - fixed_station
    rel[0] = np.clip(rel[0], min(clip_x), max(clip_x))
    rel[1] = np.clip(rel[1], min(clip_y), max(clip_y))
    return fixed_station + rel


def nearest_clip(position: np.ndarray, side: float, clips: list[Clip]) -> Clip:
    candidates = [clip for clip in clips if (clip.side >= 0.0) == (side >= 0.0)]
    best = None
    for clip in candidates:
        center = clip.ranges.mean(axis=1)
        scale = np.maximum(np.ptp(clip.ranges, axis=1), 0.05)
        # x may be forced to a planner plane in eval; classify mostly from y/z.
        score = float(np.linalg.norm((position[1:] - center[1:]) / scale[1:]))
        if best is None or score < best[0]:
            best = (score, clip)
    assert best is not None
    return best[1]


def make_targets(
    clips: list[Clip],
    kin: A3Kinematics,
    joint_names: list[str],
    planner_csv: Path,
    samples_per_clip: int,
    rng: np.random.Generator,
    station_clip_x: tuple[float, float],
    station_clip_y: tuple[float, float],
) -> list[Target]:
    for clip in clips:
        q = clip_q(clip, joint_names)
        kin.set_pose(clip.root_pos, clip.root_quat, q)
        _, rotation = kin.racket_pose()
        clip.motion_normal = unit(rotation[:, 1] * clip.side)

    targets: list[Target] = []
    fixed_training = np.zeros(2, dtype=np.float64)
    for clip in clips:
        points = [clip.ranges.mean(axis=1)]
        for bits in range(8):
            points.append(
                np.array(
                    [clip.ranges[axis, (bits >> axis) & 1] for axis in range(3)],
                    dtype=np.float64,
                )
            )
        while len(points) < samples_per_clip:
            points.append(rng.uniform(clip.ranges[:, 0], clip.ranges[:, 1]))
        for sample_id, position in enumerate(points[:samples_per_clip]):
            targets.append(
                Target(
                    source="training_box",
                    clip=clip,
                    sample_id=sample_id,
                    position=position,
                    normal=np.asarray(clip.motion_normal),
                    station=dynamic_station(
                        position,
                        clip.racket_offset_xy,
                        fixed_training,
                        station_clip_x,
                        station_clip_y,
                    ),
                )
            )

    if planner_csv.is_file():
        fixed_mujoco = kin.default_root[:2].copy()
        with planner_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                position = np.array(
                    [f(row.get(f"target_pos_{axis}")) for axis in "xyz"], dtype=np.float64
                )
                normal = np.array(
                    [f(row.get(f"target_normal_{axis}")) for axis in "xyz"], dtype=np.float64
                )
                if not np.all(np.isfinite(position)) or not np.all(np.isfinite(normal)):
                    continue
                side = 1.0 if row.get("side") == "forehand" else -1.0
                clip = nearest_clip(position, side, clips)
                targets.append(
                    Target(
                        source="real_planner_contact_command",
                        clip=clip,
                        sample_id=int(row.get("trial") or -1),
                        position=position,
                        normal=unit(normal),
                        station=dynamic_station(
                            position,
                            clip.racket_offset_xy,
                            fixed_mujoco,
                            station_clip_x,
                            station_clip_y,
                        ),
                        planner_trial=int(row.get("trial") or -1),
                    )
                )
    return targets


def row_from_result(target: Target, posture: str, result: dict) -> dict:
    root_quat = result.pop("_root_quat")
    root_pos = result.pop("_root_pos")
    rpy = quat_to_rpy_deg(root_quat)
    normal_error = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(unit(target.normal), unit(target.clip.motion_normal)),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    box_violation = np.maximum(
        np.maximum(target.clip.ranges[:, 0] - target.position, 0.0),
        np.maximum(target.position - target.clip.ranges[:, 1], 0.0),
    )
    return {
        "source": target.source,
        "clip_index": target.clip.index,
        "clip_name": target.clip.name,
        "side": "forehand" if target.clip.side >= 0.0 else "backhand",
        "sample_id": target.sample_id,
        "planner_trial": target.planner_trial,
        "posture": posture,
        "target_x": float(target.position[0]),
        "target_y": float(target.position[1]),
        "target_z": float(target.position[2]),
        "station_x": float(target.station[0]),
        "station_y": float(target.station[1]),
        "target_rel_station_x": float(target.position[0] - target.station[0]),
        "target_rel_station_y": float(target.position[1] - target.station[1]),
        "planner_vs_motion_normal_deg": normal_error,
        "target_box_violation_x_m": float(box_violation[0]),
        "target_box_violation_y_m": float(box_violation[1]),
        "target_box_violation_z_m": float(box_violation[2]),
        "root_z": float(root_pos[2]),
        "root_roll_deg": float(rpy[0]),
        "root_pitch_deg": float(rpy[1]),
        **{key: value for key, value in result.items() if key not in {"q", "score"}},
    }


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['source']}|{row['posture']}|{row['side']}"
        groups.setdefault(key, []).append(row)
    output = {}
    for key, values in sorted(groups.items()):
        def arr(name: str) -> np.ndarray:
            return np.array([f(value.get(name)) for value in values], dtype=np.float64)

        output[key] = {
            "count": len(values),
            "position_reachable_rate": float(np.mean(arr("position_reachable"))),
            "pose_reachable_rate": float(np.mean(arr("pose_reachable"))),
            "position_error_p50_m": float(np.nanpercentile(arr("position_error_m"), 50)),
            "position_error_p95_m": float(np.nanpercentile(arr("position_error_m"), 95)),
            "normal_error_p50_deg": float(np.nanpercentile(arr("normal_error_deg"), 50)),
            "normal_error_p95_deg": float(np.nanpercentile(arr("normal_error_deg"), 95)),
            "com_support_margin_p05_m": float(np.nanpercentile(arr("com_support_margin_m"), 5)),
            "com_outside_support_rate": float(np.mean(arr("com_support_margin_m") < 0.0)),
            "joint_margin_p05": float(np.nanpercentile(arr("min_joint_margin_fraction"), 5)),
            "right_chain_margin_p05": float(
                np.nanpercentile(arr("min_right_chain_margin_fraction"), 5)
            ),
            "leg_margin_p05": float(np.nanpercentile(arr("min_leg_margin_fraction"), 5)),
            "ankle_margin_p05": float(np.nanpercentile(arr("min_ankle_margin_fraction"), 5)),
            "planner_vs_motion_normal_p50_deg": float(
                np.nanpercentile(arr("planner_vs_motion_normal_deg"), 50)
            ),
            "planner_vs_motion_normal_p95_deg": float(
                np.nanpercentile(arr("planner_vs_motion_normal_deg"), 95)
            ),
            "outside_motion_box_rate": float(
                np.mean(
                    np.maximum.reduce(
                        (
                            arr("target_box_violation_x_m"),
                            arr("target_box_violation_y_m"),
                            arr("target_box_violation_z_m"),
                        )
                    )
                    > 1.0e-6
                )
            ),
            "table_zone_collision_rate": float(np.mean(arr("table_zone_collision"))),
            "target_rel_station_x_range_m": [
                float(np.nanmin(arr("target_rel_station_x"))),
                float(np.nanmax(arr("target_rel_station_x"))),
            ],
            "target_rel_station_y_range_m": [
                float(np.nanmin(arr("target_rel_station_y"))),
                float(np.nanmax(arr("target_rel_station_y"))),
            ],
        }
    return output


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joint_names = list(yaml.safe_load(args.joint_order.read_text())["joint_order"])
    adapter = yaml.safe_load(args.adapter.read_text())
    default_q = {name: float(value) for name, value in adapter["default_q"].items()}
    kin = A3Kinematics(args.robot_xml, joint_names, default_q)
    clips = load_clips(args.manifest, joint_names)
    rng = np.random.default_rng(args.seed)
    targets = make_targets(
        clips,
        kin,
        joint_names,
        args.planner_csv,
        args.samples_per_clip,
        rng,
        tuple(args.station_clip_x),
        tuple(args.station_clip_y),
    )

    rows: list[dict] = []
    default_support = {name: default_q[name] for name in joint_names if name not in RIGHT_CHAIN}
    for target_index, target in enumerate(targets):
        motion_q = clip_q(target.clip, joint_names)
        postures = (
            (
                "ready_upright",
                np.array([target.station[0], target.station[1], kin.default_root[2]]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                default_support,
                motion_q,
            ),
            (
                "motion_strike",
                np.array([target.station[0], target.station[1], target.clip.root_pos[2]]),
                target.clip.root_quat,
                {name: motion_q[name] for name in joint_names if name not in RIGHT_CHAIN},
                motion_q,
            ),
        )
        for posture, root_pos, root_quat, support, seed in postures:
            result = kin.solve_chain(
                target.position,
                target.normal,
                target.clip.side,
                root_pos,
                root_quat,
                support,
                seed,
            )
            result["_root_pos"] = root_pos
            result["_root_quat"] = root_quat
            rows.append(row_from_result(target, posture, result))
        if (target_index + 1) % 50 == 0:
            print(f"[planner-audit] solved {target_index + 1}/{len(targets)} targets", flush=True)

    write_csv(args.output_dir / "planner_reachability_rows.csv", rows)
    summary = {
        "inputs": {
            "robot_xml": str(args.robot_xml.resolve()),
            "manifest": str(args.manifest.resolve()),
            "planner_csv": str(args.planner_csv.resolve()),
            "samples_per_clip": args.samples_per_clip,
            "table_near_x_training": args.table_near_x,
            "dynamic_station_clip_x": list(args.station_clip_x),
            "dynamic_station_clip_y": list(args.station_clip_y),
        },
        "thresholds": {
            "position_reachable_m": 0.035,
            "pose_normal_error_deg": 20.0,
            "support_margin_safe": ">= 0 m",
        },
        "groups": summarize(rows),
    }
    (args.output_dir / "planner_reachability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
