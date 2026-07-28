# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo no-ball standing and no-ball swing-cycle diagnostic for exported HOPE ONNX policies.

This script intentionally reuses the reference deploy observation builder, ONNX
policy wrapper, ActionAdapter, lifecycle and MuJoCo scene used by
``mujoco_eval_onnx.py``.  It answers a narrow question: if no real ball task is
present, does the policy keep the A3 standing from the same reset state used by
MuJoCo eval?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys

import numpy as np


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "a3_deploy").is_dir() and (parent / "hope_training").is_dir():
            return parent
    return here.parents[3]


def _reference_pkg_dir(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "a3_deploy" / "a3_deploy_example" / "reference"


def _default_runtime_config(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "a3_deploy" / "a3_deploy_example" / "config" / "hope_pingpong_runtime.yaml"


def _load_success_metric(repo_root: pathlib.Path):
    import importlib.util

    path = (
        repo_root
        / "hope_training"
        / "whole_body_tracking"
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "utils"
        / "success_metric.py"
    )
    spec = importlib.util.spec_from_file_location("hope_success_metric", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quat_to_euler_wxyz(q: np.ndarray) -> tuple[float, float, float]:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1.0e-12:
        return 0.0, 0.0, 0.0
    w, x, y, z = q / n
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _rotation_projected_gravity_body(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1.0e-12:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    w = q[0] / n
    xyz = q[1:4] / n
    v = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(xyz, v) * (2.0 * w)
    c = xyz * (2.0 * float(np.dot(xyz, v)))
    return a - b + c


def _load_serve_manifest(path: str | None) -> list[dict]:
    if not path:
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ranges = []
            ok = True
            for axis in ("x", "y", "z"):
                lo = row.get(f"racket_pos_{axis}_lo")
                hi = row.get(f"racket_pos_{axis}_hi")
                if lo in (None, "") or hi in (None, ""):
                    ok = False
                    break
                lo_f = float(lo)
                hi_f = float(hi)
                ranges.append((min(lo_f, hi_f), max(lo_f, hi_f)))
            side = row.get("swing_side")
            if not ok or side in (None, ""):
                continue
            rows.append(
                {
                    "side": 1.0 if float(side) >= 0.0 else -1.0,
                    "ranges": tuple(ranges),
                    "source": row.get("output") or row.get("source") or "",
                }
            )
    return rows


def _make_no_ball_command(rng, rows: list[dict], side: float, RacketCommand, task_id: int, revision: int):
    candidates = [r for r in rows if (r["side"] >= 0.0) == (side >= 0.0)]
    if not candidates:
        candidates = rows
    if candidates:
        row = candidates[int(rng.integers(0, len(candidates)))]
        pos = np.array([rng.uniform(lo, hi) for lo, hi in row["ranges"]], dtype=np.float64)
    else:
        y = -0.28 if side >= 0 else 0.22
        pos = np.array([0.30, y, 0.92], dtype=np.float64)
    # A mild forward/up target is enough to trigger the trained swing without any real ball impulse.
    vel = np.array([1.6, 0.0, 1.1], dtype=np.float64)
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return RacketCommand(
        task_id=task_id,
        task_revision=revision,
        swing_side=side,
        position=pos,
        velocity=vel,
        time_to_strike=0.45,
        target_normal=normal,
    )


def run(args) -> dict:
    repo_root = _repo_root()
    if args.record_video:
        os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    ref_dir = pathlib.Path(args.reference_dir) if args.reference_dir else _reference_pkg_dir(repo_root)
    sys.path.insert(0, str(ref_dir))
    from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
    from a3_deploy_onnx_ref_pingpong.joint_order import HEAD_INDICES, JOINT_NAMES
    from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle
    from a3_deploy_onnx_ref_pingpong.observation import (
        OBS_DIM_NORMAL114,
        OBS_DIM_STABILITY122,
        build_observation,
        build_observation_normal114,
        build_observation_stability122,
    )
    from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy
    from a3_deploy_onnx_ref_pingpong.racket_command import BACKHAND, FOREHAND, QueueRacketCommandSource, RacketCommand

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mujoco_eval_onnx import _MujocoVideoRecorder
    from mujoco_pingpong_scene import PingPongRealPhysicsScene

    metric = _load_success_metric(repo_root)
    load_ball_physics_config = metric.load_ball_physics_config

    runtime_cfg = RuntimeConfig.load(args.runtime_config or _default_runtime_config(repo_root))
    robot_xml = args.model_xml or str(runtime_cfg.model_xml_path)
    ball_cfg = load_ball_physics_config()
    policy = OnnxPolicy(args.onnx or str(runtime_cfg.onnx_path))
    feedback_mode = getattr(args, "last_action_feedback_mode", "auto")
    if feedback_mode == "auto":
        feedback_mode = policy.last_action_feedback_mode or runtime_cfg.last_action_feedback_mode
    scene = PingPongRealPhysicsScene(
        robot_xml,
        ball_cfg,
        JOINT_NAMES,
        control_dt=runtime_cfg.control_dt,
        near_edge_x=args.near_edge_x,
        launch_viewer=args.view,
    )
    recorder = (
        _MujocoVideoRecorder(
            scene,
            args.record_video,
            width=args.video_width,
            height=args.video_height,
            fps=args.video_fps,
        )
        if args.record_video
        else None
    )

    default_q = runtime_cfg.action_adapter.default_q.copy()
    kp = runtime_cfg.sim_kp.copy() * float(args.kp_scale)
    kd = runtime_cfg.sim_kd.copy() * float(args.kd_scale)
    head_idx = list(HEAD_INDICES)
    dt = runtime_cfg.control_dt
    ticks = max(1, int(round(args.seconds / dt)))
    rng = np.random.default_rng(args.seed)
    rows = _load_serve_manifest(args.serve_manifest)

    scene.reset_stand()
    scene.set_ball(
        [
            scene.near_edge_x + scene.length + 1.0,
            scene.table_center_y,
            scene.table_surface_z + 1.0,
        ],
        [0.0, 0.0, 0.0],
    )
    lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
    source = QueueRacketCommandSource()
    last_action = np.zeros(31, dtype=np.float64)
    fixed_station_xy = scene.base_pos_w()[:2].copy()

    csv_fh = None
    writer = None
    if args.trace_csv:
        trace_path = pathlib.Path(args.trace_csv)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fh = trace_path.open("w", encoding="utf-8", newline="")
        joint_trace_fields = []
        if args.trace_joints:
            for name in JOINT_NAMES:
                joint_trace_fields.extend([f"q_{name}", f"qd_{name}", f"qdes_{name}", f"raw_{name}"])
        fields = [
            "tick",
            "time_s",
            "phase",
            "base_x",
            "base_y",
            "base_z",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "proj_grav_x",
            "proj_grav_y",
            "proj_grav_xy_norm",
            "base_ang_vel_norm",
            "base_lin_vel_xy_norm",
            "racket_speed",
            "raw_action_l2",
            "raw_action_max",
            "q_des_l2_from_default",
            "leg_q_des_l2_from_default",
            "floor_contact_count",
            "floor_contact_min_dist",
            "fallen_low",
            "fallen_tilt",
        ] + joint_trace_fields
        writer = csv.DictWriter(csv_fh, fieldnames=fields)
        writer.writeheader()

    def _floor_contact_summary():
        vals = []
        for ci in range(scene.data.ncon):
            con = scene.data.contact[ci]
            g1 = scene._mj.mj_id2name(scene.model, scene._mj.mjtObj.mjOBJ_GEOM, int(con.geom1)) or ""
            g2 = scene._mj.mj_id2name(scene.model, scene._mj.mjtObj.mjOBJ_GEOM, int(con.geom2)) or ""
            if g1 == "floor" or g2 == "floor":
                vals.append(float(con.dist))
        return len(vals), (min(vals) if vals else None)

    task_id = 0
    side = FOREHAND
    first_low_tick = None
    first_tilt_tick = None
    max_abs_pitch = 0.0
    max_abs_roll = 0.0
    max_proj_xy = 0.0
    max_base_ang_vel = 0.0
    max_base_lin_xy = 0.0
    max_racket_speed = 0.0
    max_raw_action = 0.0
    final_row = {}

    for tick in range(ticks):
        if args.mode == "fake-cycle" and tick % max(1, int(round(args.cycle_period_s / dt))) == 0:
            task_id += 1
            if args.side_mode == "mixed":
                side = FOREHAND if (task_id % 4) < 2 else BACKHAND
            elif args.side_mode == "forehand":
                side = FOREHAND
            else:
                side = BACKHAND
            source.submit(_make_no_ball_command(rng, rows, side, RacketCommand, task_id, 0))

        state = scene.read_robot_state()
        target = lifecycle.update(source.poll(), state)
        station_xy = fixed_station_xy
        if getattr(policy, "obs_dim", 111) == OBS_DIM_STABILITY122:
            obs = build_observation_stability122(
                state, target, last_action, default_q, station_xy
            )
        elif getattr(policy, "obs_dim", 111) == OBS_DIM_NORMAL114:
            obs = build_observation_normal114(state, target, last_action, default_q, station_xy)
        else:
            obs = build_observation(state, target, last_action, default_q, station_xy)
        if args.controller == "policy":
            raw_action = policy.infer(obs)
        else:
            raw_action = np.zeros(31, dtype=np.float32)
        applied_action = np.asarray(raw_action, dtype=np.float64).reshape(31).copy()
        if runtime_cfg.passive_neck:
            applied_action[head_idx] = 0.0
        q_des = runtime_cfg.action_adapter.decode(applied_action)
        if runtime_cfg.passive_neck:
            q_des[head_idx] = default_q[head_idx]
        scene.write_targets(q_des, kp, kd)
        scene.step()
        if feedback_mode == "effective":
            last_action = runtime_cfg.action_adapter.encode_effective(q_des)
            if runtime_cfg.passive_neck:
                last_action[head_idx] = 0.0
        else:
            last_action = applied_action
        if recorder is not None:
            recorder.capture(scene)

        state_post = scene.read_robot_state()
        base_q = np.asarray(state_post.base_quat_w, dtype=np.float64)
        roll, pitch, yaw = _quat_to_euler_wxyz(base_q)
        proj = _rotation_projected_gravity_body(base_q)
        proj_xy = float(np.linalg.norm(proj[:2]))
        base_z = float(state_post.base_pos_w[2])
        base_ang = float(np.linalg.norm(state_post.base_ang_vel_b))
        base_lin_xy = float(np.linalg.norm(scene.data.qvel[scene._base_vadr:scene._base_vadr + 2]))
        racket_speed = float(np.linalg.norm(scene.racket_site_state()[1]))
        raw_max = float(np.max(np.abs(raw_action)))
        q_delta = q_des - default_q
        leg_delta = q_delta[19:31]
        floor_count, floor_min = _floor_contact_summary()
        low = base_z < float(args.low_height)
        tilt = proj_xy > float(args.tilt_threshold)
        if low and first_low_tick is None:
            first_low_tick = tick
        if tilt and first_tilt_tick is None:
            first_tilt_tick = tick
        max_abs_pitch = max(max_abs_pitch, abs(math.degrees(pitch)))
        max_abs_roll = max(max_abs_roll, abs(math.degrees(roll)))
        max_proj_xy = max(max_proj_xy, proj_xy)
        max_base_ang_vel = max(max_base_ang_vel, base_ang)
        max_base_lin_xy = max(max_base_lin_xy, base_lin_xy)
        max_racket_speed = max(max_racket_speed, racket_speed)
        max_raw_action = max(max_raw_action, raw_max)
        final_row = {
            "tick": int(tick),
            "time_s": float((tick + 1) * dt),
            "phase": lifecycle.phase.value,
            "base_x": float(state_post.base_pos_w[0]),
            "base_y": float(state_post.base_pos_w[1]),
            "base_z": base_z,
            "roll_deg": float(math.degrees(roll)),
            "pitch_deg": float(math.degrees(pitch)),
            "yaw_deg": float(math.degrees(yaw)),
            "proj_grav_x": float(proj[0]),
            "proj_grav_y": float(proj[1]),
            "proj_grav_xy_norm": proj_xy,
            "base_ang_vel_norm": base_ang,
            "base_lin_vel_xy_norm": base_lin_xy,
            "racket_speed": racket_speed,
            "raw_action_l2": float(np.linalg.norm(raw_action)),
            "raw_action_max": raw_max,
            "q_des_l2_from_default": float(np.linalg.norm(q_delta)),
            "leg_q_des_l2_from_default": float(np.linalg.norm(leg_delta)),
            "floor_contact_count": int(floor_count),
            "floor_contact_min_dist": "" if floor_min is None else float(floor_min),
            "fallen_low": int(low),
            "fallen_tilt": int(tilt),
        }
        if args.trace_joints:
            for i, name in enumerate(JOINT_NAMES):
                final_row[f"q_{name}"] = float(state_post.q[i])
                final_row[f"qd_{name}"] = float(state_post.qd[i])
                final_row[f"qdes_{name}"] = float(q_des[i])
                final_row[f"raw_{name}"] = float(np.asarray(raw_action).reshape(31)[i])
        if writer is not None:
            writer.writerow(final_row)
        if args.stop_on_fall and (low or tilt):
            break

    if recorder is not None:
        recorder.close()
    if csv_fh is not None:
        csv_fh.close()
    scene.close()

    elapsed_s = float(final_row.get("time_s", 0.0))
    result = {
        "mode": args.mode,
        "controller": args.controller,
        "last_action_feedback_mode": feedback_mode,
        "table_frame_origin_world_xyz": [float(value) for value in scene.offset],
        "seconds_requested": float(args.seconds),
        "seconds_simulated": elapsed_s,
        "fell_low": first_low_tick is not None,
        "fell_tilt": first_tilt_tick is not None,
        "first_low_s": None if first_low_tick is None else float((first_low_tick + 1) * dt),
        "first_tilt_s": None if first_tilt_tick is None else float((first_tilt_tick + 1) * dt),
        "low_height_m": float(args.low_height),
        "tilt_threshold_proj_gravity_xy": float(args.tilt_threshold),
        "max_abs_pitch_deg": max_abs_pitch,
        "max_abs_roll_deg": max_abs_roll,
        "max_projected_gravity_xy": max_proj_xy,
        "max_base_ang_vel_norm": max_base_ang_vel,
        "max_base_lin_vel_xy_norm": max_base_lin_xy,
        "max_racket_speed": max_racket_speed,
        "max_abs_raw_action": max_raw_action,
        "final": final_row,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--model-xml", default=None)
    parser.add_argument("--runtime-config", default=None)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--serve-manifest", default=None)
    parser.add_argument("--mode", choices=["ready-only", "fake-cycle"], default="ready-only")
    parser.add_argument(
        "--controller",
        choices=["policy", "zero-action"],
        default="policy",
        help="policy runs the exported ONNX; zero-action holds ActionAdapter default_q.",
    )
    parser.add_argument("--side-mode", choices=["mixed", "forehand", "backhand"], default="mixed")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--cycle-period-s", type=float, default=2.2)
    parser.add_argument("--low-height", type=float, default=0.55)
    parser.add_argument("--tilt-threshold", type=float, default=0.85)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument(
        "--last-action-feedback-mode",
        choices=["auto", "raw", "effective"],
        default="auto",
        help="Actor last-action feedback contract. auto prefers ONNX metadata, then runtime config.",
    )
    parser.add_argument(
        "--near-edge-x",
        type=float,
        default=None,
        help="Optional eval-only override; default uses configs/table_frame.yaml.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--record-video", default=None)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--trace-csv", default=None)
    parser.add_argument("--trace-joints", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--fail-on-fall",
        action="store_true",
        help="Exit non-zero if base height or tilt crosses the configured fall thresholds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result))
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    if args.fail_on_fall and (result["fell_low"] or result["fell_tilt"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
