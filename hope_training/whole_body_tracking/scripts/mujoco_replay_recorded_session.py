#!/usr/bin/env python3
"""Replay recorded hardware commands and ball tracks in MuJoCo without a planner.

Three actuator-side experiments share the same recorded command lifecycle and
measured incoming ball trajectory:

``closed-loop``
    Rebuild observations from MuJoCo state and run the deployed ONNX.
``open-loop-action``
    Apply the recorded actor action through the deployed ActionAdapter.
``open-loop-qdes``
    Apply the recorded, already-smoothed hardware q_des samples directly.

The first two modes reproduce the 500 Hz alpha=0.5 command filter. Comparing
them separates policy/state-feedback divergence from PD/actuator/dynamics and
contact-model divergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np


HEAD_INDICES = np.array([3, 4], dtype=np.int64)
GROUPS = {
    "waist": np.arange(0, 3),
    "right_arm": np.arange(12, 19),
    "legs": np.arange(19, 31),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--onnx", type=pathlib.Path, required=True)
    parser.add_argument("--runtime-config", type=pathlib.Path, required=True)
    parser.add_argument("--reference-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-xml", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["closed-loop", "open-loop-action", "open-loop-qdes"],
        default=["closed-loop", "open-loop-action", "open-loop-qdes"],
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated planner task IDs. Empty evaluates every replayable task.",
    )
    parser.add_argument("--post-strike-s", type=float, default=1.20)
    parser.add_argument("--command-filter-alpha", type=float, default=0.5)
    parser.add_argument("--physics-dt", type=float, default=0.002)
    parser.add_argument("--contact-radius", type=float, default=0.10)
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record one MP4 per mode.",
    )
    parser.add_argument("--record-tasks", type=int, default=8)
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--mujoco-gl", default="egl")
    return parser.parse_args()


@dataclass
class RecordedBall:
    time_ns: np.ndarray
    position_table: np.ndarray

    def state(self, time_ns: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self.time_ns.size < 2:
            return None
        if time_ns < self.time_ns[0] or time_ns > self.time_ns[-1]:
            return None

        def interpolate(query_ns: int) -> np.ndarray:
            query = float(np.clip(query_ns, self.time_ns[0], self.time_ns[-1]))
            return np.asarray(
                [
                    np.interp(query, self.time_ns, self.position_table[:, axis])
                    for axis in range(3)
                ]
            )

        position = interpolate(time_ns)
        half_window_ns = 10_000_000
        before = max(int(self.time_ns[0]), int(time_ns - half_window_ns))
        after = min(int(self.time_ns[-1]), int(time_ns + half_window_ns))
        if after <= before:
            velocity = np.zeros(3)
        else:
            velocity = (interpolate(after) - interpolate(before)) / (
                (after - before) * 1.0e-9
            )
        return position, velocity


@dataclass
class RecordedTCP:
    time_ns: np.ndarray
    position_table: np.ndarray
    normal_table: np.ndarray

    def state(self, time_ns: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self.time_ns.size < 2:
            return None
        if time_ns < self.time_ns[0] or time_ns > self.time_ns[-1]:
            return None
        query = float(np.clip(time_ns, self.time_ns[0], self.time_ns[-1]))
        position = np.asarray(
            [
                np.interp(query, self.time_ns, self.position_table[:, axis])
                for axis in range(3)
            ]
        )
        normal = np.asarray(
            [
                np.interp(query, self.time_ns, self.normal_table[:, axis])
                for axis in range(3)
            ]
        )
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-9:
            return None
        return position, normal / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1.0e-9 or nb < 1.0e-9:
        return None
    value = float(np.dot(a / na, b / nb))
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def _face_normal(xmat: np.ndarray, side: float) -> np.ndarray:
    normal = np.asarray(xmat, dtype=np.float64).reshape(3, 3)[:, 1].copy()
    if side < 0.0:
        normal = -normal
    norm = float(np.linalg.norm(normal))
    return normal / norm if norm > 1.0e-9 else np.array([1.0, 0.0, 0.0])


def _merge_step_events(total: dict, step, scene, after_contact: bool) -> None:
    if step.ball_racket_contact and not total["contact"]:
        total["contact"] = True
        total["contact_time_offset_s"] = step.contact_time_offset_s
        total["contact_ball_vel_pre_w"] = step.contact_ball_vel_pre_w
        total["contact_ball_vel_post_w"] = step.contact_ball_vel_post_w
        total["contact_racket_pos_pre_w"] = step.contact_racket_site_pos_pre_w
        total["contact_racket_vel_pre_w"] = step.contact_racket_site_vel_pre_w
    if not after_contact and not total["contact"]:
        return
    for z_table, direction in step.net_crossings:
        if direction > 0 and not total["net_cross"]:
            total["net_cross"] = True
            total["net_z_table"] = float(z_table)
            total["net_clear"] = bool(
                z_table > scene.net_height + scene.ball_radius
            )
    for x_table, y_table, direction in step.surface_crossings:
        if direction < 0 and total["contact"] and total["first_bounce"] is None:
            total["first_bounce"] = (float(x_table), float(y_table))


def _set_recorded_robot_state(scene, data: dict[str, np.ndarray], index: int) -> None:
    """Initialize physical pose from the split deploy attitude observation.

    Hardware uses pelvis IMU roll/pitch for projected gravity and the FDU heading
    for base_forward_xy. ``base_quat`` alone therefore is not a unified physical
    world pose. Synthesize the unique yaw-aligned orientation that reproduces
    both actor features before inserting the robot into MuJoCo.
    """
    offset = np.asarray(scene.offset, dtype=np.float64)
    scene.reset_stand()
    scene.data.qpos[scene._base_qadr : scene._base_qadr + 3] = (
        np.asarray(data["base_pos"][index], dtype=np.float64) + offset
    )
    gravity_b = np.asarray(data["observation"][index, 96:99], dtype=np.float64)
    gravity_b /= max(float(np.linalg.norm(gravity_b)), 1.0e-9)
    up_b = -gravity_b
    up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(up_b, up_w)
    cosine = float(np.clip(np.dot(up_b, up_w), -1.0, 1.0))
    if cosine < -1.0 + 1.0e-9:
        tilt = np.diag([1.0, -1.0, -1.0])
    else:
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ]
        )
        tilt = np.eye(3) + skew + (skew @ skew) / max(1.0 + cosine, 1.0e-9)
    forward = tilt[:, 0]
    current_yaw = float(np.arctan2(forward[1], forward[0]))
    heading = np.asarray(data["observation"][index, 99:101], dtype=np.float64)
    desired_yaw = float(np.arctan2(heading[1], heading[0]))
    yaw = desired_yaw - current_yaw
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    yaw_rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotation = yaw_rotation @ tilt
    quaternion = np.empty(4, dtype=np.float64)
    scene._mj.mju_mat2Quat(quaternion, rotation.reshape(-1))
    scene.data.qpos[scene._base_qadr + 3 : scene._base_qadr + 7] = quaternion
    scene.data.qvel[scene._base_vadr : scene._base_vadr + 6] = 0.0
    if 0 < index < data["system_ns"].size - 1:
        before = max(0, index - 2)
        after = min(data["system_ns"].size - 1, index + 2)
        elapsed = (
            int(data["system_ns"][after]) - int(data["system_ns"][before])
        ) * 1.0e-9
        if elapsed > 1.0e-6:
            base_velocity_w = (
                np.asarray(data["base_pos"][after], dtype=np.float64)
                - np.asarray(data["base_pos"][before], dtype=np.float64)
            ) / elapsed
            scene.data.qvel[scene._base_vadr : scene._base_vadr + 3] = np.clip(
                base_velocity_w, -2.0, 2.0
            )
    scene.data.qvel[scene._base_vadr + 3 : scene._base_vadr + 6] = np.asarray(
        data["observation"][index, 0:3], dtype=np.float64
    )
    scene.data.qpos[scene._q_adr] = np.asarray(data["q"][index], dtype=np.float64)
    scene.data.qvel[scene._v_adr] = np.asarray(data["dq"][index], dtype=np.float64)
    scene._mj.mj_forward(scene.model, scene.data)


def _task_ball(data: dict[str, np.ndarray], task_id: int) -> RecordedBall | None:
    mask = (
        (data["ball_task_id"] == task_id)
        & np.all(np.isfinite(data["ball_pos"]), axis=1)
        & (data["ball_time_ns"] > 0)
    )
    if np.sum(mask) < 10:
        return None
    time_ns = np.asarray(data["ball_time_ns"][mask], dtype=np.int64)
    positions = np.asarray(data["ball_pos"][mask], dtype=np.float64)
    order = np.argsort(time_ns)
    time_ns = time_ns[order]
    positions = positions[order]
    unique = np.concatenate(([True], np.diff(time_ns) > 0))
    return RecordedBall(time_ns[unique], positions[unique])


def _task_tcp(data: dict[str, np.ndarray], task_id: int) -> RecordedTCP | None:
    mask = (
        (data["ball_task_id"] == task_id)
        & np.all(np.isfinite(data["tcp_pos"]), axis=1)
        & np.all(np.isfinite(data["tcp_normal"]), axis=1)
        & (data["ball_time_ns"] > 0)
    )
    if np.sum(mask) < 2:
        return None
    time_ns = np.asarray(data["ball_time_ns"][mask], dtype=np.int64)
    positions = np.asarray(data["tcp_pos"][mask], dtype=np.float64)
    normals = np.asarray(data["tcp_normal"][mask], dtype=np.float64)
    order = np.argsort(time_ns)
    time_ns = time_ns[order]
    positions = positions[order]
    normals = normals[order]
    unique = np.concatenate(([True], np.diff(time_ns) > 0))
    return RecordedTCP(time_ns[unique], positions[unique], normals[unique])


def _task_windows(
    data: dict[str, np.ndarray],
    selected: set[int] | None,
    post_strike_s: float,
) -> list[tuple[int, np.ndarray]]:
    windows: list[tuple[int, np.ndarray]] = []
    task_ids = np.asarray(data["active_task_id"], dtype=np.int64)
    phase = np.asarray(data["phase"], dtype=np.int8)
    system_ns = np.asarray(data["system_ns"], dtype=np.int64)
    for task_id in sorted(set(int(value) for value in task_ids if value > 0)):
        if selected is not None and task_id not in selected:
            continue
        lifecycle = np.flatnonzero(task_ids == task_id)
        active = lifecycle[(phase[lifecycle] == 1) | (phase[lifecycle] == 2)]
        ball = _task_ball(data, task_id)
        if lifecycle.size == 0 or active.size == 0 or ball is None:
            continue
        start = int(lifecycle[0])
        strike_candidates = active[np.asarray(data["time_to_strike"][active]) <= 0.0]
        strike_index = int(strike_candidates[0]) if strike_candidates.size else int(active[-1])
        end_time = max(
            system_ns[strike_index] + int(post_strike_s * 1.0e9),
            system_ns[int(lifecycle[-1])],
        )
        end = int(np.searchsorted(system_ns, end_time, side="right"))
        indices = np.arange(start, min(end, system_ns.size), dtype=np.int64)
        windows.append((task_id, indices))
    return windows


def _write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> dict:
    attempts = len(rows)
    contact = sum(int(row["contact"]) for row in rows)
    net_clear = sum(int(row["net_clear"]) for row in rows)
    opponent = sum(int(row["opponent_bounce"]) for row in rows)
    success = sum(int(row["success"]) for row in rows)
    fall = sum(int(row["fallen"]) for row in rows)
    return {
        "attempts": attempts,
        "contact": contact,
        "contact_rate": contact / attempts if attempts else None,
        "net_clear": net_clear,
        "net_clear_rate": net_clear / attempts if attempts else None,
        "opponent_bounce": opponent,
        "opponent_bounce_rate": opponent / attempts if attempts else None,
        "success": success,
        "success_rate": success / attempts if attempts else None,
        "fall": fall,
        "fall_rate": fall / attempts if attempts else None,
        "min_ball_racket_distance_m": (
            float(min(row["min_ball_racket_distance_m"] for row in rows))
            if rows
            else None
        ),
        "action_l2_vs_real_median": (
            float(
                np.median(
                    [
                        row["action_l2_vs_real_median"]
                        for row in rows
                        if row["action_l2_vs_real_median"] is not None
                    ]
                )
            )
            if any(row["action_l2_vs_real_median"] is not None for row in rows)
            else None
        ),
    }


def _run_mode(
    args: argparse.Namespace,
    mode: str,
    data: dict[str, np.ndarray],
    windows: list[tuple[int, np.ndarray]],
    runtime,
    policy,
    build_observation_normal114,
    ObsTarget,
    Scene,
    ball_cfg: dict,
    Recorder,
) -> dict:
    from a3_deploy_onnx_ref_pingpong.joint_order import JOINT_NAMES

    scene = Scene(
        str(args.model_xml),
        ball_cfg,
        JOINT_NAMES,
        control_dt=args.physics_dt,
        launch_viewer=False,
    )
    recorder = None
    if args.record_video:
        recorder = Recorder(
            scene,
            str(args.output_dir / mode / f"{mode}_recorded_ball_command.mp4"),
            width=args.video_width,
            height=args.video_height,
            fps=args.video_fps,
        )

    task_results: list[dict] = []
    trace_rows: list[dict] = []
    substeps = max(1, int(round(runtime.control_dt / args.physics_dt)))
    alpha = float(args.command_filter_alpha)
    offset = np.asarray(scene.offset, dtype=np.float64)

    try:
        for trial, (task_id, indices) in enumerate(windows):
            first = int(indices[0])
            _set_recorded_robot_state(scene, data, first)
            ball_track = _task_ball(data, task_id)
            tcp_track = _task_tcp(data, task_id)
            assert ball_track is not None
            initial_ball = ball_track.state(int(data["system_ns"][first]))
            if initial_ball is None:
                continue
            scene.set_ball(initial_ball[0] + offset, initial_ball[1])

            last_action = np.asarray(data["observation"][first, 65:96], dtype=np.float64)
            command_q = np.asarray(data["q_des"][first], dtype=np.float64).copy()
            command_q[HEAD_INDICES] = runtime.action_adapter.default_q[HEAD_INDICES]
            scene.write_targets(command_q, runtime.sim_kp, runtime.sim_kd)
            initial_state = scene.read_robot_state()
            initial_target = ObsTarget(
                pos_w=np.asarray(data["target_pos"][first], dtype=np.float64) + offset,
                vel_w=np.asarray(data["target_vel"][first], dtype=np.float64),
                time_to_strike=float(data["time_to_strike"][first]),
                swing_side=float(data["swing_side"][first]),
                normal_w=np.asarray(data["target_normal"][first], dtype=np.float64),
            )
            initial_station = (
                np.asarray(data["station"][first], dtype=np.float64) + offset[:2]
            )
            initial_observation = build_observation_normal114(
                initial_state,
                initial_target,
                last_action,
                runtime.action_adapter.default_q,
                initial_station,
            )
            initial_obs_delta = initial_observation - np.asarray(
                data["observation"][first], dtype=np.float64
            )
            initial_obs_group_error = {
                "initial_obs_base_ang_vel_l2_error": float(
                    np.linalg.norm(initial_obs_delta[0:3])
                ),
                "initial_obs_q_l2_error": float(
                    np.linalg.norm(initial_obs_delta[3:34])
                ),
                "initial_obs_dq_l2_error": float(
                    np.linalg.norm(initial_obs_delta[34:65])
                ),
                "initial_obs_gravity_l2_error": float(
                    np.linalg.norm(initial_obs_delta[96:99])
                ),
                "initial_obs_heading_l2_error": float(
                    np.linalg.norm(initial_obs_delta[99:101])
                ),
                "initial_obs_station_l2_error": float(
                    np.linalg.norm(initial_obs_delta[101:103])
                ),
                "initial_obs_command_l2_error": float(
                    np.linalg.norm(initial_obs_delta[103:114])
                ),
            }
            initial_racket_pos, _, initial_racket_xmat = scene.racket_site_pose()
            initial_racket_normal = _face_normal(
                initial_racket_xmat, float(data["swing_side"][first])
            )
            initial_recorded_tcp = (
                tcp_track.state(int(data["system_ns"][first]))
                if tcp_track is not None
                else None
            )
            initial_tcp_position_error = None
            initial_tcp_normal_error = None
            initial_tcp_normal_axis_error = None
            if initial_recorded_tcp is not None:
                recorded_tcp_pos, recorded_tcp_normal = initial_recorded_tcp
                initial_tcp_position_error = float(
                    np.linalg.norm(
                        scene.to_table(initial_racket_pos) - recorded_tcp_pos
                    )
                )
                initial_tcp_normal_error = _angle_deg(
                    initial_racket_normal, recorded_tcp_normal
                )
                if initial_tcp_normal_error is not None:
                    initial_tcp_normal_axis_error = min(
                        initial_tcp_normal_error, 180.0 - initial_tcp_normal_error
                    )

            events = {
                "contact": False,
                "contact_time_offset_s": None,
                "contact_ball_vel_pre_w": None,
                "contact_ball_vel_post_w": None,
                "contact_racket_pos_pre_w": None,
                "contact_racket_vel_pre_w": None,
                "net_cross": False,
                "net_z_table": None,
                "net_clear": False,
                "first_bounce": None,
            }
            min_distance = float("inf")
            action_errors: list[float] = []
            group_errors = {name: [] for name in GROUPS}
            observation_errors: list[float] = []
            q_state_errors: list[float] = []
            dq_state_errors: list[float] = []
            base_position_errors: list[float] = []
            tcp_position_errors: list[float] = []
            tcp_normal_errors: list[float] = []
            tcp_normal_axis_errors: list[float] = []
            contact_diag: dict[str, float | None] = {}
            released = False
            start_ns = int(data["system_ns"][first])

            for local_tick, index_value in enumerate(indices):
                index = int(index_value)
                state = scene.read_robot_state()
                target = ObsTarget(
                    pos_w=np.asarray(data["target_pos"][index], dtype=np.float64) + offset,
                    vel_w=np.asarray(data["target_vel"][index], dtype=np.float64),
                    time_to_strike=float(data["time_to_strike"][index]),
                    swing_side=float(data["swing_side"][index]),
                    normal_w=np.asarray(data["target_normal"][index], dtype=np.float64),
                )
                station_w = np.asarray(data["station"][index], dtype=np.float64) + offset[:2]
                observation_last_action = (
                    last_action
                    if mode == "closed-loop"
                    else np.asarray(
                        data["observation"][index, 65:96], dtype=np.float64
                    )
                )
                simulated_observation = build_observation_normal114(
                    state,
                    target,
                    observation_last_action,
                    runtime.action_adapter.default_q,
                    station_w,
                )
                observation_error = float(
                    np.linalg.norm(
                        simulated_observation
                        - np.asarray(data["observation"][index], dtype=np.float64)
                    )
                )
                q_state_error = float(
                    np.linalg.norm(
                        np.asarray(state.q, dtype=np.float64)
                        - np.asarray(data["q"][index], dtype=np.float64)
                    )
                )
                dq_state_error = float(
                    np.linalg.norm(
                        np.asarray(state.qd, dtype=np.float64)
                        - np.asarray(data["dq"][index], dtype=np.float64)
                    )
                )
                base_position_error = float(
                    np.linalg.norm(
                        scene.to_table(state.base_pos_w)
                        - np.asarray(data["base_pos"][index], dtype=np.float64)
                    )
                )
                observation_errors.append(observation_error)
                q_state_errors.append(q_state_error)
                dq_state_errors.append(dq_state_error)
                base_position_errors.append(base_position_error)
                recorded_tcp = (
                    tcp_track.state(int(data["system_ns"][index]))
                    if tcp_track is not None
                    else None
                )
                if recorded_tcp is not None:
                    simulated_tcp_pos, _, simulated_tcp_xmat = scene.racket_site_pose()
                    simulated_tcp_normal = _face_normal(
                        simulated_tcp_xmat, float(data["swing_side"][index])
                    )
                    recorded_tcp_pos, recorded_tcp_normal = recorded_tcp
                    tcp_position_errors.append(
                        float(
                            np.linalg.norm(
                                scene.to_table(simulated_tcp_pos) - recorded_tcp_pos
                            )
                        )
                    )
                    normal_error = _angle_deg(
                        simulated_tcp_normal, recorded_tcp_normal
                    )
                    if normal_error is not None:
                        tcp_normal_errors.append(normal_error)
                        tcp_normal_axis_errors.append(
                            min(normal_error, 180.0 - normal_error)
                        )

                if mode == "closed-loop":
                    raw = policy.infer(simulated_observation).astype(np.float64)
                    applied = raw.copy()
                    applied[HEAD_INDICES] = 0.0
                    target_q = runtime.action_adapter.decode(applied)
                    recorded_raw = np.asarray(data["raw_action"][index], dtype=np.float64)
                    delta = raw - recorded_raw
                    action_errors.append(float(np.linalg.norm(delta)))
                    for group, group_indices in GROUPS.items():
                        group_errors[group].append(
                            float(np.linalg.norm(delta[group_indices]))
                        )
                    last_action = applied
                elif mode == "open-loop-action":
                    applied = np.asarray(data["applied_action"][index], dtype=np.float64)
                    target_q = runtime.action_adapter.decode(applied)
                else:
                    target_q = np.asarray(data["q_des"][index], dtype=np.float64)
                current_action_error = (
                    action_errors[-1] if mode == "closed-loop" else None
                )

                target_q = np.asarray(target_q, dtype=np.float64).copy()
                target_q[HEAD_INDICES] = runtime.action_adapter.default_q[HEAD_INDICES]

                for substep in range(substeps):
                    absolute_ns = int(
                        data["system_ns"][index] + substep * args.physics_dt * 1.0e9
                    )
                    tts_substep = float(data["time_to_strike"][index]) - (
                        substep * args.physics_dt
                    )
                    if (
                        not released
                        and not events["contact"]
                        and int(data["phase"][index]) == 1
                        and tts_substep > 0.0
                    ):
                        measured = ball_track.state(absolute_ns)
                        if measured is not None:
                            scene.set_ball(measured[0] + offset, measured[1])
                    else:
                        released = True

                    if mode == "open-loop-qdes":
                        command_q = target_q
                    else:
                        command_q = (1.0 - alpha) * command_q + alpha * target_q
                    scene.write_targets(command_q, runtime.sim_kp, runtime.sim_kd)
                    step = scene.step()
                    had_contact = bool(events["contact"])
                    _merge_step_events(events, step, scene, had_contact)
                    if step.ball_racket_contact and not had_contact:
                        released = True
                        racket_pos, racket_vel, racket_xmat = scene.racket_site_pose()
                        racket_normal = _face_normal(
                            racket_xmat, float(data["swing_side"][index])
                        )
                        contact_diag = {
                            "target_pos_error_m": float(
                                np.linalg.norm(racket_pos - target.pos_w)
                            ),
                            "target_vel_error_mps": float(
                                np.linalg.norm(racket_vel - target.vel_w)
                            ),
                            "target_vel_angle_deg": _angle_deg(
                                racket_vel, target.vel_w
                            ),
                            "target_normal_error_deg": _angle_deg(
                                racket_normal, target.normal_w
                            ),
                            "tts_s": tts_substep,
                        }
                    ball_pos, _ = scene.ball_state()
                    racket_pos, _ = scene.racket_site_state()
                    min_distance = min(
                        min_distance, float(np.linalg.norm(ball_pos - racket_pos))
                    )

                balance = scene.robot_balance_diagnostics()
                if recorder is not None and trial < args.record_tasks:
                    recorder.capture(scene)
                if local_tick % 2 == 0:
                    trace_rows.append(
                        {
                            "mode": mode,
                            "trial": trial,
                            "task_id": task_id,
                            "tick": local_tick,
                            "elapsed_s": float(
                                (int(data["system_ns"][index]) - start_ns) * 1.0e-9
                            ),
                            "phase": int(data["phase"][index]),
                            "tts_s": float(data["time_to_strike"][index]),
                            "contact": int(events["contact"]),
                            "base_z": float(scene.base_pos_w()[2]),
                            "pelvis_pitch_deg": float(balance["pelvis_rpy_deg"][1]),
                            "torso_pitch_deg": float(balance["torso_rpy_deg"][1]),
                            "support_margin_m": float(balance["support_margin"]),
                            "min_ball_racket_distance_m": min_distance,
                            "observation_l2_vs_real": observation_error,
                            "q_l2_vs_real": q_state_error,
                            "dq_l2_vs_real": dq_state_error,
                            "base_position_error_m": base_position_error,
                            "action_l2_vs_real": current_action_error,
                        }
                    )

            first_bounce = events["first_bounce"]
            opponent_bounce = bool(
                first_bounce is not None
                and scene.net_x_table <= first_bounce[0] <= scene.length
                and -scene.width <= first_bounce[1] <= 0.0
            )
            success = bool(events["contact"] and events["net_clear"] and opponent_bounce)
            balance = scene.robot_balance_diagnostics()
            fallen = bool(scene.base_fallen())
            row = {
                "mode": mode,
                "trial": trial,
                "task_id": task_id,
                "side": (
                    "forehand"
                    if float(np.median(data["swing_side"][indices])) >= 0.0
                    else "backhand"
                ),
                "contact": int(events["contact"]),
                "net_cross": int(events["net_cross"]),
                "net_clear": int(events["net_clear"]),
                "opponent_bounce": int(opponent_bounce),
                "success": int(success),
                "fallen": int(fallen),
                "min_ball_racket_distance_m": min_distance,
                "first_bounce_x_table": (
                    first_bounce[0] if first_bounce is not None else None
                ),
                "first_bounce_y_table": (
                    first_bounce[1] if first_bounce is not None else None
                ),
                "net_z_table": events["net_z_table"],
                "action_l2_vs_real_median": (
                    float(np.median(action_errors)) if action_errors else None
                ),
                "action_l2_vs_real_p95": (
                    float(np.quantile(action_errors, 0.95)) if action_errors else None
                ),
                "observation_l2_vs_real_median": float(
                    np.median(observation_errors)
                ),
                "observation_l2_vs_real_p95": float(
                    np.quantile(observation_errors, 0.95)
                ),
                "q_l2_vs_real_median": float(np.median(q_state_errors)),
                "q_l2_vs_real_p95": float(np.quantile(q_state_errors, 0.95)),
                "dq_l2_vs_real_median": float(np.median(dq_state_errors)),
                "base_position_error_median_m": float(
                    np.median(base_position_errors)
                ),
                "base_position_error_p95_m": float(
                    np.quantile(base_position_errors, 0.95)
                ),
                "base_z_end": float(scene.base_pos_w()[2]),
                "pelvis_pitch_deg_end": float(balance["pelvis_rpy_deg"][1]),
                "torso_pitch_deg_end": float(balance["torso_rpy_deg"][1]),
                "support_margin_end_m": float(balance["support_margin"]),
                "initial_obs_l2_error": float(np.linalg.norm(initial_obs_delta)),
                "initial_obs_max_abs_error": float(np.max(np.abs(initial_obs_delta))),
                **initial_obs_group_error,
                "initial_tcp_position_error_m": initial_tcp_position_error,
                "initial_tcp_normal_error_deg": initial_tcp_normal_error,
                "initial_tcp_normal_axis_error_deg": initial_tcp_normal_axis_error,
                "tcp_position_error_median_m": (
                    float(np.median(tcp_position_errors))
                    if tcp_position_errors
                    else None
                ),
                "tcp_position_error_p95_m": (
                    float(np.quantile(tcp_position_errors, 0.95))
                    if tcp_position_errors
                    else None
                ),
                "tcp_normal_error_median_deg": (
                    float(np.median(tcp_normal_errors))
                    if tcp_normal_errors
                    else None
                ),
                "tcp_normal_axis_error_median_deg": (
                    float(np.median(tcp_normal_axis_errors))
                    if tcp_normal_axis_errors
                    else None
                ),
                **contact_diag,
            }
            for group, values in group_errors.items():
                row[f"{group}_action_l2_vs_real_median"] = (
                    float(np.median(values)) if values else None
                )
            task_results.append(row)
    finally:
        if recorder is not None:
            recorder.close()

    mode_dir = args.output_dir / mode
    _write_rows(mode_dir / "tasks.csv", task_results)
    _write_rows(mode_dir / "trace.csv", trace_rows)
    summary = _summarize(task_results)
    result = {
        "mode": mode,
        "planner_used": False,
        "command_source": "recorded final MDU lifecycle target",
        "ball_source": "recorded Motive path, kinematic-guided until strike",
        "table_to_mujoco_world_translation_xyz": [float(value) for value in offset],
        "command_filter": (
            None
            if mode == "open-loop-qdes"
            else {
                "publish_hz": 1.0 / args.physics_dt,
                "alpha": alpha,
            }
        ),
        **summary,
        "tasks_csv": str((mode_dir / "tasks.csv").resolve()),
        "trace_csv": str((mode_dir / "trace.csv").resolve()),
    }
    (mode_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    args = parse_args()
    if not 0.0 < args.command_filter_alpha <= 1.0:
        raise ValueError("--command-filter-alpha must be in (0, 1]")
    if args.physics_dt <= 0.0:
        raise ValueError("--physics-dt must be positive")
    if args.record_video:
        import os

        os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_npz = np.load(args.dataset)
    data = {name: data_npz[name] for name in data_npz.files}
    selected = (
        {int(value) for value in args.tasks.split(",") if value.strip()}
        if args.tasks
        else None
    )
    windows = _task_windows(data, selected, args.post_strike_s)
    if not windows:
        raise ValueError("no replayable task has both an active policy window and ball samples")

    sys.path.insert(0, str(args.reference_dir.resolve()))
    from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
    from a3_deploy_onnx_ref_pingpong.observation import (
        ObsTarget,
        build_observation_normal114,
    )
    from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy

    scripts_dir = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from mujoco_eval_onnx import _MujocoVideoRecorder, _load_success_metric
    from mujoco_pingpong_scene import PingPongRealPhysicsScene

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    metric = _load_success_metric(repo_root)
    ball_cfg = metric.load_ball_physics_config()
    runtime = RuntimeConfig.load(args.runtime_config)
    policy = OnnxPolicy(args.onnx)
    if policy.obs_dim != 114:
        raise ValueError(f"recorded replay requires a 114-D actor, got {policy.obs_dim}")

    results = []
    for mode in args.modes:
        results.append(
            _run_mode(
                args,
                mode,
                data,
                windows,
                runtime,
                policy,
                build_observation_normal114,
                ObsTarget,
                PingPongRealPhysicsScene,
                ball_cfg,
                _MujocoVideoRecorder,
            )
        )
    aggregate = {
        "dataset": str(args.dataset.resolve()),
        "onnx": str(args.onnx.resolve()),
        "planner_used": False,
        "replayable_task_ids": [task_id for task_id, _ in windows],
        "modes": results,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
