#!/usr/bin/env python3
"""Render a HOPE/A3 body-drive raw MCAP recording into an MP4 video."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def load_converter(repo_root: Path):
    candidates = [
        repo_root
        / "agibot/code_deployment/a3_deploy_example/dist/codex_x86_64/tools/a3_body_drive_debug_convert.py",
        repo_root
        / "agibot/code_deployment/a3_deploy_example/src/a3/a3_deploy_onnx_ref/scripts/tools/a3_body_drive_debug_convert.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("a3_body_drive_debug_convert", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("could not locate a3_body_drive_debug_convert.py")


def mcap_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".mcap":
        return [path]
    files = sorted(path.rglob("*.mcap"))
    if not files:
        raise FileNotFoundError(f"no .mcap files found under {path}")
    return files


def values_by_name(joints, names: list[str], fields: list[str]) -> list[np.ndarray]:
    values = {joint.name: joint for joint in joints}
    arrays: list[list[float]] = [[] for _ in fields]
    for name in names:
        joint = values.get(name)
        for out, field in zip(arrays, fields):
            out.append(float(getattr(joint, field, 0.0)) if joint is not None else 0.0)
    return [np.asarray(out, dtype=np.float64) for out in arrays]


def decode_recording(
    raw_path: Path, converter
) -> tuple[
    list[tuple[int, str, np.ndarray, np.ndarray]],
    list[tuple[int, np.ndarray]],
    list[tuple[int, np.ndarray, np.ndarray]],
    int,
    int,
]:
    typestore = converter.get_ros2_typestore()
    state_events: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    imu_events: list[tuple[int, np.ndarray]] = []
    pose_events: list[tuple[int, np.ndarray, np.ndarray]] = []
    first_command_ns: int | None = None
    last_command_ns: int | None = None

    for file in mcap_files(raw_path):
        with file.open("rb") as stream:
            reader = make_reader(stream)
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                timestamp_ns = int(message.log_time)
                if topic in converter.STATE_TOPIC_TO_GROUP and schema.name == "joint_msgs/msg/JointState":
                    group = converter.STATE_TOPIC_TO_GROUP[topic]
                    cfg = converter.GROUPS[group]
                    decoded = typestore.deserialize_cdr(message.data, "joint_msgs/msg/JointState")
                    q, dq = values_by_name(decoded.joints, list(cfg["names"]), ["position", "velocity"])
                    state_events.append((timestamp_ns, group, q, dq))
                elif topic in converter.COMMAND_TOPIC_TO_GROUP and schema.name == "joint_msgs/msg/JointCommand":
                    first_command_ns = timestamp_ns if first_command_ns is None else min(first_command_ns, timestamp_ns)
                    last_command_ns = timestamp_ns if last_command_ns is None else max(last_command_ns, timestamp_ns)
                elif topic == "/body_drive/pelvis_imu/data" and schema.name == "sensor_msgs/msg/Imu":
                    decoded = typestore.deserialize_cdr(message.data, "sensor_msgs/msg/Imu")
                    quat = np.asarray(
                        [
                            float(decoded.orientation.w),
                            float(decoded.orientation.x),
                            float(decoded.orientation.y),
                            float(decoded.orientation.z),
                        ],
                        dtype=np.float64,
                    )
                    norm = float(np.linalg.norm(quat))
                    if norm > 1e-9:
                        imu_events.append((timestamp_ns, quat / norm))
                elif topic == "/sim/a3/pelvis_pose" and schema.name == "geometry_msgs/msg/PoseStamped":
                    decoded = typestore.deserialize_cdr(message.data, "geometry_msgs/msg/PoseStamped")
                    pos = np.asarray(
                        [
                            float(decoded.pose.position.x),
                            float(decoded.pose.position.y),
                            float(decoded.pose.position.z),
                        ],
                        dtype=np.float64,
                    )
                    quat = np.asarray(
                        [
                            float(decoded.pose.orientation.w),
                            float(decoded.pose.orientation.x),
                            float(decoded.pose.orientation.y),
                            float(decoded.pose.orientation.z),
                        ],
                        dtype=np.float64,
                    )
                    norm = float(np.linalg.norm(quat))
                    if norm > 1e-9:
                        pose_events.append((timestamp_ns, pos, quat / norm))

    if not state_events:
        raise RuntimeError(f"no body-drive joint state messages found under {raw_path}")

    state_events.sort(key=lambda item: item[0])
    imu_events.sort(key=lambda item: item[0])
    pose_events.sort(key=lambda item: item[0])
    if first_command_ns is None or last_command_ns is None or last_command_ns <= first_command_ns:
        first_command_ns = state_events[0][0]
        last_command_ns = state_events[-1][0]
    return state_events, imu_events, pose_events, int(first_command_ns), int(last_command_ns)


def sample_joint_states(
    state_events: list[tuple[int, str, np.ndarray, np.ndarray]],
    imu_events: list[tuple[int, np.ndarray]],
    pose_events: list[tuple[int, np.ndarray, np.ndarray]],
    converter,
    start_ns: int,
    end_ns: int,
    fps: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    q = np.zeros(31, dtype=np.float64)
    dq = np.zeros(31, dtype=np.float64)
    base_pos = np.asarray([0.0, 0.0, 1.3], dtype=np.float64)
    base_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    seen: set[str] = set()
    frames_q: list[np.ndarray] = []
    frames_dq: list[np.ndarray] = []
    frames_base_pos: list[np.ndarray] = []
    frames_base_quat: list[np.ndarray] = []

    event_index = 0
    imu_index = 0
    pose_index = 0
    step_ns = int(round(1_000_000_000 / fps))
    sample_ns = start_ns
    groups = set(converter.GROUPS)

    while sample_ns <= end_ns:
        while event_index < len(state_events) and state_events[event_index][0] <= sample_ns:
            _timestamp_ns, group, group_q, group_dq = state_events[event_index]
            cfg = converter.GROUPS[group]
            start = int(cfg["start"])
            stop = start + len(cfg["names"])
            q[start:stop] = group_q
            dq[start:stop] = group_dq
            seen.add(group)
            event_index += 1
        while imu_index < len(imu_events) and imu_events[imu_index][0] <= sample_ns:
            base_quat = imu_events[imu_index][1]
            imu_index += 1
        while pose_index < len(pose_events) and pose_events[pose_index][0] <= sample_ns:
            base_pos = pose_events[pose_index][1]
            base_quat = pose_events[pose_index][2]
            pose_index += 1
        if seen == groups:
            frames_q.append(q.copy())
            frames_dq.append(dq.copy())
            frames_base_pos.append(base_pos.copy())
            frames_base_quat.append(base_quat.copy())
        sample_ns += step_ns

    if not frames_q:
        raise RuntimeError("no complete 31-joint frames were available in the requested time range")
    return frames_q, frames_dq, frames_base_pos, frames_base_quat


def open_ffmpeg(output: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(output),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_video(
    xml_path: Path,
    output: Path,
    frames_q: list[np.ndarray],
    frames_dq: list[np.ndarray],
    frames_base_pos: list[np.ndarray],
    frames_base_quat: list[np.ndarray],
    converter,
    width: int,
    height: int,
    fps: float,
    camera: str,
) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    scene_option = mujoco.MjvOption()
    if len(scene_option.geomgroup) > 3:
        scene_option.geomgroup[3] = 0

    joint_qpos_addr: dict[str, int] = {}
    joint_qvel_addr: dict[str, int] = {}
    for name in converter.A3_JOINT_NAMES_31:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        joint_qpos_addr[name] = int(model.jnt_qposadr[joint_id])
        joint_qvel_addr[name] = int(model.jnt_dofadr[joint_id])

    ffmpeg = open_ffmpeg(output, width, height, fps)
    assert ffmpeg.stdin is not None
    try:
        for q, dq, base_pos, base_quat in zip(frames_q, frames_dq, frames_base_pos, frames_base_quat):
            mujoco.mj_resetData(model, data)
            if model.nq >= 7:
                data.qpos[0:3] = base_pos
                data.qpos[3:7] = base_quat
            for index, name in enumerate(converter.A3_JOINT_NAMES_31):
                qpos_addr = joint_qpos_addr.get(name)
                qvel_addr = joint_qvel_addr.get(name)
                if qpos_addr is not None:
                    data.qpos[qpos_addr] = q[index]
                if qvel_addr is not None:
                    data.qvel[qvel_addr] = dq[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            ffmpeg.stdin.write(renderer.render().tobytes())
    finally:
        ffmpeg.stdin.close()
        rc = ffmpeg.wait()
        renderer.close()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    default_xml = (
        repo_root
        / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/codex_x86_64/install/bin/cfg/model/a3_pingpong/a3_pingpong.xml"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_path", type=Path, help="Raw body-drive MCAP file or raw directory")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--xml", type=Path, default=default_xml, help="MuJoCo XML path")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", default="torso_follow")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = repo_root_from_script()
    converter = load_converter(repo_root)
    state_events, imu_events, pose_events, start_ns, end_ns = decode_recording(args.raw_path.resolve(), converter)
    frames_q, frames_dq, frames_base_pos, frames_base_quat = sample_joint_states(
        state_events, imu_events, pose_events, converter, start_ns, end_ns, args.fps
    )
    duration_s = len(frames_q) / args.fps
    print(
        f"[render-body-drive-video] frames={len(frames_q)} fps={args.fps:g} "
        f"duration={duration_s:.2f}s output={args.output}"
    )
    render_video(
        args.xml.resolve(),
        args.output.resolve(),
        frames_q,
        frames_dq,
        frames_base_pos,
        frames_base_quat,
        converter,
        args.width,
        args.height,
        args.fps,
        args.camera,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
