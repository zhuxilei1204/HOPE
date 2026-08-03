#!/usr/bin/env python3
"""Forward OptiTrack base pose and raw HOPE planner commands to MDU.

HP14 v2 preserves the planner/capture timestamp.  TTS age correction belongs
to the MDU, where UDP receive time and actuation lead are both known.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import threading
import time
from typing import Sequence

PACKET = struct.Struct("<4sHHQQQQQQII20f")
JOINT_PACKET = struct.Struct("<4sHHQQ62f")
COMMAND_STATUS_PACKET = struct.Struct("<4sHH10Q2I3iI2f")
COMMAND_STATUS_FLAGS = {
    "planner_packet_received": 1 << 0,
    "planner_packet_fresh": 1 << 1,
    "wire_command_valid": 1 << 2,
    "transport_timestamp_valid": 1 << 3,
    "command_timestamp_valid": 1 << 4,
    "lifecycle_command_applied": 1 << 5,
    "command_entered_model": 1 << 6,
    "model_inference_ok": 1 << 7,
    "publish_commands": 1 << 8,
    "policy_output_active": 1 << 9,
    "shadow_probe": 1 << 10,
    "pelvis_window_ready": 1 << 11,
}
LIVE_MODES = {-1: "PROBE", 0: "IDLE", 1: "PASSIVE", 2: "PD_STAND", 3: "PREP_STAND", 4: "POLICY"}
LIFECYCLE_PHASES = {0: "ready", 1: "swing", 2: "follow_through", 3: "recovery"}
REVISION_DECISIONS = {
    0: "no_new_command", 1: "accepted_initial", 2: "accepted_revision",
    3: "late_initial", 4: "late_freeze", 5: "old_or_duplicate",
    6: "phase_blocked",
}
MAGIC = b"HP14"
VERSION = 2
JOINT_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def rotate_vector_xyzw(quat: Sequence[float], vec: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = (float(v) for v in quat)
    vx, vy, vz = (float(v) for v in vec)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid base quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def normalize_quat_xyzw(quat: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid quaternion")
    return x / norm, y / norm, z / norm, w / norm


def multiply_quat_xyzw(
    lhs: Sequence[float], rhs: Sequence[float]
) -> tuple[float, float, float, float]:
    """Return lhs * rhs, e.g. q_world_fdu * q_fdu_pelvis."""
    x1, y1, z1, w1 = normalize_quat_xyzw(lhs)
    x2, y2, z2, w2 = normalize_quat_xyzw(rhs)
    return normalize_quat_xyzw((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdu-host", default="10.42.10.12")
    parser.add_argument("--mdu-port", type=int, default=17240)
    parser.add_argument("--base-topic", default="/FDU_a3/pose")
    parser.add_argument("--ball-topic", default="/ball/point")
    parser.add_argument("--marker-topic", default="/optitrack/markerMetadata")
    parser.add_argument("--ball-min-size-m", type=float, default=0.015)
    parser.add_argument("--ball-max-size-m", type=float, default=0.050)
    parser.add_argument("--racket-topic", default="/FDU_pai/pose")
    parser.add_argument("--command-topic", default="/racket/command")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--max-base-age", type=float, default=1.00)
    parser.add_argument("--tracking-warning-age", type=float, default=0.10)
    parser.add_argument("--max-command-age", type=float, default=0.35)
    parser.add_argument("--bias-body-x", type=float, default=0.0)
    parser.add_argument("--bias-body-y", type=float, default=0.0)
    parser.add_argument("--bias-body-z", type=float, default=0.0)
    parser.add_argument("--fdu-to-pelvis-qx", type=float, default=0.0)
    parser.add_argument("--fdu-to-pelvis-qy", type=float, default=0.0)
    parser.add_argument("--fdu-to-pelvis-qz", type=float, default=0.0)
    parser.add_argument("--fdu-to-pelvis-qw", type=float, default=1.0)
    parser.add_argument("--station-x", type=float, default=None)
    parser.add_argument("--station-y", type=float, default=None)
    parser.add_argument("--monitor-host", default="")
    parser.add_argument("--monitor-port", type=int, default=17650)
    parser.add_argument("--joint-listen-host", default="0.0.0.0")
    parser.add_argument("--joint-listen-port", type=int, default=17242)
    parser.add_argument("--mdu-status-listen-host", default="0.0.0.0")
    parser.add_argument("--mdu-status-listen-port", type=int, default=17241)
    parser.add_argument("--marker-process-period", type=float, default=0.10)
    parser.add_argument("--max-joint-drain", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rate <= 0.0 or not 0 < args.mdu_port < 65536:
        raise SystemExit("rate and mdu-port must be positive")
    if args.max_base_age <= 0.0 or args.tracking_warning_age <= 0.0:
        raise SystemExit("base age limits must be > 0")
    if args.marker_process_period < 0.0 or args.max_joint_drain <= 0:
        raise SystemExit("marker-process-period must be >= 0 and max-joint-drain > 0")
    fdu_to_pelvis_quat = normalize_quat_xyzw((
        args.fdu_to_pelvis_qx,
        args.fdu_to_pelvis_qy,
        args.fdu_to_pelvis_qz,
        args.fdu_to_pelvis_qw,
    ))

    import rclpy
    from geometry_msgs.msg import PointStamped, PoseStamped
    from hope_msgs.msg import RacketCommand
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2

    class Bridge(Node):
        def __init__(self) -> None:
            super().__init__("hope_hdu_planner_udp_bridge")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.destination = (args.mdu_host, args.mdu_port)
            self.monitor_destinations = [
                (host.strip(), args.monitor_port)
                for host in args.monitor_host.split(",")
                if host.strip()
            ]
            self.joint_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.joint_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.joint_sock.bind((args.joint_listen_host, args.joint_listen_port))
            self.joint_sock.setblocking(False)
            self.mdu_status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.mdu_status_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.mdu_status_sock.bind(
                (args.mdu_status_listen_host, args.mdu_status_listen_port)
            )
            self.mdu_status_sock.setblocking(False)
            self.rigid_body = None
            self.bias_world = None
            self.base = None
            self.ball = None
            self.marker_stats = None
            self.racket = None
            self.command = None
            self.base_rx = 0.0
            self.ball_rx = 0.0
            self.marker_rx = 0.0
            self.racket_rx = 0.0
            self.command_rx = 0.0
            self.command_rx_unix_ns = 0
            self.command_header_unix_ns = 0
            self.planner_node_active = False
            self.joint_q = None
            self.joint_dq = None
            self.joint_sequence = 0
            self.joint_rx = 0.0
            self.mdu_status = None
            self.mdu_status_rx = 0.0
            self.mdu_status_rx_count = 0
            self.mdu_status_invalid_count = 0
            self.sequence = 0
            self.marker_process_rx = 0.0
            self.command_rx_count = 0
            self.command_tx_valid_count = 0
            self.tx_count = 0
            self.tx_gap_max_ms = 0.0
            self.last_tx = 0.0
            self.last_tx_report = 0.0
            self.last_mdu_tx_sequence = 0
            self.last_mdu_tx_unix_ns = 0
            self.last_mdu_tx_task_id = 0
            self.last_mdu_tx_revision = 0
            self.last_mdu_tx_command_valid = False
            self.sender_deadline_misses = 0
            self.sender_stop = threading.Event()
            self.sender_thread = None
            self.station = None
            if args.station_x is not None and args.station_y is not None:
                self.station = (args.station_x, args.station_y)
            elif (args.station_x is None) != (args.station_y is None):
                raise ValueError("station-x and station-y must be provided together")
            self.create_subscription(
                PoseStamped, args.base_topic, self.on_base, qos_profile_sensor_data
            )
            self.create_subscription(
                PointStamped, args.ball_topic, self.on_ball, qos_profile_sensor_data
            )
            self.create_subscription(
                PointCloud2, args.marker_topic, self.on_markers, qos_profile_sensor_data
            )
            self.create_subscription(
                PoseStamped, args.racket_topic, self.on_racket, qos_profile_sensor_data
            )
            self.create_subscription(RacketCommand, args.command_topic, self.on_command, 10)
            # dedicated_sender_thread_v1: control heartbeat must not share
            # the overloaded rclpy callback queue.
            self.create_timer(0.5, self.update_graph_status)
            self.last_wait_log = 0.0
            self.get_logger().info(
                f"base={args.base_topic} racket={args.racket_topic} command={args.command_topic} "
                f"destination={args.mdu_host}:{args.mdu_port} "
                f"monitor={','.join(host for host, _ in self.monitor_destinations) or 'disabled'}:"
                f"{args.monitor_port} "
                f"joints={args.joint_listen_host}:{args.joint_listen_port} "
                f"mdu_status={args.mdu_status_listen_host}:"
                f"{args.mdu_status_listen_port} "
                f"q_fdu_pelvis_xyzw={fdu_to_pelvis_quat}"
            )
            self.sender_thread = threading.Thread(
                target=self.sender_loop, name="hope_udp_sender", daemon=True
            )
            self.sender_thread.start()

        def sender_loop(self) -> None:
            period = 1.0 / args.rate
            deadline = time.monotonic()
            while not self.sender_stop.is_set():
                deadline += period
                self.send()
                delay = deadline - time.monotonic()
                if delay > 0.0:
                    self.sender_stop.wait(delay)
                else:
                    self.sender_deadline_misses += 1
                    deadline = time.monotonic()

        def stop_sender(self) -> None:
            self.sender_stop.set()
            if self.sender_thread is not None:
                self.sender_thread.join(timeout=2.0)

        def on_base(self, msg: PoseStamped) -> None:
            p = msg.pose.position
            q = msg.pose.orientation
            quat = normalize_quat_xyzw((q.x, q.y, q.z, q.w))
            pelvis_quat = multiply_quat_xyzw(quat, fdu_to_pelvis_quat)
            offset = rotate_vector_xyzw(
                quat, (args.bias_body_x, args.bias_body_y, args.bias_body_z)
            )
            self.rigid_body = (p.x, p.y, p.z, q.w, q.x, q.y, q.z)
            self.bias_world = offset
            self.base = (
                p.x + offset[0], p.y + offset[1], p.z + offset[2],
                pelvis_quat[3], pelvis_quat[0], pelvis_quat[1], pelvis_quat[2],
            )
            self.base_rx = time.monotonic()
            if self.station is None:
                self.station = (self.base[0], self.base[1])
                self.get_logger().info(
                    f"station locked at ({self.station[0]:.4f}, {self.station[1]:.4f})"
                )

        def on_ball(self, msg: PointStamped) -> None:
            self.ball = (msg.point.x, msg.point.y, msg.point.z)
            self.ball_rx = time.monotonic()

        def on_markers(self, msg: PointCloud2) -> None:
            now = time.monotonic()
            self.marker_rx = now
            if now - self.marker_process_rx < args.marker_process_period:
                return
            self.marker_process_rx = now
            candidates = []
            total = 0
            for row in point_cloud2.read_points(
                msg, field_names=("x", "y", "z", "size", "id", "flags"), skip_nans=True
            ):
                total += 1
                x, y, z, size, marker_id, flags = row
                values = (float(x), float(y), float(z), float(size))
                if (all(math.isfinite(value) for value in values)
                        and int(flags) & 1
                        and args.ball_min_size_m <= values[3] <= args.ball_max_size_m):
                    candidates.append((values[3], values[0], values[1], values[2], int(marker_id), int(flags)))
            if candidates and self.ball is not None and now - self.ball_rx <= 0.25:
                selected = min(
                    candidates,
                    key=lambda item: sum(
                        (item[index + 1] - self.ball[index]) ** 2
                        for index in range(3)
                    ),
                )
            else:
                selected = max(candidates, default=None)
            self.marker_stats = {
                "total_count": total,
                "candidate_count": len(candidates),
                "threshold_mm": args.ball_min_size_m * 1000.0,
                "max_threshold_mm": args.ball_max_size_m * 1000.0,
                "selected_id": None if selected is None else selected[4],
                "selected_size_mm": None if selected is None else selected[0] * 1000.0,
                "selected_xyz": None if selected is None else list(selected[1:4]),
                "selected_unlabeled": None if selected is None else bool(selected[5] & 1),
                "selected_flags": None if selected is None else selected[5],
            }
            self.marker_rx = time.monotonic()

        def on_racket(self, msg: PoseStamped) -> None:
            p = msg.pose.position
            q = msg.pose.orientation
            values = (p.x, p.y, p.z, q.w, q.x, q.y, q.z)
            if all(math.isfinite(value) for value in values):
                self.racket = values
                self.racket_rx = time.monotonic()

        def on_command(self, msg: RacketCommand) -> None:
            self.command = msg
            self.command_rx = time.monotonic()
            self.command_rx_unix_ns = time.time_ns()
            self.command_header_unix_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )
            self.command_rx_count += 1

        def update_graph_status(self) -> None:
            self.planner_node_active = any(
                name == "hope_planner" for name, _ in self.get_node_names_and_namespaces()
            )

        def send(self) -> None:
            now = time.monotonic()
            self.send_monitor(now)
            base_age = now - self.base_rx if self.base is not None else math.inf
            base_ready = (
                self.base is not None
                and self.station is not None
                and base_age <= args.max_base_age
            )
            if not base_ready:
                if now - self.last_wait_log >= 1.0:
                    self.last_wait_log = now
                    self.get_logger().warning("waiting for fresh base pose")
                return
            command_valid = (
                self.command is not None
                and now - self.command_rx <= args.max_command_age
            )
            cmd = self.command if command_valid else None
            position = (0.0, 0.0, 0.0) if cmd is None else (
                cmd.position.x, cmd.position.y, cmd.position.z
            )
            velocity = (0.0, 0.0, 0.0) if cmd is None else (
                cmd.velocity.x, cmd.velocity.y, cmd.velocity.z
            )
            normal = (1.0, 0.0, 0.0)
            if cmd is not None:
                candidate = (
                    float(cmd.target_normal.x),
                    float(cmd.target_normal.y),
                    float(cmd.target_normal.z),
                )
                normal_norm = math.sqrt(sum(value * value for value in candidate))
                if math.isfinite(normal_norm) and normal_norm > 1.0e-6:
                    normal = tuple(value / normal_norm for value in candidate)
            values = (
                *self.base,
                self.station[0], self.station[1],
                *position, *velocity, *normal,
                1.0 if cmd is None else float(cmd.time_to_strike),
                1.0 if cmd is None or int(cmd.swing_side) >= 0 else -1.0,
            )
            if not all(math.isfinite(float(v)) for v in values):
                self.get_logger().error("non-finite planner packet rejected")
                return
            self.sequence += 1
            bridge_send_steady_ns = time.monotonic_ns()
            bridge_send_unix_ns = time.time_ns()
            packet = PACKET.pack(
                MAGIC, VERSION, 20, self.sequence, bridge_send_steady_ns,
                bridge_send_unix_ns,
                0 if cmd is None else self.command_header_unix_ns,
                0 if cmd is None else self.command_rx_unix_ns,
                0 if cmd is None else int(cmd.task_id),
                0 if cmd is None else int(cmd.task_revision),
                1 if command_valid else 0,
                *values,
            )
            self.sock.sendto(packet, self.destination)
            self.last_mdu_tx_sequence = self.sequence
            self.last_mdu_tx_unix_ns = bridge_send_unix_ns
            self.last_mdu_tx_task_id = 0 if cmd is None else int(cmd.task_id)
            self.last_mdu_tx_revision = (
                0 if cmd is None else int(cmd.task_revision)
            )
            self.last_mdu_tx_command_valid = command_valid
            if self.last_tx > 0.0:
                self.tx_gap_max_ms = max(
                    self.tx_gap_max_ms, (now - self.last_tx) * 1000.0
                )
            self.last_tx = now
            self.tx_count += 1
            if command_valid:
                self.command_tx_valid_count += 1
            if now - self.last_tx_report >= 2.0:
                self.last_tx_report = now
                self.get_logger().info(
                    f"sender tx={self.tx_count} valid_tx={self.command_tx_valid_count} "
                    f"command_rx={self.command_rx_count} gap_max_ms={self.tx_gap_max_ms:.2f} "
                    f"deadline_miss={self.sender_deadline_misses}"
                )
                self.tx_gap_max_ms = 0.0

        def send_monitor(self, now: float) -> None:
            self.read_joint_packets(now)
            self.read_mdu_status_packets(now)
            if not self.monitor_destinations:
                return
            base_age = now - self.base_rx if self.base is not None else math.inf
            base_fresh = (
                self.base is not None and base_age <= args.tracking_warning_age
            )
            ball_fresh = self.ball is not None and now - self.ball_rx <= 0.25
            ball_age = now - self.ball_rx if self.ball is not None else math.inf
            marker_age = now - self.marker_rx if self.marker_stats is not None else math.inf
            marker_fresh = self.marker_stats is not None and marker_age <= 0.10
            racket_age = now - self.racket_rx if self.racket is not None else math.inf
            racket_fresh = self.racket is not None and racket_age <= 0.10
            command_fresh = (
                self.command is not None and now - self.command_rx <= args.max_command_age
            )
            command_age = now - self.command_rx if self.command is not None else math.inf
            joints_fresh = self.joint_q is not None and now - self.joint_rx <= 0.12
            joints_age = now - self.joint_rx if self.joint_q is not None else math.inf
            cmd = self.command
            payload = {
                "schema": "hope726.telemetry.v1",
                "time_unix_ns": time.time_ns(),
                "base_fresh": base_fresh,
                "base_age_ms": None if not math.isfinite(base_age) else base_age * 1000.0,
                "ball_fresh": ball_fresh,
                "ball_age_ms": None if not math.isfinite(ball_age) else ball_age * 1000.0,
                "marker_fresh": marker_fresh,
                "marker_age_ms": None if not math.isfinite(marker_age) else marker_age * 1000.0,
                "ball_marker": self.marker_stats,
                "racket_fresh": racket_fresh,
                "racket_age_ms": None if not math.isfinite(racket_age) else racket_age * 1000.0,
                "planner_fresh": command_fresh,
                "planner_node_active": self.planner_node_active,
                "planner_input_fresh": ball_fresh,
                "planner_age_ms": None if not math.isfinite(command_age) else command_age * 1000.0,
                "joints_fresh": joints_fresh,
                "joints_age_ms": None if not math.isfinite(joints_age) else joints_age * 1000.0,
                "rigid_body_wxyz": (
                    None if self.rigid_body is None else list(self.rigid_body)
                ),
                "bias_body_xyz": [
                    args.bias_body_x, args.bias_body_y, args.bias_body_z
                ],
                "bias_world_xyz": (
                    None if self.bias_world is None else list(self.bias_world)
                ),
                "fdu_to_pelvis_quat_xyzw": list(fdu_to_pelvis_quat),
                "base_wxyz": None if self.base is None else list(self.base),
                "station_xy": None if self.station is None else list(self.station),
                "ball_xyz": None if self.ball is None else list(self.ball),
                "racket_wxyz": None if self.racket is None else list(self.racket),
                "planner": None if cmd is None else {
                    "task_id": int(cmd.task_id),
                    "revision": int(cmd.task_revision),
                    "swing_side": int(cmd.swing_side),
                    "position_xyz": [cmd.position.x, cmd.position.y, cmd.position.z],
                    "velocity_xyz": [cmd.velocity.x, cmd.velocity.y, cmd.velocity.z],
                    "normal_xyz": [
                        cmd.target_normal.x, cmd.target_normal.y, cmd.target_normal.z
                    ],
                    "time_to_strike": float(cmd.time_to_strike),
                    "capture_header_unix_ns": self.command_header_unix_ns,
                    "bridge_command_rx_unix_ns": self.command_rx_unix_ns,
                },
                "command_pipeline": {
                    "planner_ros_received": self.command is not None,
                    "planner_ros_fresh": command_fresh,
                    "planner_ros_rx_count": self.command_rx_count,
                    "hdu_udp_send_ok": self.last_mdu_tx_sequence > 0,
                    "hdu_udp_tx_count": self.tx_count,
                    "hdu_udp_valid_tx_count": self.command_tx_valid_count,
                    "hdu_udp_last_sequence": self.last_mdu_tx_sequence,
                    "hdu_udp_last_send_unix_ns": self.last_mdu_tx_unix_ns,
                    "hdu_udp_last_task_id": self.last_mdu_tx_task_id,
                    "hdu_udp_last_revision": self.last_mdu_tx_revision,
                    "hdu_udp_last_command_valid": self.last_mdu_tx_command_valid,
                },
                "mdu_command_status_fresh": (
                    self.mdu_status is not None
                    and now - self.mdu_status_rx <= 0.25
                ),
                "mdu_command_status_age_ms": (
                    None if self.mdu_status is None
                    else (now - self.mdu_status_rx) * 1000.0
                ),
                "mdu_command_status_rx_count": self.mdu_status_rx_count,
                "mdu_command_status_invalid_count": self.mdu_status_invalid_count,
                "mdu_command_status": self.mdu_status,
                "joints": None if self.joint_q is None else {
                    "sequence": self.joint_sequence,
                    "names": JOINT_NAMES,
                    "position": self.joint_q,
                    "velocity": self.joint_dq,
                },
            }
            packet = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            for destination in self.monitor_destinations:
                try:
                    self.sock.sendto(packet, destination)
                except OSError as exc:
                    self.get_logger().warning(
                        f"monitor UDP send to {destination[0]}:{destination[1]} failed: {exc}",
                        throttle_duration_sec=2.0,
                    )

        def read_joint_packets(self, now: float) -> None:
            for _ in range(args.max_joint_drain):
                try:
                    packet, _ = self.joint_sock.recvfrom(JOINT_PACKET.size)
                except BlockingIOError:
                    return
                if len(packet) != JOINT_PACKET.size:
                    continue
                decoded = JOINT_PACKET.unpack(packet)
                if decoded[0] != b"HJ31" or decoded[1] != 1 or decoded[2] != 31:
                    continue
                values = decoded[5:]
                if not all(math.isfinite(value) for value in values):
                    continue
                self.joint_sequence = int(decoded[3])
                self.joint_q = list(values[:31])
                self.joint_dq = list(values[31:])
                self.joint_rx = now

        def read_mdu_status_packets(self, now: float) -> None:
            for _ in range(args.max_joint_drain):
                try:
                    packet, source = self.mdu_status_sock.recvfrom(
                        COMMAND_STATUS_PACKET.size
                    )
                except BlockingIOError:
                    return
                if len(packet) != COMMAND_STATUS_PACKET.size:
                    self.mdu_status_invalid_count += 1
                    continue
                decoded = COMMAND_STATUS_PACKET.unpack(packet)
                if decoded[0] != b"HCMD" or decoded[1] != 1 or decoded[2] != COMMAND_STATUS_PACKET.size:
                    self.mdu_status_invalid_count += 1
                    continue
                if not all(math.isfinite(value) for value in decoded[19:21]):
                    self.mdu_status_invalid_count += 1
                    continue
                flags = int(decoded[18])
                status = {
                    "schema": "hope726.command_status.v1",
                    "source_ip": source[0],
                    "status_sequence": int(decoded[3]),
                    "system_time_unix_ns": int(decoded[4]),
                    "planner_packet_sequence": int(decoded[5]),
                    "planner_task_id": int(decoded[6]),
                    "model_packet_sequence": int(decoded[7]),
                    "model_task_id": int(decoded[8]),
                    "model_inference_count": int(decoded[9]),
                    "received_datagram_count": int(decoded[10]),
                    "decoded_packet_count": int(decoded[11]),
                    "invalid_datagram_count": int(decoded[12]),
                    "planner_task_revision": int(decoded[13]),
                    "model_task_revision": int(decoded[14]),
                    "live_mode_code": int(decoded[15]),
                    "live_mode": LIVE_MODES.get(int(decoded[15]), "UNKNOWN"),
                    "lifecycle_phase_code": int(decoded[16]),
                    "lifecycle_phase": LIFECYCLE_PHASES.get(int(decoded[16]), "unknown"),
                    "revision_decision_code": int(decoded[17]),
                    "revision_decision": REVISION_DECISIONS.get(int(decoded[17]), "unknown"),
                    "planner_local_age_ms": float(decoded[19]),
                    "cross_host_packet_age_ms": float(decoded[20]),
                }
                status.update({
                    name: bool(flags & bit)
                    for name, bit in COMMAND_STATUS_FLAGS.items()
                })
                self.mdu_status = status
                self.mdu_status_rx = now
                self.mdu_status_rx_count += 1

    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.stop_sender()
        node.joint_sock.close()
        node.mdu_status_sock.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
