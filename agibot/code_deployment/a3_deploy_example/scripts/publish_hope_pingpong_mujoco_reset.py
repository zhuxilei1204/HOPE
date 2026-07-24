#!/usr/bin/env python3
"""Publish a MuJoCo SimReset for the HOPE/A3 pingpong standing pose."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import rclpy
import yaml
from geometry_msgs.msg import Pose, Twist
from mujoco_sim_msgs.msg import SimReset
from sensor_msgs.msg import JointState


HOPE_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
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
]


def load_default_q(adapter_path: Path) -> list[float]:
    data = yaml.safe_load(adapter_path.read_text(encoding="utf-8"))
    default_q = data.get("default_q", {})
    missing = [name for name in HOPE_JOINT_NAMES if name not in default_q]
    if missing:
        raise ValueError(f"default_q missing joints: {missing}")
    return [float(default_q[name]) for name in HOPE_JOINT_NAMES]


def make_absolute_reset(default_q: list[float], height: float) -> SimReset:
    msg = SimReset()
    msg.mode = SimReset.MODE_ABSOLUTE
    msg.keyframe_id = 0

    msg.set_base = True
    msg.pelvis_pose = Pose()
    msg.pelvis_pose.position.x = 0.0
    msg.pelvis_pose.position.y = 0.0
    msg.pelvis_pose.position.z = float(height)
    msg.pelvis_pose.orientation.w = 1.0
    msg.pelvis_pose.orientation.x = 0.0
    msg.pelvis_pose.orientation.y = 0.0
    msg.pelvis_pose.orientation.z = 0.0

    msg.set_base_twist = True
    msg.pelvis_twist = Twist()

    msg.set_joints = True
    msg.joint_state = JointState()
    msg.joint_state.name = list(HOPE_JOINT_NAMES)
    msg.joint_state.position = list(default_q)
    msg.joint_state.velocity = [0.0] * len(default_q)

    msg.zero_all_velocities = True
    msg.clear_ctrl = True
    return msg


def make_keyframe_reset(keyframe_id: int) -> SimReset:
    msg = SimReset()
    msg.mode = SimReset.MODE_KEYFRAME
    msg.keyframe_id = int(keyframe_id)
    msg.set_base = False
    msg.set_base_twist = False
    msg.set_joints = False
    msg.zero_all_velocities = True
    msg.clear_ctrl = True
    return msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--topic", default="/sim/a3/reset")
    parser.add_argument("--mode", choices=("keyframe", "absolute"), default="keyframe")
    parser.add_argument("--keyframe-id", type=int, default=0)
    parser.add_argument("--height", type=float, default=1.3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--wait-subscriber-s", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "absolute":
        if args.adapter is None:
            raise ValueError("--adapter is required for --mode absolute")
        default_q = load_default_q(args.adapter)
        msg = make_absolute_reset(default_q, args.height)
    else:
        msg = make_keyframe_reset(args.keyframe_id)

    rclpy.init()
    node = rclpy.create_node("hope_pingpong_mujoco_reset")
    publisher = node.create_publisher(SimReset, args.topic, 10)
    deadline = time.monotonic() + max(args.wait_subscriber_s, 0.0)
    while args.wait_subscriber_s > 0.0 and publisher.get_subscription_count() == 0:
        if time.monotonic() >= deadline:
            break
        rclpy.spin_once(node, timeout_sec=0.05)
    period = 1.0 / max(args.rate_hz, 1.0)
    for _ in range(max(args.repeat, 1)):
        msg.header.stamp = node.get_clock().now().to_msg()
        publisher.publish(msg)
        rclpy.spin_once(node, timeout_sec=period)
    subscriber_count = publisher.get_subscription_count()
    node.destroy_node()
    rclpy.shutdown()
    if args.mode == "absolute":
        detail = f"height={args.height}"
    else:
        detail = f"keyframe_id={args.keyframe_id}"
    print(
        f"[hope-mujoco-reset] published {max(args.repeat, 1)} {args.mode} reset "
        f"messages to {args.topic} {detail} subscribers={subscriber_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
