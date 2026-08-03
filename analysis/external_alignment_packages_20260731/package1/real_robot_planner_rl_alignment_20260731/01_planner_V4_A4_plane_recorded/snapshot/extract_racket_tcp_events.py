#!/usr/bin/env python3
"""Extract synchronized ball/planner/physical-racket TCP diagnostics from an HDU rosbag.

This is read-only post-processing.  It never publishes a ROS message.  The Motive
rigid-body origin-to-paddle-center offset and paddle normal axis are explicit so
that raw recordings stay useful even before the physical calibration is final.
"""

import argparse
import bisect
import csv
import json
import math
import os
from collections import defaultdict

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPICS = ("/ball/point", "/FDU_pai/pose", "/racket/command")


def stamp_ns(header):
    stamp = header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


def parse_vec(text, name):
    try:
        out = tuple(float(x.strip()) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("%s must be x,y,z" % name) from exc
    if len(out) != 3 or not all(math.isfinite(x) for x in out):
        raise argparse.ArgumentTypeError("%s must contain three finite values" % name)
    return out


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def normalize(v):
    length = norm(v)
    if length < 1.0e-9:
        raise ValueError("normal axis cannot be zero")
    return tuple(x / length for x in v)


def add(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def scale(v, k):
    return tuple(x * k for x in v)


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def quat_normalize(q):
    length = math.sqrt(sum(x * x for x in q))
    if length < 1.0e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(x / length for x in q)


def quat_rotate(q, v):
    x, y, z, w = quat_normalize(q)
    return (
        (1 - 2*y*y - 2*z*z) * v[0] + (2*x*y - 2*z*w) * v[1] + (2*x*z + 2*y*w) * v[2],
        (2*x*y + 2*z*w) * v[0] + (1 - 2*x*x - 2*z*z) * v[1] + (2*y*z - 2*x*w) * v[2],
        (2*x*z - 2*y*w) * v[0] + (2*y*z + 2*x*w) * v[1] + (1 - 2*x*x - 2*y*y) * v[2],
    )


def nlerp_quat(a, b, u):
    if sum(a[i] * b[i] for i in range(4)) < 0.0:
        b = tuple(-x for x in b)
    return quat_normalize(tuple(a[i] + u * (b[i] - a[i]) for i in range(4)))


def interp_pose(poses, pose_times, when_ns, max_gap_ns):
    idx = bisect.bisect_left(pose_times, when_ns)
    if idx == 0 or idx >= len(poses):
        return None
    a, b = poses[idx - 1], poses[idx]
    if b["t_ns"] - a["t_ns"] > max_gap_ns:
        return None
    if when_ns - a["t_ns"] > max_gap_ns or b["t_ns"] - when_ns > max_gap_ns:
        return None
    u = (when_ns - a["t_ns"]) / float(b["t_ns"] - a["t_ns"] or 1)
    pos = tuple(a["pos"][i] + u * (b["pos"][i] - a["pos"][i]) for i in range(3))
    quat = nlerp_quat(a["quat"], b["quat"], u)
    return pos, quat, min(when_ns - a["t_ns"], b["t_ns"] - when_ns)


def bag_storage_id(bag):
    metadata = os.path.join(bag, "metadata.yaml")
    if os.path.isfile(metadata):
        with open(metadata, "r", encoding="utf-8") as handle:
            text = handle.read()
        if "storage_identifier: mcap" in text:
            return "mcap"
        if "storage_identifier: sqlite3" in text:
            return "sqlite3"
    return "mcap"


def read_bag(bag):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=bag_storage_id(bag)),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {row.name: row.type for row in reader.get_all_topics_and_types()}
    missing = [topic for topic in TOPICS if topic not in types]
    if missing:
        raise RuntimeError("bag missing required topics: " + ", ".join(missing))
    msg_types = {topic: get_message(types[topic]) for topic in TOPICS}
    balls, poses, commands = [], [], []
    while reader.has_next():
        topic, raw, bag_ns = reader.read_next()
        if topic not in msg_types:
            continue
        msg = deserialize_message(raw, msg_types[topic])
        source_ns = stamp_ns(msg.header) or int(bag_ns)
        base = {"t_ns": source_ns, "bag_ns": int(bag_ns), "header_ns": stamp_ns(msg.header) or 0}
        if topic == "/ball/point":
            base["pos"] = (float(msg.point.x), float(msg.point.y), float(msg.point.z))
            balls.append(base)
        elif topic == "/FDU_pai/pose":
            p, q = msg.pose.position, msg.pose.orientation
            base["pos"] = (float(p.x), float(p.y), float(p.z))
            base["quat"] = (float(q.x), float(q.y), float(q.z), float(q.w))
            poses.append(base)
        else:
            base.update({
                "task": int(msg.task_id), "revision": int(msg.task_revision),
                # Current ROS RacketCommand has no command_valid field: the
                # existence of a published command is itself the valid signal.
                "command_valid": bool(getattr(msg, "command_valid", True)),
                "tts": float(msg.time_to_strike),
                "target": (float(msg.position.x), float(msg.position.y), float(msg.position.z)),
                "target_velocity": (float(msg.velocity.x), float(msg.velocity.y), float(msg.velocity.z)),
                "target_normal": (float(msg.target_normal.x), float(msg.target_normal.y), float(msg.target_normal.z)),
            })
            base["strike_ns"] = source_ns + int(base["tts"] * 1.0e9)
            commands.append(base)
    balls.sort(key=lambda x: x["t_ns"])
    poses.sort(key=lambda x: x["t_ns"])
    commands.sort(key=lambda x: x["t_ns"])
    return balls, poses, commands


SAMPLE_COLUMNS = [
    "time_ns", "bag_time_ns", "ball_header_ns", "pose_sync_error_ms",
    "task_id", "revision", "command_valid", "planner_tts_at_sample_s", "planner_strike_time_ns",
    "ball_x", "ball_y", "ball_z", "planner_x", "planner_y", "planner_z",
    "planner_normal_x", "planner_normal_y", "planner_normal_z",
    "rigid_body_x", "rigid_body_y", "rigid_body_z", "rigid_qx", "rigid_qy", "rigid_qz", "rigid_qw",
    "tcp_x", "tcp_y", "tcp_z", "normal_x", "normal_y", "normal_z",
    "ball_tcp_dx", "ball_tcp_dy", "ball_tcp_dz", "ball_tcp_distance_m",
    "normal_distance_m", "tangent_distance_m", "contact_candidate",
]


def produce_rows(balls, poses, commands, offset_body, normal_body, max_pose_gap_s,
                 contact_plane_m, contact_radius_m):
    pose_times = [row["t_ns"] for row in poses]
    valid_commands = [row for row in commands if row["command_valid"] and row["task"] > 0]
    command_times = [row["t_ns"] for row in valid_commands]
    max_gap_ns = int(max_pose_gap_s * 1.0e9)
    rows = []
    by_task = defaultdict(list)
    for ball in balls:
        pose = interp_pose(poses, pose_times, ball["t_ns"], max_gap_ns)
        if pose is None:
            continue
        idx = bisect.bisect_right(command_times, ball["t_ns"]) - 1
        command = valid_commands[idx] if idx >= 0 else None
        rigid_pos, quat, sync_ns = pose
        tcp = add(rigid_pos, quat_rotate(quat, offset_body))
        normal_world = normalize(quat_rotate(quat, normal_body))
        delta = sub(ball["pos"], tcp)
        normal_distance = dot(delta, normal_world)
        tangent = sub(delta, scale(normal_world, normal_distance))
        tangent_distance = norm(tangent)
        contact = abs(normal_distance) <= contact_plane_m and tangent_distance <= contact_radius_m
        row = {
            "time_ns": ball["t_ns"], "bag_time_ns": ball["bag_ns"], "ball_header_ns": ball["header_ns"],
            "pose_sync_error_ms": sync_ns / 1.0e6,
            "task_id": command["task"] if command else 0,
            "revision": command["revision"] if command else -1,
            "command_valid": int(bool(command)),
            "planner_tts_at_sample_s": (command["strike_ns"] - ball["t_ns"]) / 1.0e9 if command else math.nan,
            "planner_strike_time_ns": command["strike_ns"] if command else 0,
            "ball_x": ball["pos"][0], "ball_y": ball["pos"][1], "ball_z": ball["pos"][2],
            "planner_x": command["target"][0] if command else math.nan,
            "planner_y": command["target"][1] if command else math.nan,
            "planner_z": command["target"][2] if command else math.nan,
            "planner_normal_x": command["target_normal"][0] if command else math.nan,
            "planner_normal_y": command["target_normal"][1] if command else math.nan,
            "planner_normal_z": command["target_normal"][2] if command else math.nan,
            "rigid_body_x": rigid_pos[0], "rigid_body_y": rigid_pos[1], "rigid_body_z": rigid_pos[2],
            "rigid_qx": quat[0], "rigid_qy": quat[1], "rigid_qz": quat[2], "rigid_qw": quat[3],
            "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
            "normal_x": normal_world[0], "normal_y": normal_world[1], "normal_z": normal_world[2],
            "ball_tcp_dx": delta[0], "ball_tcp_dy": delta[1], "ball_tcp_dz": delta[2],
            "ball_tcp_distance_m": norm(delta), "normal_distance_m": normal_distance,
            "tangent_distance_m": tangent_distance, "contact_candidate": int(contact),
        }
        rows.append(row)
        if command:
            # Only associate samples reasonably close to this task's predicted strike.
            if abs(row["planner_tts_at_sample_s"]) <= 0.45:
                by_task[command["task"]].append(row)
    return rows, by_task


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="rosbag directory (normally SESSION/planner)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tcp-offset-body", default="0,0,0",
                        help="Motive rigid-body origin to physical paddle center, body frame, metres")
    parser.add_argument("--normal-axis-body", default="0,1,0",
                        help="physical paddle front normal in Motive rigid-body frame")
    parser.add_argument("--calibration-id", default="INFERRED_LOCAL_Y_UNVERIFIED_TCP_ORIGIN")
    parser.add_argument("--max-pose-gap-s", type=float, default=0.05)
    parser.add_argument("--contact-plane-m", type=float, default=0.035)
    parser.add_argument("--contact-radius-m", type=float, default=0.10)
    args = parser.parse_args()

    offset = parse_vec(args.tcp_offset_body, "tcp-offset-body")
    normal_body = normalize(parse_vec(args.normal_axis_body, "normal-axis-body"))
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.bag))
    os.makedirs(output_dir, exist_ok=True)
    balls, poses, commands = read_bag(args.bag)
    rows, by_task = produce_rows(balls, poses, commands, offset, normal_body,
                                 args.max_pose_gap_s, args.contact_plane_m, args.contact_radius_m)
    samples_path = os.path.join(output_dir, "racket_tcp_samples.csv")
    events_path = os.path.join(output_dir, "racket_tcp_events.csv")
    metadata_path = os.path.join(output_dir, "racket_tcp_metadata.json")
    write_csv(samples_path, SAMPLE_COLUMNS, rows)

    events = []
    for task, task_rows in sorted(by_task.items()):
        # Geometry-first closest approach; contact_candidate is an observable label,
        # not a claim that a force sensor detected impact.
        chosen = min(task_rows, key=lambda row: row["ball_tcp_distance_m"])
        event = dict(chosen)
        event["event_type"] = "contact_candidate" if chosen["contact_candidate"] else "closest_approach"
        event["planner_error_x"] = chosen["ball_x"] - chosen["planner_x"]
        event["planner_error_y"] = chosen["ball_y"] - chosen["planner_y"]
        event["planner_error_z"] = chosen["ball_z"] - chosen["planner_z"]
        event["tcp_error_x"] = chosen["tcp_x"] - chosen["planner_x"]
        event["tcp_error_y"] = chosen["tcp_y"] - chosen["planner_y"]
        event["tcp_error_z"] = chosen["tcp_z"] - chosen["planner_z"]
        planner_normal = (chosen["planner_normal_x"], chosen["planner_normal_y"], chosen["planner_normal_z"])
        actual_normal = (chosen["normal_x"], chosen["normal_y"], chosen["normal_z"])
        try:
            cos_angle = max(-1.0, min(1.0, dot(normalize(planner_normal), normalize(actual_normal))))
            event["normal_error_deg"] = math.degrees(math.acos(cos_angle))
        except ValueError:
            event["normal_error_deg"] = math.nan
        events.append(event)
    event_columns = ["event_type"] + SAMPLE_COLUMNS + [
        "planner_error_x", "planner_error_y", "planner_error_z",
        "tcp_error_x", "tcp_error_y", "tcp_error_z", "normal_error_deg",
    ]
    write_csv(events_path, event_columns, events)
    metadata = {
        "bag": os.path.abspath(args.bag), "calibration_id": args.calibration_id,
        "tcp_offset_body_m": offset, "normal_axis_body": normal_body,
        "contact_candidate_definition": {
            "abs_plane_distance_max_m": args.contact_plane_m,
            "tangent_distance_max_m": args.contact_radius_m,
            "note": "geometric candidate only; no force/contact sensor",
        },
        "counts": {"ball": len(balls), "racket_pose": len(poses), "planner_command": len(commands),
                   "synchronized_samples": len(rows), "task_events": len(events)},
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("racket_tcp_extract samples=%d events=%d output=%s calibration=%s" %
          (len(rows), len(events), output_dir, args.calibration_id))


if __name__ == "__main__":
    main()
