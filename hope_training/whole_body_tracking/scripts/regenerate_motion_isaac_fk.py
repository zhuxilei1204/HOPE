#!/usr/bin/env python3
"""Validate and regenerate canonical motion body arrays with the Isaac A3 articulation.

Contact-aware IK is solved against the deployment MuJoCo model. Training body
imitation rewards, however, consume Isaac articulation FK. This utility keeps
the root and canonical 31-joint trajectory unchanged, evaluates it in Isaac,
reports MuJoCo/Isaac body FK differences, and writes final NPZ body arrays that
are exactly self-consistent with the training asset.
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
from pathlib import Path
import signal
from typing import Any

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT / "hope_training/motions/supplemental_lower_body_contact_aware_20260728"
)
DEFAULT_OUTPUT = (
    ROOT
    / "hope_training/motions/supplemental_lower_body_contact_aware_isaacfk_20260728"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", default="HOPE-PingPong-AgibotA3-v0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of articulation environments used per FK batch.",
    )
    return parser.parse_args()


def gradient(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=np.float64), dt, axis=0, edge_order=1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    output = q.copy()
    output[..., 1:] *= -1.0
    return output


def angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)
    previous = np.concatenate((q[:1], q[:-1]), axis=0)
    following = np.concatenate((q[1:], q[-1:]), axis=0)
    span = np.full((len(q),), 2.0 * dt)
    span[[0, -1]] = dt
    delta = quat_mul(following, quat_inv(previous))
    delta *= np.where(delta[..., :1] < 0.0, -1.0, 1.0)
    vector = delta[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[..., :1], -1.0, 1.0))
    axis = vector / np.maximum(vector_norm, 1.0e-12)
    result = axis * angle / span[:, None, None]
    result[vector_norm[..., 0] < 1.0e-10] = 0.0
    return result


def orientation_error_deg(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    reference_xyzw = reference[..., [1, 2, 3, 0]].reshape(-1, 4)
    actual_xyzw = actual[..., [1, 2, 3, 0]].reshape(-1, 4)
    error = Rotation.from_quat(actual_xyzw) * Rotation.from_quat(reference_xyzw).inv()
    return np.rad2deg(np.linalg.norm(error.as_rotvec(), axis=1)).reshape(reference.shape[:-1])


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    faulthandler.register(signal.SIGUSR1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields, rows = load_rows(args.input_dir / "manifest.tsv")
    if not rows:
        raise RuntimeError("input manifest has no rows")
    max_frames = max(int(float(row["frames"])) for row in rows)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    batch_size = min(args.batch_size, max_frames)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device).app
    try:
        import gymnasium as gym
        import whole_body_tracking.tasks.tracking  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        from whole_body_tracking.robots.agibot_a3 import A3_TRACKED_BODIES
        from whole_body_tracking.utils.action_adapter_config import (
            load_joint_order,
            resolve_joint_order_mapping,
        )

        env_cfg = parse_env_cfg(
            args.task, device=args.device, num_envs=batch_size
        )
        if hasattr(env_cfg, "events"):
            for name in ("link_mass", "pd_gains"):
                if hasattr(env_cfg.events, name):
                    setattr(env_cfg.events, name, None)
        print(f"creating Isaac FK environment with {batch_size} envs", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        print("resetting Isaac FK environment", flush=True)
        env.reset(seed=0)
        print("Isaac FK environment ready", flush=True)
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        tracked_ids = torch.tensor(
            robot.find_bodies(A3_TRACKED_BODIES, preserve_order=True)[0],
            dtype=torch.long,
            device=robot.device,
        )
        mapping = resolve_joint_order_mapping(
            robot.data.joint_names, canonical_joint_names=load_joint_order()
        )
        canonical_to_art = torch.tensor(
            mapping.canonical_to_articulation, dtype=torch.long, device=robot.device
        )
        env_origins = unwrapped.scene.env_origins

        output_rows: list[dict[str, str]] = []
        all_position_errors = []
        all_orientation_errors = []
        for row in rows:
            print(f"regenerating {row['name']}", flush=True)
            source = Path(row["output"])
            source_data = np.load(source)
            arrays: dict[str, Any] = {
                key: np.asarray(source_data[key]) for key in source_data.files
            }
            joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float32)
            frames = len(joint_pos)
            source_body_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)
            source_body_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)
            isaac_body_pos = np.empty_like(source_body_pos)
            isaac_body_quat = np.empty_like(source_body_quat)
            for start in range(0, frames, batch_size):
                end = min(frames, start + batch_size)
                count = end - start
                env_ids = torch.arange(
                    count, dtype=torch.long, device=robot.device
                )
                root_state = robot.data.default_root_state[:count].clone()
                root_state[:, :3] = (
                    torch.tensor(
                        source_body_pos[start:end, 0],
                        dtype=torch.float32,
                        device=robot.device,
                    )
                    + env_origins[:count]
                )
                root_state[:, 3:7] = torch.tensor(
                    source_body_quat[start:end, 0],
                    dtype=torch.float32,
                    device=robot.device,
                )
                root_state[:, 7:] = 0.0
                articulation_joint = robot.data.default_joint_pos[:count].clone()
                articulation_joint[:, canonical_to_art] = torch.tensor(
                    joint_pos[start:end],
                    dtype=torch.float32,
                    device=robot.device,
                )
                articulation_velocity = torch.zeros_like(articulation_joint)

                robot.write_root_state_to_sim(root_state, env_ids=env_ids)
                robot.write_joint_state_to_sim(
                    articulation_joint, articulation_velocity, env_ids=env_ids
                )
                unwrapped.scene.write_data_to_sim()
                unwrapped.sim.forward()
                unwrapped.scene.update(dt=0.0)

                isaac_body_pos[start:end] = (
                    robot.data.body_pos_w[:count, tracked_ids]
                    - env_origins[:count, None, :]
                ).detach().cpu().numpy().astype(np.float64)
                isaac_body_quat[start:end] = (
                    robot.data.body_quat_w[:count, tracked_ids]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            position_error = np.linalg.norm(
                isaac_body_pos - source_body_pos, axis=-1
            )
            rotation_error = orientation_error_deg(
                source_body_quat, isaac_body_quat
            )
            all_position_errors.append(position_error)
            all_orientation_errors.append(rotation_error)

            fps = float(arrays["fps"])
            dt = 1.0 / fps
            arrays["body_pos_w"] = isaac_body_pos.astype(np.float32)
            arrays["body_quat_w"] = isaac_body_quat.astype(np.float32)
            arrays["body_lin_vel_w"] = gradient(isaac_body_pos, dt).astype(np.float32)
            arrays["body_ang_vel_w"] = angular_velocity(
                isaac_body_quat, dt
            ).astype(np.float32)
            destination = args.output_dir / source.name
            np.savez(destination, **arrays)

            source_yaml = source.with_suffix(".yaml")
            sidecar: dict[str, Any] = {}
            if source_yaml.is_file():
                with source_yaml.open("r", encoding="utf-8") as handle:
                    sidecar = yaml.safe_load(handle) or {}
            sidecar["isaac_fk"] = {
                "method": "direct_articulation_fk_v1",
                "task": args.task,
                "source_motion": str(source.resolve()),
                "body_position_error_p95_m_before_regeneration": float(
                    np.percentile(position_error, 95)
                ),
                "body_orientation_error_p95_deg_before_regeneration": float(
                    np.percentile(rotation_error, 95)
                ),
                "root_and_joint_trajectory_changed": False,
            }
            with destination.with_suffix(".yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(sidecar, handle, sort_keys=False, allow_unicode=False)

            output_row = dict(row)
            output_row["source"] = str(source.resolve())
            output_row["output"] = str(destination.resolve())
            output_row["isaac_fk_pos_error_p95_m"] = (
                f"{np.percentile(position_error, 95):.9f}"
            )
            output_row["isaac_fk_rot_error_p95_deg"] = (
                f"{np.percentile(rotation_error, 95):.9f}"
            )
            output_rows.append(output_row)
            print(
                f"{row['name']}: position_p95={np.percentile(position_error, 95):.5f}m "
                f"rotation_p95={np.percentile(rotation_error, 95):.3f}deg"
            )

        for field in ("isaac_fk_pos_error_p95_m", "isaac_fk_rot_error_p95_deg"):
            if field not in fields:
                fields.append(field)
        write_rows(args.output_dir / "manifest.tsv", fields, output_rows)
        ready_rows = [row for row in output_rows if row["acceptance"] == "pass"]
        write_rows(args.output_dir / "manifest_train_ready.tsv", fields, ready_rows)
        position_error = np.concatenate(all_position_errors)
        rotation_error = np.concatenate(all_orientation_errors)
        summary = {
            "clips": len(output_rows),
            "train_ready": len(ready_rows),
            "body_position_error_p50_m_before_regeneration": float(
                np.percentile(position_error, 50)
            ),
            "body_position_error_p95_m_before_regeneration": float(
                np.percentile(position_error, 95)
            ),
            "body_position_error_max_m_before_regeneration": float(
                np.max(position_error)
            ),
            "body_orientation_error_p50_deg_before_regeneration": float(
                np.percentile(rotation_error, 50)
            ),
            "body_orientation_error_p95_deg_before_regeneration": float(
                np.percentile(rotation_error, 95)
            ),
            "body_orientation_error_max_deg_before_regeneration": float(
                np.max(rotation_error)
            ),
        }
        with (args.output_dir / "isaac_fk_summary.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(summary, handle, sort_keys=False)
        print(summary)
        env.close()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
