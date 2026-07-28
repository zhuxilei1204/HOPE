#!/usr/bin/env python3
"""Render a canonical HOPE motion NPZ with the shipped A3 MuJoCo model."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = (
    ROOT
    / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
DEFAULT_JOINT_ORDER = ROOT / "hope_training/config/joint_order_agibot_a3.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint-order", type=Path, default=DEFAULT_JOINT_ORDER)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=145.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument("--distance", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.joint_order.open("r", encoding="utf-8") as handle:
        joint_names = list((yaml.safe_load(handle) or {})["joint_order"])
    motion = np.load(args.motion)
    joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)
    fps = float(motion["fps"])

    model = mujoco.MjModel.from_xml_path(str(args.robot_xml))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    root_jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint"
    )
    root_qadr = int(model.jnt_qposadr[root_jid])
    joint_qadr = np.array(
        [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in joint_names
        ],
        dtype=np.int32,
    )

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    camera.distance = args.distance
    center = np.median(body_pos[:, 0], axis=0)
    camera.lookat[:] = [center[0], center[1], 0.85]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=fps, codec="libx264", quality=8)
    try:
        for frame in range(len(joint_pos)):
            data.qpos[:] = model.qpos0
            data.qpos[root_qadr : root_qadr + 3] = body_pos[frame, 0]
            data.qpos[root_qadr + 3 : root_qadr + 7] = body_quat[frame, 0]
            data.qpos[joint_qadr] = joint_pos[frame]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    print(f"frames={len(joint_pos)} fps={fps:.3f} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
