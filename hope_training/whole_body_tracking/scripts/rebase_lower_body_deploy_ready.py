#!/usr/bin/env python3
"""Rebase contact-aware A3 motions onto a deploy-ready lower-body posture.

The source motion contributes only temporal root and leg residuals around its
initial ready window. Waist, head, and arm joint trajectories remain bitwise
unchanged. The rebased trajectory is then solved again for stored support-foot
contacts, COM support, ground alignment, and joint limits.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from contact_aware_retarget import (
    A3Kinematics,
    ClipInput,
    acceptance,
    correct_clip,
    load_joint_names,
    write_npz,
    write_tsv,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT
    / "hope_training/motions/"
    "supplemental_lower_body_contact_aware_isaacfk_20260728"
)
DEFAULT_XML = (
    ROOT
    / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3_pingpong/a3_pingpong.xml"
)
DEFAULT_JOINT_ORDER = ROOT / "hope_training/config/joint_order_agibot_a3.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--joint-order", type=Path, default=DEFAULT_JOINT_ORDER)
    parser.add_argument(
        "--ready-blend",
        type=float,
        required=True,
        help="0 keeps the source ready baseline; 1 uses the XML deploy-ready baseline.",
    )
    parser.add_argument("--ready-frames-s", type=float, default=0.40)
    parser.add_argument("--ground-z", type=float, default=0.001)
    parser.add_argument("--swing-clearance-m", type=float, default=0.030)
    parser.add_argument("--support-margin-m", type=float, default=0.005)
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--correction-smoothing-frames", type=int, default=11)
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def rotation_blend(source: Rotation, target: Rotation, amount: float) -> Rotation:
    delta = target * source.inv()
    return Rotation.from_rotvec(amount * delta.as_rotvec()) * source


def rebase_clip(
    model: A3Kinematics,
    source: ClipInput,
    ready_blend: float,
    ready_frames_s: float,
) -> tuple[ClipInput, dict[str, float]]:
    frames = max(1, min(len(source.root_pos), int(round(ready_frames_s * source.fps))))
    leg_ids = model.leg_canonical
    source_leg_ready = np.median(source.joint_pos[:frames, leg_ids], axis=0)
    default_leg = model.default_qpos[model.leg_qadr]
    target_leg_ready = (
        (1.0 - ready_blend) * source_leg_ready + ready_blend * default_leg
    )

    joint_pos = source.joint_pos.copy()
    leg_residual = source.joint_pos[:, leg_ids] - source_leg_ready[None, :]
    joint_pos[:, leg_ids] = np.clip(
        target_leg_ready[None, :] + leg_residual,
        model.leg_lower + model.leg_safety_margin,
        model.leg_upper - model.leg_safety_margin,
    )

    source_root_ready = np.median(source.root_pos[:frames], axis=0)
    default_root = model.default_qpos[model.root_qadr : model.root_qadr + 3]
    target_root_ready = (
        (1.0 - ready_blend) * source_root_ready + ready_blend * default_root
    )
    root_pos = target_root_ready[None, :] + (
        source.root_pos - source_root_ready[None, :]
    )

    source_rotation = Rotation.from_quat(
        source.root_quat_wxyz[:, [1, 2, 3, 0]]
    )
    source_ready_rotation = source_rotation[:frames].mean()
    default_quat = model.default_qpos[
        model.root_qadr + 3 : model.root_qadr + 7
    ]
    default_rotation = Rotation.from_quat(default_quat[[1, 2, 3, 0]])
    target_ready_rotation = rotation_blend(
        source_ready_rotation, default_rotation, ready_blend
    )
    local_rotation = source_ready_rotation.inv() * source_rotation
    root_xyzw = (target_ready_rotation * local_rotation).as_quat()
    root_quat = root_xyzw[:, [3, 0, 1, 2]]

    metadata = {
        "source_ready_leg_rms_from_default_rad": float(
            np.sqrt(np.mean(np.square(source_leg_ready - default_leg)))
        ),
        "target_ready_leg_rms_from_default_rad": float(
            np.sqrt(np.mean(np.square(target_leg_ready - default_leg)))
        ),
        "source_ready_root_z_m": float(source_root_ready[2]),
        "target_ready_root_z_before_ground_ik_m": float(target_root_ready[2]),
    }
    return ClipInput(source.fps, root_pos, root_quat, joint_pos), metadata


def signed_pitch_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_euler(
        "xyz", degrees=True
    )[:, 1]


def racket_positions(
    model: A3Kinematics, clip: ClipInput
) -> np.ndarray:
    site_id = mujoco.mj_name2id(
        model.model, mujoco.mjtObj.mjOBJ_SITE, "right_racket"
    )
    if site_id < 0:
        raise ValueError("right_racket site is missing from the A3 model")
    positions = np.empty((len(clip.root_pos), 3), dtype=np.float64)
    for frame in range(len(clip.root_pos)):
        model.set_state(
            clip.root_pos[frame],
            clip.root_quat_wxyz[frame],
            clip.joint_pos[frame],
        )
        positions[frame] = model.data.site_xpos[site_id]
    return positions


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.ready_blend <= 1.0:
        raise ValueError("--ready-blend must be within [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fields, rows = load_rows(args.input_dir / "manifest_train_ready.tsv")
    if not rows:
        raise RuntimeError("input manifest has no train-ready rows")
    model = A3Kinematics(
        args.robot_xml, load_joint_names(args.joint_order)
    )

    output_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        source_path = Path(row["output"])
        source_data = np.load(source_path)
        source_clip = ClipInput(
            float(source_data["fps"]),
            np.asarray(source_data["body_pos_w"][:, 0], dtype=np.float64),
            np.asarray(source_data["body_quat_w"][:, 0], dtype=np.float64),
            np.asarray(source_data["joint_pos"], dtype=np.float64),
        )
        contacts = np.column_stack(
            (
                np.asarray(source_data["left_foot_contact"], dtype=bool),
                np.asarray(source_data["right_foot_contact"], dtype=bool),
            )
        )
        strike_frame = int(source_data["strike_frame"])
        rebased, rebase_metrics = rebase_clip(
            model, source_clip, args.ready_blend, args.ready_frames_s
        )
        corrected, trajectory, contact, summary = correct_clip(
            model,
            rebased,
            args.ground_z,
            contact_height=0.040,
            contact_speed=0.10,
            swing_clearance=args.swing_clearance_m,
            support_margin=args.support_margin_m,
            max_iterations=args.max_iterations,
            correction_smoothing_frames=args.correction_smoothing_frames,
            contact_override=contacts,
        )

        racket = racket_positions(model, corrected)
        pelvis_pitch = signed_pitch_deg(corrected.root_quat_wxyz)
        torso_pitch = signed_pitch_deg(trajectory.body_quat[:, 7])
        ready_frames = max(
            1,
            min(
                len(corrected.root_pos),
                int(round(args.ready_frames_s * corrected.fps)),
            ),
        )
        default_leg = model.default_qpos[model.leg_qadr]
        final_metrics = {
            **summary,
            **rebase_metrics,
            "final_ready_leg_rms_from_default_rad": float(
                np.sqrt(
                    np.mean(
                        np.square(
                            corrected.joint_pos[:ready_frames, model.leg_canonical]
                            - default_leg[None, :]
                        )
                    )
                )
            ),
            "final_ready_root_z_m": float(
                np.median(corrected.root_pos[:ready_frames, 2])
            ),
            "strike_racket_x_m": float(racket[strike_frame, 0]),
            "strike_racket_y_m": float(racket[strike_frame, 1]),
            "strike_racket_z_m": float(racket[strike_frame, 2]),
            "strike_pelvis_pitch_deg": float(pelvis_pitch[strike_frame]),
            "strike_torso_pitch_deg": float(torso_pitch[strike_frame]),
        }

        destination = (
            args.output_dir / f"{row['name']}_deploy_ready_residual.npz"
        )
        write_npz(destination, corrected, trajectory, contact, strike_frame)
        tier = row.get("train_tier", "tier1")
        status, reasons = acceptance(summary, tier)
        sidecar = {
            "name": row["name"],
            "source_motion": str(source_path.resolve()),
            "fps": corrected.fps,
            "frame_count": len(corrected.root_pos),
            "strike_frame": strike_frame,
            "strike_phase": strike_frame / max(len(corrected.root_pos) - 1, 1),
            "swing_side": int(float(row["swing_side"])),
            "motion_type": row.get("motion_type"),
            "train_tier": tier,
            "acceptance": status,
            "review_reasons": reasons,
            "retarget": {
                "method": "a3_deploy_ready_lower_body_temporal_residual_v1",
                "ready_blend": args.ready_blend,
                "stored_support_contacts_preserved": True,
                "ground_and_com_ik_reapplied": True,
                "upper_body_joint_trajectory_preserved": True,
            },
            "metrics": final_metrics,
        }
        with destination.with_suffix(".yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            yaml.safe_dump(sidecar, handle, sort_keys=False, allow_unicode=False)

        output_row: dict[str, Any] = dict(row)
        output_row.update(
            {
                "source": str(source_path.resolve()),
                "output": str(destination.resolve()),
                "acceptance": status,
                "review_reasons": ",".join(reasons),
                "ready_blend": f"{args.ready_blend:.6f}",
            }
        )
        output_row.update(
            {
                key: f"{value:.9f}"
                for key, value in final_metrics.items()
                if key not in ("frames", "fps")
            }
        )
        output_rows.append(output_row)
        summaries.append(
            {
                "name": row["name"],
                "status": status,
                "review_reasons": reasons,
                "ready_blend": args.ready_blend,
                **final_metrics,
            }
        )
        print(
            f"{row['name']}: {status} "
            f"root_z={final_metrics['final_ready_root_z_m']:.3f} "
            f"racket_z={final_metrics['strike_racket_z_m']:.3f} "
            f"torso_pitch={final_metrics['strike_torso_pitch_deg']:.1f}deg "
            f"slip_p95={summary['contact_foot_speed_p95_m_s']:.4f}"
        )

    output_fields = list(dict.fromkeys(fields + list(output_rows[0].keys())))
    write_tsv(args.output_dir / "manifest.tsv", output_fields, output_rows)
    ready_rows = [row for row in output_rows if row["acceptance"] == "pass"]
    write_tsv(
        args.output_dir / "manifest_train_ready.tsv",
        output_fields,
        ready_rows,
    )
    with (args.output_dir / "rebase_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summaries, handle, indent=2)
    print(f"train_ready={len(ready_rows)}/{len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
