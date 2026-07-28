#!/usr/bin/env python3
"""Add NPZ-derived motion racket offsets to a TSV manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


OFFSET_FIELDS = ("motion_racket_offset_x", "motion_racket_offset_y")


def _vec3(text: str) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split(",")], dtype=np.float64)
    if values.shape != (3,):
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return values


def _quat_apply_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    scalar = float(quat[0])
    axis = np.asarray(quat[1:4], dtype=np.float64)
    return vector + 2.0 * scalar * np.cross(axis, vector) + 2.0 * np.cross(
        axis, np.cross(axis, vector)
    )


def _motion_path(row: dict[str, str], manifest: Path) -> Path:
    value = row.get("output") or row.get("source")
    if not value:
        raise ValueError(f"{row.get('name', '<unnamed>')}: no output/source motion path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-body-index", type=int, default=0)
    parser.add_argument("--wrist-body-index", type=int, default=13)
    parser.add_argument("--mount-offset", type=_vec3, default=np.array([0.21, 0.032, 0.032]))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"manifest has no rows: {args.input}")
    for field in OFFSET_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    for row in rows:
        path = _motion_path(row, args.input)
        with np.load(path) as motion:
            body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
            body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)
            if "strike_frame" in motion:
                strike_frame = int(np.asarray(motion["strike_frame"]).item())
            elif row.get("strike_frame"):
                strike_frame = int(float(row["strike_frame"]))
            else:
                raise ValueError(f"{path}: no strike_frame in NPZ or manifest")
        if not 0 <= strike_frame < body_pos.shape[0]:
            raise ValueError(f"{path}: invalid strike frame {strike_frame}")
        if max(args.root_body_index, args.wrist_body_index) >= body_pos.shape[1]:
            raise ValueError(
                f"{path}: body count {body_pos.shape[1]} does not contain requested indices"
            )
        root = body_pos[strike_frame, args.root_body_index]
        wrist = body_pos[strike_frame, args.wrist_body_index]
        wrist_quat = body_quat[strike_frame, args.wrist_body_index]
        racket = wrist + _quat_apply_wxyz(wrist_quat, args.mount_offset)
        offset = racket[:2] - root[:2]
        row["motion_racket_offset_x"] = f"{offset[0]:.9f}"
        row["motion_racket_offset_y"] = f"{offset[1]:.9f}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
