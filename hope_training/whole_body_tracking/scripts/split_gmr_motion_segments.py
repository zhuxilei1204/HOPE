#!/usr/bin/env python3
"""Split multi-hit GMR pickle recordings into one-strike motion candidates."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def parse_hits(value: str) -> list[int]:
    hits = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not hits or hits != sorted(set(hits)):
        raise ValueError(f"hit_frames must be sorted unique integers, got {value!r}")
    return hits


def split_bounds(frame_count: int, hits: list[int]) -> list[int]:
    if any(hit < 0 or hit >= frame_count for hit in hits):
        raise ValueError(f"hit frame outside [0, {frame_count}): {hits}")
    return [0, *[(left + right + 1) // 2 for left, right in zip(hits, hits[1:])], frame_count]


def slice_motion(data: dict, start: int, end: int) -> dict:
    output = dict(data)
    for key in ("root_pos", "root_rot", "dof_pos", "local_body_pos"):
        value = data.get(key)
        if value is not None:
            output[key] = np.asarray(value)[start:end].copy()
    return output


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.output_dir / "manifest.tsv"
    with args.recordings.open("r", encoding="utf-8", newline="") as handle:
        recordings = list(csv.DictReader(handle, delimiter="\t"))

    rows = []
    for recording in recordings:
        source = Path(recording["gmr_output"]).expanduser().resolve()
        with source.open("rb") as handle:
            data = pickle.load(handle)
        frame_count = int(np.asarray(data["dof_pos"]).shape[0])
        if int(recording["frames"]) != frame_count:
            raise ValueError(
                f"{recording['name']}: manifest frames={recording['frames']} != GMR {frame_count}"
            )
        hits = parse_hits(recording["hit_frames"])
        bounds = split_bounds(frame_count, hits)
        for index, hit in enumerate(hits):
            start, end = bounds[index], bounds[index + 1]
            name = f"{recording['name']}_hit{index + 1}"
            output = (args.output_dir / f"{name}_agibot_a3.pkl").resolve()
            with output.open("wb") as handle:
                pickle.dump(slice_motion(data, start, end), handle)
            rows.append(
                {
                    "name": name,
                    "output": str(output),
                    "source_video": recording["source_video"],
                    "source_motion": str(source),
                    "source_frame_start": start,
                    "source_frame_end_exclusive": end,
                    "source_strike_frame": hit,
                    "strike_frame": hit - start,
                    "frames": end - start,
                    "fps": float(data["fps"]),
                    "swing_side": recording["swing_side"],
                    "motion_type": recording["motion_type"],
                }
            )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"segments={len(rows)}")
    print(f"manifest={manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
