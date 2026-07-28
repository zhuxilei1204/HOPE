#!/usr/bin/env python3
"""Add explicit side-conditioned racket target boxes to a motion TSV manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "racket_pos_x_lo",
    "racket_pos_x_hi",
    "racket_pos_y_lo",
    "racket_pos_y_hi",
    "racket_pos_z_lo",
    "racket_pos_z_hi",
    "racket_vel_x_lo",
    "racket_vel_x_hi",
    "racket_vel_y_lo",
    "racket_vel_y_hi",
    "racket_vel_z_lo",
    "racket_vel_z_hi",
)


def _values(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) != len(FIELDS):
        raise argparse.ArgumentTypeError(
            f"expected {len(FIELDS)} comma-separated values, got {len(values)}"
        )
    return values


def _triple(text: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in text.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            f"expected 3 comma-separated values, got {len(values)}"
        )
    if any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("half widths must be non-negative")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forehand", type=_values, required=True)
    parser.add_argument("--backhand", type=_values, required=True)
    parser.add_argument(
        "--center-position-on-strike",
        action="store_true",
        help=(
            "Center each position box on strike_racket_[xyz]_m while retaining "
            "the side-conditioned velocity ranges."
        ),
    )
    parser.add_argument(
        "--static-position-half-widths",
        type=_triple,
        default=(0.08, 0.16, 0.09),
    )
    parser.add_argument(
        "--translation-position-half-widths",
        type=_triple,
        default=(0.11, 0.20, 0.09),
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"manifest has no rows: {args.input}")
    if "swing_side" not in fieldnames:
        raise ValueError(f"manifest has no swing_side column: {args.input}")
    for field in FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    for row in rows:
        side = float(row["swing_side"])
        box = list(args.forehand if side >= 0.0 else args.backhand)
        if args.center_position_on_strike:
            required = (
                "strike_racket_x_m",
                "strike_racket_y_m",
                "strike_racket_z_m",
            )
            missing = [field for field in required if not row.get(field)]
            if missing:
                raise ValueError(
                    f"{row.get('name', '<unnamed>')}: missing strike fields {missing}"
                )
            center = tuple(float(row[field]) for field in required)
            motion_type = row.get("motion_type", "")
            half = (
                args.translation_position_half_widths
                if "translate" in motion_type
                else args.static_position_half_widths
            )
            box[:6] = (
                center[0] - half[0],
                center[0] + half[0],
                center[1] - half[1],
                center[1] + half[1],
                center[2] - half[2],
                center[2] + half[2],
            )
        row.update({field: f"{value:.9f}" for field, value in zip(FIELDS, box)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
