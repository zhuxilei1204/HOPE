#!/usr/bin/env python3
"""Summarize Isaac/MuJoCo wide per-joint action diagnostics by joint and phase."""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--csv-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = pathlib.Path(args.csv_path)
    with source.open(newline="") as fh:
        reader = csv.DictReader(fh)
        joint_names = [
            name.removeprefix("raw__")
            for name in (reader.fieldnames or ())
            if name.startswith("raw__")
        ]
        if not joint_names:
            raise ValueError(f"{source} has no raw__<joint> columns")
        rows = list(reader)

    groups = {
        "waist": joint_names[0:3],
        "right_arm": joint_names[12:19],
        "legs": joint_names[19:31],
    }
    per_joint: dict[str, dict[str, list[float]]] = {
        joint: defaultdict(list) for joint in joint_names
    }
    per_phase_group: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        phase = row.get("phase", "unknown")
        for joint in joint_names:
            stats = per_joint[joint]
            for key in (
                "raw",
                "effective",
                "overflow",
                "position_clamped",
                "tracking_error",
                "torque_requested",
                "torque_applied",
                "torque_clipped",
            ):
                stats[key].append(float(row[f"{key}__{joint}"]))
        for group, names in groups.items():
            stats = per_phase_group[(phase, group)]
            for joint in names:
                stats["clamp"].append(float(row[f"position_clamped__{joint}"]))
                stats["overflow"].append(abs(float(row[f"overflow__{joint}"])))
                stats["tracking"].append(abs(float(row[f"tracking_error__{joint}"])))
                stats["torque_clip"].append(float(row[f"torque_clipped__{joint}"]))

    summaries = []
    for joint, stats in per_joint.items():
        summaries.append(
            {
                "joint": joint,
                "clamp_rate": _mean(stats["position_clamped"]),
                "overflow_mean": _mean([abs(v) for v in stats["overflow"]]),
                "overflow_p95": _percentile([abs(v) for v in stats["overflow"]], 0.95),
                "raw_abs_max": max((abs(v) for v in stats["raw"]), default=0.0),
                "effective_abs_max": max((abs(v) for v in stats["effective"]), default=0.0),
                "tracking_mean": _mean([abs(v) for v in stats["tracking_error"]]),
                "tracking_p95": _percentile([abs(v) for v in stats["tracking_error"]], 0.95),
                "torque_clip_rate": _mean(stats["torque_clipped"]),
            }
        )
    summaries.sort(key=lambda row: (row["clamp_rate"], row["overflow_p95"]), reverse=True)

    lines = [
        f"# Joint action diagnostic: {source.name}",
        "",
        f"Rows: `{len(rows)}`; joints: `{len(joint_names)}`.",
        "",
        "## Per-joint ranking",
        "",
        "| joint | clamp rate | overflow mean | overflow p95 | raw abs max | effective abs max | q tracking mean | q tracking p95 | torque clip rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['joint']} | {row['clamp_rate']:.4f} | {row['overflow_mean']:.4f} | "
            f"{row['overflow_p95']:.4f} | {row['raw_abs_max']:.4f} | "
            f"{row['effective_abs_max']:.4f} | {row['tracking_mean']:.4f} | "
            f"{row['tracking_p95']:.4f} | {row['torque_clip_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Phase/group summary",
            "",
            "| phase | group | clamp rate | overflow mean | overflow p95 | q tracking mean | q tracking p95 | torque clip rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (phase, group), stats in sorted(per_phase_group.items()):
        lines.append(
            f"| {phase} | {group} | {_mean(stats['clamp']):.4f} | "
            f"{_mean(stats['overflow']):.4f} | {_percentile(stats['overflow'], 0.95):.4f} | "
            f"{_mean(stats['tracking']):.4f} | {_percentile(stats['tracking'], 0.95):.4f} | "
            f"{_mean(stats['torque_clip']):.4f} |"
        )

    failure_key = "fallen" if rows and "fallen" in rows[0] else "done"
    first_failure = next(
        (
            row
            for row in rows
            if row.get(failure_key, "") not in ("", "0", "0.0")
        ),
        None,
    )
    lines.extend(["", "## First recorded failure/reset", ""])
    if first_failure is None:
        lines.append("No failure/reset marker was present in this trace.")
    else:
        marker = first_failure.get("sequence_step", first_failure.get("step", "?"))
        lines.append(
            f"`{failure_key}=1` first appears at row step `{marker}`, "
            f"phase `{first_failure.get('phase', 'unknown')}`."
        )

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.markdown_out:
        target = pathlib.Path(args.markdown_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    if args.csv_out:
        target = pathlib.Path(args.csv_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
