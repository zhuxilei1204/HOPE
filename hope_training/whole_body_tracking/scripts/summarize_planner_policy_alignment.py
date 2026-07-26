"""Summarize planner-command vs policy-execution alignment from MuJoCo contact diagnostics.

The MuJoCo evaluator writes one row per serve with planner targets, achieved
racket state at contact, real post-contact ball velocity, and task outcome. This
script turns that CSV into stable per-run metrics so we can separate:

* planner/command quality: target position, target velocity, target normal;
* policy execution quality: achieved racket position, timing, velocity, normal;
* collision outcome quality: real post-contact ball velocity and return result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _int_bool(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _vec(row: dict[str, str], prefix: str) -> tuple[float, float, float] | None:
    vals = [_float(row, f"{prefix}_{axis}") for axis in ("x", "y", "z")]
    if any(v is None for v in vals):
        return None
    return vals[0], vals[1], vals[2]  # type: ignore[return-value]


def _norm(v: Iterable[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _diff_norm(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    return _norm((a[0] - b[0], a[1] - b[1], a[2] - b[2]))


def _angle_deg(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    na = _norm(a)
    nb = _norm(b)
    if na < 1.0e-9 or nb < 1.0e-9:
        return None
    dot = sum(float(x) * float(y) for x, y in zip(a, b)) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _stats(values: list[float]) -> dict[str, float | int | None]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    p90_idx = min(len(vals) - 1, math.ceil(0.90 * len(vals)) - 1)
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": median(vals),
        "p90": vals[p90_idx],
        "min": vals[0],
        "max": vals[-1],
    }


def _rate(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return sum(int(bool(r.get(key, 0))) for r in rows) / len(rows)


def _dominant_failure(rows: list[dict[str, str]]) -> str | None:
    counts: dict[str, int] = {}
    for row in rows:
        if _int_bool(row, "success"):
            continue
        reason = row.get("miss_reason") or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _with_threshold(values: list[float], threshold: float, *, abs_value: bool = False) -> float | None:
    vals = [abs(v) if abs_value else v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return sum(v <= threshold for v in vals) / len(vals)


def summarize_rows(
    rows: list[dict[str, str]],
    *,
    pos_threshold: float,
    time_threshold: float,
    vel_threshold: float,
    vel_angle_threshold: float,
    outgoing_threshold: float,
    outgoing_angle_threshold: float,
    normal_angle_threshold: float,
) -> dict:
    attempts = len(rows)
    contact_rows = [r for r in rows if (r.get("contact_kind") or "none") != "none"]
    real_contact_rows = [r for r in rows if (r.get("contact_kind") or "none") == "real"]

    derived_rows = []
    for row in rows:
        target_pos = _vec(row, "target_pos")
        contact_pos = _vec(row, "contact_pos")
        racket_pos = _vec(row, "racket_post_pos")
        ball_post_vel = _vec(row, "ball_post_vel")
        target_outgoing = _vec(row, "target_outgoing_vel")
        racket_normal = _vec(row, "racket_normal")
        target_normal = _vec(row, "target_normal")
        contact_normal = _vec(row, "mujoco_contact_normal")

        target_speed = _float(row, "racket_target_speed")
        racket_speed_post = _float(row, "racket_speed_post")
        speed_ratio = None
        if target_speed is not None and target_speed > 1.0e-6 and racket_speed_post is not None:
            speed_ratio = racket_speed_post / target_speed

        derived_rows.append(
            {
                "racket_target_pos_error_m": _diff_norm(racket_pos, target_pos),
                "contact_target_pos_error_m": _diff_norm(contact_pos, target_pos),
                "abs_time_to_strike_cmd_s": None
                if _float(row, "time_to_strike_cmd") is None
                else abs(_float(row, "time_to_strike_cmd")),  # type: ignore[arg-type]
                "racket_vel_target_error_mps": _float(row, "racket_vel_target_error"),
                "racket_vel_target_angle_deg": _float(row, "racket_vel_target_angle_deg"),
                "racket_speed_ratio": speed_ratio,
                "ball_post_target_outgoing_error_mps": _diff_norm(ball_post_vel, target_outgoing),
                "ball_post_target_outgoing_angle_deg": _angle_deg(ball_post_vel, target_outgoing),
                "racket_normal_target_angle_deg": _angle_deg(racket_normal, target_normal),
                "contact_normal_target_angle_deg": _angle_deg(contact_normal, target_normal),
                "closing_speed_pre_mps": _float(row, "closing_speed_pre"),
                "outgoing_normal_speed_post_mps": _float(row, "outgoing_normal_speed_post"),
                "min_ball_racket_distance_m": _float(row, "min_ball_racket_distance"),
                "raw_action_max": _float(row, "raw_action_max_at_contact"),
                "applied_action_max": _float(row, "applied_action_max_at_contact"),
                "q_des_max_rad": _float(row, "q_des_max_at_contact"),
            }
        )

    def values(key: str, source_rows: list[dict] | None = None) -> list[float]:
        selected = source_rows if source_rows is not None else derived_rows
        return [r[key] for r in selected if r.get(key) is not None]

    contact_derived = [d for d, r in zip(derived_rows, rows) if (r.get("contact_kind") or "none") != "none"]
    real_contact_derived = [d for d, r in zip(derived_rows, rows) if (r.get("contact_kind") or "none") == "real"]

    return {
        "attempts": attempts,
        "success_rate": None if attempts == 0 else sum(_int_bool(r, "success") for r in rows) / attempts,
        "contact_rate": None if attempts == 0 else len(contact_rows) / attempts,
        "real_contact_rate": None if attempts == 0 else len(real_contact_rows) / attempts,
        "net_clear_rate": None if attempts == 0 else sum(_int_bool(r, "net_clear") for r in rows) / attempts,
        "opponent_bounce_rate": None if attempts == 0 else sum(_int_bool(r, "opponent_bounce") for r in rows) / attempts,
        "fall_rate": None if attempts == 0 else sum(_int_bool(r, "fallen") for r in rows) / attempts,
        "dominant_failure": _dominant_failure(rows),
        "all_attempts": {
            "min_ball_racket_distance_m": _stats(values("min_ball_racket_distance_m")),
        },
        "contact_alignment": {
            "contact_count": len(contact_rows),
            "racket_target_pos_error_m": _stats(values("racket_target_pos_error_m", contact_derived)),
            "contact_target_pos_error_m": _stats(values("contact_target_pos_error_m", contact_derived)),
            "abs_time_to_strike_cmd_s": _stats(values("abs_time_to_strike_cmd_s", contact_derived)),
            "racket_vel_target_error_mps": _stats(values("racket_vel_target_error_mps", contact_derived)),
            "racket_vel_target_angle_deg": _stats(values("racket_vel_target_angle_deg", contact_derived)),
            "racket_speed_ratio": _stats(values("racket_speed_ratio", contact_derived)),
            "ball_post_target_outgoing_error_mps": _stats(values("ball_post_target_outgoing_error_mps", contact_derived)),
            "ball_post_target_outgoing_angle_deg": _stats(values("ball_post_target_outgoing_angle_deg", contact_derived)),
            "racket_normal_target_angle_deg": _stats(values("racket_normal_target_angle_deg", contact_derived)),
            "contact_normal_target_angle_deg": _stats(values("contact_normal_target_angle_deg", real_contact_derived)),
            "closing_speed_pre_mps": _stats(values("closing_speed_pre_mps", contact_derived)),
            "outgoing_normal_speed_post_mps": _stats(values("outgoing_normal_speed_post_mps", contact_derived)),
            "raw_action_max": _stats(values("raw_action_max", contact_derived)),
            "applied_action_max": _stats(values("applied_action_max", contact_derived)),
            "q_des_max_rad": _stats(values("q_des_max_rad", contact_derived)),
            "within_default_thresholds": {
                "racket_pos_le_threshold": _with_threshold(
                    values("racket_target_pos_error_m", contact_derived), pos_threshold
                ),
                "abs_tts_le_threshold": _with_threshold(
                    values("abs_time_to_strike_cmd_s", contact_derived), time_threshold
                ),
                "racket_vel_error_le_threshold": _with_threshold(
                    values("racket_vel_target_error_mps", contact_derived), vel_threshold
                ),
                "racket_vel_angle_le_threshold": _with_threshold(
                    values("racket_vel_target_angle_deg", contact_derived), vel_angle_threshold
                ),
                "outgoing_error_le_threshold": _with_threshold(
                    values("ball_post_target_outgoing_error_mps", contact_derived), outgoing_threshold
                ),
                "outgoing_angle_le_threshold": _with_threshold(
                    values("ball_post_target_outgoing_angle_deg", contact_derived), outgoing_angle_threshold
                ),
                "normal_angle_le_threshold": _with_threshold(
                    values("racket_normal_target_angle_deg", contact_derived), normal_angle_threshold
                ),
            },
        },
    }


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "na"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _mean(summary: dict, path: tuple[str, ...]) -> float | None:
    node = summary
    for key in path:
        node = node[key]
    return node.get("mean")


def _markdown_table(name: str, summary: dict) -> str:
    ca = summary["contact_alignment"]
    return (
        f"| {name} | {_fmt(summary['success_rate'])} | {_fmt(summary['contact_rate'])} | "
        f"{_fmt(summary['net_clear_rate'])} | {_fmt(summary['fall_rate'])} | "
        f"{_fmt(_mean(summary, ('contact_alignment', 'racket_target_pos_error_m')))} | "
        f"{_fmt(_mean(summary, ('contact_alignment', 'abs_time_to_strike_cmd_s')))} | "
        f"{_fmt(_mean(summary, ('contact_alignment', 'racket_vel_target_error_mps')))} | "
        f"{_fmt(_mean(summary, ('contact_alignment', 'racket_vel_target_angle_deg')))} | "
        f"{_fmt(_mean(summary, ('contact_alignment', 'ball_post_target_outgoing_error_mps')))} | "
        f"{_fmt(_mean(summary, ('contact_alignment', 'ball_post_target_outgoing_angle_deg')))} | "
        f"{_fmt(_mean(summary, ('contact_alignment', 'racket_normal_target_angle_deg')))} | "
        f"{ca['contact_count']} | {summary['dominant_failure'] or 'none'} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contact_diag_csv", help="CSV written by mujoco_eval_onnx.py --contact-diag-csv.")
    parser.add_argument("--json-out", default=None, help="Write the full summary JSON.")
    parser.add_argument("--markdown-out", default=None, help="Write a compact markdown table.")
    parser.add_argument("--pos-threshold", type=float, default=0.08)
    parser.add_argument("--time-threshold", type=float, default=0.04)
    parser.add_argument("--vel-threshold", type=float, default=1.0)
    parser.add_argument("--vel-angle-threshold", type=float, default=30.0)
    parser.add_argument("--outgoing-threshold", type=float, default=2.0)
    parser.add_argument("--outgoing-angle-threshold", type=float, default=25.0)
    parser.add_argument("--normal-angle-threshold", type=float, default=30.0)
    args = parser.parse_args()

    csv_path = Path(args.contact_diag_csv).resolve()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    groups = {"all": rows}
    for side in ("forehand", "backhand"):
        groups[side] = [r for r in rows if (r.get("side") or "").lower() == side]

    result = {
        "schema_version": 1,
        "source_csv": str(csv_path),
        "thresholds": {
            "pos_m": args.pos_threshold,
            "time_s": args.time_threshold,
            "vel_mps": args.vel_threshold,
            "vel_angle_deg": args.vel_angle_threshold,
            "outgoing_mps": args.outgoing_threshold,
            "outgoing_angle_deg": args.outgoing_angle_threshold,
            "normal_angle_deg": args.normal_angle_threshold,
        },
        "groups": {
            name: summarize_rows(
                group_rows,
                pos_threshold=args.pos_threshold,
                time_threshold=args.time_threshold,
                vel_threshold=args.vel_threshold,
                vel_angle_threshold=args.vel_angle_threshold,
                outgoing_threshold=args.outgoing_threshold,
                outgoing_angle_threshold=args.outgoing_angle_threshold,
                normal_angle_threshold=args.normal_angle_threshold,
            )
            for name, group_rows in groups.items()
        },
    }

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = (
        "| group | success | contact | net | fall | racket_pos_err_m | abs_tts_s | "
        "racket_vel_err_mps | racket_vel_ang_deg | ball_out_err_mps | ball_out_ang_deg | "
        "normal_ang_deg | contacts | dominant_failure |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    table = "\n".join([header] + [_markdown_table(name, result["groups"][name]) for name in ("all", "forehand", "backhand")])

    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")

    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
