"""Diagnose planner rollout error by estimator confidence gates.

This is an offline-only analysis tool.  It does not modify planner parameters.
The goal is to check whether poor long-tail predictions are correlated with
low estimator sample count after a table-bounce reset.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_real_planner_mujoco import _analytic_rollout, _interp_real, _stratified_sample  # noqa: E402
from evaluate_planner_on_segments import (  # noqa: E402
    REPO_ROOT,
    _crossings,
    _diagnose_geometry,
    _load_segment,
    _make_config,
    _manifest_metadata,
    _next_crossing,
    _planner_frame_issue,
    _segment_files,
)

PLANNER_SRC = REPO_ROOT / "hope_ws" / "src" / "hope_planner"
if str(PLANNER_SRC) not in sys.path:
    sys.path.insert(0, str(PLANNER_SRC))

from hope_planner.ball_state_estimator import BallStateEstimator  # noqa: E402
from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor  # noqa: E402


@dataclass(frozen=True)
class ConfidenceCase:
    file: str
    sample_i: int
    t: np.ndarray
    pos: np.ndarray
    p_est: np.ndarray
    v_est: np.ndarray
    t_est: float
    actual_t: float
    actual_p: np.ndarray
    horizon_s: float
    estimator_samples: int
    samples_since_bounce: int | None
    seconds_since_bounce: float | None


def _collect_confidence_cases(
    files: list[Path],
    cfg: Any,
    eval_period_s: float,
    min_horizon_s: float,
    max_horizon_s: float,
) -> list[ConfidenceCase]:
    cases: list[ConfidenceCase] = []
    for path in files:
        t, pos = _load_segment(path)
        if len(t) < max(cfg.fit_window, 6):
            continue
        crossings = _crossings(t, pos, cfg.x_hit)
        if not crossings:
            continue

        est = BallStateEstimator(cfg)
        next_eval_t = -np.inf
        last_bounce_i: int | None = None
        last_bounce_t: float | None = None
        for i, (ti, pi) in enumerate(zip(t, pos)):
            ti = float(ti)
            est.push(ti, pi)
            if est.bounce_detected:
                last_bounce_i = i
                last_bounce_t = ti
            if eval_period_s > 0.0 and ti < next_eval_t:
                continue
            next_eval_t = ti + eval_period_s
            if not est.ready:
                continue
            p_est, v_est, t_est = est.estimate()
            if v_est[0] >= 0.0:
                continue
            actual = _next_crossing(crossings, ti, min_horizon_s)
            if actual is None:
                continue
            horizon = float(actual["t"] - t_est)
            if horizon < min_horizon_s or horizon > max_horizon_s:
                continue
            cases.append(
                ConfidenceCase(
                    file=path.name,
                    sample_i=int(i),
                    t=t,
                    pos=pos,
                    p_est=p_est,
                    v_est=v_est,
                    t_est=float(t_est),
                    actual_t=float(actual["t"]),
                    actual_p=np.asarray(actual["p"], dtype=float),
                    horizon_s=horizon,
                    estimator_samples=len(est.t_buffer),
                    samples_since_bounce=(i - last_bounce_i if last_bounce_i is not None else None),
                    seconds_since_bounce=(ti - last_bounce_t if last_bounce_t is not None else None),
                )
            )
    return cases


def _actual_bounces(case: ConfidenceCase, z_max: float) -> int:
    keep = (case.t >= case.t_est) & (case.t <= case.actual_t)
    z = case.pos[keep, 2]
    if len(z) < 5:
        return 0
    count = 0
    for i in range(2, len(z) - 2):
        if z[i] <= z_max and z[i] <= z[i - 1] and z[i] <= z[i + 1]:
            count += 1
    return int(count)


def _summary(values: list[float], scale: float = 1e3, unit: str = "mm") -> dict[str, Any]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        f"median_{unit}": float(np.median(arr) * scale),
        f"p90_{unit}": float(np.percentile(arr, 90) * scale),
        f"mean_{unit}": float(np.mean(arr) * scale),
        f"max_{unit}": float(np.max(arr) * scale),
    }


def _case_metrics(case: ConfidenceCase, physics: Any, cfg: Any, table: Any, args: argparse.Namespace) -> dict[str, Any] | None:
    predictor = BallTrajectoryPredictor(physics, cfg, table)
    strike = predictor.predict(case.p_est, case.v_est, case.t_est)
    if not strike.valid:
        return None

    offsets = np.arange(0.0, case.horizon_s + 0.5 * args.sample_dt_s, args.sample_dt_s)
    if offsets[-1] < case.horizon_s:
        offsets = np.append(offsets, case.horizon_s)
    else:
        offsets[-1] = case.horizon_s

    pred = _analytic_rollout(case.p_est, case.v_est, offsets, physics, cfg, table)
    real = _interp_real(case.t, case.pos, case.t_est + offsets)
    point_yz = np.linalg.norm(pred[:, 1:3] - real[:, 1:3], axis=1)
    endpoint_yz = float(np.linalg.norm(strike.p_ball[1:3] - case.actual_p[1:3]))
    actual_bounces = _actual_bounces(case, physics.radius + args.actual_bounce_z_margin_m)
    return {
        "endpoint_yz_m": endpoint_yz,
        "point_yz_median_m": float(np.median(point_yz)),
        "time_abs_s": float(abs(strike.t_strike - case.actual_t)),
        "pred_bounces": int(strike.num_bounces),
        "actual_bounces": int(actual_bounces),
        "bounce_match": int(strike.num_bounces == actual_bounces),
    }


def _evaluate_group(cases: list[ConfidenceCase], physics: Any, cfg: Any, table: Any, args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    invalid = 0
    for case in cases:
        row = _case_metrics(case, physics, cfg, table, args)
        if row is None:
            invalid += 1
        else:
            rows.append(row)
    return {
        "num_cases": int(len(cases)),
        "num_valid": int(len(rows)),
        "invalid": int(invalid),
        "valid_rate": float(len(rows) / max(len(cases), 1)),
        "kept_fraction_of_all": float(len(cases) / max(args._total_cases, 1)),
        "endpoint_yz": _summary([r["endpoint_yz_m"] for r in rows]),
        "point_yz_median": _summary([r["point_yz_median_m"] for r in rows]),
        "time_abs": _summary([r["time_abs_s"] for r in rows], scale=1e3, unit="ms"),
        "bounce_match_rate": float(np.mean([r["bounce_match"] for r in rows])) if rows else None,
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    physics, cfg, table = _make_config(args)
    files = _segment_files(Path(args.segments))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no segment CSVs found under {args.segments}")

    geometry = _diagnose_geometry(files, table, physics)
    frame_issue = _planner_frame_issue(geometry)
    if frame_issue and not args.allow_bad_geometry:
        raise SystemExit(
            f"{frame_issue} Use --allow-bad-geometry only for intentional diagnostics."
        )

    min_h = args.min_horizon_ms * 1e-3
    max_h = args.max_horizon_ms * 1e-3
    cfg.max_predict_time = max(float(cfg.max_predict_time), max_h)
    cases_full = _collect_confidence_cases(files, cfg, args.eval_period_s, min_h, max_h)
    cases = _stratified_sample(cases_full, args.max_cases, args.seed)
    if not cases:
        raise SystemExit("no comparable incoming-ball cases")
    args._total_cases = len(cases)

    thresholds = [int(v) for v in args.min_estimator_samples.split(",") if v.strip()]
    rows = []
    for threshold in thresholds:
        subset = [case for case in cases if case.estimator_samples >= threshold]
        rows.append(
            {
                "gate": f"estimator_samples>={threshold}",
                "threshold": int(threshold),
                **_evaluate_group(subset, physics, cfg, table, args),
            }
        )

    post_bounce_rows = []
    for threshold in thresholds:
        subset = [
            case
            for case in cases
            if case.samples_since_bounce is not None and case.samples_since_bounce >= threshold
        ]
        post_bounce_rows.append(
            {
                "gate": f"samples_since_detected_bounce>={threshold}",
                "threshold": int(threshold),
                **_evaluate_group(subset, physics, cfg, table, args),
            }
        )

    result = {
        "segments": str(Path(args.segments).resolve()),
        "planner_yaml": str(Path(args.planner_yaml).resolve()),
        "geometry": geometry,
        "segments_manifest": _manifest_metadata(Path(args.segments)),
        "planner_frame_geometry_ok": frame_issue is None,
        "config": {
            "x_hit": float(cfg.x_hit),
            "fit_window": int(cfg.fit_window),
            "min_ready_samples": int(cfg.min_ready_samples),
            "bounce_center_z_max": float(cfg.bounce_center_z_max),
            "eval_period_s": float(args.eval_period_s),
            "sample_dt_s": float(args.sample_dt_s),
            "horizon_range_ms": [float(args.min_horizon_ms), float(args.max_horizon_ms)],
        },
        "physics": {
            "drag_k": float(physics.k),
            "table_C_h": float(physics.C_h),
            "table_C_v": float(physics.C_v),
        },
        "num_cases_full": int(len(cases_full)),
        "num_cases_used": int(len(cases)),
        "by_estimator_samples": rows,
        "by_samples_since_detected_bounce": post_bounce_rows,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    printable = {
        "num_cases_full": result["num_cases_full"],
        "num_cases_used": result["num_cases_used"],
        "by_estimator_samples": rows,
        "by_samples_since_detected_bounce": post_bounce_rows,
    }
    print(json.dumps(printable, indent=1))
    print(f"-> {out_path}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segments", help="directory of planner-frame t,x,y,z segment CSVs")
    parser.add_argument(
        "--planner-yaml",
        default=str(REPO_ROOT / "hope_ws" / "src" / "hope_planner" / "config" / "hope_planner.yaml"),
    )
    parser.add_argument("--physics-path", default=str(REPO_ROOT / "configs" / "ball_physics.yaml"))
    parser.add_argument("--out-json", default=str(REPO_ROOT / "analysis" / "estimator_confidence_diagnosis.json"))
    parser.add_argument("--x-hit", type=float, default=None)
    parser.add_argument("--table-y-max", type=float, default=None)
    parser.add_argument("--fit-window", type=int, default=None)
    parser.add_argument("--min-ready-samples", type=int, default=None)
    parser.add_argument("--min-horizon-ms", type=float, default=50.0)
    parser.add_argument("--max-horizon-ms", type=float, default=500.0)
    parser.add_argument("--eval-period-s", type=float, default=0.02)
    parser.add_argument("--sample-dt-s", type=float, default=0.04)
    parser.add_argument("--actual-bounce-z-margin-m", type=float, default=0.06)
    parser.add_argument("--min-estimator-samples", default="6,12,20,30,45,60,67")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-bad-geometry",
        action="store_true",
        help="do not fail when near-table points fall outside planner table bounds",
    )
    return parser


def main() -> None:
    diagnose(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
