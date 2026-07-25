"""Optimize incoming-ball planner parameters against real trajectories.

This tunes only the part identifiable from ball-only Motive data:

* estimator window and centre-bounce reset threshold;
* no-spin incoming flight drag;
* effective table-bounce tangential/normal coefficients.

The objective compares full future trajectory samples plus the final measured
crossing of the strike plane.  It intentionally does not tune paddle/contact
return parameters; those require racket pose/velocity data.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_real_planner_mujoco import (  # noqa: E402
    RolloutCase,
    _analytic_rollout,
    _collect_cases,
    _interp_real,
    _stratified_sample,
)
from evaluate_planner_on_segments import (  # noqa: E402
    REPO_ROOT,
    _diagnose_geometry,
    _make_config,
    _manifest_metadata,
    _planner_frame_issue,
    _segment_files,
)

PLANNER_SRC = REPO_ROOT / "hope_ws" / "src" / "hope_planner"
if str(PLANNER_SRC) not in sys.path:
    sys.path.insert(0, str(PLANNER_SRC))

from hope_planner.ball_trajectory_predictor import BallTrajectoryPredictor  # noqa: E402
from hope_planner.constants import BallPhysics, load_ball_physics  # noqa: E402


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_bounds(text: str) -> list[tuple[float, float]]:
    bounds = []
    for part in text.split(","):
        lo_hi = [float(v.strip()) for v in part.split(":") if v.strip()]
        if len(lo_hi) != 2 or lo_hi[0] > lo_hi[1]:
            raise argparse.ArgumentTypeError(
                "--bounds must look like k_lo:k_hi,ch_lo:ch_hi,cv_lo:cv_hi"
            )
        bounds.append((lo_hi[0], lo_hi[1]))
    if len(bounds) != 3:
        raise argparse.ArgumentTypeError(
            "--bounds must provide exactly three ranges: drag_k,table_C_h,table_C_v"
        )
    return bounds


def _clone_physics(base: Any, k: float, c_h: float, c_v: float) -> BallPhysics:
    return BallPhysics(
        k=float(k),
        C_h=float(c_h),
        C_v=float(c_v),
        g=np.asarray(base.g, dtype=float).copy(),
        radius=float(base.radius),
        mass=float(base.mass),
    )


def _split_files(files: list[Path], train_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(files))
    rng.shuffle(order)
    n_train = int(round(len(files) * train_fraction))
    n_train = min(max(n_train, 1), max(len(files) - 1, 1))
    train = [files[i] for i in sorted(order[:n_train])]
    val = [files[i] for i in sorted(order[n_train:])]
    return train, val


def _actual_bounces(case: RolloutCase, z_max: float = 0.08) -> int:
    keep = (case.t >= case.t_est) & (case.t <= case.actual_t)
    t = case.t[keep]
    pos = case.pos[keep]
    if len(t) < 5:
        return 0
    z = pos[:, 2]
    count = 0
    for i in range(2, len(z) - 2):
        if z[i] <= z_max and z[i] <= z[i - 1] and z[i] <= z[i + 1]:
            count += 1
    return int(count)


def _velocity_at(t: np.ndarray, pos: np.ndarray, t_query: float, window_s: float = 0.035) -> np.ndarray | None:
    keep = np.abs(t - t_query) <= window_s
    if np.count_nonzero(keep) < 6:
        return None
    tt = t[keep] - t_query
    pp = pos[keep]
    vel = np.zeros(3)
    for axis in range(3):
        coeff = np.polyfit(tt, pp[:, axis], deg=2)
        vel[axis] = coeff[-2]
    return vel


def _case_loss_and_row(
    case: RolloutCase,
    physics: BallPhysics,
    cfg: Any,
    table: Any,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any] | None]:
    predictor = BallTrajectoryPredictor(physics, cfg, table)
    strike = predictor.predict(case.p_est, case.v_est, case.t_est)
    if not strike.valid:
        return float(args.invalid_penalty_m), None

    offsets = np.arange(0.0, case.horizon_s + 0.5 * args.sample_dt_s, args.sample_dt_s)
    if offsets[-1] < case.horizon_s:
        offsets = np.append(offsets, case.horizon_s)
    else:
        offsets[-1] = case.horizon_s
    pred = _analytic_rollout(case.p_est, case.v_est, offsets, physics, cfg, table)
    real = _interp_real(case.t, case.pos, case.t_est + offsets)
    point_yz = np.linalg.norm(pred[:, 1:3] - real[:, 1:3], axis=1)

    endpoint_yz = float(np.linalg.norm(strike.p_ball[1:3] - case.actual_p[1:3]))
    point_yz_med = float(np.median(point_yz))
    time_error_s = float(strike.t_strike - case.actual_t)
    time_equiv_m = abs(time_error_s) * float(args.time_weight_mps)
    pred_bounces = int(strike.num_bounces)
    actual_bounces = _actual_bounces(case, z_max=physics.radius + args.actual_bounce_z_margin_m)
    bounce_penalty = abs(pred_bounces - actual_bounces) * float(args.bounce_penalty_m)

    velocity_equiv_m = 0.0
    actual_v = _velocity_at(case.t, case.pos, case.actual_t)
    if actual_v is not None:
        velocity_equiv_m = (
            float(np.linalg.norm(strike.v_ball[1:3] - actual_v[1:3]))
            * float(args.velocity_weight_s)
        )

    loss = (
        float(args.endpoint_weight) * endpoint_yz
        + float(args.pointwise_weight) * point_yz_med
        + time_equiv_m
        + bounce_penalty
        + velocity_equiv_m
    )
    row = {
        "loss_m": float(loss),
        "endpoint_yz_m": endpoint_yz,
        "point_yz_median_m": point_yz_med,
        "time_error_s": time_error_s,
        "velocity_equiv_m": float(velocity_equiv_m),
        "pred_bounces": pred_bounces,
        "actual_bounces": actual_bounces,
        "bounce_mismatch": int(pred_bounces != actual_bounces),
        "horizon_s": float(case.horizon_s),
    }
    return float(loss), row


def _percentiles(values: list[float], scale: float = 1e3, suffix: str = "mm") -> dict[str, Any]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        f"median_{suffix}": float(np.median(arr) * scale),
        f"p90_{suffix}": float(np.percentile(arr, 90) * scale),
        f"mean_{suffix}": float(np.mean(arr) * scale),
        f"max_{suffix}": float(np.max(arr) * scale),
    }


def _evaluate_cases(
    cases: list[RolloutCase],
    physics: BallPhysics,
    cfg: Any,
    table: Any,
    args: argparse.Namespace,
    keep_rows: int = 0,
) -> dict[str, Any]:
    losses = []
    rows = []
    invalid = 0
    kept = []
    for case in cases:
        loss, row = _case_loss_and_row(case, physics, cfg, table, args)
        losses.append(loss)
        if row is None:
            invalid += 1
            continue
        rows.append(row)
        if len(kept) < keep_rows:
            kept.append({"file": case.file, "sample_i": case.sample_i, **row})
    if losses:
        arr = np.asarray(losses, dtype=float)
        objective = float(np.median(arr) + 0.25 * np.percentile(arr, 90))
    else:
        objective = float(args.invalid_penalty_m * 2.0)
    return {
        "objective": objective,
        "valid_rate": float(len(rows) / max(len(cases), 1)),
        "invalid": int(invalid),
        "num_cases": int(len(cases)),
        "num_valid": int(len(rows)),
        "loss": _percentiles([r["loss_m"] for r in rows]),
        "endpoint_yz": _percentiles([r["endpoint_yz_m"] for r in rows]),
        "point_yz_median": _percentiles([r["point_yz_median_m"] for r in rows]),
        "time_abs": _percentiles([abs(r["time_error_s"]) for r in rows], scale=1e3, suffix="ms"),
        "velocity_equiv": _percentiles([r["velocity_equiv_m"] for r in rows]),
        "bounce_mismatch_rate": (
            float(np.mean([r["bounce_mismatch"] for r in rows])) if rows else None
        ),
        "sample_rows": kept,
    }


def _candidate_dict(x: np.ndarray, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "drag_k": float(x[0]),
        "table_C_h": float(x[1]),
        "table_C_v": float(x[2]),
    }
    if metrics is not None:
        row.update(
            {
                "objective": float(metrics["objective"]),
                "valid_rate": float(metrics["valid_rate"]),
                "num_cases": int(metrics["num_cases"]),
                "num_valid": int(metrics["num_valid"]),
                "loss": metrics["loss"],
                "endpoint_yz": metrics["endpoint_yz"],
                "point_yz_median": metrics["point_yz_median"],
                "time_abs": metrics["time_abs"],
                "bounce_mismatch_rate": metrics["bounce_mismatch_rate"],
            }
        )
    return row


def _initial_points(bounds: list[tuple[float, float]], base: BallPhysics, args: argparse.Namespace) -> list[np.ndarray]:
    rng = np.random.default_rng(args.seed)
    points = [
        np.array([base.k, base.C_h, base.C_v], dtype=float),
        np.array([0.18, 0.80, 0.95], dtype=float),
        np.array([0.15, 0.75, 0.95], dtype=float),
        np.array([0.1261, 0.631, 0.9215], dtype=float),
        np.array([0.1261, 0.70, 0.9215], dtype=float),
    ]
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    for _ in range(args.random_candidates):
        points.append(rng.uniform(lows, highs))
    out = []
    seen = set()
    for point in points:
        clipped = np.clip(point, lows, highs)
        key = tuple(np.round(clipped, 6))
        if key not in seen:
            out.append(clipped)
            seen.add(key)
    return out


def _objective(x: np.ndarray, cases: list[RolloutCase], base: BallPhysics, cfg: Any, table: Any, args: argparse.Namespace) -> float:
    physics = _clone_physics(base, x[0], x[1], x[2])
    return float(_evaluate_cases(cases, physics, cfg, table, args)["objective"])


def _minimize_from_starts(
    starts: list[np.ndarray],
    bounds: list[tuple[float, float]],
    train_cases: list[RolloutCase],
    base: BallPhysics,
    cfg: Any,
    table: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    try:
        from scipy.optimize import minimize
    except ImportError:
        return []
    rows = []
    for i, x0 in enumerate(starts[: args.num_starts], start=1):
        print(f"  powell start {i}/{min(args.num_starts, len(starts))}: {x0.tolist()}")
        result = minimize(
            _objective,
            x0,
            args=(train_cases, base, cfg, table, args),
            method="Powell",
            bounds=bounds,
            options={"maxfev": args.max_evals_per_start, "xtol": 1e-3, "ftol": 1e-3, "disp": False},
        )
        x = np.asarray(result.x, dtype=float)
        metrics = _evaluate_cases(train_cases, _clone_physics(base, x[0], x[1], x[2]), cfg, table, args)
        rows.append(
            {
                "source": "powell",
                "success": bool(result.success),
                "message": str(result.message),
                "nfev": int(result.nfev),
                **_candidate_dict(x, metrics),
            }
        )
        print(f"    -> obj={metrics['objective'] * 1e3:.1f} mm, x={x.tolist()}")
    return rows


def optimize(args: argparse.Namespace) -> dict[str, Any]:
    planner_physics, base_cfg, table = _make_config(args)
    shared_physics = load_ball_physics(args.physics_path)
    base_physics = planner_physics if args.physics_source == "planner-yaml" else shared_physics

    files = _segment_files(Path(args.segments))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no segment CSVs found under {args.segments}")

    geometry = _diagnose_geometry(files, table, planner_physics)
    frame_issue = _planner_frame_issue(geometry)
    if frame_issue and not args.allow_bad_geometry:
        raise SystemExit(
            f"{frame_issue} Use --allow-bad-geometry only for intentional diagnostics."
        )

    train_files, val_files = _split_files(files, args.train_fraction, args.seed)
    min_h = args.min_horizon_ms * 1e-3
    max_h = args.max_horizon_ms * 1e-3
    bounds = _parse_bounds(args.bounds)
    fit_windows = _parse_int_list(args.fit_windows)
    bounce_values = _parse_float_list(args.bounce_center_z_max_values)

    all_estimator_rows = []
    final_candidates = []
    for fw in fit_windows:
        for bounce_z in bounce_values:
            cfg = copy.deepcopy(base_cfg)
            cfg.fit_window = int(fw)
            cfg.bounce_center_z_max = float(bounce_z)
            cfg.max_predict_time = max(float(cfg.max_predict_time), max_h)
            train_full = _collect_cases(train_files, cfg, args.eval_period_s, min_h, max_h)
            val_cases = _collect_cases(val_files, cfg, args.eval_period_s, min_h, max_h)
            train_sample = _stratified_sample(train_full, args.train_sample_cases, args.seed)
            if not train_sample or not val_cases:
                continue
            current_metrics = _evaluate_cases(train_sample, planner_physics, cfg, table, args)
            all_estimator_rows.append(
                {
                    "fit_window": int(fw),
                    "bounce_center_z_max": float(bounce_z),
                    "train_cases_full": int(len(train_full)),
                    "train_cases_used": int(len(train_sample)),
                    "validation_cases": int(len(val_cases)),
                    "current_train_sample": _candidate_dict(
                        np.array([planner_physics.k, planner_physics.C_h, planner_physics.C_v]),
                        current_metrics,
                    ),
                }
            )
            print(
                f"estimator fw={fw}, bounce={bounce_z:.3f}: "
                f"train={len(train_sample)}/{len(train_full)}, val={len(val_cases)}, "
                f"current_obj={current_metrics['objective'] * 1e3:.1f} mm"
            )

            starts = _initial_points(bounds, base_physics, args)
            seed_rows = []
            for point in starts:
                metrics = _evaluate_cases(train_sample, _clone_physics(base_physics, point[0], point[1], point[2]), cfg, table, args)
                seed_rows.append({"source": "seed", **_candidate_dict(point, metrics)})
            seed_rows.sort(key=lambda row: row["objective"])
            powell_rows = _minimize_from_starts(
                [np.array([r["drag_k"], r["table_C_h"], r["table_C_v"]], dtype=float) for r in seed_rows],
                bounds,
                train_sample,
                base_physics,
                cfg,
                table,
                args,
            )
            candidate_rows = seed_rows + powell_rows
            candidate_rows.sort(key=lambda row: row["objective"])

            seen = set()
            kept = 0
            for row in candidate_rows:
                x = np.array([row["drag_k"], row["table_C_h"], row["table_C_v"]], dtype=float)
                key = tuple(np.round(x, 5))
                if key in seen:
                    continue
                seen.add(key)
                train_metrics = _evaluate_cases(train_full, _clone_physics(base_physics, x[0], x[1], x[2]), cfg, table, args)
                val_metrics = _evaluate_cases(val_cases, _clone_physics(base_physics, x[0], x[1], x[2]), cfg, table, args)
                final_candidates.append(
                    {
                        "fit_window": int(fw),
                        "bounce_center_z_max": float(bounce_z),
                        **_candidate_dict(x),
                        "train_sample_objective": float(row["objective"]),
                        "train_full": train_metrics,
                        "validation": val_metrics,
                    }
                )
                kept += 1
                print(
                    f"  final fw={fw} bz={bounce_z:.3f} x={x.tolist()} "
                    f"train={train_metrics['objective'] * 1e3:.1f}mm "
                    f"val={val_metrics['objective'] * 1e3:.1f}mm"
                )
                if kept >= args.keep_top_per_estimator:
                    break

    if not final_candidates:
        raise SystemExit("no final candidates")
    final_candidates.sort(key=lambda row: row["validation"]["objective"])
    best = final_candidates[0]

    result = {
        "segments": str(Path(args.segments).resolve()),
        "planner_yaml": str(Path(args.planner_yaml).resolve()),
        "physics_path": str(Path(args.physics_path).resolve()),
        "physics_source": args.physics_source,
        "geometry": geometry,
        "segments_manifest": _manifest_metadata(Path(args.segments)),
        "planner_frame_geometry_ok": frame_issue is None,
        "objective_weights": {
            "endpoint_weight": float(args.endpoint_weight),
            "pointwise_weight": float(args.pointwise_weight),
            "time_weight_mps": float(args.time_weight_mps),
            "velocity_weight_s": float(args.velocity_weight_s),
            "bounce_penalty_m": float(args.bounce_penalty_m),
        },
        "config_search": {
            "fit_windows": fit_windows,
            "bounce_center_z_max_values": bounce_values,
            "horizon_range_ms": [float(args.min_horizon_ms), float(args.max_horizon_ms)],
            "eval_period_s": float(args.eval_period_s),
            "sample_dt_s": float(args.sample_dt_s),
            "train_fraction": float(args.train_fraction),
            "seed": int(args.seed),
        },
        "train_files": [str(p) for p in train_files],
        "validation_files": [str(p) for p in val_files],
        "estimator_rows": all_estimator_rows,
        "best_by_validation": best,
        "final_candidates": final_candidates[: args.keep_top_final],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    printable = {
        "best": {
            "fit_window": best["fit_window"],
            "bounce_center_z_max": best["bounce_center_z_max"],
            "drag_k": best["drag_k"],
            "table_C_h": best["table_C_h"],
            "table_C_v": best["table_C_v"],
            "train_objective_mm": best["train_full"]["objective"] * 1e3,
            "validation_objective_mm": best["validation"]["objective"] * 1e3,
            "validation_endpoint_yz": best["validation"]["endpoint_yz"],
            "validation_point_yz_median": best["validation"]["point_yz_median"],
            "validation_time_abs": best["validation"]["time_abs"],
            "validation_bounce_mismatch_rate": best["validation"]["bounce_mismatch_rate"],
        },
        "top_candidates": [
            {
                "fit_window": row["fit_window"],
                "bounce_center_z_max": row["bounce_center_z_max"],
                "drag_k": row["drag_k"],
                "table_C_h": row["table_C_h"],
                "table_C_v": row["table_C_v"],
                "val_objective_mm": row["validation"]["objective"] * 1e3,
                "val_endpoint_yz_median_mm": row["validation"]["endpoint_yz"].get("median_mm"),
                "val_time_median_ms": row["validation"]["time_abs"].get("median_ms"),
            }
            for row in final_candidates[: min(8, len(final_candidates))]
        ],
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
    parser.add_argument("--out-json", default=str(REPO_ROOT / "analysis" / "incoming_trajectory_optimization.json"))
    parser.add_argument("--physics-source", choices=("planner-yaml", "shared"), default="planner-yaml")
    parser.add_argument("--x-hit", type=float, default=None)
    parser.add_argument("--table-y-max", type=float, default=None)
    parser.add_argument("--fit-window", type=int, default=None)
    parser.add_argument("--fit-windows", default="61,67,73")
    parser.add_argument("--bounce-center-z-max-values", default="0.09,0.11,0.13")
    parser.add_argument("--bounds", default="0.05:0.30,0.45:0.95,0.75:1.00")
    parser.add_argument("--min-horizon-ms", type=float, default=50.0)
    parser.add_argument("--max-horizon-ms", type=float, default=500.0)
    parser.add_argument("--eval-period-s", type=float, default=0.02)
    parser.add_argument("--sample-dt-s", type=float, default=0.04)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--pointwise-weight", type=float, default=0.5)
    parser.add_argument("--time-weight-mps", type=float, default=1.0)
    parser.add_argument("--velocity-weight-s", type=float, default=0.02)
    parser.add_argument("--bounce-penalty-m", type=float, default=0.15)
    parser.add_argument("--actual-bounce-z-margin-m", type=float, default=0.06)
    parser.add_argument("--invalid-penalty-m", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--train-sample-cases", type=int, default=420)
    parser.add_argument("--random-candidates", type=int, default=5)
    parser.add_argument("--num-starts", type=int, default=2)
    parser.add_argument("--max-evals-per-start", type=int, default=24)
    parser.add_argument("--keep-top-per-estimator", type=int, default=3)
    parser.add_argument("--keep-top-final", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-bad-geometry",
        action="store_true",
        help="do not fail when near-table points fall outside planner table bounds",
    )
    return parser


def main() -> None:
    optimize(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
