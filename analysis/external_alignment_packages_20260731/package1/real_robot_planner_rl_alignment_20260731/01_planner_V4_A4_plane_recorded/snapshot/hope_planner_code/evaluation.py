"""Causal, offline evaluation for the HOPE planner.

The evaluator replays canonical motion-capture CSVs in timestamp order.  A
forecast at time ``t`` only sees samples at or before ``t``; the later samples
are used solely to construct an offline reference at the strike plane.  This
makes the report useful for debugging without pretending that noisy mocap is
perfect ground truth.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import html
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    BallPhysics,
    PlannerConfig,
    TableParams,
    load_ball_physics,
    load_paddle_params,
    load_table_params,
)
from .planner import HOPEPlanner
from .side_selection import BACKHAND, FOREHAND, select_swing_side


@dataclass
class EvalThresholds:
    intercept_error_m: float = 0.05
    timing_error_s: float = 0.03
    estimator_position_error_m: float = 0.025
    estimator_velocity_error_mps: float = 0.50
    max_racket_speed_mps: float = 6.0


@dataclass
class Crossing:
    event_id: str
    file: str
    t: float
    p: np.ndarray
    v: np.ndarray
    sample_index: int


def load_canonical_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load, sort and de-duplicate a canonical ``t,x,y,z`` CSV."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        first = next((row for row in csv.reader(handle) if any(cell.strip() for cell in row)), [])
    header = {cell.strip() for cell in first}
    missing = sorted({"t", "x", "y", "z"} - header)
    if missing:
        hint = " This looks like a Motive multi-row export." if "Format Version" in header else ""
        raise ValueError(
            f"{path}: expected canonical t,x,y,z columns; missing {', '.join(missing)}."
            f"{hint} Convert the mocap export first (see docs/PLANNER_EVALUATION.md)."
        )
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig")
    names = tuple(raw.dtype.names or ())
    missing = sorted({"t", "x", "y", "z"} - set(names))
    if missing:
        raise ValueError(
            f"{path}: expected canonical t,x,y,z columns; missing {', '.join(missing)}. "
            "Convert the mocap export first (see docs/PLANNER_EVALUATION.md)."
        )
    raw = np.atleast_1d(raw)
    t = np.asarray(raw["t"], float)
    p = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(float)
    total = len(t)
    finite = np.isfinite(t) & np.all(np.isfinite(p), axis=1)
    t, p = t[finite], p[finite]
    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]
    if len(t):
        keep = np.r_[True, np.diff(t) > 0.0]
        duplicates = int(np.count_nonzero(~keep))
        t, p = t[keep], p[keep]
    else:
        duplicates = 0
    return t, p, {"rows": total, "nonfinite_rows": int(total - finite.sum()), "duplicate_timestamps": duplicates}


def _local_poly_state(t: np.ndarray, p: np.ndarray, query_t: float, center: int, window: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """Offline reference state from a centred local polynomial fit."""
    half = max(3, window // 2)
    lo, hi = max(0, center - half), min(len(t), center + half + 1)
    if hi - lo < 4:
        lo, hi = max(0, center - 2), min(len(t), center + 3)
    tt = t[lo:hi] - query_t
    pp = p[lo:hi]
    degree = min(2, len(tt) - 1)
    pos = np.zeros(3)
    vel = np.zeros(3)
    for axis in range(3):
        coeff = np.polyfit(tt, pp[:, axis], degree)
        pos[axis] = np.polyval(coeff, 0.0)
        vel[axis] = np.polyval(np.polyder(coeff), 0.0) if degree else 0.0
    return pos, vel


def find_incoming_crossings(t: np.ndarray, p: np.ndarray, x_hit: float, file_label: str) -> List[Crossing]:
    """Measured crossings of ``x_hit`` in the incoming (negative-x) direction."""
    out: List[Crossing] = []
    for i in range(len(t) - 1):
        if not (p[i, 0] > x_hit and p[i + 1, 0] <= x_hit and p[i + 1, 0] < p[i, 0]):
            continue
        dx = p[i + 1, 0] - p[i, 0]
        if abs(dx) < 1e-12:
            continue
        frac = float(np.clip((x_hit - p[i, 0]) / dx, 0.0, 1.0))
        tc = float(t[i] + frac * (t[i + 1] - t[i]))
        pc, vc = _local_poly_state(t, p, tc, i + 1)
        pc[0] = x_hit
        # Reject marker swaps/jitter masquerading as a real incoming crossing.
        if vc[0] >= -0.05:
            continue
        out.append(Crossing(f"{file_label}:{len(out):03d}", file_label, tc, pc, vc, i + 1))
    return out


def _stats(values: Iterable[float], absolute: bool = False) -> Dict[str, Optional[float]]:
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if absolute:
        a = np.abs(a)
    if not len(a):
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(a)), "mean": float(np.mean(a)), "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)), "max": float(np.max(a)),
    }


def _reason(planner: HOPEPlanner, exception: Optional[BaseException] = None) -> str:
    if exception is not None:
        return "exception"
    if getattr(planner.estimator, "outlier_rejected", False):
        return "outlier_rejected"
    if not planner.estimator.ready:
        return "estimator_warmup"
    if planner.ball_incoming is False:
        return "ball_not_incoming"
    if planner.strike_target is None:
        return "no_valid_plane_crossing"
    return "command"


def _event_root_cause(event_rows: List[dict], event: Crossing, thresholds: EvalThresholds) -> str:
    if not event_rows:
        return "NO_PREDICTION"
    final = min(event_rows, key=lambda r: r["lead_time_s"])
    if final["estimator_position_error_m"] > thresholds.estimator_position_error_m or final["estimator_velocity_error_mps"] > thresholds.estimator_velocity_error_mps:
        return "STATE_ESTIMATION"
    if final["intercept_error_m"] > thresholds.intercept_error_m or abs(final["timing_error_s"]) > thresholds.timing_error_s:
        return "TRAJECTORY_MODEL"
    if not final["side_correct"]:
        return "SIDE_SELECTION"
    if final["racket_speed_mps"] > thresholds.max_racket_speed_mps:
        return "COMMAND_UNREACHABLE"
    if final["net_margin_m"] < final["required_net_margin_m"]:
        return "NET_CLEARANCE"
    return "OK"


def _tts_bins(rows: List[dict]) -> List[dict]:
    """Error curves at fixed TTS checkpoints, with one forecast per event.

    Selecting the closest revision per physical crossing prevents an event
    with more Planner revisions from receiving more statistical weight.
    """
    checkpoints = ((0.8, 0.10), (0.5, 0.075), (0.3, 0.05), (0.1, 0.05))
    by_event: Dict[str, List[dict]] = {}
    for row in rows:
        by_event.setdefault(row["event_id"], []).append(row)
    out = []
    for target, tolerance in checkpoints:
        selected = []
        for event_rows in by_event.values():
            nearest = min(event_rows, key=lambda r: abs(r["lead_time_s"] - target))
            if abs(nearest["lead_time_s"] - target) <= tolerance:
                selected.append(nearest)
        out.append({
            "tts_s": target,
            "tolerance_s": tolerance,
            "events": len(selected),
            "intercept_error_m": _stats(r["intercept_error_m"] for r in selected),
            "absolute_y_error_m": _stats((r["error_y_m"] for r in selected), absolute=True),
            "absolute_z_error_m": _stats((r["error_z_m"] for r in selected), absolute=True),
            "absolute_timing_error_s": _stats((r["timing_error_s"] for r in selected), absolute=True),
            "timing_bias_s": (
                float(np.mean([r["timing_error_s"] for r in selected])) if selected else None
            ),
        })
    return out


def evaluate_files(
    paths: Sequence[Path], config: PlannerConfig, physics: BallPhysics, table: TableParams,
    solve_period_s: float = 0.02, split_y: float = -0.7625,
    thresholds: Optional[EvalThresholds] = None,
) -> Tuple[dict, List[dict], List[dict]]:
    """Evaluate canonical trajectories and return summary, forecasts, events."""
    thresholds = thresholds or EvalThresholds()
    # Fit covariance/residual diagnostics are intentionally disabled in the
    # real-time node because they are not part of command generation.
    config.collect_fit_diagnostics = True
    forecasts: List[dict] = []
    event_records: List[dict] = []
    failure_reasons: Counter = Counter()
    root_causes: Counter = Counter()
    quality = Counter()
    rates, jitter_ms = [], []

    for path in paths:
        t, p, q = load_canonical_csv(path)
        quality.update(q)
        if len(t) < 2:
            continue
        dt = np.diff(t)
        med_dt = float(np.median(dt))
        rates.append(1.0 / med_dt if med_dt > 0 else float("nan"))
        jitter_ms.extend(np.abs(dt - med_dt) * 1000.0)
        quality["large_gaps"] += int(np.count_nonzero(dt > max(0.05, 3.0 * med_dt)))

        crossings = find_incoming_crossings(t, p, config.x_hit, str(path))
        crossing_times = [c.t for c in crossings]
        by_event: Dict[str, List[dict]] = {c.event_id: [] for c in crossings}
        fail_by_event: Dict[str, Counter] = {c.event_id: Counter() for c in crossings}
        locked_side: Dict[str, int] = {}
        planner = HOPEPlanner(physics=physics, config=config, table=table)
        # Keep compatibility with older planner constructors while ensuring the
        # evaluator exercises the configured bounce physics.
        if hasattr(planner.estimator, "physics"):
            planner.estimator.physics = physics
        last_solve_t: Optional[float] = None

        for i, (ti, pi) in enumerate(zip(t, p)):
            ti = float(ti)
            if solve_period_s > 0 and last_solve_t is not None and 0 <= ti - last_solve_t < solve_period_s:
                planner.estimator.push(ti, pi)
                if getattr(planner.estimator, "outlier_rejected", False):
                    failure_reasons["outlier_rejected"] += 1
                continue
            last_solve_t = ti
            exc = None
            try:
                cmd = planner.update(ti, pi)
            except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as err:
                exc, cmd = err, None
            reason = _reason(planner, exc)
            failure_reasons[reason] += 1
            if cmd is None:
                j = bisect.bisect_left(crossing_times, ti)
                if j < len(crossings) and 0 < crossings[j].t - ti <= config.max_predict_time:
                    fail_by_event[crossings[j].event_id][reason] += 1
                continue

            j = bisect.bisect_left(crossing_times, ti)
            if j >= len(crossings):
                continue
            event = crossings[j]
            lead = event.t - ti
            if lead <= 0 or lead > config.max_predict_time:
                continue
            try:
                p_est, v_est, _ = planner.estimator.estimate()
                p_ref, v_ref = _local_poly_state(t, p, ti, i)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                p_est = v_est = p_ref = v_ref = np.full(3, np.nan)

            strike = planner.strike_target
            if strike is None:
                continue
            diagnostics = planner.estimator.fit_diagnostics
            residual_rms = diagnostics["residual_rms_m"]
            velocity_std = diagnostics["velocity_std_mps"]
            condition_number = diagnostics["condition_number"]
            pred_side = locked_side.setdefault(
                event.event_id, select_swing_side(float(cmd.p_intercept[1]), split_y)
            )
            true_side = select_swing_side(float(event.p[1]), split_y)
            row = {
                "event_id": event.event_id, "file": str(path), "forecast_t": ti,
                "truth_t": event.t, "lead_time_s": lead,
                "pred_y_m": float(cmd.p_intercept[1]), "pred_z_m": float(cmd.p_intercept[2]),
                "truth_y_m": float(event.p[1]), "truth_z_m": float(event.p[2]),
                "error_y_m": float(cmd.p_intercept[1] - event.p[1]),
                "error_z_m": float(cmd.p_intercept[2] - event.p[2]),
                "intercept_error_m": float(np.linalg.norm(cmd.p_intercept[1:3] - event.p[1:3])),
                "timing_error_s": float(cmd.t_strike - event.t),
                "ball_velocity_error_mps": float(np.linalg.norm(strike.v_ball - event.v)),
                "estimator_position_error_m": float(np.linalg.norm(p_est - p_ref)),
                "estimator_velocity_error_mps": float(np.linalg.norm(v_est - v_ref)),
                "fit_residual_rms_max_m": float(np.nanmax(residual_rms)),
                "fit_velocity_std_max_mps": float(np.nanmax(velocity_std)),
                "fit_condition_number_max": float(np.nanmax(condition_number)),
                "vertical_acceleration_mps2": float(diagnostics["vertical_acceleration_mps2"]),
                "vertical_model_error_mps2": float(diagnostics["vertical_model_error_mps2"]),
                "pred_side": int(pred_side), "truth_side": int(true_side),
                "side_correct": bool(pred_side == true_side),
                "racket_speed_mps": float(np.linalg.norm(cmd.v_racket)),
                "net_margin_m": float(cmd.net_margin),
                "required_net_margin_m": float(config.net_clearance_margin),
                "num_bounces_pred": int(cmd.num_bounces),
            }
            forecasts.append(row)
            by_event[event.event_id].append(row)

        for event in crossings:
            rows = by_event[event.event_id]
            cause = _event_root_cause(rows, event, thresholds)
            root_causes[cause] += 1
            no_command = fail_by_event[event.event_id]
            dominant_no_command = no_command.most_common(1)[0][0] if no_command else None
            first = max(rows, key=lambda r: r["lead_time_s"]) if rows else None
            final = min(rows, key=lambda r: r["lead_time_s"]) if rows else None
            event_records.append({
                "event_id": event.event_id, "file": str(path), "truth_t": event.t,
                "truth_y_m": float(event.p[1]), "truth_z_m": float(event.p[2]),
                "forecast_count": len(rows), "first_lead_time_s": first["lead_time_s"] if first else None,
                "first_intercept_error_m": first["intercept_error_m"] if first else None,
                "final_lead_time_s": final["lead_time_s"] if final else None,
                "final_intercept_error_m": final["intercept_error_m"] if final else None,
                "final_timing_error_s": final["timing_error_s"] if final else None,
                "root_cause": cause, "dominant_no_command_reason": dominant_no_command,
            })

    covered = sum(e["forecast_count"] > 0 for e in event_records)
    final_rows = []
    for event in event_records:
        rows = [r for r in forecasts if r["event_id"] == event["event_id"]]
        if rows:
            final_rows.append(min(rows, key=lambda r: r["lead_time_s"]))
    summary = {
        "schema_version": 2,
        "dataset": {
            "files": len(paths), "samples": int(quality["rows"]),
            "events": len(event_records), "nonfinite_rows": int(quality["nonfinite_rows"]),
            "duplicate_timestamps": int(quality["duplicate_timestamps"]),
            "large_gaps": int(quality["large_gaps"]),
            "median_sample_rate_hz": float(np.nanmedian(rates)) if rates else None,
            "timestamp_jitter_p95_ms": float(np.percentile(jitter_ms, 95)) if jitter_ms else None,
        },
        "planner": {
            "event_coverage": covered / len(event_records) if event_records else 0.0,
            "covered_events": covered, "forecasts": len(forecasts),
            "failure_reasons": dict(failure_reasons), "root_causes": dict(root_causes),
            "final_intercept_error_m": _stats(r["intercept_error_m"] for r in final_rows),
            "final_absolute_timing_error_s": _stats((r["timing_error_s"] for r in final_rows), absolute=True),
            "final_timing_bias_s": float(np.mean([r["timing_error_s"] for r in final_rows])) if final_rows else None,
            "final_ball_velocity_error_mps": _stats(r["ball_velocity_error_mps"] for r in final_rows),
            "side_accuracy": float(np.mean([r["side_correct"] for r in final_rows])) if final_rows else None,
            "unreachable_command_rate": float(np.mean([r["racket_speed_mps"] > thresholds.max_racket_speed_mps for r in final_rows])) if final_rows else None,
            "net_clearance_failure_rate": float(np.mean([r["net_margin_m"] < r["required_net_margin_m"] for r in final_rows])) if final_rows else None,
            "by_tts": _tts_bins(forecasts),
        },
        "configuration": {
            "x_hit_m": config.x_hit, "solve_period_s": solve_period_s,
            "fit_window_s": config.fit_window_s,
            "fit_window_sample_cap": config.fit_window,
            "poly_order_xy": config.poly_order_xy,
            "poly_order_z": config.poly_order_z,
            "min_ready_samples": config.min_ready_samples,
            "post_bounce_estimation": "real_samples_only",
            "split_y_m": split_y,
            "thresholds": asdict(thresholds),
        },
        "interpretation": {
            "ground_truth": "centred local polynomial fit to later mocap at the measured x_hit crossing",
            "causal_replay": True,
            "warning": "This evaluates planner commands, not robot tracking/contact success. Add racket pose and outgoing-ball logs for full closed-loop attribution.",
        },
    }
    return summary, forecasts, event_records


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, scale=1.0, suffix="") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * scale:.2f}{suffix}"


def _html_report(summary: dict, events: List[dict]) -> str:
    p, d = summary["planner"], summary["dataset"]
    err, timing = p["final_intercept_error_m"], p["final_absolute_timing_error_s"]
    rows = "".join(
        f"<tr><td>{html.escape(e['event_id'])}</td><td>{e['forecast_count']}</td>"
        f"<td>{_fmt(e['final_intercept_error_m'],1000,' mm')}</td>"
        f"<td>{_fmt(abs(e['final_timing_error_s']) if e['final_timing_error_s'] is not None else None,1000,' ms')}</td>"
        f"<td><span class='tag'>{html.escape(e['root_cause'])}</span>"
        f"{' / ' + html.escape(e['dominant_no_command_reason']) if e['dominant_no_command_reason'] else ''}</td></tr>" for e in events
    )
    bins = "".join(
        f"<tr><td>{b['tts_s']:.1f} s</td><td>{b['events']}</td>"
        f"<td>{_fmt(b['intercept_error_m']['median'],1000,' mm')} / "
        f"{_fmt(b['intercept_error_m']['p95'],1000,' mm')}</td>"
        f"<td>{_fmt(b['absolute_timing_error_s']['median'],1000,' ms')} / "
        f"{_fmt(b['absolute_timing_error_s']['p95'],1000,' ms')}</td></tr>"
        for b in p["by_tts"]
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>HOPE Planner Evaluation</title>
<style>body{{font:14px system-ui;margin:32px;max-width:1100px;color:#17202a}}h1{{margin-bottom:4px}}.muted{{color:#667085}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.card{{border:1px solid #ddd;border-radius:10px;padding:16px}}.big{{font-size:25px;font-weight:700}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{text-align:left;border-bottom:1px solid #eee;padding:8px}}th{{background:#f7f8fa}}.tag{{font:12px ui-monospace;background:#eef2f6;padding:3px 6px;border-radius:5px}}code{{background:#f2f4f7;padding:2px 4px}}</style></head><body>
<h1>HOPE Planner Evaluation</h1><div class='muted'>因果回放：预测只使用当时已到达的样本，后续动捕仅用于离线真值。</div>
<div class='cards'><div class='card'><div class='muted'>事件覆盖率</div><div class='big'>{p['event_coverage']*100:.1f}%</div><div>{p['covered_events']}/{d['events']} crossings</div></div>
<div class='card'><div class='muted'>最终截点误差 P50 / P95</div><div class='big'>{_fmt(err['median'],1000,' mm')}</div><div>{_fmt(err['p95'],1000,' mm')}</div></div>
<div class='card'><div class='muted'>最终时间误差 P50 / P95</div><div class='big'>{_fmt(timing['median'],1000,' ms')}</div><div>{_fmt(timing['p95'],1000,' ms')}</div></div>
<div class='card'><div class='muted'>正反手准确率</div><div class='big'>{_fmt(p['side_accuracy'],100,'%')}</div><div>{d['median_sample_rate_hz'] or 0:.1f} Hz mocap</div></div></div>
<h2>按 TTS 检查点分解</h2><table><tr><th>TTS</th><th>球数</th><th>截点误差 P50 / P95</th><th>时间误差 P50 / P95</th></tr>{bins}</table>
<h2>逐球根因</h2><table><tr><th>事件</th><th>预测数</th><th>最终截点误差</th><th>最终时间误差</th><th>首要问题</th></tr>{rows or '<tr><td colspan=5>没有检测到入射击球平面事件</td></tr>'}</table>
<h2>如何看明细</h2><p><code>predictions.csv</code> 用于查看每次 planner revision；<code>events.csv</code> 是逐球结论；<code>summary.json</code> 适合回归门禁和参数对比。</p>
<p class='muted'>注意：这份报告验证 planner 的状态、截点、时序、侧别和指令可行性，不验证机器人是否实际跟踪到球拍目标。闭环归因需要同步球拍位姿及击球后球轨迹。</p></body></html>"""


def write_report(output_dir: Path, summary: dict, forecasts: List[dict], events: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "predictions.csv", forecasts)
    _write_csv(output_dir / "events.csv", events)
    tts_rows = []
    for item in summary["planner"]["by_tts"]:
        tts_rows.append({
            "tts_s": item["tts_s"], "tolerance_s": item["tolerance_s"],
            "events": item["events"],
            "intercept_error_p50_m": item["intercept_error_m"]["median"],
            "intercept_error_p95_m": item["intercept_error_m"]["p95"],
            "absolute_y_error_p95_m": item["absolute_y_error_m"]["p95"],
            "absolute_z_error_p95_m": item["absolute_z_error_m"]["p95"],
            "absolute_timing_error_p50_s": item["absolute_timing_error_s"]["median"],
            "absolute_timing_error_p95_s": item["absolute_timing_error_s"]["p95"],
            "timing_bias_s": item["timing_bias_s"],
        })
    _write_csv(output_dir / "tts_metrics.csv", tts_rows)
    (output_dir / "report.html").write_text(_html_report(summary, events), encoding="utf-8")


def _expand_inputs(inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(value)
    return list(dict.fromkeys(p.resolve() for p in files))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Causally replay canonical mocap CSVs through HOPE planner.")
    ap.add_argument("inputs", nargs="+", help="Canonical t,x,y,z CSV file(s), or directories.")
    ap.add_argument("--output", default="planner_eval_report", help="Report directory.")
    ap.add_argument("--physics", default=None, help="configs/ball_physics.yaml path (auto-discovered by default).")
    ap.add_argument("--x-hit", type=float, default=0.2, help="Strike-plane x in world frame (m).")
    ap.add_argument("--split-y", type=float, default=-0.7625, help="Forehand/backhand split y (m).")
    ap.add_argument("--solve-period", type=float, default=0.02, help="Planner solve period (s); 0 solves every sample.")
    ap.add_argument("--fit-window-s", type=float, default=0.14, help="Estimator history in seconds.")
    ap.add_argument("--fit-window", type=int, default=67, help="Estimator hard sample-count cap.")
    ap.add_argument("--poly-order-xy", type=int, default=1, help="Horizontal polynomial order.")
    ap.add_argument("--poly-order-z", type=int, default=2, help="Vertical polynomial order.")
    ap.add_argument("--min-ready-samples", type=int, default=20, help="Real samples required after startup/bounce.")
    ap.add_argument("--max-racket-speed", type=float, default=6.0, help="Debug reachability threshold (m/s).")
    args = ap.parse_args(argv)

    paths = _expand_inputs(args.inputs)
    if not paths:
        ap.error("no CSV files found")
    physics = load_ball_physics(args.physics)
    paddle = load_paddle_params(args.physics)
    table = load_table_params(args.physics)
    config = PlannerConfig(
        x_hit=args.x_hit, fit_window_s=args.fit_window_s,
        fit_window=args.fit_window, min_ready_samples=args.min_ready_samples,
        poly_order_xy=args.poly_order_xy, poly_order_z=args.poly_order_z,
        C_r=paddle["C_r"],
        paddle_a_t=paddle["paddle_a_t"], paddle_b_t=paddle["paddle_b_t"],
        paddle_mu=paddle["paddle_mu"],
    )
    thresholds = EvalThresholds(max_racket_speed_mps=args.max_racket_speed)
    summary, forecasts, events = evaluate_files(
        paths, config, physics, table, solve_period_s=args.solve_period,
        split_y=args.split_y, thresholds=thresholds,
    )
    output = Path(args.output).resolve()
    write_report(output, summary, forecasts, events)
    compact = {
        "events": summary["dataset"]["events"],
        "coverage": summary["planner"]["event_coverage"],
        "intercept_error_p95_m": summary["planner"]["final_intercept_error_m"]["p95"],
        "timing_error_p95_s": summary["planner"]["final_absolute_timing_error_s"]["p95"],
        "report": str(output / "report.html"),
    }
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == "__main__":
    main()
