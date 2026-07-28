import csv

import numpy as np

from hope_planner.constants import BallPhysics, PlannerConfig, TableParams
from hope_planner.evaluation import evaluate_files, find_incoming_crossings, write_report


def _write_trajectory(path, t, p):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "z"])
        writer.writerows(np.column_stack([t, p]))


def test_crossing_reference_interpolates_plane():
    t = np.array([0.0, 0.1, 0.2])
    p = np.array([[0.5, -0.7, 0.4], [0.3, -0.7, 0.4], [0.1, -0.7, 0.4]])
    crossings = find_incoming_crossings(t, p, 0.2, "shot.csv")
    assert len(crossings) == 1
    assert np.isclose(crossings[0].t, 0.15)
    assert np.isclose(crossings[0].p[0], 0.2)


def test_causal_replay_produces_debug_metrics(tmp_path):
    physics = BallPhysics(k=0.0)
    config = PlannerConfig(x_hit=0.2, fit_window=15, max_predict_time=1.0)
    table = TableParams()
    t = np.arange(0.0, 0.46, 0.005)
    # Exact drag-free ballistic arc, incoming and high enough to avoid a bounce.
    p0 = np.array([1.0, -0.70, 0.65])
    v0 = np.array([-2.0, 0.15, 1.0])
    p = p0 + t[:, None] * v0 + 0.5 * t[:, None] ** 2 * physics.g
    path = tmp_path / "shot.csv"
    _write_trajectory(path, t, p)

    summary, forecasts, events = evaluate_files(
        [path], config, physics, table, solve_period_s=0.02, split_y=-0.7625,
    )

    assert summary["dataset"]["events"] == 1
    assert summary["planner"]["event_coverage"] == 1.0
    assert forecasts
    assert events[0]["root_cause"] in {"OK", "COMMAND_UNREACHABLE"}
    assert summary["planner"]["final_intercept_error_m"]["p95"] < 0.02
    assert summary["planner"]["final_absolute_timing_error_s"]["p95"] < 0.02

    report_dir = tmp_path / "report"
    write_report(report_dir, summary, forecasts, events)
    assert (report_dir / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (report_dir / "summary.json").is_file()
    assert (report_dir / "events.csv").is_file()
    assert (report_dir / "predictions.csv").is_file()


def test_wrong_schema_has_actionable_error(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("Frame,Time,X\n0,0,1\n", encoding="utf-8")
    try:
        evaluate_files([path], PlannerConfig(), BallPhysics(), TableParams())
    except ValueError as exc:
        assert "canonical t,x,y,z" in str(exc)
    else:
        raise AssertionError("expected schema validation failure")
