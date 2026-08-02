#!/usr/bin/env python3
"""Audit real no-spin racket-contact residuals from synchronized replay logs.

The audit treats a contact as a weak physical label only when the ball changes
velocity near the tracked racket and its relative normal velocity reverses.
Geometric face crossings alone are never promoted to physical contacts.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterable
import zipfile

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
BALL_FIT_ROOT = REPO_ROOT / "hope_training/ball_physics_fit"
if str(BALL_FIT_ROOT) not in sys.path:
    sys.path.insert(0, str(BALL_FIT_ROOT))

from contact_model import predict_contact  # noqa: E402


DEFAULT_SESSION_ZIP = Path("/mnt/ssd/zxl/1310数据包.zip")
DEFAULT_PHYSICS = REPO_ROOT / "configs/ball_physics.yaml"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "analysis/closed_loop_v2_20260731/n2_real_contact_residuals_20260731"
)

BALL_FIELDS = ("ball_x", "ball_y", "ball_z")
RACKET_FIELDS = ("tcp_x", "tcp_y", "tcp_z")
NORMAL_FIELDS = ("normal_x", "normal_y", "normal_z")


def _find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if name.endswith(suffix) and "/raw_hdu/" in name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one raw HDU member ending in {suffix!r}, found {matches}"
        )
    return matches[0]


def load_session(zip_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with zipfile.ZipFile(zip_path) as archive:
        sample_member = _find_member(archive, "racket_tcp_samples.csv")
        event_member = _find_member(archive, "racket_tcp_events.csv")
        with archive.open(sample_member) as binary:
            samples = list(
                csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
            )
        with archive.open(event_member) as binary:
            events = list(
                csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
            )
    return samples, events


def fit_velocity(
    samples: Iterable[dict[str, str]],
    fields: tuple[str, str, str],
    event_time_ns: int,
) -> tuple[np.ndarray, float, int]:
    rows = list(samples)
    if len(rows) < 5:
        return np.full(3, np.nan), float("nan"), len(rows)
    time_s = np.asarray(
        [(int(row["time_ns"]) - event_time_ns) * 1.0e-9 for row in rows],
        dtype=np.float64,
    )
    positions = np.asarray(
        [[float(row[field]) for field in fields] for row in rows],
        dtype=np.float64,
    )
    finite = np.isfinite(time_s) & np.all(np.isfinite(positions), axis=1)
    time_s = time_s[finite]
    positions = positions[finite]
    if len(time_s) < 5:
        return np.full(3, np.nan), float("nan"), len(time_s)
    design = np.column_stack((time_s, np.ones_like(time_s)))
    coefficients = np.linalg.lstsq(design, positions, rcond=None)[0]
    residual = positions - design @ coefficients
    rms_m = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return coefficients[0], rms_m, len(time_s)


def orient_normal(
    normal: np.ndarray,
    incoming_ball_velocity: np.ndarray,
    racket_velocity: np.ndarray,
) -> np.ndarray:
    oriented = np.asarray(normal, dtype=np.float64).copy()
    norm = float(np.linalg.norm(oriented))
    if not np.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("racket normal is invalid")
    oriented /= norm
    if float(np.dot(incoming_ball_velocity - racket_velocity, oriented)) > 0.0:
        oriented *= -1.0
    return oriented


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float64)
    reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    first = np.cross(normal, reference)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    second /= np.linalg.norm(second)
    return first, second


def passes_physical_contact_gate(
    *,
    distance_m: float,
    ball_height_m: float,
    delta_velocity_mps: float,
    incoming_relative_normal_mps: float,
    outgoing_relative_normal_mps: float,
    incoming_fit_rms_m: float,
    outgoing_fit_rms_m: float,
    distance_max_m: float = 0.13,
    ball_height_min_m: float = 0.08,
    delta_velocity_min_mps: float = 1.5,
    approach_speed_min_mps: float = 0.2,
    separation_speed_min_mps: float = 0.1,
    fit_rms_max_m: float = 0.015,
) -> bool:
    values = np.asarray(
        [
            distance_m,
            ball_height_m,
            delta_velocity_mps,
            incoming_relative_normal_mps,
            outgoing_relative_normal_mps,
            incoming_fit_rms_m,
            outgoing_fit_rms_m,
        ]
    )
    return bool(
        np.all(np.isfinite(values))
        and distance_m <= distance_max_m
        and ball_height_m >= ball_height_min_m
        and delta_velocity_mps >= delta_velocity_min_mps
        and incoming_relative_normal_mps <= -approach_speed_min_mps
        and outgoing_relative_normal_mps >= separation_speed_min_mps
        and incoming_fit_rms_m <= fit_rms_max_m
        and outgoing_fit_rms_m <= fit_rms_max_m
    )


def analyze_event(
    event: dict[str, str],
    samples: list[dict[str, str]],
    paddle_parameters: dict[str, float],
) -> dict[str, Any] | None:
    event_time_ns = int(event["time_ns"])
    nearby = [
        row
        for row in samples
        if abs(int(row["time_ns"]) - event_time_ns) <= 110_000_000
    ]
    before = [
        row
        for row in nearby
        if -0.100
        <= (int(row["time_ns"]) - event_time_ns) * 1.0e-9
        <= -0.020
    ]
    after = [
        row
        for row in nearby
        if 0.020
        <= (int(row["time_ns"]) - event_time_ns) * 1.0e-9
        <= 0.100
    ]
    racket_window = [
        row
        for row in nearby
        if abs((int(row["time_ns"]) - event_time_ns) * 1.0e-9) <= 0.030
    ]
    v_in, incoming_rms, incoming_count = fit_velocity(
        before, BALL_FIELDS, event_time_ns
    )
    v_out, outgoing_rms, outgoing_count = fit_velocity(
        after, BALL_FIELDS, event_time_ns
    )
    racket_velocity, racket_rms, racket_count = fit_velocity(
        racket_window, RACKET_FIELDS, event_time_ns
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (v_in, v_out, racket_velocity)
    ):
        return None
    raw_normal = np.asarray(
        [float(event[field]) for field in NORMAL_FIELDS], dtype=np.float64
    )
    normal = orient_normal(raw_normal, v_in, racket_velocity)
    relative_in = v_in - racket_velocity
    relative_out = v_out - racket_velocity
    incoming_normal = float(np.dot(relative_in, normal))
    outgoing_normal = float(np.dot(relative_out, normal))
    delta_velocity = float(np.linalg.norm(v_out - v_in))
    physical_contact = passes_physical_contact_gate(
        distance_m=float(event["ball_tcp_distance_m"]),
        ball_height_m=float(event["ball_z"]),
        delta_velocity_mps=delta_velocity,
        incoming_relative_normal_mps=incoming_normal,
        outgoing_relative_normal_mps=outgoing_normal,
        incoming_fit_rms_m=incoming_rms,
        outgoing_fit_rms_m=outgoing_rms,
    )
    predicted = predict_contact(
        v_in,
        racket_velocity,
        normal,
        paddle_parameters["restitution"],
        paddle_parameters["tangential_damping"],
        0.0,
        paddle_parameters["tangential_cap"],
    )["v_plus"][0]
    residual = v_out - predicted
    first_tangent, second_tangent = tangent_basis(normal)
    output: dict[str, Any] = {
        "task_id": int(event["task_id"]),
        "event_time_ns": event_time_ns,
        "event_type": event["event_type"],
        "geometric_contact_candidate": event["contact_candidate"] == "1",
        "physical_contact_weak_label": physical_contact,
        "label_strength": "weak_velocity_reversal" if physical_contact else "rejected",
        "pose_sync_error_ms": float(event["pose_sync_error_ms"]),
        "planner_time_to_strike_s": float(event["planner_tts_at_sample_s"]),
        "planner_normal_error_deg": float(event["normal_error_deg"]),
        "ball_tcp_distance_m": float(event["ball_tcp_distance_m"]),
        "ball_height_m": float(event["ball_z"]),
        "v_in_mps": v_in.tolist(),
        "v_out_mps": v_out.tolist(),
        "racket_velocity_mps": racket_velocity.tolist(),
        "racket_normal": normal.tolist(),
        "delta_velocity_mps": delta_velocity,
        "incoming_relative_normal_mps": incoming_normal,
        "outgoing_relative_normal_mps": outgoing_normal,
        "incoming_fit_rms_mm": incoming_rms * 1000.0,
        "outgoing_fit_rms_mm": outgoing_rms * 1000.0,
        "racket_fit_rms_mm": racket_rms * 1000.0,
        "incoming_fit_samples": incoming_count,
        "outgoing_fit_samples": outgoing_count,
        "racket_fit_samples": racket_count,
        "predicted_v_out_mps": predicted.tolist(),
        "residual_world_mps": residual.tolist(),
        "residual_normal_mps": float(np.dot(residual, normal)),
        "residual_tangent_1_mps": float(np.dot(residual, first_tangent)),
        "residual_tangent_2_mps": float(np.dot(residual, second_tangent)),
    }
    return output


def calibration_readiness(
    events: list[dict[str, Any]],
    session_count: int,
    *,
    minimum_contacts: int = 30,
    minimum_sessions: int = 3,
    minimum_holdout_contacts: int = 10,
) -> dict[str, Any]:
    contact_count = sum(
        bool(event["physical_contact_weak_label"]) for event in events
    )
    holdout_possible = (
        session_count >= minimum_sessions
        and contact_count >= minimum_contacts
        and contact_count // session_count >= minimum_holdout_contacts
    )
    blockers: list[str] = []
    if contact_count < minimum_contacts:
        blockers.append(
            f"need at least {minimum_contacts} weak physical contacts; have {contact_count}"
        )
    if session_count < minimum_sessions:
        blockers.append(
            f"need at least {minimum_sessions} independent sessions; have {session_count}"
        )
    if not holdout_possible:
        blockers.append("session-group holdout cannot be formed with >=10 contacts")
    return {
        "calibration_ready": not blockers,
        "physical_contact_count": contact_count,
        "session_count": session_count,
        "minimum_contacts": minimum_contacts,
        "minimum_sessions": minimum_sessions,
        "minimum_holdout_contacts": minimum_holdout_contacts,
        "session_group_holdout_possible": holdout_possible,
        "blockers": blockers,
    }


def provisional_distribution(events: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        event for event in events if event["physical_contact_weak_label"]
    ]
    if not selected:
        return {"status": "unavailable", "count": 0}
    tangent = np.asarray(
        [
            [
                event["residual_tangent_1_mps"],
                event["residual_tangent_2_mps"],
            ]
            for event in selected
        ],
        dtype=np.float64,
    )
    normal = np.asarray(
        [event["residual_normal_mps"] for event in selected], dtype=np.float64
    )
    covariance = (
        np.cov(tangent, rowvar=False)
        if len(tangent) > 1
        else np.zeros((2, 2), dtype=np.float64)
    )
    return {
        "status": "descriptive_only_not_trainable",
        "label_strength": "weak_velocity_reversal",
        "count": len(selected),
        "tangent_mean_mps": np.mean(tangent, axis=0).tolist(),
        "tangent_covariance_mps2": np.asarray(covariance).tolist(),
        "tangent_residual_norm_median_mps": float(
            np.median(np.linalg.norm(tangent, axis=1))
        ),
        "tangent_residual_norm_p90_mps": float(
            np.quantile(np.linalg.norm(tangent, axis=1), 0.9)
        ),
        "normal_residual_mean_mps": float(np.mean(normal)),
        "normal_residual_abs_p90_mps": float(np.quantile(np.abs(normal), 0.9)),
        "warning": (
            "Do not use as a training distribution until independent-session "
            "holdout coverage passes the contact contract."
        ),
    }


def _flatten_event(event: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in event.items():
        if isinstance(value, list):
            for index, component in enumerate(value):
                flat[f"{key}_{index}"] = component
        else:
            flat[key] = value
    return flat


def run(
    session_zips: list[Path],
    physics_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    physics = yaml.safe_load(physics_path.read_text(encoding="utf-8"))
    paddle = physics["contact"]["paddle"]
    parameters = {
        "restitution": float(paddle["restitution"]),
        "tangential_damping": float(paddle["tangential_damping"]),
        "tangential_cap": float(paddle["tangential_cap"]),
    }
    all_events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for zip_path in session_zips:
        samples, events = load_session(zip_path)
        analyzed = [
            result
            for event in events
            if (result := analyze_event(event, samples, parameters)) is not None
        ]
        session_id = zip_path.stem
        for event in analyzed:
            event["session_id"] = session_id
        all_events.extend(analyzed)
        sources.append(
            {
                "path": str(zip_path),
                "session_id": session_id,
                "sample_rows": len(samples),
                "event_rows": len(events),
                "analyzed_events": len(analyzed),
                "weak_physical_contacts": sum(
                    bool(event["physical_contact_weak_label"])
                    for event in analyzed
                ),
            }
        )
    readiness = calibration_readiness(all_events, len(session_zips))
    distribution = provisional_distribution(all_events)
    summary = {
        "schema_version": 1,
        "audit_id": "hope_real_contact_residual_audit_n2",
        "physics_source": str(physics_path),
        "paddle_parameters": parameters,
        "contact_label_definition": (
            "tracked ball velocity reversal near tracked racket; no force sensor"
        ),
        "geometric_events_are_physical_labels": False,
        "sources": sources,
        "readiness": readiness,
        "provisional_distribution": distribution,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "provisional_distribution.yaml").write_text(
        yaml.safe_dump(distribution, sort_keys=False),
        encoding="utf-8",
    )
    rows = [_flatten_event(event) for event in all_events]
    if rows:
        with (output_dir / "per_event_audit.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-zip",
        action="append",
        type=Path,
        dest="session_zips",
        help="synchronized real replay archive; may be supplied multiple times",
    )
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_zips = args.session_zips or [DEFAULT_SESSION_ZIP]
    summary = run(session_zips, args.physics, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
