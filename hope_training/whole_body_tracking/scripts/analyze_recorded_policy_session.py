#!/usr/bin/env python3
"""Audit a recorded A3 hardware policy session without rerunning the planner.

The recorder stores the final 114-D observation seen by the actor, its 31-D
raw/applied actions, the accepted command lifecycle, and measured robot state.
This tool validates those boundaries against the deployed ONNX and exports a
compact, task-indexed dataset for deterministic MuJoCo replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import pathlib
import zipfile
from collections import Counter, defaultdict

import numpy as np
import onnxruntime as ort
import yaml


OBS_DIM = 114
ACTION_DIM = 31
COMMAND_SLICE = slice(103, 114)
JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
JOINT_GROUPS = {
    "waist": np.arange(0, 3),
    "left_arm": np.arange(5, 12),
    "right_arm": np.arange(12, 19),
    "legs": np.arange(19, 31),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=pathlib.Path, required=True)
    parser.add_argument("--session-id", default="20260731_125723_761563263")
    parser.add_argument("--onnx", type=pathlib.Path, required=True)
    parser.add_argument("--action-adapter", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--counterfactual-samples",
        type=int,
        default=2048,
        help="Maximum active-command frames used for ONNX command sensitivity.",
    )
    return parser.parse_args()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _members_for_session(archive: zipfile.ZipFile, session_id: str) -> dict[str, str]:
    suffixes = {
        "policy": [
            f"sessions/{session_id}/raw_mdu/policy_trace.csv",
            f"sessions/{session_id}/assembled/mdu/policy_trace.csv",
        ],
        "tcp_samples": [
            f"sessions/{session_id}/raw_hdu/racket_tcp_samples.csv",
            f"sessions/{session_id}/assembled/hdu/racket_tcp_samples.csv",
        ],
        "tcp_events": [
            f"sessions/{session_id}/raw_hdu/racket_tcp_events.csv",
            f"sessions/{session_id}/assembled/hdu/racket_tcp_events.csv",
        ],
        "hit_replay": [f"sessions/{session_id}/hit_replay.json"],
    }
    names = archive.namelist()
    resolved: dict[str, str] = {}
    for key, candidates in suffixes.items():
        for suffix in candidates:
            matches = [name for name in names if name.endswith(suffix)]
            if len(matches) == 1:
                resolved[key] = matches[0]
                break
            if len(matches) > 1:
                raise ValueError(
                    f"expected one archive member ending in {suffix!r}, got {matches}"
                )
        if key not in resolved:
            raise ValueError(
                f"none of the archive members for {key!r} exists: {candidates}"
            )
    return resolved


def _finite(row: dict[str, str], name: str, default: float = np.nan) -> float:
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _vector(row: dict[str, str], prefix: str, size: int) -> np.ndarray:
    return np.asarray([_finite(row, f"{prefix}{idx}") for idx in range(size)])


def _read_policy_rows(
    archive: zipfile.ZipFile, member: str
) -> tuple[list[str], list[dict[str, str]]]:
    with archive.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        reader = csv.DictReader(text)
        fields = list(reader.fieldnames or ())
        rows = [row for row in reader if None not in row and len(row) == len(fields)]
    if len(fields) != 382:
        raise ValueError(f"policy trace has {len(fields)} columns, expected 382")
    if not rows:
        raise ValueError("policy trace has no complete rows")
    return fields, rows


def _matrix(rows: list[dict[str, str]], prefix: str, size: int) -> np.ndarray:
    return np.stack([_vector(row, prefix, size) for row in rows])


def _column(rows: list[dict[str, str]], name: str, default: float = np.nan) -> np.ndarray:
    return np.asarray([_finite(row, name, default) for row in rows])


def _reconstruct_active_task_id(
    task_id: np.ndarray,
    phase: np.ndarray,
    revision_accept: np.ndarray,
    revision_reason: np.ndarray,
) -> np.ndarray:
    """Recover the task retained by the MDU lifecycle.

    ``policy_trace.task_id`` describes the newest UDP packet and falls back to
    zero when there is no fresh command. The actor target, however, is retained
    through swing, follow-through, and recovery. An accepted initial revision is
    therefore the authoritative start of an active lifecycle task.
    """
    active_task_id = np.full(task_id.shape, -1, dtype=np.int64)
    current_task = -1
    for index in range(task_id.size):
        if (
            revision_accept[index]
            and revision_reason[index] == "accepted_initial"
            and task_id[index] > 0
        ):
            current_task = int(task_id[index])
        if phase[index] == 0:
            current_task = -1
        else:
            active_task_id[index] = current_task
    missing = (phase != 0) & (active_task_id < 0)
    if np.any(missing):
        first = int(np.flatnonzero(missing)[0])
        raise ValueError(
            "cannot reconstruct lifecycle active task at policy frame "
            f"{first}: phase={phase[first]}, packet task={task_id[first]}"
        )
    return active_task_id


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _error(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1.0e-9:
        return None
    cosine = float(np.dot(a, b) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _fit_track_velocity(
    time_ns: np.ndarray,
    position: np.ndarray,
    event_ns: int,
    begin_ms: float,
    end_ms: float,
) -> np.ndarray | None:
    relative_ms = (time_ns.astype(np.float64) - float(event_ns)) * 1.0e-6
    mask = (
        (relative_ms >= begin_ms)
        & (relative_ms <= end_ms)
        & np.all(np.isfinite(position), axis=1)
    )
    if np.sum(mask) < 4:
        return None
    time_s = (time_ns[mask].astype(np.float64) - float(event_ns)) * 1.0e-9
    design = np.column_stack((time_s, np.ones(time_s.shape)))
    velocity = np.linalg.lstsq(design, position[mask], rcond=None)[0][0]
    return np.asarray(velocity, dtype=np.float64)


def _first_forward_plane_crossing(
    time_ns: np.ndarray,
    position: np.ndarray,
    event_ns: int,
    x_plane: float,
    horizon_s: float = 2.0,
) -> tuple[int, np.ndarray] | None:
    mask = (
        (time_ns >= event_ns)
        & (time_ns <= event_ns + int(horizon_s * 1.0e9))
        & np.all(np.isfinite(position), axis=1)
    )
    times = time_ns[mask]
    points = position[mask]
    for index in range(1, times.size):
        before, after = points[index - 1], points[index]
        if before[0] < x_plane <= after[0] and after[0] > before[0]:
            fraction = float(
                (x_plane - before[0]) / max(after[0] - before[0], 1.0e-12)
            )
            crossing = before + fraction * (after - before)
            crossing_ns = int(
                times[index - 1]
                + fraction * (int(times[index]) - int(times[index - 1]))
            )
            return crossing_ns, crossing
    return None


def _first_downward_surface_crossing(
    time_ns: np.ndarray,
    position: np.ndarray,
    event_ns: int,
    z_surface: float,
    horizon_s: float = 2.0,
) -> tuple[int, np.ndarray] | None:
    mask = (
        (time_ns >= event_ns)
        & (time_ns <= event_ns + int(horizon_s * 1.0e9))
        & np.all(np.isfinite(position), axis=1)
    )
    times = time_ns[mask]
    points = position[mask]
    for index in range(1, times.size):
        before, after = points[index - 1], points[index]
        if before[2] > z_surface >= after[2] and after[2] < before[2]:
            fraction = float(
                (before[2] - z_surface) / max(before[2] - after[2], 1.0e-12)
            )
            crossing = before + fraction * (after - before)
            crossing_ns = int(
                times[index - 1]
                + fraction * (int(times[index]) - int(times[index - 1]))
            )
            return crossing_ns, crossing
    return None


def _load_adapter(path: pathlib.Path) -> dict[str, np.ndarray]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    joint_names = list(doc.get("joint_order", JOINT_NAMES))

    def resolve(spec, field: str) -> np.ndarray:
        if isinstance(spec, dict):
            missing = [name for name in joint_names if name not in spec]
            if missing:
                raise ValueError(f"{field} missing joints: {missing}")
            return np.asarray([float(spec[name]) for name in joint_names])
        values = np.asarray(spec, dtype=np.float64).reshape(-1)
        if values.shape != (ACTION_DIM,):
            raise ValueError(f"{field} has shape {values.shape}, expected {(ACTION_DIM,)}")
        return values

    default_q = resolve(doc["default_q"], "default_q")
    scale_spec = doc["action_scale"]
    action_scale = (
        np.full(ACTION_DIM, float(scale_spec))
        if isinstance(scale_spec, (int, float))
        else resolve(scale_spec, "action_scale")
    )
    clamp = doc["joint_position_clamp"]
    return {
        "default_q": default_q,
        "action_scale": action_scale,
        "lower": resolve(clamp["lower"], "joint_position_clamp.lower"),
        "upper": resolve(clamp["upper"], "joint_position_clamp.upper"),
    }


def _infer(session: ort.InferenceSession, observations: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return np.stack(
        [
            np.asarray(
                session.run([output_name], {input_name: obs[None].astype(np.float32)})[0]
            ).reshape(ACTION_DIM)
            for obs in observations
        ]
    )


def _counterfactual_sensitivity(
    session: ort.InferenceSession,
    observations: np.ndarray,
    base_pos: np.ndarray,
    target_pos: np.ndarray,
    swing_side: np.ndarray,
    fixed_station: np.ndarray,
    active: np.ndarray,
    max_samples: int,
) -> list[dict[str, float | str | int]]:
    indices = np.flatnonzero(active)
    if indices.size > max_samples:
        indices = indices[np.linspace(0, indices.size - 1, max_samples, dtype=int)]
    if indices.size == 0:
        return []
    obs = observations[indices].copy()
    baseline = _infer(session, obs)
    variants: dict[str, np.ndarray] = {}

    value = obs.copy()
    value[:, 101:103] = fixed_station[indices] - base_pos[indices, :2]
    variants["fixed_station"] = value

    value = obs.copy()
    side = np.where(swing_side[indices] >= 0.0, 1.0, -1.0)
    ready_pos = base_pos[indices] + np.column_stack(
        (
            np.full(indices.size, 0.40),
            0.20 * side,
            np.full(indices.size, -0.05),
        )
    )
    value[:, 103:106] = ready_pos - base_pos[indices]
    variants["position_to_ready"] = value

    value = obs.copy()
    value[:, 106:109] = 0.0
    variants["zero_target_velocity"] = value

    value = obs.copy()
    value[:, 109] = 1.0
    variants["ready_timing"] = value

    value = obs.copy()
    value[:, 111:114] = np.array([1.0, 0.0, 0.0])
    variants["default_normal"] = value

    value = obs.copy()
    value[:, COMMAND_SLICE] = 0.0
    variants["zero_command_slice"] = value

    result: list[dict[str, float | str | int]] = []
    for name, variant in variants.items():
        action = _infer(session, variant)
        delta = action - baseline
        row: dict[str, float | str | int] = {
            "variant": name,
            "samples": int(indices.size),
            "all_action_l2_median": float(np.median(np.linalg.norm(delta, axis=1))),
            "all_action_l2_p95": float(np.quantile(np.linalg.norm(delta, axis=1), 0.95)),
            "all_action_max_abs": float(np.max(np.abs(delta))),
        }
        for group, group_indices in JOINT_GROUPS.items():
            norms = np.linalg.norm(delta[:, group_indices], axis=1)
            row[f"{group}_l2_median"] = float(np.median(norms))
            row[f"{group}_l2_p95"] = float(np.quantile(norms, 0.95))
        result.append(row)
    return result


def _read_csv_dicts(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with archive.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        return list(csv.DictReader(text))


def _load_hit_summary(archive: zipfile.ZipFile, member: str) -> tuple[dict, dict[int, int]]:
    with archive.open(member) as raw:
        doc = json.load(io.TextIOWrapper(raw, encoding="utf-8"))
    crossings = Counter(
        int(item["task_id"])
        for item in doc.get("ball_face_crossings", [])
        if item.get("task_id") is not None
    )
    return doc.get("summary", {}), dict(crossings)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        members = _members_for_session(archive, args.session_id)
        _, rows = _read_policy_rows(archive, members["policy"])
        tcp_samples = _read_csv_dicts(archive, members["tcp_samples"])
        tcp_events = _read_csv_dicts(archive, members["tcp_events"])
        hit_summary, crossing_count = _load_hit_summary(
            archive, members["hit_replay"]
        )

    system_ns = _column(rows, "system_ns").astype(np.int64)
    task_id = _column(rows, "task_id", -1).astype(np.int64)
    task_revision = _column(rows, "task_revision", -1).astype(np.int64)
    phase = _column(rows, "phase", 0).astype(np.int8)
    command_valid = _column(rows, "command_valid", 0).astype(bool)
    revision_accept = _column(rows, "revision_accept", 0).astype(bool)
    revision_reason = np.asarray(
        [str(row.get("revision_reject_reason", "")) for row in rows]
    )
    active_task_id = _reconstruct_active_task_id(
        task_id, phase, revision_accept, revision_reason
    )
    dynamic_active = _column(rows, "dynamic_station_active", 0).astype(bool)
    base_pos = _matrix(rows, "base_pos", 3)
    base_quat = _matrix(rows, "base_quat", 4)
    station = _matrix(rows, "station", 2)
    fixed_station = _matrix(rows, "fixed_station", 2)
    target_pos = _matrix(rows, "target_pos", 3)
    target_vel = _matrix(rows, "target_vel", 3)
    target_normal = _matrix(rows, "target_normal", 3)
    time_to_strike = _column(rows, "time_to_strike")
    swing_side = _column(rows, "swing_side")
    q = _matrix(rows, "q", ACTION_DIM)
    dq = _matrix(rows, "dq", ACTION_DIM)
    tau = _matrix(rows, "tau", ACTION_DIM)
    obs = _matrix(rows, "obs", OBS_DIM).astype(np.float32)
    raw_action = _matrix(rows, "raw_action", ACTION_DIM)
    applied_action = _matrix(rows, "applied_action", ACTION_DIM)
    q_des = _matrix(rows, "q_des", ACTION_DIM)

    onnx_session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    replay_action = _infer(onnx_session, obs)
    adapter = _load_adapter(args.action_adapter)
    expected_q_des = np.clip(
        adapter["default_q"] + applied_action * adapter["action_scale"],
        adapter["lower"],
        adapter["upper"],
    )

    active = (phase == 1) | (phase == 2)
    command_frames = command_valid | active
    current_station_error = station - base_pos[:, :2]
    fixed_station_error = fixed_station - base_pos[:, :2]
    target_rel = target_pos - base_pos
    last_action_expected = np.vstack(
        (np.full((1, ACTION_DIM), np.nan), applied_action[:-1])
    )
    contiguous = np.diff(system_ns) < 40_000_000
    valid_last = np.concatenate(([False], contiguous))

    preclip_q_des = adapter["default_q"] + applied_action * adapter["action_scale"]
    clamp_mask = (preclip_q_des < adapter["lower"] - 1.0e-9) | (
        preclip_q_des > adapter["upper"] + 1.0e-9
    )
    tracking_error = q_des - q
    policy_dt = np.diff(system_ns) * 1.0e-9
    valid_rate = contiguous & (policy_dt > 1.0e-6)
    swing_follow_rate = valid_rate & active[:-1] & active[1:]
    q_des_rate = np.diff(q_des, axis=0) / np.maximum(policy_dt[:, None], 1.0e-6)

    joint_rows: list[dict] = []
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        raw_active = np.abs(raw_action[active, joint_index])
        track_active = np.abs(tracking_error[active, joint_index])
        rate_active = np.abs(q_des_rate[swing_follow_rate, joint_index])
        joint_rows.append(
            {
                "joint_index": joint_index,
                "joint_name": joint_name,
                "raw_action_abs_median_swing_follow": float(np.median(raw_active)),
                "raw_action_abs_p95_swing_follow": float(
                    np.quantile(raw_active, 0.95)
                ),
                "raw_action_abs_max_swing_follow": float(np.max(raw_active)),
                "target_clamp_fraction_all": float(
                    np.mean(clamp_mask[:, joint_index])
                ),
                "target_clamp_fraction_swing_follow": float(
                    np.mean(clamp_mask[active, joint_index])
                ),
                "q_tracking_abs_median_swing_follow_rad": float(
                    np.median(track_active)
                ),
                "q_tracking_abs_p95_swing_follow_rad": float(
                    np.quantile(track_active, 0.95)
                ),
                "q_des_rate_abs_p95_swing_follow_rad_s": (
                    float(np.quantile(rate_active, 0.95))
                    if rate_active.size
                    else None
                ),
                "tau_abs_p95_swing_follow": float(
                    np.quantile(np.abs(tau[active, joint_index]), 0.95)
                ),
            }
        )
    _write_csv(args.output_dir / "joint_diagnostics.csv", joint_rows)

    latency_fields = {
        name: _stats(_column(rows, name)[command_valid])
        for name in (
            "planner_age_ms",
            "command_header_age_ms",
            "bridge_queue_age_ms",
            "packet_age_ms",
            "actuation_lead_ms",
        )
    }

    sensitivity = _counterfactual_sensitivity(
        onnx_session,
        obs,
        base_pos,
        target_pos,
        swing_side,
        fixed_station,
        command_frames,
        args.counterfactual_samples,
    )
    _write_csv(args.output_dir / "counterfactual_action_sensitivity.csv", sensitivity)

    contact_tasks = Counter(
        int(_finite(row, "task_id", -1))
        for row in tcp_events
        if str(row.get("contact_candidate", "0")) == "1"
    )
    event_by_task: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in tcp_events:
        event_by_task[int(_finite(row, "task_id", -1))].append(row)

    task_rows: list[dict] = []
    for current_task in sorted(set(active_task_id[active_task_id > 0])):
        mask = active_task_id == current_task
        swing_follow_mask = mask & active
        if not np.any(swing_follow_mask):
            continue
        indices = np.flatnonzero(mask)
        swing_follow_indices = np.flatnonzero(swing_follow_mask)
        first, last = int(indices[0]), int(indices[-1])
        base_delta = base_pos[last] - base_pos[first]
        task_tracking = np.linalg.norm(tracking_error[indices], axis=1)
        task_rows.append(
            {
                "task_id": int(current_task),
                "first_system_ns": int(system_ns[first]),
                "last_system_ns": int(system_ns[last]),
                "frames": int(indices.size),
                "swing_follow_frames": int(swing_follow_indices.size),
                "recovery_frames": int(np.sum(mask & (phase == 3))),
                "accepted_revisions": int(
                    np.sum(revision_accept & (task_id == current_task))
                ),
                "side": (
                    "forehand"
                    if np.median(swing_side[swing_follow_mask]) >= 0
                    else "backhand"
                ),
                "first_tts_s": float(time_to_strike[swing_follow_indices[0]]),
                "min_tts_s": float(np.min(time_to_strike[swing_follow_indices])),
                "target_x_median": float(
                    np.median(target_pos[swing_follow_indices, 0])
                ),
                "target_y_median": float(
                    np.median(target_pos[swing_follow_indices, 1])
                ),
                "target_z_median": float(
                    np.median(target_pos[swing_follow_indices, 2])
                ),
                "target_speed_median": float(
                    np.median(
                        np.linalg.norm(target_vel[swing_follow_indices], axis=1)
                    )
                ),
                "base_dx_m": float(base_delta[0]),
                "base_dy_m": float(base_delta[1]),
                "base_dz_m": float(base_delta[2]),
                "q_tracking_l2_median": float(np.median(task_tracking)),
                "q_tracking_l2_p95": float(np.quantile(task_tracking, 0.95)),
                "raw_action_max_abs": float(np.max(np.abs(raw_action[indices]))),
                "q_des_clamp_fraction": float(np.mean(clamp_mask[indices])),
                "tcp_contact_candidates": int(contact_tasks.get(int(current_task), 0)),
                "ball_face_crossings": int(crossing_count.get(int(current_task), 0)),
            }
        )
    _write_csv(args.output_dir / "task_summary.csv", task_rows)

    ball_time_ns = np.asarray(
        [_finite(row, "time_ns", -1) for row in tcp_samples], dtype=np.int64
    )
    ball_task_id = np.asarray(
        [_finite(row, "task_id", -1) for row in tcp_samples], dtype=np.int64
    )
    ball_command_valid = np.asarray(
        [_finite(row, "command_valid", 0) for row in tcp_samples], dtype=np.int8
    )
    ball_pos = np.asarray(
        [
            [_finite(row, f"ball_{axis}") for axis in "xyz"]
            for row in tcp_samples
        ],
        dtype=np.float32,
    )
    tcp_pos = np.asarray(
        [
            [_finite(row, f"tcp_{axis}") for axis in "xyz"]
            for row in tcp_samples
        ],
        dtype=np.float32,
    )
    tcp_normal = np.asarray(
        [
            [_finite(row, f"normal_{axis}") for axis in "xyz"]
            for row in tcp_samples
        ],
        dtype=np.float32,
    )

    impact_rows: list[dict] = []
    high_confidence_impact_tasks: list[int] = []
    for event in tcp_events:
        event_ns = int(_finite(event, "time_ns", -1))
        event_task = int(_finite(event, "task_id", -1))
        # Planner task IDs can clear immediately after impact. Use the continuous
        # Motive timeline for local kinematics and the post-impact flight; the
        # next serve is several seconds away, outside the two-second outcome
        # horizon.
        sample_mask = (
            (ball_time_ns > 0)
            & np.all(np.isfinite(ball_pos), axis=1)
            & np.all(np.isfinite(tcp_pos), axis=1)
        )
        pre_velocity = _fit_track_velocity(
            ball_time_ns[sample_mask],
            ball_pos[sample_mask],
            event_ns,
            -80.0,
            -15.0,
        )
        post_velocity = _fit_track_velocity(
            ball_time_ns[sample_mask],
            ball_pos[sample_mask],
            event_ns,
            15.0,
            80.0,
        )
        tcp_velocity = _fit_track_velocity(
            ball_time_ns[sample_mask],
            tcp_pos[sample_mask],
            event_ns,
            -35.0,
            35.0,
        )
        nearest_policy_index = int(np.argmin(np.abs(system_ns - event_ns)))
        nearest_policy_dt_ms = float(
            abs(int(system_ns[nearest_policy_index]) - event_ns) * 1.0e-6
        )
        policy_target_velocity = np.asarray(
            target_vel[nearest_policy_index], dtype=np.float64
        )
        event_tcp_position = np.asarray(
            [_finite(event, f"tcp_{axis}") for axis in "xyz"], dtype=np.float64
        )
        policy_target_position = np.asarray(
            target_pos[nearest_policy_index], dtype=np.float64
        )
        policy_target_normal = np.asarray(
            target_normal[nearest_policy_index], dtype=np.float64
        )
        target_position_error = event_tcp_position - policy_target_position
        velocity_error = (
            tcp_velocity - policy_target_velocity
            if tcp_velocity is not None
            else None
        )
        velocity_angle = (
            _angle_deg(tcp_velocity, policy_target_velocity)
            if tcp_velocity is not None
            else None
        )
        event_times = ball_time_ns[sample_mask]
        event_ball = ball_pos[sample_mask]
        event_order = np.argsort(event_times)
        event_times = event_times[event_order]
        event_ball = event_ball[event_order]
        net_crossing = _first_forward_plane_crossing(
            event_times, event_ball, event_ns, 1.37
        )
        first_bounce = _first_downward_surface_crossing(
            event_times, event_ball, event_ns, 0.02
        )
        if (
            net_crossing is not None
            and first_bounce is not None
            and net_crossing[0] >= first_bounce[0]
        ):
            net_crossing = None
        net_cross_z = (
            float(net_crossing[1][2]) if net_crossing is not None else None
        )
        net_clear = bool(net_cross_z is not None and net_cross_z > 0.1725)
        opponent_bounce = bool(
            first_bounce is not None
            and 1.37 <= first_bounce[1][0] <= 2.74
            and -1.525 <= first_bounce[1][1] <= 0.0
        )
        delta_velocity = (
            post_velocity - pre_velocity
            if pre_velocity is not None and post_velocity is not None
            else None
        )
        candidate = str(event.get("contact_candidate", "0")) == "1"
        x_reversal = bool(
            pre_velocity is not None
            and post_velocity is not None
            and pre_velocity[0] < -0.2
            and post_velocity[0] > 0.2
        )
        impulse_norm = (
            float(np.linalg.norm(delta_velocity))
            if delta_velocity is not None
            else None
        )
        high_confidence = bool(
            candidate
            and x_reversal
            and impulse_norm is not None
            and impulse_norm >= 1.0
        )
        planner_aligned_impact = bool(
            high_confidence
            and np.linalg.norm(target_position_error) <= 0.06
            and abs(float(time_to_strike[nearest_policy_index])) <= 0.05
        )
        if high_confidence:
            high_confidence_impact_tasks.append(event_task)
        impact_rows.append(
            {
                "task_id": event_task,
                "event_type": str(event.get("event_type", "")),
                "event_time_ns": event_ns,
                "contact_candidate": int(candidate),
                "ball_tcp_distance_m": _finite(event, "ball_tcp_distance_m"),
                "normal_distance_m": _finite(event, "normal_distance_m"),
                "tangent_distance_m": _finite(event, "tangent_distance_m"),
                "planner_tts_at_sample_s": _finite(
                    event, "planner_tts_at_sample_s"
                ),
                "pre_ball_vx_mps": (
                    float(pre_velocity[0]) if pre_velocity is not None else None
                ),
                "pre_ball_vy_mps": (
                    float(pre_velocity[1]) if pre_velocity is not None else None
                ),
                "pre_ball_vz_mps": (
                    float(pre_velocity[2]) if pre_velocity is not None else None
                ),
                "post_ball_vx_mps": (
                    float(post_velocity[0]) if post_velocity is not None else None
                ),
                "post_ball_vy_mps": (
                    float(post_velocity[1]) if post_velocity is not None else None
                ),
                "post_ball_vz_mps": (
                    float(post_velocity[2]) if post_velocity is not None else None
                ),
                "ball_delta_v_mps": impulse_norm,
                "ball_impulse_target_normal_angle_deg": (
                    _angle_deg(delta_velocity, policy_target_normal)
                    if delta_velocity is not None
                    else None
                ),
                "x_velocity_reversal_to_opponent": int(x_reversal),
                "high_confidence_impact": int(high_confidence),
                "planner_aligned_impact": int(planner_aligned_impact),
                "nearest_policy_frame_dt_ms": nearest_policy_dt_ms,
                "policy_tts_at_event_s": float(
                    time_to_strike[nearest_policy_index]
                ),
                "tcp_speed_mps": (
                    float(np.linalg.norm(tcp_velocity))
                    if tcp_velocity is not None
                    else None
                ),
                "target_speed_mps": float(
                    np.linalg.norm(policy_target_velocity)
                ),
                "tcp_target_velocity_error_mps": (
                    float(np.linalg.norm(velocity_error))
                    if velocity_error is not None
                    else None
                ),
                "tcp_target_velocity_angle_deg": velocity_angle,
                "tcp_target_position_error_m": float(
                    np.linalg.norm(target_position_error)
                ),
                "tcp_target_normal_error_deg": _finite(
                    event, "normal_error_deg"
                ),
                "real_net_cross": int(net_crossing is not None),
                "real_net_cross_z_table_m": net_cross_z,
                "real_net_clear": int(net_clear),
                "real_first_bounce_x_table_m": (
                    float(first_bounce[1][0]) if first_bounce is not None else None
                ),
                "real_first_bounce_y_table_m": (
                    float(first_bounce[1][1]) if first_bounce is not None else None
                ),
                "real_opponent_bounce": int(opponent_bounce),
                "real_return_success": int(
                    high_confidence and net_clear and opponent_bounce
                ),
            }
        )
    _write_csv(args.output_dir / "real_contact_kinematics.csv", impact_rows)

    np.savez_compressed(
        args.output_dir / "recorded_policy_replay.npz",
        system_ns=system_ns,
        task_id=task_id,
        active_task_id=active_task_id,
        task_revision=task_revision,
        phase=phase,
        command_valid=command_valid.astype(np.int8),
        revision_accept=revision_accept.astype(np.int8),
        dynamic_station_active=dynamic_active.astype(np.int8),
        base_pos=base_pos.astype(np.float32),
        base_quat=base_quat.astype(np.float32),
        station=station.astype(np.float32),
        fixed_station=fixed_station.astype(np.float32),
        target_pos=target_pos.astype(np.float32),
        target_vel=target_vel.astype(np.float32),
        target_normal=target_normal.astype(np.float32),
        time_to_strike=time_to_strike.astype(np.float32),
        swing_side=swing_side.astype(np.float32),
        q=q.astype(np.float32),
        dq=dq.astype(np.float32),
        tau=tau.astype(np.float32),
        observation=obs,
        raw_action=raw_action.astype(np.float32),
        applied_action=applied_action.astype(np.float32),
        q_des=q_des.astype(np.float32),
        ball_time_ns=ball_time_ns,
        ball_task_id=ball_task_id,
        ball_command_valid=ball_command_valid,
        ball_pos=ball_pos,
        tcp_pos=tcp_pos,
        tcp_normal=tcp_normal,
    )

    report = {
        "archive": str(args.archive.resolve()),
        "session_id": args.session_id,
        "archive_members": members,
        "policy_frames": int(len(rows)),
        "duration_s": float((system_ns[-1] - system_ns[0]) * 1.0e-9),
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": _sha256(args.onnx),
        "onnx_replay_raw_action_error": _error(raw_action, replay_action),
        "observation_contract": {
            "current_station_error": _error(obs[:, 101:103], current_station_error),
            "fixed_station_error": _error(obs[:, 101:103], fixed_station_error),
            "target_relative_base": _error(obs[:, 103:106], target_rel),
            "previous_applied_action": _error(
                obs[valid_last, 65:96], last_action_expected[valid_last]
            ),
        },
        "command_output_filter": {
            "instantaneous_adapter_target_vs_recorded_smoothed_q_des": _error(
                q_des, expected_q_des
            ),
            "note": (
                "Recorded q_des is the 500 Hz alpha=0.5 smoothed command, while "
                "expected_q_des is the instantaneous 50 Hz adapter target."
            ),
        },
        "policy_timing_s": _stats(policy_dt),
        "lifecycle": {
            "phase_counts": dict(Counter(int(value) for value in phase)),
            "command_valid_frames": int(np.sum(command_valid)),
            "revision_accept_frames": int(np.sum(revision_accept)),
            "dynamic_station_frames": int(np.sum(dynamic_active)),
            "active_task_count": int(len(task_rows)),
        },
        "latency_ms": latency_fields,
        "action_and_tracking": {
            "raw_action_abs": _stats(np.abs(raw_action)),
            "raw_action_abs_active": _stats(np.abs(raw_action[active])),
            "q_des_clamp_fraction_all": float(np.mean(clamp_mask)),
            "q_des_clamp_fraction_active": float(np.mean(clamp_mask[active])),
            "q_tracking_abs_rad": _stats(np.abs(tracking_error)),
            "q_tracking_l2_active": _stats(
                np.linalg.norm(tracking_error[active], axis=1)
            ),
            "highest_swing_follow_clamp_joints": [
                {
                    "joint_name": row["joint_name"],
                    "target_clamp_fraction_swing_follow": row[
                        "target_clamp_fraction_swing_follow"
                    ],
                }
                for row in sorted(
                    joint_rows,
                    key=lambda item: item["target_clamp_fraction_swing_follow"],
                    reverse=True,
                )[:8]
            ],
        },
        "recorded_geometry": {
            "hit_replay_summary": hit_summary,
            "racket_tcp_samples": int(len(tcp_samples)),
            "tcp_contact_candidate_tasks": sorted(
                int(task) for task, count in contact_tasks.items() if task >= 0 and count
            ),
            "ball_face_crossing_tasks": sorted(
                int(task) for task, count in crossing_count.items() if count
            ),
            "high_confidence_impact_tasks": sorted(
                set(task for task in high_confidence_impact_tasks if task >= 0)
            ),
            "high_confidence_impact_count": int(
                sum(row["high_confidence_impact"] for row in impact_rows)
            ),
            "measured_net_clear_after_impact_count": int(
                sum(
                    row["real_net_clear"]
                    for row in impact_rows
                    if row["high_confidence_impact"]
                )
            ),
            "measured_return_success_count": int(
                sum(row["real_return_success"] for row in impact_rows)
            ),
            "planner_aligned_impact_tasks": sorted(
                int(row["task_id"])
                for row in impact_rows
                if row["planner_aligned_impact"]
            ),
            "planner_aligned_impact_count": int(
                sum(row["planner_aligned_impact"] for row in impact_rows)
            ),
            "planner_aligned_net_clear_count": int(
                sum(
                    row["real_net_clear"]
                    for row in impact_rows
                    if row["planner_aligned_impact"]
                )
            ),
            "impact_definition": (
                "geometric contact_candidate plus measured ball vx reversal "
                "from incoming (<-0.2 m/s) to opponent-going (>0.2 m/s), with "
                "at least 1.0 m/s fitted velocity change"
            ),
        },
        "counterfactual_action_sensitivity": sensitivity,
        "outputs": {
            "dataset_npz": str((args.output_dir / "recorded_policy_replay.npz").resolve()),
            "task_summary_csv": str((args.output_dir / "task_summary.csv").resolve()),
            "sensitivity_csv": str(
                (args.output_dir / "counterfactual_action_sensitivity.csv").resolve()
            ),
            "joint_diagnostics_csv": str(
                (args.output_dir / "joint_diagnostics.csv").resolve()
            ),
            "real_contact_kinematics_csv": str(
                (args.output_dir / "real_contact_kinematics.csv").resolve()
            ),
        },
    }
    report_path = args.output_dir / "policy_session_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    markdown = [
        "# Recorded policy session audit",
        "",
        f"- Session: `{args.session_id}`",
        f"- Frames: {len(rows)} ({report['duration_s']:.3f} s)",
        f"- ONNX SHA-256: `{report['onnx_sha256']}`",
        (
            "- ONNX replay max/RMSE: "
            f"{report['onnx_replay_raw_action_error']['max_abs']:.3e} / "
            f"{report['onnx_replay_raw_action_error']['rmse']:.3e}"
        ),
        (
            "- obs[101:103] current-station RMSE: "
            f"{report['observation_contract']['current_station_error']['rmse']:.3e}; "
            "fixed-station RMSE: "
            f"{report['observation_contract']['fixed_station_error']['rmse']:.3e}"
        ),
        (
            "- Active q_des clamp fraction: "
            f"{report['action_and_tracking']['q_des_clamp_fraction_active']:.3%}"
        ),
        f"- Active tasks: {len(task_rows)}",
        (
            "- Geometric contact-candidate tasks: "
            f"{report['recorded_geometry']['tcp_contact_candidate_tasks']}"
        ),
        (
            "- Ball/face crossing tasks: "
            f"{report['recorded_geometry']['ball_face_crossing_tasks']}"
        ),
        "",
        "The crossing/candidate labels are geometric diagnostics, not force-sensor contact truth.",
    ]
    (args.output_dir / "POLICY_SESSION_AUDIT.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
