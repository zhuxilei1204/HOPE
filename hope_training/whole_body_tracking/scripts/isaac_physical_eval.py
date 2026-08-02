"""Continuous physical-ball Isaac evaluation for a trained HOPE policy.

The evaluator keeps one A3 articulation alive across all serves, simulates the
ball/table/net/racket contacts in PhysX, feeds measured ball positions through
the pure-Python HOPEPlanner, and exposes the resulting deployment lifecycle to
the existing 114-D/122-D actor observation.

Unlike ``evaluate.py``, this script does not score the analytic training return.
Contact, net crossing, opponent bounce, and falls are measured from the rigid
ball and robot state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import yaml

try:
    from scripts.planner_snapshot import (
        instantiate_with_supported_kwargs,
        load_planner_api,
    )
except ModuleNotFoundError:
    from planner_snapshot import instantiate_with_supported_kwargs, load_planner_api


_CLOSED_LOOP_V2_PHYSICAL_TASK = (
    "HOPE-PingPong-ClosedLoopV2-PhysicalEval-AgibotA3-v0"
)
_CLOSED_LOOP_V3_SCRATCH_TRAINING_TASK = (
    "HOPE-PingPong-ClosedLoopV3-ScratchMultiSkill-AgibotA3-v0"
)
_CLOSED_LOOP_V3_SCRATCH_PHYSICAL_TASK = (
    "HOPE-PingPong-ClosedLoopV3-ScratchMultiSkill-PhysicalEval-AgibotA3-v0"
)
_CLOSED_LOOP_V2_TERMINATION_TERMS = (
    "time_out",
    "base_too_low",
    "base_tilted",
    "table_touch",
    "persistent_action_overflow",
)
_RECOVERY_PHASE_BOUNDARIES_S = np.asarray(
    (0.10, 0.30, 0.60), dtype=np.float64
)
_RECOVERY_PHASE_NAMES = (
    "impact_0_100ms",
    "brake_100_300ms",
    "settle_300_600ms",
    "ready_after_600ms",
)
_RECOVERY_COMPONENT_NAMES = (
    "tilt",
    "abs_pitch",
    "com_x",
    "com_y",
    "waist_overflow",
    "leg_overflow",
    "base_ang_vel",
)
_RECOVERY_HISTOGRAM_THRESHOLDS = np.asarray(
    (
        (0.10, 0.20, 0.35, 0.50),
        (0.10, 0.20, 0.35, 0.50),
        (0.08, 0.12, 0.16, 0.20),
        (0.10, 0.14, 0.18, 0.22),
        (0.50, 1.00, 1.50, 2.00),
        (1.50, 2.00, 2.50, 3.00),
        (0.80, 1.20, 1.60, 2.00),
    ),
    dtype=np.float64,
)


def _new_physical_recovery_record(contact_time: float) -> dict:
    return {
        "contact_time": float(contact_time),
        "last_elapsed_s": 0.0,
        "phase_seen": np.zeros(4, dtype=np.bool_),
        "histogram_latches": np.zeros((4, 7, 4), dtype=np.bool_),
        "outcome_bucket": 1,
        "ready_success": False,
        "fell": False,
    }


def _observe_physical_recovery(
    record: dict,
    *,
    timestamp: float,
    values: np.ndarray,
) -> None:
    """Accumulate one actual-contact recovery sample in a deterministic phase."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError(
            f"physical recovery values must have shape (7,), got {values.shape}"
        )
    elapsed_s = max(float(timestamp) - float(record["contact_time"]), 0.0)
    phase_index = int(
        np.searchsorted(
            _RECOVERY_PHASE_BOUNDARIES_S,
            elapsed_s + 1.0e-9,
            side="right",
        )
    )
    record["last_elapsed_s"] = elapsed_s
    record["phase_seen"][phase_index] = True
    record["histogram_latches"][phase_index] |= (
        values[:, None] > _RECOVERY_HISTOGRAM_THRESHOLDS
    )


def _summarize_physical_recoveries(records: list[dict]) -> dict:
    phase_attempts = np.zeros(4, dtype=np.int64)
    phase_histograms = np.zeros((4, 7, 4), dtype=np.int64)
    outcome_resolution = np.zeros((4, 3), dtype=np.int64)
    durations = []
    for record in records:
        phase_attempts += record["phase_seen"].astype(np.int64)
        phase_histograms += record["histogram_latches"].astype(np.int64)
        bucket = int(np.clip(record["outcome_bucket"], 0, 3))
        outcome_resolution[bucket, 0] += 1
        outcome_resolution[
            bucket, 1 if bool(record["ready_success"]) else 2
        ] += 1
        durations.append(float(record["last_elapsed_s"]))

    return {
        "trigger": "actual_physx_contact",
        "attempts": len(records),
        "phase_boundaries_s": _RECOVERY_PHASE_BOUNDARIES_S.tolist(),
        "phase_histograms": {
            phase_name: {
                "attempts_reaching_phase": int(
                    phase_attempts[phase_index]
                ),
                "histograms": {
                    component_name: [
                        {
                            "threshold": float(threshold),
                            "count": int(count),
                            "rate": int(count)
                            / max(int(phase_attempts[phase_index]), 1),
                        }
                        for threshold, count in zip(
                            _RECOVERY_HISTOGRAM_THRESHOLDS[
                                component_index
                            ],
                            phase_histograms[
                                phase_index, component_index
                            ],
                        )
                    ]
                    for component_index, component_name in enumerate(
                        _RECOVERY_COMPONENT_NAMES
                    )
                },
            }
            for phase_index, phase_name in enumerate(_RECOVERY_PHASE_NAMES)
        },
        "outcome_resolution": {
            outcome_name: {
                "attempts": int(values[0]),
                "ready_success": int(values[1]),
                "ready_fail": int(values[2]),
                "ready_success_rate": int(values[1])
                / max(int(values[0]), 1),
            }
            for outcome_name, values in zip(
                (
                    "targeted_miss",
                    "contact",
                    "net_cross",
                    "opponent_bounce",
                ),
                outcome_resolution,
            )
        },
        "mean_observed_duration_s": (
            float(np.mean(durations)) if durations else 0.0
        ),
    }


def _validate_physical_task_contract(
    task_id: str, termination_terms: list[str] | tuple[str, ...]
) -> None:
    """Reject a closed-loop-v2 physical eval with leaked termination terms."""
    if task_id not in (
        _CLOSED_LOOP_V2_PHYSICAL_TASK,
        _CLOSED_LOOP_V3_SCRATCH_PHYSICAL_TASK,
    ):
        return
    actual = tuple(termination_terms)
    if actual != _CLOSED_LOOP_V2_TERMINATION_TERMS:
        raise RuntimeError(
            "closed-loop physical termination mismatch: "
            f"expected {_CLOSED_LOOP_V2_TERMINATION_TERMS}, got {actual}"
        )


def _resolve_physical_task_id(explicit_task: str, training_task: str | None) -> str:
    """Select the physical task that preserves the training config inheritance."""
    if explicit_task != "auto":
        return explicit_task
    if training_task == _CLOSED_LOOP_V3_SCRATCH_TRAINING_TASK:
        return _CLOSED_LOOP_V3_SCRATCH_PHYSICAL_TASK
    if training_task and "ClosedLoopV2" in training_task:
        return _CLOSED_LOOP_V2_PHYSICAL_TASK
    return "HOPE-PingPong-PhysicalEval-AgibotA3-v0"


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir() and (parent / "hope_ws").is_dir():
            return parent
    return here.parents[3]


def _default_checkpoint(root: pathlib.Path) -> pathlib.Path:
    return (
        root
        / "hope_training/whole_body_tracking/logs/rsl_rl/hope_pingpong"
        / "2026-07-28_06-15-15_twcyclev3_long_anchor10_old4_10000_from22693_20260728"
        / "model_22750.pt"
    )


def _default_manifest(root: pathlib.Path) -> pathlib.Path:
    return (
        root
        / "hope_training/motions/user_four_motion_manual_hits_a3fk_yaw_leg_stabilized_20260724"
        / "manifest.tsv"
    )


def _default_planner_yaml(root: pathlib.Path) -> pathlib.Path:
    current = root / "hope_ws/src/hope_planner/config/hope_planner.yaml"
    if current.is_file():
        return current
    packaged = (
        root
        / "deploy_artifacts/B17996_deploy_ready_no_command_20260726/config/hope_planner.yaml"
    )
    if packaged.is_file():
        return packaged
    return current


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=str(_default_checkpoint(root)))
    parser.add_argument(
        "--task-yaml",
        default="HOPEPingPongM22494TableWidthCycleV3Old4.yaml",
        help="Training task config used to reconstruct the checkpoint observation contract.",
    )
    parser.add_argument(
        "--physical-task",
        default="auto",
        help=(
            "Registered rigid-ball task. 'auto' maps task-yaml.gym_task to a "
            "physical evaluator with the same training-task inheritance."
        ),
    )
    parser.add_argument("--motion-manifest", default=str(_default_manifest(root)))
    parser.add_argument("--planner-yaml", default=str(_default_planner_yaml(root)))
    parser.add_argument(
        "--planner-code-dir",
        default=None,
        help=(
            "Python package directory for the planner implementation. None uses "
            "hope_ws/src/hope_planner/hope_planner; pass a versioned snapshot "
            "directory to prevent local-code drift."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-serves", type=int, default=8)
    parser.add_argument(
        "--side-mode",
        choices=("mixed", "forehand", "backhand"),
        default="mixed",
        help="Serve-side schedule. mixed preserves the legacy FH,FH,BH,BH sequence.",
    )
    parser.add_argument("--max-trial-seconds", type=float, default=2.6)
    parser.add_argument("--min-rest-seconds", type=float, default=1.0)
    parser.add_argument("--max-rest-seconds", type=float, default=2.2)
    parser.add_argument("--mocap-hz", type=float, default=300.0)
    parser.add_argument("--planner-solve-period", type=float, default=None)
    parser.add_argument("--planner-x-hit", type=float, default=None)
    parser.add_argument(
        "--dynamic-station-clip-x",
        type=float,
        nargs=2,
        default=(-0.25, 0.25),
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--dynamic-station-clip-y",
        type=float,
        nargs=2,
        default=(-0.90, 0.90),
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument("--dynamic-station-post-window", type=float, default=0.10)
    parser.add_argument("--contact-distance", type=float, default=0.14)
    parser.add_argument("--contact-force", type=float, default=0.25)
    parser.add_argument("--video", default=None, help="Output MP4 path.")
    parser.add_argument("--trace-csv", default=None)
    parser.add_argument("--trials-csv", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--experiment-name", default="hope_pingpong")
    parser.add_argument("--motion-file", default=None)
    parser.add_argument("--motion-file-2", default=None)
    return parser.parse_args()


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_manifest(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh, delimiter="\t"):
            ranges = []
            for axis in ("x", "y", "z"):
                lo = _float_or_none(raw.get(f"racket_pos_{axis}_lo"))
                hi = _float_or_none(raw.get(f"racket_pos_{axis}_hi"))
                if lo is None or hi is None:
                    ranges = []
                    break
                ranges.append((min(lo, hi), max(lo, hi)))
            if not ranges:
                continue
            side = _float_or_none(raw.get("swing_side"))
            ox = _float_or_none(raw.get("motion_racket_offset_x"))
            oy = _float_or_none(raw.get("motion_racket_offset_y"))
            rows.append(
                {
                    "side": 1 if side is None or side >= 0 else -1,
                    "ranges": tuple(ranges),
                    "motion_racket_offset_xy": (
                        np.array([ox, oy], dtype=np.float64)
                        if ox is not None and oy is not None
                        else None
                    ),
                    "source": raw.get("output") or raw.get("source") or "",
                }
            )
    if not rows:
        raise ValueError(f"motion manifest has no usable racket boxes: {path}")
    return rows


def _select_manifest_row(rows: list[dict], side: int, target_w=None) -> dict:
    candidates = [row for row in rows if row["side"] == (1 if side >= 0 else -1)]
    if not candidates:
        candidates = rows
    if target_w is None:
        return candidates[0]
    point = np.asarray(target_w, dtype=np.float64)

    def distance_sq(row):
        center = np.array(
            [0.5 * (float(lo) + float(hi)) for lo, hi in row["ranges"]]
        )
        return float(np.sum(np.square(point - center)))

    return min(candidates, key=distance_sq)


def _dynamic_station_for_observation(
    fixed_station_xy,
    target,
    phase,
    manifest_row,
    clip_x,
    clip_y,
    post_window_s,
) -> np.ndarray:
    """Return the planner-derived station using the deployment observation contract."""
    fixed = np.asarray(fixed_station_xy, dtype=np.float64).reshape(2)
    phase_value = getattr(phase, "value", str(phase))
    active = phase_value in ("swing", "follow_through")
    offset = None if manifest_row is None else manifest_row.get("motion_racket_offset_xy")
    if (
        not active
        or offset is None
        or float(target.time_to_strike) < -float(post_window_s)
    ):
        return fixed.copy()

    desired = (
        np.asarray(target.pos_w, dtype=np.float64)[:2]
        - np.asarray(offset, dtype=np.float64).reshape(2)
    )
    rel = desired - fixed
    rel[0] = np.clip(rel[0], *sorted(float(v) for v in clip_x))
    rel[1] = np.clip(rel[1], *sorted(float(v) for v in clip_y))
    return fixed + rel


def _load_table_transform(root: pathlib.Path) -> np.ndarray:
    with (root / "configs/table_frame.yaml").open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    quat = tuple(float(v) for v in doc["simulation"]["table_to_world_quaternion_wxyz"])
    if quat != (1.0, 0.0, 0.0, 0.0):
        raise ValueError("physical evaluator currently requires an axis-aligned table transform")
    return np.asarray(
        doc["simulation"]["table_to_world_translation_xyz"], dtype=np.float64
    )


def _load_planner_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return dict(doc.get("hope_planner", {}).get("ros__parameters", doc) or {})


def _acceleration(physics, velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    return np.asarray(physics.g, dtype=np.float64) - float(physics.k) * np.linalg.norm(
        velocity
    ) * velocity


def _rollout_position(origin, velocity, duration, physics, dt=0.002):
    return _rollout_state(origin, velocity, duration, physics, dt=dt)


def _rollout_state(origin, velocity, duration, physics, dt=0.002):
    """Integrate drag dynamics forward or backward with an RK4 state step."""
    p = np.asarray(origin, dtype=np.float64).copy()
    v = np.asarray(velocity, dtype=np.float64).copy()
    direction = 1.0 if float(duration) >= 0.0 else -1.0
    remaining = abs(float(duration))
    while remaining > 1.0e-12:
        h_abs = min(float(dt), remaining)
        h = direction * h_abs

        k1_p = v
        k1_v = _acceleration(physics, v)
        v2 = v + 0.5 * h * k1_v
        k2_p = v2
        k2_v = _acceleration(physics, v2)
        v3 = v + 0.5 * h * k2_v
        k3_p = v3
        k3_v = _acceleration(physics, v3)
        v4 = v + h * k3_v
        k4_p = v4
        k4_v = _acceleration(physics, v4)

        p += (h / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        v += (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        remaining -= h_abs
    return p, v


def _solve_drag_velocity(origin, target, duration, physics):
    origin = np.asarray(origin, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    accel = np.asarray(physics.g, dtype=np.float64)
    v = (target - origin) / duration - 0.5 * accel * duration
    for _ in range(8):
        p, _ = _rollout_position(origin, v, duration, physics)
        error = p - target
        if float(np.linalg.norm(error)) < 1.0e-4:
            break
        jac = np.zeros((3, 3), dtype=np.float64)
        for axis in range(3):
            eps = 1.0e-3
            dv = np.zeros(3)
            dv[axis] = eps
            p_eps, _ = _rollout_position(origin, v + dv, duration, physics)
            jac[:, axis] = (p_eps - p) / eps
        try:
            v -= np.linalg.solve(jac, error)
        except np.linalg.LinAlgError:
            v -= np.linalg.lstsq(jac, error, rcond=None)[0]
    return v


def _sample_one_bounce_serve(
    rng: np.random.Generator,
    side: int,
    row: dict,
    table_to_world: np.ndarray,
    planner_bridge,
):
    """Sample a drag-consistent one-bounce serve in canonical table coordinates."""
    physics = planner_bridge.simulation_physics
    table = planner_bridge.table
    radius = float(physics.radius)
    x_hit = float(planner_bridge.config.x_hit)

    # Manifest ranges are in the robot-centred Isaac world.  X is fixed by the
    # real planner; Y/Z retain the checkpoint's demonstrated reach distribution.
    y_lo, y_hi = row["ranges"][1]
    z_lo, z_hi = row["ranges"][2]
    strike = np.array(
        [
            x_hit,
            rng.uniform(y_lo - table_to_world[1], y_hi - table_to_world[1]),
            rng.uniform(z_lo - table_to_world[2], z_hi - table_to_world[2]),
        ],
        dtype=np.float64,
    )
    strike[1] = float(np.clip(strike[1], -table.width + 0.04, -0.04))
    side_margin = max(float(planner_bridge.hysteresis_y), 0.03) + 0.01
    if side >= 0:
        strike[1] = min(strike[1], planner_bridge.split_y - side_margin)
    else:
        strike[1] = max(strike[1], planner_bridge.split_y + side_margin)
    strike[1] = float(np.clip(strike[1], -table.width + 0.04, -0.04))
    strike[2] = float(np.clip(strike[2], 0.34, 0.68))

    lo_x = max(0.45, x_hit + 0.35)
    hi_x = min(float(table.net_x) - 0.10, x_hit + 0.72)
    for _ in range(32):
        bounce = np.array(
            [
                rng.uniform(lo_x, hi_x),
                np.clip(
                    strike[1] + rng.uniform(-0.05, 0.05),
                    -table.width + 0.06,
                    -0.06,
                ),
                radius,
            ],
            dtype=np.float64,
        )

        post_t = float(rng.uniform(0.28, 0.40))
        post_vel = _solve_drag_velocity(bounce, strike, post_t, physics)
        pre_vel = post_vel.copy()
        pre_vel[:2] /= max(float(physics.C_h), 1.0e-3)
        pre_vel[2] = -abs(post_vel[2]) / max(float(physics.C_v), 1.0e-3)

        pre_t = float(rng.uniform(0.55, 0.85))
        origin, velocity = _rollout_state(bounce, pre_vel, -pre_t, physics)
        valid = (
            table.net_x + 0.10 <= origin[0] <= table.length + 0.30
            and -table.width + 0.02 <= origin[1] <= -0.02
            and 0.20 <= origin[2] <= 1.10
        )
        if valid:
            return strike, origin, velocity

    raise ValueError(
        "failed to sample a drag-consistent one-bounce route inside the physical "
        f"serve envelope for strike={strike.tolist()}"
    )


@dataclass
class _PlannerStats:
    solve_calls: int = 0
    commands: int = 0
    no_command: int = 0


class _IsaacHopePlannerBridge:
    """Mocap-rate HOPEPlanner bridge with table/world transforms and task IDs."""

    def __init__(
        self,
        root: pathlib.Path,
        planner_yaml: str,
        table_to_world: np.ndarray,
        env_origin: np.ndarray,
        *,
        mocap_hz: float,
        solve_period_override: float | None,
        x_hit_override: float | None,
        planner_code_dir: str | None,
        RacketCommand,
    ):
        planner_api = load_planner_api(root, planner_code_dir)
        self.planner_code_dir = planner_api.package_dir
        self.planner_code_sha256 = planner_api.source_sha256
        PlannerConfig = planner_api.PlannerConfig
        load_ball_physics = planner_api.load_ball_physics
        load_paddle_params = planner_api.load_paddle_params
        load_table_params = planner_api.load_table_params
        self._HOPEPlanner = planner_api.HOPEPlanner
        self._CommandStabilityConfig = planner_api.CommandStabilityConfig
        self._CommandStabilityGate = planner_api.CommandStabilityGate
        self._select_swing_side = planner_api.select_swing_side
        self.RacketCommand = RacketCommand
        self.table_to_world = np.asarray(table_to_world, dtype=np.float64)
        self.env_origin = np.asarray(env_origin, dtype=np.float64)
        self.params = _load_planner_params(planner_yaml)

        physics_path = self.params.get("ball_physics_path") or None
        if physics_path and not os.path.isabs(str(physics_path)):
            physics_path = str(root / str(physics_path))
        self.simulation_physics = load_ball_physics(physics_path)
        self.physics = load_ball_physics(physics_path)
        for key, attr in (("drag_k", "k"), ("table_c_h", "C_h"), ("table_c_v", "C_v")):
            value = float(self.params.get(key, -1.0))
            if value >= 0.0:
                setattr(self.physics, attr, value)
        self.table = load_table_params(
            physics_path, y_max=float(self.params.get("table_y_max", 0.0))
        )
        paddle = load_paddle_params(physics_path)
        x_hit = (
            float(x_hit_override)
            if x_hit_override is not None
            else float(self.params.get("x_hit", 0.2))
        )
        self.config = instantiate_with_supported_kwargs(
            PlannerConfig,
            {
                "x_hit": x_hit,
                "target_land": np.array(
                [
                    float(self.params.get("target_land_x", 2.055)),
                    float(self.params.get("target_land_y", -0.7625)),
                    float(self.physics.radius),
                ],
                dtype=np.float64,
            ),
                "delta_t_flight": float(
                    self.params.get("delta_t_flight", 0.5)
                ),
                "max_predict_time": float(
                    self.params.get("max_predict_time", 2.0)
                ),
                "dt_integrate": float(self.params.get("dt_integrate", 0.001)),
                "fit_window_s": float(self.params.get("fit_window_s", 0.14)),
                "fit_window": int(self.params.get("fit_window", 67)),
                "poly_order_xy": int(self.params.get("poly_order_xy", 1)),
                "poly_order_z": int(self.params.get("poly_order_z", 2)),
                "min_ready_samples": int(
                    self.params.get("min_ready_samples", 6)
                ),
                "bounce_z_tol": float(self.params.get("bounce_z_tol", 0.005)),
                "bounce_center_z_max": float(
                    self.params.get("bounce_center_z_max", 0.11)
                ),
                "bounce_min_vertical_delta": float(
                    self.params.get("bounce_min_vertical_delta", 0.002)
                ),
                "bounce_refractory_s": float(
                    self.params.get("bounce_refractory_s", 0.08)
                ),
                "bounce_max_sample_gap_s": float(
                    self.params.get("bounce_max_sample_gap_s", 0.01)
                ),
                "C_r": float(paddle["C_r"]),
                "paddle_a_t": float(paddle["paddle_a_t"]),
                "paddle_b_t": float(paddle["paddle_b_t"]),
                "paddle_mu": float(paddle["paddle_mu"]),
            },
        )
        self.command_gate = self._make_command_gate()
        self.gate_reason_counts = Counter()
        self.solve_period = (
            float(solve_period_override)
            if solve_period_override is not None
            else float(self.params.get("solve_period_s", 0.02))
        )
        self.mocap_dt = 1.0 / float(mocap_hz) if mocap_hz > 0.0 else None
        self.split_y = float(self.params.get("swing_side_split_y", -0.7625))
        self.hysteresis_y = max(
            0.0, float(self.params.get("swing_side_hysteresis_y", 0.0))
        )
        self.stats = _PlannerStats()
        self._task_id = 0
        self._prev_side = 0
        self.reset_for_new_ball()

    def _make_command_gate(self):
        if (
            self._CommandStabilityConfig is None
            or self._CommandStabilityGate is None
        ):
            return None
        config = instantiate_with_supported_kwargs(
            self._CommandStabilityConfig,
            {
                "initial_consecutive": int(
                    self.params.get("revision_gate_initial_consecutive", 2)
                ),
                "initial_position_tolerance_m": float(
                    self.params.get(
                        "revision_gate_initial_position_tolerance_m", 0.10
                    )
                ),
                "max_position_jump_m": float(
                    self.params.get("revision_gate_max_position_jump_m", 0.10)
                ),
                "max_velocity_jump_mps": float(
                    self.params.get("revision_gate_max_velocity_jump_mps", 1.50)
                ),
                "freeze_time_to_strike_s": float(
                    self.params.get("revision_gate_freeze_tts_s", 0.20)
                ),
                "max_strike_time_jump_s": float(
                    self.params.get(
                        "revision_gate_max_strike_time_jump_s", 0.040
                    )
                ),
            },
        )
        return self._CommandStabilityGate(config)

    def to_table(self, p_world) -> np.ndarray:
        return (
            np.asarray(p_world, dtype=np.float64)
            - self.env_origin
            - self.table_to_world
        )

    def to_world(self, p_table) -> np.ndarray:
        return (
            np.asarray(p_table, dtype=np.float64)
            + self.table_to_world
            + self.env_origin
        )

    def reset_for_new_ball(self):
        self.planner = self._HOPEPlanner(
            physics=self.physics, config=self.config, table=self.table
        )
        self._task_revision = 0
        self._task_active = False
        self._candidate_active = False
        self._locked_side = 1
        self._last_solve_t = None
        self._last_control_t = None
        self._last_control_pos = None
        self._next_mocap_t = None
        if self.command_gate is not None:
            self.command_gate.reset()

    def _push_sample(self, t: float, p_table: np.ndarray):
        if (
            self.solve_period > 0.0
            and self._last_solve_t is not None
            and 0.0 <= t - self._last_solve_t < self.solve_period
        ):
            self.planner.estimator.push(t, p_table)
            return None
        self._last_solve_t = t
        self.stats.solve_calls += 1
        try:
            command = self.planner.update(t, p_table)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            self.stats.no_command += 1
            return None
        if command is None:
            if self.planner.ball_incoming is False:
                self._task_active = False
                self._candidate_active = False
                if self.command_gate is not None:
                    self.command_gate.reset()
            self.stats.no_command += 1
            return None

        tts = self.planner.time_to_strike
        if tts is None:
            tts = float(command.t_strike) - float(t)
        if not self._candidate_active:
            self._candidate_active = True
            if self.command_gate is not None:
                self.command_gate.reset()
        if self.command_gate is not None:
            accepted = self.command_gate.consider(
                command.p_intercept,
                command.v_racket,
                float(tts),
                candidate_time_s=float(t),
            )
            self.gate_reason_counts[self.command_gate.last_reason] += 1
            if not accepted:
                return None

        if not self._task_active:
            self._task_id += 1
            self._task_revision = 0
            side = self._select_swing_side(
                float(command.p_intercept[1]),
                self.split_y,
                self.hysteresis_y,
                self._prev_side,
            )
            self._locked_side = 1 if side >= 0 else -1
            self._prev_side = self._locked_side
            self._task_active = True
        else:
            self._task_revision += 1

        normal = np.asarray(command.n_racket, dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-9)
        out = self.RacketCommand(
            task_id=self._task_id,
            task_revision=self._task_revision,
            swing_side=self._locked_side,
            position=self.to_world(command.p_intercept),
            velocity=np.asarray(command.v_racket, dtype=np.float64),
            time_to_strike=max(0.0, float(tts)),
            target_normal=normal,
        )
        out.planner_num_bounces = int(getattr(command, "num_bounces", 0))
        out.intercept_table = np.asarray(command.p_intercept, dtype=np.float64).copy()
        out.planner_net_margin = float(getattr(command, "net_margin", float("nan")))
        try:
            flight_time = float(
                getattr(command, "flight_time", self.config.delta_t_flight)
            )
            out.outgoing_velocity = (
                self.planner.target_planner._compute_outgoing_velocity(
                    np.asarray(command.p_intercept, dtype=np.float64),
                    np.asarray(self.config.target_land, dtype=np.float64),
                    flight_time,
                )
            )
        except Exception:
            out.outgoing_velocity = None
        self.stats.commands += 1
        return out

    def update(self, p_world: np.ndarray, t: float):
        p_table = self.to_table(p_world)
        t = float(t)
        if self._last_control_t is None or t <= self._last_control_t:
            self._last_control_t = t
            self._last_control_pos = p_table.copy()
            self._next_mocap_t = None if self.mocap_dt is None else t + self.mocap_dt
            return self._push_sample(t, p_table)

        latest = None
        if self.mocap_dt is None:
            latest = self._push_sample(t, p_table)
        else:
            sample_t = (
                self._next_mocap_t
                if self._next_mocap_t is not None
                else self._last_control_t
            )
            while sample_t <= t + 1.0e-9:
                alpha = (sample_t - self._last_control_t) / max(
                    t - self._last_control_t, 1.0e-9
                )
                interp = (1.0 - alpha) * self._last_control_pos + alpha * p_table
                command = self._push_sample(float(sample_t), interp)
                if command is not None:
                    latest = command
                sample_t += self.mocap_dt
            self._next_mocap_t = sample_t
        self._last_control_t = t
        self._last_control_pos = p_table.copy()
        return latest


def _angle_deg(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1.0e-9:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))


def _quat_roll_pitch_deg(quat_wxyz) -> tuple[float, float]:
    """Return world-frame roll and pitch for a wxyz quaternion."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return (
        math.degrees(math.atan2(sin_roll, cos_roll)),
        math.degrees(math.asin(sin_pitch)),
    )


def _open_csv(path: str | None, fieldnames: list[str]):
    if not path:
        return None, None
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fh = output.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer


def main() -> int:
    args = parse_args()
    root = _repo_root()
    package_source = (
        root
        / "hope_training/whole_body_tracking/source/whole_body_tracking"
    )
    if str(package_source) not in sys.path:
        sys.path.insert(0, str(package_source))
    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    manifest = _load_manifest(args.motion_manifest)
    table_to_world = _load_table_transform(root)

    # Isaac Lab parses its own CLI from sys.argv.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(
        headless=True, device=args.device, enable_cameras=bool(args.video)
    )
    simulation_app = launcher.app

    status = 0
    env = None
    trace_fh = None
    trials_fh = None
    try:
        import gymnasium as gym
        import torch

        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import whole_body_tracking.tasks  # noqa: F401
        from whole_body_tracking.robots.agibot_a3 import A3_RACKET_BODY, A3_WRIST_BODY
        from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
        from whole_body_tracking.utils.ppo_cfg import load_ppo_params, runner_kwargs
        from evaluate import (
            _apply_training_task_overrides,
            _load_task_yaml_with_defaults,
            _resolve_task_yaml,
        )
        from train import _apply_motion_metadata, _resolve_motion_plan

        ref_dir = root / "a3_deploy/a3_deploy_example/reference"
        if str(ref_dir) not in sys.path:
            sys.path.insert(0, str(ref_dir))
        from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
        from a3_deploy_onnx_ref_pingpong.lifecycle import Phase, SwingLifecycle
        from a3_deploy_onnx_ref_pingpong.observation import RobotState
        from a3_deploy_onnx_ref_pingpong.racket_command import RacketCommand

        task_cfg = _load_task_yaml_with_defaults(_resolve_task_yaml(args.task_yaml))
        training_task = task_cfg.get("gym_task")
        task_id = _resolve_physical_task_id(
            str(args.physical_task),
            None if training_task is None else str(training_task),
        )
        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        applied = _apply_training_task_overrides(env_cfg, args.task_yaml)
        motion_args = argparse.Namespace(**vars(args))
        motion_args.task = task_cfg
        # ``--motion-manifest`` defines the deterministic physical serve set.
        # Reconstruct the hidden reference command from the training task
        # instead of accidentally replacing it with that serve manifest.
        motion_args.motion_manifest = None
        motion_args.motion_files = []
        motion_args.motion_file = None
        motion_args.motion_file_2 = None
        clips, metadata = _resolve_motion_plan(motion_args)
        env_cfg.commands.motion.motion_file = clips if len(clips) > 1 else clips[0]
        _apply_motion_metadata(env_cfg, clips, metadata, applied)

        # Deterministic deploy-style entry.  The hidden motion reference remains
        # frozen at default ready; only the physical planner lifecycle drives the
        # actor-visible strike command.
        env_cfg.seed = int(args.seed)
        env_cfg.episode_length_s = max(
            300.0,
            args.num_serves * (args.max_trial_seconds + args.max_rest_seconds) + 30.0,
        )
        # Force the same deploy-entry reset contract used by the strict
        # no-command diagnostic.  stand_start_prob alone is insufficient when
        # a scratch task enables motion-start warmup, because that branch is
        # evaluated first and can replace the default stand with a motion frame.
        env_cfg.commands.motion.stand_episode_prob = 1.0
        env_cfg.commands.motion.stand_episode_hold_steps = 1_000_000
        env_cfg.commands.motion.stand_start_prob = 1.0
        env_cfg.commands.motion.stand_start_min_hold = 1_000_000
        env_cfg.commands.motion.hold_steps_range = (1_000_000, 1_000_000)
        env_cfg.commands.motion.motion_start_warmup_enabled = False
        env_cfg.commands.motion.wrap_teleport = False
        env_cfg.commands.motion.pose_range = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        env_cfg.commands.motion.velocity_range = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
        env_cfg.observations.policy.enable_corruption = False
        if not args.video:
            env_cfg.scene.table_visual = None
        for event_name in (
            "physics_material",
            "base_com",
            "link_mass",
            "joint_default_pos",
            "pd_gains",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

        print(
            f"[isaac_physical_eval] checkpoint={checkpoint}\n"
            f"[isaac_physical_eval] task_yaml={args.task_yaml} overrides={len(applied)}\n"
            f"[isaac_physical_eval] planner_yaml={args.planner_yaml}",
            file=sys.stderr,
            flush=True,
        )

        render_mode = "rgb_array" if args.video else None
        raw_env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)
        base_env = raw_env.unwrapped
        termination_terms = list(base_env.termination_manager.active_terms)
        _validate_physical_task_contract(task_id, termination_terms)
        motion_command = base_env.command_manager.get_term("motion")
        if args.video:
            video_path = pathlib.Path(args.video).expanduser().resolve()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            max_video_steps = int(
                math.ceil(
                    args.num_serves
                    * (args.max_trial_seconds + args.max_rest_seconds)
                    / base_env.step_dt
                )
            ) + 200
            raw_env = gym.wrappers.RecordVideo(
                raw_env,
                video_folder=str(video_path.parent),
                step_trigger=lambda step: step == 0,
                video_length=max_video_steps,
                disable_logger=True,
                name_prefix=video_path.stem,
            )
        env = RslRlVecEnvWrapper(raw_env)

        agent_cfg = RslRlOnPolicyRunnerCfg(
            **runner_kwargs(load_ppo_params(), args.experiment_name)
        )
        agent_cfg.device = args.device
        runner = HOPEOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=args.device
        )
        # Physical evaluation consumes only the actor.  Actor-only loading maps
        # tensors onto the requested device and keeps checkpoints produced on a
        # different CUDA index portable.
        runner.load_actor_only(str(checkpoint))
        policy = runner.get_inference_policy(device=base_env.device)

        runtime = RuntimeConfig.load(
            root / "a3_deploy/a3_deploy_example/config/hope_pingpong_runtime.yaml"
        )
        lifecycle = SwingLifecycle(runtime.lifecycle)
        ball = base_env.scene["ball"]
        robot = base_env.scene["robot"]
        command = base_env.command_manager.get_term("racket_target")
        torso_index = robot.body_names.index("torso_Link")
        waist_pitch_index = (
            robot.joint_names.index("waist_pitch_joint")
            if "waist_pitch_joint" in robot.joint_names
            else None
        )
        env_origin = (
            base_env.scene.env_origins[0].detach().cpu().numpy().astype(np.float64)
        )
        planner = _IsaacHopePlannerBridge(
            root,
            args.planner_yaml,
            table_to_world,
            env_origin,
            mocap_hz=args.mocap_hz,
            solve_period_override=args.planner_solve_period,
            x_hit_override=args.planner_x_hit,
            planner_code_dir=args.planner_code_dir,
            RacketCommand=RacketCommand,
        )
        rng = np.random.default_rng(args.seed)

        sensor = base_env.scene.sensors["contact_forces"]
        racket_sensor_index = None
        for name in (A3_RACKET_BODY, A3_WRIST_BODY):
            if name in sensor.body_names:
                racket_sensor_index = sensor.body_names.index(name)
                break
        if racket_sensor_index is None:
            print(
                "[isaac_physical_eval] WARNING: racket link absent from contact sensor; "
                "using physical velocity-jump + proximity contact detection.",
                file=sys.stderr,
            )

        fixed_station = env_origin[:2].copy()
        latest_command = None
        latest_manifest_row = None

        def set_ball(table_pos, table_vel):
            pos_w = planner.to_world(table_pos)
            pose = torch.tensor(
                [[pos_w[0], pos_w[1], pos_w[2], 1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device=base_env.device,
            )
            velocity = torch.tensor(
                [[table_vel[0], table_vel[1], table_vel[2], 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device=base_env.device,
            )
            ids = torch.tensor([0], dtype=torch.long, device=base_env.device)
            ball.write_root_pose_to_sim(pose, env_ids=ids)
            ball.write_root_velocity_to_sim(velocity, env_ids=ids)

        park_pos = np.array([3.35, 0.45, 1.35], dtype=np.float64)

        def robot_state():
            return RobotState(
                base_pos_w=robot.data.root_pos_w[0].detach().cpu().numpy(),
                base_quat_w=robot.data.root_quat_w[0].detach().cpu().numpy(),
                base_ang_vel_b=robot.data.root_ang_vel_b[0].detach().cpu().numpy(),
                q=np.zeros(31),
                qd=np.zeros(31),
            )

        def apply_actor_target(target, phase):
            # Command tensors are advanced by Isaac under inference mode during
            # env.step(), so subsequent in-place external planner updates must
            # use the same mode.
            with torch.inference_mode():
                pos = torch.as_tensor(
                    target.pos_w, dtype=torch.float32, device=base_env.device
                )
                vel = torch.as_tensor(
                    target.vel_w, dtype=torch.float32, device=base_env.device
                )
                normal_np = (
                    np.asarray(target.normal_w, dtype=np.float64)
                    if target.normal_w is not None
                    else np.asarray(target.vel_w, dtype=np.float64)
                )
                normal_np /= max(float(np.linalg.norm(normal_np)), 1.0e-9)
                normal = torch.as_tensor(
                    normal_np, dtype=torch.float32, device=base_env.device
                )
                station = _dynamic_station_for_observation(
                    fixed_station,
                    target,
                    phase,
                    latest_manifest_row,
                    args.dynamic_station_clip_x,
                    args.dynamic_station_clip_y,
                    args.dynamic_station_post_window,
                )

                command.racket_target_pos_w[0] = pos
                command.racket_target_vel_w[0] = vel
                command.racket_impact_target_vel_w[0] = vel
                command.racket_target_normal_w[0] = normal
                command.time_to_strike[0] = float(target.time_to_strike)
                command.true_time_to_strike[0] = float(target.time_to_strike)
                command.swing_sign[0] = float(target.swing_side)
                command.dynamic_station_w[0] = torch.as_tensor(
                    station, dtype=torch.float32, device=base_env.device
                )
                is_swing = phase in (Phase.SWING, Phase.FOLLOW_THROUGH)
                command.pre_strike[0] = bool(is_swing and target.time_to_strike > 0.10)
                command.strike_window[0] = bool(
                    is_swing and abs(float(target.time_to_strike)) <= 0.10
                )
                command.no_command_ready_active[0] = phase in (
                    Phase.READY,
                    Phase.RECOVERY,
                )
                if hasattr(command, "station_relocation_rehearsal"):
                    command.station_relocation_rehearsal[0] = False

        trace_fields = [
            "trial",
            "tick",
            "session_time",
            "phase",
            "planner_command",
            "planner_tts",
            "planner_side",
            "planner_bounces",
            "ball_x",
            "ball_y",
            "ball_z",
            "ball_world_x",
            "ball_world_y",
            "ball_world_z",
            "ball_vx",
            "ball_vy",
            "ball_vz",
            "intended_strike_table_x",
            "intended_strike_table_y",
            "intended_strike_table_z",
            "racket_x",
            "racket_y",
            "racket_z",
            "racket_vx",
            "racket_vy",
            "racket_vz",
            "racket_normal_x",
            "racket_normal_y",
            "racket_normal_z",
            "target_x",
            "target_y",
            "target_z",
            "target_vx",
            "target_vy",
            "target_vz",
            "target_normal_x",
            "target_normal_y",
            "target_normal_z",
            "ball_racket_distance",
            "racket_contact_force",
            "incoming_bounce",
            "contact",
            "net_cross",
            "base_x",
            "base_y",
            "base_z",
            "base_pitch_deg",
            "base_roll_deg",
            "torso_pitch_deg",
            "waist_pitch_rad",
            "base_vx",
            "base_vy",
            "station_error_x",
            "station_error_y",
            "base_lin_speed",
            "base_ang_speed",
            "feet_contact_frac",
            "done",
            "termination_reason",
        ]
        trial_fields = [
            "trial",
            "requested_side",
            "planner_side",
            "planner_command_seen",
            "planner_num_bounces",
            "incoming_bounce",
            "contact",
            "net_cross",
            "opponent_bounce",
            "success",
            "fell",
            "termination_reason",
            "timed_out",
            "min_ball_racket_distance",
            "intended_strike_table_x",
            "intended_strike_table_y",
            "intended_strike_table_z",
            "contact_force_peak",
            "contact_target_pos_error",
            "contact_target_vel_error",
            "contact_target_normal_error_deg",
            "contact_ball_target_error",
            "contact_pre_ball_vx",
            "contact_pre_ball_vy",
            "contact_pre_ball_vz",
            "contact_racket_speed",
            "contact_target_racket_speed",
            "contact_racket_normal_x",
            "contact_racket_normal_y",
            "contact_racket_normal_z",
            "contact_target_normal_x",
            "contact_target_normal_y",
            "contact_target_normal_z",
            "contact_ball_out_speed",
            "contact_desired_out_speed",
            "contact_desired_out_vx",
            "contact_desired_out_vy",
            "contact_desired_out_vz",
            "contact_ball_out_error",
            "contact_ball_out_angle_deg",
            "contact_planner_net_margin",
            "contact_base_pitch_deg",
            "contact_torso_pitch_deg",
            "contact_waist_pitch_rad",
            "contact_base_vx",
            "contact_station_error_x",
            "contact_station_error_y",
            "contact_base_ang_speed",
            "final_base_pitch_deg",
            "final_base_ang_speed",
            "rest_max_abs_base_pitch_deg",
            "rest_max_abs_base_roll_deg",
            "rest_max_base_ang_speed",
            "rest_end_base_pitch_deg",
            "rest_end_base_roll_deg",
            "rest_end_base_ang_speed",
            "rest_reached_ready",
            "rest_healthy_ready",
        ]
        trace_fh, trace_writer = _open_csv(args.trace_csv, trace_fields)
        trials_fh, trials_writer = _open_csv(args.trials_csv, trial_fields)

        obs, _ = env.get_observations()
        ready_target = lifecycle.update(None, robot_state())
        apply_actor_target(ready_target, lifecycle.phase)
        obs, _ = env.get_observations()
        step_dt = float(base_env.step_dt)
        min_rest_ticks = max(0, int(round(args.min_rest_seconds / step_dt)))
        max_rest_ticks = max(min_rest_ticks, int(round(args.max_rest_seconds / step_dt)))
        max_trial_ticks = max(1, int(round(args.max_trial_seconds / step_dt)))

        counts = {
            "attempts": 0,
            "planner_command_seen": 0,
            "incoming_bounce": 0,
            "contact": 0,
            "net_cross": 0,
            "opponent_bounce": 0,
            "success": 0,
            "fall": 0,
            "healthy_ready": 0,
        }
        side_counts = {
            "forehand": {"attempts": 0, "contact": 0, "success": 0},
            "backhand": {"attempts": 0, "contact": 0, "success": 0},
        }
        physical_recovery_records: list[dict] = []
        session_time = 0.0
        session_done = False
        termination_counts = {name: 0 for name in termination_terms}
        last_done_terms: list[str] = []
        deploy_entry_contract_checked = False

        def step_policy(target):
            nonlocal obs, session_time, last_done_terms, deploy_entry_contract_checked
            apply_actor_target(target, lifecycle.phase)
            obs, _ = env.get_observations()
            with torch.inference_mode():
                action = policy(obs)
                obs, _, dones, _ = env.step(action)
            if not deploy_entry_contract_checked:
                if not bool(torch.all(motion_command.stand_episode)) or not bool(
                    torch.all(motion_command.default_stand_reset)
                ):
                    raise RuntimeError(
                        "physical eval deploy-entry contract violated after first step: "
                        "expected a default-stand episode with motion-start warmup disabled"
                    )
                deploy_entry_contract_checked = True
            session_time += step_dt
            done = bool(dones.reshape(-1)[0].item())
            last_done_terms = []
            if done:
                for name in termination_terms:
                    term = base_env.termination_manager.get_term(name)
                    if bool(term.reshape(-1)[0].item()):
                        last_done_terms.append(name)
                        termination_counts[name] += 1
            return done

        def physical_recovery_values() -> np.ndarray:
            gravity = robot.data.projected_gravity_b[0]
            return np.asarray(
                (
                    float(torch.norm(gravity[:2]).item()),
                    abs(float(gravity[0].item())),
                    float(command.metrics["impact_health_com_x"][0].item()),
                    float(command.metrics["impact_health_com_y"][0].item()),
                    float(
                        command.metrics["waist_action_overflow_rms"][0].item()
                    ),
                    float(
                        command.metrics["leg_action_overflow_rms"][0].item()
                    ),
                    float(torch.norm(robot.data.root_ang_vel_w[0]).item()),
                ),
                dtype=np.float64,
            )

        def finish_physical_recovery(
            record: dict | None,
            *,
            net_crossed: bool,
            bounced_on_opponent: bool,
            ready_success: bool,
            fell: bool,
        ) -> None:
            if record is None:
                return
            record["outcome_bucket"] = (
                3 if bounced_on_opponent else 2 if net_crossed else 1
            )
            record["ready_success"] = bool(ready_success)
            record["fell"] = bool(fell)
            physical_recovery_records.append(record)

        # Settle from the deterministic default pose before the first physical ball.
        for _ in range(max(25, min_rest_ticks)):
            set_ball(park_pos, np.zeros(3))
            target = lifecycle.update(None, robot_state())
            if step_policy(target):
                counts["fall"] += 1
                session_done = True
                break

        for trial in range(args.num_serves):
            if session_done:
                break
            if args.side_mode == "forehand":
                requested_side = 1
            elif args.side_mode == "backhand":
                requested_side = -1
            else:
                requested_side = 1 if (trial % 4) < 2 else -1
            row = _select_manifest_row(manifest, requested_side)
            strike_table, serve_table, serve_velocity = _sample_one_bounce_serve(
                rng, requested_side, row, table_to_world, planner
            )
            planner.reset_for_new_ball()
            latest_command = None
            latest_manifest_row = row
            set_ball(serve_table, serve_velocity)

            counts["attempts"] += 1
            requested_name = "forehand" if requested_side >= 0 else "backhand"
            incoming_bounce = False
            contacted = False
            net_cross = False
            opponent_bounce = False
            command_seen = False
            planner_side = 0
            planner_num_bounces = -1
            min_distance = float("inf")
            force_peak = 0.0
            timed_out = True
            trial_done = False
            physical_recovery_record = None
            contact_diag = {
                "contact_target_pos_error": float("nan"),
                "contact_target_vel_error": float("nan"),
                "contact_target_normal_error_deg": float("nan"),
                "contact_ball_target_error": float("nan"),
                "contact_pre_ball_vx": float("nan"),
                "contact_pre_ball_vy": float("nan"),
                "contact_pre_ball_vz": float("nan"),
                "contact_racket_speed": float("nan"),
                "contact_target_racket_speed": float("nan"),
                "contact_racket_normal_x": float("nan"),
                "contact_racket_normal_y": float("nan"),
                "contact_racket_normal_z": float("nan"),
                "contact_target_normal_x": float("nan"),
                "contact_target_normal_y": float("nan"),
                "contact_target_normal_z": float("nan"),
                "contact_ball_out_speed": float("nan"),
                "contact_desired_out_speed": float("nan"),
                "contact_desired_out_vx": float("nan"),
                "contact_desired_out_vy": float("nan"),
                "contact_desired_out_vz": float("nan"),
                "contact_ball_out_error": float("nan"),
                "contact_ball_out_angle_deg": float("nan"),
                "contact_planner_net_margin": float("nan"),
                "contact_base_pitch_deg": float("nan"),
                "contact_torso_pitch_deg": float("nan"),
                "contact_waist_pitch_rad": float("nan"),
                "contact_base_vx": float("nan"),
                "contact_station_error_x": float("nan"),
                "contact_station_error_y": float("nan"),
                "contact_base_ang_speed": float("nan"),
            }

            previous_ball = (
                ball.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            )
            previous_velocity = (
                ball.data.root_lin_vel_w[0].detach().cpu().numpy().astype(np.float64)
            )

            for tick in range(max_trial_ticks):
                ball_pos_w = (
                    ball.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
                )
                planner_update = planner.update(ball_pos_w, session_time)
                if planner_update is not None:
                    latest_command = planner_update
                    command_seen = True
                    planner_side = int(planner_update.swing_side)
                    planner_num_bounces = max(
                        planner_num_bounces,
                        int(getattr(planner_update, "planner_num_bounces", -1)),
                    )
                    latest_manifest_row = _select_manifest_row(
                        manifest, planner_side, planner_update.position
                    )

                target = lifecycle.update(latest_command, robot_state())
                done = step_policy(target)

                ball_pos_w = (
                    ball.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
                )
                ball_vel_w = (
                    ball.data.root_lin_vel_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                ball_table = planner.to_table(ball_pos_w)
                previous_table = planner.to_table(previous_ball)
                racket_pos = (
                    command.racket_pos_w[0].detach().cpu().numpy().astype(np.float64)
                )
                racket_vel = (
                    command.racket_lin_vel_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                racket_normal = (
                    command.racket_normal_w[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                target_normal = (
                    np.asarray(target.normal_w, dtype=np.float64)
                    if target.normal_w is not None
                    else np.asarray(target.vel_w, dtype=np.float64)
                )
                target_normal /= max(float(np.linalg.norm(target_normal)), 1.0e-9)
                # The frozen internal motion still owns the URDF blade-face
                # sign, while the external lifecycle owns the requested side.
                internal_sign = 1.0
                motion_term = command._motion()
                sign_table = getattr(command, "_mount_sign_per_clip", None)
                if sign_table is not None and motion_term._multiseg:
                    internal_sign = float(
                        sign_table[motion_term.clip_id[0]].item()
                    )
                external_sign = (
                    1.0 if float(target.swing_side) >= 0.0 else -1.0
                )
                racket_normal_for_side = (
                    racket_normal * external_sign / internal_sign
                )
                distance = float(np.linalg.norm(ball_pos_w - racket_pos))
                min_distance = min(min_distance, distance)

                contact_force = 0.0
                if racket_sensor_index is not None:
                    history = getattr(sensor.data, "net_forces_w_history", None)
                    if history is not None:
                        contact_force = float(
                            torch.norm(history[0, :, racket_sensor_index, :], dim=-1)
                            .max()
                            .item()
                        )
                    else:
                        contact_force = float(
                            torch.norm(
                                sensor.data.net_forces_w[0, racket_sensor_index]
                            ).item()
                        )
                force_peak = max(force_peak, contact_force)

                near_surface = (
                    ball_table[2] <= float(planner.physics.radius) + 0.065
                )
                bounce_now = bool(
                    near_surface
                    and previous_velocity[2] < -0.10
                    and ball_vel_w[2] > 0.10
                    and 0.0 <= ball_table[0] <= planner.table.length
                    and -planner.table.width <= ball_table[1] <= 0.0
                )
                if not contacted and bounce_now and ball_table[0] <= planner.table.net_x:
                    incoming_bounce = True

                velocity_jump = float(np.linalg.norm(ball_vel_w - previous_velocity))
                physical_contact = bool(
                    not contacted
                    and distance <= args.contact_distance
                    and (
                        contact_force >= args.contact_force
                        or (
                            velocity_jump >= 0.65
                            and previous_velocity[0] < -0.1
                            and ball_vel_w[0] > previous_velocity[0] + 0.25
                        )
                    )
                )
                if physical_contact:
                    contacted = True
                    physical_recovery_record = _new_physical_recovery_record(
                        session_time
                    )
                    # Use the exact lifecycle target that produced this action.
                    # The internal training command manager advances after
                    # physics and may already hold its next analytic target.
                    target_pos = np.asarray(target.pos_w, dtype=np.float64)
                    target_vel = np.asarray(target.vel_w, dtype=np.float64)
                    desired_out = (
                        None
                        if latest_command is None
                        else getattr(latest_command, "outgoing_velocity", None)
                    )
                    gravity = robot.data.projected_gravity_b[0]
                    pitch = float(
                        torch.rad2deg(
                            torch.asin(torch.clamp(gravity[0], -1.0, 1.0))
                        ).item()
                    )
                    _, torso_pitch = _quat_roll_pitch_deg(
                        robot.data.body_quat_w[0, torso_index]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    base_xy_w = (
                        robot.data.root_pos_w[0, :2]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    station_xy_w = (
                        command.dynamic_station_w[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    station_error = base_xy_w - station_xy_w
                    contact_diag = {
                        "contact_target_pos_error": float(
                            np.linalg.norm(racket_pos - target_pos)
                        ),
                        "contact_target_vel_error": float(
                            np.linalg.norm(racket_vel - target_vel)
                        ),
                        "contact_target_normal_error_deg": _angle_deg(
                            racket_normal_for_side, target_normal
                        ),
                        "contact_ball_target_error": float(
                            np.linalg.norm(ball_pos_w - target_pos)
                        ),
                        "contact_pre_ball_vx": float(previous_velocity[0]),
                        "contact_pre_ball_vy": float(previous_velocity[1]),
                        "contact_pre_ball_vz": float(previous_velocity[2]),
                        "contact_racket_speed": float(np.linalg.norm(racket_vel)),
                        "contact_target_racket_speed": float(
                            np.linalg.norm(target_vel)
                        ),
                        "contact_racket_normal_x": float(
                            racket_normal_for_side[0]
                        ),
                        "contact_racket_normal_y": float(
                            racket_normal_for_side[1]
                        ),
                        "contact_racket_normal_z": float(
                            racket_normal_for_side[2]
                        ),
                        "contact_target_normal_x": float(target_normal[0]),
                        "contact_target_normal_y": float(target_normal[1]),
                        "contact_target_normal_z": float(target_normal[2]),
                        "contact_ball_out_speed": float(np.linalg.norm(ball_vel_w)),
                        "contact_desired_out_speed": (
                            float("nan")
                            if desired_out is None
                            else float(np.linalg.norm(desired_out))
                        ),
                        "contact_desired_out_vx": (
                            float("nan")
                            if desired_out is None
                            else float(desired_out[0])
                        ),
                        "contact_desired_out_vy": (
                            float("nan")
                            if desired_out is None
                            else float(desired_out[1])
                        ),
                        "contact_desired_out_vz": (
                            float("nan")
                            if desired_out is None
                            else float(desired_out[2])
                        ),
                        "contact_ball_out_error": (
                            float("nan")
                            if desired_out is None
                            else float(np.linalg.norm(ball_vel_w - desired_out))
                        ),
                        "contact_ball_out_angle_deg": (
                            float("nan")
                            if desired_out is None
                            else _angle_deg(ball_vel_w, desired_out)
                        ),
                        "contact_planner_net_margin": (
                            float("nan")
                            if latest_command is None
                            else float(
                                getattr(
                                    latest_command,
                                    "planner_net_margin",
                                    float("nan"),
                                )
                            )
                        ),
                        "contact_base_pitch_deg": pitch,
                        "contact_torso_pitch_deg": torso_pitch,
                        "contact_waist_pitch_rad": (
                            float("nan")
                            if waist_pitch_index is None
                            else float(
                                robot.data.joint_pos[0, waist_pitch_index].item()
                            )
                        ),
                        "contact_base_vx": float(
                            robot.data.root_lin_vel_w[0, 0].item()
                        ),
                        "contact_station_error_x": float(station_error[0]),
                        "contact_station_error_y": float(station_error[1]),
                        "contact_base_ang_speed": float(
                            torch.norm(robot.data.root_ang_vel_w[0]).item()
                        ),
                    }

                net_x = float(planner.table_to_world[0] + planner.table.net_x)
                if (
                    contacted
                    and not net_cross
                    and previous_ball[0] < net_x <= ball_pos_w[0]
                ):
                    frac = (net_x - previous_ball[0]) / max(
                        ball_pos_w[0] - previous_ball[0], 1.0e-9
                    )
                    z_cross = previous_ball[2] + frac * (
                        ball_pos_w[2] - previous_ball[2]
                    )
                    net_top = (
                        planner.table_to_world[2]
                        + planner.table.net_height
                        + planner.physics.radius
                    )
                    net_cross = bool(z_cross > net_top)

                if contacted and bounce_now:
                    opponent_bounce = bool(
                        net_cross
                        and planner.table.net_x < ball_table[0] < planner.table.length
                        and -planner.table.width < ball_table[1] < 0.0
                    )
                    timed_out = False
                    trial_done = True

                gravity = robot.data.projected_gravity_b[0]
                base_pitch = float(
                    torch.rad2deg(
                        torch.asin(torch.clamp(gravity[0], -1.0, 1.0))
                    ).item()
                )
                base_roll = float(
                    torch.rad2deg(
                        torch.asin(torch.clamp(gravity[1], -1.0, 1.0))
                    ).item()
                )
                base_pos = (
                    robot.data.root_pos_w[0] - base_env.scene.env_origins[0]
                )
                _, torso_pitch = _quat_roll_pitch_deg(
                    robot.data.body_quat_w[0, torso_index]
                    .detach()
                    .cpu()
                    .numpy()
                )
                station_error = (
                    robot.data.root_pos_w[0, :2] - command.dynamic_station_w[0]
                )
                if physical_recovery_record is not None:
                    _observe_physical_recovery(
                        physical_recovery_record,
                        timestamp=session_time,
                        values=physical_recovery_values(),
                    )
                if trace_writer is not None:
                    trace_writer.writerow(
                        {
                            "trial": trial,
                            "tick": tick,
                            "session_time": session_time,
                            "phase": lifecycle.phase.value,
                            "planner_command": int(command_seen),
                            "planner_tts": float(target.time_to_strike),
                            "planner_side": float(target.swing_side),
                            "planner_bounces": planner_num_bounces,
                            "ball_x": ball_table[0],
                            "ball_y": ball_table[1],
                            "ball_z": ball_table[2],
                            "ball_world_x": ball_pos_w[0],
                            "ball_world_y": ball_pos_w[1],
                            "ball_world_z": ball_pos_w[2],
                            "ball_vx": ball_vel_w[0],
                            "ball_vy": ball_vel_w[1],
                            "ball_vz": ball_vel_w[2],
                            "intended_strike_table_x": strike_table[0],
                            "intended_strike_table_y": strike_table[1],
                            "intended_strike_table_z": strike_table[2],
                            "racket_x": racket_pos[0],
                            "racket_y": racket_pos[1],
                            "racket_z": racket_pos[2],
                            "racket_vx": racket_vel[0],
                            "racket_vy": racket_vel[1],
                            "racket_vz": racket_vel[2],
                            "racket_normal_x": racket_normal_for_side[0],
                            "racket_normal_y": racket_normal_for_side[1],
                            "racket_normal_z": racket_normal_for_side[2],
                            "target_x": float(target.pos_w[0]),
                            "target_y": float(target.pos_w[1]),
                            "target_z": float(target.pos_w[2]),
                            "target_vx": float(target.vel_w[0]),
                            "target_vy": float(target.vel_w[1]),
                            "target_vz": float(target.vel_w[2]),
                            "target_normal_x": float(target_normal[0]),
                            "target_normal_y": float(target_normal[1]),
                            "target_normal_z": float(target_normal[2]),
                            "ball_racket_distance": distance,
                            "racket_contact_force": contact_force,
                            "incoming_bounce": int(incoming_bounce),
                            "contact": int(contacted),
                            "net_cross": int(net_cross),
                            "base_x": float(base_pos[0].item()),
                            "base_y": float(base_pos[1].item()),
                            "base_z": float(base_pos[2].item()),
                            "base_pitch_deg": base_pitch,
                            "base_roll_deg": base_roll,
                            "torso_pitch_deg": torso_pitch,
                            "waist_pitch_rad": (
                                float("nan")
                                if waist_pitch_index is None
                                else float(
                                    robot.data.joint_pos[
                                        0, waist_pitch_index
                                    ].item()
                                )
                            ),
                            "base_vx": float(
                                robot.data.root_lin_vel_w[0, 0].item()
                            ),
                            "base_vy": float(
                                robot.data.root_lin_vel_w[0, 1].item()
                            ),
                            "station_error_x": float(station_error[0].item()),
                            "station_error_y": float(station_error[1].item()),
                            "base_lin_speed": float(
                                torch.norm(robot.data.root_lin_vel_w[0]).item()
                            ),
                            "base_ang_speed": float(
                                torch.norm(robot.data.root_ang_vel_w[0]).item()
                            ),
                            "feet_contact_frac": float(
                                command.feet_contact_frac[0].item()
                            ),
                            "done": int(done),
                            "termination_reason": ";".join(last_done_terms),
                        }
                    )

                previous_ball = ball_pos_w.copy()
                previous_velocity = ball_vel_w.copy()
                if done:
                    counts["fall"] += 1
                    session_done = True
                    timed_out = False
                    trial_done = True
                if ball_table[0] < -0.65 or ball_table[2] < -0.20:
                    timed_out = False
                    trial_done = True
                if trial_done:
                    break

            if command_seen:
                counts["planner_command_seen"] += 1
            if planner_side > 0:
                eval_side_name = "forehand"
            elif planner_side < 0:
                eval_side_name = "backhand"
            else:
                eval_side_name = requested_name
            side_counts[eval_side_name]["attempts"] += 1
            if incoming_bounce:
                counts["incoming_bounce"] += 1
            if contacted:
                counts["contact"] += 1
                side_counts[eval_side_name]["contact"] += 1
            if net_cross:
                counts["net_cross"] += 1
            if opponent_bounce:
                counts["opponent_bounce"] += 1
                counts["success"] += 1
                side_counts[eval_side_name]["success"] += 1

            gravity = robot.data.projected_gravity_b[0]
            final_pitch = float(
                torch.rad2deg(torch.asin(torch.clamp(gravity[0], -1.0, 1.0))).item()
            )
            final_ang_speed = float(torch.norm(robot.data.root_ang_vel_w[0]).item())
            trial_row = {
                "trial": trial,
                "requested_side": requested_name,
                "planner_side": (
                    "forehand"
                    if planner_side > 0
                    else "backhand"
                    if planner_side < 0
                    else "none"
                ),
                "planner_command_seen": int(command_seen),
                "planner_num_bounces": planner_num_bounces,
                "incoming_bounce": int(incoming_bounce),
                "contact": int(contacted),
                "net_cross": int(net_cross),
                "opponent_bounce": int(opponent_bounce),
                "success": int(opponent_bounce),
                "fell": int(session_done),
                "termination_reason": ";".join(last_done_terms),
                "timed_out": int(timed_out),
                "min_ball_racket_distance": min_distance,
                "intended_strike_table_x": float(strike_table[0]),
                "intended_strike_table_y": float(strike_table[1]),
                "intended_strike_table_z": float(strike_table[2]),
                "contact_force_peak": force_peak,
                **contact_diag,
                "final_base_pitch_deg": final_pitch,
                "final_base_ang_speed": final_ang_speed,
                "rest_max_abs_base_pitch_deg": float("nan"),
                "rest_max_abs_base_roll_deg": float("nan"),
                "rest_max_base_ang_speed": float("nan"),
                "rest_end_base_pitch_deg": float("nan"),
                "rest_end_base_roll_deg": float("nan"),
                "rest_end_base_ang_speed": float("nan"),
                "rest_reached_ready": 0,
                "rest_healthy_ready": 0,
            }
            print(
                "[isaac_physical_eval] "
                f"trial={trial} requested={requested_name} "
                f"planner={'yes' if command_seen else 'no'} side={trial_row['planner_side']} "
                f"bounce={int(incoming_bounce)} contact={int(contacted)} "
                f"net={int(net_cross)} success={int(opponent_bounce)} "
                f"fall={int(session_done)} min_dist={min_distance:.3f}",
                file=sys.stderr,
                flush=True,
            )

            if session_done:
                finish_physical_recovery(
                    physical_recovery_record,
                    net_crossed=net_cross,
                    bounced_on_opponent=opponent_bounce,
                    ready_success=False,
                    fell=True,
                )
                if trials_writer is not None:
                    trials_writer.writerow(trial_row)
                break

            # Continuous recovery: park only the ball.  Robot, policy action
            # history, lifecycle, station, and articulation state remain live.
            rest_max_abs_pitch = 0.0
            rest_max_abs_roll = 0.0
            rest_max_ang_speed = 0.0
            rest_reached_ready = False
            for rest_tick in range(max_rest_ticks):
                set_ball(park_pos, np.zeros(3))
                target = lifecycle.update(latest_command, robot_state())
                done = step_policy(target)
                rest_gravity = robot.data.projected_gravity_b[0]
                rest_pitch = float(
                    torch.rad2deg(
                        torch.asin(torch.clamp(rest_gravity[0], -1.0, 1.0))
                    ).item()
                )
                rest_roll = float(
                    torch.rad2deg(
                        torch.asin(torch.clamp(rest_gravity[1], -1.0, 1.0))
                    ).item()
                )
                rest_ang_speed = float(
                    torch.norm(robot.data.root_ang_vel_w[0]).item()
                )
                if physical_recovery_record is not None:
                    _observe_physical_recovery(
                        physical_recovery_record,
                        timestamp=session_time,
                        values=physical_recovery_values(),
                    )
                rest_max_abs_pitch = max(rest_max_abs_pitch, abs(rest_pitch))
                rest_max_abs_roll = max(rest_max_abs_roll, abs(rest_roll))
                rest_max_ang_speed = max(rest_max_ang_speed, rest_ang_speed)
                if done:
                    counts["fall"] += 1
                    session_done = True
                    break
                if rest_tick + 1 >= min_rest_ticks and lifecycle.phase == Phase.READY:
                    rest_reached_ready = True
                    break
            rest_gravity = robot.data.projected_gravity_b[0]
            rest_end_pitch = float(
                torch.rad2deg(
                    torch.asin(torch.clamp(rest_gravity[0], -1.0, 1.0))
                ).item()
            )
            rest_end_roll = float(
                torch.rad2deg(
                    torch.asin(torch.clamp(rest_gravity[1], -1.0, 1.0))
                ).item()
            )
            rest_end_ang_speed = float(
                torch.norm(robot.data.root_ang_vel_w[0]).item()
            )
            rest_end_base_z = float(
                (
                    robot.data.root_pos_w[0, 2]
                    - base_env.scene.env_origins[0, 2]
                ).item()
            )
            rest_healthy_ready = bool(
                rest_reached_ready
                and rest_end_base_z >= 0.85
                and abs(rest_end_pitch) <= 10.0
                and abs(rest_end_roll) <= 10.0
                and rest_end_ang_speed <= 0.5
                and float(command.feet_contact_frac[0].item()) >= 0.99
            )
            if rest_healthy_ready:
                counts["healthy_ready"] += 1
            finish_physical_recovery(
                physical_recovery_record,
                net_crossed=net_cross,
                bounced_on_opponent=opponent_bounce,
                ready_success=rest_healthy_ready,
                fell=session_done,
            )
            trial_row.update(
                {
                    "fell": int(session_done),
                    "termination_reason": ";".join(last_done_terms),
                    "rest_max_abs_base_pitch_deg": rest_max_abs_pitch,
                    "rest_max_abs_base_roll_deg": rest_max_abs_roll,
                    "rest_max_base_ang_speed": rest_max_ang_speed,
                    "rest_end_base_pitch_deg": rest_end_pitch,
                    "rest_end_base_roll_deg": rest_end_roll,
                    "rest_end_base_ang_speed": rest_end_ang_speed,
                    "rest_reached_ready": int(rest_reached_ready),
                    "rest_healthy_ready": int(rest_healthy_ready),
                }
            )
            if trials_writer is not None:
                trials_writer.writerow(trial_row)

        attempts = max(counts["attempts"], 1)
        result = {
            **counts,
            "checkpoint": str(checkpoint),
            "task_yaml": str(args.task_yaml),
            "physical_task": task_id,
            "termination_terms": termination_terms,
            "termination_counts": termination_counts,
            "terminal_trace_state": "post_auto_reset_when_done",
            "planner_yaml": str(pathlib.Path(args.planner_yaml).expanduser().resolve()),
            "planner_code_dir": str(planner.planner_code_dir),
            "planner_code_sha256": planner.planner_code_sha256,
            "motion_manifest": str(
                pathlib.Path(args.motion_manifest).expanduser().resolve()
            ),
            "trace_coordinate_frames": {
                "ball_x_y_z": "canonical_table",
                "ball_world_x_y_z": "isaac_world",
                "racket_x_y_z": "isaac_world",
                "target_x_y_z": "isaac_world",
                "intended_strike_table_x_y_z": "canonical_table",
            },
            "planner_command_rate": counts["planner_command_seen"] / attempts,
            "incoming_bounce_rate": counts["incoming_bounce"] / attempts,
            "contact_rate": counts["contact"] / attempts,
            "net_cross_rate": counts["net_cross"] / attempts,
            "success_rate": counts["success"] / attempts,
            "healthy_ready_rate": counts["healthy_ready"] / attempts,
            "completed_without_fall": not session_done,
            "planner_solve_calls": planner.stats.solve_calls,
            "planner_commands": planner.stats.commands,
            "planner_no_command": planner.stats.no_command,
            "planner_revision_gate_enabled": planner.command_gate is not None,
            "planner_revision_gate_reasons": dict(planner.gate_reason_counts),
            "physical_recovery_diagnostics": (
                _summarize_physical_recoveries(physical_recovery_records)
            ),
            "by_side": {},
        }
        for side_name, values in side_counts.items():
            n = max(values["attempts"], 1)
            result["by_side"][side_name] = {
                **values,
                "contact_rate": values["contact"] / n,
                "success_rate": values["success"] / n,
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.json_out:
            output = pathlib.Path(args.json_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, sort_keys=True)
                fh.write("\n")
    except Exception:
        import traceback

        print("[isaac_physical_eval] ERROR", file=sys.stderr)
        traceback.print_exc()
        status = 1
    finally:
        if trace_fh is not None:
            trace_fh.close()
        if trials_fh is not None:
            trials_fh.close()
        if env is not None:
            env.close()
        simulation_app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
