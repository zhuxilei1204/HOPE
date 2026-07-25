# Copyright (c) 2025, Intelligent Racing Inc. (dba Hitch Interactive).
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo sim-to-sim evaluation of the exported HOPE ONNX policy.

This drives the exported ``hope_pingpong.onnx`` (single layout: observation[1, 111]
-> raw_action[1, 31], no observation normalization) through the SAME 111-D
observation builder, ActionAdapter, swing lifecycle and RacketCommand that the
clean-room reference deploy runner uses, so this is a faithful test of the deploy
contract rather than a second, divergent implementation.

``success_rate`` is measured from an ACTUAL simulated ball. A real ball (free joint,
sphere) is served from the opponent side toward the robot; it flies under gravity +
no-spin aerodynamic drag and really bounces off the racket, table and net inside
MuJoCo. A return succeeds when ALL of the following are observed on the real
trajectory:

  * **contact**   -- an actual MuJoCo contact between the ball geom and the racket
                     collision geom (or the ball passes within the contact radius of
                     the racket site while the racket is moving toward it);
  * **net clear** -- after contact the ball's x crosses the net plane with its centre
                     above the net top; and
  * **opponent-half first bounce** -- the ball's first downward crossing of the table
                     surface (z reaches ball_radius descending) lands on the opponent
                     half, inside the table bounds.

    success_rate = successful_return_tasks / incoming_balls_that_entered_a_strike_task

There is NO analytic predicted-landing substitute: every event above is read off the
real simulated ball. Forehand, backhand and all serves merge into the one number.
Emits only ``{"success_rate": <float>}`` to stdout (and optionally --json-out); a
one-line human summary goes to stderr.

Two evaluation modes (``--eval-mode``):

  * ``continuous`` (default) — deploy-faithful continuous rally: the robot, policy
    ``last_action``, lifecycle and fixed station are initialized ONCE; between serves
    the policy keeps running through its follow-through/recovery with no state reset
    and no teleport, exactly like the deploy runner. The serve side follows the
    pattern FH, FH, BH, BH, ... so all four adjacent transitions (FH->FH, FH->BH,
    BH->BH, BH->FH) are exercised; the transitions actually seen are reported on
    stderr. A fall keeps affecting later serves — that is the honest continuous
    measurement.
  * ``independent`` — independent-strike evaluation: every serve resets the robot to
    its stand with a fresh lifecycle/last_action/station. This measures isolated
    swings only; it does NOT validate between-swing transitions.

Requires ``mujoco``, ``onnxruntime``, ``pyyaml`` and ``numpy`` plus the shipped
reference deploy package (``a3_deploy/a3_deploy_example``) and the
``a3_pingpong`` MJCF. Runs OUTSIDE Isaac.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np


_ISAAC_A3_CANONICAL_TO_ARTICULATION = np.array(
    [2, 5, 8, 11, 16, 12, 17, 21, 23, 25, 27, 29, 13, 18, 22, 24,
     26, 28, 30, 0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20],
    dtype=np.int64,
)
_ISAAC_A3_ARTICULATION_TO_CANONICAL = np.empty_like(_ISAAC_A3_CANONICAL_TO_ARTICULATION)
for _canonical_i, _articulation_i in enumerate(_ISAAC_A3_CANONICAL_TO_ARTICULATION):
    _ISAAC_A3_ARTICULATION_TO_CANONICAL[_articulation_i] = _canonical_i


def _obs_for_policy_joint_order(obs: np.ndarray, last_action_canonical: np.ndarray, mode: str) -> np.ndarray:
    """Return the observation layout expected by the ONNX policy for MuJoCo diagnostics.

    Current exported policies should be canonical. Older checkpoints may have been
    trained before the Isaac articulation order was gated; for those, only MuJoCo
    eval can opt into the legacy articulation-order slices without changing the
    training code or the public canonical deploy contract.
    """
    if mode != "isaac-articulation-obs-action":
        return obs
    out = np.asarray(obs, dtype=np.float32).copy()
    art_to_can = _ISAAC_A3_ARTICULATION_TO_CANONICAL
    out[3:34] = obs[3:34][art_to_can]
    out[34:65] = obs[34:65][art_to_can]
    out[65:96] = np.asarray(last_action_canonical, dtype=np.float32)[art_to_can]
    return out


def _raw_action_to_canonical(raw_action: np.ndarray, mode: str) -> np.ndarray:
    """Map policy raw-action columns into the canonical ActionAdapter order."""
    raw = np.asarray(raw_action, dtype=np.float64).reshape(31)
    if mode in ("isaac-articulation-action", "isaac-articulation-obs-action"):
        return raw[_ISAAC_A3_CANONICAL_TO_ARTICULATION].copy()
    return raw.copy()


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "a3_deploy").is_dir() and (parent / "hope_training").is_dir():
            return parent
    return here.parents[3]


def _reference_pkg_dir(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "a3_deploy" / "a3_deploy_example" / "reference"


def _default_runtime_config(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "a3_deploy" / "a3_deploy_example" / "config" / "hope_pingpong_runtime.yaml"


def _default_real_planner_yaml(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "hope_ws" / "src" / "hope_planner" / "config" / "hope_planner.yaml"


def _resolve_repo_path(repo_root: pathlib.Path, value: str | None) -> pathlib.Path | None:
    if value is None or value == "":
        return None
    path = pathlib.Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def _load_success_metric(repo_root: pathlib.Path):
    """Import the shared ``success_metric`` module by file path (pure NumPy).

    Importing it as ``whole_body_tracking.utils.success_metric`` would execute the
    training package ``__init__`` (which registers Isaac Lab gym tasks and needs
    gymnasium); this eval only needs the metric's SuccessRate accumulator, the
    TableGeometry.on_opponent_half definition, and the physics-config loader, so we
    load the single file standalone.
    """
    import importlib.util

    path = (
        repo_root / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
        / "whole_body_tracking" / "utils" / "success_metric.py"
    )
    spec = importlib.util.spec_from_file_location("hope_success_metric", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection (sys.modules[__module__]) works.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_serve_manifest(path: str | None) -> list[dict]:
    """Load training motion-manifest racket target boxes for MuJoCo serve sampling.

    The target boxes are treated as robot/MuJoCo-world racket intercept positions.
    This is intentionally closer to the current Isaac training distribution than the
    script's neutral example serve distribution.
    """
    if not path:
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ranges = []
            ok = True
            for axis in ("x", "y", "z"):
                lo = _float_or_none(row.get(f"racket_pos_{axis}_lo"))
                hi = _float_or_none(row.get(f"racket_pos_{axis}_hi"))
                if lo is None or hi is None:
                    ok = False
                    break
                ranges.append((min(lo, hi), max(lo, hi)))
            if not ok:
                continue
            side = _float_or_none(row.get("swing_side"))
            offset_x = _float_or_none(row.get("motion_racket_offset_x"))
            offset_y = _float_or_none(row.get("motion_racket_offset_y"))
            rows.append(
                {
                    "side": 1.0 if side is None or side >= 0.0 else -1.0,
                    "ranges": tuple(ranges),
                    "motion_racket_offset_xy": (
                        np.array([offset_x, offset_y], dtype=np.float64)
                        if offset_x is not None and offset_y is not None
                        else None
                    ),
                    "source": row.get("output") or row.get("source") or "",
                }
            )
    if not rows:
        raise ValueError(f"serve manifest has no complete racket_pos boxes: {path}")
    return rows


def _new_eval_counts() -> dict:
    return {
        "attempts": 0,
        "successes": 0,
        "incoming_bounce": 0,
        "contact": 0,
        "real_contact": 0,
        "proximity_contact": 0,
        "net_clear": 0,
        "first_bounce": 0,
        "opponent_bounce": 0,
        "miss_no_incoming_bounce": 0,
        "miss_no_contact": 0,
        "miss_contact_no_net": 0,
        "miss_net_no_bounce": 0,
        "miss_wrong_bounce": 0,
        "miss_fell": 0,
        "miss_timeout": 0,
        "nonfinite_action_ticks": 0,
        "min_ball_racket_distance": float("inf"),
        "max_abs_raw_action": 0.0,
        "max_abs_applied_action": 0.0,
        "max_abs_q_des": 0.0,
    }


def _update_max(counts: dict, key: str, value: float) -> None:
    if np.isfinite(value):
        counts[key] = max(float(counts[key]), float(value))


def _finalize_counts(counts: dict) -> dict:
    attempts = int(counts["attempts"])
    out = dict(counts)
    if not np.isfinite(out["min_ball_racket_distance"]):
        out["min_ball_racket_distance"] = None
    for key in (
        "successes",
        "incoming_bounce",
        "contact",
        "real_contact",
        "proximity_contact",
        "net_clear",
        "first_bounce",
        "opponent_bounce",
        "miss_no_incoming_bounce",
        "miss_no_contact",
        "miss_contact_no_net",
        "miss_net_no_bounce",
        "miss_wrong_bounce",
        "miss_fell",
        "miss_timeout",
    ):
        out[f"{key}_rate"] = (float(counts[key]) / attempts) if attempts > 0 else None
    return out


def _norm(v) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def _unit_or_none(v):
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return None
    return arr / n


def _angle_deg(a, b):
    a_u = _unit_or_none(a)
    b_u = _unit_or_none(b)
    if a_u is None or b_u is None:
        return None
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a_u, b_u)), -1.0, 1.0))))


def _xyz(row: dict, prefix: str, value) -> None:
    if value is None:
        row[f"{prefix}_x"] = ""
        row[f"{prefix}_y"] = ""
        row[f"{prefix}_z"] = ""
        return
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    row[f"{prefix}_x"] = float(arr[0])
    row[f"{prefix}_y"] = float(arr[1])
    row[f"{prefix}_z"] = float(arr[2])


def _outgoing_flight_diag(start_pos_table: np.ndarray, ball_vel_w: np.ndarray, physics, table) -> dict:
    """Predict net crossing and first bounce from a post-contact ball state.

    This is only a diagnostic readout. The public MuJoCo eval result still uses the
    simulated ball events observed later in the same trial.
    """
    p = np.asarray(start_pos_table, dtype=np.float64).copy()
    v = np.asarray(ball_vel_w, dtype=np.float64).copy()
    dt = 0.002
    surface_z = physics.ball_radius
    out = {
        "pred_net_cross": 0,
        "pred_net_z": "",
        "pred_net_clear": 0,
        "pred_bounce_x": "",
        "pred_bounce_y": "",
        "pred_opponent_bounce": 0,
    }
    for _ in range(int(3.0 / dt)):
        a = physics.acceleration(v)
        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v + a * dt
        if not out["pred_net_cross"] and p[0] < table.net_x <= p_new[0]:
            dx = p_new[0] - p[0]
            frac = (table.net_x - p[0]) / dx if abs(dx) > 1e-12 else 0.5
            net_z = float(p[2] + frac * (p_new[2] - p[2]))
            out["pred_net_cross"] = 1
            out["pred_net_z"] = net_z
            out["pred_net_clear"] = int(net_z > (table.net_height + physics.ball_radius))
        if p[2] > surface_z >= p_new[2] and v[2] < 0.0:
            dz = p_new[2] - p[2]
            frac = (surface_z - p[2]) / dz if abs(dz) > 1e-12 else 0.5
            land_x = float(p[0] + frac * (p_new[0] - p[0]))
            land_y = float(p[1] + frac * (p_new[1] - p[1]))
            out["pred_bounce_x"] = land_x
            out["pred_bounce_y"] = land_y
            out["pred_opponent_bounce"] = int(table.on_opponent_half(land_x, land_y))
            break
        p, v = p_new, v_new
    return out


def _choose_racket_normal(racket_xmat: np.ndarray, to_ball: np.ndarray) -> np.ndarray:
    """Pick the racket geom axis most aligned with the ball-side direction.

    The collision mesh/site convention is model-specific, so diagnostics avoid
    assuming a named local face-normal axis. The chosen normal is always oriented
    from the racket toward the ball at contact.
    """
    axes = [racket_xmat[:, i].copy() for i in range(3)]
    to_ball_u = _unit_or_none(to_ball)
    if to_ball_u is None:
        return axes[0]
    axis = max(axes, key=lambda a: abs(float(np.dot(a, to_ball_u))))
    if float(np.dot(axis, to_ball_u)) < 0.0:
        axis = -axis
    return axis


def _new_contact_diag_row(trial: int, side_name: str, strike_pt, serve_pos, serve_vel) -> dict:
    row = {
        "trial": int(trial),
        "side": side_name,
        "contact_kind": "none",
        "contact_tick": "",
        "contact_substep": "",
        "contact_time_offset_s": "",
        "contact_dist": "",
        "phase_at_contact": "",
        "success": 0,
        "miss_reason": "",
        "net_clear": 0,
        "first_bounce": 0,
        "first_bounce_x": "",
        "first_bounce_y": "",
        "opponent_bounce": 0,
        "fallen": 0,
        "timed_out": 0,
        "min_ball_racket_distance": "",
        "base_z_at_contact": "",
        "base_z_end": "",
        "time_to_strike_cmd": "",
        "racket_target_speed": "",
        "racket_speed_pre": "",
        "racket_speed_post": "",
        "ball_speed_pre": "",
        "ball_speed_post": "",
        "ball_speed_delta": "",
        "racket_vel_target_error": "",
        "racket_vel_target_angle_deg": "",
        "ball_post_vs_target_vel_error": "",
        "ball_post_vs_target_vel_angle_deg": "",
        "ball_post_vs_target_outgoing_vel_error": "",
        "ball_post_vs_target_outgoing_vel_angle_deg": "",
        "closing_speed_pre": "",
        "outgoing_normal_speed_post": "",
        "raw_action_max_at_contact": "",
        "applied_action_max_at_contact": "",
        "q_des_max_at_contact": "",
    }
    for prefix, value in (
        ("strike_sample", strike_pt),
        ("serve_pos", serve_pos),
        ("serve_vel", serve_vel),
        ("contact_pos", None),
        ("mujoco_contact_normal", None),
        ("racket_normal", None),
        ("target_pos", None),
        ("target_vel", None),
        ("target_outgoing_vel", None),
        ("target_normal", None),
        ("ball_pre_pos", None),
        ("ball_pre_vel", None),
        ("ball_post_pos", None),
        ("ball_post_vel", None),
        ("racket_pre_pos", None),
        ("racket_pre_vel", None),
        ("racket_post_pos", None),
        ("racket_post_vel", None),
    ):
        _xyz(row, prefix, value)
    row.update(
        {
            "pred_net_cross": 0,
            "pred_net_z": "",
            "pred_net_clear": 0,
            "pred_bounce_x": "",
            "pred_bounce_y": "",
            "pred_opponent_bounce": 0,
        }
    )
    return row


def _contact_diag_fields() -> list[str]:
    base = [
        "trial",
        "side",
        "contact_kind",
        "contact_tick",
        "contact_substep",
        "contact_time_offset_s",
        "contact_dist",
        "phase_at_contact",
        "success",
        "miss_reason",
        "net_clear",
        "first_bounce",
        "first_bounce_x",
        "first_bounce_y",
        "opponent_bounce",
        "fallen",
        "timed_out",
        "min_ball_racket_distance",
        "base_z_at_contact",
        "base_z_end",
    ]
    xyz_prefixes = [
        "strike_sample",
        "serve_pos",
        "serve_vel",
        "contact_pos",
        "mujoco_contact_normal",
        "racket_normal",
        "target_pos",
        "target_vel",
        "target_outgoing_vel",
        "target_normal",
        "ball_pre_pos",
        "ball_pre_vel",
        "ball_post_pos",
        "ball_post_vel",
        "racket_pre_pos",
        "racket_pre_vel",
        "racket_post_pos",
        "racket_post_vel",
    ]
    xyz_fields = [f"{prefix}_{axis}" for prefix in xyz_prefixes for axis in ("x", "y", "z")]
    scalars = [
        "time_to_strike_cmd",
        "racket_target_speed",
        "racket_speed_pre",
        "racket_speed_post",
        "ball_speed_pre",
        "ball_speed_post",
        "ball_speed_delta",
        "racket_vel_target_error",
        "racket_vel_target_angle_deg",
        "ball_post_vs_target_vel_error",
        "ball_post_vs_target_vel_angle_deg",
        "ball_post_vs_target_outgoing_vel_error",
        "ball_post_vs_target_outgoing_vel_angle_deg",
        "closing_speed_pre",
        "outgoing_normal_speed_post",
        "pred_net_cross",
        "pred_net_z",
        "pred_net_clear",
        "pred_bounce_x",
        "pred_bounce_y",
        "pred_opponent_bounce",
        "raw_action_max_at_contact",
        "applied_action_max_at_contact",
        "q_des_max_at_contact",
    ]
    return base + xyz_fields + scalars


class _MujocoVideoRecorder:
    def __init__(self, scene, path: str, *, width: int, height: int, fps: int) -> None:
        import imageio.v2 as imageio
        import mujoco

        pathlib.Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self._renderer = mujoco.Renderer(scene.model, height=height, width=width)
        self._writer = imageio.get_writer(path, fps=fps)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[:] = [scene.near_edge_x + 1.15, 0.0, scene.table_height + 0.35]
        self._camera.distance = 3.3
        self._camera.azimuth = -140.0
        self._camera.elevation = -18.0

    def capture(self, scene) -> None:
        self._renderer.update_scene(scene.data, camera=self._camera)
        self._writer.append_data(self._renderer.render())

    def close(self) -> None:
        self._writer.close()
        self._renderer.close()


# ---------------------------------------------------------------------------------------------------
# Example serve distribution (planner-less). These are NEUTRAL example incoming balls that arrive in
# front of the robot with a return toward the opponent half; they are NOT the training sampling boxes.
# Replace them (or feed your own planner's RacketCommand stream) for a deployment-faithful evaluation.
# All positions/velocities below are in the MuJoCo world (robot) frame, metres / m·s^-1.
# ---------------------------------------------------------------------------------------------------
def _rollout_drag_position(origin: np.ndarray, velocity: np.ndarray, flight_t: float, physics, dt: float = 0.002):
    """Forward-integrate no-spin drag flight to ``flight_t`` with the eval physics."""
    p = np.asarray(origin, dtype=np.float64).copy()
    v = np.asarray(velocity, dtype=np.float64).copy()
    remaining = float(flight_t)
    while remaining > 1e-12:
        h = min(dt, remaining)
        a = physics.acceleration(v)
        p = p + v * h + 0.5 * a * h * h
        v = v + a * h
        remaining -= h
    return p, v


def _solve_drag_serve_velocity(origin: np.ndarray, target: np.ndarray, flight_t: float, physics) -> np.ndarray:
    """Shoot an initial velocity whose drag trajectory reaches ``target`` at ``flight_t``.

    The eval predictor and MuJoCo scene both include quadratic drag. The old
    no-drag ballistic serve made the ball reach the strike plane too late/low,
    sometimes producing racket targets metres below the table. This small Newton
    shooting loop keeps the sampled serve and the predictor on the same physics.
    """
    accel = np.array([0.0, 0.0, -physics.gravity])
    v = (target - origin) / flight_t - 0.5 * accel * flight_t
    eps = 1e-3
    for _ in range(8):
        p, _ = _rollout_drag_position(origin, v, flight_t, physics)
        err = p - target
        if float(np.linalg.norm(err)) < 1e-4:
            break
        jac = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            dv = np.zeros(3, dtype=np.float64)
            dv[j] = eps
            p_eps, _ = _rollout_drag_position(origin, v + dv, flight_t, physics)
            jac[:, j] = (p_eps - p) / eps
        try:
            step = np.linalg.solve(jac, err)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jac, err, rcond=None)[0]
        v = v - step
        speed = float(np.linalg.norm(v))
        if speed > physics.velocity_clip:
            v *= physics.velocity_clip / speed
    return v


def _table_contact_params(args) -> tuple[float, float]:
    """Return (normal restitution, tangential speed retain) for the table bounce model."""
    e = float(getattr(args, "_table_restitution", 0.9215))
    damping = float(getattr(args, "_table_tangential_damping", 0.369))
    return e, max(0.0, min(1.0, 1.0 - damping))


def _paddle_contact_params(args) -> tuple[float, float]:
    """Return (normal restitution, tangential speed retain) for the planner-side paddle model."""
    e = float(getattr(args, "_paddle_restitution", 0.654))
    damping = float(getattr(args, "_paddle_tangential_damping", 0.52))
    return e, max(0.0, min(1.0, 1.0 - damping))


def _bounce_ball_velocity(v_in: np.ndarray, args) -> np.ndarray:
    """Simple no-spin table bounce used by the planner-side predictor."""
    e, tangential_retain = _table_contact_params(args)
    out = np.asarray(v_in, dtype=np.float64).copy()
    out[:2] *= tangential_retain
    out[2] = -e * out[2]
    return out


def _impact_inverse_racket_command(v_in: np.ndarray, v_out: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    """Invert the moving-paddle no-spin contact model into racket velocity + blade normal."""
    v_in = np.asarray(v_in, dtype=np.float64)
    v_out = np.asarray(v_out, dtype=np.float64)
    normal = _unit_or_none(v_out - v_in)
    if normal is None:
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    e, retain = _paddle_contact_params(args)
    vin_n = float(np.dot(v_in, normal))
    vout_n = float(np.dot(v_out, normal))
    racket_n = (vout_n + e * vin_n) / max(1.0 + e, 1.0e-6)
    vin_t = v_in - vin_n * normal
    vout_t = v_out - vout_n * normal
    if abs(1.0 - retain) > 1.0e-4:
        racket_t = (vout_t - retain * vin_t) / (1.0 - retain)
    else:
        racket_t = vout_t
    racket_vel = racket_n * normal + racket_t
    speed = float(np.linalg.norm(racket_vel))
    max_speed = float(getattr(args, "planner_max_racket_speed", 2.8))
    min_speed = float(getattr(args, "planner_min_racket_speed", 0.3))
    if speed > max_speed:
        racket_vel *= max_speed / speed
    elif 1.0e-6 < speed < min_speed:
        racket_vel *= min_speed / speed
    return racket_vel, normal


def _sample_one_bounce_serve(rng, strike_pt: np.ndarray, y: float, scene, physics, args):
    """Sample an incoming ball that bounces once on the robot half before the strike point.

    The serve is constructed in the table/MuJoCo world frame with a lightweight no-spin bounce
    model.  MuJoCo still owns the authoritative physics: this only chooses a plausible initial
    state whose real simulated ball should land on the near half, rebound, then pass the policy's
    strike plane.
    """
    surface_z = scene.table_height + physics.ball_radius
    target_table_x = float(strike_pt[0] - scene.near_edge_x)
    target_table_y = float(strike_pt[1] - scene.offset[1])
    margin = 0.08
    lo_x = max(0.18, target_table_x + 0.18)
    hi_x = min(scene.net_x_table - margin, target_table_x + 0.78)
    if hi_x <= lo_x:
        lo_x = max(0.12, min(scene.net_x_table - margin, target_table_x + 0.20))
        hi_x = min(scene.net_x_table - margin, lo_x + 0.30)
    bounce_table_x = float(rng.uniform(lo_x, hi_x))
    bounce_table_y = float(np.clip(target_table_y + rng.uniform(-0.18, 0.18), -scene.width + 0.08, -0.08))
    bounce = np.array(
        [scene.near_edge_x + bounce_table_x, bounce_table_y + scene.offset[1], surface_z],
        dtype=np.float64,
    )

    # Desired post-bounce flight from bounce point to the strike point.
    post_t = float(rng.uniform(float(args.bounce_post_time[0]), float(args.bounce_post_time[1])))
    accel = np.array([0.0, 0.0, -physics.gravity], dtype=np.float64)
    post_vel = (strike_pt - bounce) / post_t - 0.5 * accel * post_t

    e, tangential_retain = _table_contact_params(args)
    pre_vel = post_vel.copy()
    pre_vel[:2] = post_vel[:2] / max(tangential_retain, 1.0e-3)
    pre_vel[2] = -abs(post_vel[2]) / max(e, 1.0e-3)

    # Choose an origin on/above the opponent side so the incoming path really has a near-half bounce.
    pre_t = float(rng.uniform(float(args.bounce_pre_time[0]), float(args.bounce_pre_time[1])))
    origin = bounce - pre_vel * pre_t + 0.5 * accel * pre_t * pre_t
    origin[0] = max(origin[0], scene.near_edge_x + scene.net_x_table + 0.25)
    origin[0] = min(origin[0], scene.near_edge_x + scene.length - 0.12)
    origin[1] = float(np.clip(origin[1], scene.offset[1] - scene.width + 0.08, scene.offset[1] - 0.08))
    origin[2] = float(np.clip(origin[2], scene.table_height + 0.18, scene.table_height + 0.55))

    serve_vel = _solve_drag_serve_velocity(origin, bounce, pre_t, physics)
    return origin, serve_vel


def _sample_serve(rng, side, scene, physics, args, serve_manifest=None):
    """Sample (strike_point, serve_origin, serve_velocity) in the MuJoCo world frame.

    A strike point in the robot's reachable zone is chosen (forehand to the robot's
    right, backhand to its left), a serve origin is placed on the opponent side, and
    the serve velocity is solved with the same no-spin drag model used by the
    predictor/evaluator so the ball actually reaches the sampled strike point.
    """
    if serve_manifest:
        candidates = [r for r in serve_manifest if (r["side"] >= 0.0) == (side >= 0.0)]
        if not candidates:
            candidates = serve_manifest
        row = candidates[int(rng.integers(0, len(candidates)))]
        strike_pt = np.array([rng.uniform(lo, hi) for lo, hi in row["ranges"]], dtype=np.float64)
        if bool(getattr(args, "_force_serve_strike_plane_x", False)):
            strike_pt[0] = scene.near_edge_x + float(getattr(args, "_serve_strike_plane_x", args.strike_plane_x))
        y = float(strike_pt[1])
    else:
        row = None
        strike_table_x = float(getattr(args, "_serve_strike_plane_x", args.strike_plane_x))
        strike_x = scene.near_edge_x + strike_table_x  # MuJoCo x of the fixed strike plane
        if side >= 0:  # FOREHAND -> robot's right = -y in the MuJoCo/robot frame
            y = rng.uniform(-0.45, -0.12)
        else:          # BACKHAND -> robot's left = +y
            y = rng.uniform(0.05, 0.38)
        z = scene.table_height + rng.uniform(0.10, 0.22)
        strike_pt = np.array([strike_x, y, z], dtype=np.float64)

    if args.incoming_trajectory == "one-bounce":
        origin, serve_vel = _sample_one_bounce_serve(rng, strike_pt, y, scene, physics, args)
    else:
        origin = np.array(
            [
                scene.near_edge_x + rng.uniform(1.9, 2.4),
                y + rng.uniform(-0.15, 0.15),
                scene.table_height + rng.uniform(0.15, 0.35),
            ],
            dtype=np.float64,
        )
        flight_t = rng.uniform(0.75, 1.0)
        serve_vel = _solve_drag_serve_velocity(origin, strike_pt, flight_t, physics)
    return strike_pt, origin, serve_vel, row


def _station_xy_for_observation(fixed_station_xy, target, phase, manifest_row, args):
    """Return the station XY that should occupy obs[101:103].

    Training keeps the public 111-D observation layout unchanged, but in dynamic-station
    mode the old ``fixed_station_error_xy`` slot contains the current station target
    before impact.  MuJoCo eval must mirror that or the policy sees a different task
    than it was trained on.
    """
    fixed_station_xy = np.asarray(fixed_station_xy, dtype=np.float64)[:2]
    mode = args.station_mode
    if mode == "fixed":
        return fixed_station_xy
    if mode == "auto":
        has_offset = manifest_row is not None and manifest_row.get("motion_racket_offset_xy") is not None
        if not has_offset:
            return fixed_station_xy
    elif mode != "dynamic-from-manifest":
        raise ValueError(f"Unsupported station mode: {mode}")

    if manifest_row is None:
        return fixed_station_xy
    offset_xy = manifest_row.get("motion_racket_offset_xy")
    if offset_xy is None:
        return fixed_station_xy
    phase_value = getattr(phase, "value", str(phase))
    if phase_value not in ("swing", "follow_through"):
        return fixed_station_xy
    if float(target.time_to_strike) < -float(args.dynamic_station_post_window):
        return fixed_station_xy

    desired = np.asarray(target.pos_w, dtype=np.float64)[:2] - np.asarray(offset_xy, dtype=np.float64)[:2]
    rel = desired - fixed_station_xy
    clip_x = sorted((float(args.dynamic_station_clip_x[0]), float(args.dynamic_station_clip_x[1])))
    clip_y = sorted((float(args.dynamic_station_clip_y[0]), float(args.dynamic_station_clip_y[1])))
    rel[0] = np.clip(rel[0], clip_x[0], clip_x[1])
    rel[1] = np.clip(rel[1], clip_y[0], clip_y[1])
    return fixed_station_xy + float(args.dynamic_station_blend) * rel


def _predict_command(
    ball_pos,
    ball_vel,
    side,
    scene,
    physics,
    args,
    task_id,
    revision,
    RacketCommand,
    *,
    strike_x_w: float | None = None,
):
    """Lightweight inline no-spin intercept predictor -> a RacketCommand.

    Forward-integrates the true ball with the shared no-spin model (gravity +
    quadratic drag) until it crosses the fixed strike plane, producing the racket
    target position (where the ball will be), the time-to-strike, and a target racket
    velocity aimed at the fixed opponent-half landing point. ``swing_side`` is the
    side locked for this task. This mirrors the planner's job but reads the ball
    directly (the evaluator is the ground truth, so no mocap emulation is needed).
    """
    strike_x = float(strike_x_w) if strike_x_w is not None else scene.near_edge_x + args.strike_plane_x
    p = np.asarray(ball_pos, dtype=np.float64).copy()
    v = np.asarray(ball_vel, dtype=np.float64).copy()
    dt = 0.002
    t = 0.0
    y_cross, z_cross, tts = float(p[1]), float(p[2]), 0.0
    v_cross = v.copy()
    bounced = False
    surface_z = scene.table_height + physics.ball_radius
    for _ in range(int(2.0 / dt)):
        a = physics.acceleration(v)
        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v + a * dt
        if args.planner_mode == "bounce-aware" and not bounced:
            z0 = p[2]
            z1 = p_new[2]
            if z0 > surface_z >= z1 and v[2] < 0.0:
                dz = z1 - z0
                frac = (surface_z - z0) / dz if abs(dz) > 1e-12 else 0.5
                x_b = float(p[0] + frac * (p_new[0] - p[0]))
                y_b = float(p[1] + frac * (p_new[1] - p[1]))
                x_t = x_b - scene.offset[0]
                y_t = y_b - scene.offset[1]
                on_near_half = (0.0 <= x_t <= scene.net_x_table) and (-scene.width <= y_t <= 0.0)
                if on_near_half:
                    v_impact = v + a * (frac * dt)
                    p_new = np.array([x_b, y_b, surface_z], dtype=np.float64)
                    v_new = _bounce_ball_velocity(v_impact, args)
                    bounced = True
        if p[0] >= strike_x > p_new[0]:  # ball moving -x through the strike plane
            dx = p_new[0] - p[0]
            frac = (strike_x - p[0]) / dx if abs(dx) > 1e-12 else 0.5
            y_cross = float(p[1] + frac * (p_new[1] - p[1]))
            z_cross = float(p[2] + frac * (p_new[2] - p[2]))
            tts = t + frac * dt
            v_cross = v + a * (frac * dt)
            break
        p, v = p_new, v_new
        t += dt

    target_pos = np.array([strike_x, y_cross, z_cross], dtype=np.float64)

    # Target racket velocity: the ballistic velocity that would send the ball from the
    # strike point to the fixed opponent-half landing target in a fixed time.
    landing_table = np.array(
        [(scene.net_x_table + scene.length) / 2.0, -scene.width / 2.0, 0.0], dtype=np.float64
    )
    landing_mujoco = landing_table + scene.offset
    t_out = 0.6
    accel = np.array([0.0, 0.0, -physics.gravity])
    target_outgoing_vel = (landing_mujoco - target_pos) / t_out - 0.5 * accel * t_out
    target_vel, target_normal = _impact_inverse_racket_command(v_cross, target_outgoing_vel, args)

    cmd = RacketCommand(
        task_id=task_id,
        task_revision=revision,
        swing_side=side,
        position=target_pos,
        velocity=target_vel,
        time_to_strike=float(tts),
        target_normal=target_normal,
    )
    cmd.outgoing_velocity = target_outgoing_vel
    return cmd


def _perturb_command(cmd, args, rng, RacketCommand):
    """Apply planner-command perturbations for robustness/sensitivity evaluation."""
    pos = np.asarray(cmd.position, dtype=np.float64).copy()
    vel = np.asarray(cmd.velocity, dtype=np.float64).copy()
    outgoing_vel = getattr(cmd, "outgoing_velocity", None)
    target_normal = getattr(cmd, "target_normal", None)
    tts = float(cmd.time_to_strike)

    pos_offset = np.asarray(args.planner_target_pos_offset, dtype=np.float64)
    pos_noise = np.asarray(args.planner_target_pos_noise_std, dtype=np.float64)
    vel_offset = np.asarray(args.planner_target_vel_offset, dtype=np.float64)
    vel_noise = np.asarray(args.planner_target_vel_noise_std, dtype=np.float64)

    pos = pos + pos_offset
    if np.any(pos_noise > 0.0):
        pos = pos + rng.normal(0.0, pos_noise, size=3)

    vel = vel * float(args.planner_target_vel_scale) + vel_offset
    if abs(float(args.planner_target_vel_yaw_deg)) > 1.0e-9:
        theta = np.deg2rad(float(args.planner_target_vel_yaw_deg))
        c, s = np.cos(theta), np.sin(theta)
        rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        vel = rot_z @ vel
    if np.any(vel_noise > 0.0):
        vel = vel + rng.normal(0.0, vel_noise, size=3)

    tts = max(0.0, tts + float(args.planner_tts_offset))
    if float(args.planner_tts_noise_std) > 0.0:
        tts = max(0.0, tts + float(rng.normal(0.0, float(args.planner_tts_noise_std))))

    out = RacketCommand(
        task_id=cmd.task_id,
        task_revision=cmd.task_revision,
        swing_side=cmd.swing_side,
        position=pos,
        velocity=vel,
        time_to_strike=tts,
        target_normal=target_normal,
    )
    if outgoing_vel is not None:
        out.outgoing_velocity = np.asarray(outgoing_vel, dtype=np.float64).copy()
    return out


class _RealHopePlannerBridge:
    """Node-faithful pure-Python HOPE planner bridge for MuJoCo eval.

    The real planner consumes table-frame ball positions at mocap rate.  MuJoCo eval
    owns the physics state at 50 Hz, so this bridge linearly interpolates table-frame
    ball positions to a configurable mocap rate, applies the same solve-rate limiter
    and task lifecycle as ``hope_planner.node.HOPEPlannerNode``, then converts the
    planner command back to the deploy runner's world-frame ``RacketCommand``.
    """

    def __init__(self, repo_root, scene, args, DeployRacketCommand, forehand: int, backhand: int):
        self.repo_root = pathlib.Path(repo_root)
        self.scene = scene
        self.args = args
        self.DeployRacketCommand = DeployRacketCommand
        self.forehand = int(forehand)
        self.backhand = int(backhand)

        pkg_parent = self.repo_root / "hope_ws" / "src" / "hope_planner"
        if str(pkg_parent) not in sys.path:
            sys.path.insert(0, str(pkg_parent))

        from hope_planner.constants import PlannerConfig, load_ball_physics, load_paddle_params, load_table_params
        from hope_planner.planner import HOPEPlanner
        from hope_planner.side_selection import select_swing_side

        self._PlannerConfig = PlannerConfig
        self._HOPEPlanner = HOPEPlanner
        self._load_ball_physics = load_ball_physics
        self._load_paddle_params = load_paddle_params
        self._load_table_params = load_table_params
        self._select_swing_side = select_swing_side

        self.yaml_path = _resolve_repo_path(self.repo_root, args.real_planner_yaml) or _default_real_planner_yaml(self.repo_root)
        self.params = self._load_yaml_params(self.yaml_path)
        self.physics_path = self._resolve_physics_path(args.real_planner_physics_path)
        self.physics = self._make_physics()
        self.table = self._make_table()
        self.config = self._make_config()
        self.x_hit_table = float(self.config.x_hit)
        self.solve_period = (
            float(args.real_planner_solve_period)
            if args.real_planner_solve_period is not None
            else float(self.params.get("solve_period_s", 0.02))
        )
        self.mocap_hz = float(args.real_planner_mocap_hz)
        self.mocap_dt = 1.0 / self.mocap_hz if self.mocap_hz > 0.0 else None
        self.split_y = float(self.params.get("swing_side_split_y", -0.7625))
        self.hysteresis_y = max(0.0, float(self.params.get("swing_side_hysteresis_y", 0.0)))

        self._task_id = 0
        self._task_revision = 0
        self._task_active = False
        self._locked_side = self.forehand
        self._prev_side = 0
        self._last_solve_t = None
        self._last_control_t = None
        self._last_control_pos = None
        self._next_mocap_t = None
        self.solve_calls = 0
        self.command_count = 0
        self.no_command_count = 0
        self.reset_for_new_ball()

    @staticmethod
    def _load_yaml_params(path: pathlib.Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"real planner yaml not found: {path}")
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        params = doc.get("hope_planner", {}).get("ros__parameters", doc)
        return dict(params or {})

    def _resolve_physics_path(self, value: str | None) -> str | None:
        candidate = _resolve_repo_path(self.repo_root, value)
        if candidate is None:
            candidate = _resolve_repo_path(self.repo_root, self.params.get("ball_physics_path"))
        if candidate is None:
            return None
        return str(candidate)

    def _make_physics(self):
        physics = self._load_ball_physics(self.physics_path)
        for arg_name, param_name, attr_name in (
            ("real_planner_drag_k", "drag_k", "k"),
            ("real_planner_table_c_h", "table_c_h", "C_h"),
            ("real_planner_table_c_v", "table_c_v", "C_v"),
        ):
            override = getattr(self.args, arg_name, None)
            if override is None:
                override = self.params.get(param_name, -1.0)
            override = float(override)
            if override >= 0.0:
                setattr(physics, attr_name, override)
        return physics

    def _make_table(self):
        return self._load_table_params(self.physics_path, y_max=float(self.params.get("table_y_max", 0.0)))

    def _make_config(self):
        paddle = self._load_paddle_params(self.physics_path)
        x_hit = self.args.real_planner_x_hit
        if x_hit is None:
            x_hit = self.params.get("x_hit", 0.2)
        target_land = np.array(
            [
                float(self.params.get("target_land_x", 2.055)),
                float(self.params.get("target_land_y", -0.7625)),
                float(self.physics.radius),
            ],
            dtype=np.float64,
        )
        return self._PlannerConfig(
            x_hit=float(x_hit),
            target_land=target_land,
            delta_t_flight=float(self.params.get("delta_t_flight", 0.5)),
            max_predict_time=float(self.params.get("max_predict_time", 2.0)),
            fit_window=int(self.params.get("fit_window", 67)),
            min_ready_samples=int(self.params.get("min_ready_samples", 6)),
            bounce_z_tol=float(self.params.get("bounce_z_tol", 0.005)),
            bounce_center_z_max=float(self.params.get("bounce_center_z_max", 0.11)),
            C_r=float(paddle["C_r"]),
            paddle_a_t=float(paddle["paddle_a_t"]),
            paddle_b_t=float(paddle["paddle_b_t"]),
            paddle_mu=float(paddle["paddle_mu"]),
        )

    def reset_for_new_ball(self) -> None:
        self.planner = self._HOPEPlanner(physics=self.physics, config=self.config, table=self.table)
        self._task_revision = 0
        self._task_active = False
        self._last_solve_t = None
        self._last_control_t = None
        self._last_control_pos = None
        self._next_mocap_t = None

    def _select_side(self, intercept_y: float) -> int:
        side = self._select_swing_side(float(intercept_y), self.split_y, self.hysteresis_y, self._prev_side)
        return self.forehand if side >= 0 else self.backhand

    def _push_sample(self, t: float, p_table: np.ndarray):
        if (
            self.solve_period > 0.0
            and self._last_solve_t is not None
            and 0.0 <= (float(t) - self._last_solve_t) < self.solve_period
        ):
            self.planner.estimator.push(float(t), np.asarray(p_table, dtype=np.float64))
            return None
        self._last_solve_t = float(t)
        self.solve_calls += 1
        try:
            cmd = self.planner.update(float(t), np.asarray(p_table, dtype=np.float64))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            self.no_command_count += 1
            return None

        if cmd is None:
            if self.planner.ball_incoming is False:
                self._task_active = False
            self.no_command_count += 1
            return None

        if not self._task_active:
            self._task_id += 1
            self._task_revision = 0
            self._locked_side = self._select_side(float(cmd.p_intercept[1]))
            self._prev_side = self._locked_side
            self._task_active = True
        else:
            self._task_revision += 1

        tts = self.planner.time_to_strike
        if tts is None:
            latest_t = getattr(self.planner, "_latest_t", float(t))
            tts = float(cmd.t_strike) - float(latest_t)
        normal = np.asarray(cmd.n_racket, dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-9)
        deploy_cmd = self.DeployRacketCommand(
            task_id=self._task_id,
            task_revision=self._task_revision,
            swing_side=self._locked_side,
            position=np.asarray(cmd.p_intercept, dtype=np.float64) + self.scene.offset,
            velocity=np.asarray(cmd.v_racket, dtype=np.float64),
            time_to_strike=max(0.0, float(tts)),
            target_normal=normal,
        )
        try:
            deploy_cmd.outgoing_velocity = self.planner.target_planner._compute_outgoing_velocity(
                np.asarray(cmd.p_intercept, dtype=np.float64),
                np.asarray(self.config.target_land, dtype=np.float64),
                float(self.config.delta_t_flight),
            )
        except Exception:
            pass
        deploy_cmd.planner_num_bounces = int(getattr(cmd, "num_bounces", 0))
        self.command_count += 1
        return deploy_cmd

    def update(self, ball_pos_w: np.ndarray, t: float):
        p_table = self.scene.to_table(np.asarray(ball_pos_w, dtype=np.float64))
        t = float(t)
        if self._last_control_t is None or self._last_control_pos is None or t <= self._last_control_t:
            self._last_control_t = t
            self._last_control_pos = p_table.copy()
            self._next_mocap_t = None if self.mocap_dt is None else t + self.mocap_dt
            return self._push_sample(t, p_table)

        latest_cmd = None
        if self.mocap_dt is None:
            latest_cmd = self._push_sample(t, p_table)
        else:
            sample_t = self._next_mocap_t if self._next_mocap_t is not None else self._last_control_t
            while sample_t <= t + 1.0e-9:
                alpha = (sample_t - self._last_control_t) / max(t - self._last_control_t, 1.0e-9)
                interp = (1.0 - alpha) * self._last_control_pos + alpha * p_table
                cmd = self._push_sample(sample_t, interp)
                if cmd is not None:
                    latest_cmd = cmd
                sample_t += self.mocap_dt
            self._next_mocap_t = sample_t

        self._last_control_t = t
        self._last_control_pos = p_table.copy()
        return latest_cmd


def run_eval(args) -> dict:
    repo_root = _repo_root()
    if args.record_video:
        os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)

    # Make the reference deploy package importable (shared 111-D obs / ActionAdapter /
    # lifecycle / RacketCommand / ONNX wrapper).
    ref_dir = pathlib.Path(args.reference_dir) if args.reference_dir else _reference_pkg_dir(repo_root)
    sys.path.insert(0, str(ref_dir))
    from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
    from a3_deploy_onnx_ref_pingpong.joint_order import HEAD_INDICES, JOINT_NAMES
    from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle
    from a3_deploy_onnx_ref_pingpong.observation import OBS_DIM_NORMAL114, build_observation, build_observation_normal114
    from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy
    from a3_deploy_onnx_ref_pingpong.racket_command import (
        BACKHAND,
        FOREHAND,
        QueueRacketCommandSource,
        RacketCommand,
    )

    # Shared success metric + physics config (pure NumPy; the DEFINITION only -- fed
    # real simulated events here, never an analytic predicted-landing rollout). Loaded
    # directly by file path so importing it does NOT drag in the Isaac Lab training
    # package (its __init__ registers gym tasks that need gymnasium/Isaac).
    metric = _load_success_metric(repo_root)
    BallPhysics = metric.BallPhysics
    SuccessRate = metric.SuccessRate
    TableGeometry = metric.TableGeometry
    load_ball_physics_config = metric.load_ball_physics_config

    # The MuJoCo scene builder lives next to this script.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mujoco_pingpong_scene import PingPongRealPhysicsScene

    runtime_cfg = RuntimeConfig.load(args.runtime_config or _default_runtime_config(repo_root))
    if float(args.lifecycle_recovery_blend_seconds) > 0.0:
        runtime_cfg.lifecycle.recovery_blend_s = float(args.lifecycle_recovery_blend_seconds)
        runtime_cfg.lifecycle.recovery_blend_velocity = bool(args.lifecycle_recovery_blend_velocity)
    onnx_path = args.onnx or str(runtime_cfg.onnx_path)
    robot_xml = args.model_xml or str(runtime_cfg.model_xml_path)

    ball_cfg = load_ball_physics_config()
    table_contact = ball_cfg.get("contact", {}).get("table", {})
    args._table_restitution = float(table_contact.get("restitution", 0.9215))
    args._table_tangential_damping = float(table_contact.get("tangential_damping", 0.369))
    paddle_contact = ball_cfg.get("contact", {}).get("paddle", {})
    args._paddle_restitution = float(paddle_contact.get("restitution", 0.654))
    args._paddle_tangential_damping = float(paddle_contact.get("tangential_damping", 0.52))
    physics = BallPhysics.from_config(ball_cfg)
    table = TableGeometry.from_config(ball_cfg)
    accumulator = SuccessRate()
    serve_manifest = _load_serve_manifest(args.serve_manifest)

    policy = OnnxPolicy(onnx_path)
    scene = PingPongRealPhysicsScene(
        robot_xml,
        ball_cfg,
        JOINT_NAMES,
        control_dt=runtime_cfg.control_dt,
        near_edge_x=args.near_edge_x,
        launch_viewer=args.view,
    )
    recorder = (
        _MujocoVideoRecorder(
            scene,
            args.record_video,
            width=args.video_width,
            height=args.video_height,
            fps=args.video_fps,
        )
        if args.record_video
        else None
    )
    real_planner = None
    if args.planner_mode == "real-hope-planner":
        real_planner = _RealHopePlannerBridge(repo_root, scene, args, RacketCommand, FOREHAND, BACKHAND)
        # Sample/eval serves at the same fixed table-frame strike plane the real planner uses.  The
        # manifest still supplies side/y/z boxes and dynamic-station metadata.
        args._force_serve_strike_plane_x = True
        args._serve_strike_plane_x = float(real_planner.x_hit_table)

    default_q = runtime_cfg.action_adapter.default_q.copy()
    kp = runtime_cfg.sim_kp.copy() * float(args.kp_scale)
    kd = runtime_cfg.sim_kd.copy() * float(args.kd_scale)
    head_idx = list(HEAD_INDICES)
    net_clear_z = table.net_height + physics.ball_radius
    contact_radius = args.contact_radius
    dt = runtime_cfg.control_dt
    max_ticks = max(1, int(round(args.max_trial_seconds / dt)))
    rng = np.random.default_rng(args.seed)
    continuous = args.eval_mode == "continuous"
    counts = _new_eval_counts()
    by_side = {"forehand": _new_eval_counts(), "backhand": _new_eval_counts()}
    trace_fh = None
    trace_writer = None
    if args.trace_csv:
        trace_path = pathlib.Path(args.trace_csv)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fh = trace_path.open("w", encoding="utf-8", newline="")
        trace_fields = [
            "trial",
            "tick",
            "side",
            "phase",
            "base_x",
            "base_y",
            "base_z",
            "ball_x",
            "ball_y",
            "ball_z",
            "racket_x",
            "racket_y",
            "racket_z",
            "ball_racket_distance",
            "target_x",
            "target_y",
            "target_z",
            "time_to_strike",
            "incoming_bounce",
            "contacted",
            "real_contact",
            "proximity_contact",
            "net_clear",
            "first_bounce",
            "fallen",
            "ncon",
            "floor_contact_count",
            "floor_contact_min_dist",
            "max_abs_raw_action",
            "max_abs_applied_action",
            "max_abs_q_des",
        ]
        trace_writer = csv.DictWriter(trace_fh, fieldnames=trace_fields)
        trace_writer.writeheader()
    contact_diag_fh = None
    contact_diag_writer = None
    if args.contact_diag_csv:
        contact_diag_path = pathlib.Path(args.contact_diag_csv)
        contact_diag_path.parent.mkdir(parents=True, exist_ok=True)
        contact_diag_fh = contact_diag_path.open("w", encoding="utf-8", newline="")
        contact_diag_writer = csv.DictWriter(contact_diag_fh, fieldnames=_contact_diag_fields())
        contact_diag_writer.writeheader()

    def _floor_contact_summary():
        vals = []
        for ci in range(scene.data.ncon):
            con = scene.data.contact[ci]
            g1 = scene._mj.mj_id2name(scene.model, scene._mj.mjtObj.mjOBJ_GEOM, int(con.geom1)) or ""
            g2 = scene._mj.mj_id2name(scene.model, scene._mj.mjtObj.mjOBJ_GEOM, int(con.geom2)) or ""
            if g1 == "floor" or g2 == "floor":
                vals.append(float(con.dist))
        return len(vals), (min(vals) if vals else None)

    def _record_contact_diag(
        row: dict,
        *,
        kind: str,
        tick: int,
        phase: str,
        events,
        cmd,
        ball_pre_pos,
        ball_pre_vel,
        racket_pre_pos,
        racket_pre_vel,
        racket_post_pos,
        racket_post_vel,
        racket_geom_xmat,
        ball_post_pos,
        ball_post_vel,
        base_at_contact,
        diag: dict,
    ) -> None:
        real = kind == "real"
        contact_ball_pre_pos = events.contact_ball_pos_pre_w if real and events.contact_ball_pos_pre_w is not None else ball_pre_pos
        contact_ball_pre_vel = events.contact_ball_vel_pre_w if real and events.contact_ball_vel_pre_w is not None else ball_pre_vel
        contact_ball_post_pos = ball_post_pos
        contact_ball_post_vel = ball_post_vel
        contact_pos = events.contact_pos_w if real and events.contact_pos_w is not None else contact_ball_post_pos

        to_ball = np.asarray(contact_ball_pre_pos, dtype=np.float64) - np.asarray(racket_post_pos, dtype=np.float64)
        racket_normal = _choose_racket_normal(np.asarray(racket_geom_xmat, dtype=np.float64), to_ball)
        mujoco_normal = events.contact_normal_w if real and events.contact_normal_w is not None else None
        if mujoco_normal is not None:
            mujoco_normal = np.asarray(mujoco_normal, dtype=np.float64).copy()
            to_ball_u = _unit_or_none(to_ball)
            if to_ball_u is not None and float(np.dot(mujoco_normal, to_ball_u)) < 0.0:
                mujoco_normal = -mujoco_normal

        target_vel = np.asarray(cmd.velocity, dtype=np.float64)
        target_outgoing_vel = np.asarray(getattr(cmd, "outgoing_velocity", target_vel), dtype=np.float64)
        target_normal = getattr(cmd, "target_normal", None)
        if target_normal is not None:
            target_normal = np.asarray(target_normal, dtype=np.float64)
        racket_pre_vel = np.asarray(racket_pre_vel, dtype=np.float64)
        racket_post_vel = np.asarray(racket_post_vel, dtype=np.float64)
        contact_ball_pre_vel = np.asarray(contact_ball_pre_vel, dtype=np.float64)
        contact_ball_post_vel = np.asarray(contact_ball_post_vel, dtype=np.float64)
        rel_pre = contact_ball_pre_vel - racket_pre_vel
        rel_post = contact_ball_post_vel - racket_post_vel
        prediction = _outgoing_flight_diag(scene.to_table(contact_ball_post_pos), contact_ball_post_vel, physics, table)

        row["contact_kind"] = kind
        row["contact_tick"] = int(tick)
        row["contact_substep"] = "" if events.contact_substep is None else int(events.contact_substep)
        row["contact_time_offset_s"] = "" if events.contact_time_offset_s is None else float(events.contact_time_offset_s)
        row["contact_dist"] = "" if events.contact_dist is None else float(events.contact_dist)
        row["phase_at_contact"] = phase
        row["base_z_at_contact"] = float(np.asarray(base_at_contact, dtype=np.float64)[2])
        row["time_to_strike_cmd"] = float(cmd.time_to_strike)
        row["racket_target_speed"] = _norm(target_vel)
        row["racket_speed_pre"] = _norm(racket_pre_vel)
        row["racket_speed_post"] = _norm(racket_post_vel)
        row["ball_speed_pre"] = _norm(contact_ball_pre_vel)
        row["ball_speed_post"] = _norm(contact_ball_post_vel)
        row["ball_speed_delta"] = _norm(contact_ball_post_vel) - _norm(contact_ball_pre_vel)
        row["racket_vel_target_error"] = _norm(racket_post_vel - target_vel)
        angle = _angle_deg(racket_post_vel, target_vel)
        row["racket_vel_target_angle_deg"] = "" if angle is None else angle
        row["ball_post_vs_target_vel_error"] = _norm(contact_ball_post_vel - target_outgoing_vel)
        ball_angle = _angle_deg(contact_ball_post_vel, target_outgoing_vel)
        row["ball_post_vs_target_vel_angle_deg"] = "" if ball_angle is None else ball_angle
        row["ball_post_vs_target_outgoing_vel_error"] = _norm(contact_ball_post_vel - target_outgoing_vel)
        outgoing_angle = _angle_deg(contact_ball_post_vel, target_outgoing_vel)
        row["ball_post_vs_target_outgoing_vel_angle_deg"] = "" if outgoing_angle is None else outgoing_angle
        row["closing_speed_pre"] = float(-np.dot(rel_pre, racket_normal))
        row["outgoing_normal_speed_post"] = float(np.dot(rel_post, racket_normal))
        row["raw_action_max_at_contact"] = float(diag["max_abs_raw_action"])
        row["applied_action_max_at_contact"] = float(diag["max_abs_applied_action"])
        row["q_des_max_at_contact"] = float(diag["max_abs_q_des"])
        row.update(prediction)

        for prefix, value in (
            ("contact_pos", contact_pos),
            ("mujoco_contact_normal", mujoco_normal),
            ("racket_normal", racket_normal),
            ("target_pos", cmd.position),
            ("target_vel", target_vel),
            ("target_outgoing_vel", target_outgoing_vel),
            ("target_normal", target_normal),
            ("ball_pre_pos", contact_ball_pre_pos),
            ("ball_pre_vel", contact_ball_pre_vel),
            ("ball_post_pos", contact_ball_post_pos),
            ("ball_post_vel", contact_ball_post_vel),
            ("racket_pre_pos", racket_pre_pos),
            ("racket_pre_vel", racket_pre_vel),
            ("racket_post_pos", racket_post_pos),
            ("racket_post_vel", racket_post_vel),
        ):
            _xyz(row, prefix, value)

    def _policy_tick(lifecycle, source, last_action, fixed_station_xy, manifest_row=None):
        """One 50 Hz policy step (identical to the deploy runner's tick).

        Returns (events, applied_action) — the caller keeps ``applied_action`` as
        the next tick's ``last_action``.
        """
        state = scene.read_robot_state()
        target = lifecycle.update(source.poll(), state)
        station_xy = _station_xy_for_observation(fixed_station_xy, target, lifecycle.phase, manifest_row, args)
        if getattr(policy, "obs_dim", 111) == OBS_DIM_NORMAL114:
            obs = build_observation_normal114(state, target, last_action, default_q, station_xy)
        else:
            obs = build_observation(state, target, last_action, default_q, station_xy)
        policy_obs = _obs_for_policy_joint_order(obs, last_action, args.policy_joint_order)
        raw_policy_action = policy.infer(policy_obs)
        raw_action = _raw_action_to_canonical(raw_policy_action, args.policy_joint_order)
        # Applied action = raw with the passive head columns zeroed, matching both
        # the deploy runner and training's zeroed last_action feedback.
        applied_action = np.asarray(raw_action, dtype=np.float64).copy()
        if runtime_cfg.passive_neck:
            applied_action[head_idx] = 0.0
        q_des = runtime_cfg.action_adapter.decode(applied_action)
        if runtime_cfg.passive_neck:
            q_des[head_idx] = default_q[head_idx]
        scene.write_targets(q_des, kp, kd)
        diag = {
            "finite": bool(np.all(np.isfinite(raw_policy_action)) and np.all(np.isfinite(raw_action)) and np.all(np.isfinite(applied_action)) and np.all(np.isfinite(q_des))),
            "max_abs_raw_action": float(np.nanmax(np.abs(raw_policy_action))),
            "max_abs_applied_action": float(np.nanmax(np.abs(applied_action))),
            "max_abs_q_des": float(np.nanmax(np.abs(q_des))),
        }
        return scene.step(), applied_action, diag

    def _park_ball():
        """Drop the ball out of play (past the far edge) between rally serves."""
        scene.set_ball(
            [scene.near_edge_x + scene.length + 1.0, 0.0, scene.table_height + 0.5],
            [0.0, 0.0, 0.0],
        )

    # Continuous mode: ONE initialization for the whole session — robot state,
    # last_action, lifecycle and the fixed station all persist across serves
    # (matching the deploy runner). Independent mode re-creates them per serve.
    scene.reset_stand()
    lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
    source = QueueRacketCommandSource()
    last_action = np.zeros(31, dtype=np.float64)
    fixed_station_xy = scene.base_pos_w()[:2].copy()
    max_rest_ticks = max(1, int(round(args.max_rest_seconds / dt)))
    transitions_seen: set = set()
    prev_side = None
    task_counter = 0

    for trial in range(args.num_serves):
        # Side pattern FH, FH, BH, BH, ... exercises all four adjacent side
        # transitions (FH->FH, FH->BH, BH->BH, BH->FH) across the session.
        if args.side_mode == "forehand":
            side = FOREHAND
        elif args.side_mode == "backhand":
            side = BACKHAND
        else:
            side = FOREHAND if (trial % 4) < 2 else BACKHAND
        if prev_side is not None:
            transitions_seen.add((prev_side, side))
        prev_side = side
        strike_pt, serve_pos, serve_vel, serve_row = _sample_serve(rng, side, scene, physics, args, serve_manifest)
        if real_planner is not None:
            real_planner.reset_for_new_ball()

        if not continuous:
            # Independent-strike evaluation: fresh robot + policy state per serve.
            scene.reset_stand()
            lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
            source = QueueRacketCommandSource()
            last_action = np.zeros(31, dtype=np.float64)
            fixed_station_xy = scene.base_pos_w()[:2].copy()
        elif trial > 0:
            # Continuous rally: keep the policy running (no reset, no teleport)
            # through follow-through/recovery until it is ready for the next ball.
            _park_ball()
            for _ in range(max_rest_ticks):
                _events, last_action, _diag = _policy_tick(
                    lifecycle, source, last_action, fixed_station_xy
                )
                if recorder is not None and trial < args.record_serves:
                    recorder.capture(scene)
                if lifecycle.phase.value == "ready":
                    break

        scene.set_ball(serve_pos, serve_vel)
        if recorder is not None and trial < args.record_serves:
            recorder.capture(scene)
        task_counter += 1
        task_id = task_counter
        revision = 0
        contacted = False
        real_contacted = False
        proximity_contacted = False
        net_clear = False
        first_bounce = None
        incoming_bounced = args.incoming_trajectory != "one-bounce"
        fell = False
        timed_out = True
        min_distance = float("inf")
        trial_nonfinite_ticks = 0
        trial_max_raw = 0.0
        trial_max_applied = 0.0
        trial_max_q_des = 0.0
        side_name = "forehand" if side >= 0 else "backhand"
        contact_diag_row = _new_contact_diag_row(trial, side_name, strike_pt, serve_pos, serve_vel)

        for _tick in range(max_ticks):
            ball_pos, ball_vel = scene.ball_state()
            if real_planner is not None:
                cmd = real_planner.update(ball_pos, float(scene.data.time))
                if cmd is not None:
                    cmd = _perturb_command(cmd, args, rng, RacketCommand)
                    source.submit(cmd)
                else:
                    cmd = source.poll()
            else:
                cmd = _predict_command(
                    ball_pos,
                    ball_vel,
                    side,
                    scene,
                    physics,
                    args,
                    task_id,
                    revision,
                    RacketCommand,
                    strike_x_w=float(strike_pt[0]) if serve_manifest else None,
                )
                cmd = _perturb_command(cmd, args, rng, RacketCommand)
                revision += 1
                source.submit(cmd)

            racket_pre_pos, racket_pre_vel, _racket_site_xmat_pre = scene.racket_site_pose()
            events, last_action, diag = _policy_tick(lifecycle, source, last_action, fixed_station_xy, serve_row)
            if recorder is not None and trial < args.record_serves:
                recorder.capture(scene)
            if not diag["finite"]:
                trial_nonfinite_ticks += 1
            trial_max_raw = max(trial_max_raw, diag["max_abs_raw_action"])
            trial_max_applied = max(trial_max_applied, diag["max_abs_applied_action"])
            trial_max_q_des = max(trial_max_q_des, diag["max_abs_q_des"])

            r_pos, r_vel, _racket_site_xmat = scene.racket_site_pose()
            _racket_geom_pos, racket_geom_xmat = scene.racket_geom_pose()
            b_pos, b_vel = scene.ball_state()
            delta = b_pos - r_pos
            distance = float(np.linalg.norm(delta))
            min_distance = min(min_distance, distance)

            if args.incoming_trajectory == "one-bounce" and not contacted and not incoming_bounced:
                for x_t, y_t, z_sign in events.surface_crossings:
                    if z_sign < 0.0 and (0.0 <= x_t <= scene.net_x_table) and (-scene.width <= y_t <= 0.0):
                        incoming_bounced = True
                        break

            # --- contact: real MuJoCo ball<->racket contact, or the allowed proximity
            #     fallback (ball within contact_radius of the racket site with the
            #     racket moving toward it). ---
            new_contact_kind = None
            if events.ball_racket_contact:
                real_contacted = True
                if not contacted:
                    contacted = True
                new_contact_kind = "real"
            elif not contacted and np.linalg.norm(delta) <= contact_radius and float(np.dot(r_vel, delta)) > 0.0:
                contacted = True
                proximity_contacted = True
                new_contact_kind = "proximity"
            if new_contact_kind == "real" or (
                new_contact_kind == "proximity" and contact_diag_row["contact_kind"] == "none"
            ):
                contact_cmd = cmd if cmd is not None else source.poll()
                if contact_cmd is not None:
                    _record_contact_diag(
                        contact_diag_row,
                        kind=new_contact_kind,
                        tick=_tick,
                        phase=lifecycle.phase.value,
                        events=events,
                        cmd=contact_cmd,
                        ball_pre_pos=ball_pos,
                        ball_pre_vel=ball_vel,
                        racket_pre_pos=racket_pre_pos,
                        racket_pre_vel=racket_pre_vel,
                        racket_post_pos=r_pos,
                        racket_post_vel=r_vel,
                        racket_geom_xmat=racket_geom_xmat,
                        ball_post_pos=b_pos,
                        ball_post_vel=b_vel,
                        base_at_contact=scene.base_pos_w(),
                        diag=diag,
                    )

            # --- after contact, watch the REAL outgoing ball for net clearance and its
            #     first bounce (both read off the simulated trajectory). ---
            if contacted and first_bounce is None:
                if not net_clear:
                    for z_cross, x_sign in events.net_crossings:
                        if x_sign > 0.0 and z_cross > net_clear_z:
                            net_clear = True
                            break
                for x_t, y_t, z_sign in events.surface_crossings:
                    if z_sign < 0.0:
                        first_bounce = (x_t, y_t)
                        break

            if first_bounce is not None:
                timed_out = False
                break
            if scene.base_fallen():
                fell = True
            if trace_writer is not None and trial < args.trace_serves:
                floor_count, floor_min = _floor_contact_summary()
                base = scene.base_pos_w()
                trace_cmd = cmd if cmd is not None else source.poll()
                trace_writer.writerow(
                    {
                        "trial": trial,
                        "tick": _tick,
                        "side": side_name,
                        "phase": lifecycle.phase.value,
                        "base_x": float(base[0]),
                        "base_y": float(base[1]),
                        "base_z": float(base[2]),
                        "ball_x": float(b_pos[0]),
                        "ball_y": float(b_pos[1]),
                        "ball_z": float(b_pos[2]),
                        "racket_x": float(r_pos[0]),
                        "racket_y": float(r_pos[1]),
                        "racket_z": float(r_pos[2]),
                        "ball_racket_distance": float(distance),
                        "target_x": "" if trace_cmd is None else float(trace_cmd.position[0]),
                        "target_y": "" if trace_cmd is None else float(trace_cmd.position[1]),
                        "target_z": "" if trace_cmd is None else float(trace_cmd.position[2]),
                        "time_to_strike": "" if trace_cmd is None else float(trace_cmd.time_to_strike),
                        "incoming_bounce": int(incoming_bounced),
                        "contacted": int(contacted),
                        "real_contact": int(real_contacted),
                        "proximity_contact": int(proximity_contacted),
                        "net_clear": int(net_clear),
                        "first_bounce": int(first_bounce is not None),
                        "fallen": int(fell),
                        "ncon": int(scene.data.ncon),
                        "floor_contact_count": int(floor_count),
                        "floor_contact_min_dist": "" if floor_min is None else float(floor_min),
                        "max_abs_raw_action": float(diag["max_abs_raw_action"]),
                        "max_abs_applied_action": float(diag["max_abs_applied_action"]),
                        "max_abs_q_des": float(diag["max_abs_q_des"]),
                    }
                )
            # Incoming ball that flew past the robot without ever being contacted is a
            # definite miss (it can no longer produce an opponent-half return): stop the
            # trial early instead of waiting out the timeout.
            if not contacted and scene.ball_state()[0][0] < scene.near_edge_x - 0.8:
                timed_out = False
                break

        opponent = bool(first_bounce is not None and table.on_opponent_half(first_bounce[0], first_bounce[1]))
        success = bool(
            incoming_bounced
            and contacted
            and net_clear
            and first_bounce is not None
            and opponent
        )
        if success:
            miss_reason = "success"
        elif not incoming_bounced:
            miss_reason = "no_incoming_bounce"
        elif not contacted:
            miss_reason = "no_contact"
        elif not net_clear:
            miss_reason = "contact_no_net"
        elif first_bounce is None:
            miss_reason = "net_no_bounce"
        elif not opponent:
            miss_reason = "wrong_bounce"
        else:
            miss_reason = "unknown"
        if fell and not success:
            miss_reason = f"{miss_reason}+fell"
        if timed_out and not success:
            miss_reason = f"{miss_reason}+timeout"
        contact_diag_row["success"] = int(success)
        contact_diag_row["miss_reason"] = miss_reason
        contact_diag_row["net_clear"] = int(net_clear)
        contact_diag_row["first_bounce"] = int(first_bounce is not None)
        contact_diag_row["first_bounce_x"] = "" if first_bounce is None else float(first_bounce[0])
        contact_diag_row["first_bounce_y"] = "" if first_bounce is None else float(first_bounce[1])
        contact_diag_row["opponent_bounce"] = int(opponent)
        contact_diag_row["fallen"] = int(fell)
        contact_diag_row["timed_out"] = int(timed_out)
        contact_diag_row["min_ball_racket_distance"] = float(min_distance)
        contact_diag_row["base_z_end"] = float(scene.base_pos_w()[2])
        if contact_diag_writer is not None:
            contact_diag_writer.writerow(contact_diag_row)
        accumulator.add_bool(success)
        for bucket in (counts, by_side[side_name]):
            bucket["attempts"] += 1
            bucket["successes"] += int(success)
            bucket["incoming_bounce"] += int(incoming_bounced)
            bucket["contact"] += int(contacted)
            bucket["real_contact"] += int(real_contacted)
            bucket["proximity_contact"] += int(proximity_contacted)
            bucket["net_clear"] += int(net_clear)
            bucket["first_bounce"] += int(first_bounce is not None)
            bucket["opponent_bounce"] += int(opponent)
            bucket["nonfinite_action_ticks"] += int(trial_nonfinite_ticks)
            if not incoming_bounced:
                bucket["miss_no_incoming_bounce"] += 1
            elif not contacted:
                bucket["miss_no_contact"] += 1
            elif not net_clear:
                bucket["miss_contact_no_net"] += 1
            elif first_bounce is None:
                bucket["miss_net_no_bounce"] += 1
            elif not opponent:
                bucket["miss_wrong_bounce"] += 1
            if fell:
                bucket["miss_fell"] += 1
            if timed_out:
                bucket["miss_timeout"] += 1
            bucket["min_ball_racket_distance"] = min(bucket["min_ball_racket_distance"], min_distance)
            _update_max(bucket, "max_abs_raw_action", trial_max_raw)
            _update_max(bucket, "max_abs_applied_action", trial_max_applied)
            _update_max(bucket, "max_abs_q_des", trial_max_q_des)

    if recorder is not None:
        recorder.close()
    if trace_fh is not None:
        trace_fh.close()
    if contact_diag_fh is not None:
        contact_diag_fh.close()
    scene.close()
    if continuous:
        _names = {FOREHAND: "FH", BACKHAND: "BH"}
        seen = sorted(f"{_names[a]}->{_names[b]}" for a, b in transitions_seen)
        all_four = len(transitions_seen) == 4
        print(
            f"[mujoco_eval] adjacent side transitions exercised: {', '.join(seen) or 'none'}"
            + ("" if all_four else "  (increase --num-serves >= 5 to cover all four)"),
            file=sys.stderr,
        )
    print(
        f"[mujoco_eval] mode={args.eval_mode} serves={accumulator.attempts} "
        f"returns={accumulator.successes} success_rate={accumulator.value:.4f} "
        f"incoming_bounce_rate={counts['incoming_bounce'] / max(1, counts['attempts']):.4f} "
        f"contact_rate={counts['contact'] / max(1, counts['attempts']):.4f} "
        f"net_clear_rate={counts['net_clear'] / max(1, counts['attempts']):.4f} "
        f"opponent_bounce_rate={counts['opponent_bounce'] / max(1, counts['attempts']):.4f}",
        file=sys.stderr,
    )
    result = accumulator.as_dict()
    if args.detailed:
        result.update(
            {
                "counts": _finalize_counts(counts),
                "by_side": {name: _finalize_counts(v) for name, v in by_side.items()},
                "eval_mode": args.eval_mode,
                "side_mode": args.side_mode,
                "incoming_trajectory": args.incoming_trajectory,
                "planner_mode": args.planner_mode,
                "real_planner": (
                    {
                        "yaml": str(real_planner.yaml_path),
                        "physics_path": real_planner.physics_path,
                        "x_hit_table": float(real_planner.x_hit_table),
                        "mocap_hz": float(real_planner.mocap_hz),
                        "solve_period_s": float(real_planner.solve_period),
                        "solve_calls": int(real_planner.solve_calls),
                        "command_count": int(real_planner.command_count),
                        "no_command_count": int(real_planner.no_command_count),
                    }
                    if real_planner is not None
                    else None
                ),
                "planner_perturbation": {
                    "target_pos_offset": [float(v) for v in args.planner_target_pos_offset],
                    "target_pos_noise_std": [float(v) for v in args.planner_target_pos_noise_std],
                    "tts_offset": float(args.planner_tts_offset),
                    "tts_noise_std": float(args.planner_tts_noise_std),
                    "target_vel_scale": float(args.planner_target_vel_scale),
                    "target_vel_offset": [float(v) for v in args.planner_target_vel_offset],
                    "target_vel_noise_std": [float(v) for v in args.planner_target_vel_noise_std],
                    "target_vel_yaw_deg": float(args.planner_target_vel_yaw_deg),
                    "min_racket_speed": float(args.planner_min_racket_speed),
                    "max_racket_speed": float(args.planner_max_racket_speed),
                },
                "policy_joint_order": args.policy_joint_order,
                "policy_obs_dim": int(getattr(policy, "obs_dim", 111)),
                "kp_scale": float(args.kp_scale),
                "kd_scale": float(args.kd_scale),
                "lifecycle_recovery_blend_seconds": float(args.lifecycle_recovery_blend_seconds),
                "lifecycle_recovery_blend_velocity": bool(args.lifecycle_recovery_blend_velocity),
                "serve_manifest": str(args.serve_manifest) if args.serve_manifest else None,
                "contact_diag_csv": str(args.contact_diag_csv) if args.contact_diag_csv else None,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", default=None, help="Exported hope_pingpong.onnx (default: from runtime config).")
    parser.add_argument("--model-xml", default=None, help="a3_pingpong MJCF (default: from runtime config).")
    parser.add_argument("--runtime-config", default=None, help="hope_pingpong_runtime.yaml (default: shipped).")
    parser.add_argument("--reference-dir", default=None, help="Dir containing the reference deploy package.")
    parser.add_argument("--num-serves", type=int, default=50, help="Number of served balls (denominator).")
    parser.add_argument(
        "--side-mode",
        choices=["mixed", "forehand", "backhand"],
        default="mixed",
        help="Serve side schedule. mixed keeps FH,FH,BH,BH; forehand/backhand force one side.",
    )
    parser.add_argument(
        "--serve-manifest",
        default=None,
        help="Optional TSV motion manifest; sample strike points from its racket_pos target boxes.",
    )
    parser.add_argument(
        "--incoming-trajectory",
        choices=["direct", "one-bounce"],
        default="direct",
        help="direct preserves the old sampled flight-to-strike eval; one-bounce serves a ball that "
             "must bounce on the robot half before the strike.",
    )
    parser.add_argument(
        "--planner-mode",
        choices=["no-bounce", "bounce-aware", "real-hope-planner"],
        default="no-bounce",
        help="Planner used to turn the live ball state into RacketCommand targets. "
             "real-hope-planner calls hope_ws/src/hope_planner with table-frame mocap emulation.",
    )
    parser.add_argument(
        "--real-planner-yaml",
        default=None,
        help="Planner YAML for --planner-mode real-hope-planner "
             "(default: hope_ws/src/hope_planner/config/hope_planner.yaml).",
    )
    parser.add_argument(
        "--real-planner-physics-path",
        default=None,
        help="Optional ball_physics.yaml path passed to the real planner loaders.",
    )
    parser.add_argument(
        "--real-planner-x-hit",
        type=float,
        default=None,
        help="Override the real planner fixed strike plane in table-frame x metres.",
    )
    parser.add_argument(
        "--real-planner-mocap-hz",
        type=float,
        default=300.0,
        help="Mocap sample rate emulated from the 50 Hz MuJoCo ball state before calling the planner.",
    )
    parser.add_argument(
        "--real-planner-solve-period",
        type=float,
        default=None,
        help="Override planner solve_period_s. None uses the YAML value.",
    )
    parser.add_argument(
        "--real-planner-drag-k",
        type=float,
        default=None,
        help="Override planner drag_k. None uses the YAML value; negative disables override.",
    )
    parser.add_argument(
        "--real-planner-table-c-h",
        type=float,
        default=None,
        help="Override planner table_c_h. None uses the YAML value; negative disables override.",
    )
    parser.add_argument(
        "--real-planner-table-c-v",
        type=float,
        default=None,
        help="Override planner table_c_v. None uses the YAML value; negative disables override.",
    )
    parser.add_argument(
        "--planner-target-pos-offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Add a fixed offset to the planner command target position in world metres.",
    )
    parser.add_argument(
        "--planner-target-pos-noise-std",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("SX", "SY", "SZ"),
        help="Per-command Gaussian noise std for target position in metres.",
    )
    parser.add_argument(
        "--planner-tts-offset",
        type=float,
        default=0.0,
        help="Add a fixed time-to-strike offset in seconds; negative values emulate late commands.",
    )
    parser.add_argument(
        "--planner-tts-noise-std",
        type=float,
        default=0.0,
        help="Per-command Gaussian noise std for time-to-strike in seconds.",
    )
    parser.add_argument(
        "--planner-target-vel-scale",
        type=float,
        default=1.0,
        help="Scale the planner command target velocity vector.",
    )
    parser.add_argument(
        "--planner-target-vel-offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("DVX", "DVY", "DVZ"),
        help="Add a fixed offset to the planner command target velocity in m/s.",
    )
    parser.add_argument(
        "--planner-target-vel-noise-std",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("SVX", "SVY", "SVZ"),
        help="Per-command Gaussian noise std for target velocity in m/s.",
    )
    parser.add_argument(
        "--planner-target-vel-yaw-deg",
        type=float,
        default=0.0,
        help="Rotate the planner command target velocity around world z by this many degrees.",
    )
    parser.add_argument(
        "--planner-min-racket-speed",
        type=float,
        default=0.3,
        help="Lower clamp for impact-inverted planner racket speed in m/s.",
    )
    parser.add_argument(
        "--planner-max-racket-speed",
        type=float,
        default=2.8,
        help="Upper clamp for impact-inverted planner racket speed in m/s.",
    )
    parser.add_argument(
        "--bounce-pre-time",
        nargs=2,
        type=float,
        default=(0.42, 0.65),
        metavar=("LO", "HI"),
        help="one-bounce serve: flight time from opponent origin to the robot-half bounce.",
    )
    parser.add_argument(
        "--bounce-post-time",
        nargs=2,
        type=float,
        default=(0.28, 0.44),
        metavar=("LO", "HI"),
        help="one-bounce serve: approximate time from robot-half bounce to strike point.",
    )
    parser.add_argument(
        "--station-mode",
        choices=["auto", "fixed", "dynamic-from-manifest"],
        default="auto",
        help="Observation station source. auto uses dynamic station only when --serve-manifest rows contain "
             "motion_racket_offset_x/y; fixed preserves the legacy fixed-station eval.",
    )
    parser.add_argument(
        "--dynamic-station-clip-x",
        nargs=2,
        type=float,
        default=(-0.18, 0.18),
        metavar=("LO", "HI"),
        help="Dynamic station x clip relative to fixed station, matching training.",
    )
    parser.add_argument(
        "--dynamic-station-clip-y",
        nargs=2,
        type=float,
        default=(-0.22, 0.22),
        metavar=("LO", "HI"),
        help="Dynamic station y clip relative to fixed station, matching training.",
    )
    parser.add_argument(
        "--dynamic-station-blend",
        type=float,
        default=1.0,
        help="Blend from fixed station to manifest-derived dynamic station, matching training.",
    )
    parser.add_argument(
        "--dynamic-station-post-window",
        type=float,
        default=0.12,
        help="Keep dynamic station for this many seconds after contact, matching strike_window_s.",
    )
    parser.add_argument("--detailed", action="store_true", help="Emit detailed stage rates and diagnostics.")
    parser.add_argument(
        "--policy-joint-order",
        choices=["canonical", "isaac-articulation-action", "isaac-articulation-obs-action"],
        default="canonical",
        help="ONNX policy column semantics for MuJoCo diagnostics. canonical is the public contract; "
             "isaac-articulation-action treats only raw_action as the old Isaac articulation order; "
             "isaac-articulation-obs-action also reorders the 31-D joint_pos/joint_vel/last_action obs slices.",
    )
    parser.add_argument(
        "--eval-mode", choices=["continuous", "independent"], default="continuous",
        help="continuous (default): one uninterrupted rally session — robot/policy state persist "
             "across serves and all four adjacent side transitions are exercised. "
             "independent: reset the robot per serve (isolated-swing evaluation only).",
    )
    parser.add_argument(
        "--max-rest-seconds", type=float, default=2.0,
        help="continuous mode: max seconds the policy gets between serves to finish "
             "follow-through/recovery before the next ball is served.",
    )
    parser.add_argument(
        "--max-trial-seconds", type=float, default=3.0, help="Max simulated seconds per serve before scoring."
    )
    parser.add_argument("--contact-radius", type=float, default=0.10, help="Racket-site proximity contact fallback (m).")
    parser.add_argument("--kp-scale", type=float, default=1.0, help="Scale MuJoCo bridge PD stiffness gains.")
    parser.add_argument("--kd-scale", type=float, default=1.0, help="Scale MuJoCo bridge PD damping gains.")
    parser.add_argument(
        "--lifecycle-recovery-blend-seconds",
        type=float,
        default=0.0,
        help="Optional deploy/eval experiment: blend the recovery observation target from the last "
             "strike target into the ready target for this many seconds. Default 0 keeps the "
             "original abrupt recovery target switch.",
    )
    parser.add_argument(
        "--lifecycle-recovery-blend-velocity",
        action="store_true",
        help="With --lifecycle-recovery-blend-seconds, also decay the last strike target velocity "
             "during recovery. By default only target position is blended and recovery velocity is zero.",
    )
    parser.add_argument(
        "--near-edge-x", type=float, default=0.30,
        help="MuJoCo x of the table's near edge (sets the robot-to-table placement).",
    )
    parser.add_argument(
        "--strike-plane-x", type=float, default=0.0,
        help="Strike plane, table-frame x (0 = the near table edge); the fixed intercept plane.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the example serve distribution.")
    parser.add_argument("--view", action="store_true", help="Launch the MuJoCo passive viewer (debug only).")
    parser.add_argument("--record-video", default=None, help="Write an MP4 of the first eval serves.")
    parser.add_argument("--record-serves", type=int, default=5, help="Number of initial serves to record.")
    parser.add_argument("--trace-csv", default=None, help="Write per-tick MuJoCo eval diagnostics to CSV.")
    parser.add_argument("--trace-serves", type=int, default=5, help="Number of initial serves to trace.")
    parser.add_argument(
        "--contact-diag-csv",
        default=None,
        help="Write one row per serve with contact-instant ball/racket/outgoing-flight diagnostics.",
    )
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--mujoco-gl", default="egl", help="MUJOCO_GL value used when recording video.")
    parser.add_argument("--json-out", default=None, help="Also write {'success_rate': ...} to this file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_eval(args)
    print(json.dumps(result))
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
