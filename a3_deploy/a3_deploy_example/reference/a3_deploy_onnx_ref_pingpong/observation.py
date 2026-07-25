# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Assemble the actor observation.

The baseline 111-D layout is fixed and must byte-for-byte match the training/export contract.
The experimental 114-D layout appends the planner racket target normal. No normalization is applied
(raw observation).

| slice      | term                    | dim | frame / units          |
|------------|-------------------------|-----|------------------------|
| [0:3]      | base_ang_vel            | 3   | pelvis body, rad/s     |
| [3:34]     | joint_pos               | 31  | rad, q - default_q     |
| [34:65]    | joint_vel               | 31  | rad/s                  |
| [65:96]    | last_action             | 31  | applied (prev tick)(*) |
| [96:99]    | projected_gravity       | 3   | base frame, unit       |
| [99:101]   | base_forward_xy         | 2   | world xy, unit         |
| [101:103]  | fixed_station_error_xy  | 2   | world xy, m            |
| [103:106]  | racket_target_rel_base  | 3   | world, m               |
| [106:109]  | racket_target_vel_w     | 3   | world, m/s             |
| [109:110]  | time_to_strike          | 1   | s                      |
| [110:111]  | swing_side              | 1   | +1 forehand / -1 back  |
| [111:114]  | racket_target_normal_w  | 3   | world, unit            |

``fixed_station_error_xy`` keeps its historical name for the 111-D contract, but
the value is current_station_xy - current base xy. For fixed-station policies the
current station is the startup ready station; for dynamic-station policies the
runner/planner may supply a per-swing station before impact.

(*) ``last_action`` is the previous tick's APPLIED action: the actor's raw output
with the passive head columns (idx 3, 4) zeroed, exactly as training zeroes them
in its applied-action feedback. Those two columns are therefore always 0 here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quaternion as quat
from .joint_order import NUM_JOINTS

OBS_DIM: int = 111
OBS_DIM_NORMAL114: int = 114
_ACTION_DIM: int = NUM_JOINTS


@dataclass
class RobotState:
    """Proprioceptive robot state consumed by the observation builder.

    All world poses are expressed in the same frame the planner uses for the
    racket target. On hardware this is the operator/table frame; in the reference
    MuJoCo sim it is the simulator world frame.
    """

    base_pos_w: np.ndarray      # (3,) pelvis position, world m
    base_quat_w: np.ndarray     # (4,) pelvis orientation, (w, x, y, z)
    base_ang_vel_b: np.ndarray  # (3,) pelvis gyro, body frame rad/s
    q: np.ndarray               # (31,) joint positions, joint order
    qd: np.ndarray              # (31,) joint velocities, joint order


@dataclass
class ObsTarget:
    """The strike goal fed into the observation for the current tick."""

    pos_w: np.ndarray   # (3,) racket target position, world m
    vel_w: np.ndarray   # (3,) racket target velocity, world m/s
    time_to_strike: float
    swing_side: float   # +1 forehand / -1 backhand
    normal_w: np.ndarray | None = None  # (3,) racket target normal, world unit vector


def build_observation(
    state: RobotState,
    target: ObsTarget,
    last_action: np.ndarray,
    default_q: np.ndarray,
    fixed_station_xy: np.ndarray,
) -> np.ndarray:
    """Return the 111-D observation as a contiguous ``float32`` vector."""
    q = np.asarray(state.q, dtype=np.float64)
    qd = np.asarray(state.qd, dtype=np.float64)
    default_q = np.asarray(default_q, dtype=np.float64)
    last_action = np.asarray(last_action, dtype=np.float64)
    base_pos = np.asarray(state.base_pos_w, dtype=np.float64)
    base_quat = quat.normalize(state.base_quat_w)

    if q.shape[0] != NUM_JOINTS or qd.shape[0] != NUM_JOINTS:
        raise ValueError(f"joint vectors must be length {NUM_JOINTS}")
    if last_action.shape[0] != _ACTION_DIM:
        raise ValueError(f"last_action must be length {_ACTION_DIM}")

    obs = np.empty(OBS_DIM, dtype=np.float64)
    o = 0
    obs[o:o + 3] = state.base_ang_vel_b;                       o += 3          # base_ang_vel
    obs[o:o + NUM_JOINTS] = q - default_q;                     o += NUM_JOINTS  # joint_pos
    obs[o:o + NUM_JOINTS] = qd;                                o += NUM_JOINTS  # joint_vel
    obs[o:o + NUM_JOINTS] = last_action;                       o += NUM_JOINTS  # last_action
    obs[o:o + 3] = quat.projected_gravity_body(base_quat);     o += 3          # projected_gravity
    obs[o:o + 2] = quat.base_forward_xy(base_quat);            o += 2          # base_forward_xy
    obs[o:o + 2] = np.asarray(fixed_station_xy)[:2] - base_pos[:2]; o += 2     # fixed_station_error_xy
    obs[o:o + 3] = np.asarray(target.pos_w) - base_pos;        o += 3          # racket_target_rel_base
    obs[o:o + 3] = np.asarray(target.vel_w);                   o += 3          # racket_target_vel_w
    obs[o] = float(target.time_to_strike);                     o += 1          # time_to_strike
    obs[o] = float(target.swing_side);                         o += 1          # swing_side
    assert o == OBS_DIM, o
    return obs.astype(np.float32)


def _target_normal_or_fallback(target: ObsTarget) -> np.ndarray:
    normal = target.normal_w
    if normal is None:
        normal = target.vel_w
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(normal))
    if n < 1.0e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return normal / n


def build_observation_normal114(
    state: RobotState,
    target: ObsTarget,
    last_action: np.ndarray,
    default_q: np.ndarray,
    fixed_station_xy: np.ndarray,
) -> np.ndarray:
    """Return the 114-D normal-visible observation as a contiguous ``float32`` vector."""
    obs111 = build_observation(state, target, last_action, default_q, fixed_station_xy)
    out = np.empty(OBS_DIM_NORMAL114, dtype=np.float32)
    out[:OBS_DIM] = obs111
    out[OBS_DIM:OBS_DIM_NORMAL114] = _target_normal_or_fallback(target).astype(np.float32)
    return out
