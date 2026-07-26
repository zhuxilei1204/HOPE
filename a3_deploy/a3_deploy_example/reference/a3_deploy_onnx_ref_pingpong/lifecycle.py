# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Continuous multi-rally swing lifecycle.

Per tick the runner advances one state machine that turns the latest
``RacketCommand`` and the robot state into the strike goal fed to the observation:

    ready -> swing -> follow-through -> recovery -> ready -> (next task_id)

Contract points enforced here:
  * A new ``task_id`` engages a swing (only from ready or recovery). ``swing_side``
    is locked for the whole task.
  * A higher ``task_revision`` under the active ``task_id`` updates the target and
    time-to-strike, but only before contact.
  * There is exactly one swing per ``task_id``.
  * Between balls the robot pose, joint state, and policy history are NOT reset --
    the lifecycle never touches them, it only chooses what goal to observe.
  * Recovery is in-place recentring only (a fixed ready reach); it is never
    locomotion or footstep planning.

The reference clock: ``time_to_strike`` is seeded from the command and counts down
by ``dt`` each tick, passing through zero at contact and continuing negative through
the follow-through (matching the training reference clock).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .observation import ObsTarget, RobotState
from .racket_command import FOREHAND, RacketCommand


class Phase(Enum):
    READY = "ready"
    SWING = "swing"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


@dataclass
class LifecycleConfig:
    dt: float = 0.02
    follow_through_s: float = 0.6   # goal held after contact so the swing completes
    recovery_s: float = 0.8         # in-place recentring window before the next ball
    ready_time_to_strike: float = 1.0
    # Base-relative ready reach (example values): the racket goal while idle sits a
    # comfortable distance in front of the robot, offset laterally by swing side.
    ready_reach_x: float = 0.40
    ready_reach_y: float = 0.20
    ready_reach_z: float = -0.05    # relative to the pelvis height
    # Optional deploy/eval experiment: blend from the last strike target into
    # the ready target during recovery instead of switching abruptly. Default 0
    # preserves the original lifecycle exactly.
    recovery_blend_s: float = 0.0
    recovery_blend_velocity: bool = False


class SwingLifecycle:
    def __init__(self, cfg: LifecycleConfig | None = None) -> None:
        self.cfg = cfg or LifecycleConfig()
        self.phase = Phase.READY
        self.active_task_id: int | None = None
        self.swing_side: int = FOREHAND         # last locked side (default forehand)
        self._target_pos_w = np.zeros(3)
        self._target_vel_w = np.zeros(3)
        self._target_normal_w = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._tts = self.cfg.ready_time_to_strike
        self._follow_t = 0.0
        self._recover_t = 0.0
        self._recovery_start_pos_w = np.zeros(3)
        self._recovery_start_vel_w = np.zeros(3)
        # task_id of the most recently engaged ball; task_ids increase monotonically
        # (one ball, one increasing id), so we only ever engage a strictly newer id --
        # this enforces exactly one swing per task_id.
        self._last_engaged_task_id: int = -1
        # highest task_revision already applied to the active task (pre-contact only).
        self._applied_revision: int = -1

    # -- helpers ------------------------------------------------------------
    def _ready_target_pos_w(self, state: RobotState) -> np.ndarray:
        side = 1.0 if self.swing_side >= 0 else -1.0
        base = np.asarray(state.base_pos_w, dtype=np.float64)
        return base + np.array(
            [self.cfg.ready_reach_x, side * self.cfg.ready_reach_y, self.cfg.ready_reach_z],
            dtype=np.float64,
        )

    def _can_engage(self) -> bool:
        return self.phase in (Phase.READY, Phase.RECOVERY)

    @staticmethod
    def _normal_from_command(cmd: RacketCommand | None, fallback_vel: np.ndarray) -> np.ndarray:
        normal = getattr(cmd, "target_normal", None) if cmd is not None else None
        if normal is None:
            normal = fallback_vel
        normal = np.asarray(normal, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(normal))
        if n < 1.0e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return normal / n

    # -- main tick ----------------------------------------------------------
    def update(self, cmd: RacketCommand | None, state: RobotState) -> ObsTarget:
        c = self.cfg

        # 1) Engage a new ball, or refine the active one before contact.
        if cmd is not None:
            if cmd.task_id > self._last_engaged_task_id and self._can_engage():
                # A strictly newer ball: engage once and lock the side.
                self.active_task_id = cmd.task_id
                self._last_engaged_task_id = cmd.task_id
                self._applied_revision = cmd.task_revision
                self.swing_side = cmd.swing_side          # locked for this task
                self._target_pos_w = np.asarray(cmd.position, dtype=np.float64).copy()
                self._target_vel_w = np.asarray(cmd.velocity, dtype=np.float64).copy()
                self._target_normal_w = self._normal_from_command(cmd, self._target_vel_w)
                self._tts = float(cmd.time_to_strike)
                self.phase = Phase.SWING
            elif (
                cmd.task_id == self.active_task_id
                and self.phase == Phase.SWING
                and self._tts > 0.0
                and cmd.task_revision > self._applied_revision
            ):
                # Pre-contact revision (must be newer): update where/when, never the
                # locked side.
                self._applied_revision = cmd.task_revision
                self._target_pos_w = np.asarray(cmd.position, dtype=np.float64).copy()
                self._target_vel_w = np.asarray(cmd.velocity, dtype=np.float64).copy()
                self._target_normal_w = self._normal_from_command(cmd, self._target_vel_w)
                self._tts = float(cmd.time_to_strike)

        # 2) Advance the reference clock and step the phase machine.
        if self.phase == Phase.SWING:
            self._tts -= c.dt
            if self._tts <= 0.0:
                self.phase = Phase.FOLLOW_THROUGH
                self._follow_t = 0.0
        elif self.phase == Phase.FOLLOW_THROUGH:
            self._tts -= c.dt
            self._follow_t += c.dt
            if self._follow_t >= c.follow_through_s:
                self.phase = Phase.RECOVERY
                self._recover_t = 0.0
                self._recovery_start_pos_w = self._target_pos_w.copy()
                self._recovery_start_vel_w = self._target_vel_w.copy()
        elif self.phase == Phase.RECOVERY:
            self._recover_t += c.dt
            if self._recover_t >= c.recovery_s:
                self.phase = Phase.READY
                self.active_task_id = None

        # 3) Emit the goal to observe this tick.
        if self.phase in (Phase.SWING, Phase.FOLLOW_THROUGH):
            return ObsTarget(
                pos_w=self._target_pos_w,
                vel_w=self._target_vel_w,
                time_to_strike=self._tts,
                swing_side=float(self.swing_side),
                normal_w=self._target_normal_w,
            )
        # READY / RECOVERY -> in-place ready reach, clock pinned.  If requested,
        # smooth only the recovery observation target; the state machine and
        # one-swing-per-task contract are unchanged.
        ready_pos_w = self._ready_target_pos_w(state)
        ready_vel_w = np.zeros(3)
        if self.phase == Phase.RECOVERY and c.recovery_blend_s > 1.0e-9 and self._recover_t < c.recovery_blend_s:
            a = max(0.0, min(float(self._recover_t / c.recovery_blend_s), 1.0))
            # Smoothstep avoids a sharp acceleration at both ends of the blend.
            a = a * a * (3.0 - 2.0 * a)
            ready_pos_w = (1.0 - a) * self._recovery_start_pos_w + a * ready_pos_w
            if c.recovery_blend_velocity:
                ready_vel_w = (1.0 - a) * self._recovery_start_vel_w
        return ObsTarget(
            pos_w=ready_pos_w,
            vel_w=ready_vel_w,
            time_to_strike=c.ready_time_to_strike,
            swing_side=float(self.swing_side),
            normal_w=self._target_normal_w,
        )
