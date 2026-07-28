# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""The 50 Hz reference control loop.

Per tick, in order:
  1. read robot state from the sim bridge;
  2. poll the latest RacketCommand and advance the swing lifecycle;
  3. assemble the 111-D observation;
  4. run the ONNX actor -> raw_action[31];
  5. zero the passive head columns (idx 3, 4) to form the APPLIED raw action;
  6. pass the applied action through the shared ActionAdapter -> 31 joint targets
     (holding the passive neck at its default);
  7. feed back either the applied raw action or the residual represented by the
     clamped target, according to the runtime contract;
  8. write the targets (with the example PD gains) and step the sim.

There are deliberately NO gates, failure checks, rejections, state resets between
tasks, or reference playback here -- a single continuous 111-D control path.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from .config import RuntimeConfig
from .joint_order import HEAD_INDICES, NUM_JOINTS
from .lifecycle import SwingLifecycle
from .observation import (
    OBS_DIM_NORMAL114,
    OBS_DIM_STABILITY122,
    build_observation,
    build_observation_normal114,
    build_observation_stability122,
)
from .onnx_policy import OnnxPolicy
from .racket_command import RacketCommandSource
from .sim_bridge import SimBridge

_HEAD_IDX = list(HEAD_INDICES)


class PingPongReferenceRunner:
    def __init__(
        self,
        cfg: RuntimeConfig,
        bridge: SimBridge,
        command_source: RacketCommandSource,
        policy: OnnxPolicy | None = None,
    ) -> None:
        self.cfg = cfg
        self.bridge = bridge
        self.source = command_source
        self.policy = policy or OnnxPolicy(cfg.onnx_path)
        policy_feedback_mode = getattr(self.policy, "last_action_feedback_mode", None)
        if (
            policy_feedback_mode is not None
            and policy_feedback_mode != cfg.last_action_feedback_mode
        ):
            raise ValueError(
                "Runtime last_action feedback does not match the ONNX policy contract: "
                f"runtime={cfg.last_action_feedback_mode!r}, "
                f"policy={policy_feedback_mode!r}"
            )
        self.lifecycle = SwingLifecycle(cfg.lifecycle)

        self.default_q = cfg.action_adapter.default_q.copy()
        self.kp = cfg.sim_kp.copy()
        self.kd = cfg.sim_kd.copy()

        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.fixed_station_xy: np.ndarray | None = None

    def run(self, max_ticks: int | None = None, realtime: bool = False,
            status_every: int = 100) -> None:
        dt = self.cfg.control_dt
        self.bridge.reset()

        state = self.bridge.read_state()
        # Capture the startup station once; the fixed_station_error_xy obs term is
        # measured against this constant for the rest of the session.
        self.fixed_station_xy = np.asarray(state.base_pos_w[:2], dtype=np.float64).copy()

        tick = 0
        try:
            while max_ticks is None or tick < max_ticks:
                loop_start = time.perf_counter()

                state = self.bridge.read_state()
                cmd = self.source.poll()
                target = self.lifecycle.update(cmd, state)

                policy_obs_dim = getattr(self.policy, "obs_dim", 111)
                if policy_obs_dim == OBS_DIM_STABILITY122:
                    obs = build_observation_stability122(
                        state, target, self.last_action, self.default_q, self.fixed_station_xy
                    )
                elif policy_obs_dim == OBS_DIM_NORMAL114:
                    obs = build_observation_normal114(
                        state, target, self.last_action, self.default_q, self.fixed_station_xy
                    )
                else:
                    obs = build_observation(
                        state, target, self.last_action, self.default_q, self.fixed_station_xy
                    )
                raw_action = self.policy.infer(obs)
                # The APPLIED action: with a passive neck the head columns are never
                # actuated, so they are zeroed before feedback — training exposes the
                # same zeroed columns in its last_action observation.
                applied_action = np.asarray(raw_action, dtype=np.float64).copy()
                if self.cfg.passive_neck:
                    applied_action[_HEAD_IDX] = 0.0

                q_des = self.cfg.action_adapter.decode(applied_action)
                if self.cfg.passive_neck:
                    q_des[_HEAD_IDX] = self.default_q[_HEAD_IDX]
                if self.cfg.last_action_feedback_mode == "effective":
                    self.last_action = self.cfg.action_adapter.encode_effective(q_des)
                    if self.cfg.passive_neck:
                        self.last_action[_HEAD_IDX] = 0.0
                else:
                    self.last_action = applied_action.copy()

                self.bridge.write_targets(q_des, self.kp, self.kd)
                self.bridge.step()
                self.bridge.sync_viewer()

                if not self.bridge.is_viewer_running():
                    break

                if status_every and tick % status_every == 0:
                    self._print_status(tick, target)

                tick += 1
                if realtime:
                    self._sleep_to_rate(loop_start, dt)
        finally:
            self.bridge.close()

    def _print_status(self, tick: int, target) -> None:
        side = "forehand" if target.swing_side >= 0 else "backhand"
        print(
            f"[ref] t={tick * self.cfg.control_dt:6.2f}s "
            f"phase={self.lifecycle.phase.value:<14} "
            f"task={self.lifecycle.active_task_id} side={side} "
            f"tts={target.time_to_strike:+.2f}",
            file=sys.stderr,
        )

    @staticmethod
    def _sleep_to_rate(loop_start: float, dt: float) -> None:
        remaining = dt - (time.perf_counter() - loop_start)
        if remaining > 0:
            time.sleep(remaining)
