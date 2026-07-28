"""Racket-target command: the ping-pong goal on top of motion imitation.

:class:`RacketTargetCommand` rides on the :class:`~whole_body_tracking.tasks.tracking.mdp.commands.MotionCommand`.
Each swing it samples the quantities the model-based planner supplies at deploy time — a desired racket
position, a desired racket velocity, and a time-to-strike — plus the swing side (forehand/backhand),
which is locked for the duration of that swing. It also:

* holds a ready station target (a startup constant = the environment origin), and can optionally
  expose a per-swing dynamic station before impact.  The observation slot remains the same 2-D
  station-error contract: in fixed mode it is ready-station error; in dynamic mode it is the base
  target needed to hit the current racket intercept with the reference motion's natural strike pose.
* computes the ACTUAL racket state in simulation by forward kinematics through the fixed racket mount
  (wrist -> paddle center), so the reward can compare actual vs desired.
* derives the strike timing from the reference clip phase, and evaluates a simple no-spin outgoing
  ball at the strike (contact + net crossing + opponent-half first bounce) for the return rewards.

There is no measured racket feedback at deploy: the racket FK, its face normal, and the ball
evaluation are simulation-only signals used by rewards/critic, never by the actor observation.
Swing side selection is uniform per swing and follows the imitated clip (clip 0 = forehand -> +1,
clip 1 = backhand -> -1), so all four forehand/backhand transitions appear across the batch.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_error_magnitude, quat_mul, sample_uniform

from whole_body_tracking.tasks.tracking.mdp.ballistics import (
    GRAVITY as _GRAVITY,
    ballistic_velocity_from_landing as _ballistic_velocity_from_landing,
    ballistic_z_at_x as _ballistic_z_at_x,
)
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.table_workspace import (
    interpolate_bounds,
    table_side_lateral_bounds,
    validate_table_workspace,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class RacketTargetCommand(CommandTerm):
    """Samples desired racket/station targets and computes the actual racket state by FK."""

    cfg: RacketTargetCommandCfg

    def __init__(self, cfg: RacketTargetCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Racket FK source: prefer a dedicated racket body, else (wrist pose) * (fixed mount offset).
        if cfg.racket_body_name in self.robot.body_names:
            self._racket_mode = "body"
            self._racket_body_index = self.robot.find_bodies(cfg.racket_body_name, preserve_order=True)[0][0]
            self._wrist_body_index = -1
        else:
            assert cfg.wrist_body_name in self.robot.body_names, (
                f"RacketTargetCommand: neither racket body '{cfg.racket_body_name}' nor wrist body "
                f"'{cfg.wrist_body_name}' found on asset '{cfg.asset_name}'."
            )
            self._racket_mode = "wrist_offset"
            self._racket_body_index = -1
            self._wrist_body_index = self.robot.find_bodies(cfg.wrist_body_name, preserve_order=True)[0][0]
        self._mount_offset = torch.tensor(cfg.mount_offset, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self._mount_quat = torch.tensor(cfg.mount_quat, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )

        self._motion_term: MotionCommand | None = None

        # Desired (sampled) targets, world frame.
        self.racket_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_impact_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w[:, 2] = 1.0
        # Strike-command backups.  Public racket_target_* tensors may be temporarily overridden
        # during deploy-style no-command holds; the true sampled strike command is restored as soon
        # as the hold ends, so the swing/reward task remains unchanged.
        self._strike_racket_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._strike_racket_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._strike_racket_impact_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._strike_racket_target_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._strike_racket_target_normal_w[:, 2] = 1.0
        # Hidden true ball task.  The actor sees the planner command above; rewards use these
        # fields so training can tolerate planner error instead of treating the planner as truth.
        self.ball_strike_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.incoming_ball_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_outgoing_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.swing_sign = torch.ones(self.num_envs, device=self.device)
        self.dynamic_station_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.dynamic_station_w[:] = self.fixed_station_w

        # Actual racket state (FK), world frame.
        self.racket_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.racket_quat_w[:, 0] = 1.0
        self.racket_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_normal_w[:, 2] = 1.0

        # Strike timing.
        self.true_time_to_strike = torch.zeros(self.num_envs, device=self.device)
        self.time_to_strike = torch.zeros(self.num_envs, device=self.device)
        self.pre_strike = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.strike_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._planner_tts_offset = torch.zeros(self.num_envs, device=self.device)
        self.steps_since_target_resample = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.no_command_ready_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.no_command_ready_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Reward helper signals.
        self.racket_target_distance = torch.zeros(self.num_envs, device=self.device)
        self.feet_contact_frac = torch.zeros(self.num_envs, device=self.device)
        self.feet_contact_state = torch.zeros(self.num_envs, 2, device=self.device)
        # No-spin return evaluation caches (one-shot at the exact strike frame).
        self.strike_fired = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_net_cross = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_on_opponent = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_swing_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_swing_net_cross = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_swing_on_opponent = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_swing_attempt = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_swing_unsafe = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_swing_station_saturated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.prev_swing_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_swing_net_cross = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_swing_on_opponent = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Cycle-v2 state machine: after a useful previous return, require the next
        # command-visible pre-strike state to become ready before a deadline.
        self.cycle_v2_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_attempt_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_ready_success_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_ready_fail_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_attempt_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_ready_success_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_ready_fail_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_unresolved_resample_fail_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.cycle_v2_command_visible_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cycle_v2_elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cycle_v2_ready_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_ready_score = torch.zeros(self.num_envs, device=self.device)
        self.cycle_v2_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_ready_score_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_height_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_upright_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_base_lin_vel_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_base_ang_vel_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_feet_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_core_ready_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cycle_v2_outcome_tier = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.cycle_v2_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.impact_ball_out_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.impact_ball_out_error = torch.zeros(self.num_envs, device=self.device)
        self._ability_curriculum_initialized = False
        self._ability_curriculum_level = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_contact_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_net_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_success_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_recovery_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_cycle_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_cycle_attempt_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_cycle_resolved_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_forehand_contact_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_forehand_net_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_forehand_success_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_backhand_contact_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_backhand_net_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_backhand_success_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_attempt_ema = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_safety_ema = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self._ability_station_saturation_ema = torch.tensor(
            0.0, dtype=torch.float32, device=self.device
        )

        # Per-clip strike phase / target boxes (resolved lazily once the motion term is available).
        self._strike_phase_per_clip = None
        self._swing_side_per_clip = (
            torch.tensor([float(s) for s in cfg.swing_side_per_clip], device=self.device)
            if cfg.swing_side_per_clip
            else None
        )
        self._pos_box = _boxes_to_tensor(cfg.racket_pos_range_per_clip, self.device)  # (C,3,2) or None
        self._vel_box = _boxes_to_tensor(cfg.racket_vel_range_per_clip, self.device)
        self._mount_sign_per_clip = (
            torch.tensor([float(s) for s in cfg.mount_normal_sign_per_clip], device=self.device)
            if cfg.mount_normal_sign_per_clip
            else None
        )
        self._motion_racket_offset_xy = None
        if str(cfg.strike_position_mode) == "table_workspace":
            validate_table_workspace(
                cfg.table_width,
                cfg.table_workspace_edge_margin,
                cfg.table_workspace_side_overlap,
                cfg.table_workspace_forehand_core_y_range,
                cfg.table_workspace_backhand_core_y_range,
            )
            interpolate_bounds(
                cfg.table_workspace_x_jitter_core_range,
                cfg.table_workspace_x_jitter_full_range,
                0.0,
            )
            interpolate_bounds(
                cfg.table_workspace_z_core_above_surface_range,
                cfg.table_workspace_z_full_above_surface_range,
                0.0,
            )

        # Feet resolution for contact fraction (degrades to 0 if it cannot resolve — never crashes).
        try:
            self._contact_sensor = env.scene.sensors["contact_forces"]
        except (KeyError, AttributeError, TypeError):
            self._contact_sensor = None
        self._foot_idx_contact: list[int] = []
        if self._contact_sensor is not None:
            sensor_bodies = list(self._contact_sensor.body_names)
            self._foot_idx_contact = [sensor_bodies.index(n) for n in cfg.feet_body_names if n in sensor_bodies]

        for key in (
            "racket_pos_error",
            "racket_vel_error",
            "time_to_strike",
            "return_success",
            "prev_return_success",
            "prev_net_cross",
            "current_net_cross",
            "racket_pos_curriculum_scale",
            "steps_since_target_resample",
            "recovery_height_error",
            "recovery_upright_error",
            "recovery_base_lin_vel",
            "recovery_base_ang_vel",
            "recovery_station_error",
            "recovery_feet_contact_frac",
            "recovery_racket_speed",
            "recovery_height_score",
            "recovery_upright_score",
            "recovery_lin_vel_score",
            "recovery_ang_vel_score",
            "recovery_station_score",
            "recovery_racket_score",
            "recovery_arm_score",
            "recovery_ready_score",
            "recovery_phase_gate",
            "recovery_phase_ready_score",
            "recovery_contact_ready_score",
            "recovery_net_ready_score",
            "recovery_return_ready_score",
            "cycle_success_ready_score",
            "cycle_success_soft",
            "cycle_success_binary",
            "cycle_v2_pending",
            "cycle_v2_attempt_latch",
            "cycle_v2_ready_success_latch",
            "cycle_v2_ready_fail_latch",
            "cycle_v2_attempt_event",
            "cycle_v2_ready_success_event",
            "cycle_v2_ready_fail_event",
            "cycle_v2_unresolved_resample_fail_event",
            "cycle_v2_resolved_event",
            "cycle_v2_command_visible_steps",
            "cycle_v2_elapsed_steps",
            "cycle_v2_ready_score",
            "cycle_v2_ready_ok",
            "cycle_v2_visible",
            "cycle_v2_ready_score_ok",
            "cycle_v2_height_ok",
            "cycle_v2_upright_ok",
            "cycle_v2_base_lin_vel_ok",
            "cycle_v2_base_ang_vel_ok",
            "cycle_v2_feet_ok",
            "cycle_v2_core_ready_ok",
            "no_command_ready_active",
            "ability_curriculum_level",
            "ability_curriculum_scale",
            "ability_contact_ema",
            "ability_net_ema",
            "ability_success_ema",
            "ability_recovery_ema",
            "ability_cycle_ema",
            "ability_cycle_attempt_ema",
            "ability_cycle_resolved_ema",
            "ability_forehand_contact_ema",
            "ability_forehand_net_ema",
            "ability_forehand_success_ema",
            "ability_backhand_contact_ema",
            "ability_backhand_net_ema",
            "ability_backhand_success_ema",
            "ability_attempt_ema",
            "ability_safety_ema",
            "ability_station_saturation_ema",
            "planner_hit_plane_blend",
            "planner_hit_plane_x_delta",
            "true_strike_table_x",
            "planner_target_table_x",
            "workspace_level",
            "workspace_fringe_sample",
            "workspace_target_lateral_norm",
            "workspace_target_table_y",
            "workspace_playable_y_min",
            "workspace_playable_y_max",
            "workspace_active_y_lo",
            "workspace_active_y_hi",
            "station_saturated",
            "station_raw_clip_excess",
            "current_swing_attempt",
            "current_swing_unsafe",
            "current_swing_station_saturated",
            "cycle_v2_outcome_tier",
            "cycle_v2_streak",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    # --- helpers -------------------------------------------------------------------------------- #
    def _motion(self) -> MotionCommand:
        if self._motion_term is None:
            self._motion_term = self._env.command_manager.get_term(self.cfg.motion_command_name)
        return self._motion_term

    @property
    def base_pos_w(self) -> torch.Tensor:
        return self.robot.data.root_pos_w

    @property
    def base_quat_w(self) -> torch.Tensor:
        return self.robot.data.root_quat_w

    @property
    def fixed_station_w(self) -> torch.Tensor:
        """Fixed ready station XY = the environment origin plus a nominal offset (constant)."""
        off = torch.tensor(self.cfg.station_nominal_offset_xy, device=self.device)
        return self._env.scene.env_origins[:, :2] + off

    @property
    def station_w(self) -> torch.Tensor:
        """Current base-station target exposed to the actor.

        In dynamic mode, the station is dynamic before impact so the base can move under the
        reference strike pose; after impact and during recovery it switches back to the fixed ready
        station.  This keeps the 111-D observation layout unchanged while giving random racket
        targets a base-motion target instead of encouraging wrist-only chasing.
        """
        if self.cfg.station_mode == "fixed":
            return self.fixed_station_w
        if self.cfg.station_mode != "dynamic_from_motion":
            raise ValueError(f"Unsupported station_mode: {self.cfg.station_mode}")
        use_dynamic = ((self.pre_strike | self.strike_window) & (~self.no_command_ready_active)).unsqueeze(-1)
        return torch.where(use_dynamic, self.dynamic_station_w, self.fixed_station_w)

    @property
    def command(self) -> torch.Tensor:
        """Raw target vector (world): [pos(3), vel(3), tts(1), station(2), swing(1)]."""
        return torch.cat(
            [
                self.racket_target_pos_w,
                self.racket_target_vel_w,
                self.time_to_strike.unsqueeze(-1),
                self.station_w,
                self.swing_sign.unsqueeze(-1),
            ],
            dim=-1,
        )

    # --- observation accessors ------------------------------------------------------- #
    def base_forward_xy(self) -> torch.Tensor:
        """Base forward unit vector e_base,x, world XY (2)."""
        fwd = quat_apply(
            self.base_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        )[:, :2]
        return fwd / (torch.norm(fwd, dim=-1, keepdim=True) + 1e-6)

    def fixed_station_error_xy(self) -> torch.Tensor:
        """Current station XY minus current base XY, world frame (2)."""
        return self.station_w - self.base_pos_w[:, :2]

    def racket_target_rel_base_w(self) -> torch.Tensor:
        """Target racket position minus base position, world frame (3)."""
        return self.racket_target_pos_w - self.base_pos_w

    # --- sampling ------------------------------------------------------------------------------- #
    def _sample_targets(self, env_ids: torch.Tensor):
        motion = self._motion()
        n = len(env_ids)
        clip = motion.clip_id[env_ids] if motion._multiseg else torch.zeros(n, dtype=torch.long, device=self.device)
        fixed_station = self.fixed_station_w[env_ids]  # (n, 2)
        self._assign_swing_side(env_ids, clip, motion)

        pos_box = self._resolve_box(self._pos_box, clip, self.cfg.racket_pos_range)  # (n, 3, 2)
        pos_box = self._apply_racket_pos_curriculum(pos_box)
        vel_box = self._resolve_box(self._vel_box, clip, self.cfg.racket_vel_range)

        # True ball strike position: x/y are ready-station-relative (fixed table/strike frame),
        # z is absolute.  The planner command may be perturbed away from this hidden truth.
        true_pos, plane_blend, plane_delta = self._sample_true_strike_pos(env_ids, clip, fixed_station, pos_box)
        true_pos[:, 0] = fixed_station[:, 0] + true_pos[:, 0]
        true_pos[:, 1] = fixed_station[:, 1] + true_pos[:, 1]
        self.ball_strike_pos_w[env_ids] = true_pos
        self.metrics["planner_hit_plane_blend"][env_ids] = plane_blend
        self.metrics["planner_hit_plane_x_delta"][env_ids] = plane_delta
        self.metrics["true_strike_table_x"][env_ids] = (
            true_pos[:, 0] - self._env.scene.env_origins[env_ids, 0] - float(self.cfg.table_near_x)
        )

        planner_pos_offset, planner_vel_offset, planner_vel_scale, planner_yaw, planner_tts_offset = (
            self._sample_planner_perturbations(n)
        )
        self._planner_tts_offset[env_ids] = planner_tts_offset
        planner_pos = true_pos + planner_pos_offset
        self.racket_target_pos_w[env_ids] = planner_pos
        self.metrics["planner_target_table_x"][env_ids] = (
            planner_pos[:, 0] - self._env.scene.env_origins[env_ids, 0] - float(self.cfg.table_near_x)
        )
        self._update_dynamic_station(env_ids, clip, planner_pos, fixed_station)

        if self.cfg.racket_velocity_mode == "range":
            outgoing_vel = sample_uniform(vel_box[..., 0], vel_box[..., 1], (n, 3), self.device)
        elif self.cfg.racket_velocity_mode in ("ballistic_landing", "impact_inverse_landing"):
            outgoing_vel = self._sample_ballistic_target_velocity(env_ids, true_pos, fixed_station)
        else:
            raise ValueError(f"Unsupported racket_velocity_mode: {self.cfg.racket_velocity_mode}")
        outgoing_vel = self._apply_outgoing_target_calibration(outgoing_vel)

        if self.cfg.incoming_trajectory_mode == "direct":
            incoming_vel = self._sample_incoming_ball_velocity(env_ids, true_pos)
        elif self.cfg.incoming_trajectory_mode == "one_bounce":
            incoming_vel = self._sample_one_bounce_incoming_ball_velocity(env_ids, true_pos, fixed_station)
        else:
            raise ValueError(f"Unsupported incoming_trajectory_mode: {self.cfg.incoming_trajectory_mode}")
        self.incoming_ball_vel_w[env_ids] = incoming_vel

        self.ball_outgoing_target_vel_w[env_ids] = outgoing_vel
        if self.cfg.racket_velocity_mode == "impact_inverse_landing":
            racket_vel, normal = self._solve_impact_racket_command(incoming_vel, outgoing_vel)
        else:
            # Legacy mode: the target velocity is interpreted directly as racket velocity.
            # The outgoing target remains the same vector for the simple analytic return model.
            racket_vel = outgoing_vel
            normal = outgoing_vel - incoming_vel
            normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        self.racket_impact_target_vel_w[env_ids] = racket_vel
        self.racket_target_normal_w[env_ids] = normal
        planner_vel = self._apply_planner_velocity_perturbation(
            racket_vel, planner_vel_offset, planner_vel_scale, planner_yaw
        )
        self.racket_target_vel_w[env_ids] = planner_vel
        self._strike_racket_target_pos_w[env_ids] = self.racket_target_pos_w[env_ids]
        self._strike_racket_target_vel_w[env_ids] = self.racket_target_vel_w[env_ids]
        self._strike_racket_impact_target_vel_w[env_ids] = self.racket_impact_target_vel_w[env_ids]
        self._strike_racket_target_normal_w[env_ids] = self.racket_target_normal_w[env_ids]
        self._sample_no_command_ready_hold(env_ids)

    def _assign_swing_side(
        self, env_ids: torch.Tensor, clip: torch.Tensor, motion: MotionCommand
    ) -> None:
        """Assign the side before target sampling so table-workspace ranges can be side compatible."""
        if self._swing_side_per_clip is not None and motion._multiseg:
            if self._swing_side_per_clip.shape[0] != motion.motion.num_segments:
                raise ValueError(
                    "RacketTargetCommandCfg.swing_side_per_clip length "
                    f"({self._swing_side_per_clip.shape[0]}) must match motion clips "
                    f"({motion.motion.num_segments})."
                )
            self.swing_sign[env_ids] = self._swing_side_per_clip[clip]
        elif motion._multiseg:
            self.swing_sign[env_ids] = torch.where(clip == 0, 1.0, -1.0)
        else:
            self.swing_sign[env_ids] = 1.0

    def _sample_no_command_ready_hold(self, env_ids: torch.Tensor) -> None:
        """Mark a subset of pre-swing holds as deploy no-command ready states."""
        self.no_command_ready_hold[env_ids] = False
        prob = float(self.cfg.deploy_ready_hold_prob)
        if prob <= 0.0:
            return
        motion = self._motion()
        hold = motion.in_hold[env_ids]
        if not bool(torch.any(hold)):
            return
        draw = torch.rand(len(env_ids), device=self.device) < min(max(prob, 0.0), 1.0)
        self.no_command_ready_hold[env_ids] = hold & draw

    def _apply_no_command_ready_targets(self) -> None:
        """Expose deploy lifecycle ready/no-command targets while the motion clock is held."""
        self.racket_target_pos_w.copy_(self._strike_racket_target_pos_w)
        self.racket_target_vel_w.copy_(self._strike_racket_target_vel_w)
        self.racket_impact_target_vel_w.copy_(self._strike_racket_impact_target_vel_w)
        self.racket_target_normal_w.copy_(self._strike_racket_target_normal_w)

        motion = self._motion()
        active = self.no_command_ready_hold & motion.in_hold
        self.no_command_ready_active.copy_(active)
        if not bool(torch.any(active)):
            return

        ready_reach = torch.tensor(self.cfg.deploy_ready_reach, dtype=torch.float32, device=self.device).expand(
            self.num_envs, 3
        )
        side_y = self.swing_sign.sign().clamp(min=-1.0, max=1.0) * ready_reach[:, 1].abs()
        ready_pos = self.base_pos_w + torch.stack((ready_reach[:, 0], side_y, ready_reach[:, 2]), dim=-1)
        ready_vel = torch.tensor(self.cfg.deploy_ready_velocity_w, dtype=torch.float32, device=self.device).expand(
            self.num_envs, 3
        )
        ready_normal = torch.tensor(self.cfg.deploy_ready_normal_w, dtype=torch.float32, device=self.device)
        ready_normal = ready_normal / torch.norm(ready_normal).clamp_min(1.0e-6)

        self.racket_target_pos_w[active] = ready_pos[active]
        self.racket_target_vel_w[active] = ready_vel[active]
        self.racket_impact_target_vel_w[active] = ready_vel[active]
        self.racket_target_normal_w[active] = ready_normal
        self.time_to_strike[active] = float(self.cfg.deploy_ready_time_to_strike)

    def _motion_strike_phase_per_clip(self) -> torch.Tensor:
        """Return strike phase per motion segment without requiring per-step timing to be initialized."""
        ml = self._motion().motion
        sp = tuple(self.cfg.strike_phase_per_clip)
        if sp and len(sp) == ml.num_segments:
            return torch.tensor([float(x) for x in sp], device=self.device)
        return torch.full((ml.num_segments,), float(self.cfg.strike_phase), device=self.device)

    def _ensure_motion_racket_offsets(self) -> torch.Tensor:
        """Natural racket XY offset from the motion root at each clip's strike frame."""
        if self._motion_racket_offset_xy is not None:
            return self._motion_racket_offset_xy
        motion = self._motion()
        ml = motion.motion
        try:
            wrist_idx = motion.cfg.body_names.index(self.cfg.wrist_body_name)
        except ValueError as exc:
            raise ValueError(
                f"dynamic station requires wrist body {self.cfg.wrist_body_name!r} in motion body_names"
            ) from exc
        root_idx = 0
        phases = self._motion_strike_phase_per_clip()
        strike_steps = ml.seg_start + (phases * (ml.seg_len - 1).float()).round().long()
        wrist_pos = ml.body_pos_w[strike_steps, wrist_idx]
        wrist_quat = ml.body_quat_w[strike_steps, wrist_idx]
        mount_offset = torch.tensor(self.cfg.mount_offset, dtype=torch.float32, device=self.device).expand(
            ml.num_segments, 3
        )
        racket_pos = wrist_pos + quat_apply(wrist_quat, mount_offset)
        root_pos = ml.body_pos_w[strike_steps, root_idx]
        self._motion_racket_offset_xy = racket_pos[:, :2] - root_pos[:, :2]
        return self._motion_racket_offset_xy

    def _update_dynamic_station(
        self,
        env_ids: torch.Tensor,
        clip: torch.Tensor,
        target_pos_w: torch.Tensor,
        fixed_station_w: torch.Tensor,
    ) -> None:
        if self.cfg.station_mode == "fixed":
            self.dynamic_station_w[env_ids] = fixed_station_w
            self.current_swing_station_saturated[env_ids] = False
            self.metrics["station_saturated"][env_ids] = 0.0
            self.metrics["station_raw_clip_excess"][env_ids] = 0.0
            return
        if self.cfg.station_mode != "dynamic_from_motion":
            raise ValueError(f"Unsupported station_mode: {self.cfg.station_mode}")
        offset_xy = self._ensure_motion_racket_offsets()[clip]
        desired = target_pos_w[:, :2] - offset_xy
        rel = desired - fixed_station_w
        clip_box = torch.tensor(self.cfg.dynamic_station_xy_clip, dtype=torch.float32, device=self.device)
        clipped_rel = torch.clamp(rel, clip_box[:, 0], clip_box[:, 1])
        clip_excess = torch.norm(rel - clipped_rel, dim=-1)
        saturated = clip_excess > 1.0e-5
        self.current_swing_station_saturated[env_ids] = saturated
        self.metrics["station_saturated"][env_ids] = saturated.float()
        self.metrics["station_raw_clip_excess"][env_ids] = clip_excess
        blend = float(self.cfg.dynamic_station_blend)
        self.dynamic_station_w[env_ids] = fixed_station_w + blend * clipped_rel

    def _resolve_box(self, per_clip, clip: torch.Tensor, shared_range) -> torch.Tensor:
        """Return an (n, 3, 2) [lo, hi] box per env: per-clip if configured, else the shared box."""
        if per_clip is not None:
            return per_clip[clip]
        shared = torch.tensor(shared_range, dtype=torch.float32, device=self.device)  # (3, 2)
        return shared.unsqueeze(0).expand(len(clip), 3, 2)

    def _racket_pos_curriculum_scale(self) -> torch.Tensor:
        if bool(self.cfg.ability_curriculum_enabled):
            start = self.cfg.ability_curriculum_start_racket_pos_scale
            if isinstance(start, (float, int)):
                start_scale = torch.full((3,), float(start), dtype=torch.float32, device=self.device)
            else:
                if len(start) != 3:
                    raise ValueError("ability_curriculum_start_racket_pos_scale must be a scalar or a 3-tuple")
                start_scale = torch.tensor([float(v) for v in start], dtype=torch.float32, device=self.device)
            start_scale = torch.clamp(start_scale, 0.0, 1.0)
            level = torch.clamp(self._ability_curriculum_level, 0.0, 1.0)
            return start_scale + level * (torch.ones_like(start_scale) - start_scale)
        steps = int(self.cfg.racket_pos_curriculum_steps)
        if steps <= 0:
            return torch.ones(3, dtype=torch.float32, device=self.device)
        start = self.cfg.racket_pos_curriculum_start_scale
        if isinstance(start, (float, int)):
            start_scale = torch.full((3,), float(start), dtype=torch.float32, device=self.device)
        else:
            if len(start) != 3:
                raise ValueError("racket_pos_curriculum_start_scale must be a scalar or a 3-tuple")
            start_scale = torch.tensor([float(v) for v in start], dtype=torch.float32, device=self.device)
        start_scale = torch.clamp(start_scale, 0.0, 1.0)
        step_count = float(getattr(self._env, "common_step_counter", 0))
        alpha = min(max(step_count / float(steps), 0.0), 1.0)
        return start_scale + alpha * (torch.ones_like(start_scale) - start_scale)

    def _ability_blend(self, start_scale: float) -> float:
        """Return an ability-gated scalar in [start_scale, 1]."""
        if not bool(self.cfg.ability_curriculum_enabled):
            return 1.0
        start = min(max(float(start_scale), 0.0), 1.0)
        level = float(torch.clamp(self._ability_curriculum_level, 0.0, 1.0).item())
        return start + level * (1.0 - start)

    @staticmethod
    def _scaled_float_range(value_range, scale: float) -> tuple[float, float]:
        lo, hi = (float(v) for v in value_range)
        if hi < lo:
            lo, hi = hi, lo
        scale = min(max(float(scale), 0.0), 1.0)
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * scale
        return center - half, center + half

    def _apply_racket_pos_curriculum(self, box: torch.Tensor) -> torch.Tensor:
        """Shrink target-position boxes around their centers early in training, then grow to full size."""
        scale = self._racket_pos_curriculum_scale()
        if bool(torch.all(scale >= 0.999)):
            return box
        center = 0.5 * (box[..., 0] + box[..., 1])
        half = 0.5 * (box[..., 1] - box[..., 0]) * scale.unsqueeze(0)
        return torch.stack((center - half, center + half), dim=-1)

    def _clip_scalar_or_default(self, values, clip: torch.Tensor, default: float) -> torch.Tensor:
        if values:
            table = torch.tensor([float(v) for v in values], dtype=torch.float32, device=self.device)
            max_clip = int(torch.max(clip).item()) if len(clip) > 0 else -1
            if table.shape[0] <= max_clip:
                raise ValueError(
                    f"Per-clip value length {table.shape[0]} is too short for clip id {max_clip}: {values!r}"
                )
            return table[clip]
        return torch.full((len(clip),), float(default), dtype=torch.float32, device=self.device)

    def _planner_hit_plane_blend(self, clip: torch.Tensor) -> torch.Tensor:
        """Return the current blend from motion-derived x sampling to planner table x_hit sampling."""
        mode = str(self.cfg.planner_hit_plane_mode)
        if mode == "motion_box":
            return torch.zeros((len(clip),), dtype=torch.float32, device=self.device)
        if mode != "fixed_x_hit":
            raise ValueError(f"Unsupported planner_hit_plane_mode: {self.cfg.planner_hit_plane_mode}")
        start = self._clip_scalar_or_default(
            self.cfg.planner_hit_plane_blend_start_per_clip, clip, float(self.cfg.planner_hit_plane_blend_start)
        )
        end = self._clip_scalar_or_default(
            self.cfg.planner_hit_plane_blend_per_clip, clip, float(self.cfg.planner_hit_plane_blend)
        )
        warmup_steps = int(self.cfg.planner_hit_plane_blend_warmup_steps)
        if warmup_steps <= 0:
            return torch.clamp(end, 0.0, 1.0)
        step_count = float(getattr(self._env, "common_step_counter", 0))
        alpha = min(max(step_count / float(warmup_steps), 0.0), 1.0)
        return torch.clamp(start + alpha * (end - start), 0.0, 1.0)

    def _sample_true_strike_pos(
        self,
        env_ids: torch.Tensor,
        clip: torch.Tensor,
        fixed_station_w: torch.Tensor,
        pos_box: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample hidden strike position, optionally moving x onto the planner table hit plane.

        ``pos_box`` uses the historical command convention: x/y are fixed-station-relative and z is
        absolute.  ``planner_hit_plane_x`` follows the planner/MuJoCo convention: table-frame x from
        the robot-side table edge.  Convert it into station-relative x before blending.
        """
        mode = str(self.cfg.strike_position_mode)
        if mode == "table_workspace":
            return self._sample_table_workspace_strike_pos(env_ids, fixed_station_w, pos_box)
        if mode != "motion_box":
            raise ValueError(
                "strike_position_mode must be one of 'motion_box' or 'table_workspace', "
                f"got {mode!r}"
            )
        n = len(env_ids)
        true_pos = sample_uniform(pos_box[..., 0], pos_box[..., 1], (n, 3), self.device)
        blend = self._planner_hit_plane_blend(clip)
        if bool(torch.all(blend <= 1.0e-6)):
            return true_pos, blend, torch.zeros_like(blend)

        origins = self._env.scene.env_origins[env_ids]
        fixed_station_local_x = fixed_station_w[:, 0] - origins[:, 0]
        jitter = self._sample_scalar_range(self.cfg.planner_hit_plane_x_jitter_range, n)
        planner_plane_station_rel_x = (
            float(self.cfg.table_near_x) + float(self.cfg.planner_hit_plane_x) + jitter - fixed_station_local_x
        )
        raw_x = true_pos[:, 0].clone()
        true_pos[:, 0] = raw_x + blend * (planner_plane_station_rel_x - raw_x)
        return true_pos, blend, true_pos[:, 0] - raw_x

    def _sample_table_workspace_strike_pos(
        self,
        env_ids: torch.Tensor,
        fixed_station_w: torch.Tensor,
        motion_box: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a side-compatible target across the regulation table width."""
        n = len(env_ids)
        if bool(self.cfg.ability_curriculum_enabled):
            level = float(torch.clamp(self._ability_curriculum_level, 0.0, 1.0).item())
        else:
            level = 1.0

        forehand_full = table_side_lateral_bounds(
            self.cfg.table_width,
            self.cfg.table_workspace_edge_margin,
            self.cfg.table_workspace_side_overlap,
            1.0,
        )
        backhand_full = table_side_lateral_bounds(
            self.cfg.table_width,
            self.cfg.table_workspace_edge_margin,
            self.cfg.table_workspace_side_overlap,
            -1.0,
        )
        forehand_current = interpolate_bounds(
            self.cfg.table_workspace_forehand_core_y_range, forehand_full, level
        )
        backhand_current = interpolate_bounds(
            self.cfg.table_workspace_backhand_core_y_range, backhand_full, level
        )
        x_current = interpolate_bounds(
            self.cfg.table_workspace_x_jitter_core_range,
            self.cfg.table_workspace_x_jitter_full_range,
            level,
        )
        z_current = interpolate_bounds(
            self.cfg.table_workspace_z_core_above_surface_range,
            self.cfg.table_workspace_z_full_above_surface_range,
            level,
        )

        forehand = self.swing_sign[env_ids] >= 0.0
        y_lo = torch.where(
            forehand,
            torch.full((n,), forehand_current[0], device=self.device),
            torch.full((n,), backhand_current[0], device=self.device),
        )
        y_hi = torch.where(
            forehand,
            torch.full((n,), forehand_current[1], device=self.device),
            torch.full((n,), backhand_current[1], device=self.device),
        )
        full_y_lo = torch.where(
            forehand,
            torch.full((n,), forehand_full[0], device=self.device),
            torch.full((n,), backhand_full[0], device=self.device),
        )
        full_y_hi = torch.where(
            forehand,
            torch.full((n,), forehand_full[1], device=self.device),
            torch.full((n,), backhand_full[1], device=self.device),
        )

        fringe_prob = min(max(float(self.cfg.table_workspace_fringe_prob), 0.0), 1.0)
        fringe = torch.rand(n, device=self.device) < fringe_prob
        y_lo = torch.where(fringe, full_y_lo, y_lo)
        y_hi = torch.where(fringe, full_y_hi, y_hi)

        x_lo = torch.full((n,), x_current[0], device=self.device)
        x_hi = torch.full((n,), x_current[1], device=self.device)
        z_lo = torch.full((n,), z_current[0], device=self.device)
        z_hi = torch.full((n,), z_current[1], device=self.device)
        x_lo[fringe] = float(self.cfg.table_workspace_x_jitter_full_range[0])
        x_hi[fringe] = float(self.cfg.table_workspace_x_jitter_full_range[1])
        z_lo[fringe] = float(self.cfg.table_workspace_z_full_above_surface_range[0])
        z_hi[fringe] = float(self.cfg.table_workspace_z_full_above_surface_range[1])

        x_jitter = sample_uniform(x_lo, x_hi, (n,), self.device)
        y_station = sample_uniform(y_lo, y_hi, (n,), self.device)
        z_above_table = sample_uniform(z_lo, z_hi, (n,), self.device)
        origins = self._env.scene.env_origins[env_ids]
        fixed_station_local_x = fixed_station_w[:, 0] - origins[:, 0]
        table_x_station = (
            float(self.cfg.table_near_x)
            + float(self.cfg.planner_hit_plane_x)
            + x_jitter
            - fixed_station_local_x
        )
        true_pos = torch.stack(
            (
                table_x_station,
                y_station,
                origins[:, 2] + float(self.cfg.table_surface_z) + z_above_table,
            ),
            dim=-1,
        )

        playable_half = 0.5 * float(self.cfg.table_width) - float(self.cfg.table_workspace_edge_margin)
        self.metrics["workspace_level"][env_ids] = level
        self.metrics["workspace_fringe_sample"][env_ids] = fringe.float()
        self.metrics["workspace_target_lateral_norm"][env_ids] = (
            torch.abs(y_station) / max(playable_half, 1.0e-6)
        )
        self.metrics["workspace_target_table_y"][env_ids] = y_station
        self.metrics["workspace_playable_y_min"][env_ids] = -playable_half
        self.metrics["workspace_playable_y_max"][env_ids] = playable_half
        self.metrics["workspace_active_y_lo"][env_ids] = y_lo
        self.metrics["workspace_active_y_hi"][env_ids] = y_hi
        raw_x = 0.5 * (motion_box[:, 0, 0] + motion_box[:, 0, 1])
        return true_pos, torch.ones(n, device=self.device), true_pos[:, 0] - raw_x

    def _sample_box_range(self, ranges, n: int) -> torch.Tensor:
        box = torch.tensor(ranges, dtype=torch.float32, device=self.device)
        if box.shape != (3, 2):
            raise ValueError(f"Expected a 3x2 range, got shape {tuple(box.shape)} for {ranges!r}")
        return sample_uniform(box[:, 0], box[:, 1], (n, 3), self.device)

    def _sample_scalar_range(self, value_range, n: int) -> torch.Tensor:
        lo, hi = (float(v) for v in value_range)
        return sample_uniform(lo, hi, (n,), self.device)

    def _sample_planner_perturbations(
        self, n: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample per-swing planner command errors.

        These perturb the actor-visible command only.  The hidden ball task remains unchanged and
        is used by contact / impact / net / opponent-bounce rewards.
        """
        pos_offset = self._sample_box_range(self.cfg.planner_target_pos_offset_range, n)
        vel_offset = self._sample_box_range(self.cfg.planner_target_vel_offset_range, n)
        vel_scale = self._sample_scalar_range(self.cfg.planner_target_vel_scale_range, n)
        yaw = self._sample_scalar_range(self.cfg.planner_target_vel_yaw_deg_range, n)
        tts_offset = self._sample_scalar_range(self.cfg.planner_time_to_strike_offset_range, n)
        perturb_scale = self._ability_blend(float(self.cfg.ability_curriculum_start_planner_perturb_scale))
        if perturb_scale < 0.999:
            pos_offset = pos_offset * perturb_scale
            vel_offset = vel_offset * perturb_scale
            vel_scale = 1.0 + (vel_scale - 1.0) * perturb_scale
            yaw = yaw * perturb_scale
            tts_offset = tts_offset * perturb_scale
        return pos_offset, vel_offset, vel_scale, yaw, tts_offset

    def _apply_planner_velocity_perturbation(
        self,
        vel: torch.Tensor,
        vel_offset: torch.Tensor,
        vel_scale: torch.Tensor,
        yaw_deg: torch.Tensor,
    ) -> torch.Tensor:
        out = vel * vel_scale.unsqueeze(-1)
        theta = yaw_deg * (torch.pi / 180.0)
        c = torch.cos(theta)
        s = torch.sin(theta)
        x = c * out[:, 0] - s * out[:, 1]
        y = s * out[:, 0] + c * out[:, 1]
        out = torch.stack((x, y, out[:, 2]), dim=-1)
        return out + vel_offset

    def _sample_ballistic_target_velocity(
        self,
        env_ids: torch.Tensor,
        target_pos_w: torch.Tensor,
        station_xy_w: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a target velocity that is self-consistent with the simplified return evaluator."""
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]
        p0 = target_pos_w - origins
        center_y = station_xy_w[:, 1] - origins[:, 1]

        ball_scale = self._ability_blend(float(self.cfg.ability_curriculum_start_ball_scale))
        x_lo, x_hi = self._scaled_float_range(self.cfg.ballistic_land_x_range, ball_scale)
        y_lo, y_hi = self._scaled_float_range(self.cfg.ballistic_land_y_range, ball_scale)
        t_lo, t_hi = self._scaled_float_range(self.cfg.ballistic_flight_time_range, ball_scale)
        net_x = float(self.cfg.table_near_x) + float(self.cfg.net_x)
        net_top = float(self.cfg.table_surface_z) + float(self.cfg.net_height) + float(self.cfg.net_margin)

        out = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        resolved = torch.zeros(n, dtype=torch.bool, device=self.device)
        attempts = max(int(self.cfg.ballistic_sample_attempts), 1)

        for _ in range(attempts):
            flight_time = sample_uniform(t_lo, t_hi, (n,), self.device)
            land_x = sample_uniform(x_lo, x_hi, (n,), self.device)
            land_y = center_y + sample_uniform(y_lo, y_hi, (n,), self.device)
            land_xy = torch.stack((land_x, land_y), dim=-1)
            candidate = _ballistic_velocity_from_landing(
                p0, land_xy, flight_time, float(self.cfg.table_surface_z)
            )
            t_net = (net_x - p0[:, 0]) / candidate[:, 0].clamp_min(1.0e-3)
            z_net = _ballistic_z_at_x(p0, candidate, net_x)
            valid = (
                (candidate[:, 0] > float(self.cfg.ballistic_min_forward_speed))
                & (t_net > 0.0)
                & (z_net > net_top)
            )
            take = (~resolved) & valid
            out[take] = candidate[take]
            resolved |= take
            if bool(torch.all(resolved)):
                break

        if not bool(torch.all(resolved)):
            flight_time = torch.full((n,), 0.5 * (t_lo + t_hi), dtype=torch.float32, device=self.device)
            land_x = torch.full((n,), 0.5 * (x_lo + x_hi), dtype=torch.float32, device=self.device)
            land_y = center_y
            land_xy = torch.stack((land_x, land_y), dim=-1)
            fallback = _ballistic_velocity_from_landing(
                p0, land_xy, flight_time, float(self.cfg.table_surface_z)
            )
            out[~resolved] = fallback[~resolved]
        return out

    def _apply_outgoing_target_calibration(self, outgoing_vel: torch.Tensor) -> torch.Tensor:
        """Apply small training-side corrections inferred from MuJoCo impact diagnostics."""
        scale = float(self.cfg.target_outgoing_vel_scale)
        z_bias = float(self.cfg.target_outgoing_vel_z_bias)
        if abs(scale - 1.0) < 1.0e-6 and abs(z_bias) < 1.0e-6:
            return outgoing_vel
        out = outgoing_vel * scale
        out[:, 2] = out[:, 2] + z_bias
        return out

    def _sample_incoming_ball_velocity(self, env_ids: torch.Tensor, target_pos_w: torch.Tensor) -> torch.Tensor:
        """Nominal incoming ball velocity at the strike point.

        The actor does not observe this extra signal; it is used only to shape the critic/rewards
        around a more realistic moving-racket impact model. The distribution mirrors the MuJoCo
        evaluator's serve geometry closely enough to make the desired racket face normal meaningful.
        """
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]
        ball_scale = self._ability_blend(float(self.cfg.ability_curriculum_start_ball_scale))
        x_lo, x_hi = self._scaled_float_range(self.cfg.incoming_origin_x_range, ball_scale)
        y_lo, y_hi = self._scaled_float_range(self.cfg.incoming_origin_y_jitter_range, ball_scale)
        z_lo, z_hi = self._scaled_float_range(self.cfg.incoming_origin_z_above_table_range, ball_scale)
        t_lo, t_hi = self._scaled_float_range(self.cfg.incoming_flight_time_range, ball_scale)

        origin = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        origin[:, 0] = origins[:, 0] + float(self.cfg.table_near_x) + sample_uniform(x_lo, x_hi, (n,), self.device)
        origin[:, 1] = target_pos_w[:, 1] + sample_uniform(y_lo, y_hi, (n,), self.device)
        origin[:, 2] = origins[:, 2] + float(self.cfg.table_surface_z) + sample_uniform(z_lo, z_hi, (n,), self.device)

        flight_time = sample_uniform(t_lo, t_hi, (n,), self.device)
        accel = torch.tensor([0.0, 0.0, -_GRAVITY], dtype=torch.float32, device=self.device)
        v0 = (target_pos_w - origin) / flight_time.unsqueeze(-1) - 0.5 * accel * flight_time.unsqueeze(-1)
        return v0 + accel * flight_time.unsqueeze(-1)

    def _sample_one_bounce_incoming_ball_velocity(
        self,
        env_ids: torch.Tensor,
        target_pos_w: torch.Tensor,
        fixed_station_w: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the true incoming velocity after one robot-half table bounce.

        This mirrors the MuJoCo one-bounce serve geometry at the command/reward level.  It does not
        spawn a rigid ball in Isaac; it generates the hidden incoming velocity that the moving-racket
        impact model sees at the true strike point.
        """
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]
        p_target = target_pos_w - origins
        center_y = fixed_station_w[:, 1] - origins[:, 1]

        near_x = float(self.cfg.table_near_x)
        net_x = near_x + float(self.cfg.net_x)
        half_w = 0.5 * float(self.cfg.table_width)
        margin = 0.08

        target_table_x = p_target[:, 0] - near_x
        ball_scale = self._ability_blend(float(self.cfg.ability_curriculum_start_ball_scale))
        dx_lo, dx_hi = self._scaled_float_range(
            (self.cfg.one_bounce_min_post_bounce_dx, self.cfg.one_bounce_max_post_bounce_dx),
            ball_scale,
        )
        lo_x = torch.maximum(
            torch.full((n,), 0.18, dtype=torch.float32, device=self.device),
            target_table_x + float(dx_lo),
        )
        hi_x = torch.minimum(
            torch.full((n,), float(self.cfg.net_x) - margin, dtype=torch.float32, device=self.device),
            target_table_x + float(dx_hi),
        )
        fallback_lo = torch.maximum(
            torch.full((n,), 0.12, dtype=torch.float32, device=self.device),
            torch.minimum(
                torch.full((n,), float(self.cfg.net_x) - margin, dtype=torch.float32, device=self.device),
                target_table_x + 0.20,
            ),
        )
        bad = hi_x <= lo_x
        lo_x = torch.where(bad, fallback_lo, lo_x)
        hi_x = torch.where(bad, torch.minimum(torch.full_like(lo_x, float(self.cfg.net_x) - margin), lo_x + 0.30), hi_x)
        bounce_x = near_x + sample_uniform(lo_x, hi_x, (n,), self.device)

        jitter_lo, jitter_hi = self._scaled_float_range(self.cfg.one_bounce_lateral_jitter_range, ball_scale)
        bounce_y = p_target[:, 1] + sample_uniform(jitter_lo, jitter_hi, (n,), self.device)
        bounce_y = torch.maximum(torch.minimum(bounce_y, center_y + half_w - margin), center_y - half_w + margin)
        bounce_z = torch.full((n,), float(self.cfg.table_surface_z) + float(self.cfg.ball_radius), device=self.device)
        bounce = torch.stack((bounce_x, bounce_y, bounce_z), dim=-1)

        post_lo, post_hi = self._scaled_float_range(self.cfg.one_bounce_post_time_range, ball_scale)
        post_t = sample_uniform(post_lo, post_hi, (n,), self.device).clamp_min(1.0e-3)
        accel = torch.tensor([0.0, 0.0, -_GRAVITY], dtype=torch.float32, device=self.device)
        post_vel = (p_target - bounce) / post_t.unsqueeze(-1) - 0.5 * accel * post_t.unsqueeze(-1)
        return post_vel + accel * post_t.unsqueeze(-1)

    def _moving_racket_impact_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Approximate no-spin ball velocity after contact with a moving racket plane."""
        normal = self.racket_normal_w / (torch.norm(self.racket_normal_w, dim=-1, keepdim=True) + 1e-6)
        rel_in = self.incoming_ball_vel_w - self.racket_lin_vel_w
        rel_n = torch.sum(rel_in * normal, dim=-1, keepdim=True)
        rel_t = rel_in - rel_n * normal
        rel_out = float(self.cfg.paddle_tangent_retain) * rel_t - float(self.cfg.paddle_restitution) * rel_n * normal
        return self.racket_lin_vel_w + rel_out, rel_n.squeeze(-1)

    def _solve_impact_racket_command(
        self, incoming_vel: torch.Tensor, outgoing_vel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the moving-racket impact model into a feasible racket velocity and blade normal.

        ``outgoing_vel`` is the desired post-impact ball velocity.  The policy should not chase that
        vector as the racket velocity; for a moving paddle, the needed racket velocity is smaller and
        depends on the incoming ball velocity and the blade normal.  We use the impulse direction
        ``outgoing - incoming`` as the target normal, then solve the normal/tangential components of the
        moving-plane model.  A speed clamp keeps the command inside the A3's observed reachable range.
        """
        delta = outgoing_vel - incoming_vel
        delta_norm = torch.norm(delta, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand_as(delta)
        normal = torch.where(delta_norm > 1.0e-6, delta / delta_norm.clamp_min(1.0e-6), fallback)

        e = float(self.cfg.paddle_restitution)
        retain = float(self.cfg.paddle_tangent_retain)
        vin_n = torch.sum(incoming_vel * normal, dim=-1, keepdim=True)
        vout_n = torch.sum(outgoing_vel * normal, dim=-1, keepdim=True)
        racket_n = (vout_n + e * vin_n) / max(1.0 + e, 1.0e-6)

        vin_t = incoming_vel - vin_n * normal
        vout_t = outgoing_vel - vout_n * normal
        if abs(1.0 - retain) > 1.0e-4:
            racket_t_exact = (vout_t - retain * vin_t) / (1.0 - retain)
        else:
            racket_t_exact = vout_t
        blend = float(self.cfg.impact_inverse_tangent_blend)
        racket_t = (1.0 - blend) * vout_t + blend * racket_t_exact
        racket_vel = racket_n * normal + racket_t

        speed = torch.norm(racket_vel, dim=-1, keepdim=True)
        speed_scale = float(self.cfg.impact_inverse_racket_speed_scale)
        speed_bias = float(self.cfg.impact_inverse_racket_speed_bias)
        if abs(speed_scale - 1.0) >= 1.0e-6 or abs(speed_bias) >= 1.0e-6:
            direction = racket_vel / speed.clamp_min(1.0e-6)
            speed = speed * speed_scale + speed_bias
            racket_vel = direction * speed
        min_speed = float(self.cfg.impact_inverse_min_racket_speed)
        max_speed = float(self.cfg.impact_inverse_max_racket_speed)
        clamped = torch.clamp(speed, min=min_speed, max=max_speed)
        racket_vel = racket_vel * (clamped / speed.clamp_min(1.0e-6))
        return racket_vel, normal

    def _resample_command(self, env_ids: Sequence[int], carry_previous: bool = False):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if carry_previous:
            self._update_ability_curriculum(env_ids)
            self.prev_swing_contact[env_ids] = self.current_swing_contact[env_ids]
            self.prev_swing_net_cross[env_ids] = self.current_swing_net_cross[env_ids]
            self.prev_swing_on_opponent[env_ids] = self.current_swing_on_opponent[env_ids]
            self._begin_cycle_v2_check(env_ids)
        else:
            self._record_reset_safety_events(env_ids)
            self.prev_swing_contact[env_ids] = False
            self.prev_swing_net_cross[env_ids] = False
            self.prev_swing_on_opponent[env_ids] = False
            self._reset_cycle_v2(env_ids)
            self.cycle_v2_streak[env_ids] = 0
        self.current_swing_contact[env_ids] = False
        self.current_swing_net_cross[env_ids] = False
        self.current_swing_on_opponent[env_ids] = False
        self.current_swing_attempt[env_ids] = False
        self.current_swing_unsafe[env_ids] = False
        self.current_swing_station_saturated[env_ids] = False
        self.steps_since_target_resample[env_ids] = 0
        self.target_just_resampled[env_ids] = True
        self._sample_targets(env_ids)

    def _record_reset_safety_events(self, env_ids: torch.Tensor) -> None:
        """Count unsafe true resets so the curriculum cannot learn only from survivors."""
        if (
            not bool(self.cfg.ability_curriculum_enabled)
            or not self._ability_curriculum_initialized
            or len(env_ids) == 0
        ):
            return
        valid = self.steps_since_target_resample[env_ids] > 0
        unsafe = valid & self.current_swing_unsafe[env_ids]
        if not bool(torch.any(unsafe)):
            return
        rate = min(max(float(self.cfg.ability_curriculum_ema_rate), 0.0), 1.0)
        unsafe_fraction = unsafe.float().sum() / valid.float().sum().clamp_min(1.0)
        observed_safe = 1.0 - unsafe_fraction
        self._ability_safety_ema.lerp_(observed_safe, rate)

    def _cycle_v2_outcome_tier(self) -> int:
        """Return 0=contact, 1=net, 2=opponent bounce for the current ability."""
        if not bool(self.cfg.cycle_v2_outcome_by_ability):
            return {"contact": 0, "net_cross": 1, "opponent_bounce": 2}[str(self.cfg.cycle_v2_required_outcome)]
        level = float(torch.clamp(self._ability_curriculum_level, 0.0, 1.0).item())
        if level < float(self.cfg.cycle_v2_net_outcome_level):
            return 0
        if level < float(self.cfg.cycle_v2_bounce_outcome_level):
            return 1
        return 2

    def _cycle_v2_outcome(self, env_ids: torch.Tensor) -> torch.Tensor:
        tier = self._cycle_v2_outcome_tier()
        self.cycle_v2_outcome_tier[env_ids] = tier
        if tier == 0:
            return self.current_swing_contact[env_ids]
        if tier == 1:
            return self.current_swing_net_cross[env_ids]
        if tier == 2:
            return self.current_swing_on_opponent[env_ids]
        raise RuntimeError(f"Unsupported cycle-v2 outcome tier {tier}")

    def _reset_cycle_v2(self, env_ids: torch.Tensor) -> None:
        self.cycle_v2_pending[env_ids] = False
        self.cycle_v2_attempt_latch[env_ids] = False
        self.cycle_v2_ready_success_latch[env_ids] = False
        self.cycle_v2_ready_fail_latch[env_ids] = False
        self.cycle_v2_attempt_event[env_ids] = False
        self.cycle_v2_ready_success_event[env_ids] = False
        self.cycle_v2_ready_fail_event[env_ids] = False
        self.cycle_v2_unresolved_resample_fail_event[env_ids] = False
        self.cycle_v2_command_visible_steps[env_ids] = 0
        self.cycle_v2_elapsed_steps[env_ids] = 0
        self.cycle_v2_ready_ok[env_ids] = False
        self.cycle_v2_ready_score[env_ids] = 0.0
        self.cycle_v2_visible[env_ids] = False
        self.cycle_v2_ready_score_ok[env_ids] = False
        self.cycle_v2_height_ok[env_ids] = False
        self.cycle_v2_upright_ok[env_ids] = False
        self.cycle_v2_base_lin_vel_ok[env_ids] = False
        self.cycle_v2_base_ang_vel_ok[env_ids] = False
        self.cycle_v2_feet_ok[env_ids] = False
        self.cycle_v2_core_ready_ok[env_ids] = False
        self.cycle_v2_outcome_tier[env_ids] = -1

    def _begin_cycle_v2_check(self, env_ids: torch.Tensor) -> None:
        unresolved_fail = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if bool(self.cfg.cycle_v2_enabled) and bool(self.cfg.cycle_v2_fail_unresolved_on_resample):
            unresolved_fail = (
                self.cycle_v2_pending[env_ids]
                & self.cycle_v2_attempt_latch[env_ids]
                & (~self.cycle_v2_ready_success_latch[env_ids])
                & (~self.cycle_v2_ready_fail_latch[env_ids])
            )
            if bool(torch.any(unresolved_fail)):
                fail_event = torch.zeros_like(self.cycle_v2_pending)
                fail_event[env_ids] = unresolved_fail
                self.cycle_v2_ready_fail_latch[env_ids] |= unresolved_fail
                self.cycle_v2_streak[env_ids[unresolved_fail]] = 0
                self._update_cycle_v2_ability_events(torch.zeros_like(fail_event), fail_event)

        self._reset_cycle_v2(env_ids)
        self.cycle_v2_unresolved_resample_fail_event[env_ids] = unresolved_fail
        if not bool(self.cfg.cycle_v2_enabled):
            return
        attempt = self._cycle_v2_outcome(env_ids)
        self.cycle_v2_streak[env_ids[~attempt]] = 0
        self.cycle_v2_pending[env_ids] = attempt
        self.cycle_v2_attempt_latch[env_ids] = attempt
        self.cycle_v2_attempt_event[env_ids] = attempt

    def _update_ability_curriculum(self, env_ids: torch.Tensor) -> None:
        """Advance ball/planner difficulty from measured ability, not iteration count."""
        if not bool(self.cfg.ability_curriculum_enabled) or len(env_ids) == 0:
            return
        rate = min(max(float(self.cfg.ability_curriculum_ema_rate), 0.0), 1.0)
        if rate <= 0.0:
            return

        contact = self.current_swing_contact[env_ids].float().mean()
        net = self.current_swing_net_cross[env_ids].float().mean()
        success = self.current_swing_on_opponent[env_ids].float().mean()
        attempt = self.current_swing_attempt[env_ids].float().mean()
        safety = (~self.current_swing_unsafe[env_ids]).float().mean()
        station_saturation = self.current_swing_station_saturated[env_ids].float().mean()
        side = self.swing_sign[env_ids]
        forehand = side >= 0.0
        backhand = side < 0.0

        def _masked_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
            if bool(torch.any(mask)):
                return values[mask].float().mean()
            return fallback.detach()

        contact_values = self.current_swing_contact[env_ids].float()
        net_values = self.current_swing_net_cross[env_ids].float()
        success_values = self.current_swing_on_opponent[env_ids].float()
        forehand_contact = _masked_mean(contact_values, forehand, contact)
        forehand_net = _masked_mean(net_values, forehand, net)
        forehand_success = _masked_mean(success_values, forehand, success)
        backhand_contact = _masked_mean(contact_values, backhand, contact)
        backhand_net = _masked_mean(net_values, backhand, net)
        backhand_success = _masked_mean(success_values, backhand, success)

        ready_score = self.metrics["recovery_ready_score"][env_ids].mean()
        healthy = (
            (self.metrics["recovery_height_error"][env_ids] <= float(self.cfg.ability_curriculum_max_height_error))
            & (self.metrics["recovery_upright_error"][env_ids] <= float(self.cfg.ability_curriculum_max_upright_error))
        ).float().mean()
        recovery = 0.7 * ready_score + 0.3 * healthy
        cycle_v2_event_mode = (
            bool(self.cfg.ability_curriculum_use_cycle_v2)
            and str(self.cfg.ability_curriculum_cycle_v2_update_mode) == "events"
        )
        if bool(self.cfg.ability_curriculum_use_cycle_v2) and not cycle_v2_event_mode:
            attempts = self.cycle_v2_attempt_latch[env_ids].float()
            successes = self.cycle_v2_ready_success_latch[env_ids].float()
            attempt_count = attempts.sum()
            cycle_attempt = attempts.mean()
            if float(attempt_count.item()) > 0.0:
                cycle = successes.sum() / attempt_count
            else:
                cycle = torch.zeros((), dtype=torch.float32, device=self.device)
        elif cycle_v2_event_mode:
            cycle_attempt = self._ability_cycle_attempt_ema.detach()
            cycle = self._ability_cycle_ema.detach()
        else:
            cycle_attempt = success
            cycle = success * (recovery >= float(self.cfg.ability_curriculum_cycle_ready_threshold)).float()

        if not self._ability_curriculum_initialized:
            self._ability_contact_ema.copy_(contact)
            self._ability_net_ema.copy_(net)
            self._ability_success_ema.copy_(success)
            self._ability_recovery_ema.copy_(recovery)
            self._ability_cycle_ema.copy_(cycle)
            self._ability_cycle_attempt_ema.copy_(cycle_attempt)
            self._ability_cycle_resolved_ema.copy_(torch.zeros((), dtype=torch.float32, device=self.device))
            self._ability_forehand_contact_ema.copy_(forehand_contact)
            self._ability_forehand_net_ema.copy_(forehand_net)
            self._ability_forehand_success_ema.copy_(forehand_success)
            self._ability_backhand_contact_ema.copy_(backhand_contact)
            self._ability_backhand_net_ema.copy_(backhand_net)
            self._ability_backhand_success_ema.copy_(backhand_success)
            self._ability_attempt_ema.copy_(attempt)
            self._ability_safety_ema.copy_(safety)
            self._ability_station_saturation_ema.copy_(station_saturation)
            self._ability_curriculum_initialized = True
        else:
            self._ability_contact_ema.lerp_(contact, rate)
            self._ability_net_ema.lerp_(net, rate)
            self._ability_success_ema.lerp_(success, rate)
            self._ability_recovery_ema.lerp_(recovery, rate)
            self._ability_forehand_contact_ema.lerp_(forehand_contact, rate)
            self._ability_forehand_net_ema.lerp_(forehand_net, rate)
            self._ability_forehand_success_ema.lerp_(forehand_success, rate)
            self._ability_backhand_contact_ema.lerp_(backhand_contact, rate)
            self._ability_backhand_net_ema.lerp_(backhand_net, rate)
            self._ability_backhand_success_ema.lerp_(backhand_success, rate)
            self._ability_attempt_ema.lerp_(attempt, rate)
            self._ability_safety_ema.lerp_(safety, rate)
            self._ability_station_saturation_ema.lerp_(station_saturation, rate)
            if not cycle_v2_event_mode:
                self._ability_cycle_ema.lerp_(cycle, rate)
                self._ability_cycle_attempt_ema.lerp_(cycle_attempt, rate)
                self._ability_cycle_resolved_ema.lerp_(cycle_attempt, rate)

        can_advance = (
            (self._ability_contact_ema >= float(self.cfg.ability_curriculum_contact_threshold))
            & (self._ability_net_ema >= float(self.cfg.ability_curriculum_net_threshold))
            & (self._ability_success_ema >= float(self.cfg.ability_curriculum_success_threshold))
            & (self._ability_recovery_ema >= float(self.cfg.ability_curriculum_recovery_threshold))
        )
        if bool(self.cfg.ability_curriculum_require_side_contact):
            can_advance = can_advance & (
                self._ability_forehand_contact_ema >= float(self.cfg.ability_curriculum_forehand_contact_threshold)
            )
            can_advance = can_advance & (
                self._ability_backhand_contact_ema >= float(self.cfg.ability_curriculum_backhand_contact_threshold)
            )
        if bool(self.cfg.ability_curriculum_require_side_net):
            can_advance = can_advance & (
                self._ability_forehand_net_ema >= float(self.cfg.ability_curriculum_forehand_net_threshold)
            )
            can_advance = can_advance & (
                self._ability_backhand_net_ema >= float(self.cfg.ability_curriculum_backhand_net_threshold)
            )
        if bool(self.cfg.ability_curriculum_require_side_success):
            can_advance = can_advance & (
                self._ability_forehand_success_ema >= float(self.cfg.ability_curriculum_forehand_success_threshold)
            )
            can_advance = can_advance & (
                self._ability_backhand_success_ema >= float(self.cfg.ability_curriculum_backhand_success_threshold)
            )
        if bool(self.cfg.ability_curriculum_use_cycle_success):
            can_advance = can_advance & (
                self._ability_cycle_ema >= float(self.cfg.ability_curriculum_cycle_success_threshold)
            )
        if bool(self.cfg.ability_curriculum_use_cycle_v2):
            can_advance = can_advance & (
                self._ability_cycle_ema >= float(self.cfg.ability_curriculum_cycle_success_threshold)
            )
            can_advance = can_advance & (
                self._ability_cycle_attempt_ema >= float(self.cfg.ability_curriculum_cycle_attempt_threshold)
            )
        if bool(self.cfg.ability_curriculum_use_safety_gates):
            can_advance = can_advance & (
                self._ability_attempt_ema >= float(self.cfg.ability_curriculum_attempt_threshold)
            )
            can_advance = can_advance & (
                self._ability_safety_ema >= float(self.cfg.ability_curriculum_safety_threshold)
            )
            can_advance = can_advance & (
                self._ability_station_saturation_ema
                <= float(self.cfg.ability_curriculum_station_saturation_threshold)
            )
        should_regress = self._ability_recovery_ema < float(self.cfg.ability_curriculum_recovery_floor)
        outcome_regress_ratio = float(self.cfg.ability_curriculum_outcome_regress_ratio)
        if outcome_regress_ratio > 0.0:
            should_regress = should_regress | (
                self._ability_contact_ema
                < outcome_regress_ratio * float(self.cfg.ability_curriculum_contact_threshold)
            )
            should_regress = should_regress | (
                self._ability_net_ema
                < outcome_regress_ratio * float(self.cfg.ability_curriculum_net_threshold)
            )
            should_regress = should_regress | (
                self._ability_success_ema
                < outcome_regress_ratio * float(self.cfg.ability_curriculum_success_threshold)
            )
            if bool(self.cfg.ability_curriculum_require_side_contact):
                should_regress = should_regress | (
                    self._ability_forehand_contact_ema
                    < outcome_regress_ratio
                    * float(self.cfg.ability_curriculum_forehand_contact_threshold)
                )
                should_regress = should_regress | (
                    self._ability_backhand_contact_ema
                    < outcome_regress_ratio
                    * float(self.cfg.ability_curriculum_backhand_contact_threshold)
                )
            if bool(self.cfg.ability_curriculum_require_side_net):
                should_regress = should_regress | (
                    self._ability_forehand_net_ema
                    < outcome_regress_ratio * float(self.cfg.ability_curriculum_forehand_net_threshold)
                )
                should_regress = should_regress | (
                    self._ability_backhand_net_ema
                    < outcome_regress_ratio * float(self.cfg.ability_curriculum_backhand_net_threshold)
                )
        if bool(self.cfg.ability_curriculum_use_safety_gates):
            should_regress = should_regress | (
                self._ability_safety_ema < float(self.cfg.ability_curriculum_safety_floor)
            )
            should_regress = should_regress | (
                self._ability_station_saturation_ema
                > float(self.cfg.ability_curriculum_station_saturation_regress_threshold)
            )
        if bool(can_advance):
            self._ability_curriculum_level += float(self.cfg.ability_curriculum_advance_rate)
        elif bool(should_regress):
            self._ability_curriculum_level -= float(self.cfg.ability_curriculum_regress_rate)
        self._ability_curriculum_level.clamp_(0.0, 1.0)

    def _update_cycle_v2_ability_events(self, success_event: torch.Tensor, fail_event: torch.Tensor) -> None:
        """Update cycle ability from all cycle-v2 events, including failures that may reset later."""
        if (
            not bool(self.cfg.ability_curriculum_enabled)
            or not bool(self.cfg.ability_curriculum_use_cycle_v2)
            or str(self.cfg.ability_curriculum_cycle_v2_update_mode) != "events"
            or not self._ability_curriculum_initialized
        ):
            return
        rate = min(max(float(self.cfg.ability_curriculum_ema_rate), 0.0), 1.0)
        if rate <= 0.0:
            return

        # Event mode uses the current all-env cycle latches instead of only the
        # envs that survive to the next wrap.  Pending cycles count as not-yet
        # successful, so the curriculum cannot unlock from a few early successes
        # before the failure deadline has had a chance to fire.
        attempts = self.cycle_v2_attempt_latch.float()
        attempt_count = attempts.sum()
        attempt_rate = attempts.mean()
        if float(attempt_count.item()) > 0.0:
            success_rate = self.cycle_v2_ready_success_latch.float().sum() / attempt_count
            resolved_rate = (
                (self.cycle_v2_ready_success_latch | self.cycle_v2_ready_fail_latch).float().sum()
                / attempt_count
            )
        else:
            success_rate = torch.zeros((), dtype=torch.float32, device=self.device)
            resolved_rate = torch.zeros((), dtype=torch.float32, device=self.device)
        self._ability_cycle_attempt_ema.lerp_(attempt_rate, rate)
        self._ability_cycle_ema.lerp_(success_rate, rate)
        self._ability_cycle_resolved_ema.lerp_(resolved_rate, rate)

    # --- per-step updates ----------------------------------------------------------------------- #
    def _compute_strike_timing(self):
        motion = self._motion()
        ml = motion.motion
        if self._strike_phase_per_clip is None:
            sp = tuple(self.cfg.strike_phase_per_clip)
            if sp and len(sp) == ml.num_segments:
                self._strike_phase_per_clip = torch.tensor([float(x) for x in sp], device=self.device)
            else:
                self._strike_phase_per_clip = torch.full((ml.num_segments,), float(self.cfg.strike_phase), device=self.device)
        clip = motion.clip_id
        seg_start = ml.seg_start[clip]
        seg_len = ml.seg_len[clip]
        phase = self._strike_phase_per_clip[clip]
        strike_step = seg_start + (phase * (seg_len - 1).float()).round().long()
        self.true_time_to_strike = (strike_step - motion.time_steps).float() * self._env.step_dt
        self.time_to_strike = self.true_time_to_strike + self._planner_tts_offset
        self.pre_strike = self.true_time_to_strike > 0.0
        self.strike_window = self.true_time_to_strike.abs() <= self.cfg.strike_window_s
        self._apply_no_command_ready_targets()

    def _compute_racket_state(self):
        data = self.robot.data
        if self._racket_mode == "body":
            idx = self._racket_body_index
            self.racket_pos_w = data.body_pos_w[:, idx]
            self.racket_quat_w = data.body_quat_w[:, idx]
            self.racket_lin_vel_w = data.body_lin_vel_w[:, idx]
        else:
            widx = self._wrist_body_index
            wpos = data.body_pos_w[:, widx]
            wquat = data.body_quat_w[:, widx]
            wlin = data.body_lin_vel_w[:, widx]
            wang = data.body_ang_vel_w[:, widx]
            offset_w = quat_apply(wquat, self._mount_offset)
            self.racket_pos_w = wpos + offset_w
            self.racket_lin_vel_w = wlin + torch.cross(wang, offset_w, dim=-1)
            self.racket_quat_w = quat_mul(wquat, self._mount_quat)
        # Face normal = a chosen local axis of the racket frame, times the striking-face sign (the
        # forehand and backhand strike with opposite faces).
        axis_w = matrix_from_quat(self.racket_quat_w)[:, :, self.cfg.mount_normal_axis]
        if self._mount_sign_per_clip is not None and self._motion()._multiseg:
            clip = self._motion().clip_id.clamp(max=self._mount_sign_per_clip.shape[0] - 1)
            sign = self._mount_sign_per_clip[clip].unsqueeze(-1)
        else:
            sign = self.cfg.mount_normal_sign
        self.racket_normal_w = axis_w * sign

    def _update_feet_contact(self):
        if self._contact_sensor is None or not self._foot_idx_contact:
            return
        forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._foot_idx_contact, :], dim=-1)
        in_contact = (forces > self.cfg.contact_force_threshold).float()
        self.feet_contact_frac = in_contact.mean(dim=-1)
        self.feet_contact_state.zero_()
        count = min(in_contact.shape[1], self.feet_contact_state.shape[1])
        self.feet_contact_state[:, :count] = in_contact[:, :count]

    def _recovery_arm_pose_score(self, motion: MotionCommand) -> torch.Tensor:
        body_names = tuple(self.cfg.recovery_diag_arm_body_names)
        if not body_names:
            return torch.ones(self.num_envs, device=self.device)
        ids = [motion.cfg.body_names.index(name) for name in body_names if name in motion.cfg.body_names]
        if not ids:
            return torch.ones(self.num_envs, device=self.device)
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)
        pos_err = torch.norm(motion.robot_body_pos_w[:, idx] - motion.body_pos_relative_w[:, idx], dim=-1).mean(dim=-1)
        pos = torch.exp(-torch.square(pos_err / max(float(self.cfg.recovery_diag_arm_pos_std), 1.0e-6)))

        q_ref = motion.body_quat_relative_w[:, idx].reshape(-1, 4)
        q_robot = motion.robot_body_quat_w[:, idx].reshape(-1, 4)
        ori_err = quat_error_magnitude(q_ref, q_robot).reshape(self.num_envs, len(ids)).mean(dim=-1)
        ori = torch.exp(-torch.square(ori_err / max(float(self.cfg.recovery_diag_arm_ori_std), 1.0e-6)))
        return 0.5 * pos + 0.5 * ori

    def _update_recovery_diagnostics(self, motion: MotionCommand):
        data = self.robot.data
        default_z = data.default_root_state[:, 2] + self._env.scene.env_origins[:, 2]
        height_error = torch.abs(data.root_pos_w[:, 2] - default_z)
        upright_error = torch.norm(data.projected_gravity_b[:, :2], dim=-1)
        base_lin_vel = torch.norm(data.root_lin_vel_w[:, :2], dim=-1)
        base_ang_vel = torch.norm(data.root_ang_vel_w, dim=-1)
        station_error = torch.norm(self.base_pos_w[:, :2] - self.station_w, dim=-1)
        racket_speed = torch.norm(self.racket_lin_vel_w, dim=-1)
        action = self._env.action_manager.action
        previous_action = self._env.action_manager.prev_action
        action_delta = action - previous_action
        action_term = self._env.action_manager.get_term("joint_pos")
        applied_raw_action = action_term.applied_raw_actions
        effective_action = action_term.effective_raw_actions
        overflow_action = action_term.overflow_actions
        position_clamped = torch.abs(overflow_action) > 1.0e-6
        action_joint_ids = list(action_term.action_joint_ids)
        q_des = action_term.processed_actions
        q_tracking_error = q_des - data.joint_pos[:, action_joint_ids]
        torque_requested = data.computed_torque[:, action_joint_ids]
        torque_applied = data.applied_torque[:, action_joint_ids]
        torque_clipped = torch.abs(torque_requested - torque_applied) > 1.0e-6

        def _group_rms(values: torch.Tensor, start: int, end: int) -> torch.Tensor:
            return torch.sqrt(torch.mean(torch.square(values[:, start:end]), dim=-1))

        def _group_fraction(values: torch.Tensor, start: int, end: int) -> torch.Tensor:
            return values[:, start:end].float().mean(dim=-1)

        waist_action = _group_rms(action, 0, 3)
        right_arm_action = _group_rms(action, 12, 19)
        leg_action = _group_rms(action, 19, 31)
        action_abs_max = torch.amax(torch.abs(action), dim=-1)
        applied_action_abs_max = torch.amax(torch.abs(applied_raw_action), dim=-1)
        effective_action_abs_max = torch.amax(torch.abs(effective_action), dim=-1)
        leg_action_abs_max = torch.amax(torch.abs(action[:, 19:31]), dim=-1)
        waist_action_delta = _group_rms(action_delta, 0, 3)
        right_arm_action_delta = _group_rms(action_delta, 12, 19)
        leg_action_delta = _group_rms(action_delta, 19, 31)
        joint_vel = motion.robot_joint_vel
        waist_joint_vel = _group_rms(joint_vel, 0, 3)
        right_arm_joint_vel = _group_rms(joint_vel, 12, 19)
        leg_joint_vel = _group_rms(joint_vel, 19, 31)

        height = torch.exp(-torch.square(height_error / max(float(self.cfg.recovery_diag_height_std), 1.0e-6)))
        upright = torch.exp(-torch.square(upright_error / max(float(self.cfg.recovery_diag_upright_std), 1.0e-6)))
        lin = torch.exp(-torch.square(base_lin_vel / max(float(self.cfg.recovery_diag_lin_vel_std), 1.0e-6)))
        ang = torch.exp(-torch.square(base_ang_vel / max(float(self.cfg.recovery_diag_ang_vel_std), 1.0e-6)))
        station = torch.exp(-torch.square(station_error / max(float(self.cfg.recovery_diag_station_std), 1.0e-6)))
        racket = torch.exp(-torch.square(racket_speed / max(float(self.cfg.recovery_diag_racket_vel_std), 1.0e-6)))
        arm = self._recovery_arm_pose_score(motion)
        feet = torch.clamp(self.feet_contact_frac, 0.0, 1.0)
        ready = 0.18 * height + 0.18 * upright + 0.15 * lin + 0.15 * ang + 0.12 * feet + 0.10 * station + 0.07 * racket + 0.05 * arm

        recovery_or_hold = ((~self.pre_strike) & (~self.strike_window)) | motion.in_hold
        early_next = self.pre_strike & (
            self.steps_since_target_resample <= int(self.cfg.recovery_diag_early_prestrike_steps)
        )
        phase_gate = (recovery_or_hold | early_next).float()

        self.metrics["recovery_height_error"] = height_error
        self.metrics["recovery_upright_error"] = upright_error
        self.metrics["recovery_base_lin_vel"] = base_lin_vel
        self.metrics["recovery_base_ang_vel"] = base_ang_vel
        self.metrics["recovery_station_error"] = station_error
        self.metrics["recovery_feet_contact_frac"] = feet
        self.metrics["recovery_racket_speed"] = racket_speed
        self.metrics["base_pitch_like_signed"] = data.projected_gravity_b[:, 0]
        self.metrics["base_backward_velocity"] = (-data.root_lin_vel_w[:, 0]).clamp_min(0.0)
        self.metrics["waist_action_rms"] = waist_action
        self.metrics["right_arm_action_rms"] = right_arm_action
        self.metrics["leg_action_rms"] = leg_action
        self.metrics["action_abs_max"] = action_abs_max
        self.metrics["applied_action_abs_max"] = applied_action_abs_max
        self.metrics["leg_action_abs_max"] = leg_action_abs_max
        self.metrics["effective_action_abs_max"] = effective_action_abs_max
        self.metrics["action_clamp_fraction"] = position_clamped.float().mean(dim=-1)
        self.metrics["waist_action_clamp_fraction"] = _group_fraction(position_clamped, 0, 3)
        self.metrics["right_arm_action_clamp_fraction"] = _group_fraction(position_clamped, 12, 19)
        self.metrics["leg_action_clamp_fraction"] = _group_fraction(position_clamped, 19, 31)
        self.metrics["waist_action_overflow_rms"] = _group_rms(overflow_action, 0, 3)
        self.metrics["right_arm_action_overflow_rms"] = _group_rms(overflow_action, 12, 19)
        self.metrics["leg_action_overflow_rms"] = _group_rms(overflow_action, 19, 31)
        self.metrics["waist_q_tracking_error_rms"] = _group_rms(q_tracking_error, 0, 3)
        self.metrics["right_arm_q_tracking_error_rms"] = _group_rms(q_tracking_error, 12, 19)
        self.metrics["leg_q_tracking_error_rms"] = _group_rms(q_tracking_error, 19, 31)
        self.metrics["waist_torque_clip_fraction"] = _group_fraction(torque_clipped, 0, 3)
        self.metrics["right_arm_torque_clip_fraction"] = _group_fraction(torque_clipped, 12, 19)
        self.metrics["leg_torque_clip_fraction"] = _group_fraction(torque_clipped, 19, 31)
        self.metrics["waist_action_delta_rms"] = waist_action_delta
        self.metrics["right_arm_action_delta_rms"] = right_arm_action_delta
        self.metrics["leg_action_delta_rms"] = leg_action_delta
        self.metrics["waist_joint_vel_rms"] = waist_joint_vel
        self.metrics["right_arm_joint_vel_rms"] = right_arm_joint_vel
        self.metrics["leg_joint_vel_rms"] = leg_joint_vel
        self.metrics["recovery_leg_action_abs_max"] = leg_action_abs_max * phase_gate
        self.metrics["recovery_leg_action_delta_rms"] = leg_action_delta * phase_gate
        self.metrics["recovery_leg_joint_vel_rms"] = leg_joint_vel * phase_gate
        self.metrics["recovery_height_score"] = height
        self.metrics["recovery_upright_score"] = upright
        self.metrics["recovery_lin_vel_score"] = lin
        self.metrics["recovery_ang_vel_score"] = ang
        self.metrics["recovery_station_score"] = station
        self.metrics["recovery_racket_score"] = racket
        self.metrics["recovery_arm_score"] = arm
        self.metrics["recovery_ready_score"] = ready
        self.metrics["recovery_phase_gate"] = phase_gate
        self.metrics["recovery_phase_ready_score"] = ready * phase_gate
        self.metrics["recovery_contact_ready_score"] = ready * phase_gate * self.current_swing_contact.float()
        self.metrics["recovery_net_ready_score"] = ready * phase_gate * self.current_swing_net_cross.float()
        self.metrics["recovery_return_ready_score"] = ready * phase_gate * self.current_swing_on_opponent.float()
        cycle_window = (
            self.pre_strike
            & (self.steps_since_target_resample <= int(self.cfg.cycle_success_diag_window_steps))
            & (~self.no_command_ready_active)
        )
        threshold = float(self.cfg.cycle_success_diag_ready_threshold)
        temperature = max(float(self.cfg.cycle_success_diag_ready_temperature), 1.0e-6)
        cycle_soft = torch.sigmoid((ready - threshold) / temperature)
        prev_success = self.prev_swing_on_opponent.float()
        self.metrics["cycle_success_ready_score"] = ready * cycle_window.float() * prev_success
        self.metrics["cycle_success_soft"] = cycle_soft * cycle_window.float() * prev_success
        self.metrics["cycle_success_binary"] = (
            (ready >= threshold).float() * cycle_window.float() * prev_success
        )
        self.metrics["no_command_ready_active"] = self.no_command_ready_active.float()

    def _update_cycle_v2_state(self) -> None:
        self.cycle_v2_ready_success_event.zero_()
        self.cycle_v2_ready_fail_event.zero_()

        if not bool(self.cfg.cycle_v2_enabled):
            self.cycle_v2_ready_ok.zero_()
            self.cycle_v2_ready_score.zero_()
            self.cycle_v2_visible.zero_()
            self.cycle_v2_ready_score_ok.zero_()
            self.cycle_v2_height_ok.zero_()
            self.cycle_v2_upright_ok.zero_()
            self.cycle_v2_base_lin_vel_ok.zero_()
            self.cycle_v2_base_ang_vel_ok.zero_()
            self.cycle_v2_feet_ok.zero_()
            self.cycle_v2_core_ready_ok.zero_()
            return

        pending = self.cycle_v2_pending.clone()
        if not bool(torch.any(pending)):
            self.cycle_v2_ready_ok.zero_()
            self.cycle_v2_ready_score.copy_(self.metrics["recovery_ready_score"])
            self.cycle_v2_visible.zero_()
            self.cycle_v2_ready_score_ok.zero_()
            self.cycle_v2_height_ok.zero_()
            self.cycle_v2_upright_ok.zero_()
            self.cycle_v2_base_lin_vel_ok.zero_()
            self.cycle_v2_base_ang_vel_ok.zero_()
            self.cycle_v2_feet_ok.zero_()
            self.cycle_v2_core_ready_ok.zero_()
            empty = torch.zeros_like(self.cycle_v2_pending)
            self._update_cycle_v2_ability_events(empty, empty)
            return

        visible = pending & self.pre_strike
        if not bool(self.cfg.cycle_v2_count_no_command_as_visible):
            visible = visible & (~self.no_command_ready_active)
        elapsed = pending & self.pre_strike
        self.cycle_v2_elapsed_steps[elapsed] += 1
        self.cycle_v2_command_visible_steps[visible] += 1

        ready_score = self.metrics["recovery_ready_score"]
        self.cycle_v2_ready_score.copy_(ready_score)
        ready_score_ok = pending & visible & (ready_score >= float(self.cfg.cycle_v2_ready_threshold))
        height_ok = pending & visible & (
            self.metrics["recovery_height_error"] <= float(self.cfg.cycle_v2_max_height_error)
        )
        upright_ok = pending & visible & (
            self.metrics["recovery_upright_error"] <= float(self.cfg.cycle_v2_max_upright_error)
        )
        base_lin_vel_ok = pending & visible & (
            self.metrics["recovery_base_lin_vel"] <= float(self.cfg.cycle_v2_max_base_lin_vel)
        )
        base_ang_vel_ok = pending & visible & (
            self.metrics["recovery_base_ang_vel"] <= float(self.cfg.cycle_v2_max_base_ang_vel)
        )
        feet_ok = pending & visible & (
            self.metrics["recovery_feet_contact_frac"] >= float(self.cfg.cycle_v2_min_feet_contact)
        )
        core_ready = (
            ready_score_ok & height_ok & upright_ok & base_lin_vel_ok & base_ang_vel_ok & feet_ok
        )
        self.cycle_v2_visible.copy_(visible)
        self.cycle_v2_ready_score_ok.copy_(ready_score_ok)
        self.cycle_v2_height_ok.copy_(height_ok)
        self.cycle_v2_upright_ok.copy_(upright_ok)
        self.cycle_v2_base_lin_vel_ok.copy_(base_lin_vel_ok)
        self.cycle_v2_base_ang_vel_ok.copy_(base_ang_vel_ok)
        self.cycle_v2_feet_ok.copy_(feet_ok)
        self.cycle_v2_core_ready_ok.copy_(core_ready)
        ready_mode = str(self.cfg.cycle_v2_ready_mode)
        if ready_mode == "all_hard":
            ready_ok = (
                core_ready
                & (self.metrics["recovery_station_error"] <= float(self.cfg.cycle_v2_max_station_error))
                & (self.metrics["recovery_racket_speed"] <= float(self.cfg.cycle_v2_max_racket_speed))
                & (self.metrics["recovery_arm_score"] >= float(self.cfg.cycle_v2_min_arm_score))
            )
        elif ready_mode == "core_hard_soft":
            ready_ok = core_ready
        else:
            raise ValueError(
                "cycle_v2_ready_mode must be one of 'all_hard' or 'core_hard_soft', "
                f"got {ready_mode!r}"
            )
        self.cycle_v2_ready_ok.copy_(ready_ok)

        success_event = pending & ready_ok & (~self.cycle_v2_ready_success_latch)
        remaining = pending & (~success_event)
        visible_timeout = (
            remaining
            & visible
            & (self.cycle_v2_command_visible_steps >= int(self.cfg.cycle_v2_visible_deadline_steps))
        )
        total_deadline = int(self.cfg.cycle_v2_total_deadline_steps)
        if total_deadline > 0:
            total_timeout = remaining & elapsed & (self.cycle_v2_elapsed_steps >= total_deadline)
        else:
            total_timeout = torch.zeros_like(remaining)
        if bool(self.cfg.cycle_v2_fail_on_strike_without_ready):
            strike_timeout = remaining & self.strike_window
        else:
            strike_timeout = torch.zeros_like(remaining)
        fail_event = remaining & (visible_timeout | total_timeout | strike_timeout)

        self.cycle_v2_ready_success_event.copy_(success_event)
        self.cycle_v2_ready_fail_event.copy_(fail_event)
        self.cycle_v2_ready_success_latch |= success_event
        self.cycle_v2_ready_fail_latch |= fail_event
        self.cycle_v2_streak[success_event] += 1
        self.cycle_v2_streak[fail_event] = 0
        self.cycle_v2_pending[success_event | fail_event] = False
        self._update_cycle_v2_ability_events(success_event, fail_event)

    def _evaluate_return(self):
        """Simple no-spin outgoing-ball evaluation at the exact strike frame (contact/net/bounce).

        The paddle is assumed to carry the ball off at the achieved racket velocity (no spin). The
        outgoing flight is a gravity-only ballistic arc; net clearance and the first table bounce are
        solved in closed form. All quantities are example approximations for training shaping.
        """
        exact = self.true_time_to_strike.abs() <= (0.5 * self._env.step_dt + 1e-6)
        self.strike_fired = exact

        pos_err = torch.norm(self.racket_pos_w - self.ball_strike_pos_w, dim=-1)
        self.racket_target_distance = pos_err
        # Contact requires the racket to be near the target and moving through the strike.  The
        # target-velocity mode avoids a singular "toward the target point" test when the racket is
        # already at the contact point.
        if self.cfg.contact_approach_mode == "target_velocity":
            approach_dir = self.racket_impact_target_vel_w / (
                torch.norm(self.racket_impact_target_vel_w, dim=-1, keepdim=True) + 1e-6
            )
        elif self.cfg.contact_approach_mode == "target_point":
            to_target = self.ball_strike_pos_w - self.racket_pos_w
            approach_dir = to_target / (torch.norm(to_target, dim=-1, keepdim=True) + 1e-6)
        else:
            raise ValueError(f"Unsupported contact_approach_mode: {self.cfg.contact_approach_mode}")
        approach = torch.sum(self.racket_lin_vel_w * approach_dir, dim=-1)
        attempt = exact & (
            torch.norm(self.racket_lin_vel_w, dim=-1)
            >= float(self.cfg.ability_curriculum_attempt_min_racket_speed)
        )
        self.current_swing_attempt |= attempt
        contact = exact & (pos_err < self.cfg.contact_radius) & (approach > self.cfg.min_approach_speed)

        if self.cfg.return_model == "racket_velocity":
            out_vel = self.racket_lin_vel_w
        elif self.cfg.return_model == "moving_racket_impact":
            out_vel, rel_normal_speed = self._moving_racket_impact_velocity()
            contact = contact & ((-rel_normal_speed) > float(self.cfg.min_normal_closing_speed))
        else:
            raise ValueError(f"Unsupported return_model: {self.cfg.return_model}")
        self.impact_ball_out_vel_w = out_vel
        self.impact_ball_out_error = torch.norm(out_vel - self.ball_outgoing_target_vel_w, dim=-1)

        # Outgoing ballistic arc from the true ball strike point (env-local frame) at the predicted
        # post-impact ball velocity.
        p0 = self.ball_strike_pos_w - self._env.scene.env_origins
        v = out_vel
        x0, y0, z0 = p0[:, 0], p0[:, 1], p0[:, 2]
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

        near_x = float(self.cfg.table_near_x)
        net_x = near_x + float(self.cfg.net_x)
        far_x = near_x + float(self.cfg.table_length)
        half_w = 0.5 * float(self.cfg.table_width)
        surface_z = float(self.cfg.table_surface_z)
        net_top = surface_z + float(self.cfg.net_height) + float(self.cfg.net_margin)
        center_y = self.fixed_station_w[:, 1] - self._env.scene.env_origins[:, 1]  # env-local table center y

        # Net crossing height (ball travels in +x toward the opponent).
        moving_fwd = vx > 0.1
        t_net = (net_x - x0) / vx.clamp_min(1e-3)
        z_net = z0 + vz * t_net - 0.5 * _GRAVITY * t_net**2
        net_cross = contact & moving_fwd & (t_net > 0) & (z_net > net_top)

        # First bounce on the table surface (descending root of the ballistic arc).
        disc = (vz**2 + 2.0 * _GRAVITY * (z0 - surface_z)).clamp_min(0.0)
        t_bounce = (vz + torch.sqrt(disc)) / _GRAVITY
        land_x = x0 + vx * t_bounce
        land_y = y0 + vy * t_bounce
        on_opponent = net_cross & (land_x > net_x) & (land_x < far_x) & ((land_y - center_y).abs() < half_w)

        self.ball_contact = contact
        self.ball_net_cross = net_cross
        self.ball_on_opponent = on_opponent
        self.current_swing_contact[:] = self.current_swing_contact | contact
        self.current_swing_net_cross[:] = self.current_swing_net_cross | net_cross
        self.current_swing_on_opponent[:] = self.current_swing_on_opponent | on_opponent

    def _update_metrics(self):
        # Timing + FK must be fresh before the reward reads them (motion updated first this step).
        self._compute_strike_timing()
        self._compute_racket_state()
        self._update_feet_contact()
        self._evaluate_return()
        self._update_recovery_diagnostics(self._motion())
        unsafe = (
            self.robot.data.root_pos_w[:, 2] < float(self.cfg.ability_curriculum_safe_min_base_height)
        ) | (
            torch.norm(self.robot.data.projected_gravity_b[:, :2], dim=-1)
            > float(self.cfg.ability_curriculum_safe_max_tilt)
        )
        self.current_swing_unsafe |= unsafe
        self._update_cycle_v2_state()
        self.metrics["racket_pos_error"] = torch.where(
            self.strike_window, self.racket_target_distance, self.metrics["racket_pos_error"]
        )
        self.metrics["racket_vel_error"] = torch.where(
            self.strike_window,
            torch.norm(self.racket_lin_vel_w - self.racket_impact_target_vel_w, dim=-1),
            self.metrics["racket_vel_error"],
        )
        self.metrics["time_to_strike"] = self.time_to_strike
        self.metrics["return_success"] = torch.where(
            self.strike_fired, self.ball_on_opponent.float(), self.metrics["return_success"]
        )
        self.metrics["prev_return_success"] = self.prev_swing_on_opponent.float()
        self.metrics["prev_net_cross"] = self.prev_swing_net_cross.float()
        self.metrics["current_net_cross"] = self.current_swing_net_cross.float()
        self.metrics["racket_pos_curriculum_scale"] = torch.full_like(
            self.metrics["racket_pos_curriculum_scale"], float(torch.mean(self._racket_pos_curriculum_scale()).item())
        )
        self.metrics["steps_since_target_resample"] = self.steps_since_target_resample.float()
        self.metrics["ability_curriculum_level"] = torch.full_like(
            self.metrics["ability_curriculum_level"], float(self._ability_curriculum_level.item())
        )
        self.metrics["ability_curriculum_scale"] = torch.full_like(
            self.metrics["ability_curriculum_scale"], float(torch.mean(self._racket_pos_curriculum_scale()).item())
        )
        self.metrics["ability_contact_ema"] = torch.full_like(
            self.metrics["ability_contact_ema"], float(self._ability_contact_ema.item())
        )
        self.metrics["ability_net_ema"] = torch.full_like(
            self.metrics["ability_net_ema"], float(self._ability_net_ema.item())
        )
        self.metrics["ability_success_ema"] = torch.full_like(
            self.metrics["ability_success_ema"], float(self._ability_success_ema.item())
        )
        self.metrics["ability_recovery_ema"] = torch.full_like(
            self.metrics["ability_recovery_ema"], float(self._ability_recovery_ema.item())
        )
        self.metrics["ability_cycle_ema"] = torch.full_like(
            self.metrics["ability_cycle_ema"], float(self._ability_cycle_ema.item())
        )
        self.metrics["ability_cycle_attempt_ema"] = torch.full_like(
            self.metrics["ability_cycle_attempt_ema"], float(self._ability_cycle_attempt_ema.item())
        )
        self.metrics["ability_cycle_resolved_ema"] = torch.full_like(
            self.metrics["ability_cycle_resolved_ema"], float(self._ability_cycle_resolved_ema.item())
        )
        self.metrics["ability_forehand_contact_ema"] = torch.full_like(
            self.metrics["ability_forehand_contact_ema"], float(self._ability_forehand_contact_ema.item())
        )
        self.metrics["ability_forehand_net_ema"] = torch.full_like(
            self.metrics["ability_forehand_net_ema"], float(self._ability_forehand_net_ema.item())
        )
        self.metrics["ability_forehand_success_ema"] = torch.full_like(
            self.metrics["ability_forehand_success_ema"], float(self._ability_forehand_success_ema.item())
        )
        self.metrics["ability_backhand_contact_ema"] = torch.full_like(
            self.metrics["ability_backhand_contact_ema"], float(self._ability_backhand_contact_ema.item())
        )
        self.metrics["ability_backhand_net_ema"] = torch.full_like(
            self.metrics["ability_backhand_net_ema"], float(self._ability_backhand_net_ema.item())
        )
        self.metrics["ability_backhand_success_ema"] = torch.full_like(
            self.metrics["ability_backhand_success_ema"], float(self._ability_backhand_success_ema.item())
        )
        self.metrics["ability_attempt_ema"] = torch.full_like(
            self.metrics["ability_attempt_ema"], float(self._ability_attempt_ema.item())
        )
        self.metrics["ability_safety_ema"] = torch.full_like(
            self.metrics["ability_safety_ema"], float(self._ability_safety_ema.item())
        )
        self.metrics["ability_station_saturation_ema"] = torch.full_like(
            self.metrics["ability_station_saturation_ema"],
            float(self._ability_station_saturation_ema.item()),
        )
        self.metrics["current_swing_attempt"] = self.current_swing_attempt.float()
        self.metrics["current_swing_unsafe"] = self.current_swing_unsafe.float()
        self.metrics["current_swing_station_saturated"] = (
            self.current_swing_station_saturated.float()
        )
        self.metrics["cycle_v2_pending"] = self.cycle_v2_pending.float()
        self.metrics["cycle_v2_attempt_latch"] = self.cycle_v2_attempt_latch.float()
        self.metrics["cycle_v2_ready_success_latch"] = self.cycle_v2_ready_success_latch.float()
        self.metrics["cycle_v2_ready_fail_latch"] = self.cycle_v2_ready_fail_latch.float()
        self.metrics["cycle_v2_attempt_event"] = self.cycle_v2_attempt_event.float()
        self.metrics["cycle_v2_ready_success_event"] = self.cycle_v2_ready_success_event.float()
        cycle_v2_fail_event = self.cycle_v2_ready_fail_event | self.cycle_v2_unresolved_resample_fail_event
        self.metrics["cycle_v2_ready_fail_event"] = cycle_v2_fail_event.float()
        self.metrics["cycle_v2_unresolved_resample_fail_event"] = (
            self.cycle_v2_unresolved_resample_fail_event.float()
        )
        self.metrics["cycle_v2_resolved_event"] = (
            self.cycle_v2_ready_success_event | cycle_v2_fail_event
        ).float()
        self.metrics["cycle_v2_command_visible_steps"] = self.cycle_v2_command_visible_steps.float()
        self.metrics["cycle_v2_elapsed_steps"] = self.cycle_v2_elapsed_steps.float()
        self.metrics["cycle_v2_ready_score"] = self.cycle_v2_ready_score
        self.metrics["cycle_v2_ready_ok"] = self.cycle_v2_ready_ok.float()
        self.metrics["cycle_v2_visible"] = self.cycle_v2_visible.float()
        self.metrics["cycle_v2_ready_score_ok"] = self.cycle_v2_ready_score_ok.float()
        self.metrics["cycle_v2_height_ok"] = self.cycle_v2_height_ok.float()
        self.metrics["cycle_v2_upright_ok"] = self.cycle_v2_upright_ok.float()
        self.metrics["cycle_v2_base_lin_vel_ok"] = self.cycle_v2_base_lin_vel_ok.float()
        self.metrics["cycle_v2_base_ang_vel_ok"] = self.cycle_v2_base_ang_vel_ok.float()
        self.metrics["cycle_v2_feet_ok"] = self.cycle_v2_feet_ok.float()
        self.metrics["cycle_v2_core_ready_ok"] = self.cycle_v2_core_ready_ok.float()
        self.metrics["cycle_v2_outcome_tier"] = self.cycle_v2_outcome_tier.float()
        self.metrics["cycle_v2_streak"] = self.cycle_v2_streak.float()

    def _update_command(self):
        self.cycle_v2_attempt_event.zero_()
        self.cycle_v2_unresolved_resample_fail_event.zero_()
        self.steps_since_target_resample += 1
        self.target_just_resampled.zero_()
        self._compute_strike_timing()
        # Re-sample the target at each new swing (the motion command sets just_resampled this step
        # when it wrapped a swing). Reset-time resampling is handled by the manager's reset -> _resample.
        motion = self._motion()
        wrapped = torch.where(motion.just_resampled)[0]
        if len(wrapped) > 0:
            self._resample_command(wrapped, carry_previous=True)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


def _boxes_to_tensor(per_clip, device):
    """Convert ((xlo,xhi),(ylo,yhi),(zlo,zhi)) x num_clips into an (C, 3, 2) tensor, or None."""
    if per_clip is None:
        return None
    return torch.tensor(
        [[[float(lo), float(hi)] for (lo, hi) in clip_rng] for clip_rng in per_clip],
        dtype=torch.float32,
        device=device,
    )


@configclass
class RacketTargetCommandCfg(CommandTermCfg):
    """Configuration for :class:`RacketTargetCommand`."""

    class_type: type = RacketTargetCommand

    asset_name: str = MISSING
    motion_command_name: str = "motion"
    # Targets are re-sampled per swing (on wrap / reset), not on a timer.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    # --- racket mount FK ---
    racket_body_name: str = "pingpang_red_Link"
    wrist_body_name: str = "right_wrist_yaw_Link"
    mount_offset: tuple[float, float, float] = (0.21, 0.032, 0.032)
    mount_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mount_normal_axis: int = 1  # racket-local +Y is the blade face normal
    mount_normal_sign: float = 1.0
    # Per-clip striking-face sign (forehand and backhand strike with opposite faces), e.g. (1.0, -1.0).
    mount_normal_sign_per_clip: tuple = ()
    # Per-clip swing side exposed to the actor: forehand +1, backhand -1.
    swing_side_per_clip: tuple = ()

    # --- station target ---
    station_nominal_offset_xy: tuple[float, float] = (0.0, 0.0)
    station_mode: str = "fixed"  # fixed | dynamic_from_motion
    dynamic_station_xy_clip: tuple = ((-0.08, 0.08), (-0.25, 0.25))
    dynamic_station_blend: float = 1.0
    # Deploy lifecycle alignment.  With probability deploy_ready_hold_prob, a pre-swing hold exposes
    # the same no-command READY observation as the reference deploy runner: base-relative ready reach,
    # zero racket velocity, fixed ready time-to-strike, and fixed station.  The sampled strike command
    # is kept hidden and restored when the hold ends.  Default 0.0 preserves all existing tasks.
    deploy_ready_hold_prob: float = 0.0
    deploy_ready_reach: tuple[float, float, float] = (0.40, 0.20, -0.05)
    deploy_ready_velocity_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    deploy_ready_normal_w: tuple[float, float, float] = (1.0, 0.0, 0.0)
    deploy_ready_time_to_strike: float = 1.0

    # --- feet (for the contact fraction used by the follow-through/recovery reward) ---
    feet_body_names: tuple[str, ...] = ("left_ankle_roll_Link", "right_ankle_roll_Link")
    contact_force_threshold: float = 10.0

    # --- strike timing (fraction of the reference clip at which the paddle meets the ball) ---
    strike_phase: float = 0.5
    strike_phase_per_clip: tuple = ()  # e.g. (0.47, 0.33); empty -> scalar strike_phase for every clip
    strike_window_s: float = 0.12  # half-window in which the racket-tracking rewards are active

    # --- racket target boxes ---
    # x/y are STATION-RELATIVE (fixed striking plane in front + swing-side band), z is absolute height.
    racket_pos_range: tuple = ((0.45, 0.55), (-0.35, 0.35), (0.7, 1.1))
    racket_vel_range: tuple = ((1.0, 2.5), (-1.5, 1.5), (0.0, 1.0))
    racket_velocity_mode: str = "range"  # range | ballistic_landing | impact_inverse_landing
    # ``table_workspace`` samples the physical strike route from regulation-table geometry and only
    # uses the selected motion clip as a side-compatible action/station prior.
    strike_position_mode: str = "motion_box"  # motion_box | table_workspace
    table_workspace_edge_margin: float = 0.04
    table_workspace_side_overlap: float = 0.12
    table_workspace_forehand_core_y_range: tuple[float, float] = (-0.55, -0.20)
    table_workspace_backhand_core_y_range: tuple[float, float] = (-0.12, 0.18)
    table_workspace_x_jitter_core_range: tuple[float, float] = (-0.015, 0.015)
    table_workspace_x_jitter_full_range: tuple[float, float] = (-0.04, 0.04)
    table_workspace_z_core_above_surface_range: tuple[float, float] = (0.40, 0.58)
    table_workspace_z_full_above_surface_range: tuple[float, float] = (0.34, 0.70)
    table_workspace_fringe_prob: float = 0.15
    # Optional per-clip boxes (indexed by clip_id 0=forehand, 1=backhand). None -> shared boxes above.
    racket_pos_range_per_clip: tuple | None = None
    racket_vel_range_per_clip: tuple | None = None
    # Optional planner-plane aware true target sampling.  Default keeps the historical motion/metadata
    # x box.  ``fixed_x_hit`` moves true strike x toward table_near_x + planner_hit_plane_x, matching
    # the MuJoCo/HOPEPlanner x_hit convention while preserving y/z motion priors.
    planner_hit_plane_mode: str = "motion_box"  # motion_box | fixed_x_hit
    planner_hit_plane_x: float = 0.20
    planner_hit_plane_x_jitter_range: tuple = (0.0, 0.0)
    planner_hit_plane_blend: float = 1.0
    planner_hit_plane_blend_start: float = 1.0
    planner_hit_plane_blend_per_clip: tuple = ()
    planner_hit_plane_blend_start_per_clip: tuple = ()
    planner_hit_plane_blend_warmup_steps: int = 0
    # Shrink each sampled racket-position box around its center at the beginning of training, then
    # linearly widen it to the configured full range over this many environment control steps.
    racket_pos_curriculum_steps: int = 0
    racket_pos_curriculum_start_scale: tuple[float, float, float] | float = (1.0, 1.0, 1.0)
    # Ability-based curriculum: difficulty is advanced from measured contact/net/success/recovery
    # at swing boundaries instead of from global iteration count. Disabled by default for full
    # backwards compatibility with existing tasks.
    ability_curriculum_enabled: bool = False
    ability_curriculum_start_racket_pos_scale: tuple[float, float, float] | float = (1.0, 1.0, 1.0)
    ability_curriculum_start_ball_scale: float = 1.0
    ability_curriculum_start_planner_perturb_scale: float = 1.0
    ability_curriculum_ema_rate: float = 0.02
    ability_curriculum_advance_rate: float = 0.015
    ability_curriculum_regress_rate: float = 0.03
    ability_curriculum_contact_threshold: float = 0.55
    ability_curriculum_net_threshold: float = 0.35
    ability_curriculum_success_threshold: float = 0.20
    ability_curriculum_recovery_threshold: float = 0.70
    ability_curriculum_recovery_floor: float = 0.45
    # Optional hysteresis: regress when an outcome EMA falls below this fraction
    # of its corresponding advance threshold. Zero preserves legacy behavior.
    ability_curriculum_outcome_regress_ratio: float = 0.0
    ability_curriculum_max_height_error: float = 0.10
    ability_curriculum_max_upright_error: float = 0.28
    ability_curriculum_use_cycle_success: bool = False
    ability_curriculum_cycle_success_threshold: float = 0.20
    ability_curriculum_cycle_ready_threshold: float = 0.72
    ability_curriculum_use_cycle_v2: bool = False
    ability_curriculum_cycle_attempt_threshold: float = 0.20
    ability_curriculum_cycle_v2_update_mode: str = "wrap"  # wrap | events
    ability_curriculum_require_side_contact: bool = False
    ability_curriculum_require_side_net: bool = False
    ability_curriculum_require_side_success: bool = False
    ability_curriculum_forehand_contact_threshold: float = 0.55
    ability_curriculum_backhand_contact_threshold: float = 0.55
    ability_curriculum_forehand_net_threshold: float = 0.35
    ability_curriculum_backhand_net_threshold: float = 0.35
    ability_curriculum_forehand_success_threshold: float = 0.20
    ability_curriculum_backhand_success_threshold: float = 0.20
    ability_curriculum_use_safety_gates: bool = False
    ability_curriculum_attempt_threshold: float = 0.45
    ability_curriculum_attempt_min_racket_speed: float = 0.35
    ability_curriculum_safety_threshold: float = 0.995
    ability_curriculum_safety_floor: float = 0.985
    ability_curriculum_station_saturation_threshold: float = 0.08
    ability_curriculum_station_saturation_regress_threshold: float = 0.15
    ability_curriculum_safe_min_base_height: float = 0.70
    ability_curriculum_safe_max_tilt: float = 0.65

    # Cycle-v2 deploy-faithful continuous constraint.  When enabled, a useful previous swing opens
    # a per-env pending check.  The next command-visible pre-strike state must become ready before
    # the deadline, otherwise a one-shot fail event is emitted for rewards/terminations.
    cycle_v2_enabled: bool = False
    cycle_v2_required_outcome: str = "opponent_bounce"  # contact | net_cross | opponent_bounce
    cycle_v2_outcome_by_ability: bool = False
    cycle_v2_net_outcome_level: float = 0.35
    cycle_v2_bounce_outcome_level: float = 0.70
    cycle_v2_visible_deadline_steps: int = 24
    cycle_v2_total_deadline_steps: int = 0
    cycle_v2_count_no_command_as_visible: bool = False
    cycle_v2_fail_on_strike_without_ready: bool = True
    cycle_v2_fail_unresolved_on_resample: bool = False
    cycle_v2_ready_mode: str = "all_hard"  # all_hard | core_hard_soft
    cycle_v2_ready_threshold: float = 0.72
    cycle_v2_max_height_error: float = 0.10
    cycle_v2_max_upright_error: float = 0.28
    cycle_v2_max_base_lin_vel: float = 0.25
    cycle_v2_max_base_ang_vel: float = 0.80
    cycle_v2_max_station_error: float = 0.24
    cycle_v2_min_feet_contact: float = 0.45
    cycle_v2_max_racket_speed: float = 0.80
    cycle_v2_min_arm_score: float = 0.55

    # --- no-spin return evaluation (example table placement in the env frame; tune to your scene) ---
    contact_radius: float = 0.095   # racket radius + ball radius
    min_approach_speed: float = 0.3  # racket must be moving into the target this fast to "contact"
    contact_approach_mode: str = "target_point"  # target_point | target_velocity
    return_model: str = "moving_racket_impact"  # racket_velocity | moving_racket_impact
    paddle_restitution: float = 0.654
    paddle_tangent_retain: float = 0.85
    min_normal_closing_speed: float = 0.2
    impact_inverse_tangent_blend: float = 1.0
    impact_inverse_min_racket_speed: float = 0.3
    impact_inverse_max_racket_speed: float = 3.2
    impact_inverse_racket_speed_scale: float = 1.0
    impact_inverse_racket_speed_bias: float = 0.0
    target_outgoing_vel_scale: float = 1.0
    target_outgoing_vel_z_bias: float = 0.0
    incoming_origin_x_range: tuple = (1.9, 2.4)
    incoming_origin_y_jitter_range: tuple = (-0.15, 0.15)
    incoming_origin_z_above_table_range: tuple = (0.15, 0.35)
    incoming_flight_time_range: tuple = (0.75, 1.0)
    incoming_trajectory_mode: str = "direct"  # direct | one_bounce
    ball_radius: float = 0.02
    # One-bounce incoming geometry. The hidden true ball bounces on the robot half, then reaches
    # the true strike point. The actor still receives only the planner target fields.
    one_bounce_post_time_range: tuple = (0.28, 0.44)
    one_bounce_lateral_jitter_range: tuple = (-0.18, 0.18)
    one_bounce_min_post_bounce_dx: float = 0.18
    one_bounce_max_post_bounce_dx: float = 0.78
    # Planner perturbations affect the actor-visible command only. Rewards/evaluation use the
    # hidden true ball task so small planner errors train robustness instead of changing the task.
    planner_target_pos_offset_range: tuple = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    planner_time_to_strike_offset_range: tuple = (0.0, 0.0)
    planner_target_vel_scale_range: tuple = (1.0, 1.0)
    planner_target_vel_offset_range: tuple = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    planner_target_vel_yaw_deg_range: tuple = (0.0, 0.0)
    table_near_x: float = 0.5       # x of the robot's own table end (robot sits behind it)
    table_surface_z: float = 0.76   # table surface height above the env origin
    table_length: float = 2.74      # ITTF table length (+x)
    table_width: float = 1.525      # ITTF table width (y)
    net_x: float = 1.37             # net plane from the near edge
    net_height: float = 0.1525      # net height above the surface
    net_margin: float = 0.02        # required clearance above the net top
    ballistic_flight_time_range: tuple = (0.45, 0.75)
    ballistic_land_x_range: tuple = (2.05, 2.95)
    ballistic_land_y_range: tuple = (-0.45, 0.45)
    ballistic_min_forward_speed: float = 0.3
    ballistic_sample_attempts: int = 8

    # Recovery diagnostics only affect logging.  They mirror the readiness reward components so we can
    # identify whether continuous failures come from base height, tilt, residual velocity, station drift,
    # foot contact, racket settling, or right-arm pose.
    recovery_diag_height_std: float = 0.095
    recovery_diag_upright_std: float = 0.26
    recovery_diag_lin_vel_std: float = 0.24
    recovery_diag_ang_vel_std: float = 0.70
    recovery_diag_station_std: float = 0.21
    recovery_diag_racket_vel_std: float = 0.68
    recovery_diag_arm_pos_std: float = 0.34
    recovery_diag_arm_ori_std: float = 0.78
    recovery_diag_arm_body_names: tuple[str, ...] = ()
    recovery_diag_early_prestrike_steps: int = 18
    cycle_success_diag_window_steps: int = 20
    cycle_success_diag_ready_threshold: float = 0.72
    cycle_success_diag_ready_temperature: float = 0.04
