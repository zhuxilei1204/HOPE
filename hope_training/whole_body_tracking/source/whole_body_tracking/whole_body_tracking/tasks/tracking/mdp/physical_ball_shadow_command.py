"""Vectorized rigid-ball shadow lifecycle for HOPE training diagnostics."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul, quat_rotate

from .physical_ball_shadow import (
    PhysicalShadowPhase,
    build_one_bounce_route,
    detect_incoming_net_collision,
    detect_net_cross,
    detect_outgoing_landing,
    detect_racket_contact,
    detect_table_bounce,
    measured_normal_restitution,
    moving_plane_impact_velocity,
    predict_drag_plane_crossing,
    predict_linearized_drag_plane_crossing,
    rigid_contact_point_velocity,
)
from .table_workspace import interpolate_bounds, windowed_curriculum_level

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .hope_commands import RacketTargetCommand


class PhysicalBallShadowCommand(CommandTerm):
    """Run a hidden rigid ball beside the unchanged analytic HOPE task."""

    cfg: "PhysicalBallShadowCommandCfg"

    def __init__(
        self,
        cfg: "PhysicalBallShadowCommandCfg",
        env: "ManagerBasedRLEnv",
    ):
        super().__init__(cfg, env)
        if bool(cfg.route_ability_curriculum_enabled):
            windowed_curriculum_level(
                0.0,
                float(cfg.route_ability_curriculum_start_level),
                float(cfg.route_ability_curriculum_full_level),
            )
            for easy, full in (
                (
                    cfg.route_easy_pre_bounce_time_range,
                    cfg.pre_bounce_time_range,
                ),
                (
                    cfg.route_easy_post_bounce_time_range,
                    cfg.post_bounce_time_range,
                ),
                (cfg.route_easy_bounce_dx_range, cfg.bounce_dx_range),
                (
                    cfg.route_easy_bounce_y_jitter_range,
                    cfg.bounce_y_jitter_range,
                ),
            ):
                interpolate_bounds(easy, full, 0.0)
        self.ball: RigidObject = env.scene[cfg.ball_asset_name]
        self._target_term: RacketTargetCommand | None = None
        self._racket_sensor_index: int | None = None
        self._detailed_contact_view = None
        self._detailed_contact_init_attempted = False
        self._detailed_contact_init_error: str | None = None
        self.physics_substep_capture_registered = False
        self._substep_racket_body_index: int | None = None
        self._substep_racket_com_offset_b: torch.Tensor | None = None
        self._substep_capture_error: str | None = None
        self._substep_capture_disabled = False
        self._route_batch_tick = max(
            int(cfg.route_batch_interval_steps) - 1, 0
        )

        self.phase = torch.full(
            (self.num_envs,),
            int(PhysicalShadowPhase.PARKED),
            dtype=torch.long,
            device=self.device,
        )
        self.pending_route = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.route_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.route_origin = torch.zeros(self.num_envs, 3, device=self.device)
        self.route_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.route_bounce = torch.zeros(self.num_envs, 3, device=self.device)
        self.route_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.route_total_time = torch.zeros(self.num_envs, device=self.device)
        self.route_incoming_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        initial_route_level = (
            0.0 if bool(cfg.route_ability_curriculum_enabled) else 1.0
        )
        self.route_curriculum_level = torch.full(
            (self.num_envs,), initial_route_level, device=self.device
        )
        self.flight_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.previous_position = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.previous_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )

        self.incoming_bounce_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.contact_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.net_cross_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.opponent_bounce_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._init_substep_contact_tensors()
        self._init_event_tensors()
        self._init_diagnostic_tensors()
        self._publish_metrics()

    def _init_substep_contact_tensors(self) -> None:
        self._substep_previous_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._substep_contact_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._substep_contact_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._substep_previous_ball_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_previous_ball_position = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_previous_ball_ang_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_previous_racket_com_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_previous_racket_ang_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_previous_racket_com_position = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_previous_racket_normal = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_pre_ball_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_pre_ball_position = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_pre_ball_ang_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_post_ball_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_post_ball_ang_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_pre_racket_com_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_pre_racket_ang_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_pre_racket_com_position = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_pre_racket_point_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_post_racket_point_velocity = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_point = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_normal = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_racket_normal = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_racket_center_position = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._substep_contact_force_peak = torch.zeros(
            self.num_envs, device=self.device
        )
        self._substep_contact_normal_impulse = torch.zeros(
            self.num_envs, device=self.device
        )
        self._substep_contact_separation = torch.zeros(
            self.num_envs, device=self.device
        )
        self._substep_contact_patch_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._substep_contact_age_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def _init_event_tensors(self) -> None:
        for name in (
            "serve_event",
            "incoming_bounce_event",
            "contact_event",
            "net_cross_event",
            "outgoing_landing_event",
            "opponent_bounce_event",
            "landing_short_event",
            "landing_long_event",
            "landing_side_event",
            "landing_no_net_event",
            "abort_event",
            "timeout_event",
            "route_unalignable_event",
            "route_invalid_event",
            "incoming_net_collision_event",
            "reset_abort_event",
            "command_refresh_event",
        ):
            setattr(
                self,
                name,
                torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                ),
            )

    def _init_diagnostic_tensors(self) -> None:
        for name in (
            "launch_timing_error_s",
            "contact_true_target_error",
            "contact_planner_position_error",
            "contact_planner_velocity_error",
            "contact_planner_velocity_direction_error_deg",
            "contact_planner_normal_error_deg",
            "contact_outgoing_velocity_error",
            "contact_outgoing_speed",
            "contact_target_outgoing_speed",
            "contact_outgoing_speed_ratio",
            "contact_outgoing_direction_error_deg",
            "contact_outgoing_velocity_x",
            "contact_outgoing_velocity_y",
            "contact_outgoing_velocity_z",
            "contact_target_outgoing_velocity_x",
            "contact_target_outgoing_velocity_y",
            "contact_target_outgoing_velocity_z",
            "contact_incoming_velocity_error",
            "contact_incoming_velocity_direction_error_deg",
            "contact_route_incoming_velocity_error",
            "contact_route_incoming_velocity_direction_error_deg",
            "contact_actual_route_incoming_velocity_error",
            "contact_command_normal_only_error",
            "contact_command_training_model_error",
            "contact_command_physics_model_error",
            "contact_actual_training_model_residual",
            "contact_actual_physics_model_residual",
            "contact_actual_normal_only_model_residual",
            "contact_actual_training_predicted_target_error",
            "contact_actual_physics_predicted_target_error",
            "contact_actual_normal_only_predicted_target_error",
            "contact_force_direction_valid",
            "contact_force_vs_racket_normal_angle_deg",
            "contact_actual_force_direction_model_residual",
            "contact_actual_force_direction_predicted_target_error",
            "contact_physx_data_valid",
            "contact_physx_force",
            "contact_physx_separation",
            "contact_physx_normal_error_deg",
            "contact_physx_point_radial_error",
            "contact_actual_physx_normal_model_residual",
            "contact_actual_physx_normal_predicted_target_error",
            "contact_physx_substep_valid",
            "contact_physx_substep_normal_error_deg",
            "contact_physx_substep_point_radial_error",
            "contact_physx_substep_point_normal_offset",
            "contact_physx_patch_substeps",
            "contact_physx_normal_impulse",
            "contact_physx_capture_lag_s",
            "contact_physx_contact_point_speed_delta",
            "contact_physx_substep_link_model_residual",
            "contact_physx_substep_point_model_residual",
            "contact_physx_substep_outgoing_target_error",
            "contact_physx_measured_restitution",
            "contact_physx_pre_ball_speed",
            "contact_physx_post_ball_speed",
            "contact_physx_point_x",
            "contact_physx_point_y",
            "contact_physx_point_z",
            "contact_physx_normal_x",
            "contact_physx_normal_y",
            "contact_physx_normal_z",
            "contact_physx_pre_ball_velocity_x",
            "contact_physx_pre_ball_velocity_y",
            "contact_physx_pre_ball_velocity_z",
            "contact_physx_post_ball_velocity_x",
            "contact_physx_post_ball_velocity_y",
            "contact_physx_post_ball_velocity_z",
            "contact_physx_pre_racket_point_velocity_x",
            "contact_physx_pre_racket_point_velocity_y",
            "contact_physx_pre_racket_point_velocity_z",
            "contact_physx_post_racket_point_velocity_x",
            "contact_physx_post_racket_point_velocity_y",
            "contact_physx_post_racket_point_velocity_z",
            "contact_physx_pre_ball_angular_velocity_x",
            "contact_physx_pre_ball_angular_velocity_y",
            "contact_physx_pre_ball_angular_velocity_z",
            "contact_physx_post_ball_angular_velocity_x",
            "contact_physx_post_ball_angular_velocity_y",
            "contact_physx_post_ball_angular_velocity_z",
            "contact_physx_pre_ball_surface_tangent_speed",
            "contact_physx_post_ball_spin_speed",
            "contact_impact_normal_error_deg",
            "contact_wire_impact_normal_gap_deg",
            "contact_face_radial_error",
            "contact_time_to_strike",
            "landing_target_error",
            "landing_target_x_error",
            "landing_target_y_error",
            "landing_position_x",
            "landing_position_y",
            "minimum_ball_racket_distance",
            "command_refresh_valid",
            "command_refresh_count",
            "command_refresh_tts_s",
            "command_refresh_position_delta",
            "command_refresh_incoming_velocity_delta",
            "command_refresh_timing_delta_s",
            "command_refresh_racket_velocity_delta",
            "command_refresh_normal_delta_deg",
        ):
            setattr(
                self,
                name,
                torch.zeros(self.num_envs, device=self.device),
            )
        self.minimum_ball_racket_distance[:] = float("inf")
        self._reward_snapshot_prepared = False
        self._pre_physics_prepare_pending = True

    def _target(self) -> "RacketTargetCommand":
        if self._target_term is None:
            self._target_term = self._env.command_manager.get_term(
                self.cfg.target_command_name
            )
        return self._target_term

    @property
    def command(self) -> torch.Tensor:
        return self.phase.float().unsqueeze(-1)

    def _table_geometry(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        origins = self._env.scene.env_origins
        target_cfg = self._target().cfg
        table_near_x = origins[:, 0] + float(target_cfg.table_near_x)
        net_x = table_near_x + float(target_cfg.net_x)
        table_far_x = table_near_x + float(target_cfg.table_length)
        surface_z = origins[:, 2] + float(target_cfg.table_surface_z)
        table_y_min = origins[:, 1] - 0.5 * float(target_cfg.table_width)
        table_y_max = origins[:, 1] + 0.5 * float(target_cfg.table_width)
        return (
            table_near_x,
            net_x,
            table_far_x,
            surface_z,
            table_y_min,
            table_y_max,
        )

    def _write_ball_state(
        self,
        env_ids: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
    ) -> None:
        pose = torch.zeros(len(env_ids), 7, device=self.device)
        pose[:, :3] = position
        pose[:, 3] = 1.0
        root_velocity = torch.zeros(len(env_ids), 6, device=self.device)
        root_velocity[:, :3] = velocity
        self.ball.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(
            root_velocity, env_ids=env_ids
        )

    def _park(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        origins = self._env.scene.env_origins[env_ids]
        target_cfg = self._target().cfg
        position = origins.clone()
        position[:, 0] += (
            float(target_cfg.table_near_x)
            + float(target_cfg.table_length)
            + float(self.cfg.park_offset_x)
        )
        position[:, 1] += float(self.cfg.park_offset_y)
        position[:, 2] += (
            float(target_cfg.table_surface_z)
            + float(self.cfg.park_height_above_table)
        )
        self._write_ball_state(
            env_ids,
            position,
            torch.zeros_like(position),
        )
        self.previous_position[env_ids] = position
        self.previous_velocity[env_ids] = 0.0

    def _clear_swing_state(self, env_ids: torch.Tensor) -> None:
        self.phase[env_ids] = int(PhysicalShadowPhase.PARKED)
        self.pending_route[env_ids] = True
        self.route_valid[env_ids] = False
        self.flight_elapsed[env_ids] = 0.0
        self.incoming_bounce_latch[env_ids] = False
        self.contact_latch[env_ids] = False
        self.net_cross_latch[env_ids] = False
        self.opponent_bounce_latch[env_ids] = False
        self.minimum_ball_racket_distance[env_ids] = float("inf")
        self._clear_substep_contact_state(env_ids)
        for name in (
            "launch_timing_error_s",
            "contact_true_target_error",
            "contact_planner_position_error",
            "contact_planner_velocity_error",
            "contact_planner_velocity_direction_error_deg",
            "contact_planner_normal_error_deg",
            "contact_outgoing_velocity_error",
            "contact_outgoing_speed",
            "contact_target_outgoing_speed",
            "contact_outgoing_speed_ratio",
            "contact_outgoing_direction_error_deg",
            "contact_outgoing_velocity_x",
            "contact_outgoing_velocity_y",
            "contact_outgoing_velocity_z",
            "contact_target_outgoing_velocity_x",
            "contact_target_outgoing_velocity_y",
            "contact_target_outgoing_velocity_z",
            "contact_incoming_velocity_error",
            "contact_incoming_velocity_direction_error_deg",
            "contact_route_incoming_velocity_error",
            "contact_route_incoming_velocity_direction_error_deg",
            "contact_actual_route_incoming_velocity_error",
            "contact_command_normal_only_error",
            "contact_command_training_model_error",
            "contact_command_physics_model_error",
            "contact_actual_training_model_residual",
            "contact_actual_physics_model_residual",
            "contact_actual_normal_only_model_residual",
            "contact_actual_training_predicted_target_error",
            "contact_actual_physics_predicted_target_error",
            "contact_actual_normal_only_predicted_target_error",
            "contact_force_direction_valid",
            "contact_force_vs_racket_normal_angle_deg",
            "contact_actual_force_direction_model_residual",
            "contact_actual_force_direction_predicted_target_error",
            "contact_physx_data_valid",
            "contact_physx_force",
            "contact_physx_separation",
            "contact_physx_normal_error_deg",
            "contact_physx_point_radial_error",
            "contact_actual_physx_normal_model_residual",
            "contact_actual_physx_normal_predicted_target_error",
            "contact_physx_substep_valid",
            "contact_physx_substep_normal_error_deg",
            "contact_physx_substep_point_radial_error",
            "contact_physx_substep_point_normal_offset",
            "contact_physx_patch_substeps",
            "contact_physx_normal_impulse",
            "contact_physx_capture_lag_s",
            "contact_physx_contact_point_speed_delta",
            "contact_physx_substep_link_model_residual",
            "contact_physx_substep_point_model_residual",
            "contact_physx_substep_outgoing_target_error",
            "contact_physx_measured_restitution",
            "contact_physx_pre_ball_speed",
            "contact_physx_post_ball_speed",
            "contact_physx_point_x",
            "contact_physx_point_y",
            "contact_physx_point_z",
            "contact_physx_normal_x",
            "contact_physx_normal_y",
            "contact_physx_normal_z",
            "contact_physx_pre_ball_velocity_x",
            "contact_physx_pre_ball_velocity_y",
            "contact_physx_pre_ball_velocity_z",
            "contact_physx_post_ball_velocity_x",
            "contact_physx_post_ball_velocity_y",
            "contact_physx_post_ball_velocity_z",
            "contact_physx_pre_racket_point_velocity_x",
            "contact_physx_pre_racket_point_velocity_y",
            "contact_physx_pre_racket_point_velocity_z",
            "contact_physx_post_racket_point_velocity_x",
            "contact_physx_post_racket_point_velocity_y",
            "contact_physx_post_racket_point_velocity_z",
            "contact_impact_normal_error_deg",
            "contact_wire_impact_normal_gap_deg",
            "contact_face_radial_error",
            "contact_time_to_strike",
            "landing_target_error",
            "landing_target_x_error",
            "landing_target_y_error",
            "landing_position_x",
            "landing_position_y",
            "command_refresh_valid",
            "command_refresh_count",
            "command_refresh_tts_s",
            "command_refresh_position_delta",
            "command_refresh_incoming_velocity_delta",
            "command_refresh_timing_delta_s",
            "command_refresh_racket_velocity_delta",
            "command_refresh_normal_delta_deg",
        ):
            getattr(self, name)[env_ids] = 0.0

    def _clear_substep_contact_state(self, env_ids: torch.Tensor) -> None:
        for name in (
            "_substep_previous_valid",
            "_substep_contact_active",
            "_substep_contact_latched",
        ):
            getattr(self, name)[env_ids] = False
        for name in (
            "_substep_previous_ball_velocity",
            "_substep_previous_ball_position",
            "_substep_previous_ball_ang_velocity",
            "_substep_previous_racket_com_velocity",
            "_substep_previous_racket_ang_velocity",
            "_substep_previous_racket_com_position",
            "_substep_previous_racket_normal",
            "_substep_contact_pre_ball_velocity",
            "_substep_contact_pre_ball_position",
            "_substep_contact_pre_ball_ang_velocity",
            "_substep_contact_post_ball_velocity",
            "_substep_contact_post_ball_ang_velocity",
            "_substep_contact_pre_racket_com_velocity",
            "_substep_contact_pre_racket_ang_velocity",
            "_substep_contact_pre_racket_com_position",
            "_substep_contact_pre_racket_point_velocity",
            "_substep_contact_post_racket_point_velocity",
            "_substep_contact_point",
            "_substep_contact_normal",
            "_substep_contact_racket_normal",
            "_substep_contact_racket_center_position",
            "_substep_contact_force_peak",
            "_substep_contact_normal_impulse",
            "_substep_contact_separation",
            "_substep_contact_patch_steps",
            "_substep_contact_age_steps",
        ):
            getattr(self, name)[env_ids] = 0

    def _resample_command(self, env_ids) -> None:
        env_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        )
        active = (
            (self.phase[env_ids] == int(
                PhysicalShadowPhase.INCOMING_PRE_BOUNCE
            ))
            | (self.phase[env_ids] == int(
                PhysicalShadowPhase.INCOMING_POST_BOUNCE
            ))
            | (self.phase[env_ids] == int(PhysicalShadowPhase.OUTGOING))
        )
        self._clear_swing_state(env_ids)
        self.reset_abort_event[env_ids] = active
        self._park(env_ids)
        self._pre_physics_prepare_pending = True

    def _route_sampling_ranges(
        self, target: "RacketTargetCommand"
    ) -> tuple[
        float,
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """Return ability-scaled physical-route ranges.

        Route difficulty follows the shared measured ability level, but the
        route remains independent of motion and planner implementation.
        """
        if not bool(self.cfg.route_ability_curriculum_enabled):
            return (
                1.0,
                tuple(float(v) for v in self.cfg.pre_bounce_time_range),
                tuple(float(v) for v in self.cfg.post_bounce_time_range),
                tuple(float(v) for v in self.cfg.bounce_dx_range),
                tuple(float(v) for v in self.cfg.bounce_y_jitter_range),
            )
        level = windowed_curriculum_level(
            float(target._ability_curriculum_level.item()),
            float(self.cfg.route_ability_curriculum_start_level),
            float(self.cfg.route_ability_curriculum_full_level),
        )
        return (
            level,
            interpolate_bounds(
                self.cfg.route_easy_pre_bounce_time_range,
                self.cfg.pre_bounce_time_range,
                level,
            ),
            interpolate_bounds(
                self.cfg.route_easy_post_bounce_time_range,
                self.cfg.post_bounce_time_range,
                level,
            ),
            interpolate_bounds(
                self.cfg.route_easy_bounce_dx_range,
                self.cfg.bounce_dx_range,
                level,
            ),
            interpolate_bounds(
                self.cfg.route_easy_bounce_y_jitter_range,
                self.cfg.bounce_y_jitter_range,
                level,
            ),
        )

    def _sample_routes(self, env_ids: torch.Tensor) -> None:
        target = self._target()
        if len(env_ids) == 0:
            return
        (
            table_near_x,
            net_x,
            table_far_x,
            surface_z,
            table_y_min,
            table_y_max,
        ) = self._table_geometry()

        self.pending_route[env_ids] = False
        self.route_valid[env_ids] = False

        (
            route_level,
            (pre_lo, pre_hi),
            (post_lo, post_hi),
            bounce_dx_range,
            bounce_y_jitter_range,
        ) = self._route_sampling_ranges(target)
        self.route_curriculum_level[env_ids] = route_level
        available_time = target.true_time_to_strike[env_ids]
        if self.cfg.route_geometry_mode == "target_hidden":
            hidden_post_time = (
                target.incoming_ball_post_bounce_time_s[env_ids]
            )
            alignable = (
                target.incoming_ball_route_valid[env_ids]
                & (
                    available_time
                    >= (
                        pre_lo
                        + hidden_post_time
                        - 0.5 * float(self._env.step_dt)
                    )
                )
            )
        elif self.cfg.route_geometry_mode == "independent":
            alignable = available_time >= (
                pre_lo + post_lo - 0.5 * float(self._env.step_dt)
            )
        else:
            raise ValueError(
                "route_geometry_mode must be 'target_hidden' or "
                "'independent'"
            )
        unalignable_ids = env_ids[~alignable]
        self.route_unalignable_event[unalignable_ids] = True
        candidate_ids = env_ids[alignable]
        unresolved = torch.ones(
            len(candidate_ids), dtype=torch.bool, device=self.device
        )

        for _ in range(int(self.cfg.route_sample_attempts)):
            trial_ids = candidate_ids[unresolved]
            if len(trial_ids) == 0:
                break
            count = len(trial_ids)
            available = target.true_time_to_strike[trial_ids]
            if self.cfg.route_geometry_mode == "target_hidden":
                post_time = (
                    target.incoming_ball_post_bounce_time_s[
                        trial_ids
                    ]
                )
            else:
                post_upper = torch.minimum(
                    torch.full_like(available, post_hi),
                    available - pre_lo,
                ).clamp_min(post_lo)
                post_time = post_lo + torch.rand(
                    count, device=self.device
                ) * (post_upper - post_lo)
            pre_upper = torch.minimum(
                torch.full_like(available, pre_hi),
                available - post_time,
            ).clamp_min(pre_lo)
            pre_time = pre_lo + torch.rand(
                count, device=self.device
            ) * (pre_upper - pre_lo)

            strike = target.ball_strike_pos_w[trial_ids].clone()
            if self.cfg.route_geometry_mode == "target_hidden":
                bounce = target.incoming_ball_bounce_pos_w[
                    trial_ids
                ].clone()
            else:
                bounce_dx = torch.empty(
                    count, device=self.device
                ).uniform_(*bounce_dx_range)
                bounce_y_jitter = torch.empty(
                    count, device=self.device
                ).uniform_(*bounce_y_jitter_range)
                bounce = strike.clone()
                bounce[:, 0] = torch.minimum(
                    strike[:, 0] + bounce_dx,
                    net_x[trial_ids]
                    - float(self.cfg.net_bounce_margin),
                )
                bounce[:, 0] = torch.maximum(
                    bounce[:, 0],
                    torch.maximum(
                        strike[:, 0]
                        + float(self.cfg.minimum_bounce_dx),
                        table_near_x[trial_ids]
                        + float(self.cfg.near_edge_margin),
                    ),
                )
                bounce[:, 1] = torch.clamp(
                    strike[:, 1] + bounce_y_jitter,
                    min=table_y_min[trial_ids]
                    + float(self.cfg.side_margin),
                    max=table_y_max[trial_ids]
                    - float(self.cfg.side_margin),
                )
                bounce[:, 2] = (
                    surface_z[trial_ids]
                    + float(target.cfg.ball_radius)
                )

            route = build_one_bounce_route(
                strike,
                bounce,
                pre_time,
                post_time,
                horizontal_retain=float(self.cfg.horizontal_retain),
                vertical_restitution=float(self.cfg.vertical_restitution),
                gravity=float(self.cfg.gravity),
                drag_k=float(self.cfg.drag_k),
                max_dt=float(self.cfg.route_integrator_max_dt),
                pre_max_duration=pre_hi,
                post_max_duration=post_hi,
            )
            valid = (
                torch.isfinite(route.origin).all(dim=-1)
                & torch.isfinite(route.serve_velocity).all(dim=-1)
                & (
                    route.origin[:, 0]
                    >= net_x[trial_ids] + float(self.cfg.origin_net_margin)
                )
                & (
                    route.origin[:, 0]
                    <= table_far_x[trial_ids]
                    + float(self.cfg.origin_far_margin)
                )
                & (
                    route.origin[:, 1]
                    >= table_y_min[trial_ids]
                    + float(self.cfg.origin_side_margin)
                )
                & (
                    route.origin[:, 1]
                    <= table_y_max[trial_ids]
                    - float(self.cfg.origin_side_margin)
                )
                & (
                    route.origin[:, 2]
                    >= surface_z[trial_ids]
                    + float(self.cfg.origin_height_range[0])
                )
                & (
                    route.origin[:, 2]
                    <= surface_z[trial_ids]
                    + float(self.cfg.origin_height_range[1])
                )
            )
            accepted_ids = trial_ids[valid]
            if len(accepted_ids) > 0:
                self.route_origin[accepted_ids] = route.origin[valid]
                self.route_velocity[accepted_ids] = (
                    route.serve_velocity[valid]
                )
                self.route_bounce[accepted_ids] = route.bounce[valid]
                self.route_target[accepted_ids] = strike[valid]
                self.route_total_time[accepted_ids] = (
                    route.total_time[valid]
                )
                self.route_incoming_velocity[accepted_ids] = (
                    route.incoming_velocity[valid]
                )
                self.route_valid[accepted_ids] = True
            unresolved_indices = torch.where(unresolved)[0]
            unresolved[unresolved_indices[valid]] = False

        invalid_ids = candidate_ids[unresolved]
        self.route_invalid_event[invalid_ids] = True

    def _launch_due_routes(self) -> None:
        target = self._target()
        interval = max(int(self.cfg.route_batch_interval_steps), 1)
        self._route_batch_tick = (self._route_batch_tick + 1) % interval
        if self._route_batch_tick == 0:
            candidate = (
                (self.phase == int(PhysicalShadowPhase.PARKED))
                & self.pending_route
                & (target.true_time_to_strike > 0.0)
            )
            sample_ids = torch.where(candidate)[0]
            if len(sample_ids) > 0:
                self._sample_routes(sample_ids)
        expired = (
            (self.phase == int(PhysicalShadowPhase.PARKED))
            & self.pending_route
            & (target.true_time_to_strike <= 0.0)
        )
        self.route_unalignable_event |= expired
        self.pending_route[expired] = False

        due = (
            (self.phase == int(PhysicalShadowPhase.PARKED))
            & self.route_valid
            & (
                target.true_time_to_strike
                <= self.route_total_time + 0.5 * float(self._env.step_dt)
            )
            & (target.true_time_to_strike >= -float(self._env.step_dt))
        )
        env_ids = torch.where(due)[0]
        if len(env_ids) == 0:
            return
        self._write_ball_state(
            env_ids,
            self.route_origin[env_ids],
            self.route_velocity[env_ids],
        )
        self.previous_position[env_ids] = self.route_origin[env_ids]
        self.previous_velocity[env_ids] = self.route_velocity[env_ids]
        self.flight_elapsed[env_ids] = 0.0
        self.launch_timing_error_s[env_ids] = (
            target.true_time_to_strike[env_ids]
            - self.route_total_time[env_ids]
        )
        self.phase[env_ids] = int(
            PhysicalShadowPhase.INCOMING_PRE_BOUNCE
        )
        self.serve_event[env_ids] = True

    def _racket_force_vector(self) -> torch.Tensor:
        target = self._target()
        sensor = getattr(target, "_contact_sensor", None)
        if sensor is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        if self._racket_sensor_index is None:
            names = list(sensor.body_names)
            for name in (
                target.cfg.racket_body_name,
                target.cfg.wrist_body_name,
            ):
                if name in names:
                    self._racket_sensor_index = names.index(name)
                    break
            if self._racket_sensor_index is None:
                self._racket_sensor_index = -1
        if self._racket_sensor_index < 0:
            return torch.zeros(self.num_envs, 3, device=self.device)
        history = getattr(sensor.data, "net_forces_w_history", None)
        if history is not None:
            vectors = history[:, :, self._racket_sensor_index, :]
            indices = torch.linalg.norm(
                vectors, dim=-1
            ).argmax(dim=1)
            return torch.gather(
                vectors,
                1,
                indices[:, None, None].expand(-1, 1, 3),
            ).squeeze(1)
        return sensor.data.net_forces_w[:, self._racket_sensor_index]

    def _empty_detailed_contact_data(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        force = torch.zeros(self.num_envs, device=self.device)
        point = torch.zeros(self.num_envs, 3, device=self.device)
        normal = torch.zeros(self.num_envs, 3, device=self.device)
        separation = torch.zeros(self.num_envs, device=self.device)
        return valid, force, point, normal, separation

    def _ensure_detailed_contact_view(self) -> bool:
        if not self.cfg.detailed_contact_enabled:
            return False
        if not self._detailed_contact_init_attempted:
            self._detailed_contact_init_attempted = True
            try:
                from isaacsim.core.simulation_manager import (
                    SimulationManager,
                )

                robot_root = self._env.scene["robot"].cfg.prim_path.replace(
                    ".*", "*"
                )
                ball_path = self.ball.cfg.prim_path.replace(".*", "*")
                target = self._target()
                sensor_body_name = (
                    target.cfg.racket_body_name
                    if target._racket_mode == "body"
                    else target.cfg.wrist_body_name
                )
                sensor_pattern = f"{robot_root}/{sensor_body_name}"
                sim_view = SimulationManager.get_physics_sim_view()
                self._detailed_contact_view = (
                    sim_view.create_rigid_contact_view(
                        sensor_pattern,
                        filter_patterns=[ball_path],
                        max_contact_data_count=(
                            self.num_envs
                            * int(self.cfg.max_contact_points_per_env)
                        ),
                    )
                )
                sensor_count = int(
                    self._detailed_contact_view.sensor_count
                )
                filter_count = int(
                    self._detailed_contact_view.filter_count
                )
                if sensor_count != self.num_envs or filter_count != 1:
                    raise RuntimeError(
                        "Unexpected detailed contact view shape: "
                        f"sensors={sensor_count}, filters={filter_count}, "
                        f"envs={self.num_envs}"
                    )
            except Exception as exc:
                self._detailed_contact_init_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                self._detailed_contact_view = None
        return self._detailed_contact_view is not None

    def _read_current_physx_contact_data(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Read and aggregate the current filtered PhysX racket-ball patch."""
        valid, force, point, normal, separation = (
            self._empty_detailed_contact_data()
        )
        if not self._ensure_detailed_contact_view():
            return valid, force, point, normal, separation

        try:
            (
                force_buffer,
                point_buffer,
                normal_buffer,
                separation_buffer,
                count_buffer,
                start_buffer,
            ) = self._detailed_contact_view.get_contact_data(
                float(self._env.scene.physics_dt)
            )
            counts = count_buffer.reshape(self.num_envs, 1)[:, 0]
            starts = start_buffer.reshape(self.num_envs, 1)[:, 0]
            force_buffer = force_buffer.reshape(-1)
            point_buffer = point_buffer.reshape(-1, 3)
            normal_buffer = normal_buffer.reshape(-1, 3)
            separation_buffer = separation_buffer.reshape(-1)
            max_points = int(self.cfg.max_contact_points_per_env)
            offsets = torch.arange(max_points, device=self.device)
            limited_counts = counts.long().clamp(min=0, max=max_points)
            indices = starts.long().unsqueeze(-1) + offsets.unsqueeze(0)
            sample_mask = offsets.unsqueeze(0) < limited_counts.unsqueeze(-1)
            indices = indices.clamp(min=0, max=force_buffer.shape[0] - 1)
            sample_force = torch.where(
                sample_mask,
                force_buffer[indices].clamp_min(0.0),
                torch.zeros_like(indices, dtype=force_buffer.dtype),
            )
            force_sum = sample_force.sum(dim=-1)
            weight = sample_force / force_sum.unsqueeze(-1).clamp_min(1.0e-9)
            point = torch.sum(point_buffer[indices] * weight.unsqueeze(-1), dim=1)
            normal = torch.sum(
                normal_buffer[indices] * weight.unsqueeze(-1), dim=1
            )
            normal = normal / torch.linalg.norm(
                normal, dim=-1, keepdim=True
            ).clamp_min(1.0e-9)
            separation_samples = torch.where(
                sample_mask,
                separation_buffer[indices],
                torch.full_like(
                    indices, float("inf"), dtype=separation_buffer.dtype
                ),
            )
            separation = separation_samples.min(dim=-1).values
            valid = (limited_counts > 0) & (force_sum > 0.0)
            force = force_sum
            point = torch.where(valid.unsqueeze(-1), point, torch.zeros_like(point))
            normal = torch.where(
                valid.unsqueeze(-1), normal, torch.zeros_like(normal)
            )
            separation = torch.where(
                valid, separation, torch.zeros_like(separation)
            )
        except Exception as exc:
            self._detailed_contact_init_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self._detailed_contact_view = None
        return valid, force, point, normal, separation

    def _read_detailed_contact_data(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prefer a physics-rate latched patch and retain control-rate fallback."""
        current = self._read_current_physx_contact_data()
        if (
            not self.physics_substep_capture_registered
            or self._substep_capture_disabled
        ):
            return current
        current_valid, current_force, current_point, current_normal, current_sep = (
            current
        )
        latched = self._substep_contact_latched
        valid = latched | current_valid
        force = torch.where(
            latched, self._substep_contact_force_peak, current_force
        )
        point = torch.where(
            latched.unsqueeze(-1), self._substep_contact_point, current_point
        )
        normal = torch.where(
            latched.unsqueeze(-1), self._substep_contact_normal, current_normal
        )
        separation = torch.where(
            latched, self._substep_contact_separation, current_sep
        )
        return valid, force, point, normal, separation

    def _read_racket_physx_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        target = self._target()
        robot = target.robot
        if self._substep_racket_body_index is None:
            sensor_body_name = (
                target.cfg.racket_body_name
                if target._racket_mode == "body"
                else target.cfg.wrist_body_name
            )
            body_ids = robot.find_bodies(
                sensor_body_name, preserve_order=True
            )[0]
            if len(body_ids) != 1:
                raise RuntimeError(
                    "physics-rate contact capture requires exactly one "
                    f"racket body, found {body_ids}"
                )
            self._substep_racket_body_index = int(body_ids[0])
            self._substep_racket_com_offset_b = robot.data.com_pos_b[
                :, self._substep_racket_body_index
            ].clone()

        body_index = self._substep_racket_body_index
        transforms = robot.root_physx_view.get_link_transforms()
        velocities = robot.root_physx_view.get_link_velocities()
        link_position = transforms[:, body_index, :3]
        quat_xyzw = transforms[:, body_index, 3:7]
        quat_wxyz = torch.cat(
            (quat_xyzw[:, 3:4], quat_xyzw[:, :3]), dim=-1
        )
        com_position = link_position + quat_rotate(
            quat_wxyz, self._substep_racket_com_offset_b
        )
        com_velocity = velocities[:, body_index, :3]
        angular_velocity = velocities[:, body_index, 3:6]
        local_axis = torch.zeros(
            self.num_envs, 3, dtype=transforms.dtype, device=self.device
        )
        local_axis[:, int(target.cfg.mount_normal_axis)] = 1.0
        racket_quat_wxyz = quat_wxyz
        racket_center_position = link_position
        if target._racket_mode == "wrist_offset":
            racket_quat_wxyz = quat_mul(quat_wxyz, target._mount_quat)
            racket_center_position = link_position + quat_rotate(
                quat_wxyz, target._mount_offset
            )
        unsigned_normal = quat_rotate(racket_quat_wxyz, local_axis)
        cached_alignment = torch.sum(
            unsigned_normal * target.racket_normal_w, dim=-1
        )
        default_sign = 1.0 if float(target.cfg.mount_normal_sign) >= 0.0 else -1.0
        face_sign = torch.where(
            cached_alignment.abs() > 0.25,
            torch.sign(cached_alignment),
            torch.full_like(cached_alignment, default_sign),
        )
        racket_normal = unsigned_normal * face_sign.unsqueeze(-1)
        return (
            com_position,
            com_velocity,
            angular_velocity,
            racket_normal,
            racket_center_position,
        )

    def capture_physics_substep(self, dt: float) -> None:
        """Latch exact racket-ball contact state at the PhysX simulation rate."""
        if self._substep_capture_disabled:
            return
        try:
            (
                contact_valid,
                contact_force,
                contact_point,
                contact_normal,
                contact_separation,
            ) = self._read_current_physx_contact_data()
            ball_state = self.ball.root_physx_view.get_velocities()
            ball_velocity = ball_state[:, :3]
            ball_ang_velocity = ball_state[:, 3:6]
            ball_position = self.ball.root_physx_view.get_transforms()[:, :3]
            (
                racket_com_position,
                racket_com_velocity,
                racket_angular_velocity,
                racket_normal,
                racket_center_position,
            ) = self._read_racket_physx_state()

            new_contact = (
                contact_valid
                & (~self._substep_contact_active)
                & (~self._substep_contact_latched)
                & self._substep_previous_valid
            )
            if torch.any(new_contact):
                normal_alignment = torch.sum(
                    contact_normal
                    * self._substep_previous_racket_normal,
                    dim=-1,
                    keepdim=True,
                )
                aligned_normal = torch.where(
                    normal_alignment < 0.0,
                    -contact_normal,
                    contact_normal,
                )
                self._substep_contact_active[new_contact] = True
                self._substep_contact_latched[new_contact] = True
                self._substep_contact_pre_ball_velocity[new_contact] = (
                    self._substep_previous_ball_velocity[new_contact]
                )
                self._substep_contact_pre_ball_position[new_contact] = (
                    self._substep_previous_ball_position[new_contact]
                )
                self._substep_contact_pre_ball_ang_velocity[new_contact] = (
                    self._substep_previous_ball_ang_velocity[new_contact]
                )
                self._substep_contact_pre_racket_com_velocity[new_contact] = (
                    self._substep_previous_racket_com_velocity[new_contact]
                )
                self._substep_contact_pre_racket_ang_velocity[new_contact] = (
                    self._substep_previous_racket_ang_velocity[new_contact]
                )
                self._substep_contact_pre_racket_com_position[new_contact] = (
                    self._substep_previous_racket_com_position[new_contact]
                )
                self._substep_contact_point[new_contact] = contact_point[
                    new_contact
                ]
                self._substep_contact_normal[new_contact] = aligned_normal[
                    new_contact
                ]
                self._substep_contact_racket_normal[new_contact] = (
                    self._substep_previous_racket_normal[new_contact]
                )
                self._substep_contact_racket_center_position[new_contact] = (
                    racket_center_position[new_contact]
                )
                self._substep_contact_pre_racket_point_velocity[new_contact] = (
                    rigid_contact_point_velocity(
                        self._substep_previous_racket_com_velocity[new_contact],
                        self._substep_previous_racket_ang_velocity[new_contact],
                        contact_point[new_contact],
                        self._substep_previous_racket_com_position[new_contact],
                    )
                )
                self._substep_contact_force_peak[new_contact] = 0.0
                self._substep_contact_normal_impulse[new_contact] = 0.0
                self._substep_contact_separation[new_contact] = (
                    contact_separation[new_contact]
                )
                self._substep_contact_patch_steps[new_contact] = 0
                self._substep_contact_age_steps[new_contact] = 0

            ongoing = contact_valid & self._substep_contact_active
            replace_peak = ongoing & (
                contact_force > self._substep_contact_force_peak
            )
            if torch.any(replace_peak):
                normal_alignment = torch.sum(
                    contact_normal * racket_normal, dim=-1, keepdim=True
                )
                aligned_normal = torch.where(
                    normal_alignment < 0.0,
                    -contact_normal,
                    contact_normal,
                )
                self._substep_contact_point[replace_peak] = contact_point[
                    replace_peak
                ]
                self._substep_contact_normal[replace_peak] = aligned_normal[
                    replace_peak
                ]
                self._substep_contact_racket_normal[replace_peak] = (
                    racket_normal[replace_peak]
                )
                self._substep_contact_racket_center_position[replace_peak] = (
                    racket_center_position[replace_peak]
                )
                self._substep_contact_pre_racket_point_velocity[
                    replace_peak
                ] = rigid_contact_point_velocity(
                    self._substep_contact_pre_racket_com_velocity[replace_peak],
                    self._substep_contact_pre_racket_ang_velocity[replace_peak],
                    contact_point[replace_peak],
                    self._substep_contact_pre_racket_com_position[replace_peak],
                )

            if torch.any(ongoing):
                self._substep_contact_post_ball_velocity[ongoing] = (
                    ball_velocity[ongoing]
                )
                self._substep_contact_post_ball_ang_velocity[ongoing] = (
                    ball_ang_velocity[ongoing]
                )
                self._substep_contact_post_racket_point_velocity[ongoing] = (
                    rigid_contact_point_velocity(
                        racket_com_velocity[ongoing],
                        racket_angular_velocity[ongoing],
                        self._substep_contact_point[ongoing],
                        racket_com_position[ongoing],
                    )
                )
                self._substep_contact_force_peak[ongoing] = torch.maximum(
                    self._substep_contact_force_peak[ongoing],
                    contact_force[ongoing],
                )
                self._substep_contact_normal_impulse[ongoing] += (
                    contact_force[ongoing] * float(dt)
                )
                self._substep_contact_separation[ongoing] = torch.minimum(
                    self._substep_contact_separation[ongoing],
                    contact_separation[ongoing],
                )
                self._substep_contact_patch_steps[ongoing] += 1

            ended = self._substep_contact_active & (~contact_valid)
            self._substep_contact_active[ended] = False
            self._substep_contact_age_steps[
                self._substep_contact_latched
            ] += 1
            self._substep_previous_ball_velocity[:] = ball_velocity
            self._substep_previous_ball_position[:] = ball_position
            self._substep_previous_ball_ang_velocity[:] = ball_ang_velocity
            self._substep_previous_racket_com_velocity[:] = racket_com_velocity
            self._substep_previous_racket_ang_velocity[:] = (
                racket_angular_velocity
            )
            self._substep_previous_racket_com_position[:] = racket_com_position
            self._substep_previous_racket_normal[:] = racket_normal
            self._substep_previous_valid[:] = True
        except Exception as exc:
            self._substep_capture_error = f"{type(exc).__name__}: {exc}"
            self._substep_capture_disabled = True

    def _record_contact_diagnostics(
        self,
        contact: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        racket_force_vector: torch.Tensor,
        detailed_contact: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        env_ids = torch.where(contact)[0]
        if len(env_ids) == 0:
            return
        target = self._target()
        racket_pos = target.racket_pos_w[env_ids]
        racket_vel = target.racket_lin_vel_w[env_ids]
        racket_normal = target.racket_normal_w[env_ids]
        planner_vel = target.racket_target_vel_w[env_ids]
        planner_normal = target.racket_target_normal_w[env_ids]
        incoming_velocity = self.previous_velocity[env_ids]
        intended_incoming_velocity = target.incoming_ball_vel_w[env_ids]

        self.contact_true_target_error[env_ids] = torch.linalg.norm(
            position[env_ids] - target.ball_strike_pos_w[env_ids], dim=-1
        )
        self.contact_planner_position_error[env_ids] = torch.linalg.norm(
            racket_pos - target.racket_target_pos_w[env_ids], dim=-1
        )
        self.contact_planner_velocity_error[env_ids] = torch.linalg.norm(
            racket_vel - planner_vel, dim=-1
        )
        racket_speed = torch.linalg.norm(
            racket_vel, dim=-1
        ).clamp_min(1.0e-6)
        planner_speed = torch.linalg.norm(
            planner_vel, dim=-1
        ).clamp_min(1.0e-6)
        velocity_cos = torch.sum(
            racket_vel * planner_vel, dim=-1
        ) / (racket_speed * planner_speed)
        self.contact_planner_velocity_direction_error_deg[env_ids] = (
            torch.rad2deg(torch.acos(velocity_cos.clamp(-1.0, 1.0)))
        )
        normal_cos = torch.abs(
            torch.sum(racket_normal * planner_normal, dim=-1)
        )
        self.contact_planner_normal_error_deg[env_ids] = torch.rad2deg(
            torch.acos(normal_cos.clamp(-1.0, 1.0))
        )
        self.contact_outgoing_velocity_error[env_ids] = torch.linalg.norm(
            velocity[env_ids] - target.ball_outgoing_target_vel_w[env_ids],
            dim=-1,
        )
        outgoing_velocity = velocity[env_ids]
        target_outgoing_velocity = target.ball_outgoing_target_vel_w[
            env_ids
        ]
        outgoing_speed = torch.linalg.norm(
            outgoing_velocity, dim=-1
        ).clamp_min(1.0e-6)
        target_outgoing_speed = torch.linalg.norm(
            target_outgoing_velocity, dim=-1
        ).clamp_min(1.0e-6)
        outgoing_cosine = torch.sum(
            outgoing_velocity * target_outgoing_velocity, dim=-1
        ) / (outgoing_speed * target_outgoing_speed)
        self.contact_outgoing_speed[env_ids] = outgoing_speed
        self.contact_target_outgoing_speed[env_ids] = (
            target_outgoing_speed
        )
        self.contact_outgoing_speed_ratio[env_ids] = (
            outgoing_speed / target_outgoing_speed
        )
        self.contact_outgoing_direction_error_deg[env_ids] = (
            torch.rad2deg(
                torch.acos(outgoing_cosine.clamp(-1.0, 1.0))
            )
        )
        self.contact_outgoing_velocity_x[env_ids] = outgoing_velocity[:, 0]
        self.contact_outgoing_velocity_y[env_ids] = outgoing_velocity[:, 1]
        self.contact_outgoing_velocity_z[env_ids] = outgoing_velocity[:, 2]
        self.contact_target_outgoing_velocity_x[env_ids] = (
            target_outgoing_velocity[:, 0]
        )
        self.contact_target_outgoing_velocity_y[env_ids] = (
            target_outgoing_velocity[:, 1]
        )
        self.contact_target_outgoing_velocity_z[env_ids] = (
            target_outgoing_velocity[:, 2]
        )
        impact_normal = target.racket_impact_target_normal_w[env_ids]
        impact_velocity = target.racket_impact_target_vel_w[env_ids]
        restitution = float(target.cfg.paddle_restitution)
        self.contact_incoming_velocity_error[env_ids] = torch.linalg.norm(
            incoming_velocity - intended_incoming_velocity, dim=-1
        )
        incoming_speed = torch.linalg.norm(
            incoming_velocity, dim=-1
        ).clamp_min(1.0e-6)
        intended_incoming_speed = torch.linalg.norm(
            intended_incoming_velocity, dim=-1
        ).clamp_min(1.0e-6)
        incoming_cosine = torch.sum(
            incoming_velocity * intended_incoming_velocity, dim=-1
        ) / (incoming_speed * intended_incoming_speed)
        self.contact_incoming_velocity_direction_error_deg[env_ids] = (
            torch.rad2deg(
                torch.acos(incoming_cosine.clamp(-1.0, 1.0))
            )
        )
        route_incoming_velocity = self.route_incoming_velocity[env_ids]
        self.contact_route_incoming_velocity_error[
            env_ids
        ] = torch.linalg.norm(
            route_incoming_velocity - intended_incoming_velocity,
            dim=-1,
        )
        route_incoming_speed = torch.linalg.norm(
            route_incoming_velocity, dim=-1
        ).clamp_min(1.0e-6)
        route_incoming_cosine = torch.sum(
            route_incoming_velocity * intended_incoming_velocity,
            dim=-1,
        ) / (route_incoming_speed * intended_incoming_speed)
        self.contact_route_incoming_velocity_direction_error_deg[
            env_ids
        ] = torch.rad2deg(
            torch.acos(route_incoming_cosine.clamp(-1.0, 1.0))
        )
        self.contact_actual_route_incoming_velocity_error[
            env_ids
        ] = torch.linalg.norm(
            incoming_velocity - route_incoming_velocity, dim=-1
        )

        def predict(
            candidate_incoming_velocity: torch.Tensor,
            candidate_racket_velocity: torch.Tensor,
            candidate_normal: torch.Tensor,
            tangential_damping: float,
            tangential_cap: float,
        ) -> torch.Tensor:
            return moving_plane_impact_velocity(
                candidate_incoming_velocity,
                candidate_racket_velocity,
                candidate_normal,
                restitution=restitution,
                tangential_damping=tangential_damping,
                tangential_cap=tangential_cap,
            )

        command_predictions = {
            "normal_only": predict(
                intended_incoming_velocity,
                impact_velocity,
                impact_normal,
                0.0,
                0.0,
            ),
            "training": predict(
                intended_incoming_velocity,
                impact_velocity,
                impact_normal,
                1.0 - float(target.cfg.paddle_tangent_retain),
                1.0e6,
            ),
            "physics": predict(
                intended_incoming_velocity,
                impact_velocity,
                impact_normal,
                float(self.cfg.paddle_tangential_damping),
                float(self.cfg.paddle_tangential_cap),
            ),
        }
        actual_predictions = {
            "normal_only": predict(
                incoming_velocity,
                racket_vel,
                racket_normal,
                0.0,
                0.0,
            ),
            "training": predict(
                incoming_velocity,
                racket_vel,
                racket_normal,
                1.0 - float(target.cfg.paddle_tangent_retain),
                1.0e6,
            ),
            "physics": predict(
                incoming_velocity,
                racket_vel,
                racket_normal,
                float(self.cfg.paddle_tangential_damping),
                float(self.cfg.paddle_tangential_cap),
            ),
        }
        command_metric_names = {
            "normal_only": "contact_command_normal_only_error",
            "training": "contact_command_training_model_error",
            "physics": "contact_command_physics_model_error",
        }
        for model_name, prediction in command_predictions.items():
            getattr(self, command_metric_names[model_name])[
                env_ids
            ] = torch.linalg.norm(
                prediction - target_outgoing_velocity, dim=-1
            )
        for model_name, prediction in actual_predictions.items():
            getattr(
                self, f"contact_actual_{model_name}_model_residual"
            )[env_ids] = torch.linalg.norm(
                prediction - outgoing_velocity, dim=-1
            )
            getattr(
                self,
                f"contact_actual_{model_name}_predicted_target_error",
            )[env_ids] = torch.linalg.norm(
                prediction - target_outgoing_velocity, dim=-1
            )
        force_vector = racket_force_vector[env_ids]
        force_norm = torch.linalg.norm(
            force_vector, dim=-1, keepdim=True
        )
        force_valid = force_norm.squeeze(-1) >= float(
            self.cfg.contact_force
        )
        force_direction = torch.where(
            force_valid.unsqueeze(-1),
            force_vector / force_norm.clamp_min(1.0e-9),
            racket_normal,
        )
        force_direction_cosine = torch.abs(
            torch.sum(force_direction * racket_normal, dim=-1)
        )
        self.contact_force_direction_valid[env_ids] = force_valid.float()
        self.contact_force_vs_racket_normal_angle_deg[
            env_ids
        ] = torch.rad2deg(
            torch.acos(force_direction_cosine.clamp(-1.0, 1.0))
        )
        force_physics_prediction = predict(
            incoming_velocity,
            racket_vel,
            force_direction,
            float(self.cfg.paddle_tangential_damping),
            float(self.cfg.paddle_tangential_cap),
        )
        self.contact_actual_force_direction_model_residual[
            env_ids
        ] = torch.linalg.norm(
            force_physics_prediction - outgoing_velocity, dim=-1
        )
        self.contact_actual_force_direction_predicted_target_error[
            env_ids
        ] = torch.linalg.norm(
            force_physics_prediction - target_outgoing_velocity,
            dim=-1,
        )

        (
            detailed_valid_all,
            detailed_force_all,
            detailed_point_all,
            detailed_normal_all,
            detailed_separation_all,
        ) = detailed_contact
        detailed_valid = detailed_valid_all[env_ids]
        detailed_normal = detailed_normal_all[env_ids]
        normal_alignment = torch.sum(
            detailed_normal * racket_normal, dim=-1, keepdim=True
        )
        detailed_normal = torch.where(
            normal_alignment < 0.0,
            -detailed_normal,
            detailed_normal,
        )
        detailed_normal = detailed_normal / torch.linalg.norm(
            detailed_normal, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)
        detailed_normal_cosine = torch.abs(
            torch.sum(detailed_normal * racket_normal, dim=-1)
        )
        self.contact_physx_data_valid[env_ids] = detailed_valid.float()
        self.contact_physx_force[env_ids] = detailed_force_all[env_ids]
        self.contact_physx_separation[env_ids] = (
            detailed_separation_all[env_ids]
        )
        self.contact_physx_normal_error_deg[env_ids] = torch.where(
            detailed_valid,
            torch.rad2deg(
                torch.acos(detailed_normal_cosine.clamp(-1.0, 1.0))
            ),
            torch.zeros_like(detailed_normal_cosine),
        )
        detailed_point_offset = detailed_point_all[env_ids] - racket_pos
        detailed_point_normal_offset = torch.sum(
            detailed_point_offset * racket_normal,
            dim=-1,
            keepdim=True,
        )
        detailed_point_tangent_offset = (
            detailed_point_offset
            - detailed_point_normal_offset * racket_normal
        )
        self.contact_physx_point_radial_error[env_ids] = torch.where(
            detailed_valid,
            torch.linalg.norm(detailed_point_tangent_offset, dim=-1),
            torch.zeros_like(detailed_normal_cosine),
        )
        detailed_prediction = predict(
            incoming_velocity,
            racket_vel,
            detailed_normal,
            float(self.cfg.paddle_tangential_damping),
            float(self.cfg.paddle_tangential_cap),
        )
        detailed_residual = torch.linalg.norm(
            detailed_prediction - outgoing_velocity, dim=-1
        )
        detailed_target_error = torch.linalg.norm(
            detailed_prediction - target_outgoing_velocity, dim=-1
        )
        self.contact_actual_physx_normal_model_residual[
            env_ids
        ] = torch.where(
            detailed_valid,
            detailed_residual,
            torch.zeros_like(detailed_residual),
        )
        self.contact_actual_physx_normal_predicted_target_error[
            env_ids
        ] = torch.where(
            detailed_valid,
            detailed_target_error,
            torch.zeros_like(detailed_target_error),
        )

        substep_valid = self._substep_contact_latched[env_ids]
        substep_pre_ball = self._substep_contact_pre_ball_velocity[env_ids]
        substep_post_ball = self._substep_contact_post_ball_velocity[env_ids]
        substep_pre_ball_ang = (
            self._substep_contact_pre_ball_ang_velocity[env_ids]
        )
        substep_post_ball_ang = (
            self._substep_contact_post_ball_ang_velocity[env_ids]
        )
        substep_pre_com_velocity = (
            self._substep_contact_pre_racket_com_velocity[env_ids]
        )
        substep_pre_point_velocity = (
            self._substep_contact_pre_racket_point_velocity[env_ids]
        )
        substep_post_point_velocity = (
            self._substep_contact_post_racket_point_velocity[env_ids]
        )
        substep_normal = self._substep_contact_normal[env_ids]
        substep_racket_normal = self._substep_contact_racket_normal[env_ids]
        substep_point_offset = (
            self._substep_contact_point[env_ids]
            - self._substep_contact_racket_center_position[env_ids]
        )
        substep_point_normal_offset = torch.sum(
            substep_point_offset * substep_racket_normal, dim=-1
        )
        substep_point_tangent_offset = (
            substep_point_offset
            - substep_point_normal_offset.unsqueeze(-1) * substep_racket_normal
        )
        substep_normal_cosine = torch.abs(
            torch.sum(substep_normal * substep_racket_normal, dim=-1)
        )
        ball_contact_offset = (
            self._substep_contact_point[env_ids]
            - self._substep_contact_pre_ball_position[env_ids]
        )
        pre_ball_surface_velocity = (
            substep_pre_ball
            + torch.linalg.cross(
                substep_pre_ball_ang,
                ball_contact_offset,
                dim=-1,
            )
        )
        pre_relative_surface_velocity = (
            pre_ball_surface_velocity - substep_pre_point_velocity
        )
        pre_relative_surface_normal = torch.sum(
            pre_relative_surface_velocity * substep_normal,
            dim=-1,
            keepdim=True,
        )
        pre_relative_surface_tangent = (
            pre_relative_surface_velocity
            - pre_relative_surface_normal * substep_normal
        )
        substep_link_prediction = predict(
            substep_pre_ball,
            substep_pre_com_velocity,
            substep_normal,
            float(self.cfg.paddle_tangential_damping),
            float(self.cfg.paddle_tangential_cap),
        )
        substep_point_prediction = predict(
            substep_pre_ball,
            substep_pre_point_velocity,
            substep_normal,
            float(self.cfg.paddle_tangential_damping),
            float(self.cfg.paddle_tangential_cap),
        )
        substep_link_residual = torch.linalg.norm(
            substep_link_prediction - substep_post_ball, dim=-1
        )
        substep_point_residual = torch.linalg.norm(
            substep_point_prediction - substep_post_ball, dim=-1
        )
        substep_target_error = torch.linalg.norm(
            substep_post_ball - target_outgoing_velocity, dim=-1
        )
        restitution_measured, restitution_valid = measured_normal_restitution(
            substep_pre_ball,
            substep_post_ball,
            substep_pre_point_velocity,
            substep_post_point_velocity,
            substep_normal,
        )
        metric_values = {
            "contact_physx_substep_normal_error_deg": torch.rad2deg(
                torch.acos(substep_normal_cosine.clamp(-1.0, 1.0))
            ),
            "contact_physx_substep_point_radial_error": torch.linalg.norm(
                substep_point_tangent_offset, dim=-1
            ),
            "contact_physx_substep_point_normal_offset": (
                substep_point_normal_offset
            ),
            "contact_physx_patch_substeps": (
                self._substep_contact_patch_steps[env_ids].float()
            ),
            "contact_physx_normal_impulse": (
                self._substep_contact_normal_impulse[env_ids]
            ),
            "contact_physx_capture_lag_s": (
                self._substep_contact_age_steps[env_ids].float()
                * float(self._env.scene.physics_dt)
            ),
            "contact_physx_contact_point_speed_delta": torch.linalg.norm(
                substep_pre_point_velocity - substep_pre_com_velocity,
                dim=-1,
            ),
            "contact_physx_substep_link_model_residual": substep_link_residual,
            "contact_physx_substep_point_model_residual": (
                substep_point_residual
            ),
            "contact_physx_substep_outgoing_target_error": (
                substep_target_error
            ),
            "contact_physx_measured_restitution": torch.where(
                restitution_valid,
                restitution_measured,
                torch.zeros_like(restitution_measured),
            ),
            "contact_physx_pre_ball_speed": torch.linalg.norm(
                substep_pre_ball, dim=-1
            ),
            "contact_physx_post_ball_speed": torch.linalg.norm(
                substep_post_ball, dim=-1
            ),
            "contact_physx_point_x": self._substep_contact_point[env_ids, 0],
            "contact_physx_point_y": self._substep_contact_point[env_ids, 1],
            "contact_physx_point_z": self._substep_contact_point[env_ids, 2],
            "contact_physx_normal_x": substep_normal[:, 0],
            "contact_physx_normal_y": substep_normal[:, 1],
            "contact_physx_normal_z": substep_normal[:, 2],
            "contact_physx_pre_ball_velocity_x": substep_pre_ball[:, 0],
            "contact_physx_pre_ball_velocity_y": substep_pre_ball[:, 1],
            "contact_physx_pre_ball_velocity_z": substep_pre_ball[:, 2],
            "contact_physx_post_ball_velocity_x": substep_post_ball[:, 0],
            "contact_physx_post_ball_velocity_y": substep_post_ball[:, 1],
            "contact_physx_post_ball_velocity_z": substep_post_ball[:, 2],
            "contact_physx_pre_racket_point_velocity_x": (
                substep_pre_point_velocity[:, 0]
            ),
            "contact_physx_pre_racket_point_velocity_y": (
                substep_pre_point_velocity[:, 1]
            ),
            "contact_physx_pre_racket_point_velocity_z": (
                substep_pre_point_velocity[:, 2]
            ),
            "contact_physx_post_racket_point_velocity_x": (
                substep_post_point_velocity[:, 0]
            ),
            "contact_physx_post_racket_point_velocity_y": (
                substep_post_point_velocity[:, 1]
            ),
            "contact_physx_post_racket_point_velocity_z": (
                substep_post_point_velocity[:, 2]
            ),
            "contact_physx_pre_ball_angular_velocity_x": (
                substep_pre_ball_ang[:, 0]
            ),
            "contact_physx_pre_ball_angular_velocity_y": (
                substep_pre_ball_ang[:, 1]
            ),
            "contact_physx_pre_ball_angular_velocity_z": (
                substep_pre_ball_ang[:, 2]
            ),
            "contact_physx_post_ball_angular_velocity_x": (
                substep_post_ball_ang[:, 0]
            ),
            "contact_physx_post_ball_angular_velocity_y": (
                substep_post_ball_ang[:, 1]
            ),
            "contact_physx_post_ball_angular_velocity_z": (
                substep_post_ball_ang[:, 2]
            ),
            "contact_physx_pre_ball_surface_tangent_speed": (
                torch.linalg.norm(
                    pre_relative_surface_tangent,
                    dim=-1,
                )
            ),
            "contact_physx_post_ball_spin_speed": torch.linalg.norm(
                substep_post_ball_ang,
                dim=-1,
            ),
        }
        self.contact_physx_substep_valid[env_ids] = substep_valid.float()
        for metric_name, metric_value in metric_values.items():
            getattr(self, metric_name)[env_ids] = torch.where(
                substep_valid,
                metric_value,
                torch.zeros_like(metric_value),
            )
        impact_normal_cosine = torch.abs(
            torch.sum(racket_normal * impact_normal, dim=-1)
        )
        self.contact_impact_normal_error_deg[env_ids] = torch.rad2deg(
            torch.acos(impact_normal_cosine.clamp(-1.0, 1.0))
        )
        wire_impact_cosine = torch.abs(
            torch.sum(planner_normal * impact_normal, dim=-1)
        )
        self.contact_wire_impact_normal_gap_deg[env_ids] = torch.rad2deg(
            torch.acos(wire_impact_cosine.clamp(-1.0, 1.0))
        )
        ball_offset = position[env_ids] - racket_pos
        normal_offset = torch.sum(
            ball_offset * racket_normal, dim=-1, keepdim=True
        )
        tangent_offset = ball_offset - normal_offset * racket_normal
        self.contact_face_radial_error[env_ids] = torch.linalg.norm(
            tangent_offset, dim=-1
        )
        self.contact_time_to_strike[env_ids] = target.time_to_strike[env_ids]

    def _refresh_post_bounce_commands(
        self,
        refresh_candidate: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
    ) -> None:
        """Refresh actor commands from the latest post-bounce PhysX state."""
        if not bool(self.cfg.post_bounce_command_refresh_enabled):
            return
        candidate_ids = torch.where(refresh_candidate)[0]
        if len(candidate_ids) == 0:
            return

        target = self._target()
        old_strike_position = target.ball_strike_pos_w[candidate_ids].clone()
        old_incoming_velocity = target.incoming_ball_vel_w[candidate_ids].clone()
        old_true_tts = target.true_time_to_strike[candidate_ids].clone()
        old_racket_velocity = target._strike_racket_target_vel_w[
            candidate_ids
        ].clone()
        old_racket_normal = target._strike_racket_target_normal_w[
            candidate_ids
        ].clone()
        prediction_kwargs = {
            "gravity": float(self.cfg.gravity),
            "drag_k": float(self.cfg.drag_k),
            "max_time": float(self.cfg.command_prediction_max_time),
            "minimum_x_speed": float(
                self.cfg.command_prediction_minimum_x_speed
            ),
        }
        if str(self.cfg.command_prediction_mode) == "linearized":
            prediction = predict_linearized_drag_plane_crossing(
                position[candidate_ids],
                velocity[candidate_ids],
                old_strike_position[:, 0],
                iterations=int(self.cfg.command_prediction_iterations),
                **prediction_kwargs,
            )
        elif str(self.cfg.command_prediction_mode) == "rk4":
            prediction = predict_drag_plane_crossing(
                position[candidate_ids],
                velocity[candidate_ids],
                old_strike_position[:, 0],
                max_dt=float(self.cfg.command_prediction_max_dt),
                **prediction_kwargs,
            )
        else:
            raise ValueError(
                "command_prediction_mode must be 'linearized' or 'rk4'"
            )
        valid = (
            prediction.valid
            & (prediction.time > float(self.cfg.command_refresh_freeze_tts_s))
            & torch.isfinite(prediction.position).all(dim=-1)
            & torch.isfinite(prediction.velocity).all(dim=-1)
        )
        refresh_ids = candidate_ids[valid]
        if len(refresh_ids) == 0:
            return

        predicted_position = prediction.position[valid]
        predicted_velocity = prediction.velocity[valid]
        predicted_tts = prediction.time[valid]
        target.refresh_from_physical_prediction(
            refresh_ids,
            predicted_position,
            predicted_velocity,
            predicted_tts,
            solution_blend=float(self.cfg.command_refresh_solution_blend),
            update_dynamic_station=bool(
                self.cfg.command_refresh_update_dynamic_station
            ),
        )

        old_position = old_strike_position[valid]
        old_incoming = old_incoming_velocity[valid]
        old_tts = old_true_tts[valid]
        old_velocity_command = old_racket_velocity[valid]
        old_normal_command = old_racket_normal[valid]
        new_velocity_command = target._strike_racket_target_vel_w[refresh_ids]
        new_normal_command = target._strike_racket_target_normal_w[refresh_ids]
        normal_cosine = torch.sum(
            old_normal_command * new_normal_command, dim=-1
        ) / (
            torch.linalg.norm(old_normal_command, dim=-1)
            * torch.linalg.norm(new_normal_command, dim=-1)
        ).clamp_min(1.0e-6)
        self.command_refresh_event[refresh_ids] = True
        self.command_refresh_valid[refresh_ids] = 1.0
        self.command_refresh_count[refresh_ids] += 1.0
        self.command_refresh_tts_s[refresh_ids] = predicted_tts
        self.command_refresh_position_delta[refresh_ids] = torch.linalg.norm(
            predicted_position - old_position, dim=-1
        )
        self.command_refresh_incoming_velocity_delta[
            refresh_ids
        ] = torch.linalg.norm(predicted_velocity - old_incoming, dim=-1)
        self.command_refresh_timing_delta_s[refresh_ids] = (
            predicted_tts - old_tts
        )
        self.command_refresh_racket_velocity_delta[
            refresh_ids
        ] = torch.linalg.norm(
            new_velocity_command - old_velocity_command, dim=-1
        )
        self.command_refresh_normal_delta_deg[refresh_ids] = torch.rad2deg(
            torch.acos(normal_cosine.clamp(-1.0, 1.0))
        )

    def _update_active_flights(self) -> None:
        active = (
            (self.phase == int(PhysicalShadowPhase.INCOMING_PRE_BOUNCE))
            | (self.phase == int(PhysicalShadowPhase.INCOMING_POST_BOUNCE))
            | (self.phase == int(PhysicalShadowPhase.OUTGOING))
        ) & (~self.serve_event)
        if not bool(torch.any(active)):
            return
        target = self._target()
        position = self.ball.data.root_pos_w
        velocity = self.ball.data.root_lin_vel_w
        racket_position = target.racket_pos_w
        distance = torch.linalg.norm(position - racket_position, dim=-1)
        self.minimum_ball_racket_distance = torch.where(
            active,
            torch.minimum(self.minimum_ball_racket_distance, distance),
            self.minimum_ball_racket_distance,
        )
        (
            table_near_x,
            net_x,
            table_far_x,
            surface_z,
            table_y_min,
            table_y_max,
        ) = self._table_geometry()

        bounce = detect_table_bounce(
            self.phase,
            self.previous_position,
            self.previous_velocity,
            position,
            velocity,
            table_surface_z=surface_z,
            ball_radius=float(target.cfg.ball_radius),
            table_near_x=table_near_x,
            net_x=net_x,
            table_y_min=table_y_min,
            table_y_max=table_y_max,
        )
        self.incoming_bounce_event |= bounce
        self.incoming_bounce_latch |= bounce
        self.phase[bounce] = int(
            PhysicalShadowPhase.INCOMING_POST_BOUNCE
        )
        refresh_candidate = bounce
        if bool(self.cfg.command_refresh_continuous_post_bounce):
            refresh_candidate = (
                self.phase
                == int(PhysicalShadowPhase.INCOMING_POST_BOUNCE)
            )
        self._refresh_post_bounce_commands(
            refresh_candidate, position, velocity
        )

        racket_force_vector = self._racket_force_vector()
        detailed_contact = self._read_detailed_contact_data()
        contact = detect_racket_contact(
            self.phase,
            self.previous_velocity,
            position,
            velocity,
            racket_position,
            torch.linalg.norm(racket_force_vector, dim=-1),
            contact_distance=float(self.cfg.contact_distance),
            contact_force=float(self.cfg.contact_force),
            velocity_jump=float(self.cfg.contact_velocity_jump),
        )
        contact |= self._substep_contact_latched & (
            (
                self.phase
                == int(PhysicalShadowPhase.INCOMING_PRE_BOUNCE)
            )
            | (
                self.phase
                == int(PhysicalShadowPhase.INCOMING_POST_BOUNCE)
            )
        )
        self._record_contact_diagnostics(
            contact,
            position,
            velocity,
            racket_force_vector,
            detailed_contact,
        )
        self.contact_event |= contact
        self.contact_latch |= contact
        self.phase[contact] = int(PhysicalShadowPhase.OUTGOING)

        net_cross = detect_net_cross(
            self.phase,
            self.previous_position,
            position,
            net_x=net_x,
            net_top_z=surface_z + float(target.cfg.net_height),
            ball_radius=float(target.cfg.ball_radius),
        ) & (~self.net_cross_latch)
        self.net_cross_event |= net_cross
        self.net_cross_latch |= net_cross

        incoming_net_collision = detect_incoming_net_collision(
            self.phase,
            self.previous_position,
            self.previous_velocity,
            position,
            velocity,
            net_x=net_x,
            net_top_z=surface_z + float(target.cfg.net_height),
            ball_radius=float(target.cfg.ball_radius),
        )
        self.incoming_net_collision_event |= incoming_net_collision
        self.phase[incoming_net_collision] = int(
            PhysicalShadowPhase.TERMINAL
        )

        landing = detect_outgoing_landing(
            self.phase,
            self.net_cross_latch,
            self.previous_position,
            self.previous_velocity,
            position,
            velocity,
            table_surface_z=surface_z,
            ball_radius=float(target.cfg.ball_radius),
            net_x=net_x,
            table_far_x=table_far_x,
            table_y_min=table_y_min,
            table_y_max=table_y_max,
        )
        landing_event = landing.event & (~self.opponent_bounce_latch)
        opponent_bounce = landing.opponent & landing_event
        self.outgoing_landing_event |= landing_event
        self.opponent_bounce_event |= opponent_bounce
        self.landing_short_event |= landing.short & landing_event
        self.landing_long_event |= landing.long & landing_event
        self.landing_side_event |= landing.side & landing_event
        self.landing_no_net_event |= landing.no_net & landing_event
        self.opponent_bounce_latch |= opponent_bounce
        landing_ids = torch.where(landing_event)[0]
        if len(landing_ids) > 0:
            landing_position = landing.position[landing_ids]
            self.landing_position_x[landing_ids] = landing_position[:, 0]
            self.landing_position_y[landing_ids] = landing_position[:, 1]
            valid_target = target.ball_target_landing_valid[landing_ids]
            target_xy = target.ball_target_landing_w[landing_ids, :2]
            landing_delta = landing_position[:, :2] - target_xy
            landing_error = torch.linalg.norm(landing_delta, dim=-1)
            self.landing_target_error[landing_ids] = torch.where(
                valid_target,
                landing_error,
                torch.zeros_like(landing_error),
            )
            self.landing_target_x_error[landing_ids] = torch.where(
                valid_target,
                landing_delta[:, 0],
                torch.zeros_like(landing_delta[:, 0]),
            )
            self.landing_target_y_error[landing_ids] = torch.where(
                valid_target,
                landing_delta[:, 1],
                torch.zeros_like(landing_delta[:, 1]),
            )
        self.phase[landing_event] = int(PhysicalShadowPhase.TERMINAL)

        self.flight_elapsed[active] += float(self._env.step_dt)
        timed_out = active & (~landing_event) & (
            ~incoming_net_collision
        ) & (
            self.flight_elapsed >= float(self.cfg.maximum_flight_time)
        )
        self.timeout_event |= timed_out
        self.phase[timed_out] = int(PhysicalShadowPhase.TERMINAL)

        self.previous_position[active] = position[active]
        self.previous_velocity[active] = velocity[active]
        terminal_ids = torch.where(
            landing_event | incoming_net_collision | timed_out
        )[0]
        if len(terminal_ids) > 0:
            self._park(terminal_ids)

    def _publish_metrics(self) -> None:
        self.metrics["phase"] = self.phase.float()
        self.metrics["route_valid"] = self.route_valid.float()
        self.metrics["route_curriculum_level"] = self.route_curriculum_level
        self.metrics["serve_event"] = self.serve_event.float()
        self.metrics["incoming_bounce_event"] = (
            self.incoming_bounce_event.float()
        )
        self.metrics["contact_event"] = self.contact_event.float()
        self.metrics["net_cross_event"] = self.net_cross_event.float()
        self.metrics["outgoing_landing_event"] = (
            self.outgoing_landing_event.float()
        )
        self.metrics["opponent_bounce_event"] = (
            self.opponent_bounce_event.float()
        )
        self.metrics["landing_short_event"] = (
            self.landing_short_event.float()
        )
        self.metrics["landing_long_event"] = (
            self.landing_long_event.float()
        )
        self.metrics["landing_side_event"] = (
            self.landing_side_event.float()
        )
        self.metrics["landing_no_net_event"] = (
            self.landing_no_net_event.float()
        )
        self.metrics["abort_event"] = self.abort_event.float()
        self.metrics["timeout_event"] = self.timeout_event.float()
        self.metrics["route_unalignable_event"] = (
            self.route_unalignable_event.float()
        )
        self.metrics["route_invalid_event"] = (
            self.route_invalid_event.float()
        )
        self.metrics["incoming_net_collision_event"] = (
            self.incoming_net_collision_event.float()
        )
        self.metrics["reset_abort_event"] = self.reset_abort_event.float()
        self.metrics["command_refresh_event"] = (
            self.command_refresh_event.float()
        )
        self.metrics["incoming_bounce_latch"] = (
            self.incoming_bounce_latch.float()
        )
        self.metrics["contact_latch"] = self.contact_latch.float()
        self.metrics["net_cross_latch"] = self.net_cross_latch.float()
        self.metrics["opponent_bounce_latch"] = (
            self.opponent_bounce_latch.float()
        )
        for name in (
            "launch_timing_error_s",
            "contact_true_target_error",
            "contact_planner_position_error",
            "contact_planner_velocity_error",
            "contact_planner_velocity_direction_error_deg",
            "contact_planner_normal_error_deg",
            "contact_outgoing_velocity_error",
            "contact_outgoing_speed",
            "contact_target_outgoing_speed",
            "contact_outgoing_speed_ratio",
            "contact_outgoing_direction_error_deg",
            "contact_outgoing_velocity_x",
            "contact_outgoing_velocity_y",
            "contact_outgoing_velocity_z",
            "contact_target_outgoing_velocity_x",
            "contact_target_outgoing_velocity_y",
            "contact_target_outgoing_velocity_z",
            "contact_incoming_velocity_error",
            "contact_incoming_velocity_direction_error_deg",
            "contact_route_incoming_velocity_error",
            "contact_route_incoming_velocity_direction_error_deg",
            "contact_actual_route_incoming_velocity_error",
            "contact_command_normal_only_error",
            "contact_command_training_model_error",
            "contact_command_physics_model_error",
            "contact_actual_training_model_residual",
            "contact_actual_physics_model_residual",
            "contact_actual_normal_only_model_residual",
            "contact_actual_training_predicted_target_error",
            "contact_actual_physics_predicted_target_error",
            "contact_actual_normal_only_predicted_target_error",
            "contact_force_direction_valid",
            "contact_force_vs_racket_normal_angle_deg",
            "contact_actual_force_direction_model_residual",
            "contact_actual_force_direction_predicted_target_error",
            "contact_physx_data_valid",
            "contact_physx_force",
            "contact_physx_separation",
            "contact_physx_normal_error_deg",
            "contact_physx_point_radial_error",
            "contact_actual_physx_normal_model_residual",
            "contact_actual_physx_normal_predicted_target_error",
            "contact_physx_substep_valid",
            "contact_physx_substep_normal_error_deg",
            "contact_physx_substep_point_radial_error",
            "contact_physx_substep_point_normal_offset",
            "contact_physx_patch_substeps",
            "contact_physx_normal_impulse",
            "contact_physx_capture_lag_s",
            "contact_physx_contact_point_speed_delta",
            "contact_physx_substep_link_model_residual",
            "contact_physx_substep_point_model_residual",
            "contact_physx_substep_outgoing_target_error",
            "contact_physx_measured_restitution",
            "contact_physx_pre_ball_speed",
            "contact_physx_post_ball_speed",
            "contact_physx_point_x",
            "contact_physx_point_y",
            "contact_physx_point_z",
            "contact_physx_normal_x",
            "contact_physx_normal_y",
            "contact_physx_normal_z",
            "contact_physx_pre_ball_velocity_x",
            "contact_physx_pre_ball_velocity_y",
            "contact_physx_pre_ball_velocity_z",
            "contact_physx_post_ball_velocity_x",
            "contact_physx_post_ball_velocity_y",
            "contact_physx_post_ball_velocity_z",
            "contact_physx_pre_racket_point_velocity_x",
            "contact_physx_pre_racket_point_velocity_y",
            "contact_physx_pre_racket_point_velocity_z",
            "contact_physx_post_racket_point_velocity_x",
            "contact_physx_post_racket_point_velocity_y",
            "contact_physx_post_racket_point_velocity_z",
            "contact_physx_pre_ball_angular_velocity_x",
            "contact_physx_pre_ball_angular_velocity_y",
            "contact_physx_pre_ball_angular_velocity_z",
            "contact_physx_post_ball_angular_velocity_x",
            "contact_physx_post_ball_angular_velocity_y",
            "contact_physx_post_ball_angular_velocity_z",
            "contact_physx_pre_ball_surface_tangent_speed",
            "contact_physx_post_ball_spin_speed",
            "contact_impact_normal_error_deg",
            "contact_wire_impact_normal_gap_deg",
            "contact_face_radial_error",
            "contact_time_to_strike",
            "landing_target_error",
            "landing_target_x_error",
            "landing_target_y_error",
            "landing_position_x",
            "landing_position_y",
            "command_refresh_valid",
            "command_refresh_count",
            "command_refresh_tts_s",
            "command_refresh_position_delta",
            "command_refresh_incoming_velocity_delta",
            "command_refresh_timing_delta_s",
            "command_refresh_racket_velocity_delta",
            "command_refresh_normal_delta_deg",
        ):
            self.metrics[name] = getattr(self, name)
        self.metrics["minimum_ball_racket_distance"] = torch.where(
            torch.isfinite(self.minimum_ball_racket_distance),
            self.minimum_ball_racket_distance,
            torch.zeros_like(self.minimum_ball_racket_distance),
        )

    def _clear_step_events(self) -> None:
        for name in (
            "serve_event",
            "incoming_bounce_event",
            "contact_event",
            "net_cross_event",
            "outgoing_landing_event",
            "opponent_bounce_event",
            "landing_short_event",
            "landing_long_event",
            "landing_side_event",
            "landing_no_net_event",
            "abort_event",
            "timeout_event",
            "route_unalignable_event",
            "route_invalid_event",
            "incoming_net_collision_event",
            "reset_abort_event",
            "command_refresh_event",
        ):
            getattr(self, name).zero_()

    def _update_metrics(self) -> None:
        self._publish_metrics()

    def prepare_pre_physics(self) -> None:
        """Launch an initial/reset route before the first policy action."""
        if not self._pre_physics_prepare_pending:
            return
        self._clear_step_events()
        self._launch_due_routes()
        self._publish_metrics()
        self._pre_physics_prepare_pending = False

    def prepare_reward_snapshot(self) -> None:
        """Resolve rigid-ball events once, after physics and before reward."""
        if self._reward_snapshot_prepared:
            return
        self._update_active_flights()
        self._publish_metrics()
        self._reward_snapshot_prepared = True

    def _update_command(self) -> None:
        self._clear_step_events()
        target = self._target()
        new_target = target.target_just_resampled
        active = self.phase != int(PhysicalShadowPhase.PARKED)
        aborted = new_target & active & (
            self.phase != int(PhysicalShadowPhase.TERMINAL)
        )
        self.abort_event |= aborted
        reset_ids = torch.where(new_target)[0]
        if len(reset_ids) > 0:
            self._clear_swing_state(reset_ids)
            self._park(reset_ids)
        self._launch_due_routes()
        # In the opt-in environment protocol the same post-physics state was
        # already resolved before reward. Do not advance contact/recovery twice.
        if self._reward_snapshot_prepared:
            self._reward_snapshot_prepared = False
        else:
            self._update_active_flights()
        self._publish_metrics()
        self._pre_physics_prepare_pending = False

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass


@configclass
class PhysicalBallShadowCommandCfg(CommandTermCfg):
    """Configuration for the reward-free rigid-ball shadow lifecycle."""

    class_type: type = PhysicalBallShadowCommand
    ball_asset_name: str = MISSING
    target_command_name: str = "racket_target"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    gravity: float = 9.81
    drag_k: float = 0.1261
    horizontal_retain: float = 0.631
    vertical_restitution: float = 0.9215
    route_integrator_max_dt: float = 0.005
    route_sample_attempts: int = 4
    route_batch_interval_steps: int = 5
    route_geometry_mode: str = "target_hidden"
    # Expand the independent rigid-ball route from a compact Core range to the
    # configured full range using measured task ability, never iteration count.
    route_ability_curriculum_enabled: bool = False
    route_ability_curriculum_start_level: float = 0.0
    route_ability_curriculum_full_level: float = 1.0
    route_easy_pre_bounce_time_range: tuple[float, float] = (0.55, 0.65)
    route_easy_post_bounce_time_range: tuple[float, float] = (0.34, 0.42)
    route_easy_bounce_dx_range: tuple[float, float] = (0.35, 0.62)
    route_easy_bounce_y_jitter_range: tuple[float, float] = (-0.05, 0.05)
    # Optional closed-loop command bridge.  The rigid ball stays hidden from
    # the actor; only the existing planner-command fields are revised after a
    # measured table bounce.
    post_bounce_command_refresh_enabled: bool = False
    # Re-solve from the same rigid ball's latest state at every control step
    # until the freeze horizon.  Keeping this opt-in preserves all historical
    # tasks while matching the deployed command lifecycle for V5 tasks.
    command_refresh_continuous_post_bounce: bool = False
    command_refresh_freeze_tts_s: float = 0.25
    command_refresh_solution_blend: float = 1.0
    command_refresh_update_dynamic_station: bool = True
    command_prediction_mode: str = "linearized"
    command_prediction_iterations: int = 3
    command_prediction_max_dt: float = 0.005
    command_prediction_max_time: float = 0.65
    command_prediction_minimum_x_speed: float = 0.05
    pre_bounce_time_range: tuple[float, float] = (0.55, 0.72)
    post_bounce_time_range: tuple[float, float] = (0.28, 0.44)
    bounce_dx_range: tuple[float, float] = (0.35, 0.72)
    bounce_y_jitter_range: tuple[float, float] = (-0.05, 0.05)
    minimum_bounce_dx: float = 0.18
    near_edge_margin: float = 0.08
    net_bounce_margin: float = 0.08
    side_margin: float = 0.05
    origin_net_margin: float = 0.10
    origin_far_margin: float = 0.30
    origin_side_margin: float = 0.02
    origin_height_range: tuple[float, float] = (0.20, 1.10)

    contact_distance: float = 0.12
    contact_force: float = 2.0
    contact_velocity_jump: float = 0.65
    detailed_contact_enabled: bool = True
    max_contact_points_per_env: int = 8
    paddle_tangential_damping: float = 0.52
    paddle_tangential_cap: float = 0.50
    maximum_flight_time: float = 2.5

    park_offset_x: float = 0.60
    park_offset_y: float = 0.30
    park_height_above_table: float = 0.80
