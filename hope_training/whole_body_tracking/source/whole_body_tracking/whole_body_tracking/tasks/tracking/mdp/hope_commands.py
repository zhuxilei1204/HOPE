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
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_mul,
    quat_rotate_inverse,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.mdp.ballistics import (
    GRAVITY as _GRAVITY,
    ballistic_velocity_from_landing as _ballistic_velocity_from_landing,
    ballistic_z_at_x as _ballistic_z_at_x,
)
from whole_body_tracking.tasks.tracking.mdp.closed_loop_v2 import (
    VALID_RECOVERY_TRIGGERS,
    achieved_outcome_tier,
    closed_cycle_success_event,
    deploy_ready_hold_mask,
    durable_ready_resolution_events,
    face_contact_region_masks,
    face_center_quality,
    lifecycle_phase_ids,
    lifecycle_hold_gate,
    one_hot_lifecycle,
    operational_terminal_events,
    recovery_outcome_bucket,
    recovery_phase_indices,
    recovery_peak_excess_increment,
    recovery_safety_envelope_violations,
    recovered_planner_velocity_settlement_components,
    recovery_trigger_event,
    interpolate_curriculum_range,
    station_relocation_resolution,
    terminal_quality_window_mask,
    wire_compatible_velocity,
)
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.command_metrics import (
    batched_metric_means_and_clear,
)
from whole_body_tracking.tasks.tracking.mdp.functional_ready import (
    update_consecutive_steps,
    weighted_geometric_mean,
)
from whole_body_tracking.tasks.tracking.mdp.ready_recovery import (
    active_score_progress,
    bounded_gaussian_score,
    directional_error_progress,
    joint_deadband_score,
    ready_curriculum_guards_satisfied,
    ready_curriculum_should_advance,
    ready_curriculum_stage_value,
    strike_health_floor_progress,
    update_survival_milestones,
    validate_survival_milestones,
)
from whole_body_tracking.tasks.tracking.mdp.single_cycle_curriculum import (
    SingleCycleAbility,
    ability_driven_single_cycle_deadline,
    ability_driven_single_cycle_probability,
)
from whole_body_tracking.tasks.tracking.mdp.table_workspace import (
    event_gated_scalar_curriculum_transition,
    hysteretic_curriculum_transition,
    interpolate_bounds,
    motion_seed_blend,
    windowed_curriculum_level,
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
        strike_health_floor_progress(
            source=str(cfg.impact_health_floor_progress_source),
            ability_level=0.0,
            targeted_attempt_ema=0.0,
            contact_ema=0.0,
            targeted_attempt_threshold=float(
                cfg.impact_health_floor_targeted_attempt_threshold
            ),
            contact_threshold=float(cfg.impact_health_floor_contact_threshold),
        )
        if float(cfg.no_command_ready_progress_clip) <= 0.0:
            raise ValueError("no_command_ready_progress_clip must be positive")
        if int(cfg.active_ready_required_consecutive_steps) < 1:
            raise ValueError("active_ready_required_consecutive_steps must be at least 1")
        if bool(cfg.active_ready_survival_milestones_enabled):
            validate_survival_milestones(
                cfg.active_ready_survival_milestone_steps,
                cfg.active_ready_survival_milestone_values,
            )
        if int(cfg.cycle_v2_required_consecutive_ready_steps) < 1:
            raise ValueError("cycle_v2_required_consecutive_ready_steps must be at least 1")
        if int(cfg.post_contact_ready_required_consecutive_steps) < 1:
            raise ValueError(
                "post_contact_ready_required_consecutive_steps must be at least 1"
            )
        if int(cfg.post_contact_ready_deadline_steps) < int(
            cfg.post_contact_ready_required_consecutive_steps
        ):
            raise ValueError(
                "post_contact_ready_deadline_steps must be no shorter than the "
                "required consecutive READY interval"
            )
        if float(cfg.post_contact_ready_diagnostic_horizon_s) <= 0.0:
            raise ValueError(
                "post_contact_ready_diagnostic_horizon_s must be positive"
            )
        if float(cfg.post_contact_ready_durable_deadline_s) > float(
            cfg.post_contact_ready_diagnostic_horizon_s
        ):
            raise ValueError(
                "durable READY deadline must not exceed the diagnostic horizon"
            )
        if not (
            0.0
            <= float(cfg.post_contact_ready_durable_min_delay_s)
            < float(cfg.post_contact_ready_durable_deadline_s)
        ):
            raise ValueError(
                "durable READY delay/deadline must satisfy "
                "0 <= delay < deadline"
            )
        if int(
            cfg.post_contact_ready_durable_required_consecutive_steps
        ) < 1:
            raise ValueError(
                "durable READY required consecutive steps must be positive"
            )
        if float(cfg.post_contact_ready_torso_x_min) > float(
            cfg.post_contact_ready_torso_x_max
        ):
            raise ValueError(
                "post_contact_ready_torso_x_min must not exceed its maximum"
            )
        if str(cfg.post_contact_ready_progress_error_mode) not in (
            "backlean",
            "bounded_interval",
        ):
            raise ValueError(
                "post_contact_ready_progress_error_mode must be 'backlean' "
                "or 'bounded_interval'"
            )
        if str(cfg.post_contact_ready_trigger) not in VALID_RECOVERY_TRIGGERS:
            raise ValueError(
                "post_contact_ready_trigger must be one of "
                f"{VALID_RECOVERY_TRIGGERS}, got {cfg.post_contact_ready_trigger!r}"
            )
        diagnostic_boundaries = tuple(
            float(value)
            for value in cfg.post_contact_ready_diagnostic_phase_boundaries_s
        )
        if len(diagnostic_boundaries) != 3:
            raise ValueError(
                "post_contact_ready_diagnostic_phase_boundaries_s must contain "
                "three boundaries"
            )
        if any(value <= 0.0 for value in diagnostic_boundaries) or any(
            right <= left
            for left, right in zip(
                diagnostic_boundaries, diagnostic_boundaries[1:]
            )
        ):
            raise ValueError(
                "post-contact recovery diagnostic boundaries must be positive "
                "and strictly increasing"
            )
        if str(cfg.planner_command_mode) not in ("legacy", "v4_wire_compatible"):
            raise ValueError(
                "planner_command_mode must be 'legacy' or 'v4_wire_compatible'"
            )
        if float(cfg.actuator_safety_overflow_threshold) <= 0.0:
            raise ValueError(
                "actuator_safety_overflow_threshold must be positive"
            )
        if int(cfg.actuator_safety_overflow_consecutive_steps) < 1:
            raise ValueError(
                "actuator_safety_overflow_consecutive_steps must be positive"
            )
        if (
            float(cfg.face_quality_inner_radius) < 0.0
            or float(cfg.face_quality_outer_radius)
            <= float(cfg.face_quality_inner_radius)
            or float(cfg.contact_radius)
            <= float(cfg.face_quality_outer_radius)
        ):
            raise ValueError(
                "face/contact radii must satisfy "
                "0 <= inner_radius < outer_radius < contact_radius"
            )
        workspace_level = float(cfg.table_workspace_fixed_level)
        if workspace_level != -1.0 and not 0.0 <= workspace_level <= 1.0:
            raise ValueError(
                "table_workspace_fixed_level must be -1 or a value in [0, 1]"
            )
        if str(cfg.table_workspace_level_source) not in (
            "ability",
            "station_relocation",
        ):
            raise ValueError(
                "table_workspace_level_source must be 'ability' or "
                "'station_relocation'"
            )
        if str(cfg.planner_perturb_curriculum_source) not in (
            "ability",
            "fixed",
        ):
            raise ValueError(
                "planner_perturb_curriculum_source must be 'ability' or 'fixed'"
            )
        if not 0.0 <= float(cfg.planner_perturb_fixed_scale) <= 1.0:
            raise ValueError("planner_perturb_fixed_scale must be in [0, 1]")
        curriculum_windows = {
            "table_workspace": (
                cfg.table_workspace_curriculum_start_level,
                cfg.table_workspace_curriculum_full_level,
            ),
            "planner_perturb": (
                cfg.planner_perturb_curriculum_start_level,
                cfg.planner_perturb_curriculum_full_level,
            ),
            "impact_inverse": (
                cfg.impact_inverse_command_curriculum_start_level,
                cfg.impact_inverse_command_curriculum_full_level,
            ),
            "one_bounce_speed": (
                cfg.one_bounce_speed_curriculum_start_level,
                cfg.one_bounce_speed_curriculum_full_level,
            ),
        }
        for name, (start_level, full_level) in curriculum_windows.items():
            try:
                windowed_curriculum_level(0.0, start_level, full_level)
            except ValueError as exc:
                raise ValueError(f"invalid {name} curriculum window") from exc
        if str(cfg.cycle_v2_settlement_tier_mode) not in (
            "required",
            "achieved",
        ):
            raise ValueError(
                "cycle_v2_settlement_tier_mode must be 'required' or 'achieved'"
            )
        if bool(cfg.one_bounce_speed_curriculum_enabled):
            interpolate_curriculum_range(
                cfg.one_bounce_easy_horizontal_speed_range,
                cfg.one_bounce_full_horizontal_speed_range,
                0.0,
            )
            interpolate_curriculum_range(
                cfg.one_bounce_easy_horizontal_speed_range,
                cfg.one_bounce_full_horizontal_speed_range,
                1.0,
            )
            easy_t_lo, easy_t_hi = (
                float(value)
                for value in cfg.one_bounce_easy_post_time_range
            )
            if easy_t_lo <= 0.0 or easy_t_hi < easy_t_lo:
                raise ValueError(
                    "one_bounce_easy_post_time_range must be positive and ordered"
                )
        if bool(cfg.station_relocation_enabled):
            relocation_ranges = tuple(cfg.station_relocation_abs_y_ranges)
            if not relocation_ranges:
                raise ValueError(
                    "station_relocation_abs_y_ranges must contain at least one range"
                )
            for schedule_name, schedule in (
                (
                    "station_relocation_settle_max_lin_vel_by_level",
                    cfg.station_relocation_settle_max_lin_vel_by_level,
                ),
                (
                    "station_relocation_settle_max_ang_vel_by_level",
                    cfg.station_relocation_settle_max_ang_vel_by_level,
                ),
                (
                    "station_relocation_settle_min_steps_by_level",
                    cfg.station_relocation_settle_min_steps_by_level,
                ),
                (
                    "station_relocation_hold_deadline_steps_by_level",
                    cfg.station_relocation_hold_deadline_steps_by_level,
                ),
            ):
                if schedule and len(schedule) != len(relocation_ranges):
                    raise ValueError(
                        f"{schedule_name} has {len(schedule)} values, expected "
                        f"{len(relocation_ranges)}"
                    )
            if any(
                int(value) < 1
                for value in cfg.station_relocation_hold_deadline_steps_by_level
            ):
                raise ValueError(
                    "station relocation hold deadlines must be positive"
                )
            if not (
                0.0
                <= float(cfg.station_relocation_arrival_floor)
                <= float(cfg.station_relocation_arrival_threshold)
                <= 1.0
            ):
                raise ValueError(
                    "station relocation arrival floor/threshold must be ordered in [0, 1]"
                )
            if not (
                0.0
                <= float(cfg.station_relocation_settle_floor)
                <= float(cfg.station_relocation_settle_threshold)
                <= 1.0
            ):
                raise ValueError(
                    "station relocation settle floor/threshold must be ordered in [0, 1]"
                )
            if int(cfg.station_relocation_required_advance_checks) < 1:
                raise ValueError(
                    "station_relocation_required_advance_checks must be positive"
                )
            if int(cfg.station_relocation_required_regress_checks) < 1:
                raise ValueError(
                    "station_relocation_required_regress_checks must be positive"
                )
        if bool(cfg.post_contact_ready_curriculum_enabled):
            schedules = {
                "torso_x_min": cfg.post_contact_ready_curriculum_torso_x_min,
                "torso_x_max": cfg.post_contact_ready_curriculum_torso_x_max,
                "max_torso_ang_vel": cfg.post_contact_ready_curriculum_max_torso_ang_vel,
                "max_base_lin_vel": cfg.post_contact_ready_curriculum_max_base_lin_vel,
                "max_base_ang_vel": cfg.post_contact_ready_curriculum_max_base_ang_vel,
                "max_racket_speed": cfg.post_contact_ready_curriculum_max_racket_speed,
                "min_feet_contact": cfg.post_contact_ready_curriculum_min_feet_contact,
                "required_consecutive_steps": (
                    cfg.post_contact_ready_curriculum_required_consecutive_steps
                ),
                "deadline_steps": cfg.post_contact_ready_curriculum_deadline_steps,
            }
            stage_count = len(cfg.post_contact_ready_curriculum_max_base_ang_vel)
            if stage_count < 2:
                raise ValueError(
                    "post-contact READY curriculum must contain at least two stages"
                )
            for name, values in schedules.items():
                if len(values) != stage_count:
                    raise ValueError(
                        f"post-contact READY curriculum {name} has {len(values)} "
                        f"values, expected {stage_count}"
                    )
            if any(
                float(lo) > float(hi)
                for lo, hi in zip(
                    cfg.post_contact_ready_curriculum_torso_x_min,
                    cfg.post_contact_ready_curriculum_torso_x_max,
                    strict=True,
                )
            ):
                raise ValueError(
                    "every post-contact READY torso-x curriculum stage must "
                    "satisfy min <= max"
                )
            if not 0 <= int(cfg.post_contact_ready_curriculum_start_level) < stage_count:
                raise ValueError(
                    "post_contact_ready_curriculum_start_level is outside the schedule"
                )
            transition_count = stage_count - 1
            if len(cfg.post_contact_ready_curriculum_advance_success_thresholds) != transition_count:
                raise ValueError(
                    "post-contact READY curriculum needs one success threshold per transition"
                )
            if len(cfg.post_contact_ready_curriculum_min_resolved_events) != transition_count:
                raise ValueError(
                    "post-contact READY curriculum needs one resolved-event count per transition"
                )
            if (
                len(cfg.post_contact_ready_curriculum_shadow_success_thresholds)
                != transition_count
            ):
                raise ValueError(
                    "post-contact READY curriculum needs one next-stage shadow "
                    "threshold per transition"
                )
            if any(
                float(value) < 0.0 or float(value) > 1.0
                for value in cfg.post_contact_ready_curriculum_advance_success_thresholds
            ):
                raise ValueError("READY curriculum success thresholds must be in [0, 1]")
            if any(
                float(value) < 0.0 or float(value) > 1.0
                for value in cfg.post_contact_ready_curriculum_shadow_success_thresholds
            ):
                raise ValueError(
                    "READY curriculum shadow success thresholds must be in [0, 1]"
                )
            if any(
                int(value) < 1
                for value in cfg.post_contact_ready_curriculum_min_resolved_events
            ):
                raise ValueError("READY curriculum minimum event counts must be positive")
            if int(cfg.post_contact_ready_curriculum_min_completed_swings) < 1:
                raise ValueError(
                    "READY curriculum minimum completed swings must be positive"
                )
            if int(cfg.post_contact_ready_curriculum_required_advance_checks) < 1:
                raise ValueError(
                    "READY curriculum required advance checks must be positive"
                )
            for name, value in (
                (
                    "minimum targeted-attempt EMA",
                    cfg.post_contact_ready_curriculum_min_targeted_attempt_ema,
                ),
                (
                    "minimum return-success EMA",
                    cfg.post_contact_ready_curriculum_min_return_success_ema,
                ),
            ):
                if float(value) < 0.0 or float(value) > 1.0:
                    raise ValueError(f"READY curriculum {name} must be in [0, 1]")
            for name, value in (
                ("recovery EMA rate", cfg.post_contact_ready_curriculum_ema_rate),
                ("hit EMA rate", cfg.post_contact_ready_curriculum_hit_ema_rate),
            ):
                if float(value) <= 0.0 or float(value) > 1.0:
                    raise ValueError(
                        f"READY curriculum {name} must be in (0, 1]"
                    )
            required_steps = tuple(
                int(value)
                for value in cfg.post_contact_ready_curriculum_required_consecutive_steps
            )
            deadlines = tuple(
                int(value) for value in cfg.post_contact_ready_curriculum_deadline_steps
            )
            if any(value < 1 for value in required_steps):
                raise ValueError("READY curriculum consecutive-step counts must be positive")
            if any(
                deadline < required
                for deadline, required in zip(deadlines, required_steps, strict=True)
            ):
                raise ValueError(
                    "each READY curriculum deadline must cover its consecutive-step count"
                )
        self._functional_ready_joint_ids: list[int] = []
        self._functional_ready_joint_positions: torch.Tensor | None = None
        self._functional_ready_joint_tolerances: torch.Tensor | None = None
        if cfg.functional_ready_joint_names:
            if len(cfg.functional_ready_joint_names) != len(cfg.functional_ready_joint_positions):
                raise ValueError(
                    "functional_ready_joint_names and functional_ready_joint_positions "
                    "must have the same length"
                )
            if cfg.functional_ready_joint_tolerances and (
                len(cfg.functional_ready_joint_names)
                != len(cfg.functional_ready_joint_tolerances)
            ):
                raise ValueError(
                    "functional_ready_joint_tolerances must be empty or have the "
                    "same length as functional_ready_joint_names"
                )
            self._functional_ready_joint_ids = self.robot.find_joints(
                list(cfg.functional_ready_joint_names), preserve_order=True
            )[0]
            if len(self._functional_ready_joint_ids) != len(cfg.functional_ready_joint_names):
                raise ValueError(
                    "failed to resolve every functional READY anchor joint: "
                    f"{cfg.functional_ready_joint_names}"
                )
            self._functional_ready_joint_positions = torch.tensor(
                cfg.functional_ready_joint_positions,
                dtype=torch.float32,
                device=self.device,
            )
            tolerances = (
                cfg.functional_ready_joint_tolerances
                if cfg.functional_ready_joint_tolerances
                else (0.0,) * len(cfg.functional_ready_joint_names)
            )
            self._functional_ready_joint_tolerances = torch.tensor(
                tolerances,
                dtype=torch.float32,
                device=self.device,
            )

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
        self.racket_impact_target_normal_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.racket_impact_target_normal_w[:, 2] = 1.0
        self.racket_target_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.racket_target_normal_w[:, 2] = 1.0
        # Strike-command backups.  Public racket_target_* tensors may be temporarily overridden
        # during deploy-style no-command holds; the true sampled strike command is restored as soon
        # as the hold ends, so the swing/reward task remains unchanged.
        self._strike_racket_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._strike_racket_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._strike_racket_impact_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._strike_racket_impact_target_normal_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._strike_racket_impact_target_normal_w[:, 2] = 1.0
        self._strike_racket_target_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._strike_racket_target_normal_w[:, 2] = 1.0
        # Hidden true ball task.  The actor sees the planner command above; rewards use these
        # fields so training can tolerate planner error instead of treating the planner as truth.
        self.ball_strike_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.incoming_ball_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.incoming_ball_bounce_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.incoming_ball_post_bounce_time_s = torch.zeros(
            self.num_envs, device=self.device
        )
        self.incoming_ball_route_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.ball_outgoing_target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_target_landing_w = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.ball_target_landing_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
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
        # A rigid-ball command source may revise the strike command after a
        # measured table bounce.  Keep that revision alive across subsequent
        # command-manager steps instead of restoring motion-clock timing.
        self.physical_command_override_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.physical_command_override_tts = torch.zeros(
            self.num_envs, device=self.device
        )
        self.steps_since_target_resample = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reward_snapshot_prepared = False
        self.no_command_ready_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.no_command_ready_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.healthy_stage_planted_rehearsal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.healthy_stage_station_hold_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.healthy_stage_station_arrived = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.healthy_stage_station_settled = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.healthy_stage_station_settle_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Optional deployment-compatible lateral relocation rehearsal.  The
        # actor contract is unchanged: station error remains the only command
        # input, while these tensors track whether the base actually arrives
        # and settles before the hidden strike command is revealed.
        self.station_relocation_rehearsal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.station_relocation_station_w = self.fixed_station_w.clone()
        self.station_relocation_arrival_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.station_relocation_settle_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.station_relocation_release_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.station_relocation_timeout_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.station_relocation_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.lifecycle_recovery_hold_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.lifecycle_recovery_release_success_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.lifecycle_recovery_release_fail_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._station_relocation_level = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._station_relocation_arrival_ema = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._station_relocation_settle_ema = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._station_relocation_contact_ema = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._station_relocation_safety_ema = torch.ones(
            (), dtype=torch.float32, device=self.device
        )
        self._station_relocation_resolved_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._station_relocation_terminal_reset_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._station_relocation_advance_streak = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._station_relocation_regress_streak = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._station_relocation_initialized = False

        # Reward helper signals.
        self.racket_target_distance = torch.zeros(self.num_envs, device=self.device)
        self.impact_face_radial_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self.impact_face_normal_gap = torch.zeros(
            self.num_envs, device=self.device
        )
        self.impact_face_quality = torch.zeros(
            self.num_envs, device=self.device
        )
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
        self.current_swing_center_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_center_net_cross = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_center_on_opponent = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_rim_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_rim_net_cross = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_rim_on_opponent = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_analytic_outer_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_analytic_outer_net_cross = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_analytic_outer_on_opponent = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.impact_health_score = torch.zeros(self.num_envs, device=self.device)
        self.impact_health_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_swing_impact_health_score = torch.zeros(self.num_envs, device=self.device)
        self.current_swing_healthy_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_healthy_net_cross = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_healthy_on_opponent = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_healthy_normal_ok = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_opportunity = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_attempt = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_swing_targeted_attempt = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.current_swing_unsafe = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.current_swing_station_saturated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.actuator_overflow_consecutive_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.prev_swing_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_swing_net_cross = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_swing_on_opponent = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_swing_targeted_attempt = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
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
        self.cycle_v2_ready_consecutive_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Ability-driven one-swing bootstrap.  The motion term selects the
        # population on true reset; this command owns post-wrap READY
        # settlement and the clean timeout latch.
        self.single_cycle_bootstrap = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.single_cycle_swing_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.single_cycle_recovery_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.single_cycle_recovery_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.single_cycle_ready_consecutive_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.single_cycle_safe_timeout_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.single_cycle_deadline_timeout_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.single_cycle_timeout_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.single_cycle_hard_failure_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._single_cycle_curriculum_level = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._single_cycle_probability = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._single_cycle_effective_deadline_steps = torch.full(
            (),
            int(self.cfg.single_cycle_deadline_steps),
            dtype=torch.long,
            device=self.device,
        )
        self._single_cycle_selected_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._single_cycle_completed_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._single_cycle_safe_timeout_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._single_cycle_deadline_timeout_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._single_cycle_hard_failure_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self.active_ready_consecutive_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.active_ready_success_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.active_ready_success_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.active_ready_survival_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.active_ready_survival_next_milestone = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.active_ready_survival_milestone_event = torch.zeros(
            self.num_envs, device=self.device
        )
        self._active_ready_survival_milestone_steps = torch.tensor(
            tuple(cfg.active_ready_survival_milestone_steps),
            dtype=torch.long,
            device=self.device,
        )
        self._active_ready_survival_milestone_values = torch.tensor(
            tuple(cfg.active_ready_survival_milestone_values),
            dtype=torch.float32,
            device=self.device,
        )
        self.no_command_ready_previous_score = torch.zeros(
            self.num_envs, device=self.device
        )
        self.no_command_ready_previous_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Contact-local recovery state. Unlike cycle-v2, this begins at the exact
        # impact and settles before the current clip is allowed to disappear.
        self.post_contact_ready_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_success_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_fail_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_consecutive_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_diagnostic_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_diagnostic_elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_durable_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_durable_consecutive_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_durable_ready_now = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_durable_success_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_durable_fail_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._post_contact_ready_durable_component_ok = torch.zeros(
            self.num_envs, 12, dtype=torch.bool, device=self.device
        )
        self._post_contact_ready_durable_component_consecutive_steps = (
            torch.zeros(
                self.num_envs, 12, dtype=torch.long, device=self.device
            )
        )
        self._post_contact_ready_durable_failure_component_counts = (
            torch.zeros(12, dtype=torch.long, device=self.device)
        )
        self.post_contact_ready_terminal_window_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_terminal_quality_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_terminal_quality_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_terminal_quality = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_terminal_settlement_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_operational_ready_now = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_operational_consecutive_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_safe_settlement_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_unsafe_settlement_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_incomplete_settlement_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_safe_net_cycle_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._post_contact_ready_terminal_settlement_counts = torch.zeros(
            4, dtype=torch.long, device=self.device
        )
        self._post_contact_ready_terminal_quality_sums = torch.zeros(
            2, device=self.device
        )
        self._post_contact_ready_safe_net_cycle_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._post_contact_ready_durable_outcome_resolution_counts = (
            torch.zeros(4, 3, dtype=torch.long, device=self.device)
        )
        self._post_contact_ready_durable_resolution_latency_steps = (
            torch.zeros(2, dtype=torch.long, device=self.device)
        )
        self._post_contact_ready_durable_resolution_counts = torch.zeros(
            2, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_outcome_tier = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_face_quality = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_planner_speed_ratio = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_planner_direction_error_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_planner_normal_error_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_planner_position_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_impact_health_score = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_peak_base_ang_vel = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_peak_ang_vel_excess_increment = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_envelope_violation_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_envelope_violation_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.post_contact_ready_envelope_first_violation_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_max_tilt = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_max_abs_pitch = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_max_com_x = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_max_com_y = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_max_waist_overflow = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_max_leg_overflow = torch.zeros(
            self.num_envs, device=self.device
        )
        self._post_contact_ready_envelope_component_latches = torch.zeros(
            self.num_envs, 6, dtype=torch.bool, device=self.device
        )
        self._post_contact_ready_envelope_attempts = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._post_contact_ready_envelope_violations = torch.zeros(
            7, dtype=torch.long, device=self.device
        )
        self._post_contact_ready_envelope_histogram_thresholds = torch.tensor(
            (
                (0.10, 0.20, 0.35, 0.50),
                (0.10, 0.20, 0.35, 0.50),
                (0.08, 0.12, 0.16, 0.20),
                (0.10, 0.14, 0.18, 0.22),
                (0.50, 1.00, 1.50, 2.00),
                (1.50, 2.00, 2.50, 3.00),
                (0.80, 1.20, 1.60, 2.00),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self._post_contact_ready_envelope_histogram_latches = torch.zeros(
            self.num_envs, 7, 4, dtype=torch.bool, device=self.device
        )
        self._post_contact_ready_envelope_histogram_counts = torch.zeros(
            7, 4, dtype=torch.long, device=self.device
        )
        self._post_contact_ready_phase_boundaries_s = torch.tensor(
            cfg.post_contact_ready_diagnostic_phase_boundaries_s,
            dtype=torch.float32,
            device=self.device,
        )
        self._post_contact_ready_phase_seen_latches = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self._post_contact_ready_phase_attempt_counts = torch.zeros(
            4, dtype=torch.long, device=self.device
        )
        self._post_contact_ready_phase_histogram_latches = torch.zeros(
            self.num_envs, 4, 7, 4, dtype=torch.bool, device=self.device
        )
        self._post_contact_ready_phase_histogram_counts = torch.zeros(
            4, 7, 4, dtype=torch.long, device=self.device
        )
        # Rows are targeted miss/contact/net/bounce. Columns are
        # attempt/resolved READY success/resolved READY failure.
        self._post_contact_ready_outcome_resolution_counts = torch.zeros(
            4, 3, dtype=torch.long, device=self.device
        )
        self._post_contact_ready_resolution_latency_steps = torch.zeros(
            2, dtype=torch.long, device=self.device
        )
        self._post_contact_ready_resolution_counts = torch.zeros(
            2, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_previous_backlean_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self.post_contact_ready_shadow_consecutive_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.post_contact_ready_shadow_success_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._post_contact_ready_curriculum_level = int(
            cfg.post_contact_ready_curriculum_start_level
        )
        self._post_contact_ready_curriculum_success_ema = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._post_contact_ready_curriculum_shadow_success_ema = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._post_contact_ready_curriculum_resolved_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._post_contact_ready_curriculum_successful_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._post_contact_ready_curriculum_targeted_attempt_ema = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._post_contact_ready_curriculum_return_success_ema = torch.zeros(
            (), dtype=torch.float32, device=self.device
        )
        self._post_contact_ready_curriculum_completed_swings = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._post_contact_ready_curriculum_hit_ema_initialized = False
        self._post_contact_ready_curriculum_advance_streak = 0
        self._post_contact_ready_curriculum_advance_guard_ok = False
        self._post_contact_ready_settlement_component_ema = torch.zeros(
            5, dtype=torch.float32, device=self.device
        )
        self._post_contact_ready_settlement_ema_initialized = False
        self.impact_ball_out_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.impact_ball_out_error = torch.zeros(self.num_envs, device=self.device)
        self._ability_curriculum_initialized = False
        self._ability_curriculum_level = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._ability_curriculum_resolved_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._ability_curriculum_total_resolved_events = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._ability_curriculum_advance_streak = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._ability_curriculum_regress_streak = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
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
        self._ability_targeted_attempt_ema = torch.tensor(
            0.0, dtype=torch.float32, device=self.device
        )
        self._ability_safety_ema = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self._ability_station_saturation_ema = torch.tensor(
            0.0, dtype=torch.float32, device=self.device
        )
        self._ability_healthy_normal_ema = torch.tensor(
            0.0, dtype=torch.float32, device=self.device
        )
        self._ability_station_arrival_ema = torch.tensor(
            0.0, dtype=torch.float32, device=self.device
        )
        self._ability_station_settle_ema = torch.tensor(
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
            motion_seed_blend(
                0.0,
                cfg.table_workspace_motion_seed_blend_start,
                cfg.table_workspace_motion_seed_end_level,
            )
        if float(cfg.targeted_attempt_radius) < float(cfg.contact_radius):
            raise ValueError(
                "targeted_attempt_radius must be at least contact_radius: "
                f"{cfg.targeted_attempt_radius} < {cfg.contact_radius}"
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
        torso_ids = self.robot.find_bodies(cfg.impact_health_torso_body_name, preserve_order=True)[0]
        if not torso_ids:
            raise ValueError(
                f"impact health torso body not found: {cfg.impact_health_torso_body_name}"
            )
        self._impact_health_torso_index = torso_ids[0]
        self._impact_health_foot_indices = [
            self.robot.find_bodies(name, preserve_order=True)[0][0]
            for name in cfg.feet_body_names
        ]
        waist_ids = self.robot.find_joints(
            cfg.impact_health_waist_pitch_joint_name, preserve_order=True
        )[0]
        if not waist_ids:
            raise ValueError(
                "impact health waist pitch joint not found: "
                f"{cfg.impact_health_waist_pitch_joint_name}"
            )
        self._impact_health_waist_pitch_index = waist_ids[0]

        for key in (
            "racket_pos_error",
            "racket_vel_error",
            "time_to_strike",
            "return_success",
            "prev_return_success",
            "prev_net_cross",
            "current_net_cross",
            "racket_pos_curriculum_scale",
            "impact_inverse_command_blend",
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
            "recovery_anchor_arm_score",
            "recovery_core_ready_score",
            "recovery_functional_ready_score",
            "waist_action_overflow_rms",
            "right_arm_action_overflow_rms",
            "leg_action_overflow_rms",
            "operational_margin_fraction",
            "waist_operational_margin_fraction",
            "right_arm_operational_margin_fraction",
            "leg_operational_margin_fraction",
            "waist_operational_excess_rms",
            "right_arm_operational_excess_rms",
            "leg_operational_excess_rms",
            "q_des_velocity_violation_fraction",
            "waist_q_des_velocity_violation_fraction",
            "right_arm_q_des_velocity_violation_fraction",
            "leg_q_des_velocity_violation_fraction",
            "q_des_acceleration_violation_fraction",
            "waist_q_des_acceleration_violation_fraction",
            "right_arm_q_des_acceleration_violation_fraction",
            "leg_q_des_acceleration_violation_fraction",
            "waist_q_des_velocity_rms",
            "right_arm_q_des_velocity_rms",
            "leg_q_des_velocity_rms",
            "waist_q_des_acceleration_rms",
            "right_arm_q_des_acceleration_rms",
            "leg_q_des_acceleration_rms",
            "action_operational_feasibility_score",
            "actuator_overflow_consecutive_steps",
            "actuator_overflow_severe",
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
            "cycle_v2_ready_consecutive_steps",
            "single_cycle_bootstrap",
            "single_cycle_swing_completed",
            "single_cycle_recovery_active",
            "single_cycle_recovery_elapsed_steps",
            "single_cycle_ready_consecutive_steps",
            "single_cycle_safe_timeout_event",
            "single_cycle_deadline_timeout_event",
            "single_cycle_timeout_latch",
            "single_cycle_hard_failure_event",
            "single_cycle_probability",
            "single_cycle_curriculum_level",
            "single_cycle_effective_deadline_steps",
            "single_cycle_selected_events",
            "single_cycle_completed_events",
            "single_cycle_safe_timeout_events",
            "single_cycle_deadline_timeout_events",
            "single_cycle_hard_failure_events",
            "active_ready_consecutive_steps",
            "active_ready_success_event",
            "active_ready_success_latch",
            "active_ready_survival_steps",
            "active_ready_survival_next_milestone",
            "active_ready_survival_milestone_event",
            "no_command_ready_active",
            "no_command_from_stand_episode",
            "no_command_from_default_stand_reset",
            "no_command_ready_upright_ok",
            "no_command_ready_motion_ok",
            "no_command_ready_score",
            "no_command_ready_progress",
            "no_command_ready_station_error",
            "no_command_ready_fixed_station_error",
            "healthy_training_stage",
            "healthy_stage_planted_rehearsal",
            "healthy_stage_station_hold_seen",
            "healthy_stage_station_arrived",
            "healthy_stage_station_settled",
            "healthy_stage_station_settle_steps",
            "station_relocation_active",
            "station_relocation_target_offset_y",
            "station_relocation_error",
            "station_relocation_arrival_event",
            "station_relocation_settle_event",
            "station_relocation_release_event",
            "station_relocation_timeout_event",
            "station_relocation_elapsed_steps",
            "station_relocation_hold_deadline_steps",
            "lifecycle_recovery_hold_active",
            "lifecycle_recovery_release_success_event",
            "lifecycle_recovery_release_fail_event",
            "station_relocation_level",
            "station_relocation_arrival_ema",
            "station_relocation_settle_ema",
            "station_relocation_contact_ema",
            "station_relocation_safety_ema",
            "station_relocation_resolved_events",
            "station_relocation_terminal_reset_events",
            "station_relocation_advance_streak",
            "station_relocation_regress_streak",
            "station_relocation_effective_max_lin_vel",
            "station_relocation_effective_max_ang_vel",
            "station_relocation_effective_min_settle_steps",
            "ability_curriculum_level",
            "ability_curriculum_scale",
            "ability_curriculum_resolved_events",
            "ability_curriculum_total_resolved_events",
            "ability_curriculum_advance_streak",
            "ability_curriculum_regress_streak",
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
            "ability_targeted_attempt_ema",
            "impact_health_floor_progress",
            "ability_safety_ema",
            "ability_station_saturation_ema",
            "ability_healthy_normal_ema",
            "ability_station_arrival_ema",
            "ability_station_settle_ema",
            "safe_outcome_capability_gate",
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
            "workspace_motion_seed_blend",
            "planner_perturb_curriculum_scale",
            "incoming_ball_horizontal_speed",
            "incoming_ball_speed",
            "incoming_ball_speed_curriculum_level",
            "incoming_ball_speed_sample_low",
            "incoming_ball_speed_sample_high",
            "station_saturated",
            "station_raw_clip_excess",
            "station_saturated_x",
            "station_saturated_y",
            "station_raw_clip_excess_x",
            "station_raw_clip_excess_y",
            "current_swing_opportunity",
            "current_swing_attempt",
            "current_swing_targeted_attempt",
            "prev_swing_targeted_attempt",
            "current_swing_unsafe",
            "current_swing_station_saturated",
            "impact_health_score",
            "impact_health_ok",
            "impact_health_backlean_violation",
            "impact_health_waist_backfold",
            "impact_health_base_retreat",
            "impact_health_torso_gravity_x",
            "impact_health_torso_ang_vel",
            "impact_health_com_x",
            "impact_health_com_y",
            "post_contact_ready_pending",
            "post_contact_ready_elapsed_steps",
            "post_contact_ready_consecutive_steps",
            "post_contact_ready_diagnostic_pending",
            "post_contact_ready_diagnostic_elapsed_steps",
            "post_contact_ready_durable_pending",
            "post_contact_ready_durable_consecutive_steps",
            "post_contact_ready_durable_ready_now",
            "post_contact_ready_durable_success_event",
            "post_contact_ready_durable_fail_event",
            "post_contact_ready_durable_resolution_success_rate",
            "post_contact_ready_durable_resolution_fail_rate",
            "post_contact_ready_durable_resolution_success_latency_steps",
            "post_contact_ready_durable_resolution_fail_latency_steps",
            "post_contact_ready_durable_fail_backlean_rate",
            "post_contact_ready_durable_fail_forward_lean_rate",
            "post_contact_ready_durable_fail_torso_ang_vel_rate",
            "post_contact_ready_durable_fail_base_lin_vel_rate",
            "post_contact_ready_durable_fail_base_ang_vel_rate",
            "post_contact_ready_durable_fail_racket_speed_rate",
            "post_contact_ready_durable_fail_height_rate",
            "post_contact_ready_durable_fail_com_x_rate",
            "post_contact_ready_durable_fail_com_y_rate",
            "post_contact_ready_durable_fail_feet_rate",
            "post_contact_ready_durable_fail_station_rate",
            "post_contact_ready_durable_fail_arm_rate",
            "post_contact_ready_terminal_window_active",
            "post_contact_ready_terminal_quality",
            "post_contact_ready_terminal_settlement_event",
            "post_contact_ready_operational_ready_now",
            "post_contact_ready_operational_consecutive_steps",
            "post_contact_ready_safe_settlement_event",
            "post_contact_ready_unsafe_settlement_event",
            "post_contact_ready_incomplete_settlement_event",
            "post_contact_ready_safe_net_cycle_event",
            "post_contact_ready_terminal_quality_mean",
            "post_contact_ready_safe_terminal_quality_mean",
            "post_contact_ready_terminal_safe_rate",
            "post_contact_ready_terminal_unsafe_rate",
            "post_contact_ready_terminal_incomplete_rate",
            "post_contact_ready_safe_net_cycle_rate",
            "post_contact_ready_score",
            "post_contact_ready_now",
            "post_contact_ready_success_event",
            "post_contact_ready_fail_event",
            "post_contact_ready_outcome_tier",
            "post_contact_ready_face_quality",
            "post_contact_ready_safe_face_net_cycle_value",
            "post_contact_ready_planner_speed_ratio",
            "post_contact_ready_planner_direction_error_rad",
            "post_contact_ready_planner_normal_error_rad",
            "post_contact_ready_planner_position_error",
            "post_contact_ready_impact_health_score",
            "post_contact_ready_peak_base_ang_vel",
            "post_contact_ready_peak_ang_vel_excess_increment",
            "post_contact_ready_envelope_violation_latch",
            "post_contact_ready_envelope_violation_event",
            "post_contact_ready_envelope_first_violation_step",
            "post_contact_ready_envelope_attempts",
            "post_contact_ready_envelope_violation_rate",
            "post_contact_ready_envelope_tilt_violation_rate",
            "post_contact_ready_envelope_pitch_violation_rate",
            "post_contact_ready_envelope_com_violation_rate",
            "post_contact_ready_envelope_waist_violation_rate",
            "post_contact_ready_envelope_leg_violation_rate",
            "post_contact_ready_envelope_base_ang_vel_violation_rate",
            "post_contact_ready_resolution_success_rate",
            "post_contact_ready_resolution_fail_rate",
            "post_contact_ready_resolution_success_latency_steps",
            "post_contact_ready_resolution_fail_latency_steps",
            "post_contact_ready_max_tilt",
            "post_contact_ready_max_abs_pitch",
            "post_contact_ready_max_com_x",
            "post_contact_ready_max_com_y",
            "post_contact_ready_max_waist_overflow",
            "post_contact_ready_max_leg_overflow",
            "post_contact_ready_settlement_velocity_score_ema",
            "post_contact_ready_settlement_position_gate_ema",
            "post_contact_ready_settlement_health_gate_ema",
            "post_contact_ready_settlement_recovery_gate_ema",
            "post_contact_ready_settlement_total_score_ema",
            "post_contact_recovery_backlean_error",
            "post_contact_recovery_directional_progress",
            "post_contact_ready_torso_ok",
            "post_contact_ready_motion_ok",
            "post_contact_ready_support_ok",
            "post_contact_ready_arm_ok",
            "post_contact_ready_base_lin_vel_ok",
            "post_contact_ready_base_ang_vel_ok",
            "post_contact_ready_racket_speed_ok",
            "post_contact_ready_height_ok",
            "post_contact_ready_com_x_ok",
            "post_contact_ready_com_y_ok",
            "post_contact_ready_feet_ok",
            "post_contact_ready_station_ok",
            "post_contact_ready_fail_torso",
            "post_contact_ready_fail_base_lin_vel",
            "post_contact_ready_fail_base_ang_vel",
            "post_contact_ready_fail_racket_speed",
            "post_contact_ready_fail_support",
            "post_contact_ready_fail_arm",
            "post_contact_ready_shadow_now",
            "post_contact_ready_shadow_consecutive_steps",
            "post_contact_ready_shadow_success_latch",
            "post_contact_ready_curriculum_level",
            "post_contact_ready_curriculum_fraction",
            "post_contact_ready_curriculum_success_ema",
            "post_contact_ready_curriculum_shadow_success_ema",
            "post_contact_ready_curriculum_resolved_events",
            "post_contact_ready_curriculum_successful_events",
            "post_contact_ready_curriculum_targeted_attempt_ema",
            "post_contact_ready_curriculum_return_success_ema",
            "post_contact_ready_curriculum_completed_swings",
            "post_contact_ready_curriculum_advance_streak",
            "post_contact_ready_curriculum_advance_guard_ok",
            "post_contact_ready_effective_max_torso_ang_vel",
            "post_contact_ready_effective_torso_x_min",
            "post_contact_ready_effective_torso_x_max",
            "post_contact_ready_effective_max_base_lin_vel",
            "post_contact_ready_effective_max_base_ang_vel",
            "post_contact_ready_effective_max_racket_speed",
            "post_contact_ready_effective_min_feet_contact",
            "post_contact_ready_effective_required_consecutive_steps",
            "post_contact_ready_effective_deadline_steps",
            "post_contact_ready_curriculum_advance_threshold",
            "post_contact_ready_curriculum_advance_min_events",
            "post_contact_ready_curriculum_shadow_threshold",
            "post_contact_ready_curriculum_min_targeted_attempt_ema",
            "post_contact_ready_curriculum_min_return_success_ema",
            "post_contact_ready_curriculum_min_completed_swings",
            "post_contact_ready_curriculum_required_advance_checks",
            "racket_normal_error_deg",
            "impact_normal_error_deg",
            "impact_racket_vel_error_mps",
            "impact_racket_speed_ratio",
            "impact_racket_vel_angle_deg",
            "impact_racket_actual_vel_x",
            "impact_racket_actual_vel_y",
            "impact_racket_actual_vel_z",
            "impact_racket_target_vel_x",
            "impact_racket_target_vel_y",
            "impact_racket_target_vel_z",
            "impact_ball_out_error_mps",
            "impact_face_radial_error_m",
            "impact_face_normal_gap_m",
            "impact_face_quality",
            "impact_center_contact",
            "impact_center_net_cross",
            "impact_center_on_opponent",
            "impact_rim_contact",
            "impact_rim_net_cross",
            "impact_rim_on_opponent",
            "impact_analytic_outer_contact",
            "impact_analytic_outer_net_cross",
            "impact_analytic_outer_on_opponent",
            "impact_planner_pos_error_m",
            "impact_planner_vel_error_mps",
            "impact_planner_speed_ratio",
            "impact_planner_vel_angle_deg",
            "impact_healthy_normal_ok",
            "impact_healthy_contact",
            "impact_healthy_net_cross",
            "impact_healthy_on_opponent",
            "current_swing_impact_health_score",
            "current_swing_healthy_contact",
            "current_swing_healthy_net_cross",
            "current_swing_healthy_on_opponent",
            "current_swing_healthy_normal_ok",
            "current_swing_contact",
            "current_swing_net_cross",
            "current_swing_on_opponent",
            "current_swing_center_contact",
            "current_swing_center_net_cross",
            "current_swing_center_on_opponent",
            "current_swing_rim_contact",
            "current_swing_rim_net_cross",
            "current_swing_rim_on_opponent",
            "current_swing_analytic_outer_contact",
            "current_swing_analytic_outer_net_cross",
            "current_swing_analytic_outer_on_opponent",
            "cycle_v2_outcome_tier",
            "cycle_v2_streak",
            "closed_loop_recovery_trigger_event",
            "closed_loop_cycle_success_event",
            "closed_loop_durable_cycle_success_event",
            "closed_loop_phase_id",
            "closed_loop_phase_ready_no_command",
            "closed_loop_phase_command_acquire",
            "closed_loop_phase_pre_strike",
            "closed_loop_phase_strike",
            "closed_loop_phase_follow_through",
            "closed_loop_phase_recovery",
            "closed_loop_phase_next_ready",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
        for phase_name in (
            "impact_0_100ms",
            "brake_100_300ms",
            "settle_300_600ms",
            "ready_after_600ms",
        ):
            for metric_name in (
                "attempts",
                "tilt_gt_0p1_rate",
                "pitch_gt_0p1_rate",
                "waist_overflow_gt_1p0_rate",
                "base_ang_vel_gt_0p8_rate",
            ):
                self.metrics[
                    f"post_contact_phase_{phase_name}_{metric_name}"
                ] = torch.zeros(self.num_envs, device=self.device)

    # --- helpers -------------------------------------------------------------------------------- #
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if not bool(self.cfg.batched_metric_reset_logging):
            return super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        extras = batched_metric_means_and_clear(self.metrics, env_ids)
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

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

    def healthy_training_stage(self) -> int:
        """Return the global ability-gated training stage in ``{1, 2, 3}``."""
        if not bool(self.cfg.healthy_three_stage_enabled):
            return 3
        level = float(torch.clamp(self._ability_curriculum_level, 0.0, 1.0).item())
        if level < float(self.cfg.healthy_stage2_start_level):
            return 1
        if level < float(self.cfg.healthy_stage3_start_level):
            return 2
        return 3

    def impact_health_floor_progress(self) -> float:
        """Return measured strike capability used to tighten health shaping."""
        return strike_health_floor_progress(
            source=str(self.cfg.impact_health_floor_progress_source),
            ability_level=float(self._ability_curriculum_level.item()),
            targeted_attempt_ema=float(self._ability_targeted_attempt_ema.item()),
            contact_ema=float(self._ability_contact_ema.item()),
            targeted_attempt_threshold=float(
                self.cfg.impact_health_floor_targeted_attempt_threshold
            ),
            contact_threshold=float(self.cfg.impact_health_floor_contact_threshold),
        )

    def healthy_workspace_level(self) -> float:
        """Expand the physical workspace only after combined training unlocks."""
        if str(self.cfg.table_workspace_level_source) == "station_relocation":
            ranges = tuple(self.cfg.station_relocation_abs_y_ranges)
            max_level = max(len(ranges) - 1, 0)
            if max_level == 0:
                return 0.0
            return min(
                max(
                    float(self._station_relocation_level.item())
                    / float(max_level),
                    0.0,
                ),
                1.0,
            )
        if not bool(self.cfg.healthy_three_stage_enabled):
            return windowed_curriculum_level(
                float(self._ability_curriculum_level.item()),
                float(self.cfg.table_workspace_curriculum_start_level),
                float(self.cfg.table_workspace_curriculum_full_level),
            )
        level = float(torch.clamp(self._ability_curriculum_level, 0.0, 1.0).item())
        start = min(max(float(self.cfg.healthy_stage3_start_level), 0.0), 0.999999)
        return min(max((level - start) / (1.0 - start), 0.0), 1.0)

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
        use_dynamic_phase = self.pre_strike | self.strike_window
        if bool(
            self.cfg.healthy_three_stage_enabled
            and self.cfg.healthy_stage_use_dynamic_station_during_ready
        ):
            use_dynamic_phase = use_dynamic_phase | self.no_command_ready_active
        else:
            use_dynamic_phase = use_dynamic_phase & (~self.no_command_ready_active)
        station = torch.where(
            use_dynamic_phase.unsqueeze(-1),
            self.dynamic_station_w,
            self.fixed_station_w,
        )
        if bool(self.cfg.station_relocation_enabled):
            recovery = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            if bool(self.cfg.lifecycle_recovery_hold_gate_enabled) and hasattr(
                self, "cycle_v2_pending"
            ):
                recovery = self.cycle_v2_pending
                station = torch.where(
                    recovery.unsqueeze(-1), self.fixed_station_w, station
                )
            relocation = (
                self.no_command_ready_active
                & self.station_relocation_rehearsal
                & (~recovery)
            ).unsqueeze(-1)
            station = torch.where(
                relocation, self.station_relocation_station_w, station
            )
        return station

    def station_relocation_active_mask(self) -> torch.Tensor:
        """Return relocation-only hold steps, excluding prior-swing recovery."""
        active = (
            self.no_command_ready_active
            & self.station_relocation_rehearsal
        )
        if bool(self.cfg.lifecycle_recovery_hold_gate_enabled):
            active &= ~self.cycle_v2_pending
        return active

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
        """Active station XY minus current base XY, world frame (2)."""
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
        self.physical_command_override_active[env_ids] = False
        self.physical_command_override_tts[env_ids] = 0.0
        self._assign_swing_side(env_ids, clip, motion)
        stage = self.healthy_training_stage()
        if bool(self.cfg.healthy_three_stage_enabled):
            if stage == 1:
                planted = torch.ones(n, dtype=torch.bool, device=self.device)
            elif stage == 2:
                planted = torch.zeros(n, dtype=torch.bool, device=self.device)
            else:
                prob = min(max(float(self.cfg.healthy_stage3_planted_rehearsal_prob), 0.0), 1.0)
                planted = torch.rand(n, device=self.device) < prob
            self.healthy_stage_planted_rehearsal[env_ids] = planted
        else:
            self.healthy_stage_planted_rehearsal[env_ids] = False

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

        target_landing_xy = None
        if self.cfg.racket_velocity_mode == "range":
            outgoing_vel = sample_uniform(vel_box[..., 0], vel_box[..., 1], (n, 3), self.device)
        elif self.cfg.racket_velocity_mode in ("ballistic_landing", "impact_inverse_landing"):
            outgoing_vel, target_landing_xy = (
                self._sample_ballistic_target_velocity(
                    env_ids, true_pos, fixed_station
                )
            )
        else:
            raise ValueError(f"Unsupported racket_velocity_mode: {self.cfg.racket_velocity_mode}")
        self.ball_target_landing_valid[env_ids] = (
            target_landing_xy is not None
        )
        if target_landing_xy is not None:
            origins = self._env.scene.env_origins[env_ids]
            self.ball_target_landing_w[env_ids, :2] = (
                target_landing_xy + origins[:, :2]
            )
            self.ball_target_landing_w[env_ids, 2] = (
                origins[:, 2] + float(self.cfg.table_surface_z)
            )
        else:
            self.ball_target_landing_w[env_ids] = 0.0
        outgoing_vel = self._apply_outgoing_target_calibration(outgoing_vel)

        if self.cfg.incoming_trajectory_mode == "direct":
            incoming_vel = self._sample_incoming_ball_velocity(env_ids, true_pos)
            self.incoming_ball_bounce_pos_w[env_ids] = 0.0
            self.incoming_ball_post_bounce_time_s[env_ids] = 0.0
            self.incoming_ball_route_valid[env_ids] = False
        elif self.cfg.incoming_trajectory_mode == "one_bounce":
            incoming_vel, bounce_pos, post_bounce_time = (
                self._sample_one_bounce_incoming_ball_velocity(
                    env_ids, true_pos, fixed_station
                )
            )
            self.incoming_ball_bounce_pos_w[env_ids] = (
                bounce_pos + self._env.scene.env_origins[env_ids]
            )
            self.incoming_ball_post_bounce_time_s[env_ids] = (
                post_bounce_time
            )
            self.incoming_ball_route_valid[env_ids] = True
        else:
            raise ValueError(f"Unsupported incoming_trajectory_mode: {self.cfg.incoming_trajectory_mode}")
        self.incoming_ball_vel_w[env_ids] = incoming_vel
        self.metrics["incoming_ball_horizontal_speed"][env_ids] = torch.norm(
            incoming_vel[:, :2], dim=-1
        )
        self.metrics["incoming_ball_speed"][env_ids] = torch.norm(
            incoming_vel, dim=-1
        )
        if self.cfg.incoming_trajectory_mode != "one_bounce":
            self.metrics["incoming_ball_speed_curriculum_level"][env_ids] = 0.0
            self.metrics["incoming_ball_speed_sample_low"][env_ids] = 0.0
            self.metrics["incoming_ball_speed_sample_high"][env_ids] = 0.0

        self.ball_outgoing_target_vel_w[env_ids] = outgoing_vel
        if self.cfg.racket_velocity_mode == "impact_inverse_landing":
            racket_vel, normal = self._solve_impact_racket_command(incoming_vel, outgoing_vel)
            if bool(self.cfg.impact_inverse_command_curriculum_enabled):
                bootstrap_vel = sample_uniform(
                    vel_box[..., 0], vel_box[..., 1], (n, 3), self.device
                )
                bootstrap_incoming = self._sample_incoming_ball_velocity(
                    env_ids, true_pos
                )
                bootstrap_normal = bootstrap_vel - bootstrap_incoming
                bootstrap_normal = bootstrap_normal / torch.norm(
                    bootstrap_normal, dim=-1, keepdim=True
                ).clamp_min(1.0e-6)
                bootstrap_vel = wire_compatible_velocity(
                    bootstrap_vel, bootstrap_normal
                )
                start = min(
                    max(
                        float(self.cfg.impact_inverse_command_start_blend),
                        0.0,
                    ),
                    1.0,
                )
                level = float(
                    windowed_curriculum_level(
                        float(self._ability_curriculum_level.item()),
                        float(
                            self.cfg.impact_inverse_command_curriculum_start_level
                        ),
                        float(
                            self.cfg.impact_inverse_command_curriculum_full_level
                        ),
                    )
                )
                exponent = max(
                    float(
                        self.cfg.impact_inverse_command_curriculum_exponent
                    ),
                    1.0e-6,
                )
                blend = start + (level**exponent) * (1.0 - start)
                racket_vel = (
                    (1.0 - blend) * bootstrap_vel + blend * racket_vel
                )
                normal = (
                    (1.0 - blend) * bootstrap_normal + blend * normal
                )
                normal = normal / torch.norm(
                    normal, dim=-1, keepdim=True
                ).clamp_min(1.0e-6)
                self.metrics["impact_inverse_command_blend"][env_ids] = (
                    blend
                )
            else:
                self.metrics["impact_inverse_command_blend"][env_ids] = 1.0
        else:
            # Legacy mode: the target velocity is interpreted directly as racket velocity.
            # The outgoing target remains the same vector for the simple analytic return model.
            racket_vel = outgoing_vel
            normal = outgoing_vel - incoming_vel
            normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + 1e-6)
        if self.cfg.planner_command_mode == "v4_wire_compatible":
            racket_vel = wire_compatible_velocity(racket_vel, normal)
        self.racket_impact_target_vel_w[env_ids] = racket_vel
        self.racket_impact_target_normal_w[env_ids] = normal
        planner_vel = self._apply_planner_velocity_perturbation(
            racket_vel, planner_vel_offset, planner_vel_scale, planner_yaw
        )
        self.racket_target_vel_w[env_ids] = planner_vel
        if self.cfg.planner_command_mode == "v4_wire_compatible":
            planner_normal = planner_vel / torch.norm(
                planner_vel, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
        else:
            planner_normal = normal
        self.racket_target_normal_w[env_ids] = planner_normal
        self._strike_racket_target_pos_w[env_ids] = self.racket_target_pos_w[env_ids]
        self._strike_racket_target_vel_w[env_ids] = self.racket_target_vel_w[env_ids]
        self._strike_racket_impact_target_vel_w[env_ids] = self.racket_impact_target_vel_w[env_ids]
        self._strike_racket_impact_target_normal_w[env_ids] = self.racket_impact_target_normal_w[env_ids]
        self._strike_racket_target_normal_w[env_ids] = self.racket_target_normal_w[env_ids]
        sampled_tts = self._motion_time_to_strike(env_ids)
        self.true_time_to_strike[env_ids] = sampled_tts
        self.time_to_strike[env_ids] = (
            sampled_tts + self._planner_tts_offset[env_ids]
        )
        self.pre_strike[env_ids] = sampled_tts > 0.0
        self.strike_window[env_ids] = (
            sampled_tts.abs() <= float(self.cfg.strike_window_s)
        )
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
        self.station_relocation_rehearsal[env_ids] = False
        self.station_relocation_station_w[env_ids] = self.fixed_station_w[env_ids]
        self.station_relocation_elapsed_steps[env_ids] = 0
        self.station_relocation_release_event[env_ids] = False
        self.station_relocation_timeout_event[env_ids] = False
        self.lifecycle_recovery_hold_active[env_ids] = False
        self.lifecycle_recovery_release_success_event[env_ids] = False
        self.lifecycle_recovery_release_fail_event[env_ids] = False
        prob = float(self.cfg.deploy_ready_hold_prob)
        if bool(self.cfg.healthy_three_stage_enabled):
            stage = self.healthy_training_stage()
            if stage == 1:
                prob = float(self.cfg.healthy_stage1_ready_hold_prob)
            elif stage == 2:
                prob = float(self.cfg.healthy_stage2_ready_hold_prob)
            else:
                prob = float(self.cfg.healthy_stage3_ready_hold_prob)
        motion = self._motion()
        hold = motion.in_hold[env_ids]
        if not bool(torch.any(hold)):
            return
        draw = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if prob > 0.0:
            draw = (
                torch.rand(len(env_ids), device=self.device)
                < min(max(prob, 0.0), 1.0)
            )
        self.no_command_ready_hold[env_ids] = deploy_ready_hold_mask(
            in_hold=hold,
            sampled_hold=draw,
            stand_episode=motion.stand_episode[env_ids],
            default_stand_reset=motion.default_stand_reset[env_ids],
            force_stand_episode=bool(
                self.cfg.deploy_ready_force_stand_episode
            ),
            force_default_stand_reset=bool(
                self.cfg.deploy_ready_force_default_stand_reset
            ),
        )
        if bool(self.cfg.lifecycle_recovery_hold_gate_enabled):
            recovery = hold & self.cycle_v2_pending[env_ids]
            self.no_command_ready_hold[env_ids] |= recovery
            self.lifecycle_recovery_hold_active[env_ids] = recovery
        if not bool(self.cfg.station_relocation_enabled):
            return

        relocation_prob = min(
            max(float(self.cfg.station_relocation_rehearsal_prob), 0.0), 1.0
        )
        if relocation_prob <= 0.0:
            return
        relocation_draw = torch.rand(len(env_ids), device=self.device) < relocation_prob
        if bool(self.cfg.station_relocation_apply_to_swing_holds):
            eligible = hold & (~motion.stand_episode[env_ids])
        else:
            eligible = self.no_command_ready_hold[env_ids]
        selected = eligible & relocation_draw
        self.station_relocation_rehearsal[env_ids] = selected
        self.no_command_ready_hold[env_ids] |= selected
        if not bool(torch.any(selected)):
            return

        if bool(self.cfg.station_relocation_use_dynamic_target):
            selected_ids = env_ids[selected]
            self.station_relocation_station_w[selected_ids] = (
                self.dynamic_station_w[selected_ids]
            )
            return

        ranges = tuple(self.cfg.station_relocation_abs_y_ranges)
        if len(ranges) == 0:
            raise ValueError(
                "station_relocation_abs_y_ranges must contain at least one range"
            )
        level = min(int(self._station_relocation_level.item()), len(ranges) - 1)
        lo, hi = (float(v) for v in ranges[level])
        lo, hi = max(min(lo, hi), 0.0), max(lo, hi)

        fixed = self.fixed_station_w[env_ids]
        desired_y = self.dynamic_station_w[env_ids, 1] - fixed[:, 1]
        random_sign = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones_like(desired_y),
            torch.ones_like(desired_y),
        )
        direction = torch.where(desired_y.abs() > 1.0e-4, desired_y.sign(), random_sign)
        magnitude = desired_y.abs().clamp(min=lo, max=hi)
        relocation_station = fixed.clone()
        relocation_station[:, 1] = fixed[:, 1] + direction * magnitude
        self.station_relocation_station_w[env_ids[selected]] = relocation_station[selected]

    def _apply_no_command_ready_targets(self) -> None:
        """Expose deploy lifecycle ready/no-command targets while the motion clock is held."""
        self.racket_target_pos_w.copy_(self._strike_racket_target_pos_w)
        self.racket_target_vel_w.copy_(self._strike_racket_target_vel_w)
        self.racket_impact_target_vel_w.copy_(self._strike_racket_impact_target_vel_w)
        self.racket_impact_target_normal_w.copy_(
            self._strike_racket_impact_target_normal_w
        )
        self.racket_target_normal_w.copy_(self._strike_racket_target_normal_w)

        motion = self._motion()
        # A lifecycle gate may restore ``hold_counter`` after MotionCommand has
        # decremented its final held step. Keep the no-command observation active
        # through that boundary so the actor never sees a one-frame strike-command
        # leak while relocation or prior-swing recovery is still unresolved.
        gated = self.lifecycle_recovery_hold_active.clone()
        gated |= self.single_cycle_recovery_active
        if bool(self.cfg.station_relocation_apply_to_swing_holds):
            gated |= self.station_relocation_rehearsal & self.no_command_ready_hold
        active = self.no_command_ready_hold & (motion.in_hold | gated)
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

    def _station_relocation_hold_deadline(self) -> int:
        schedule = tuple(
            self.cfg.station_relocation_hold_deadline_steps_by_level
        )
        if schedule:
            level = min(
                int(self._station_relocation_level.item()),
                len(schedule) - 1,
            )
            return int(schedule[level])
        return max(int(self.cfg.station_relocation_settle_min_steps), 1)

    def _update_lifecycle_hold_gate(self) -> None:
        """Release recovery/relocation holds only after success or a bounded timeout."""
        self.station_relocation_release_event.zero_()
        self.station_relocation_timeout_event.zero_()
        self.lifecycle_recovery_release_success_event.zero_()
        self.lifecycle_recovery_release_fail_event.zero_()
        motion = self._motion()

        # A bootstrap episode remains in deploy-style no-command READY until
        # its settlement latch is consumed by the timeout termination on the
        # following environment step.
        single_cycle_hold = self.single_cycle_recovery_active
        motion.hold_counter[single_cycle_hold] = torch.clamp(
            motion.hold_counter[single_cycle_hold], min=1
        )
        self.no_command_ready_hold[single_cycle_hold] = True

        if bool(self.cfg.lifecycle_recovery_hold_gate_enabled):
            # Do not require ``motion.in_hold`` here. MotionCommand decrements
            # its counter before this command term runs; a counter of one is
            # therefore observed as zero even though the reference clock did not
            # advance on that step.
            recovery_active = self.lifecycle_recovery_hold_active
            recovery_resolved = recovery_active & (~self.cycle_v2_pending)
            recovery_success = (
                recovery_resolved & self.cycle_v2_ready_success_latch
            )
            recovery_fail = recovery_resolved & (~recovery_success)
            self.lifecycle_recovery_release_success_event |= recovery_success
            self.lifecycle_recovery_release_fail_event |= recovery_fail
            self.lifecycle_recovery_hold_active[recovery_resolved] = False

            keep_recovery = recovery_active & self.cycle_v2_pending
            motion.hold_counter[keep_recovery] = torch.clamp(
                motion.hold_counter[keep_recovery], min=1
            )

            resolved_without_relocation = recovery_resolved & (
                ~self.station_relocation_rehearsal
            )
            motion.hold_counter[resolved_without_relocation] = 0
            resolved_with_relocation = recovery_resolved & (
                self.station_relocation_rehearsal
            )
            motion.hold_counter[resolved_with_relocation] = torch.clamp(
                motion.hold_counter[resolved_with_relocation], min=1
            )

        if not (
            bool(self.cfg.station_relocation_enabled)
            and bool(self.cfg.station_relocation_apply_to_swing_holds)
        ):
            return

        relocation_active = (
            self.station_relocation_rehearsal
            & self.no_command_ready_hold
            & (~self.lifecycle_recovery_hold_active)
        )
        self.station_relocation_elapsed_steps[relocation_active] += 1
        keep, released, timeout = lifecycle_hold_gate(
            active=relocation_active,
            complete=self.healthy_stage_station_settled,
            elapsed_steps=self.station_relocation_elapsed_steps,
            deadline_steps=self._station_relocation_hold_deadline(),
        )
        motion.hold_counter[keep] = torch.clamp(
            motion.hold_counter[keep], min=1
        )
        release = released | timeout
        self.station_relocation_release_event |= released
        self.station_relocation_timeout_event |= timeout
        self.no_command_ready_hold[release] = False
        motion.hold_counter[release] = 0

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
            self.metrics["station_saturated_x"][env_ids] = 0.0
            self.metrics["station_saturated_y"][env_ids] = 0.0
            self.metrics["station_raw_clip_excess_x"][env_ids] = 0.0
            self.metrics["station_raw_clip_excess_y"][env_ids] = 0.0
            return
        if self.cfg.station_mode != "dynamic_from_motion":
            raise ValueError(f"Unsupported station_mode: {self.cfg.station_mode}")
        route_station = (
            str(self.cfg.table_workspace_level_source)
            == "station_relocation"
        )
        if route_station:
            target_rel_y = target_pos_w[:, 1] - fixed_station_w[:, 1]
            fh_center = 0.5 * sum(
                float(value)
                for value in self.cfg.table_workspace_forehand_core_y_range
            )
            bh_center = 0.5 * sum(
                float(value)
                for value in self.cfg.table_workspace_backhand_core_y_range
            )
            local_anchor = torch.where(
                self.swing_sign[env_ids] >= 0.0,
                torch.full_like(target_rel_y, fh_center),
                torch.full_like(target_rel_y, bh_center),
            )
            rel = torch.zeros_like(fixed_station_w)
            rel[:, 1] = target_rel_y - local_anchor
        elif bool(self.cfg.healthy_three_stage_enabled):
            stage = self.healthy_training_stage()
            if stage == 1:
                self.dynamic_station_w[env_ids] = fixed_station_w
                self.current_swing_station_saturated[env_ids] = False
                self.metrics["station_saturated"][env_ids] = 0.0
                self.metrics["station_raw_clip_excess"][env_ids] = 0.0
                self.metrics["station_saturated_x"][env_ids] = 0.0
                self.metrics["station_saturated_y"][env_ids] = 0.0
                self.metrics["station_raw_clip_excess_x"][env_ids] = 0.0
                self.metrics["station_raw_clip_excess_y"][env_ids] = 0.0
                return

            # Keep station x fixed. Lateral offsets are selected from the
            # locked-waist reachability audit, not copied from motion root drift.
            rel = torch.zeros_like(fixed_station_w)
            bias = float(self.cfg.healthy_stage_lateral_station_bias)
            rel[:, 1] = -self.swing_sign[env_ids].sign() * bias
            if stage == 3:
                target_rel_y = target_pos_w[:, 1] - fixed_station_w[:, 1]
                fh_center = 0.5 * sum(
                    float(v) for v in self.cfg.table_workspace_forehand_core_y_range
                )
                bh_center = 0.5 * sum(
                    float(v) for v in self.cfg.table_workspace_backhand_core_y_range
                )
                core_center = torch.where(
                    self.swing_sign[env_ids] >= 0.0,
                    torch.full_like(target_rel_y, fh_center),
                    torch.full_like(target_rel_y, bh_center),
                )
                rel[:, 1] += float(self.cfg.healthy_stage3_station_target_gain) * (
                    target_rel_y - core_center
                )
            rel[self.healthy_stage_planted_rehearsal[env_ids]] = 0.0
        else:
            offset_xy = self._ensure_motion_racket_offsets()[clip]
            desired = target_pos_w[:, :2] - offset_xy
            rel = desired - fixed_station_w
        clip_box = torch.tensor(self.cfg.dynamic_station_xy_clip, dtype=torch.float32, device=self.device)
        if route_station:
            ranges = tuple(self.cfg.station_relocation_abs_y_ranges)
            level = min(
                int(self._station_relocation_level.item()),
                len(ranges) - 1,
            )
            max_y = max(abs(float(value)) for value in ranges[level])
            clip_box[0] = 0.0
            clip_box[1, 0] = -max_y
            clip_box[1, 1] = max_y
        elif bool(self.cfg.healthy_three_stage_enabled):
            clip_box[0] = 0.0
            if self.healthy_training_stage() == 2:
                stage2_y = abs(float(self.cfg.healthy_stage2_station_y_clip))
                clip_box[1, 0] = -stage2_y
                clip_box[1, 1] = stage2_y
        clipped_rel = torch.clamp(rel, clip_box[:, 0], clip_box[:, 1])
        clip_excess_axis = (rel - clipped_rel).abs()
        clip_excess = torch.norm(clip_excess_axis, dim=-1)
        saturated = clip_excess > 1.0e-5
        self.current_swing_station_saturated[env_ids] = saturated
        self.metrics["station_saturated"][env_ids] = saturated.float()
        self.metrics["station_raw_clip_excess"][env_ids] = clip_excess
        self.metrics["station_saturated_x"][env_ids] = (
            clip_excess_axis[:, 0] > 1.0e-5
        ).float()
        self.metrics["station_saturated_y"][env_ids] = (
            clip_excess_axis[:, 1] > 1.0e-5
        ).float()
        self.metrics["station_raw_clip_excess_x"][env_ids] = clip_excess_axis[:, 0]
        self.metrics["station_raw_clip_excess_y"][env_ids] = clip_excess_axis[:, 1]
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
            level = self.healthy_workspace_level()
        else:
            level = 1.0
        if float(self.cfg.table_workspace_fixed_level) >= 0.0:
            level = float(self.cfg.table_workspace_fixed_level)

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
        if bool(self.cfg.healthy_three_stage_enabled) or str(
            self.cfg.table_workspace_level_source
        ) == "station_relocation":
            fringe_prob *= level
        fringe = torch.rand(n, device=self.device) < fringe_prob
        planted = self.healthy_stage_planted_rehearsal[env_ids]
        if bool(torch.any(planted)):
            forehand_core = self.cfg.table_workspace_forehand_core_y_range
            backhand_core = self.cfg.table_workspace_backhand_core_y_range
            planted_y_lo = torch.where(
                forehand,
                torch.full((n,), float(forehand_core[0]), device=self.device),
                torch.full((n,), float(backhand_core[0]), device=self.device),
            )
            planted_y_hi = torch.where(
                forehand,
                torch.full((n,), float(forehand_core[1]), device=self.device),
                torch.full((n,), float(backhand_core[1]), device=self.device),
            )
            y_lo = torch.where(planted, planted_y_lo, y_lo)
            y_hi = torch.where(planted, planted_y_hi, y_hi)
            fringe &= ~planted
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
        seed_blend = motion_seed_blend(
            level,
            float(self.cfg.table_workspace_motion_seed_blend_start),
            float(self.cfg.table_workspace_motion_seed_end_level),
        )
        motion_sample = None
        if seed_blend > 0.0:
            motion_sample = sample_uniform(
                motion_box[..., 0], motion_box[..., 1], (n, 3), self.device
            )
            motion_y = torch.minimum(torch.maximum(motion_sample[:, 1], full_y_lo), full_y_hi)
            motion_z_above = (
                motion_sample[:, 2] - float(self.cfg.table_surface_z)
            ).clamp(
                min=float(self.cfg.table_workspace_z_full_above_surface_range[0]),
                max=float(self.cfg.table_workspace_z_full_above_surface_range[1]),
            )
            y_station = (1.0 - seed_blend) * y_station + seed_blend * motion_y
            z_above_table = (
                (1.0 - seed_blend) * z_above_table + seed_blend * motion_z_above
            )
        fixed_station_local_x = fixed_station_w[:, 0] - origins[:, 0]
        table_x_station = (
            float(self.cfg.table_near_x)
            + float(self.cfg.planner_hit_plane_x)
            + x_jitter
            - fixed_station_local_x
        )
        if motion_sample is not None:
            table_x_station = (
                (1.0 - seed_blend) * table_x_station
                + seed_blend * motion_sample[:, 0]
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
        self.metrics["workspace_motion_seed_blend"][env_ids] = seed_blend
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
        if str(self.cfg.planner_perturb_curriculum_source) == "fixed":
            perturb_scale = float(self.cfg.planner_perturb_fixed_scale)
        else:
            start_scale = float(
                self.cfg.ability_curriculum_start_planner_perturb_scale
            )
            perturb_progress = windowed_curriculum_level(
                float(self._ability_curriculum_level.item()),
                float(self.cfg.planner_perturb_curriculum_start_level),
                float(self.cfg.planner_perturb_curriculum_full_level),
            )
            perturb_scale = start_scale + perturb_progress * (1.0 - start_scale)
        self.metrics["planner_perturb_curriculum_scale"].fill_(perturb_scale)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a velocity and retain its hidden env-local landing target."""
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
        target_landing_xy = torch.zeros(
            (n, 2), dtype=torch.float32, device=self.device
        )
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
            target_landing_xy[take] = land_xy[take]
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
            target_landing_xy[~resolved] = land_xy[~resolved]
        return out, target_landing_xy

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if bool(self.cfg.one_bounce_speed_curriculum_enabled):
            speed_level = windowed_curriculum_level(
                float(self._ability_curriculum_level.item()),
                float(self.cfg.one_bounce_speed_curriculum_start_level),
                float(self.cfg.one_bounce_speed_curriculum_full_level),
            )
            speed_lo, speed_hi = interpolate_curriculum_range(
                self.cfg.one_bounce_easy_horizontal_speed_range,
                self.cfg.one_bounce_full_horizontal_speed_range,
                speed_level,
            )
            post_lo, post_hi = interpolate_curriculum_range(
                self.cfg.one_bounce_easy_post_time_range,
                self.cfg.one_bounce_post_time_range,
                speed_level,
            )
            post_t = sample_uniform(
                post_lo, post_hi, (n,), self.device
            ).clamp_min(1.0e-3)
            sampled_speed = sample_uniform(
                speed_lo, speed_hi, (n,), self.device
            )
            desired_dx = (sampled_speed * post_t).clamp(
                min=float(self.cfg.one_bounce_min_post_bounce_dx),
                max=float(self.cfg.one_bounce_max_post_bounce_dx),
            )
            bounce_table_x = target_table_x + desired_dx
            bounce_table_x = bounce_table_x.clamp(
                min=0.18,
                max=float(self.cfg.net_x) - margin,
            )
            bounce_x = near_x + bounce_table_x

            # Lateral route width is controlled by relocation/workspace ability,
            # not by incoming speed ability. This keeps the two curricula
            # independently diagnosable.
            route_level = self.healthy_workspace_level()
            start_scale = min(
                max(float(self.cfg.ability_curriculum_start_ball_scale), 0.0),
                1.0,
            )
            ball_scale = start_scale + route_level * (1.0 - start_scale)
            self.metrics["incoming_ball_speed_curriculum_level"][env_ids] = (
                speed_level
            )
            self.metrics["incoming_ball_speed_sample_low"][env_ids] = speed_lo
            self.metrics["incoming_ball_speed_sample_high"][env_ids] = speed_hi
        else:
            ball_scale = self._ability_blend(
                float(self.cfg.ability_curriculum_start_ball_scale)
            )
            dx_lo, dx_hi = self._scaled_float_range(
                (
                    self.cfg.one_bounce_min_post_bounce_dx,
                    self.cfg.one_bounce_max_post_bounce_dx,
                ),
                ball_scale,
            )
            lo_x = torch.maximum(
                torch.full(
                    (n,), 0.18, dtype=torch.float32, device=self.device
                ),
                target_table_x + float(dx_lo),
            )
            hi_x = torch.minimum(
                torch.full(
                    (n,),
                    float(self.cfg.net_x) - margin,
                    dtype=torch.float32,
                    device=self.device,
                ),
                target_table_x + float(dx_hi),
            )
            fallback_lo = torch.maximum(
                torch.full(
                    (n,), 0.12, dtype=torch.float32, device=self.device
                ),
                torch.minimum(
                    torch.full(
                        (n,),
                        float(self.cfg.net_x) - margin,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    target_table_x + 0.20,
                ),
            )
            bad = hi_x <= lo_x
            lo_x = torch.where(bad, fallback_lo, lo_x)
            hi_x = torch.where(
                bad,
                torch.minimum(
                    torch.full_like(lo_x, float(self.cfg.net_x) - margin),
                    lo_x + 0.30,
                ),
                hi_x,
            )
            bounce_x = near_x + sample_uniform(
                lo_x, hi_x, (n,), self.device
            )
            post_lo, post_hi = self._scaled_float_range(
                self.cfg.one_bounce_post_time_range, ball_scale
            )
            post_t = sample_uniform(
                post_lo, post_hi, (n,), self.device
            ).clamp_min(1.0e-3)
            self.metrics["incoming_ball_speed_curriculum_level"][env_ids] = 0.0
            self.metrics["incoming_ball_speed_sample_low"][env_ids] = 0.0
            self.metrics["incoming_ball_speed_sample_high"][env_ids] = 0.0

        jitter_lo, jitter_hi = self._scaled_float_range(
            self.cfg.one_bounce_lateral_jitter_range, ball_scale
        )
        bounce_y = p_target[:, 1] + sample_uniform(jitter_lo, jitter_hi, (n,), self.device)
        bounce_y = torch.maximum(torch.minimum(bounce_y, center_y + half_w - margin), center_y - half_w + margin)
        bounce_z = torch.full((n,), float(self.cfg.table_surface_z) + float(self.cfg.ball_radius), device=self.device)
        bounce = torch.stack((bounce_x, bounce_y, bounce_z), dim=-1)

        accel = torch.tensor([0.0, 0.0, -_GRAVITY], dtype=torch.float32, device=self.device)
        post_vel = (p_target - bounce) / post_t.unsqueeze(-1) - 0.5 * accel * post_t.unsqueeze(-1)
        incoming_vel = post_vel + accel * post_t.unsqueeze(-1)
        return incoming_vel, bounce, post_t

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

    def refresh_from_physical_prediction(
        self,
        env_ids: torch.Tensor,
        strike_pos_w: torch.Tensor,
        incoming_vel_w: torch.Tensor,
        true_time_to_strike: torch.Tensor,
        *,
        solution_blend: float = 1.0,
        update_dynamic_station: bool = True,
    ) -> None:
        """Revise an active strike command from a measured rigid-ball state.

        The rigid ball remains hidden from the actor.  This method updates the
        hidden physical target first, then reapplies the already sampled
        planner residual to the actor-visible command.  Consequently planner
        perturbation remains independent from the physical route and outcome
        rewards cannot be satisfied by rewriting the visible command alone.
        """
        env_ids = torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        )
        if len(env_ids) == 0:
            return
        expected_vector_shape = (len(env_ids), 3)
        if strike_pos_w.shape != expected_vector_shape:
            raise ValueError(
                f"strike_pos_w must have shape {expected_vector_shape}"
            )
        if incoming_vel_w.shape != expected_vector_shape:
            raise ValueError(
                f"incoming_vel_w must have shape {expected_vector_shape}"
            )
        if true_time_to_strike.shape != (len(env_ids),):
            raise ValueError(
                f"true_time_to_strike must have shape {(len(env_ids),)}"
            )
        correction_scale = min(max(float(solution_blend), 0.0), 1.0)

        old_true_pos = self.ball_strike_pos_w[env_ids].clone()
        planner_pos_residual = (
            self._strike_racket_target_pos_w[env_ids] - old_true_pos
        )
        old_incoming = self.incoming_ball_vel_w[env_ids].clone()
        outgoing = self.ball_outgoing_target_vel_w[env_ids]
        old_solution_vel, old_solution_normal = (
            self._solve_impact_racket_command(old_incoming, outgoing)
        )
        new_solution_vel, new_solution_normal = (
            self._solve_impact_racket_command(incoming_vel_w, outgoing)
        )
        if self.cfg.planner_command_mode == "v4_wire_compatible":
            old_solution_vel = wire_compatible_velocity(
                old_solution_vel, old_solution_normal
            )
            new_solution_vel = wire_compatible_velocity(
                new_solution_vel, new_solution_normal
            )

        impact_blend = self.metrics["impact_inverse_command_blend"][
            env_ids
        ].clamp(0.0, 1.0)
        correction_weight = correction_scale * impact_blend
        old_impact_vel = self._strike_racket_impact_target_vel_w[
            env_ids
        ].clone()
        old_impact_normal = self._strike_racket_impact_target_normal_w[
            env_ids
        ].clone()
        planner_vel_residual = (
            self._strike_racket_target_vel_w[env_ids] - old_impact_vel
        )
        revised_impact_vel = old_impact_vel + correction_weight.unsqueeze(
            -1
        ) * (new_solution_vel - old_solution_vel)
        revised_impact_normal = old_impact_normal + correction_weight.unsqueeze(
            -1
        ) * (new_solution_normal - old_solution_normal)
        revised_impact_normal = revised_impact_normal / torch.linalg.norm(
            revised_impact_normal, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        if self.cfg.planner_command_mode == "v4_wire_compatible":
            revised_impact_vel = wire_compatible_velocity(
                revised_impact_vel, revised_impact_normal
            )

        planner_pos = strike_pos_w + planner_pos_residual
        planner_vel = revised_impact_vel + planner_vel_residual
        if self.cfg.planner_command_mode == "v4_wire_compatible":
            planner_normal = planner_vel / torch.linalg.norm(
                planner_vel, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
        else:
            planner_normal_residual = (
                self._strike_racket_target_normal_w[env_ids]
                - old_impact_normal
            )
            planner_normal = revised_impact_normal + planner_normal_residual
            planner_normal = planner_normal / torch.linalg.norm(
                planner_normal, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)

        self.ball_strike_pos_w[env_ids] = strike_pos_w
        self.incoming_ball_vel_w[env_ids] = incoming_vel_w
        self.racket_target_pos_w[env_ids] = planner_pos
        self.racket_impact_target_vel_w[env_ids] = revised_impact_vel
        self.racket_impact_target_normal_w[env_ids] = revised_impact_normal
        self.racket_target_vel_w[env_ids] = planner_vel
        self.racket_target_normal_w[env_ids] = planner_normal
        self._strike_racket_target_pos_w[env_ids] = planner_pos
        self._strike_racket_impact_target_vel_w[env_ids] = revised_impact_vel
        self._strike_racket_impact_target_normal_w[env_ids] = (
            revised_impact_normal
        )
        self._strike_racket_target_vel_w[env_ids] = planner_vel
        self._strike_racket_target_normal_w[env_ids] = planner_normal

        self.physical_command_override_active[env_ids] = True
        self.physical_command_override_tts[env_ids] = true_time_to_strike
        self.true_time_to_strike[env_ids] = true_time_to_strike
        self.time_to_strike[env_ids] = (
            true_time_to_strike + self._planner_tts_offset[env_ids]
        )
        self.pre_strike[env_ids] = true_time_to_strike > 0.0
        self.strike_window[env_ids] = (
            true_time_to_strike.abs() <= float(self.cfg.strike_window_s)
        )
        self.metrics["incoming_ball_horizontal_speed"][env_ids] = (
            torch.linalg.norm(incoming_vel_w[:, :2], dim=-1)
        )
        self.metrics["incoming_ball_speed"][env_ids] = torch.linalg.norm(
            incoming_vel_w, dim=-1
        )
        self.metrics["planner_target_table_x"][env_ids] = (
            planner_pos[:, 0]
            - self._env.scene.env_origins[env_ids, 0]
            - float(self.cfg.table_near_x)
        )
        self.metrics["true_strike_table_x"][env_ids] = (
            strike_pos_w[:, 0]
            - self._env.scene.env_origins[env_ids, 0]
            - float(self.cfg.table_near_x)
        )

        if update_dynamic_station:
            motion = self._motion()
            clip = (
                motion.clip_id[env_ids]
                if motion._multiseg
                else torch.zeros(
                    len(env_ids), dtype=torch.long, device=self.device
                )
            )
            self._update_dynamic_station(
                env_ids, clip, planner_pos, self.fixed_station_w[env_ids]
            )

    def _resample_command(self, env_ids: Sequence[int], carry_previous: bool = False):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        previous_single_cycle = self.single_cycle_bootstrap[env_ids].clone()
        intentional_single_cycle_timeout = (
            previous_single_cycle
            & self.single_cycle_timeout_latch[env_ids]
        )
        if carry_previous:
            self._update_station_relocation_curriculum(env_ids)
            self._update_ability_curriculum(env_ids)
            self.prev_swing_contact[env_ids] = self.current_swing_contact[env_ids]
            self.prev_swing_net_cross[env_ids] = self.current_swing_net_cross[env_ids]
            self.prev_swing_on_opponent[env_ids] = self.current_swing_on_opponent[env_ids]
            self.prev_swing_targeted_attempt[env_ids] = (
                self.current_swing_targeted_attempt[env_ids]
            )
            self._begin_cycle_v2_check(env_ids)
            unresolved_ready = torch.zeros(
                len(env_ids), dtype=torch.bool, device=self.device
            )
            if bool(self.cfg.post_contact_ready_enabled):
                unresolved_ready = self.post_contact_ready_pending[env_ids].clone()
            if bool(torch.any(unresolved_ready)):
                unresolved_global = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
                unresolved_global[env_ids] = unresolved_ready
                self._record_post_contact_ready_resolution(
                    torch.zeros_like(unresolved_global),
                    unresolved_global,
                )
            unresolved_durable = self.post_contact_ready_durable_pending[
                env_ids
            ].clone()
            if bool(torch.any(unresolved_durable)):
                unresolved_durable_global = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
                unresolved_durable_global[env_ids] = unresolved_durable
                self._record_post_contact_durable_resolution(
                    torch.zeros_like(unresolved_durable_global),
                    unresolved_durable_global,
                )
            settled_ready = self.post_contact_ready_success_event[env_ids].clone()
            settled_tier = self.post_contact_ready_outcome_tier[env_ids].clone()
            settled_impact_values = {
                name: getattr(self, name)[env_ids].clone()
                for name in (
                    "post_contact_ready_planner_speed_ratio",
                    "post_contact_ready_planner_direction_error_rad",
                    "post_contact_ready_planner_normal_error_rad",
                    "post_contact_ready_planner_position_error",
                    "post_contact_ready_impact_health_score",
                    "post_contact_ready_face_quality",
                    "post_contact_ready_peak_base_ang_vel",
                )
            }
            self._reset_post_contact_ready(env_ids)
            self.post_contact_ready_outcome_tier[env_ids] = torch.where(
                settled_ready,
                settled_tier,
                self.post_contact_ready_outcome_tier[env_ids],
            )
            for name, settled_value in settled_impact_values.items():
                target = getattr(self, name)
                target[env_ids] = torch.where(
                    settled_ready,
                    settled_value,
                    target[env_ids],
                )
            self.post_contact_ready_fail_event[env_ids] |= unresolved_ready
            curriculum_unresolved_ready = unresolved_ready
            if bool(self.cfg.single_cycle_curriculum_enabled) and bool(
                self.cfg.single_cycle_continuous_only_curriculum
            ):
                curriculum_unresolved_ready = (
                    curriculum_unresolved_ready & (~previous_single_cycle)
                )
            self._update_post_contact_ready_curriculum(
                torch.zeros_like(curriculum_unresolved_ready),
                curriculum_unresolved_ready,
            )
        else:
            self._update_station_relocation_curriculum(
                env_ids, terminal_reset=True
            )
            self._record_reset_safety_events(env_ids)
            single_cycle_hard_failure = (
                previous_single_cycle & (~intentional_single_cycle_timeout)
            )
            self._single_cycle_hard_failure_events += (
                single_cycle_hard_failure.long().sum()
            )
            reset_pending = self.post_contact_ready_pending[env_ids].clone()
            if bool(torch.any(reset_pending)):
                reset_pending_global = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
                reset_pending_global[env_ids] = reset_pending
                self._record_post_contact_ready_resolution(
                    torch.zeros_like(reset_pending_global),
                    reset_pending_global,
                )
            reset_durable_pending = self.post_contact_ready_durable_pending[
                env_ids
            ].clone()
            if bool(torch.any(reset_durable_pending)):
                reset_durable_global = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
                reset_durable_global[env_ids] = reset_durable_pending
                self._record_post_contact_durable_resolution(
                    torch.zeros_like(reset_durable_global),
                    reset_durable_global,
                )
            curriculum_reset_pending = reset_pending
            if bool(self.cfg.single_cycle_curriculum_enabled) and bool(
                self.cfg.single_cycle_continuous_only_curriculum
            ):
                curriculum_reset_pending = (
                    curriculum_reset_pending & (~previous_single_cycle)
                )
            self._update_post_contact_ready_curriculum(
                torch.zeros_like(curriculum_reset_pending),
                curriculum_reset_pending,
            )
            self.prev_swing_contact[env_ids] = False
            self.prev_swing_net_cross[env_ids] = False
            self.prev_swing_on_opponent[env_ids] = False
            self.prev_swing_targeted_attempt[env_ids] = False
            self._reset_cycle_v2(env_ids)
            self.cycle_v2_streak[env_ids] = 0
            self.active_ready_consecutive_steps[env_ids] = 0
            self.active_ready_success_latch[env_ids] = False
            self.active_ready_success_event[env_ids] = False
            self.active_ready_survival_steps[env_ids] = 0
            self.active_ready_survival_next_milestone[env_ids] = 0
            self.active_ready_survival_milestone_event[env_ids] = 0.0
            self._reset_post_contact_ready(env_ids)
            self.post_contact_ready_success_event[env_ids] = False
            self.post_contact_ready_fail_event[env_ids] = False
            self.actuator_overflow_consecutive_steps[env_ids] = 0
            self.single_cycle_bootstrap[env_ids] = False
            self.single_cycle_swing_completed[env_ids] = False
            self.single_cycle_recovery_active[env_ids] = False
            self.single_cycle_recovery_elapsed_steps[env_ids] = 0
            self.single_cycle_ready_consecutive_steps[env_ids] = 0
            self.single_cycle_safe_timeout_event[env_ids] = False
            self.single_cycle_deadline_timeout_event[env_ids] = False
            self.single_cycle_timeout_latch[env_ids] = False
            self.single_cycle_hard_failure_event[env_ids] = (
                single_cycle_hard_failure
            )
        # A target/lifecycle resample starts a new potential segment. Do not
        # pay for a score discontinuity caused by changing the command itself.
        self.no_command_ready_previous_score[env_ids] = 0.0
        self.no_command_ready_previous_active[env_ids] = False
        self.current_swing_contact[env_ids] = False
        self.current_swing_net_cross[env_ids] = False
        self.current_swing_on_opponent[env_ids] = False
        self.current_swing_center_contact[env_ids] = False
        self.current_swing_center_net_cross[env_ids] = False
        self.current_swing_center_on_opponent[env_ids] = False
        self.current_swing_rim_contact[env_ids] = False
        self.current_swing_rim_net_cross[env_ids] = False
        self.current_swing_rim_on_opponent[env_ids] = False
        self.current_swing_analytic_outer_contact[env_ids] = False
        self.current_swing_analytic_outer_net_cross[env_ids] = False
        self.current_swing_analytic_outer_on_opponent[env_ids] = False
        self.current_swing_impact_health_score[env_ids] = 0.0
        self.current_swing_healthy_contact[env_ids] = False
        self.current_swing_healthy_net_cross[env_ids] = False
        self.current_swing_healthy_on_opponent[env_ids] = False
        self.current_swing_healthy_normal_ok[env_ids] = False
        self.current_swing_opportunity[env_ids] = False
        self.current_swing_attempt[env_ids] = False
        self.current_swing_targeted_attempt[env_ids] = False
        self.current_swing_unsafe[env_ids] = False
        self.current_swing_station_saturated[env_ids] = False
        self.healthy_stage_station_hold_seen[env_ids] = False
        self.healthy_stage_station_arrived[env_ids] = False
        self.healthy_stage_station_settled[env_ids] = False
        self.healthy_stage_station_settle_steps[env_ids] = 0
        self.station_relocation_arrival_event[env_ids] = False
        self.station_relocation_settle_event[env_ids] = False
        self.metrics["impact_normal_error_deg"][env_ids] = 0.0
        self.steps_since_target_resample[env_ids] = 0
        self.target_just_resampled[env_ids] = True
        self._sample_targets(env_ids)
        if carry_previous:
            first_single_cycle_wrap = (
                self.single_cycle_bootstrap[env_ids]
                & (~self.single_cycle_swing_completed[env_ids])
            )
            first_single_cycle_ids = env_ids[first_single_cycle_wrap]
            if len(first_single_cycle_ids) > 0:
                self.single_cycle_swing_completed[first_single_cycle_ids] = True
                self.single_cycle_recovery_active[first_single_cycle_ids] = True
                self.single_cycle_recovery_elapsed_steps[first_single_cycle_ids] = 0
                self.single_cycle_ready_consecutive_steps[first_single_cycle_ids] = 0
                self.no_command_ready_hold[first_single_cycle_ids] = True
                self.station_relocation_rehearsal[first_single_cycle_ids] = False
                self.station_relocation_station_w[first_single_cycle_ids] = (
                    self.fixed_station_w[first_single_cycle_ids]
                )
                motion = self._motion()
                motion.hold_counter[first_single_cycle_ids] = torch.clamp(
                    motion.hold_counter[first_single_cycle_ids], min=1
                )
                self._apply_no_command_ready_targets()
                self._single_cycle_completed_events += int(
                    len(first_single_cycle_ids)
                )
        else:
            new_single_cycle = self._motion().single_cycle_reset[env_ids]
            self.single_cycle_bootstrap[env_ids] = new_single_cycle
            self._single_cycle_selected_events += new_single_cycle.long().sum()

    def _reset_post_contact_ready(self, env_ids: torch.Tensor) -> None:
        self.post_contact_ready_pending[env_ids] = False
        self.post_contact_ready_elapsed_steps[env_ids] = 0
        self.post_contact_ready_consecutive_steps[env_ids] = 0
        self.post_contact_ready_diagnostic_pending[env_ids] = False
        self.post_contact_ready_diagnostic_elapsed_steps[env_ids] = 0
        self.post_contact_ready_durable_pending[env_ids] = False
        self.post_contact_ready_durable_consecutive_steps[env_ids] = 0
        self.post_contact_ready_durable_ready_now[env_ids] = False
        self.post_contact_ready_durable_success_event[env_ids] = False
        self.post_contact_ready_durable_fail_event[env_ids] = False
        self._post_contact_ready_durable_component_ok[env_ids] = False
        self._post_contact_ready_durable_component_consecutive_steps[
            env_ids
        ] = 0
        self.post_contact_ready_terminal_window_active[env_ids] = False
        self.post_contact_ready_terminal_quality_sum[env_ids] = 0.0
        self.post_contact_ready_terminal_quality_count[env_ids] = 0
        self.post_contact_ready_terminal_quality[env_ids] = 0.0
        self.post_contact_ready_terminal_settlement_event[env_ids] = False
        self.post_contact_ready_operational_ready_now[env_ids] = False
        self.post_contact_ready_operational_consecutive_steps[env_ids] = 0
        self.post_contact_ready_safe_settlement_event[env_ids] = False
        self.post_contact_ready_unsafe_settlement_event[env_ids] = False
        self.post_contact_ready_incomplete_settlement_event[env_ids] = False
        self.post_contact_ready_safe_net_cycle_event[env_ids] = False
        self.post_contact_ready_shadow_consecutive_steps[env_ids] = 0
        self.post_contact_ready_shadow_success_latch[env_ids] = False
        self.post_contact_ready_outcome_tier[env_ids] = -1
        self.post_contact_ready_face_quality[env_ids] = 0.0
        self.post_contact_ready_planner_speed_ratio[env_ids] = 0.0
        self.post_contact_ready_planner_direction_error_rad[env_ids] = 0.0
        self.post_contact_ready_planner_normal_error_rad[env_ids] = 0.0
        self.post_contact_ready_planner_position_error[env_ids] = 0.0
        self.post_contact_ready_impact_health_score[env_ids] = 0.0
        self.post_contact_ready_peak_base_ang_vel[env_ids] = 0.0
        self.post_contact_ready_peak_ang_vel_excess_increment[env_ids] = 0.0
        self.post_contact_ready_envelope_violation_latch[env_ids] = False
        self.post_contact_ready_envelope_violation_event[env_ids] = False
        self.post_contact_ready_envelope_first_violation_step[env_ids] = -1
        self.post_contact_ready_max_tilt[env_ids] = 0.0
        self.post_contact_ready_max_abs_pitch[env_ids] = 0.0
        self.post_contact_ready_max_com_x[env_ids] = 0.0
        self.post_contact_ready_max_com_y[env_ids] = 0.0
        self.post_contact_ready_max_waist_overflow[env_ids] = 0.0
        self.post_contact_ready_max_leg_overflow[env_ids] = 0.0
        self._post_contact_ready_envelope_component_latches[env_ids] = False
        self._post_contact_ready_envelope_histogram_latches[env_ids] = False
        self._post_contact_ready_phase_seen_latches[env_ids] = False
        self._post_contact_ready_phase_histogram_latches[env_ids] = False
        self.post_contact_ready_previous_backlean_error[env_ids] = 0.0

    def _post_contact_ready_gate_for_level(
        self,
        level: int,
    ) -> dict[str, float | int]:
        if not bool(self.cfg.post_contact_ready_curriculum_enabled):
            return {
                "torso_x_min": float(
                    self.cfg.post_contact_ready_torso_x_min
                ),
                "torso_x_max": float(
                    self.cfg.post_contact_ready_torso_x_max
                ),
                "max_torso_ang_vel": float(
                    self.cfg.post_contact_ready_max_torso_ang_vel
                ),
                "max_base_lin_vel": float(
                    self.cfg.post_contact_ready_max_base_lin_vel
                ),
                "max_base_ang_vel": float(
                    self.cfg.post_contact_ready_max_base_ang_vel
                ),
                "max_racket_speed": float(
                    self.cfg.post_contact_ready_max_racket_speed
                ),
                "min_feet_contact": float(
                    self.cfg.post_contact_ready_min_feet_contact
                ),
                "required_consecutive_steps": int(
                    self.cfg.post_contact_ready_required_consecutive_steps
                ),
                "deadline_steps": int(self.cfg.post_contact_ready_deadline_steps),
            }

        return {
            "torso_x_min": ready_curriculum_stage_value(
                self.cfg.post_contact_ready_curriculum_torso_x_min, level
            ),
            "torso_x_max": ready_curriculum_stage_value(
                self.cfg.post_contact_ready_curriculum_torso_x_max, level
            ),
            "max_torso_ang_vel": ready_curriculum_stage_value(
                self.cfg.post_contact_ready_curriculum_max_torso_ang_vel, level
            ),
            "max_base_lin_vel": ready_curriculum_stage_value(
                self.cfg.post_contact_ready_curriculum_max_base_lin_vel, level
            ),
            "max_base_ang_vel": ready_curriculum_stage_value(
                self.cfg.post_contact_ready_curriculum_max_base_ang_vel, level
            ),
            "max_racket_speed": ready_curriculum_stage_value(
                self.cfg.post_contact_ready_curriculum_max_racket_speed, level
            ),
            "min_feet_contact": ready_curriculum_stage_value(
                self.cfg.post_contact_ready_curriculum_min_feet_contact, level
            ),
            "required_consecutive_steps": int(
                ready_curriculum_stage_value(
                    self.cfg.post_contact_ready_curriculum_required_consecutive_steps,
                    level,
                )
            ),
            "deadline_steps": int(
                ready_curriculum_stage_value(
                    self.cfg.post_contact_ready_curriculum_deadline_steps, level
                )
            ),
        }

    def _post_contact_ready_effective_gate(self) -> dict[str, float | int]:
        return self._post_contact_ready_gate_for_level(
            self._post_contact_ready_curriculum_level
        )

    def _update_post_contact_ready_hit_retention(
        self,
        opportunity_event: torch.Tensor,
        targeted_attempt_event: torch.Tensor,
        return_success_event: torch.Tensor,
    ) -> None:
        """Track true strike frames so READY cannot advance by idling."""
        if not bool(self.cfg.post_contact_ready_curriculum_enabled):
            return
        valid = opportunity_event & (~self.no_command_ready_active)
        if bool(self.cfg.single_cycle_curriculum_enabled) and bool(
            self.cfg.single_cycle_continuous_only_curriculum
        ):
            valid &= ~self.single_cycle_bootstrap
        if not bool(torch.any(valid)):
            return

        targeted_attempt = targeted_attempt_event[valid].float().mean()
        return_success = return_success_event[valid].float().mean()
        rate = min(
            max(
                float(self.cfg.post_contact_ready_curriculum_hit_ema_rate),
                0.0,
            ),
            1.0,
        )
        if not self._post_contact_ready_curriculum_hit_ema_initialized:
            self._post_contact_ready_curriculum_targeted_attempt_ema.copy_(
                targeted_attempt
            )
            self._post_contact_ready_curriculum_return_success_ema.copy_(
                return_success
            )
            self._post_contact_ready_curriculum_hit_ema_initialized = True
        else:
            self._post_contact_ready_curriculum_targeted_attempt_ema.lerp_(
                targeted_attempt,
                rate,
            )
            self._post_contact_ready_curriculum_return_success_ema.lerp_(
                return_success,
                rate,
            )
        self._post_contact_ready_curriculum_completed_swings += int(
            valid.sum().item()
        )

    def _update_post_contact_ready_curriculum(
        self,
        success_event: torch.Tensor,
        fail_event: torch.Tensor,
        shadow_success_event: torch.Tensor | None = None,
    ) -> None:
        """Advance the hard READY gate only from resolved recovery ability."""
        if not bool(self.cfg.post_contact_ready_curriculum_enabled):
            return
        if (
            bool(self.cfg.single_cycle_curriculum_enabled)
            and bool(self.cfg.single_cycle_continuous_only_curriculum)
            and success_event.shape == self.single_cycle_bootstrap.shape
        ):
            eligible = ~self.single_cycle_bootstrap
            success_event = success_event & eligible
            fail_event = fail_event & eligible
            if shadow_success_event is not None:
                shadow_success_event = shadow_success_event & eligible
        resolved_event = success_event | fail_event
        resolved_count = int(resolved_event.sum().item())
        if resolved_count == 0:
            return

        successful_count = int(success_event.sum().item())
        if shadow_success_event is None:
            shadow_success_event = torch.zeros_like(resolved_event)
        shadow_successful_count = int(
            (shadow_success_event & resolved_event).sum().item()
        )
        observed_success = torch.tensor(
            successful_count / resolved_count,
            dtype=torch.float32,
            device=self.device,
        )
        observed_shadow_success = torch.tensor(
            shadow_successful_count / resolved_count,
            dtype=torch.float32,
            device=self.device,
        )
        rate = min(
            max(float(self.cfg.post_contact_ready_curriculum_ema_rate), 0.0),
            1.0,
        )
        self._post_contact_ready_curriculum_success_ema.lerp_(
            observed_success,
            rate,
        )
        self._post_contact_ready_curriculum_shadow_success_ema.lerp_(
            observed_shadow_success,
            rate,
        )
        self._post_contact_ready_curriculum_resolved_events += resolved_count
        self._post_contact_ready_curriculum_successful_events += successful_count

        stage_count = len(
            self.cfg.post_contact_ready_curriculum_max_base_ang_vel
        )
        level = self._post_contact_ready_curriculum_level
        base_ready = ready_curriculum_should_advance(
            self._post_contact_ready_curriculum_level,
            stage_count,
            float(self._post_contact_ready_curriculum_success_ema.item()),
            int(self._post_contact_ready_curriculum_resolved_events.item()),
            self.cfg.post_contact_ready_curriculum_advance_success_thresholds,
            self.cfg.post_contact_ready_curriculum_min_resolved_events,
        )
        if level < stage_count - 1:
            guards_ready = ready_curriculum_guards_satisfied(
                float(
                    self._post_contact_ready_curriculum_shadow_success_ema.item()
                ),
                float(
                    self.cfg.post_contact_ready_curriculum_shadow_success_thresholds[
                        level
                    ]
                ),
                float(
                    self._post_contact_ready_curriculum_targeted_attempt_ema.item()
                ),
                float(
                    self.cfg.post_contact_ready_curriculum_min_targeted_attempt_ema
                ),
                float(
                    self._post_contact_ready_curriculum_return_success_ema.item()
                ),
                float(
                    self.cfg.post_contact_ready_curriculum_min_return_success_ema
                ),
                int(
                    self._post_contact_ready_curriculum_completed_swings.item()
                ),
                int(
                    self.cfg.post_contact_ready_curriculum_min_completed_swings
                ),
            )
        else:
            guards_ready = False
        self._post_contact_ready_curriculum_advance_guard_ok = (
            base_ready and guards_ready
        )
        if self._post_contact_ready_curriculum_advance_guard_ok:
            self._post_contact_ready_curriculum_advance_streak += 1
        else:
            self._post_contact_ready_curriculum_advance_streak = 0

        if (
            self._post_contact_ready_curriculum_advance_streak
            >= int(
                self.cfg.post_contact_ready_curriculum_required_advance_checks
            )
        ):
            self._post_contact_ready_curriculum_level += 1
            self._post_contact_ready_curriculum_success_ema.zero_()
            self._post_contact_ready_curriculum_shadow_success_ema.zero_()
            self._post_contact_ready_curriculum_resolved_events.zero_()
            self._post_contact_ready_curriculum_successful_events.zero_()
            self._post_contact_ready_curriculum_completed_swings.zero_()
            self._post_contact_ready_curriculum_advance_streak = 0
            self._post_contact_ready_curriculum_advance_guard_ok = False

    def _record_reset_safety_events(self, env_ids: torch.Tensor) -> None:
        """Count unsafe true resets so the curriculum cannot learn only from survivors."""
        if bool(self.cfg.single_cycle_curriculum_enabled) and bool(
            self.cfg.single_cycle_continuous_only_curriculum
        ):
            env_ids = env_ids[~self.single_cycle_bootstrap[env_ids]]
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

    def single_cycle_curriculum_probability(self) -> float:
        """Return the reset probability derived only from continuous ability."""
        if not bool(self.cfg.single_cycle_curriculum_enabled):
            self._single_cycle_probability.zero_()
            self._single_cycle_curriculum_level.zero_()
            self._single_cycle_effective_deadline_steps.fill_(
                int(self.cfg.single_cycle_deadline_steps)
            )
            return 0.0
        if int(self.cfg.single_cycle_deadline_steps) <= int(
            self.cfg.single_cycle_min_recovery_steps
        ):
            raise ValueError(
                "single_cycle_deadline_steps must exceed "
                "single_cycle_min_recovery_steps"
            )
        probability, level = ability_driven_single_cycle_probability(
            SingleCycleAbility(
                targeted_attempt=float(self._ability_targeted_attempt_ema.item()),
                contact=float(self._ability_contact_ema.item()),
                recovery=float(self._ability_recovery_ema.item()),
                safety=float(self._ability_safety_ema.item()),
                cycle_ready=float(self._ability_cycle_ema.item()),
                resolved_events=int(
                    self._ability_curriculum_total_resolved_events.item()
                ),
            ),
            probabilities=self.cfg.single_cycle_probabilities,
            targeted_attempt_thresholds=(
                self.cfg.single_cycle_targeted_attempt_thresholds
            ),
            contact_thresholds=self.cfg.single_cycle_contact_thresholds,
            recovery_thresholds=self.cfg.single_cycle_recovery_thresholds,
            safety_thresholds=self.cfg.single_cycle_safety_thresholds,
            cycle_ready_thresholds=(
                self.cfg.single_cycle_cycle_ready_thresholds
            ),
            min_resolved_events=int(self.cfg.single_cycle_min_resolved_events),
            min_continuous_fraction=float(
                self.cfg.single_cycle_min_continuous_fraction
            ),
        )
        self._single_cycle_probability.fill_(probability)
        self._single_cycle_curriculum_level.fill_(level)
        self._single_cycle_effective_deadline_steps.fill_(
            ability_driven_single_cycle_deadline(
                level,
                deadlines_by_level=self.cfg.single_cycle_deadline_steps_by_level,
                fallback_deadline_steps=int(self.cfg.single_cycle_deadline_steps),
                min_recovery_steps=int(self.cfg.single_cycle_min_recovery_steps),
                expected_levels=len(self.cfg.single_cycle_probabilities),
            )
        )
        return probability

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
        required_tier = self._cycle_v2_outcome_tier()
        if str(self.cfg.cycle_v2_settlement_tier_mode) == "achieved":
            if bool(self.cfg.cycle_v2_require_healthy_impact):
                contact = self.current_swing_healthy_contact[env_ids]
                net_cross = self.current_swing_healthy_net_cross[env_ids]
                opponent_bounce = self.current_swing_healthy_on_opponent[env_ids]
            else:
                contact = self.current_swing_contact[env_ids]
                net_cross = self.current_swing_net_cross[env_ids]
                opponent_bounce = self.current_swing_on_opponent[env_ids]
            achieved_tier = achieved_outcome_tier(
                contact=contact,
                net_cross=net_cross,
                opponent_bounce=opponent_bounce,
            )
            self.cycle_v2_outcome_tier[env_ids] = achieved_tier
            return achieved_tier >= required_tier

        tier = required_tier
        self.cycle_v2_outcome_tier[env_ids] = tier
        if bool(self.cfg.cycle_v2_require_healthy_impact):
            if tier == 0:
                return self.current_swing_healthy_contact[env_ids]
            if tier == 1:
                return self.current_swing_healthy_net_cross[env_ids]
            if tier == 2:
                return self.current_swing_healthy_on_opponent[env_ids]
            raise RuntimeError(f"Unsupported cycle-v2 outcome tier {tier}")
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
        self.cycle_v2_ready_consecutive_steps[env_ids] = 0
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

    def _update_station_relocation_curriculum(
        self,
        env_ids: torch.Tensor,
        *,
        terminal_reset: bool = False,
    ) -> None:
        """Advance lateral relocation difficulty from resolved hold outcomes."""
        if bool(self.cfg.single_cycle_curriculum_enabled) and bool(
            self.cfg.single_cycle_continuous_only_curriculum
        ):
            env_ids = env_ids[~self.single_cycle_bootstrap[env_ids]]
        if not bool(self.cfg.station_relocation_enabled) or len(env_ids) == 0:
            return
        valid = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        if terminal_reset:
            valid &= self.steps_since_target_resample[env_ids] > 0
        (
            resolved,
            arrival_success,
            settle_success,
            contact_success,
            safety_success,
        ) = station_relocation_resolution(
            rehearsal=self.station_relocation_rehearsal[env_ids],
            hold_seen=self.healthy_stage_station_hold_seen[env_ids],
            arrived=self.healthy_stage_station_arrived[env_ids],
            settled=self.healthy_stage_station_settled[env_ids],
            contact=self.current_swing_contact[env_ids],
            unsafe=self.current_swing_unsafe[env_ids],
            valid=valid,
            terminal_reset=terminal_reset,
        )
        if not bool(torch.any(resolved)):
            return

        resolved_count = resolved.float().sum().clamp_min(1.0)
        arrival = arrival_success.float().sum() / resolved_count
        settled = settle_success.float().sum() / resolved_count
        contact = contact_success.float().sum() / resolved_count
        safety = safety_success.float().sum() / resolved_count
        rate = min(max(float(self.cfg.station_relocation_ema_rate), 0.0), 1.0)
        if not self._station_relocation_initialized:
            self._station_relocation_arrival_ema.copy_(arrival)
            self._station_relocation_settle_ema.copy_(settled)
            self._station_relocation_contact_ema.copy_(contact)
            self._station_relocation_safety_ema.copy_(safety)
            self._station_relocation_initialized = True
        else:
            self._station_relocation_arrival_ema.lerp_(arrival, rate)
            self._station_relocation_settle_ema.lerp_(settled, rate)
            self._station_relocation_contact_ema.lerp_(contact, rate)
            self._station_relocation_safety_ema.lerp_(safety, rate)
        self._station_relocation_resolved_events += resolved.long().sum()
        if terminal_reset:
            self._station_relocation_terminal_reset_events += resolved.long().sum()

        good = (
            self._station_relocation_arrival_ema
            >= float(self.cfg.station_relocation_arrival_threshold)
        ) & (
            self._station_relocation_settle_ema
            >= float(self.cfg.station_relocation_settle_threshold)
        ) & (
            self._station_relocation_contact_ema
            >= float(self.cfg.station_relocation_contact_threshold)
        ) & (
            self._station_relocation_safety_ema
            >= float(self.cfg.station_relocation_safety_threshold)
        )
        bad = (
            self._station_relocation_arrival_ema
            < float(self.cfg.station_relocation_arrival_floor)
        ) | (
            self._station_relocation_settle_ema
            < float(self.cfg.station_relocation_settle_floor)
        ) | (
            self._station_relocation_contact_ema
            < float(self.cfg.station_relocation_contact_floor)
        ) | (
            self._station_relocation_safety_ema
            < float(self.cfg.station_relocation_safety_floor)
        )
        max_level = max(len(tuple(self.cfg.station_relocation_abs_y_ranges)) - 1, 0)
        next_level, advance_streak, regress_streak, changed = (
            hysteretic_curriculum_transition(
                level=int(self._station_relocation_level.item()),
                max_level=max_level,
                good=bool(good.item()),
                bad=bool(bad.item()),
                resolved_events=int(self._station_relocation_resolved_events.item()),
                min_resolved_events=int(
                    self.cfg.station_relocation_min_resolved_events
                ),
                advance_streak=int(self._station_relocation_advance_streak.item()),
                regress_streak=int(self._station_relocation_regress_streak.item()),
                required_advance_checks=int(
                    self.cfg.station_relocation_required_advance_checks
                ),
                required_regress_checks=int(
                    self.cfg.station_relocation_required_regress_checks
                ),
            )
        )
        self._station_relocation_level.fill_(next_level)
        self._station_relocation_advance_streak.fill_(advance_streak)
        self._station_relocation_regress_streak.fill_(regress_streak)
        if changed:
            self._station_relocation_resolved_events.zero_()

    @staticmethod
    def _capability_interval_gate(
        value: torch.Tensor, low: float, high: float
    ) -> torch.Tensor:
        """Map an ability scalar to [0, 1] without an iteration schedule."""
        lo = float(low)
        hi = float(high)
        if hi <= lo:
            raise ValueError(
                f"capability gate high must exceed low, got {lo} >= {hi}"
            )
        return ((value - lo) / (hi - lo)).clamp(0.0, 1.0)

    def safe_outcome_capability_gate(self) -> torch.Tensor:
        """Unlock higher return tiers only while foundation skills persist."""
        if not bool(self.cfg.safe_outcome_capability_gate_enabled):
            return torch.ones((), dtype=torch.float32, device=self.device)
        safe_rate = self.metrics[
            "post_contact_ready_terminal_safe_rate"
        ].mean()
        components = torch.stack(
            (
                self._capability_interval_gate(
                    self._ability_contact_ema,
                    self.cfg.safe_outcome_gate_contact_low,
                    self.cfg.safe_outcome_gate_contact_high,
                ),
                self._capability_interval_gate(
                    self._ability_forehand_contact_ema,
                    self.cfg.safe_outcome_gate_forehand_low,
                    self.cfg.safe_outcome_gate_forehand_high,
                ),
                self._capability_interval_gate(
                    self._ability_backhand_contact_ema,
                    self.cfg.safe_outcome_gate_backhand_low,
                    self.cfg.safe_outcome_gate_backhand_high,
                ),
                self._capability_interval_gate(
                    self._ability_safety_ema,
                    self.cfg.safe_outcome_gate_safety_low,
                    self.cfg.safe_outcome_gate_safety_high,
                ),
                self._capability_interval_gate(
                    self._ability_recovery_ema,
                    self.cfg.safe_outcome_gate_recovery_low,
                    self.cfg.safe_outcome_gate_recovery_high,
                ),
                self._capability_interval_gate(
                    safe_rate,
                    self.cfg.safe_outcome_gate_settlement_low,
                    self.cfg.safe_outcome_gate_settlement_high,
                ),
            )
        )
        return torch.min(components)

    def _update_ability_curriculum(self, env_ids: torch.Tensor) -> None:
        """Advance ball/planner difficulty from measured ability, not iteration count."""
        if bool(self.cfg.single_cycle_curriculum_enabled) and bool(
            self.cfg.single_cycle_continuous_only_curriculum
        ):
            env_ids = env_ids[~self.single_cycle_bootstrap[env_ids]]
        if (
            not bool(self.cfg.ability_curriculum_enabled)
            or bool(self.cfg.ability_curriculum_external_update)
            or len(env_ids) == 0
        ):
            return
        rate = min(max(float(self.cfg.ability_curriculum_ema_rate), 0.0), 1.0)
        if rate <= 0.0:
            return
        resolved_count = int(len(env_ids))
        self._ability_curriculum_resolved_events += resolved_count
        self._ability_curriculum_total_resolved_events += resolved_count

        if bool(self.cfg.ability_curriculum_require_healthy_impact):
            contact_source = self.current_swing_healthy_contact
            net_source = self.current_swing_healthy_net_cross
            success_source = self.current_swing_healthy_on_opponent
        else:
            contact_source = self.current_swing_contact
            net_source = self.current_swing_net_cross
            success_source = self.current_swing_on_opponent
        contact = contact_source[env_ids].float().mean()
        net = net_source[env_ids].float().mean()
        success = success_source[env_ids].float().mean()
        attempt = self.current_swing_attempt[env_ids].float().mean()
        targeted_attempt = self.current_swing_targeted_attempt[env_ids].float().mean()
        safety = (~self.current_swing_unsafe[env_ids]).float().mean()
        station_saturation = self.current_swing_station_saturated[env_ids].float().mean()
        healthy_normal = self.current_swing_healthy_normal_ok[env_ids].float().mean()
        station_seen = self.healthy_stage_station_hold_seen[env_ids]
        if bool(torch.any(station_seen)):
            station_arrival = self.healthy_stage_station_arrived[env_ids][station_seen].float().mean()
            station_settle = self.healthy_stage_station_settled[env_ids][station_seen].float().mean()
        else:
            station_arrival = self._ability_station_arrival_ema.detach()
            station_settle = self._ability_station_settle_ema.detach()
        side = self.swing_sign[env_ids]
        forehand = side >= 0.0
        backhand = side < 0.0

        def _masked_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
            if bool(torch.any(mask)):
                return values[mask].float().mean()
            return fallback.detach()

        contact_values = contact_source[env_ids].float()
        net_values = net_source[env_ids].float()
        success_values = success_source[env_ids].float()
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
            self._ability_targeted_attempt_ema.copy_(targeted_attempt)
            self._ability_safety_ema.copy_(safety)
            self._ability_station_saturation_ema.copy_(station_saturation)
            self._ability_healthy_normal_ema.copy_(healthy_normal)
            self._ability_station_arrival_ema.copy_(station_arrival)
            self._ability_station_settle_ema.copy_(station_settle)
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
            self._ability_targeted_attempt_ema.lerp_(targeted_attempt, rate)
            self._ability_safety_ema.lerp_(safety, rate)
            self._ability_station_saturation_ema.lerp_(station_saturation, rate)
            self._ability_healthy_normal_ema.lerp_(healthy_normal, rate)
            self._ability_station_arrival_ema.lerp_(station_arrival, rate)
            self._ability_station_settle_ema.lerp_(station_settle, rate)
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
        if bool(self.cfg.healthy_three_stage_enabled):
            stage = self.healthy_training_stage()
            side_contact_ok = (
                self._ability_forehand_contact_ema
                >= float(self.cfg.healthy_stage_side_contact_threshold)
            ) & (
                self._ability_backhand_contact_ema
                >= float(self.cfg.healthy_stage_side_contact_threshold)
            )
            normal_ok = (
                self._ability_healthy_normal_ema
                >= float(self.cfg.healthy_stage_normal_threshold)
            )
            safety_ok = (
                self._ability_safety_ema
                >= float(self.cfg.healthy_stage_safety_threshold)
            )
            recovery_ok = (
                self._ability_recovery_ema
                >= float(self.cfg.healthy_stage_recovery_threshold)
            )
            if stage == 1:
                can_advance = side_contact_ok & normal_ok & safety_ok & recovery_ok
            elif stage == 2:
                station_ok = (
                    self._ability_station_arrival_ema
                    >= float(self.cfg.healthy_stage_station_arrival_threshold)
                ) & (
                    self._ability_station_settle_ema
                    >= float(self.cfg.healthy_stage_station_settle_threshold)
                )
                can_advance = (
                    side_contact_ok & normal_ok & safety_ok & recovery_ok & station_ok
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
            attempt_ema = (
                self._ability_targeted_attempt_ema
                if bool(self.cfg.ability_curriculum_use_targeted_attempt)
                else self._ability_attempt_ema
            )
            can_advance = can_advance & (
                attempt_ema >= float(self.cfg.ability_curriculum_attempt_threshold)
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
        next_level, advance_streak, regress_streak, _changed, checked = (
            event_gated_scalar_curriculum_transition(
                level=float(self._ability_curriculum_level.item()),
                good=bool(can_advance),
                bad=bool(should_regress),
                resolved_events=int(self._ability_curriculum_resolved_events.item()),
                min_resolved_events=int(
                    self.cfg.ability_curriculum_min_resolved_events
                ),
                advance_streak=int(self._ability_curriculum_advance_streak.item()),
                regress_streak=int(self._ability_curriculum_regress_streak.item()),
                required_advance_checks=int(
                    self.cfg.ability_curriculum_required_advance_checks
                ),
                required_regress_checks=int(
                    self.cfg.ability_curriculum_required_regress_checks
                ),
                advance_rate=float(self.cfg.ability_curriculum_advance_rate),
                regress_rate=float(self.cfg.ability_curriculum_regress_rate),
            )
        )
        self._ability_curriculum_level.fill_(next_level)
        self._ability_curriculum_advance_streak.fill_(advance_streak)
        self._ability_curriculum_regress_streak.fill_(regress_streak)
        if checked:
            self._ability_curriculum_resolved_events.zero_()

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
        eligible = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if bool(self.cfg.single_cycle_curriculum_enabled) and bool(
            self.cfg.single_cycle_continuous_only_curriculum
        ):
            eligible &= ~self.single_cycle_bootstrap
        eligible_count = eligible.float().sum().clamp_min(1.0)
        attempts = self.cycle_v2_attempt_latch.float() * eligible.float()
        attempt_count = attempts.sum()
        attempt_rate = attempts.sum() / eligible_count
        if float(attempt_count.item()) > 0.0:
            success_rate = (
                (self.cycle_v2_ready_success_latch & eligible).float().sum()
                / attempt_count
            )
            resolved_rate = (
                (
                    (
                        self.cycle_v2_ready_success_latch
                        | self.cycle_v2_ready_fail_latch
                    )
                    & eligible
                )
                .float()
                .sum()
                / attempt_count
            )
        else:
            success_rate = torch.zeros((), dtype=torch.float32, device=self.device)
            resolved_rate = torch.zeros((), dtype=torch.float32, device=self.device)
        self._ability_cycle_attempt_ema.lerp_(attempt_rate, rate)
        self._ability_cycle_ema.lerp_(success_rate, rate)
        self._ability_cycle_resolved_ema.lerp_(resolved_rate, rate)

    # --- per-step updates ----------------------------------------------------------------------- #
    def _motion_time_to_strike(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        motion = self._motion()
        ml = motion.motion
        if self._strike_phase_per_clip is None:
            sp = tuple(self.cfg.strike_phase_per_clip)
            if sp and len(sp) == ml.num_segments:
                self._strike_phase_per_clip = torch.tensor([float(x) for x in sp], device=self.device)
            else:
                self._strike_phase_per_clip = torch.full((ml.num_segments,), float(self.cfg.strike_phase), device=self.device)
        if env_ids is None:
            clip = motion.clip_id
            time_steps = motion.time_steps
        else:
            env_ids = torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
            clip = motion.clip_id[env_ids]
            time_steps = motion.time_steps[env_ids]
        seg_start = ml.seg_start[clip]
        seg_len = ml.seg_len[clip]
        phase = self._strike_phase_per_clip[clip]
        strike_step = seg_start + (phase * (seg_len - 1).float()).round().long()
        return (strike_step - time_steps).float() * self._env.step_dt

    def _compute_strike_timing(self):
        motion_time_to_strike = self._motion_time_to_strike()
        active_override = self.physical_command_override_active
        self.true_time_to_strike = torch.where(
            active_override,
            self.physical_command_override_tts,
            motion_time_to_strike,
        )
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

    def _compute_impact_health(self) -> None:
        """Compute a smooth and hard trunk/support health gate at the current step."""
        data = self.robot.data
        torso_idx = self._impact_health_torso_index
        gravity_w = torch.zeros(self.num_envs, 3, device=self.device)
        gravity_w[:, 2] = -1.0
        torso_gravity = quat_rotate_inverse(data.body_quat_w[:, torso_idx], gravity_w)
        backlean_violation = (
            -torso_gravity[:, 0] - float(self.cfg.impact_health_backlean_tolerance)
        ).clamp_min(0.0)
        torso_roll = torch.abs(torso_gravity[:, 1])
        torso_ang_vel = torch.norm(data.body_ang_vel_w[:, torso_idx, :2], dim=-1)

        masses = data.default_mass.to(device=data.body_com_pos_w.device)
        com_w = (data.body_com_pos_w * masses.unsqueeze(-1)).sum(dim=1) / masses.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        support_xy = data.body_pos_w[:, self._impact_health_foot_indices, :2].mean(dim=1)
        com_delta_w = torch.zeros_like(com_w)
        com_delta_w[:, :2] = com_w[:, :2] - support_xy
        com_delta_b = quat_rotate_inverse(yaw_quat(self.base_quat_w), com_delta_w)
        com_x = torch.abs(com_delta_b[:, 0])
        com_y = torch.abs(com_delta_b[:, 1])
        backward_velocity = (-data.root_lin_vel_w[:, 0]).clamp_min(0.0)
        waist_pitch = data.joint_pos[:, self._impact_health_waist_pitch_index]
        waist_default = data.default_joint_pos[:, self._impact_health_waist_pitch_index]
        waist_backfold = (
            waist_default
            - waist_pitch
            - float(self.cfg.impact_health_waist_backfold_tolerance)
        ).clamp_min(0.0)
        base_retreat = (self.fixed_station_w[:, 0] - self.base_pos_w[:, 0]).clamp_min(0.0)
        feet = torch.clamp(self.feet_contact_frac, 0.0, 1.0)

        def _gaussian(value: torch.Tensor, std: float) -> torch.Tensor:
            return torch.exp(-torch.square(value / max(float(std), 1.0e-6)))

        legacy_waist_retreat = bool(self.cfg.impact_health_include_waist_retreat)
        include_waist_backfold = (
            legacy_waist_retreat
            or bool(self.cfg.impact_health_include_waist_backfold)
        )
        if legacy_waist_retreat:
            components = (
                (0.23, _gaussian(backlean_violation, self.cfg.impact_health_backlean_std)),
                (0.13, _gaussian(waist_backfold, self.cfg.impact_health_waist_backfold_std)),
                (0.12, _gaussian(base_retreat, self.cfg.impact_health_base_retreat_std)),
                (0.07, _gaussian(torso_roll, self.cfg.impact_health_roll_std)),
                (0.10, _gaussian(torso_ang_vel, self.cfg.impact_health_ang_vel_std)),
                (0.12, _gaussian(com_x, self.cfg.impact_health_com_x_std)),
                (0.08, _gaussian(com_y, self.cfg.impact_health_com_y_std)),
                (0.08, _gaussian(backward_velocity, self.cfg.impact_health_backward_vel_std)),
                (0.07, 0.5 + 0.5 * feet),
            )
        else:
            components = (
                (0.28, _gaussian(backlean_violation, self.cfg.impact_health_backlean_std)),
                (0.10, _gaussian(torso_roll, self.cfg.impact_health_roll_std)),
                (0.14, _gaussian(torso_ang_vel, self.cfg.impact_health_ang_vel_std)),
                (0.18, _gaussian(com_x, self.cfg.impact_health_com_x_std)),
                (0.12, _gaussian(com_y, self.cfg.impact_health_com_y_std)),
                (0.10, _gaussian(backward_velocity, self.cfg.impact_health_backward_vel_std)),
                (0.08, 0.5 + 0.5 * feet),
            )
        log_score = torch.zeros(self.num_envs, device=self.device)
        for weight, component in components:
            log_score += weight * torch.log(component.clamp_min(1.0e-6))
        if include_waist_backfold and not legacy_waist_retreat:
            waist_score = _gaussian(
                waist_backfold, self.cfg.impact_health_waist_backfold_std
            )
            log_score = 0.88 * log_score + 0.12 * torch.log(
                waist_score.clamp_min(1.0e-6)
            )
        self.impact_health_score = torch.exp(log_score)
        self.impact_health_ok = (
            (torso_gravity[:, 0] >= -float(self.cfg.impact_health_max_backlean))
            & (torso_roll <= float(self.cfg.impact_health_max_roll))
            & (torso_ang_vel <= float(self.cfg.impact_health_max_ang_vel))
            & (com_x <= float(self.cfg.impact_health_max_com_x))
            & (com_y <= float(self.cfg.impact_health_max_com_y))
            & (backward_velocity <= float(self.cfg.impact_health_max_backward_vel))
            & (feet >= float(self.cfg.impact_health_min_feet_contact))
        )
        if include_waist_backfold:
            self.impact_health_ok &= waist_backfold <= float(
                self.cfg.impact_health_max_waist_backfold
            )
        if legacy_waist_retreat:
            self.impact_health_ok &= base_retreat <= float(
                self.cfg.impact_health_max_base_retreat
            )
        self.metrics["impact_health_backlean_violation"] = backlean_violation
        self.metrics["impact_health_waist_backfold"] = waist_backfold
        self.metrics["impact_health_base_retreat"] = base_retreat
        self.metrics["impact_health_torso_gravity_x"] = torso_gravity[:, 0]
        self.metrics["impact_health_torso_ang_vel"] = torso_ang_vel
        self.metrics["impact_health_com_x"] = com_x
        self.metrics["impact_health_com_y"] = com_y

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

    def _functional_ready_anchor_score(self) -> torch.Tensor:
        """Track only the configured upper-body READY anchor; legs and waist remain free."""
        if (
            self._functional_ready_joint_positions is None
            or not self._functional_ready_joint_ids
        ):
            return torch.ones(self.num_envs, device=self.device)
        joint_pos = self.robot.data.joint_pos[:, self._functional_ready_joint_ids]
        if self._functional_ready_joint_tolerances is not None:
            return joint_deadband_score(
                joint_pos,
                self._functional_ready_joint_positions.unsqueeze(0).expand_as(joint_pos),
                self._functional_ready_joint_tolerances.unsqueeze(0).expand_as(joint_pos),
                self.cfg.functional_ready_joint_std,
            )
        error = torch.mean(
            torch.square(joint_pos - self._functional_ready_joint_positions),
            dim=-1,
        )
        std = max(float(self.cfg.functional_ready_joint_std), 1.0e-6)
        return torch.exp(-error / std**2)

    def _update_active_ready_state(self) -> None:
        """Emit one event only after READY remains valid for consecutive frames."""
        self.active_ready_success_event.zero_()
        active = self.no_command_ready_active
        ready_now = active & (
            self.metrics["recovery_functional_ready_score"]
            >= float(self.cfg.active_ready_score_threshold)
        )
        ready_now &= (
            self.metrics["recovery_height_error"]
            <= float(self.cfg.active_ready_max_height_error)
        )
        ready_now &= (
            self.metrics["recovery_upright_error"]
            <= float(self.cfg.active_ready_max_upright_error)
        )
        ready_now &= (
            self.metrics["recovery_base_lin_vel"]
            <= float(self.cfg.active_ready_max_base_lin_vel)
        )
        ready_now &= (
            self.metrics["recovery_base_ang_vel"]
            <= float(self.cfg.active_ready_max_base_ang_vel)
        )
        ready_now &= (
            self.metrics["recovery_feet_contact_frac"]
            >= float(self.cfg.active_ready_min_feet_contact)
        )
        self.active_ready_consecutive_steps.copy_(
            update_consecutive_steps(self.active_ready_consecutive_steps, ready_now)
        )
        self.active_ready_consecutive_steps[~active] = 0
        self.active_ready_success_latch[~active] = False
        reached = (
            active
            & (
                self.active_ready_consecutive_steps
                >= int(self.cfg.active_ready_required_consecutive_steps)
            )
            & (~self.active_ready_success_latch)
        )
        self.active_ready_success_event.copy_(reached)
        self.active_ready_success_latch |= reached

        if bool(self.cfg.active_ready_survival_milestones_enabled):
            survival_safe_now = active.clone()
            if bool(self.cfg.active_ready_survival_require_soft_envelope):
                survival_safe_now &= (
                    self.metrics["recovery_height_error"]
                    <= float(self.cfg.active_ready_survival_max_height_error)
                )
                survival_safe_now &= (
                    self.metrics["recovery_upright_error"]
                    <= float(self.cfg.active_ready_survival_max_upright_error)
                )
                survival_safe_now &= (
                    self.metrics["recovery_base_lin_vel"]
                    <= float(self.cfg.active_ready_survival_max_base_lin_vel)
                )
                survival_safe_now &= (
                    self.metrics["recovery_base_ang_vel"]
                    <= float(self.cfg.active_ready_survival_max_base_ang_vel)
                )
                survival_safe_now &= (
                    self.metrics["recovery_feet_contact_frac"]
                    >= float(self.cfg.active_ready_survival_min_feet_contact)
                )
            steps, next_index, event = update_survival_milestones(
                self.active_ready_survival_steps,
                self.active_ready_survival_next_milestone,
                active,
                survival_safe_now,
                self._active_ready_survival_milestone_steps,
                self._active_ready_survival_milestone_values,
            )
            self.active_ready_survival_steps.copy_(steps)
            self.active_ready_survival_next_milestone.copy_(next_index)
            self.active_ready_survival_milestone_event.copy_(event)
        else:
            self.active_ready_survival_steps.zero_()
            self.active_ready_survival_next_milestone.zero_()
            self.active_ready_survival_milestone_event.zero_()

    def _update_post_contact_safety_envelope(
        self,
        active: torch.Tensor,
    ) -> None:
        """Accumulate attempt-level recovery tail-risk diagnostics."""
        if not bool(torch.any(active)):
            return

        projected_gravity = self.robot.data.projected_gravity_b
        tilt = torch.norm(projected_gravity[:, :2], dim=-1)
        abs_pitch = torch.abs(projected_gravity[:, 0])
        com_x = self.metrics["impact_health_com_x"]
        com_y = self.metrics["impact_health_com_y"]
        waist_overflow = self.metrics["waist_action_overflow_rms"]
        leg_overflow = self.metrics["leg_action_overflow_rms"]
        base_ang_vel = self.metrics["recovery_base_ang_vel"]

        for target, value in (
            (self.post_contact_ready_max_tilt, tilt),
            (self.post_contact_ready_max_abs_pitch, abs_pitch),
            (self.post_contact_ready_max_com_x, com_x),
            (self.post_contact_ready_max_com_y, com_y),
            (self.post_contact_ready_max_waist_overflow, waist_overflow),
            (self.post_contact_ready_max_leg_overflow, leg_overflow),
        ):
            target[active] = torch.maximum(target[active], value[active])

        violations = recovery_safety_envelope_violations(
            tilt=tilt,
            abs_pitch=abs_pitch,
            com_x=com_x,
            com_y=com_y,
            waist_overflow=waist_overflow,
            leg_overflow=leg_overflow,
            base_ang_vel=base_ang_vel,
            max_tilt=float(
                self.cfg.post_contact_ready_envelope_max_tilt
            ),
            max_abs_pitch=float(
                self.cfg.post_contact_ready_envelope_max_abs_pitch
            ),
            max_com_x=float(
                self.cfg.post_contact_ready_envelope_max_com_x
            ),
            max_com_y=float(
                self.cfg.post_contact_ready_envelope_max_com_y
            ),
            max_waist_overflow=float(
                self.cfg.post_contact_ready_envelope_max_waist_overflow
            ),
            max_leg_overflow=float(
                self.cfg.post_contact_ready_envelope_max_leg_overflow
            ),
            max_base_ang_vel=float(
                self.cfg.post_contact_ready_envelope_max_base_ang_vel
            ),
        )
        (
            any_bad,
            tilt_bad,
            pitch_bad,
            com_bad,
            waist_bad,
            leg_bad,
            base_ang_bad,
            _,
        ) = violations
        first_bad = (
            active
            & any_bad
            & (~self.post_contact_ready_envelope_violation_latch)
        )
        self.post_contact_ready_envelope_violation_event.copy_(first_bad)
        self.post_contact_ready_envelope_first_violation_step[first_bad] = (
            self.post_contact_ready_diagnostic_elapsed_steps[first_bad]
        )
        self.post_contact_ready_envelope_violation_latch |= active & any_bad

        component_bad = torch.stack(
            (
                tilt_bad,
                pitch_bad,
                com_bad,
                waist_bad,
                leg_bad,
                base_ang_bad,
            ),
            dim=-1,
        )
        new_component_bad = (
            active.unsqueeze(-1)
            & component_bad
            & (~self._post_contact_ready_envelope_component_latches)
        )
        self._post_contact_ready_envelope_component_latches |= (
            active.unsqueeze(-1) & component_bad
        )
        self._post_contact_ready_envelope_violations[0] += first_bad.sum()
        self._post_contact_ready_envelope_violations[1:] += (
            new_component_bad.sum(dim=0)
        )
        histogram_values = torch.stack(
            (
                tilt,
                abs_pitch,
                com_x,
                com_y,
                waist_overflow,
                leg_overflow,
                base_ang_vel,
            ),
            dim=-1,
        )
        histogram_bad = (
            histogram_values.unsqueeze(-1)
            > self._post_contact_ready_envelope_histogram_thresholds.unsqueeze(0)
        )
        new_histogram_bad = (
            active[:, None, None]
            & histogram_bad
            & (~self._post_contact_ready_envelope_histogram_latches)
        )
        self._post_contact_ready_envelope_histogram_latches |= (
            active[:, None, None] & histogram_bad
        )
        self._post_contact_ready_envelope_histogram_counts += (
            new_histogram_bad.sum(dim=0)
        )
        phase_ids = recovery_phase_indices(
            self.post_contact_ready_diagnostic_elapsed_steps,
            step_dt=float(self._env.step_dt),
            boundaries_s=tuple(
                float(value)
                for value in self.cfg.post_contact_ready_diagnostic_phase_boundaries_s
            ),
        )
        for phase_index in range(4):
            phase_active = active & (phase_ids == phase_index)
            first_phase_sample = (
                phase_active
                & (~self._post_contact_ready_phase_seen_latches[:, phase_index])
            )
            self._post_contact_ready_phase_seen_latches[
                first_phase_sample, phase_index
            ] = True
            self._post_contact_ready_phase_attempt_counts[
                phase_index
            ] += first_phase_sample.sum()

            phase_histogram_bad = (
                phase_active[:, None, None] & histogram_bad
            )
            new_phase_histogram_bad = (
                phase_histogram_bad
                & (
                    ~self._post_contact_ready_phase_histogram_latches[
                        :, phase_index
                    ]
                )
            )
            self._post_contact_ready_phase_histogram_latches[
                :, phase_index
            ] |= phase_histogram_bad
            self._post_contact_ready_phase_histogram_counts[
                phase_index
            ] += new_phase_histogram_bad.sum(dim=0)

    def _record_post_contact_ready_resolution(
        self,
        success_event: torch.Tensor,
        fail_event: torch.Tensor,
    ) -> None:
        """Accumulate outcome-conditioned resolution and latency diagnostics."""
        for column, latency_index, event in (
            (1, 0, success_event),
            (2, 1, fail_event),
        ):
            if not bool(torch.any(event)):
                continue
            buckets = recovery_outcome_bucket(
                self.post_contact_ready_outcome_tier[event]
            )
            self._post_contact_ready_outcome_resolution_counts[
                :, column
            ] += torch.bincount(buckets, minlength=4)
            self._post_contact_ready_resolution_latency_steps[
                latency_index
            ] += self.post_contact_ready_elapsed_steps[event].sum()
            self._post_contact_ready_resolution_counts[
                latency_index
            ] += event.sum()

    def _record_post_contact_durable_resolution(
        self,
        success_event: torch.Tensor,
        fail_event: torch.Tensor,
    ) -> None:
        """Accumulate fixed-deadline durable READY shadow outcomes."""
        for column, latency_index, event in (
            (1, 0, success_event),
            (2, 1, fail_event),
        ):
            if not bool(torch.any(event)):
                continue
            buckets = recovery_outcome_bucket(
                self.post_contact_ready_outcome_tier[event]
            )
            self._post_contact_ready_durable_outcome_resolution_counts[
                :, column
            ] += torch.bincount(buckets, minlength=4)
            self._post_contact_ready_durable_resolution_latency_steps[
                latency_index
            ] += self.post_contact_ready_diagnostic_elapsed_steps[event].sum()
            self._post_contact_ready_durable_resolution_counts[
                latency_index
            ] += event.sum()
            if latency_index == 1:
                self._post_contact_ready_durable_failure_component_counts += (
                    (~self._post_contact_ready_durable_component_ok[event])
                    .sum(dim=0)
                )

    def _write_post_contact_ready_diagnostic_metrics(self) -> None:
        """Expose cumulative recovery-tail rates to the training logger."""
        envelope_denominator = (
            self._post_contact_ready_envelope_attempts.float().clamp_min(1.0)
        )
        envelope_rates = (
            self._post_contact_ready_envelope_violations.float()
            / envelope_denominator
        )
        envelope_metric_names = (
            "post_contact_ready_envelope_violation_rate",
            "post_contact_ready_envelope_tilt_violation_rate",
            "post_contact_ready_envelope_pitch_violation_rate",
            "post_contact_ready_envelope_com_violation_rate",
            "post_contact_ready_envelope_waist_violation_rate",
            "post_contact_ready_envelope_leg_violation_rate",
            "post_contact_ready_envelope_base_ang_vel_violation_rate",
        )
        self.metrics["post_contact_ready_envelope_attempts"] = (
            self._post_contact_ready_envelope_attempts.float()
            .view(1)
            .repeat(self.num_envs)
        )
        for index, name in enumerate(envelope_metric_names):
            self.metrics[name] = (
                envelope_rates[index].view(1).repeat(self.num_envs)
            )

        total_attempts = (
            self._post_contact_ready_outcome_resolution_counts[:, 0]
            .sum()
            .float()
            .clamp_min(1.0)
        )
        resolved_success = (
            self._post_contact_ready_outcome_resolution_counts[:, 1]
            .sum()
            .float()
        )
        resolved_fail = (
            self._post_contact_ready_outcome_resolution_counts[:, 2]
            .sum()
            .float()
        )
        success_latency = (
            self._post_contact_ready_resolution_latency_steps[0].float()
            / self._post_contact_ready_resolution_counts[0].float().clamp_min(1.0)
        )
        fail_latency = (
            self._post_contact_ready_resolution_latency_steps[1].float()
            / self._post_contact_ready_resolution_counts[1].float().clamp_min(1.0)
        )
        for name, scalar in (
            (
                "post_contact_ready_resolution_success_rate",
                resolved_success / total_attempts,
            ),
            (
                "post_contact_ready_resolution_fail_rate",
                resolved_fail / total_attempts,
            ),
            (
                "post_contact_ready_resolution_success_latency_steps",
                success_latency,
            ),
            (
                "post_contact_ready_resolution_fail_latency_steps",
                fail_latency,
            ),
        ):
            self.metrics[name] = scalar.view(1).repeat(self.num_envs)
        durable_failure_denominator = (
            self._post_contact_ready_durable_resolution_counts[1]
            .float()
            .clamp_min(1.0)
        )
        durable_failure_component_rates = (
            self._post_contact_ready_durable_failure_component_counts.float()
            / durable_failure_denominator
        )
        for index, name in enumerate(
            (
                "post_contact_ready_durable_fail_backlean_rate",
                "post_contact_ready_durable_fail_forward_lean_rate",
                "post_contact_ready_durable_fail_torso_ang_vel_rate",
                "post_contact_ready_durable_fail_base_lin_vel_rate",
                "post_contact_ready_durable_fail_base_ang_vel_rate",
                "post_contact_ready_durable_fail_racket_speed_rate",
                "post_contact_ready_durable_fail_height_rate",
                "post_contact_ready_durable_fail_com_x_rate",
                "post_contact_ready_durable_fail_com_y_rate",
                "post_contact_ready_durable_fail_feet_rate",
                "post_contact_ready_durable_fail_station_rate",
                "post_contact_ready_durable_fail_arm_rate",
            )
        ):
            self.metrics[name] = (
                durable_failure_component_rates[index]
                .view(1)
                .repeat(self.num_envs)
            )

        terminal_total = (
            self._post_contact_ready_terminal_settlement_counts[0]
            .float()
            .clamp_min(1.0)
        )
        terminal_safe = (
            self._post_contact_ready_terminal_settlement_counts[1].float()
        )
        terminal_unsafe = (
            self._post_contact_ready_terminal_settlement_counts[2].float()
        )
        terminal_incomplete = (
            self._post_contact_ready_terminal_settlement_counts[3].float()
        )
        for name, scalar in (
            (
                "post_contact_ready_terminal_quality_mean",
                self._post_contact_ready_terminal_quality_sums[0]
                / terminal_total,
            ),
            (
                "post_contact_ready_safe_terminal_quality_mean",
                self._post_contact_ready_terminal_quality_sums[1]
                / terminal_safe.clamp_min(1.0),
            ),
            (
                "post_contact_ready_terminal_safe_rate",
                terminal_safe / terminal_total,
            ),
            (
                "post_contact_ready_terminal_unsafe_rate",
                terminal_unsafe / terminal_total,
            ),
            (
                "post_contact_ready_terminal_incomplete_rate",
                terminal_incomplete / terminal_total,
            ),
            (
                "post_contact_ready_safe_net_cycle_rate",
                self._post_contact_ready_safe_net_cycle_count.float()
                / terminal_total,
            ),
        ):
            self.metrics[name] = scalar.view(1).repeat(self.num_envs)

        durable_total_attempts = (
            self._post_contact_ready_durable_outcome_resolution_counts[:, 0]
            .sum()
            .float()
            .clamp_min(1.0)
        )
        durable_resolved_success = (
            self._post_contact_ready_durable_outcome_resolution_counts[:, 1]
            .sum()
            .float()
        )
        durable_resolved_fail = (
            self._post_contact_ready_durable_outcome_resolution_counts[:, 2]
            .sum()
            .float()
        )
        durable_success_latency = (
            self._post_contact_ready_durable_resolution_latency_steps[0].float()
            / self._post_contact_ready_durable_resolution_counts[0]
            .float()
            .clamp_min(1.0)
        )
        durable_fail_latency = (
            self._post_contact_ready_durable_resolution_latency_steps[1].float()
            / self._post_contact_ready_durable_resolution_counts[1]
            .float()
            .clamp_min(1.0)
        )
        for name, scalar in (
            (
                "post_contact_ready_durable_resolution_success_rate",
                durable_resolved_success / durable_total_attempts,
            ),
            (
                "post_contact_ready_durable_resolution_fail_rate",
                durable_resolved_fail / durable_total_attempts,
            ),
            (
                "post_contact_ready_durable_resolution_success_latency_steps",
                durable_success_latency,
            ),
            (
                "post_contact_ready_durable_resolution_fail_latency_steps",
                durable_fail_latency,
            ),
        ):
            self.metrics[name] = scalar.view(1).repeat(self.num_envs)

        phase_names = (
            "impact_0_100ms",
            "brake_100_300ms",
            "settle_300_600ms",
            "ready_after_600ms",
        )
        # Component/threshold pairs use the shared histogram tensor:
        # tilt>0.1, abs-pitch>0.1, waist-overflow>1.0, base-ang>0.8.
        selected_rates = (
            ("tilt_gt_0p1_rate", 0, 0),
            ("pitch_gt_0p1_rate", 1, 0),
            ("waist_overflow_gt_1p0_rate", 4, 1),
            ("base_ang_vel_gt_0p8_rate", 6, 0),
        )
        for phase_index, phase_name in enumerate(phase_names):
            phase_attempts = self._post_contact_ready_phase_attempt_counts[
                phase_index
            ].float()
            denominator = phase_attempts.clamp_min(1.0)
            self.metrics[
                f"post_contact_phase_{phase_name}_attempts"
            ] = phase_attempts.view(1).repeat(self.num_envs)
            for metric_name, component_index, threshold_index in selected_rates:
                rate = (
                    self._post_contact_ready_phase_histogram_counts[
                        phase_index, component_index, threshold_index
                    ].float()
                    / denominator
                )
                self.metrics[
                    f"post_contact_phase_{phase_name}_{metric_name}"
                ] = rate.view(1).repeat(self.num_envs)

    def _update_post_contact_ready_state(self) -> None:
        """Require a directional, sustained return to a reusable READY region."""
        self.post_contact_ready_success_event.zero_()
        self.post_contact_ready_fail_event.zero_()
        self.post_contact_ready_durable_success_event.zero_()
        self.post_contact_ready_durable_fail_event.zero_()
        self.post_contact_ready_terminal_window_active.zero_()
        self.post_contact_ready_terminal_settlement_event.zero_()
        self.post_contact_ready_operational_ready_now.zero_()
        self.post_contact_ready_safe_settlement_event.zero_()
        self.post_contact_ready_unsafe_settlement_event.zero_()
        self.post_contact_ready_incomplete_settlement_event.zero_()
        self.post_contact_ready_safe_net_cycle_event.zero_()
        self.post_contact_ready_peak_ang_vel_excess_increment.zero_()
        self.post_contact_ready_envelope_violation_event.zero_()

        if not bool(self.cfg.post_contact_ready_enabled):
            all_envs = torch.arange(self.num_envs, device=self.device)
            self._reset_post_contact_ready(all_envs)
            for key in (
                "post_contact_ready_pending",
                "post_contact_ready_elapsed_steps",
                "post_contact_ready_consecutive_steps",
                "post_contact_ready_diagnostic_pending",
                "post_contact_ready_diagnostic_elapsed_steps",
                "post_contact_ready_durable_pending",
                "post_contact_ready_durable_consecutive_steps",
                "post_contact_ready_durable_ready_now",
                "post_contact_ready_durable_success_event",
                "post_contact_ready_durable_fail_event",
                "post_contact_ready_durable_resolution_success_rate",
                "post_contact_ready_durable_resolution_fail_rate",
                "post_contact_ready_durable_resolution_success_latency_steps",
                "post_contact_ready_durable_resolution_fail_latency_steps",
                "post_contact_ready_durable_fail_backlean_rate",
                "post_contact_ready_durable_fail_forward_lean_rate",
                "post_contact_ready_durable_fail_torso_ang_vel_rate",
                "post_contact_ready_durable_fail_base_lin_vel_rate",
                "post_contact_ready_durable_fail_base_ang_vel_rate",
                "post_contact_ready_durable_fail_racket_speed_rate",
                "post_contact_ready_durable_fail_height_rate",
                "post_contact_ready_durable_fail_com_x_rate",
                "post_contact_ready_durable_fail_com_y_rate",
                "post_contact_ready_durable_fail_feet_rate",
                "post_contact_ready_durable_fail_station_rate",
                "post_contact_ready_durable_fail_arm_rate",
                "post_contact_ready_terminal_window_active",
                "post_contact_ready_terminal_quality",
                "post_contact_ready_terminal_settlement_event",
                "post_contact_ready_operational_ready_now",
                "post_contact_ready_operational_consecutive_steps",
                "post_contact_ready_safe_settlement_event",
                "post_contact_ready_unsafe_settlement_event",
                "post_contact_ready_incomplete_settlement_event",
                "post_contact_ready_safe_net_cycle_event",
                "post_contact_ready_terminal_quality_mean",
                "post_contact_ready_safe_terminal_quality_mean",
                "post_contact_ready_terminal_safe_rate",
                "post_contact_ready_terminal_unsafe_rate",
                "post_contact_ready_terminal_incomplete_rate",
                "post_contact_ready_safe_net_cycle_rate",
                "post_contact_ready_score",
                "post_contact_ready_now",
                "post_contact_ready_success_event",
                "post_contact_ready_fail_event",
                "post_contact_ready_outcome_tier",
                "post_contact_ready_planner_speed_ratio",
                "post_contact_ready_planner_direction_error_rad",
                "post_contact_ready_planner_normal_error_rad",
                "post_contact_ready_planner_position_error",
                "post_contact_ready_impact_health_score",
                "post_contact_ready_peak_base_ang_vel",
                "post_contact_ready_peak_ang_vel_excess_increment",
                "post_contact_ready_envelope_violation_latch",
                "post_contact_ready_envelope_violation_event",
                "post_contact_ready_envelope_first_violation_step",
                "post_contact_ready_envelope_attempts",
                "post_contact_ready_envelope_violation_rate",
                "post_contact_ready_envelope_tilt_violation_rate",
                "post_contact_ready_envelope_pitch_violation_rate",
                "post_contact_ready_envelope_com_violation_rate",
                "post_contact_ready_envelope_waist_violation_rate",
                "post_contact_ready_envelope_leg_violation_rate",
                "post_contact_ready_envelope_base_ang_vel_violation_rate",
                "post_contact_ready_max_tilt",
                "post_contact_ready_max_abs_pitch",
                "post_contact_ready_max_com_x",
                "post_contact_ready_max_com_y",
                "post_contact_ready_max_waist_overflow",
                "post_contact_ready_max_leg_overflow",
                "post_contact_recovery_backlean_error",
                "post_contact_recovery_directional_progress",
                "post_contact_ready_torso_ok",
                "post_contact_ready_motion_ok",
                "post_contact_ready_support_ok",
                "post_contact_ready_arm_ok",
                "post_contact_ready_base_lin_vel_ok",
                "post_contact_ready_base_ang_vel_ok",
                "post_contact_ready_racket_speed_ok",
                "post_contact_ready_height_ok",
                "post_contact_ready_com_x_ok",
                "post_contact_ready_com_y_ok",
                "post_contact_ready_feet_ok",
                "post_contact_ready_station_ok",
                "post_contact_ready_fail_torso",
                "post_contact_ready_fail_base_lin_vel",
                "post_contact_ready_fail_base_ang_vel",
                "post_contact_ready_fail_racket_speed",
                "post_contact_ready_fail_support",
                "post_contact_ready_fail_arm",
                "post_contact_ready_shadow_now",
                "post_contact_ready_shadow_consecutive_steps",
                "post_contact_ready_shadow_success_latch",
                "post_contact_ready_curriculum_level",
                "post_contact_ready_curriculum_fraction",
                "post_contact_ready_curriculum_success_ema",
                "post_contact_ready_curriculum_shadow_success_ema",
                "post_contact_ready_curriculum_resolved_events",
                "post_contact_ready_curriculum_successful_events",
                "post_contact_ready_curriculum_targeted_attempt_ema",
                "post_contact_ready_curriculum_return_success_ema",
                "post_contact_ready_curriculum_completed_swings",
                "post_contact_ready_curriculum_advance_streak",
                "post_contact_ready_curriculum_advance_guard_ok",
                "post_contact_ready_effective_max_torso_ang_vel",
                "post_contact_ready_effective_torso_x_min",
                "post_contact_ready_effective_torso_x_max",
                "post_contact_ready_effective_max_base_lin_vel",
                "post_contact_ready_effective_max_base_ang_vel",
                "post_contact_ready_effective_max_racket_speed",
                "post_contact_ready_effective_min_feet_contact",
                "post_contact_ready_effective_required_consecutive_steps",
                "post_contact_ready_effective_deadline_steps",
                "post_contact_ready_curriculum_advance_threshold",
                "post_contact_ready_curriculum_advance_min_events",
                "post_contact_ready_curriculum_shadow_threshold",
                "post_contact_ready_curriculum_min_targeted_attempt_ema",
                "post_contact_ready_curriculum_min_return_success_ema",
                "post_contact_ready_curriculum_min_completed_swings",
                "post_contact_ready_curriculum_required_advance_checks",
            ):
                self.metrics[key].zero_()
            self.metrics["closed_loop_recovery_trigger_event"].zero_()
            return

        effective_gate = self._post_contact_ready_effective_gate()
        torso_x = self.metrics["impact_health_torso_gravity_x"]
        torso_x_min = float(effective_gate["torso_x_min"])
        torso_x_max = float(effective_gate["torso_x_max"])
        backlean_error = (torso_x_min - torso_x).clamp_min(0.0)
        if str(self.cfg.post_contact_ready_progress_error_mode) == "bounded_interval":
            backlean_error = backlean_error + (
                torso_x - torso_x_max
            ).clamp_min(0.0)
        new_recovery_trigger = recovery_trigger_event(
            str(self.cfg.post_contact_ready_trigger),
            strike_fired=self.strike_fired,
            targeted_attempt=self.current_swing_targeted_attempt,
            ball_contact=self.ball_contact,
        )
        self.metrics["closed_loop_recovery_trigger_event"] = (
            new_recovery_trigger.float()
        )
        new_ids = torch.nonzero(
            new_recovery_trigger, as_tuple=False
        ).squeeze(-1)
        if new_ids.numel() > 0:
            replaced_durable = self.post_contact_ready_durable_pending[
                new_ids
            ].clone()
            if bool(torch.any(replaced_durable)):
                replaced_global = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
                replaced_global[new_ids] = replaced_durable
                self._record_post_contact_durable_resolution(
                    torch.zeros_like(replaced_global),
                    replaced_global,
                )
            self._reset_post_contact_ready(new_ids)
            self.post_contact_ready_pending[new_ids] = True
            self.post_contact_ready_diagnostic_pending[new_ids] = True
            if bool(self.cfg.post_contact_ready_durable_diagnostic_enabled):
                self.post_contact_ready_durable_pending[new_ids] = True
            self._post_contact_ready_envelope_attempts += new_ids.numel()
            self.post_contact_ready_previous_backlean_error[new_ids] = (
                backlean_error[new_ids]
            )
            actual_speed = torch.norm(
                self.racket_lin_vel_w[new_ids], dim=-1
            )
            target_speed = torch.norm(
                self.racket_target_vel_w[new_ids], dim=-1
            ).clamp_min(1.0e-6)
            velocity_cosine = torch.sum(
                self.racket_lin_vel_w[new_ids]
                * self.racket_target_vel_w[new_ids],
                dim=-1,
            ) / (actual_speed.clamp_min(1.0e-6) * target_speed)
            self.post_contact_ready_planner_speed_ratio[new_ids] = (
                actual_speed / target_speed
            )
            self.post_contact_ready_planner_direction_error_rad[new_ids] = (
                torch.acos(velocity_cosine.clamp(-1.0, 1.0))
            )
            actual_normal = self.racket_normal_w[new_ids]
            actual_normal = actual_normal / torch.norm(
                actual_normal, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            target_normal = self.racket_target_normal_w[new_ids]
            target_normal = target_normal / torch.norm(
                target_normal, dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            self.post_contact_ready_planner_normal_error_rad[new_ids] = (
                torch.acos(
                    torch.sum(actual_normal * target_normal, dim=-1).clamp(
                        -1.0, 1.0
                    )
                )
            )
            self.post_contact_ready_planner_position_error[new_ids] = (
                torch.norm(
                    self.racket_pos_w[new_ids]
                    - self.racket_target_pos_w[new_ids],
                    dim=-1,
                )
            )
            self.post_contact_ready_impact_health_score[new_ids] = (
                self.impact_health_score[new_ids]
            )
            self.post_contact_ready_face_quality[new_ids] = (
                self.impact_face_quality[new_ids]
            )
            self.post_contact_ready_peak_base_ang_vel[new_ids] = (
                self.metrics["recovery_base_ang_vel"][new_ids]
            )
            peak_budget = torch.full_like(
                self.post_contact_ready_peak_base_ang_vel[new_ids],
                float(effective_gate["max_base_ang_vel"]),
            )
            self.post_contact_ready_peak_ang_vel_excess_increment[new_ids] = (
                recovery_peak_excess_increment(
                    torch.zeros_like(
                        self.post_contact_ready_peak_base_ang_vel[new_ids]
                    ),
                    self.post_contact_ready_peak_base_ang_vel[new_ids],
                    peak_budget,
                    excess_std=float(
                        self.cfg.post_contact_ready_peak_excess_std
                    ),
                    max_potential=float(
                        self.cfg.post_contact_ready_peak_excess_max_potential
                    ),
                )
            )
            tier = torch.full(
                (new_ids.numel(),), -1, dtype=torch.long, device=self.device
            )
            tier = torch.where(
                self.ball_contact[new_ids], torch.zeros_like(tier), tier
            )
            tier = torch.where(self.ball_net_cross[new_ids], torch.ones_like(tier), tier)
            tier = torch.where(
                self.ball_on_opponent[new_ids],
                torch.full_like(tier, 2),
                tier,
            )
            self.post_contact_ready_outcome_tier[new_ids] = tier
            outcome_buckets = recovery_outcome_bucket(tier)
            self._post_contact_ready_outcome_resolution_counts[
                :, 0
            ] += torch.bincount(outcome_buckets, minlength=4)
            if bool(self.cfg.post_contact_ready_durable_diagnostic_enabled):
                self._post_contact_ready_durable_outcome_resolution_counts[
                    :, 0
                ] += torch.bincount(outcome_buckets, minlength=4)

        active = self.post_contact_ready_pending & (~new_recovery_trigger)
        self.post_contact_ready_elapsed_steps[active] += 1
        diagnostic_active = (
            self.post_contact_ready_diagnostic_pending
            & (~new_recovery_trigger)
        )
        self.post_contact_ready_diagnostic_elapsed_steps[
            diagnostic_active
        ] += 1
        previous_peak = self.post_contact_ready_peak_base_ang_vel[active].clone()
        current_peak = torch.maximum(
            previous_peak,
            self.metrics["recovery_base_ang_vel"][active],
        )
        self.post_contact_ready_peak_base_ang_vel[active] = current_peak
        peak_budget = torch.full_like(
            current_peak,
            float(effective_gate["max_base_ang_vel"]),
        )
        self.post_contact_ready_peak_ang_vel_excess_increment[active] = (
            recovery_peak_excess_increment(
                previous_peak,
                current_peak,
                peak_budget,
                excess_std=float(
                    self.cfg.post_contact_ready_peak_excess_std
                ),
                max_potential=float(
                    self.cfg.post_contact_ready_peak_excess_max_potential
                ),
            )
        )
        self._update_post_contact_safety_envelope(
            self.post_contact_ready_diagnostic_pending
        )
        diagnostic_horizon_steps = max(
            1,
            int(
                round(
                    float(
                        self.cfg.post_contact_ready_diagnostic_horizon_s
                    )
                    / float(self._env.step_dt)
                )
            ),
        )
        diagnostic_finished = (
            self.post_contact_ready_diagnostic_pending
            & (
                self.post_contact_ready_diagnostic_elapsed_steps
                >= diagnostic_horizon_steps
            )
        )
        self.post_contact_ready_diagnostic_pending[
            diagnostic_finished
        ] = False
        progress = directional_error_progress(
            self.post_contact_ready_previous_backlean_error,
            backlean_error,
            self.cfg.post_contact_ready_progress_clip,
        )
        progress = torch.where(active, progress, torch.zeros_like(progress))

        torso_score = bounded_gaussian_score(
            torso_x,
            torso_x_min,
            torso_x_max,
            self.cfg.post_contact_ready_torso_x_std,
        )
        torso_ang_vel = self.metrics["impact_health_torso_ang_vel"]
        torso_ang_score = torch.exp(
            -torch.square(
                torso_ang_vel
                / max(float(effective_gate["max_torso_ang_vel"]), 1.0e-6)
            )
        )
        base_lin_score = torch.exp(
            -torch.square(
                self.metrics["recovery_base_lin_vel"]
                / max(float(effective_gate["max_base_lin_vel"]), 1.0e-6)
            )
        )
        base_ang_score = torch.exp(
            -torch.square(
                self.metrics["recovery_base_ang_vel"]
                / max(float(effective_gate["max_base_ang_vel"]), 1.0e-6)
            )
        )
        racket_score = torch.exp(
            -torch.square(
                self.metrics["recovery_racket_speed"]
                / max(float(effective_gate["max_racket_speed"]), 1.0e-6)
            )
        )
        com_x = self.metrics["impact_health_com_x"]
        com_y = self.metrics["impact_health_com_y"]
        com_x_score = torch.exp(
            -torch.square(
                com_x / max(float(self.cfg.post_contact_ready_max_com_x), 1.0e-6)
            )
        )
        com_y_score = torch.exp(
            -torch.square(
                com_y / max(float(self.cfg.post_contact_ready_max_com_y), 1.0e-6)
            )
        )
        anchor_score = self.metrics["recovery_anchor_arm_score"]
        components = torch.stack(
            (
                torso_score,
                torso_ang_score,
                self.metrics["recovery_height_score"],
                base_lin_score,
                base_ang_score,
                com_x_score,
                com_y_score,
                self.metrics["recovery_feet_contact_frac"].clamp(0.0, 1.0),
                self.metrics["recovery_station_score"],
                racket_score,
                anchor_score,
            ),
            dim=-1,
        )
        weights = torch.tensor(
            (0.18, 0.12, 0.06, 0.08, 0.12, 0.10, 0.08, 0.10, 0.05, 0.05, 0.06),
            device=self.device,
        )
        ready_score = weighted_geometric_mean(components, weights)

        torso_ok = (
            (torso_x >= torso_x_min)
            & (torso_x <= torso_x_max)
            & (
                torso_ang_vel
                <= float(effective_gate["max_torso_ang_vel"])
            )
        )
        base_lin_vel_ok = (
            self.metrics["recovery_base_lin_vel"]
            <= float(effective_gate["max_base_lin_vel"])
        )
        base_ang_vel_ok = (
            self.metrics["recovery_base_ang_vel"]
            <= float(effective_gate["max_base_ang_vel"])
        )
        racket_speed_ok = (
            self.metrics["recovery_racket_speed"]
            <= float(effective_gate["max_racket_speed"])
        )
        motion_ok = base_lin_vel_ok & base_ang_vel_ok & racket_speed_ok
        height_ok = (
            self.metrics["recovery_height_error"]
            <= float(self.cfg.post_contact_ready_max_height_error)
        )
        com_x_ok = com_x <= float(self.cfg.post_contact_ready_max_com_x)
        com_y_ok = com_y <= float(self.cfg.post_contact_ready_max_com_y)
        feet_ok = (
            self.metrics["recovery_feet_contact_frac"]
            >= float(effective_gate["min_feet_contact"])
        )
        station_ok = (
            self.metrics["recovery_station_error"]
            <= float(self.cfg.post_contact_ready_max_station_error)
        )
        support_ok = (
            height_ok & com_x_ok & com_y_ok & feet_ok & station_ok
        )
        arm_ok = anchor_score >= float(self.cfg.post_contact_ready_min_arm_score)
        ready_now = active & torso_ok & motion_ok & support_ok & arm_ok
        self.post_contact_ready_consecutive_steps.copy_(
            update_consecutive_steps(
                self.post_contact_ready_consecutive_steps,
                ready_now,
            )
        )

        durable_active = (
            self.post_contact_ready_durable_pending
            & (~new_recovery_trigger)
        )
        durable_min_delay_steps = max(
            0,
            int(
                round(
                    float(self.cfg.post_contact_ready_durable_min_delay_s)
                    / float(self._env.step_dt)
                )
            ),
        )
        durable_deadline_steps = max(
            1,
            int(
                round(
                    float(self.cfg.post_contact_ready_durable_deadline_s)
                    / float(self._env.step_dt)
                )
            ),
        )
        durable_use_effective = bool(
            self.cfg.post_contact_ready_durable_use_effective_gate
        )
        durable_torso_x_min = (
            torso_x_min
            if durable_use_effective
            else float(self.cfg.post_contact_ready_torso_x_min)
        )
        durable_torso_x_max = (
            torso_x_max
            if durable_use_effective
            else float(self.cfg.post_contact_ready_torso_x_max)
        )
        durable_backlean_ok = torso_x >= durable_torso_x_min
        durable_forward_lean_ok = torso_x <= durable_torso_x_max
        durable_max_torso_ang_vel = (
            float(effective_gate["max_torso_ang_vel"])
            if durable_use_effective
            else float(self.cfg.post_contact_ready_max_torso_ang_vel)
        )
        durable_max_base_lin_vel = (
            float(effective_gate["max_base_lin_vel"])
            if durable_use_effective
            else float(self.cfg.post_contact_ready_max_base_lin_vel)
        )
        durable_max_base_ang_vel = (
            float(effective_gate["max_base_ang_vel"])
            if durable_use_effective
            else float(self.cfg.post_contact_ready_max_base_ang_vel)
        )
        durable_max_racket_speed = (
            float(effective_gate["max_racket_speed"])
            if durable_use_effective
            else float(self.cfg.post_contact_ready_max_racket_speed)
        )
        durable_min_feet_contact = (
            float(effective_gate["min_feet_contact"])
            if durable_use_effective
            else float(self.cfg.post_contact_ready_min_feet_contact)
        )
        durable_required_steps = (
            int(effective_gate["required_consecutive_steps"])
            if durable_use_effective
            else int(self.cfg.post_contact_ready_durable_required_consecutive_steps)
        )
        durable_torso_ang_vel_ok = (
            torso_ang_vel <= durable_max_torso_ang_vel
        )
        durable_torso_ok = (
            durable_backlean_ok
            & durable_forward_lean_ok
            & durable_torso_ang_vel_ok
        )
        durable_base_lin_vel_ok = (
            self.metrics["recovery_base_lin_vel"]
            <= durable_max_base_lin_vel
        )
        durable_base_ang_vel_ok = (
            self.metrics["recovery_base_ang_vel"]
            <= durable_max_base_ang_vel
        )
        durable_racket_speed_ok = (
            self.metrics["recovery_racket_speed"]
            <= durable_max_racket_speed
        )
        durable_motion_ok = (
            durable_base_lin_vel_ok
            & durable_base_ang_vel_ok
            & durable_racket_speed_ok
        )
        durable_feet_ok = (
            self.metrics["recovery_feet_contact_frac"]
            >= durable_min_feet_contact
        )
        durable_support_ok = (
            height_ok
            & com_x_ok
            & com_y_ok
            & durable_feet_ok
            & station_ok
        )
        durable_eligible = (
            self.post_contact_ready_diagnostic_elapsed_steps
            >= durable_min_delay_steps
        )
        terminal_window_active = terminal_quality_window_mask(
            pending=durable_active,
            elapsed_steps=self.post_contact_ready_diagnostic_elapsed_steps,
            deadline_steps=durable_deadline_steps,
            window_steps=durable_required_steps,
        )
        self.post_contact_ready_terminal_window_active.copy_(
            terminal_window_active
        )
        self.post_contact_ready_terminal_quality_sum += (
            ready_score * terminal_window_active.float()
        )
        self.post_contact_ready_terminal_quality_count += (
            terminal_window_active.long()
        )
        self.post_contact_ready_terminal_quality.copy_(
            torch.where(
                self.post_contact_ready_terminal_quality_count > 0,
                self.post_contact_ready_terminal_quality_sum
                / self.post_contact_ready_terminal_quality_count
                .float()
                .clamp_min(1.0),
                torch.zeros_like(
                    self.post_contact_ready_terminal_quality_sum
                ),
            )
        )
        base_tilt = torch.norm(
            self.robot.data.projected_gravity_b[:, :2], dim=-1
        )
        operational_ready_now = (
            terminal_window_active
            & (
                base_tilt
                <= float(
                    self.cfg.post_contact_ready_operational_max_tilt
                )
            )
            & (
                torso_ang_vel
                <= float(
                    self.cfg
                    .post_contact_ready_operational_max_torso_ang_vel
                )
            )
            & (
                self.metrics["recovery_base_ang_vel"]
                <= float(
                    self.cfg
                    .post_contact_ready_operational_max_base_ang_vel
                )
            )
            & (
                self.metrics["recovery_height_error"]
                <= float(
                    self.cfg
                    .post_contact_ready_operational_max_height_error
                )
            )
            & (
                com_x
                <= float(
                    self.cfg.post_contact_ready_operational_max_com_x
                )
            )
            & (
                com_y
                <= float(
                    self.cfg.post_contact_ready_operational_max_com_y
                )
            )
            & (
                self.metrics["recovery_feet_contact_frac"]
                >= float(
                    self.cfg
                    .post_contact_ready_operational_min_feet_contact
                )
            )
            & (
                self.metrics["recovery_station_error"]
                <= float(
                    self.cfg
                    .post_contact_ready_operational_max_station_error
                )
            )
        )
        self.post_contact_ready_operational_ready_now.copy_(
            operational_ready_now
        )
        self.post_contact_ready_operational_consecutive_steps.copy_(
            update_consecutive_steps(
                self.post_contact_ready_operational_consecutive_steps,
                operational_ready_now,
            )
        )
        durable_component_now = torch.stack(
            (
                durable_backlean_ok,
                durable_forward_lean_ok,
                durable_torso_ang_vel_ok,
                durable_base_lin_vel_ok,
                durable_base_ang_vel_ok,
                durable_racket_speed_ok,
                height_ok,
                com_x_ok,
                com_y_ok,
                durable_feet_ok,
                station_ok,
                arm_ok,
            ),
            dim=-1,
        )
        durable_component_now &= (
            durable_active & durable_eligible
        ).unsqueeze(-1)
        self._post_contact_ready_durable_component_consecutive_steps.copy_(
            update_consecutive_steps(
                self._post_contact_ready_durable_component_consecutive_steps,
                durable_component_now,
            )
        )
        self._post_contact_ready_durable_component_ok.copy_(
            self._post_contact_ready_durable_component_consecutive_steps
            >= int(
                durable_required_steps
            )
        )
        durable_ready_now = (
            durable_active
            & durable_eligible
            & durable_torso_ok
            & durable_motion_ok
            & durable_support_ok
            & arm_ok
        )
        self.post_contact_ready_durable_ready_now.copy_(durable_ready_now)
        self.post_contact_ready_durable_consecutive_steps.copy_(
            update_consecutive_steps(
                self.post_contact_ready_durable_consecutive_steps,
                durable_ready_now,
            )
        )
        durable_success_event, durable_fail_event = (
            durable_ready_resolution_events(
                pending=self.post_contact_ready_durable_pending,
                elapsed_steps=(
                    self.post_contact_ready_diagnostic_elapsed_steps
                ),
                deadline_steps=durable_deadline_steps,
                consecutive_ready_steps=(
                    self.post_contact_ready_durable_consecutive_steps
                ),
                required_consecutive_steps=int(
                    durable_required_steps
                ),
            )
        )
        self.post_contact_ready_durable_success_event.copy_(
            durable_success_event
        )
        self.post_contact_ready_durable_fail_event.copy_(durable_fail_event)
        self._record_post_contact_durable_resolution(
            durable_success_event,
            durable_fail_event,
        )
        terminal_settlement_event = (
            durable_success_event | durable_fail_event
        )
        operational_ready = (
            self.post_contact_ready_operational_consecutive_steps
            >= int(
                    durable_required_steps
            )
        )
        (
            safe_settlement_event,
            unsafe_settlement_event,
            incomplete_settlement_event,
        ) = operational_terminal_events(
                settlement_event=terminal_settlement_event,
                catastrophic_violation_latch=(
                    self.post_contact_ready_envelope_violation_latch
                ),
                operational_ready=operational_ready,
            )
        safe_net_cycle_event = (
            safe_settlement_event
            & (self.post_contact_ready_outcome_tier >= 1)
        )
        self.post_contact_ready_terminal_settlement_event.copy_(
            terminal_settlement_event
        )
        self.post_contact_ready_safe_settlement_event.copy_(
            safe_settlement_event
        )
        self.post_contact_ready_unsafe_settlement_event.copy_(
            unsafe_settlement_event
        )
        self.post_contact_ready_incomplete_settlement_event.copy_(
            incomplete_settlement_event
        )
        self.post_contact_ready_safe_net_cycle_event.copy_(
            safe_net_cycle_event
        )
        if bool(torch.any(terminal_settlement_event)):
            self._post_contact_ready_terminal_settlement_counts[0] += (
                terminal_settlement_event.sum()
            )
            self._post_contact_ready_terminal_settlement_counts[1] += (
                safe_settlement_event.sum()
            )
            self._post_contact_ready_terminal_settlement_counts[2] += (
                unsafe_settlement_event.sum()
            )
            self._post_contact_ready_terminal_settlement_counts[3] += (
                incomplete_settlement_event.sum()
            )
            self._post_contact_ready_terminal_quality_sums[0] += (
                self.post_contact_ready_terminal_quality[
                    terminal_settlement_event
                ].sum()
            )
            self._post_contact_ready_terminal_quality_sums[1] += (
                self.post_contact_ready_terminal_quality[
                    safe_settlement_event
                ].sum()
            )
            self._post_contact_ready_safe_net_cycle_count += (
                safe_net_cycle_event.sum()
            )
        self.post_contact_ready_durable_pending[
            durable_success_event | durable_fail_event
        ] = False

        stage_count = len(
            self.cfg.post_contact_ready_curriculum_max_base_ang_vel
        )
        level = self._post_contact_ready_curriculum_level
        transition_available = (
            bool(self.cfg.post_contact_ready_curriculum_enabled)
            and level < stage_count - 1
        )
        if transition_available:
            shadow_gate = self._post_contact_ready_gate_for_level(level + 1)
            shadow_torso_ok = (
                (torso_x >= float(shadow_gate["torso_x_min"]))
                & (torso_x <= float(shadow_gate["torso_x_max"]))
                & (
                    torso_ang_vel
                    <= float(shadow_gate["max_torso_ang_vel"])
                )
            )
            shadow_motion_ok = (
                (
                    self.metrics["recovery_base_lin_vel"]
                    <= float(shadow_gate["max_base_lin_vel"])
                )
                & (
                    self.metrics["recovery_base_ang_vel"]
                    <= float(shadow_gate["max_base_ang_vel"])
                )
                & (
                    self.metrics["recovery_racket_speed"]
                    <= float(shadow_gate["max_racket_speed"])
                )
            )
            shadow_support_ok = (
                height_ok
                & com_x_ok
                & com_y_ok
                & (
                    self.metrics["recovery_feet_contact_frac"]
                    >= float(shadow_gate["min_feet_contact"])
                )
                & station_ok
            )
            shadow_now = (
                active
                & shadow_torso_ok
                & shadow_motion_ok
                & shadow_support_ok
                & arm_ok
            )
            self.post_contact_ready_shadow_consecutive_steps.copy_(
                update_consecutive_steps(
                    self.post_contact_ready_shadow_consecutive_steps,
                    shadow_now,
                )
            )
            # The current stage closes as soon as its sustained event resolves.
            # Shadow evaluation therefore checks the next physical envelope for
            # the current stage's duration, then lets the next stage lengthen it.
            shadow_required_steps = min(
                int(effective_gate["required_consecutive_steps"]),
                int(shadow_gate["required_consecutive_steps"]),
            )
            self.post_contact_ready_shadow_success_latch |= (
                active
                & (
                    self.post_contact_ready_shadow_consecutive_steps
                    >= shadow_required_steps
                )
            )
        else:
            shadow_now = torch.zeros_like(active)
            self.post_contact_ready_shadow_consecutive_steps.zero_()
            self.post_contact_ready_shadow_success_latch.zero_()

        success_event = (
            active
            & (
                self.post_contact_ready_consecutive_steps
                >= int(effective_gate["required_consecutive_steps"])
            )
        )
        fail_event = (
            active
            & (~success_event)
            & (
                self.post_contact_ready_elapsed_steps
                >= int(effective_gate["deadline_steps"])
            )
        )
        resolved_event = success_event | fail_event
        shadow_success_event = (
            resolved_event & self.post_contact_ready_shadow_success_latch
        )
        self.post_contact_ready_success_event.copy_(success_event)
        self.post_contact_ready_fail_event.copy_(fail_event)
        self._record_post_contact_ready_resolution(success_event, fail_event)
        self.post_contact_ready_pending[resolved_event] = False
        self.post_contact_ready_previous_backlean_error[active] = backlean_error[active]

        self.metrics["post_contact_ready_pending"] = (
            self.post_contact_ready_pending.float()
        )
        self.metrics["post_contact_ready_elapsed_steps"] = (
            self.post_contact_ready_elapsed_steps.float()
        )
        self.metrics["post_contact_ready_consecutive_steps"] = (
            self.post_contact_ready_consecutive_steps.float()
        )
        self.metrics["post_contact_ready_diagnostic_pending"] = (
            self.post_contact_ready_diagnostic_pending.float()
        )
        self.metrics["post_contact_ready_diagnostic_elapsed_steps"] = (
            self.post_contact_ready_diagnostic_elapsed_steps.float()
        )
        self.metrics["post_contact_ready_durable_pending"] = (
            self.post_contact_ready_durable_pending.float()
        )
        self.metrics["post_contact_ready_durable_consecutive_steps"] = (
            self.post_contact_ready_durable_consecutive_steps.float()
        )
        self.metrics["post_contact_ready_durable_ready_now"] = (
            self.post_contact_ready_durable_ready_now.float()
        )
        self.metrics["post_contact_ready_durable_success_event"] = (
            durable_success_event.float()
        )
        self.metrics["post_contact_ready_durable_fail_event"] = (
            durable_fail_event.float()
        )
        self.metrics["post_contact_ready_terminal_window_active"] = (
            terminal_window_active.float()
        )
        self.metrics["post_contact_ready_terminal_quality"] = (
            self.post_contact_ready_terminal_quality
        )
        self.metrics["post_contact_ready_terminal_settlement_event"] = (
            terminal_settlement_event.float()
        )
        self.metrics["post_contact_ready_operational_ready_now"] = (
            operational_ready_now.float()
        )
        self.metrics[
            "post_contact_ready_operational_consecutive_steps"
        ] = self.post_contact_ready_operational_consecutive_steps.float()
        self.metrics["post_contact_ready_safe_settlement_event"] = (
            safe_settlement_event.float()
        )
        self.metrics["post_contact_ready_unsafe_settlement_event"] = (
            unsafe_settlement_event.float()
        )
        self.metrics["post_contact_ready_incomplete_settlement_event"] = (
            incomplete_settlement_event.float()
        )
        self.metrics["post_contact_ready_safe_net_cycle_event"] = (
            safe_net_cycle_event.float()
        )
        self.metrics["post_contact_ready_score"] = ready_score
        self.metrics["post_contact_ready_now"] = ready_now.float()
        self.metrics["post_contact_ready_success_event"] = success_event.float()
        self.metrics["post_contact_ready_fail_event"] = fail_event.float()
        self.metrics["post_contact_ready_outcome_tier"] = (
            self.post_contact_ready_outcome_tier.float()
        )
        self.metrics["post_contact_ready_face_quality"] = (
            self.post_contact_ready_face_quality
        )
        self.metrics["post_contact_ready_safe_face_net_cycle_value"] = (
            safe_net_cycle_event.float()
            * self.post_contact_ready_face_quality
        )
        self.metrics["post_contact_ready_planner_speed_ratio"] = (
            self.post_contact_ready_planner_speed_ratio
        )
        self.metrics["post_contact_ready_planner_direction_error_rad"] = (
            self.post_contact_ready_planner_direction_error_rad
        )
        self.metrics["post_contact_ready_planner_normal_error_rad"] = (
            self.post_contact_ready_planner_normal_error_rad
        )
        self.metrics["post_contact_ready_planner_position_error"] = (
            self.post_contact_ready_planner_position_error
        )
        self.metrics["post_contact_ready_impact_health_score"] = (
            self.post_contact_ready_impact_health_score
        )
        self.metrics["post_contact_ready_peak_base_ang_vel"] = (
            self.post_contact_ready_peak_base_ang_vel
        )
        self.metrics["post_contact_ready_peak_ang_vel_excess_increment"] = (
            self.post_contact_ready_peak_ang_vel_excess_increment
        )
        self.metrics["post_contact_ready_envelope_violation_latch"] = (
            self.post_contact_ready_envelope_violation_latch.float()
        )
        self.metrics["post_contact_ready_envelope_violation_event"] = (
            self.post_contact_ready_envelope_violation_event.float()
        )
        self.metrics["post_contact_ready_envelope_first_violation_step"] = (
            self.post_contact_ready_envelope_first_violation_step.float()
        )
        self._write_post_contact_ready_diagnostic_metrics()
        self.metrics["post_contact_ready_max_tilt"] = (
            self.post_contact_ready_max_tilt
        )
        self.metrics["post_contact_ready_max_abs_pitch"] = (
            self.post_contact_ready_max_abs_pitch
        )
        self.metrics["post_contact_ready_max_com_x"] = (
            self.post_contact_ready_max_com_x
        )
        self.metrics["post_contact_ready_max_com_y"] = (
            self.post_contact_ready_max_com_y
        )
        self.metrics["post_contact_ready_max_waist_overflow"] = (
            self.post_contact_ready_max_waist_overflow
        )
        self.metrics["post_contact_ready_max_leg_overflow"] = (
            self.post_contact_ready_max_leg_overflow
        )
        self.metrics["post_contact_recovery_backlean_error"] = backlean_error
        self.metrics["post_contact_recovery_directional_progress"] = progress
        self.metrics["post_contact_ready_torso_ok"] = torso_ok.float()
        self.metrics["post_contact_ready_motion_ok"] = motion_ok.float()
        self.metrics["post_contact_ready_support_ok"] = support_ok.float()
        self.metrics["post_contact_ready_arm_ok"] = arm_ok.float()
        self.metrics["post_contact_ready_base_lin_vel_ok"] = (
            base_lin_vel_ok.float()
        )
        self.metrics["post_contact_ready_base_ang_vel_ok"] = (
            base_ang_vel_ok.float()
        )
        self.metrics["post_contact_ready_racket_speed_ok"] = (
            racket_speed_ok.float()
        )
        self.metrics["post_contact_ready_height_ok"] = height_ok.float()
        self.metrics["post_contact_ready_com_x_ok"] = com_x_ok.float()
        self.metrics["post_contact_ready_com_y_ok"] = com_y_ok.float()
        self.metrics["post_contact_ready_feet_ok"] = feet_ok.float()
        self.metrics["post_contact_ready_station_ok"] = station_ok.float()
        self.metrics["post_contact_ready_fail_torso"] = (
            fail_event & (~torso_ok)
        ).float()
        self.metrics["post_contact_ready_fail_base_lin_vel"] = (
            fail_event & (~base_lin_vel_ok)
        ).float()
        self.metrics["post_contact_ready_fail_base_ang_vel"] = (
            fail_event & (~base_ang_vel_ok)
        ).float()
        self.metrics["post_contact_ready_fail_racket_speed"] = (
            fail_event & (~racket_speed_ok)
        ).float()
        self.metrics["post_contact_ready_fail_support"] = (
            fail_event & (~support_ok)
        ).float()
        self.metrics["post_contact_ready_fail_arm"] = (
            fail_event & (~arm_ok)
        ).float()
        self.metrics["post_contact_ready_shadow_now"] = shadow_now.float()
        self.metrics["post_contact_ready_shadow_consecutive_steps"] = (
            self.post_contact_ready_shadow_consecutive_steps.float()
        )
        self.metrics["post_contact_ready_shadow_success_latch"] = (
            self.post_contact_ready_shadow_success_latch.float()
        )
        if bool(self.cfg.post_contact_ready_curriculum_enabled):
            advance_threshold = (
                float(
                    self.cfg.post_contact_ready_curriculum_advance_success_thresholds[
                        level
                    ]
                )
                if transition_available
                else 1.0
            )
            advance_min_events = (
                int(
                    self.cfg.post_contact_ready_curriculum_min_resolved_events[
                        level
                    ]
                )
                if transition_available
                else 0
            )
            shadow_threshold = (
                float(
                    self.cfg.post_contact_ready_curriculum_shadow_success_thresholds[
                        level
                    ]
                )
                if transition_available
                else 1.0
            )
            curriculum_fraction = level / max(stage_count - 1, 1)
        else:
            level = 0
            curriculum_fraction = 1.0
            advance_threshold = 1.0
            advance_min_events = 0
            shadow_threshold = 1.0
        curriculum_metrics = {
            "post_contact_ready_curriculum_level": float(level),
            "post_contact_ready_curriculum_fraction": float(curriculum_fraction),
            "post_contact_ready_curriculum_success_ema": float(
                self._post_contact_ready_curriculum_success_ema.item()
            ),
            "post_contact_ready_curriculum_shadow_success_ema": float(
                self._post_contact_ready_curriculum_shadow_success_ema.item()
            ),
            "post_contact_ready_curriculum_resolved_events": float(
                self._post_contact_ready_curriculum_resolved_events.item()
            ),
            "post_contact_ready_curriculum_successful_events": float(
                self._post_contact_ready_curriculum_successful_events.item()
            ),
            "post_contact_ready_curriculum_targeted_attempt_ema": float(
                self._post_contact_ready_curriculum_targeted_attempt_ema.item()
            ),
            "post_contact_ready_curriculum_return_success_ema": float(
                self._post_contact_ready_curriculum_return_success_ema.item()
            ),
            "post_contact_ready_curriculum_completed_swings": float(
                self._post_contact_ready_curriculum_completed_swings.item()
            ),
            "post_contact_ready_curriculum_advance_streak": float(
                self._post_contact_ready_curriculum_advance_streak
            ),
            "post_contact_ready_curriculum_advance_guard_ok": float(
                self._post_contact_ready_curriculum_advance_guard_ok
            ),
            "post_contact_ready_effective_max_torso_ang_vel": float(
                effective_gate["max_torso_ang_vel"]
            ),
            "post_contact_ready_effective_torso_x_min": torso_x_min,
            "post_contact_ready_effective_torso_x_max": torso_x_max,
            "post_contact_ready_effective_max_base_lin_vel": float(
                effective_gate["max_base_lin_vel"]
            ),
            "post_contact_ready_effective_max_base_ang_vel": float(
                effective_gate["max_base_ang_vel"]
            ),
            "post_contact_ready_effective_max_racket_speed": float(
                effective_gate["max_racket_speed"]
            ),
            "post_contact_ready_effective_min_feet_contact": float(
                effective_gate["min_feet_contact"]
            ),
            "post_contact_ready_effective_required_consecutive_steps": float(
                effective_gate["required_consecutive_steps"]
            ),
            "post_contact_ready_effective_deadline_steps": float(
                effective_gate["deadline_steps"]
            ),
            "post_contact_ready_curriculum_advance_threshold": advance_threshold,
            "post_contact_ready_curriculum_advance_min_events": float(
                advance_min_events
            ),
            "post_contact_ready_curriculum_shadow_threshold": shadow_threshold,
            "post_contact_ready_curriculum_min_targeted_attempt_ema": float(
                self.cfg.post_contact_ready_curriculum_min_targeted_attempt_ema
            ),
            "post_contact_ready_curriculum_min_return_success_ema": float(
                self.cfg.post_contact_ready_curriculum_min_return_success_ema
            ),
            "post_contact_ready_curriculum_min_completed_swings": float(
                self.cfg.post_contact_ready_curriculum_min_completed_swings
            ),
            "post_contact_ready_curriculum_required_advance_checks": float(
                self.cfg.post_contact_ready_curriculum_required_advance_checks
            ),
        }
        for key, value in curriculum_metrics.items():
            self.metrics[key].fill_(value)

        self._update_post_contact_ready_curriculum(
            success_event,
            fail_event,
            shadow_success_event,
        )

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
        operational_excess = action_term.operational_excess
        operational_outside = operational_excess > 1.0e-8
        q_des_velocity = action_term.q_des_velocity
        q_des_acceleration = action_term.q_des_acceleration
        q_des_velocity_excess = action_term.q_des_velocity_excess_ratio
        q_des_acceleration_excess = (
            action_term.q_des_acceleration_excess_ratio
        )
        q_des_velocity_violated = q_des_velocity_excess > 0.0
        q_des_acceleration_violated = q_des_acceleration_excess > 0.0
        severe_overflow = torch.amax(torch.abs(overflow_action), dim=-1) >= float(
            self.cfg.actuator_safety_overflow_threshold
        )
        self.actuator_overflow_consecutive_steps.copy_(
            update_consecutive_steps(
                self.actuator_overflow_consecutive_steps,
                severe_overflow,
            )
        )
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
        anchor_arm = self._functional_ready_anchor_score()
        feet = torch.clamp(self.feet_contact_frac, 0.0, 1.0)
        ready = 0.18 * height + 0.18 * upright + 0.15 * lin + 0.15 * ang + 0.12 * feet + 0.10 * station + 0.07 * racket + 0.05 * arm
        core_ready = weighted_geometric_mean(
            torch.stack((height, upright, lin, ang, feet), dim=-1),
            torch.tensor((0.18, 0.23, 0.18, 0.23, 0.18), device=self.device),
        )
        secondary_ready = (
            0.45
            + 0.20 * station
            + 0.15 * racket
            + 0.20 * anchor_arm
        )
        functional_ready = core_ready * secondary_ready

        no_command_active = self.no_command_ready_active
        no_command_progress = active_score_progress(
            self.no_command_ready_previous_score,
            ready,
            no_command_active,
            self.no_command_ready_previous_active,
            float(self.cfg.no_command_ready_progress_clip),
        )
        self.no_command_ready_previous_score.copy_(
            torch.where(no_command_active, ready, torch.zeros_like(ready))
        )
        self.no_command_ready_previous_active.copy_(no_command_active)

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
        waist_operational_excess = _group_rms(operational_excess, 0, 3)
        right_arm_operational_excess = _group_rms(operational_excess, 12, 19)
        leg_operational_excess = _group_rms(operational_excess, 19, 31)
        self.metrics["operational_margin_fraction"] = (
            operational_outside.float().mean(dim=-1)
        )
        self.metrics["waist_operational_margin_fraction"] = _group_fraction(
            operational_outside, 0, 3
        )
        self.metrics["right_arm_operational_margin_fraction"] = _group_fraction(
            operational_outside, 12, 19
        )
        self.metrics["leg_operational_margin_fraction"] = _group_fraction(
            operational_outside, 19, 31
        )
        self.metrics["waist_operational_excess_rms"] = waist_operational_excess
        self.metrics["right_arm_operational_excess_rms"] = (
            right_arm_operational_excess
        )
        self.metrics["leg_operational_excess_rms"] = leg_operational_excess
        self.metrics["q_des_velocity_violation_fraction"] = (
            q_des_velocity_violated.float().mean(dim=-1)
        )
        self.metrics["waist_q_des_velocity_violation_fraction"] = _group_fraction(
            q_des_velocity_violated, 0, 3
        )
        self.metrics["right_arm_q_des_velocity_violation_fraction"] = _group_fraction(
            q_des_velocity_violated, 12, 19
        )
        self.metrics["leg_q_des_velocity_violation_fraction"] = _group_fraction(
            q_des_velocity_violated, 19, 31
        )
        self.metrics["q_des_acceleration_violation_fraction"] = (
            q_des_acceleration_violated.float().mean(dim=-1)
        )
        self.metrics["waist_q_des_acceleration_violation_fraction"] = _group_fraction(
            q_des_acceleration_violated, 0, 3
        )
        self.metrics["right_arm_q_des_acceleration_violation_fraction"] = _group_fraction(
            q_des_acceleration_violated, 12, 19
        )
        self.metrics["leg_q_des_acceleration_violation_fraction"] = _group_fraction(
            q_des_acceleration_violated, 19, 31
        )
        self.metrics["waist_q_des_velocity_rms"] = _group_rms(q_des_velocity, 0, 3)
        self.metrics["right_arm_q_des_velocity_rms"] = _group_rms(
            q_des_velocity, 12, 19
        )
        self.metrics["leg_q_des_velocity_rms"] = _group_rms(q_des_velocity, 19, 31)
        self.metrics["waist_q_des_acceleration_rms"] = _group_rms(
            q_des_acceleration, 0, 3
        )
        self.metrics["right_arm_q_des_acceleration_rms"] = _group_rms(
            q_des_acceleration, 12, 19
        )
        self.metrics["leg_q_des_acceleration_rms"] = _group_rms(
            q_des_acceleration, 19, 31
        )
        self.metrics["action_operational_feasibility_score"] = torch.exp(
            -torch.square(waist_operational_excess / 0.03)
            -torch.square(leg_operational_excess / 0.03)
            -0.20 * torch.square(right_arm_operational_excess / 0.03)
        )
        self.metrics["actuator_overflow_consecutive_steps"] = (
            self.actuator_overflow_consecutive_steps.float()
        )
        self.metrics["actuator_overflow_severe"] = severe_overflow.float()
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
        self.metrics["recovery_anchor_arm_score"] = anchor_arm
        self.metrics["recovery_core_ready_score"] = core_ready
        self.metrics["recovery_functional_ready_score"] = functional_ready
        self.metrics["no_command_ready_score"] = (
            ready * no_command_active.float()
        )
        self.metrics["no_command_ready_progress"] = no_command_progress
        self.metrics["no_command_ready_station_error"] = (
            station_error * no_command_active.float()
        )
        self.metrics["no_command_ready_fixed_station_error"] = (
            torch.norm(self.base_pos_w[:, :2] - self.fixed_station_w, dim=-1)
            * no_command_active.float()
        )
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
        motion = self._motion()
        self.metrics["no_command_from_stand_episode"] = (
            self.no_command_ready_active & motion.stand_episode
        ).float()
        self.metrics["no_command_from_default_stand_reset"] = (
            self.no_command_ready_active & motion.default_stand_reset
        ).float()
        self.metrics["no_command_ready_upright_ok"] = (
            self.no_command_ready_active
            & (
                self.metrics["recovery_upright_error"]
                <= float(self.cfg.active_ready_max_upright_error)
            )
        ).float()
        self.metrics["no_command_ready_motion_ok"] = (
            self.no_command_ready_active
            & (
                self.metrics["recovery_base_lin_vel"]
                <= float(self.cfg.active_ready_max_base_lin_vel)
            )
            & (
                self.metrics["recovery_base_ang_vel"]
                <= float(self.cfg.active_ready_max_base_ang_vel)
            )
        ).float()
        self._update_active_ready_state()
        self.metrics["active_ready_consecutive_steps"] = (
            self.active_ready_consecutive_steps.float()
        )
        self.metrics["active_ready_success_event"] = (
            self.active_ready_success_event.float()
        )
        self.metrics["active_ready_success_latch"] = (
            self.active_ready_success_latch.float()
        )
        self.metrics["active_ready_survival_steps"] = (
            self.active_ready_survival_steps.float()
        )
        self.metrics["active_ready_survival_next_milestone"] = (
            self.active_ready_survival_next_milestone.float()
        )
        self.metrics["active_ready_survival_milestone_event"] = (
            self.active_ready_survival_milestone_event
        )
        self.metrics["healthy_training_stage"] = torch.full_like(
            self.metrics["healthy_training_stage"], float(self.healthy_training_stage())
        )
        self.metrics["healthy_stage_planted_rehearsal"] = (
            self.healthy_stage_planted_rehearsal.float()
        )
        self.metrics["healthy_stage_station_hold_seen"] = (
            self.healthy_stage_station_hold_seen.float()
        )
        self.metrics["healthy_stage_station_arrived"] = (
            self.healthy_stage_station_arrived.float()
        )
        self.metrics["healthy_stage_station_settled"] = (
            self.healthy_stage_station_settled.float()
        )
        self.metrics["healthy_stage_station_settle_steps"] = (
            self.healthy_stage_station_settle_steps.float()
        )

    def _update_healthy_stage_station_diagnostics(self) -> None:
        """Latch lateral READY arrival and sustained settle events."""
        relocation_enabled = bool(self.cfg.station_relocation_enabled)
        if not bool(self.cfg.healthy_three_stage_enabled) and not relocation_enabled:
            return
        self.station_relocation_arrival_event.zero_()
        self.station_relocation_settle_event.zero_()
        active = self.no_command_ready_active
        if relocation_enabled:
            active = self.station_relocation_active_mask()
        self.healthy_stage_station_hold_seen |= active
        station_error = torch.norm(self.base_pos_w[:, :2] - self.station_w, dim=-1)
        arrival_radius = (
            float(self.cfg.station_relocation_arrival_radius)
            if relocation_enabled
            else float(self.cfg.healthy_stage_station_arrival_radius)
        )
        previous_arrived = self.healthy_stage_station_arrived.clone()
        arrived = active & (
            station_error <= arrival_radius
        )
        self.healthy_stage_station_arrived |= arrived
        self.station_relocation_arrival_event |= arrived & (~previous_arrived)
        if relocation_enabled:
            level = int(self._station_relocation_level.item())
            lin_by_level = tuple(
                self.cfg.station_relocation_settle_max_lin_vel_by_level
            )
            ang_by_level = tuple(
                self.cfg.station_relocation_settle_max_ang_vel_by_level
            )
            steps_by_level = tuple(
                self.cfg.station_relocation_settle_min_steps_by_level
            )
            max_lin_vel = (
                float(lin_by_level[min(level, len(lin_by_level) - 1)])
                if lin_by_level
                else float(self.cfg.station_relocation_settle_max_lin_vel)
            )
            max_ang_vel = (
                float(ang_by_level[min(level, len(ang_by_level) - 1)])
                if ang_by_level
                else float(self.cfg.station_relocation_settle_max_ang_vel)
            )
            min_steps = (
                int(steps_by_level[min(level, len(steps_by_level) - 1)])
                if steps_by_level
                else int(self.cfg.station_relocation_settle_min_steps)
            )
        else:
            max_lin_vel = float(self.cfg.healthy_stage_station_settle_max_lin_vel)
            max_ang_vel = float(self.cfg.healthy_stage_station_settle_max_ang_vel)
            min_steps = int(self.cfg.healthy_stage_station_settle_min_steps)
        feet_ok = torch.ones_like(active)
        if relocation_enabled:
            feet_ok = (
                self.feet_contact_frac
                >= float(self.cfg.station_relocation_min_feet_contact)
            )
        stable = (
            arrived
            & self.impact_health_ok
            & feet_ok
            & (
                torch.norm(self.robot.data.root_lin_vel_w[:, :2], dim=-1)
                <= max_lin_vel
            )
            & (
                torch.norm(self.robot.data.root_ang_vel_w, dim=-1)
                <= max_ang_vel
            )
        )
        self.healthy_stage_station_settle_steps = torch.where(
            stable,
            self.healthy_stage_station_settle_steps + 1,
            torch.zeros_like(self.healthy_stage_station_settle_steps),
        )
        previous_settled = self.healthy_stage_station_settled.clone()
        self.healthy_stage_station_settled |= (
            self.healthy_stage_station_settle_steps >= min_steps
        )
        self.station_relocation_settle_event |= (
            self.healthy_stage_station_settled & (~previous_settled)
        )
        self.metrics["station_relocation_active"] = active.float()
        self.metrics["station_relocation_target_offset_y"] = (
            torch.where(
                active,
                self.station_relocation_station_w[:, 1]
                - self.fixed_station_w[:, 1],
                torch.zeros_like(station_error),
            )
        )
        self.metrics["station_relocation_error"] = torch.where(
            active, station_error, torch.zeros_like(station_error)
        )
        self.metrics["station_relocation_arrival_event"] = (
            self.station_relocation_arrival_event.float()
        )
        self.metrics["station_relocation_settle_event"] = (
            self.station_relocation_settle_event.float()
        )
        self.metrics["station_relocation_release_event"] = (
            self.station_relocation_release_event.float()
        )
        self.metrics["station_relocation_timeout_event"] = (
            self.station_relocation_timeout_event.float()
        )
        self.metrics["station_relocation_elapsed_steps"] = (
            self.station_relocation_elapsed_steps.float()
        )
        self.metrics["station_relocation_hold_deadline_steps"] = (
            torch.full_like(
                station_error,
                float(self._station_relocation_hold_deadline()),
            )
        )
        self.metrics["lifecycle_recovery_hold_active"] = (
            self.lifecycle_recovery_hold_active.float()
        )
        self.metrics["lifecycle_recovery_release_success_event"] = (
            self.lifecycle_recovery_release_success_event.float()
        )
        self.metrics["lifecycle_recovery_release_fail_event"] = (
            self.lifecycle_recovery_release_fail_event.float()
        )
        self.metrics["station_relocation_level"] = torch.full_like(
            self.metrics["station_relocation_level"],
            0.0,
        ) + self._station_relocation_level.float()
        self.metrics["station_relocation_arrival_ema"] = torch.full_like(
            self.metrics["station_relocation_arrival_ema"],
            0.0,
        ) + self._station_relocation_arrival_ema
        self.metrics["station_relocation_settle_ema"] = torch.full_like(
            self.metrics["station_relocation_settle_ema"],
            0.0,
        ) + self._station_relocation_settle_ema
        self.metrics["station_relocation_contact_ema"] = torch.full_like(
            self.metrics["station_relocation_contact_ema"],
            0.0,
        ) + self._station_relocation_contact_ema
        self.metrics["station_relocation_safety_ema"] = torch.full_like(
            self.metrics["station_relocation_safety_ema"],
            0.0,
        ) + self._station_relocation_safety_ema
        self.metrics["station_relocation_resolved_events"] = torch.full_like(
            self.metrics["station_relocation_resolved_events"],
            0.0,
        ) + self._station_relocation_resolved_events.float()
        self.metrics["station_relocation_terminal_reset_events"] = torch.full_like(
            self.metrics["station_relocation_terminal_reset_events"],
            0.0,
        ) + self._station_relocation_terminal_reset_events.float()
        self.metrics["station_relocation_advance_streak"] = torch.full_like(
            self.metrics["station_relocation_advance_streak"],
            0.0,
        ) + self._station_relocation_advance_streak.float()
        self.metrics["station_relocation_regress_streak"] = torch.full_like(
            self.metrics["station_relocation_regress_streak"],
            0.0,
        ) + self._station_relocation_regress_streak.float()
        self.metrics["station_relocation_effective_max_lin_vel"] = torch.full_like(
            self.metrics["station_relocation_effective_max_lin_vel"], max_lin_vel
        )
        self.metrics["station_relocation_effective_max_ang_vel"] = torch.full_like(
            self.metrics["station_relocation_effective_max_ang_vel"], max_ang_vel
        )
        self.metrics["station_relocation_effective_min_settle_steps"] = torch.full_like(
            self.metrics["station_relocation_effective_min_settle_steps"],
            float(min_steps),
        )

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
            self.cycle_v2_ready_consecutive_steps.zero_()
            return

        pending = self.cycle_v2_pending.clone()
        if not bool(torch.any(pending)):
            self.cycle_v2_ready_ok.zero_()
            ready_score_key = (
                "recovery_functional_ready_score"
                if bool(self.cfg.cycle_v2_use_functional_ready)
                else "recovery_ready_score"
            )
            self.cycle_v2_ready_score.copy_(self.metrics[ready_score_key])
            self.cycle_v2_visible.zero_()
            self.cycle_v2_ready_score_ok.zero_()
            self.cycle_v2_height_ok.zero_()
            self.cycle_v2_upright_ok.zero_()
            self.cycle_v2_base_lin_vel_ok.zero_()
            self.cycle_v2_base_ang_vel_ok.zero_()
            self.cycle_v2_feet_ok.zero_()
            self.cycle_v2_core_ready_ok.zero_()
            self.cycle_v2_ready_consecutive_steps.zero_()
            empty = torch.zeros_like(self.cycle_v2_pending)
            self._update_cycle_v2_ability_events(empty, empty)
            return

        visible = pending & self.pre_strike
        if not bool(self.cfg.cycle_v2_count_no_command_as_visible):
            visible = visible & (~self.no_command_ready_active)
        elapsed = pending & self.pre_strike
        self.cycle_v2_elapsed_steps[elapsed] += 1
        self.cycle_v2_command_visible_steps[visible] += 1

        ready_score_key = (
            "recovery_functional_ready_score"
            if bool(self.cfg.cycle_v2_use_functional_ready)
            else "recovery_ready_score"
        )
        ready_score = self.metrics[ready_score_key]
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
        self.cycle_v2_ready_consecutive_steps.copy_(
            update_consecutive_steps(
                self.cycle_v2_ready_consecutive_steps,
                pending & visible & ready_ok,
            )
        )
        sustained_ready = ready_ok & (
            self.cycle_v2_ready_consecutive_steps
            >= int(self.cfg.cycle_v2_required_consecutive_ready_steps)
        )
        self.cycle_v2_ready_ok.copy_(sustained_ready)

        success_event = pending & sustained_ready & (~self.cycle_v2_ready_success_latch)
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

    def _update_single_cycle_state(self) -> None:
        """Set a clean timeout only after first-wrap READY has settled."""
        self.single_cycle_safe_timeout_event.zero_()
        self.single_cycle_deadline_timeout_event.zero_()
        if not bool(self.cfg.single_cycle_curriculum_enabled):
            return

        active = (
            self.single_cycle_bootstrap
            & self.single_cycle_swing_completed
            & self.single_cycle_recovery_active
            & (~self.single_cycle_timeout_latch)
        )
        if not bool(torch.any(active)):
            return
        self.single_cycle_recovery_elapsed_steps[active] += 1

        ready_now = (
            active
            & (
                self.metrics["recovery_ready_score"]
                >= float(self.cfg.cycle_v2_ready_threshold)
            )
            & (
                self.metrics["recovery_height_error"]
                <= float(self.cfg.cycle_v2_max_height_error)
            )
            & (
                self.metrics["recovery_upright_error"]
                <= float(self.cfg.cycle_v2_max_upright_error)
            )
            & (
                self.metrics["recovery_base_lin_vel"]
                <= float(self.cfg.cycle_v2_max_base_lin_vel)
            )
            & (
                self.metrics["recovery_base_ang_vel"]
                <= float(self.cfg.cycle_v2_max_base_ang_vel)
            )
            & (
                self.metrics["recovery_feet_contact_frac"]
                >= float(self.cfg.cycle_v2_min_feet_contact)
            )
        )
        self.single_cycle_ready_consecutive_steps.copy_(
            update_consecutive_steps(
                self.single_cycle_ready_consecutive_steps,
                ready_now,
            )
        )

        attempted = self.cycle_v2_attempt_latch
        settlement_resolved = (
            (~attempted)
            | self.cycle_v2_ready_success_latch
            | self.cycle_v2_ready_fail_latch
        )
        minimum_elapsed = (
            self.single_cycle_recovery_elapsed_steps
            >= int(self.cfg.single_cycle_min_recovery_steps)
        )
        ready_sustained = (
            self.single_cycle_ready_consecutive_steps
            >= int(self.cfg.single_cycle_required_ready_steps)
        )
        safe = active & settlement_resolved & minimum_elapsed & ready_sustained
        # An attempted return remains blocked by ``settlement_resolved`` until
        # cycle-v2 escrow has paid or failed.  No-attempt bootstrap samples can
        # therefore use a short deadline without truncating impact credit.
        deadline = (
            active
            & settlement_resolved
            & (~safe)
            & (
                self.single_cycle_recovery_elapsed_steps
                >= self._single_cycle_effective_deadline_steps
            )
        )
        self.single_cycle_safe_timeout_event.copy_(safe)
        self.single_cycle_deadline_timeout_event.copy_(deadline)
        resolved = safe | deadline
        self.single_cycle_timeout_latch |= resolved
        self._single_cycle_safe_timeout_events += safe.long().sum()
        self._single_cycle_deadline_timeout_events += deadline.long().sum()

    def _evaluate_return(self):
        """Simple no-spin outgoing-ball evaluation at the exact strike frame (contact/net/bounce).

        The paddle is assumed to carry the ball off at the achieved racket velocity (no spin). The
        outgoing flight is a gravity-only ballistic arc; net clearance and the first table bounce are
        solved in closed form. All quantities are example approximations for training shaping.
        """
        exact = self.true_time_to_strike.abs() <= (0.5 * self._env.step_dt + 1e-6)
        self.strike_fired = exact
        self.current_swing_opportunity |= exact

        pos_err = torch.norm(self.racket_pos_w - self.ball_strike_pos_w, dim=-1)
        self.racket_target_distance = pos_err
        face_normal = self.racket_normal_w / torch.norm(
            self.racket_normal_w, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        ball_center_offset = self.ball_strike_pos_w - self.racket_pos_w
        signed_normal_gap = torch.sum(
            ball_center_offset * face_normal, dim=-1
        )
        tangent_offset = (
            ball_center_offset
            - signed_normal_gap.unsqueeze(-1) * face_normal
        )
        self.impact_face_radial_error = torch.norm(
            tangent_offset, dim=-1
        )
        self.impact_face_normal_gap = signed_normal_gap.abs()
        self.impact_face_quality = face_center_quality(
            self.impact_face_radial_error,
            inner_radius=float(self.cfg.face_quality_inner_radius),
            outer_radius=float(self.cfg.face_quality_outer_radius),
        )
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
        targeted_attempt = (
            attempt
            & (pos_err <= float(self.cfg.targeted_attempt_radius))
            & (approach >= float(self.cfg.targeted_attempt_min_approach_speed))
        )
        self.current_swing_targeted_attempt |= targeted_attempt
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

        self._update_post_contact_ready_hit_retention(
            exact,
            targeted_attempt,
            on_opponent,
        )
        self.ball_contact = contact
        self.ball_net_cross = net_cross
        self.ball_on_opponent = on_opponent
        center_zone, rim_zone, analytic_outer_zone = (
            face_contact_region_masks(
                self.impact_face_radial_error,
                inner_radius=float(self.cfg.face_quality_inner_radius),
                outer_radius=float(self.cfg.face_quality_outer_radius),
                contact_radius=float(self.cfg.contact_radius),
            )
        )
        impact_region_events = {
            "center": (
                contact & center_zone,
                net_cross & center_zone,
                on_opponent & center_zone,
            ),
            "rim": (
                contact & rim_zone,
                net_cross & rim_zone,
                on_opponent & rim_zone,
            ),
            "analytic_outer": (
                contact & analytic_outer_zone,
                net_cross & analytic_outer_zone,
                on_opponent & analytic_outer_zone,
            ),
        }
        for region, (
            region_contact,
            region_net,
            region_opponent,
        ) in impact_region_events.items():
            self.metrics[f"impact_{region}_contact"] = region_contact.float()
            self.metrics[f"impact_{region}_net_cross"] = region_net.float()
            self.metrics[f"impact_{region}_on_opponent"] = (
                region_opponent.float()
            )
            getattr(self, f"current_swing_{region}_contact")[:] |= (
                region_contact
            )
            getattr(self, f"current_swing_{region}_net_cross")[:] |= (
                region_net
            )
            getattr(self, f"current_swing_{region}_on_opponent")[:] |= (
                region_opponent
            )
        self.current_swing_contact[:] = self.current_swing_contact | contact
        self.current_swing_net_cross[:] = self.current_swing_net_cross | net_cross
        self.current_swing_on_opponent[:] = self.current_swing_on_opponent | on_opponent
        self.metrics["current_swing_contact"] = (
            self.current_swing_contact.float()
        )
        self.metrics["current_swing_net_cross"] = (
            self.current_swing_net_cross.float()
        )
        self.metrics["current_swing_on_opponent"] = (
            self.current_swing_on_opponent.float()
        )
        self.current_swing_impact_health_score[:] = torch.where(
            contact,
            self.impact_health_score,
            self.current_swing_impact_health_score,
        )
        healthy_contact = contact & self.impact_health_ok
        healthy_net_cross = net_cross & self.impact_health_ok
        healthy_on_opponent = on_opponent & self.impact_health_ok
        self.current_swing_healthy_contact |= healthy_contact
        self.current_swing_healthy_net_cross |= healthy_net_cross
        self.current_swing_healthy_on_opponent |= healthy_on_opponent
        for region in ("center", "rim", "analytic_outer"):
            self.metrics[f"current_swing_{region}_contact"] = getattr(
                self, f"current_swing_{region}_contact"
            ).float()
            self.metrics[f"current_swing_{region}_net_cross"] = getattr(
                self, f"current_swing_{region}_net_cross"
            ).float()
            self.metrics[f"current_swing_{region}_on_opponent"] = getattr(
                self, f"current_swing_{region}_on_opponent"
            ).float()
        actual_normal = self.racket_normal_w / torch.norm(
            self.racket_normal_w, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        target_normal = self.racket_target_normal_w / torch.norm(
            self.racket_target_normal_w, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        normal_error_deg = torch.rad2deg(
            torch.acos(torch.sum(actual_normal * target_normal, dim=-1).clamp(-1.0, 1.0))
        )
        actual_racket_speed = torch.norm(self.racket_lin_vel_w, dim=-1)
        target_racket_speed = torch.norm(self.racket_impact_target_vel_w, dim=-1).clamp_min(1.0e-6)
        racket_velocity_cosine = torch.sum(
            self.racket_lin_vel_w * self.racket_impact_target_vel_w, dim=-1
        ) / (actual_racket_speed.clamp_min(1.0e-6) * target_racket_speed)
        racket_velocity_angle_deg = torch.rad2deg(
            torch.acos(racket_velocity_cosine.clamp(-1.0, 1.0))
        )
        exact_impact_metrics = {
            "impact_racket_vel_error_mps": torch.norm(
                self.racket_lin_vel_w - self.racket_impact_target_vel_w, dim=-1
            ),
            "impact_racket_speed_ratio": actual_racket_speed / target_racket_speed,
            "impact_racket_vel_angle_deg": racket_velocity_angle_deg,
            "impact_racket_actual_vel_x": self.racket_lin_vel_w[:, 0],
            "impact_racket_actual_vel_y": self.racket_lin_vel_w[:, 1],
            "impact_racket_actual_vel_z": self.racket_lin_vel_w[:, 2],
            "impact_racket_target_vel_x": self.racket_impact_target_vel_w[:, 0],
            "impact_racket_target_vel_y": self.racket_impact_target_vel_w[:, 1],
            "impact_racket_target_vel_z": self.racket_impact_target_vel_w[:, 2],
            "impact_ball_out_error_mps": self.impact_ball_out_error,
            "impact_face_radial_error_m": self.impact_face_radial_error,
            "impact_face_normal_gap_m": self.impact_face_normal_gap,
            "impact_face_quality": self.impact_face_quality,
        }
        planner_target_speed = torch.norm(self.racket_target_vel_w, dim=-1).clamp_min(1.0e-6)
        planner_velocity_cosine = torch.sum(
            self.racket_lin_vel_w * self.racket_target_vel_w, dim=-1
        ) / (actual_racket_speed.clamp_min(1.0e-6) * planner_target_speed)
        exact_impact_metrics.update(
            {
                "impact_planner_pos_error_m": torch.norm(
                    self.racket_pos_w - self.racket_target_pos_w, dim=-1
                ),
                "impact_planner_vel_error_mps": torch.norm(
                    self.racket_lin_vel_w - self.racket_target_vel_w, dim=-1
                ),
                "impact_planner_speed_ratio": actual_racket_speed / planner_target_speed,
                "impact_planner_vel_angle_deg": torch.rad2deg(
                    torch.acos(planner_velocity_cosine.clamp(-1.0, 1.0))
                ),
            }
        )
        for key, value in exact_impact_metrics.items():
            self.metrics[key] = torch.where(exact, value, self.metrics[key])
        healthy_normal_ok = (
            exact
            & self.impact_health_ok
            & (normal_error_deg <= float(self.cfg.healthy_stage_normal_error_max_deg))
        )
        self.current_swing_healthy_normal_ok |= healthy_normal_ok
        self.metrics["racket_normal_error_deg"] = normal_error_deg
        self.metrics["impact_normal_error_deg"] = torch.where(
            exact, normal_error_deg, self.metrics["impact_normal_error_deg"]
        )
        self.metrics["impact_healthy_normal_ok"] = healthy_normal_ok.float()

    def _update_metrics(self):
        # Timing + FK must be fresh before the reward reads them (motion updated first this step).
        self._compute_strike_timing()
        self._compute_racket_state()
        self._update_feet_contact()
        self._compute_impact_health()
        self._evaluate_return()
        self._update_recovery_diagnostics(self._motion())
        self._update_post_contact_ready_state()
        settlement_components = (
            recovered_planner_velocity_settlement_components(
                speed_ratio=self.post_contact_ready_planner_speed_ratio,
                direction_error_rad=(
                    self.post_contact_ready_planner_direction_error_rad
                ),
                position_error=self.post_contact_ready_planner_position_error,
                impact_health_score=(
                    self.post_contact_ready_impact_health_score
                ),
                recovery_peak_base_ang_vel=(
                    self.post_contact_ready_peak_base_ang_vel
                ),
            )
        )
        settlement_event = self.post_contact_ready_success_event
        if bool(torch.any(settlement_event)):
            observed = torch.stack(
                [
                    component[settlement_event].mean()
                    for component in settlement_components
                ]
            )
            if not self._post_contact_ready_settlement_ema_initialized:
                self._post_contact_ready_settlement_component_ema.copy_(
                    observed
                )
                self._post_contact_ready_settlement_ema_initialized = True
            else:
                self._post_contact_ready_settlement_component_ema.lerp_(
                    observed,
                    0.05,
                )
        for index, name in enumerate(
            (
                "post_contact_ready_settlement_velocity_score_ema",
                "post_contact_ready_settlement_position_gate_ema",
                "post_contact_ready_settlement_health_gate_ema",
                "post_contact_ready_settlement_recovery_gate_ema",
                "post_contact_ready_settlement_total_score_ema",
            )
        ):
            self.metrics[name] = (
                self._post_contact_ready_settlement_component_ema[index]
                .view(1)
                .repeat(self.num_envs)
            )
        unsafe = (
            self.robot.data.root_pos_w[:, 2]
            < float(self.cfg.ability_curriculum_safe_min_base_height)
        ) | (
            torch.norm(self.robot.data.projected_gravity_b[:, :2], dim=-1)
            > float(self.cfg.ability_curriculum_safe_max_tilt)
        )
        self.current_swing_unsafe |= unsafe
        closed_cycle_success = closed_cycle_success_event(
            recovery_success=self.post_contact_ready_success_event,
            outcome_tier=self.post_contact_ready_outcome_tier,
            healthy_net_cross=self.current_swing_healthy_net_cross,
            unsafe=self.current_swing_unsafe,
            minimum_outcome_tier=1,
        )
        self.metrics["closed_loop_cycle_success_event"] = (
            closed_cycle_success.float()
        )
        durable_cycle_success = closed_cycle_success_event(
            recovery_success=(
                self.post_contact_ready_durable_success_event
            ),
            outcome_tier=self.post_contact_ready_outcome_tier,
            healthy_net_cross=self.current_swing_healthy_net_cross,
            unsafe=self.current_swing_unsafe,
            minimum_outcome_tier=1,
        )
        self.metrics["closed_loop_durable_cycle_success_event"] = (
            durable_cycle_success.float()
        )
        phase_ids = lifecycle_phase_ids(
            no_command=self.no_command_ready_active,
            time_to_strike=self.true_time_to_strike,
            strike_window=self.strike_window,
            recovery_pending=self.post_contact_ready_pending,
            recovery_success=self.post_contact_ready_success_event,
        )
        phase_one_hot = one_hot_lifecycle(phase_ids)
        self.metrics["closed_loop_phase_id"] = phase_ids.float()
        phase_metric_names = (
            "closed_loop_phase_ready_no_command",
            "closed_loop_phase_command_acquire",
            "closed_loop_phase_pre_strike",
            "closed_loop_phase_strike",
            "closed_loop_phase_follow_through",
            "closed_loop_phase_recovery",
            "closed_loop_phase_next_ready",
        )
        for phase_index, metric_name in enumerate(phase_metric_names):
            self.metrics[metric_name] = phase_one_hot[:, phase_index]
        self._update_healthy_stage_station_diagnostics()
        self._update_cycle_v2_state()
        self._update_single_cycle_state()
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
        self.metrics["ability_curriculum_resolved_events"] = torch.full_like(
            self.metrics["ability_curriculum_resolved_events"],
            float(self._ability_curriculum_resolved_events.item()),
        )
        self.metrics["ability_curriculum_total_resolved_events"] = torch.full_like(
            self.metrics["ability_curriculum_total_resolved_events"],
            float(self._ability_curriculum_total_resolved_events.item()),
        )
        self.metrics["ability_curriculum_advance_streak"] = torch.full_like(
            self.metrics["ability_curriculum_advance_streak"],
            float(self._ability_curriculum_advance_streak.item()),
        )
        self.metrics["ability_curriculum_regress_streak"] = torch.full_like(
            self.metrics["ability_curriculum_regress_streak"],
            float(self._ability_curriculum_regress_streak.item()),
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
        self.metrics["ability_targeted_attempt_ema"] = torch.full_like(
            self.metrics["ability_targeted_attempt_ema"],
            float(self._ability_targeted_attempt_ema.item()),
        )
        self.metrics["impact_health_floor_progress"] = torch.full_like(
            self.metrics["impact_health_floor_progress"],
            self.impact_health_floor_progress(),
        )
        self.metrics["ability_safety_ema"] = torch.full_like(
            self.metrics["ability_safety_ema"], float(self._ability_safety_ema.item())
        )
        self.metrics["ability_station_saturation_ema"] = torch.full_like(
            self.metrics["ability_station_saturation_ema"],
            float(self._ability_station_saturation_ema.item()),
        )
        self.metrics["ability_healthy_normal_ema"] = torch.full_like(
            self.metrics["ability_healthy_normal_ema"],
            float(self._ability_healthy_normal_ema.item()),
        )
        self.metrics["ability_station_arrival_ema"] = torch.full_like(
            self.metrics["ability_station_arrival_ema"],
            float(self._ability_station_arrival_ema.item()),
        )
        self.metrics["ability_station_settle_ema"] = torch.full_like(
            self.metrics["ability_station_settle_ema"],
            float(self._ability_station_settle_ema.item()),
        )
        self.metrics["safe_outcome_capability_gate"] = torch.full_like(
            self.metrics["safe_outcome_capability_gate"],
            float(self.safe_outcome_capability_gate().item()),
        )
        self.metrics["current_swing_opportunity"] = (
            self.current_swing_opportunity.float()
        )
        self.metrics["current_swing_attempt"] = self.current_swing_attempt.float()
        self.metrics["current_swing_targeted_attempt"] = (
            self.current_swing_targeted_attempt.float()
        )
        self.metrics["prev_swing_targeted_attempt"] = (
            self.prev_swing_targeted_attempt.float()
        )
        self.metrics["current_swing_unsafe"] = self.current_swing_unsafe.float()
        self.metrics["current_swing_station_saturated"] = (
            self.current_swing_station_saturated.float()
        )
        self.metrics["impact_health_score"] = self.impact_health_score
        self.metrics["impact_health_ok"] = self.impact_health_ok.float()
        self.metrics["impact_healthy_contact"] = (
            self.ball_contact & self.impact_health_ok
        ).float()
        self.metrics["impact_healthy_net_cross"] = (
            self.ball_net_cross & self.impact_health_ok
        ).float()
        self.metrics["impact_healthy_on_opponent"] = (
            self.ball_on_opponent & self.impact_health_ok
        ).float()
        self.metrics["current_swing_impact_health_score"] = (
            self.current_swing_impact_health_score
        )
        self.metrics["current_swing_healthy_contact"] = (
            self.current_swing_healthy_contact.float()
        )
        self.metrics["current_swing_healthy_net_cross"] = (
            self.current_swing_healthy_net_cross.float()
        )
        self.metrics["current_swing_healthy_on_opponent"] = (
            self.current_swing_healthy_on_opponent.float()
        )
        self.metrics["current_swing_healthy_normal_ok"] = (
            self.current_swing_healthy_normal_ok.float()
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
        self.metrics["cycle_v2_ready_consecutive_steps"] = (
            self.cycle_v2_ready_consecutive_steps.float()
        )
        self.metrics["cycle_v2_outcome_tier"] = self.cycle_v2_outcome_tier.float()
        self.metrics["cycle_v2_streak"] = self.cycle_v2_streak.float()
        self.metrics["single_cycle_bootstrap"] = (
            self.single_cycle_bootstrap.float()
        )
        self.metrics["single_cycle_swing_completed"] = (
            self.single_cycle_swing_completed.float()
        )
        self.metrics["single_cycle_recovery_active"] = (
            self.single_cycle_recovery_active.float()
        )
        self.metrics["single_cycle_recovery_elapsed_steps"] = (
            self.single_cycle_recovery_elapsed_steps.float()
        )
        self.metrics["single_cycle_ready_consecutive_steps"] = (
            self.single_cycle_ready_consecutive_steps.float()
        )
        self.metrics["single_cycle_safe_timeout_event"] = (
            self.single_cycle_safe_timeout_event.float()
        )
        self.metrics["single_cycle_deadline_timeout_event"] = (
            self.single_cycle_deadline_timeout_event.float()
        )
        self.metrics["single_cycle_timeout_latch"] = (
            self.single_cycle_timeout_latch.float()
        )
        self.metrics["single_cycle_hard_failure_event"] = (
            self.single_cycle_hard_failure_event.float()
        )
        for metric_name, value in (
            ("single_cycle_probability", self._single_cycle_probability),
            ("single_cycle_curriculum_level", self._single_cycle_curriculum_level),
            (
                "single_cycle_effective_deadline_steps",
                self._single_cycle_effective_deadline_steps,
            ),
            ("single_cycle_selected_events", self._single_cycle_selected_events),
            ("single_cycle_completed_events", self._single_cycle_completed_events),
            ("single_cycle_safe_timeout_events", self._single_cycle_safe_timeout_events),
            ("single_cycle_deadline_timeout_events", self._single_cycle_deadline_timeout_events),
            ("single_cycle_hard_failure_events", self._single_cycle_hard_failure_events),
        ):
            self.metrics[metric_name] = torch.full_like(
                self.metrics[metric_name], float(value.item())
            )

    def prepare_reward_snapshot(self) -> None:
        """Publish one post-physics target/FK/lifecycle snapshot for reward."""
        if self._reward_snapshot_prepared:
            return
        self._update_metrics()
        self._reward_snapshot_prepared = True

    def refresh_kinematic_snapshot(self) -> None:
        """Refresh reset-state tensors without advancing lifecycle counters."""
        self._compute_strike_timing()
        self._compute_racket_state()
        self._update_feet_contact()
        self._compute_impact_health()

    def compute(self, dt: float) -> None:
        """Avoid a second lifecycle update after a pre-reward snapshot."""
        if self._reward_snapshot_prepared:
            self._reward_snapshot_prepared = False
        else:
            self._update_metrics()
        self.time_left -= dt
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(resample_env_ids)
        self._update_command()

    def _update_command(self):
        self.cycle_v2_attempt_event.zero_()
        self.cycle_v2_unresolved_resample_fail_event.zero_()
        self.steps_since_target_resample += 1
        self.target_just_resampled.zero_()
        # ``_compute_strike_timing`` is also called by ``_update_metrics``.
        # Advance a physical override only here so one 20 ms policy step cannot
        # consume 40 ms of time-to-strike.
        active_override = self.physical_command_override_active
        if bool(torch.any(active_override)):
            self.physical_command_override_tts[active_override] -= float(
                self._env.step_dt
            )
        self._compute_strike_timing()
        # Re-sample the target at each new swing (the motion command sets just_resampled this step
        # when it wrapped a swing). Reset-time resampling is handled by the manager's reset -> _resample.
        motion = self._motion()
        wrapped = torch.where(motion.just_resampled)[0]
        if len(wrapped) > 0:
            self._resample_command(wrapped, carry_previous=True)
        if (
            bool(self.cfg.lifecycle_recovery_hold_gate_enabled)
            or bool(self.cfg.station_relocation_apply_to_swing_holds)
            or bool(self.cfg.single_cycle_curriculum_enabled)
        ):
            self._update_lifecycle_hold_gate()

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
    deploy_ready_force_stand_episode: bool = False
    deploy_ready_force_default_stand_reset: bool = False
    deploy_ready_reach: tuple[float, float, float] = (0.40, 0.20, -0.05)
    deploy_ready_velocity_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    deploy_ready_normal_w: tuple[float, float, float] = (1.0, 0.0, 0.0)
    deploy_ready_time_to_strike: float = 1.0
    # Mixed lateral relocation drill. A sampled no-command hold exposes a
    # non-zero current station while retaining the ready racket command; the
    # original strike command is restored after the hold. Difficulty advances
    # only after arrival, braking, contact retention, and safety all pass.
    station_relocation_enabled: bool = False
    station_relocation_rehearsal_prob: float = 0.0
    station_relocation_abs_y_ranges: tuple = (
        (0.04, 0.06),
        (0.06, 0.10),
        (0.10, 0.16),
    )
    station_relocation_arrival_radius: float = 0.03
    station_relocation_settle_max_lin_vel: float = 0.14
    station_relocation_settle_max_ang_vel: float = 0.55
    station_relocation_settle_min_steps: int = 12
    station_relocation_settle_max_lin_vel_by_level: tuple = ()
    station_relocation_settle_max_ang_vel_by_level: tuple = ()
    station_relocation_settle_min_steps_by_level: tuple = ()
    station_relocation_min_feet_contact: float = 0.90
    station_relocation_ema_rate: float = 0.05
    station_relocation_arrival_threshold: float = 0.85
    station_relocation_arrival_floor: float = 0.60
    station_relocation_settle_threshold: float = 0.72
    station_relocation_settle_floor: float = 0.25
    station_relocation_contact_threshold: float = 0.55
    station_relocation_contact_floor: float = 0.42
    station_relocation_safety_threshold: float = 0.995
    station_relocation_safety_floor: float = 0.985
    station_relocation_min_resolved_events: int = 12288
    station_relocation_required_advance_checks: int = 6
    station_relocation_required_regress_checks: int = 6
    # Optional phase gate used by the closed-loop-v3 curriculum. Normal swing
    # holds expose the dynamic station, stay frozen until settle/timeout, then
    # reveal the strike command. Motion-seeded starts still bypass this gate so
    # a scratch policy keeps seeing strike opportunities.
    station_relocation_apply_to_swing_holds: bool = False
    station_relocation_use_dynamic_target: bool = False
    station_relocation_hold_deadline_steps_by_level: tuple[int, ...] = ()
    lifecycle_recovery_hold_gate_enabled: bool = False

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
    # -1 follows the ability curriculum (or the legacy full range when it is
    # disabled). A fixed value isolates one workspace stage for validation.
    table_workspace_fixed_level: float = -1.0
    # ``station_relocation`` decouples lateral workspace expansion from the
    # incoming-ball/strike curriculum.
    table_workspace_level_source: str = "ability"  # ability | station_relocation
    table_workspace_curriculum_start_level: float = 0.0
    table_workspace_curriculum_full_level: float = 1.0
    # Scratch-only contact seeding. At zero ability, blend sampled y/z toward
    # the selected motion's strike box. The blend reaches zero at end_level,
    # after which ball routes are fully independent of motion.
    table_workspace_motion_seed_blend_start: float = 0.0
    table_workspace_motion_seed_end_level: float = 0.0
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
    # A physical observer can own level updates while this command continues
    # to use the shared level to scale workspace, ball, and planner noise.
    ability_curriculum_external_update: bool = False
    ability_curriculum_start_racket_pos_scale: tuple[float, float, float] | float = (1.0, 1.0, 1.0)
    ability_curriculum_start_ball_scale: float = 1.0
    ability_curriculum_start_planner_perturb_scale: float = 1.0
    planner_perturb_curriculum_source: str = "ability"  # ability | fixed
    planner_perturb_fixed_scale: float = 1.0
    planner_perturb_curriculum_start_level: float = 0.0
    planner_perturb_curriculum_full_level: float = 1.0
    ability_curriculum_ema_rate: float = 0.02
    ability_curriculum_advance_rate: float = 0.015
    ability_curriculum_regress_rate: float = 0.03
    # A decision consumes one disjoint batch of completed swings. Defaults
    # preserve the historical one-check-per-wrap behavior.
    ability_curriculum_min_resolved_events: int = 1
    ability_curriculum_required_advance_checks: int = 1
    ability_curriculum_required_regress_checks: int = 1
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
    ability_curriculum_use_targeted_attempt: bool = False
    ability_curriculum_safety_threshold: float = 0.995
    ability_curriculum_safety_floor: float = 0.985
    ability_curriculum_station_saturation_threshold: float = 0.08
    ability_curriculum_station_saturation_regress_threshold: float = 0.15
    ability_curriculum_safe_min_base_height: float = 0.70
    ability_curriculum_safe_max_tilt: float = 0.65
    # When enabled, curriculum outcome EMAs count only impacts made while the
    # trunk and COM remain inside the deployable support envelope.
    ability_curriculum_require_healthy_impact: bool = False

    # Dense strike shaping starts permissive so a scratch policy can discover
    # contact, then becomes strictly health-conditioned as strike capability is
    # measured. ``ability`` preserves the historical global-curriculum coupling.
    impact_health_floor_progress_source: str = "ability"
    impact_health_floor_targeted_attempt_threshold: float = 0.05
    impact_health_floor_contact_threshold: float = 0.01
    # Exponent applied to strike health before it gates task rewards. One keeps
    # historical linear behavior; values above one sharply discount marginal
    # postures without imposing a direct arm or leg constraint.
    impact_health_reward_power: float = 1.0

    # Optional capability gate for delayed net/bounce value. The contact tier
    # remains available below the gate; only higher outcome tiers are locked.
    # Thresholds are task-specific and must be set from a frozen-policy audit.
    safe_outcome_capability_gate_enabled: bool = False
    safe_outcome_gate_contact_low: float = 0.45
    safe_outcome_gate_contact_high: float = 0.60
    safe_outcome_gate_forehand_low: float = 0.45
    safe_outcome_gate_forehand_high: float = 0.60
    safe_outcome_gate_backhand_low: float = 0.40
    safe_outcome_gate_backhand_high: float = 0.55
    safe_outcome_gate_safety_low: float = 0.90
    safe_outcome_gate_safety_high: float = 0.95
    safe_outcome_gate_recovery_low: float = 0.70
    safe_outcome_gate_recovery_high: float = 0.82
    safe_outcome_gate_settlement_low: float = 0.55
    safe_outcome_gate_settlement_high: float = 0.70

    # Ability-gated healthy task structure. Stage 1 learns planted, upright
    # forehand/backhand strikes in the audited core workspace. Stage 2 adds a
    # lateral no-command READY hold before the strike. Stage 3 combines those
    # skills and expands the table workspace. No transition depends on iteration.
    healthy_three_stage_enabled: bool = False
    healthy_stage2_start_level: float = 0.20
    healthy_stage3_start_level: float = 0.60
    healthy_stage1_ready_hold_prob: float = 0.20
    healthy_stage2_ready_hold_prob: float = 0.50
    healthy_stage3_ready_hold_prob: float = 0.20
    healthy_stage3_planted_rehearsal_prob: float = 0.20
    healthy_stage_use_dynamic_station_during_ready: bool = True
    healthy_stage_lateral_station_bias: float = 0.10
    healthy_stage2_station_y_clip: float = 0.10
    healthy_stage3_station_target_gain: float = 0.85
    healthy_stage_station_arrival_radius: float = 0.07
    healthy_stage_station_settle_max_lin_vel: float = 0.18
    healthy_stage_station_settle_max_ang_vel: float = 0.65
    healthy_stage_station_settle_min_steps: int = 12
    healthy_stage_side_contact_threshold: float = 0.40
    healthy_stage_normal_threshold: float = 0.35
    healthy_stage_safety_threshold: float = 0.995
    healthy_stage_recovery_threshold: float = 0.68
    healthy_stage_station_arrival_threshold: float = 0.75
    healthy_stage_station_settle_threshold: float = 0.65
    healthy_stage_normal_error_max_deg: float = 30.0

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
    cycle_v2_use_functional_ready: bool = False
    cycle_v2_required_consecutive_ready_steps: int = 1
    cycle_v2_ready_threshold: float = 0.72
    cycle_v2_max_height_error: float = 0.10
    cycle_v2_max_upright_error: float = 0.28
    cycle_v2_max_base_lin_vel: float = 0.25
    cycle_v2_max_base_ang_vel: float = 0.80
    cycle_v2_max_station_error: float = 0.24
    cycle_v2_min_feet_contact: float = 0.45
    cycle_v2_max_racket_speed: float = 0.80
    cycle_v2_min_arm_score: float = 0.55
    cycle_v2_require_healthy_impact: bool = False
    # ``achieved`` stores the best actual contact/net/bounce tier for deferred
    # settlement instead of paying only the globally required tier.
    cycle_v2_settlement_tier_mode: str = "required"  # required | achieved

    # Ability-driven one-swing bootstrap. A bounded subset of true resets is
    # forced to an audited pre-strike frame, then cleanly times out only after
    # the first swing reaches a no-command READY settlement. Continuous
    # environments remain the source of task-difficulty promotion.
    single_cycle_curriculum_enabled: bool = False
    single_cycle_probabilities: tuple[float, ...] = (0.65, 0.45, 0.25, 0.10, 0.0)
    single_cycle_targeted_attempt_thresholds: tuple[float, ...] = (0.01, 0.03, 0.06, 0.10)
    single_cycle_contact_thresholds: tuple[float, ...] = (0.005, 0.015, 0.03, 0.05)
    single_cycle_recovery_thresholds: tuple[float, ...] = (0.35, 0.48, 0.60, 0.68)
    single_cycle_safety_thresholds: tuple[float, ...] = (0.90, 0.94, 0.96, 0.98)
    single_cycle_cycle_ready_thresholds: tuple[float, ...] = (0.0, 0.01, 0.03, 0.08)
    single_cycle_min_resolved_events: int = 4096
    single_cycle_min_continuous_fraction: float = 0.40
    single_cycle_required_ready_steps: int = 5
    single_cycle_min_recovery_steps: int = 10
    single_cycle_deadline_steps: int = 100
    single_cycle_deadline_steps_by_level: tuple[int, ...] = ()
    single_cycle_continuous_only_curriculum: bool = True

    # Functional READY is a stable region rather than an exact full-body pose. The configured
    # joints should contain only the upper-body forehand anchor; waist and legs remain available
    # for balance corrections.
    functional_ready_joint_names: tuple[str, ...] = ()
    functional_ready_joint_positions: tuple[float, ...] = ()
    functional_ready_joint_tolerances: tuple[float, ...] = ()
    functional_ready_joint_std: float = 0.30
    no_command_ready_progress_clip: float = 0.05
    active_ready_required_consecutive_steps: int = 15
    active_ready_score_threshold: float = 0.55
    active_ready_max_height_error: float = 0.11
    active_ready_max_upright_error: float = 0.30
    active_ready_max_base_lin_vel: float = 0.28
    active_ready_max_base_ang_vel: float = 0.90
    active_ready_min_feet_contact: float = 0.50
    active_ready_survival_milestones_enabled: bool = False
    active_ready_survival_require_soft_envelope: bool = True
    active_ready_survival_milestone_steps: tuple[int, ...] = (
        50,
        75,
        100,
        150,
        250,
        400,
    )
    active_ready_survival_milestone_values: tuple[float, ...] = (
        0.10,
        0.15,
        0.25,
        0.45,
        0.80,
        1.25,
    )
    active_ready_survival_max_height_error: float = 0.20
    active_ready_survival_max_upright_error: float = 0.55
    active_ready_survival_max_base_lin_vel: float = 1.00
    active_ready_survival_max_base_ang_vel: float = 1.80
    active_ready_survival_min_feet_contact: float = 0.50

    # Contact-local recovery settlement. Torso gravity-x is signed: backward
    # lean is negative, while a small positive value is a useful forward stance.
    post_contact_ready_enabled: bool = False
    # ``targeted_attempt`` also trains recovery after a miss. ``contact``
    # preserves the historical behavior for existing tasks.
    post_contact_ready_trigger: str = "contact"
    post_contact_ready_torso_x_min: float = -0.035
    post_contact_ready_torso_x_max: float = 0.14
    post_contact_ready_torso_x_std: float = 0.06
    # Legacy tasks shape only backward lean. New closed-loop tasks can use the
    # distance outside the full healthy torso interval so excessive forward
    # lean has a directional recovery signal as well.
    post_contact_ready_progress_error_mode: str = "backlean"
    post_contact_ready_max_torso_ang_vel: float = 0.60
    post_contact_ready_max_height_error: float = 0.08
    post_contact_ready_max_base_lin_vel: float = 0.18
    post_contact_ready_max_base_ang_vel: float = 0.55
    post_contact_ready_max_com_x: float = 0.11
    post_contact_ready_max_com_y: float = 0.13
    post_contact_ready_min_feet_contact: float = 0.99
    post_contact_ready_max_station_error: float = 0.22
    post_contact_ready_max_racket_speed: float = 0.90
    post_contact_ready_min_arm_score: float = 0.45
    post_contact_ready_required_consecutive_steps: int = 10
    post_contact_ready_deadline_steps: int = 70
    post_contact_ready_progress_clip: float = 0.035
    post_contact_ready_peak_excess_std: float = 0.60
    post_contact_ready_peak_excess_max_potential: float = 4.0
    post_contact_ready_envelope_max_tilt: float = 0.35
    post_contact_ready_envelope_max_abs_pitch: float = 0.25
    post_contact_ready_envelope_max_com_x: float = 0.16
    post_contact_ready_envelope_max_com_y: float = 0.18
    post_contact_ready_envelope_max_waist_overflow: float = 1.50
    post_contact_ready_envelope_max_leg_overflow: float = 2.50
    post_contact_ready_envelope_max_base_ang_vel: float = 2.00
    post_contact_ready_diagnostic_horizon_s: float = 1.80
    post_contact_ready_diagnostic_phase_boundaries_s: tuple[float, ...] = (
        0.10,
        0.30,
        0.60,
    )
    # Read-only validation gate: do not count an early READY crossing as a
    # reusable state unless it remains terminal-ready near a fixed deadline.
    post_contact_ready_durable_diagnostic_enabled: bool = False
    post_contact_ready_durable_min_delay_s: float = 0.60
    # Keep this before the nominal 1.18 s command lifecycle boundary so the
    # result cannot be censored by the next command resample.
    post_contact_ready_durable_deadline_s: float = 1.10
    post_contact_ready_durable_required_consecutive_steps: int = 15
    post_contact_ready_durable_use_effective_gate: bool = False
    # Operational deployment gate for deferred strike value. Unlike the
    # strict posture-quality diagnostic, it is symmetric and does not require
    # a specific signed lean pose.
    post_contact_ready_operational_max_tilt: float = 0.20
    post_contact_ready_operational_max_torso_ang_vel: float = 1.20
    post_contact_ready_operational_max_base_ang_vel: float = 1.20
    post_contact_ready_operational_max_height_error: float = 0.12
    post_contact_ready_operational_max_com_x: float = 0.16
    post_contact_ready_operational_max_com_y: float = 0.18
    post_contact_ready_operational_min_feet_contact: float = 0.50
    post_contact_ready_operational_max_station_error: float = 0.24
    # Monotonic ability curriculum for the hard settlement event. Advancement
    # depends only on resolved contact-local attempts, never on iteration.
    post_contact_ready_curriculum_enabled: bool = False
    post_contact_ready_curriculum_start_level: int = 0
    post_contact_ready_curriculum_torso_x_min: tuple[float, ...] = (
        -0.10,
        -0.075,
        -0.05,
        -0.035,
    )
    post_contact_ready_curriculum_torso_x_max: tuple[float, ...] = (
        0.38,
        0.30,
        0.22,
        0.14,
    )
    post_contact_ready_curriculum_max_torso_ang_vel: tuple[float, ...] = (
        0.90,
        0.78,
        0.68,
        0.60,
    )
    post_contact_ready_curriculum_max_base_lin_vel: tuple[float, ...] = (
        0.22,
        0.20,
        0.19,
        0.18,
    )
    post_contact_ready_curriculum_max_base_ang_vel: tuple[float, ...] = (
        1.00,
        0.82,
        0.68,
        0.55,
    )
    post_contact_ready_curriculum_max_racket_speed: tuple[float, ...] = (
        1.20,
        1.05,
        0.95,
        0.90,
    )
    post_contact_ready_curriculum_min_feet_contact: tuple[float, ...] = (
        0.90,
        0.95,
        0.98,
        0.99,
    )
    post_contact_ready_curriculum_required_consecutive_steps: tuple[int, ...] = (
        3,
        5,
        7,
        10,
    )
    post_contact_ready_curriculum_deadline_steps: tuple[int, ...] = (
        90,
        85,
        80,
        70,
    )
    post_contact_ready_curriculum_advance_success_thresholds: tuple[float, ...] = (
        0.30,
        0.40,
        0.50,
    )
    post_contact_ready_curriculum_min_resolved_events: tuple[int, ...] = (
        1024,
        2048,
        4096,
    )
    post_contact_ready_curriculum_shadow_success_thresholds: tuple[
        float, ...
    ] = (
        0.0,
        0.0,
        0.0,
    )
    post_contact_ready_curriculum_min_targeted_attempt_ema: float = 0.0
    post_contact_ready_curriculum_min_return_success_ema: float = 0.0
    post_contact_ready_curriculum_min_completed_swings: int = 1
    post_contact_ready_curriculum_required_advance_checks: int = 1
    post_contact_ready_curriculum_ema_rate: float = 0.05
    post_contact_ready_curriculum_hit_ema_rate: float = 0.05

    # Impact-health gate. Backward pitch is one-sided so a useful forward weight
    # transfer is not punished; arm joints are intentionally absent.
    impact_health_torso_body_name: str = "torso_Link"
    impact_health_backlean_tolerance: float = 0.05
    impact_health_backlean_std: float = 0.12
    impact_health_roll_std: float = 0.22
    impact_health_ang_vel_std: float = 1.40
    impact_health_com_x_std: float = 0.14
    impact_health_com_y_std: float = 0.18
    impact_health_backward_vel_std: float = 0.30
    impact_health_max_backlean: float = 0.20
    impact_health_max_roll: float = 0.28
    impact_health_max_ang_vel: float = 2.50
    impact_health_max_com_x: float = 0.20
    impact_health_max_com_y: float = 0.22
    impact_health_max_backward_vel: float = 0.35
    impact_health_min_feet_contact: float = 0.50
    impact_health_include_waist_retreat: bool = False
    impact_health_include_waist_backfold: bool = False
    impact_health_waist_pitch_joint_name: str = "waist_pitch_joint"
    impact_health_waist_backfold_tolerance: float = 0.05
    impact_health_waist_backfold_std: float = 0.12
    impact_health_max_waist_backfold: float = 0.20
    impact_health_base_retreat_std: float = 0.08
    impact_health_max_base_retreat: float = 0.10

    # --- no-spin return evaluation (example table placement in the env frame; tune to your scene) ---
    targeted_attempt_radius: float = 0.20
    targeted_attempt_min_approach_speed: float = 0.05
    contact_radius: float = 0.095   # racket radius + ball radius
    # Ball-center radial quality on the actual racket plane. The inner radius
    # keeps a full-value central plateau; the physical collision rim is zero.
    face_quality_inner_radius: float = 0.061
    face_quality_outer_radius: float = 0.081
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
    # Bootstrap from the bounded velocity distribution learned before adding
    # the rigid ball, then blend toward the impact-inverse planner command.
    impact_inverse_command_curriculum_enabled: bool = False
    impact_inverse_command_start_blend: float = 0.0
    impact_inverse_command_curriculum_exponent: float = 1.0
    impact_inverse_command_curriculum_start_level: float = 0.0
    impact_inverse_command_curriculum_full_level: float = 1.0
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
    # Optional ability-based incoming horizontal-speed curriculum. The bounce
    # distance is reconstructed from sampled speed and post-bounce time, so a
    # narrow early interval is genuinely slower rather than merely less random.
    one_bounce_speed_curriculum_enabled: bool = False
    one_bounce_speed_curriculum_start_level: float = 0.0
    one_bounce_speed_curriculum_full_level: float = 1.0
    one_bounce_easy_horizontal_speed_range: tuple = (0.90, 1.40)
    one_bounce_full_horizontal_speed_range: tuple = (0.80, 2.20)
    one_bounce_easy_post_time_range: tuple = (0.38, 0.44)
    # Planner perturbations affect the actor-visible command only. Rewards/evaluation use the
    # hidden true ball task so small planner errors train robustness instead of changing the task.
    planner_target_pos_offset_range: tuple = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    planner_time_to_strike_offset_range: tuple = (0.0, 0.0)
    planner_target_vel_scale_range: tuple = (1.0, 1.0)
    planner_target_vel_offset_range: tuple = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    planner_target_vel_yaw_deg_range: tuple = (0.0, 0.0)
    planner_command_mode: str = "legacy"  # legacy | v4_wire_compatible
    actuator_safety_overflow_threshold: float = 6.0
    actuator_safety_overflow_consecutive_steps: int = 5
    # Runtime-only equivalent of CommandTerm.reset() with one metric sync.
    batched_metric_reset_logging: bool = False
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
