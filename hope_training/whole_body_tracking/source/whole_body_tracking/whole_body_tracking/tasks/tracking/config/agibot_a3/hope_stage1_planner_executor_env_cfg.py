"""Ball-free Stage-1 foundation for stable planner-command execution."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_closed_loop_v2_env_cfg import (
    _zero_all_reward_terms,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_stage1_command_tracking_env_cfg import (
    STAGE1_ARM_PRIOR_BODIES,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_stage1_plane020_escrow_env_cfg import (
    HOPEStage1Plane020EscrowImpulseEnvCfg,
    Plane020EscrowRewardsCfg,
)


@configclass
class Stage1PlannerExecutorRewardsCfg(Plane020EscrowRewardsCfg):
    """One-shot command-cycle terms, separate from physical return outcomes."""

    recovery_peak_ang_vel_excess = RewTerm(
        func=mdp.closed_loop_v2_recovery_peak_ang_vel_excess,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "impulse": True,
            "ability_scaled": True,
            "ability_start_scale": 0.08,
            "ability_attempt_start": 0.20,
            "ability_attempt_full": 0.45,
        },
    )

    safe_recovered_planner_command = RewTerm(
        func=mdp.stage1_safe_recovered_planner_command,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "speed_ratio_std": 0.30,
            "direction_std_rad": 0.35,
            "normal_std_rad": 0.30,
            "component_floor": 0.02,
            "normal_floor": 0.05,
            "position_std": 0.12,
            "position_floor": 0.10,
            "impact_health_floor": 0.25,
            "recovery_peak_ang_vel_budget": 0.80,
            "recovery_peak_ang_vel_excess_std": 0.60,
            "recovery_gate_floor": 0.10,
            "require_contact": True,
            "impulse": True,
        },
    )
    command_cycle_failure = RewTerm(
        func=mdp.stage1_command_cycle_failure,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "unsafe_scale": 2.0,
            "incomplete_scale": 0.50,
            "impulse": True,
        },
    )


@configclass
class HOPEStage1PlannerExecutorEnvCfg(
    HOPEStage1Plane020EscrowImpulseEnvCfg
):
    """Execute coherent Planner commands and recover without a rigid ball."""

    rewards: Stage1PlannerExecutorRewardsCfg = (
        Stage1PlannerExecutorRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0

        motion = self.commands.motion
        motion.wrap_teleport = False
        motion.hold_steps_range = (16, 28)
        motion.core_clip_count = 2
        motion.clip_sampling_weights = (0.5, 0.5)
        motion.stand_start_prob = 0.20
        motion.stand_start_min_hold = 15
        motion.stand_episode_prob = 0.02
        motion.motion_start_warmup_enabled = True
        motion.motion_start_warmup_start_prob = 0.80
        motion.motion_start_warmup_min_prob = 0.80
        motion.motion_start_warmup_prestrike_enabled = False
        motion.motion_start_warmup_recovery_enabled = False
        motion.motion_start_warmup_lifecycle_curriculum_enabled = False

        command = self.commands.racket_target
        command.station_mode = "dynamic_from_motion"
        command.dynamic_station_xy_clip = ((-0.02, 0.05), (-0.42, 0.42))
        command.dynamic_station_blend = 1.0
        command.deploy_ready_hold_prob = 0.05
        command.deploy_ready_force_stand_episode = True
        command.deploy_ready_force_default_stand_reset = True

        # Keep one carried-state command lifecycle. Outcome tiers are virtual
        # diagnostics in this stage; no rigid-ball, net, or landing objective is
        # allowed to enter the reward.
        command.station_relocation_enabled = False
        command.lifecycle_recovery_hold_gate_enabled = False
        command.healthy_three_stage_enabled = False
        command.cycle_v2_enabled = False
        command.single_cycle_curriculum_enabled = False
        command.safe_outcome_capability_gate_enabled = False

        command.strike_position_mode = "table_workspace"
        command.planner_hit_plane_mode = "fixed_x_hit"
        command.planner_hit_plane_x = 0.20
        command.planner_hit_plane_x_jitter_range = (0.0, 0.0)
        command.planner_hit_plane_blend = 1.0
        command.table_workspace_forehand_core_y_range = (-0.53, -0.38)
        command.table_workspace_backhand_core_y_range = (-0.12, 0.00)
        command.table_workspace_x_jitter_core_range = (0.0, 0.0)
        command.table_workspace_x_jitter_full_range = (0.0, 0.0)
        command.table_workspace_z_core_above_surface_range = (0.44, 0.54)
        command.table_workspace_z_full_above_surface_range = (0.36, 0.64)
        command.table_workspace_fringe_prob = 0.0
        command.table_workspace_fixed_level = -1.0
        command.table_workspace_level_source = "ability"
        command.table_workspace_motion_seed_blend_start = 0.0
        command.table_workspace_motion_seed_end_level = 0.0

        # Generate a coherent ideal command with the same no-spin impact model
        # used by Stage 2. The early blend narrows command difficulty; it does
        # not add Planner noise or rigid-ball outcome credit.
        command.racket_velocity_mode = "impact_inverse_landing"
        command.planner_command_mode = "v4_wire_compatible"
        command.impact_inverse_command_curriculum_enabled = True
        command.impact_inverse_command_start_blend = 0.35
        command.impact_inverse_command_curriculum_exponent = 1.25
        command.impact_inverse_command_curriculum_start_level = 0.0
        command.impact_inverse_command_curriculum_full_level = 0.75
        command.incoming_trajectory_mode = "one_bounce"
        command.one_bounce_speed_curriculum_enabled = True
        command.one_bounce_speed_curriculum_start_level = 0.55
        command.one_bounce_speed_curriculum_full_level = 1.0
        command.one_bounce_easy_horizontal_speed_range = (0.90, 1.20)
        command.one_bounce_full_horizontal_speed_range = (0.80, 2.20)
        command.one_bounce_easy_post_time_range = (0.40, 0.46)
        command.strike_window_s = 0.12

        command.planner_target_pos_offset_range = (
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        )
        command.planner_time_to_strike_offset_range = (0.0, 0.0)
        command.planner_target_vel_scale_range = (1.0, 1.0)
        command.planner_target_vel_offset_range = (
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        )
        command.planner_target_vel_yaw_deg_range = (0.0, 0.0)
        command.planner_perturb_curriculum_source = "fixed"
        command.planner_perturb_fixed_scale = 0.0

        # Only execution, recovery, safety, and both-side contact unlock a
        # broader command manifold. Analytic net/landing rates are excluded.
        command.ability_curriculum_enabled = True
        command.ability_curriculum_external_update = False
        command.ability_curriculum_ema_rate = 0.08
        command.ability_curriculum_advance_rate = 0.10
        command.ability_curriculum_regress_rate = 0.10
        command.ability_curriculum_min_resolved_events = 4096
        command.ability_curriculum_required_advance_checks = 2
        command.ability_curriculum_required_regress_checks = 2
        command.ability_curriculum_contact_threshold = 0.30
        command.ability_curriculum_net_threshold = 0.0
        command.ability_curriculum_success_threshold = 0.0
        command.ability_curriculum_recovery_threshold = 0.62
        command.ability_curriculum_recovery_floor = 0.44
        command.ability_curriculum_outcome_regress_ratio = 0.45
        command.ability_curriculum_require_side_contact = True
        command.ability_curriculum_forehand_contact_threshold = 0.24
        command.ability_curriculum_backhand_contact_threshold = 0.24
        command.ability_curriculum_require_side_net = False
        command.ability_curriculum_require_side_success = False
        command.ability_curriculum_use_safety_gates = True
        command.ability_curriculum_use_targeted_attempt = True
        command.ability_curriculum_attempt_threshold = 0.32
        command.ability_curriculum_safety_threshold = 0.985
        command.ability_curriculum_safety_floor = 0.965
        command.ability_curriculum_station_saturation_threshold = 0.20
        command.ability_curriculum_station_saturation_regress_threshold = 0.35
        command.ability_curriculum_require_healthy_impact = True
        command.ability_curriculum_start_racket_pos_scale = (
            0.45,
            0.40,
            0.45,
        )
        command.ability_curriculum_start_ball_scale = 0.35
        command.ability_curriculum_start_planner_perturb_scale = 0.0
        command.impact_health_floor_progress_source = "contact"
        command.impact_health_floor_contact_threshold = 0.30
        command.impact_health_reward_power = 2.0
        command.batched_metric_reset_logging = True

        # A targeted attempt opens recovery even when the virtual contact gate
        # is missed. The terminal path envelope therefore evaluates every
        # meaningful command execution, including misses.
        command.post_contact_ready_enabled = True
        command.post_contact_ready_trigger = "targeted_attempt"
        command.post_contact_ready_curriculum_enabled = False
        command.post_contact_ready_progress_error_mode = "bounded_interval"
        command.post_contact_ready_required_consecutive_steps = 3
        command.post_contact_ready_deadline_steps = 40
        command.post_contact_ready_durable_diagnostic_enabled = True
        command.post_contact_ready_durable_min_delay_s = 0.30
        command.post_contact_ready_durable_deadline_s = 0.80
        command.post_contact_ready_durable_required_consecutive_steps = 5
        command.post_contact_ready_durable_use_effective_gate = False
        command.post_contact_ready_diagnostic_horizon_s = 0.85
        command.post_contact_ready_peak_excess_std = 2.00

        command.post_contact_ready_torso_x_min = -0.12
        command.post_contact_ready_torso_x_max = 0.30
        command.post_contact_ready_max_torso_ang_vel = 1.50
        command.post_contact_ready_max_height_error = 0.14
        command.post_contact_ready_max_base_lin_vel = 0.45
        command.post_contact_ready_max_base_ang_vel = 1.30
        command.post_contact_ready_max_com_x = 0.18
        command.post_contact_ready_max_com_y = 0.20
        command.post_contact_ready_min_feet_contact = 0.50
        command.post_contact_ready_max_station_error = 0.30
        command.post_contact_ready_max_racket_speed = 3.00
        command.post_contact_ready_min_arm_score = 0.0

        command.post_contact_ready_envelope_max_tilt = 0.45
        command.post_contact_ready_envelope_max_abs_pitch = 0.35
        command.post_contact_ready_envelope_max_com_x = 0.24
        command.post_contact_ready_envelope_max_com_y = 0.25
        command.post_contact_ready_envelope_max_waist_overflow = 1.50
        command.post_contact_ready_envelope_max_leg_overflow = 2.50
        command.post_contact_ready_envelope_max_base_ang_vel = 2.50

        command.post_contact_ready_operational_max_tilt = 0.24
        command.post_contact_ready_operational_max_torso_ang_vel = 1.50
        command.post_contact_ready_operational_max_base_ang_vel = 1.30
        command.post_contact_ready_operational_max_height_error = 0.14
        command.post_contact_ready_operational_max_com_x = 0.18
        command.post_contact_ready_operational_max_com_y = 0.20
        command.post_contact_ready_operational_min_feet_contact = 0.50
        command.post_contact_ready_operational_max_station_error = 0.30

        _zero_all_reward_terms(self.rewards)

        # Dense terms provide a reachable path. The hard terminations already
        # remove unsafe trajectories, so their extra scalar penalty must not
        # dominate the initially sparse command signal. Component terms are
        # only a bootstrap; the large values still require joint execution and
        # a safe terminal settlement.
        self.rewards.alive.weight = 0.010
        self.rewards.upright.weight = -0.30
        self.rewards.imitation.weight = 0.65
        self.rewards.imitation.params.update(
            {
                "body_names": list(STAGE1_ARM_PRIOR_BODIES),
                "racket_command_name": "racket_target",
                "pre_strike_scale": 0.80,
                "strike_scale": 0.35,
                "recovery_scale": 0.55,
            }
        )
        self.rewards.planner_racket_task_space_crossfade.weight = 3.0
        self.rewards.planner_racket_task_space_crossfade.params.update(
            {
                "pre_start_s": 0.42,
                "pre_full_s": 0.16,
                "post_full_s": 0.08,
                "post_end_s": 0.22,
                "position_std": 0.30,
                "velocity_std": 1.90,
                "normal_std_rad": 0.75,
                "component_floor": 0.15,
                "minimum_health_multiplier": 0.30,
                "final_minimum_health_multiplier": 0.30,
                "action_feasibility_metric": (
                    "action_operational_feasibility_score"
                ),
                "action_feasibility_floor": 0.40,
            }
        )
        self.rewards.prestrike_racket_progress.weight = 0.90
        self.rewards.prestrike_racket_progress.params.update(
            {
                "minimum_health_multiplier": 0.30,
                "final_minimum_health_multiplier": 0.30,
            }
        )
        self.rewards.near_impact_planner_velocity_progress.weight = 1.10
        self.rewards.near_impact_planner_velocity_progress.params.update(
            {
                "minimum_health_multiplier": 0.30,
                "final_minimum_health_multiplier": 0.30,
            }
        )
        # Bootstrap only the missing impact-position entry. With zero Planner
        # perturbation, the hidden strike point is exactly the actor-visible
        # target. This shared term is deliberately not side-specific and must
        # not be inherited unchanged by noisy Stage 2.
        self.rewards.racket_position.weight = 1.20
        self.rewards.racket_position.params.update(
            {
                "std": 0.30,
                "minimum_health_multiplier": 0.30,
                "final_minimum_health_multiplier": 0.30,
            }
        )
        self.rewards.racket_velocity.weight = 0.60
        self.rewards.racket_velocity.params.update(
            {
                "std": 2.50,
                "minimum_health_multiplier": 0.25,
                "final_minimum_health_multiplier": 0.25,
            }
        )
        self.rewards.blade_direction.weight = 0.40
        self.rewards.blade_direction.params.update(
            {
                "std": 0.80,
                "minimum_health_multiplier": 0.25,
                "final_minimum_health_multiplier": 0.25,
            }
        )
        self.rewards.health_gated_soft_ball_contact.weight = 1.00
        self.rewards.health_gated_soft_ball_contact.params.update(
            {
                "minimum_health_multiplier": 0.20,
                "final_minimum_health_multiplier": 0.20,
                "pos_std": 0.20,
                "window_s": 0.20,
            }
        )
        self.rewards.prestrike_station_progress.weight = 0.65

        self.rewards.exact_impact_planner_task_space_alignment.weight = 3.0
        self.rewards.exact_impact_planner_task_space_alignment.params.update(
            {
                "position_std": 0.10,
                "speed_ratio_std": 0.30,
                "direction_std_rad": 0.35,
                "normal_std_rad": 0.30,
                "component_floor": 0.03,
                "require_contact": True,
                "minimum_health_multiplier": 0.05,
                "final_minimum_health_multiplier": 0.0,
                "impulse": True,
            }
        )
        self.rewards.safe_recovered_planner_command.weight = 6.0
        self.rewards.command_cycle_failure.weight = -1.0
        self.rewards.targeted_contact_miss.weight = -0.25
        self.rewards.targeted_contact_miss.params["impulse"] = True
        self.rewards.safe_strike_inactivity.weight = -0.25
        self.rewards.safe_strike_inactivity.params["impulse"] = True

        # Recovery and balance are low-rate shaping terms. They cannot produce
        # the main Stage-1 return without executing a command and settling it.
        self.rewards.lower_body_support.weight = 0.20
        self.rewards.strike_balance.weight = 0.18
        self.rewards.healthy_trunk_support.weight = 0.08
        self.rewards.healthy_trunk_support.params.update(
            {
                "pre_window_s": 0.34,
                "post_window_s": 0.22,
                "include_no_command_ready": True,
            }
        )
        self.rewards.post_strike_base_ang_vel.weight = 0.15
        self.rewards.post_strike_base_ang_vel.params.update(
            {
                "std": 1.20,
                "include_hold": False,
                "early_prestrike_window_steps": 0,
                "ability_scaled_std": True,
                "initial_std": 3.00,
                "ability_attempt_start": 0.01,
                "ability_attempt_full": 0.08,
            }
        )
        # This is a one-step potential increment, so impulse accounting keeps
        # its per-recovery magnitude independent of the 50 Hz control dt.
        self.rewards.recovery_peak_ang_vel_excess.weight = -0.25
        self.rewards.post_contact_ready_region.weight = 0.15
        self.rewards.no_command_ready_stability.weight = 0.05

        self.rewards.operational_joint_margin.weight = -0.60
        self.rewards.joint_target_slew.weight = -0.035
        self.rewards.actuator_waist_feasibility.weight = -0.020
        self.rewards.actuator_right_arm_feasibility.weight = -0.005
        self.rewards.actuator_leg_feasibility.weight = -0.015
        self.rewards.phase_action_overflow.weight = -0.30
        self.rewards.phase_action_rate_waist.weight = -0.035
        self.rewards.phase_action_rate_upper.weight = -0.005
        self.rewards.phase_action_rate_legs.weight = -0.007
        self.rewards.joint_limit.weight = -1.0
        self.rewards.undesired_contacts.weight = -0.10
        self.rewards.feet_contact_slip.weight = -0.05
        self.rewards.table_no_touch.weight = -1.50
        self.rewards.table_no_touch.params.update(
            {
                "ability_scaled": True,
                "ability_start_scale": 1.0 / 3.0,
                "ability_attempt_start": 0.01,
                "ability_attempt_full": 0.08,
            }
        )
        self.rewards.termination_penalty.weight = -12.0
        self.rewards.termination_penalty.params["term_keys"] = [
            "base_too_low",
            "base_tilted",
            "table_touch",
            "persistent_action_overflow",
        ]

        # Net, bounce, outgoing-ball, and physical-contact rewards remain zero.
        # They belong exclusively to Stage 2.
        self.terminations.base_too_low.params.update(
            {"min_height": 0.65, "min_steps": 25}
        )
        self.terminations.base_tilted.params.update(
            {"threshold": 0.70, "min_steps": 25}
        )
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
        self.terminations.cycle_v2_ready_timeout = None
        self.terminations.single_cycle_curriculum_timeout = None
        self.terminations.table_touch.params.update(
            {"enabled": True, "min_steps": 25}
        )
