"""Clean from-scratch multi-skill task for the HOPE closed-loop-v3 experiment."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.agibot_a3 import AGIBOT_A3_JOINT_NAMES
from whole_body_tracking.robots.agibot_a3_actuator_contract import (
    A3_LEG_JOINT_NAMES,
    A3_PARALLEL_JOINT_NAMES,
    A3_RIGHT_ARM_JOINT_NAMES,
    A3_SERIAL_JOINT_NAMES,
    A3_WAIST_JOINT_NAMES,
    actuator_values,
    validate_actuator_contract,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_actuator_robust_env_cfg import (
    ActuatorRobustEventCfg,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_closed_loop_v2_env_cfg import (
    ClosedLoopV2ImpactRewardsCfg,
    HOPEClosedLoopV2ImpactEnvCfg,
    _zero_all_reward_terms,
)


def _actuator_feasibility_term(joint_names: tuple[str, ...]) -> RewTerm:
    return RewTerm(
        func=mdp.phase_actuator_feasibility,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(joint_names), preserve_order=True
            ),
            "rated_torque": actuator_values(joint_names, "rated_torque"),
            "peak_torque": actuator_values(joint_names, "peak_torque"),
            "rated_speed": actuator_values(joint_names, "rated_speed"),
            "peak_speed": actuator_values(joint_names, "peak_speed"),
            "pre_strike_scale": 0.20,
            "strike_scale": 0.04,
            "recovery_scale": 0.80,
            "hold_scale": 1.00,
            "corner_weight": 1.0,
            "peak_weight": 4.0,
            "free_margin": 0.05,
            "delta": 0.25,
            "maximum": 4.0,
        },
    )


@configclass
class ClosedLoopV3ScratchRewardsCfg(ClosedLoopV2ImpactRewardsCfg):
    prestrike_racket_progress = RewTerm(
        func=mdp.prestrike_racket_progress,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "speed_scale": 0.60,
            "arrival_radius": 0.18,
            "stop_window_s": 0.20,
            "path_lookahead_s": 0.35,
            "idle_cost": 0.06,
            "velocity_frame": "torso_relative",
            "minimum_health_multiplier": 0.15,
            "final_minimum_health_multiplier": 0.0,
        },
    )
    prestrike_station_progress = RewTerm(
        func=mdp.nonfarmable_prestrike_station_progress,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "speed_scale": 0.25,
            "arrival_radius": 0.05,
            "stop_window_s": 0.20,
            "idle_cost": 0.03,
            "minimum_health_multiplier": 0.30,
            "final_minimum_health_multiplier": 0.10,
        },
    )
    near_impact_planner_velocity_progress = RewTerm(
        func=mdp.near_impact_planner_velocity_progress,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "pre_start_s": 0.22,
            "pre_full_s": 0.10,
            "post_full_s": 0.04,
            "post_end_s": 0.08,
            "position_std": 0.30,
            "position_floor": 0.18,
            "normal_std_rad": 0.65,
            "normal_floor": 0.18,
            "projection_ratio_scale": 0.65,
            "lateral_ratio_scale": 0.50,
            "lateral_weight": 0.25,
            "idle_cost": 0.05,
            "minimum_health_multiplier": 0.15,
            "final_minimum_health_multiplier": 0.0,
        },
    )
    actuator_waist_feasibility = _actuator_feasibility_term(
        A3_WAIST_JOINT_NAMES
    )
    actuator_right_arm_feasibility = _actuator_feasibility_term(
        A3_RIGHT_ARM_JOINT_NAMES
    )
    actuator_leg_feasibility = _actuator_feasibility_term(A3_LEG_JOINT_NAMES)


@configclass
class ClosedLoopV3ScratchEventCfg(ActuatorRobustEventCfg):
    # Serial inertia is supplier-bounded to about 3%. Parallel mechanisms use
    # a wider approximation band because scalar joint armature omits coupling.
    serial_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(A3_SERIAL_JOINT_NAMES),
                preserve_order=True,
            ),
            "armature_distribution_params": (0.97, 1.03),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    parallel_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(A3_PARALLEL_JOINT_NAMES),
                preserve_order=True,
            ),
            "armature_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class HOPEClosedLoopV3ScratchMultiSkillEnvCfg(HOPEClosedLoopV2ImpactEnvCfg):
    """Independent 114-D scratch task; no historical policy is loaded."""

    rewards: ClosedLoopV3ScratchRewardsCfg = ClosedLoopV3ScratchRewardsCfg()
    events: ClosedLoopV3ScratchEventCfg = ClosedLoopV3ScratchEventCfg()

    def __post_init__(self):
        validate_actuator_contract(AGIBOT_A3_JOINT_NAMES)
        super().__post_init__()

        self.episode_length_s = 45.0
        self.actions.joint_pos.feedback_mode = "effective"

        motion = self.commands.motion
        motion.wrap_teleport = False
        motion.hold_steps_range = (25, 55)
        motion.stand_start_prob = 0.40
        motion.stand_start_min_hold = 35
        motion.stand_episode_prob = 0.10
        motion.stand_episode_hold_steps = 100000
        motion.motion_start_warmup_enabled = True
        motion.motion_start_warmup_start_prob = 0.60
        motion.motion_start_warmup_min_prob = 0.20
        motion.motion_start_warmup_contact_low = 0.04
        motion.motion_start_warmup_contact_high = 0.32
        motion.motion_start_warmup_recovery_low = 0.55
        motion.motion_start_warmup_recovery_high = 0.72
        motion.motion_start_warmup_prestrike_enabled = True
        motion.motion_start_warmup_prestrike_fraction = 0.60
        # Manifest order: BH static 1/2, FH static, FH translation 1/2.
        motion.clip_sampling_weights = (0.25, 0.25, 0.20, 0.15, 0.15)
        motion.core_clip_count = 3

        command = self.commands.racket_target
        command.strike_position_mode = "table_workspace"
        command.planner_hit_plane_mode = "fixed_x_hit"
        command.planner_hit_plane_x = 0.20
        command.table_workspace_fixed_level = -1.0
        command.table_workspace_motion_seed_blend_start = 0.0
        command.table_workspace_motion_seed_end_level = 0.0
        command.table_workspace_edge_margin = 0.02
        command.table_workspace_side_overlap = 0.12
        command.table_workspace_forehand_core_y_range = (-0.48, -0.22)
        command.table_workspace_backhand_core_y_range = (-0.10, 0.18)
        command.table_workspace_x_jitter_core_range = (-0.012, 0.012)
        command.table_workspace_x_jitter_full_range = (-0.04, 0.04)
        command.table_workspace_z_core_above_surface_range = (0.39, 0.54)
        command.table_workspace_z_full_above_surface_range = (0.34, 0.68)
        command.table_workspace_fringe_prob = 0.08

        command.station_mode = "dynamic_from_motion"
        command.dynamic_station_xy_clip = ((0.0, 0.0), (-0.30, 0.30))
        command.dynamic_station_blend = 1.0
        command.deploy_ready_hold_prob = 0.25
        command.deploy_ready_force_stand_episode = True
        command.deploy_ready_force_default_stand_reset = True

        command.racket_velocity_mode = "impact_inverse_landing"
        command.planner_command_mode = "v4_wire_compatible"
        command.incoming_trajectory_mode = "one_bounce"
        command.paddle_restitution = 0.654
        command.paddle_tangent_retain = 0.48
        command.impact_inverse_racket_speed_scale = 1.18
        command.impact_inverse_racket_speed_bias = 0.10
        command.impact_inverse_max_racket_speed = 3.35
        command.target_outgoing_vel_scale = 1.04
        command.target_outgoing_vel_z_bias = 0.26
        command.net_margin = 0.06

        # These perturb the accepted V4 command jointly. In V4-compatible mode
        # target normal is always reconstructed from the perturbed velocity.
        command.planner_target_pos_offset_range = (
            (-0.035, 0.035),
            (-0.045, 0.045),
            (-0.030, 0.030),
        )
        command.planner_time_to_strike_offset_range = (-0.10, 0.02)
        command.planner_target_vel_scale_range = (0.92, 1.08)
        command.planner_target_vel_offset_range = (
            (-0.10, 0.10),
            (-0.10, 0.10),
            (-0.08, 0.08),
        )
        command.planner_target_vel_yaw_deg_range = (-3.0, 3.0)

        command.ability_curriculum_enabled = True
        command.ability_curriculum_start_racket_pos_scale = (0.50, 0.45, 0.45)
        command.ability_curriculum_start_ball_scale = 0.45
        command.ability_curriculum_start_planner_perturb_scale = 0.10
        command.ability_curriculum_ema_rate = 0.02
        command.ability_curriculum_advance_rate = 0.02
        command.ability_curriculum_regress_rate = 0.04
        command.ability_curriculum_min_resolved_events = 4096
        command.ability_curriculum_required_advance_checks = 2
        command.ability_curriculum_required_regress_checks = 2
        command.ability_curriculum_contact_threshold = 0.32
        command.ability_curriculum_net_threshold = 0.16
        command.ability_curriculum_success_threshold = 0.07
        command.ability_curriculum_recovery_threshold = 0.62
        command.ability_curriculum_recovery_floor = 0.48
        command.ability_curriculum_outcome_regress_ratio = 0.55
        command.ability_curriculum_require_side_contact = True
        command.ability_curriculum_forehand_contact_threshold = 0.25
        command.ability_curriculum_backhand_contact_threshold = 0.25
        command.ability_curriculum_use_cycle_success = False
        command.ability_curriculum_use_cycle_v2 = False
        command.ability_curriculum_use_safety_gates = True
        command.ability_curriculum_use_targeted_attempt = True
        command.ability_curriculum_attempt_threshold = 0.30
        command.ability_curriculum_safety_threshold = 0.990
        command.ability_curriculum_safety_floor = 0.975
        command.ability_curriculum_station_saturation_threshold = 0.10
        command.ability_curriculum_station_saturation_regress_threshold = 0.18
        command.ability_curriculum_require_healthy_impact = True

        command.healthy_three_stage_enabled = True
        command.healthy_stage2_start_level = 0.25
        command.healthy_stage3_start_level = 0.60
        command.healthy_stage1_ready_hold_prob = 0.15
        command.healthy_stage2_ready_hold_prob = 0.45
        command.healthy_stage3_ready_hold_prob = 0.25
        command.healthy_stage3_planted_rehearsal_prob = 0.25
        command.healthy_stage_use_dynamic_station_during_ready = True
        command.healthy_stage_lateral_station_bias = 0.08
        command.healthy_stage2_station_y_clip = 0.10
        command.healthy_stage3_station_target_gain = 0.85
        command.healthy_stage_station_arrival_radius = 0.07
        command.healthy_stage_station_settle_max_lin_vel = 0.20
        command.healthy_stage_station_settle_max_ang_vel = 0.70
        command.healthy_stage_station_settle_min_steps = 10
        command.healthy_stage_side_contact_threshold = 0.25
        command.healthy_stage_normal_threshold = 0.20
        command.healthy_stage_safety_threshold = 0.990
        command.healthy_stage_recovery_threshold = 0.62
        command.healthy_stage_station_arrival_threshold = 0.65
        command.healthy_stage_station_settle_threshold = 0.50
        command.healthy_stage_normal_error_max_deg = 35.0

        command.impact_health_include_waist_retreat = True
        command.impact_health_include_waist_backfold = True
        command.impact_health_backlean_tolerance = 0.05
        command.impact_health_backlean_std = 0.12
        command.impact_health_max_backlean = 0.18
        command.impact_health_waist_backfold_tolerance = 0.05
        command.impact_health_waist_backfold_std = 0.13
        command.impact_health_max_waist_backfold = 0.20
        command.impact_health_base_retreat_std = 0.09
        command.impact_health_max_base_retreat = 0.12
        command.impact_health_max_com_x = 0.19
        command.impact_health_max_com_y = 0.20
        command.impact_health_max_backward_vel = 0.32
        command.impact_health_max_ang_vel = 2.50
        command.impact_health_min_feet_contact = 0.50

        command.cycle_v2_enabled = True
        command.cycle_v2_required_outcome = "contact"
        command.cycle_v2_outcome_by_ability = True
        command.cycle_v2_net_outcome_level = 0.55
        command.cycle_v2_bounce_outcome_level = 0.82
        command.cycle_v2_visible_deadline_steps = 70
        command.cycle_v2_total_deadline_steps = 110
        command.cycle_v2_count_no_command_as_visible = True
        command.cycle_v2_fail_on_strike_without_ready = True
        command.cycle_v2_fail_unresolved_on_resample = True
        command.cycle_v2_ready_mode = "core_hard_soft"
        command.cycle_v2_use_functional_ready = False
        command.cycle_v2_required_consecutive_ready_steps = 5
        command.cycle_v2_ready_threshold = 0.66
        command.cycle_v2_max_height_error = 0.10
        command.cycle_v2_max_upright_error = 0.28
        command.cycle_v2_max_base_lin_vel = 0.25
        command.cycle_v2_max_base_ang_vel = 0.80
        command.cycle_v2_min_feet_contact = 0.65
        command.cycle_v2_require_healthy_impact = True

        # Do not impose the old forehand-only READY arm anchor on backhand.
        command.functional_ready_joint_names = ()
        command.functional_ready_joint_positions = ()
        command.functional_ready_joint_tolerances = ()
        command.post_contact_ready_enabled = True
        command.post_contact_ready_trigger = "targeted_attempt"
        command.post_contact_ready_durable_diagnostic_enabled = True
        command.post_contact_ready_curriculum_enabled = False
        command.actuator_safety_overflow_threshold = 4.0
        command.actuator_safety_overflow_consecutive_steps = 5

        # Rebuild the reward surface from an explicit whitelist after all v2
        # initialization. This prevents hidden historical reward inheritance.
        _zero_all_reward_terms(self.rewards)

        self.rewards.upright.weight = -0.20
        self.rewards.imitation.weight = 0.45
        self.rewards.imitation.params.update(
            {
                "racket_command_name": "racket_target",
                "pre_strike_scale": 0.65,
                "strike_scale": 0.10,
                "recovery_scale": 0.35,
                "core_clip_scale": 1.0,
                "supplemental_clip_scale": 1.10,
            }
        )
        self.rewards.phase_lower_body_motion_prior.weight = 0.20
        self.rewards.phase_lower_body_motion_prior.params.update(
            {
                "pre_strike_scale": 0.10,
                "strike_scale": 0.02,
                "recovery_scale": 0.38,
                "hold_scale": 0.45,
                "core_clip_scale": 0.90,
                "supplemental_clip_scale": 1.25,
            }
        )
        self.rewards.racket_wrist_motion_pos.weight = 0.12
        self.rewards.racket_wrist_motion_pos.params.update(
            {"release_window_s": 0.30, "release_scale": 0.05}
        )
        self.rewards.racket_wrist_motion_ori.weight = 0.10
        self.rewards.racket_wrist_motion_ori.params.update(
            {"release_window_s": 0.30, "release_scale": 0.05}
        )

        self.rewards.racket_position.weight = 1.6
        self.rewards.racket_position.params.update(
            {
                "std": 0.20,
                "minimum_health_multiplier": 0.30,
                "final_minimum_health_multiplier": 0.08,
            }
        )
        self.rewards.planner_racket_task_space_crossfade.weight = 3.0
        self.rewards.planner_racket_task_space_crossfade.params.update(
            {
                "pre_start_s": 0.42,
                "pre_full_s": 0.16,
                "position_std": 0.15,
                "velocity_std": 1.00,
                "normal_std_rad": 0.34,
                "ability_scaled_stds": True,
                "initial_position_std": 0.45,
                "initial_velocity_std": 2.50,
                "initial_normal_std_rad": 1.20,
                "minimum_health_multiplier": 0.30,
                "final_minimum_health_multiplier": 0.08,
            }
        )
        self.rewards.prestrike_racket_progress.weight = 1.2
        self.rewards.prestrike_station_progress.weight = 0.8
        self.rewards.near_impact_planner_velocity_progress.weight = 3.0
        self.rewards.planner_velocity_band.weight = 0.0
        self.rewards.exact_impact_planner_task_space_alignment.weight = 2.5
        self.rewards.exact_impact_planner_task_space_alignment.params.update(
            {
                "position_std": 0.10,
                "speed_ratio_std": 0.25,
                "direction_std_rad": 0.32,
                "normal_std_rad": 0.24,
                "require_contact": True,
                "minimum_health_multiplier": 0.25,
                "final_minimum_health_multiplier": 0.05,
            }
        )
        self.rewards.health_gated_soft_ball_contact.weight = 2.8
        self.rewards.health_gated_soft_ball_contact.params.update(
            {
                "minimum_health_multiplier": 0.30,
                "final_minimum_health_multiplier": 0.08,
                "swing_side": 0.0,
            }
        )
        self.rewards.targeted_strike_attempt.weight = 1.0
        self.rewards.targeted_strike_attempt.params[
            "minimum_health_multiplier"
        ] = 0.35
        self.rewards.physical_outcome.weight = 1.5
        self.rewards.physical_outcome.params.update(
            {
                "contact_scale": 0.7,
                "net_cross_scale": 1.2,
                "opponent_bounce_scale": 1.7,
                "minimum_health_multiplier": 0.25,
                "overflow_std": 0.40,
            }
        )
        self.rewards.safe_strike_inactivity.weight = -3.0

        self.rewards.no_command_instability.weight = -0.35
        self.rewards.active_ready_sustained_bonus.weight = 2.0
        self.rewards.post_contact_directional_recovery.weight = 0.8
        self.rewards.cycle_v2_ready_success_bonus.weight = 12.0
        self.rewards.cycle_v2_ready_success_bonus.params[
            "tier_multipliers"
        ] = (0.6, 1.0, 1.5)
        self.rewards.cycle_v2_ready_fail.weight = -8.0
        self.rewards.cycle_v2_streak_bonus.weight = 3.0
        self.rewards.cycle_v2_streak_bonus.params.update(
            {"min_streak": 2, "min_ability_level": 0.55}
        )

        self.rewards.phase_action_overflow.weight = -0.15
        self.rewards.phase_action_rate_waist.weight = -0.025
        self.rewards.phase_action_rate_waist.params.update(
            {
                "pre_strike_scale": 0.01,
                "strike_scale": 0.002,
                "recovery_scale": 0.08,
                "hold_scale": 0.10,
            }
        )
        self.rewards.phase_action_rate_upper.weight = -0.005
        self.rewards.phase_action_rate_upper.params.update(
            {
                "pre_strike_scale": 0.004,
                "strike_scale": 0.001,
                "recovery_scale": 0.025,
                "hold_scale": 0.035,
            }
        )
        self.rewards.phase_action_rate_legs.weight = -0.003
        self.rewards.phase_action_rate_legs.params.update(
            {
                "pre_strike_scale": 0.002,
                "strike_scale": 0.0,
                "recovery_scale": 0.012,
                "hold_scale": 0.018,
            }
        )
        self.rewards.actuator_waist_feasibility.weight = -0.025
        self.rewards.actuator_right_arm_feasibility.weight = -0.015
        self.rewards.actuator_leg_feasibility.weight = -0.025
        self.rewards.joint_limit.weight = -0.50
        self.rewards.undesired_contacts.weight = -0.15
        self.rewards.feet_contact_slip.weight = -0.05
        self.rewards.table_no_touch.weight = -1.20
        self.rewards.termination_penalty.weight = -30.0
        self.rewards.termination_penalty.params["term_keys"] = [
            "base_too_low",
            "base_tilted",
            "table_touch",
            "persistent_action_overflow",
        ]

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
        self.terminations.table_touch.params.update(
            {"enabled": True, "min_steps": 25}
        )
        self.terminations.persistent_action_overflow.params.update(
            {"enabled": True, "min_steps": 25}
        )

        self.events.physics_material.params.update(
            {
                "static_friction_range": (0.80, 1.20),
                "dynamic_friction_range": (0.75, 1.15),
                "restitution_range": (0.0, 0.08),
            }
        )
        self.events.base_com.params["com_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
        }
        self.events.link_mass.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.pd_gains.params[
            "stiffness_distribution_params"
        ] = (0.95, 1.05)
        self.events.pd_gains.params[
            "damping_distribution_params"
        ] = (0.95, 1.05)
