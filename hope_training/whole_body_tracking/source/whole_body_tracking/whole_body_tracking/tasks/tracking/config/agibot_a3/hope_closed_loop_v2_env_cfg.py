"""Isolated Stage-1 environment for the HOPE closed-loop-v2 design."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
    HOPEPingPongEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class ClosedLoopV2RewardsCfg(RewardsCfg):
    physical_outcome = RewTerm(
        func=mdp.closed_loop_v2_physical_outcome,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "contact_scale": 1.0,
            "net_cross_scale": 2.0,
            "opponent_bounce_scale": 3.0,
            "minimum_health_multiplier": 0.25,
            "overflow_std": 0.40,
        },
    )
    closed_cycle_success = RewTerm(
        func=mdp.closed_loop_v2_cycle_success_bonus,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    recovery_progress = RewTerm(
        func=mdp.closed_loop_v2_recovery_progress,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    recovery_resolved = RewTerm(
        func=mdp.closed_loop_v2_recovery_success,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    recovery_failed = RewTerm(
        func=mdp.closed_loop_v2_recovery_failure,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    durable_cycle_success = RewTerm(
        func=mdp.closed_loop_v2_durable_cycle_success_bonus,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    durable_recovery_progress = RewTerm(
        func=mdp.closed_loop_v2_durable_recovery_progress,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    durable_recovery_resolved = RewTerm(
        func=mdp.closed_loop_v2_durable_recovery_success,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    durable_recovery_failed = RewTerm(
        func=mdp.closed_loop_v2_durable_recovery_failure,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    durable_outcome_bonus = RewTerm(
        func=mdp.durable_recovery_outcome_bonus,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "tier_multipliers": (0.5, 1.0, 1.5),
        },
    )
    no_command_instability = RewTerm(
        func=mdp.closed_loop_v2_no_command_instability,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    safe_strike_inactivity = RewTerm(
        func=mdp.closed_loop_v2_safe_inactivity,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "minimum_health": 0.50,
            "maximum_target_distance_from_base": 1.40,
        },
    )


@configclass
class ClosedLoopV2ImpactRewardsCfg(ClosedLoopV2RewardsCfg):
    planner_velocity_band = RewTerm(
        func=mdp.closed_loop_v2_planner_velocity_band,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "timing_std_s": 0.055,
            "active_window_s": 0.12,
            "position_std": 0.12,
            "position_floor": 0.10,
            "speed_ratio_std": 0.30,
            "direction_std_rad": 0.35,
            "component_floor": 0.02,
        },
    )


@configclass
class ClosedLoopV2SafeQualityRewardsCfg(ClosedLoopV2ImpactRewardsCfg):
    terminal_quality_window = RewTerm(
        func=mdp.closed_loop_v2_terminal_quality_window,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    safe_terminal_quality = RewTerm(
        func=mdp.closed_loop_v2_safe_terminal_quality,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    unsafe_terminal_recovery = RewTerm(
        func=mdp.closed_loop_v2_unsafe_terminal_recovery,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    safe_terminal_outcome = RewTerm(
        func=mdp.closed_loop_v2_safe_terminal_outcome,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "tier_multipliers": (0.5, 1.0, 1.5),
        },
    )
    safe_terminal_cycle = RewTerm(
        func=mdp.closed_loop_v2_safe_terminal_cycle,
        weight=0.0,
        params={"command_name": "racket_target"},
    )


@configclass
class ClosedLoopV2ImpactRecoveryRewardsCfg(ClosedLoopV2ImpactRewardsCfg):
    recovered_planner_velocity = RewTerm(
        func=mdp.closed_loop_v2_recovered_planner_velocity,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "speed_ratio_std": 0.30,
            "direction_std_rad": 0.35,
            "component_floor": 0.02,
            "position_std": 0.12,
            "position_floor": 0.10,
            "impact_health_floor": 0.25,
            "recovery_peak_ang_vel_budget": 0.80,
            "recovery_peak_ang_vel_excess_std": 0.60,
            "recovery_gate_floor": 0.10,
        },
    )


@configclass
class ClosedLoopV2ImpactConstraintRewardsCfg(ClosedLoopV2ImpactRewardsCfg):
    recovery_peak_ang_vel_excess = RewTerm(
        func=mdp.closed_loop_v2_recovery_peak_ang_vel_excess,
        weight=0.0,
        params={"command_name": "racket_target"},
    )


@configclass
class ClosedLoopV2TerminationsCfg(TerminationsCfg):
    persistent_action_overflow = DoneTerm(
        func=mdp.persistent_action_overflow,
        params={
            "command_name": "racket_target",
            "enabled": True,
            "min_steps": 25,
        },
    )


def _zero_all_reward_terms(rewards: ClosedLoopV2RewardsCfg) -> None:
    """Prevent any historical experiment reward from leaking into v2."""
    for name in dir(rewards):
        if name.startswith("_"):
            continue
        term = getattr(rewards, name)
        if hasattr(term, "weight"):
            term.weight = 0.0


@configclass
class HOPEClosedLoopV2EnvCfg(HOPEPingPongEnvCfg):
    """Fixed-station, forehand-core validation task with a closed cycle."""

    rewards: ClosedLoopV2RewardsCfg = ClosedLoopV2RewardsCfg()
    terminations: ClosedLoopV2TerminationsCfg = ClosedLoopV2TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 45.0

        motion = self.commands.motion
        motion.wrap_teleport = False
        motion.hold_steps_range = (20, 60)
        motion.stand_start_prob = 0.35
        motion.stand_start_min_hold = 20
        motion.stand_episode_prob = 0.10
        motion.stand_episode_hold_steps = 100000
        motion.core_clip_count = 1
        motion.clip_sampling_weights = ()
        motion.stand_start_pose_range = {
            "x": (-0.025, 0.025),
            "y": (-0.025, 0.025),
            "z": (-0.005, 0.005),
            "roll": (-0.06, 0.06),
            "pitch": (-0.10, 0.10),
            "yaw": (-0.04, 0.04),
        }
        motion.stand_start_velocity_range = {
            "x": (-0.12, 0.12),
            "y": (-0.08, 0.08),
            "z": (-0.03, 0.03),
            "roll": (-0.24, 0.24),
            "pitch": (-0.36, 0.36),
            "yaw": (-0.14, 0.14),
        }
        motion.stand_start_joint_position_range = (-0.02, 0.02)

        command = self.commands.racket_target
        command.station_mode = "fixed"
        command.deploy_ready_hold_prob = 1.0
        command.station_relocation_enabled = False
        command.healthy_three_stage_enabled = False
        command.strike_position_mode = "table_workspace"
        command.table_workspace_fixed_level = 0.0
        command.table_workspace_fringe_prob = 0.0
        command.table_workspace_motion_seed_blend_start = 0.0
        command.table_workspace_motion_seed_end_level = 0.0
        command.table_workspace_forehand_core_y_range = (-0.55, -0.20)
        command.table_workspace_x_jitter_core_range = (-0.015, 0.015)
        command.table_workspace_z_core_above_surface_range = (0.40, 0.58)
        command.planner_hit_plane_x = 0.20
        command.racket_velocity_mode = "impact_inverse_landing"
        command.planner_command_mode = "v4_wire_compatible"
        command.incoming_trajectory_mode = "one_bounce"
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
        command.strike_window_s = 0.12
        command.ability_curriculum_enabled = False
        command.cycle_v2_enabled = False

        command.functional_ready_joint_names = (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        )
        command.functional_ready_joint_positions = (
            -0.246565,
            0.618356,
            0.026664,
            0.333463,
            -0.591667,
            -0.344093,
            0.642940,
            -1.124400,
            -0.267800,
            0.441000,
            0.792700,
            0.866700,
            -0.142800,
            -0.238400,
        )
        command.functional_ready_joint_tolerances = (
            0.35,
            0.30,
            0.35,
            0.35,
            0.45,
            0.30,
            0.40,
            0.20,
            0.18,
            0.20,
            0.18,
            0.30,
            0.15,
            0.20,
        )
        command.functional_ready_joint_std = 0.22
        command.post_contact_ready_enabled = True
        command.post_contact_ready_trigger = "targeted_attempt"
        command.post_contact_ready_torso_x_min = -0.035
        command.post_contact_ready_torso_x_max = 0.14
        command.post_contact_ready_max_torso_ang_vel = 0.80
        command.post_contact_ready_max_height_error = 0.10
        command.post_contact_ready_max_base_lin_vel = 0.22
        command.post_contact_ready_max_base_ang_vel = 0.80
        command.post_contact_ready_max_com_x = 0.13
        command.post_contact_ready_max_com_y = 0.15
        command.post_contact_ready_min_feet_contact = 0.90
        command.post_contact_ready_max_station_error = 0.22
        command.post_contact_ready_max_racket_speed = 1.10
        command.post_contact_ready_min_arm_score = 0.40
        command.post_contact_ready_required_consecutive_steps = 5
        command.post_contact_ready_deadline_steps = 60
        command.post_contact_ready_progress_clip = 0.035
        command.post_contact_ready_durable_diagnostic_enabled = True
        command.post_contact_ready_curriculum_enabled = True
        command.post_contact_ready_curriculum_start_level = 0
        command.post_contact_ready_curriculum_max_torso_ang_vel = (
            1.20,
            0.95,
            0.80,
        )
        command.post_contact_ready_curriculum_max_base_lin_vel = (
            0.35,
            0.28,
            0.22,
        )
        command.post_contact_ready_curriculum_max_base_ang_vel = (
            1.40,
            1.05,
            0.80,
        )
        command.post_contact_ready_curriculum_max_racket_speed = (
            1.80,
            1.40,
            1.10,
        )
        command.post_contact_ready_curriculum_min_feet_contact = (
            0.75,
            0.85,
            0.90,
        )
        command.post_contact_ready_curriculum_required_consecutive_steps = (
            2,
            3,
            5,
        )
        command.post_contact_ready_curriculum_deadline_steps = (
            90,
            75,
            60,
        )
        command.post_contact_ready_curriculum_advance_success_thresholds = (
            0.45,
            0.60,
        )
        command.post_contact_ready_curriculum_min_resolved_events = (
            1024,
            2048,
        )
        command.post_contact_ready_curriculum_shadow_success_thresholds = (
            0.15,
            0.25,
        )
        command.post_contact_ready_curriculum_min_targeted_attempt_ema = 0.25
        command.post_contact_ready_curriculum_min_return_success_ema = 0.05
        command.post_contact_ready_curriculum_min_completed_swings = 2048
        command.post_contact_ready_curriculum_required_advance_checks = 2
        command.post_contact_ready_curriculum_ema_rate = 0.05
        command.post_contact_ready_curriculum_hit_ema_rate = 0.05

        command.impact_health_include_waist_backfold = True
        command.impact_health_include_waist_retreat = False
        command.impact_health_max_backlean = 0.20
        command.impact_health_max_waist_backfold = 0.20
        command.impact_health_max_ang_vel = 2.50
        command.impact_health_min_feet_contact = 0.50
        command.actuator_safety_overflow_threshold = 6.0
        command.actuator_safety_overflow_consecutive_steps = 5

        _zero_all_reward_terms(self.rewards)

        self.rewards.upright.weight = -0.15
        self.rewards.imitation.weight = 0.55
        self.rewards.imitation.params.update(
            {
                "racket_command_name": "racket_target",
                "pre_strike_scale": 0.70,
                "strike_scale": 0.15,
                "recovery_scale": 0.45,
            }
        )
        self.rewards.racket_wrist_motion_pos.weight = 0.10
        self.rewards.racket_wrist_motion_pos.params.update(
            {"release_window_s": 0.24, "release_scale": 0.08}
        )
        self.rewards.racket_wrist_motion_ori.weight = 0.08
        self.rewards.racket_wrist_motion_ori.params.update(
            {"release_window_s": 0.24, "release_scale": 0.08}
        )

        self.rewards.racket_position.weight = 2.0
        self.rewards.racket_position.params.update(
            {
                "std": 0.18,
                "minimum_health_multiplier": 0.25,
                "final_minimum_health_multiplier": 0.25,
            }
        )
        self.rewards.racket_velocity_projection.weight = 1.0
        self.rewards.racket_velocity_projection.params.update(
            {
                "minimum_health_multiplier": 0.25,
                "final_minimum_health_multiplier": 0.25,
            }
        )
        self.rewards.prestrike_blade_direction.weight = 0.35
        self.rewards.planner_racket_task_space_crossfade.weight = 1.0
        self.rewards.planner_racket_task_space_crossfade.params.update(
            {
                "minimum_health_multiplier": 0.25,
                "final_minimum_health_multiplier": 0.25,
            }
        )
        self.rewards.exact_impact_planner_task_space_alignment.weight = 1.5
        self.rewards.exact_impact_planner_task_space_alignment.params.update(
            {
                "minimum_health_multiplier": 0.25,
                "final_minimum_health_multiplier": 0.25,
                "require_contact": True,
            }
        )
        self.rewards.health_gated_soft_ball_contact.weight = 3.0
        self.rewards.health_gated_soft_ball_contact.params.update(
            {
                "minimum_health_multiplier": 0.25,
                "final_minimum_health_multiplier": 0.25,
                "swing_side": 1.0,
            }
        )
        self.rewards.physical_outcome.weight = 1.0
        self.rewards.physical_outcome.params.update(
            {
                "contact_scale": 1.0,
                "net_cross_scale": 1.0,
                "opponent_bounce_scale": 1.0,
            }
        )
        self.rewards.closed_cycle_success.weight = 24.0
        self.rewards.recovery_progress.weight = 0.8
        self.rewards.recovery_resolved.weight = 6.0
        self.rewards.recovery_failed.weight = -6.0
        self.rewards.deferred_recovery_outcome_bonus.weight = 8.0
        self.rewards.deferred_recovery_outcome_bonus.params.update(
            {"tier_multipliers": (0.5, 1.0, 1.5)}
        )
        self.rewards.no_command_instability.weight = -0.40
        self.rewards.safe_strike_inactivity.weight = -6.0

        self.rewards.phase_action_overflow.weight = -0.15
        self.rewards.phase_action_rate_waist.weight = -0.04
        self.rewards.phase_action_rate_waist.params.update(
            {
                "pre_strike_scale": 0.02,
                "strike_scale": 0.005,
                "recovery_scale": 0.12,
                "hold_scale": 0.15,
            }
        )
        self.rewards.phase_action_rate_upper.weight = -0.008
        self.rewards.phase_action_rate_legs.weight = -0.004
        self.rewards.joint_limit.weight = -0.8
        self.rewards.undesired_contacts.weight = -0.1
        self.rewards.feet_contact_slip.weight = -0.05
        self.rewards.table_no_touch.weight = -1.0
        self.rewards.termination_penalty.weight = -24.0
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

        self.events.physics_material.params.update(
            {
                "static_friction_range": (0.80, 1.20),
                "dynamic_friction_range": (0.75, 1.15),
                "restitution_range": (0.0, 0.10),
            }
        )
        self.events.base_com.params["com_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
        }
        self.events.link_mass.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.pd_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.events.pd_gains.params["damping_distribution_params"] = (0.95, 1.05)


@configclass
class HOPEClosedLoopV2ImpactEnvCfg(HOPEClosedLoopV2EnvCfg):
    """V2 task plus isolated planner impact-velocity feedback."""

    rewards: ClosedLoopV2ImpactRewardsCfg = ClosedLoopV2ImpactRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.planner_velocity_band.weight = 6.0


@configclass
class HOPEClosedLoopV2DurableCycleEnvCfg(HOPEClosedLoopV2ImpactEnvCfg):
    """ImpactV1 strike feedback with fixed-deadline next-cycle settlement."""

    rewards: ClosedLoopV2ImpactRewardsCfg = ClosedLoopV2ImpactRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.racket_target

        # The durable gate is immutable and checkpoint-independent. The old
        # mutable, early-READY curriculum remains available only as diagnosis.
        command.post_contact_ready_durable_diagnostic_enabled = True
        command.post_contact_ready_curriculum_enabled = False

        # Remove every payoff tied to the brief early READY crossing.
        self.rewards.closed_cycle_success.weight = 0.0
        self.rewards.recovery_progress.weight = 0.0
        self.rewards.recovery_resolved.weight = 0.0
        self.rewards.recovery_failed.weight = 0.0
        self.rewards.deferred_recovery_outcome_bonus.weight = 0.0

        # Preserve dense impact exploration. Full outcome/cycle value is paid
        # only when the state at 1.10 s remains reusable for the next command.
        self.rewards.durable_recovery_progress.weight = 0.8
        self.rewards.durable_recovery_resolved.weight = 2.0
        self.rewards.durable_recovery_failed.weight = -8.0
        self.rewards.durable_outcome_bonus.weight = 8.0
        self.rewards.durable_cycle_success.weight = 24.0


@configclass
class HOPEClosedLoopV2SafeQualityCycleEnvCfg(HOPEClosedLoopV2ImpactEnvCfg):
    """Impact feedback plus safety-conditioned terminal recovery quality."""

    rewards: ClosedLoopV2SafeQualityRewardsCfg = (
        ClosedLoopV2SafeQualityRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.racket_target
        command.post_contact_ready_durable_diagnostic_enabled = True
        command.post_contact_ready_curriculum_enabled = False

        # Disable both brief READY settlement and V1's all-or-nothing durable
        # objective. The strict durable event remains read-only diagnosis.
        for term in (
            self.rewards.closed_cycle_success,
            self.rewards.recovery_progress,
            self.rewards.recovery_resolved,
            self.rewards.recovery_failed,
            self.rewards.deferred_recovery_outcome_bonus,
            self.rewards.durable_cycle_success,
            self.rewards.durable_recovery_progress,
            self.rewards.durable_recovery_resolved,
            self.rewards.durable_recovery_failed,
            self.rewards.durable_outcome_bonus,
        ):
            term.weight = 0.0

        # The fixed terminal window contributes at most four reward units.
        # Catastrophic envelope violations revoke every deferred strike term.
        self.rewards.terminal_quality_window.weight = 4.0
        self.rewards.safe_terminal_quality.weight = 2.0
        self.rewards.unsafe_terminal_recovery.weight = -12.0
        self.rewards.safe_terminal_outcome.weight = 8.0
        self.rewards.safe_terminal_cycle.weight = 24.0


@configclass
class HOPEClosedLoopV2SafeFaceQualityCycleEnvCfg(
    HOPEClosedLoopV2SafeQualityCycleEnvCfg
):
    """Safe-quality cycle with rigid-contact-informed usable-face credit."""

    rewards: ClosedLoopV2SafeQualityRewardsCfg = (
        ClosedLoopV2SafeQualityRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.racket_target
        command.face_quality_inner_radius = 0.061
        command.face_quality_outer_radius = 0.081

        # Dense approach/contact shaping stays unchanged. Only exact analytic
        # impact and deferred outcome value are discounted near the blade rim.
        self.rewards.exact_impact_planner_task_space_alignment.params.update(
            {
                "face_quality_power": 1.0,
                "face_quality_floor": 0.05,
            }
        )
        self.rewards.physical_outcome.params.update(
            {
                "face_quality_power": 1.0,
                "face_quality_floor": 0.10,
            }
        )
        self.rewards.safe_terminal_outcome.params.update(
            {
                "face_quality_power": 1.0,
                "face_quality_floor": 0.0,
            }
        )
        self.rewards.safe_terminal_cycle.params.update(
            {
                "face_quality_power": 1.0,
                "face_quality_floor": 0.0,
            }
        )


@configclass
class HOPEClosedLoopV2ImpactRecoveryEnvCfg(HOPEClosedLoopV2ImpactEnvCfg):
    """V2 impact feedback with planner quality settled after recovery."""

    rewards: ClosedLoopV2ImpactRecoveryRewardsCfg = (
        ClosedLoopV2ImpactRecoveryRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.planner_velocity_band.weight = 3.0
        self.rewards.recovered_planner_velocity.weight = 12.0


@configclass
class HOPEClosedLoopV2ImpactConstraintEnvCfg(HOPEClosedLoopV2ImpactEnvCfg):
    """ImpactV1 plus a direct, incremental recovery-peak constraint."""

    rewards: ClosedLoopV2ImpactConstraintRewardsCfg = (
        ClosedLoopV2ImpactConstraintRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.recovery_peak_ang_vel_excess.weight = -2.0
