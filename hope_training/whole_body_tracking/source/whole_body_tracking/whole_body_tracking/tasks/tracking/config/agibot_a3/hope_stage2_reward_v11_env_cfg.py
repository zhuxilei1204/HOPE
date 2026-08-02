"""Isolated Stage-2 V11 task with same-frame physical reward settlement."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp

from .hope_closed_loop_v2_env_cfg import _zero_all_reward_terms
from .hope_stage1_command_tracking_env_cfg import STAGE1_ARM_PRIOR_BODIES
from .hope_stage2_physical_env_cfg import (
    HOPEStage2PhysicalEnvCfg,
    Stage2PhysicalRewardsCfg,
)


@configclass
class Stage2RewardV11RewardsCfg(Stage2PhysicalRewardsCfg):
    """Small reward whitelist: prior, command, physics, recovery, safety."""

    durable_recovery_progress = RewTerm(
        func=mdp.closed_loop_v2_durable_recovery_progress,
        weight=0.0,
        params={"command_name": "racket_target"},
    )
    durable_recovery_success = RewTerm(
        func=mdp.closed_loop_v2_durable_recovery_success,
        weight=0.0,
        params={"command_name": "racket_target", "impulse": True},
    )
    durable_recovery_failure = RewTerm(
        func=mdp.closed_loop_v2_durable_recovery_failure,
        weight=0.0,
        params={"command_name": "racket_target", "impulse": True},
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
            "minimum_health": 0.45,
            "maximum_target_distance_from_base": 1.40,
            "impulse": True,
        },
    )


@configclass
class HOPEStage2RewardV11EnvCfg(HOPEStage2PhysicalEnvCfg):
    """Auditable physical task that cannot farm READY while skipping impact."""

    rewards: Stage2RewardV11RewardsCfg = Stage2RewardV11RewardsCfg()
    pre_reward_command_snapshot_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 14.0

        motion = self.commands.motion
        motion.wrap_teleport = False
        motion.hold_steps_range = (30, 50)
        motion.stand_start_prob = 0.25
        motion.stand_episode_prob = 0.08
        motion.core_clip_count = 2
        motion.clip_sampling_weights = (0.5, 0.5)

        target = self.commands.racket_target
        # Keep command geometry fixed during the first V11 screen. Independent
        # route, planner noise, and recovery curricula are promoted separately.
        target.ability_curriculum_enabled = False
        target.table_workspace_fixed_level = 0.0
        target.table_workspace_motion_seed_blend_start = 0.0
        target.table_workspace_motion_seed_end_level = 0.0
        target.planner_perturb_curriculum_source = "fixed"
        target.planner_perturb_fixed_scale = 0.0
        target.impact_inverse_command_curriculum_enabled = False
        target.impact_inverse_command_start_blend = 1.0

        # A targeted attempt opens recovery even when it misses the physical
        # ball. Durable settlement is measured at a fixed post-impact delay.
        target.post_contact_ready_enabled = True
        target.post_contact_ready_trigger = "targeted_attempt"
        target.post_contact_ready_progress_error_mode = "bounded_interval"
        target.post_contact_ready_curriculum_enabled = True
        target.post_contact_ready_curriculum_start_level = 0
        target.post_contact_ready_curriculum_torso_x_min = (
            -0.10,
            -0.075,
            -0.05,
            -0.035,
        )
        target.post_contact_ready_curriculum_torso_x_max = (
            0.38,
            0.30,
            0.22,
            0.14,
        )
        target.post_contact_ready_curriculum_max_torso_ang_vel = (
            0.90,
            0.78,
            0.68,
            0.60,
        )
        target.post_contact_ready_curriculum_max_base_lin_vel = (
            0.22,
            0.20,
            0.19,
            0.18,
        )
        target.post_contact_ready_curriculum_max_base_ang_vel = (
            1.00,
            0.82,
            0.68,
            0.55,
        )
        target.post_contact_ready_curriculum_max_racket_speed = (
            1.20,
            1.05,
            0.95,
            0.90,
        )
        target.post_contact_ready_curriculum_min_feet_contact = (
            0.90,
            0.95,
            0.98,
            0.99,
        )
        target.post_contact_ready_curriculum_required_consecutive_steps = (
            3,
            5,
            7,
            10,
        )
        target.post_contact_ready_curriculum_deadline_steps = (
            90,
            85,
            80,
            70,
        )
        target.post_contact_ready_curriculum_advance_success_thresholds = (
            0.20,
            0.35,
            0.50,
        )
        target.post_contact_ready_curriculum_shadow_success_thresholds = (
            0.08,
            0.15,
            0.25,
        )
        target.post_contact_ready_curriculum_min_resolved_events = (
            512,
            1024,
            2048,
        )
        target.post_contact_ready_curriculum_min_targeted_attempt_ema = 0.12
        target.post_contact_ready_curriculum_min_return_success_ema = 0.005
        target.post_contact_ready_curriculum_min_completed_swings = 512
        target.post_contact_ready_curriculum_required_advance_checks = 2
        target.post_contact_ready_required_consecutive_steps = 6
        target.post_contact_ready_deadline_steps = 80
        target.post_contact_ready_durable_diagnostic_enabled = True
        target.post_contact_ready_durable_min_delay_s = 0.45
        target.post_contact_ready_durable_deadline_s = 1.05
        target.post_contact_ready_durable_required_consecutive_steps = 10
        target.post_contact_ready_durable_use_effective_gate = True

        physical = self.commands.physical_shadow
        physical.route_geometry_mode = "independent"
        physical.route_ability_curriculum_enabled = False
        physical.pre_bounce_time_range = (0.50, 0.62)
        physical.post_bounce_time_range = (0.34, 0.46)
        physical.bounce_dx_range = (0.30, 0.68)
        physical.bounce_y_jitter_range = (-0.10, 0.10)
        physical.route_sample_attempts = 8
        physical.route_batch_interval_steps = 1
        physical.post_bounce_command_refresh_enabled = True
        physical.command_refresh_continuous_post_bounce = True
        physical.command_refresh_freeze_tts_s = 0.20
        physical.command_refresh_solution_blend = 1.0
        physical.command_refresh_update_dynamic_station = True

        _zero_all_reward_terms(self.rewards)

        # Motion is a posture/coordination prior only. Waist, legs, and racket
        # execution remain under task and physical-outcome authority.
        self.rewards.imitation.weight = 0.50
        self.rewards.imitation.params.update(
            {
                "body_names": list(STAGE1_ARM_PRIOR_BODIES),
                "racket_command_name": "racket_target",
                "pre_strike_scale": 0.70,
                "strike_scale": 0.18,
                "recovery_scale": 0.45,
            }
        )

        # One task-space objective plus non-farmable approach/relocation terms.
        self.rewards.planner_racket_task_space_crossfade.weight = 2.50
        self.rewards.planner_racket_task_space_crossfade.params.update(
            {
                "ability_scaled_stds": False,
                "position_std": 0.18,
                "velocity_std": 1.15,
                "normal_std_rad": 0.45,
                "component_floor": 0.08,
                "minimum_health_multiplier": 0.30,
                "final_minimum_health_multiplier": 0.30,
                "action_feasibility_floor": 0.30,
            }
        )
        self.rewards.prestrike_racket_progress.weight = 0.65
        self.rewards.prestrike_station_progress.weight = 0.25

        # Real PhysX events are impulses. Immediate contact value is deliberately
        # small; most return value is paid only after durable recovery.
        # Contact and command alignment remain immediate exploration signals.
        # Net/bounce carry only small provisional value here; their large value
        # is settled below only after fixed-deadline durable recovery.
        self.rewards.physical_outcome_events.weight = 1.50
        self.rewards.physical_outcome_events.params.update(
            {
                "contact_scale": 0.25,
                "contact_quality_scale": 0.50,
                "net_cross_scale": 0.25,
                "opponent_bounce_scale": 0.50,
                "impulse": True,
            }
        )
        self.rewards.physical_contact_planner_alignment.weight = 3.00
        self.rewards.physical_contact_planner_alignment.params.update(
            {
                "position_std": 0.12,
                "velocity_std": 1.00,
                "direction_std_deg": 28.0,
                "normal_std_deg": 24.0,
                "timing_std_s": 0.09,
                "impulse": True,
            }
        )
        self.rewards.physical_recovery_settlement.weight = 1.00
        self.rewards.physical_recovery_settlement.params.update(
            {
                "ready_threshold": 0.55,
                "minimum_resolved_steps": 3,
                "required_ready_steps": 6,
                "deadline_steps": 60,
                "contact_value": 1.0,
                "net_cross_value": 3.0,
                "opponent_bounce_value": 6.0,
                "failure_cost": 0.50,
                "terminal_failure_cost": 1.0,
                "impulse": True,
                "require_durable_recovery": True,
                "require_safe_settlement": True,
            }
        )

        # Recovery has bounded directional shaping and one-shot resolution.
        # No positive per-frame READY reward is active.
        self.rewards.durable_recovery_progress.weight = 0.35
        # A missed token swing must not earn a positive closed-loop return.
        # Miss recovery still receives bounded directional shaping and avoids
        # the failure debit. Contact/net/bounce value is owned exclusively by
        # the safe physical settlement above.
        self.rewards.durable_recovery_success.weight = 0.0
        self.rewards.durable_recovery_failure.weight = -1.25
        self.rewards.no_command_instability.weight = -0.30
        self.rewards.safe_strike_inactivity.weight = -0.40
        self.rewards.healthy_trunk_support.weight = 0.30
        self.rewards.healthy_trunk_support.params[
            "include_no_command_ready"
        ] = False
        self.rewards.lower_body_support.weight = 0.25

        # Operational costs preserve feasible leg compensation and release the
        # right arm around impact instead of globally stiffening the robot.
        self.rewards.operational_joint_margin.weight = -0.60
        self.rewards.joint_target_slew.weight = -0.10
        self.rewards.actuator_waist_feasibility.weight = -0.025
        self.rewards.actuator_right_arm_feasibility.weight = -0.008
        self.rewards.actuator_leg_feasibility.weight = -0.020
        self.rewards.phase_action_overflow.weight = -0.20
        self.rewards.phase_action_rate_waist.weight = -0.030
        self.rewards.phase_action_rate_upper.weight = -0.002
        self.rewards.phase_action_rate_legs.weight = -0.006
        self.rewards.joint_limit.weight = -1.00
        self.rewards.undesired_contacts.weight = -0.10
        self.rewards.feet_contact_slip.weight = -0.05
        self.rewards.table_no_touch.weight = -0.75

        self.rewards.alive.weight = 0.01
        # At 50 Hz RewardManager integrates this pulse by 0.02. The resulting
        # hard failure cost is about -5, while the termination itself is kept.
        self.rewards.termination_penalty.weight = -250.0
        self.rewards.termination_penalty.params["term_keys"] = [
            "base_too_low",
            "base_tilted",
            "table_touch",
            "persistent_action_overflow",
        ]

        # Keep the physical controller alive for diagnostics, but freeze all
        # difficulty promotion during the first screen.
        self.rewards.physical_capability_curriculum.weight = 1.0
        self.rewards.physical_capability_curriculum.params["fixed_level"] = 0.0
