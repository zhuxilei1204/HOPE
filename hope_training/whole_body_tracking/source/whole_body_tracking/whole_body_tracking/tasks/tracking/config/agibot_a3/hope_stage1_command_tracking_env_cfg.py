"""Stage-1 planner-command tracking foundation for Agibot A3.

This task deliberately excludes ball contact outcomes and long-horizon cycle
settlement. It first teaches one policy to remain deployable while executing a
small, feasible forehand/backhand racket-command distribution.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_closed_loop_v2_env_cfg import (
    _zero_all_reward_terms,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
    HOPEPingPongEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)


# The audited clips keep torso pitch below 8 degrees with low angular velocity,
# so torso is a useful healthy swing prior. Waist, wrists, and legs remain free;
# task-space commands own the paddle and the lower body owns balance corrections.
STAGE1_ARM_PRIOR_BODIES = (
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
)


@configclass
class Stage1CommandTrackingRewardsCfg(RewardsCfg):
    prestrike_racket_progress = RewTerm(
        func=mdp.prestrike_racket_progress,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "speed_scale": 0.60,
            "arrival_radius": 0.18,
            "stop_window_s": 0.20,
            "path_lookahead_s": 0.35,
            "idle_cost": 0.05,
            "velocity_frame": "torso_relative",
            "minimum_health_multiplier": 0.35,
            "final_minimum_health_multiplier": 0.35,
        },
    )
    prestrike_station_progress = RewTerm(
        func=mdp.nonfarmable_prestrike_station_progress,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "speed_scale": 0.25,
            "arrival_radius": 0.04,
            "stop_window_s": 0.20,
            "idle_cost": 0.02,
            "minimum_health_multiplier": 0.40,
            "final_minimum_health_multiplier": 0.40,
            "include_no_command_ready": False,
        },
    )
    near_impact_planner_velocity_progress = RewTerm(
        func=mdp.near_impact_planner_velocity_progress,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "pre_start_s": 0.24,
            "pre_full_s": 0.10,
            "post_full_s": 0.04,
            "post_end_s": 0.08,
            "position_std": 0.32,
            "position_floor": 0.25,
            "normal_std_rad": 0.75,
            "normal_floor": 0.25,
            "projection_ratio_scale": 0.65,
            "lateral_ratio_scale": 0.55,
            "lateral_weight": 0.20,
            "idle_cost": 0.04,
            "minimum_health_multiplier": 0.30,
            "final_minimum_health_multiplier": 0.30,
        },
    )


@configclass
class Stage1CommandTrackingTerminationsCfg(TerminationsCfg):
    persistent_action_overflow = DoneTerm(
        func=mdp.persistent_action_overflow,
        params={
            "command_name": "racket_target",
            "enabled": True,
            "min_steps": 25,
        },
    )


@configclass
class HOPEStage1CommandTrackingEnvCfg(HOPEPingPongEnvCfg):
    """Ten-second, continuous, ball-free command-tracking foundation."""

    rewards: Stage1CommandTrackingRewardsCfg = Stage1CommandTrackingRewardsCfg()
    terminations: Stage1CommandTrackingTerminationsCfg = (
        Stage1CommandTrackingTerminationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0

        motion = self.commands.motion
        motion.wrap_teleport = False
        motion.hold_steps_range = (15, 30)
        motion.stand_start_prob = 0.25
        motion.stand_start_min_hold = 20
        motion.stand_episode_prob = 0.08
        motion.stand_episode_hold_steps = 100000
        motion.core_clip_count = 2
        motion.clip_sampling_weights = (0.5, 0.5)
        motion.batched_metric_reset_logging = True

        # Exactly two reset modes: clean clip start or deploy-ready stand. The
        # equal start/min probabilities disable capability coupling and remove
        # random mid-clip RSI from this first-stage distribution.
        motion.motion_start_warmup_enabled = True
        motion.motion_start_warmup_start_prob = 0.75
        motion.motion_start_warmup_min_prob = 0.75
        motion.motion_start_warmup_prestrike_enabled = False
        motion.motion_start_warmup_recovery_enabled = False
        motion.motion_start_warmup_lifecycle_curriculum_enabled = False
        motion.stand_start_pose_range = {
            "x": (-0.015, 0.015),
            "y": (-0.015, 0.015),
            "z": (-0.004, 0.004),
            "roll": (-0.035, 0.035),
            "pitch": (-0.050, 0.050),
            "yaw": (-0.035, 0.035),
        }
        motion.stand_start_velocity_range = {
            "x": (-0.08, 0.08),
            "y": (-0.06, 0.06),
            "z": (-0.02, 0.02),
            "roll": (-0.16, 0.16),
            "pitch": (-0.22, 0.22),
            "yaw": (-0.12, 0.12),
        }
        motion.stand_start_joint_position_range = (-0.012, 0.012)

        command = self.commands.racket_target
        command.station_mode = "dynamic_from_motion"
        command.dynamic_station_xy_clip = ((-0.03, 0.03), (-0.08, 0.08))
        command.dynamic_station_blend = 1.0
        command.deploy_ready_hold_prob = 0.20
        command.deploy_ready_force_stand_episode = True
        command.deploy_ready_force_default_stand_reset = True
        command.station_relocation_enabled = False
        command.lifecycle_recovery_hold_gate_enabled = False
        command.healthy_three_stage_enabled = False
        command.cycle_v2_enabled = False
        command.single_cycle_curriculum_enabled = False
        command.post_contact_ready_enabled = False
        command.post_contact_ready_curriculum_enabled = False
        command.ability_curriculum_enabled = False

        # Fixed feasible Level-0 command set. Position boxes are overridden by
        # the audited two-motion manifest; velocity and normal remain coupled.
        command.strike_position_mode = "motion_box"
        command.planner_hit_plane_mode = "motion_box"
        command.racket_pos_curriculum_steps = 0
        command.racket_velocity_mode = "range"
        command.racket_vel_range_per_clip = (
            ((1.20, 2.20), (0.35, 1.10), (0.15, 0.75)),
            ((1.20, 2.20), (-1.00, -0.25), (0.10, 0.70)),
        )
        command.planner_command_mode = "v4_wire_compatible"
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
        command.incoming_trajectory_mode = "direct"
        command.batched_metric_reset_logging = True

        _zero_all_reward_terms(self.rewards)
        self.rewards.alive.weight = 0.02
        self.rewards.upright.weight = -0.40
        self.rewards.imitation.weight = 1.20
        self.rewards.imitation.params.update(
            {
                "body_names": list(STAGE1_ARM_PRIOR_BODIES),
                "racket_command_name": "racket_target",
                "pre_strike_scale": 0.85,
                "strike_scale": 0.45,
                "recovery_scale": 0.70,
            }
        )
        self.rewards.planner_racket_task_space_crossfade.weight = 4.0
        self.rewards.planner_racket_task_space_crossfade.params.update(
            {
                "pre_start_s": 0.42,
                "pre_full_s": 0.16,
                "post_full_s": 0.08,
                "post_end_s": 0.22,
                "position_std": 0.28,
                "velocity_std": 1.80,
                "normal_std_rad": 0.70,
                "component_floor": 0.12,
                "minimum_health_multiplier": 0.35,
                "final_minimum_health_multiplier": 0.35,
            }
        )
        self.rewards.prestrike_racket_progress.weight = 1.20
        self.rewards.near_impact_planner_velocity_progress.weight = 1.50
        self.rewards.racket_position.weight = 0.80
        self.rewards.racket_position.params.update(
            {
                "std": 0.30,
                "minimum_health_multiplier": 0.35,
                "final_minimum_health_multiplier": 0.35,
            }
        )
        self.rewards.racket_velocity.weight = 0.45
        self.rewards.racket_velocity.params.update(
            {
                "std": 2.00,
                "minimum_health_multiplier": 0.35,
                "final_minimum_health_multiplier": 0.35,
            }
        )
        self.rewards.blade_direction.weight = 0.45
        self.rewards.blade_direction.params.update(
            {
                "std": 0.85,
                "minimum_health_multiplier": 0.35,
                "final_minimum_health_multiplier": 0.35,
            }
        )
        self.rewards.prestrike_station_progress.weight = 0.45
        self.rewards.pre_strike_station.weight = 0.15
        self.rewards.pre_strike_station.params.update(
            {"station_std": 0.16, "stop_window_s": 0.12}
        )
        self.rewards.healthy_trunk_support.weight = 0.80
        self.rewards.lower_body_support.weight = 0.65
        self.rewards.recovery_health.weight = 0.35
        self.rewards.recovery_health.params.update(
            {
                "use_dynamic_station": True,
                "require_targeted_attempt": False,
            }
        )
        self.rewards.no_command_ready_stability.weight = 0.20
        self.rewards.phase_action_overflow.weight = -0.08
        self.rewards.phase_action_rate_waist.weight = -0.025
        self.rewards.phase_action_rate_waist.params.update(
            {
                "pre_strike_scale": 0.020,
                "strike_scale": 0.004,
                "recovery_scale": 0.090,
                "hold_scale": 0.120,
            }
        )
        self.rewards.phase_action_rate_upper.weight = -0.006
        self.rewards.phase_action_rate_upper.params.update(
            {
                "pre_strike_scale": 0.010,
                "strike_scale": 0.002,
                "recovery_scale": 0.050,
                "hold_scale": 0.070,
            }
        )
        self.rewards.phase_action_rate_legs.weight = -0.003
        self.rewards.phase_action_rate_legs.params.update(
            {
                "pre_strike_scale": 0.010,
                "strike_scale": 0.002,
                "recovery_scale": 0.035,
                "hold_scale": 0.050,
            }
        )
        self.rewards.joint_limit.weight = -1.0
        self.rewards.undesired_contacts.weight = -0.10
        self.rewards.feet_contact_slip.weight = -0.05
        self.rewards.table_no_touch.weight = -0.50
        self.rewards.termination_penalty.weight = -12.0
        self.rewards.termination_penalty.params["term_keys"] = [
            "base_too_low",
            "base_tilted",
            "table_touch",
            "persistent_action_overflow",
        ]

        # Physical safety only. Reference-divergence and cycle timeouts would
        # censor precisely the carried-state transitions Stage 1 must learn.
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

        # Mild first-stage DR; wider actuator/system-ID ranges belong after the
        # command skill is established.
        self.events.physics_material.params.update(
            {
                "static_friction_range": (0.85, 1.15),
                "dynamic_friction_range": (0.80, 1.10),
                "restitution_range": (0.0, 0.05),
            }
        )
        self.events.base_com.params["com_range"] = {
            "x": (-0.008, 0.008),
            "y": (-0.015, 0.015),
            "z": (-0.015, 0.015),
        }
        self.events.link_mass.params["mass_distribution_params"] = (0.97, 1.03)
        self.events.pd_gains.params["stiffness_distribution_params"] = (0.97, 1.03)
        self.events.pd_gains.params["damping_distribution_params"] = (0.97, 1.03)
