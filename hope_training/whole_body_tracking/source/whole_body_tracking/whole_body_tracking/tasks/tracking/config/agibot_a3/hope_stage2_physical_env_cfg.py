"""Stage-2 rigid-ball task initialized from the Stage-1 operational policy."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.table_tennis.ball import BallAerodynamicsCfg

from .hope_env_cfg import CommandsCfg
from .hope_physical_eval_env_cfg import (
    HOPEPhysicalShadowSceneCfg,
    _configure_physical_eval,
)
from .hope_stage1_operational_env_cfg import (
    HOPEStage1OperationalEnvCfg,
    Stage1OperationalRewardsCfg,
)


@configclass
class Stage2PhysicalCommandsCfg(CommandsCfg):
    """Policy commands plus an unobserved rigid-ball lifecycle."""

    physical_shadow = mdp.PhysicalBallShadowCommandCfg(
        ball_asset_name="ball",
        target_command_name="racket_target",
        route_geometry_mode="target_hidden",
        pre_bounce_time_range=(0.48, 0.64),
        post_bounce_time_range=(0.28, 0.46),
        route_batch_interval_steps=3,
        debug_vis=False,
    )


@configclass
class Stage2PhysicalRewardsCfg(Stage1OperationalRewardsCfg):
    """Stage-1 shaping plus real PhysX outcome and settlement value."""

    physical_outcome_events = RewTerm(
        func=mdp.physical_outcome_events,
        weight=3.0,
        params={
            "physical_command_name": "physical_shadow",
            "target_command_name": "racket_target",
            "contact_scale": 1.0,
            "net_cross_scale": 3.0,
            "opponent_bounce_scale": 6.0,
            "contact_quality_scale": 1.5,
            "outgoing_velocity_error_std": 2.0,
            "outgoing_direction_error_std_deg": 45.0,
            "face_inner_radius": 0.040,
            "face_outer_radius": 0.095,
            "minimum_face_multiplier": 0.05,
            "minimum_health_multiplier": 0.05,
            "minimum_recovery_multiplier": 0.40,
            # Keep physical outcome feedback active while the impact command
            # crossfades away from the motion-derived bootstrap.
            "impact_inverse_quality_floor": 0.30,
        },
    )
    physical_contact_planner_alignment = RewTerm(
        func=mdp.physical_contact_planner_alignment,
        weight=0.0,
        params={
            "physical_command_name": "physical_shadow",
            "target_command_name": "racket_target",
            "position_std": 0.10,
            "velocity_std": 0.65,
            "direction_std_deg": 18.0,
            "normal_std_deg": 15.0,
            "timing_std_s": 0.06,
            "component_floor": 0.04,
            "face_inner_radius": 0.040,
            "face_outer_radius": 0.095,
            "minimum_face_multiplier": 0.05,
            "minimum_health_multiplier": 0.15,
        },
    )
    physical_recovery_settlement = RewTerm(
        func=mdp.physical_outcome_recovery_settlement,
        weight=2.0,
        params={
            "physical_command_name": "physical_shadow",
            "target_command_name": "racket_target",
            "ready_threshold": 0.62,
            "minimum_resolved_steps": 3,
            "required_ready_steps": 5,
            "deadline_steps": 55,
            "contact_value": 1.0,
            "net_cross_value": 2.0,
            "opponent_bounce_value": 4.0,
            "failure_cost": 0.20,
            # Negative preserves the historical shared cost. V10 separates a
            # post-contact hard reset from an ordinary recovery timeout.
            "terminal_failure_cost": -1.0,
        },
    )
    physical_capability_curriculum = RewTerm(
        func=mdp.physical_capability_curriculum,
        # The controller returns zero. A nonzero weight keeps the manager term
        # alive when runtime zero-weight pruning is enabled.
        weight=1.0,
        params={
            "physical_command_name": "physical_shadow",
            "target_command_name": "racket_target",
            "minimum_events": 512,
            "ema_rate": 0.35,
            # Contact capability means usable face-center contact, not any
            # PhysX collision with the racket rim.
            "contact_threshold": 0.28,
            # Negative bootstrap preserves the historical constant threshold.
            # V6 tasks override it with a measured low-to-high schedule.
            "bootstrap_contact_threshold": -1.0,
            "full_contact_threshold_level": 1.0,
            "aligned_contact_threshold": 0.0,
            "bootstrap_aligned_contact_threshold": 0.0,
            "full_aligned_contact_threshold_level": 1.0,
            "net_threshold": 0.25,
            "bounce_threshold": 0.10,
            "recovery_threshold": 0.52,
            "bootstrap_recovery_threshold": 0.30,
            "full_recovery_threshold_level": 0.70,
            "safety_threshold": 0.58,
            "recovery_floor": 0.38,
            "safety_floor": 0.45,
            # Only the first level increment can bootstrap on contact alone.
            # Every later increment must preserve an increasing net rate.
            "contact_only_until_level": 0.0,
            "net_full_threshold_level": 0.50,
            "net_only_until_level": 0.45,
            "bounce_full_threshold_level": 1.0,
            "contact_regress_ratio": 0.80,
            "aligned_contact_regress_ratio": 0.0,
            "net_regress_ratio": 0.60,
            "bounce_regress_ratio": 0.60,
            "center_contact_radius": 0.061,
            "aligned_position_error_max": 0.14,
            "aligned_velocity_error_max": 1.25,
            "aligned_velocity_direction_error_max_deg": 40.0,
            "aligned_normal_error_max_deg": 32.0,
            "aligned_timing_error_max_s": 0.07,
            "level_step": 0.05,
            "regress_step": 0.10,
            "required_advance_checks": 3,
            "required_regress_checks": 2,
            # -1 enables the event-driven controller. Calibration experiments
            # can hold a level while retaining the same event EMA diagnostics.
            "fixed_level": -1.0,
        },
    )


@configclass
class HOPEStage2PhysicalEnvCfg(HOPEStage1OperationalEnvCfg):
    """Single-policy one-bounce task with physics-owned outcome rewards."""

    scene: HOPEPhysicalShadowSceneCfg = HOPEPhysicalShadowSceneCfg(
        num_envs=256, env_spacing=6.0
    )
    commands: Stage2PhysicalCommandsCfg = Stage2PhysicalCommandsCfg()
    rewards: Stage2PhysicalRewardsCfg = Stage2PhysicalRewardsCfg()
    ball_aerodynamics: BallAerodynamicsCfg = (
        BallAerodynamicsCfg.from_physics_config(enabled=True)
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_physical_eval(self)
        self.episode_length_s = 12.0

        motion = self.commands.motion
        motion.hold_steps_range = (30, 50)
        motion.stand_start_prob = 0.25
        motion.stand_episode_prob = 0.05

        command = self.commands.racket_target
        command.strike_position_mode = "table_workspace"
        command.table_workspace_fixed_level = -1.0
        command.table_workspace_level_source = "ability"
        command.table_workspace_curriculum_start_level = 0.10
        command.table_workspace_curriculum_full_level = 1.0
        # Level zero exactly preserves the Stage-1 endpoint distribution. The
        # physical ball route remains independent and ability unlocks the
        # planner/table workspace after the policy adapts to URDF dynamics.
        command.table_workspace_motion_seed_blend_start = 1.0
        command.table_workspace_motion_seed_end_level = 0.90
        command.table_workspace_fringe_prob = 0.0
        command.table_workspace_forehand_core_y_range = (-0.42, -0.30)
        command.table_workspace_backhand_core_y_range = (-0.02, 0.10)
        command.table_workspace_x_jitter_core_range = (-0.008, 0.008)
        command.table_workspace_z_core_above_surface_range = (0.46, 0.54)
        command.planner_hit_plane_x = 0.20

        command.racket_velocity_mode = "impact_inverse_landing"
        command.impact_inverse_command_curriculum_enabled = True
        command.impact_inverse_command_start_blend = 0.30
        command.impact_inverse_command_curriculum_exponent = 1.5
        command.impact_inverse_command_curriculum_start_level = 0.0
        command.impact_inverse_command_curriculum_full_level = 0.75
        command.incoming_trajectory_mode = "one_bounce"
        command.one_bounce_speed_curriculum_enabled = True
        command.one_bounce_speed_curriculum_start_level = 0.60
        command.one_bounce_speed_curriculum_full_level = 1.0
        command.one_bounce_easy_horizontal_speed_range = (0.90, 1.25)
        command.one_bounce_full_horizontal_speed_range = (0.80, 2.20)
        command.one_bounce_easy_post_time_range = (0.40, 0.46)
        command.strike_window_s = 0.12

        # Full ranges represent deployment uncertainty. At level zero the
        # external physical curriculum removes this noise and then expands it.
        command.planner_target_pos_offset_range = (
            (-0.012, 0.012),
            (-0.018, 0.018),
            (-0.015, 0.015),
        )
        command.planner_time_to_strike_offset_range = (-0.025, 0.025)
        command.planner_target_vel_scale_range = (0.92, 1.08)
        command.planner_target_vel_offset_range = (
            (-0.12, 0.12),
            (-0.12, 0.12),
            (-0.10, 0.10),
        )
        command.planner_target_vel_yaw_deg_range = (-4.0, 4.0)
        command.planner_perturb_curriculum_source = "ability"
        command.planner_perturb_curriculum_start_level = 0.75
        command.planner_perturb_curriculum_full_level = 1.0

        command.ability_curriculum_enabled = True
        command.ability_curriculum_external_update = True
        command.ability_curriculum_start_racket_pos_scale = (
            0.35,
            0.35,
            0.40,
        )
        command.ability_curriculum_start_ball_scale = 0.35
        command.ability_curriculum_start_planner_perturb_scale = 0.0

        # Stage 2 remains a sequence of independently resolved shots. Long
        # continuous-rally success becomes the Stage-3 task contract.
        command.cycle_v2_enabled = False
        command.single_cycle_curriculum_enabled = False
        command.post_contact_ready_enabled = False

        # Keep the Stage-1 skill visible while giving physical outcomes enough
        # value to correct impact speed and face direction.
        # Stage1CommandTracking clears every reward term during its own
        # post-init, including subclass terms, so restore Stage-2 terms here.
        self.rewards.physical_outcome_events.weight = 3.0
        self.rewards.physical_recovery_settlement.weight = 2.0
        self.rewards.physical_capability_curriculum.weight = 1.0
        self.rewards.planner_racket_task_space_crossfade.weight = 3.5
        self.rewards.planner_racket_task_space_crossfade.params.update(
            {
                "ability_scaled_stds": True,
                "initial_position_std": 0.18,
                "initial_velocity_std": 1.20,
                "initial_normal_std_rad": 0.50,
                "position_std": 0.07,
                "velocity_std": 0.65,
                "normal_std_rad": 0.22,
            }
        )
        self.rewards.prestrike_racket_progress.weight = 1.0
        self.rewards.near_impact_planner_velocity_progress.weight = 1.25
        self.rewards.recovery_health.weight = 0.55
        self.rewards.strike_balance.weight = 0.22
        self.rewards.post_strike_base_ang_vel.weight = 0.14
