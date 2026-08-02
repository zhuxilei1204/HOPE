"""Merged command-tracking and footwork foundation on the planner hit plane.

This task is intentionally ball-free at the simulator level.  The command term
still evaluates one-shot analytic contact/return events, which provides a
stable curriculum signal without introducing rigid-ball collision variance.
The sampled target is independent of the motion box; motion supplies only an
upper-body action prior and side/timing labels.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_closed_loop_v2_env_cfg import (
    _zero_all_reward_terms,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_stage1_command_tracking_env_cfg import (
    STAGE1_ARM_PRIOR_BODIES,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_stage1_operational_env_cfg import (
    HOPEStage1OperationalEnvCfg,
)


@configclass
class HOPEStage1Plane020MergedEnvCfg(HOPEStage1OperationalEnvCfg):
    """Continuous fore/back command execution with capability-gated footwork."""

    def __post_init__(self):
        super().__post_init__()

        # Three or more carried-state swings per episode.  The rigid ball is
        # deferred to a later stage, so this remains substantially cheaper than
        # the physical task despite the ten-second horizon.
        self.episode_length_s = 10.0

        motion = self.commands.motion
        motion.wrap_teleport = False
        motion.hold_steps_range = (10, 20)
        motion.core_clip_count = 2
        motion.clip_sampling_weights = (0.5, 0.5)
        motion.stand_start_prob = 0.20
        motion.stand_start_min_hold = 15
        motion.stand_episode_prob = 0.02
        motion.stand_episode_hold_steps = 100000
        motion.motion_start_warmup_enabled = True
        motion.motion_start_warmup_start_prob = 0.80
        motion.motion_start_warmup_min_prob = 0.80
        motion.motion_start_warmup_prestrike_enabled = False
        motion.motion_start_warmup_recovery_enabled = False
        motion.motion_start_warmup_lifecycle_curriculum_enabled = False

        command = self.commands.racket_target
        command.station_mode = "dynamic_from_motion"
        # The fixed x=+0.20 m plane is about 3 cm beyond the natural racket
        # reach of both audited clips.  Lateral motion is available from the
        # first update and grows to full-table targets only after capability
        # gates pass.
        command.dynamic_station_xy_clip = ((-0.02, 0.05), (-0.42, 0.42))
        command.dynamic_station_blend = 1.0
        command.deploy_ready_hold_prob = 0.05
        command.deploy_ready_force_stand_episode = True
        command.deploy_ready_force_default_stand_reset = True

        # Keep the foundation lifecycle small.  Recovery is represented by the
        # carried post-strike state and the next sampled swing, rather than a
        # second collection of timeout/escrow state machines.
        command.station_relocation_enabled = False
        command.lifecycle_recovery_hold_gate_enabled = False
        command.healthy_three_stage_enabled = False
        command.cycle_v2_enabled = False
        command.single_cycle_curriculum_enabled = False
        command.post_contact_ready_enabled = False
        command.post_contact_ready_curriculum_enabled = False

        # HITTER-style separation: motion chooses a feasible swing prior;
        # table geometry chooses the racket command and explicit base target.
        command.strike_position_mode = "table_workspace"
        command.planner_hit_plane_mode = "fixed_x_hit"
        command.planner_hit_plane_x = 0.20
        command.planner_hit_plane_x_jitter_range = (0.0, 0.0)
        command.planner_hit_plane_blend = 1.0
        command.planner_hit_plane_blend_start = 1.0
        command.planner_hit_plane_blend_warmup_steps = 0
        # Level 0 stays inside the previous actor's roughly +/-8 cm station
        # support while still requiring nonzero relocation.  Later levels
        # interpolate these intervals to the full playable table width.
        command.table_workspace_forehand_core_y_range = (-0.53, -0.38)
        command.table_workspace_backhand_core_y_range = (-0.12, 0.00)
        command.table_workspace_x_jitter_core_range = (0.0, 0.0)
        command.table_workspace_x_jitter_full_range = (0.0, 0.0)
        command.table_workspace_z_core_above_surface_range = (0.40, 0.56)
        command.table_workspace_z_full_above_surface_range = (0.34, 0.68)
        command.table_workspace_fringe_prob = 0.0
        command.table_workspace_fixed_level = -1.0
        command.table_workspace_level_source = "ability"
        command.table_workspace_motion_seed_blend_start = 0.0
        command.table_workspace_motion_seed_end_level = 0.0

        # The command skill starts with the audited bounded racket-speed boxes.
        # Planner noise and the physical one-bounce ball remain later-stage
        # difficulties; adding them here would conflate execution and planning.
        command.racket_velocity_mode = "range"
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

        # Expand y/z only from measured virtual contact, return, recovery and
        # safety.  Each decision consumes a fresh set of resolved swings and
        # two good checks, preventing a transient spike from unlocking the
        # whole table.
        command.ability_curriculum_enabled = True
        command.ability_curriculum_external_update = False
        command.ability_curriculum_ema_rate = 0.08
        command.ability_curriculum_advance_rate = 0.10
        command.ability_curriculum_regress_rate = 0.10
        command.ability_curriculum_min_resolved_events = 4096
        command.ability_curriculum_required_advance_checks = 2
        command.ability_curriculum_required_regress_checks = 2
        command.ability_curriculum_contact_threshold = 0.35
        command.ability_curriculum_net_threshold = 0.12
        command.ability_curriculum_success_threshold = 0.05
        command.ability_curriculum_recovery_threshold = 0.60
        command.ability_curriculum_recovery_floor = 0.42
        command.ability_curriculum_outcome_regress_ratio = 0.45
        command.ability_curriculum_require_side_contact = True
        command.ability_curriculum_forehand_contact_threshold = 0.28
        command.ability_curriculum_backhand_contact_threshold = 0.28
        command.ability_curriculum_require_side_net = False
        command.ability_curriculum_require_side_success = False
        command.ability_curriculum_use_safety_gates = True
        command.ability_curriculum_use_targeted_attempt = True
        command.ability_curriculum_attempt_threshold = 0.35
        command.ability_curriculum_safety_threshold = 0.985
        command.ability_curriculum_safety_floor = 0.965
        command.ability_curriculum_station_saturation_threshold = 0.20
        command.ability_curriculum_station_saturation_regress_threshold = 0.35
        command.ability_curriculum_require_healthy_impact = True
        command.ability_curriculum_start_racket_pos_scale = 1.0
        command.ability_curriculum_start_ball_scale = 1.0
        command.ability_curriculum_start_planner_perturb_scale = 1.0
        command.batched_metric_reset_logging = True
        command.impact_health_reward_power = 2.0

        # One reviewed reward whitelist. Task-space terms are short-window or
        # one-shot; standing has no large persistent payoff. The selected
        # initialization already has contact skill, so planner and outcome
        # payoff is almost fully multiplicative with strike health. A small
        # dense floor remains only to provide a recovery path after mistakes.
        _zero_all_reward_terms(self.rewards)
        self.rewards.alive.weight = 0.01
        self.rewards.upright.weight = -0.40
        self.rewards.imitation.weight = 0.90
        self.rewards.imitation.params.update(
            {
                "body_names": list(STAGE1_ARM_PRIOR_BODIES),
                "racket_command_name": "racket_target",
                "pre_strike_scale": 0.80,
                "strike_scale": 0.40,
                "recovery_scale": 0.65,
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
                "minimum_health_multiplier": 0.10,
                "final_minimum_health_multiplier": 0.02,
                "action_feasibility_metric": (
                    "action_operational_feasibility_score"
                ),
                "action_feasibility_floor": 0.40,
            }
        )
        self.rewards.prestrike_racket_progress.weight = 0.90
        self.rewards.prestrike_racket_progress.params.update(
            {
                "minimum_health_multiplier": 0.15,
                "final_minimum_health_multiplier": 0.03,
            }
        )
        self.rewards.near_impact_planner_velocity_progress.weight = 1.10
        self.rewards.near_impact_planner_velocity_progress.params.update(
            {
                "minimum_health_multiplier": 0.08,
                "final_minimum_health_multiplier": 0.0,
            }
        )
        self.rewards.racket_position.weight = 1.20
        self.rewards.racket_position.params.update(
            {
                "std": 0.30,
                "minimum_health_multiplier": 0.10,
                "final_minimum_health_multiplier": 0.0,
            }
        )
        # Backhand is kinematically harder and was the first capability lost
        # under average-return PPO. This term is itself fully health-gated, so
        # it preserves the weak side without paying for an unsafe reach.
        self.rewards.backhand_racket_position.weight = 1.00
        self.rewards.backhand_racket_position.params.update(
            {
                "std": 0.24,
                "swing_side": -1.0,
                "minimum_health_multiplier": 0.05,
                "final_minimum_health_multiplier": 0.0,
            }
        )
        self.rewards.racket_velocity.weight = 0.80
        self.rewards.racket_velocity.params.update(
            {
                "std": 2.00,
                "minimum_health_multiplier": 0.10,
                "final_minimum_health_multiplier": 0.0,
            }
        )
        self.rewards.blade_direction.weight = 0.70
        self.rewards.blade_direction.params.update(
            {
                "std": 0.85,
                "minimum_health_multiplier": 0.10,
                "final_minimum_health_multiplier": 0.0,
            }
        )
        # This reward is one-shot and contact-gated. It prevents position,
        # speed, direction, and normal from being optimized independently into
        # a combination that cannot execute the planner command at impact.
        self.rewards.exact_impact_planner_task_space_alignment.weight = 4.0
        self.rewards.exact_impact_planner_task_space_alignment.params.update(
            {
                "position_std": 0.10,
                "speed_ratio_std": 0.25,
                "direction_std_rad": 0.30,
                "normal_std_rad": 0.24,
                "component_floor": 0.04,
                "require_contact": True,
                "minimum_health_multiplier": 0.05,
                "final_minimum_health_multiplier": 0.0,
            }
        )
        self.rewards.prestrike_station_progress.weight = 0.80
        self.rewards.pre_strike_station.weight = 0.20
        self.rewards.pre_strike_station.params.update(
            {"station_std": 0.14, "stop_window_s": 0.12}
        )

        self.rewards.health_gated_soft_ball_contact.weight = 1.20
        self.rewards.health_gated_soft_ball_contact.params.update(
            {
                "minimum_health_multiplier": 0.20,
                "final_minimum_health_multiplier": 0.02,
                "pos_std": 0.20,
                "window_s": 0.20,
            }
        )
        self.rewards.health_gated_backhand_soft_ball_contact.weight = 1.20
        self.rewards.health_gated_backhand_soft_ball_contact.params.update(
            {
                "minimum_health_multiplier": 0.05,
                "final_minimum_health_multiplier": 0.0,
                "pos_std": 0.21,
                "window_s": 0.22,
                "swing_side": -1.0,
            }
        )
        self.rewards.targeted_strike_attempt.weight = 2.0
        self.rewards.targeted_strike_attempt.params[
            "minimum_health_multiplier"
        ] = 0.05
        self.rewards.strike_inactivity.weight = -0.60
        self.rewards.health_gated_ball_contact.weight = 6.0
        self.rewards.health_gated_ball_contact.params.update(
            {
                "minimum_health_multiplier": 0.03,
                "final_minimum_health_multiplier": 0.0,
            }
        )
        self.rewards.health_gated_net_cross.weight = 2.5
        self.rewards.health_gated_net_cross.params.update(
            {
                "minimum_health_multiplier": 0.0,
                "final_minimum_health_multiplier": 0.0,
            }
        )
        self.rewards.health_gated_opponent_bounce.weight = 2.0
        self.rewards.health_gated_opponent_bounce.params.update(
            {
                "minimum_health_multiplier": 0.0,
                "final_minimum_health_multiplier": 0.0,
            }
        )

        # Balance shaping is local to the swing or weak during recovery.  It
        # never constrains the racket arm, and the lower body has no motion
        # prior, so stepping and fast stabilizing corrections remain available.
        self.rewards.healthy_trunk_support.weight = 0.40
        self.rewards.healthy_trunk_support.params[
            "include_no_command_ready"
        ] = True
        self.rewards.lower_body_support.weight = 0.25
        self.rewards.strike_balance.weight = 0.30
        self.rewards.post_strike_base_ang_vel.weight = 0.06
        self.rewards.recovery_health.weight = 0.12
        self.rewards.recovery_health.params.update(
            {
                "use_dynamic_station": False,
                "require_targeted_attempt": False,
            }
        )
        self.rewards.no_command_ready_stability.weight = 0.08

        # Existing operational envelope.  Strike scales remain very small for
        # the legs and racket arm; infeasible clipping is costly in every phase.
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
        self.rewards.table_no_touch.weight = -1.00
        self.rewards.termination_penalty.weight = -60.0
        self.rewards.termination_penalty.params["term_keys"] = [
            "base_too_low",
            "base_tilted",
            "table_touch",
            "persistent_action_overflow",
        ]

        # Hard safety remains identical to the audited operational foundation.
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
