"""Plane020 foundation with delayed, safety-conditioned outcome credit."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_stage1_operational_env_cfg import (
    Stage1OperationalRewardsCfg,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_stage1_plane020_merged_env_cfg import (
    HOPEStage1Plane020MergedEnvCfg,
)


@configclass
class Plane020EscrowRewardsCfg(Stage1OperationalRewardsCfg):
    """Recovery escrow terms kept separate from the reviewed merged baseline."""

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
            "tier_multipliers": (0.40, 1.00, 1.50),
        },
    )
    capability_gated_safe_terminal_outcome = RewTerm(
        func=mdp.capability_gated_safe_terminal_outcome,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "contact_value": 0.80,
            "net_extra_value": 0.30,
            "bounce_extra_value": 0.40,
        },
    )
    targeted_contact_miss = RewTerm(
        func=mdp.targeted_contact_miss,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "minimum_health_multiplier": 0.50,
            "impulse": False,
        },
    )
    safe_strike_inactivity = RewTerm(
        func=mdp.closed_loop_v2_safe_inactivity,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "minimum_health": 0.45,
            "maximum_target_distance_from_base": 1.40,
            "impulse": False,
        },
    )


@configclass
class HOPEStage1Plane020EscrowEnvCfg(HOPEStage1Plane020MergedEnvCfg):
    """Delay high-value outcomes until a reusable post-impact state exists."""

    rewards: Plane020EscrowRewardsCfg = Plane020EscrowRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.racket_target
        command.post_contact_ready_enabled = True
        command.post_contact_ready_trigger = "contact"
        command.post_contact_ready_curriculum_enabled = False
        command.post_contact_ready_progress_error_mode = "bounded_interval"

        # The escrow gate is deliberately easier than the final deployment
        # READY gate. It asks for a safe, reusable state 0.35--0.70 s after
        # contact without prescribing one rigid arm or lower-body posture.
        command.post_contact_ready_torso_x_min = -0.12
        command.post_contact_ready_torso_x_max = 0.32
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
        command.post_contact_ready_required_consecutive_steps = 3
        command.post_contact_ready_deadline_steps = 40

        command.post_contact_ready_durable_diagnostic_enabled = True
        command.post_contact_ready_durable_min_delay_s = 0.35
        command.post_contact_ready_durable_deadline_s = 0.70
        command.post_contact_ready_durable_required_consecutive_steps = 5
        command.post_contact_ready_durable_use_effective_gate = False
        command.post_contact_ready_diagnostic_horizon_s = 0.75

        # A transient compensation is allowed, but a genuinely dangerous
        # trajectory cannot settle the escrow even if its final frame happens
        # to cross the READY region.
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

        # Contact remains available as a low-value exploration signal. Most
        # net/bounce value is paid by safe_terminal_outcome after escrow.
        self.rewards.health_gated_ball_contact.weight = 2.50
        self.rewards.health_gated_net_cross.weight = 0.50
        self.rewards.health_gated_opponent_bounce.weight = 0.25
        self.rewards.terminal_quality_window.weight = 0.40
        self.rewards.safe_terminal_quality.weight = 1.00
        self.rewards.unsafe_terminal_recovery.weight = -0.50
        self.rewards.safe_terminal_outcome.weight = 8.00
        self.rewards.safe_terminal_outcome.params.update(
            {"tier_multipliers": (0.40, 1.00, 1.50)}
        )


@configclass
class HOPEStage1Plane020EscrowGuardedEnvCfg(
    HOPEStage1Plane020EscrowEnvCfg
):
    """Protect foundation capability while higher return tiers improve."""

    rewards: Plane020EscrowRewardsCfg = Plane020EscrowRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        command = self.commands.racket_target
        command.safe_outcome_capability_gate_enabled = True
        # These are preservation floors, not new optimization targets.  The
        # frozen model_1800 audit sits above every high threshold, while each
        # previously observed long-run collapse crosses at least one low
        # threshold.  Keeping the normal policy near one avoids injecting a
        # noisy global reward scale from short-batch safety fluctuations.
        command.safe_outcome_gate_contact_low = 0.50
        command.safe_outcome_gate_contact_high = 0.60
        command.safe_outcome_gate_forehand_low = 0.58
        command.safe_outcome_gate_forehand_high = 0.68
        command.safe_outcome_gate_backhand_low = 0.38
        command.safe_outcome_gate_backhand_high = 0.50
        command.safe_outcome_gate_safety_low = 0.89
        command.safe_outcome_gate_safety_high = 0.92
        command.safe_outcome_gate_recovery_low = 0.74
        command.safe_outcome_gate_recovery_high = 0.79
        command.safe_outcome_gate_settlement_low = 0.60
        command.safe_outcome_gate_settlement_high = 0.67

        self.rewards.health_gated_net_cross.weight = 0.0
        self.rewards.health_gated_opponent_bounce.weight = 0.0
        self.rewards.safe_terminal_outcome.weight = 0.0
        self.rewards.capability_gated_safe_terminal_outcome.weight = 8.0


@configclass
class HOPEStage1Plane020EscrowImpulseEnvCfg(
    HOPEStage1Plane020EscrowGuardedEnvCfg
):
    """Use per-transition units for sparse closed-loop resolution events."""

    rewards: Plane020EscrowRewardsCfg = Plane020EscrowRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # RewardManager integrates every term by step_dt. These one-frame
        # events divide by step_dt so their configured weights are actual
        # transition values, matching the reviewed Stage-2 physical contract.
        self.rewards.health_gated_ball_contact.weight = 0.0
        self.rewards.targeted_strike_attempt.weight = 0.0
        self.rewards.strike_inactivity.weight = 0.0
        self.rewards.safe_terminal_quality.weight = 0.0

        self.rewards.capability_gated_safe_terminal_outcome.weight = 1.0
        self.rewards.capability_gated_safe_terminal_outcome.params.update(
            {
                "contact_value": 0.50,
                "net_extra_value": 1.50,
                "bounce_extra_value": 3.00,
                "impulse": True,
            }
        )
        self.rewards.unsafe_terminal_recovery.weight = -1.0
        self.rewards.unsafe_terminal_recovery.params["impulse"] = True
        self.rewards.safe_strike_inactivity.weight = -0.25
        self.rewards.safe_strike_inactivity.params["impulse"] = True
        self.rewards.targeted_contact_miss.params["impulse"] = True
