"""Operationally feasible Stage-1 command-tracking task for Agibot A3.

This is an isolated revision of Stage-1 v1. It preserves the same 114-D actor,
two-motion command distribution, and physical safety terminations while making
raw q-des saturation and target slew visible to the objective.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.agibot_a3_actuator_contract import (
    A3_LEG_JOINT_NAMES,
    A3_RIGHT_ARM_JOINT_NAMES,
    A3_WAIST_JOINT_NAMES,
    actuator_values,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
    A3_CANONICAL_JOINT_ORDER,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_stage1_command_tracking_env_cfg import (
    HOPEStage1CommandTrackingEnvCfg,
    Stage1CommandTrackingRewardsCfg,
)


_PASSIVE_HEAD = ("head_yaw_joint", "head_pitch_joint")
_LEFT_ARM = tuple(
    name
    for name in A3_CANONICAL_JOINT_ORDER
    if name.startswith("left_shoulder")
    or name.startswith("left_elbow")
    or name.startswith("left_wrist")
)


def _operational_margin_by_name() -> dict[str, float]:
    """Conservative soft margins that retain both audited motion envelopes."""
    values: dict[str, float] = {}
    for name in A3_CANONICAL_JOINT_ORDER:
        if name in _PASSIVE_HEAD:
            values[name] = 0.0
        elif name == "waist_yaw_joint":
            values[name] = 0.08
        elif name in ("waist_roll_joint", "waist_pitch_joint"):
            values[name] = 0.10
        elif name in A3_LEG_JOINT_NAMES:
            values[name] = 0.08
        elif name in A3_RIGHT_ARM_JOINT_NAMES:
            values[name] = 0.03 if "wrist" in name else 0.04
        elif name in _LEFT_ARM:
            values[name] = 0.04
        else:
            raise RuntimeError(f"Unclassified A3 policy joint: {name}")
    return values


def _rated_speed_by_name() -> dict[str, float]:
    speeds = actuator_values(A3_CANONICAL_JOINT_ORDER, "rated_speed")
    return dict(zip(A3_CANONICAL_JOINT_ORDER, speeds, strict=True))


def _target_acceleration_by_name(rise_time_s: float = 0.10) -> dict[str, float]:
    return {
        name: speed / float(rise_time_s)
        for name, speed in _rated_speed_by_name().items()
    }


def _actuator_term(joint_names: tuple[str, ...]) -> RewTerm:
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
            "pre_strike_scale": 0.25,
            "strike_scale": 0.05,
            "recovery_scale": 1.0,
            "hold_scale": 1.0,
            "corner_weight": 1.0,
            "peak_weight": 4.0,
            "free_margin": 0.05,
            "delta": 0.25,
            "maximum": 4.0,
        },
    )


@configclass
class Stage1OperationalRewardsCfg(Stage1CommandTrackingRewardsCfg):
    operational_joint_margin = RewTerm(
        func=mdp.phase_operational_joint_margin,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "action_name": "joint_pos",
            "waist_scale": 1.0,
            "other_upper_scale": 0.20,
            "right_arm_scale": 0.15,
            "leg_scale": 1.0,
            "pre_strike_scale": 0.80,
            "strike_scale": 0.70,
            "recovery_scale": 1.0,
            "hold_scale": 1.20,
            "excess_scale": 0.03,
            "maximum": 4.0,
        },
    )
    joint_target_slew = RewTerm(
        func=mdp.phase_joint_target_slew,
        weight=0.0,
        params={
            "command_name": "racket_target",
            "action_name": "joint_pos",
            "waist_scale": 1.0,
            "other_upper_scale": 0.20,
            "right_arm_scale": 0.12,
            "leg_scale": 0.85,
            "pre_strike_scale": 0.45,
            "strike_scale": 0.12,
            "recovery_scale": 1.0,
            "hold_scale": 1.20,
            "velocity_excess_scale": 0.30,
            "acceleration_excess_scale": 0.50,
            "acceleration_weight": 0.25,
            "acceleration_cost_mode": "bounded_squared",
            "maximum": 4.0,
        },
    )
    actuator_waist_feasibility = _actuator_term(A3_WAIST_JOINT_NAMES)
    actuator_right_arm_feasibility = _actuator_term(A3_RIGHT_ARM_JOINT_NAMES)
    actuator_leg_feasibility = _actuator_term(A3_LEG_JOINT_NAMES)


@configclass
class HOPEStage1OperationalEnvCfg(HOPEStage1CommandTrackingEnvCfg):
    """Stage-1 v2 with soft operational bounds and actuator-aware target slew."""

    rewards: Stage1OperationalRewardsCfg = Stage1OperationalRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        action = self.actions.joint_pos
        action.operational_margin_fraction = _operational_margin_by_name()
        action.q_des_velocity_limit = _rated_speed_by_name()
        action.q_des_acceleration_limit = _target_acceleration_by_name()

        self.rewards.operational_joint_margin.weight = -0.60
        self.rewards.joint_target_slew.weight = -0.035
        self.rewards.actuator_waist_feasibility.weight = -0.020
        self.rewards.actuator_right_arm_feasibility.weight = -0.005
        self.rewards.actuator_leg_feasibility.weight = -0.015

        # Hard-clamp overflow remains a separate, stronger cost. The right arm
        # retains strike freedom while waist/legs cannot use clipping cheaply.
        self.rewards.phase_action_overflow.weight = -0.30
        self.rewards.phase_action_overflow.params.update(
            {
                "waist_scale": 1.0,
                "right_arm_scale": 0.15,
                "leg_scale": 1.0,
                "pre_strike_scale": 0.80,
                "strike_scale": 0.60,
                "recovery_scale": 1.0,
                "hold_scale": 1.20,
                "free_margin": 0.0,
            }
        )

        # Command reward remains available under poor exploration, but only a
        # feasible command receives the full task-space payoff.
        self.rewards.planner_racket_task_space_crossfade.params.update(
            {
                "action_feasibility_metric": (
                    "action_operational_feasibility_score"
                ),
                "action_feasibility_floor": 0.35,
            }
        )

        # Damping is concentrated after the strike. Strike-time waist/leg
        # compensation and the racket arm remain substantially freer.
        self.rewards.phase_action_rate_waist.weight = -0.040
        self.rewards.phase_action_rate_waist.params.update(
            {
                "pre_strike_scale": 0.015,
                "strike_scale": 0.002,
                "recovery_scale": 0.150,
                "hold_scale": 0.200,
            }
        )
        self.rewards.phase_action_rate_legs.weight = -0.008
        self.rewards.phase_action_rate_legs.params.update(
            {
                "pre_strike_scale": 0.008,
                "strike_scale": 0.001,
                "recovery_scale": 0.080,
                "hold_scale": 0.120,
            }
        )
        self.rewards.strike_balance.weight = 0.20
        self.rewards.post_strike_base_ang_vel.weight = 0.12
        self.rewards.recovery_health.weight = 0.50
