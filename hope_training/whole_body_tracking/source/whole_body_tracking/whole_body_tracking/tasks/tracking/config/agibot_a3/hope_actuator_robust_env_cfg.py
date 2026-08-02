"""Isolated A3 actuator-robust variant of the existing HOPE task."""

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
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
    EventCfg,
    HOPEPingPongEnvCfg,
    RewardsCfg,
)


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
class ActuatorRobustRewardsCfg(RewardsCfg):
    # These terms default to zero.  Only the isolated Hydra experiment enables
    # them, so existing checkpoints/configurations retain byte-for-byte reward
    # semantics.
    actuator_waist_feasibility = _actuator_term(A3_WAIST_JOINT_NAMES)
    actuator_right_arm_feasibility = _actuator_term(A3_RIGHT_ARM_JOINT_NAMES)
    actuator_leg_feasibility = _actuator_term(A3_LEG_JOINT_NAMES)


@configclass
class ActuatorRobustEventCfg(EventCfg):
    # Supplier variation is documented as <=3% for serial joints.  Parallel
    # ankle/waist equivalents omit posture-dependent coupling, so use a wider
    # envelope to stop the policy relying on one diagonal approximation.
    serial_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(A3_SERIAL_JOINT_NAMES), preserve_order=True
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
                "robot", joint_names=list(A3_PARALLEL_JOINT_NAMES), preserve_order=True
            ),
            "armature_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class HOPEActuatorRobustEnvCfg(HOPEPingPongEnvCfg):
    """Baseline HOPE task with only actuator robustness added."""

    rewards: ActuatorRobustRewardsCfg = ActuatorRobustRewardsCfg()
    events: ActuatorRobustEventCfg = ActuatorRobustEventCfg()

    def __post_init__(self):
        validate_actuator_contract(AGIBOT_A3_JOINT_NAMES)
        super().__post_init__()
