"""Physical actuator contract for the Agibot A3 policy joints.

The values in this module are output-side values.  They are intentionally kept
separate from the simulator configuration: the simulator may use a conservative
peak effort cap, while this contract describes the rated/peak operating envelope
used by diagnostics, rewards, and domain randomization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable


@dataclass(frozen=True)
class A3JointActuatorSpec:
    motor_model: str
    armature: float
    rated_torque: float
    peak_torque: float
    rated_speed: float
    peak_speed: float
    parallel: bool = False


def _rpm(value: float) -> float:
    return float(value) * 2.0 * pi / 60.0


_PFP_110_75 = A3JointActuatorSpec(
    "PFP-110-75", 0.1203404, 70.0, 320.0, _rpm(140.0), _rpm(190.0)
)
_PFP_93_65 = A3JointActuatorSpec(
    "PFP-93-65", 0.06646569891, 45.0, 220.0, _rpm(115.0), _rpm(206.0)
)
_PFP_78_58 = A3JointActuatorSpec(
    "PFP-78-58", 0.01208336871, 17.0, 60.0, _rpm(130.0), _rpm(240.0)
)
_PFP_59_60 = A3JointActuatorSpec(
    "PFP-59-60", 0.004967351303, 10.0, 36.0, _rpm(140.0), _rpm(170.0)
)
_PFP_41_48 = A3JointActuatorSpec(
    "PFP-41-48", 0.0008100893338, 2.0, 6.0, _rpm(150.0), _rpm(200.0)
)


def _parallel_spec(
    *,
    armature: float,
    peak_torque: float,
    rated_speed: float,
    peak_speed: float,
) -> A3JointActuatorSpec:
    # The supplied A3 table gives equivalent peak torque but not equivalent
    # rated torque.  Preserve the PFP-78-58 rated/peak ratio rather than
    # treating peak torque as continuously available.
    rated_torque = peak_torque * _PFP_78_58.rated_torque / _PFP_78_58.peak_torque
    return A3JointActuatorSpec(
        "PFP-78-58-parallel",
        armature,
        rated_torque,
        peak_torque,
        rated_speed,
        peak_speed,
        parallel=True,
    )


_ANKLE_PITCH = _parallel_spec(
    armature=0.06444060531,
    peak_torque=118.2,
    rated_speed=10.8,
    peak_speed=10.8,
)
_ANKLE_ROLL = _parallel_spec(
    armature=0.02012630058,
    peak_torque=54.75,
    rated_speed=19.37,
    peak_speed=19.37,
)
_WAIST_PITCH = _parallel_spec(
    armature=0.08820859156,
    peak_torque=115.0,
    rated_speed=9.24785,
    peak_speed=9.24785,
)
_WAIST_ROLL = _parallel_spec(
    armature=0.01462087613,
    peak_torque=46.0,
    rated_speed=22.7,
    peak_speed=22.7,
)


A3_ACTUATOR_SPECS: dict[str, A3JointActuatorSpec] = {
    "waist_yaw_joint": _PFP_93_65,
    "waist_roll_joint": _WAIST_ROLL,
    "waist_pitch_joint": _WAIST_PITCH,
    "head_yaw_joint": _PFP_41_48,
    "head_pitch_joint": _PFP_41_48,
    "left_shoulder_pitch_joint": _PFP_78_58,
    "left_shoulder_roll_joint": _PFP_78_58,
    "left_shoulder_yaw_joint": _PFP_59_60,
    "left_elbow_joint": _PFP_59_60,
    "left_wrist_roll_joint": _PFP_59_60,
    "left_wrist_pitch_joint": _PFP_41_48,
    "left_wrist_yaw_joint": _PFP_41_48,
    "right_shoulder_pitch_joint": _PFP_78_58,
    "right_shoulder_roll_joint": _PFP_78_58,
    "right_shoulder_yaw_joint": _PFP_59_60,
    "right_elbow_joint": _PFP_59_60,
    "right_wrist_roll_joint": _PFP_59_60,
    "right_wrist_pitch_joint": _PFP_41_48,
    "right_wrist_yaw_joint": _PFP_41_48,
    "left_hip_pitch_joint": _PFP_93_65,
    "left_hip_roll_joint": _PFP_93_65,
    "left_hip_yaw_joint": _PFP_93_65,
    "left_knee_joint": _PFP_110_75,
    "left_ankle_pitch_joint": _ANKLE_PITCH,
    "left_ankle_roll_joint": _ANKLE_ROLL,
    "right_hip_pitch_joint": _PFP_93_65,
    "right_hip_roll_joint": _PFP_93_65,
    "right_hip_yaw_joint": _PFP_93_65,
    "right_knee_joint": _PFP_110_75,
    "right_ankle_pitch_joint": _ANKLE_PITCH,
    "right_ankle_roll_joint": _ANKLE_ROLL,
}


A3_PARALLEL_JOINT_NAMES = tuple(
    name for name, spec in A3_ACTUATOR_SPECS.items() if spec.parallel
)
A3_SERIAL_JOINT_NAMES = tuple(
    name for name, spec in A3_ACTUATOR_SPECS.items() if not spec.parallel
)
A3_WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
A3_RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
A3_LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


def actuator_values(joint_names: Iterable[str], field: str) -> tuple[float, ...]:
    """Return one physical field in exactly the requested joint order."""
    names = tuple(joint_names)
    missing = [name for name in names if name not in A3_ACTUATOR_SPECS]
    if missing:
        raise KeyError(f"A3 actuator contract is missing joints: {missing}")
    numeric_fields = {
        "armature",
        "rated_torque",
        "peak_torque",
        "rated_speed",
        "peak_speed",
    }
    if field not in numeric_fields:
        raise KeyError(f"Unknown A3 actuator field: {field}")
    return tuple(float(getattr(A3_ACTUATOR_SPECS[name], field)) for name in names)


def validate_actuator_contract(joint_order: Iterable[str]) -> None:
    """Require one and only one physical specification for every policy joint."""
    names = tuple(joint_order)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    missing = sorted(set(names) - set(A3_ACTUATOR_SPECS))
    extra = sorted(set(A3_ACTUATOR_SPECS) - set(names))
    if duplicates or missing or extra:
        raise ValueError(
            "A3 actuator contract/order mismatch: "
            f"duplicates={duplicates}, missing={missing}, extra={extra}"
        )
