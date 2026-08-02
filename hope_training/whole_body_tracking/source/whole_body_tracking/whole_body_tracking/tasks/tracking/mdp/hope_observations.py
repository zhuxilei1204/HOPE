"""HOPE observation terms.

The baseline actor (policy) observation is the ``hope_pingpong`` contract (111 dims). Most
terms are standard proprioception from ``isaaclab.envs.mdp`` (base_ang_vel, joint_pos_rel,
joint_vel_rel, projected_gravity); the goal/target terms below wrap :class:`RacketTargetCommand`, and
``last_action`` reads the action term's deploy-faithful applied action.

The actual racket FK terms remain critic-only. ``racket_target_normal_w`` is critic-only in the
baseline and can be enabled as an experimental actor term by the 114-D normal-visible contract.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


# --- actor (policy) terms --------------------------------------------------------- #
def base_forward_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Base forward unit vector e_base,x, world XY (2)."""
    return _cmd(env, command_name).base_forward_xy()


def fixed_station_error_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Active station XY minus current base XY, world frame (2).

    The public term name is retained for existing 111/114-D checkpoints.  In
    dynamic-station tasks this is the per-swing base target before impact and
    the fixed ready station during recovery.
    """
    return _cmd(env, command_name).fixed_station_error_xy()


def racket_target_rel_base(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target racket position minus base position, world frame (3)."""
    return _cmd(env, command_name).racket_target_rel_base_w()


def racket_target_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket velocity, world frame (3)."""
    return _cmd(env, command_name).racket_target_vel_w


def time_to_strike(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Remaining time until the strike, s (1)."""
    return _cmd(env, command_name).time_to_strike.unsqueeze(-1)


def swing_side(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Forehand (+1) / backhand (-1), locked per swing (1)."""
    return _cmd(env, command_name).swing_sign.unsqueeze(-1)


def applied_last_action(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Previous tick's configured action feedback (31), with passive-head columns zero.

    ``raw`` remains the default contract. Experimental tasks may select ``effective``, the residual
    represented by the clamped q_des, to remove unrealizable action aliases from the feedback loop.
    """
    action_term = env.action_manager.get_term(action_name)
    feedback = getattr(action_term, "feedback_actions", None)
    if feedback is None:
        feedback = getattr(action_term, "applied_raw_actions", None)
    if feedback is None:
        raise RuntimeError(f"Action term {action_name!r} does not expose action feedback")
    return feedback


def stability_feedback(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Deployable base/torso/support feedback appended by the 122-D contract.

    Layout: base linear velocity body XY, torso projected gravity XY, whole-body
    COM relative to the midpoint of the feet in the base-heading frame XY, and
    left/right foot contact flags.
    """
    cmd = _cmd(env, command_name)
    motion = cmd._motion()
    data = cmd.robot.data

    torso_idx = motion.cfg.body_names.index("torso_Link")
    torso_quat_w = motion.robot_body_quat_w[:, torso_idx]
    gravity_w = torch.zeros(cmd.num_envs, 3, device=cmd.device)
    gravity_w[:, 2] = -1.0
    torso_gravity_xy = quat_rotate_inverse(torso_quat_w, gravity_w)[:, :2]

    masses = data.default_mass.to(device=data.body_com_pos_w.device)
    com_w = (data.body_com_pos_w * masses.unsqueeze(-1)).sum(dim=1) / masses.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    foot_ids = [
        motion.cfg.body_names.index("left_ankle_roll_Link"),
        motion.cfg.body_names.index("right_ankle_roll_Link"),
    ]
    support_xy = motion.robot_body_pos_w[:, foot_ids, :2].mean(dim=1)
    com_delta_w = torch.zeros_like(com_w)
    com_delta_w[:, :2] = com_w[:, :2] - support_xy
    com_support_b_xy = quat_rotate_inverse(yaw_quat(cmd.base_quat_w), com_delta_w)[:, :2]

    return torch.cat(
        (
            data.root_lin_vel_b[:, :2],
            torso_gravity_xy,
            com_support_b_xy,
            cmd.feet_contact_state,
        ),
        dim=-1,
    )


# --- privileged (critic-only) terms --------------------------------------------------------- #
def racket_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket position relative to the base (FK, yaw-heading frame). Privileged."""
    cmd = _cmd(env, command_name)
    return quat_rotate_inverse(yaw_quat(cmd.base_quat_w), cmd.racket_pos_w - cmd.base_pos_w)


def racket_lin_vel_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket linear velocity (FK), world frame. Privileged."""
    return _cmd(env, command_name).racket_lin_vel_w


def racket_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Actual racket face normal (FK), world frame. Privileged."""
    return _cmd(env, command_name).racket_normal_w


def racket_target_normal_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Desired racket face normal, world frame."""
    return _cmd(env, command_name).racket_target_normal_w


def episode_time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time remaining in the episode (s). Privileged critic input."""
    buf = getattr(env, "episode_length_buf", None)
    if buf is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return ((env.max_episode_length - buf).float() * env.step_dt).unsqueeze(-1)
