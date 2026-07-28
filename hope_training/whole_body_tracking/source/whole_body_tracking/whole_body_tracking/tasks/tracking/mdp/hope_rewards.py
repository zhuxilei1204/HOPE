"""HOPE reward terms.

The public task uses eleven reward terms. Eight are defined here; three (upright balance,
action smoothness, joint-limit regularization) are standard ``isaaclab.envs.mdp`` terms wired in the
env config. All example weights and kernel widths live in the env config / task YAML and are meant to
be tuned — they are illustrative, not performance-tuned values.

  1. upright / balance            -> mdp.flat_orientation_l2       (env config)
  2. forehand/backhand imitation  -> sample_imitation
  3. racket position              -> racket_position
  4. racket velocity              -> racket_velocity
  5. impact outgoing velocity     -> impact_outgoing_velocity
  6. simplified blade direction   -> racket_blade_direction
  7. soft contact proximity       -> soft_ball_contact
  8. actual ball contact          -> ball_contact
  9. net crossing                 -> ball_net_cross
 10. opponent-half first bounce   -> ball_opponent_bounce
 11. in-place follow-through/recovery -> follow_through_recovery / recovery_health
 12. action smoothness            -> mdp.action_rate_l2            (env config)
 13. joint-limit regularization   -> mdp.joint_pos_limits          (env config)

The racket position/velocity/blade terms are active only in a short window around the strike; the
contact/net/bounce terms fire once at the exact strike frame; imitation is active during the swing
(not the frozen pre-swing hold); recovery is active through the follow-through and the hold.
"""

from __future__ import annotations

from functools import lru_cache
import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_error_magnitude, quat_rotate_inverse, yaw_quat

from whole_body_tracking.tasks.tracking.mdp import rewards as _imitation
from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand
from whole_body_tracking.utils.action_adapter_config import load_joint_order

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


def _phase_scale(
    env: ManagerBasedRLEnv,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 0.0,
    end_scale: float = 1.0,
) -> torch.Tensor | float:
    """Linear reward warmup based on the environment control-step counter."""
    if warmup_steps <= 0:
        return float(end_scale)
    step = float(getattr(env, "common_step_counter", 0))
    alpha = min(max((step - float(start_step)) / float(warmup_steps), 0.0), 1.0)
    return float(start_scale) + alpha * (float(end_scale) - float(start_scale))


def _recovery_or_hold_gate(
    cmd: RacketTargetCommand,
    include_hold: bool = True,
    early_prestrike_window_steps: int = 0,
) -> torch.Tensor:
    motion = cmd._motion()
    gate = (~cmd.pre_strike) & (~cmd.strike_window)
    if include_hold:
        gate = gate | motion.in_hold
    if early_prestrike_window_steps > 0:
        gate = gate | (cmd.pre_strike & (cmd.steps_since_target_resample <= int(early_prestrike_window_steps)))
    return gate


@lru_cache(maxsize=16)
def _canonical_joint_indices(joint_names: tuple[str, ...]) -> tuple[int, ...]:
    canonical = tuple(load_joint_order())
    name_to_index = {name: index for index, name in enumerate(canonical)}
    return tuple(name_to_index[name] for name in joint_names if name in name_to_index)


def _table_monitor_points(
    cmd: RacketTargetCommand,
    body_names: list[str] | None,
    include_racket: bool,
) -> torch.Tensor | None:
    points = []
    motion = cmd._motion()
    if body_names:
        ids = [motion.cfg.body_names.index(name) for name in body_names if name in motion.cfg.body_names]
        if ids:
            idx = torch.tensor(ids, dtype=torch.long, device=cmd.device)
            points.append(motion.robot_body_pos_w[:, idx])
    if include_racket:
        points.append(cmd.racket_pos_w.unsqueeze(1))
    if not points:
        return None
    return torch.cat(points, dim=1)


def _table_zone_score(
    cmd: RacketTargetCommand,
    points_w: torch.Tensor,
    x_margin: float,
    y_margin: float,
    below_surface_margin: float,
    above_surface_margin: float,
    distance_std: float,
) -> torch.Tensor:
    origins = cmd._env.scene.env_origins
    points_l = points_w - origins.unsqueeze(1)
    x = points_l[..., 0]
    y = points_l[..., 1]
    z = points_l[..., 2]

    center_y = cmd.fixed_station_w[:, 1] - origins[:, 1]
    half_width = 0.5 * float(cmd.cfg.table_width)
    x_min = x.new_tensor(float(cmd.cfg.table_near_x) - float(x_margin))
    x_max = x.new_tensor(float(cmd.cfg.table_near_x) + float(cmd.cfg.table_length) + float(x_margin))
    y_min = center_y.unsqueeze(1) - half_width - float(y_margin)
    y_max = center_y.unsqueeze(1) + half_width + float(y_margin)
    z_min = z.new_tensor(float(cmd.cfg.table_surface_z) - float(below_surface_margin))
    z_max = z.new_tensor(float(cmd.cfg.table_surface_z) + float(above_surface_margin))

    inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    outside_x = torch.maximum(x_min - x, x - x_max).clamp_min(0.0)
    outside_y = torch.maximum(y_min - y, y - y_max).clamp_min(0.0)
    outside_z = torch.maximum(z_min - z, z - z_max).clamp_min(0.0)
    outside_dist = torch.sqrt(outside_x.square() + outside_y.square() + outside_z.square())
    std = max(float(distance_std), 1.0e-6)
    score = torch.exp(-torch.square(outside_dist / std))
    score = torch.where(inside, torch.ones_like(score), score)
    return score.max(dim=1).values


def table_proximity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_names: list[str] | None = None,
    include_racket: bool = True,
    x_margin: float = 0.06,
    y_margin: float = 0.06,
    below_surface_margin: float = 0.08,
    above_surface_margin: float = 0.05,
    distance_std: float = 0.05,
) -> torch.Tensor:
    """Soft penalty for tracked points entering or grazing the analytic table-top no-touch zone."""
    cmd = _cmd(env, command_name)
    points = _table_monitor_points(cmd, body_names, include_racket)
    if points is None:
        return torch.zeros(cmd.num_envs, device=cmd.device)
    return _table_zone_score(
        cmd,
        points,
        x_margin,
        y_margin,
        below_surface_margin,
        above_surface_margin,
        distance_std,
    )


def _body_pose_score(
    cmd: RacketTargetCommand,
    body_names: list[str] | None,
    pos_std: float,
    ori_std: float,
) -> torch.Tensor:
    motion = cmd._motion()
    if not body_names:
        return torch.ones(cmd.num_envs, device=cmd.device)
    ids = [motion.cfg.body_names.index(name) for name in body_names if name in motion.cfg.body_names]
    if not ids:
        return torch.ones(cmd.num_envs, device=cmd.device)
    idx = torch.tensor(ids, dtype=torch.long, device=cmd.device)
    pos_err = torch.norm(motion.robot_body_pos_w[:, idx] - motion.body_pos_relative_w[:, idx], dim=-1).mean(dim=-1)
    pos = torch.exp(-torch.square(pos_err / pos_std))

    q_ref = motion.body_quat_relative_w[:, idx].reshape(-1, 4)
    q_robot = motion.robot_body_quat_w[:, idx].reshape(-1, 4)
    ori_err = quat_error_magnitude(q_ref, q_robot).reshape(cmd.num_envs, len(ids)).mean(dim=-1)
    ori = torch.exp(-torch.square(ori_err / ori_std))
    return 0.5 * pos + 0.5 * ori


def _next_ready_score(
    env: ManagerBasedRLEnv,
    cmd: RacketTargetCommand,
    height_std: float,
    upright_std: float,
    lin_vel_std: float,
    ang_vel_std: float,
    station_std: float,
    racket_vel_std: float,
    arm_pos_std: float,
    arm_ori_std: float,
    arm_body_names: list[str] | None,
) -> torch.Tensor:
    data = cmd.robot.data
    default_z = data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    height = torch.exp(-torch.square((data.root_pos_w[:, 2] - default_z) / height_std))
    upright_err = torch.norm(data.projected_gravity_b[:, :2], dim=-1)
    upright = torch.exp(-torch.square(upright_err / upright_std))
    lin = torch.exp(-torch.square(torch.norm(data.root_lin_vel_w[:, :2], dim=-1) / lin_vel_std))
    ang = torch.exp(-torch.square(torch.norm(data.root_ang_vel_w, dim=-1) / ang_vel_std))
    feet = torch.clamp(cmd.feet_contact_frac, 0.0, 1.0)
    station_err = torch.norm(cmd.base_pos_w[:, :2] - cmd.station_w, dim=-1)
    station = torch.exp(-torch.square(station_err / station_std))
    racket = torch.exp(-torch.square(torch.norm(cmd.racket_lin_vel_w, dim=-1) / racket_vel_std))
    arm = _body_pose_score(cmd, arm_body_names, arm_pos_std, arm_ori_std)
    return (
        0.18 * height
        + 0.18 * upright
        + 0.15 * lin
        + 0.15 * ang
        + 0.12 * feet
        + 0.10 * station
        + 0.07 * racket
        + 0.05 * arm
    )


# --- (2) forehand/backhand sample imitation ------------------------------------------------- #
def sample_imitation(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_pos: float = 0.3,
    std_ori: float = 0.4,
    body_names: list[str] | None = None,
    core_clip_scale: float = 1.0,
    supplemental_clip_scale: float = 1.0,
) -> torch.Tensor:
    """Track the imitated clip's (upper-)body pose during the swing.

    Combines the anchor-relative body position and orientation tracking kernels and gates them to the
    active swing (zero during the frozen pre-swing hold). ``body_names`` selects the tracked subset
    (the env config passes the upper-body bodies so the legs stay free to step)."""
    rp = _imitation.motion_relative_body_position_error_exp(env, command_name, std_pos, body_names)
    ro = _imitation.motion_relative_body_orientation_error_exp(env, command_name, std_ori, body_names)
    motion = env.command_manager.get_term(command_name)
    score = (0.5 * rp + 0.5 * ro) * (~motion.in_hold).float()
    core_count = int(getattr(motion.cfg, "core_clip_count", 0))
    if core_count > 0:
        scale = torch.where(
            motion.clip_id >= core_count,
            torch.full_like(score, float(supplemental_clip_scale)),
            torch.full_like(score, float(core_clip_scale)),
        )
    else:
        scale = torch.full_like(score, float(core_clip_scale))
    return score * scale


def phase_lower_body_motion_prior(
    env: ManagerBasedRLEnv,
    motion_command_name: str,
    racket_command_name: str,
    std_pos: float = 0.28,
    std_ori: float = 0.45,
    body_names: list[str] | None = None,
    pre_strike_scale: float = 0.35,
    strike_scale: float = 0.15,
    recovery_scale: float = 1.0,
    hold_scale: float = 1.15,
    core_clip_scale: float = 0.8,
    supplemental_clip_scale: float = 1.15,
) -> torch.Tensor:
    """Track lower-body support timing strongly after impact, lightly during the strike.

    This preserves hip/knee/ankle compensation freedom at impact while making the
    accepted supplemental clips useful for braking and recovery instead of globally
    constraining the hitting arm.
    """
    motion = env.command_manager.get_term(motion_command_name)
    racket = _cmd(env, racket_command_name)
    rp = _imitation.motion_relative_body_position_error_exp(
        env, motion_command_name, std_pos, body_names
    )
    ro = _imitation.motion_relative_body_orientation_error_exp(
        env, motion_command_name, std_ori, body_names
    )
    score = 0.55 * rp + 0.45 * ro
    phase = torch.full_like(score, float(recovery_scale))
    phase = torch.where(racket.pre_strike, torch.full_like(phase, float(pre_strike_scale)), phase)
    phase = torch.where(racket.strike_window, torch.full_like(phase, float(strike_scale)), phase)
    phase = torch.where(motion.in_hold, torch.full_like(phase, float(hold_scale)), phase)

    core_count = int(getattr(motion.cfg, "core_clip_count", 0))
    if core_count > 0:
        clip_scale = torch.where(
            motion.clip_id >= core_count,
            torch.full_like(score, float(supplemental_clip_scale)),
            torch.full_like(score, float(core_clip_scale)),
        )
    else:
        clip_scale = torch.full_like(score, float(core_clip_scale))
    return score * phase * clip_scale


def motion_anchor_height_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.12,
) -> torch.Tensor:
    """Track only reference torso height, leaving clip root XY to the station objective."""
    motion = env.command_manager.get_term(command_name)
    error = torch.square(motion.anchor_pos_w[:, 2] - motion.robot_anchor_pos_w[:, 2])
    return torch.exp(-error / max(float(std), 1.0e-6) ** 2)


def wrist_motion_pos_release(
    env: ManagerBasedRLEnv,
    motion_command_name: str,
    racket_command_name: str,
    std: float,
    body_names: list[str] | None = None,
    release_window_s: float = 0.20,
    release_scale: float = 0.35,
) -> torch.Tensor:
    """Track the racket wrist from the clip, but soften it near impact.

    ``release_scale`` keeps a non-zero wrist reference through the strike window.  A value of 0.0
    reproduces the old full release; values in roughly [0.25, 0.5] allow hit correction without
    letting the policy solve the task by destroying the forehand/backhand wrist pose.
    """
    value = _imitation.motion_relative_body_position_error_exp(env, motion_command_name, std, body_names)
    racket = _cmd(env, racket_command_name)
    release = racket.true_time_to_strike.abs() <= release_window_s
    scale = torch.where(
        release,
        torch.full_like(value, float(release_scale)),
        torch.ones_like(value),
    )
    return value * scale


def wrist_motion_ori_release(
    env: ManagerBasedRLEnv,
    motion_command_name: str,
    racket_command_name: str,
    std: float,
    body_names: list[str] | None = None,
    release_window_s: float = 0.20,
    release_scale: float = 0.35,
) -> torch.Tensor:
    """Track the racket wrist orientation from the clip, but soften it near impact."""
    value = _imitation.motion_relative_body_orientation_error_exp(env, motion_command_name, std, body_names)
    racket = _cmd(env, racket_command_name)
    release = racket.true_time_to_strike.abs() <= release_window_s
    scale = torch.where(
        release,
        torch.full_like(value, float(release_scale)),
        torch.ones_like(value),
    )
    return value * scale


# --- (3,4,5) racket goal tracking, active in the strike window ------------------------------ #
def racket_position(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the racket center against the hidden true strike point near impact."""
    cmd = _cmd(env, command_name)
    target_now = cmd.ball_strike_pos_w - cmd.racket_impact_target_vel_w * cmd.true_time_to_strike.unsqueeze(-1)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_now), dim=-1)
    return torch.exp(-error / std**2) * cmd.strike_window.float()


def _swing_side_gate(cmd: RacketTargetCommand, swing_side: float) -> torch.Tensor:
    if abs(float(swing_side)) < 1.0e-6:
        return torch.ones(cmd.num_envs, dtype=torch.bool, device=cmd.device)
    return cmd.swing_sign >= 0.0 if float(swing_side) > 0.0 else cmd.swing_sign < 0.0


def side_racket_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    swing_side: float = -1.0,
) -> torch.Tensor:
    """Side-gated racket-position shaping; ``swing_side=-1`` targets backhand."""
    cmd = _cmd(env, command_name)
    return racket_position(env, command_name, std) * _swing_side_gate(cmd, swing_side).float()


def racket_velocity(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the racket linear velocity against the impact-inverted racket velocity."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_impact_target_vel_w), dim=-1)
    return torch.exp(-error / std**2) * cmd.strike_window.float()


def racket_velocity_projection(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_speed_ratio: float = 0.82,
    speed_std: float = 0.35,
    lateral_std: float = 0.75,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward realized racket speed along the impact-inverted target direction.

    The isotropic velocity kernel can still be generous when the policy undershoots a fast racket
    target.  This term makes that failure mode explicit: near impact, get enough velocity projection
    in the requested direction while keeping lateral velocity modest.
    """
    cmd = _cmd(env, command_name)
    target = cmd.racket_impact_target_vel_w
    target_speed = torch.norm(target, dim=-1).clamp_min(1.0e-6)
    target_dir = target / target_speed.unsqueeze(-1)
    proj = torch.sum(cmd.racket_lin_vel_w * target_dir, dim=-1)
    lateral = cmd.racket_lin_vel_w - proj.unsqueeze(-1) * target_dir
    speed_score = torch.sigmoid((proj - float(min_speed_ratio) * target_speed) / float(speed_std))
    lateral_score = torch.exp(-torch.square(torch.norm(lateral, dim=-1) / float(lateral_std)))
    return speed_score * lateral_score * cmd.strike_window.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def impact_forward_lift(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_forward_speed: float = 1.9,
    min_upward_speed: float = 1.0,
    speed_std: float = 0.45,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Dense physics-facing shaping for a return that has enough forward speed and lift.

    This uses the same moving-racket impact prediction as the sparse net/opponent rewards.  It does
    not replace the real success terms; it gives PPO a smoother slope for the dominant MuJoCo failure
    mode where the ball contacts the racket but leaves too low/slow to clear the net.
    """
    cmd = _cmd(env, command_name)
    out = cmd.impact_ball_out_vel_w
    forward = torch.sigmoid((out[:, 0] - float(min_forward_speed)) / float(speed_std))
    upward = torch.sigmoid((out[:, 2] - float(min_upward_speed)) / float(speed_std))
    return forward * upward * cmd.strike_window.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def impact_outgoing_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Track the predicted post-impact ball velocity against the desired outgoing velocity."""
    cmd = _cmd(env, command_name)
    value = torch.exp(-torch.square(cmd.impact_ball_out_error / std)) * cmd.strike_window.float()
    return value * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def racket_blade_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Align the racket face normal with the desired blade direction near the strike (``std`` in rad)."""
    cmd = _cmd(env, command_name)
    cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    value = torch.exp(-(angle**2) / std**2) * cmd.strike_window.float()
    return value * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def soft_ball_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float = 0.18,
    approach_speed: float = 0.15,
    approach_std: float = 0.75,
    normal_speed: float = 0.0,
    normal_std: float = 0.75,
    window_s: float = 0.20,
) -> torch.Tensor:
    """Dense pre-contact shaping: be near the impact point and move into the incoming ball.

    The hard contact/net/bounce terms stay as the success metric. This term only creates a learnable
    slope around the sparse one-frame contact event, so a standing policy cannot get the contact reward
    without actually bringing the racket into the strike neighborhood.
    """
    cmd = _cmd(env, command_name)
    return _soft_ball_contact_score(
        cmd,
        pos_std=pos_std,
        approach_speed=approach_speed,
        approach_std=approach_std,
        normal_speed=normal_speed,
        normal_std=normal_std,
        window_s=window_s,
    )


def _soft_ball_contact_score(
    cmd: RacketTargetCommand,
    pos_std: float = 0.18,
    approach_speed: float = 0.15,
    approach_std: float = 0.75,
    normal_speed: float = 0.0,
    normal_std: float = 0.75,
    window_s: float = 0.20,
) -> torch.Tensor:
    time_abs = cmd.true_time_to_strike.abs()
    gate = time_abs <= window_s
    timing = torch.exp(-torch.square(time_abs / max(window_s, 1e-6)))

    pos_err = torch.norm(cmd.racket_pos_w - cmd.ball_strike_pos_w, dim=-1)
    proximity = torch.exp(-torch.square(pos_err / pos_std))

    if cmd.cfg.contact_approach_mode == "target_velocity":
        approach_dir = cmd.racket_impact_target_vel_w / (
            torch.norm(cmd.racket_impact_target_vel_w, dim=-1, keepdim=True) + 1e-6
        )
    else:
        to_target = cmd.ball_strike_pos_w - cmd.racket_pos_w
        approach_dir = to_target / (torch.norm(to_target, dim=-1, keepdim=True) + 1e-6)
    approach = torch.sum(cmd.racket_lin_vel_w * approach_dir, dim=-1)
    approach_score = torch.sigmoid((approach - approach_speed) / approach_std)

    normal = cmd.racket_normal_w / (torch.norm(cmd.racket_normal_w, dim=-1, keepdim=True) + 1e-6)
    rel_in = cmd.incoming_ball_vel_w - cmd.racket_lin_vel_w
    closing = -torch.sum(rel_in * normal, dim=-1)
    normal_score = torch.sigmoid((closing - normal_speed) / normal_std)

    return proximity * approach_score * normal_score * timing * gate.float()


def side_soft_ball_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float = 0.18,
    approach_speed: float = 0.15,
    approach_std: float = 0.75,
    normal_speed: float = 0.0,
    normal_std: float = 0.75,
    window_s: float = 0.20,
    swing_side: float = -1.0,
) -> torch.Tensor:
    """Side-gated dense contact reward; ``swing_side=-1`` targets backhand."""
    cmd = _cmd(env, command_name)
    score = _soft_ball_contact_score(
        cmd,
        pos_std=pos_std,
        approach_speed=approach_speed,
        approach_std=approach_std,
        normal_speed=normal_speed,
        normal_std=normal_std,
        window_s=window_s,
    )
    return score * _swing_side_gate(cmd, swing_side).float()


def contact_net_margin(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float = 0.16,
    approach_speed: float = 0.10,
    approach_std: float = 0.75,
    normal_speed: float = 0.0,
    normal_std: float = 0.75,
    window_s: float = 0.18,
    min_forward_speed: float = 2.05,
    forward_std: float = 0.35,
    min_upward_speed: float = 0.95,
    upward_std: float = 0.35,
    min_clearance: float = 0.03,
    clearance_std: float = 0.06,
    swing_side: float = 0.0,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Dense contact-conditioned signal for clearing the net with margin.

    This targets the MuJoCo failure mode where the racket reaches the ball but the
    outgoing velocity is too low/slow.  It is gated by a soft contact score so it
    does not reward a stable policy that never brings the racket to the ball.
    """
    cmd = _cmd(env, command_name)
    contact_like = _soft_ball_contact_score(
        cmd,
        pos_std=pos_std,
        approach_speed=approach_speed,
        approach_std=approach_std,
        normal_speed=normal_speed,
        normal_std=normal_std,
        window_s=window_s,
    )

    p0 = cmd.ball_strike_pos_w - cmd._env.scene.env_origins
    v = cmd.impact_ball_out_vel_w
    vx = v[:, 0]
    vz = v[:, 2]
    net_x = float(cmd.cfg.table_near_x) + float(cmd.cfg.net_x)
    net_top = float(cmd.cfg.table_surface_z) + float(cmd.cfg.net_height) + float(cmd.cfg.net_margin)

    t_net = (net_x - p0[:, 0]) / vx.clamp_min(1.0e-3)
    z_net = p0[:, 2] + vz * t_net - 0.5 * 9.81 * t_net.square()
    clearance = z_net - net_top

    forward = torch.sigmoid((vx - float(min_forward_speed)) / max(float(forward_std), 1.0e-6))
    upward = torch.sigmoid((vz - float(min_upward_speed)) / max(float(upward_std), 1.0e-6))
    margin = torch.sigmoid((clearance - float(min_clearance)) / max(float(clearance_std), 1.0e-6))
    valid = (t_net > 0.0).float()
    side = _swing_side_gate(cmd, swing_side).float()
    return contact_like * forward * upward * margin * valid * side * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def _strike_balance_score(
    cmd: RacketTargetCommand,
    pre_window_s: float = 0.34,
    post_window_s: float = 0.18,
    pitch_std: float = 0.16,
    upright_std: float = 0.26,
    ang_vel_std: float = 1.05,
    backward_std: float = 0.16,
    backward_vel_std: float = 0.42,
) -> torch.Tensor:
    data = cmd.robot.data
    tts = cmd.true_time_to_strike
    gate = (tts <= float(pre_window_s)) & (tts >= -float(post_window_s)) & (~cmd.no_command_ready_active)

    projected_gravity_xy = data.projected_gravity_b[:, :2]
    pitch_like = torch.abs(projected_gravity_xy[:, 0])
    pitch = torch.exp(-torch.square(pitch_like / max(float(pitch_std), 1.0e-6)))
    upright = torch.exp(-torch.square(torch.norm(projected_gravity_xy, dim=-1) / max(float(upright_std), 1.0e-6)))
    ang = torch.exp(-torch.square(torch.norm(data.root_ang_vel_b[:, :2], dim=-1) / max(float(ang_vel_std), 1.0e-6)))
    backward_offset = (cmd.station_w[:, 0] - cmd.base_pos_w[:, 0]).clamp_min(0.0)
    backward = torch.exp(-torch.square(backward_offset / max(float(backward_std), 1.0e-6)))
    backward_vel = (-data.root_lin_vel_w[:, 0]).clamp_min(0.0)
    back_vel = torch.exp(-torch.square(backward_vel / max(float(backward_vel_std), 1.0e-6)))

    score = 0.26 * pitch + 0.24 * upright + 0.22 * ang + 0.16 * backward + 0.12 * back_vel
    return score * gate.float()


def balanced_soft_ball_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float = 0.18,
    approach_speed: float = 0.10,
    approach_std: float = 0.75,
    normal_speed: float = 0.0,
    normal_std: float = 0.75,
    window_s: float = 0.22,
    pre_window_s: float = 0.34,
    post_window_s: float = 0.18,
    pitch_std: float = 0.18,
    upright_std: float = 0.30,
    ang_vel_std: float = 1.25,
    backward_std: float = 0.20,
    backward_vel_std: float = 0.55,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Soft contact reward gated by trunk/base balance near the strike."""
    cmd = _cmd(env, command_name)
    contact_like = _soft_ball_contact_score(
        cmd,
        pos_std=pos_std,
        approach_speed=approach_speed,
        approach_std=approach_std,
        normal_speed=normal_speed,
        normal_std=normal_std,
        window_s=window_s,
    )
    balance = _strike_balance_score(
        cmd,
        pre_window_s=pre_window_s,
        post_window_s=post_window_s,
        pitch_std=pitch_std,
        upright_std=upright_std,
        ang_vel_std=ang_vel_std,
        backward_std=backward_std,
        backward_vel_std=backward_vel_std,
    )
    return contact_like * balance * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def balanced_ball_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    pre_window_s: float = 0.34,
    post_window_s: float = 0.18,
    pitch_std: float = 0.18,
    upright_std: float = 0.30,
    ang_vel_std: float = 1.25,
    backward_std: float = 0.20,
    backward_vel_std: float = 0.55,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Actual contact event gated by balanced trunk/base state."""
    cmd = _cmd(env, command_name)
    balance = _strike_balance_score(
        cmd,
        pre_window_s=pre_window_s,
        post_window_s=post_window_s,
        pitch_std=pitch_std,
        upright_std=upright_std,
        ang_vel_std=ang_vel_std,
        backward_std=backward_std,
        backward_vel_std=backward_vel_std,
    )
    return cmd.ball_contact.float() * balance * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def balanced_net_cross(
    env: ManagerBasedRLEnv,
    command_name: str,
    pre_window_s: float = 0.34,
    post_window_s: float = 0.18,
    pitch_std: float = 0.18,
    upright_std: float = 0.30,
    ang_vel_std: float = 1.25,
    backward_std: float = 0.20,
    backward_vel_std: float = 0.55,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Net-clear event gated by balanced trunk/base state."""
    cmd = _cmd(env, command_name)
    balance = _strike_balance_score(
        cmd,
        pre_window_s=pre_window_s,
        post_window_s=post_window_s,
        pitch_std=pitch_std,
        upright_std=upright_std,
        ang_vel_std=ang_vel_std,
        backward_std=backward_std,
        backward_vel_std=backward_vel_std,
    )
    return cmd.ball_net_cross.float() * balance * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def balanced_opponent_bounce(
    env: ManagerBasedRLEnv,
    command_name: str,
    pre_window_s: float = 0.34,
    post_window_s: float = 0.18,
    pitch_std: float = 0.18,
    upright_std: float = 0.30,
    ang_vel_std: float = 1.25,
    backward_std: float = 0.20,
    backward_vel_std: float = 0.55,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Opponent-bounce return event gated by balanced trunk/base state."""
    cmd = _cmd(env, command_name)
    balance = _strike_balance_score(
        cmd,
        pre_window_s=pre_window_s,
        post_window_s=post_window_s,
        pitch_std=pitch_std,
        upright_std=upright_std,
        ang_vel_std=ang_vel_std,
        backward_std=backward_std,
        backward_vel_std=backward_vel_std,
    )
    return cmd.ball_on_opponent.float() * balance * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


# --- (6,7,8) no-spin return outcome, one-shot at the strike --------------------------------- #
def ball_contact(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """+1 on the strike frame when the racket actually contacts the target ball (near + approaching)."""
    return _cmd(env, command_name).ball_contact.float()


def ball_net_cross(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """+1 on the strike frame when the (no-spin) outgoing ball clears the net."""
    return _cmd(env, command_name).ball_net_cross.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def ball_opponent_bounce(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """+1 on the strike frame when the outgoing ball's first bounce lands on the opponent half."""
    return _cmd(env, command_name).ball_on_opponent.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def pre_strike_station_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    station_std: float = 0.25,
    stop_window_s: float = 0.10,
) -> torch.Tensor:
    """Reward moving the base to the current swing's desired station before impact.

    In fixed-station mode this is simply an in-place ready reward.  In dynamic-station mode the
    command supplies a per-swing base target derived from the sampled racket intercept and the
    reference motion's natural racket offset, so lateral balls create a learnable base-motion signal.
    """
    cmd = _cmd(env, command_name)
    station_err = torch.norm(cmd.base_pos_w[:, :2] - cmd.station_w, dim=-1)
    station = torch.exp(-torch.square(station_err / station_std))
    gate = cmd.pre_strike & (cmd.true_time_to_strike.abs() > float(stop_window_s))
    return station * gate.float()


# --- (9) in-place follow-through / recovery ------------------------------------------------- #
def follow_through_recovery(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.5,
    station_std: float = 0.3,
    use_dynamic_station: bool = False,
) -> torch.Tensor:
    """Reward settling calmly AT the fixed station through the follow-through and the pre-swing hold.

    ``exp(-(|v_base_xy|/std)^2) * exp(-(station_err/station_std)^2) * feet_contact_frac`` active in the
    follow-through ((~pre_strike) & (~strike_window)) and during the hold. This is in-place recentring
    and balance only — it never rewards walking, footstep planning, or leaving the station.
    """
    cmd = _cmd(env, command_name)
    v_xy = torch.norm(cmd.robot.data.root_lin_vel_w[:, :2], dim=-1)
    calm = torch.exp(-torch.square(v_xy / std))
    station_target = cmd.station_w if use_dynamic_station else cmd.fixed_station_w
    station_err = torch.norm(cmd.base_pos_w[:, :2] - station_target, dim=-1)
    at_station = torch.exp(-torch.square(station_err / station_std))
    in_hold = cmd._motion().in_hold
    gate = ((~cmd.pre_strike) & (~cmd.strike_window)) | in_hold
    return calm * at_station * cmd.feet_contact_frac * gate.float()


def recovery_health(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_std: float = 0.12,
    upright_std: float = 0.35,
    lin_vel_std: float = 0.35,
    ang_vel_std: float = 1.0,
    station_std: float = 0.25,
    use_dynamic_station: bool = False,
) -> torch.Tensor:
    """Reward a deploy-ready stance after the strike and during the ready hold.

    This complements ``follow_through_recovery``: it explicitly scores base height, uprightness,
    residual base motion, station drift, and foot contact. It is gated off before the strike so it
    does not suppress the swing itself.
    """
    cmd = _cmd(env, command_name)
    data = cmd.robot.data
    in_hold = cmd._motion().in_hold
    gate = (((~cmd.pre_strike) & (~cmd.strike_window)) | in_hold).float()

    default_z = data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    height = torch.exp(-torch.square((data.root_pos_w[:, 2] - default_z) / height_std))
    upright_err = torch.norm(data.projected_gravity_b[:, :2], dim=-1)
    upright = torch.exp(-torch.square(upright_err / upright_std))
    lin = torch.exp(-torch.square(torch.norm(data.root_lin_vel_w[:, :2], dim=-1) / lin_vel_std))
    ang = torch.exp(-torch.square(torch.norm(data.root_ang_vel_w, dim=-1) / ang_vel_std))
    station_target = cmd.station_w if use_dynamic_station else cmd.fixed_station_w
    station_err = torch.norm(cmd.base_pos_w[:, :2] - station_target, dim=-1)
    station = torch.exp(-torch.square(station_err / station_std))
    feet = torch.clamp(cmd.feet_contact_frac, 0.0, 1.0)

    score = 0.25 * height + 0.25 * upright + 0.20 * lin + 0.15 * ang + 0.10 * station + 0.05 * feet
    return score * gate


def next_ball_readiness(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_std: float = 0.10,
    upright_std: float = 0.28,
    lin_vel_std: float = 0.25,
    ang_vel_std: float = 0.75,
    station_std: float = 0.22,
    racket_vel_std: float = 0.75,
    arm_pos_std: float = 0.35,
    arm_ori_std: float = 0.8,
    arm_body_names: list[str] | None = None,
    early_prestrike_window_steps: int = 12,
) -> torch.Tensor:
    """Reward being physically ready for the next sampled ball.

    This is stricter than generic post-strike recovery: it is active after the strike, during the hold,
    and for the first few control steps after a new target is sampled.  During that early pre-strike
    window the station term uses the new swing's dynamic station, so the policy is rewarded for being
    able to launch the next forehand/backhand rather than merely standing somewhere stable.
    """
    cmd = _cmd(env, command_name)
    motion = cmd._motion()
    score = _next_ready_score(
        env,
        cmd,
        height_std,
        upright_std,
        lin_vel_std,
        ang_vel_std,
        station_std,
        racket_vel_std,
        arm_pos_std,
        arm_ori_std,
        arm_body_names,
    )
    recovery_or_hold = ((~cmd.pre_strike) & (~cmd.strike_window)) | motion.in_hold
    early_next = cmd.pre_strike & (cmd.steps_since_target_resample <= int(early_prestrike_window_steps))
    return score * (recovery_or_hold | early_next).float()


def next_swing_ready_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    window_steps: int = 10,
    height_std: float = 0.09,
    upright_std: float = 0.25,
    lin_vel_std: float = 0.22,
    ang_vel_std: float = 0.65,
    station_std: float = 0.18,
    racket_vel_std: float = 0.65,
    arm_pos_std: float = 0.32,
    arm_ori_std: float = 0.75,
    arm_body_names: list[str] | None = None,
) -> torch.Tensor:
    """Explicit readiness bonus at the beginning of each newly sampled swing."""
    cmd = _cmd(env, command_name)
    score = _next_ready_score(
        env,
        cmd,
        height_std,
        upright_std,
        lin_vel_std,
        ang_vel_std,
        station_std,
        racket_vel_std,
        arm_pos_std,
        arm_ori_std,
        arm_body_names,
    )
    gate = cmd.pre_strike & (cmd.steps_since_target_resample <= int(window_steps))
    return score * gate.float()


def no_command_ready_stability(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_std: float = 0.09,
    upright_std: float = 0.22,
    lin_vel_std: float = 0.18,
    ang_vel_std: float = 0.55,
    station_std: float = 0.18,
    racket_vel_std: float = 0.55,
    arm_pos_std: float = 0.34,
    arm_ori_std: float = 0.78,
    arm_body_names: list[str] | None = None,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward stable deploy-ready posture only while no planner command is exposed.

    This makes the deploy/no-target hold a direct learning target without suppressing
    pre-strike or strike motion.  The gate is the synthetic no-command READY state
    used by ``RacketTargetCommand`` for deployment-lifecycle alignment.
    """
    cmd = _cmd(env, command_name)
    score = _next_ready_score(
        env,
        cmd,
        height_std,
        upright_std,
        lin_vel_std,
        ang_vel_std,
        station_std,
        racket_vel_std,
        arm_pos_std,
        arm_ori_std,
        arm_body_names,
    )
    return score * cmd.no_command_ready_active.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def next_prestrike_reachable(
    env: ManagerBasedRLEnv,
    command_name: str,
    window_steps: int = 20,
    reach_std: float = 0.30,
    feasible_speed: float = 2.60,
    feasible_speed_std: float = 0.45,
    min_time_to_strike: float = 0.18,
    max_time_to_strike: float = 1.20,
    station_std: float = 0.24,
    upright_std: float = 0.28,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward the next sampled ball being reachable from the carried recovery state.

    The term fires only during the first few pre-strike ticks after a new target is
    sampled.  It directly scores whether the racket is close enough, the required
    catch-up speed is physically reasonable, the base is near the dynamic station,
    and the trunk is upright.  It is not active during impact, so it should not
    damp the actual hit.
    """
    cmd = _cmd(env, command_name)
    tts = torch.clamp(cmd.true_time_to_strike, min=float(min_time_to_strike), max=float(max_time_to_strike))
    target_now = cmd.ball_strike_pos_w - cmd.racket_impact_target_vel_w * cmd.true_time_to_strike.unsqueeze(-1)
    racket_dist = torch.norm(cmd.racket_pos_w - target_now, dim=-1)
    reach = torch.exp(-torch.square(racket_dist / max(float(reach_std), 1.0e-6)))

    required_speed = racket_dist / tts
    feasible = torch.sigmoid(
        (float(feasible_speed) - required_speed) / max(float(feasible_speed_std), 1.0e-6)
    )

    station_err = torch.norm(cmd.base_pos_w[:, :2] - cmd.station_w, dim=-1)
    station = torch.exp(-torch.square(station_err / max(float(station_std), 1.0e-6)))
    upright_err = torch.norm(cmd.robot.data.projected_gravity_b[:, :2], dim=-1)
    upright = torch.exp(-torch.square(upright_err / max(float(upright_std), 1.0e-6)))

    score = 0.35 * reach + 0.25 * feasible + 0.20 * station + 0.20 * upright
    gate = cmd.pre_strike & (cmd.steps_since_target_resample <= int(window_steps)) & (~cmd.no_command_ready_active)
    return score * gate.float() * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def _previous_swing_outcome(cmd: RacketTargetCommand, outcome: str) -> torch.Tensor:
    if outcome == "opponent_bounce":
        return cmd.prev_swing_on_opponent.float()
    if outcome == "net_cross":
        return cmd.prev_swing_net_cross.float()
    if outcome == "contact":
        return cmd.prev_swing_contact.float()
    raise ValueError(
        "cycle_return_readiness outcome must be one of "
        f"'opponent_bounce', 'net_cross', or 'contact'; got {outcome!r}."
    )


def _current_swing_outcome(cmd: RacketTargetCommand, outcome: str) -> torch.Tensor:
    if outcome == "opponent_bounce":
        return cmd.current_swing_on_opponent.float()
    if outcome == "net_cross":
        return cmd.current_swing_net_cross.float()
    if outcome == "contact":
        return cmd.current_swing_contact.float()
    raise ValueError(
        "post_outcome_recovery_readiness outcome must be one of "
        f"'opponent_bounce', 'net_cross', or 'contact'; got {outcome!r}."
    )


def post_outcome_recovery_readiness(
    env: ManagerBasedRLEnv,
    command_name: str,
    outcome: str = "contact",
    height_std: float = 0.10,
    upright_std: float = 0.28,
    lin_vel_std: float = 0.25,
    ang_vel_std: float = 0.75,
    station_std: float = 0.22,
    racket_vel_std: float = 0.75,
    arm_pos_std: float = 0.35,
    arm_ori_std: float = 0.80,
    arm_body_names: list[str] | None = None,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward immediately recovering after a useful current-swing outcome.

    The cycle rewards below fire at the beginning of the next sampled swing.  This term fills the
    credit-assignment gap right after contact: once the current swing has contacted / cleared the net /
    landed, the remaining follow-through frames are rewarded for getting back to a reusable state.
    """
    cmd = _cmd(env, command_name)
    score = _next_ready_score(
        env,
        cmd,
        height_std,
        upright_std,
        lin_vel_std,
        ang_vel_std,
        station_std,
        racket_vel_std,
        arm_pos_std,
        arm_ori_std,
        arm_body_names,
    )
    post_strike = (~cmd.pre_strike) & (~cmd.strike_window)
    return (
        _current_swing_outcome(cmd, outcome)
        * score
        * post_strike.float()
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def post_outcome_arm_racket_readiness(
    env: ManagerBasedRLEnv,
    command_name: str,
    outcome: str = "contact",
    arm_pos_std: float = 0.30,
    arm_ori_std: float = 0.70,
    racket_vel_std: float = 0.55,
    arm_body_names: list[str] | None = None,
    include_hold: bool = True,
    early_prestrike_window_steps: int = 0,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward the right arm/racket returning to a reusable motion-prior pose after a useful outcome."""
    cmd = _cmd(env, command_name)
    motion = cmd._motion()
    arm = _body_pose_score(cmd, arm_body_names, arm_pos_std, arm_ori_std)
    racket = torch.exp(-torch.square(torch.norm(cmd.racket_lin_vel_w, dim=-1) / max(float(racket_vel_std), 1.0e-6)))
    score = 0.65 * arm + 0.35 * racket
    post_strike = (~cmd.pre_strike) & (~cmd.strike_window)
    if include_hold:
        post_strike = post_strike | motion.in_hold
    if early_prestrike_window_steps > 0:
        post_strike = post_strike | (
            cmd.pre_strike & (cmd.steps_since_target_resample <= int(early_prestrike_window_steps))
        )
    return (
        _current_swing_outcome(cmd, outcome)
        * score
        * post_strike.float()
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def cycle_return_readiness(
    env: ManagerBasedRLEnv,
    command_name: str,
    outcome: str = "opponent_bounce",
    window_steps: int = 14,
    height_std: float = 0.10,
    upright_std: float = 0.28,
    lin_vel_std: float = 0.25,
    ang_vel_std: float = 0.75,
    station_std: float = 0.22,
    racket_vel_std: float = 0.75,
    arm_pos_std: float = 0.35,
    arm_ori_std: float = 0.80,
    arm_body_names: list[str] | None = None,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward completing one rally cycle: previous return outcome plus next-swing readiness.

    The outcome is latched when a motion clip wraps into the next swing.  The reward then fires only
    in the first few pre-strike steps of the new swing, using that new swing's station target.  This
    ties hit quality to recoverability without changing the actor observation or hard terminations.
    """
    cmd = _cmd(env, command_name)
    score = _next_ready_score(
        env,
        cmd,
        height_std,
        upright_std,
        lin_vel_std,
        ang_vel_std,
        station_std,
        racket_vel_std,
        arm_pos_std,
        arm_ori_std,
        arm_body_names,
    )
    previous = _previous_swing_outcome(cmd, outcome)
    gate = cmd.pre_strike & (cmd.steps_since_target_resample <= int(window_steps))
    return previous * score * gate.float() * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def cycle_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    outcome: str = "opponent_bounce",
    window_steps: int = 20,
    ready_threshold: float = 0.72,
    ready_temperature: float = 0.04,
    height_std: float = 0.095,
    upright_std: float = 0.26,
    lin_vel_std: float = 0.24,
    ang_vel_std: float = 0.70,
    station_std: float = 0.21,
    racket_vel_std: float = 0.68,
    arm_pos_std: float = 0.34,
    arm_ori_std: float = 0.78,
    arm_body_names: list[str] | None = None,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward a complete rally cycle: useful previous return plus next-swing readiness.

    ``cycle_return_readiness`` gives a dense readiness score after a previous outcome.  This term is
    intentionally closer to an episode-success condition: it only fires in the first pre-strike ticks
    of the next swing, requires a previous hit outcome, and gates the readiness score through a sharp
    sigmoid around ``ready_threshold``.  It keeps gradients usable while making "hit and recover into a
    reusable next ball" the event that receives the large sparse bonus.
    """
    cmd = _cmd(env, command_name)
    score = _next_ready_score(
        env,
        cmd,
        height_std,
        upright_std,
        lin_vel_std,
        ang_vel_std,
        station_std,
        racket_vel_std,
        arm_pos_std,
        arm_ori_std,
        arm_body_names,
    )
    previous = _previous_swing_outcome(cmd, outcome)
    gate = (
        cmd.pre_strike
        & (cmd.steps_since_target_resample <= int(window_steps))
        & (~cmd.no_command_ready_active)
    )
    temp = max(float(ready_temperature), 1.0e-6)
    ready = torch.sigmoid((score - float(ready_threshold)) / temp)
    return previous * ready * gate.float() * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def cycle_v2_ready_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    tier_multipliers: tuple[float, float, float] = (1.0, 1.0, 1.0),
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """One-shot bonus when a previous useful return reaches next-swing ready before the deadline."""
    cmd = _cmd(env, command_name)
    multipliers = torch.tensor(tier_multipliers, dtype=torch.float32, device=cmd.device)
    if multipliers.shape != (3,):
        raise ValueError(f"tier_multipliers must contain contact/net/bounce values, got {tier_multipliers}")
    tier = torch.clamp(cmd.cycle_v2_outcome_tier, min=0, max=2)
    return cmd.cycle_v2_ready_success_event.float() * multipliers[tier] * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def cycle_v2_streak_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_streak: int = 2,
    min_ability_level: float = 0.70,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Extra one-shot bonus for consecutive closed cycles at high ability."""
    cmd = _cmd(env, command_name)
    event = cmd.cycle_v2_ready_success_event & (cmd.cycle_v2_streak >= int(min_streak))
    ability_ok = (cmd._ability_curriculum_level >= float(min_ability_level)).float()
    return event.float() * ability_ok * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def cycle_v2_ready_fail(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """One-shot event for previous useful return that fails to reach next-swing ready in time."""
    cmd = _cmd(env, command_name)
    fail_event = cmd.cycle_v2_ready_fail_event | cmd.cycle_v2_unresolved_resample_fail_event
    return fail_event.float() * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def post_strike_base_angular_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.60,
    include_hold: bool = True,
    early_prestrike_window_steps: int = 0,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward quickly damping pelvis/base angular velocity after a swing."""
    cmd = _cmd(env, command_name)
    gate = _recovery_or_hold_gate(cmd, include_hold, early_prestrike_window_steps).float()
    ang_vel = torch.norm(cmd.robot.data.root_ang_vel_w, dim=-1)
    calm = torch.exp(-torch.square(ang_vel / max(float(std), 1.0e-6)))
    return calm * gate * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def strike_balance(
    env: ManagerBasedRLEnv,
    command_name: str,
    pre_window_s: float = 0.34,
    post_window_s: float = 0.18,
    pitch_std: float = 0.16,
    upright_std: float = 0.26,
    ang_vel_std: float = 1.05,
    backward_std: float = 0.16,
    backward_vel_std: float = 0.42,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Keep the trunk from leaning/falling backward during the swing window.

    This is intentionally not an arm or lower-body action-rate term.  It targets the observed failure
    where the upper body throws angular momentum backward during pre-strike/impact, while still
    allowing the legs to make quick compensating motions.
    """
    cmd = _cmd(env, command_name)
    return _strike_balance_score(
        cmd,
        pre_window_s=pre_window_s,
        post_window_s=post_window_s,
        pitch_std=pitch_std,
        upright_std=upright_std,
        ang_vel_std=ang_vel_std,
        backward_std=backward_std,
        backward_vel_std=backward_vel_std,
    ) * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def _torso_com_balance_score(
    cmd: RacketTargetCommand,
    torso_pitch_std: float,
    torso_roll_std: float,
    torso_ang_vel_std: float,
    com_x_std: float,
    com_y_std: float,
) -> torch.Tensor:
    """Measure torso and support balance without constraining arm motion."""
    motion = cmd._motion()
    data = cmd.robot.data
    torso_idx = motion.cfg.body_names.index("torso_Link")
    torso_quat_w = motion.robot_body_quat_w[:, torso_idx]
    gravity_w = torch.zeros(cmd.num_envs, 3, device=cmd.device)
    gravity_w[:, 2] = -1.0
    torso_gravity = quat_rotate_inverse(torso_quat_w, gravity_w)
    torso_pitch = torch.exp(
        -torch.square(torso_gravity[:, 0] / max(float(torso_pitch_std), 1.0e-6))
    )
    torso_roll = torch.exp(
        -torch.square(torso_gravity[:, 1] / max(float(torso_roll_std), 1.0e-6))
    )
    torso_ang_vel = torch.exp(
        -torch.square(
            torch.norm(motion.robot_body_ang_vel_w[:, torso_idx, :2], dim=-1)
            / max(float(torso_ang_vel_std), 1.0e-6)
        )
    )

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
    com_delta_b = quat_rotate_inverse(yaw_quat(cmd.base_quat_w), com_delta_w)
    com_x = torch.exp(-torch.square(com_delta_b[:, 0] / max(float(com_x_std), 1.0e-6)))
    com_y = torch.exp(-torch.square(com_delta_b[:, 1] / max(float(com_y_std), 1.0e-6)))
    feet = torch.clamp(cmd.feet_contact_frac, 0.0, 1.0)
    return (
        0.22 * torso_pitch
        + 0.16 * torso_roll
        + 0.18 * torso_ang_vel
        + 0.20 * com_x
        + 0.14 * com_y
        + 0.10 * feet
    )


def torso_com_balance(
    env: ManagerBasedRLEnv,
    command_name: str,
    phase: str = "strike",
    pre_window_s: float = 0.36,
    post_window_s: float = 0.22,
    torso_pitch_std: float = 0.20,
    torso_roll_std: float = 0.20,
    torso_ang_vel_std: float = 1.30,
    com_x_std: float = 0.13,
    com_y_std: float = 0.16,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward torso/COM balance during strike or recovery without damping the arm."""
    cmd = _cmd(env, command_name)
    if phase == "strike":
        gate = (
            (cmd.true_time_to_strike <= float(pre_window_s))
            & (cmd.true_time_to_strike >= -float(post_window_s))
            & (~cmd.no_command_ready_active)
        )
    elif phase == "recovery_hold":
        gate = _recovery_or_hold_gate(cmd, include_hold=True)
    elif phase == "all":
        gate = torch.ones(cmd.num_envs, dtype=torch.bool, device=cmd.device)
    else:
        raise ValueError(f"unsupported torso_com_balance phase {phase!r}")
    score = _torso_com_balance_score(
        cmd,
        torso_pitch_std,
        torso_roll_std,
        torso_ang_vel_std,
        com_x_std,
        com_y_std,
    )
    return score * gate.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def prestrike_station_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_scale: float = 0.35,
    arrival_radius: float = 0.12,
    stop_window_s: float = 0.10,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward base velocity toward the dynamic station before the strike.

    Unlike a Gaussian station score, this remains informative when the sampled
    target is far away. It turns off near the station and near impact so it does
    not reward overshoot or interfere with the strike.
    """
    cmd = _cmd(env, command_name)
    delta = cmd.station_w - cmd.base_pos_w[:, :2]
    distance = torch.norm(delta, dim=-1)
    direction = delta / distance.unsqueeze(-1).clamp_min(1.0e-6)
    progress_speed = torch.sum(cmd.robot.data.root_lin_vel_w[:, :2] * direction, dim=-1)
    progress = torch.tanh(progress_speed / max(float(speed_scale), 1.0e-6))
    gate = (
        cmd.pre_strike
        & (cmd.true_time_to_strike > float(stop_window_s))
        & (distance > float(arrival_radius))
        & (~cmd.no_command_ready_active)
    )
    return progress * gate.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def lower_body_support(
    env: ManagerBasedRLEnv,
    command_name: str,
    pre_window_s: float = 0.36,
    post_window_s: float = 0.22,
    station_std: float = 0.28,
    backward_std: float = 0.18,
    backward_vel_std: float = 0.48,
    min_feet_contact: float = 0.50,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward support quality during the swing without making the legs globally stiff."""
    cmd = _cmd(env, command_name)
    data = cmd.robot.data
    tts = cmd.true_time_to_strike
    gate = (tts <= float(pre_window_s)) & (tts >= -float(post_window_s)) & (~cmd.no_command_ready_active)

    station_err = torch.norm(cmd.base_pos_w[:, :2] - cmd.station_w, dim=-1)
    station = torch.exp(-torch.square(station_err / max(float(station_std), 1.0e-6)))
    backward_offset = (cmd.station_w[:, 0] - cmd.base_pos_w[:, 0]).clamp_min(0.0)
    backward = torch.exp(-torch.square(backward_offset / max(float(backward_std), 1.0e-6)))
    backward_vel = (-data.root_lin_vel_w[:, 0]).clamp_min(0.0)
    back_vel = torch.exp(-torch.square(backward_vel / max(float(backward_vel_std), 1.0e-6)))
    feet = torch.clamp((cmd.feet_contact_frac - float(min_feet_contact)) / max(1.0 - float(min_feet_contact), 1.0e-6), 0.0, 1.0)

    score = 0.36 * feet + 0.26 * station + 0.22 * backward + 0.16 * back_vel
    return score * gate.float() * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def post_strike_lower_body_action_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_names: list[str] | None = None,
    include_hold: bool = True,
    early_prestrike_window_steps: int = 0,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Action-rate penalty for waist/leg joints during recovery and ready hold.

    Keep the reward term weight negative. The strike and most of pre-strike remain ungated so this
    does not suppress the policy's ability to step, shift, and swing toward a new ball.
    """
    if joint_names is None:
        joint_names = [
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
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
        ]
    indices = _canonical_joint_indices(tuple(joint_names))
    action = env.action_manager.action
    if len(indices) == 0:
        rate = torch.zeros(action.shape[0], device=action.device)
    else:
        idx = torch.tensor(indices, dtype=torch.long, device=action.device)
        rate = torch.sum(torch.square(action[:, idx] - env.action_manager.prev_action[:, idx]), dim=1)
    gate = _recovery_or_hold_gate(_cmd(env, command_name), include_hold, early_prestrike_window_steps).float()
    return rate * gate * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def phase_action_rate_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    joint_names: list[str] | None = None,
    pre_strike_scale: float = 0.04,
    strike_scale: float = 0.015,
    recovery_scale: float = 0.12,
    hold_scale: float = 0.16,
) -> torch.Tensor:
    """Action-rate penalty with different strength per swing phase.

    The reward term weight should be negative.  Strike/pre-strike stay light enough to allow a swing;
    recovery and hold are heavier so large residual changes after contact do not carry bad posture into
    the next ball.
    """
    cmd = _cmd(env, command_name)
    motion = cmd._motion()
    delta = env.action_manager.action - env.action_manager.prev_action
    if joint_names:
        indices = _canonical_joint_indices(tuple(joint_names))
        if indices:
            idx = torch.tensor(indices, dtype=torch.long, device=delta.device)
            delta = delta[:, idx]
        else:
            return torch.zeros(delta.shape[0], dtype=delta.dtype, device=delta.device)
    rate = torch.sum(torch.square(delta), dim=1)
    scale = torch.full_like(rate, float(pre_strike_scale))
    scale = torch.where(cmd.strike_window, torch.full_like(scale, float(strike_scale)), scale)
    recovery = (~cmd.pre_strike) & (~cmd.strike_window)
    scale = torch.where(recovery, torch.full_like(scale, float(recovery_scale)), scale)
    scale = torch.where(motion.in_hold, torch.full_like(scale, float(hold_scale)), scale)
    return rate * scale
