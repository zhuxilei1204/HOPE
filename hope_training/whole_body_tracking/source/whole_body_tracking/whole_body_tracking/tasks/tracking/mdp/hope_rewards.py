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
from whole_body_tracking.tasks.tracking.mdp.attempt_gating import (
    attempt_conditioned_phase_gate,
)
from whole_body_tracking.tasks.tracking.mdp.closed_loop_v2 import (
    health_floor_multiplier,
    outcome_tier_multiplier,
    planner_velocity_alignment_score,
    recovered_planner_command_settlement_score,
    recovered_planner_velocity_settlement_score,
    rigid_body_relative_point_velocity,
    safe_command_cycle_value,
    safety_conditioned_cycle_value,
    safety_conditioned_outcome_value,
    signed_directional_velocity_progress,
)
from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand
from whole_body_tracking.tasks.tracking.mdp.functional_ready import smooth_overflow_penalty
from whole_body_tracking.tasks.tracking.mdp.trunk_waist_stability import (
    one_sided_trunk_waist_score,
)
from whole_body_tracking.tasks.tracking.mdp.wrist_prestrike import (
    asymmetric_wrist_release_scale,
    planner_task_space_crossfade_gate,
    prestrike_alignment_ramp,
)
from whole_body_tracking.utils.action_adapter_config import load_joint_order

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env: ManagerBasedRLEnv, command_name: str) -> RacketTargetCommand:
    return env.command_manager.get_term(command_name)


def _event_manager_value(
    env: ManagerBasedRLEnv, value: torch.Tensor, *, impulse: bool
) -> torch.Tensor:
    """Convert a one-frame event to Isaac RewardManager units."""
    if not bool(impulse):
        return value
    step_dt = float(env.step_dt)
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive for impulse rewards")
    return value / step_dt


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


def _impact_health_multiplier(
    cmd: RacketTargetCommand,
    minimum_start: float,
    minimum_end: float | None = None,
) -> torch.Tensor:
    """Blend dense exploration into strict health-gated striking by capability."""
    start = min(max(float(minimum_start), 0.0), 1.0)
    end = start if minimum_end is None else min(max(float(minimum_end), 0.0), 1.0)
    level = cmd.impact_health_floor_progress()
    floor = start + level * (end - start)
    power = max(float(getattr(cmd.cfg, "impact_health_reward_power", 1.0)), 1.0)
    health = cmd.impact_health_score.clamp(0.0, 1.0).pow(power)
    return floor + (1.0 - floor) * health


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


def feet_contact_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    asset_cfg,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal foot velocity only while that foot supports the robot."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .amax(dim=1)
        > float(contact_force_threshold)
    )
    asset = env.scene[asset_cfg.name]
    foot_speed_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return torch.sum(foot_speed_xy * contact.float(), dim=-1)


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
    ability_scaled: bool = False,
    ability_start_scale: float = 1.0,
    ability_attempt_start: float = 0.0,
    ability_attempt_full: float = 1.0,
) -> torch.Tensor:
    """Soft penalty for tracked points entering or grazing the analytic table-top no-touch zone."""
    cmd = _cmd(env, command_name)
    points = _table_monitor_points(cmd, body_names, include_racket)
    if points is None:
        return torch.zeros(cmd.num_envs, device=cmd.device)
    value = _table_zone_score(
        cmd,
        points,
        x_margin,
        y_margin,
        below_surface_margin,
        above_surface_margin,
        distance_std,
    )
    if not ability_scaled:
        return value
    start = min(max(float(ability_start_scale), 0.0), 1.0)
    lo = float(ability_attempt_start)
    hi = max(float(ability_attempt_full), lo + 1.0e-6)
    progress = torch.clamp(
        (cmd._ability_targeted_attempt_ema - lo) / (hi - lo),
        0.0,
        1.0,
    )
    scale = start + (1.0 - start) * progress
    return value * scale


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
    racket_command_name: str | None = None,
    pre_strike_scale: float = 1.0,
    strike_scale: float = 1.0,
    recovery_scale: float = 1.0,
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
    if racket_command_name is not None:
        racket = _cmd(env, racket_command_name)
        phase_scale = torch.full_like(score, float(recovery_scale))
        phase_scale = torch.where(
            racket.pre_strike,
            torch.full_like(phase_scale, float(pre_strike_scale)),
            phase_scale,
        )
        phase_scale = torch.where(
            racket.strike_window,
            torch.full_like(phase_scale, float(strike_scale)),
            phase_scale,
        )
        scale = scale * phase_scale
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
    pre_release_start_s: float | None = None,
    post_full_release_s: float | None = None,
    post_release_end_s: float | None = None,
    use_planner_time_to_strike: bool = False,
    hold_scale: float = 1.0,
) -> torch.Tensor:
    """Track the racket wrist from the clip, but soften it near impact.

    ``release_scale`` keeps a non-zero wrist reference through the strike window.  A value of 0.0
    reproduces the old full release; values in roughly [0.25, 0.5] allow hit correction without
    letting the policy solve the task by destroying the forehand/backhand wrist pose.
    """
    value = _imitation.motion_relative_body_position_error_exp(env, motion_command_name, std, body_names)
    racket = _cmd(env, racket_command_name)
    if pre_release_start_s is not None or post_release_end_s is not None:
        if pre_release_start_s is None or post_release_end_s is None:
            raise ValueError(
                "pre_release_start_s and post_release_end_s must be configured together"
            )
        time_to_strike = (
            racket.time_to_strike
            if use_planner_time_to_strike
            else racket.true_time_to_strike
        )
        scale = asymmetric_wrist_release_scale(
            time_to_strike,
            pre_release_start_s=pre_release_start_s,
            strike_half_window_s=release_window_s,
            post_strike_full_release_s=post_full_release_s,
            post_release_end_s=post_release_end_s,
            strike_scale=release_scale,
        )
    else:
        release = racket.true_time_to_strike.abs() <= release_window_s
        scale = torch.where(
            release,
            torch.full_like(value, float(release_scale)),
            torch.ones_like(value),
        )
    motion = env.command_manager.get_term(motion_command_name)
    hold = torch.where(
        motion.in_hold,
        torch.full_like(value, float(hold_scale)),
        torch.ones_like(value),
    )
    return value * scale * hold


def wrist_motion_ori_release(
    env: ManagerBasedRLEnv,
    motion_command_name: str,
    racket_command_name: str,
    std: float,
    body_names: list[str] | None = None,
    release_window_s: float = 0.20,
    release_scale: float = 0.35,
    pre_release_start_s: float | None = None,
    post_full_release_s: float | None = None,
    post_release_end_s: float | None = None,
    use_planner_time_to_strike: bool = False,
    hold_scale: float = 1.0,
) -> torch.Tensor:
    """Track the racket wrist orientation from the clip, but soften it near impact."""
    value = _imitation.motion_relative_body_orientation_error_exp(env, motion_command_name, std, body_names)
    racket = _cmd(env, racket_command_name)
    if pre_release_start_s is not None or post_release_end_s is not None:
        if pre_release_start_s is None or post_release_end_s is None:
            raise ValueError(
                "pre_release_start_s and post_release_end_s must be configured together"
            )
        time_to_strike = (
            racket.time_to_strike
            if use_planner_time_to_strike
            else racket.true_time_to_strike
        )
        scale = asymmetric_wrist_release_scale(
            time_to_strike,
            pre_release_start_s=pre_release_start_s,
            strike_half_window_s=release_window_s,
            post_strike_full_release_s=post_full_release_s,
            post_release_end_s=post_release_end_s,
            strike_scale=release_scale,
        )
    else:
        release = racket.true_time_to_strike.abs() <= release_window_s
        scale = torch.where(
            release,
            torch.full_like(value, float(release_scale)),
            torch.ones_like(value),
        )
    motion = env.command_manager.get_term(motion_command_name)
    hold = torch.where(
        motion.in_hold,
        torch.full_like(value, float(hold_scale)),
        torch.ones_like(value),
    )
    return value * scale * hold


# --- (3,4,5) racket goal tracking, active in the strike window ------------------------------ #
def racket_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    minimum_health_multiplier: float = 1.0,
    final_minimum_health_multiplier: float | None = None,
) -> torch.Tensor:
    """Track the racket center against the hidden true strike point near impact."""
    cmd = _cmd(env, command_name)
    target_now = cmd.ball_strike_pos_w - cmd.racket_impact_target_vel_w * cmd.true_time_to_strike.unsqueeze(-1)
    error = torch.sum(torch.square(cmd.racket_pos_w - target_now), dim=-1)
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return torch.exp(-error / std**2) * cmd.strike_window.float() * health


def _swing_side_gate(cmd: RacketTargetCommand, swing_side: float) -> torch.Tensor:
    if abs(float(swing_side)) < 1.0e-6:
        return torch.ones(cmd.num_envs, dtype=torch.bool, device=cmd.device)
    return cmd.swing_sign >= 0.0 if float(swing_side) > 0.0 else cmd.swing_sign < 0.0


def side_racket_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    swing_side: float = -1.0,
    minimum_health_multiplier: float = 1.0,
    final_minimum_health_multiplier: float | None = None,
) -> torch.Tensor:
    """Side-gated racket-position shaping; ``swing_side=-1`` targets backhand."""
    cmd = _cmd(env, command_name)
    return racket_position(
        env,
        command_name,
        std,
        minimum_health_multiplier,
        final_minimum_health_multiplier,
    ) * _swing_side_gate(cmd, swing_side).float()


def racket_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    minimum_health_multiplier: float = 1.0,
    final_minimum_health_multiplier: float | None = None,
) -> torch.Tensor:
    """Track the racket linear velocity against the impact-inverted racket velocity."""
    cmd = _cmd(env, command_name)
    error = torch.sum(torch.square(cmd.racket_lin_vel_w - cmd.racket_impact_target_vel_w), dim=-1)
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return torch.exp(-error / std**2) * cmd.strike_window.float() * health


def racket_velocity_projection(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_speed_ratio: float = 0.82,
    speed_std: float = 0.35,
    lateral_std: float = 0.75,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
    minimum_health_multiplier: float = 1.0,
    final_minimum_health_multiplier: float | None = None,
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
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return speed_score * lateral_score * cmd.strike_window.float() * health * _phase_scale(
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
    minimum_health_multiplier: float = 1.0,
    final_minimum_health_multiplier: float | None = None,
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
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return forward * upward * cmd.strike_window.float() * health * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def impact_outgoing_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
    minimum_health_multiplier: float = 1.0,
    final_minimum_health_multiplier: float | None = None,
) -> torch.Tensor:
    """Track the predicted post-impact ball velocity against the desired outgoing velocity."""
    cmd = _cmd(env, command_name)
    value = torch.exp(-torch.square(cmd.impact_ball_out_error / std)) * cmd.strike_window.float()
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return value * health * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def _soft_geometric_score(scores: tuple[torch.Tensor, ...], floor: float) -> torch.Tensor:
    """Combine required components without losing all gradient when one is initially poor."""
    minimum = min(max(float(floor), 0.0), 0.95)
    stacked = torch.stack(tuple(score.clamp(0.0, 1.0) for score in scores), dim=-1)
    lifted = minimum + (1.0 - minimum) * stacked
    geometric = torch.exp(torch.mean(torch.log(lifted.clamp_min(1.0e-8)), dim=-1))
    return ((geometric - minimum) / (1.0 - minimum)).clamp(0.0, 1.0)


def _face_quality_multiplier(
    quality: torch.Tensor,
    *,
    power: float,
    floor: float,
) -> torch.Tensor:
    """Leave legacy rewards unchanged at power zero; otherwise gate by face quality."""
    exponent = float(power)
    if exponent < 0.0:
        raise ValueError("face_quality_power must be non-negative")
    if exponent == 0.0:
        return torch.ones_like(quality)
    minimum = min(max(float(floor), 0.0), 1.0)
    return minimum + (1.0 - minimum) * quality.clamp(0.0, 1.0).pow(
        exponent
    )


def planner_racket_task_space_crossfade(
    env: ManagerBasedRLEnv,
    command_name: str,
    pre_start_s: float = 0.40,
    pre_full_s: float = 0.15,
    post_full_s: float = 0.08,
    post_end_s: float = 0.22,
    position_std: float = 0.14,
    velocity_std: float = 0.95,
    normal_std_rad: float = 0.30,
    ability_scaled_stds: bool = False,
    initial_position_std: float = 0.45,
    initial_velocity_std: float = 2.50,
    initial_normal_std_rad: float = 1.20,
    component_floor: float = 0.04,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
    minimum_health_multiplier: float = 0.25,
    final_minimum_health_multiplier: float | None = 0.10,
    action_feasibility_metric: str | None = None,
    action_feasibility_floor: float = 1.0,
) -> torch.Tensor:
    """Hand authority from wrist motion prior to the actor-visible planner command.

    Position follows a clipped moving pre-impact target, velocity ramps from
    zero to the planner velocity with the same gate, and blade normal aligns
    throughout the active gate. The gate is disabled during motion holds and
    deploy ready/no-command states.
    """
    cmd = _cmd(env, command_name)
    if ability_scaled_stds:
        level = torch.clamp(cmd._ability_curriculum_level, 0.0, 1.0)
        position_std = float(initial_position_std) + level * (
            float(position_std) - float(initial_position_std)
        )
        velocity_std = float(initial_velocity_std) + level * (
            float(velocity_std) - float(initial_velocity_std)
        )
        normal_std_rad = float(initial_normal_std_rad) + level * (
            float(normal_std_rad) - float(initial_normal_std_rad)
        )
    gate = planner_task_space_crossfade_gate(
        cmd.time_to_strike,
        pre_start_s=pre_start_s,
        pre_full_s=pre_full_s,
        post_full_s=post_full_s,
        post_end_s=post_end_s,
    )
    motion = cmd._motion()
    active = (~motion.in_hold) & (~cmd.no_command_ready_active)
    gate = gate * active.float()

    path_time = cmd.time_to_strike.clamp(min=-float(post_full_s), max=float(pre_full_s))
    path_target = cmd.racket_target_pos_w - cmd.racket_target_vel_w * path_time.unsqueeze(-1)
    position_error = torch.norm(cmd.racket_pos_w - path_target, dim=-1)
    position_scale = torch.as_tensor(position_std, device=cmd.device).clamp_min(1.0e-6)
    position_score = torch.exp(-torch.square(position_error / position_scale))

    velocity_target = gate.unsqueeze(-1) * cmd.racket_target_vel_w
    velocity_error = torch.norm(cmd.racket_lin_vel_w - velocity_target, dim=-1)
    velocity_scale = torch.as_tensor(velocity_std, device=cmd.device).clamp_min(1.0e-6)
    velocity_score = torch.exp(-torch.square(velocity_error / velocity_scale))

    actual_normal = cmd.racket_normal_w / torch.norm(
        cmd.racket_normal_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    target_normal = cmd.racket_target_normal_w / torch.norm(
        cmd.racket_target_normal_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    normal_angle = torch.acos(
        torch.sum(actual_normal * target_normal, dim=-1).clamp(-1.0, 1.0)
    )
    normal_scale = torch.as_tensor(normal_std_rad, device=cmd.device).clamp_min(1.0e-6)
    normal_score = torch.exp(-torch.square(normal_angle / normal_scale))
    joint_score = _soft_geometric_score(
        (position_score, velocity_score, normal_score), component_floor
    )
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    feasibility = torch.ones_like(health)
    if action_feasibility_metric is not None:
        if action_feasibility_metric not in cmd.metrics:
            raise KeyError(
                "planner_racket_task_space_crossfade is missing action "
                f"feasibility metric {action_feasibility_metric!r}"
            )
        floor = min(max(float(action_feasibility_floor), 0.0), 1.0)
        score = cmd.metrics[action_feasibility_metric].clamp(0.0, 1.0)
        feasibility = floor + (1.0 - floor) * score
    return (
        gate
        * joint_score
        * health
        * feasibility
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def prestrike_racket_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_scale: float = 0.60,
    arrival_radius: float = 0.18,
    stop_window_s: float = 0.20,
    path_lookahead_s: float = 0.35,
    idle_cost: float = 0.06,
    velocity_frame: str = "world",
    minimum_health_multiplier: float = 0.30,
    final_minimum_health_multiplier: float | None = 0.08,
) -> torch.Tensor:
    """Reward commanded racket motion toward the visible pre-impact path.

    A stationary racket outside the arrival neighborhood pays a small cost;
    healthy motion toward the path must exceed that cost to become positive.
    The term switches off where precise position/contact rewards take over.
    """
    cmd = _cmd(env, command_name)
    path_time = cmd.time_to_strike.clamp(
        min=0.0, max=max(float(path_lookahead_s), 0.0)
    )
    path_target = (
        cmd.racket_target_pos_w
        - cmd.racket_target_vel_w * path_time.unsqueeze(-1)
    )
    delta = path_target - cmd.racket_pos_w
    distance = torch.norm(delta, dim=-1)
    direction = delta / distance.unsqueeze(-1).clamp_min(1.0e-6)
    progress_velocity = cmd.racket_lin_vel_w
    if velocity_frame == "torso_relative":
        data = cmd.robot.data
        torso_idx = cmd._impact_health_torso_index
        progress_velocity = rigid_body_relative_point_velocity(
            cmd.racket_pos_w,
            cmd.racket_lin_vel_w,
            data.body_pos_w[:, torso_idx],
            data.body_lin_vel_w[:, torso_idx],
            data.body_ang_vel_w[:, torso_idx],
        )
    elif velocity_frame != "world":
        raise ValueError(
            "velocity_frame must be 'world' or 'torso_relative', "
            f"got {velocity_frame!r}"
        )
    progress_speed = torch.sum(progress_velocity * direction, dim=-1)
    progress = torch.tanh(
        progress_speed / max(float(speed_scale), 1.0e-6)
    )
    active = (
        cmd.pre_strike
        & (cmd.time_to_strike > float(stop_window_s))
        & (distance > float(arrival_radius))
        & (~cmd.no_command_ready_active)
    )
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return (progress * health - float(idle_cost)) * active.float()


def nonfarmable_prestrike_station_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_scale: float = 0.25,
    arrival_radius: float = 0.05,
    stop_window_s: float = 0.20,
    idle_cost: float = 0.03,
    minimum_health_multiplier: float = 0.30,
    final_minimum_health_multiplier: float | None = 0.10,
    include_no_command_ready: bool = False,
) -> torch.Tensor:
    """Reward active base relocation without paying for standing at a station."""
    cmd = _cmd(env, command_name)
    delta = cmd.station_w - cmd.base_pos_w[:, :2]
    distance = torch.norm(delta, dim=-1)
    direction = delta / distance.unsqueeze(-1).clamp_min(1.0e-6)
    progress_speed = torch.sum(
        cmd.robot.data.root_lin_vel_w[:, :2] * direction,
        dim=-1,
    )
    progress = torch.tanh(
        progress_speed / max(float(speed_scale), 1.0e-6)
    )
    prestrike_active = (
        cmd.pre_strike
        & (cmd.time_to_strike > float(stop_window_s))
        & (~cmd.no_command_ready_active)
    )
    active = prestrike_active
    if include_no_command_ready:
        active |= cmd.station_relocation_active_mask()
    active &= distance > float(arrival_radius)
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return (progress * health - float(idle_cost)) * active.float()


def near_impact_planner_velocity_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    pre_start_s: float = 0.22,
    pre_full_s: float = 0.10,
    post_full_s: float = 0.04,
    post_end_s: float = 0.08,
    position_std: float = 0.30,
    position_floor: float = 0.18,
    normal_std_rad: float = 0.65,
    normal_floor: float = 0.18,
    projection_ratio_scale: float = 0.65,
    lateral_ratio_scale: float = 0.50,
    lateral_weight: float = 0.25,
    idle_cost: float = 0.05,
    minimum_health_multiplier: float = 0.15,
    final_minimum_health_multiplier: float | None = 0.0,
) -> torch.Tensor:
    """Provide signed planner-velocity gradient in the final strike phase."""
    cmd = _cmd(env, command_name)
    gate = planner_task_space_crossfade_gate(
        cmd.time_to_strike,
        pre_start_s=pre_start_s,
        pre_full_s=pre_full_s,
        post_full_s=post_full_s,
        post_end_s=post_end_s,
    )
    velocity_progress = signed_directional_velocity_progress(
        cmd.racket_lin_vel_w,
        cmd.racket_target_vel_w,
        projection_ratio_scale=projection_ratio_scale,
        lateral_ratio_scale=lateral_ratio_scale,
        lateral_weight=lateral_weight,
    )

    position_error = torch.norm(
        cmd.racket_pos_w - cmd.racket_target_pos_w, dim=-1
    )
    position_score = torch.exp(
        -torch.square(position_error / max(float(position_std), 1.0e-6))
    )
    position_gate = float(position_floor) + (
        1.0 - float(position_floor)
    ) * position_score

    actual_normal = cmd.racket_normal_w / torch.norm(
        cmd.racket_normal_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    target_normal = cmd.racket_target_normal_w / torch.norm(
        cmd.racket_target_normal_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    normal_error = torch.acos(
        torch.sum(actual_normal * target_normal, dim=-1).clamp(-1.0, 1.0)
    )
    normal_score = torch.exp(
        -torch.square(normal_error / max(float(normal_std_rad), 1.0e-6))
    )
    normal_gate = float(normal_floor) + (
        1.0 - float(normal_floor)
    ) * normal_score

    task_gate = torch.sqrt(
        position_gate.clamp(0.0, 1.0) * normal_gate.clamp(0.0, 1.0)
    )
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    active = (~cmd.no_command_ready_active) & (~cmd._motion().in_hold)
    return gate * (
        velocity_progress * task_gate * health - float(idle_cost)
    ) * active.float()


def exact_impact_planner_task_space_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_std: float = 0.08,
    speed_ratio_std: float = 0.18,
    direction_std_rad: float = 0.25,
    normal_std_rad: float = 0.18,
    component_floor: float = 0.03,
    require_contact: bool = True,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
    minimum_health_multiplier: float = 0.15,
    final_minimum_health_multiplier: float | None = 0.05,
    face_quality_power: float = 0.0,
    face_quality_floor: float = 0.0,
    impulse: bool = False,
) -> torch.Tensor:
    """Jointly settle planner position, velocity, normal, timing, and health at impact."""
    cmd = _cmd(env, command_name)
    actual_velocity = cmd.racket_lin_vel_w
    target_velocity = cmd.racket_target_vel_w
    actual_speed = torch.norm(actual_velocity, dim=-1)
    target_speed = torch.norm(target_velocity, dim=-1).clamp_min(1.0e-6)

    position_error = torch.norm(cmd.racket_pos_w - cmd.racket_target_pos_w, dim=-1)
    position_score = torch.exp(
        -torch.square(position_error / max(float(position_std), 1.0e-6))
    )
    speed_ratio = actual_speed / target_speed
    speed_score = torch.exp(
        -torch.square((speed_ratio - 1.0) / max(float(speed_ratio_std), 1.0e-6))
    )
    velocity_cosine = torch.sum(actual_velocity * target_velocity, dim=-1) / (
        actual_speed.clamp_min(1.0e-6) * target_speed
    )
    velocity_angle = torch.acos(velocity_cosine.clamp(-1.0, 1.0))
    direction_score = torch.exp(
        -torch.square(velocity_angle / max(float(direction_std_rad), 1.0e-6))
    )

    actual_normal = cmd.racket_normal_w / torch.norm(
        cmd.racket_normal_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    target_normal = cmd.racket_target_normal_w / torch.norm(
        cmd.racket_target_normal_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    normal_angle = torch.acos(
        torch.sum(actual_normal * target_normal, dim=-1).clamp(-1.0, 1.0)
    )
    normal_score = torch.exp(
        -torch.square(normal_angle / max(float(normal_std_rad), 1.0e-6))
    )
    joint_score = _soft_geometric_score(
        (position_score, speed_score, direction_score, normal_score),
        component_floor,
    )
    contact_gate = cmd.ball_contact.float() if require_contact else torch.ones_like(joint_score)
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    face_quality = _face_quality_multiplier(
        cmd.impact_face_quality,
        power=face_quality_power,
        floor=face_quality_floor,
    )
    value = (
        joint_score
        * cmd.strike_fired.float()
        * contact_gate
        * health
        * face_quality
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )
    return _event_manager_value(env, value, impulse=impulse)


def exact_impact_racket_velocity_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_ratio_std: float = 0.20,
    direction_std_rad: float = 0.35,
    position_std: float = 0.16,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
    minimum_health_multiplier: float = 0.15,
    final_minimum_health_multiplier: float | None = 0.05,
) -> torch.Tensor:
    """Align racket speed magnitude and direction at the exact planned impact step.

    The existing velocity rewards shape the full strike window. This term closes the timing
    loophole by paying only when ``strike_fired`` is true, and only when the racket is also near
    the planned contact point.
    """
    cmd = _cmd(env, command_name)
    target = cmd.racket_impact_target_vel_w
    actual = cmd.racket_lin_vel_w
    target_speed = torch.norm(target, dim=-1).clamp_min(1.0e-6)
    actual_speed = torch.norm(actual, dim=-1)
    speed_ratio = actual_speed / target_speed
    speed_score = torch.exp(
        -torch.square((speed_ratio - 1.0) / max(float(speed_ratio_std), 1.0e-6))
    )
    cosine = torch.sum(actual * target, dim=-1) / (
        actual_speed.clamp_min(1.0e-6) * target_speed
    )
    angle = torch.acos(cosine.clamp(-1.0, 1.0))
    direction_score = torch.exp(
        -torch.square(angle / max(float(direction_std_rad), 1.0e-6))
    )
    position_error = torch.norm(cmd.racket_pos_w - cmd.ball_strike_pos_w, dim=-1)
    position_score = torch.exp(
        -torch.square(position_error / max(float(position_std), 1.0e-6))
    )
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return (
        speed_score
        * direction_score
        * position_score
        * cmd.strike_fired.float()
        * health
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def exact_impact_outgoing_velocity_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.5,
    position_std: float = 0.16,
    require_contact: bool = True,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
    minimum_health_multiplier: float = 0.15,
    final_minimum_health_multiplier: float | None = 0.05,
) -> torch.Tensor:
    """Match the analytic moving-racket post-impact velocity at the exact impact step.

    This is physics-facing shaping from the command term's no-spin collision model. It is not a
    replacement for a simulated ball-racket contact; contact, net-cross, and opponent-bounce
    rewards remain the outcome gates.
    """
    cmd = _cmd(env, command_name)
    velocity_score = torch.exp(
        -torch.square(cmd.impact_ball_out_error / max(float(std), 1.0e-6))
    )
    position_error = torch.norm(cmd.racket_pos_w - cmd.ball_strike_pos_w, dim=-1)
    position_score = torch.exp(
        -torch.square(position_error / max(float(position_std), 1.0e-6))
    )
    contact_gate = cmd.ball_contact.float() if require_contact else torch.ones_like(velocity_score)
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return (
        velocity_score
        * position_score
        * contact_gate
        * cmd.strike_fired.float()
        * health
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def closed_loop_v2_planner_velocity_band(
    env: ManagerBasedRLEnv,
    command_name: str,
    timing_std_s: float = 0.055,
    active_window_s: float = 0.12,
    position_std: float = 0.12,
    position_floor: float = 0.10,
    speed_ratio_std: float = 0.30,
    direction_std_rad: float = 0.35,
    component_floor: float = 0.02,
) -> torch.Tensor:
    """Track the actor-visible planner velocity in a narrow impact band.

    The exact-impact reward is contact-gated and settles on one control step.
    This term exposes a short, smooth correction signal before contact while
    retaining task-space position and impact-health gates. It never uses the
    hidden ideal command.
    """
    cmd = _cmd(env, command_name)
    velocity_score = planner_velocity_alignment_score(
        cmd.racket_lin_vel_w,
        cmd.racket_target_vel_w,
        speed_ratio_std=speed_ratio_std,
        direction_std_rad=direction_std_rad,
        component_floor=component_floor,
    )
    position_error = torch.norm(
        cmd.racket_pos_w - cmd.racket_target_pos_w, dim=-1
    )
    position_score = torch.exp(
        -torch.square(position_error / max(float(position_std), 1.0e-6))
    )
    minimum_position = min(max(float(position_floor), 0.0), 1.0)
    position_gate = (
        minimum_position + (1.0 - minimum_position) * position_score
    )
    timing = torch.exp(
        -torch.square(
            cmd.time_to_strike / max(float(timing_std_s), 1.0e-6)
        )
    )
    active = (
        (cmd.time_to_strike.abs() <= float(active_window_s))
        & (~cmd.no_command_ready_active)
        & (~cmd._motion().in_hold)
    )
    return (
        velocity_score
        * position_gate
        * timing
        * cmd.impact_health_score.clamp(0.0, 1.0)
        * active.float()
    )


def racket_blade_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
    minimum_health_multiplier: float = 1.0,
    final_minimum_health_multiplier: float | None = None,
) -> torch.Tensor:
    """Align the racket face normal with the desired blade direction near the strike (``std`` in rad)."""
    cmd = _cmd(env, command_name)
    cos_ang = torch.sum(cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    value = torch.exp(-(angle**2) / std**2) * cmd.strike_window.float()
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return value * health * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def prestrike_blade_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    start_s: float = 0.35,
    full_s: float = 0.12,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Prepare the blade normal before impact without rewarding early target arrival."""
    cmd = _cmd(env, command_name)
    cos_ang = torch.sum(
        cmd.racket_normal_w * cmd.racket_target_normal_w, dim=-1
    ).clamp(-1.0, 1.0)
    angle = torch.acos(cos_ang)
    ramp = prestrike_alignment_ramp(
        cmd.true_time_to_strike,
        start_s=start_s,
        full_s=full_s,
    )
    value = torch.exp(-(angle**2) / std**2) * ramp
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
    minimum_health_multiplier: float = 1.0,
    final_minimum_health_multiplier: float | None = None,
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
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return (
        contact_like
        * forward
        * upward
        * margin
        * valid
        * side
        * health
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def robust_contact_net_margin(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_std: float = 0.17,
    approach_speed: float = 0.10,
    approach_std: float = 0.75,
    normal_speed: float = 0.0,
    normal_std: float = 0.75,
    window_s: float = 0.18,
    restitution_range: tuple[float, float] = (0.50, 0.70),
    tangent_retain_range: tuple[float, float] = (0.45, 0.75),
    min_forward_speed: float = 2.70,
    forward_std: float = 0.35,
    min_clearance: float = 0.05,
    clearance_std: float = 0.06,
    swing_side: float = 0.0,
    minimum_health_multiplier: float = 1.0,
    final_minimum_health_multiplier: float | None = None,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward net clearance across a small ensemble of plausible paddle collisions.

    The training command normally predicts one outgoing velocity using one fixed
    restitution/tangential-retention pair. Physical contacts vary across PhysX,
    MuJoCo, and hardware. This term evaluates the four corners of a configured
    parameter box and optimizes the worst score, so a strike cannot earn the
    full reward by exploiting only one collision model.
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

    normal = cmd.racket_normal_w / (
        torch.norm(cmd.racket_normal_w, dim=-1, keepdim=True) + 1.0e-6
    )
    relative_in = cmd.incoming_ball_vel_w - cmd.racket_lin_vel_w
    relative_normal = torch.sum(relative_in * normal, dim=-1, keepdim=True) * normal
    relative_tangent = relative_in - relative_normal

    restitution_lo, restitution_hi = (float(v) for v in restitution_range)
    retain_lo, retain_hi = (float(v) for v in tangent_retain_range)
    if restitution_lo > restitution_hi or retain_lo > retain_hi:
        raise ValueError("robust collision parameter ranges must be ordered")

    p0 = cmd.ball_strike_pos_w - cmd._env.scene.env_origins
    net_x = float(cmd.cfg.table_near_x) + float(cmd.cfg.net_x)
    net_top = (
        float(cmd.cfg.table_surface_z)
        + float(cmd.cfg.net_height)
        + float(cmd.cfg.net_margin)
    )
    scores = []
    for restitution in (restitution_lo, restitution_hi):
        for retain in (retain_lo, retain_hi):
            out = (
                cmd.racket_lin_vel_w
                + retain * relative_tangent
                - restitution * relative_normal
            )
            vx = out[:, 0]
            vz = out[:, 2]
            t_net = (net_x - p0[:, 0]) / vx.clamp_min(1.0e-3)
            z_net = p0[:, 2] + vz * t_net - 0.5 * 9.81 * t_net.square()
            clearance = z_net - net_top
            forward = torch.sigmoid(
                (vx - float(min_forward_speed)) / max(float(forward_std), 1.0e-6)
            )
            margin = torch.sigmoid(
                (clearance - float(min_clearance)) / max(float(clearance_std), 1.0e-6)
            )
            scores.append(forward * margin * (t_net > 0.0).float())

    robust_score = torch.stack(scores, dim=0).amin(dim=0)
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return (
        contact_like
        * robust_score
        * _swing_side_gate(cmd, swing_side).float()
        * health
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
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


def health_gated_soft_ball_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_health_multiplier: float = 0.0,
    final_minimum_health_multiplier: float | None = None,
    pos_std: float = 0.18,
    approach_speed: float = 0.15,
    approach_std: float = 0.75,
    normal_speed: float = 0.0,
    normal_std: float = 0.75,
    window_s: float = 0.20,
    swing_side: float = 0.0,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Dense contact shaping multiplied by direct torso/COM impact health."""
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
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return (
        contact_like
        * health
        * _swing_side_gate(cmd, swing_side).float()
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def health_gated_ball_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_health_multiplier: float = 0.0,
    final_minimum_health_multiplier: float | None = None,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Actual contact event multiplied by impact health from the same frame."""
    cmd = _cmd(env, command_name)
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return cmd.ball_contact.float() * health * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def health_gated_net_cross(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_health_multiplier: float = 0.0,
    final_minimum_health_multiplier: float | None = None,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Net-clear event multiplied by impact health from the same frame."""
    cmd = _cmd(env, command_name)
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return cmd.ball_net_cross.float() * health * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def health_gated_opponent_bounce(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_health_multiplier: float = 0.0,
    final_minimum_health_multiplier: float | None = None,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Opponent-bounce event multiplied by impact health from the same frame."""
    cmd = _cmd(env, command_name)
    health = _impact_health_multiplier(
        cmd, minimum_health_multiplier, final_minimum_health_multiplier
    )
    return cmd.ball_on_opponent.float() * health * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def healthy_trunk_support(
    env: ManagerBasedRLEnv,
    command_name: str,
    pre_window_s: float = 0.34,
    post_window_s: float = 0.20,
    include_no_command_ready: bool = True,
) -> torch.Tensor:
    """Reward one-sided trunk/COM support without constraining arms or legs."""
    cmd = _cmd(env, command_name)
    gate = (
        (cmd.true_time_to_strike <= float(pre_window_s))
        & (cmd.true_time_to_strike >= -float(post_window_s))
    )
    if include_no_command_ready:
        gate |= cmd.no_command_ready_active
    return cmd.impact_health_score * gate.float()


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


def targeted_strike_attempt(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_health_multiplier: float = 0.40,
) -> torch.Tensor:
    """One-shot reward for a target-directed swing at the true strike frame."""
    cmd = _cmd(env, command_name)
    minimum = min(max(float(minimum_health_multiplier), 0.0), 1.0)
    health = minimum + (1.0 - minimum) * cmd.impact_health_score
    event = cmd.strike_fired & cmd.current_swing_targeted_attempt
    return event.float() * health


def strike_inactivity(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """One-shot event when a visible strike opportunity receives no targeted swing."""
    cmd = _cmd(env, command_name)
    return (
        cmd.strike_fired
        & (~cmd.current_swing_targeted_attempt)
        & (~cmd.no_command_ready_active)
    ).float()


def targeted_contact_miss(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_health_multiplier: float = 0.50,
    impulse: bool = False,
) -> torch.Tensor:
    """One-shot cost when a real targeted attempt misses the contact region."""
    cmd = _cmd(env, command_name)
    minimum = min(max(float(minimum_health_multiplier), 0.0), 1.0)
    health = minimum + (1.0 - minimum) * cmd.impact_health_score
    event = (
        cmd.strike_fired
        & cmd.current_swing_targeted_attempt
        & (~cmd.ball_contact)
        & (~cmd.no_command_ready_active)
    )
    return _event_manager_value(
        env, event.float() * health, impulse=impulse
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
    require_targeted_attempt: bool = False,
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
    gate = attempt_conditioned_phase_gate(
        gate,
        torch.zeros_like(gate),
        cmd.current_swing_targeted_attempt,
        cmd.prev_swing_targeted_attempt,
        require_targeted_attempt,
    )
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
    require_targeted_attempt: bool = False,
) -> torch.Tensor:
    """Reward a deploy-ready stance after the strike and during the ready hold.

    This complements ``follow_through_recovery``: it explicitly scores base height, uprightness,
    residual base motion, station drift, and foot contact. It is gated off before the strike so it
    does not suppress the swing itself.
    """
    cmd = _cmd(env, command_name)
    data = cmd.robot.data
    in_hold = cmd._motion().in_hold
    gate = ((~cmd.pre_strike) & (~cmd.strike_window)) | in_hold
    gate = attempt_conditioned_phase_gate(
        gate,
        torch.zeros_like(gate),
        cmd.current_swing_targeted_attempt,
        cmd.prev_swing_targeted_attempt,
        require_targeted_attempt,
    ).float()

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
    require_targeted_attempt: bool = False,
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
    gate = attempt_conditioned_phase_gate(
        recovery_or_hold,
        early_next,
        cmd.current_swing_targeted_attempt,
        cmd.prev_swing_targeted_attempt,
        require_targeted_attempt,
    )
    return score * gate.float()


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
    require_targeted_attempt: bool = False,
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
    gate = attempt_conditioned_phase_gate(
        torch.zeros_like(gate),
        gate,
        cmd.current_swing_targeted_attempt,
        cmd.prev_swing_targeted_attempt,
        require_targeted_attempt,
    )
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
    ability_scaled_std: bool = False,
    initial_std: float = 3.0,
    ability_attempt_start: float = 0.0,
    ability_attempt_full: float = 1.0,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward quickly damping pelvis/base angular velocity after a swing."""
    cmd = _cmd(env, command_name)
    gate = _recovery_or_hold_gate(cmd, include_hold, early_prestrike_window_steps).float()
    ang_vel = torch.norm(cmd.robot.data.root_ang_vel_w, dim=-1)
    effective_std = torch.as_tensor(
        max(float(std), 1.0e-6), device=cmd.device
    )
    if ability_scaled_std:
        lo = float(ability_attempt_start)
        hi = max(float(ability_attempt_full), lo + 1.0e-6)
        progress = torch.clamp(
            (cmd._ability_targeted_attempt_ema - lo) / (hi - lo),
            0.0,
            1.0,
        )
        initial = max(float(initial_std), float(std), 1.0e-6)
        effective_std = initial + progress * (float(std) - initial)
    calm = torch.exp(
        -torch.square(ang_vel / effective_std.clamp_min(1.0e-6))
    )
    return calm * gate * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def no_command_ready_balance(
    env: ManagerBasedRLEnv,
    command_name: str,
    pitch_std: float = 0.11,
    roll_std: float = 0.13,
    lin_vel_std: float = 0.18,
    ang_vel_std: float = 0.50,
    height_std: float = 0.09,
    station_std: float = 0.18,
    min_feet_contact: float = 0.50,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward active balance recovery during deploy-style no-command READY."""
    cmd = _cmd(env, command_name)
    data = cmd.robot.data
    gate = cmd.no_command_ready_active.float()

    gravity_xy = data.projected_gravity_b[:, :2]
    pitch = torch.exp(-torch.square(gravity_xy[:, 0] / max(float(pitch_std), 1.0e-6)))
    roll = torch.exp(-torch.square(gravity_xy[:, 1] / max(float(roll_std), 1.0e-6)))
    lin = torch.exp(
        -torch.square(
            torch.norm(data.root_lin_vel_w[:, :2], dim=-1) / max(float(lin_vel_std), 1.0e-6)
        )
    )
    ang = torch.exp(
        -torch.square(
            torch.norm(data.root_ang_vel_b[:, :2], dim=-1) / max(float(ang_vel_std), 1.0e-6)
        )
    )
    default_z = data.default_root_state[:, 2] + env.scene.env_origins[:, 2]
    height = torch.exp(
        -torch.square((data.root_pos_w[:, 2] - default_z) / max(float(height_std), 1.0e-6))
    )
    station_err = torch.norm(cmd.base_pos_w[:, :2] - cmd.fixed_station_w, dim=-1)
    station = torch.exp(-torch.square(station_err / max(float(station_std), 1.0e-6)))
    feet = torch.clamp(
        (cmd.feet_contact_frac - float(min_feet_contact))
        / max(1.0 - float(min_feet_contact), 1.0e-6),
        0.0,
        1.0,
    )

    score = (
        0.26 * pitch
        + 0.14 * roll
        + 0.17 * ang
        + 0.13 * lin
        + 0.12 * height
        + 0.10 * station
        + 0.08 * feet
    )
    return score * gate * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)


def functional_no_command_ready(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Dense multiplicative READY score; one weak component cannot be hidden by others."""
    cmd = _cmd(env, command_name)
    return (
        cmd.metrics["recovery_functional_ready_score"]
        * cmd.no_command_ready_active.float()
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def no_command_joint_anchor(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Keep only selected upper-body joints near a forehand READY anchor."""
    cmd = _cmd(env, command_name)
    return (
        cmd.metrics["recovery_anchor_arm_score"]
        * cmd.no_command_ready_active.float()
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def no_command_ready_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Pay only signed step-to-step progress toward deploy READY.

    The command resets this potential when READY starts or the target is
    resampled. A static READY state therefore earns no recurring payoff, while
    deterioration cancels earlier improvement.
    """
    cmd = _cmd(env, command_name)
    return cmd.metrics["no_command_ready_progress"] * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def active_ready_sustained_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """One-shot bonus after a no-command READY state remains stable long enough."""
    cmd = _cmd(env, command_name)
    return cmd.active_ready_success_event.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def active_ready_survival_milestone_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Pay one-shot rewards for progressively longer safe no-command holds."""
    cmd = _cmd(env, command_name)
    return cmd.active_ready_survival_milestone_event * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def post_contact_directional_recovery(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Signed shaping for reducing backward lean after contact.

    The signal is positive only while the backward error decreases, negative
    when it grows, and zero once the torso has entered the configured READY
    band. It therefore cannot keep paying the policy for leaning farther
    forward after recovery.
    """
    cmd = _cmd(env, command_name)
    return cmd.metrics["post_contact_recovery_directional_progress"] * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def post_contact_ready_region(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Dense multiplicative settlement score while a contact is pending."""
    cmd = _cmd(env, command_name)
    return (
        cmd.metrics["post_contact_ready_score"]
        * cmd.post_contact_ready_pending.float()
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def post_contact_ready_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """One-shot reward after the strict READY region is sustained."""
    cmd = _cmd(env, command_name)
    return cmd.post_contact_ready_success_event.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def post_contact_ready_fail(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """One-shot failure signal when contact never settles before the deadline."""
    cmd = _cmd(env, command_name)
    return cmd.post_contact_ready_fail_event.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def deferred_recovery_outcome_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    tier_multipliers: tuple[float, float, float] = (0.5, 1.0, 1.3),
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Settle contact/net/bounce payoff only after strict post-contact READY."""
    cmd = _cmd(env, command_name)
    multipliers = torch.tensor(
        tier_multipliers,
        dtype=torch.float32,
        device=cmd.device,
    )
    if multipliers.shape != (3,):
        raise ValueError(
            "tier_multipliers must contain contact/net/bounce values, "
            f"got {tier_multipliers}"
        )
    tier_multiplier = outcome_tier_multiplier(
        cmd.post_contact_ready_outcome_tier,
        multipliers,
    )
    return (
        cmd.post_contact_ready_success_event.float()
        * tier_multiplier
        * _phase_scale(env, start_step, warmup_steps, start_scale, 1.0)
    )


def durable_recovery_outcome_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    tier_multipliers: tuple[float, float, float] = (0.5, 1.0, 1.5),
) -> torch.Tensor:
    """Settle contact/net/bounce value only after durable next-cycle READY."""
    cmd = _cmd(env, command_name)
    multipliers = torch.tensor(
        tier_multipliers,
        dtype=torch.float32,
        device=cmd.device,
    )
    if multipliers.shape != (3,):
        raise ValueError(
            "tier_multipliers must contain contact/net/bounce values, "
            f"got {tier_multipliers}"
        )
    return (
        cmd.post_contact_ready_durable_success_event.float()
        * outcome_tier_multiplier(
            cmd.post_contact_ready_outcome_tier,
            multipliers,
        )
    )


def closed_loop_v2_recovered_planner_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_ratio_std: float = 0.30,
    direction_std_rad: float = 0.35,
    component_floor: float = 0.02,
    position_std: float = 0.12,
    position_floor: float = 0.10,
    impact_health_floor: float = 0.25,
    recovery_peak_ang_vel_budget: float = 0.80,
    recovery_peak_ang_vel_excess_std: float = 0.60,
    recovery_gate_floor: float = 0.10,
) -> torch.Tensor:
    """Settle latched planner execution only after a controlled READY return."""
    cmd = _cmd(env, command_name)
    score = recovered_planner_velocity_settlement_score(
        speed_ratio=cmd.post_contact_ready_planner_speed_ratio,
        direction_error_rad=(
            cmd.post_contact_ready_planner_direction_error_rad
        ),
        position_error=cmd.post_contact_ready_planner_position_error,
        impact_health_score=cmd.post_contact_ready_impact_health_score,
        recovery_peak_base_ang_vel=(
            cmd.post_contact_ready_peak_base_ang_vel
        ),
        speed_ratio_std=speed_ratio_std,
        direction_std_rad=direction_std_rad,
        component_floor=component_floor,
        position_std=position_std,
        position_floor=position_floor,
        impact_health_floor=impact_health_floor,
        recovery_peak_ang_vel_budget=recovery_peak_ang_vel_budget,
        recovery_peak_ang_vel_excess_std=recovery_peak_ang_vel_excess_std,
        recovery_gate_floor=recovery_gate_floor,
    )
    return cmd.post_contact_ready_success_event.float() * score


def closed_loop_v2_recovery_peak_ang_vel_excess(
    env: ManagerBasedRLEnv,
    command_name: str,
    impulse: bool = False,
    ability_scaled: bool = False,
    ability_start_scale: float = 1.0,
    ability_attempt_start: float = 0.0,
    ability_attempt_full: float = 1.0,
) -> torch.Tensor:
    """Penalize only the incremental recovery angular-speed peak excess."""
    cmd = _cmd(env, command_name)
    value = cmd.post_contact_ready_peak_ang_vel_excess_increment
    if ability_scaled:
        start = min(max(float(ability_start_scale), 0.0), 1.0)
        lo = float(ability_attempt_start)
        hi = max(float(ability_attempt_full), lo + 1.0e-6)
        progress = torch.clamp(
            (cmd._ability_targeted_attempt_ema - lo) / (hi - lo),
            0.0,
            1.0,
        )
        value = value * (start + (1.0 - start) * progress)
    return _event_manager_value(
        env,
        value,
        impulse=impulse,
    )


def closed_loop_v2_physical_outcome(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_scale: float = 1.0,
    net_cross_scale: float = 2.0,
    opponent_bounce_scale: float = 3.0,
    minimum_health_multiplier: float = 0.25,
    overflow_std: float = 0.40,
    face_quality_power: float = 0.0,
    face_quality_floor: float = 0.0,
) -> torch.Tensor:
    """Settle each physical result once, gated by impact and actuator health."""
    cmd = _cmd(env, command_name)
    overflow = torch.maximum(
        cmd.metrics["waist_action_overflow_rms"],
        torch.maximum(
            cmd.metrics["right_arm_action_overflow_rms"],
            cmd.metrics["leg_action_overflow_rms"],
        ),
    )
    actuator_health = torch.exp(
        -torch.square(overflow / max(float(overflow_std), 1.0e-6))
    )
    combined_health = torch.sqrt(
        cmd.impact_health_score.clamp_min(1.0e-6)
        * actuator_health.clamp_min(1.0e-6)
    )
    multiplier = health_floor_multiplier(
        combined_health, minimum_health_multiplier
    )
    outcome = (
        float(contact_scale) * cmd.ball_contact.float()
        + float(net_cross_scale) * cmd.ball_net_cross.float()
        + float(opponent_bounce_scale) * cmd.ball_on_opponent.float()
    )
    face_quality = _face_quality_multiplier(
        cmd.impact_face_quality,
        power=face_quality_power,
        floor=face_quality_floor,
    )
    return outcome * multiplier * face_quality


def closed_loop_v2_cycle_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """One-shot reward for net-cross, healthy impact, and resolved recovery."""
    return _cmd(env, command_name).metrics[
        "closed_loop_cycle_success_event"
    ]


def closed_loop_v2_durable_cycle_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """One-shot healthy net-cross credit after durable recovery."""
    return _cmd(env, command_name).metrics[
        "closed_loop_durable_cycle_success_event"
    ]


def closed_loop_v2_recovery_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Bounded directional progress while an attempted swing is recovering."""
    cmd = _cmd(env, command_name)
    return (
        cmd.metrics["post_contact_recovery_directional_progress"]
        * cmd.post_contact_ready_pending.float()
    )


def closed_loop_v2_durable_recovery_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Keep bounded directional recovery shaping active until durable settlement."""
    cmd = _cmd(env, command_name)
    return (
        cmd.metrics["post_contact_recovery_directional_progress"]
        * cmd.post_contact_ready_durable_pending.float()
    )


def closed_loop_v2_recovery_success(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    return _cmd(env, command_name).post_contact_ready_success_event.float()


def closed_loop_v2_recovery_failure(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    return _cmd(env, command_name).post_contact_ready_fail_event.float()


def closed_loop_v2_durable_recovery_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    impulse: bool = False,
) -> torch.Tensor:
    value = (
        _cmd(env, command_name)
        .post_contact_ready_durable_success_event.float()
    )
    return value / float(env.step_dt) if impulse else value


def closed_loop_v2_durable_recovery_failure(
    env: ManagerBasedRLEnv,
    command_name: str,
    impulse: bool = False,
) -> torch.Tensor:
    value = (
        _cmd(env, command_name)
        .post_contact_ready_durable_fail_event.float()
    )
    return value / float(env.step_dt) if impulse else value


def closed_loop_v2_terminal_quality_window(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Bound terminal READY shaping to one fixed-duration recovery window."""
    cmd = _cmd(env, command_name)
    window_steps = max(
        int(cmd.cfg.post_contact_ready_durable_required_consecutive_steps),
        1,
    )
    return (
        cmd.post_contact_ready_terminal_window_active.float()
        * cmd.metrics["post_contact_ready_score"].clamp(0.0, 1.0)
        / float(window_steps)
    )


def closed_loop_v2_safe_terminal_quality(
    env: ManagerBasedRLEnv,
    command_name: str,
    impulse: bool = False,
) -> torch.Tensor:
    """Settle graded recovery quality once when the full interval was safe."""
    cmd = _cmd(env, command_name)
    value = (
        cmd.post_contact_ready_safe_settlement_event.float()
        * cmd.post_contact_ready_terminal_quality.clamp(0.0, 1.0)
    )
    return _event_manager_value(env, value, impulse=impulse)


def stage1_safe_recovered_planner_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    speed_ratio_std: float = 0.30,
    direction_std_rad: float = 0.35,
    normal_std_rad: float = 0.30,
    component_floor: float = 0.02,
    normal_floor: float = 0.05,
    position_std: float = 0.12,
    position_floor: float = 0.10,
    impact_health_floor: float = 0.25,
    recovery_peak_ang_vel_budget: float = 0.80,
    recovery_peak_ang_vel_excess_std: float = 0.60,
    recovery_gate_floor: float = 0.10,
    require_contact: bool = True,
    impulse: bool = False,
) -> torch.Tensor:
    """Settle one complete planner command only after safe reusable recovery."""
    cmd = _cmd(env, command_name)
    command_score = recovered_planner_command_settlement_score(
        speed_ratio=cmd.post_contact_ready_planner_speed_ratio,
        direction_error_rad=(
            cmd.post_contact_ready_planner_direction_error_rad
        ),
        normal_error_rad=(
            cmd.post_contact_ready_planner_normal_error_rad
        ),
        position_error=cmd.post_contact_ready_planner_position_error,
        impact_health_score=cmd.post_contact_ready_impact_health_score,
        recovery_peak_base_ang_vel=(
            cmd.post_contact_ready_peak_base_ang_vel
        ),
        speed_ratio_std=speed_ratio_std,
        direction_std_rad=direction_std_rad,
        normal_std_rad=normal_std_rad,
        component_floor=component_floor,
        normal_floor=normal_floor,
        position_std=position_std,
        position_floor=position_floor,
        impact_health_floor=impact_health_floor,
        recovery_peak_ang_vel_budget=recovery_peak_ang_vel_budget,
        recovery_peak_ang_vel_excess_std=(
            recovery_peak_ang_vel_excess_std
        ),
        recovery_gate_floor=recovery_gate_floor,
    )
    value = safe_command_cycle_value(
        safe_settlement_event=(
            cmd.post_contact_ready_safe_settlement_event
        ),
        terminal_quality=cmd.post_contact_ready_terminal_quality,
        command_score=command_score,
        outcome_tier=cmd.post_contact_ready_outcome_tier,
        require_contact=require_contact,
    )
    return _event_manager_value(env, value, impulse=impulse)


def stage1_command_cycle_failure(
    env: ManagerBasedRLEnv,
    command_name: str,
    unsafe_scale: float = 2.0,
    incomplete_scale: float = 0.50,
    impulse: bool = False,
) -> torch.Tensor:
    """One-shot debit for unsafe or non-reusable command execution."""
    cmd = _cmd(env, command_name)
    value = (
        float(unsafe_scale)
        * cmd.post_contact_ready_unsafe_settlement_event.float()
        + float(incomplete_scale)
        * cmd.post_contact_ready_incomplete_settlement_event.float()
    )
    return _event_manager_value(env, value, impulse=impulse)


def closed_loop_v2_unsafe_terminal_recovery(
    env: ManagerBasedRLEnv,
    command_name: str,
    impulse: bool = False,
) -> torch.Tensor:
    """One-shot catastrophic recovery-envelope event."""
    value = (
        _cmd(env, command_name)
        .post_contact_ready_unsafe_settlement_event.float()
    )
    return _event_manager_value(env, value, impulse=impulse)


def closed_loop_v2_safe_terminal_outcome(
    env: ManagerBasedRLEnv,
    command_name: str,
    tier_multipliers: tuple[float, float, float] = (0.5, 1.0, 1.5),
    face_quality_power: float = 0.0,
    face_quality_floor: float = 0.0,
    impulse: bool = False,
) -> torch.Tensor:
    """Defer graded contact/net/bounce value until safe terminal recovery."""
    cmd = _cmd(env, command_name)
    multipliers = torch.tensor(
        tier_multipliers,
        dtype=torch.float32,
        device=cmd.device,
    )
    if multipliers.shape != (3,):
        raise ValueError(
            "tier_multipliers must contain contact/net/bounce values, "
            f"got {tier_multipliers}"
        )
    value = safety_conditioned_outcome_value(
        safe_settlement_event=(
            cmd.post_contact_ready_safe_settlement_event
        ),
        terminal_quality=cmd.post_contact_ready_terminal_quality,
        outcome_tier=cmd.post_contact_ready_outcome_tier,
        tier_multipliers=multipliers,
    )
    value = value * _face_quality_multiplier(
        cmd.post_contact_ready_face_quality,
        power=face_quality_power,
        floor=face_quality_floor,
    )
    return _event_manager_value(env, value, impulse=impulse)


def capability_gated_safe_terminal_outcome(
    env: ManagerBasedRLEnv,
    command_name: str,
    contact_value: float = 0.80,
    net_extra_value: float = 0.30,
    bounce_extra_value: float = 0.40,
    face_quality_power: float = 0.0,
    face_quality_floor: float = 0.0,
    impulse: bool = False,
) -> torch.Tensor:
    """Pay safe contact always and unlock higher tiers from retained ability."""
    cmd = _cmd(env, command_name)
    tier = cmd.post_contact_ready_outcome_tier
    capability = cmd.safe_outcome_capability_gate()
    multiplier = torch.full_like(
        cmd.post_contact_ready_terminal_quality, float(contact_value)
    )
    multiplier += capability * float(net_extra_value) * (tier >= 1).float()
    multiplier += capability * float(bounce_extra_value) * (tier >= 2).float()
    value = (
        cmd.post_contact_ready_safe_settlement_event.float()
        * cmd.post_contact_ready_terminal_quality.clamp(0.0, 1.0)
        * multiplier
    )
    value = value * _face_quality_multiplier(
        cmd.post_contact_ready_face_quality,
        power=face_quality_power,
        floor=face_quality_floor,
    )
    return _event_manager_value(env, value, impulse=impulse)


def closed_loop_v2_safe_terminal_cycle(
    env: ManagerBasedRLEnv,
    command_name: str,
    face_quality_power: float = 0.0,
    face_quality_floor: float = 0.0,
) -> torch.Tensor:
    """Pay full-cycle value only for safe net-cross recovery."""
    cmd = _cmd(env, command_name)
    value = safety_conditioned_cycle_value(
        safe_settlement_event=(
            cmd.post_contact_ready_safe_settlement_event
        ),
        terminal_quality=cmd.post_contact_ready_terminal_quality,
        outcome_tier=cmd.post_contact_ready_outcome_tier,
    )
    return value * _face_quality_multiplier(
        cmd.post_contact_ready_face_quality,
        power=face_quality_power,
        floor=face_quality_floor,
    )


def closed_loop_v2_no_command_instability(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """A bounded cost, rather than an accumulating positive READY payoff."""
    cmd = _cmd(env, command_name)
    instability = 1.0 - cmd.metrics[
        "recovery_functional_ready_score"
    ].clamp(0.0, 1.0)
    return instability * cmd.no_command_ready_active.float()


def closed_loop_v2_safe_inactivity(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimum_health: float = 0.50,
    maximum_target_distance_from_base: float = 1.40,
    impulse: bool = False,
) -> torch.Tensor:
    """Penalize skipping only a healthy, visible, geometrically feasible shot."""
    cmd = _cmd(env, command_name)
    target_distance = torch.norm(
        cmd.ball_strike_pos_w - cmd.base_pos_w, dim=-1
    )
    eligible = (
        cmd.strike_fired
        & (~cmd.no_command_ready_active)
        & (cmd.impact_health_score >= float(minimum_health))
        & (target_distance <= float(maximum_target_distance_from_base))
    )
    value = (eligible & (~cmd.current_swing_targeted_attempt)).float()
    return value / float(env.step_dt) if impulse else value


def phase_action_overflow(
    env: ManagerBasedRLEnv,
    command_name: str,
    waist_scale: float = 0.35,
    right_arm_scale: float = 0.15,
    leg_scale: float = 1.0,
    pre_strike_scale: float = 0.35,
    strike_scale: float = 0.10,
    recovery_scale: float = 1.0,
    hold_scale: float = 1.0,
    free_margin: float = 0.05,
    delta: float = 0.25,
    maximum: float = 4.0,
) -> torch.Tensor:
    """Penalize only infeasible raw actions, preserving feasible leg compensation."""
    cmd = _cmd(env, command_name)
    overflow = (
        float(waist_scale)
        * smooth_overflow_penalty(
            cmd.metrics["waist_action_overflow_rms"], free_margin, delta, maximum
        )
        + float(right_arm_scale)
        * smooth_overflow_penalty(
            cmd.metrics["right_arm_action_overflow_rms"], free_margin, delta, maximum
        )
        + float(leg_scale)
        * smooth_overflow_penalty(
            cmd.metrics["leg_action_overflow_rms"], free_margin, delta, maximum
        )
    )
    phase = torch.full_like(overflow, float(recovery_scale))
    phase = torch.where(
        cmd.pre_strike, torch.full_like(phase, float(pre_strike_scale)), phase
    )
    phase = torch.where(
        cmd.strike_window, torch.full_like(phase, float(strike_scale)), phase
    )
    phase = torch.where(
        cmd._motion().in_hold, torch.full_like(phase, float(hold_scale)), phase
    )
    return overflow * phase


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


def one_sided_trunk_waist_stability(
    env: ManagerBasedRLEnv,
    command_name: str,
    phase: str = "strike",
    pre_window_s: float = 0.34,
    post_window_s: float = 0.18,
    torso_backlean_tolerance: float = 0.06,
    torso_backlean_std: float = 0.12,
    waist_backfold_tolerance: float = 0.14,
    waist_backfold_std: float = 0.14,
    torso_pitch_rate_std: float = 1.40,
    waist_pitch_rate_std: float = 1.20,
    rate_weight: float = 0.0,
    start_step: int = 0,
    warmup_steps: int = 0,
    start_scale: float = 1.0,
) -> torch.Tensor:
    """Reward non-backward trunk/waist pose without constraining arms or legs."""
    cmd = _cmd(env, command_name)
    data = cmd.robot.data
    torso_idx = cmd._impact_health_torso_index
    gravity_w = torch.zeros(cmd.num_envs, 3, device=cmd.device)
    gravity_w[:, 2] = -1.0
    torso_quat_w = data.body_quat_w[:, torso_idx]
    torso_gravity = quat_rotate_inverse(torso_quat_w, gravity_w)

    waist_idx = cmd._impact_health_waist_pitch_index
    waist_pitch = data.joint_pos[:, waist_idx]
    waist_default = data.default_joint_pos[:, waist_idx]

    torso_ang_vel_b = quat_rotate_inverse(
        torso_quat_w, data.body_ang_vel_w[:, torso_idx]
    )
    score = one_sided_trunk_waist_score(
        torso_gravity[:, 0],
        waist_pitch,
        waist_default,
        torso_backlean_tolerance=torso_backlean_tolerance,
        torso_backlean_std=torso_backlean_std,
        waist_backfold_tolerance=waist_backfold_tolerance,
        waist_backfold_std=waist_backfold_std,
        torso_pitch_rate=torch.abs(torso_ang_vel_b[:, 1]),
        waist_pitch_rate=torch.abs(data.joint_vel[:, waist_idx]),
        torso_pitch_rate_std=torso_pitch_rate_std,
        waist_pitch_rate_std=waist_pitch_rate_std,
        rate_weight=rate_weight,
    )

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
        raise ValueError(
            f"unsupported one_sided_trunk_waist_stability phase {phase!r}"
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
    include_no_command_ready: bool = False,
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
    phase = cmd.pre_strike & (cmd.true_time_to_strike > float(stop_window_s))
    if include_no_command_ready:
        phase |= cmd.no_command_ready_active
    else:
        phase &= ~cmd.no_command_ready_active
    gate = phase & (distance > float(arrival_radius))
    return progress * gate.float() * _phase_scale(
        env, start_step, warmup_steps, start_scale, 1.0
    )


def station_relocation_arrival_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """One-shot reward when a relocation rehearsal reaches its pelvis target."""
    return _cmd(env, command_name).station_relocation_arrival_event.float()


def station_relocation_settle_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """One-shot reward after arrival remains upright, supported, and slow."""
    return _cmd(env, command_name).station_relocation_settle_event.float()


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
