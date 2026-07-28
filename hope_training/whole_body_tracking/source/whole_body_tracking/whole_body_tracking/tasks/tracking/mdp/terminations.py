"""Termination helpers for physical falls and reference tracking failures."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.hope_commands import RacketTargetCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _past_grace_steps(env: ManagerBasedRLEnv, command: MotionCommand, min_steps: int) -> torch.Tensor:
    steps = getattr(command, "steps_since_resample", env.episode_length_buf)
    return steps >= int(min_steps)


def _past_episode_steps(env: ManagerBasedRLEnv, min_steps: int) -> torch.Tensor:
    return env.episode_length_buf >= int(min_steps)


def base_tilted(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_steps: int = 0,
) -> torch.Tensor:
    """True when the base is tilted past ``threshold`` (horizontal component of projected gravity)."""
    asset: Articulation = env.scene[asset_cfg.name]
    bad = torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1) > threshold
    return bad & _past_episode_steps(env, min_steps)


def base_too_low(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_steps: int = 0,
) -> torch.Tensor:
    """True when the base root height drops below ``min_height`` (the robot has fallen/collapsed)."""
    asset: Articulation = env.scene[asset_cfg.name]
    bad = asset.data.root_pos_w[:, 2] < min_height
    return bad & _past_episode_steps(env, min_steps)


def bad_anchor_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, min_steps: int = 0
) -> torch.Tensor:
    """True when the reference anchor and robot anchor differ too much in height."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    bad = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    return bad & _past_grace_steps(env, command, min_steps) & (~command.in_hold)


def bad_anchor_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
    min_steps: int = 0,
) -> torch.Tensor:
    """True when the robot anchor tilt diverges too far from the reference anchor tilt."""
    asset: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = math_utils.quat_rotate_inverse(
        command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W
    )
    bad = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
    return bad & _past_grace_steps(env, command, min_steps) & (~command.in_hold)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
    min_steps: int = 0,
) -> torch.Tensor:
    """True when selected tracked bodies drift too far from the reference in vertical position."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, idx, -1] - command.robot_body_pos_w[:, idx, -1])
    bad = torch.any(error > threshold, dim=-1)
    return bad & _past_grace_steps(env, command, min_steps) & (~command.in_hold)


def _racket_table_points(command: RacketTargetCommand, body_names: list[str] | None, include_racket: bool):
    points = []
    motion = command._motion()
    if body_names:
        ids = [motion.cfg.body_names.index(name) for name in body_names if name in motion.cfg.body_names]
        if ids:
            idx = torch.tensor(ids, dtype=torch.long, device=command.device)
            points.append(motion.robot_body_pos_w[:, idx])
    if include_racket:
        points.append(command.racket_pos_w.unsqueeze(1))
    if not points:
        return None
    return torch.cat(points, dim=1)


def body_inside_table_zone(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_names: list[str] | None = None,
    include_racket: bool = True,
    x_margin: float = 0.04,
    y_margin: float = 0.04,
    below_surface_margin: float = 0.08,
    above_surface_margin: float = 0.04,
    min_steps: int = 0,
    enabled: bool = False,
) -> torch.Tensor:
    """True when monitored points enter the analytic table-top no-touch volume."""
    command: RacketTargetCommand = env.command_manager.get_term(command_name)
    if not enabled:
        return torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
    points_w = _racket_table_points(command, body_names, include_racket)
    if points_w is None:
        return torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)

    origins = env.scene.env_origins
    points_l = points_w - origins.unsqueeze(1)
    x = points_l[..., 0]
    y = points_l[..., 1]
    z = points_l[..., 2]

    center_y = command.fixed_station_w[:, 1] - origins[:, 1]
    half_width = 0.5 * float(command.cfg.table_width)
    x_min = x.new_tensor(float(command.cfg.table_near_x) - float(x_margin))
    x_max = x.new_tensor(float(command.cfg.table_near_x) + float(command.cfg.table_length) + float(x_margin))
    y_min = center_y.unsqueeze(1) - half_width - float(y_margin)
    y_max = center_y.unsqueeze(1) + half_width + float(y_margin)
    z_min = z.new_tensor(float(command.cfg.table_surface_z) - float(below_surface_margin))
    z_max = z.new_tensor(float(command.cfg.table_surface_z) + float(above_surface_margin))

    inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    return inside.any(dim=1) & _past_episode_steps(env, min_steps)


def cycle_v2_ready_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
    enabled: bool = False,
    start_step: int = 0,
) -> torch.Tensor:
    """Terminate when cycle-v2 marks a useful previous return as not ready for the next swing."""
    command: RacketTargetCommand = env.command_manager.get_term(command_name)
    if not enabled or int(getattr(env, "common_step_counter", 0)) < int(start_step):
        return torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
    return command.cycle_v2_ready_fail_latch
