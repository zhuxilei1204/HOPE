"""Actuator feasibility terms derived from the A3 output-side specification."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from whole_body_tracking.utils.actuator_envelope import (
    actuator_envelope_components,
)
from whole_body_tracking.utils.operational_action import (
    bounded_squared_excess,
    grouped_joint_cost,
    robust_log_excess,
)


def _phase_scale(
    command,
    value: torch.Tensor,
    *,
    pre_strike_scale: float,
    strike_scale: float,
    recovery_scale: float,
    hold_scale: float,
) -> torch.Tensor:
    phase = torch.full_like(value, float(recovery_scale))
    phase = torch.where(
        command.pre_strike,
        torch.full_like(phase, float(pre_strike_scale)),
        phase,
    )
    phase = torch.where(
        command.strike_window,
        torch.full_like(phase, float(strike_scale)),
        phase,
    )
    phase = torch.where(
        command._motion().in_hold,
        torch.full_like(phase, float(hold_scale)),
        phase,
    )
    return phase


def phase_operational_joint_margin(
    env,
    command_name: str,
    action_name: str = "joint_pos",
    waist_scale: float = 1.0,
    other_upper_scale: float = 0.20,
    right_arm_scale: float = 0.15,
    leg_scale: float = 1.0,
    pre_strike_scale: float = 0.80,
    strike_scale: float = 0.70,
    recovery_scale: float = 1.0,
    hold_scale: float = 1.20,
    excess_scale: float = 0.03,
    maximum: float = 4.0,
) -> torch.Tensor:
    """Penalize raw q-des outside the soft working range before hard clipping."""
    command = env.command_manager.get_term(command_name)
    action_term = env.action_manager.get_term(action_name)
    cost = grouped_joint_cost(
        bounded_squared_excess(
            action_term.operational_excess,
            scale=excess_scale,
            maximum=maximum,
        ),
        waist_scale=waist_scale,
        other_upper_scale=other_upper_scale,
        right_arm_scale=right_arm_scale,
        leg_scale=leg_scale,
    )
    return cost * _phase_scale(
        command,
        cost,
        pre_strike_scale=pre_strike_scale,
        strike_scale=strike_scale,
        recovery_scale=recovery_scale,
        hold_scale=hold_scale,
    )


def phase_joint_target_slew(
    env,
    command_name: str,
    action_name: str = "joint_pos",
    waist_scale: float = 1.0,
    other_upper_scale: float = 0.20,
    right_arm_scale: float = 0.12,
    leg_scale: float = 0.85,
    pre_strike_scale: float = 0.45,
    strike_scale: float = 0.12,
    recovery_scale: float = 1.0,
    hold_scale: float = 1.20,
    velocity_excess_scale: float = 0.30,
    acceleration_excess_scale: float = 0.50,
    acceleration_weight: float = 0.25,
    acceleration_cost_mode: str = "bounded_squared",
    maximum: float = 4.0,
) -> torch.Tensor:
    """Penalize target slew beyond the A3 rated-speed-derived command envelope."""
    command = env.command_manager.get_term(command_name)
    action_term = env.action_manager.get_term(action_name)
    velocity_cost = bounded_squared_excess(
        action_term.q_des_velocity_excess_ratio,
        scale=velocity_excess_scale,
        maximum=maximum,
    )
    if acceleration_cost_mode == "bounded_squared":
        acceleration_cost = bounded_squared_excess(
            action_term.q_des_acceleration_excess_ratio,
            scale=acceleration_excess_scale,
            maximum=maximum,
        )
    elif acceleration_cost_mode == "robust_log":
        acceleration_cost = robust_log_excess(
            action_term.q_des_acceleration_excess_ratio,
            scale=acceleration_excess_scale,
            maximum=maximum,
        )
    else:
        raise ValueError(
            "acceleration_cost_mode must be 'bounded_squared' or "
            f"'robust_log', got {acceleration_cost_mode!r}"
        )
    cost = grouped_joint_cost(
        velocity_cost + float(acceleration_weight) * acceleration_cost,
        waist_scale=waist_scale,
        other_upper_scale=other_upper_scale,
        right_arm_scale=right_arm_scale,
        leg_scale=leg_scale,
    )
    return cost * _phase_scale(
        command,
        cost,
        pre_strike_scale=pre_strike_scale,
        strike_scale=strike_scale,
        recovery_scale=recovery_scale,
        hold_scale=hold_scale,
    )


class phase_actuator_feasibility(ManagerTermBase):
    """Phase-aware soft penalty for sustained and physically impossible loads."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        self._asset = env.scene[params["asset_cfg"].name]
        joint_ids = params["asset_cfg"].joint_ids
        if isinstance(joint_ids, slice):
            joint_ids = list(range(self._asset.num_joints))[joint_ids]
        self._joint_ids = torch.as_tensor(
            joint_ids, dtype=torch.long, device=env.device
        )
        count = int(self._joint_ids.numel())
        self._rated_torque = self._as_row(params["rated_torque"], count)
        self._peak_torque = self._as_row(params["peak_torque"], count)
        self._rated_speed = self._as_row(params["rated_speed"], count)
        self._peak_speed = self._as_row(params["peak_speed"], count)

    def _as_row(self, values, count: int) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if tensor.numel() != count:
            raise ValueError(
                f"Actuator envelope has {tensor.numel()} values for {count} joints"
            )
        return tensor.reshape(1, count)

    def __call__(
        self,
        env,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        rated_torque,
        peak_torque,
        rated_speed,
        peak_speed,
        pre_strike_scale: float = 0.25,
        strike_scale: float = 0.05,
        recovery_scale: float = 1.0,
        hold_scale: float = 1.0,
        corner_weight: float = 1.0,
        peak_weight: float = 4.0,
        free_margin: float = 0.05,
        delta: float = 0.25,
        maximum: float = 4.0,
    ) -> torch.Tensor:
        del asset_cfg, rated_torque, peak_torque, rated_speed, peak_speed
        command = env.command_manager.get_term(command_name)
        data = self._asset.data
        requested_torque = data.computed_torque.index_select(-1, self._joint_ids)
        joint_velocity = data.joint_vel.index_select(-1, self._joint_ids)
        rated, corner, peak = actuator_envelope_components(
            requested_torque,
            joint_velocity,
            self._rated_torque,
            self._peak_torque,
            self._rated_speed,
            self._peak_speed,
            free_margin=free_margin,
            delta=delta,
            maximum=maximum,
        )
        # Mean over joints keeps weights comparable across body groups.
        rated = rated.mean(dim=-1)
        corner = corner.mean(dim=-1)
        peak = peak.mean(dim=-1)

        phase = torch.full_like(rated, float(recovery_scale))
        phase = torch.where(
            command.pre_strike,
            torch.full_like(phase, float(pre_strike_scale)),
            phase,
        )
        phase = torch.where(
            command.strike_window,
            torch.full_like(phase, float(strike_scale)),
            phase,
        )
        phase = torch.where(
            command._motion().in_hold,
            torch.full_like(phase, float(hold_scale)),
            phase,
        )
        return rated * phase + float(corner_weight) * corner + float(peak_weight) * peak
