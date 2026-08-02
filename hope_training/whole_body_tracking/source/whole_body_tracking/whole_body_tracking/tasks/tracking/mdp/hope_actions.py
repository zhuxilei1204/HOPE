"""The HOPE action term.

:class:`ClampedJointPositionAction` is a 31-D joint-position residual action:

    q_des = default_q + raw_action * action_scale

followed by a deterministic clamp of the position targets (a numeric transform, not a rejection
gate — it emits no failure status). When ``position_clamp`` is configured (the HOPE task
sets it from the shared deploy ``action_adapter.yaml``) the clamp bounds are those exact per-joint
values, so training and deployment clamp identically; without it the term falls back to the
articulation's soft joint limits. The policy still emits 31 raw values, but the two passive head
joints (idx 3, 4) are held at their articulation default and their applied-action feedback is
zeroed, matching a deploy runner that holds the neck passive. ``applied_raw_actions`` (the exact
31 values that were applied, with the passive columns zeroed) is fed back as the next tick's
``last_action`` observation.

The residual form, ``action_scale``, ``default_q`` offset, and clamp are the shared public action
adapter: scale/default/clamp all come from ONE file
(``a3_deploy/a3_deploy_example/config/action_adapter.yaml``) and the reference runner
applies the identical transform, so training and deployment agree
(see ``tests/test_action_adapter_parity.py``).
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


def resolve_action_columns_by_joint_name(action_term, joint_names) -> list[int]:
    """Resolve exact joint names to action columns, raising on any contract mismatch."""
    action_joint_names = getattr(action_term, "_joint_names", None)
    if action_joint_names is None:
        raise RuntimeError("Passive-joint action contract requires the action term's resolved joint names")
    action_joint_names = list(action_joint_names)
    requested = list(joint_names)
    if not requested:
        return []
    missing = [name for name in requested if name not in action_joint_names]
    if missing:
        raise RuntimeError(
            f"Passive joints missing from the configured action term: {missing}; "
            f"resolved action joints are {action_joint_names}"
        )
    return [action_joint_names.index(name) for name in requested]


def _action_joint_ids_in_column_order(action_term) -> list[int]:
    """Return articulation joint ids in action-column order (order-assumption free)."""
    joint_ids = getattr(action_term, "_joint_ids", None)
    if joint_ids is None:
        raise RuntimeError("Passive-joint action contract cannot resolve articulation joint ids")
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(action_term._asset.joint_names)))[joint_ids]
    elif torch.is_tensor(joint_ids):
        joint_ids = joint_ids.detach().cpu().tolist()
    else:
        joint_ids = list(joint_ids)
    return [int(j) for j in joint_ids]


class ClampedJointPositionAction(JointPositionAction):
    """Joint-position residual action with a deterministic limit clamp and optional passive joints."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.cfg.feedback_mode not in ("raw", "effective"):
            raise ValueError(
                f"feedback_mode must be 'raw' or 'effective', got {self.cfg.feedback_mode!r}"
            )
        # Keep both representations even when only one is exposed to the actor.
        # Their difference is the part of the policy output erased by q_des clamping.
        self._applied_raw_actions = torch.zeros_like(self._raw_actions)
        self._effective_raw_actions = torch.zeros_like(self._raw_actions)
        self._feedback_actions = torch.zeros_like(self._raw_actions)
        self._unclamped_processed_actions = torch.zeros_like(self._processed_actions)
        self._q_des_velocity = torch.zeros_like(self._processed_actions)
        self._q_des_acceleration = torch.zeros_like(self._processed_actions)
        self._previous_q_des_velocity = torch.zeros_like(self._processed_actions)
        self._previous_processed_actions = self._asset.data.default_joint_pos[
            :, _action_joint_ids_in_column_order(self)
        ].clone()
        self._step_dt = float(env.step_dt)
        if self._step_dt <= 0.0:
            raise RuntimeError(f"Environment step_dt must be positive, got {self._step_dt}")

        # Explicit per-joint clamp from the shared action-adapter config (exact joint names).
        # When absent, process_actions falls back to the articulation soft joint limits.
        self._clamp_lower: torch.Tensor | None = None
        self._clamp_upper: torch.Tensor | None = None
        clamp_cfg = self.cfg.position_clamp
        if clamp_cfg is not None:
            action_joint_names = list(self._joint_names)
            missing = [n for n in action_joint_names if n not in clamp_cfg]
            if missing:
                raise RuntimeError(
                    f"position_clamp is missing joints resolved by the action term: {missing}"
                )
            lo = torch.tensor(
                [float(clamp_cfg[n][0]) for n in action_joint_names],
                dtype=torch.float32, device=self.device,
            )
            hi = torch.tensor(
                [float(clamp_cfg[n][1]) for n in action_joint_names],
                dtype=torch.float32, device=self.device,
            )
            if torch.any(lo > hi):
                raise RuntimeError("position_clamp lower must be <= upper for every joint")
            self._clamp_lower = lo.unsqueeze(0)
            self._clamp_upper = hi.unsqueeze(0)

        self._operational_lower: torch.Tensor | None = None
        self._operational_upper: torch.Tensor | None = None
        margin_cfg = self.cfg.operational_margin_fraction
        if margin_cfg is not None:
            if self._clamp_lower is None or self._clamp_upper is None:
                raise RuntimeError(
                    "operational_margin_fraction requires an explicit position_clamp"
                )
            action_joint_names = list(self._joint_names)
            missing = [name for name in action_joint_names if name not in margin_cfg]
            if missing:
                raise RuntimeError(
                    "operational_margin_fraction is missing joints resolved by the "
                    f"action term: {missing}"
                )
            margin = torch.tensor(
                [float(margin_cfg[name]) for name in action_joint_names],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            if torch.any((margin < 0.0) | (margin >= 0.5)):
                raise RuntimeError(
                    "operational margin fractions must lie in [0, 0.5)"
                )
            span = self._clamp_upper - self._clamp_lower
            self._operational_lower = self._clamp_lower + margin * span
            self._operational_upper = self._clamp_upper - margin * span
            default_q = self._asset.data.default_joint_pos[
                :, _action_joint_ids_in_column_order(self)
            ]
            if torch.any(
                (default_q < self._operational_lower)
                | (default_q > self._operational_upper)
            ):
                raise RuntimeError(
                    "default_q must lie inside every configured operational range"
                )

        self._q_des_velocity_limit = self._resolve_positive_joint_map(
            self.cfg.q_des_velocity_limit,
            "q_des_velocity_limit",
        )
        self._q_des_acceleration_limit = self._resolve_positive_joint_map(
            self.cfg.q_des_acceleration_limit,
            "q_des_acceleration_limit",
        )

        passive_names = tuple(self.cfg.passive_joint_names)
        passive_cols = resolve_action_columns_by_joint_name(self, passive_names)
        action_joint_ids = _action_joint_ids_in_column_order(self)
        self._passive_joint_names = passive_names
        self._passive_action_cols = torch.tensor(passive_cols, dtype=torch.long, device=self.device)
        self._passive_joint_ids = torch.tensor(
            [action_joint_ids[c] for c in passive_cols], dtype=torch.long, device=self.device
        )

    def _resolve_positive_joint_map(
        self,
        values: dict[str, float] | None,
        field: str,
    ) -> torch.Tensor | None:
        if values is None:
            return None
        action_joint_names = list(self._joint_names)
        missing = [name for name in action_joint_names if name not in values]
        if missing:
            raise RuntimeError(f"{field} is missing joints: {missing}")
        tensor = torch.tensor(
            [float(values[name]) for name in action_joint_names],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        if torch.any(tensor <= 0.0):
            raise RuntimeError(f"{field} must be positive for every joint")
        return tensor

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._applied_raw_actions.copy_(self._raw_actions)
        self._unclamped_processed_actions.copy_(self._processed_actions)
        if self._clamp_lower is not None:
            # Shared-adapter clamp: identical bounds to the deploy ActionAdapter.
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clamp_lower, max=self._clamp_upper
            )
        else:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
            self._processed_actions = torch.clamp(
                self._processed_actions, min=limits[..., 0], max=limits[..., 1]
            )
        if self._passive_action_cols.numel() > 0:
            default_q = self._asset.data.default_joint_pos.index_select(-1, self._passive_joint_ids)
            self._processed_actions.index_copy_(-1, self._passive_action_cols, default_q)
            self._unclamped_processed_actions.index_copy_(
                -1, self._passive_action_cols, default_q
            )
            self._applied_raw_actions.index_fill_(-1, self._passive_action_cols, 0.0)
        scale = torch.as_tensor(self._scale, device=self.device, dtype=self._processed_actions.dtype)
        offset = torch.as_tensor(self._offset, device=self.device, dtype=self._processed_actions.dtype)
        if torch.any(scale == 0.0):
            raise RuntimeError("Action scale must be nonzero to compute effective raw actions")
        self._effective_raw_actions.copy_((self._processed_actions - offset) / scale)
        if self._passive_action_cols.numel() > 0:
            self._effective_raw_actions.index_fill_(-1, self._passive_action_cols, 0.0)
        if self.cfg.feedback_mode == "effective":
            self._feedback_actions.copy_(self._effective_raw_actions)
        else:
            self._feedback_actions.copy_(self._applied_raw_actions)
        self._q_des_velocity.copy_(
            (self._processed_actions - self._previous_processed_actions)
            / self._step_dt
        )
        self._q_des_acceleration.copy_(
            (self._q_des_velocity - self._previous_q_des_velocity)
            / self._step_dt
        )
        self._previous_processed_actions.copy_(self._processed_actions)
        self._previous_q_des_velocity.copy_(self._q_des_velocity)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._applied_raw_actions.zero_()
            self._effective_raw_actions.zero_()
            self._feedback_actions.zero_()
            self._unclamped_processed_actions.zero_()
            self._q_des_velocity.zero_()
            self._q_des_acceleration.zero_()
            self._previous_q_des_velocity.zero_()
            self._previous_processed_actions.copy_(
                self._asset.data.default_joint_pos[
                    :, _action_joint_ids_in_column_order(self)
                ]
            )
        else:
            self._applied_raw_actions[env_ids] = 0.0
            self._effective_raw_actions[env_ids] = 0.0
            self._feedback_actions[env_ids] = 0.0
            self._unclamped_processed_actions[env_ids] = 0.0
            self._q_des_velocity[env_ids] = 0.0
            self._q_des_acceleration[env_ids] = 0.0
            self._previous_q_des_velocity[env_ids] = 0.0
            action_joint_ids = _action_joint_ids_in_column_order(self)
            self._previous_processed_actions[env_ids] = (
                self._asset.data.default_joint_pos[
                    env_ids
                ][:, action_joint_ids]
            )

    @property
    def applied_raw_actions(self) -> torch.Tensor:
        """Raw-domain action actually applied, with the passive columns set to zero."""
        return self._applied_raw_actions

    @property
    def effective_raw_actions(self) -> torch.Tensor:
        """Residual action represented by the clamped joint-position target."""
        return self._effective_raw_actions

    @property
    def feedback_actions(self) -> torch.Tensor:
        """Action representation selected for the next actor observation."""
        return self._feedback_actions

    @property
    def overflow_actions(self) -> torch.Tensor:
        """Raw residual erased by the joint-position clamp."""
        return self._applied_raw_actions - self._effective_raw_actions

    @property
    def unclamped_processed_actions(self) -> torch.Tensor:
        """Joint targets before the shared mechanical-position clamp."""
        return self._unclamped_processed_actions

    @property
    def operational_excess(self) -> torch.Tensor:
        """Raw q-des excess outside the configured operational range, normalized by hard span."""
        if self._operational_lower is None or self._operational_upper is None:
            return torch.zeros_like(self._processed_actions)
        if self._clamp_lower is None or self._clamp_upper is None:
            raise RuntimeError("Operational bounds require explicit mechanical clamps")
        span = (self._clamp_upper - self._clamp_lower).clamp_min(1.0e-6)
        lower = torch.relu(self._operational_lower - self._unclamped_processed_actions)
        upper = torch.relu(self._unclamped_processed_actions - self._operational_upper)
        return (lower + upper) / span

    @property
    def q_des_velocity(self) -> torch.Tensor:
        return self._q_des_velocity

    @property
    def q_des_acceleration(self) -> torch.Tensor:
        return self._q_des_acceleration

    @property
    def q_des_velocity_excess_ratio(self) -> torch.Tensor:
        if self._q_des_velocity_limit is None:
            return torch.zeros_like(self._q_des_velocity)
        return torch.relu(
            torch.abs(self._q_des_velocity) / self._q_des_velocity_limit - 1.0
        )

    @property
    def q_des_acceleration_excess_ratio(self) -> torch.Tensor:
        if self._q_des_acceleration_limit is None:
            return torch.zeros_like(self._q_des_acceleration)
        return torch.relu(
            torch.abs(self._q_des_acceleration)
            / self._q_des_acceleration_limit
            - 1.0
        )

    @property
    def feedback_mode(self) -> str:
        return self.cfg.feedback_mode

    @property
    def action_joint_names(self) -> tuple[str, ...]:
        return tuple(self._joint_names)

    @property
    def action_joint_ids(self) -> tuple[int, ...]:
        return tuple(_action_joint_ids_in_column_order(self))

    @property
    def passive_joint_names(self) -> tuple[str, ...]:
        return self._passive_joint_names


@configclass
class ClampedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ClampedJointPositionAction

    passive_joint_names: tuple[str, ...] = ()
    """Exact joint names held at default q and zeroed in the applied-action feedback (e.g. the head)."""

    feedback_mode: str = "raw"
    """Next-tick actor feedback: ``raw`` or the residual represented by clamped q_des."""

    position_clamp: dict[str, tuple[float, float]] | None = None
    """Explicit per-joint (lower, upper) clamp in rad, keyed by exact joint name.

    Set from the shared deploy ``action_adapter.yaml`` so training clamps position targets with
    exactly the same bounds as the deploy ActionAdapter. ``None`` falls back to the articulation's
    soft joint limits.
    """

    operational_margin_fraction: dict[str, float] | None = None
    """Per-joint soft margin as a fraction of the mechanical position span."""

    q_des_velocity_limit: dict[str, float] | None = None
    """Optional per-joint target slew limits in rad/s; diagnostics and rewards only."""

    q_des_acceleration_limit: dict[str, float] | None = None
    """Optional per-joint target acceleration limits in rad/s^2; diagnostics and rewards only."""
