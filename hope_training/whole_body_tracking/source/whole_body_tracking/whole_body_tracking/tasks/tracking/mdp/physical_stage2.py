"""Physical-outcome rewards and capability curriculum for HOPE Stage 2."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase

from whole_body_tracking.tasks.tracking.mdp.table_workspace import (
    ramped_curriculum_threshold,
)
from whole_body_tracking.tasks.tracking.mdp.closed_loop_v2 import (
    face_center_quality,
)


def _command(env, name: str):
    return env.command_manager.get_term(name)


def _env_ids_tensor(env_ids, *, num_envs: int, device: str) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, dtype=torch.long, device=device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=device)


def _event_manager_value(
    env, value: torch.Tensor, *, impulse: bool
) -> torch.Tensor:
    """Convert a one-shot event value to Isaac RewardManager units.

    RewardManager multiplies every term by ``step_dt``. Dense rates should keep
    that integration, while a one-control-frame event must divide by ``step_dt``
    so its configured weight is the actual transition value.
    """
    if not impulse:
        return value
    step_dt = float(env.step_dt)
    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive for impulse rewards")
    return value / step_dt


class physical_outcome_events(ManagerTermBase):
    """Reward real PhysX contact/net/bounce pulses with strike-health gating."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._contact_health = torch.zeros(self.num_envs, device=self.device)
        self._contact_face_quality = torch.zeros(
            self.num_envs, device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = _env_ids_tensor(
            env_ids, num_envs=self.num_envs, device=self.device
        )
        self._contact_health[ids] = 0.0
        self._contact_face_quality[ids] = 0.0

    def __call__(
        self,
        env,
        physical_command_name: str,
        target_command_name: str,
        contact_scale: float = 1.0,
        net_cross_scale: float = 3.0,
        opponent_bounce_scale: float = 6.0,
        contact_quality_scale: float = 0.0,
        outgoing_velocity_error_std: float = 0.90,
        outgoing_direction_error_std_deg: float = 30.0,
        face_inner_radius: float = 0.040,
        face_outer_radius: float = 0.095,
        minimum_face_multiplier: float = 0.05,
        minimum_health_multiplier: float = 0.25,
        minimum_recovery_multiplier: float = 0.35,
        impact_inverse_quality_floor: float = 0.0,
        impulse: bool = False,
    ) -> torch.Tensor:
        shadow = _command(env, physical_command_name)
        target = _command(env, target_command_name)

        new_route = shadow.serve_event | target.target_just_resampled
        self._contact_health[new_route] = 0.0
        self._contact_face_quality[new_route] = 0.0
        contact = shadow.contact_event
        self._contact_health[contact] = target.impact_health_score[contact].clamp(
            0.0, 1.0
        )
        current_face_quality = face_center_quality(
            shadow.contact_face_radial_error,
            inner_radius=float(face_inner_radius),
            outer_radius=float(face_outer_radius),
        )
        self._contact_face_quality[contact] = current_face_quality[contact]

        health_floor = float(minimum_health_multiplier)
        health = health_floor + (1.0 - health_floor) * self._contact_health
        ready = target.metrics["recovery_functional_ready_score"].clamp(0.0, 1.0)
        recovery_floor = float(minimum_recovery_multiplier)
        recovery = recovery_floor + (1.0 - recovery_floor) * ready
        face_floor = min(max(float(minimum_face_multiplier), 0.0), 1.0)
        face = face_floor + (1.0 - face_floor) * self._contact_face_quality

        velocity_quality = torch.exp(
            -torch.square(
                shadow.contact_outgoing_velocity_error
                / max(float(outgoing_velocity_error_std), 1.0e-6)
            )
        )
        direction_quality = torch.exp(
            -torch.square(
                shadow.contact_outgoing_direction_error_deg
                / max(float(outgoing_direction_error_std_deg), 1.0e-6)
            )
        )
        contact_quality = velocity_quality * direction_quality
        impact_inverse_blend = target.metrics[
            "impact_inverse_command_blend"
        ].clamp(0.0, 1.0)
        quality_floor = min(
            max(float(impact_inverse_quality_floor), 0.0), 1.0
        )
        quality_command_scale = impact_inverse_blend.clamp_min(quality_floor)

        target.metrics["physical_contact_health_score"] = self._contact_health
        target.metrics["physical_contact_face_quality"] = (
            self._contact_face_quality
        )
        target.metrics["physical_contact_outcome_quality"] = (
            contact.float() * contact_quality
        )
        target.metrics["physical_contact_quality_command_scale"] = (
            quality_command_scale
        )
        value = (
            float(contact_scale) * contact.float() * health * face
            + float(contact_quality_scale)
            * contact.float()
            * contact_quality
            * quality_command_scale
            * health
            * face
            + float(net_cross_scale)
            * shadow.net_cross_event.float()
            * health
            * face
            + float(opponent_bounce_scale)
            * shadow.opponent_bounce_event.float()
            * health
            * recovery
            * face
        )
        return _event_manager_value(env, value, impulse=bool(impulse))


def physical_contact_planner_alignment(
    env,
    physical_command_name: str,
    target_command_name: str,
    position_std: float = 0.10,
    velocity_std: float = 0.65,
    direction_std_deg: float = 18.0,
    normal_std_deg: float = 15.0,
    timing_std_s: float = 0.06,
    component_floor: float = 0.04,
    face_inner_radius: float = 0.040,
    face_outer_radius: float = 0.095,
    minimum_face_multiplier: float = 0.05,
    minimum_health_multiplier: float = 0.15,
    impulse: bool = False,
) -> torch.Tensor:
    """Reward planner execution only when a real PhysX contact occurs.

    Dense strike-window terms can be collected without ever touching the ball.
    This term uses the diagnostics latched by the rigid-ball command at the
    physical contact frame, so position, velocity, normal, and timing must all
    describe the same realized impact. The base contact reward remains separate,
    which preserves exploration while making a precise contact more valuable.
    """
    shadow = _command(env, physical_command_name)
    target = _command(env, target_command_name)

    scales = (
        max(float(position_std), 1.0e-6),
        max(float(velocity_std), 1.0e-6),
        max(float(direction_std_deg), 1.0e-6),
        max(float(normal_std_deg), 1.0e-6),
        max(float(timing_std_s), 1.0e-6),
    )
    errors = (
        shadow.contact_planner_position_error,
        shadow.contact_planner_velocity_error,
        shadow.contact_planner_velocity_direction_error_deg,
        shadow.contact_planner_normal_error_deg,
        shadow.contact_time_to_strike.abs(),
    )
    scores = torch.stack(
        tuple(
            torch.exp(-torch.square(error / scale))
            for error, scale in zip(errors, scales, strict=True)
        ),
        dim=-1,
    )
    floor = min(max(float(component_floor), 0.0), 0.95)
    lifted = floor + (1.0 - floor) * scores.clamp(0.0, 1.0)
    geometric = torch.exp(torch.mean(torch.log(lifted.clamp_min(1.0e-8)), dim=-1))
    alignment = ((geometric - floor) / (1.0 - floor)).clamp(0.0, 1.0)

    face_quality = face_center_quality(
        shadow.contact_face_radial_error,
        inner_radius=float(face_inner_radius),
        outer_radius=float(face_outer_radius),
    )
    face_floor = min(max(float(minimum_face_multiplier), 0.0), 1.0)
    face = face_floor + (1.0 - face_floor) * face_quality
    health_floor = min(max(float(minimum_health_multiplier), 0.0), 1.0)
    health = health_floor + (1.0 - health_floor) * target.impact_health_score.clamp(
        0.0, 1.0
    )
    contact = shadow.contact_event.float()
    value = contact * alignment * face * health
    target.metrics["physical_contact_planner_alignment"] = value
    target.metrics["physical_contact_planner_alignment_raw"] = contact * alignment
    return _event_manager_value(env, value, impulse=bool(impulse))


class physical_outcome_recovery_settlement(ManagerTermBase):
    """Settle a physical shot only after its flight resolves and READY returns."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._tier = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._route_resolved = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._resolved_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._ready_steps = torch.zeros_like(self._resolved_steps)
        self._durable_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._durable_failure = torch.zeros_like(self._durable_success)
        self._safe_settlement = torch.zeros_like(self._durable_success)
        self._unsafe_settlement = torch.zeros_like(self._durable_success)

    def _clear(self, ids: torch.Tensor) -> None:
        self._tier[ids] = 0
        self._route_resolved[ids] = False
        self._resolved_steps[ids] = 0
        self._ready_steps[ids] = 0
        self._durable_success[ids] = False
        self._durable_failure[ids] = False
        self._safe_settlement[ids] = False
        self._unsafe_settlement[ids] = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._clear(
            _env_ids_tensor(
                env_ids, num_envs=self.num_envs, device=self.device
            )
        )

    def __call__(
        self,
        env,
        physical_command_name: str,
        target_command_name: str,
        ready_threshold: float = 0.62,
        minimum_resolved_steps: int = 3,
        required_ready_steps: int = 5,
        deadline_steps: int = 55,
        contact_value: float = 1.0,
        net_cross_value: float = 2.0,
        opponent_bounce_value: float = 4.0,
        failure_cost: float = 0.20,
        terminal_failure_cost: float = -1.0,
        impulse: bool = False,
        require_durable_recovery: bool = False,
        require_safe_settlement: bool = False,
    ) -> torch.Tensor:
        shadow = _command(env, physical_command_name)
        target = _command(env, target_command_name)

        self._tier = torch.maximum(
            self._tier,
            shadow.contact_event.long(),
        )
        self._tier = torch.maximum(
            self._tier,
            2 * shadow.net_cross_event.long(),
        )
        self._tier = torch.maximum(
            self._tier,
            3 * shadow.opponent_bounce_event.long(),
        )
        resolved_now = (
            shadow.outgoing_landing_event
            | shadow.incoming_net_collision_event
            | shadow.timeout_event
            | shadow.abort_event
        )
        self._route_resolved |= resolved_now

        active = self._tier > 0
        resolved_active = active & self._route_resolved
        self._resolved_steps[resolved_active] += 1
        ready = (
            target.metrics["recovery_functional_ready_score"]
            >= float(ready_threshold)
        )
        eligible = resolved_active & (
            self._resolved_steps >= int(minimum_resolved_steps)
        )
        self._ready_steps = torch.where(
            eligible & ready,
            self._ready_steps + 1,
            torch.zeros_like(self._ready_steps),
        )

        self._durable_success |= (
            target.post_contact_ready_durable_success_event
        )
        self._durable_failure |= (
            target.post_contact_ready_durable_fail_event
        )
        self._safe_settlement |= (
            target.post_contact_ready_safe_settlement_event
        )
        self._unsafe_settlement |= (
            target.post_contact_ready_unsafe_settlement_event
            | target.post_contact_ready_incomplete_settlement_event
        )
        if bool(require_durable_recovery):
            # V11 settles physical value only after the fixed-deadline READY
            # contract resolves, never on an early soft READY crossing.
            success = resolved_active & self._durable_success
            if bool(require_safe_settlement):
                # A safe endpoint cannot erase a dangerous post-impact path.
                # The command term latches envelope violations from impact to
                # settlement and classifies that terminal event as unsafe.
                success &= self._safe_settlement
        else:
            success = eligible & (
                self._ready_steps >= int(required_ready_steps)
            )
        expired = resolved_active & (
            self._resolved_steps >= int(deadline_steps)
        )
        interrupted = active & target.target_just_resampled
        # Terminations are computed before rewards in ManagerBasedRLEnv.step().
        # Charge a hit that ends in a hard reset on this step; otherwise reset()
        # clears the pending settlement without ever assigning failure credit.
        terminal_failed = active & env.termination_manager.terminated & (~success)
        durable_failed = (
            active
            & (
                self._durable_failure
                | (
                    self._unsafe_settlement
                    if bool(require_safe_settlement)
                    else torch.zeros_like(active)
                )
            )
            if bool(require_durable_recovery)
            else torch.zeros_like(active)
        )
        nonterminal_failed = (
            (expired | interrupted | durable_failed)
            & (~success)
            & (~terminal_failed)
        )
        failed = nonterminal_failed | terminal_failed

        values = torch.zeros(self.num_envs, device=self.device)
        values = torch.where(
            self._tier == 1,
            torch.full_like(values, float(contact_value)),
            values,
        )
        values = torch.where(
            self._tier == 2,
            torch.full_like(values, float(net_cross_value)),
            values,
        )
        values = torch.where(
            self._tier == 3,
            torch.full_like(values, float(opponent_bounce_value)),
            values,
        )
        hard_cost = (
            float(failure_cost)
            if float(terminal_failure_cost) < 0.0
            else float(terminal_failure_cost)
        )
        reward = (
            success.float() * values
            - nonterminal_failed.float() * values * float(failure_cost)
            - terminal_failed.float() * values * hard_cost
        )

        target.metrics["physical_recovery_pending"] = active.float()
        target.metrics["physical_recovery_tier"] = self._tier.float()
        target.metrics["physical_recovery_success_event"] = success.float()
        target.metrics["physical_recovery_failure_event"] = failed.float()
        target.metrics["physical_recovery_nonterminal_failure_event"] = (
            nonterminal_failed.float()
        )
        target.metrics["physical_recovery_terminal_failure_event"] = (
            terminal_failed.float()
        )
        self._clear(torch.where(success | failed)[0])
        return _event_manager_value(env, reward, impulse=bool(impulse))


class physical_capability_curriculum(ManagerTermBase):
    """Drive command difficulty only from completed rigid-ball attempts."""

    _METRIC_NAMES = (
        "raw_contact",
        "contact",
        "aligned_contact",
        "net",
        "bounce",
        "recovery",
        "safety",
    )

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._served = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._contact = torch.zeros_like(self._served)
        self._aligned_contact = torch.zeros_like(self._served)
        self._raw_contact = torch.zeros_like(self._served)
        self._net = torch.zeros_like(self._served)
        self._bounce = torch.zeros_like(self._served)
        self._route_resolved = torch.zeros_like(self._served)
        self._best_post_route_recovery = torch.zeros(
            self.num_envs, device=self.device
        )
        self._batch_count = 0
        self._batch_sums = {
            name: torch.zeros((), device=self.device)
            for name in self._METRIC_NAMES
        }
        self._ema = {
            name: torch.zeros((), device=self.device)
            for name in self._METRIC_NAMES
        }
        self._initialized = False
        self._advance_streak = 0
        self._regress_streak = 0

    def _clear_trials(self, ids: torch.Tensor) -> None:
        self._served[ids] = False
        self._contact[ids] = False
        self._aligned_contact[ids] = False
        self._raw_contact[ids] = False
        self._net[ids] = False
        self._bounce[ids] = False
        self._route_resolved[ids] = False
        self._best_post_route_recovery[ids] = 0.0

    def _accumulate(
        self,
        ids: torch.Tensor,
        recovery: torch.Tensor,
        safety: torch.Tensor,
    ) -> None:
        if len(ids) == 0:
            return
        self._batch_count += int(len(ids))
        self._batch_sums["raw_contact"] += self._raw_contact[ids].float().sum()
        self._batch_sums["contact"] += self._contact[ids].float().sum()
        self._batch_sums["aligned_contact"] += self._aligned_contact[
            ids
        ].float().sum()
        self._batch_sums["net"] += self._net[ids].float().sum()
        self._batch_sums["bounce"] += self._bounce[ids].float().sum()
        self._batch_sums["recovery"] += recovery.float().sum()
        self._batch_sums["safety"] += safety.float().sum()

    def _maybe_update(
        self,
        target,
        *,
        minimum_events: int,
        ema_rate: float,
        contact_threshold: float,
        bootstrap_contact_threshold: float,
        full_contact_threshold_level: float,
        aligned_contact_threshold: float,
        bootstrap_aligned_contact_threshold: float,
        full_aligned_contact_threshold_level: float,
        net_threshold: float,
        bounce_threshold: float,
        recovery_threshold: float,
        bootstrap_recovery_threshold: float,
        full_recovery_threshold_level: float,
        safety_threshold: float,
        recovery_floor: float,
        safety_floor: float,
        contact_only_until_level: float,
        net_full_threshold_level: float,
        net_only_until_level: float,
        bounce_full_threshold_level: float,
        contact_regress_ratio: float,
        aligned_contact_regress_ratio: float,
        net_regress_ratio: float,
        bounce_regress_ratio: float,
        level_step: float,
        regress_step: float,
        required_advance_checks: int,
        required_regress_checks: int,
    ) -> None:
        if self._batch_count < int(minimum_events):
            return
        count = float(self._batch_count)
        rate = min(max(float(ema_rate), 0.0), 1.0)
        for name in self._METRIC_NAMES:
            batch = self._batch_sums[name] / count
            if self._initialized:
                self._ema[name].mul_(1.0 - rate).add_(rate * batch)
            else:
                self._ema[name].copy_(batch)
            self._batch_sums[name].zero_()
        self._batch_count = 0
        self._initialized = True

        level = float(target._ability_curriculum_level.item())
        bootstrap_contact = (
            float(contact_threshold)
            if float(bootstrap_contact_threshold) < 0.0
            else float(bootstrap_contact_threshold)
        )
        contact_alpha = min(
            max(level / max(float(full_contact_threshold_level), 1.0e-6), 0.0),
            1.0,
        )
        effective_contact_threshold = bootstrap_contact + contact_alpha * (
            float(contact_threshold) - bootstrap_contact
        )
        alignment_alpha = min(
            max(
                level
                / max(float(full_aligned_contact_threshold_level), 1.0e-6),
                0.0,
            ),
            1.0,
        )
        effective_alignment_threshold = float(
            bootstrap_aligned_contact_threshold
        ) + alignment_alpha * (
            float(aligned_contact_threshold)
            - float(bootstrap_aligned_contact_threshold)
        )
        effective_net_threshold = ramped_curriculum_threshold(
            level,
            float(contact_only_until_level),
            float(net_full_threshold_level),
            float(net_threshold),
        )
        effective_bounce_threshold = ramped_curriculum_threshold(
            level,
            float(net_only_until_level),
            float(bounce_full_threshold_level),
            float(bounce_threshold),
        )
        threshold_level = max(float(full_recovery_threshold_level), 1.0e-6)
        recovery_alpha = min(max(level / threshold_level, 0.0), 1.0)
        effective_recovery_threshold = (
            float(bootstrap_recovery_threshold)
            + recovery_alpha
            * (float(recovery_threshold) - float(bootstrap_recovery_threshold))
        )
        advance = (
            float(self._ema["contact"].item())
            >= effective_contact_threshold
            and float(self._ema["aligned_contact"].item())
            >= effective_alignment_threshold
            and float(self._ema["net"].item()) >= effective_net_threshold
            and float(self._ema["bounce"].item()) >= effective_bounce_threshold
            and float(self._ema["recovery"].item())
            >= effective_recovery_threshold
            and float(self._ema["safety"].item()) >= float(safety_threshold)
        )
        contact_regress = float(contact_regress_ratio) > 0.0 and float(
            self._ema["contact"].item()
        ) < effective_contact_threshold * float(contact_regress_ratio)
        aligned_contact_regress = (
            effective_alignment_threshold > 0.0
            and float(aligned_contact_regress_ratio) > 0.0
            and float(self._ema["aligned_contact"].item())
            < effective_alignment_threshold
            * float(aligned_contact_regress_ratio)
        )
        net_regress = effective_net_threshold > 0.0 and float(
            net_regress_ratio
        ) > 0.0 and float(self._ema["net"].item()) < (
            effective_net_threshold * float(net_regress_ratio)
        )
        bounce_regress = effective_bounce_threshold > 0.0 and float(
            bounce_regress_ratio
        ) > 0.0 and float(self._ema["bounce"].item()) < (
            effective_bounce_threshold * float(bounce_regress_ratio)
        )
        regress = (
            float(self._ema["recovery"].item()) < float(recovery_floor)
            or float(self._ema["safety"].item()) < float(safety_floor)
            or contact_regress
            or aligned_contact_regress
            or net_regress
            or bounce_regress
        )
        self._advance_streak = self._advance_streak + 1 if advance else 0
        self._regress_streak = self._regress_streak + 1 if regress else 0

        if self._regress_streak >= int(required_regress_checks):
            level = max(0.0, level - float(regress_step))
            self._regress_streak = 0
            self._advance_streak = 0
        elif self._advance_streak >= int(required_advance_checks):
            level = min(1.0, level + float(level_step))
            self._advance_streak = 0
        target._ability_curriculum_level.fill_(level)
        target.metrics["physical_ability_recovery_threshold"] = torch.full(
            (self.num_envs,),
            effective_recovery_threshold,
            device=self.device,
        )
        target.metrics["physical_ability_contact_threshold"] = torch.full(
            (self.num_envs,), effective_contact_threshold, device=self.device
        )
        target.metrics["physical_ability_aligned_contact_threshold"] = (
            torch.full(
                (self.num_envs,),
                effective_alignment_threshold,
                device=self.device,
            )
        )
        target.metrics["physical_ability_net_threshold"] = torch.full(
            (self.num_envs,), effective_net_threshold, device=self.device
        )
        target.metrics["physical_ability_bounce_threshold"] = torch.full(
            (self.num_envs,), effective_bounce_threshold, device=self.device
        )

    def _publish(self, target) -> None:
        level = float(target._ability_curriculum_level.item())
        target.metrics["physical_ability_level"] = torch.full(
            (self.num_envs,), level, device=self.device
        )
        target.metrics["physical_ability_batch_events"] = torch.full(
            (self.num_envs,), float(self._batch_count), device=self.device
        )
        for name in self._METRIC_NAMES:
            target.metrics[f"physical_ability_{name}_ema"] = torch.full(
                (self.num_envs,),
                float(self._ema[name].item()),
                device=self.device,
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = _env_ids_tensor(
            env_ids, num_envs=self.num_envs, device=self.device
        )
        failed = ids[self._served[ids]]
        if len(failed) > 0:
            zeros = torch.zeros(len(failed), device=self.device)
            self._accumulate(failed, zeros, zeros)
        self._clear_trials(ids)

    def __call__(
        self,
        env,
        physical_command_name: str,
        target_command_name: str,
        minimum_events: int = 512,
        ema_rate: float = 0.35,
        contact_threshold: float = 0.45,
        bootstrap_contact_threshold: float = -1.0,
        full_contact_threshold_level: float = 1.0,
        aligned_contact_threshold: float = 0.0,
        bootstrap_aligned_contact_threshold: float = 0.0,
        full_aligned_contact_threshold_level: float = 1.0,
        net_threshold: float = 0.25,
        bounce_threshold: float = 0.10,
        recovery_threshold: float = 0.60,
        bootstrap_recovery_threshold: float = 0.35,
        full_recovery_threshold_level: float = 0.40,
        safety_threshold: float = 0.65,
        recovery_floor: float = 0.38,
        safety_floor: float = 0.45,
        contact_only_until_level: float = 0.40,
        net_full_threshold_level: float = 1.0,
        net_only_until_level: float = 0.70,
        bounce_full_threshold_level: float = 1.0,
        contact_regress_ratio: float = 0.0,
        aligned_contact_regress_ratio: float = 0.0,
        net_regress_ratio: float = 0.0,
        bounce_regress_ratio: float = 0.0,
        center_contact_radius: float = 0.061,
        aligned_position_error_max: float = 0.14,
        aligned_velocity_error_max: float = 1.25,
        aligned_velocity_direction_error_max_deg: float = 40.0,
        aligned_normal_error_max_deg: float = 32.0,
        aligned_timing_error_max_s: float = 0.07,
        level_step: float = 0.20,
        regress_step: float = 0.10,
        required_advance_checks: int = 2,
        required_regress_checks: int = 2,
        fixed_level: float = -1.0,
    ) -> torch.Tensor:
        shadow = _command(env, physical_command_name)
        target = _command(env, target_command_name)

        resolved_now = (
            shadow.outgoing_landing_event
            | shadow.incoming_net_collision_event
            | shadow.timeout_event
            | shadow.abort_event
        )
        self._route_resolved |= resolved_now
        current_recovery = target.metrics[
            "recovery_functional_ready_score"
        ].clamp(0.0, 1.0)
        resolved = self._served & self._route_resolved
        self._best_post_route_recovery[resolved] = torch.maximum(
            self._best_post_route_recovery[resolved],
            current_recovery[resolved],
        )

        finished = target.target_just_resampled & self._served
        finished_ids = torch.where(finished)[0]
        if len(finished_ids) > 0:
            recovery = self._best_post_route_recovery[finished_ids]
            self._accumulate(
                finished_ids,
                recovery,
                torch.ones(len(finished_ids), device=self.device),
            )
            self._clear_trials(finished_ids)

        serve_ids = torch.where(shadow.serve_event)[0]
        self._clear_trials(serve_ids)
        self._served[serve_ids] = True
        self._raw_contact |= shadow.contact_event
        center_contact = shadow.contact_event & (
            shadow.contact_face_radial_error <= float(center_contact_radius)
        )
        self._contact |= center_contact
        aligned_contact = (
            center_contact
            & (
                shadow.contact_planner_position_error
                <= float(aligned_position_error_max)
            )
            & (
                shadow.contact_planner_velocity_error
                <= float(aligned_velocity_error_max)
            )
            & (
                shadow.contact_planner_velocity_direction_error_deg
                <= float(aligned_velocity_direction_error_max_deg)
            )
            & (
                shadow.contact_planner_normal_error_deg
                <= float(aligned_normal_error_max_deg)
            )
            & (
                torch.abs(shadow.contact_time_to_strike)
                <= float(aligned_timing_error_max_s)
            )
        )
        self._aligned_contact |= aligned_contact
        self._net |= shadow.net_cross_event & self._contact
        self._bounce |= shadow.opponent_bounce_event & self._net
        target.metrics["physical_center_contact_event"] = center_contact.float()
        target.metrics["physical_aligned_contact_event"] = (
            aligned_contact.float()
        )

        self._maybe_update(
            target,
            minimum_events=minimum_events,
            ema_rate=ema_rate,
            contact_threshold=contact_threshold,
            bootstrap_contact_threshold=bootstrap_contact_threshold,
            full_contact_threshold_level=full_contact_threshold_level,
            aligned_contact_threshold=aligned_contact_threshold,
            bootstrap_aligned_contact_threshold=(
                bootstrap_aligned_contact_threshold
            ),
            full_aligned_contact_threshold_level=(
                full_aligned_contact_threshold_level
            ),
            net_threshold=net_threshold,
            bounce_threshold=bounce_threshold,
            recovery_threshold=recovery_threshold,
            bootstrap_recovery_threshold=bootstrap_recovery_threshold,
            full_recovery_threshold_level=full_recovery_threshold_level,
            safety_threshold=safety_threshold,
            recovery_floor=recovery_floor,
            safety_floor=safety_floor,
            contact_only_until_level=contact_only_until_level,
            net_full_threshold_level=net_full_threshold_level,
            net_only_until_level=net_only_until_level,
            bounce_full_threshold_level=bounce_full_threshold_level,
            contact_regress_ratio=contact_regress_ratio,
            aligned_contact_regress_ratio=aligned_contact_regress_ratio,
            net_regress_ratio=net_regress_ratio,
            bounce_regress_ratio=bounce_regress_ratio,
            level_step=level_step,
            regress_step=regress_step,
            required_advance_checks=required_advance_checks,
            required_regress_checks=required_regress_checks,
        )
        if float(fixed_level) >= 0.0:
            if float(fixed_level) > 1.0:
                raise ValueError("fixed_level must be -1 or a value in [0, 1]")
            # Keep collecting physical-event EMAs for acceptance diagnostics,
            # but do not let those events change the command distribution in
            # an isolated calibration run.
            target._ability_curriculum_level.fill_(float(fixed_level))
        self._publish(target)
        return torch.zeros(self.num_envs, device=self.device)
