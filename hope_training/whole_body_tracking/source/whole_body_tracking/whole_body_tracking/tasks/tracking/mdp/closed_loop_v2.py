"""Pure closed-loop lifecycle helpers shared by training and offline replay."""

from __future__ import annotations

from enum import IntEnum

import torch


class ClosedLoopPhase(IntEnum):
    READY_NO_COMMAND = 0
    COMMAND_ACQUIRE = 1
    PRE_STRIKE = 2
    STRIKE = 3
    FOLLOW_THROUGH = 4
    RECOVERY = 5
    NEXT_READY = 6


VALID_RECOVERY_TRIGGERS = ("contact", "targeted_attempt")


def deploy_ready_hold_mask(
    *,
    in_hold: torch.Tensor,
    sampled_hold: torch.Tensor,
    stand_episode: torch.Tensor,
    default_stand_reset: torch.Tensor,
    force_stand_episode: bool,
    force_default_stand_reset: bool,
) -> torch.Tensor:
    """Select deploy READY without independently thinning no-ball resets."""
    shapes = {
        tuple(value.shape)
        for value in (in_hold, sampled_hold, stand_episode, default_stand_reset)
    }
    if len(shapes) != 1:
        raise ValueError("deploy READY masks must have identical shapes")
    selected = sampled_hold.bool()
    if bool(force_stand_episode):
        selected = selected | stand_episode.bool()
    if bool(force_default_stand_reset):
        selected = selected | default_stand_reset.bool()
    return in_hold.bool() & selected


def face_center_quality(
    radial_error: torch.Tensor,
    *,
    inner_radius: float,
    outer_radius: float,
) -> torch.Tensor:
    """Smooth usable-face score with a central plateau and zero-valued rim."""
    inner = float(inner_radius)
    outer = float(outer_radius)
    if inner < 0.0 or outer <= inner:
        raise ValueError(
            "face radii must satisfy 0 <= inner_radius < outer_radius"
        )
    fraction = ((outer - radial_error) / (outer - inner)).clamp(0.0, 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def face_contact_region_masks(
    radial_error: torch.Tensor,
    *,
    inner_radius: float,
    outer_radius: float,
    contact_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition analytic contact radii into center, rim and outer-only zones."""
    inner = float(inner_radius)
    outer = float(outer_radius)
    contact = float(contact_radius)
    if inner < 0.0 or outer <= inner or contact <= outer:
        raise ValueError(
            "face/contact radii must satisfy "
            "0 <= inner_radius < outer_radius < contact_radius"
        )
    center = radial_error <= inner
    rim = (radial_error > inner) & (radial_error <= outer)
    analytic_outer = (radial_error > outer) & (radial_error < contact)
    return center, rim, analytic_outer


def recovery_phase_indices(
    elapsed_steps: torch.Tensor,
    *,
    step_dt: float,
    boundaries_s: tuple[float, ...] = (0.10, 0.30, 0.60),
) -> torch.Tensor:
    """Map recovery elapsed steps into ordered, right-open time windows."""
    if float(step_dt) <= 0.0:
        raise ValueError("recovery diagnostic step_dt must be positive")
    if not boundaries_s:
        raise ValueError("recovery diagnostic boundaries must not be empty")
    if any(float(value) <= 0.0 for value in boundaries_s):
        raise ValueError("recovery diagnostic boundaries must be positive")
    if any(
        float(right) <= float(left)
        for left, right in zip(boundaries_s, boundaries_s[1:])
    ):
        raise ValueError(
            "recovery diagnostic boundaries must be strictly increasing"
        )
    boundaries = torch.tensor(
        boundaries_s,
        dtype=torch.float64,
        device=elapsed_steps.device,
    )
    boundary_steps = torch.ceil(
        boundaries / float(step_dt) - 1.0e-9
    ).to(torch.long)
    return torch.bucketize(
        elapsed_steps.to(torch.long), boundary_steps, right=True
    )


def recovery_outcome_bucket(outcome_tier: torch.Tensor) -> torch.Tensor:
    """Map targeted miss/contact/net/bounce tiers to contiguous buckets."""
    return (outcome_tier.to(torch.long) + 1).clamp(0, 3)


def durable_ready_resolution_events(
    *,
    pending: torch.Tensor,
    elapsed_steps: torch.Tensor,
    deadline_steps: int,
    consecutive_ready_steps: torch.Tensor,
    required_consecutive_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve a durable READY check only at its fixed observation deadline."""
    if int(deadline_steps) < 1:
        raise ValueError("durable READY deadline_steps must be positive")
    if int(required_consecutive_steps) < 1:
        raise ValueError(
            "durable READY required_consecutive_steps must be positive"
        )
    deadline = pending & (elapsed_steps >= int(deadline_steps))
    success = deadline & (
        consecutive_ready_steps >= int(required_consecutive_steps)
    )
    return success, deadline & (~success)


def terminal_quality_window_mask(
    *,
    pending: torch.Tensor,
    elapsed_steps: torch.Tensor,
    deadline_steps: int,
    window_steps: int,
) -> torch.Tensor:
    """Select a fixed terminal window ending at the recovery deadline."""
    if int(deadline_steps) < 1:
        raise ValueError("terminal quality deadline_steps must be positive")
    if int(window_steps) < 1 or int(window_steps) > int(deadline_steps):
        raise ValueError(
            "terminal quality window_steps must be within the deadline"
        )
    first_step = int(deadline_steps) - int(window_steps) + 1
    return (
        pending
        & (elapsed_steps >= first_step)
        & (elapsed_steps <= int(deadline_steps))
    )


def safety_conditioned_terminal_events(
    *,
    settlement_event: torch.Tensor,
    safety_violation_latch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fixed-deadline settlement into safe and unsafe one-shot events."""
    safe = settlement_event & (~safety_violation_latch)
    return safe, settlement_event & safety_violation_latch


def operational_terminal_events(
    *,
    settlement_event: torch.Tensor,
    catastrophic_violation_latch: torch.Tensor,
    operational_ready: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify terminal recovery as operational, catastrophic or incomplete."""
    catastrophic = settlement_event & catastrophic_violation_latch
    operational = (
        settlement_event
        & (~catastrophic_violation_latch)
        & operational_ready
    )
    incomplete = (
        settlement_event
        & (~catastrophic_violation_latch)
        & (~operational_ready)
    )
    return operational, catastrophic, incomplete


def safety_conditioned_outcome_value(
    *,
    safe_settlement_event: torch.Tensor,
    terminal_quality: torch.Tensor,
    outcome_tier: torch.Tensor,
    tier_multipliers: torch.Tensor,
) -> torch.Tensor:
    """Settle graded strike value only after a safe recovery interval."""
    return (
        safe_settlement_event.float()
        * terminal_quality.clamp(0.0, 1.0)
        * outcome_tier_multiplier(outcome_tier, tier_multipliers)
    )


def safety_conditioned_cycle_value(
    *,
    safe_settlement_event: torch.Tensor,
    terminal_quality: torch.Tensor,
    outcome_tier: torch.Tensor,
) -> torch.Tensor:
    """Require net crossing and safe terminal quality for full-cycle value."""
    return (
        safe_settlement_event.float()
        * terminal_quality.clamp(0.0, 1.0)
        * (outcome_tier >= 1).float()
    )


def wire_compatible_velocity(
    velocity: torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    """Preserve commanded speed while coupling velocity direction to normal."""
    speed = torch.norm(velocity, dim=-1, keepdim=True)
    velocity_direction = velocity / speed.clamp_min(1.0e-6)
    normal_norm = torch.norm(normal, dim=-1, keepdim=True)
    direction = torch.where(
        normal_norm > 1.0e-6,
        normal / normal_norm.clamp_min(1.0e-6),
        velocity_direction,
    )
    return speed * direction


def recovery_trigger_event(
    mode: str,
    *,
    strike_fired: torch.Tensor,
    targeted_attempt: torch.Tensor,
    ball_contact: torch.Tensor,
) -> torch.Tensor:
    """Select the one-shot event that opens a recovery window."""
    if mode == "contact":
        return ball_contact
    if mode == "targeted_attempt":
        return strike_fired & targeted_attempt
    raise ValueError(
        f"recovery trigger must be one of {VALID_RECOVERY_TRIGGERS}, got {mode!r}"
    )


def health_floor_multiplier(
    health_score: torch.Tensor,
    minimum: float,
) -> torch.Tensor:
    """Keep an exploration floor while preserving a monotonic health gate."""
    floor = min(max(float(minimum), 0.0), 1.0)
    return floor + (1.0 - floor) * health_score.clamp(0.0, 1.0)


def rigid_body_relative_point_velocity(
    point_position: torch.Tensor,
    point_velocity: torch.Tensor,
    body_position: torch.Tensor,
    body_linear_velocity: torch.Tensor,
    body_angular_velocity: torch.Tensor,
) -> torch.Tensor:
    """Remove the rigid motion of ``body`` from a world-frame point velocity."""
    body_point_velocity = body_linear_velocity + torch.cross(
        body_angular_velocity,
        point_position - body_position,
        dim=-1,
    )
    return point_velocity - body_point_velocity


def signed_directional_velocity_progress(
    actual_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    projection_ratio_scale: float = 0.65,
    lateral_ratio_scale: float = 0.50,
    lateral_weight: float = 0.25,
) -> torch.Tensor:
    """Return a signed, non-saturating velocity signal around zero speed.

    Unlike a Gaussian target-velocity kernel, this has useful gradient at a
    stationary racket. Motion along the command is positive, reverse motion is
    negative, and lateral motion is explicitly discounted.
    """
    target_speed = torch.norm(target_velocity, dim=-1).clamp_min(1.0e-6)
    target_direction = target_velocity / target_speed.unsqueeze(-1)
    projection = torch.sum(actual_velocity * target_direction, dim=-1)
    lateral = actual_velocity - projection.unsqueeze(-1) * target_direction
    projection_ratio = projection / target_speed
    lateral_ratio = torch.norm(lateral, dim=-1) / target_speed
    forward = torch.tanh(
        projection_ratio / max(float(projection_ratio_scale), 1.0e-6)
    )
    lateral_cost = torch.tanh(
        lateral_ratio / max(float(lateral_ratio_scale), 1.0e-6)
    )
    return forward - float(lateral_weight) * lateral_cost


def planner_velocity_alignment_score(
    actual_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    speed_ratio_std: float = 0.30,
    direction_std_rad: float = 0.35,
    component_floor: float = 0.02,
) -> torch.Tensor:
    """Score planner velocity magnitude and direction without a contact gate."""
    target_speed = torch.norm(target_velocity, dim=-1).clamp_min(1.0e-6)
    actual_speed = torch.norm(actual_velocity, dim=-1)
    speed_ratio = actual_speed / target_speed
    cosine = torch.sum(actual_velocity * target_velocity, dim=-1) / (
        actual_speed.clamp_min(1.0e-6) * target_speed
    )
    direction_error = torch.acos(cosine.clamp(-1.0, 1.0))
    return planner_velocity_diagnostic_score(
        speed_ratio,
        direction_error,
        speed_ratio_std=speed_ratio_std,
        direction_std_rad=direction_std_rad,
        component_floor=component_floor,
    )


def planner_velocity_diagnostic_score(
    speed_ratio: torch.Tensor,
    direction_error_rad: torch.Tensor,
    *,
    speed_ratio_std: float = 0.30,
    direction_std_rad: float = 0.35,
    component_floor: float = 0.02,
) -> torch.Tensor:
    """Score latched impact diagnostics using the live alignment contract."""
    speed_score = torch.exp(
        -torch.square(
            (speed_ratio - 1.0) / max(float(speed_ratio_std), 1.0e-6)
        )
    )
    direction_score = torch.exp(
        -torch.square(
            direction_error_rad / max(float(direction_std_rad), 1.0e-6)
        )
    )

    floor = min(max(float(component_floor), 0.0), 0.95)
    lifted_speed = floor + (1.0 - floor) * speed_score
    lifted_direction = floor + (1.0 - floor) * direction_score
    geometric = torch.sqrt(lifted_speed * lifted_direction)
    return ((geometric - floor) / (1.0 - floor)).clamp(0.0, 1.0)


def recovered_planner_velocity_settlement_score(
    *,
    speed_ratio: torch.Tensor,
    direction_error_rad: torch.Tensor,
    position_error: torch.Tensor,
    impact_health_score: torch.Tensor,
    recovery_peak_base_ang_vel: torch.Tensor,
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
    """Settle impact quality only when the following recovery stayed controlled."""
    return recovered_planner_velocity_settlement_components(
        speed_ratio=speed_ratio,
        direction_error_rad=direction_error_rad,
        position_error=position_error,
        impact_health_score=impact_health_score,
        recovery_peak_base_ang_vel=recovery_peak_base_ang_vel,
        speed_ratio_std=speed_ratio_std,
        direction_std_rad=direction_std_rad,
        component_floor=component_floor,
        position_std=position_std,
        position_floor=position_floor,
        impact_health_floor=impact_health_floor,
        recovery_peak_ang_vel_budget=recovery_peak_ang_vel_budget,
        recovery_peak_ang_vel_excess_std=recovery_peak_ang_vel_excess_std,
        recovery_gate_floor=recovery_gate_floor,
    )[-1]


def recovered_planner_command_settlement_score(
    *,
    speed_ratio: torch.Tensor,
    direction_error_rad: torch.Tensor,
    normal_error_rad: torch.Tensor,
    position_error: torch.Tensor,
    impact_health_score: torch.Tensor,
    recovery_peak_base_ang_vel: torch.Tensor,
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
) -> torch.Tensor:
    """Score a complete planner command after the resulting motion recovers."""
    base_score = recovered_planner_velocity_settlement_score(
        speed_ratio=speed_ratio,
        direction_error_rad=direction_error_rad,
        position_error=position_error,
        impact_health_score=impact_health_score,
        recovery_peak_base_ang_vel=recovery_peak_base_ang_vel,
        speed_ratio_std=speed_ratio_std,
        direction_std_rad=direction_std_rad,
        component_floor=component_floor,
        position_std=position_std,
        position_floor=position_floor,
        impact_health_floor=impact_health_floor,
        recovery_peak_ang_vel_budget=recovery_peak_ang_vel_budget,
        recovery_peak_ang_vel_excess_std=(
            recovery_peak_ang_vel_excess_std
        ),
        recovery_gate_floor=recovery_gate_floor,
    )
    normal_score = torch.exp(
        -torch.square(
            normal_error_rad / max(float(normal_std_rad), 1.0e-6)
        )
    )
    floor = min(max(float(normal_floor), 0.0), 1.0)
    normal_gate = floor + (1.0 - floor) * normal_score
    return base_score * normal_gate


def safe_command_cycle_value(
    *,
    safe_settlement_event: torch.Tensor,
    terminal_quality: torch.Tensor,
    command_score: torch.Tensor,
    outcome_tier: torch.Tensor,
    require_contact: bool = True,
) -> torch.Tensor:
    """Settle a command only after its carried state becomes safely reusable."""
    value = (
        safe_settlement_event.float()
        * terminal_quality.clamp(0.0, 1.0)
        * command_score.clamp(0.0, 1.0)
    )
    if bool(require_contact):
        value = value * (outcome_tier >= 0).float()
    return value


def recovered_planner_velocity_settlement_components(
    *,
    speed_ratio: torch.Tensor,
    direction_error_rad: torch.Tensor,
    position_error: torch.Tensor,
    impact_health_score: torch.Tensor,
    recovery_peak_base_ang_vel: torch.Tensor,
    speed_ratio_std: float = 0.30,
    direction_std_rad: float = 0.35,
    component_floor: float = 0.02,
    position_std: float = 0.12,
    position_floor: float = 0.10,
    impact_health_floor: float = 0.25,
    recovery_peak_ang_vel_budget: float = 0.80,
    recovery_peak_ang_vel_excess_std: float = 0.60,
    recovery_gate_floor: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return velocity, position, health, recovery and total settlement scores."""
    velocity_score = planner_velocity_diagnostic_score(
        speed_ratio,
        direction_error_rad,
        speed_ratio_std=speed_ratio_std,
        direction_std_rad=direction_std_rad,
        component_floor=component_floor,
    )
    position_score = torch.exp(
        -torch.square(
            position_error / max(float(position_std), 1.0e-6)
        )
    )
    floor = min(max(float(position_floor), 0.0), 1.0)
    position_gate = floor + (1.0 - floor) * position_score
    recovery_peak_excess = (
        recovery_peak_base_ang_vel
        - max(float(recovery_peak_ang_vel_budget), 0.0)
    ).clamp_min(0.0)
    recovery_gate_raw = torch.exp(
        -torch.square(
            recovery_peak_excess
            / max(float(recovery_peak_ang_vel_excess_std), 1.0e-6)
        )
    )
    recovery_floor = min(max(float(recovery_gate_floor), 0.0), 1.0)
    recovery_gate = (
        recovery_floor + (1.0 - recovery_floor) * recovery_gate_raw
    )
    health_gate = health_floor_multiplier(
        impact_health_score,
        impact_health_floor,
    )
    total = velocity_score * position_gate * health_gate * recovery_gate
    return (
        velocity_score,
        position_gate,
        health_gate,
        recovery_gate,
        total,
    )


def recovery_peak_excess_potential(
    peak_base_ang_vel: torch.Tensor,
    budget: torch.Tensor | float,
    *,
    excess_std: float = 0.60,
    max_potential: float = 4.0,
) -> torch.Tensor:
    """Bound the squared angular-speed excess accumulated by one recovery."""
    budget_tensor = torch.as_tensor(
        budget,
        dtype=peak_base_ang_vel.dtype,
        device=peak_base_ang_vel.device,
    )
    excess = (peak_base_ang_vel - budget_tensor).clamp_min(0.0)
    potential = torch.square(excess / max(float(excess_std), 1.0e-6))
    return potential.clamp_max(max(float(max_potential), 0.0))


def recovery_peak_excess_increment(
    previous_peak_base_ang_vel: torch.Tensor,
    current_peak_base_ang_vel: torch.Tensor,
    budget: torch.Tensor | float,
    *,
    excess_std: float = 0.60,
    max_potential: float = 4.0,
) -> torch.Tensor:
    """Charge only newly created peak excess, never recovery duration."""
    previous = recovery_peak_excess_potential(
        previous_peak_base_ang_vel,
        budget,
        excess_std=excess_std,
        max_potential=max_potential,
    )
    current = recovery_peak_excess_potential(
        current_peak_base_ang_vel,
        budget,
        excess_std=excess_std,
        max_potential=max_potential,
    )
    return (current - previous).clamp_min(0.0)


def recovery_safety_envelope_violations(
    *,
    tilt: torch.Tensor,
    abs_pitch: torch.Tensor,
    com_x: torch.Tensor,
    com_y: torch.Tensor,
    waist_overflow: torch.Tensor,
    leg_overflow: torch.Tensor,
    base_ang_vel: torch.Tensor,
    max_tilt: float,
    max_abs_pitch: float,
    max_com_x: float,
    max_com_y: float,
    max_waist_overflow: float,
    max_leg_overflow: float,
    max_base_ang_vel: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return aggregate and component recovery-envelope violations."""
    tilt_bad = tilt > float(max_tilt)
    pitch_bad = abs_pitch > float(max_abs_pitch)
    com_bad = (com_x > float(max_com_x)) | (com_y > float(max_com_y))
    waist_bad = waist_overflow > float(max_waist_overflow)
    leg_bad = leg_overflow > float(max_leg_overflow)
    base_ang_bad = base_ang_vel > float(max_base_ang_vel)
    support_bad = tilt_bad | pitch_bad | com_bad
    any_bad = support_bad | waist_bad | leg_bad | base_ang_bad
    return (
        any_bad,
        tilt_bad,
        pitch_bad,
        com_bad,
        waist_bad,
        leg_bad,
        base_ang_bad,
        support_bad,
    )


def outcome_tier_multiplier(
    outcome_tier: torch.Tensor,
    multipliers: torch.Tensor,
) -> torch.Tensor:
    """Map miss/contact/net/bounce tiers without rewarding a recovered miss."""
    if multipliers.numel() != 3:
        raise ValueError("outcome multipliers must contain contact/net/bounce")
    valid = outcome_tier >= 0
    index = outcome_tier.clamp(min=0, max=2).to(torch.long)
    return torch.where(valid, multipliers[index], torch.zeros_like(outcome_tier))


def achieved_outcome_tier(
    *,
    contact: torch.Tensor,
    net_cross: torch.Tensor,
    opponent_bounce: torch.Tensor,
) -> torch.Tensor:
    """Return the highest achieved outcome tier: miss=-1, contact=0, net=1, bounce=2."""
    shapes = {
        tuple(value.shape)
        for value in (contact, net_cross, opponent_bounce)
    }
    if len(shapes) != 1:
        raise ValueError("outcome event tensors must have identical shapes")
    tier = torch.full_like(contact, -1, dtype=torch.long)
    tier = torch.where(contact, torch.zeros_like(tier), tier)
    tier = torch.where(net_cross, torch.ones_like(tier), tier)
    tier = torch.where(
        opponent_bounce,
        torch.full_like(tier, 2),
        tier,
    )
    return tier


def lifecycle_hold_gate(
    *,
    active: torch.Tensor,
    complete: torch.Tensor,
    elapsed_steps: torch.Tensor,
    deadline_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify a bounded hold into keep, successful release and timeout release.

    The deadline prevents a random policy from being trapped before it can see a
    strike. Completion wins on the deadline step so a valid settle is never
    reported as a timeout.
    """
    shapes = {
        tuple(value.shape)
        for value in (active, complete, elapsed_steps)
    }
    if len(shapes) != 1:
        raise ValueError("lifecycle hold tensors must have identical shapes")
    if int(deadline_steps) < 1:
        raise ValueError("lifecycle hold deadline_steps must be positive")
    success = active & complete
    timeout = active & (~complete) & (
        elapsed_steps >= int(deadline_steps)
    )
    keep = active & (~success) & (~timeout)
    return keep, success, timeout


def station_relocation_resolution(
    *,
    rehearsal: torch.Tensor,
    hold_seen: torch.Tensor,
    arrived: torch.Tensor,
    settled: torch.Tensor,
    contact: torch.Tensor,
    unsafe: torch.Tensor,
    valid: torch.Tensor,
    terminal_reset: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Resolve relocation outcomes without dropping terminal-reset failures.

    A normal motion wrap uses the measured unsafe latch. Any valid environment
    reset before wrap is a failed deploy cycle even if the last sampled frame
    did not yet trip the geometric unsafe latch (for example table contact).
    """
    values = (
        rehearsal,
        hold_seen,
        arrived,
        settled,
        contact,
        unsafe,
        valid,
    )
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("station relocation tensors must have identical shapes")
    resolved = rehearsal & hold_seen & valid
    arrival_success = resolved & arrived
    settle_success = resolved & settled
    contact_success = resolved & contact
    if terminal_reset:
        safety_success = torch.zeros_like(resolved)
    else:
        safety_success = resolved & (~unsafe)
    return (
        resolved,
        arrival_success,
        settle_success,
        contact_success,
        safety_success,
    )


def interpolate_curriculum_range(
    easy: tuple[float, float],
    full: tuple[float, float],
    level: float,
) -> tuple[float, float]:
    """Interpolate a valid scalar sampling interval by measured ability."""
    easy_lo, easy_hi = (float(value) for value in easy)
    full_lo, full_hi = (float(value) for value in full)
    if easy_hi < easy_lo or full_hi < full_lo:
        raise ValueError(
            f"invalid curriculum range: easy={easy}, full={full}"
        )
    alpha = min(max(float(level), 0.0), 1.0)
    lo = easy_lo + alpha * (full_lo - easy_lo)
    hi = easy_hi + alpha * (full_hi - easy_hi)
    if hi < lo:
        raise ValueError(
            "interpolated curriculum range is inverted; choose compatible "
            f"easy/full bounds, got {(lo, hi)}"
        )
    return lo, hi


def closed_cycle_success_event(
    *,
    recovery_success: torch.Tensor,
    outcome_tier: torch.Tensor,
    healthy_net_cross: torch.Tensor,
    unsafe: torch.Tensor,
    minimum_outcome_tier: int = 1,
) -> torch.Tensor:
    """A useful return is complete only after the same swing settles safely."""
    return (
        recovery_success
        & (outcome_tier >= int(minimum_outcome_tier))
        & healthy_net_cross
        & (~unsafe)
    )


def lifecycle_phase_ids(
    *,
    no_command: torch.Tensor,
    time_to_strike: torch.Tensor,
    strike_window: torch.Tensor,
    recovery_pending: torch.Tensor,
    recovery_success: torch.Tensor,
    acquire_threshold_s: float = 0.35,
    follow_through_s: float = 0.25,
) -> torch.Tensor:
    """Infer a diagnostic phase with deterministic priority.

    This phase is not fed to the actor. It exists so training and replay use the
    same lifecycle definition when reporting occupancy and event settlement.
    """
    phase = torch.full_like(
        time_to_strike,
        int(ClosedLoopPhase.PRE_STRIKE),
        dtype=torch.long,
    )
    phase = torch.where(
        time_to_strike > float(acquire_threshold_s),
        torch.full_like(phase, int(ClosedLoopPhase.COMMAND_ACQUIRE)),
        phase,
    )
    follow = (
        (time_to_strike < 0.0)
        & (time_to_strike >= -float(follow_through_s))
        & (~recovery_pending)
    )
    phase = torch.where(
        follow,
        torch.full_like(phase, int(ClosedLoopPhase.FOLLOW_THROUGH)),
        phase,
    )
    phase = torch.where(
        recovery_pending,
        torch.full_like(phase, int(ClosedLoopPhase.RECOVERY)),
        phase,
    )
    phase = torch.where(
        recovery_success,
        torch.full_like(phase, int(ClosedLoopPhase.NEXT_READY)),
        phase,
    )
    phase = torch.where(
        strike_window,
        torch.full_like(phase, int(ClosedLoopPhase.STRIKE)),
        phase,
    )
    phase = torch.where(
        no_command,
        torch.full_like(phase, int(ClosedLoopPhase.READY_NO_COMMAND)),
        phase,
    )
    return phase


def one_hot_lifecycle(phase_ids: torch.Tensor) -> torch.Tensor:
    """Return a stable seven-column diagnostic phase encoding."""
    return torch.nn.functional.one_hot(
        phase_ids.to(torch.long),
        num_classes=len(ClosedLoopPhase),
    ).to(dtype=torch.float32)
