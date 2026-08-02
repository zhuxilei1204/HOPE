"""Pure helpers for a vectorized rigid-ball shadow lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch


class PhysicalShadowPhase(IntEnum):
    PARKED = 0
    INCOMING_PRE_BOUNCE = 1
    INCOMING_POST_BOUNCE = 2
    OUTGOING = 3
    TERMINAL = 4


@dataclass(frozen=True)
class OneBounceRoute:
    origin: torch.Tensor
    serve_velocity: torch.Tensor
    bounce: torch.Tensor
    pre_bounce_velocity: torch.Tensor
    post_bounce_velocity: torch.Tensor
    incoming_velocity: torch.Tensor
    pre_time: torch.Tensor
    post_time: torch.Tensor

    @property
    def total_time(self) -> torch.Tensor:
        return self.pre_time + self.post_time


@dataclass(frozen=True)
class OutgoingLanding:
    event: torch.Tensor
    position: torch.Tensor
    opponent: torch.Tensor
    short: torch.Tensor
    long: torch.Tensor
    side: torch.Tensor
    no_net: torch.Tensor


@dataclass(frozen=True)
class PlaneCrossingPrediction:
    """Predicted state when a free-flying ball reaches a world-X plane."""

    position: torch.Tensor
    velocity: torch.Tensor
    time: torch.Tensor
    valid: torch.Tensor


def moving_plane_impact_velocity(
    incoming_velocity: torch.Tensor,
    racket_velocity: torch.Tensor,
    racket_normal: torch.Tensor,
    *,
    restitution: float,
    tangential_damping: float,
    tangential_cap: float,
) -> torch.Tensor:
    """Predict a no-spin moving-paddle impact with capped tangential impulse."""
    if incoming_velocity.ndim != 2 or incoming_velocity.shape[-1] != 3:
        raise ValueError("incoming_velocity must have shape (N, 3)")
    if racket_velocity.shape != incoming_velocity.shape:
        raise ValueError("racket_velocity must match incoming_velocity")
    if racket_normal.shape != incoming_velocity.shape:
        raise ValueError("racket_normal must match incoming_velocity")

    normal = racket_normal / torch.linalg.norm(
        racket_normal, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    relative = incoming_velocity - racket_velocity
    normal_velocity = torch.sum(
        relative * normal, dim=-1, keepdim=True
    )
    tangent_velocity = relative - normal_velocity * normal
    tangent_speed = torch.linalg.norm(
        tangent_velocity, dim=-1, keepdim=True
    )
    raw_damping = float(tangential_damping) * tangent_speed
    capped_damping = float(tangential_cap) * (
        1.0 + float(restitution)
    ) * normal_velocity.abs()
    removed_speed = torch.minimum(raw_damping, capped_damping)
    tangent_impulse = -removed_speed * (
        tangent_velocity / tangent_speed.clamp_min(1.0e-9)
    )
    normal_impulse = (
        -(1.0 + float(restitution)) * normal_velocity * normal
    )
    return incoming_velocity + normal_impulse + tangent_impulse


def rigid_contact_point_velocity(
    com_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    contact_point: torch.Tensor,
    com_position: torch.Tensor,
) -> torch.Tensor:
    """Return rigid-body velocity at a world-frame contact point."""
    if com_velocity.ndim != 2 or com_velocity.shape[-1] != 3:
        raise ValueError("com_velocity must have shape (N, 3)")
    for name, value in (
        ("angular_velocity", angular_velocity),
        ("contact_point", contact_point),
        ("com_position", com_position),
    ):
        if value.shape != com_velocity.shape:
            raise ValueError(f"{name} must match com_velocity")
    return com_velocity + torch.linalg.cross(
        angular_velocity,
        contact_point - com_position,
        dim=-1,
    )


def measured_normal_restitution(
    incoming_velocity: torch.Tensor,
    outgoing_velocity: torch.Tensor,
    incoming_surface_velocity: torch.Tensor,
    outgoing_surface_velocity: torch.Tensor,
    contact_normal: torch.Tensor,
    *,
    minimum_approach_speed: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure normal restitution from pre/post relative contact velocities.

    Returns the restitution estimate and a mask indicating samples whose
    pre-impact relative velocity is approaching the surface.
    """
    if incoming_velocity.ndim != 2 or incoming_velocity.shape[-1] != 3:
        raise ValueError("incoming_velocity must have shape (N, 3)")
    for name, value in (
        ("outgoing_velocity", outgoing_velocity),
        ("incoming_surface_velocity", incoming_surface_velocity),
        ("outgoing_surface_velocity", outgoing_surface_velocity),
        ("contact_normal", contact_normal),
    ):
        if value.shape != incoming_velocity.shape:
            raise ValueError(f"{name} must match incoming_velocity")
    normal = contact_normal / torch.linalg.norm(
        contact_normal, dim=-1, keepdim=True
    ).clamp_min(1.0e-9)
    incoming_relative_normal = torch.sum(
        (incoming_velocity - incoming_surface_velocity) * normal,
        dim=-1,
    )
    outgoing_relative_normal = torch.sum(
        (outgoing_velocity - outgoing_surface_velocity) * normal,
        dim=-1,
    )
    valid = incoming_relative_normal < -float(minimum_approach_speed)
    restitution = torch.where(
        valid,
        -outgoing_relative_normal
        / incoming_relative_normal.clamp_max(-float(minimum_approach_speed)),
        torch.zeros_like(incoming_relative_normal),
    )
    return restitution, valid


def _duration_tensor(
    duration: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(
        duration,
        dtype=reference.dtype,
        device=reference.device,
    )
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.shape != (reference.shape[0],):
        raise ValueError("duration must be scalar or have shape (N,)")
    return value


def _drag_acceleration(
    velocity: torch.Tensor,
    *,
    gravity: float,
    drag_k: float,
) -> torch.Tensor:
    gravity_vector = torch.zeros_like(velocity)
    gravity_vector[:, 2] = -float(gravity)
    speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
    return gravity_vector - float(drag_k) * speed * velocity


def predict_drag_plane_crossing(
    position: torch.Tensor,
    velocity: torch.Tensor,
    plane_x: torch.Tensor | float,
    *,
    gravity: float = 9.81,
    drag_k: float = 0.1261,
    max_dt: float = 0.005,
    max_time: float = 0.65,
    minimum_x_speed: float = 0.05,
) -> PlaneCrossingPrediction:
    """Integrate no-spin drag dynamics to the first crossing of ``plane_x``.

    The helper consumes the measured post-bounce state rather than the route
    generator's intended state.  Interpolation within the crossing RK4 step
    keeps the returned position and arrival time continuous at control rate.
    """
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError("position must have shape (N, 3)")
    if velocity.shape != position.shape:
        raise ValueError("velocity must match position shape")
    if float(max_dt) <= 0.0 or float(max_time) <= 0.0:
        raise ValueError("max_dt and max_time must be positive")

    target_x = torch.as_tensor(
        plane_x, dtype=position.dtype, device=position.device
    )
    if target_x.ndim == 0:
        target_x = target_x.expand(position.shape[0])
    if target_x.shape != (position.shape[0],):
        raise ValueError("plane_x must be scalar or have shape (N,)")

    p = position.clone()
    v = velocity.clone()
    prediction_position = torch.zeros_like(position)
    prediction_velocity = torch.zeros_like(velocity)
    prediction_time = torch.zeros(
        position.shape[0], dtype=position.dtype, device=position.device
    )
    delta_x = target_x - position[:, 0]
    active = (
        (delta_x * velocity[:, 0] > 0.0)
        & (velocity[:, 0].abs() >= float(minimum_x_speed))
        & torch.isfinite(position).all(dim=-1)
        & torch.isfinite(velocity).all(dim=-1)
    )
    valid = torch.zeros_like(active)
    elapsed = torch.zeros_like(prediction_time)
    step_dt = float(max_dt)
    max_steps = int(math.ceil(float(max_time) / step_dt))

    for _ in range(max_steps):
        if not bool(torch.any(active)):
            break
        remaining = (float(max_time) - elapsed).clamp_min(0.0)
        h_scalar = remaining.clamp(max=step_dt)
        h = h_scalar.unsqueeze(-1)
        previous_p = p
        previous_v = v

        k1_p = previous_v
        k1_v = _drag_acceleration(
            previous_v, gravity=gravity, drag_k=drag_k
        )
        v2 = previous_v + 0.5 * h * k1_v
        k2_p = v2
        k2_v = _drag_acceleration(v2, gravity=gravity, drag_k=drag_k)
        v3 = previous_v + 0.5 * h * k2_v
        k3_p = v3
        k3_v = _drag_acceleration(v3, gravity=gravity, drag_k=drag_k)
        v4 = previous_v + h * k3_v
        k4_p = v4
        k4_v = _drag_acceleration(v4, gravity=gravity, drag_k=drag_k)
        next_p = previous_p + (h / 6.0) * (
            k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
        )
        next_v = previous_v + (h / 6.0) * (
            k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
        )

        before = target_x - previous_p[:, 0]
        after = target_x - next_p[:, 0]
        crossed = active & (before * after <= 0.0)
        step_dx = next_p[:, 0] - previous_p[:, 0]
        safe_step_dx = torch.where(
            step_dx.abs() > 1.0e-12, step_dx, torch.ones_like(step_dx)
        )
        alpha = (
            (target_x - previous_p[:, 0]) / safe_step_dx
        ).clamp(0.0, 1.0)
        crossing_p = previous_p + alpha.unsqueeze(-1) * (next_p - previous_p)
        crossing_v = previous_v + alpha.unsqueeze(-1) * (next_v - previous_v)
        crossing_p[:, 0] = target_x
        prediction_position[crossed] = crossing_p[crossed]
        prediction_velocity[crossed] = crossing_v[crossed]
        prediction_time[crossed] = (
            elapsed[crossed] + alpha[crossed] * h_scalar[crossed]
        )
        valid |= crossed
        active &= ~crossed
        p = torch.where(active.unsqueeze(-1), next_p, p)
        v = torch.where(active.unsqueeze(-1), next_v, v)
        elapsed = torch.where(active, elapsed + h_scalar, elapsed)
        active &= elapsed < float(max_time) - 1.0e-9

    return PlaneCrossingPrediction(
        position=prediction_position,
        velocity=prediction_velocity,
        time=prediction_time,
        valid=valid,
    )


def predict_linearized_drag_plane_crossing(
    position: torch.Tensor,
    velocity: torch.Tensor,
    plane_x: torch.Tensor | float,
    *,
    gravity: float = 9.81,
    drag_k: float = 0.1261,
    max_time: float = 0.65,
    minimum_x_speed: float = 0.05,
    iterations: int = 3,
) -> PlaneCrossingPrediction:
    """Fast crossing estimate for quadratic drag using iterated linear drag.

    For each iteration, ``k * |v|`` is held constant, which gives a closed-form
    plane crossing and gravity trajectory.  Updating that coefficient from the
    average start/end speed captures most quadratic-drag curvature without a
    policy-step RK4 loop or CPU synchronization.
    """
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError("position must have shape (N, 3)")
    if velocity.shape != position.shape:
        raise ValueError("velocity must match position shape")
    if float(max_time) <= 0.0:
        raise ValueError("max_time must be positive")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")

    target_x = torch.as_tensor(
        plane_x, dtype=position.dtype, device=position.device
    )
    if target_x.ndim == 0:
        target_x = target_x.expand(position.shape[0])
    if target_x.shape != (position.shape[0],):
        raise ValueError("plane_x must be scalar or have shape (N,)")

    delta_x = target_x - position[:, 0]
    vx = velocity[:, 0]
    finite = torch.isfinite(position).all(dim=-1) & torch.isfinite(
        velocity
    ).all(dim=-1)
    moving_toward = (
        (delta_x * vx > 0.0)
        & (vx.abs() >= float(minimum_x_speed))
        & finite
    )
    ballistic_time = delta_x / torch.where(
        vx.abs() >= float(minimum_x_speed), vx, torch.ones_like(vx)
    )
    gravity_vector = torch.zeros_like(velocity)
    gravity_vector[:, 2] = -float(gravity)

    if abs(float(drag_k)) <= 1.0e-12:
        time = ballistic_time
        prediction_position = (
            position
            + velocity * time.unsqueeze(-1)
            + 0.5 * gravity_vector * time.square().unsqueeze(-1)
        )
        prediction_velocity = velocity + gravity_vector * time.unsqueeze(-1)
    else:
        start_speed = torch.linalg.norm(velocity, dim=-1)
        decay_rate = (
            abs(float(drag_k)) * start_speed
        ).clamp_min(1.0e-6)
        prediction_position = torch.zeros_like(position)
        prediction_velocity = torch.zeros_like(velocity)
        time = ballistic_time
        for iteration in range(int(iterations)):
            ratio = 1.0 - decay_rate * delta_x / torch.where(
                vx.abs() >= float(minimum_x_speed), vx, torch.ones_like(vx)
            )
            ratio_safe = ratio.clamp_min(1.0e-6)
            time = -torch.log(ratio_safe) / decay_rate
            exponential = torch.exp(-decay_rate * time)
            travel_factor = (1.0 - exponential) / decay_rate
            prediction_position = (
                position
                + velocity * travel_factor.unsqueeze(-1)
                + gravity_vector
                / decay_rate.unsqueeze(-1)
                * (time - travel_factor).unsqueeze(-1)
            )
            prediction_velocity = (
                velocity * exponential.unsqueeze(-1)
                + gravity_vector
                / decay_rate.unsqueeze(-1)
                * (1.0 - exponential).unsqueeze(-1)
            )
            if iteration + 1 < int(iterations):
                end_speed = torch.linalg.norm(prediction_velocity, dim=-1)
                decay_rate = (
                    abs(float(drag_k)) * 0.5 * (start_speed + end_speed)
                ).clamp_min(1.0e-6)

        # Refine the closed-form estimate with two constant-cost quadratic-drag
        # integrations.  Three variable-size RK4 substeps are sufficient over
        # the sub-0.65 s post-bounce horizon and avoid the old O(time / dt)
        # control-loop cost.
        time = time.clamp(min=1.0e-5, max=float(max_time))
        for refinement in range(2):
            p_refined = position.clone()
            v_refined = velocity.clone()
            h = (time / 3.0).unsqueeze(-1)
            for _ in range(3):
                k1_p = v_refined
                k1_v = _drag_acceleration(
                    v_refined, gravity=gravity, drag_k=drag_k
                )
                v2 = v_refined + 0.5 * h * k1_v
                k2_p = v2
                k2_v = _drag_acceleration(
                    v2, gravity=gravity, drag_k=drag_k
                )
                v3 = v_refined + 0.5 * h * k2_v
                k3_p = v3
                k3_v = _drag_acceleration(
                    v3, gravity=gravity, drag_k=drag_k
                )
                v4 = v_refined + h * k3_v
                k4_p = v4
                k4_v = _drag_acceleration(
                    v4, gravity=gravity, drag_k=drag_k
                )
                p_refined = p_refined + (h / 6.0) * (
                    k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
                )
                v_refined = v_refined + (h / 6.0) * (
                    k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
                )
            prediction_position = p_refined
            prediction_velocity = v_refined
            if refinement == 0:
                time = (
                    time
                    - (p_refined[:, 0] - target_x)
                    / torch.where(
                        v_refined[:, 0].abs()
                        >= float(minimum_x_speed),
                        v_refined[:, 0],
                        torch.ones_like(v_refined[:, 0]),
                    )
                ).clamp(min=1.0e-5, max=float(max_time))

    valid = (
        moving_toward
        & (time > 0.0)
        & (time <= float(max_time))
        & torch.isfinite(time)
        & torch.isfinite(prediction_position).all(dim=-1)
        & torch.isfinite(prediction_velocity).all(dim=-1)
    )
    prediction_position[:, 0] = target_x
    prediction_position = torch.where(
        valid.unsqueeze(-1), prediction_position, torch.zeros_like(position)
    )
    prediction_velocity = torch.where(
        valid.unsqueeze(-1), prediction_velocity, torch.zeros_like(velocity)
    )
    time = torch.where(valid, time, torch.zeros_like(time))
    return PlaneCrossingPrediction(
        position=prediction_position,
        velocity=prediction_velocity,
        time=time,
        valid=valid,
    )


def integrate_drag_state(
    position: torch.Tensor,
    velocity: torch.Tensor,
    duration: float | torch.Tensor,
    *,
    gravity: float = 9.81,
    drag_k: float = 0.1261,
    max_dt: float = 0.005,
    max_duration: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate no-spin drag dynamics forward or backward with RK4."""
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError("position must have shape (N, 3)")
    if velocity.shape != position.shape:
        raise ValueError("velocity must match position shape")
    if float(max_dt) <= 0.0:
        raise ValueError("max_dt must be positive")

    p = position.clone()
    v = velocity.clone()
    signed_duration = _duration_tensor(duration, position)
    direction = torch.where(
        signed_duration >= 0.0,
        torch.ones_like(signed_duration),
        -torch.ones_like(signed_duration),
    )
    remaining = signed_duration.abs()
    if max_duration is None:
        max_steps = int(
            torch.ceil(remaining.max() / float(max_dt)).item()
        )
    else:
        if float(max_duration) < 0.0:
            raise ValueError("max_duration must be non-negative")
        max_steps = int(math.ceil(float(max_duration) / float(max_dt)))
    for _ in range(max_steps):
        h_abs = remaining.clamp(max=float(max_dt))
        h = (direction * h_abs).unsqueeze(-1)

        k1_p = v
        k1_v = _drag_acceleration(v, gravity=gravity, drag_k=drag_k)
        v2 = v + 0.5 * h * k1_v
        k2_p = v2
        k2_v = _drag_acceleration(v2, gravity=gravity, drag_k=drag_k)
        v3 = v + 0.5 * h * k2_v
        k3_p = v3
        k3_v = _drag_acceleration(v3, gravity=gravity, drag_k=drag_k)
        v4 = v + h * k3_v
        k4_p = v4
        k4_v = _drag_acceleration(v4, gravity=gravity, drag_k=drag_k)

        p = p + (h / 6.0) * (
            k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
        )
        v = v + (h / 6.0) * (
            k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v
        )
        remaining = (remaining - h_abs).clamp_min(0.0)
    return p, v


def solve_drag_velocity(
    origin: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
    *,
    gravity: float = 9.81,
    drag_k: float = 0.1261,
    max_dt: float = 0.005,
    iterations: int = 6,
    tolerance: float = 1.0e-4,
    finite_difference: float = 1.0e-3,
) -> torch.Tensor:
    """Solve the launch velocity that reaches ``target`` under drag."""
    if origin.shape != target.shape or origin.ndim != 2 or origin.shape[-1] != 3:
        raise ValueError("origin and target must have matching shape (N, 3)")
    time = _duration_tensor(duration, origin)
    if torch.any(time <= 0.0):
        raise ValueError("shooting duration must be positive")

    gravity_vector = torch.zeros_like(origin)
    gravity_vector[:, 2] = -float(gravity)
    time_column = time.unsqueeze(-1)
    velocity = (
        (target - origin) / time_column
        - 0.5 * gravity_vector * time_column
    )

    for _ in range(int(iterations)):
        reached, _ = integrate_drag_state(
            origin,
            velocity,
            time,
            gravity=gravity,
            drag_k=drag_k,
            max_dt=max_dt,
        )
        error = reached - target
        if float(torch.linalg.norm(error, dim=-1).max().item()) <= float(
            tolerance
        ):
            break

        columns = []
        for axis in range(3):
            perturbation = torch.zeros_like(velocity)
            perturbation[:, axis] = float(finite_difference)
            reached_eps, _ = integrate_drag_state(
                origin,
                velocity + perturbation,
                time,
                gravity=gravity,
                drag_k=drag_k,
                max_dt=max_dt,
            )
            columns.append(
                (reached_eps - reached) / float(finite_difference)
            )
        jacobian = torch.stack(columns, dim=-1)
        try:
            correction = torch.linalg.solve(
                jacobian,
                error.unsqueeze(-1),
            ).squeeze(-1)
        except RuntimeError:
            correction = (
                torch.linalg.pinv(jacobian) @ error.unsqueeze(-1)
            ).squeeze(-1)
        velocity = velocity - correction
    return velocity


def solve_drag_velocity_fixed_point(
    origin: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
    *,
    gravity: float = 9.81,
    drag_k: float = 0.1261,
    max_dt: float = 0.005,
    iterations: int = 4,
    correction_gain: float = 1.2,
    max_duration: float | None = None,
) -> torch.Tensor:
    """GPU-friendly batched shooting without finite-difference Jacobians."""
    if origin.shape != target.shape or origin.ndim != 2 or origin.shape[-1] != 3:
        raise ValueError("origin and target must have matching shape (N, 3)")
    time = _duration_tensor(duration, origin)
    if max_duration is None and torch.any(time <= 0.0):
        raise ValueError("shooting duration must be positive")

    gravity_vector = torch.zeros_like(origin)
    gravity_vector[:, 2] = -float(gravity)
    time_column = time.unsqueeze(-1)
    velocity = (
        (target - origin) / time_column
        - 0.5 * gravity_vector * time_column
    )
    for _ in range(int(iterations)):
        reached, _ = integrate_drag_state(
            origin,
            velocity,
            time,
            gravity=gravity,
            drag_k=drag_k,
            max_dt=max_dt,
            max_duration=max_duration,
        )
        velocity = velocity + (
            float(correction_gain)
            * (target - reached)
            / time_column
        )
    return velocity


def build_one_bounce_route(
    target: torch.Tensor,
    bounce: torch.Tensor,
    pre_time: torch.Tensor,
    post_time: torch.Tensor,
    *,
    horizontal_retain: float,
    vertical_restitution: float,
    gravity: float = 9.81,
    drag_k: float = 0.1261,
    max_dt: float = 0.005,
    pre_max_duration: float | None = None,
    post_max_duration: float | None = None,
) -> OneBounceRoute:
    """Construct a drag-consistent serve around an analytic table bounce."""
    horizontal = float(horizontal_retain)
    vertical = float(vertical_restitution)
    if not 0.0 < horizontal <= 1.0:
        raise ValueError("horizontal_retain must be in (0, 1]")
    if not 0.0 < vertical <= 1.0:
        raise ValueError("vertical_restitution must be in (0, 1]")

    post_bounce_velocity = solve_drag_velocity_fixed_point(
        bounce,
        target,
        post_time,
        gravity=gravity,
        drag_k=drag_k,
        max_dt=max_dt,
        max_duration=post_max_duration,
    )
    _, incoming_velocity = integrate_drag_state(
        bounce,
        post_bounce_velocity,
        post_time,
        gravity=gravity,
        drag_k=drag_k,
        max_dt=max_dt,
        max_duration=post_max_duration,
    )
    pre_bounce_velocity = post_bounce_velocity.clone()
    pre_bounce_velocity[:, :2] /= horizontal
    pre_bounce_velocity[:, 2] = (
        -post_bounce_velocity[:, 2].abs() / vertical
    )
    origin, serve_velocity = integrate_drag_state(
        bounce,
        pre_bounce_velocity,
        -_duration_tensor(pre_time, target),
        gravity=gravity,
        drag_k=drag_k,
        max_dt=max_dt,
        max_duration=pre_max_duration,
    )
    return OneBounceRoute(
        origin=origin,
        serve_velocity=serve_velocity,
        bounce=bounce,
        pre_bounce_velocity=pre_bounce_velocity,
        post_bounce_velocity=post_bounce_velocity,
        incoming_velocity=incoming_velocity,
        pre_time=_duration_tensor(pre_time, target),
        post_time=_duration_tensor(post_time, target),
    )


def detect_table_bounce(
    phase: torch.Tensor,
    previous_position: torch.Tensor,
    previous_velocity: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    *,
    table_surface_z: torch.Tensor,
    ball_radius: float,
    table_near_x: torch.Tensor,
    net_x: torch.Tensor,
    table_y_min: torch.Tensor,
    table_y_max: torch.Tensor,
    surface_band: float = 0.065,
) -> torch.Tensor:
    near_surface = position[:, 2] <= (
        table_surface_z + float(ball_radius) + float(surface_band)
    )
    on_robot_half = (
        (position[:, 0] >= table_near_x)
        & (position[:, 0] <= net_x)
        & (position[:, 1] >= table_y_min)
        & (position[:, 1] <= table_y_max)
    )
    return (
        (phase == int(PhysicalShadowPhase.INCOMING_PRE_BOUNCE))
        & near_surface
        & on_robot_half
        & (previous_velocity[:, 2] < -0.10)
        & (velocity[:, 2] > 0.10)
    )


def detect_racket_contact(
    phase: torch.Tensor,
    previous_velocity: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    racket_position: torch.Tensor,
    racket_force: torch.Tensor,
    *,
    contact_distance: float = 0.12,
    contact_force: float = 2.0,
    velocity_jump: float = 0.65,
) -> torch.Tensor:
    distance = torch.linalg.norm(position - racket_position, dim=-1)
    jump = torch.linalg.norm(velocity - previous_velocity, dim=-1)
    dynamic_contact = (
        (jump >= float(velocity_jump))
        & (previous_velocity[:, 0] < -0.10)
        & (velocity[:, 0] > previous_velocity[:, 0] + 0.25)
    )
    incoming = (
        (phase == int(PhysicalShadowPhase.INCOMING_PRE_BOUNCE))
        | (phase == int(PhysicalShadowPhase.INCOMING_POST_BOUNCE))
    )
    return (
        incoming
        & (distance <= float(contact_distance))
        & ((racket_force >= float(contact_force)) | dynamic_contact)
    )


def detect_net_cross(
    phase: torch.Tensor,
    previous_position: torch.Tensor,
    position: torch.Tensor,
    *,
    net_x: torch.Tensor,
    net_top_z: torch.Tensor,
    ball_radius: float,
) -> torch.Tensor:
    delta_x = position[:, 0] - previous_position[:, 0]
    crossing_fraction = (
        (net_x - previous_position[:, 0])
        / delta_x.clamp_min(1.0e-6)
    ).clamp(0.0, 1.0)
    crossing_z = previous_position[:, 2] + crossing_fraction * (
        position[:, 2] - previous_position[:, 2]
    )
    return (
        (phase == int(PhysicalShadowPhase.OUTGOING))
        & (previous_position[:, 0] < net_x)
        & (position[:, 0] >= net_x)
        & (crossing_z > net_top_z + float(ball_radius))
    )


def detect_incoming_net_collision(
    phase: torch.Tensor,
    previous_position: torch.Tensor,
    previous_velocity: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    *,
    net_x: torch.Tensor,
    net_top_z: torch.Tensor,
    ball_radius: float,
) -> torch.Tensor:
    delta_x = previous_position[:, 0] - position[:, 0]
    crossing_fraction = (
        (previous_position[:, 0] - net_x)
        / delta_x.clamp_min(1.0e-6)
    ).clamp(0.0, 1.0)
    crossing_z = previous_position[:, 2] + crossing_fraction * (
        position[:, 2] - previous_position[:, 2]
    )
    incoming = (
        (phase == int(PhysicalShadowPhase.INCOMING_PRE_BOUNCE))
        | (phase == int(PhysicalShadowPhase.INCOMING_POST_BOUNCE))
    )
    crossing_low = (
        (previous_position[:, 0] > net_x)
        & (position[:, 0] <= net_x)
        & (crossing_z <= net_top_z + float(ball_radius))
    )
    reversal_near_net = (
        ((position[:, 0] - net_x).abs() <= 0.08)
        & (position[:, 2] <= net_top_z + float(ball_radius) + 0.03)
        & (previous_velocity[:, 0] < -0.10)
        & (velocity[:, 0] > 0.10)
    )
    return incoming & (crossing_low | reversal_near_net)


def detect_opponent_bounce(
    phase: torch.Tensor,
    net_crossed: torch.Tensor,
    previous_velocity: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    *,
    table_surface_z: torch.Tensor,
    ball_radius: float,
    net_x: torch.Tensor,
    table_far_x: torch.Tensor,
    table_y_min: torch.Tensor,
    table_y_max: torch.Tensor,
    surface_band: float = 0.065,
) -> torch.Tensor:
    near_surface = position[:, 2] <= (
        table_surface_z + float(ball_radius) + float(surface_band)
    )
    return (
        (phase == int(PhysicalShadowPhase.OUTGOING))
        & net_crossed
        & near_surface
        & (position[:, 0] > net_x)
        & (position[:, 0] <= table_far_x)
        & (position[:, 1] >= table_y_min)
        & (position[:, 1] <= table_y_max)
        & (previous_velocity[:, 2] < -0.10)
        & (velocity[:, 2] > 0.10)
    )


def detect_outgoing_landing(
    phase: torch.Tensor,
    net_crossed: torch.Tensor,
    previous_position: torch.Tensor,
    previous_velocity: torch.Tensor,
    position: torch.Tensor,
    velocity: torch.Tensor,
    *,
    table_surface_z: torch.Tensor,
    ball_radius: float,
    net_x: torch.Tensor,
    table_far_x: torch.Tensor,
    table_y_min: torch.Tensor,
    table_y_max: torch.Tensor,
    surface_band: float = 0.065,
) -> OutgoingLanding:
    """Detect the first descending table-height crossing, including misses."""
    center_surface_z = table_surface_z + float(ball_radius)
    descending_crossing = (
        (previous_position[:, 2] > center_surface_z)
        & (position[:, 2] <= center_surface_z)
        & (previous_velocity[:, 2] < -0.10)
    )
    physical_bounce = (
        (
            position[:, 2]
            <= center_surface_z + float(surface_band)
        )
        & (previous_velocity[:, 2] < -0.10)
        & (velocity[:, 2] > 0.10)
    )
    outgoing = phase == int(PhysicalShadowPhase.OUTGOING)
    event = outgoing & (descending_crossing | physical_bounce)

    delta_z = position[:, 2] - previous_position[:, 2]
    safe_delta_z = torch.where(
        delta_z.abs() > 1.0e-6,
        delta_z,
        torch.ones_like(delta_z),
    )
    fraction = (
        (center_surface_z - previous_position[:, 2])
        / safe_delta_z
    ).clamp(0.0, 1.0)
    interpolated = previous_position + fraction.unsqueeze(-1) * (
        position - previous_position
    )
    landing_position = torch.where(
        descending_crossing.unsqueeze(-1),
        interpolated,
        position,
    )

    x = landing_position[:, 0]
    y = landing_position[:, 1]
    in_y = (y >= table_y_min) & (y <= table_y_max)
    on_opponent = (
        event
        & net_crossed
        & (x > net_x)
        & (x <= table_far_x)
        & in_y
    )
    short = event & (x <= net_x)
    long = event & (x > table_far_x)
    side = event & (x > net_x) & (x <= table_far_x) & (~in_y)
    no_net = event & (~net_crossed)
    return OutgoingLanding(
        event=event,
        position=landing_position,
        opponent=on_opponent,
        short=short,
        long=long,
        side=side,
        no_net=no_net,
    )
