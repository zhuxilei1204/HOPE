import importlib.util
import pathlib
import sys

import pytest
import torch


_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "physical_ball_shadow.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "physical_ball_shadow_under_test",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
shadow = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = shadow
_SPEC.loader.exec_module(shadow)


def test_drag_integrator_is_forward_backward_consistent() -> None:
    position = torch.tensor([[1.8, -0.7, 0.8]], dtype=torch.float64)
    velocity = torch.tensor([[-4.2, 0.2, 0.1]], dtype=torch.float64)
    forward_position, forward_velocity = shadow.integrate_drag_state(
        position,
        velocity,
        0.35,
        max_dt=0.002,
    )
    recovered_position, recovered_velocity = shadow.integrate_drag_state(
        forward_position,
        forward_velocity,
        -0.35,
        max_dt=0.002,
    )
    torch.testing.assert_close(
        recovered_position,
        position,
        atol=2.0e-8,
        rtol=0.0,
    )
    torch.testing.assert_close(
        recovered_velocity,
        velocity,
        atol=2.0e-8,
        rtol=0.0,
    )


def test_one_bounce_route_closes_both_flight_segments() -> None:
    target = torch.tensor(
        [[0.20, -1.10, 0.48], [0.20, -0.95, 0.55]],
        dtype=torch.float64,
    )
    bounce = torch.tensor(
        [[0.72, -1.08, 0.02], [0.80, -0.98, 0.02]],
        dtype=torch.float64,
    )
    pre_time = torch.tensor([0.65, 0.75], dtype=torch.float64)
    post_time = torch.tensor([0.32, 0.38], dtype=torch.float64)
    route = shadow.build_one_bounce_route(
        target,
        bounce,
        pre_time,
        post_time,
        horizontal_retain=0.631,
        vertical_restitution=0.9215,
        max_dt=0.002,
    )

    reached_bounce, pre_velocity = shadow.integrate_drag_state(
        route.origin,
        route.serve_velocity,
        pre_time,
        max_dt=0.002,
    )
    reached_target, incoming_velocity = shadow.integrate_drag_state(
        route.bounce,
        route.post_bounce_velocity,
        post_time,
        max_dt=0.002,
    )
    torch.testing.assert_close(
        reached_bounce,
        bounce,
        atol=2.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        pre_velocity,
        route.pre_bounce_velocity,
        atol=2.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        reached_target,
        target,
        atol=2.0e-4,
        rtol=0.0,
    )
    torch.testing.assert_close(
        incoming_velocity,
        route.incoming_velocity,
        atol=2.0e-7,
        rtol=0.0,
    )


def test_fixed_point_drag_shooting_is_accurate_in_float32() -> None:
    count = 256
    origin = torch.zeros(count, 3)
    origin[:, 0] = torch.linspace(1.05, 1.42, count)
    origin[:, 1] = torch.linspace(-0.60, 0.20, count)
    origin[:, 2] = 0.78
    target = origin.clone()
    target[:, 0] = 0.70
    target[:, 1] += 0.04
    target[:, 2] = torch.linspace(1.10, 1.46, count)
    duration = torch.linspace(0.28, 0.40, count)
    velocity = shadow.solve_drag_velocity_fixed_point(
        origin,
        target,
        duration,
    )
    reached, _ = shadow.integrate_drag_state(
        origin,
        velocity,
        duration,
    )
    error = torch.linalg.norm(reached - target, dim=-1)
    assert float(error.max()) < 1.0e-4


def test_drag_plane_crossing_recovers_integrated_state_and_time() -> None:
    position = torch.tensor(
        [[1.10, -0.20, 0.48], [1.00, 0.15, 0.52]],
        dtype=torch.float64,
    )
    velocity = torch.tensor(
        [[-2.40, 0.18, 1.20], [-1.85, -0.12, 0.95]],
        dtype=torch.float64,
    )
    duration = torch.tensor([0.34, 0.41], dtype=torch.float64)
    expected_position, expected_velocity = shadow.integrate_drag_state(
        position,
        velocity,
        duration,
        max_dt=0.001,
    )
    prediction = shadow.predict_drag_plane_crossing(
        position,
        velocity,
        expected_position[:, 0],
        max_dt=0.002,
        max_time=0.55,
    )
    assert prediction.valid.all()
    torch.testing.assert_close(
        prediction.time, duration, atol=2.5e-5, rtol=0.0
    )
    torch.testing.assert_close(
        prediction.position, expected_position, atol=3.0e-5, rtol=0.0
    )
    torch.testing.assert_close(
        prediction.velocity, expected_velocity, atol=3.0e-5, rtol=0.0
    )


def test_drag_plane_crossing_rejects_motion_away_from_plane() -> None:
    prediction = shadow.predict_drag_plane_crossing(
        torch.tensor([[1.0, 0.0, 0.5]]),
        torch.tensor([[2.0, 0.0, 0.0]]),
        torch.tensor([0.2]),
    )
    assert not prediction.valid.item()


def test_linearized_drag_plane_crossing_is_control_rate_accurate() -> None:
    position = torch.tensor(
        [[1.10, -0.20, 0.48], [1.00, 0.15, 0.52]],
        dtype=torch.float64,
    )
    velocity = torch.tensor(
        [[-2.40, 0.18, 1.20], [-1.85, -0.12, 0.95]],
        dtype=torch.float64,
    )
    duration = torch.tensor([0.34, 0.41], dtype=torch.float64)
    expected_position, expected_velocity = shadow.integrate_drag_state(
        position, velocity, duration, max_dt=0.001
    )
    prediction = shadow.predict_linearized_drag_plane_crossing(
        position, velocity, expected_position[:, 0]
    )
    assert prediction.valid.all()
    torch.testing.assert_close(
        prediction.time, duration, atol=2.0e-4, rtol=0.0
    )
    torch.testing.assert_close(
        prediction.position, expected_position, atol=3.0e-4, rtol=0.0
    )
    torch.testing.assert_close(
        prediction.velocity, expected_velocity, atol=2.0e-3, rtol=0.0
    )


def test_one_bounce_route_rejects_invalid_bounce_coefficients() -> None:
    value = torch.zeros((1, 3))
    time = torch.ones(1)
    with pytest.raises(ValueError):
        shadow.build_one_bounce_route(
            value,
            value,
            time,
            time,
            horizontal_retain=0.0,
            vertical_restitution=0.9,
        )


def test_moving_plane_impact_separates_normal_and_tangential_models() -> None:
    incoming = torch.tensor([[-3.0, 1.0, 0.5]], dtype=torch.float64)
    racket = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    normal = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

    normal_only = shadow.moving_plane_impact_velocity(
        incoming,
        racket,
        normal,
        restitution=0.5,
        tangential_damping=0.0,
        tangential_cap=0.0,
    )
    damped = shadow.moving_plane_impact_velocity(
        incoming,
        racket,
        normal,
        restitution=0.5,
        tangential_damping=0.5,
        tangential_cap=10.0,
    )

    torch.testing.assert_close(
        normal_only,
        torch.tensor([[3.0, 1.0, 0.5]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        damped,
        torch.tensor([[3.0, 0.5, 0.25]], dtype=torch.float64),
    )


def test_rigid_contact_point_velocity_includes_angular_motion() -> None:
    point_velocity = shadow.rigid_contact_point_velocity(
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
    )
    torch.testing.assert_close(
        point_velocity,
        torch.tensor([[-1.0, 2.0, 3.0]], dtype=torch.float64),
    )


def test_measured_normal_restitution_uses_relative_surface_velocity() -> None:
    restitution, valid = shadow.measured_normal_restitution(
        torch.tensor([[-3.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert valid.item()
    torch.testing.assert_close(
        restitution,
        torch.tensor([0.25], dtype=torch.float64),
    )

    _, separating_valid = shadow.measured_normal_restitution(
        torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert not separating_valid.item()


def test_shadow_events_follow_bounce_contact_net_and_opponent_bounce() -> None:
    pre = int(shadow.PhysicalShadowPhase.INCOMING_PRE_BOUNCE)
    post = int(shadow.PhysicalShadowPhase.INCOMING_POST_BOUNCE)
    outgoing = int(shadow.PhysicalShadowPhase.OUTGOING)

    bounce = shadow.detect_table_bounce(
        torch.tensor([pre]),
        torch.tensor([[1.0, 0.0, 0.05]]),
        torch.tensor([[-3.0, 0.0, -1.0]]),
        torch.tensor([[0.9, 0.0, 0.03]]),
        torch.tensor([[-2.8, 0.0, 0.8]]),
        table_surface_z=torch.tensor([0.0]),
        ball_radius=0.02,
        table_near_x=torch.tensor([0.0]),
        net_x=torch.tensor([1.37]),
        table_y_min=torch.tensor([-1.525]),
        table_y_max=torch.tensor([0.0]),
    )
    assert bounce.item()

    outside_table = shadow.detect_table_bounce(
        torch.tensor([pre]),
        torch.tensor([[-0.10, 0.0, 0.05]]),
        torch.tensor([[-3.0, 0.0, -1.0]]),
        torch.tensor([[-0.20, 0.0, 0.03]]),
        torch.tensor([[-2.8, 0.0, 0.8]]),
        table_surface_z=torch.tensor([0.0]),
        ball_radius=0.02,
        table_near_x=torch.tensor([0.0]),
        net_x=torch.tensor([1.37]),
        table_y_min=torch.tensor([-1.525]),
        table_y_max=torch.tensor([0.0]),
    )
    assert not outside_table.item()

    contact = shadow.detect_racket_contact(
        torch.tensor([post]),
        torch.tensor([[-3.0, 0.0, 0.2]]),
        torch.tensor([[0.2, -1.0, 0.5]]),
        torch.tensor([[2.0, 0.0, 0.4]]),
        torch.tensor([[0.21, -1.0, 0.5]]),
        torch.tensor([5.0]),
    )
    assert contact.item()

    net = shadow.detect_net_cross(
        torch.tensor([outgoing]),
        torch.tensor([[1.30, -1.0, 0.30]]),
        torch.tensor([[1.42, -1.0, 0.28]]),
        net_x=torch.tensor([1.37]),
        net_top_z=torch.tensor([0.1525]),
        ball_radius=0.02,
    )
    assert net.item()

    net_collision = shadow.detect_net_cross(
        torch.tensor([outgoing]),
        torch.tensor([[1.30, -1.0, 0.14]]),
        torch.tensor([[1.42, -1.0, 0.18]]),
        net_x=torch.tensor([1.37]),
        net_top_z=torch.tensor([0.1525]),
        ball_radius=0.02,
    )
    assert not net_collision.item()

    incoming_net_collision = shadow.detect_incoming_net_collision(
        torch.tensor([pre]),
        torch.tensor([[1.45, -1.0, 0.16]]),
        torch.tensor([[-3.0, 0.0, -0.1]]),
        torch.tensor([[1.30, -1.0, 0.18]]),
        torch.tensor([[-2.8, 0.0, -0.2]]),
        net_x=torch.tensor([1.37]),
        net_top_z=torch.tensor([0.1525]),
        ball_radius=0.02,
    )
    assert incoming_net_collision.item()

    opponent = shadow.detect_opponent_bounce(
        torch.tensor([outgoing]),
        torch.tensor([True]),
        torch.tensor([[3.0, 0.0, -1.0]]),
        torch.tensor([[2.1, -1.0, 0.03]]),
        torch.tensor([[2.8, 0.0, 0.8]]),
        table_surface_z=torch.tensor([0.0]),
        ball_radius=0.02,
        net_x=torch.tensor([1.37]),
        table_far_x=torch.tensor([2.74]),
        table_y_min=torch.tensor([-1.525]),
        table_y_max=torch.tensor([0.0]),
    )
    assert opponent.item()


def test_outgoing_landing_classifies_opponent_short_long_and_side() -> None:
    outgoing = int(shadow.PhysicalShadowPhase.OUTGOING)
    phase = torch.full((4,), outgoing)
    previous_position = torch.tensor(
        [
            [2.00, -0.70, 0.06],
            [1.20, -0.70, 0.06],
            [2.80, -0.70, 0.06],
            [2.00, 0.10, 0.06],
        ]
    )
    position = previous_position.clone()
    position[:, 2] = 0.01
    previous_velocity = torch.zeros(4, 3)
    previous_velocity[:, 2] = -1.0
    velocity = previous_velocity.clone()
    landing = shadow.detect_outgoing_landing(
        phase,
        torch.tensor([True, False, True, True]),
        previous_position,
        previous_velocity,
        position,
        velocity,
        table_surface_z=torch.zeros(4),
        ball_radius=0.02,
        net_x=torch.full((4,), 1.37),
        table_far_x=torch.full((4,), 2.74),
        table_y_min=torch.full((4,), -1.525),
        table_y_max=torch.zeros(4),
    )
    assert landing.event.tolist() == [True, True, True, True]
    assert landing.opponent.tolist() == [True, False, False, False]
    assert landing.short.tolist() == [False, True, False, False]
    assert landing.long.tolist() == [False, False, True, False]
    assert landing.side.tolist() == [False, False, False, True]
    assert landing.no_net.tolist() == [False, True, False, False]
    torch.testing.assert_close(
        landing.position[:, 2],
        torch.full((4,), 0.02),
    )
