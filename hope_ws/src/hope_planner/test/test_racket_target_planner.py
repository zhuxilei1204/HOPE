"""Racket target planner unit tests (constant paddle restitution, no-spin)."""

import numpy as np

from hope_planner.ball_trajectory_predictor import StrikeTarget
from hope_planner.constants import BallPhysics, PlannerConfig, TableParams
from hope_planner.racket_target_planner import RacketTargetPlanner


def _planner():
    return RacketTargetPlanner(BallPhysics(), PlannerConfig(), TableParams())


def _incoming_strike():
    return StrikeTarget(
        p_ball=np.array([0.0, -0.7625, 0.3]),
        v_ball=np.array([-3.0, 0.0, -0.5]),
        t_strike=0.4, num_bounces=1, valid=True,
    )


def test_incoming_ball_produces_finite_command():
    cmd = _planner().plan(_incoming_strike())
    assert cmd.num_bounces == 1
    assert np.all(np.isfinite(cmd.v_racket))
    assert np.all(np.isfinite(cmd.p_intercept))


def test_normal_vector_is_unit_length():
    cmd = _planner().plan(_incoming_strike())
    assert np.isclose(np.linalg.norm(cmd.n_racket), 1.0, atol=1e-9)


def test_normal_vector_faces_opponent_side():
    cmd = _planner().plan(_incoming_strike())
    assert np.dot(cmd.n_racket, np.array([1.0, 0.0, 0.0])) > 0.0


def test_degenerate_sideways_and_reversed_normals_face_opponent():
    pl = _planner()
    cases = [
        (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),   # delta_v ~= 0
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # pure sideways delta_v
        (np.array([3.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),  # raw delta_v points -x
    ]
    for v_in, v_out in cases:
        _, n = pl._compute_racket_velocity(v_in, v_out, pl.config.C_r)
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-9)
        assert n[0] > 0.0


def test_outgoing_velocity_with_drag_lands_near_target():
    pl = _planner()
    p_strike = np.array([0.0, -0.7625, 0.3])
    p_land = np.array([2.055, -0.7625, 0.0])
    dt = 0.55
    v_out = pl._compute_outgoing_velocity(p_strike, p_land, dt)
    p_end, _ = pl._integrate_free_flight(p_strike, v_out, dt)
    assert np.allclose(p_end, p_land, atol=5e-4)


def test_net_margin_matches_analytic_estimate_without_drag():
    """With zero drag, the stepped (accurate) and closed-form (analytic) net
    margins should agree almost exactly -- they integrate the same physics."""
    pl = RacketTargetPlanner(BallPhysics(k=0.0), PlannerConfig(), TableParams())
    p_strike = np.array([0.0, -0.7625, 0.3])
    v_outgoing = np.array([4.0, 0.0, 1.5])
    accurate = pl._net_margin(p_strike, v_outgoing)
    analytic = pl._analytic_net_margin(p_strike, v_outgoing)
    assert np.isclose(accurate, analytic, atol=1e-3)


def test_net_margin_increases_with_flight_time():
    """For a fixed p_strike -> p_land pair, a longer flight time lofts the arc
    higher at the net (more hang time), so margin must increase with
    delta_t -- this is the direction the search in _solve_flight_time relies
    on; if it ever regresses, the search would extend flight time in the
    wrong direction and make net clearance worse, not better."""
    pl = _planner()
    p_strike = np.array([0.0, -0.7625, 0.3])
    p_land = np.array([2.055, -0.7625, 0.02])
    margins = []
    for dt in [0.2, 0.35, 0.5, 0.65, 0.8]:
        v = pl._compute_outgoing_velocity(p_strike, p_land, dt)
        margins.append(pl._net_margin(p_strike, v))
    assert all(b > a for a, b in zip(margins, margins[1:]))


def test_plan_extends_flight_time_when_nominal_would_clip_the_net():
    """A short nominal delta_t_flight forces a flat, low arc that clips the
    net; plan() must lengthen the flight time (not shorten it) until the
    configured clearance is met."""
    config = PlannerConfig(delta_t_flight=0.2, delta_t_flight_max=1.0, net_clearance_margin=0.03)
    pl = RacketTargetPlanner(BallPhysics(), config, TableParams())
    cmd = pl.plan(_incoming_strike())
    assert cmd.flight_time > config.delta_t_flight
    assert cmd.net_margin >= config.net_clearance_margin - 1e-6
    assert np.all(np.isfinite(cmd.v_racket))


def test_plan_leaves_flight_time_unchanged_when_nominal_already_clears():
    """The common case (nominal already clears comfortably) must be
    untouched: no search, exact nominal flight time, unchanged behaviour."""
    pl = _planner()
    cmd = pl.plan(_incoming_strike())
    assert cmd.flight_time == pl.config.delta_t_flight
    assert cmd.net_margin >= pl.config.net_clearance_margin


def test_plan_falls_back_to_best_effort_when_ceiling_cannot_clear():
    """Even when delta_t_flight_max is too tight to fully satisfy
    net_clearance_margin, plan() must still return a finite command (the
    best-available margin) rather than raising or stalling."""
    config = PlannerConfig(delta_t_flight=0.1, delta_t_flight_max=0.15, net_clearance_margin=0.03)
    pl = RacketTargetPlanner(BallPhysics(), config, TableParams())
    strike = StrikeTarget(
        p_ball=np.array([0.0, -0.7625, 0.05]),
        v_ball=np.array([-3.0, 0.0, -0.5]),
        t_strike=0.4, num_bounces=1, valid=True,
    )
    cmd = pl.plan(strike)
    assert cmd.flight_time <= config.delta_t_flight_max + 1e-9
    assert np.all(np.isfinite(cmd.v_racket))
    assert np.all(np.isfinite(cmd.p_intercept))


def test_constant_restitution_is_self_consistent():
    """The commanded racket normal speed must satisfy the restitution identity
    v_o_n - v_r_n = -C_r (v_i_n - v_r_n)."""
    pl = _planner()
    v_in = np.array([-6.0, 0.3, -1.0])
    v_out = np.array([4.0, -0.2, 2.5])
    v_r, n = pl._compute_racket_velocity(v_in, v_out, pl.config.C_r)
    v_r_n = float(np.dot(v_r, n))
    v_i_n = float(np.dot(v_in, n))
    v_o_n = float(np.dot(v_out, n))
    assert abs((v_o_n - v_r_n) + pl.config.C_r * (v_i_n - v_r_n)) < 1e-9
