import numpy as np
import pytest
from dataclasses import replace

from scripts.replay_impact_contact_model import (
    ContactSample,
    evaluate_parameters,
    fit_parameters,
    moving_racket_impact,
    target_execution_decomposition,
)


def _sample(
    ball_in,
    racket_velocity,
    normal,
    restitution,
    tangent_retain,
) -> ContactSample:
    ball_out = moving_racket_impact(
        np.asarray(ball_in, dtype=np.float64),
        np.asarray(racket_velocity, dtype=np.float64),
        np.asarray(normal, dtype=np.float64),
        restitution=restitution,
        tangent_retain=tangent_retain,
    )
    return ContactSample(
        label="synthetic",
        trial=0,
        success=True,
        miss_reason="",
        ball_pre_pos=np.array([0.02, 0.0, 0.0]),
        ball_pre_vel=np.asarray(ball_in, dtype=np.float64),
        ball_out_vel=ball_out,
        target_pos=np.zeros(3),
        contact_pos=np.array([0.02, 0.0, 0.0]),
        racket_site_pos=np.zeros(3),
        racket_tick_vel=np.asarray(racket_velocity, dtype=np.float64),
        racket_site_vel=np.asarray(racket_velocity, dtype=np.float64),
        racket_point_vel=np.asarray(racket_velocity, dtype=np.float64),
        racket_normal=np.asarray(normal, dtype=np.float64),
        contact_normal=np.asarray(normal, dtype=np.float64),
        angular_velocity=np.zeros(3),
    )


def test_moving_racket_impact_reflects_normal_relative_velocity() -> None:
    result = moving_racket_impact(
        np.array([-2.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        restitution=0.5,
        tangent_retain=0.8,
    )

    np.testing.assert_allclose(result, [2.5, 0.0, 0.0])


def test_fit_recovers_synthetic_contact_parameters() -> None:
    expected_restitution = 0.72
    expected_retain = 0.41
    samples = [
        _sample(
            ball_in=[-2.0, 0.4 + index * 0.1, -0.2],
            racket_velocity=[1.0, -0.1, 0.3],
            normal=[1.0, 0.1, 0.2],
            restitution=expected_restitution,
            tangent_retain=expected_retain,
        )
        for index in range(6)
    ]

    fitted = fit_parameters(
        samples,
        velocity_field="racket_site_vel",
        normal_field="racket_normal",
    )
    metrics = evaluate_parameters(
        samples,
        velocity_field="racket_site_vel",
        normal_field="racket_normal",
        restitution=fitted["restitution"],
        tangent_retain=fitted["tangent_retain"],
    )

    assert fitted["restitution"] == pytest.approx(expected_restitution)
    assert fitted["tangent_retain"] == pytest.approx(expected_retain)
    assert metrics["vector_error_mean_mps"] < 1.0e-10


def test_target_execution_decomposition_closes_in_racket_plane() -> None:
    sample = replace(
        _sample(
            ball_in=[-2.0, 0.0, 0.0],
            racket_velocity=[1.0, 0.0, 0.0],
            normal=[1.0, 0.0, 0.0],
            restitution=0.5,
            tangent_retain=0.8,
        ),
        ball_pre_pos=np.array([0.02, 0.05, 0.0]),
        target_pos=np.array([0.0, 0.03, 0.0]),
        racket_site_pos=np.zeros(3),
    )
    result = target_execution_decomposition([sample])
    assert result["planner_ball_tangent_error_mean_m"] == pytest.approx(0.02)
    assert result["actor_target_tangent_error_mean_m"] == pytest.approx(0.03)
    assert result["ball_racket_tangent_error_mean_m"] == pytest.approx(0.05)
    assert result["planner_actor_cross_term_mean_m2"] == pytest.approx(0.0012)
    assert result["ball_racket_normal_offset_mean_m"] == pytest.approx(0.02)
