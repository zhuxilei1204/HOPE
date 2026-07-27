"""Ball state estimator unit tests."""

import numpy as np

from hope_planner.ball_state_estimator import BallStateEstimator
from hope_planner.constants import BallPhysics, PlannerConfig


def _dt():
    return 1.0 / 300.0


def test_constant_velocity_returns_correct_velocity():
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    v_true = np.array([1.0, -0.5, 0.0])
    p0 = np.array([0.2, -0.3, 0.5])  # z constant > bounce tol -> no false bounce
    dt = _dt()
    for i in range(20):
        est.push(i * dt, p0 + v_true * (i * dt))
    assert est.ready
    _, v_est, _ = est.estimate()
    assert np.allclose(v_est, v_true, atol=1e-3)


def test_parabolic_z_returns_correct_vertical_velocity():
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = _dt()
    z0, vz0, g = 1.0, 0.0, -9.81
    for i in range(20):
        t = i * dt
        p = np.array([0.0, 0.0, z0 + vz0 * t + 0.5 * g * t * t])
        est.push(t, p)
    _, v_est, t_ref = est.estimate()
    expected_vz = vz0 + g * t_ref  # derivative at the latest (reference) sample
    assert abs(v_est[2] - expected_vz) < 1e-3


def test_bounce_pattern_clears_buffer():
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = _dt()
    zs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.002, 0.1]  # descend -> contact -> rise
    for i, z in enumerate(zs):
        est.push(i * dt, np.array([0.0, 0.0, z]))
    assert est.bounce_detected
    # reset() runs before the rising sample is appended -> only that sample remains
    assert len(est.t_buffer) == 1
    assert not est.ready


def test_center_geometry_bounce_clears_buffer():
    """A centre-tracked ball's minimum height at contact is the ball radius, which
    the point-ball dip condition never sees. A physically consistent 300 Hz
    centre-height bounce (descent to z = radius, restitution rebound) must clear
    the buffer via the local-minimum detector."""
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = 1.0 / 300.0
    radius, vz_in, e_n = 0.02, 3.0, 0.92
    zs_down = [radius + vz_in * dt * k for k in (4, 3, 2, 1)]      # 0.06 -> 0.03
    zs_up = [radius + e_n * vz_in * dt * k for k in (1, 2, 3, 4)]  # rebound
    zs = zs_down + [radius] + zs_up                                # local min at z = 0.02
    detected = False
    for i, z in enumerate(zs):
        est.push(i * dt, np.array([0.0, 0.0, z]))
        detected = detected or est.bounce_detected
    assert detected
    assert len(est.t_buffer) <= len(zs_up) + 1


def test_no_false_bounce_on_low_flat_or_monotonic_flight():
    """Low but flat / monotonically descending tracks (no local minimum) must NOT clear."""
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = 1.0 / 300.0
    for i in range(12):  # flat inside the centre band
        est.push(i * dt, np.array([0.0, 0.0, 0.04]))
        assert not est.bounce_detected
    est2 = BallStateEstimator(cfg)
    for i, z in enumerate([0.08, 0.06, 0.045, 0.035, 0.028, 0.024]):  # still descending
        est2.push(i * dt, np.array([0.0, 0.0, z]))
        assert not est2.bounce_detected
    assert len(est.t_buffer) == 12 and len(est2.t_buffer) == 6


def test_low_amplitude_mocap_jitter_does_not_repeatedly_reset():
    cfg = PlannerConfig(
        min_ready_samples=20,
        bounce_min_vertical_delta=0.002,
    )
    est = BallStateEstimator(cfg)
    dt = _dt()
    # Alternating 1 mm local minima are below the deployed 2 mm evidence
    # threshold even though every other frame satisfies the old detector.
    for i in range(30):
        z = 0.04 + (0.0005 if i % 2 == 0 else -0.0005)
        est.push(i * dt, np.array([0.2 - i * dt, 0.0, z]))
        assert not est.bounce_detected
    assert est.ready
    assert len(est.t_buffer) == 30


def test_bounce_is_not_inferred_across_tracking_gap():
    cfg = PlannerConfig(bounce_max_sample_gap_s=0.01)
    est = BallStateEstimator(cfg)
    est.push(0.000, np.array([0.0, 0.0, 0.04]))
    est.push(0.003, np.array([0.0, 0.0, 0.02]))
    est.push(0.033, np.array([0.0, 0.0, 0.04]))
    assert not est.bounce_detected


def test_fewer_than_six_samples_not_ready():
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = _dt()
    for i in range(5):
        est.push(i * dt, np.array([0.0, 0.0, 0.5]))
    assert not est.ready


def test_outlier_sample_is_rejected_and_does_not_corrupt_the_fit():
    """A single implausible frame (mis-tracked marker / reflection) must be
    dropped from the fit buffer, leaving the velocity estimate exactly as if
    the outlier had never arrived."""
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    v_true = np.array([1.0, -0.5, 0.0])
    p0 = np.array([0.2, -0.3, 0.5])
    dt = _dt()
    for i in range(6):
        est.push(i * dt, p0 + v_true * (i * dt))
    assert est.ready
    n_before = len(est.t_buffer)

    est.push(6 * dt, np.array([5.0, 5.0, 5.0]))  # teleport: far beyond any plausible step
    assert est.outlier_rejected
    assert len(est.t_buffer) == n_before  # dropped, not appended

    # The stream continues normally; the estimate must be unaffected by the outlier.
    for i in range(7, 13):
        est.push(i * dt, p0 + v_true * (i * dt))
        assert not est.outlier_rejected
    _, v_est, _ = est.estimate()
    assert np.allclose(v_est, v_true, atol=1e-3)


def test_nonfinite_sample_is_rejected():
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = _dt()
    for i in range(6):
        est.push(i * dt, np.array([0.0, 0.0, 0.5]))
    n_before = len(est.t_buffer)

    est.push(6 * dt, np.array([np.nan, 0.0, 0.5]))
    assert est.outlier_rejected
    assert len(est.t_buffer) == n_before


def test_persistent_disagreement_resynchronizes_instead_of_freezing():
    """A run of `outlier_max_consecutive_reject` mutually-consistent samples that
    disagree with stale history (rig re-lock, or a genuinely large step) must
    eventually be trusted, discarding the stale pre-gap window rather than
    rejecting forever."""
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = _dt()
    for i in range(3):
        est.push(i * dt, np.array([0.0, 0.0, 0.5]))

    new_p = np.array([2.0, 0.0, 0.5])  # 2 m away: far beyond the speed gate
    for k in range(cfg.outlier_max_consecutive_reject - 1):
        est.push((3 + k) * dt, new_p)
        assert est.outlier_rejected

    est.push((3 + cfg.outlier_max_consecutive_reject - 1) * dt, new_p)
    assert not est.outlier_rejected
    assert len(est.t_buffer) == 1  # stale window discarded, fresh start at new_p


def test_bounce_detection_is_unaffected_by_a_rejected_sample():
    """The outlier gate must never starve bounce detection of the dip sample:
    the legacy point-ball pattern still fires even though the dip itself is far
    enough from the pre-bounce trend to fail the speed gate on its own."""
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg)
    dt = _dt()
    zs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.002, 0.1]
    for i, z in enumerate(zs):
        est.push(i * dt, np.array([0.0, 0.0, z]))
    assert est.bounce_detected
    assert len(est.t_buffer) == 1


def test_bridge_after_bounce_gives_immediate_post_bounce_estimate():
    """A bounce landing close to the robot must not cost ~6 real post-bounce
    samples (~20 ms at 300 Hz) of estimator downtime: if the pre-bounce buffer
    was `ready`, `ready`/`estimate()` must be usable again on the very same
    push that detects the bounce, with a velocity close to the analytically
    correct (reflected) post-bounce value -- not merely "some value"."""
    cfg = PlannerConfig()
    physics = BallPhysics()
    est = BallStateEstimator(cfg, physics)
    dt = _dt()
    radius = physics.radius
    vx_in, vz_in = -2.0, 3.0

    def x_of(t):
        return vx_in * t

    # Smooth, physically-plausible constant-rate descent (small per-step
    # displacement, unlike the toy dip in test_bounce_pattern_clears_buffer)
    # with enough samples that the pre-bounce buffer is `ready` (>= 6) when
    # the bounce fires -- this is what makes the bridge eligible to engage.
    n_pre = 8
    for k in range(n_pre, 0, -1):
        t = (n_pre - k) * dt
        est.push(t, np.array([x_of(t), 0.0, radius + vz_in * dt * k]))
    assert est.ready

    t_dip = n_pre * dt
    est.push(t_dip, np.array([x_of(t_dip), 0.0, radius]))
    assert not est.bounce_detected  # pattern needs one more, rising sample

    # Post-bounce, x-velocity is also reflected (scaled by C_h) -- a real
    # measurement stream would show this kink at the bounce too.
    t_rise = t_dip + dt
    x_rise = x_of(t_dip) + physics.C_h * vx_in * dt
    p_rise = np.array([x_rise, 0.0, radius + physics.C_v * vz_in * dt])
    est.push(t_rise, p_rise)

    assert est.bounce_detected
    assert est.ready  # <- the fix: usable immediately, not ~6 real samples later

    v_post_true = np.array([physics.C_h * vx_in, 0.0, physics.C_v * vz_in])
    _, v_est, _ = est.estimate()
    assert np.allclose(v_est, v_post_true, atol=0.15)


def test_high_confidence_bounce_prior_does_not_pollute_real_sample_buffer():
    """With the deployed 20-sample confidence gate, a trusted pre-impact
    state must remain usable immediately after reset, but model-generated
    samples must never be inserted into the post-impact measurement buffer."""
    cfg = PlannerConfig(min_ready_samples=20, fit_window=67)
    physics = BallPhysics()
    est = BallStateEstimator(cfg, physics)
    dt = _dt()
    radius = physics.radius
    vx_in, vz_in = -2.0, 3.0
    n_pre = 24

    for k in range(n_pre, 0, -1):
        t = (n_pre - k) * dt
        est.push(t, np.array([vx_in * t, 0.0, radius + vz_in * dt * k]))
    assert est.ready

    t_dip = n_pre * dt
    x_dip = vx_in * t_dip
    est.push(t_dip, np.array([x_dip, 0.0, radius]))
    t_rise = t_dip + dt
    p_rise = np.array([
        x_dip + physics.C_h * vx_in * dt,
        0.0,
        radius + physics.C_v * vz_in * dt,
    ])
    est.push(t_rise, p_rise)

    assert est.bounce_detected
    assert est.ready
    assert len(est.t_buffer) == 1
    assert np.allclose(est.p_buffer[0], p_rise)

    v_post_true = np.array([physics.C_h * vx_in, 0.0, physics.C_v * vz_in])
    _, v_est, _ = est.estimate()
    assert np.allclose(v_est, v_post_true, atol=0.15)


def test_bridge_does_not_engage_when_pre_bounce_buffer_was_not_ready():
    """The bridge must not fabricate a velocity from an unreliable pre-bounce
    fit: with < 6 pre-bounce samples (as in the existing toy dip tests above),
    the estimator must fall back to the old cold-start behaviour."""
    cfg = PlannerConfig()
    est = BallStateEstimator(cfg, BallPhysics())
    dt = _dt()
    zs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.002, 0.1]
    for i, z in enumerate(zs):
        est.push(i * dt, np.array([0.0, 0.0, z]))
    assert est.bounce_detected
    assert not est.ready
    assert len(est.t_buffer) == 1


def test_min_ready_samples_delays_ready_without_changing_fit_window():
    cfg = PlannerConfig(min_ready_samples=12, fit_window=67)
    est = BallStateEstimator(cfg)
    dt = _dt()
    v = np.array([1.0, 0.0, 0.0])  # plausible speed for the outlier gate
    p0 = np.array([0.0, 0.0, 0.5])
    for i in range(11):
        est.push(i * dt, p0 + v * (i * dt))
    assert not est.ready
    est.push(11 * dt, p0 + v * (11 * dt))
    assert est.ready


def test_reset_stream_removes_old_ball_and_outlier_reference():
    cfg = PlannerConfig(min_ready_samples=6)
    est = BallStateEstimator(cfg)
    dt = _dt()
    for i in range(8):
        est.push(i * dt, np.array([0.2 + i * dt, 0.0, 0.5]))
    assert est.ready

    est.reset_stream()
    assert not est.ready
    assert len(est.t_buffer) == 0
    assert not est.outlier_rejected

    # A distant first sample from the next physical ball is a new reference,
    # not an implausible jump from the old ball.
    est.push(5.0, np.array([2.0, -1.0, 1.2]))
    assert len(est.t_buffer) == 1
    assert not est.outlier_rejected
