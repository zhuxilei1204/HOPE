"""Ball state estimation.

Fits a low-order polynomial to the most recent position samples and
differentiates it analytically to obtain a smoothed position and velocity.
The buffer is cleared on each detected table bounce so the fit never spans the
velocity discontinuity.

Before a sample reaches the fit buffer it passes an outlier gate: a frame that
implies an implausible speed relative to the last accepted sample (a
mis-tracked marker, a reflection, a stale/duplicate frame) is dropped rather
than corrupting the polynomial fit. Bounce detection is deliberately exempt --
see :meth:`BallStateEstimator._is_outlier`.

A cleared buffer would otherwise need many fresh post-bounce samples before
``ready``/``estimate()`` are usable again.  When the pre-bounce fit is trusted,
the estimator instead carries a physics-predicted post-bounce *state prior*.
The prior supplies velocity immediately, while real post-bounce samples take
over continuously as their own polynomial fit becomes observable.  Unlike the
older bridge implementation, predicted positions are never inserted into the
measurement buffer as if they were real mocap samples.
"""

from typing import List, Optional, Tuple

import numpy as np

from .constants import BallPhysics, PlannerConfig


class BallStateEstimator:
    """Estimate ball position and velocity from a stream of positions.

    Maintains a sliding window of recent position measurements and performs a
    least-squares polynomial fit to extract smoothed position and velocity.

    Bounce detection uses a three-sample z-height pattern to identify a table
    impact and clear the buffer, covering two geometries:

    * a point-ball dip that touches ``bounce_z_tol``;
    * a local z-minimum below ``bounce_center_z_max`` for centre-tracked balls,
      whose minimum height at contact is the ball radius.

    An outlier gate runs ahead of the fit buffer: a sample that is not finite,
    or that implies a frame-to-frame speed above ``config.outlier_max_speed``,
    is dropped instead of being fit. A run of ``config.outlier_max_consecutive_reject``
    rejections in a row is treated as a resynchronization instead (see
    :meth:`_is_outlier`), so a genuine large step never gets stuck rejected
    forever. Bounce detection always sees the raw stream regardless of the
    gate's verdict -- it must never be starved of the sample that signals a
    real table contact.

    ``physics`` (diagonal restitution ``C_h``/``C_v`` and ball ``radius``) is
    only needed for the post-bounce bridge described above; it defaults to
    :class:`~.constants.BallPhysics`'s nominal values so existing callers that
    construct this class with just a config are unaffected.
    """

    def __init__(self, config: PlannerConfig, physics: Optional[BallPhysics] = None):
        self.config = config
        self.physics = physics if physics is not None else BallPhysics()
        self.t_buffer: List[float] = []
        self.p_buffer: List[np.ndarray] = []

        # Three-sample z ring buffer for bounce detection; None suppresses
        # false triggers before enough measurements are collected.
        self._z_hist: List[Optional[float]] = [None, None, None]
        self._t_hist: List[Optional[float]] = [None, None, None]
        self._bounce_detected: bool = False
        self._last_bounce_t: Optional[float] = None

        # Outlier gate state, tracked independently of the fit buffer so a
        # bounce-triggered reset() does not itself relax the gate.
        self._last_accepted_t: Optional[float] = None
        self._last_accepted_p: Optional[np.ndarray] = None
        self._consecutive_outliers: int = 0
        self._outlier_rejected: bool = False

        # Temporary post-bounce state prior.  It is deliberately separate from
        # the real measurement buffers: model-generated positions must not be
        # given the same weight as mocap observations in the polynomial fit.
        self._bounce_prior_p: Optional[np.ndarray] = None
        self._bounce_prior_v: Optional[np.ndarray] = None
        self._bounce_prior_t: Optional[float] = None

    def reset(self) -> None:
        """Clear the estimation buffer (called on bounce detection)."""
        self.t_buffer.clear()
        self.p_buffer.clear()
        self._clear_bounce_prior()

    def _clear_bounce_prior(self) -> None:
        self._bounce_prior_p = None
        self._bounce_prior_v = None
        self._bounce_prior_t = None

    def reset_stream(self) -> None:
        """Clear all state after a tracking gap or a new physical ball."""
        self.reset()
        self._z_hist = [None, None, None]
        self._t_hist = [None, None, None]
        self._bounce_detected = False
        self._last_bounce_t = None
        self._last_accepted_t = None
        self._last_accepted_p = None
        self._consecutive_outliers = 0
        self._outlier_rejected = False

    def _is_outlier(self, t: float, p: np.ndarray) -> bool:
        """True if ``p`` at ``t`` is not a plausible continuation of the stream.

        A non-finite sample is always an outlier. Otherwise the check is a
        speed gate: the straight-line distance from the last *accepted*
        sample must be reachable within the elapsed time at
        ``config.outlier_max_speed``. This does not fight bounce detection --
        ball position is continuous through a bounce (only the velocity
        changes sign/magnitude), so the frame-to-frame displacement stays
        small even when the z-height pattern used for bounce detection sees a
        sharp turn.
        """
        if not np.all(np.isfinite(p)) or not np.isfinite(t):
            return True
        if self._last_accepted_p is None or self._last_accepted_t is None:
            return False
        dt = t - self._last_accepted_t
        dist = float(np.linalg.norm(p - self._last_accepted_p))
        return dist > self.config.outlier_max_speed * dt

    def push(self, t: float, p: np.ndarray) -> None:
        """Add a new position measurement.

        Parameters
        ----------
        t : float
            Timestamp in seconds (monotonic).
        p : np.ndarray, shape (3,)
            Ball position [x, y, z] in the play frame.
        """
        if not np.all(np.isfinite(p)) or not np.isfinite(t):
            # A corrupt frame must never touch bounce detection or the fit
            # buffer; there is no plausible "resync" for non-finite data.
            self._outlier_rejected = True
            return

        accept = not self._is_outlier(t, p)
        if accept:
            self._consecutive_outliers = 0
        else:
            self._consecutive_outliers += 1
            # Repeated disagreement with the stale reference means the new
            # stream is real (a rig re-lock, or a genuinely large step that
            # outran the speed gate) -- trust it and resynchronize rather than
            # freezing the fit on stale data forever.
            accept = self._consecutive_outliers >= self.config.outlier_max_consecutive_reject
            if accept:
                self._consecutive_outliers = 0
                # Discard the stale pre-gap window instead of blending it with
                # the newly-trusted position: the two are, by construction,
                # farther apart than a real flight step could cover.
                self.reset()
        self._outlier_rejected = not accept

        # Bounce detection runs on the raw stream unconditionally, independent
        # of the outlier gate's verdict on this sample.
        self._z_hist[0] = self._z_hist[1]
        self._z_hist[1] = self._z_hist[2]
        self._z_hist[2] = p[2]
        self._t_hist[0] = self._t_hist[1]
        self._t_hist[1] = self._t_hist[2]
        self._t_hist[2] = t

        self._bounce_detected = False
        z_pp, z_p, z_c = self._z_hist
        t_pp, t_p, t_c = self._t_hist
        tol = self.config.bounce_z_tol
        center_max = getattr(self.config, "bounce_center_z_max", 0.11)
        if z_pp is not None and z_p is not None and z_c is not None:
            max_gap = float(getattr(self.config, "bounce_max_sample_gap_s", 0.01))
            contiguous = (
                t_pp is not None and t_p is not None and t_c is not None
                and 0.0 < t_p - t_pp <= max_gap
                and 0.0 < t_c - t_p <= max_gap
            )
            refractory = float(getattr(self.config, "bounce_refractory_s", 0.08))
            outside_refractory = (
                self._last_bounce_t is None or t - self._last_bounce_t >= refractory
            )
            min_delta = float(getattr(self.config, "bounce_min_vertical_delta", 0.002))
            legacy_dip = z_pp > tol and z_p <= tol and z_c > tol
            center_min = (
                z_p <= center_max
                and z_pp - z_p >= min_delta
                and z_c - z_p >= min_delta
            )
            if contiguous and outside_refractory and (legacy_dip or center_min):
                self._bounce_detected = True
                self._last_bounce_t = t
                # Compute a post-bounce state prior before reset() clears the
                # pre-impact measurements.  Keep it separate from the new real
                # buffer so model positions never masquerade as observations.
                min_samples = max(6, int(getattr(self.config, "min_ready_samples", 6)))
                prior = self._state_after_bounce(t) if len(self.t_buffer) >= min_samples else None
                self.reset()
                if prior is not None:
                    self._bounce_prior_p, self._bounce_prior_v, self._bounce_prior_t = prior

        if not accept:
            return

        self._last_accepted_t = t
        self._last_accepted_p = p.copy()

        self.t_buffer.append(t)
        self.p_buffer.append(p.copy())

        if len(self.t_buffer) > self.config.fit_window:
            self.t_buffer.pop(0)
            self.p_buffer.pop(0)

        # By this point the post-bounce measurement fit has reached the normal
        # confidence threshold, so retire the model prior completely.
        min_samples = max(6, int(getattr(self.config, "min_ready_samples", 6)))
        if self._bounce_prior_t is not None and len(self.t_buffer) >= min_samples:
            self._clear_bounce_prior()

    @property
    def bounce_detected(self) -> bool:
        """True if the most recent push() detected a table bounce."""
        return self._bounce_detected

    @property
    def outlier_rejected(self) -> bool:
        """True if the most recent push() dropped its sample (outlier gate)."""
        return self._outlier_rejected

    @property
    def ready(self) -> bool:
        """True once enough samples exist for a stable fit."""
        min_samples = max(6, int(getattr(self.config, "min_ready_samples", 6)))
        return len(self.t_buffer) >= min_samples or (
            self._bounce_prior_t is not None and len(self.t_buffer) >= 1
        )

    def _fit(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Polynomial fit over the current buffer (caller must ensure `ready`)."""
        t_arr = np.array(self.t_buffer)
        p_arr = np.array(self.p_buffer)

        # Normalize time about the latest sample to improve conditioning.
        t_ref = t_arr[-1]
        t_norm = t_arr - t_ref

        p_est = np.zeros(3)
        v_est = np.zeros(3)
        for axis in range(3):
            coeffs = np.polyfit(t_norm, p_arr[:, axis], deg=self.config.poly_order)
            p_est[axis] = coeffs[-1]   # value at t_norm = 0
            v_est[axis] = coeffs[-2]   # first derivative at t_norm = 0

        return p_est, v_est, t_ref

    def _propagate_state(
        self, p0: np.ndarray, v0: np.ndarray, t0: float, t1: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate one state with the planner's drag + gravity model."""
        p = np.asarray(p0, dtype=float).copy()
        v = np.asarray(v0, dtype=float).copy()
        remaining = max(float(t1 - t0), 0.0)
        max_step = max(float(getattr(self.config, "dt_integrate", 0.005)), 1e-4)
        while remaining > 1e-12:
            dt = min(max_step, remaining)
            speed = float(np.linalg.norm(v))
            a = -self.physics.k * speed * v + self.physics.g
            p = p + v * dt + 0.5 * a * dt * dt
            v = v + a * dt
            remaining -= dt
        return p, v

    def _state_after_bounce(
        self, t_now: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Build a post-bounce state prior without fabricating measurements."""
        p_pre, v_pre, t_pre = self._fit()

        v_seed = np.array([
            self.physics.C_h * v_pre[0],
            self.physics.C_h * v_pre[1],
            -self.physics.C_v * v_pre[2],
        ])
        p_seed = p_pre.copy()
        p_seed[2] = max(p_seed[2], self.physics.radius)
        p_now, v_now = self._propagate_state(p_seed, v_seed, t_pre, t_now)
        return p_now, v_now, float(t_now)

    def estimate(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute smoothed ball position and velocity at the latest timestamp.

        Returns
        -------
        p_est : np.ndarray, shape (3,)
            Smoothed position estimate [x, y, z].
        v_est : np.ndarray, shape (3,)
            Velocity estimate [vx, vy, vz] (m/s).
        t_est : float
            Timestamp of the estimate (latest sample time).
        """
        if not self.ready:
            min_samples = max(6, int(getattr(self.config, "min_ready_samples", 6)))
            raise RuntimeError(f"Need >= {min_samples} samples, have {len(self.t_buffer)}")

        if self._bounce_prior_t is None:
            return self._fit()

        t_ref = float(self.t_buffer[-1])
        p_model, v_model = self._propagate_state(
            self._bounce_prior_p, self._bounce_prior_v,
            float(self._bounce_prior_t), t_ref,
        )
        n_real = len(self.t_buffer)
        min_samples = max(6, int(getattr(self.config, "min_ready_samples", 6)))

        # Position is directly observed and should correct the model promptly;
        # velocity needs at least six real samples before a polynomial derivative
        # is meaningful.  Thereafter, fade continuously from the impact-model
        # prior to the measurement-only fit, reaching the latter at the normal
        # readiness threshold.
        position_gain = min(0.85, 0.25 + 0.60 * n_real / min_samples)
        p_est = p_model + position_gain * (self.p_buffer[-1] - p_model)
        v_est = v_model
        if n_real >= 6:
            p_fit, v_fit, _ = self._fit()
            denom = max(min_samples - 5, 1)
            measurement_weight = float(np.clip((n_real - 5) / denom, 0.0, 1.0))
            p_est = (1.0 - measurement_weight) * p_est + measurement_weight * p_fit
            v_est = (1.0 - measurement_weight) * v_model + measurement_weight * v_fit
        return p_est, v_est, t_ref
