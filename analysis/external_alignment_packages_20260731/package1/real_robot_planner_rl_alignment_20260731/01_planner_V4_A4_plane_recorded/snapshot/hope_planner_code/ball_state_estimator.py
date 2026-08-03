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

After a bounce the estimator deliberately cold-starts from real post-impact
measurements. It does not publish a physics-propagated pre-impact prior as a
post-impact estimate: a wrong bounce time or restitution coefficient would
otherwise create confident but systematically wrong early revisions.
"""

from typing import Dict, List, Optional, Tuple

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

    ``physics`` remains accepted for API compatibility with existing callers;
    the estimator itself now uses measurements only.
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
        self._fit_residual_rms_m = np.full(3, np.nan)
        self._position_std_m = np.full(3, np.nan)
        self._velocity_std_mps = np.full(3, np.nan)
        self._fit_condition_number = np.full(3, np.nan)
        self._vertical_acceleration_mps2 = float("nan")
        self._vertical_model_error_mps2 = float("nan")

    def reset(self) -> None:
        """Clear the estimation buffer (called on bounce detection)."""
        self.t_buffer.clear()
        self.p_buffer.clear()
        self._fit_residual_rms_m.fill(np.nan)
        self._position_std_m.fill(np.nan)
        self._velocity_std_mps.fill(np.nan)
        self._fit_condition_number.fill(np.nan)
        self._vertical_acceleration_mps2 = float("nan")
        self._vertical_model_error_mps2 = float("nan")

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
                self.reset()

        if not accept:
            return

        self._last_accepted_t = t
        self._last_accepted_p = p.copy()

        self.t_buffer.append(t)
        self.p_buffer.append(p.copy())

        # The time span is primary so estimator behaviour does not silently
        # change with Motive/relay rate. The sample limit remains a hard cap.
        fit_window_s = float(getattr(self.config, "fit_window_s", 0.0))
        if fit_window_s > 0.0:
            cutoff = t - fit_window_s
            while len(self.t_buffer) > 1 and self.t_buffer[0] < cutoff:
                self.t_buffer.pop(0)
                self.p_buffer.pop(0)
        while len(self.t_buffer) > self.config.fit_window:
            self.t_buffer.pop(0)
            self.p_buffer.pop(0)

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
        return len(self.t_buffer) >= min_samples

    def _fit(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Polynomial fit over the current buffer (caller must ensure `ready`)."""
        t_arr = np.array(self.t_buffer)
        p_arr = np.array(self.p_buffer)

        # Normalize time about the latest sample to improve conditioning.
        t_ref = t_arr[-1]
        t_norm = t_arr - t_ref

        p_est = np.zeros(3)
        v_est = np.zeros(3)
        z_acceleration = float("nan")
        collect_diagnostics = bool(getattr(self.config, "collect_fit_diagnostics", False))
        for axis in range(3):
            configured = (
                getattr(self.config, "poly_order_xy", None) if axis < 2
                else getattr(self.config, "poly_order_z", None)
            )
            degree = self.config.poly_order if configured is None else int(configured)
            degree = max(1, min(degree, len(t_arr) - 1))
            coeffs = np.polyfit(t_norm, p_arr[:, axis], deg=degree)
            p_est[axis] = coeffs[-1]   # value at t_norm = 0
            v_est[axis] = coeffs[-2]   # first derivative at t_norm = 0
            if collect_diagnostics:
                design = np.vander(t_norm, N=degree + 1)
                residual = p_arr[:, axis] - design @ coeffs
                self._fit_residual_rms_m[axis] = float(np.sqrt(np.mean(residual * residual)))
                dof = len(t_arr) - (degree + 1)
                sigma2 = float(residual @ residual) / dof if dof > 0 else 0.0
                covariance = sigma2 * np.linalg.pinv(design.T @ design)
                self._position_std_m[axis] = float(np.sqrt(max(covariance[-1, -1], 0.0)))
                self._velocity_std_mps[axis] = float(np.sqrt(max(covariance[-2, -2], 0.0)))
                self._fit_condition_number[axis] = float(np.linalg.cond(design))
            if collect_diagnostics and axis == 2 and degree >= 2:
                z_acceleration = float(2.0 * coeffs[-3])

        self._vertical_acceleration_mps2 = z_acceleration
        if collect_diagnostics and np.isfinite(z_acceleration):
            speed = float(np.linalg.norm(v_est))
            model_acceleration = float(
                -self.physics.k * speed * v_est[2] + self.physics.g[2]
            )
            self._vertical_model_error_mps2 = z_acceleration - model_acceleration
        else:
            self._vertical_model_error_mps2 = float("nan")

        return p_est, v_est, t_ref

    @property
    def fit_diagnostics(self) -> Dict[str, object]:
        """Online-only fit quality indicators; no future samples are used."""
        return {
            "residual_rms_m": self._fit_residual_rms_m.copy(),
            "position_std_m": self._position_std_m.copy(),
            "velocity_std_mps": self._velocity_std_mps.copy(),
            "condition_number": self._fit_condition_number.copy(),
            "vertical_acceleration_mps2": self._vertical_acceleration_mps2,
            "vertical_model_error_mps2": self._vertical_model_error_mps2,
        }

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

        return self._fit()
