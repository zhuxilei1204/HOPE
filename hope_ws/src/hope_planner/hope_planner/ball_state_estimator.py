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

A cleared buffer would otherwise need >= 6 fresh post-bounce samples (~20 ms
at 300 Hz) before ``ready``/``estimate()`` are usable again, purely from
data -- a real cost against ``time_to_strike`` for a bounce that lands close
to the robot. If the pre-bounce buffer was itself ready, the clear is
immediately followed by a physics-predicted bridge (see
:meth:`BallStateEstimator._bridge_after_bounce`): the pre-bounce velocity
reflected through the same diagonal restitution model
:class:`~.ball_trajectory_predictor.BallTrajectoryPredictor` uses, then
forward-integrated a few synthetic samples ahead. ``ready`` therefore becomes
true again on the very same push that detects the bounce; real samples
afterward simply age the synthetic ones out of the fit window like any other
sample, so the bridge only ever covers the gap, never permanently biases the
fit.
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
        self._bounce_detected: bool = False

        # Outlier gate state, tracked independently of the fit buffer so a
        # bounce-triggered reset() does not itself relax the gate.
        self._last_accepted_t: Optional[float] = None
        self._last_accepted_p: Optional[np.ndarray] = None
        self._consecutive_outliers: int = 0
        self._outlier_rejected: bool = False

    def reset(self) -> None:
        """Clear the estimation buffer (called on bounce detection)."""
        self.t_buffer.clear()
        self.p_buffer.clear()

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

        self._bounce_detected = False
        z_pp, z_p, z_c = self._z_hist
        tol = self.config.bounce_z_tol
        center_max = getattr(self.config, "bounce_center_z_max", 0.11)
        if z_pp is not None and z_p is not None and z_c is not None:
            legacy_dip = z_pp > tol and z_p <= tol and z_c > tol
            center_min = z_p <= center_max and z_pp > z_p and z_c > z_p
            if legacy_dip or center_min:
                self._bounce_detected = True
                # Compute the bridge from the pre-bounce buffer before reset()
                # clears it; ready() below requires >= 6, matching the fit's
                # own reliability bar for a pre-bounce velocity worth reflecting.
                bridge = self._bridge_after_bounce(t) if len(self.t_buffer) >= 6 else []
                self.reset()
                for t_b, p_b in bridge:
                    self.t_buffer.append(t_b)
                    self.p_buffer.append(p_b)
                if bridge:
                    self._last_accepted_t, self._last_accepted_p = bridge[-1]

        if not accept:
            return

        self._last_accepted_t = t
        self._last_accepted_p = p.copy()

        self.t_buffer.append(t)
        self.p_buffer.append(p.copy())

        if len(self.t_buffer) > self.config.fit_window:
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
        return len(self.t_buffer) >= 6

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

    def _bridge_after_bounce(self, t_now: float) -> List[Tuple[float, np.ndarray]]:
        """Physics-predicted samples bridging a just-detected bounce.

        Fits the (still pre-clear) buffer for the pre-bounce state, reflects
        its velocity through the same diagonal restitution model
        :class:`~.ball_trajectory_predictor.BallTrajectoryPredictor` uses
        (``v+ = diag(C_h, C_h, -C_v) @ v-``), then forward-integrates the same
        drag + gravity flight model a few synthetic samples ahead, evenly
        spaced up to (but not including) ``t_now``. Caller is responsible for
        only calling this while the pre-bounce buffer is still intact (i.e.
        before :meth:`reset`) and only when it is `ready`.
        """
        p_pre, v_pre, t_pre = self._fit()

        v_seed = np.array([
            self.physics.C_h * v_pre[0],
            self.physics.C_h * v_pre[1],
            -self.physics.C_v * v_pre[2],
        ])
        p_seed = p_pre.copy()
        p_seed[2] = max(p_seed[2], self.physics.radius)

        n_bridge = 5  # a handful of samples; real ones age these out fast
        dt = max(t_now - t_pre, 1e-4) / (n_bridge + 1)

        samples: List[Tuple[float, np.ndarray]] = []
        p, v, t = p_seed, v_seed, t_pre
        for _ in range(n_bridge):
            speed = float(np.linalg.norm(v))
            a = -self.physics.k * speed * v + self.physics.g
            p = p + v * dt + 0.5 * a * dt * dt
            v = v + a * dt
            t = t + dt
            samples.append((t, p.copy()))
        return samples

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
            raise RuntimeError(f"Need >= 6 samples, have {len(self.t_buffer)}")
        return self._fit()
