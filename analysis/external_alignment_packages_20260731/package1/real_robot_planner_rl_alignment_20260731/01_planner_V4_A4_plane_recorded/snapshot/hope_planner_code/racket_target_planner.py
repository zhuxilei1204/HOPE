"""Racket target planning.

Given the predicted ball state at the hitting plane, compute the desired racket
velocity (and face normal) to return the ball toward a fixed opponent-half
landing target, using the same no-spin drag + gravity flight model as the
trajectory predictor.

The outgoing arc is also checked against the net (``table.net_x``/``net_height``):
see :meth:`RacketTargetPlanner._solve_flight_time`, which lengthens the
post-strike flight time -- and so lofts the arc higher -- just enough to
clear the net when the nominal ``config.delta_t_flight`` would not.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .ball_trajectory_predictor import StrikeTarget
from .constants import BallPhysics, PlannerConfig, TableParams


@dataclass
class RacketCommand:
    """Desired racket state at strike time.

    The ROS message publishes only ``p_intercept`` (target position),
    ``v_racket`` (target velocity), and the derived time-to-strike; ``n_racket``
    is an orientation hint for downstream inverse kinematics (see
    :func:`quaternion_utils.normal_to_quaternion`) and is not part of the wire
    message. ``flight_time``/``net_margin`` are internal diagnostics from the
    net-clearance search (see :meth:`RacketTargetPlanner._solve_flight_time`)
    and are likewise not part of the wire message.
    """

    p_intercept: np.ndarray   # desired racket centre position at interception
    v_racket: np.ndarray      # desired racket velocity vector [vx, vy, vz]
    n_racket: np.ndarray      # desired racket face normal (unit vector)
    t_strike: float           # predicted absolute time of the strike
    num_bounces: int = 0      # table bounces predicted before the strike
    flight_time: float = 0.0  # post-strike flight time actually used (s)
    net_margin: float = 0.0   # height above the net top at the net crossing (m)


class RacketTargetPlanner:
    """Compute the desired racket velocity and face orientation for a return."""

    def __init__(self, physics: BallPhysics, config: PlannerConfig, table: TableParams):
        self.physics = physics
        self.config = config
        self.table = table

    def _integrate_free_flight(
        self, p0: np.ndarray, v0: np.ndarray, duration: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate ball free flight for ``duration`` with the drag + gravity model.

        Uses plain floats for the per-step update rather than numpy array ops
        (equivalent to ``p, v = p + v*h + 0.5*_flight_acceleration(v)*h**2, ...``
        component-wise): this is the inner loop of the Newton solve in
        :meth:`_compute_outgoing_velocity` (a handful of calls per solve, each
        stepping at ``dt_integrate`` resolution), where numpy's per-call
        dispatch overhead dominates the actual 3-element arithmetic -- measured
        ~250ms/solve with the array form vs. a few ms with this one, for the
        same result to float precision.
        """
        x, y, z = float(p0[0]), float(p0[1]), float(p0[2])
        vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])
        k = self.physics.k
        gx, gy, gz = float(self.physics.g[0]), float(self.physics.g[1]), float(self.physics.g[2])
        remaining = max(float(duration), 0.0)
        dt = self.config.dt_integrate
        while remaining > 1e-12:
            h = min(dt, remaining)
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            ax = -k * speed * vx + gx
            ay = -k * speed * vy + gy
            az = -k * speed * vz + gz
            x += vx * h + 0.5 * ax * h * h
            y += vy * h + 0.5 * ay * h * h
            z += vz * h + 0.5 * az * h * h
            vx += ax * h
            vy += ay * h
            vz += az * h
            remaining -= h
        return np.array([x, y, z]), np.array([vx, vy, vz])

    def _analytic_outgoing_velocity(
        self, p_strike: np.ndarray, p_land: np.ndarray, delta_t: float,
    ) -> np.ndarray:
        """Closed-form (drag-free) outgoing velocity for landing at ``p_land``
        after ``delta_t``. Used as the Newton seed below, and directly by the
        net-clearance search (:meth:`_solve_flight_time`), which needs many
        cheap candidate evaluations rather than one accurate one."""
        return (p_land - p_strike) / delta_t - 0.5 * self.physics.g * delta_t

    def _compute_outgoing_velocity(
        self, p_strike: np.ndarray, p_land: np.ndarray, delta_t: float,
    ) -> np.ndarray:
        """Solve the outgoing velocity that lands at ``p_land`` after ``delta_t``.

        Seeds with the drag-free analytic solution, then refines with a
        finite-difference Newton solve on the drag + gravity flight. This is
        the expensive call in this module (a handful of full trajectory
        integrations); callers that need to scan many candidate ``delta_t``
        values should use :meth:`_analytic_outgoing_velocity` instead and only
        call this once, on the chosen value.
        """
        if delta_t <= 0.0:
            raise ValueError("delta_t must be positive")

        v = self._analytic_outgoing_velocity(p_strike, p_land, delta_t)
        if self.physics.k == 0.0:
            return v

        for _ in range(12):
            p_end, _ = self._integrate_free_flight(p_strike, v, delta_t)
            residual = p_end - p_land
            if np.linalg.norm(residual) < 1e-5:
                break

            jac = np.zeros((3, 3))
            for axis in range(3):
                eps = 1e-4 * max(1.0, abs(v[axis]))
                v_eps = v.copy()
                v_eps[axis] += eps
                p_eps, _ = self._integrate_free_flight(p_strike, v_eps, delta_t)
                jac[:, axis] = (p_eps - p_end) / eps
            try:
                step = np.linalg.solve(jac, residual)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jac, residual, rcond=None)[0]
            v = v - step
            if not np.all(np.isfinite(v)):
                raise FloatingPointError("outgoing velocity solve diverged")
        return v

    def _net_margin(self, p_strike: np.ndarray, v_outgoing: np.ndarray) -> float:
        """Height above the net top when the outgoing arc crosses ``table.net_x``.

        Positive clears the net, negative would hit it. Integrates the same
        free-flight model as :meth:`_integrate_free_flight` but stops (and
        linearly interpolates) at the spatial crossing rather than a fixed
        duration -- mirroring the hitting-plane crossing in
        :class:`~.ball_trajectory_predictor.BallTrajectoryPredictor`. Returns
        ``+inf`` (i.e. "clears") if the strike point is already at/past the net
        or the arc never reaches it within a generous horizon; both mean the
        net is not this shot's problem.

        Uses plain floats rather than :meth:`_integrate_free_flight`'s
        numpy-array stepping: this is called from the net-clearance search
        with ``dt_integrate``-sized steps, and numpy's per-call dispatch
        overhead dominates at that granularity for a 3-element vector.
        """
        net_x = self.table.net_x
        x = float(p_strike[0])
        if x >= net_x:
            return float("inf")

        z = float(p_strike[2])
        vx, vy, vz = float(v_outgoing[0]), float(v_outgoing[1]), float(v_outgoing[2])
        k = self.physics.k
        gz = float(self.physics.g[2])
        dt = self.config.dt_integrate
        for _ in range(int(2.0 / dt)):
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            ax = -k * speed * vx
            ay = -k * speed * vy
            az = -k * speed * vz + gz
            x_new = x + vx * dt + 0.5 * ax * dt * dt
            z_new = z + vz * dt + 0.5 * az * dt * dt
            if x_new >= net_x:
                dx = x_new - x
                frac = (net_x - x) / dx if abs(dx) > 1e-9 else 0.0
                frac = min(1.0, max(0.0, frac))
                return z + frac * (z_new - z) - self.table.net_height
            x, z = x_new, z_new
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
        return float("inf")

    def _analytic_net_margin(self, p_strike: np.ndarray, v_outgoing: np.ndarray) -> float:
        """Closed-form (drag-free) counterpart of :meth:`_net_margin`, O(1) via
        the projectile equation instead of stepping the integrator. Used for
        the (many-candidate) search in :meth:`_solve_flight_time`; the drag
        error is small enough over one net-crossing leg that it reliably picks
        a good ``delta_t``, and the final choice is still re-checked with the
        accurate, drag-aware :meth:`_net_margin` before it is used."""
        net_x = self.table.net_x
        x0 = float(p_strike[0])
        vx = float(v_outgoing[0])
        if vx <= 1e-9 or x0 >= net_x:
            return float("inf")
        t_net = (net_x - x0) / vx
        z0 = float(p_strike[2])
        vz = float(v_outgoing[2])
        gz = float(self.physics.g[2])
        z_net = z0 + vz * t_net + 0.5 * gz * t_net * t_net
        return z_net - self.table.net_height

    def _solve_flight_time(
        self, p_strike: np.ndarray, p_land: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Pick a post-strike flight time and matching outgoing velocity.

        Starts from ``config.delta_t_flight`` (today's fixed behaviour) and, if
        that arc would not clear the net by ``config.net_clearance_margin``,
        bisects *upward* for the smallest flight time (closest to nominal)
        within ``[config.delta_t_flight, config.delta_t_flight_max]`` that
        does -- a longer flight time lofts the same start->land arc higher at
        the net (more hang time), and as a bonus asks less peak speed of the
        racket than a fast, flat drive would. If even the ceiling cannot
        clear, it is used anyway (best available margin) -- consistent with
        the planner never producing a null/failure result.

        The search itself only evaluates the cheap closed-form
        :meth:`_analytic_outgoing_velocity`/:meth:`_analytic_net_margin` (no
        Newton solve, no stepped integration), so scanning many candidates
        stays cheap; :meth:`_compute_outgoing_velocity` (the expensive,
        drag-refined solve) is called exactly once, on the finally-chosen
        ``delta_t`` -- this keeps the net-clearance search roughly as cheap as
        today's single fixed-``delta_t_flight`` solve, rather than
        ``net_search_iters`` times as expensive.
        """
        nominal = self.config.delta_t_flight
        margin_min = self.config.net_clearance_margin

        chosen_dt = nominal
        nominal_margin = self._analytic_net_margin(
            p_strike, self._analytic_outgoing_velocity(p_strike, p_land, nominal)
        )
        if nominal_margin < margin_min:
            lo, hi = nominal, self.config.delta_t_flight_max
            hi_margin = self._analytic_net_margin(
                p_strike, self._analytic_outgoing_velocity(p_strike, p_land, hi)
            )
            chosen_dt = hi  # best available if even the ceiling cannot clear
            if hi_margin >= margin_min:
                for _ in range(self.config.net_search_iters):
                    mid = 0.5 * (lo + hi)
                    v_mid = self._analytic_outgoing_velocity(p_strike, p_land, mid)
                    if self._analytic_net_margin(p_strike, v_mid) >= margin_min:
                        chosen_dt = mid
                        hi = mid
                    else:
                        lo = mid

        v_outgoing = self._compute_outgoing_velocity(p_strike, p_land, chosen_dt)
        return chosen_dt, v_outgoing

    @staticmethod
    def _opponent_facing_normal(candidate: np.ndarray) -> np.ndarray:
        """Return a unit racket normal with a positive forward (+x) component."""
        n = np.asarray(candidate, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-9 or not np.isfinite(norm):
            return np.array([1.0, 0.0, 0.0])

        n = n / norm
        if n[0] < 0.0:
            n = -n
        if n[0] <= 1e-6:
            n = n + np.array([1.0, 0.0, 0.0])
            n = n / np.linalg.norm(n)
        return n

    def _compute_racket_velocity(
        self, v_incoming: np.ndarray, v_outgoing: np.ndarray, C_r: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Desired racket velocity and face normal from the impact model.

        Along the face normal n, restitution gives
        v_out_n - v_r_n = -C_r (v_in_n - v_r_n), so the required racket normal
        speed is v_r_n = (v_out_n + C_r v_in_n) / (1 + C_r). The racket velocity
        is taken along n (no tangential drive in this simplified model).
        """
        delta_v = v_outgoing - v_incoming
        if np.linalg.norm(delta_v) < 1e-6:
            return np.zeros(3), np.array([1.0, 0.0, 0.0])

        n = self._opponent_facing_normal(delta_v)
        v_o_n = np.dot(v_outgoing, n)
        v_i_n = np.dot(v_incoming, n)
        v_r_n = (v_o_n + C_r * v_i_n) / (1.0 + C_r)
        return v_r_n * n, n

    def plan(self, strike: StrikeTarget) -> RacketCommand:
        """Compute the racket target for a valid predicted strike.

        Callers pass only valid strikes (see :class:`HOPEPlanner`).
        """
        p_strike = strike.p_ball
        v_incoming = strike.v_ball
        p_land = self.config.target_land.copy()

        flight_time, v_outgoing = self._solve_flight_time(p_strike, p_land)
        v_racket, n_racket = self._compute_racket_velocity(
            v_incoming, v_outgoing, self.config.C_r
        )
        return RacketCommand(
            p_intercept=p_strike, v_racket=v_racket, n_racket=n_racket,
            t_strike=strike.t_strike, num_bounces=strike.num_bounces,
            flight_time=flight_time, net_margin=self._net_margin(p_strike, v_outgoing),
        )
