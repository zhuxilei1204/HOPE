"""Ball trajectory prediction (no spin).

Forward-integrates the ball trajectory with explicit Euler using a simple
flight model (quadratic drag + gravity) and a diagonal table-bounce model, and
returns the predicted ball state at the fixed virtual hitting plane.

Table contact follows the shared ``configs/ball_physics.yaml`` convention: the
ball CENTROID contacts the surface at z = ball radius (0.02 m for a 40 mm ball),
and the bounce event is interpolated to that plane within the crossing step.
"""

import math
from dataclasses import dataclass

import numpy as np

from .constants import BallPhysics, PlannerConfig, TableParams


@dataclass
class StrikeTarget:
    """Predicted ball state at the hitting plane."""

    p_ball: np.ndarray        # predicted ball position at strike [x, y, z]
    v_ball: np.ndarray        # predicted ball velocity at strike [vx, vy, vz]
    t_strike: float           # absolute time of the strike
    num_bounces: int          # number of table bounces before the strike
    valid: bool               # True if a usable plane crossing was found


class BallTrajectoryPredictor:
    """Forward-integrate the ball trajectory and find the hitting-plane crossing."""

    def __init__(self, physics: BallPhysics, config: PlannerConfig, table: TableParams):
        self.physics = physics
        self.config = config
        self.table = table

    def _is_on_table(self, p: np.ndarray) -> bool:
        """True if the ball could contact the table surface (bounds + ball radius)."""
        return self._is_on_table_xy(float(p[0]), float(p[1]))

    def _is_on_table_xy(self, x: float, y: float) -> bool:
        r = self.physics.radius
        y_hi = self.table.y_max
        return (
            -r <= x <= self.table.length + r
            and y_hi - self.table.width - r <= y <= y_hi + r
        )

    def _apply_bounce(self, v: np.ndarray) -> np.ndarray:
        """Diagonal table bounce: v+ = diag(C_h, C_h, -C_v) @ v-."""
        C = np.diag([self.physics.C_h, self.physics.C_h, -self.physics.C_v])
        return C @ v

    def predict(self, p0: np.ndarray, v0: np.ndarray, t0: float) -> StrikeTarget:
        """Forward-integrate from (p0, v0, t0) and find the hitting-plane crossing.

        Returns a :class:`StrikeTarget`. ``valid`` is False if the ball never
        crosses ``config.x_hit`` while moving toward the robot within the
        prediction horizon, or if the only crossing is a dead ball skimming the
        table on the way down.

        Uses plain floats for the per-step state rather than numpy arrays: this
        loop runs on every incoming mocap frame (up to ``max_predict_time /
        dt_integrate`` steps, e.g. 2000 at the defaults) and numpy's per-call
        dispatch overhead dominates the actual 3-element arithmetic at that
        granularity -- the same fix applied to
        :meth:`RacketTargetPlanner._integrate_free_flight`.
        """
        dt = self.config.dt_integrate
        max_steps = int(self.config.max_predict_time / dt)
        x_hit = self.config.x_hit
        # Contact plane for the ball CENTROID: the ball touches the table when its
        # centre reaches z = ball radius (configs/ball_physics.yaml convention),
        # not when the centre reaches the table surface z = 0.
        contact_z = self.physics.radius
        k = self.physics.k
        gx, gy, gz = float(self.physics.g[0]), float(self.physics.g[1]), float(self.physics.g[2])
        C_h, C_v = self.physics.C_h, self.physics.C_v

        x, y, z = float(p0[0]), float(p0[1]), float(p0[2])
        vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])
        t = t0
        bounces = 0

        # Track the most recent bounce so a plane crossing that happens in the
        # same step as a bounce interpolates along the post-bounce arc.
        pbx, pby, pbz = x, y, z
        vpx, vpy, vpz = vx, vy, vz
        remaining_dt = dt

        for _step in range(max_steps):
            p_prev_x = x

            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            ax = -k * speed * vx + gx
            ay = -k * speed * vy + gy
            az = -k * speed * vz + gz
            vx_new = vx + ax * dt
            vy_new = vy + ay * dt
            vz_new = vz + az * dt
            x_new = x + vx * dt + 0.5 * ax * dt * dt
            y_new = y + vy * dt + 0.5 * ay * dt * dt
            z_new = z + vz * dt + 0.5 * az * dt * dt
            t += dt
            bounce_this_step = False

            # --- Bounce detection (centroid contact at z = ball radius, interpolated) ---
            if z_new < contact_z and vz_new < 0.0:
                if self._is_on_table_xy(x_new, y_new):
                    dz = z - z_new
                    frac = (z - contact_z) / dz if dz > 1e-9 else 0.5
                    frac = min(1.0, max(0.0, frac))

                    pbx = x + frac * (x_new - x)
                    pby = y + frac * (y_new - y)
                    pbz = contact_z
                    vabx = vx + ax * (frac * dt)
                    vaby = vy + ay * (frac * dt)
                    vabz = vz + az * (frac * dt)
                    vpx = C_h * vabx
                    vpy = C_h * vaby
                    vpz = -C_v * vabz

                    remaining_dt = (1.0 - frac) * dt
                    speed_post = math.sqrt(vpx * vpx + vpy * vpy + vpz * vpz)
                    axp = -k * speed_post * vpx + gx
                    ayp = -k * speed_post * vpy + gy
                    azp = -k * speed_post * vpz + gz
                    x_new = pbx + vpx * remaining_dt + 0.5 * axp * remaining_dt * remaining_dt
                    y_new = pby + vpy * remaining_dt + 0.5 * ayp * remaining_dt * remaining_dt
                    z_new = pbz + vpz * remaining_dt + 0.5 * azp * remaining_dt * remaining_dt
                    vx_new = vpx + axp * remaining_dt
                    vy_new = vpy + ayp * remaining_dt
                    vz_new = vpz + azp * remaining_dt
                    bounces += 1
                    bounce_this_step = True
                else:
                    z_new = max(z_new, contact_z)

            # --- Hitting-plane crossing detection ---
            if p_prev_x > x_hit and x_new <= x_hit and vx_new < 0:
                if bounce_this_step:
                    dx_arc = pbx - x_new
                    frac_cross = (pbx - x_hit) / dx_arc if abs(dx_arc) > 1e-9 else 0.5
                    frac_cross = min(1.0, max(0.0, frac_cross))
                    px_cross = pbx + frac_cross * (x_new - pbx)
                    py_cross = pby + frac_cross * (y_new - pby)
                    pz_cross = pbz + frac_cross * (z_new - pbz)
                    vx_cross = vpx + frac_cross * (vx_new - vpx)
                    vy_cross = vpy + frac_cross * (vy_new - vpy)
                    vz_cross = vpz + frac_cross * (vz_new - vpz)
                    t_cross = (t - remaining_dt) + frac_cross * remaining_dt
                else:
                    dx_step = x - x_new
                    frac_cross = (x - x_hit) / dx_step if abs(dx_step) > 1e-9 else 0.5
                    frac_cross = min(1.0, max(0.0, frac_cross))
                    px_cross = x + frac_cross * (x_new - x)
                    py_cross = y + frac_cross * (y_new - y)
                    pz_cross = z + frac_cross * (z_new - z)
                    vx_cross = vx + frac_cross * (vx_new - vx)
                    vy_cross = vy + frac_cross * (vy_new - vy)
                    vz_cross = vz + frac_cross * (vz_new - vz)
                    t_cross = t - dt + frac_cross * dt

                px_cross = x_hit

                # A crossing at table-skim height with the ball still falling
                # means no bounce was modelled (off-table, centroid clamped at the
                # contact height z = ball radius): the ball is effectively dead, so
                # it is not a usable strike. The margin keeps the threshold strictly
                # above the clamp height.
                dead_ball = pz_cross < contact_z + 0.03 and vz_cross < 0.0
                return StrikeTarget(
                    p_ball=np.array([px_cross, py_cross, pz_cross]),
                    v_ball=np.array([vx_cross, vy_cross, vz_cross]),
                    t_strike=t_cross, num_bounces=bounces, valid=not dead_ball,
                )

            x, y, z = x_new, y_new, z_new
            vx, vy, vz = vx_new, vy_new, vz_new

        return StrikeTarget(
            p_ball=np.array([x, y, z]), v_ball=np.array([vx, vy, vz]),
            t_strike=t, num_bounces=bounces, valid=False,
        )
