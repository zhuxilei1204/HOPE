# Copyright (c) 2025, Intelligent Racing Inc. (dba Hitch Interactive).
# SPDX-License-Identifier: Apache-2.0
"""Real-physics MuJoCo ping-pong scene for the sim-to-sim success_rate evaluation.

This module assembles a single MuJoCo model that contains

  * the shipped ``a3_pingpong`` robot (loaded verbatim from its MJCF -- no copy of
    the robot model lives here), and
  * a static **table top**, **net** and **floor** contact surface, plus a dynamic
    **ball** (free joint, sphere), all sized/placed and given contact restitution +
    friction from ``configs/ball_physics.yaml``.

The point of the scene is that ``success_rate`` can be measured from an ACTUAL
simulated ball that really bounces off the racket, table and net -- not from an
analytic predicted-landing rollout. The robot's racket link already carries a
collision geom in the shipped MJCF (``right_racket_collision``), so no extra racket
contact geom has to be added; the ball collides with it directly.

Frames
------
The MuJoCo world frame is the robot frame: the robot's own floor is at ``z = 0`` and
the robot stands on it. The **table frame** used by the success metric places the
table playing surface at ``z = 0`` with its origin at the near-side left corner of
the table (``+x`` toward the opponent, ``+y`` left). The two frames differ by a pure
translation ``offset = (near_edge_x, width/2, table_height)`` so that

    table_frame_position = mujoco_world_position - offset

Velocities are identical in both frames (pure translation). All success checks
(net crossing, opponent-half first bounce) are evaluated in the table frame; the
policy-facing quantities (robot state, racket target) stay in the MuJoCo world frame
exactly as the reference deploy runner expects.

Restitution
-----------
MuJoCo has no direct restitution parameter; a bouncy contact is a lightly-damped
soft constraint. Each ball<->surface contact is added as an explicit ``<pair>`` whose
``solref`` damping ratio is derived from the configured normal restitution ``e`` via
the linear spring-damper log-decrement ``zeta = -ln(e) / sqrt(pi^2 + ln(e)^2)``.
Near-elastic surfaces use a slightly softer time constant for numerical stability.
This is an approximation of the measured restitution; the FIRST-bounce landing that
``success_rate`` depends on is set by the outgoing flight (gravity + drag + the real
racket contact) before any table-bounce restitution matters, so the metric is robust
to the approximation.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field

import numpy as np
import yaml


def _load_shared_table_origin_world() -> np.ndarray:
    """Return the table-frame origin expressed in the robot-centred simulator world."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        path = parent / "configs" / "table_frame.yaml"
        if path.is_file():
            with path.open("r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh) or {}
            translation = np.asarray(
                doc["simulation"]["table_to_world_translation_xyz"], dtype=np.float64
            ).reshape(-1)
            quaternion = np.asarray(
                doc["simulation"]["table_to_world_quaternion_wxyz"], dtype=np.float64
            ).reshape(-1)
            if translation.shape != (3,):
                raise ValueError(f"{path}: table translation must have three values")
            if quaternion.shape != (4,) or not np.allclose(
                quaternion, [1.0, 0.0, 0.0, 0.0], atol=1.0e-9
            ):
                raise ValueError(
                    f"{path}: MuJoCo scene currently requires an identity table-frame rotation"
                )
            return translation
    raise FileNotFoundError("configs/table_frame.yaml not found from MuJoCo scene")


SHARED_TABLE_ORIGIN_WORLD = _load_shared_table_origin_world()

# Physics substep applied-drag uses the ball's world-frame linear velocity. For a
# MuJoCo free joint the first three qvel entries are exactly that world velocity.


def _dampratio_from_restitution(e: float) -> float:
    """Linear spring-damper damping ratio that yields normal restitution ``e``."""
    e = float(min(max(e, 1e-3), 0.999))
    le = math.log(e)
    return -le / math.sqrt(math.pi * math.pi + le * le)


@dataclass
class StepResult:
    """Physical events observed during one 50 Hz control step (sub-step resolution).

    Everything here is a raw physical observation of the real simulated ball; the
    success DEFINITION (after-contact gating, net clearance threshold, opponent-half
    test) is applied by the evaluator, not here.
    """

    ball_racket_contact: bool = False
    contact_substep: int | None = None
    contact_time_offset_s: float | None = None
    contact_pos_w: np.ndarray | None = None
    contact_normal_w: np.ndarray | None = None
    contact_dist: float | None = None
    contact_ball_pos_pre_w: np.ndarray | None = None
    contact_ball_vel_pre_w: np.ndarray | None = None
    contact_ball_pos_post_w: np.ndarray | None = None
    contact_ball_vel_post_w: np.ndarray | None = None
    # Each net-plane (x = net_x) crossing during the step: (z_table_at_crossing, x_sign)
    # where x_sign = +1 if the ball moved in +x (outgoing toward the opponent).
    net_crossings: list = field(default_factory=list)
    # Each table-surface plane (z_table = ball_radius) crossing: (x_table, y_table, z_sign)
    # where z_sign = -1 for a downward crossing (a bounce onto the surface).
    surface_crossings: list = field(default_factory=list)


@dataclass
class RobotObsState:
    """Proprioceptive state consumed by the reference 111-D observation builder.

    Attribute names match the reference ``RobotState`` so it can be passed straight
    into ``build_observation`` without conversion.
    """

    base_pos_w: np.ndarray
    base_quat_w: np.ndarray
    base_ang_vel_b: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    stability_feedback: np.ndarray | None = None


class PingPongRealPhysicsScene:
    """Robot + table + net + floor + dynamic ball, stepped with real MuJoCo physics."""

    def __init__(
        self,
        robot_xml_path: str,
        ball_cfg: dict,
        joint_names,
        control_dt: float = 0.02,
        near_edge_x: float | None = None,
        launch_viewer: bool = False,
    ) -> None:
        import mujoco  # lazy import so the module imports without MuJoCo present

        self._mj = mujoco
        self.control_dt = float(control_dt)
        self.joint_names = list(joint_names)
        self.num_joints = len(self.joint_names)

        # --- geometry from the shared ball-physics config -----------------------
        g = float(ball_cfg.get("gravity", 9.81))
        table = ball_cfg.get("table", {})
        net = ball_cfg.get("net", {})
        ball = ball_cfg.get("ball", {})
        drag = ball_cfg.get("drag", {})
        self.length = float(table.get("length", 2.74))
        self.width = float(table.get("width", 1.525))
        self.table_height = float(table.get("height", 0.76))
        self.table_thickness = float(table.get("thickness", 0.05))
        self.net_height = float(net.get("height", 0.1525))
        self.net_x_table = float(net.get("x_position", self.length / 2.0))
        self.net_overhang = float(net.get("overhang", 0.15))
        self.net_thickness = float(net.get("thickness", 0.01))
        self.ball_radius = float(ball.get("radius", 0.020))
        self.ball_mass = float(ball.get("mass", 0.0027))
        self.drag_k = float(drag.get("k", 0.1261))
        self.velocity_clip = float(drag.get("velocity_clip", 50.0))
        self.gravity = g

        self.offset = SHARED_TABLE_ORIGIN_WORLD.copy()
        if near_edge_x is not None:
            self.offset[0] = float(near_edge_x)
        self.near_edge_x = float(self.offset[0])
        self.table_surface_z = float(self.offset[2])
        self.table_center_y = float(self.offset[1] - self.width / 2.0)
        # mujoco_world -> table_frame is a pure translation by this offset.
        self.net_x_mujoco = self.near_edge_x + self.net_x_table

        # --- build the combined model via the MuJoCo spec API -------------------
        self._build_model(mujoco, robot_xml_path, ball_cfg)

        # --- resolve addresses -------------------------------------------------
        m = self.model
        self._base_qadr = int(m.jnt_qposadr[self._joint_id("pelvis_free_joint")])
        self._base_vadr = int(m.jnt_dofadr[self._joint_id("pelvis_free_joint")])
        self._ball_qadr = int(m.jnt_qposadr[self._joint_id("ball_free_joint")])
        self._ball_vadr = int(m.jnt_dofadr[self._joint_id("ball_free_joint")])
        self._ball_bid = self._body_id("ball")
        self._pelvis_bid = self._body_id("pelvis_link")
        self._torso_bid = self._body_id("torso_Link")
        self._right_shoulder_bid = self._body_id("right_shoulder_roll_Link")
        self._left_foot_bid = self._body_id("left_ankle_roll_Link")
        self._right_foot_bid = self._body_id("right_ankle_roll_Link")
        self._racket_sid = self._site_id("right_racket")
        self._ball_gid = self._geom_id("ball_geom")
        self._racket_gid = self._geom_id("right_racket_collision")
        self._floor_gid = self._geom_id("floor")
        self._gyro_adr = self._sensor_adr("pelvis_imu_gyro")
        excluded_bodies = {
            self._ball_bid,
            self._body_id("table_top"),
            self._body_id("net_body"),
        }
        self._robot_mass_body_ids = np.array(
            [
                body_id
                for body_id in range(1, m.nbody)
                if body_id not in excluded_bodies and float(m.body_mass[body_id]) > 0.0
            ],
            dtype=np.int32,
        )
        self._left_foot_descendants = self._body_descendants(self._left_foot_bid)
        self._right_foot_descendants = self._body_descendants(self._right_foot_bid)

        # Controlled-joint qpos/qvel addresses + driving actuator indices.
        self._q_adr = np.zeros(self.num_joints, dtype=int)
        self._v_adr = np.zeros(self.num_joints, dtype=int)
        self._act_idx = np.full(self.num_joints, -1, dtype=int)
        trn_joint = m.actuator_trnid[:, 0]
        for i, name in enumerate(self.joint_names):
            jid = self._joint_id(name)
            self._q_adr[i] = int(m.jnt_qposadr[jid])
            self._v_adr[i] = int(m.jnt_dofadr[jid])
            matches = np.where(trn_joint == jid)[0]
            if matches.size == 0:
                raise ValueError(f"no actuator drives joint '{name}'")
            self._act_idx[i] = int(matches[0])
        self._ctrl_lo = m.actuator_ctrlrange[self._act_idx, 0].copy()
        self._ctrl_hi = m.actuator_ctrlrange[self._act_idx, 1].copy()
        self._ctrl_limited = m.actuator_ctrllimited[self._act_idx].astype(bool)

        self._substeps = max(1, int(round(self.control_dt / m.opt.timestep)))

        self._q_des = np.zeros(self.num_joints)
        self._kp = np.zeros(self.num_joints)
        self._kd = np.zeros(self.num_joints)

        self._viewer = None
        if launch_viewer:
            from mujoco import viewer as mj_viewer

            self._viewer = mj_viewer.launch_passive(self.model, self.data)

    # -- model construction -----------------------------------------------------
    def _build_model(self, mujoco, robot_xml_path, ball_cfg) -> None:
        spec = mujoco.MjSpec.from_file(str(robot_xml_path))
        wb = spec.worldbody

        # Collision bitmasks: the ball collides with the racket (already in the
        # robot model) + our table/net, but our table/net do NOT collide with the
        # robot (so the static surfaces never shove the robot around).
        BALL_CT, BALL_CA = 8, 15   # ball
        SURF_CT, SURF_CA = 8, 8    # table + net (share the ball's bit only)

        x0, w, h, th = self.near_edge_x, self.width, self.table_surface_z, self.table_thickness
        center_y = self.table_center_y
        L = self.length

        # Table top slab: top face at table_height (z=0 in the table frame).
        tb = wb.add_body(name="table_top", pos=[x0 + L / 2.0, center_y, h - th / 2.0])
        tb.add_geom(
            name="table_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[L / 2.0, w / 2.0, th / 2.0], contype=SURF_CT, conaffinity=SURF_CA,
            rgba=[0.10, 0.35, 0.55, 1.0],
        )
        # Net slab at the net plane, spanning the table width + overhang each side.
        nb = wb.add_body(
            name="net_body", pos=[self.net_x_mujoco, center_y, h + self.net_height / 2.0]
        )
        nb.add_geom(
            name="net_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.net_thickness / 2.0, w / 2.0 + self.net_overhang, self.net_height / 2.0],
            contype=SURF_CT, conaffinity=SURF_CA, rgba=[0.9, 0.9, 0.9, 0.35],
        )
        # Dynamic ball (free joint).
        bb = wb.add_body(name="ball", pos=[x0 + L / 2.0, center_y, h + 0.5])
        bb.add_freejoint(name="ball_free_joint")
        bb.add_geom(
            name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.ball_radius, 0.0, 0.0], mass=self.ball_mass,
            contype=BALL_CT, conaffinity=BALL_CA, rgba=[1.0, 0.55, 0.0, 1.0],
        )

        # Contact pairs: restitution + friction from the config (see module docstring).
        contact = ball_cfg.get("contact", {})

        def _friction(surface_key: str, default: float) -> float:
            return float(contact.get(surface_key, {}).get("dynamic_friction", default))

        def _restitution(surface_key: str, default: float) -> float:
            return float(contact.get(surface_key, {}).get("restitution", default))

        def _add_pair(g1: str, g2: str, e: float, fr: float) -> None:
            pair = spec.add_pair(geomname1=g1, geomname2=g2)
            timeconst = 0.03 if e > 0.8 else 0.02   # softer for near-elastic stability
            pair.solref = [timeconst, _dampratio_from_restitution(e)]
            pair.friction = [fr, fr, 0.005, 1e-4, 1e-4]
            pair.condim = 3

        _add_pair("ball_geom", "table_geom", _restitution("table", 0.9215), _friction("table", 0.40))
        _add_pair("ball_geom", "net_geom", _restitution("net", 0.10), _friction("net", 0.50))
        _add_pair("ball_geom", "right_racket_collision", _restitution("paddle", 0.654), _friction("paddle", 0.60))
        _add_pair("ball_geom", "floor", _restitution("floor", 0.40), _friction("floor", 0.80))

        # Extend the model's stand keyframe with a resting ball pose so the combined
        # nq matches (the ball's 7 qpos are overwritten per serve anyway).
        if spec.keys:
            key = spec.keys[0]
            key.qpos = list(np.array(key.qpos, dtype=np.float64)) + [
                x0 + L / 2.0, center_y, h + 0.5, 1.0, 0.0, 0.0, 0.0
            ]

        self.model = spec.compile()
        self.data = self._mj.MjData(self.model)
        # Honour the configured gravity magnitude (drag is applied on top via xfrc).
        self.model.opt.gravity[:] = [0.0, 0.0, -self.gravity]

    # -- id / address helpers ---------------------------------------------------
    def _joint_id(self, name: str) -> int:
        jid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint '{name}' not found")
        return jid

    def _body_id(self, name: str) -> int:
        return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, name)

    def _site_id(self, name: str) -> int:
        return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_SITE, name)

    def _geom_id(self, name: str) -> int:
        gid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"geom '{name}' not found")
        return gid

    def _sensor_adr(self, name: str) -> int:
        sid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid]) if sid >= 0 else -1

    def _body_descendants(self, root_body_id: int) -> set[int]:
        descendants = {int(root_body_id)}
        for body_id in range(1, self.model.nbody):
            parent = int(body_id)
            while parent > 0:
                if parent == root_body_id:
                    descendants.add(int(body_id))
                    break
                parent = int(self.model.body_parentid[parent])
        return descendants

    @staticmethod
    def _matrix_to_rpy_deg(xmat: np.ndarray) -> np.ndarray:
        rot = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
        pitch = math.asin(float(np.clip(-rot[2, 0], -1.0, 1.0)))
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        return np.rad2deg([roll, pitch, yaw])

    @staticmethod
    def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
        unique = sorted({(float(p[0]), float(p[1])) for p in np.asarray(points)})
        if len(unique) <= 1:
            return np.asarray(unique, dtype=np.float64)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)

    @staticmethod
    def _signed_polygon_margin(point_xy: np.ndarray, polygon_xy: np.ndarray) -> float:
        point = np.asarray(point_xy, dtype=np.float64)
        polygon = np.asarray(polygon_xy, dtype=np.float64)
        if polygon.shape[0] < 3:
            return float("nan")
        margins = []
        for idx in range(polygon.shape[0]):
            p0 = polygon[idx]
            p1 = polygon[(idx + 1) % polygon.shape[0]]
            edge = p1 - p0
            edge_norm = float(np.linalg.norm(edge))
            if edge_norm < 1.0e-12:
                continue
            margins.append(float(np.cross(edge, point - p0) / edge_norm))
        return min(margins) if margins else float("nan")

    # -- frame transform --------------------------------------------------------
    def to_table(self, pos_w) -> np.ndarray:
        """MuJoCo world position -> table-frame position (pure translation)."""
        return np.asarray(pos_w, dtype=np.float64) - self.offset

    # -- reset / serve ----------------------------------------------------------
    def reset_stand(self) -> None:
        """Reset the robot to its grounded stand keyframe; park the ball far away."""
        m, d = self.model, self.data
        if m.nkey > 0:
            self._mj.mj_resetDataKeyframe(m, d, 0)
        else:
            self._mj.mj_resetData(m, d)
        d.xfrc_applied[:] = 0.0
        # Park the ball out of play until a serve is set.
        d.qpos[self._ball_qadr:self._ball_qadr + 3] = [
            self.near_edge_x + self.length / 2.0,
            self.table_center_y,
            self.table_surface_z + 1.0,
        ]
        d.qpos[self._ball_qadr + 3:self._ball_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_vadr:self._ball_vadr + 6] = 0.0
        self._mj.mj_forward(m, d)

    def set_ball(self, pos_w, vel_w) -> None:
        """Place the ball at ``pos_w`` (MuJoCo world) with world linear velocity ``vel_w``."""
        d = self.data
        d.qpos[self._ball_qadr:self._ball_qadr + 3] = np.asarray(pos_w, dtype=np.float64)
        d.qpos[self._ball_qadr + 3:self._ball_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[self._ball_vadr:self._ball_vadr + 3] = np.asarray(vel_w, dtype=np.float64)
        d.qvel[self._ball_vadr + 3:self._ball_vadr + 6] = 0.0
        self._mj.mj_forward(self.model, self.data)

    # -- state readout ----------------------------------------------------------
    def read_robot_state(self) -> RobotObsState:
        d = self.data
        base_pos = d.qpos[self._base_qadr:self._base_qadr + 3].copy()
        base_quat = d.qpos[self._base_qadr + 3:self._base_qadr + 7].copy()  # (w,x,y,z)
        if self._gyro_adr >= 0:
            base_ang_vel = d.sensordata[self._gyro_adr:self._gyro_adr + 3].copy()
        else:
            base_ang_vel = d.qvel[self._base_vadr + 3:self._base_vadr + 6].copy()
        balance = self.robot_balance_diagnostics()
        pelvis_rot = np.asarray(balance["pelvis_xmat_w"], dtype=np.float64)
        base_lin_vel_b = pelvis_rot.T @ np.asarray(
            balance["pelvis_lin_vel_w"], dtype=np.float64
        )
        torso_rot = d.xmat[self._torso_bid].reshape(3, 3)
        torso_gravity_b = torso_rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        support_xy = 0.5 * (
            d.xpos[self._left_foot_bid, :2] + d.xpos[self._right_foot_bid, :2]
        )
        com_delta = np.asarray(balance["com_w"], dtype=np.float64)[:2] - support_xy
        heading = pelvis_rot[:2, 0]
        heading /= max(float(np.linalg.norm(heading)), 1.0e-9)
        left = np.array([-heading[1], heading[0]], dtype=np.float64)
        com_support_b = np.array(
            [float(np.dot(com_delta, heading)), float(np.dot(com_delta, left))],
            dtype=np.float64,
        )
        stability_feedback = np.concatenate(
            (
                base_lin_vel_b[:2],
                torso_gravity_b[:2],
                com_support_b,
                np.array(
                    [
                        balance["left_foot_contact"],
                        balance["right_foot_contact"],
                    ],
                    dtype=np.float64,
                ),
            )
        )
        return RobotObsState(
            base_pos_w=base_pos,
            base_quat_w=base_quat,
            base_ang_vel_b=base_ang_vel,
            q=d.qpos[self._q_adr].copy(),
            qd=d.qvel[self._v_adr].copy(),
            stability_feedback=stability_feedback,
        )

    def robot_balance_diagnostics(self) -> dict[str, float | np.ndarray]:
        """Return read-only whole-body balance diagnostics in the MuJoCo world frame."""
        m, d = self.model, self.data
        masses = m.body_mass[self._robot_mass_body_ids]
        mass_sum = float(np.sum(masses))
        if mass_sum > 0.0:
            com_w = np.sum(d.xipos[self._robot_mass_body_ids] * masses[:, None], axis=0) / mass_sum
        else:
            com_w = np.full(3, np.nan, dtype=np.float64)

        def body_velocity(body_id: int) -> tuple[np.ndarray, np.ndarray]:
            velocity = np.zeros(6, dtype=np.float64)
            self._mj.mj_objectVelocity(
                m, d, self._mj.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0
            )
            return velocity[:3].copy(), velocity[3:].copy()

        pelvis_ang_w, pelvis_lin_w = body_velocity(self._pelvis_bid)
        _, left_foot_lin_w = body_velocity(self._left_foot_bid)
        _, right_foot_lin_w = body_velocity(self._right_foot_bid)

        left_contact = False
        right_contact = False
        for contact_idx in range(d.ncon):
            contact = d.contact[contact_idx]
            if int(contact.geom1) == self._floor_gid:
                other_geom = int(contact.geom2)
            elif int(contact.geom2) == self._floor_gid:
                other_geom = int(contact.geom1)
            else:
                continue
            other_body = int(m.geom_bodyid[other_geom])
            left_contact |= other_body in self._left_foot_descendants
            right_contact |= other_body in self._right_foot_descendants

        support_corners = []
        # Fixed diagnostic footprint dimensions. They are deliberately conservative
        # and are used only to compare policies under the same MuJoCo model.
        foot_half_length = 0.105
        foot_half_width = 0.050
        for body_id, in_contact in (
            (self._left_foot_bid, left_contact),
            (self._right_foot_bid, right_contact),
        ):
            if not in_contact:
                continue
            center = d.xpos[body_id, :2]
            rotation = d.xmat[body_id].reshape(3, 3)
            axis_x = rotation[:2, 0]
            axis_y = rotation[:2, 1]
            for sx, sy in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)):
                support_corners.append(
                    center + sx * foot_half_length * axis_x + sy * foot_half_width * axis_y
                )
        support_hull = self._convex_hull_xy(np.asarray(support_corners, dtype=np.float64))
        support_margin = self._signed_polygon_margin(com_w[:2], support_hull)

        pelvis_rpy = self._matrix_to_rpy_deg(d.xmat[self._pelvis_bid])
        torso_rpy = self._matrix_to_rpy_deg(d.xmat[self._torso_bid])
        return {
            "com_w": com_w,
            "support_margin": support_margin,
            "left_foot_contact": float(left_contact),
            "right_foot_contact": float(right_contact),
            "left_foot_speed_xy": float(np.linalg.norm(left_foot_lin_w[:2])),
            "right_foot_speed_xy": float(np.linalg.norm(right_foot_lin_w[:2])),
            "pelvis_rpy_deg": pelvis_rpy,
            "torso_rpy_deg": torso_rpy,
            "pelvis_xmat_w": d.xmat[self._pelvis_bid].reshape(3, 3).copy(),
            "pelvis_lin_vel_w": pelvis_lin_w,
            "pelvis_ang_vel_w": pelvis_ang_w,
            "right_shoulder_pos_w": d.xpos[self._right_shoulder_bid].copy(),
        }

    def ball_state(self):
        """Return (pos_w, vel_w) of the ball in the MuJoCo world frame."""
        d = self.data
        pos = d.qpos[self._ball_qadr:self._ball_qadr + 3].copy()
        vel = d.qvel[self._ball_vadr:self._ball_vadr + 3].copy()
        return pos, vel

    def racket_site_state(self):
        """Return (pos_w, vel_w) of the racket site in the MuJoCo world frame."""
        d = self.data
        pos = d.site_xpos[self._racket_sid].copy()
        res = np.zeros(6)
        self._mj.mj_objectVelocity(self.model, d, self._mj.mjtObj.mjOBJ_SITE, self._racket_sid, res, 0)
        return pos, res[3:6].copy()

    def racket_site_pose(self):
        """Return (pos_w, vel_w, xmat_w) for the racket site."""
        d = self.data
        pos, vel = self.racket_site_state()
        xmat = d.site_xmat[self._racket_sid].reshape(3, 3).copy()
        return pos, vel, xmat

    def racket_geom_pose(self):
        """Return (pos_w, xmat_w) for the physical racket collision geom."""
        d = self.data
        pos = d.geom_xpos[self._racket_gid].copy()
        xmat = d.geom_xmat[self._racket_gid].reshape(3, 3).copy()
        return pos, xmat

    def base_pos_w(self) -> np.ndarray:
        return self.data.qpos[self._base_qadr:self._base_qadr + 3].copy()

    def base_fallen(self, min_height: float = 0.4) -> bool:
        """Crude fall check: pelvis dropped well below the stand height."""
        return bool(self.data.qpos[self._base_qadr + 2] < min_height)

    # -- control + stepping -----------------------------------------------------
    def write_targets(self, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        self._q_des = np.asarray(q_des, dtype=np.float64).reshape(self.num_joints)
        self._kp = np.asarray(kp, dtype=np.float64).reshape(self.num_joints)
        self._kd = np.asarray(kd, dtype=np.float64).reshape(self.num_joints)

    def joint_control_diagnostics(self) -> dict[str, np.ndarray]:
        """Return canonical-order PD tracking and torque-limit diagnostics."""
        d = self.data
        q = d.qpos[self._q_adr].copy()
        qd = d.qvel[self._v_adr].copy()
        torque_requested = self._kp * (self._q_des - q) - self._kd * qd
        torque_applied = np.where(
            self._ctrl_limited,
            np.clip(torque_requested, self._ctrl_lo, self._ctrl_hi),
            torque_requested,
        )
        torque_clipped = np.abs(torque_requested - torque_applied) > 1.0e-9
        return {
            "q": q,
            "qd": qd,
            "q_des": self._q_des.copy(),
            "tracking_error": self._q_des - q,
            "torque_requested": torque_requested,
            "torque_applied": torque_applied,
            "torque_clipped": torque_clipped,
        }

    def _apply_pd(self) -> None:
        d = self.data
        q = d.qpos[self._q_adr]
        qd = d.qvel[self._v_adr]
        tau = self._kp * (self._q_des - q) - self._kd * qd
        tau = np.where(self._ctrl_limited, np.clip(tau, self._ctrl_lo, self._ctrl_hi), tau)
        d.ctrl[self._act_idx] = tau

    def _apply_ball_drag(self) -> None:
        """No-spin aerodynamic drag as a Cartesian force F = -m*k*|v|*v (world frame)."""
        d = self.data
        v = d.qvel[self._ball_vadr:self._ball_vadr + 3]
        speed = float(np.linalg.norm(v))
        speed = min(speed, self.velocity_clip)
        d.xfrc_applied[self._ball_bid, :3] = -self.ball_mass * self.drag_k * speed * v

    def step(self) -> StepResult:
        """Advance one 50 Hz control tick; report physical ball events sub-step wise."""
        mj = self._mj
        m, d = self.model, self.data
        result = StepResult()
        surface_table = self.ball_radius

        for substep in range(self._substeps):
            self._apply_pd()
            self._apply_ball_drag()
            p_before = d.qpos[self._ball_qadr:self._ball_qadr + 3].copy()
            v_before = d.qvel[self._ball_vadr:self._ball_vadr + 3].copy()
            mj.mj_step(m, d)
            p_after = d.qpos[self._ball_qadr:self._ball_qadr + 3].copy()
            v_after = d.qvel[self._ball_vadr:self._ball_vadr + 3].copy()

            # real ball<->racket contact this sub-step
            if not result.ball_racket_contact:
                for c in range(d.ncon):
                    con = d.contact[c]
                    if {con.geom1, con.geom2} == {self._ball_gid, self._racket_gid}:
                        result.ball_racket_contact = True
                        result.contact_substep = int(substep)
                        result.contact_time_offset_s = float((substep + 1) * m.opt.timestep)
                        result.contact_pos_w = np.asarray(con.pos, dtype=np.float64).copy()
                        result.contact_normal_w = np.asarray(con.frame[:3], dtype=np.float64).copy()
                        result.contact_dist = float(con.dist)
                        result.contact_ball_pos_pre_w = p_before.copy()
                        result.contact_ball_vel_pre_w = v_before.copy()
                        result.contact_ball_pos_post_w = p_after.copy()
                        result.contact_ball_vel_post_w = v_after.copy()
                        break

            # net-plane (x = net_x) crossing, evaluated in the table frame
            xb = p_before[0] - self.offset[0]
            xa = p_after[0] - self.offset[0]
            if (xb < self.net_x_table <= xa) or (xa < self.net_x_table <= xb):
                dx = xa - xb
                frac = (self.net_x_table - xb) / dx if abs(dx) > 1e-12 else 0.5
                z_cross = (p_before[2] + frac * (p_after[2] - p_before[2])) - self.offset[2]
                result.net_crossings.append((float(z_cross), 1.0 if xa > xb else -1.0))

            # table-surface plane (z_table = ball_radius) crossing
            zb = p_before[2] - self.offset[2]
            za = p_after[2] - self.offset[2]
            if (zb > surface_table >= za) or (za > surface_table >= zb):
                dz = za - zb
                frac = (surface_table - zb) / dz if abs(dz) > 1e-12 else 0.5
                x_cross = (p_before[0] + frac * (p_after[0] - p_before[0])) - self.offset[0]
                y_cross = (p_before[1] + frac * (p_after[1] - p_before[1])) - self.offset[1]
                result.surface_crossings.append((float(x_cross), float(y_cross), -1.0 if za < zb else 1.0))

        # clear the applied drag so it never leaks onto other bodies/steps
        d.xfrc_applied[self._ball_bid, :3] = 0.0

        if self._viewer is not None:
            self._viewer.sync()
        return result

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
