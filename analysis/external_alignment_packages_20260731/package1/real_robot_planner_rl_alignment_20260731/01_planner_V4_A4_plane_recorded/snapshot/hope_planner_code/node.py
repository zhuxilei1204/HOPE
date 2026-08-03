"""ROS 2 node for the HOPE no-spin racket planner.

Subscribes to the mocap ball stream (``geometry_msgs/PoseArray`` on the
configured poses topic, ball at ``ball_pose_index``), estimates the ball
position/velocity, predicts the no-spin trajectory to a fixed strike plane,
selects forehand/backhand by splitting the predicted lateral (y) position, and
publishes the typed ``hope_msgs/RacketCommand``.

Lifecycle: each new incoming ball gets a new ``task_id``; pre-strike updates
keep that id and increase ``task_revision``; ``swing_side`` is chosen once per
task and locked within it. The first sample seeds position and subsequent
samples enable the velocity fit. Live commands then pass a convergence,
revision-jump, and near-strike freeze gate before publication.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from hope_msgs.msg import RacketCommand

from .command_stability_gate import CommandStabilityConfig, CommandStabilityGate
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .constants import PlannerConfig, load_ball_physics, load_paddle_params, load_table_params
from .planner import HOPEPlanner
from .side_selection import select_swing_side

_TASK_ID_WRAP = 1 << 64
_REVISION_WRAP = 1 << 32


class HOPEPlannerNode(Node):
    """ROS 2 wrapper around :class:`HOPEPlanner`."""

    def __init__(self):
        super().__init__("hope_planner")

        # --- Topics / frames ---
        self.declare_parameter("poses_topic", "/poses")
        self.declare_parameter("command_topic", "/racket/command")
        self.declare_parameter("frame_id", "world")
        # Which slot in the PoseArray is the ball (PoseArray carries no names).
        self.declare_parameter("ball_pose_index", 0)

        # --- Planner geometry / tuning ---
        self.declare_parameter("x_hit", 0.0)              # fixed strike-plane x (m)
        self.declare_parameter("swing_side_split_y", -0.7625)   # FH/BH split on predicted y (m)
        self.declare_parameter("swing_side_hysteresis_y", 0.0)  # optional band around the split (m)
        self.declare_parameter("target_land_x", 2.055)   # fixed landing target x (m)
        self.declare_parameter("target_land_y", -0.7625)  # fixed landing target y (m)
        self.declare_parameter("delta_t_flight", 0.5)     # desired post-strike flight time (s)
        self.declare_parameter("max_predict_time", 2.0)   # prediction horizon (s)
        self.declare_parameter("dt_integrate", 0.001)     # trajectory integration step (s)
        # Rate-independent short history; fit_window is only a hard count cap.
        self.declare_parameter("fit_window_s", 0.14)
        self.declare_parameter("fit_window", 67)
        self.declare_parameter("poly_order_xy", 1)
        self.declare_parameter("poly_order_z", 2)
        # Real samples required after startup and each detected table bounce.
        self.declare_parameter("min_ready_samples", 20)
        # Live revision stability gate. Values were selected by replaying the
        # 2026-07-26 real-robot HDU/MDU session, not by changing robot limits.
        self.declare_parameter("revision_gate_initial_consecutive", 2)
        self.declare_parameter("revision_gate_initial_position_tolerance_m", 0.10)
        self.declare_parameter("revision_gate_max_position_jump_m", 0.10)
        self.declare_parameter("revision_gate_max_velocity_jump_mps", 1.50)
        self.declare_parameter("revision_gate_freeze_tts_s", 0.20)
        self.declare_parameter("revision_gate_max_strike_time_jump_s", 0.040)
        self.declare_parameter("bounce_z_tol", 0.005)
        self.declare_parameter("bounce_center_z_max", 0.11)
        self.declare_parameter("bounce_min_vertical_delta", 0.002)
        self.declare_parameter("bounce_refractory_s", 0.08)
        self.declare_parameter("bounce_max_sample_gap_s", 0.01)
        # Push every mocap sample into the estimator; run the predict+plan solve
        # at most every solve_period_s (<= 50 Hz). 0.0 = solve on every sample.
        self.declare_parameter("solve_period_s", 0.02)
        # /poses pauses while the rigid body is lost. A long gap therefore
        # marks a new physical tracking stream and must not become a revision
        # of the previously accepted ball.
        self.declare_parameter("tracking_gap_reset_s", 0.25)
        # Table's +y edge in the play frame (table occupies y in [y_max - width, y_max]).
        self.declare_parameter("table_y_max", 0.0)
        # Optional explicit path to configs/ball_physics.yaml ("" = auto-discover).
        self.declare_parameter("ball_physics_path", "")
        # Planner-only physics overrides. Negative values leave the shared
        # configs/ball_physics.yaml constants untouched.
        self.declare_parameter("drag_k", -1.0)
        self.declare_parameter("table_c_h", -1.0)
        self.declare_parameter("table_c_v", -1.0)

        self._ball_index = int(self.get_parameter("ball_pose_index").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._split_y = float(self.get_parameter("swing_side_split_y").value)
        self._hysteresis_y = max(0.0, float(self.get_parameter("swing_side_hysteresis_y").value))
        self._solve_period = float(self.get_parameter("solve_period_s").value)
        self._tracking_gap_reset = max(
            0.0, float(self.get_parameter("tracking_gap_reset_s").value)
        )

        physics_path = str(self.get_parameter("ball_physics_path").value) or None
        physics = load_ball_physics(physics_path)
        for param_name, attr_name in (
            ("drag_k", "k"),
            ("table_c_h", "C_h"),
            ("table_c_v", "C_v"),
        ):
            override = float(self.get_parameter(param_name).value)
            if override >= 0.0:
                setattr(physics, attr_name, override)
        paddle = load_paddle_params(physics_path)
        table = load_table_params(physics_path, y_max=float(self.get_parameter("table_y_max").value))
        config = PlannerConfig(
            x_hit=float(self.get_parameter("x_hit").value),
            # Landing target z = ball radius: the outgoing arc is solved for the ball
            # CENTROID reaching table contact, matching the bounce-plane convention.
            target_land=np.array([
                float(self.get_parameter("target_land_x").value),
                float(self.get_parameter("target_land_y").value),
                physics.radius,
            ]),
            delta_t_flight=float(self.get_parameter("delta_t_flight").value),
            max_predict_time=float(self.get_parameter("max_predict_time").value),
            dt_integrate=float(self.get_parameter("dt_integrate").value),
            fit_window_s=float(self.get_parameter("fit_window_s").value),
            fit_window=int(self.get_parameter("fit_window").value),
            poly_order_xy=int(self.get_parameter("poly_order_xy").value),
            poly_order_z=int(self.get_parameter("poly_order_z").value),
            min_ready_samples=int(self.get_parameter("min_ready_samples").value),
            bounce_z_tol=float(self.get_parameter("bounce_z_tol").value),
            bounce_center_z_max=float(self.get_parameter("bounce_center_z_max").value),
            bounce_min_vertical_delta=float(self.get_parameter(
                "bounce_min_vertical_delta").value),
            bounce_refractory_s=float(self.get_parameter("bounce_refractory_s").value),
            bounce_max_sample_gap_s=float(self.get_parameter(
                "bounce_max_sample_gap_s").value),
            C_r=paddle["C_r"],
            paddle_a_t=paddle["paddle_a_t"],
            paddle_b_t=paddle["paddle_b_t"],
            paddle_mu=paddle["paddle_mu"],
        )
        self.planner = HOPEPlanner(physics=physics, config=config, table=table)
        self._command_gate = CommandStabilityGate(CommandStabilityConfig(
            initial_consecutive=int(self.get_parameter(
                "revision_gate_initial_consecutive").value),
            initial_position_tolerance_m=float(self.get_parameter(
                "revision_gate_initial_position_tolerance_m").value),
            max_position_jump_m=float(self.get_parameter(
                "revision_gate_max_position_jump_m").value),
            max_velocity_jump_mps=float(self.get_parameter(
                "revision_gate_max_velocity_jump_mps").value),
            freeze_time_to_strike_s=float(self.get_parameter(
                "revision_gate_freeze_tts_s").value),
            max_strike_time_jump_s=float(self.get_parameter(
                "revision_gate_max_strike_time_jump_s").value),
        ))

        # --- Task lifecycle state ---
        self._task_id = 0
        self._task_revision = 0
        self._candidate_active = False
        self._task_active = False
        self._locked_side = RacketCommand.FOREHAND
        self._prev_side = 0            # side of the previous task (for hysteresis); 0 = none
        self._last_solve_t = None
        self._last_pose_t = None

        mocap_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            PoseArray, str(self.get_parameter("poses_topic").value), self._poses_cb, mocap_qos)
        self.cmd_pub = self.create_publisher(
            RacketCommand, str(self.get_parameter("command_topic").value), command_qos)

        self.get_logger().info(
            f"HOPE planner started: x_hit={config.x_hit:.3f} m, "
            f"landing={config.target_land[:2]}, split_y={self._split_y:.3f} m, "
            f"fit_window_s={config.fit_window_s:.3f}, fit_window_cap={config.fit_window}, "
            f"poly_order_xy={config.poly_order_xy}, poly_order_z={config.poly_order_z}, "
            f"min_ready_samples={config.min_ready_samples}, post_bounce=real_samples_only, "
            f"bounce_center_z_max={config.bounce_center_z_max:.3f} m, "
            f"ball_physics=(k={physics.k:.4f}, C_h={physics.C_h:.3f}, C_v={physics.C_v:.3f}), "
            f"solve_period={self._solve_period:.3f} s, ball_pose_index={self._ball_index}")

    def _select_side(self, intercept_y: float) -> int:
        """Binary forehand/backhand split on the predicted lateral y (optional hysteresis).

        y below the split -> FOREHAND, at/above -> BACKHAND (the convention in
        docs/PLANNER_INTERFACE.md). Delegates to the pure
        :func:`~hope_planner.side_selection.select_swing_side`, whose boundary
        behaviour is pinned by test/test_side_selection.py. The ROS message
        constants match the pure module's (+1 / -1) by definition of the msg.
        """
        return select_swing_side(
            float(intercept_y), self._split_y, self._hysteresis_y, self._prev_side
        )

    def _warn_if_outlier(self) -> None:
        """Surface a dropped mocap frame (throttled) without disrupting the pipeline.

        The estimator already handles the outlier itself (see
        ``BallStateEstimator._is_outlier``); this only makes the data-cleaning
        step observable for on-site debugging (e.g. a wrong ball_pose_index,
        a reflection, or a mocap dropout showing up as repeated rejections).
        """
        # Some deployed HDU estimator builds predate the optional outlier
        # diagnostic property.  It is observability-only, so absence must not
        # terminate the planning callback (and therefore the whole stack).
        if getattr(self.planner.estimator, "outlier_rejected", False):
            self.get_logger().warning(
                "ball pose sample rejected by the outlier gate (implausible jump or "
                "non-finite value); check the mocap feed (ball_pose_index, reflections, "
                "dropouts)", throttle_duration_sec=2.0)

    def _reset_tracking_stream(self, gap_s: float) -> None:
        """End the old ball after a mocap pause without reusing stale state."""
        self.planner.reset_stream()
        self._task_active = False
        self._candidate_active = False
        self._command_gate.reset()
        self._last_solve_t = None
        self.get_logger().info(
            f"tracking stream reset after {gap_s:.3f}s pose gap; "
            "next accepted command starts a new task"
        )

    def _poses_cb(self, msg: PoseArray) -> None:
        if len(msg.poses) <= self._ball_index:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pose = msg.poses[self._ball_index]
        p_ball = np.array([pose.position.x, pose.position.y, pose.position.z])

        if self._last_pose_t is not None:
            gap_s = t - self._last_pose_t
            if (
                (self._tracking_gap_reset > 0.0
                 and gap_s > self._tracking_gap_reset)
                or gap_s < -1e-6
            ):
                self._reset_tracking_stream(gap_s)
        self._last_pose_t = t

        # Feed every sample to the estimator, but rate-limit the solve.
        if (self._solve_period > 0.0 and self._last_solve_t is not None
                and 0.0 <= (t - self._last_solve_t) < self._solve_period):
            self.planner.estimator.push(t, p_ball)
            self._warn_if_outlier()
            return
        self._last_solve_t = t

        # A degenerate mocap frame must degrade to "no command", never kill the node.
        try:
            cmd = self.planner.update(t, p_ball)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            self.get_logger().warning(
                f"planner solve skipped ({type(exc).__name__}: {exc}); check the mocap feed "
                "(units, frame, outliers)", throttle_duration_sec=2.0)
            return
        self._warn_if_outlier()

        if cmd is None:
            # Ball struck / moving away -> end the active task; next incoming
            # ball starts a fresh task_id.
            if self.planner.ball_incoming is False:
                self._task_active = False
                self._candidate_active = False
                self._command_gate.reset()
            return

        if not self._candidate_active:
            self._candidate_active = True
            self._command_gate.reset()

        tts = self.planner.time_to_strike
        if tts is None or not self._command_gate.consider(
                cmd.p_intercept, cmd.v_racket, float(tts), candidate_time_s=t):
            self.get_logger().warning(
                f"planner command withheld by revision gate: "
                f"{self._command_gate.last_reason}", throttle_duration_sec=1.0)
            return

        if not self._task_active:
            self._task_id = (self._task_id + 1) % _TASK_ID_WRAP
            self._task_revision = 0
            self._locked_side = self._select_side(float(cmd.p_intercept[1]))
            self._prev_side = self._locked_side
            self._task_active = True
        else:
            self._task_revision = (self._task_revision + 1) % _REVISION_WRAP

        self._publish(cmd, msg.header)

    def _publish(self, cmd, header) -> None:
        out = RacketCommand()
        out.header = header
        out.header.frame_id = self._frame_id
        out.task_id = self._task_id
        out.task_revision = self._task_revision
        out.swing_side = self._locked_side
        out.position.x = float(cmd.p_intercept[0])
        out.position.y = float(cmd.p_intercept[1])
        out.position.z = float(cmd.p_intercept[2])
        out.velocity.x = float(cmd.v_racket[0])
        out.velocity.y = float(cmd.v_racket[1])
        out.velocity.z = float(cmd.v_racket[2])
        out.target_normal.x = float(cmd.n_racket[0])
        out.target_normal.y = float(cmd.n_racket[1])
        out.target_normal.z = float(cmd.n_racket[2])
        tts = self.planner.time_to_strike
        out.time_to_strike = float(tts) if tts is not None else 0.0
        self.cmd_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HOPEPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
