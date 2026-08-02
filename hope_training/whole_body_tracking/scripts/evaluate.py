"""Evaluate a trained HOPE policy in-sim and report ``success_rate`` (the only metric).

Runs the policy across many parallel environments, detects each strike (the reference clock reaching
the strike frame), and rolls out the no-spin outgoing ball to decide whether the return succeeded
(racket contact AND net crossing AND opponent-half first bounce). Forehand, backhand and rally rounds
are merged into one number:

    success_rate = successful_return_tasks / incoming_balls_that_entered_a_strike_task

Scoring contract (kept consistent with the shared ``success_metric`` module):

* positions are mapped from the sim world into the TABLE frame the metric expects (origin at the
  near-side left corner of the table surface, x in [0, length], y in [-width, 0], z = 0 at the
  surface) using the same table placement the training command term uses for its return shaping
  (``table_near_x`` / ``table_surface_z`` / station-centred y);
* the outgoing ball leaves with the racket's ACHIEVED velocity at the strike frame (not the
  commanded target velocity);
* strikes that coincide with an environment reset (time-out / fall) are excluded — a reset re-seeds
  the reference clock, which is not a swing.

This remains the fast in-Isaac estimate; the authoritative physical number comes from
``mujoco_eval_onnx.py`` (real simulated ball). Emits only a machine-readable
``{"success_rate": <float>}`` to stdout (and optionally to --json-out). No thresholds, no exit-code
changes, no other metrics.

Usage:
    python scripts/evaluate.py --checkpoint logs/rsl_rl/hope_pingpong/<run>/model_<iter>.pt \
        --num-envs 256 --num-steps 4000
"""

import argparse
import csv
import json
import os
import pathlib
import sys

import yaml
from omegaconf import OmegaConf


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir():
            return parent
    return here.parents[2]


def _resolve_motion_path(value: str) -> str:
    p = pathlib.Path(str(value))
    if p.is_file():
        return str(p.resolve())
    rooted = _repo_root() / value
    return str(rooted.resolve()) if rooted.is_file() else str(rooted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint (.pt) to evaluate.")
    parser.add_argument(
        "--task",
        default=None,
        help="Gym task id. Defaults to task-yaml.gym_task.",
    )
    parser.add_argument("--num-envs", type=int, default=256, help="Parallel environments.")
    parser.add_argument("--num-steps", type=int, default=4000, help="Policy steps to roll out.")
    parser.add_argument("--device", default="cuda:0", help="Compute device.")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260731,
        help="Deterministic environment/randomization seed.",
    )
    parser.add_argument("--contact-radius", type=float, default=0.10, help="Racket-to-target contact gate (m).")
    parser.add_argument("--json-out", default=None, help="Also write {'success_rate': ...} to this file.")
    parser.add_argument(
        "--safety-envelope-json-out",
        default=None,
        help="Write cumulative targeted-recovery safety-envelope diagnostics.",
    )
    parser.add_argument("--trace-csv", default=None, help="Write one Isaac environment's per-step diagnostics.")
    parser.add_argument(
        "--physical-shadow-json-out",
        default=None,
        help="Write rigid-ball shadow lifecycle/event diagnostics when that command term is active.",
    )
    parser.add_argument("--trace-env", type=int, default=0, help="Environment index for --trace-csv.")
    parser.add_argument(
        "--joint-action-diag-csv",
        default=None,
        help="Write one Isaac environment's canonical per-joint action/clamp/tracking/torque diagnostics.",
    )
    parser.add_argument("--experiment-name", default="hope_pingpong", help="rsl_rl experiment name.")
    parser.add_argument(
        "--task-yaml",
        default="HOPEPingPong.yaml",
        help="Task YAML whose overrides should be applied for eval (name under cfg/task or a path).",
    )
    parser.add_argument("--motion-file", default=None, help="Forehand clip override.")
    parser.add_argument("--motion-file-2", default=None, help="Backhand clip override.")
    parser.add_argument(
        "--motion-manifest",
        default=None,
        help="TSV manifest with motion clips and optional strike/racket-target metadata.",
    )
    parser.add_argument(
        "--fixed-workspace-level",
        type=float,
        default=None,
        help="Diagnostic override in [0, 1] for table-workspace sampling.",
    )
    parser.add_argument(
        "--fixed-ability-level",
        type=float,
        default=None,
        help=(
            "Diagnostic override in [0, 1] for command difficulty. Disables "
            "the physical capability updater during evaluation."
        ),
    )
    return parser.parse_args()


def _first_attr(obj, names):
    """Return the first present attribute among ``names`` (else None). Coupling shim for the env API."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _resolve_task_yaml(task_yaml: str) -> pathlib.Path:
    path = pathlib.Path(str(task_yaml)).expanduser()
    if path.is_file():
        return path.resolve()
    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    rooted = pathlib.Path(__file__).resolve().parents[1] / "cfg" / "task" / path.name
    return rooted.resolve()


def _load_task_yaml_with_defaults(task_yaml_path: pathlib.Path, seen: set[pathlib.Path] | None = None):
    """Load a task YAML and recursively merge simple Hydra-style defaults."""
    seen = set() if seen is None else seen
    task_yaml_path = task_yaml_path.resolve()
    if task_yaml_path in seen:
        raise ValueError(f"cyclic task YAML defaults include {task_yaml_path}")
    seen.add(task_yaml_path)
    with task_yaml_path.open("r", encoding="utf-8") as fh:
        task_doc = yaml.safe_load(fh) or {}

    merged = OmegaConf.create({})
    defaults = task_doc.get("defaults") or []
    for item in defaults:
        if item == "_self_" or item is None:
            continue
        if isinstance(item, str):
            parent_name = item
        elif isinstance(item, dict) and len(item) == 1:
            _, parent_name = next(iter(item.items()))
        else:
            continue
        if parent_name in (None, "_self_"):
            continue
        parent_path = _resolve_task_yaml(str(parent_name))
        if parent_path.is_file():
            merged = OmegaConf.merge(merged, _load_task_yaml_with_defaults(parent_path, seen))

    task_doc = dict(task_doc)
    task_doc.pop("defaults", None)
    return OmegaConf.merge(merged, OmegaConf.create(task_doc))


def _apply_training_task_overrides(env_cfg, task_yaml: str) -> list[str]:
    """Apply the same task YAML overrides used by train.py for comparable fast eval rollouts."""
    task_yaml_path = _resolve_task_yaml(task_yaml)
    if not task_yaml_path.is_file():
        return []
    cfg = OmegaConf.create({"task": _load_task_yaml_with_defaults(task_yaml_path)})
    from train import _apply_task_overrides

    applied: list[str] = []
    _apply_task_overrides(env_cfg, cfg, applied)
    return applied


def main() -> int:
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    task_cfg = _load_task_yaml_with_defaults(_resolve_task_yaml(args.task_yaml))
    configured_task = OmegaConf.select(task_cfg, "gym_task")
    task_id = str(args.task or configured_task or "")
    if not task_id:
        raise ValueError(
            "no Gym task was provided and task YAML has no gym_task field"
        )
    if args.task is not None and configured_task is not None:
        if str(args.task) != str(configured_task):
            raise ValueError(
                "--task conflicts with task-yaml.gym_task: "
                f"{args.task!r} != {configured_task!r}"
            )

    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=args.device)
    simulation_app = app_launcher.app

    status = 0
    try:
        import gymnasium as gym
        import torch

        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg

        import whole_body_tracking.tasks  # noqa: F401
        from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
        from whole_body_tracking.utils.ppo_cfg import load_ppo_params, runner_kwargs
        from whole_body_tracking.utils.success_metric import (
            BallPhysics,
            SuccessRate,
            TableGeometry,
            evaluate_return,
        )
        from train import _apply_motion_metadata, _resolve_motion_plan

        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = int(args.seed)
        applied = _apply_training_task_overrides(env_cfg, args.task_yaml)
        motion_args = argparse.Namespace(**vars(args))
        motion_args.task = task_cfg
        clips, motion_metadata = _resolve_motion_plan(motion_args)
        env_cfg.commands.motion.motion_file = clips if len(clips) > 1 else clips[0]
        _apply_motion_metadata(env_cfg, clips, motion_metadata, applied)
        if args.fixed_workspace_level is not None:
            workspace_level = float(args.fixed_workspace_level)
            if not 0.0 <= workspace_level <= 1.0:
                raise ValueError("--fixed-workspace-level must be in [0, 1]")
            env_cfg.commands.racket_target.table_workspace_fixed_level = (
                workspace_level
            )
        if args.fixed_ability_level is not None:
            ability_level = float(args.fixed_ability_level)
            if not 0.0 <= ability_level <= 1.0:
                raise ValueError("--fixed-ability-level must be in [0, 1]")
            if hasattr(env_cfg.rewards, "physical_capability_curriculum"):
                env_cfg.rewards.physical_capability_curriculum = None
        print(f"[evaluate.py] applied {len(applied)} task override(s):", file=sys.stderr, flush=True)
        for line in applied:
            print(f"[evaluate.py]     {line}", file=sys.stderr, flush=True)

        env = gym.make(task_id, cfg=env_cfg, render_mode=None)
        base_env = env.unwrapped
        env = RslRlVecEnvWrapper(env)

        agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(load_ppo_params(), args.experiment_name))
        agent_cfg.device = args.device
        runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=base_env.device)

        physics = BallPhysics.from_config()
        table = TableGeometry.from_config()
        accumulator = SuccessRate()

        # The racket-target command term exposes the per-strike quantities we score. Attribute names
        # are read defensively (see COUPLING NOTES in the report): the command must expose the racket
        # target position (world), the achieved racket position AND velocity (world), the
        # time-to-strike, and the swing side.
        cmd = base_env.command_manager.get_term("racket_target")
        if args.fixed_ability_level is not None:
            cmd._ability_curriculum_level.fill_(float(args.fixed_ability_level))
        motion_cmd = base_env.command_manager.get_term("motion")
        physical_shadow = None
        if "physical_shadow" in base_env.command_manager.active_terms:
            physical_shadow = base_env.command_manager.get_term(
                "physical_shadow"
            )
        if args.physical_shadow_json_out and physical_shadow is None:
            raise ValueError(
                "--physical-shadow-json-out requires a task with the "
                "'physical_shadow' command term"
            )
        robot = base_env.scene["robot"]
        env_origins = base_env.scene.env_origins  # (N, 3)
        action_term = base_env.action_manager.get_term("joint_pos")
        action_joint_names = tuple(action_term.action_joint_names)
        action_joint_ids = list(action_term.action_joint_ids)

        trace_fh = None
        trace_writer = None
        trace_env = min(max(int(args.trace_env), 0), args.num_envs - 1)
        trace_metric_names = [
            "recovery_ready_score",
            "recovery_base_ang_vel",
            "recovery_feet_contact_frac",
            "base_pitch_like_signed",
            "base_backward_velocity",
            "waist_action_delta_rms",
            "right_arm_action_delta_rms",
            "leg_action_delta_rms",
            "waist_joint_vel_rms",
            "right_arm_joint_vel_rms",
            "leg_joint_vel_rms",
            "recovery_leg_action_delta_rms",
            "recovery_leg_joint_vel_rms",
        ]
        motion_metric_names = [
            "error_body_pos_lower",
            "error_body_rot_lower",
            "error_body_lin_vel_lower",
            "error_body_ang_vel_lower",
            "reference_body_lin_speed_lower",
            "robot_body_lin_speed_lower",
            "error_body_pos_core",
            "error_body_rot_core",
            "error_body_pos_upper",
            "error_body_rot_upper",
            "error_joint_pos_legs",
            "error_joint_vel_legs",
            "reference_joint_speed_legs",
        ]
        if args.trace_csv:
            trace_path = pathlib.Path(args.trace_csv)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_fh = trace_path.open("w", encoding="utf-8", newline="")
            trace_fields = [
                "step",
                "env",
                "time_to_strike",
                "struck",
                "done",
                "swing_side",
                "base_x",
                "base_y",
                "base_z",
                "base_pitch_like_deg",
                "base_roll_like_deg",
                "base_lin_vel_x",
                "base_lin_vel_y",
                "base_lin_vel_z",
                "base_ang_vel_x",
                "base_ang_vel_y",
                "base_ang_vel_z",
                "racket_x",
                "racket_y",
                "racket_z",
                "target_x",
                "target_y",
                "target_z",
            ] + trace_metric_names + motion_metric_names
            trace_writer = csv.DictWriter(trace_fh, fieldnames=trace_fields)
            trace_writer.writeheader()
        joint_diag_fh = None
        joint_diag_writer = None
        if args.joint_action_diag_csv:
            joint_diag_path = pathlib.Path(args.joint_action_diag_csv)
            joint_diag_path.parent.mkdir(parents=True, exist_ok=True)
            joint_diag_fh = joint_diag_path.open("w", encoding="utf-8", newline="")
            joint_diag_fields = [
                "step",
                "env",
                "phase",
                "feedback_mode",
                "time_to_strike",
                "swing_side",
                "done",
                "base_x",
                "base_y",
                "base_z",
                "base_pitch_like_deg",
                "base_ang_vel_norm",
                "feet_contact_fraction",
            ]
            for prefix in (
                "raw",
                "effective",
                "feedback",
                "overflow",
                "position_clamped",
                "q",
                "qd",
                "q_des",
                "tracking_error",
                "torque_requested",
                "torque_applied",
                "torque_clipped",
            ):
                joint_diag_fields.extend(f"{prefix}__{name}" for name in action_joint_names)
            joint_diag_writer = csv.DictWriter(joint_diag_fh, fieldnames=joint_diag_fields)
            joint_diag_writer.writeheader()

        # Table placement in the ENV-LOCAL frame — the same constants the command term's return
        # shaping uses (tasks/tracking/mdp/hope_commands.py _evaluate_return). The shared
        # success-metric TABLE frame has its origin at the near-side LEFT (+y) corner of the table
        # surface: x_table = x_env - table_near_x, y_table = y_env - (station_y + width/2),
        # z_table = z_env - table_surface_z. Placement (near_x / surface_z / station y) comes from
        # the command cfg; table DIMENSIONS come from the same TableGeometry the scoring uses
        # (configs/ball_physics.yaml), so a re-fitted table width keeps the two in agreement.
        table_near_x = float(cmd.cfg.table_near_x)
        table_surface_z = float(cmd.cfg.table_surface_z)
        table_half_w = 0.5 * float(table.width)

        def read_state():
            # For robust planner training the actor-visible racket target may be intentionally
            # perturbed.  Evaluate the hidden true ball task when available, so Isaac eval matches
            # Metrics/racket_target/return_success from training.
            target_pos = _first_attr(cmd, ["ball_strike_pos_w", "racket_target_pos_w", "target_pos_w", "racket_target_w"])
            racket_pos = _first_attr(cmd, ["racket_pos_w", "achieved_racket_pos_w", "current_racket_pos_w"])
            racket_vel = _first_attr(cmd, ["racket_lin_vel_w", "racket_vel_w", "achieved_racket_vel_w"])
            tts = _first_attr(cmd, ["true_time_to_strike", "time_to_strike", "tts"])
            swing = _first_attr(cmd, ["swing_side", "swing_sign"])
            missing = [n for n, v in [
                ("racket_target_pos_w", target_pos), ("racket_pos_w", racket_pos),
                ("racket_lin_vel_w", racket_vel), ("time_to_strike", tts), ("swing_side", swing)]
                if v is None]
            if missing:
                raise AttributeError(
                    "evaluate.py could not read the strike quantities from the 'racket_target' command "
                    f"term (missing: {missing}). Expose these tensors on the command term (world frame, "
                    "shape (N,3) for positions/velocities, (N,) for tts/swing) or adjust read_state()."
                )
            return target_pos, racket_pos, racket_vel, tts, swing

        def to_table_frame(pos_w_row, e):
            """Sim-world position -> the shared metric's table frame for env ``e``."""
            p = (pos_w_row - env_origins[e]).cpu().numpy().astype(float)
            station_y = float((cmd.fixed_station_w[e, 1] - env_origins[e, 1]).item())
            p[0] -= table_near_x
            p[1] -= station_y + table_half_w   # table centred on the station: left edge -> y = 0
            p[2] -= table_surface_z
            return p

        obs, _ = env.get_observations()
        prev_tts = read_state()[3].clone()
        task_counts = {"contact": 0, "net_cross": 0, "opponent_bounce": 0}
        analytic_accumulator = SuccessRate()
        shadow_event_names = (
            "serve",
            "incoming_bounce",
            "contact",
            "net_cross",
            "outgoing_landing",
            "opponent_bounce",
            "landing_short",
            "landing_long",
            "landing_side",
            "landing_no_net",
            "abort",
            "timeout",
            "route_unalignable",
            "route_invalid",
            "incoming_net_collision",
            "reset_abort",
            "command_refresh",
        )
        def scalar_counter():
            return torch.zeros(
                (), dtype=torch.long, device=base_env.device
            )

        shadow_event_counts = {
            name: scalar_counter() for name in shadow_event_names
        }
        shadow_duplicate_counts = {
            name: scalar_counter()
            for name in (
                "incoming_bounce",
                "contact",
                "net_cross",
                "outgoing_landing",
                "opponent_bounce",
            )
        }
        shadow_order_violations = {
            "bounce_without_serve": scalar_counter(),
            "contact_without_bounce": scalar_counter(),
            "net_without_contact": scalar_counter(),
            "opponent_bounce_without_net": scalar_counter(),
            "landing_without_contact": scalar_counter(),
        }
        shadow_contact_metric_names = (
            "contact_true_target_error",
            "contact_planner_position_error",
            "contact_planner_velocity_error",
            "contact_planner_velocity_direction_error_deg",
            "contact_planner_normal_error_deg",
            "contact_outgoing_velocity_error",
            "contact_outgoing_speed",
            "contact_target_outgoing_speed",
            "contact_outgoing_speed_ratio",
            "contact_outgoing_direction_error_deg",
            "contact_outgoing_velocity_x",
            "contact_outgoing_velocity_y",
            "contact_outgoing_velocity_z",
            "contact_target_outgoing_velocity_x",
            "contact_target_outgoing_velocity_y",
            "contact_target_outgoing_velocity_z",
            "contact_incoming_velocity_error",
            "contact_incoming_velocity_direction_error_deg",
            "contact_route_incoming_velocity_error",
            "contact_route_incoming_velocity_direction_error_deg",
            "contact_actual_route_incoming_velocity_error",
            "contact_command_normal_only_error",
            "contact_command_training_model_error",
            "contact_command_physics_model_error",
            "contact_actual_training_model_residual",
            "contact_actual_physics_model_residual",
            "contact_actual_normal_only_model_residual",
            "contact_actual_training_predicted_target_error",
            "contact_actual_physics_predicted_target_error",
            "contact_actual_normal_only_predicted_target_error",
            "contact_force_direction_valid",
            "contact_force_vs_racket_normal_angle_deg",
            "contact_actual_force_direction_model_residual",
            "contact_actual_force_direction_predicted_target_error",
            "contact_physx_data_valid",
            "contact_physx_force",
            "contact_physx_separation",
            "contact_physx_normal_error_deg",
            "contact_physx_point_radial_error",
            "contact_actual_physx_normal_model_residual",
            "contact_actual_physx_normal_predicted_target_error",
            "contact_physx_substep_valid",
            "contact_physx_substep_normal_error_deg",
            "contact_physx_substep_point_radial_error",
            "contact_physx_substep_point_normal_offset",
            "contact_physx_patch_substeps",
            "contact_physx_normal_impulse",
            "contact_physx_capture_lag_s",
            "contact_physx_contact_point_speed_delta",
            "contact_physx_substep_link_model_residual",
            "contact_physx_substep_point_model_residual",
            "contact_physx_substep_outgoing_target_error",
            "contact_physx_measured_restitution",
            "contact_physx_pre_ball_speed",
            "contact_physx_post_ball_speed",
            "contact_physx_point_x",
            "contact_physx_point_y",
            "contact_physx_point_z",
            "contact_physx_normal_x",
            "contact_physx_normal_y",
            "contact_physx_normal_z",
            "contact_physx_pre_ball_velocity_x",
            "contact_physx_pre_ball_velocity_y",
            "contact_physx_pre_ball_velocity_z",
            "contact_physx_post_ball_velocity_x",
            "contact_physx_post_ball_velocity_y",
            "contact_physx_post_ball_velocity_z",
            "contact_physx_pre_racket_point_velocity_x",
            "contact_physx_pre_racket_point_velocity_y",
            "contact_physx_pre_racket_point_velocity_z",
            "contact_physx_post_racket_point_velocity_x",
            "contact_physx_post_racket_point_velocity_y",
            "contact_physx_post_racket_point_velocity_z",
            "contact_physx_pre_ball_angular_velocity_x",
            "contact_physx_pre_ball_angular_velocity_y",
            "contact_physx_pre_ball_angular_velocity_z",
            "contact_physx_post_ball_angular_velocity_x",
            "contact_physx_post_ball_angular_velocity_y",
            "contact_physx_post_ball_angular_velocity_z",
            "contact_physx_pre_ball_surface_tangent_speed",
            "contact_physx_post_ball_spin_speed",
            "contact_impact_normal_error_deg",
            "contact_wire_impact_normal_gap_deg",
            "contact_face_radial_error",
            "contact_time_to_strike",
        )
        shadow_contact_metric_sums = {
            name: torch.zeros((), device=base_env.device)
            for name in shadow_contact_metric_names
        }
        shadow_contact_metric_values = {
            name: [] for name in shadow_contact_metric_names
        }
        shadow_contact_bucket_values = {
            bucket: {
                name: [] for name in shadow_contact_metric_names
            }
            for bucket in ("usable_center", "rim_or_edge")
        }
        shadow_landing_metric_names = (
            "landing_target_error",
            "landing_target_x_error",
            "landing_target_y_error",
        )
        shadow_landing_metric_sums = {
            name: torch.zeros((), device=base_env.device)
            for name in shadow_landing_metric_names
        }
        shadow_landing_metric_values = {
            name: [] for name in shadow_landing_metric_names
        }
        shadow_refresh_metric_names = (
            "command_refresh_tts_s",
            "command_refresh_position_delta",
            "command_refresh_incoming_velocity_delta",
            "command_refresh_timing_delta_s",
            "command_refresh_racket_velocity_delta",
            "command_refresh_normal_delta_deg",
        )
        shadow_refresh_metric_values = {
            name: [] for name in shadow_refresh_metric_names
        }
        shadow_contact_sample_count = scalar_counter()
        shadow_landing_sample_count = scalar_counter()
        shadow_launch_timing_abs_sum = torch.zeros(
            (), device=base_env.device
        )
        shadow_launch_sample_count = scalar_counter()
        shadow_policy_reset_count = scalar_counter()
        shadow_resolved_counts = {
            "total": scalar_counter(),
            "contact": scalar_counter(),
            "net_cross": scalar_counter(),
            "opponent_bounce": scalar_counter(),
            "no_contact": scalar_counter(),
            "contact_no_net": scalar_counter(),
            "net_no_opponent_bounce": scalar_counter(),
            "landing_short": scalar_counter(),
            "landing_long": scalar_counter(),
            "landing_side": scalar_counter(),
            "landing_no_net": scalar_counter(),
        }
        shadow_cycle_active = torch.zeros(
            args.num_envs, dtype=torch.bool, device=base_env.device
        )
        shadow_seen = {
            name: torch.zeros(
                args.num_envs, dtype=torch.bool, device=base_env.device
            )
            for name in shadow_duplicate_counts
        }
        shadow_recovery_pending = torch.zeros(
            args.num_envs, dtype=torch.bool, device=base_env.device
        )
        shadow_recovery_elapsed = torch.zeros(
            args.num_envs, dtype=torch.long, device=base_env.device
        )
        shadow_recovery_consecutive = torch.zeros(
            args.num_envs, dtype=torch.long, device=base_env.device
        )
        shadow_recovery_net_latch = torch.zeros(
            args.num_envs, dtype=torch.bool, device=base_env.device
        )
        shadow_recovery_opponent_latch = torch.zeros(
            args.num_envs, dtype=torch.bool, device=base_env.device
        )
        shadow_recovery_deadline_steps = max(
            1,
            int(
                round(
                    float(
                        cmd.cfg.post_contact_ready_durable_deadline_s
                    )
                    / float(base_env.step_dt)
                )
            ),
        )
        shadow_recovery_required_steps = int(
            cmd.cfg.post_contact_ready_durable_required_consecutive_steps
        )
        shadow_recovery_counts = {
            "attempts": scalar_counter(),
            "resolved": scalar_counter(),
            "success": scalar_counter(),
            "failure": scalar_counter(),
            "reset_failure": scalar_counter(),
            "interrupted_by_next_contact": scalar_counter(),
            "success_with_net_cross": scalar_counter(),
            "success_with_opponent_bounce": scalar_counter(),
        }
        shadow_recovery_metric_names = (
            "peak_base_tilt",
            "peak_torso_ang_vel",
            "peak_base_ang_vel",
            "peak_height_error",
            "peak_com_x",
            "peak_com_y",
            "peak_station_error",
            "minimum_feet_contact",
        )
        shadow_recovery_current = {
            name: torch.zeros(args.num_envs, device=base_env.device)
            for name in shadow_recovery_metric_names
        }
        shadow_recovery_current["minimum_feet_contact"].fill_(1.0)
        shadow_recovery_resolved_values = {
            name: [] for name in shadow_recovery_metric_names
        }
        for step in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            target_pos, racket_pos, racket_vel, tts, swing = read_state()
            reset_now = dones.reshape(-1).to(
                dtype=torch.bool, device=tts.device
            )
            shadow_policy_reset_count += reset_now.sum()
            if physical_shadow is not None:
                events = {
                    name: getattr(physical_shadow, f"{name}_event")
                    for name in shadow_event_names
                }
                serve_event = events["serve"]
                shadow_cycle_active[serve_event] = True
                for seen in shadow_seen.values():
                    seen[serve_event] = False
                shadow_launch_timing_abs_sum += (
                    physical_shadow.launch_timing_error_s[
                        serve_event
                    ].abs().sum()
                )
                shadow_launch_sample_count += serve_event.sum()

                for name, event in events.items():
                    shadow_event_counts[name] += event.sum()
                refresh_event = events["command_refresh"]
                for name in shadow_refresh_metric_names:
                    shadow_refresh_metric_values[name].append(
                        getattr(physical_shadow, name)[refresh_event]
                    )
                for name, seen in shadow_seen.items():
                    event = events[name]
                    shadow_duplicate_counts[name] += (event & seen).sum()
                    seen |= event

                shadow_order_violations["bounce_without_serve"] += (
                    events["incoming_bounce"] & (~shadow_cycle_active)
                ).sum()
                shadow_order_violations["contact_without_bounce"] += (
                    events["contact"] & (~shadow_seen["incoming_bounce"])
                ).sum()
                shadow_order_violations["net_without_contact"] += (
                    events["net_cross"] & (~shadow_seen["contact"])
                ).sum()
                shadow_order_violations[
                    "opponent_bounce_without_net"
                ] += (
                    events["opponent_bounce"]
                    & (~shadow_seen["net_cross"])
                ).sum()
                shadow_order_violations["landing_without_contact"] += (
                    events["outgoing_landing"]
                    & (~shadow_seen["contact"])
                ).sum()
                contact_event = events["contact"]
                contact_count = contact_event.sum()
                shadow_contact_sample_count += contact_count
                exact_contact = (
                    physical_shadow.contact_physx_substep_valid > 0.5
                )
                center_radial_error = torch.where(
                    exact_contact,
                    physical_shadow.contact_physx_substep_point_radial_error,
                    physical_shadow.contact_face_radial_error,
                )
                center_contact_event = contact_event & (
                    center_radial_error <= 0.061
                )
                for name in shadow_contact_metric_names:
                    samples = getattr(physical_shadow, name)[contact_event]
                    shadow_contact_metric_sums[name] += samples.sum()
                    shadow_contact_metric_values[name].append(samples)
                    shadow_contact_bucket_values["usable_center"][
                        name
                    ].append(
                        getattr(physical_shadow, name)[
                            center_contact_event
                        ]
                    )
                    shadow_contact_bucket_values["rim_or_edge"][
                        name
                    ].append(
                        getattr(physical_shadow, name)[
                            contact_event & (~center_contact_event)
                        ]
                    )
                landing_event = events["outgoing_landing"]
                shadow_landing_sample_count += landing_event.sum()
                for name in shadow_landing_metric_names:
                    samples = getattr(physical_shadow, name)[landing_event]
                    shadow_landing_metric_sums[name] += samples.sum()
                    shadow_landing_metric_values[name].append(samples)
                base_tilt = torch.linalg.norm(
                    robot.data.projected_gravity_b[:, :2], dim=-1
                )
                torso_ang_vel = cmd.metrics[
                    "impact_health_torso_ang_vel"
                ]
                base_ang_vel = cmd.metrics["recovery_base_ang_vel"]
                height_error = cmd.metrics["recovery_height_error"]
                com_x = cmd.metrics["impact_health_com_x"]
                com_y = cmd.metrics["impact_health_com_y"]
                feet_contact = cmd.metrics["recovery_feet_contact_frac"]
                station_error = cmd.metrics["recovery_station_error"]

                interrupted = contact_event & shadow_recovery_pending
                shadow_recovery_counts[
                    "interrupted_by_next_contact"
                ] += interrupted.sum()
                shadow_recovery_pending[interrupted] = False

                shadow_recovery_pending[contact_event] = True
                shadow_recovery_elapsed[contact_event] = 0
                shadow_recovery_consecutive[contact_event] = 0
                shadow_recovery_net_latch[contact_event] = False
                shadow_recovery_opponent_latch[contact_event] = False
                shadow_recovery_counts["attempts"] += contact_count

                recovery_samples = {
                    "peak_base_tilt": base_tilt,
                    "peak_torso_ang_vel": torso_ang_vel,
                    "peak_base_ang_vel": base_ang_vel,
                    "peak_height_error": height_error,
                    "peak_com_x": com_x,
                    "peak_com_y": com_y,
                    "peak_station_error": station_error,
                    "minimum_feet_contact": feet_contact,
                }
                for name, values in recovery_samples.items():
                    if name == "minimum_feet_contact":
                        shadow_recovery_current[name][contact_event] = (
                            values[contact_event]
                        )
                    else:
                        shadow_recovery_current[name][contact_event] = (
                            values[contact_event]
                        )

                shadow_recovery_net_latch |= (
                    events["net_cross"] & shadow_recovery_pending
                )
                shadow_recovery_opponent_latch |= (
                    events["opponent_bounce"]
                    & shadow_recovery_pending
                )
                recovery_active = (
                    shadow_recovery_pending & (~contact_event)
                )
                shadow_recovery_elapsed[recovery_active] += 1
                for name, values in recovery_samples.items():
                    if name == "minimum_feet_contact":
                        shadow_recovery_current[name][recovery_active] = (
                            torch.minimum(
                                shadow_recovery_current[name][recovery_active],
                                values[recovery_active],
                            )
                        )
                    else:
                        shadow_recovery_current[name][recovery_active] = (
                            torch.maximum(
                                shadow_recovery_current[name][recovery_active],
                                values[recovery_active],
                            )
                        )

                first_terminal_window_step = (
                    shadow_recovery_deadline_steps
                    - shadow_recovery_required_steps
                    + 1
                )
                in_terminal_window = (
                    recovery_active
                    & (
                        shadow_recovery_elapsed
                        >= first_terminal_window_step
                    )
                    & (
                        shadow_recovery_elapsed
                        <= shadow_recovery_deadline_steps
                    )
                )
                operational_ready = (
                    in_terminal_window
                    & (
                        base_tilt
                        <= float(
                            cmd.cfg
                            .post_contact_ready_operational_max_tilt
                        )
                    )
                    & (
                        torso_ang_vel
                        <= float(
                            cmd.cfg
                            .post_contact_ready_operational_max_torso_ang_vel
                        )
                    )
                    & (
                        base_ang_vel
                        <= float(
                            cmd.cfg
                            .post_contact_ready_operational_max_base_ang_vel
                        )
                    )
                    & (
                        height_error
                        <= float(
                            cmd.cfg
                            .post_contact_ready_operational_max_height_error
                        )
                    )
                    & (
                        com_x
                        <= float(
                            cmd.cfg.post_contact_ready_operational_max_com_x
                        )
                    )
                    & (
                        com_y
                        <= float(
                            cmd.cfg.post_contact_ready_operational_max_com_y
                        )
                    )
                    & (
                        feet_contact
                        >= float(
                            cmd.cfg
                            .post_contact_ready_operational_min_feet_contact
                        )
                    )
                    & (
                        station_error
                        <= float(
                            cmd.cfg
                            .post_contact_ready_operational_max_station_error
                        )
                    )
                )
                shadow_recovery_consecutive = torch.where(
                    recovery_active,
                    torch.where(
                        operational_ready,
                        shadow_recovery_consecutive + 1,
                        torch.zeros_like(shadow_recovery_consecutive),
                    ),
                    shadow_recovery_consecutive,
                )
                reset_failure = shadow_recovery_pending & reset_now
                deadline_reached = (
                    shadow_recovery_pending
                    & (~reset_failure)
                    & (
                        shadow_recovery_elapsed
                        >= shadow_recovery_deadline_steps
                    )
                )
                recovery_success = deadline_reached & (
                    shadow_recovery_consecutive
                    >= shadow_recovery_required_steps
                )
                recovery_failure = deadline_reached & (~recovery_success)
                recovery_resolved = (
                    reset_failure | recovery_success | recovery_failure
                )
                resolved_count = recovery_resolved.sum()
                shadow_recovery_counts["resolved"] += resolved_count
                shadow_recovery_counts["success"] += recovery_success.sum()
                shadow_recovery_counts["failure"] += recovery_failure.sum()
                shadow_recovery_counts["reset_failure"] += (
                    reset_failure.sum()
                )
                shadow_recovery_counts[
                    "success_with_net_cross"
                ] += (
                    (
                        recovery_success
                        & shadow_recovery_net_latch
                    ).sum()
                )
                shadow_recovery_counts[
                    "success_with_opponent_bounce"
                ] += (
                    (
                        recovery_success
                        & shadow_recovery_opponent_latch
                    ).sum()
                )
                for name in shadow_recovery_metric_names:
                    shadow_recovery_resolved_values[name].append(
                        shadow_recovery_current[name][recovery_resolved]
                    )
                shadow_recovery_pending[recovery_resolved] = False
                terminal_event = (
                    events["outgoing_landing"]
                    | events["abort"]
                    | events["timeout"]
                    | events["incoming_net_collision"]
                    | events["reset_abort"]
                )
                resolved = terminal_event & shadow_cycle_active
                resolved_contact = resolved & shadow_seen["contact"]
                resolved_net = resolved & shadow_seen["net_cross"]
                resolved_opponent = (
                    resolved & shadow_seen["opponent_bounce"]
                )
                shadow_resolved_counts["total"] += resolved.sum()
                shadow_resolved_counts["contact"] += resolved_contact.sum()
                shadow_resolved_counts["net_cross"] += resolved_net.sum()
                shadow_resolved_counts[
                    "opponent_bounce"
                ] += resolved_opponent.sum()
                shadow_resolved_counts["no_contact"] += (
                    resolved & (~shadow_seen["contact"])
                ).sum()
                shadow_resolved_counts["contact_no_net"] += (
                    resolved
                    & shadow_seen["contact"]
                    & (~shadow_seen["net_cross"])
                ).sum()
                shadow_resolved_counts[
                    "net_no_opponent_bounce"
                ] += (
                    (
                        resolved
                        & shadow_seen["net_cross"]
                        & (~shadow_seen["opponent_bounce"])
                    ).sum()
                )
                for name in (
                    "landing_short",
                    "landing_long",
                    "landing_side",
                    "landing_no_net",
                ):
                    shadow_resolved_counts[name] += (
                        resolved & events[name]
                    ).sum()
                shadow_cycle_active[terminal_event] = False
            # A strike happens when the reference clock crosses the strike frame (tts: >0 -> <=0).
            # Environments that RESET this step are excluded: a time-out/fall reset re-seeds the
            # clock, and counting it would contaminate the denominator with non-swings.
            struck = (prev_tts > 0.0) & (tts <= 0.0) & (~reset_now)
            if trace_writer is not None:
                e = trace_env
                gravity = robot.data.projected_gravity_b[e]
                base_pos = robot.data.root_pos_w[e] - env_origins[e]
                base_lin = robot.data.root_lin_vel_w[e]
                base_ang = robot.data.root_ang_vel_w[e]
                row = {
                    "step": step,
                    "env": e,
                    "time_to_strike": float(tts[e].item()),
                    "struck": int(struck[e].item()),
                    "done": int(reset_now[e].item()),
                    "swing_side": float(swing[e].item()),
                    "base_x": float(base_pos[0].item()),
                    "base_y": float(base_pos[1].item()),
                    "base_z": float(base_pos[2].item()),
                    "base_pitch_like_deg": float(torch.rad2deg(torch.asin(torch.clamp(gravity[0], -1.0, 1.0))).item()),
                    "base_roll_like_deg": float(torch.rad2deg(torch.asin(torch.clamp(gravity[1], -1.0, 1.0))).item()),
                    "base_lin_vel_x": float(base_lin[0].item()),
                    "base_lin_vel_y": float(base_lin[1].item()),
                    "base_lin_vel_z": float(base_lin[2].item()),
                    "base_ang_vel_x": float(base_ang[0].item()),
                    "base_ang_vel_y": float(base_ang[1].item()),
                    "base_ang_vel_z": float(base_ang[2].item()),
                    "racket_x": float((racket_pos[e, 0] - env_origins[e, 0]).item()),
                    "racket_y": float((racket_pos[e, 1] - env_origins[e, 1]).item()),
                    "racket_z": float((racket_pos[e, 2] - env_origins[e, 2]).item()),
                    "target_x": float((target_pos[e, 0] - env_origins[e, 0]).item()),
                    "target_y": float((target_pos[e, 1] - env_origins[e, 1]).item()),
                    "target_z": float((target_pos[e, 2] - env_origins[e, 2]).item()),
                }
                for name in trace_metric_names:
                    value = cmd.metrics.get(name)
                    row[name] = "" if value is None else float(value[e].item())
                for name in motion_metric_names:
                    value = motion_cmd.metrics.get(name)
                    row[name] = "" if value is None else float(value[e].item())
                trace_writer.writerow(row)
            if joint_diag_writer is not None:
                e = trace_env
                in_hold = bool(motion_cmd.in_hold[e].item())
                in_strike = bool(cmd.strike_window[e].item())
                in_pre = bool(cmd.pre_strike[e].item())
                if in_hold:
                    phase = "hold"
                elif in_strike:
                    phase = "strike"
                elif in_pre:
                    phase = "pre_strike"
                else:
                    phase = "recovery"
                raw = action_term.applied_raw_actions[e]
                effective = action_term.effective_raw_actions[e]
                feedback = action_term.feedback_actions[e]
                overflow = action_term.overflow_actions[e]
                q_des = action_term.processed_actions[e]
                q = robot.data.joint_pos[e, action_joint_ids]
                qd = robot.data.joint_vel[e, action_joint_ids]
                torque_requested = robot.data.computed_torque[e, action_joint_ids]
                torque_applied = robot.data.applied_torque[e, action_joint_ids]
                values_by_prefix = {
                    "raw": raw,
                    "effective": effective,
                    "feedback": feedback,
                    "overflow": overflow,
                    "position_clamped": (torch.abs(overflow) > 1.0e-6).to(torch.int8),
                    "q": q,
                    "qd": qd,
                    "q_des": q_des,
                    "tracking_error": q_des - q,
                    "torque_requested": torque_requested,
                    "torque_applied": torque_applied,
                    "torque_clipped": (
                        torch.abs(torque_requested - torque_applied) > 1.0e-6
                    ).to(torch.int8),
                }
                base_pos = robot.data.root_pos_w[e] - env_origins[e]
                gravity = robot.data.projected_gravity_b[e]
                joint_row = {
                    "step": step,
                    "env": e,
                    "phase": phase,
                    "feedback_mode": action_term.feedback_mode,
                    "time_to_strike": float(tts[e].item()),
                    "swing_side": float(swing[e].item()),
                    "done": int(reset_now[e].item()),
                    "base_x": float(base_pos[0].item()),
                    "base_y": float(base_pos[1].item()),
                    "base_z": float(base_pos[2].item()),
                    "base_pitch_like_deg": float(
                        torch.rad2deg(torch.asin(torch.clamp(gravity[0], -1.0, 1.0))).item()
                    ),
                    "base_ang_vel_norm": float(torch.norm(robot.data.root_ang_vel_w[e]).item()),
                    "feet_contact_fraction": float(cmd.feet_contact_frac[e].item()),
                }
                for prefix, values in values_by_prefix.items():
                    for name, value in zip(action_joint_names, values):
                        joint_row[f"{prefix}__{name}"] = float(value.item())
                joint_diag_writer.writerow(joint_row)
            idx = struck.nonzero(as_tuple=False).flatten().tolist()
            for e in idx:
                tp = to_table_frame(target_pos[e], e)
                rp = to_table_frame(racket_pos[e], e)
                rv = racket_vel[e].cpu().numpy().astype(float)  # achieved racket velocity at strike
                outcome = evaluate_return(tp, rp, rv, physics, table, contact_radius=args.contact_radius)
                analytic_accumulator.add(outcome)
                if all(hasattr(cmd, name) for name in ("ball_contact", "ball_net_cross", "ball_on_opponent")):
                    task_counts["contact"] += int(bool(cmd.ball_contact[e].item()))
                    task_counts["net_cross"] += int(bool(cmd.ball_net_cross[e].item()))
                    task_counts["opponent_bounce"] += int(bool(cmd.ball_on_opponent[e].item()))
                    accumulator.add_bool(bool(cmd.ball_on_opponent[e].item()))
                else:
                    accumulator.add(outcome)
            prev_tts = tts.clone()

        result = accumulator.as_dict()
        if accumulator.attempts:
            result.update(
                {
                    "task_contact_rate": task_counts["contact"] / accumulator.attempts,
                    "task_net_cross_rate": task_counts["net_cross"] / accumulator.attempts,
                    "task_opponent_bounce_rate": task_counts["opponent_bounce"] / accumulator.attempts,
                    "analytic_success_rate": analytic_accumulator.value,
                }
            )
        print(json.dumps(result))
        if args.json_out:
            result_path = pathlib.Path(args.json_out)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("w", encoding="utf-8") as f:
                json.dump(result, f)
                f.write("\n")
        if args.physical_shadow_json_out:
            shadow_event_counts = {
                name: int(value.item())
                for name, value in shadow_event_counts.items()
            }
            shadow_duplicate_counts = {
                name: int(value.item())
                for name, value in shadow_duplicate_counts.items()
            }
            shadow_order_violations = {
                name: int(value.item())
                for name, value in shadow_order_violations.items()
            }
            shadow_resolved_counts = {
                name: int(value.item())
                for name, value in shadow_resolved_counts.items()
            }
            shadow_recovery_counts = {
                name: int(value.item())
                for name, value in shadow_recovery_counts.items()
            }
            shadow_contact_sample_count = int(
                shadow_contact_sample_count.item()
            )
            shadow_landing_sample_count = int(
                shadow_landing_sample_count.item()
            )
            shadow_launch_sample_count = int(
                shadow_launch_sample_count.item()
            )
            shadow_policy_reset_count = int(
                shadow_policy_reset_count.item()
            )
            shadow_launch_timing_abs_sum = float(
                shadow_launch_timing_abs_sum.item()
            )
            shadow_contact_metric_sums = {
                name: float(value.item())
                for name, value in shadow_contact_metric_sums.items()
            }
            shadow_landing_metric_sums = {
                name: float(value.item())
                for name, value in shadow_landing_metric_sums.items()
            }
            serve_denominator = max(shadow_event_counts["serve"], 1)
            contact_denominator = max(shadow_contact_sample_count, 1)
            landing_denominator = max(shadow_landing_sample_count, 1)
            route_generation_count = (
                shadow_event_counts["serve"]
                + shadow_event_counts["route_invalid"]
                + shadow_event_counts["route_unalignable"]
            )
            resolved_denominator = max(
                shadow_resolved_counts["total"], 1
            )
            contact_metric_quantiles = {}
            for name, values in shadow_contact_metric_values.items():
                samples = torch.cat(values) if values else None
                if samples is not None and samples.numel() > 0:
                    contact_metric_quantiles[name] = {
                        "p50": float(torch.quantile(samples, 0.50).item()),
                        "p90": float(torch.quantile(samples, 0.90).item()),
                    }
                else:
                    contact_metric_quantiles[name] = {
                        "p50": 0.0,
                        "p90": 0.0,
                    }
            contact_metric_buckets = {}
            for bucket, metric_values in shadow_contact_bucket_values.items():
                concatenated = {
                    name: torch.cat(values) if values else None
                    for name, values in metric_values.items()
                }
                first_samples = concatenated[
                    shadow_contact_metric_names[0]
                ]
                sample_count = (
                    int(first_samples.numel())
                    if first_samples is not None
                    else 0
                )
                contact_metric_buckets[bucket] = {
                    "sample_count": sample_count,
                    "means": {
                        name: (
                            float(samples.mean().item())
                            if samples is not None
                            and samples.numel() > 0
                            else 0.0
                        )
                        for name, samples in concatenated.items()
                    },
                    "quantiles": {
                        name: {
                            "p50": (
                                float(
                                    torch.quantile(
                                        samples, 0.50
                                    ).item()
                                )
                                if samples is not None
                                and samples.numel() > 0
                                else 0.0
                            ),
                            "p90": (
                                float(
                                    torch.quantile(
                                        samples, 0.90
                                    ).item()
                                )
                                if samples is not None
                                and samples.numel() > 0
                                else 0.0
                            ),
                        }
                        for name, samples in concatenated.items()
                    },
                }
            contact_metric_sample_names = (
                "contact_physx_substep_valid",
                "contact_physx_substep_point_radial_error",
                "contact_physx_point_x",
                "contact_physx_point_y",
                "contact_physx_point_z",
                "contact_physx_normal_x",
                "contact_physx_normal_y",
                "contact_physx_normal_z",
                "contact_physx_pre_ball_velocity_x",
                "contact_physx_pre_ball_velocity_y",
                "contact_physx_pre_ball_velocity_z",
                "contact_physx_post_ball_velocity_x",
                "contact_physx_post_ball_velocity_y",
                "contact_physx_post_ball_velocity_z",
                "contact_physx_pre_racket_point_velocity_x",
                "contact_physx_pre_racket_point_velocity_y",
                "contact_physx_pre_racket_point_velocity_z",
                "contact_physx_post_racket_point_velocity_x",
                "contact_physx_post_racket_point_velocity_y",
                "contact_physx_post_racket_point_velocity_z",
                "contact_physx_pre_ball_angular_velocity_x",
                "contact_physx_pre_ball_angular_velocity_y",
                "contact_physx_pre_ball_angular_velocity_z",
                "contact_physx_post_ball_angular_velocity_x",
                "contact_physx_post_ball_angular_velocity_y",
                "contact_physx_post_ball_angular_velocity_z",
                "contact_physx_pre_ball_surface_tangent_speed",
                "contact_physx_post_ball_spin_speed",
                "contact_physx_substep_normal_error_deg",
                "contact_physx_substep_point_model_residual",
                "contact_physx_substep_link_model_residual",
            )
            contact_metric_samples = {
                name: (
                    torch.cat(shadow_contact_metric_values[name])
                    .detach()
                    .cpu()
                    .tolist()
                    if shadow_contact_metric_values[name]
                    else []
                )
                for name in contact_metric_sample_names
            }
            landing_metric_quantiles = {}
            for name, values in shadow_landing_metric_values.items():
                samples = torch.cat(values) if values else None
                if samples is not None and samples.numel() > 0:
                    landing_metric_quantiles[name] = {
                        "p10": float(torch.quantile(samples, 0.10).item()),
                        "p50": float(torch.quantile(samples, 0.50).item()),
                        "p90": float(torch.quantile(samples, 0.90).item()),
                    }
                else:
                    landing_metric_quantiles[name] = {
                        "p10": 0.0,
                        "p50": 0.0,
                        "p90": 0.0,
                    }
            refresh_metric_summary = {}
            for name, values in shadow_refresh_metric_values.items():
                samples = torch.cat(values) if values else None
                if samples is not None and samples.numel() > 0:
                    refresh_metric_summary[name] = {
                        "mean": float(samples.mean().item()),
                        "p50": float(torch.quantile(samples, 0.50).item()),
                        "p90": float(torch.quantile(samples, 0.90).item()),
                    }
                else:
                    refresh_metric_summary[name] = {
                        "mean": 0.0,
                        "p50": 0.0,
                        "p90": 0.0,
                    }
            recovery_metric_summary = {}
            for name, values in shadow_recovery_resolved_values.items():
                samples = torch.cat(values) if values else None
                if samples is not None and samples.numel() > 0:
                    recovery_metric_summary[name] = {
                        "mean": float(samples.mean().item()),
                        "p50": float(torch.quantile(samples, 0.50).item()),
                        "p90": float(torch.quantile(samples, 0.90).item()),
                    }
                else:
                    recovery_metric_summary[name] = {
                        "mean": 0.0,
                        "p50": 0.0,
                        "p90": 0.0,
                    }
            recovery_resolved_denominator = max(
                shadow_recovery_counts["resolved"], 1
            )
            shadow_result = {
                "evaluation_seed": int(args.seed),
                "num_envs": int(args.num_envs),
                "num_steps": int(args.num_steps),
                "physx_substep_capture": {
                    "registered": bool(
                        physical_shadow.physics_substep_capture_registered
                    ),
                    "disabled": bool(
                        physical_shadow._substep_capture_disabled
                    ),
                    "capture_error": physical_shadow._substep_capture_error,
                    "contact_view_error": (
                        physical_shadow._detailed_contact_init_error
                    ),
                    "physics_dt_s": float(base_env.scene.physics_dt),
                },
                "event_counts": shadow_event_counts,
                "event_rates_per_serve": {
                    name: count / serve_denominator
                    for name, count in shadow_event_counts.items()
                    if name
                    not in (
                        "serve",
                        "route_invalid",
                        "route_unalignable",
                    )
                },
                "route_generation": {
                    "attempts": route_generation_count,
                    "served": shadow_event_counts["serve"],
                    "invalid": shadow_event_counts["route_invalid"],
                    "unalignable": shadow_event_counts[
                        "route_unalignable"
                    ],
                    "served_rate": (
                        shadow_event_counts["serve"]
                        / max(route_generation_count, 1)
                    ),
                },
                "resolved_cycles": {
                    "counts": shadow_resolved_counts,
                    "rates": {
                        name: count / resolved_denominator
                        for name, count in shadow_resolved_counts.items()
                        if name != "total"
                    },
                    "active_at_eval_end": int(
                        shadow_cycle_active.sum().item()
                    ),
                },
                "duplicate_event_counts": shadow_duplicate_counts,
                "event_order_violations": shadow_order_violations,
                "policy_reset_count": shadow_policy_reset_count,
                "mean_abs_launch_timing_error_s": (
                    shadow_launch_timing_abs_sum
                    / max(shadow_launch_sample_count, 1)
                ),
                "contact_sample_count": shadow_contact_sample_count,
                "contact_metric_means": {
                    name: value / contact_denominator
                    for name, value in shadow_contact_metric_sums.items()
                },
                "contact_metric_quantiles": contact_metric_quantiles,
                "contact_metric_buckets": contact_metric_buckets,
                "contact_metric_samples": contact_metric_samples,
                "landing_sample_count": shadow_landing_sample_count,
                "landing_metric_means": {
                    name: value / landing_denominator
                    for name, value in shadow_landing_metric_sums.items()
                },
                "landing_metric_quantiles": landing_metric_quantiles,
                "command_refresh": {
                    "count": shadow_event_counts["command_refresh"],
                    "metrics": refresh_metric_summary,
                },
                "physical_contact_recovery": {
                    "deadline_s": (
                        shadow_recovery_deadline_steps
                        * float(base_env.step_dt)
                    ),
                    "required_consecutive_steps": (
                        shadow_recovery_required_steps
                    ),
                    "counts": shadow_recovery_counts,
                    "rates_per_resolved": {
                        name: count / recovery_resolved_denominator
                        for name, count in shadow_recovery_counts.items()
                        if name
                        in (
                            "success",
                            "failure",
                            "reset_failure",
                            "success_with_net_cross",
                            "success_with_opponent_bounce",
                        )
                    },
                    "pending_at_eval_end": int(
                        shadow_recovery_pending.sum().item()
                    ),
                    "resolved_metric_summary": recovery_metric_summary,
                },
                "analytic_eval": result,
            }
            shadow_path = pathlib.Path(args.physical_shadow_json_out)
            shadow_path.parent.mkdir(parents=True, exist_ok=True)
            with shadow_path.open("w", encoding="utf-8") as f:
                json.dump(shadow_result, f, indent=2, sort_keys=True)
                f.write("\n")
        if args.safety_envelope_json_out:
            attempts = int(
                cmd._post_contact_ready_envelope_attempts.item()
            )
            component_names = (
                "tilt",
                "abs_pitch",
                "com_x",
                "com_y",
                "waist_overflow",
                "leg_overflow",
                "base_ang_vel",
            )
            summary_counts = (
                cmd._post_contact_ready_envelope_violations.detach()
                .cpu()
                .tolist()
            )
            histogram_thresholds = (
                cmd._post_contact_ready_envelope_histogram_thresholds.detach()
                .cpu()
                .tolist()
            )
            histogram_counts = (
                cmd._post_contact_ready_envelope_histogram_counts.detach()
                .cpu()
                .tolist()
            )
            phase_attempt_counts = (
                cmd._post_contact_ready_phase_attempt_counts.detach()
                .cpu()
                .tolist()
            )
            phase_histogram_counts = (
                cmd._post_contact_ready_phase_histogram_counts.detach()
                .cpu()
                .tolist()
            )
            outcome_resolution_counts = (
                cmd._post_contact_ready_outcome_resolution_counts.detach()
                .cpu()
                .tolist()
            )
            resolution_latency_steps = (
                cmd._post_contact_ready_resolution_latency_steps.detach()
                .cpu()
                .tolist()
            )
            resolution_counts = (
                cmd._post_contact_ready_resolution_counts.detach()
                .cpu()
                .tolist()
            )
            durable_outcome_resolution_counts = (
                cmd._post_contact_ready_durable_outcome_resolution_counts.detach()
                .cpu()
                .tolist()
            )
            durable_resolution_latency_steps = (
                cmd._post_contact_ready_durable_resolution_latency_steps.detach()
                .cpu()
                .tolist()
            )
            durable_resolution_counts = (
                cmd._post_contact_ready_durable_resolution_counts.detach()
                .cpu()
                .tolist()
            )
            durable_failure_component_counts = (
                cmd._post_contact_ready_durable_failure_component_counts.detach()
                .cpu()
                .tolist()
            )
            terminal_settlement_counts = (
                cmd._post_contact_ready_terminal_settlement_counts.detach()
                .cpu()
                .tolist()
            )
            terminal_quality_sums = (
                cmd._post_contact_ready_terminal_quality_sums.detach()
                .cpu()
                .tolist()
            )
            safe_net_cycle_count = int(
                cmd._post_contact_ready_safe_net_cycle_count.detach()
                .cpu()
                .item()
            )
            denominator = max(attempts, 1)
            phase_names = (
                "impact_0_100ms",
                "brake_100_300ms",
                "settle_300_600ms",
                "ready_after_600ms",
            )
            safety_result = {
                "evaluation_seed": int(args.seed),
                "attempts": attempts,
                "violation_count": int(summary_counts[0]),
                "violation_rate": summary_counts[0] / denominator,
                "component_violation_counts": {
                    name: int(summary_counts[index + 1])
                    for index, name in enumerate(
                        (
                            "tilt",
                            "abs_pitch",
                            "com",
                            "waist_overflow",
                            "leg_overflow",
                            "base_ang_vel",
                        )
                    )
                },
                "histograms": {
                    name: [
                        {
                            "threshold": float(threshold),
                            "count": int(count),
                            "rate": int(count) / denominator,
                        }
                        for threshold, count in zip(
                            histogram_thresholds[index],
                            histogram_counts[index],
                        )
                    ]
                    for index, name in enumerate(component_names)
                },
                "phase_boundaries_s": (
                    cmd._post_contact_ready_phase_boundaries_s.detach()
                    .cpu()
                    .tolist()
                ),
                "phase_histograms": {
                    phase_name: {
                        "attempts_reaching_phase": int(
                            phase_attempt_counts[phase_index]
                        ),
                        "histograms": {
                            component_name: [
                                {
                                    "threshold": float(threshold),
                                    "count": int(count),
                                    "rate": int(count)
                                    / max(
                                        int(
                                            phase_attempt_counts[
                                                phase_index
                                            ]
                                        ),
                                        1,
                                    ),
                                }
                                for threshold, count in zip(
                                    histogram_thresholds[component_index],
                                    phase_histogram_counts[phase_index][
                                        component_index
                                    ],
                                )
                            ]
                            for component_index, component_name in enumerate(
                                component_names
                            )
                        },
                    }
                    for phase_index, phase_name in enumerate(phase_names)
                },
                "outcome_resolution": {
                    outcome_name: {
                        "attempts": int(values[0]),
                        "ready_success": int(values[1]),
                        "ready_fail": int(values[2]),
                        "unresolved": max(
                            int(values[0])
                            - int(values[1])
                            - int(values[2]),
                            0,
                        ),
                        "ready_success_rate": int(values[1])
                        / max(int(values[0]), 1),
                    }
                    for outcome_name, values in zip(
                        (
                            "targeted_miss",
                            "contact",
                            "net_cross",
                            "opponent_bounce",
                        ),
                        outcome_resolution_counts,
                    )
                },
                "resolution_latency": {
                    name: {
                        "count": int(resolution_counts[index]),
                        "mean_steps": int(resolution_latency_steps[index])
                        / max(int(resolution_counts[index]), 1),
                        "mean_seconds": (
                            int(resolution_latency_steps[index])
                            / max(int(resolution_counts[index]), 1)
                            * float(base_env.step_dt)
                        ),
                    }
                    for index, name in enumerate(
                        ("ready_success", "ready_fail")
                    )
                },
                "durable_ready_shadow": {
                    "definition": {
                        "minimum_delay_s": float(
                            cmd.cfg.post_contact_ready_durable_min_delay_s
                        ),
                        "fixed_deadline_s": float(
                            cmd.cfg.post_contact_ready_durable_deadline_s
                        ),
                        "required_consecutive_steps": int(
                            cmd.cfg.post_contact_ready_durable_required_consecutive_steps
                        ),
                        "step_dt_s": float(base_env.step_dt),
                        "affects_reward_or_termination": False,
                    },
                    "outcome_resolution": {
                        outcome_name: {
                            "attempts": int(values[0]),
                            "ready_success": int(values[1]),
                            "ready_fail": int(values[2]),
                            "unresolved": max(
                                int(values[0])
                                - int(values[1])
                                - int(values[2]),
                                0,
                            ),
                            "ready_success_rate": int(values[1])
                            / max(int(values[0]), 1),
                        }
                        for outcome_name, values in zip(
                            (
                                "targeted_miss",
                                "contact",
                                "net_cross",
                                "opponent_bounce",
                            ),
                            durable_outcome_resolution_counts,
                        )
                    },
                    "resolution_latency": {
                        name: {
                            "count": int(
                                durable_resolution_counts[index]
                            ),
                            "mean_steps": int(
                                durable_resolution_latency_steps[index]
                            )
                            / max(
                                int(durable_resolution_counts[index]),
                                1,
                            ),
                            "mean_seconds": (
                                int(
                                    durable_resolution_latency_steps[
                                        index
                                    ]
                                )
                                / max(
                                    int(
                                        durable_resolution_counts[
                                            index
                                        ]
                                    ),
                                    1,
                                )
                                * float(base_env.step_dt)
                            ),
                        }
                        for index, name in enumerate(
                            ("ready_success", "ready_fail")
                        )
                    },
                    "failure_components": {
                        name: {
                            "count": int(
                                durable_failure_component_counts[index]
                            ),
                            "rate_per_failure": int(
                                durable_failure_component_counts[index]
                            )
                            / max(
                                int(durable_resolution_counts[1]),
                                1,
                            ),
                        }
                        for index, name in enumerate(
                            (
                                "backlean_limit",
                                "forward_lean_limit",
                                "torso_ang_vel",
                                "base_lin_vel",
                                "base_ang_vel",
                                "racket_speed",
                                "height",
                                "com_x",
                                "com_y",
                                "feet_contact",
                                "station",
                                "arm_ready",
                            )
                        )
                    },
                },
                "safe_quality_terminal": {
                    "settlements": int(terminal_settlement_counts[0]),
                    "safe_settlements": int(terminal_settlement_counts[1]),
                    "unsafe_settlements": int(
                        terminal_settlement_counts[2]
                    ),
                    "incomplete_settlements": int(
                        terminal_settlement_counts[3]
                    ),
                    "safe_rate": int(terminal_settlement_counts[1])
                    / max(int(terminal_settlement_counts[0]), 1),
                    "unsafe_rate": int(terminal_settlement_counts[2])
                    / max(int(terminal_settlement_counts[0]), 1),
                    "incomplete_rate": int(terminal_settlement_counts[3])
                    / max(int(terminal_settlement_counts[0]), 1),
                    "mean_terminal_quality": float(
                        terminal_quality_sums[0]
                    )
                    / max(int(terminal_settlement_counts[0]), 1),
                    "mean_safe_terminal_quality": float(
                        terminal_quality_sums[1]
                    )
                    / max(int(terminal_settlement_counts[1]), 1),
                    "safe_net_cycles": safe_net_cycle_count,
                    "safe_net_cycle_rate": safe_net_cycle_count
                    / max(int(terminal_settlement_counts[0]), 1),
                },
            }
            target = pathlib.Path(args.safety_envelope_json_out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(safety_result, indent=2) + "\n",
                encoding="utf-8",
            )
        if trace_fh is not None:
            trace_fh.close()
        if joint_diag_fh is not None:
            joint_diag_fh.close()
        env.close()
    except Exception:
        import traceback

        print("\n[evaluate] ERROR:", flush=True)
        traceback.print_exc()
        status = 1
    finally:
        simulation_app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
