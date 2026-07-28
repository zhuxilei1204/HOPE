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
    parser.add_argument("--task", default="HOPE-PingPong-AgibotA3-v0", help="Gym task id.")
    parser.add_argument("--num-envs", type=int, default=256, help="Parallel environments.")
    parser.add_argument("--num-steps", type=int, default=4000, help="Policy steps to roll out.")
    parser.add_argument("--device", default="cuda:0", help="Compute device.")
    parser.add_argument("--contact-radius", type=float, default=0.10, help="Racket-to-target contact gate (m).")
    parser.add_argument("--json-out", default=None, help="Also write {'success_rate': ...} to this file.")
    parser.add_argument("--trace-csv", default=None, help="Write one Isaac environment's per-step diagnostics.")
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

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        task_cfg = _load_task_yaml_with_defaults(_resolve_task_yaml(args.task_yaml))
        applied = _apply_training_task_overrides(env_cfg, args.task_yaml)
        motion_args = argparse.Namespace(**vars(args))
        motion_args.task = task_cfg
        clips, motion_metadata = _resolve_motion_plan(motion_args)
        env_cfg.commands.motion.motion_file = clips if len(clips) > 1 else clips[0]
        _apply_motion_metadata(env_cfg, clips, motion_metadata, applied)
        print(f"[evaluate.py] applied {len(applied)} task override(s):", file=sys.stderr, flush=True)
        for line in applied:
            print(f"[evaluate.py]     {line}", file=sys.stderr, flush=True)

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
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
        motion_cmd = base_env.command_manager.get_term("motion")
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
        for step in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            target_pos, racket_pos, racket_vel, tts, swing = read_state()
            # A strike happens when the reference clock crosses the strike frame (tts: >0 -> <=0).
            # Environments that RESET this step are excluded: a time-out/fall reset re-seeds the
            # clock, and counting it would contaminate the denominator with non-swings.
            reset_now = dones.reshape(-1).to(dtype=torch.bool, device=tts.device)
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
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(result, f)
                f.write("\n")
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
