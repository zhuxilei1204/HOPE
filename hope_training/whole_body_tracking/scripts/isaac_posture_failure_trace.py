"""Trace the first Isaac posture failure for a HOPE checkpoint.

This is a diagnostic-only rollout.  It keeps a small ring buffer per env and,
when a hard posture/safety termination fires, writes the pre-failure trajectory
for that env.  It does not change the task, policy, or training code.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from collections import deque


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="HOPE-PingPong-AgibotA3-v0")
    parser.add_argument("--task-yaml", default="HOPEPingPong.yaml")
    parser.add_argument("--motion-manifest", default=None)
    parser.add_argument("--motion-file", default="hope_training/motions/preprocessed/hope_forehand.npz")
    parser.add_argument("--motion-file-2", default="hope_training/motions/preprocessed/hope_backhand.npz")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1800)
    parser.add_argument("--history-steps", type=int, default=220)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--experiment-name", default="hope_pingpong")
    parser.add_argument(
        "--terms",
        nargs="+",
        default=["base_too_low", "base_tilted", "ee_body_pos", "table_touch"],
        help="Termination terms that should trigger a trace dump.",
    )
    return parser.parse_args()


def _scalar_metric(cmd, name: str, env_id: int, default: float = 0.0) -> float:
    value = getattr(cmd, "metrics", {}).get(name)
    if value is None:
        return default
    try:
        return float(value[env_id].item())
    except Exception:
        return default


def main() -> int:
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

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
        from isaac_stand_diagnostic import _apply_training_task_overrides
        from train import _apply_motion_metadata, _resolve_motion_plan
        from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
        from whole_body_tracking.utils.ppo_cfg import load_ppo_params, runner_kwargs

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        applied = _apply_training_task_overrides(env_cfg, args.task_yaml)
        clips, motion_metadata = _resolve_motion_plan(args)
        env_cfg.commands.motion.motion_file = clips if len(clips) > 1 else clips[0]
        _apply_motion_metadata(env_cfg, clips, motion_metadata, applied)

        env_raw = gym.make(args.task, cfg=env_cfg, render_mode=None)
        base_env = env_raw.unwrapped
        env = RslRlVecEnvWrapper(env_raw)

        agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(load_ppo_params(), args.experiment_name))
        agent_cfg.device = args.device
        runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=base_env.device)

        robot = base_env.scene["robot"]
        cmd = base_env.command_manager.get_term("racket_target")
        motion = base_env.command_manager.get_term("motion")
        default_z = robot.data.default_root_state[:, 2] + base_env.scene.env_origins[:, 2]
        buffers = [deque(maxlen=max(1, int(args.history_steps))) for _ in range(args.num_envs)]
        term_names = list(base_env.termination_manager.active_terms)
        trigger_terms = [name for name in args.terms if name in term_names]

        obs, _ = env.get_observations()
        trigger_info = None
        for step in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)

            base_z = robot.data.root_pos_w[:, 2]
            height_err = torch.abs(base_z - default_z)
            upright_err = torch.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
            lin_vel = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
            ang_vel = torch.norm(robot.data.root_ang_vel_w, dim=-1)
            action_l2 = torch.norm(actions, dim=-1)
            action_max = torch.max(torch.abs(actions), dim=-1).values
            time_s = float(step * base_env.step_dt)

            for env_id in range(args.num_envs):
                buffers[env_id].append(
                    {
                        "step": step,
                        "time_s": time_s,
                        "env_id": env_id,
                        "base_z": float(base_z[env_id].item()),
                        "height_err": float(height_err[env_id].item()),
                        "upright_err": float(upright_err[env_id].item()),
                        "base_lin_vel_xy": float(lin_vel[env_id].item()),
                        "base_ang_vel": float(ang_vel[env_id].item()),
                        "action_l2": float(action_l2[env_id].item()),
                        "action_max": float(action_max[env_id].item()),
                        "time_to_strike": float(cmd.time_to_strike[env_id].item()),
                        "in_hold": int(motion.in_hold[env_id].item()),
                        "recovery_ready_score": _scalar_metric(cmd, "recovery_ready_score", env_id),
                        "recovery_height_error": _scalar_metric(cmd, "recovery_height_error", env_id),
                        "recovery_upright_error": _scalar_metric(cmd, "recovery_upright_error", env_id),
                        "recovery_base_ang_vel": _scalar_metric(cmd, "recovery_base_ang_vel", env_id),
                        "current_net_cross": _scalar_metric(cmd, "current_net_cross", env_id),
                        "return_success": _scalar_metric(cmd, "return_success", env_id),
                    }
                )

            obs, _rew, dones, _extras = env.step(actions)

            fired = []
            for name in trigger_terms:
                mask = base_env.termination_manager.get_term(name)
                if bool(torch.any(mask)):
                    env_ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                    for env_id in env_ids:
                        fired.append((int(env_id), name))
            if fired:
                env_id, term = fired[0]
                trigger_info = {
                    "trigger_step": int(step + 1),
                    "trigger_time_s": float((step + 1) * base_env.step_dt),
                    "trigger_env_id": int(env_id),
                    "trigger_term": term,
                    "triggered_terms": [{"env_id": e, "term": t} for e, t in fired],
                    "dones_count": int(torch.count_nonzero(dones).item()),
                    "applied_overrides": applied,
                }
                break

        rows = list(buffers[trigger_info["trigger_env_id"]]) if trigger_info else []
        out_csv = pathlib.Path(args.csv_out)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "step",
            "time_s",
            "env_id",
            "base_z",
            "height_err",
            "upright_err",
            "base_lin_vel_xy",
            "base_ang_vel",
            "action_l2",
            "action_max",
            "time_to_strike",
            "in_hold",
            "recovery_ready_score",
            "recovery_height_error",
            "recovery_upright_error",
            "recovery_base_ang_vel",
            "current_net_cross",
            "return_success",
        ]
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        result = trigger_info or {
            "trigger_step": None,
            "trigger_time_s": None,
            "trigger_env_id": None,
            "trigger_term": None,
            "triggered_terms": [],
            "applied_overrides": applied,
        }
        result.update({"num_envs": int(args.num_envs), "num_steps": int(args.num_steps), "csv_out": str(out_csv)})
        print(json.dumps(result))
        if args.json_out:
            out_json = pathlib.Path(args.json_out)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        env.close()
    except Exception as exc:
        print(f"[isaac_posture_failure_trace] ERROR: {exc}", file=sys.stderr, flush=True)
        status = 1
    finally:
        simulation_app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
