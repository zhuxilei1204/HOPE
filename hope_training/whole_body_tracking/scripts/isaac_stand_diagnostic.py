"""Isaac no-ball/hold standing diagnostic for HOPE checkpoints.

This is intentionally close to ``evaluate.py`` but records base stability and termination
terms instead of return success.  In ``ready-hold`` mode it forces episode resets to the
default stand and freezes the motion command in a long hold.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

import yaml
from omegaconf import OmegaConf


_PACKAGE_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"
)
if str(_PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SOURCE))


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir():
            return parent
    return here.parents[2]


def _resolve_task_yaml(task_yaml: str) -> pathlib.Path:
    path = pathlib.Path(str(task_yaml)).expanduser()
    if path.is_file():
        return path.resolve()
    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    return (pathlib.Path(__file__).resolve().parents[1] / "cfg" / "task" / path.name).resolve()


def _load_task_yaml_with_defaults(task_yaml_path: pathlib.Path, seen: set[pathlib.Path] | None = None):
    seen = set() if seen is None else seen
    task_yaml_path = task_yaml_path.resolve()
    if task_yaml_path in seen:
        raise ValueError(f"cyclic task YAML defaults include {task_yaml_path}")
    seen.add(task_yaml_path)
    with task_yaml_path.open("r", encoding="utf-8") as fh:
        task_doc = yaml.safe_load(fh) or {}

    merged = OmegaConf.create({})
    for item in task_doc.get("defaults") or []:
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
    task_yaml_path = _resolve_task_yaml(task_yaml)
    if not task_yaml_path.is_file():
        return []
    cfg = OmegaConf.create({"task": _load_task_yaml_with_defaults(task_yaml_path)})
    from train import _apply_task_overrides

    applied: list[str] = []
    _apply_task_overrides(env_cfg, cfg, applied)
    return applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Gym task id. By default this is read from task-yaml.gym_task so "
            "the evaluator cannot silently instantiate a different base task."
        ),
    )
    parser.add_argument("--task-yaml", default="HOPEPingPong.yaml")
    parser.add_argument("--motion-manifest", default=None)
    parser.add_argument("--motion-file", default="hope_training/motions/preprocessed/hope_forehand.npz")
    parser.add_argument("--motion-file-2", default="hope_training/motions/preprocessed/hope_backhand.npz")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=3000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--mode", choices=["normal", "ready-hold"], default="normal")
    parser.add_argument("--trace-csv", default=None)
    parser.add_argument("--trace-env", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--experiment-name", default="hope_pingpong")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

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
        from train import _apply_motion_metadata, _resolve_motion_plan
        from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
        from whole_body_tracking.utils.ppo_cfg import load_ppo_params, runner_kwargs

        env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
        env_cfg.seed = int(args.seed)
        applied = _apply_training_task_overrides(env_cfg, args.task_yaml)
        clips, motion_metadata = _resolve_motion_plan(args)
        env_cfg.commands.motion.motion_file = clips if len(clips) > 1 else clips[0]
        _apply_motion_metadata(env_cfg, clips, motion_metadata, applied)
        if args.mode == "ready-hold":
            env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args.num_steps * 0.02 + 5.0)
            env_cfg.commands.motion.stand_episode_prob = 1.0
            env_cfg.commands.motion.stand_episode_hold_steps = args.num_steps + 10
            env_cfg.commands.motion.stand_start_prob = 1.0
            env_cfg.commands.motion.stand_start_min_hold = args.num_steps + 10
            env_cfg.commands.motion.hold_steps_range = (args.num_steps + 10, args.num_steps + 10)
            env_cfg.commands.motion.motion_start_warmup_enabled = False
            env_cfg.commands.motion.wrap_teleport = False
            env_cfg.commands.racket_target.deploy_ready_force_stand_episode = True
            applied.extend(
                [
                    "diagnostic: env.episode_length_s >= rollout length",
                    "diagnostic: commands.motion.stand_episode_prob = 1.0",
                    "diagnostic: commands.motion.stand_start_prob = 1.0",
                    "diagnostic: commands.motion.hold_steps_range = long fixed hold",
                    "diagnostic: commands.motion.motion_start_warmup_enabled = False",
                    "diagnostic: commands.motion.wrap_teleport = False",
                    "diagnostic: commands.racket_target.deploy_ready_force_stand_episode = True",
                ]
            )

        env_raw = gym.make(task_id, cfg=env_cfg, render_mode=None)
        base_env = env_raw.unwrapped
        env = RslRlVecEnvWrapper(env_raw)

        agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(load_ppo_params(), args.experiment_name))
        agent_cfg.device = args.device
        runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=base_env.device)

        robot = base_env.scene["robot"]
        term_names = list(base_env.termination_manager.active_terms)
        term_counts = {name: 0 for name in term_names}
        reset_count = 0
        reset_counts_per_env = torch.zeros(
            args.num_envs, dtype=torch.long, device=base_env.device
        )
        first_done_steps = torch.full(
            (args.num_envs,), -1, dtype=torch.long, device=base_env.device
        )
        max_height_err = 0.0
        max_upright_err = 0.0
        max_lin_vel = 0.0
        max_ang_vel = 0.0
        min_base_z = 1.0e9
        first_manual_low = None
        first_manual_tilt = None
        first_done = None
        no_command_steps = 0
        stand_episode_steps = 0

        trace_fh = None
        writer = None
        if args.trace_csv:
            p = pathlib.Path(args.trace_csv)
            p.parent.mkdir(parents=True, exist_ok=True)
            trace_fh = p.open("w", encoding="utf-8", newline="")
            fields = [
                "step",
                "time_s",
                "done",
                "base_z",
                "height_err",
                "upright_err",
                "base_lin_vel_xy",
                "base_ang_vel",
                "episode_length",
                "motion_in_hold",
                "motion_stand_episode",
                "motion_default_stand_reset",
                "no_command_ready_active",
                "time_to_strike",
                "racket_target_error",
                "station_error",
                "recovery_ready_score",
            ] + [f"term_{name}" for name in term_names]
            writer = csv.DictWriter(trace_fh, fieldnames=fields)
            writer.writeheader()

        obs, _ = env.get_observations()
        cmd = base_env.command_manager.get_term("racket_target")
        motion = base_env.command_manager.get_term("motion")
        default_z = robot.data.default_root_state[:, 2] + base_env.scene.env_origins[:, 2]
        initial_racket_rel_base = None
        initial_target_rel_base = None
        initial_racket_target_error = None
        initial_station_error = None
        for step in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _rew, dones, _extras = env.step(actions)
            if initial_racket_rel_base is None:
                initial_racket_rel_base = (
                    cmd.racket_pos_w - cmd.base_pos_w
                ).detach().cpu()
                initial_target_rel_base = (
                    cmd.racket_target_pos_w - cmd.base_pos_w
                ).detach().cpu()
                initial_racket_target_error = torch.norm(
                    cmd.racket_target_pos_w - cmd.racket_pos_w, dim=-1
                ).detach().cpu()
                initial_station_error = torch.norm(
                    cmd.base_pos_w[:, :2] - cmd.fixed_station_w, dim=-1
                ).detach().cpu()
            no_command_steps += int(
                torch.count_nonzero(cmd.no_command_ready_active).item()
            )
            stand_episode_steps += int(
                torch.count_nonzero(motion.stand_episode).item()
            )
            reset_count += int(torch.count_nonzero(dones).item())
            done_mask = dones > 0
            reset_counts_per_env += done_mask.long()
            first_failure = done_mask & (first_done_steps < 0)
            first_done_steps[first_failure] = step + 1
            if first_done is None and bool(torch.any(dones > 0)):
                first_done = (step + 1) * base_env.step_dt
            for name in term_names:
                term_counts[name] += int(torch.count_nonzero(base_env.termination_manager.get_term(name)).item())

            base_z = robot.data.root_pos_w[:, 2]
            height_err = torch.abs(base_z - default_z)
            upright_err = torch.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
            lin_vel = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
            ang_vel = torch.norm(robot.data.root_ang_vel_w, dim=-1)
            max_height_err = max(max_height_err, float(torch.max(height_err).item()))
            max_upright_err = max(max_upright_err, float(torch.max(upright_err).item()))
            max_lin_vel = max(max_lin_vel, float(torch.max(lin_vel).item()))
            max_ang_vel = max(max_ang_vel, float(torch.max(ang_vel).item()))
            min_base_z = min(min_base_z, float(torch.min(base_z).item()))
            if first_manual_low is None and bool(torch.any(base_z < 0.55)):
                first_manual_low = (step + 1) * base_env.step_dt
            if first_manual_tilt is None and bool(torch.any(upright_err > 0.85)):
                first_manual_tilt = (step + 1) * base_env.step_dt

            if writer is not None:
                e = min(max(int(args.trace_env), 0), args.num_envs - 1)
                row = {
                    "step": step + 1,
                    "time_s": float((step + 1) * base_env.step_dt),
                    "done": int(dones[e].item()),
                    "base_z": float(base_z[e].item()),
                    "height_err": float(height_err[e].item()),
                    "upright_err": float(upright_err[e].item()),
                    "base_lin_vel_xy": float(lin_vel[e].item()),
                    "base_ang_vel": float(ang_vel[e].item()),
                    "episode_length": int(base_env.episode_length_buf[e].item()),
                    "motion_in_hold": int(motion.in_hold[e].item()),
                    "motion_stand_episode": int(motion.stand_episode[e].item()),
                    "motion_default_stand_reset": int(
                        motion.default_stand_reset[e].item()
                    ),
                    "no_command_ready_active": int(
                        cmd.no_command_ready_active[e].item()
                    ),
                    "time_to_strike": float(cmd.time_to_strike[e].item()),
                    "racket_target_error": float(
                        torch.norm(
                            cmd.racket_target_pos_w[e] - cmd.racket_pos_w[e]
                        ).item()
                    ),
                    "station_error": float(
                        torch.norm(
                            cmd.base_pos_w[e, :2] - cmd.fixed_station_w[e]
                        ).item()
                    ),
                    "recovery_ready_score": float(cmd.metrics["recovery_ready_score"][e].item()),
                }
                for name in term_names:
                    row[f"term_{name}"] = int(base_env.termination_manager.get_term(name)[e].item())
                writer.writerow(row)

        if trace_fh is not None:
            trace_fh.close()
        total_env_steps = max(1, args.num_envs * args.num_steps)
        if args.mode == "ready-hold" and (
            no_command_steps != total_env_steps
            or stand_episode_steps != total_env_steps
        ):
            raise RuntimeError(
                "ready-hold contract violated: expected every sample to be a "
                "stand episode with no planner command, got "
                f"no_command={no_command_steps}/{total_env_steps}, "
                f"stand_episode={stand_episode_steps}/{total_env_steps}"
            )
        first_done_steps_cpu = first_done_steps.cpu().tolist()
        survival_times = sorted(
            (
                args.num_steps if value < 0 else int(value)
            )
            * float(base_env.step_dt)
            for value in first_done_steps_cpu
        )
        reset_counts_cpu = sorted(reset_counts_per_env.cpu().tolist())

        def _nearest_rank(values, quantile):
            index = int(round((len(values) - 1) * float(quantile)))
            return values[min(max(index, 0), len(values) - 1)]

        survived_full_count = sum(value < 0 for value in first_done_steps_cpu)
        if initial_racket_rel_base is None:
            raise RuntimeError("diagnostic rollout produced no samples")
        env.close()

        result = {
            "task_id": task_id,
            "mode": args.mode,
            "num_envs": int(args.num_envs),
            "num_steps": int(args.num_steps),
            "seconds": float(args.num_steps * base_env.step_dt),
            "reset_count": int(reset_count),
            "reset_rate_per_env_step": float(reset_count / max(1, args.num_envs * args.num_steps)),
            "mean_resets_per_env": float(reset_count / max(1, args.num_envs)),
            "p90_resets_per_env": int(_nearest_rank(reset_counts_cpu, 0.90)),
            "no_command_fraction": float(no_command_steps / total_env_steps),
            "stand_episode_fraction": float(stand_episode_steps / total_env_steps),
            "full_horizon_survival_count": int(survived_full_count),
            "full_horizon_survival_rate": float(
                survived_full_count / max(1, args.num_envs)
            ),
            "survival_time_p10_s": float(_nearest_rank(survival_times, 0.10)),
            "survival_time_median_s": float(
                _nearest_rank(survival_times, 0.50)
            ),
            "survival_time_p90_s": float(_nearest_rank(survival_times, 0.90)),
            "initial_racket_rel_base_mean": [
                float(value)
                for value in torch.mean(initial_racket_rel_base, dim=0).tolist()
            ],
            "initial_target_rel_base_mean": [
                float(value)
                for value in torch.mean(initial_target_rel_base, dim=0).tolist()
            ],
            "initial_racket_target_error_mean_m": float(
                torch.mean(initial_racket_target_error).item()
            ),
            "initial_racket_target_error_p90_m": float(
                torch.quantile(initial_racket_target_error, 0.90).item()
            ),
            "initial_station_error_mean_m": float(
                torch.mean(initial_station_error).item()
            ),
            "first_done_s": None if first_done is None else float(first_done),
            "first_manual_low_s": None if first_manual_low is None else float(first_manual_low),
            "first_manual_tilt_s": None if first_manual_tilt is None else float(first_manual_tilt),
            "min_base_z": float(min_base_z),
            "max_height_err": float(max_height_err),
            "max_upright_err": float(max_upright_err),
            "max_base_lin_vel_xy": float(max_lin_vel),
            "max_base_ang_vel": float(max_ang_vel),
            "termination_counts": term_counts,
            "applied_overrides": applied,
        }
        print(json.dumps(result))
        if args.json_out:
            pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
    except Exception as exc:
        print(f"[isaac_stand_diagnostic] ERROR: {exc}", file=sys.stderr, flush=True)
        status = 1
    finally:
        simulation_app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
