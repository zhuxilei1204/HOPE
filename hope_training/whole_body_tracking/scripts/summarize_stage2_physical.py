#!/usr/bin/env python3
"""Summarize comparable Stage-2 physical-task TensorBoard windows."""

from __future__ import annotations

import argparse
import pathlib
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = {
    "reward": "Train/mean_reward",
    "episode_len": "Train/mean_episode_length",
    "contact_ema": "Metrics/racket_target/physical_ability_contact_ema",
    "aligned_contact_ema": "Metrics/racket_target/physical_ability_aligned_contact_ema",
    "net_ema": "Metrics/racket_target/physical_ability_net_ema",
    "bounce_ema": "Metrics/racket_target/physical_ability_bounce_ema",
    "recovery_ema": "Metrics/racket_target/physical_ability_recovery_ema",
    "safety_ema": "Metrics/racket_target/physical_ability_safety_ema",
    "ability_level": "Metrics/racket_target/physical_ability_level",
    "route_curriculum_level": "Metrics/physical_shadow/route_curriculum_level",
    "contact_threshold": "Metrics/racket_target/physical_ability_contact_threshold",
    "aligned_contact_threshold": "Metrics/racket_target/physical_ability_aligned_contact_threshold",
    "net_threshold": "Metrics/racket_target/physical_ability_net_threshold",
    "planner_speed_ratio": "Metrics/racket_target/impact_planner_speed_ratio",
    "planner_vel_angle_deg": "Metrics/racket_target/impact_planner_vel_angle_deg",
    "racket_normal_error_deg": "Metrics/racket_target/racket_normal_error_deg",
    "outgoing_speed_ratio": "Metrics/physical_shadow/contact_outgoing_speed_ratio",
    "outgoing_dir_error_deg": "Metrics/physical_shadow/contact_outgoing_direction_error_deg",
    "serve_event": "Metrics/physical_shadow/serve_event",
    "contact_event": "Metrics/physical_shadow/contact_event",
    "net_event": "Metrics/physical_shadow/net_cross_event",
    "bounce_event": "Metrics/physical_shadow/opponent_bounce_event",
    "recovery_ready": "Metrics/racket_target/recovery_functional_ready_score",
    "impact_torso_ang_vel": "Metrics/racket_target/impact_health_torso_ang_vel",
    "base_pitch": "Metrics/racket_target/base_pitch_like_signed",
    "action_clamp": "Metrics/racket_target/action_clamp_fraction",
    "waist_action_clamp": "Metrics/racket_target/waist_action_clamp_fraction",
    "right_arm_action_clamp": "Metrics/racket_target/right_arm_action_clamp_fraction",
    "leg_action_clamp": "Metrics/racket_target/leg_action_clamp_fraction",
    "operational_margin": "Metrics/racket_target/operational_margin_fraction",
    "waist_operational_margin": "Metrics/racket_target/waist_operational_margin_fraction",
    "right_arm_operational_margin": "Metrics/racket_target/right_arm_operational_margin_fraction",
    "leg_operational_margin": "Metrics/racket_target/leg_operational_margin_fraction",
    "q_des_velocity_violation": "Metrics/racket_target/q_des_velocity_violation_fraction",
    "q_des_accel_violation": "Metrics/racket_target/q_des_acceleration_violation_fraction",
    "waist_q_des_accel_violation": "Metrics/racket_target/waist_q_des_acceleration_violation_fraction",
    "right_arm_q_des_accel_violation": "Metrics/racket_target/right_arm_q_des_acceleration_violation_fraction",
    "leg_q_des_accel_violation": "Metrics/racket_target/leg_q_des_acceleration_violation_fraction",
    "action_feasibility": "Metrics/racket_target/action_operational_feasibility_score",
    "waist_torque_clip": "Metrics/racket_target/waist_torque_clip_fraction",
    "right_arm_torque_clip": "Metrics/racket_target/right_arm_torque_clip_fraction",
    "leg_torque_clip": "Metrics/racket_target/leg_torque_clip_fraction",
    "unsafe_swing": "Metrics/racket_target/current_swing_unsafe",
    "impact_health": "Metrics/racket_target/impact_health_score",
    "impact_health_ok": "Metrics/racket_target/impact_health_ok",
    "current_swing_contact": "Metrics/racket_target/current_swing_contact",
    "current_swing_healthy_contact": "Metrics/racket_target/current_swing_healthy_contact",
    "current_swing_healthy_net": "Metrics/racket_target/current_swing_healthy_net_cross",
    "physical_contact_health": "Metrics/racket_target/physical_contact_health_score",
    "physical_center_contact_event": "Metrics/racket_target/physical_center_contact_event",
    "physical_aligned_contact_event": "Metrics/racket_target/physical_aligned_contact_event",
    "recovery_nonterminal_failure": "Metrics/racket_target/physical_recovery_nonterminal_failure_event",
    "recovery_terminal_failure": "Metrics/racket_target/physical_recovery_terminal_failure_event",
    "fall_low": "Episode_Termination/base_too_low",
    "fall_tilt": "Episode_Termination/base_tilted",
    "table_touch": "Episode_Termination/table_touch",
    "overflow_done": "Episode_Termination/persistent_action_overflow",
    "velocity_projection_reward": "Episode_Reward/racket_velocity_projection",
    "velocity_progress_reward": "Episode_Reward/near_impact_planner_velocity_progress",
    "taskspace_reward": "Episode_Reward/planner_racket_task_space_crossfade",
    "prestrike_progress_reward": "Episode_Reward/prestrike_racket_progress",
    "physical_outcome_reward": "Episode_Reward/physical_outcome_events",
    "contact_planner_alignment_reward": "Episode_Reward/physical_contact_planner_alignment",
    "physical_recovery_settlement_reward": "Episode_Reward/physical_recovery_settlement",
    "termination_penalty_reward": "Episode_Reward/termination_penalty",
    "strike_balance_reward": "Episode_Reward/strike_balance",
    "imitation_reward": "Episode_Reward/imitation",
    "trunk_support_reward": "Episode_Reward/healthy_trunk_support",
    "lower_support_reward": "Episode_Reward/lower_body_support",
    "recovery_health_reward": "Episode_Reward/recovery_health",
    "value_loss": "Loss/value_function",
    "surrogate_loss": "Loss/surrogate",
    "anchor_rms": "Loss/actor_anchor_rms",
    "learning_rate": "Loss/learning_rate",
}


def _run_arg(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=/absolute/run/path")
    label, raw_path = value.split("=", 1)
    path = pathlib.Path(raw_path).expanduser().resolve()
    if not label or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return label, path


def _mean(events, start: int, end: int) -> float | None:
    values = [float(event.value) for event in events if start <= event.step <= end]
    return statistics.fmean(values) if values else None


def _summarize(
    path: pathlib.Path,
    *,
    window: int,
    requested_start: int | None,
    requested_end: int | None,
) -> tuple[int, int, dict[str, float | None]]:
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    reward_events = accumulator.Scalars(TAGS["reward"])
    if not reward_events:
        raise RuntimeError(f"no training scalars in {path}")
    end = int(reward_events[-1].step) if requested_end is None else int(requested_end)
    start = max(0, end - max(int(window), 1) + 1) if requested_start is None else int(requested_start)
    values = {
        name: (
            _mean(accumulator.Scalars(tag), start, end)
            if tag in available
            else None
        )
        for name, tag in TAGS.items()
    }
    serve = values["serve_event"]
    contact = values["contact_event"]
    net = values["net_event"]
    bounce = values["bounce_event"]
    values["contact_per_serve"] = contact / serve if serve and contact is not None else None
    values["net_per_serve"] = net / serve if serve and net is not None else None
    values["bounce_per_serve"] = bounce / serve if serve and bounce is not None else None
    values["net_per_contact"] = net / contact if contact and net is not None else None
    return start, end, values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_run_arg)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    summaries = []
    for label, path in args.run:
        start, end, values = _summarize(
            path,
            window=args.window,
            requested_start=args.start,
            requested_end=args.end,
        )
        summaries.append((label, start, end, values))

    print("| metric | " + " | ".join(label for label, *_ in summaries) + " |")
    print("|---|" + "---:|" * len(summaries))
    print("| window | " + " | ".join(f"{start}..{end}" for _, start, end, _ in summaries) + " |")
    ordered = list(TAGS) + [
        "contact_per_serve",
        "net_per_serve",
        "bounce_per_serve",
        "net_per_contact",
    ]
    for metric in ordered:
        cells = []
        for _, _, _, values in summaries:
            value = values[metric]
            cells.append("" if value is None else f"{value:.5f}")
        print(f"| {metric} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
