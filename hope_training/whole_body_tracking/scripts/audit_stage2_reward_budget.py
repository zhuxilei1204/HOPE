#!/usr/bin/env python3
"""Audit the resolved Stage-2 reward budget of one training run.

The saved ``params/env.yaml`` is the source of truth for active terms. TensorBoard
episode-reward scalars are used to measure what actually dominated optimization.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CATEGORIES = {
    "motion_prior": {"imitation"},
    "command_dense": {
        "racket_position",
        "racket_velocity",
        "planner_racket_task_space_crossfade",
        "blade_direction",
        "pre_strike_station",
        "prestrike_station_progress",
        "prestrike_racket_progress",
        "near_impact_planner_velocity_progress",
    },
    "stability_dense": {
        "alive",
        "healthy_trunk_support",
        "recovery_health",
        "no_command_ready_stability",
        "post_strike_base_ang_vel",
        "strike_balance",
        "lower_body_support",
        "durable_recovery_progress",
    },
    "recovery_sparse": {
        "durable_recovery_success",
    },
    "physical_sparse": {
        "physical_outcome_events",
        "physical_contact_planner_alignment",
        "physical_recovery_settlement",
    },
    "constraint_cost": {
        "undesired_contacts",
        "feet_contact_slip",
        "table_no_touch",
        "upright",
        "phase_action_overflow",
        "termination_penalty",
        "phase_action_rate_waist",
        "phase_action_rate_upper",
        "phase_action_rate_legs",
        "joint_limit",
        "operational_joint_margin",
        "joint_target_slew",
        "actuator_waist_feasibility",
        "actuator_right_arm_feasibility",
        "actuator_leg_feasibility",
        "durable_recovery_failure",
        "no_command_instability",
        "safe_strike_inactivity",
    },
    "controller_only": {"physical_capability_curriculum"},
}


def _category(term: str) -> str:
    matches = [name for name, terms in CATEGORIES.items() if term in terms]
    if len(matches) > 1:
        raise RuntimeError(f"reward term {term!r} is assigned more than once")
    return matches[0] if matches else "unclassified"


def _mean(events, start: int, end: int) -> float | None:
    values = [float(event.value) for event in events if start <= event.step <= end]
    return sum(values) / len(values) if values else None


def _load_resolved_env(path: Path) -> dict:
    # The file is generated locally by Isaac Lab and contains Python tuple tags.
    with path.open() as stream:
        return yaml.unsafe_load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="End update for the audit window (defaults to the latest update).",
    )
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    env_path = run / "params" / "env.yaml"
    if not env_path.is_file():
        raise FileNotFoundError(env_path)
    env = _load_resolved_env(env_path)

    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", ()))
    train_events = accumulator.Scalars("Train/mean_reward")
    if not train_events:
        raise RuntimeError(f"no Train/mean_reward scalars in {run}")
    latest = int(train_events[-1].step)
    end = latest if args.end_step is None else min(int(args.end_step), latest)
    start = max(0, end - max(int(args.window), 1) + 1)

    active = {
        name: term
        for name, term in env["rewards"].items()
        if term is not None and float(term.get("weight", 0.0)) != 0.0
    }
    values: dict[str, float] = {}
    for name in active:
        tag = f"Episode_Reward/{name}"
        if tag in scalar_tags:
            value = _mean(accumulator.Scalars(tag), start, end)
            if value is not None:
                values[name] = value

    positive = sum(value for value in values.values() if value > 0.0)
    negative_abs = -sum(value for value in values.values() if value < 0.0)
    grouped: dict[str, float] = defaultdict(float)
    grouped_positive: dict[str, float] = defaultdict(float)
    grouped_negative_abs: dict[str, float] = defaultdict(float)
    for name, value in values.items():
        category = _category(name)
        grouped[category] += value
        if value > 0.0:
            grouped_positive[category] += value
        elif value < 0.0:
            grouped_negative_abs[category] += -value

    unclassified = sorted(name for name in active if _category(name) == "unclassified")
    missing_logs = sorted(name for name in active if name not in values)

    sim = env.get("sim", {})
    physics_dt = float(sim.get("dt", 0.0))
    control_dt = physics_dt * int(env.get("decimation", 1))

    print(f"# Reward budget: {run.name}")
    print(f"window: {start}..{end}; active terms: {len(active)}; control_dt: {control_dt:.6f} s")
    print("\n| category | observed reward/s | positive share | negative share |")
    print("|---|---:|---:|---:|")
    for category in (*CATEGORIES, "unclassified"):
        value = grouped.get(category, 0.0)
        pos_share = grouped_positive.get(category, 0.0) / positive if positive else 0.0
        neg_share = (
            grouped_negative_abs.get(category, 0.0) / negative_abs
            if negative_abs
            else 0.0
        )
        print(f"| {category} | {value:+.8f} | {pos_share:.3%} | {neg_share:.3%} |")
    print(f"| **total** | {sum(values.values()):+.8f} | 100.000% | 100.000% |")

    print("\n| term | category | weight | observed reward/s | positive share | raw=1 one-step value |")
    print("|---|---|---:|---:|---:|---:|")
    ordered = sorted(active, key=lambda name: -abs(values.get(name, 0.0)))
    for name in ordered:
        weight = float(active[name]["weight"])
        observed = values.get(name, 0.0)
        share = observed / positive if observed > 0.0 and positive else 0.0
        print(
            f"| {name} | {_category(name)} | {weight:+.5f} | "
            f"{observed:+.8f} | {share:.3%} | {weight * control_dt:+.6f} |"
        )

    physical_positive = grouped_positive.get("physical_sparse", 0.0)
    largest_dense_category = max(
        (
            (category, value / positive)
            for category, value in grouped_positive.items()
            if category not in {"physical_sparse", "constraint_cost", "controller_only"}
            and value > 0.0
            and positive > 0.0
        ),
        key=lambda item: item[1],
        default=("none", 0.0),
    )
    largest_dense_term = max(
        (
            (name, value / positive)
            for name, value in values.items()
            if _category(name)
            not in {"physical_sparse", "constraint_cost", "controller_only"}
            and value > 0.0
            and positive > 0.0
        ),
        key=lambda item: item[1],
        default=("none", 0.0),
    )
    physical_share = physical_positive / positive if positive else 0.0
    impulse_terms = sorted(
        name
        for name, term in active.items()
        if bool(term.get("params", {}).get("impulse", False))
    )
    print("\n## Audit gates")
    print(f"- observed physical positive share: {physical_share:.3%}")
    print(f"- impulse-corrected one-shot terms: {', '.join(impulse_terms)}")
    print(
        f"- largest dense category: {largest_dense_category[0]} "
        f"({largest_dense_category[1]:.3%})"
    )
    print(
        f"- largest dense term: {largest_dense_term[0]} "
        f"({largest_dense_term[1]:.3%})"
    )
    print(
        "- long-run budget gate: "
        + (
            "PASS"
            if physical_share >= 0.10
            and largest_dense_category[1] <= 0.45
            and largest_dense_term[1] <= 0.25
            else "FAIL"
        )
    )
    if unclassified:
        print(f"- unclassified active terms: {', '.join(unclassified)}")
    if missing_logs:
        print(f"- active terms without scalars in this window: {', '.join(missing_logs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
