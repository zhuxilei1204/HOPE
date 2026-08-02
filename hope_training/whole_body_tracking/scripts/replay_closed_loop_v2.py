#!/usr/bin/env python3
"""Deterministic reward/lifecycle replay using the training helper functions."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "closed_loop_v2.py"
)
_SPEC = importlib.util.spec_from_file_location("hope_closed_loop_v2_replay", _MODULE_PATH)
closed_loop = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = closed_loop
_SPEC.loader.exec_module(closed_loop)

ClosedLoopPhase = closed_loop.ClosedLoopPhase
closed_cycle_success_event = closed_loop.closed_cycle_success_event
durable_ready_resolution_events = (
    closed_loop.durable_ready_resolution_events
)
health_floor_multiplier = closed_loop.health_floor_multiplier
lifecycle_phase_ids = closed_loop.lifecycle_phase_ids
operational_terminal_events = closed_loop.operational_terminal_events
planner_velocity_alignment_score = closed_loop.planner_velocity_alignment_score
recovered_planner_velocity_settlement_score = (
    closed_loop.recovered_planner_velocity_settlement_score
)
recovery_peak_excess_increment = closed_loop.recovery_peak_excess_increment
recovery_trigger_event = closed_loop.recovery_trigger_event
safety_conditioned_cycle_value = (
    closed_loop.safety_conditioned_cycle_value
)
safety_conditioned_outcome_value = (
    closed_loop.safety_conditioned_outcome_value
)
terminal_quality_window_mask = closed_loop.terminal_quality_window_mask


def _close_list(
    actual: list[float],
    expected: list[float],
    tolerance: float = 1.0e-6,
) -> bool:
    return len(actual) == len(expected) and all(
        abs(float(left) - float(right)) <= float(tolerance)
        for left, right in zip(actual, expected)
    )


def _scenario() -> list[dict]:
    # First attempt misses but must recover. The second clears the net while
    # healthy and must settle exactly one complete-cycle event.
    return [
        {"no_command": 1, "tts": 1.0},
        {"no_command": 0, "tts": 0.8},
        {"no_command": 0, "tts": 0.1},
        {"no_command": 0, "tts": 0.0, "strike": 1, "targeted": 1},
        {"no_command": 0, "tts": -0.1},
        {"no_command": 0, "tts": -0.3, "ready": 1},
        {"no_command": 1, "tts": 1.0},
        {"no_command": 0, "tts": 0.8},
        {
            "no_command": 0,
            "tts": 0.0,
            "strike": 1,
            "targeted": 1,
            "contact": 1,
            "net": 1,
            "healthy": 1,
        },
        {"no_command": 0, "tts": -0.1},
        {"no_command": 0, "tts": -0.3, "ready": 1},
        {"no_command": 1, "tts": 1.0},
    ]


def replay(rows: list[dict]) -> tuple[list[dict], dict]:
    pending = False
    outcome_tier = -1
    healthy_net = False
    unsafe = False
    output = []

    for index, row in enumerate(rows):
        strike = bool(row.get("strike", 0))
        targeted = bool(row.get("targeted", 0))
        contact = bool(row.get("contact", 0))
        net = bool(row.get("net", 0))
        healthy = bool(row.get("healthy", 0))
        trigger = bool(
            recovery_trigger_event(
                "targeted_attempt",
                strike_fired=torch.tensor([strike]),
                targeted_attempt=torch.tensor([targeted]),
                ball_contact=torch.tensor([contact]),
            )[0]
        )
        if trigger:
            pending = True
            outcome_tier = 1 if net else (0 if contact else -1)
            healthy_net = net and healthy
            unsafe = False

        recovery_success = pending and bool(row.get("ready", 0))
        cycle_success = bool(
            closed_cycle_success_event(
                recovery_success=torch.tensor([recovery_success]),
                outcome_tier=torch.tensor([outcome_tier]),
                healthy_net_cross=torch.tensor([healthy_net]),
                unsafe=torch.tensor([unsafe]),
            )[0]
        )
        phase = ClosedLoopPhase(
            int(
                lifecycle_phase_ids(
                    no_command=torch.tensor([bool(row.get("no_command", 0))]),
                    time_to_strike=torch.tensor([float(row["tts"])]),
                    strike_window=torch.tensor([strike]),
                    recovery_pending=torch.tensor([pending]),
                    recovery_success=torch.tensor([recovery_success]),
                )[0]
            )
        )
        physical_outcome = (contact + 2.0 * net) * float(
            health_floor_multiplier(
                torch.tensor([1.0 if healthy else 0.0]), 0.25
            )[0]
        )
        output.append(
            {
                "step": index,
                "phase": phase.name,
                "recovery_trigger": int(trigger),
                "recovery_pending": int(pending),
                "recovery_success": int(recovery_success),
                "cycle_success": int(cycle_success),
                "physical_outcome": physical_outcome,
            }
        )
        if recovery_success:
            pending = False

    durable_success, durable_fail = durable_ready_resolution_events(
        pending=torch.tensor([True, True, True, True]),
        elapsed_steps=torch.tensor([20, 55, 55, 55]),
        deadline_steps=55,
        consecutive_ready_steps=torch.tensor([15, 15, 14, 15]),
        required_consecutive_steps=15,
    )
    durable_cycle = closed_cycle_success_event(
        recovery_success=durable_success,
        outcome_tier=torch.tensor([1, 1, 1, 1]),
        healthy_net_cross=torch.tensor([True, True, True, True]),
        unsafe=torch.tensor([False, False, False, True]),
    )
    terminal_window = terminal_quality_window_mask(
        pending=torch.tensor([True, True, True, True]),
        elapsed_steps=torch.tensor([40, 41, 54, 55]),
        deadline_steps=55,
        window_steps=15,
    )
    terminal_safe, terminal_unsafe, terminal_incomplete = (
        operational_terminal_events(
        settlement_event=torch.tensor([True, True, True, True]),
        catastrophic_violation_latch=torch.tensor(
            [False, False, True, False]
        ),
        operational_ready=torch.tensor([True, True, True, False]),
        )
    )
    terminal_quality = torch.tensor([1.0, 0.4, 1.0, 1.0])
    terminal_tier = torch.tensor([2, 1, 2, -1])
    terminal_outcome = safety_conditioned_outcome_value(
        safe_settlement_event=terminal_safe,
        terminal_quality=terminal_quality,
        outcome_tier=terminal_tier,
        tier_multipliers=torch.tensor([0.5, 1.0, 1.5]),
    )
    terminal_cycle = safety_conditioned_cycle_value(
        safe_settlement_event=terminal_safe,
        terminal_quality=terminal_quality,
        outcome_tier=terminal_tier,
    )
    summary = {
        "steps": len(output),
        "recovery_triggers": sum(row["recovery_trigger"] for row in output),
        "recovery_successes": sum(row["recovery_success"] for row in output),
        "closed_cycle_successes": sum(row["cycle_success"] for row in output),
        "phase_counts": {
            phase.name: sum(row["phase"] == phase.name for row in output)
            for phase in ClosedLoopPhase
        },
        "planner_velocity_score": {
            "aligned": float(
                planner_velocity_alignment_score(
                    torch.tensor([[2.0, 0.2, 0.8]]),
                    torch.tensor([[2.0, 0.2, 0.8]]),
                )[0]
            ),
            "underspeed": float(
                planner_velocity_alignment_score(
                    torch.tensor([[1.0, 0.1, 0.4]]),
                    torch.tensor([[2.0, 0.2, 0.8]]),
                )[0]
            ),
            "stationary": float(
                planner_velocity_alignment_score(
                    torch.zeros(1, 3),
                    torch.tensor([[2.0, 0.2, 0.8]]),
                )[0]
            ),
        },
        "recovered_planner_velocity_score": {
            "controlled": float(
                recovered_planner_velocity_settlement_score(
                    speed_ratio=torch.tensor([1.0]),
                    direction_error_rad=torch.tensor([0.0]),
                    position_error=torch.tensor([0.02]),
                    impact_health_score=torch.tensor([1.0]),
                    recovery_peak_base_ang_vel=torch.tensor([0.2]),
                )[0]
            ),
            "unstable": float(
                recovered_planner_velocity_settlement_score(
                    speed_ratio=torch.tensor([1.0]),
                    direction_error_rad=torch.tensor([0.0]),
                    position_error=torch.tensor([0.02]),
                    impact_health_score=torch.tensor([1.0]),
                    recovery_peak_base_ang_vel=torch.tensor([1.6]),
                )[0]
            ),
        },
        "recovery_peak_excess_cost": {
            "controlled": float(
                recovery_peak_excess_increment(
                    torch.tensor([0.2]),
                    torch.tensor([0.7]),
                    torch.tensor([0.8]),
                )[0]
            ),
            "new_excess": float(
                recovery_peak_excess_increment(
                    torch.tensor([0.8]),
                    torch.tensor([1.4]),
                    torch.tensor([0.8]),
                )[0]
            ),
            "held_peak": float(
                recovery_peak_excess_increment(
                    torch.tensor([1.4]),
                    torch.tensor([1.4]),
                    torch.tensor([0.8]),
                )[0]
            ),
        },
        "durable_ready_settlement": {
            "early_success": bool(durable_success[0]),
            "deadline_success": bool(durable_success[1]),
            "deadline_failure": bool(durable_fail[2]),
            "safe_cycle_success": bool(durable_cycle[1]),
            "unsafe_cycle_success": bool(durable_cycle[3]),
        },
        "safe_quality_settlement": {
            "window_mask": terminal_window.tolist(),
            "safe_events": terminal_safe.tolist(),
            "unsafe_events": terminal_unsafe.tolist(),
            "incomplete_events": terminal_incomplete.tolist(),
            "outcome_values": terminal_outcome.tolist(),
            "cycle_values": terminal_cycle.tolist(),
        },
    }
    if summary["recovery_triggers"] != 2:
        raise RuntimeError(f"expected two recovery triggers, got {summary}")
    if summary["recovery_successes"] != 2:
        raise RuntimeError(f"expected two recovery settlements, got {summary}")
    if summary["closed_cycle_successes"] != 1:
        raise RuntimeError(f"cycle event was not settled exactly once: {summary}")
    velocity = summary["planner_velocity_score"]
    if not (
        velocity["aligned"] > velocity["underspeed"] > velocity["stationary"]
    ):
        raise RuntimeError(f"planner velocity score is not monotonic: {summary}")
    recovered = summary["recovered_planner_velocity_score"]
    if not recovered["controlled"] > recovered["unstable"]:
        raise RuntimeError(
            f"recovered planner settlement ignores recovery peak: {summary}"
        )
    peak_cost = summary["recovery_peak_excess_cost"]
    if not (
        peak_cost["controlled"] == 0.0
        and peak_cost["new_excess"] > 0.0
        and peak_cost["held_peak"] == 0.0
    ):
        raise RuntimeError(
            f"recovery peak cost is not incremental: {summary}"
        )
    durable = summary["durable_ready_settlement"]
    if (
        durable["early_success"]
        or not durable["deadline_success"]
        or not durable["deadline_failure"]
        or not durable["safe_cycle_success"]
        or durable["unsafe_cycle_success"]
    ):
        raise RuntimeError(
            f"durable cycle settled early or ignored safety: {summary}"
        )
    safe_quality = summary["safe_quality_settlement"]
    if (
        safe_quality["window_mask"] != [False, True, True, True]
        or safe_quality["safe_events"] != [True, True, False, False]
        or safe_quality["unsafe_events"] != [False, False, True, False]
        or safe_quality["incomplete_events"]
        != [False, False, False, True]
        or not _close_list(
            safe_quality["outcome_values"], [1.5, 0.4, 0.0, 0.0]
        )
        or not _close_list(
            safe_quality["cycle_values"], [1.0, 0.4, 0.0, 0.0]
        )
    ):
        raise RuntimeError(
            f"safe terminal quality settlement is invalid: {summary}"
        )
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = replay(_scenario())
    with (args.output_dir / "lifecycle_replay.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "lifecycle_replay.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
