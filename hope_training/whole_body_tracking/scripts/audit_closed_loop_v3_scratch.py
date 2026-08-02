#!/usr/bin/env python3
"""Audit a resolved closed-loop-v3 scratch run before promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


EXPECTED_NONZERO_REWARDS = {
    "undesired_contacts",
    "feet_contact_slip",
    "table_no_touch",
    "upright",
    "imitation",
    "phase_lower_body_motion_prior",
    "racket_wrist_motion_pos",
    "racket_wrist_motion_ori",
    "racket_position",
    "planner_racket_task_space_crossfade",
    "prestrike_racket_progress",
    "prestrike_station_progress",
    "near_impact_planner_velocity_progress",
    "exact_impact_planner_task_space_alignment",
    "health_gated_soft_ball_contact",
    "targeted_strike_attempt",
    "cycle_v2_ready_success_bonus",
    "cycle_v2_streak_bonus",
    "cycle_v2_ready_fail",
    "phase_action_overflow",
    "termination_penalty",
    "phase_action_rate_waist",
    "phase_action_rate_upper",
    "phase_action_rate_legs",
    "joint_limit",
    "physical_outcome",
    "no_command_instability",
    "active_ready_sustained_bonus",
    "post_contact_directional_recovery",
    "safe_strike_inactivity",
    "actuator_waist_feasibility",
    "actuator_right_arm_feasibility",
    "actuator_leg_feasibility",
}
EXPECTED_NONZERO_REWARDS_READY_POTENTIAL_V5 = (
    EXPECTED_NONZERO_REWARDS | {"no_command_ready_progress"}
)
EXPECTED_NONZERO_REWARDS_BALANCED_READY_V6D = (
    EXPECTED_NONZERO_REWARDS_READY_POTENTIAL_V5
    | {"no_command_ready_balance"}
)
EXPECTED_NONZERO_REWARDS_READY_SURVIVAL_V6F = (
    EXPECTED_NONZERO_REWARDS_READY_POTENTIAL_V5
    | {"active_ready_survival_milestone_bonus"}
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.unsafe_load(stream)


def audit(run_dir: Path, profile: str = "ready-contract-v4") -> dict:
    valid_profiles = (
        "ready-contract-v4",
        "ready-potential-v5",
        "safety-bootstrap-v6a",
        "lifecycle-motion-v6b",
        "adaptive-motion-v6c",
        "adaptive-balanced-ready-v6d",
        "balanced-ready-v6e",
        "ready-survival-v6f",
    )
    if profile not in valid_profiles:
        raise ValueError(f"unsupported audit profile: {profile}")
    is_v6 = profile in (
        "safety-bootstrap-v6a",
        "lifecycle-motion-v6b",
        "adaptive-motion-v6c",
        "adaptive-balanced-ready-v6d",
        "balanced-ready-v6e",
        "ready-survival-v6f",
    )
    uses_ready_potential = profile != "ready-contract-v4"
    uses_balanced_ready = profile in (
        "adaptive-balanced-ready-v6d",
        "balanced-ready-v6e",
    )
    uses_ready_survival = profile == "ready-survival-v6f"
    if uses_ready_survival:
        expected_rewards = EXPECTED_NONZERO_REWARDS_READY_SURVIVAL_V6F
    elif uses_balanced_ready:
        expected_rewards = EXPECTED_NONZERO_REWARDS_BALANCED_READY_V6D
    elif uses_ready_potential:
        expected_rewards = EXPECTED_NONZERO_REWARDS_READY_POTENTIAL_V5
    else:
        expected_rewards = EXPECTED_NONZERO_REWARDS
    env = _load(run_dir / "params" / "env.yaml")
    agent = _load(run_dir / "params" / "agent.yaml")
    command = env["commands"]["racket_target"]
    motion = env["commands"]["motion"]
    policy = env["observations"]["policy"]
    rewards = env["rewards"]
    terminations = env["terminations"]
    events = env["events"]
    planner_discovery = rewards["planner_racket_task_space_crossfade"]
    racket_progress = rewards["prestrike_racket_progress"]
    station_progress = rewards["prestrike_station_progress"]
    velocity_progress = rewards["near_impact_planner_velocity_progress"]

    nonzero = {
        name
        for name, term in rewards.items()
        if term is not None and abs(float(term.get("weight", 0.0))) > 0.0
    }
    checks = {
        "episode_45s": float(env["episode_length_s"]) == 45.0,
        "control_50hz": int(env["decimation"]) == 4
        and abs(float(env["sim"]["dt"]) - 0.005) < 1.0e-12,
        "normal114": policy["racket_target_normal_w"] is not None
        and policy["stability_feedback"] is None,
        "effective_action_feedback": env["actions"]["joint_pos"]["feedback_mode"]
        == "effective",
        "five_audited_motions": isinstance(motion["motion_file"], list)
        and len(motion["motion_file"]) == 5,
        "balanced_motion_weights": tuple(motion["clip_sampling_weights"])
        == (0.25, 0.25, 0.2, 0.15, 0.15),
        "continuous_lifecycle": motion["wrap_teleport"] is False,
        "scratch_reset_curriculum": motion["motion_start_warmup_enabled"] is True
        and float(motion["motion_start_warmup_start_prob"]) == 0.60
        and float(motion["motion_start_warmup_min_prob"]) == 0.20
        and motion["motion_start_warmup_prestrike_enabled"] is True
        and float(motion["motion_start_warmup_prestrike_fraction"])
        == (0.45 if profile == "lifecycle-motion-v6b" else 0.60)
        and len(motion["motion_start_warmup_phase_per_clip"]) == 5
        and tuple(motion["motion_start_warmup_prestrike_steps_range"])
        == (12, 24),
        "no_ball_regression": float(motion["stand_episode_prob"])
        == (0.25 if profile == "adaptive-balanced-ready-v6d" else 0.10)
        and command["deploy_ready_force_stand_episode"] is True
        and command["deploy_ready_force_default_stand_reset"] is True,
        "motion_lifecycle_sampling": (
            motion["motion_start_warmup_recovery_enabled"] is True
            and float(motion["motion_start_warmup_recovery_fraction"]) == 0.35
            and tuple(motion["motion_start_warmup_recovery_steps_range"])
            == (8, 30)
            and motion.get(
                "motion_start_warmup_lifecycle_curriculum_enabled", False
            )
            is False
        )
        if profile == "lifecycle-motion-v6b"
        else (
            motion["motion_start_warmup_recovery_enabled"] is True
            and float(motion["motion_start_warmup_recovery_fraction"]) == 0.0
            and tuple(motion["motion_start_warmup_recovery_steps_range"])
            == (8, 30)
            and motion["motion_start_warmup_lifecycle_curriculum_enabled"]
            is True
            and float(
                motion["motion_start_warmup_lifecycle_targeted_attempt_low"]
            )
            == 0.01
            and float(
                motion["motion_start_warmup_lifecycle_targeted_attempt_high"]
            )
            == 0.05
            and float(
                motion["motion_start_warmup_lifecycle_prestrike_fraction_end"]
            )
            == 0.50
            and float(
                motion["motion_start_warmup_lifecycle_recovery_fraction_end"]
            )
            == 0.15
        )
        if profile in (
            "adaptive-motion-v6c",
            "adaptive-balanced-ready-v6d",
            "balanced-ready-v6e",
            "ready-survival-v6f",
        )
        else motion["motion_start_warmup_recovery_enabled"] is False,
        "balanced_ready_skill": (
            float(rewards["no_command_ready_balance"]["weight"]) == 1.2
            and int(
                rewards["no_command_ready_balance"]["params"]["warmup_steps"]
            )
            == 600
        )
        if uses_balanced_ready
        else float(rewards["no_command_ready_balance"]["weight"]) == 0.0,
        "staged_ready_survival": (
            command["active_ready_survival_milestones_enabled"] is True
            and command["active_ready_survival_require_soft_envelope"] is False
            and tuple(command["active_ready_survival_milestone_steps"])
            == (50, 75, 100, 150, 250, 400)
            and tuple(command["active_ready_survival_milestone_values"])
            == (0.10, 0.15, 0.25, 0.45, 0.80, 1.25)
            and float(
                rewards["active_ready_survival_milestone_bonus"]["weight"]
            )
            == 2.0
        )
        if uses_ready_survival
        else command["active_ready_survival_milestones_enabled"] is False
        and float(
            rewards["active_ready_survival_milestone_bonus"]["weight"]
        )
        == 0.0,
        "non_farmable_ready_recovery": float(
            rewards["active_ready_sustained_bonus"]["weight"]
        )
        == 2.0
        and float(rewards["post_contact_directional_recovery"]["weight"])
        == 0.8
        and float(rewards["post_contact_ready_region"]["weight"]) == 0.0,
        "ready_potential_profile": (
            float(rewards["no_command_ready_progress"]["weight"])
            == (3.0 if is_v6 else 1.5)
            and float(command["no_command_ready_progress_clip"]) == 0.05
            and command["healthy_stage_use_dynamic_station_during_ready"] is False
            and float(command["healthy_stage1_ready_hold_prob"])
            == (0.45 if is_v6 else 0.15)
            and float(rewards["no_command_ready_stability"]["weight"]) == 0.0
            and float(rewards["no_command_ready_balance"]["weight"])
            == (1.2 if uses_balanced_ready else 0.0)
            and float(rewards["functional_no_command_ready"]["weight"]) == 0.0
        )
        if uses_ready_potential
        else float(rewards["no_command_ready_progress"]["weight"]) == 0.0,
        "capability_driven_health_floor": (
            command["impact_health_floor_progress_source"] == "targeted_attempt"
            and float(
                command["impact_health_floor_targeted_attempt_threshold"]
            )
            == 0.05
        )
        if is_v6
        else command.get("impact_health_floor_progress_source", "ability")
        == "ability",
        "ball_route_independent_of_motion": command["strike_position_mode"]
        == "table_workspace"
        and float(command["table_workspace_motion_seed_blend_start"]) == 0.0,
        "canonical_hit_plane": float(command["planner_hit_plane_x"]) == 0.20,
        "one_bounce": command["incoming_trajectory_mode"] == "one_bounce",
        "planner_v4_joint_manifold": command["planner_command_mode"]
        == "v4_wire_compatible",
        "fitted_contact_model": abs(float(command["paddle_restitution"]) - 0.654)
        < 1.0e-9
        and abs(float(command["paddle_tangent_retain"]) - 0.48) < 1.0e-9,
        "ability_not_iteration_curriculum": command["ability_curriculum_enabled"]
        is True
        and int(command["ability_curriculum_min_resolved_events"]) >= 4096
        and int(command["ability_curriculum_required_advance_checks"]) >= 2,
        "both_sides_gate_progress": command["ability_curriculum_require_side_contact"]
        is True,
        "three_stage_station": command["healthy_three_stage_enabled"] is True,
        "closed_cycle": command["cycle_v2_enabled"] is True
        and command["cycle_v2_outcome_by_ability"] is True
        and command["cycle_v2_fail_unresolved_on_resample"] is True,
        "misses_do_not_reset": terminations["cycle_v2_ready_timeout"] is None,
        "hard_safety": terminations["table_touch"] is not None
        and terminations["persistent_action_overflow"] is not None,
        "actuator_randomization": events["serial_armature"] is not None
        and events["parallel_armature"] is not None,
        "reward_whitelist_exact": nonzero == expected_rewards,
        "ability_scaled_contact_discovery": float(
            planner_discovery["weight"]
        ) == 3.0
        and planner_discovery["params"]["ability_scaled_stds"] is True
        and float(planner_discovery["params"]["initial_position_std"]) == 0.45
        and float(planner_discovery["params"]["initial_velocity_std"]) == 2.50
        and float(planner_discovery["params"]["initial_normal_std_rad"]) == 1.20,
        "phase_separated_racket_progress": float(racket_progress["weight"]) == 1.2
        and float(racket_progress["params"]["arrival_radius"]) == 0.18
        and float(racket_progress["params"]["stop_window_s"]) == 0.20
        and float(racket_progress["params"]["idle_cost"]) == 0.06
        and racket_progress["params"]["velocity_frame"] == "torso_relative"
        and float(racket_progress["params"]["minimum_health_multiplier"]) == 0.15
        and float(racket_progress["params"]["final_minimum_health_multiplier"])
        == 0.0,
        "non_farmable_station_progress": float(station_progress["weight"]) == 0.8
        and float(station_progress["params"]["arrival_radius"]) == 0.05
        and float(station_progress["params"]["stop_window_s"]) == 0.20
        and float(station_progress["params"]["idle_cost"]) == 0.03,
        "signed_near_impact_velocity": float(velocity_progress["weight"]) == 3.0
        and float(velocity_progress["params"]["pre_start_s"]) == 0.22
        and float(velocity_progress["params"]["pre_full_s"]) == 0.10
        and float(velocity_progress["params"]["idle_cost"]) == 0.05
        and float(rewards["planner_velocity_band"]["weight"]) == 0.0,
        "ppo_fresh_scale": float(agent["policy"]["init_noise_std"]) == 0.3
        and float(agent["algorithm"]["learning_rate"]) == 0.0001,
    }
    failures = sorted(name for name, ok in checks.items() if not ok)
    return {
        "profile": profile,
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "nonzero_rewards": sorted(nonzero),
        "unexpected_rewards": sorted(nonzero - expected_rewards),
        "missing_rewards": sorted(expected_rewards - nonzero),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile",
        choices=(
            "ready-contract-v4",
            "ready-potential-v5",
            "safety-bootstrap-v6a",
            "lifecycle-motion-v6b",
            "adaptive-motion-v6c",
            "adaptive-balanced-ready-v6d",
            "balanced-ready-v6e",
            "ready-survival-v6f",
        ),
        default="ready-contract-v4",
    )
    args = parser.parse_args()
    result = audit(args.run_dir.resolve(), profile=args.profile)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
