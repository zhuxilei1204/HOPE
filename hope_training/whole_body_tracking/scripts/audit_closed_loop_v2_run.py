#!/usr/bin/env python3
"""Reject a resolved run when closed-loop-v2 isolation is incomplete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


BASELINE_NONZERO_REWARDS = {
    "upright",
    "imitation",
    "racket_wrist_motion_pos",
    "racket_wrist_motion_ori",
    "racket_position",
    "racket_velocity_projection",
    "prestrike_blade_direction",
    "planner_racket_task_space_crossfade",
    "exact_impact_planner_task_space_alignment",
    "health_gated_soft_ball_contact",
    "physical_outcome",
    "closed_cycle_success",
    "recovery_progress",
    "recovery_resolved",
    "recovery_failed",
    "deferred_recovery_outcome_bonus",
    "no_command_instability",
    "safe_strike_inactivity",
    "phase_action_overflow",
    "phase_action_rate_waist",
    "phase_action_rate_upper",
    "phase_action_rate_legs",
    "joint_limit",
    "undesired_contacts",
    "feet_contact_slip",
    "table_no_touch",
    "termination_penalty",
}

BASELINE_CORE_REWARD_WEIGHTS = {
    "health_gated_soft_ball_contact": 3.0,
    "physical_outcome": 1.0,
    "closed_cycle_success": 24.0,
    "recovery_progress": 0.8,
    "recovery_resolved": 6.0,
    "recovery_failed": -6.0,
    "deferred_recovery_outcome_bonus": 8.0,
    "no_command_instability": -0.4,
    "safe_strike_inactivity": -6.0,
    "phase_action_overflow": -0.15,
    "table_no_touch": -1.0,
    "termination_penalty": -24.0,
}

IMPACT_V1_REWARD_WEIGHTS = {
    **BASELINE_CORE_REWARD_WEIGHTS,
    "planner_velocity_band": 6.0,
}

IMPACT_RECOVERY_V2_REWARD_WEIGHTS = {
    **BASELINE_CORE_REWARD_WEIGHTS,
    "planner_velocity_band": 3.0,
    "recovered_planner_velocity": 12.0,
}

IMPACT_CONSTRAINT_V3_REWARD_WEIGHTS = {
    **BASELINE_CORE_REWARD_WEIGHTS,
    "planner_velocity_band": 6.0,
    "recovery_peak_ang_vel_excess": -2.0,
}

DURABLE_EARLY_SETTLEMENT_REWARDS = {
    "closed_cycle_success",
    "recovery_progress",
    "recovery_resolved",
    "recovery_failed",
    "deferred_recovery_outcome_bonus",
}

DURABLE_CYCLE_NONZERO_REWARDS = {
    *(BASELINE_NONZERO_REWARDS - DURABLE_EARLY_SETTLEMENT_REWARDS),
    "planner_velocity_band",
    "durable_cycle_success",
    "durable_recovery_progress",
    "durable_recovery_resolved",
    "durable_recovery_failed",
    "durable_outcome_bonus",
}

DURABLE_CYCLE_REWARD_WEIGHTS = {
    **{
        name: value
        for name, value in BASELINE_CORE_REWARD_WEIGHTS.items()
        if name not in DURABLE_EARLY_SETTLEMENT_REWARDS
    },
    "planner_velocity_band": 6.0,
    "durable_cycle_success": 24.0,
    "durable_recovery_progress": 0.8,
    "durable_recovery_resolved": 2.0,
    "durable_recovery_failed": -8.0,
    "durable_outcome_bonus": 8.0,
}

SAFE_QUALITY_CYCLE_NONZERO_REWARDS = {
    *(BASELINE_NONZERO_REWARDS - DURABLE_EARLY_SETTLEMENT_REWARDS),
    "planner_velocity_band",
    "terminal_quality_window",
    "safe_terminal_quality",
    "unsafe_terminal_recovery",
    "safe_terminal_outcome",
    "safe_terminal_cycle",
}

SAFE_QUALITY_CYCLE_REWARD_WEIGHTS = {
    **{
        name: value
        for name, value in BASELINE_CORE_REWARD_WEIGHTS.items()
        if name not in DURABLE_EARLY_SETTLEMENT_REWARDS
    },
    "planner_velocity_band": 6.0,
    "terminal_quality_window": 4.0,
    "safe_terminal_quality": 2.0,
    "unsafe_terminal_recovery": -12.0,
    "safe_terminal_outcome": 8.0,
    "safe_terminal_cycle": 24.0,
}

DURABLE_PROFILES = {"durable_cycle_v1", "safe_quality_cycle_v2"}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.unsafe_load(handle)


def audit(run_dir: Path, profile: str = "baseline") -> dict:
    env = _load(run_dir / "params" / "env.yaml")
    agent = _load(run_dir / "params" / "agent.yaml")
    errors = []

    command = env["commands"]["racket_target"]
    motion = env["commands"]["motion"]
    policy = env["observations"]["policy"]
    terminations = env["terminations"]
    events = env["events"]
    nonzero_rewards = {
        name
        for name, term in env["rewards"].items()
        if term is not None and abs(float(term.get("weight", 0.0))) > 0.0
    }
    if profile == "baseline":
        expected_nonzero_rewards = BASELINE_NONZERO_REWARDS
        expected_core_reward_weights = BASELINE_CORE_REWARD_WEIGHTS
    elif profile == "impact_v1":
        expected_nonzero_rewards = {
            *BASELINE_NONZERO_REWARDS,
            "planner_velocity_band",
        }
        expected_core_reward_weights = IMPACT_V1_REWARD_WEIGHTS
    elif profile == "impact_recovery_v2":
        expected_nonzero_rewards = {
            *BASELINE_NONZERO_REWARDS,
            "planner_velocity_band",
            "recovered_planner_velocity",
        }
        expected_core_reward_weights = IMPACT_RECOVERY_V2_REWARD_WEIGHTS
    elif profile == "impact_constraint_v3":
        expected_nonzero_rewards = {
            *BASELINE_NONZERO_REWARDS,
            "planner_velocity_band",
            "recovery_peak_ang_vel_excess",
        }
        expected_core_reward_weights = IMPACT_CONSTRAINT_V3_REWARD_WEIGHTS
    elif profile == "durable_cycle_v1":
        expected_nonzero_rewards = DURABLE_CYCLE_NONZERO_REWARDS
        expected_core_reward_weights = DURABLE_CYCLE_REWARD_WEIGHTS
    elif profile == "safe_quality_cycle_v2":
        expected_nonzero_rewards = SAFE_QUALITY_CYCLE_NONZERO_REWARDS
        expected_core_reward_weights = SAFE_QUALITY_CYCLE_REWARD_WEIGHTS
    else:
        raise ValueError(f"unsupported closed-loop-v2 audit profile: {profile}")

    checks = {
        "episode_length_45s": float(env["episode_length_s"]) == 45.0,
        "rollout_64": int(agent["num_steps_per_env"]) == 64,
        "one_forehand_motion": isinstance(motion["motion_file"], str),
        "continuous_wrap": motion["wrap_teleport"] is False,
        "hold_20_60": tuple(motion["hold_steps_range"]) == (20, 60),
        "fixed_station": command["station_mode"] == "fixed",
        "independent_workspace": command["strike_position_mode"]
        == "table_workspace",
        "workspace_stage_zero": float(command["table_workspace_fixed_level"])
        == 0.0,
        "no_motion_seed": float(
            command["table_workspace_motion_seed_blend_start"]
        )
        == 0.0,
        "one_bounce": command["incoming_trajectory_mode"] == "one_bounce",
        "v4_command_manifold": command["planner_command_mode"]
        == "v4_wire_compatible",
        "attempt_recovery": command["post_contact_ready_trigger"]
        == "targeted_attempt",
        "ability_gated_recovery_curriculum": (
            profile in DURABLE_PROFILES
            and command["post_contact_ready_curriculum_enabled"] is False
            and command[
                "post_contact_ready_durable_diagnostic_enabled"
            ]
            is True
        )
        or (
            profile not in DURABLE_PROFILES
            and command["post_contact_ready_curriculum_enabled"] is True
            and int(command["post_contact_ready_curriculum_start_level"]) == 0
        ),
        "recovery_curriculum_schedule": profile in DURABLE_PROFILES
        or (
            tuple(
            float(value)
            for value in command[
                "post_contact_ready_curriculum_max_base_ang_vel"
            ]
            )
            == (1.40, 1.05, 0.80)
            and tuple(
                int(value)
                for value in command[
                    "post_contact_ready_curriculum_required_consecutive_steps"
                ]
            )
            == (2, 3, 5)
            and tuple(
                int(value)
                for value in command[
                    "post_contact_ready_curriculum_deadline_steps"
                ]
            )
            == (90, 75, 60)
        ),
        "recovery_curriculum_final_gate": profile in DURABLE_PROFILES
        or (
            float(
            command["post_contact_ready_curriculum_max_torso_ang_vel"][-1]
            )
            == 0.80
            and float(
                command["post_contact_ready_curriculum_max_base_lin_vel"][-1]
            )
            == 0.22
            and float(
                command["post_contact_ready_curriculum_max_racket_speed"][-1]
            )
            == 1.10
            and float(
                command["post_contact_ready_curriculum_min_feet_contact"][-1]
            )
            == 0.90
        ),
        "recovery_curriculum_hit_guard": profile in DURABLE_PROFILES
        or (
            float(
            command[
                "post_contact_ready_curriculum_min_targeted_attempt_ema"
            ]
            )
            == 0.25
            and float(
                command[
                    "post_contact_ready_curriculum_min_return_success_ema"
                ]
            )
            == 0.05
            and int(
                command["post_contact_ready_curriculum_min_completed_swings"]
            )
            == 2048
        ),
        "normal_visible": policy["racket_target_normal_w"] is not None,
        "no_122d_feedback": policy["stability_feedback"] is None,
        "table_touch_hard": terminations["table_touch"]["params"]["enabled"]
        is True,
        "overflow_hard": terminations["persistent_action_overflow"]["params"][
            "enabled"
        ]
        is True,
        "reference_terminations_disabled": all(
            terminations[name] is None
            for name in ("anchor_pos", "anchor_ori", "ee_body_pos")
        ),
        "narrow_link_mass_randomization": tuple(
            float(value)
            for value in events["link_mass"]["params"][
                "mass_distribution_params"
            ]
        )
        == (0.95, 1.05),
        "narrow_pd_randomization": all(
            tuple(
                float(value)
                for value in events["pd_gains"]["params"][parameter]
            )
            == (0.95, 1.05)
            for parameter in (
                "stiffness_distribution_params",
                "damping_distribution_params",
            )
        ),
        "reviewed_friction_randomization": tuple(
            float(value)
            for value in events["physics_material"]["params"][
                "static_friction_range"
            ]
        )
        == (0.80, 1.20)
        and tuple(
            float(value)
            for value in events["physics_material"]["params"][
                "dynamic_friction_range"
            ]
        )
        == (0.75, 1.15),
        "closed_loop_reward_weights": all(
            float(env["rewards"][name]["weight"]) == expected
            for name, expected in expected_core_reward_weights.items()
        ),
        "durable_finetune_optimizer_contract": (
            profile not in DURABLE_PROFILES
            or (
                float(agent["policy"]["init_noise_std"]) == 0.02
                and float(agent["algorithm"]["learning_rate"]) == 0.00005
                and float(agent["algorithm"]["desired_kl"]) == 0.003
                and float(agent["algorithm"]["entropy_coef"]) == 0.0005
                and float(agent["algorithm"]["clip_param"]) == 0.10
                and float(agent["algorithm"]["max_grad_norm"]) == 0.7
            )
        ),
        "impact_v1_planner_velocity_contract": (
            profile
            not in (
                "impact_v1",
                "impact_recovery_v2",
                "impact_constraint_v3",
                "durable_cycle_v1",
                "safe_quality_cycle_v2",
            )
            or (
                float(
                    env["rewards"]["planner_velocity_band"]["params"][
                        "timing_std_s"
                    ]
                )
                == 0.055
                and float(
                    env["rewards"]["planner_velocity_band"]["params"][
                        "active_window_s"
                    ]
                )
                == 0.12
                and float(
                    env["rewards"]["planner_velocity_band"]["params"][
                        "position_floor"
                    ]
                )
                == 0.10
            )
        ),
        "impact_recovery_v2_settlement_contract": (
            profile != "impact_recovery_v2"
            or (
                float(
                    env["rewards"]["recovered_planner_velocity"]["params"][
                        "recovery_peak_ang_vel_budget"
                    ]
                )
                == 0.80
                and float(
                    env["rewards"]["recovered_planner_velocity"]["params"][
                        "recovery_peak_ang_vel_excess_std"
                    ]
                )
                == 0.60
                and float(
                    env["rewards"]["recovered_planner_velocity"]["params"][
                        "recovery_gate_floor"
                    ]
                )
                == 0.10
                and float(
                    env["rewards"]["recovered_planner_velocity"]["params"][
                        "impact_health_floor"
                    ]
                )
                == 0.25
                and float(
                    env["rewards"]["recovered_planner_velocity"]["params"][
                        "position_std"
                    ]
                )
                == 0.12
                and float(
                    env["rewards"]["recovered_planner_velocity"]["params"][
                        "position_floor"
                    ]
                )
                == 0.10
            )
        ),
        "impact_constraint_v3_peak_contract": (
            profile != "impact_constraint_v3"
            or (
                float(command["post_contact_ready_peak_excess_std"]) == 0.60
                and float(
                    command[
                        "post_contact_ready_peak_excess_max_potential"
                    ]
                )
                == 4.0
                and float(
                    env["rewards"]["planner_velocity_band"]["weight"]
                )
                == 6.0
            )
        ),
        "durable_cycle_contract": (
            profile not in DURABLE_PROFILES
            or (
                float(command["post_contact_ready_durable_min_delay_s"])
                == 0.60
                and float(
                    command["post_contact_ready_durable_deadline_s"]
                )
                == 1.10
                and int(
                    command[
                        "post_contact_ready_durable_required_consecutive_steps"
                    ]
                )
                == 15
                and all(
                    float(env["rewards"][name]["weight"]) == 0.0
                    for name in DURABLE_EARLY_SETTLEMENT_REWARDS
                )
                and (
                    profile != "safe_quality_cycle_v2"
                    or all(
                        float(env["rewards"][name]["weight"]) == 0.0
                        for name in (
                            "durable_cycle_success",
                            "durable_recovery_progress",
                            "durable_recovery_resolved",
                            "durable_recovery_failed",
                            "durable_outcome_bonus",
                        )
                    )
                )
            )
        ),
        "safe_quality_operational_contract": (
            profile != "safe_quality_cycle_v2"
            or (
                float(
                    command[
                        "post_contact_ready_operational_max_tilt"
                    ]
                )
                == 0.20
                and float(
                    command[
                        "post_contact_ready_operational_max_torso_ang_vel"
                    ]
                )
                == 1.20
                and float(
                    command[
                        "post_contact_ready_operational_max_base_ang_vel"
                    ]
                )
                == 1.20
                and float(
                    command[
                        "post_contact_ready_operational_max_height_error"
                    ]
                )
                == 0.12
                and float(
                    command[
                        "post_contact_ready_operational_min_feet_contact"
                    ]
                )
                == 0.50
                and float(
                    command[
                        "post_contact_ready_operational_max_station_error"
                    ]
                )
                == 0.24
            )
        ),
        "deferred_outcome_tiers": tuple(
            float(value)
            for value in env["rewards"][
                "deferred_recovery_outcome_bonus"
            ]["params"]["tier_multipliers"]
        )
        == (0.5, 1.0, 1.5),
        "durable_outcome_tiers": (
            profile != "durable_cycle_v1"
            or tuple(
                float(value)
                for value in env["rewards"]["durable_outcome_bonus"][
                    "params"
                ]["tier_multipliers"]
            )
            == (0.5, 1.0, 1.5)
        ),
        "safe_terminal_outcome_tiers": (
            profile != "safe_quality_cycle_v2"
            or tuple(
                float(value)
                for value in env["rewards"]["safe_terminal_outcome"][
                    "params"
                ]["tier_multipliers"]
            )
            == (0.5, 1.0, 1.5)
        ),
        "immediate_outcome_is_bounded": all(
            float(env["rewards"]["physical_outcome"]["params"][name]) == 1.0
            for name in (
                "contact_scale",
                "net_cross_scale",
                "opponent_bounce_scale",
            )
        ),
        "reward_whitelist_exact": nonzero_rewards
        == expected_nonzero_rewards,
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(name)

    report = {
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors,
        "nonzero_rewards": sorted(nonzero_rewards),
        "unexpected_rewards": sorted(
            nonzero_rewards - expected_nonzero_rewards
        ),
        "missing_rewards": sorted(
            expected_nonzero_rewards - nonzero_rewards
        ),
        "profile": profile,
    }
    if errors:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile",
        choices=(
            "baseline",
            "impact_v1",
            "impact_recovery_v2",
            "impact_constraint_v3",
            "durable_cycle_v1",
            "safe_quality_cycle_v2",
        ),
        default="baseline",
    )
    args = parser.parse_args()
    report = audit(args.run_dir.resolve(), profile=args.profile)
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
