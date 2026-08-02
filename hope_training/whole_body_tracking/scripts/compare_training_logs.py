#!/usr/bin/env python3
"""Compare aligned trailing windows from HOPE rsl_rl text logs."""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")
VALUE_RE = re.compile(r"^\s*([^:]+):\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")

DEFAULT_METRICS = (
    "Metrics/racket_target/return_success",
    "Metrics/racket_target/racket_pos_error",
    "Metrics/racket_target/racket_vel_error",
    "Metrics/racket_target/racket_normal_error_deg",
    "Metrics/racket_target/minimum_racket_target_distance",
    "Metrics/racket_target/impact_planner_pos_error_m",
    "Metrics/racket_target/impact_planner_vel_error_mps",
    "Metrics/racket_target/impact_planner_speed_ratio",
    "Metrics/racket_target/impact_planner_vel_angle_deg",
    "Metrics/motion/error_body_pos_lower",
    "Metrics/motion/error_body_rot_lower",
    "Metrics/motion/error_body_lin_vel_lower",
    "Metrics/motion/error_body_ang_vel_lower",
    "Metrics/motion/error_joint_pos_legs",
    "Metrics/motion/error_joint_vel_legs",
    "Metrics/racket_target/ability_contact_ema",
    "Metrics/racket_target/ability_net_ema",
    "Metrics/racket_target/ability_success_ema",
    "Metrics/racket_target/ability_forehand_contact_ema",
    "Metrics/racket_target/ability_forehand_net_ema",
    "Metrics/racket_target/ability_forehand_success_ema",
    "Metrics/racket_target/ability_backhand_contact_ema",
    "Metrics/racket_target/ability_backhand_net_ema",
    "Metrics/racket_target/ability_backhand_success_ema",
    "Metrics/racket_target/ability_recovery_ema",
    "Metrics/racket_target/ability_safety_ema",
    "Metrics/racket_target/safe_outcome_capability_gate",
    "Metrics/racket_target/ability_curriculum_level",
    "Metrics/racket_target/physical_ability_level",
    "Metrics/racket_target/physical_ability_raw_contact_ema",
    "Metrics/racket_target/physical_ability_contact_ema",
    "Metrics/racket_target/physical_ability_net_ema",
    "Metrics/racket_target/physical_ability_bounce_ema",
    "Metrics/racket_target/physical_ability_recovery_ema",
    "Metrics/racket_target/physical_ability_safety_ema",
    "Metrics/racket_target/physical_ability_net_threshold",
    "Metrics/racket_target/physical_ability_bounce_threshold",
    "Metrics/racket_target/physical_contact_health_score",
    "Metrics/racket_target/physical_contact_face_quality",
    "Metrics/racket_target/physical_center_contact_event",
    "Metrics/racket_target/physical_contact_outcome_quality",
    "Metrics/racket_target/physical_contact_quality_command_scale",
    "Metrics/racket_target/impact_inverse_command_blend",
    "Metrics/racket_target/physical_recovery_success_event",
    "Metrics/racket_target/physical_recovery_failure_event",
    "Metrics/racket_target/current_swing_contact",
    "Metrics/racket_target/current_swing_net_cross",
    "Metrics/racket_target/current_swing_healthy_contact",
    "Metrics/racket_target/current_swing_healthy_net_cross",
    "Metrics/racket_target/current_swing_targeted_attempt",
    "Metrics/racket_target/cycle_v2_attempt_event",
    "Metrics/racket_target/cycle_v2_ready_success_event",
    "Metrics/racket_target/cycle_v2_ready_fail_event",
    "Metrics/racket_target/cycle_v2_ready_ok",
    "Metrics/racket_target/cycle_v2_streak",
    "Metrics/racket_target/station_relocation_active",
    "Metrics/racket_target/station_relocation_arrival_event",
    "Metrics/racket_target/station_relocation_settle_event",
    "Metrics/racket_target/station_relocation_release_event",
    "Metrics/racket_target/station_relocation_timeout_event",
    "Metrics/racket_target/station_relocation_level",
    "Metrics/racket_target/station_relocation_arrival_ema",
    "Metrics/racket_target/station_relocation_settle_ema",
    "Metrics/racket_target/station_relocation_contact_ema",
    "Metrics/racket_target/station_relocation_safety_ema",
    "Metrics/racket_target/station_relocation_terminal_reset_events",
    "Metrics/racket_target/workspace_level",
    "Metrics/racket_target/workspace_motion_seed_blend",
    "Metrics/racket_target/planner_perturb_curriculum_scale",
    "Metrics/racket_target/incoming_ball_horizontal_speed",
    "Metrics/racket_target/incoming_ball_speed_curriculum_level",
    "Metrics/racket_target/recovery_ready_score",
    "Metrics/racket_target/recovery_base_lin_vel",
    "Metrics/racket_target/recovery_base_ang_vel",
    "Metrics/racket_target/recovery_station_error",
    "Metrics/racket_target/recovery_feet_contact_frac",
    "Metrics/racket_target/recovery_racket_speed",
    "Metrics/racket_target/post_contact_ready_curriculum_success_ema",
    "Metrics/racket_target/post_contact_ready_resolution_success_rate",
    "Metrics/racket_target/post_contact_ready_resolution_fail_rate",
    "Metrics/racket_target/post_contact_ready_durable_resolution_success_rate",
    "Metrics/racket_target/post_contact_ready_durable_resolution_fail_rate",
    "Metrics/racket_target/post_contact_ready_durable_resolution_success_latency_steps",
    "Metrics/racket_target/post_contact_ready_durable_resolution_fail_latency_steps",
    "Metrics/racket_target/post_contact_ready_durable_fail_backlean_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_forward_lean_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_torso_ang_vel_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_base_lin_vel_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_base_ang_vel_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_racket_speed_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_height_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_com_x_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_com_y_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_feet_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_station_rate",
    "Metrics/racket_target/post_contact_ready_durable_fail_arm_rate",
    "Metrics/racket_target/post_contact_ready_terminal_quality_mean",
    "Metrics/racket_target/post_contact_ready_safe_terminal_quality_mean",
    "Metrics/racket_target/post_contact_ready_terminal_safe_rate",
    "Metrics/racket_target/post_contact_ready_terminal_unsafe_rate",
    "Metrics/racket_target/post_contact_ready_terminal_incomplete_rate",
    "Metrics/racket_target/post_contact_ready_safe_net_cycle_rate",
    "Metrics/racket_target/post_contact_ready_peak_base_ang_vel",
    "Metrics/racket_target/post_contact_ready_peak_ang_vel_excess_increment",
    "Metrics/racket_target/post_contact_ready_envelope_violation_rate",
    "Metrics/racket_target/post_contact_ready_envelope_tilt_violation_rate",
    "Metrics/racket_target/post_contact_ready_envelope_pitch_violation_rate",
    "Metrics/racket_target/post_contact_ready_envelope_com_violation_rate",
    "Metrics/racket_target/post_contact_ready_envelope_waist_violation_rate",
    "Metrics/racket_target/post_contact_ready_envelope_leg_violation_rate",
    "Metrics/racket_target/post_contact_ready_envelope_base_ang_vel_violation_rate",
    "Metrics/racket_target/post_contact_ready_max_tilt",
    "Metrics/racket_target/post_contact_ready_max_abs_pitch",
    "Metrics/racket_target/post_contact_ready_max_com_x",
    "Metrics/racket_target/post_contact_ready_max_com_y",
    "Metrics/racket_target/post_contact_ready_max_waist_overflow",
    "Metrics/racket_target/post_contact_ready_max_leg_overflow",
    "Metrics/racket_target/post_contact_ready_settlement_velocity_score_ema",
    "Metrics/racket_target/post_contact_ready_settlement_position_gate_ema",
    "Metrics/racket_target/post_contact_ready_settlement_health_gate_ema",
    "Metrics/racket_target/post_contact_ready_settlement_recovery_gate_ema",
    "Metrics/racket_target/post_contact_ready_settlement_total_score_ema",
    "Metrics/racket_target/base_pitch_like_signed",
    "Metrics/racket_target/base_backward_velocity",
    "Metrics/racket_target/action_abs_max",
    "Metrics/racket_target/applied_action_abs_max",
    "Metrics/racket_target/effective_action_abs_max",
    "Metrics/racket_target/action_clamp_fraction",
    "Metrics/racket_target/waist_action_clamp_fraction",
    "Metrics/racket_target/right_arm_action_clamp_fraction",
    "Metrics/racket_target/leg_action_clamp_fraction",
    "Metrics/racket_target/waist_action_overflow_rms",
    "Metrics/racket_target/right_arm_action_overflow_rms",
    "Metrics/racket_target/leg_action_overflow_rms",
    "Metrics/racket_target/waist_q_tracking_error_rms",
    "Metrics/racket_target/right_arm_q_tracking_error_rms",
    "Metrics/racket_target/leg_q_tracking_error_rms",
    "Metrics/racket_target/waist_torque_clip_fraction",
    "Metrics/racket_target/right_arm_torque_clip_fraction",
    "Metrics/racket_target/leg_torque_clip_fraction",
    "Metrics/racket_target/actuator_overflow_consecutive_steps",
    "Metrics/racket_target/actuator_overflow_severe",
    "Metrics/racket_target/q_des_velocity_violation_fraction",
    "Metrics/racket_target/q_des_acceleration_violation_fraction",
    "Metrics/physical_shadow/serve_event",
    "Metrics/physical_shadow/contact_event",
    "Metrics/physical_shadow/net_cross_event",
    "Metrics/physical_shadow/opponent_bounce_event",
    "Metrics/physical_shadow/route_invalid_event",
    "Metrics/physical_shadow/route_unalignable_event",
    "Metrics/racket_target/post_contact_phase_impact_0_100ms_waist_overflow_gt_1p0_rate",
    "Metrics/racket_target/post_contact_phase_impact_0_100ms_base_ang_vel_gt_0p8_rate",
    "Metrics/racket_target/post_contact_phase_brake_100_300ms_waist_overflow_gt_1p0_rate",
    "Metrics/racket_target/post_contact_phase_brake_100_300ms_base_ang_vel_gt_0p8_rate",
    "Metrics/racket_target/post_contact_phase_settle_300_600ms_waist_overflow_gt_1p0_rate",
    "Metrics/racket_target/post_contact_phase_settle_300_600ms_base_ang_vel_gt_0p8_rate",
    "Metrics/racket_target/post_contact_phase_ready_after_600ms_waist_overflow_gt_1p0_rate",
    "Metrics/racket_target/post_contact_phase_ready_after_600ms_base_ang_vel_gt_0p8_rate",
    "Episode_Termination/base_too_low",
    "Episode_Termination/base_tilted",
    "Episode_Termination/table_touch",
    "Episode_Termination/persistent_action_overflow",
    "Episode_Reward/health_gated_soft_ball_contact",
    "Episode_Reward/physical_outcome",
    "Episode_Reward/planner_velocity_band",
    "Episode_Reward/terminal_quality_window",
    "Episode_Reward/safe_terminal_quality",
    "Episode_Reward/unsafe_terminal_recovery",
    "Episode_Reward/safe_terminal_outcome",
    "Episode_Reward/capability_gated_safe_terminal_outcome",
    "Episode_Reward/targeted_contact_miss",
    "Episode_Reward/safe_terminal_cycle",
    "Episode_Reward/durable_cycle_success",
    "Episode_Reward/durable_recovery_progress",
    "Episode_Reward/durable_recovery_resolved",
    "Episode_Reward/durable_recovery_failed",
    "Episode_Reward/durable_outcome_bonus",
    "Episode_Reward/recovery_peak_ang_vel_excess",
    "Episode_Reward/actuator_waist_feasibility",
    "Episode_Reward/actuator_right_arm_feasibility",
    "Episode_Reward/actuator_leg_feasibility",
    "Episode_Reward/physical_outcome_events",
    "Episode_Reward/physical_recovery_settlement",
)


def _parse(path: pathlib.Path) -> list[dict]:
    blocks = []
    current = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = ITER_RE.search(line)
        if match:
            if current is not None:
                blocks.append(current)
            current = {"iteration": int(match.group(1))}
            continue
        if current is None:
            continue
        match = VALUE_RE.match(line)
        if match:
            current[match.group(1).strip()] = float(match.group(2))
    if current is not None:
        blocks.append(current)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional display labels in the same order as logs.",
    )
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.logs):
        parser.error("--labels must contain exactly one label per log")

    parsed = [(pathlib.Path(value), _parse(pathlib.Path(value))) for value in args.logs]
    labels = args.labels or [path.stem for path, _ in parsed]
    common_end = min(
        (blocks[-1]["iteration"] for _path, blocks in parsed if blocks),
        default=0,
    )
    common_start = common_end - args.window + 1
    lines = [
        f"# Training log comparison (aligned iterations {common_start}..{common_end})",
        "",
        "| metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(parsed),
    ]
    for metric in DEFAULT_METRICS:
        values = []
        for _path, blocks in parsed:
            tail = [
                block
                for block in blocks
                if common_start <= block["iteration"] <= common_end
            ]
            observed = [block[metric] for block in tail if metric in block]
            values.append("" if not observed else f"{statistics.fmean(observed):.5f}")
        lines.append(f"| {metric} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "| run | first iteration | last iteration | parsed blocks |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, (path, blocks) in zip(labels, parsed, strict=True):
        first = blocks[0]["iteration"] if blocks else "-"
        last = blocks[-1]["iteration"] if blocks else "-"
        lines.append(f"| {label} | {first} | {last} | {len(blocks)} |")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.markdown_out:
        target = pathlib.Path(args.markdown_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
