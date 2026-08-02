from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "cfg/task"
A_PATH = TASK_DIR / "HOPEPingPongStage2PhysicalBalancedA114.yaml"
B_PATH = TASK_DIR / "HOPEPingPongStage2PhysicalOutcomeB114.yaml"
CLEAN_A_PATH = TASK_DIR / "HOPEPingPongStage2PhysicalCleanA114.yaml"
CLEAN_B_PATH = TASK_DIR / "HOPEPingPongStage2PhysicalCleanSlowB114.yaml"
QUALITY_A_PATH = TASK_DIR / "HOPEPingPongStage2PhysicalQualityA114.yaml"
QUALITY_B_PATH = TASK_DIR / "HOPEPingPongStage2PhysicalQualityB114.yaml"
CALIB_A_PATH = TASK_DIR / "HOPEPingPongStage2ImpactCalibrationA114.yaml"
CALIB_B_PATH = TASK_DIR / "HOPEPingPongStage2ImpactCalibrationB114.yaml"
IMPACT_CREDIT_V2_PATH = TASK_DIR / "HOPEPingPongStage2ImpactCredit114V2.yaml"
CONTACT_ALIGNMENT_CONTROL_V3_PATH = (
    TASK_DIR / "HOPEPingPongStage2ContactAlignmentControl114V3.yaml"
)
CONTACT_ALIGNMENT_REWARD_V3_PATH = (
    TASK_DIR / "HOPEPingPongStage2ContactAlignmentReward114V3.yaml"
)
CLOSED_LOOP_ORACLE_V4_PATH = (
    TASK_DIR / "HOPEPingPongStage2ClosedLoopCommandOracle114V4.yaml"
)
CLOSED_LOOP_DEPLOY_V4_PATH = (
    TASK_DIR / "HOPEPingPongStage2ClosedLoopCommandDeploy114V4.yaml"
)
CLOSED_LOOP_CLEAN_TRAIN_V4_PATH = (
    TASK_DIR / "HOPEPingPongStage2ClosedLoopCommandCleanTrain114V4.yaml"
)
CLOSED_LOOP_ROBUST_TRAIN_V4_PATH = (
    TASK_DIR / "HOPEPingPongStage2ClosedLoopCommandRobustTrain114V4.yaml"
)
COMMAND_EXECUTOR_CORE_V5_PATH = (
    TASK_DIR / "HOPEPingPongStage2CommandExecutorCore114V5.yaml"
)
COMMAND_EXECUTOR_DIVERSE_V5_PATH = (
    TASK_DIR / "HOPEPingPongStage2CommandExecutorDiverse114V5.yaml"
)
COMMAND_CURRICULUM_OUTCOME_V6_PATH = (
    TASK_DIR / "HOPEPingPongStage2CommandCurriculumOutcome114V6A.yaml"
)
COMMAND_CURRICULUM_ALIGNED_V6_PATH = (
    TASK_DIR / "HOPEPingPongStage2CommandCurriculumAligned114V6B.yaml"
)
COMMAND_PRECISION_CREDIT_V7_PATH = (
    TASK_DIR / "HOPEPingPongStage2CommandPrecisionCredit114V7A.yaml"
)
COMMAND_PRECISION_OUTCOME_V7_PATH = (
    TASK_DIR / "HOPEPingPongStage2CommandPrecisionOutcome114V7B.yaml"
)
SAFETY_CREDIT_SOFT_V8_PATH = (
    TASK_DIR / "HOPEPingPongStage2SafetyCreditSoft114V8A.yaml"
)
SAFETY_CREDIT_DEFERRED_V8_PATH = (
    TASK_DIR / "HOPEPingPongStage2SafetyCreditDeferred114V8B.yaml"
)
SAFETY_CREDIT_TERMINAL_V9_PATH = (
    TASK_DIR / "HOPEPingPongStage2SafetyCreditTerminal114V9.yaml"
)
SAFETY_CREDIT_HARD_DEBIT_V10_PATH = (
    TASK_DIR / "HOPEPingPongStage2SafetyCreditHardDebit114V10.yaml"
)
IMPACT_CREDIT_ALGO_PATH = ROOT / "cfg/algo/ppo_stage2_impact_credit.yaml"
ENV_CFG = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "hope_stage2_physical_env_cfg.py"
)
PHYSICAL_ENV_CFG = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "hope_physical_eval_env_cfg.py"
)
REWARDS = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "physical_stage2.py"
)
COMMANDS = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "hope_commands.py"
)
EVALUATE = ROOT / "scripts/evaluate.py"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_stage2_ab_preserves_114d_motion_and_physical_task_contract() -> None:
    a = _load(A_PATH)
    b = _load(B_PATH)
    for cfg in (a, b):
        assert cfg["gym_task"] == "HOPE-PingPong-Stage2-Physical-AgibotA3-v0"
        assert cfg["actor_obs_contract"] == "hope_pingpong_normal114"
        assert cfg["motion_manifest"].endswith(
            "stage1_core_two_motion_20260801/manifest.tsv"
        )
        assert cfg["env"]["num_envs"] == 256
        assert cfg["motion"]["wrap_teleport"] is False


def test_stage2_clean_restart_configs_share_the_same_policy_contract() -> None:
    clean_a = _load(CLEAN_A_PATH)
    clean_b = _load(CLEAN_B_PATH)
    for cfg in (clean_a, clean_b):
        assert cfg["gym_task"] == "HOPE-PingPong-Stage2-Physical-AgibotA3-v0"
        assert cfg["actor_obs_contract"] == "hope_pingpong_normal114"
        assert cfg["motion_manifest"].endswith(
            "stage1_core_two_motion_20260801/manifest.tsv"
        )
        assert cfg["motion"]["wrap_teleport"] is False
    assert clean_a["overrides"] == {}
    assert set(clean_b["overrides"]) == {
        "commands.racket_target.table_workspace_curriculum_start_level",
        "commands.racket_target.table_workspace_motion_seed_end_level",
        "commands.racket_target.impact_inverse_command_curriculum_start_level",
        "commands.racket_target.one_bounce_speed_curriculum_start_level",
        "commands.racket_target.planner_perturb_curriculum_start_level",
    }


def test_stage2_quality_ab_only_changes_initial_physical_blend() -> None:
    a = _load(QUALITY_A_PATH)
    b = _load(QUALITY_B_PATH)
    expected = {
        "commands.racket_target.impact_inverse_command_start_blend",
        "rewards.physical_outcome_events.params.impact_inverse_quality_floor",
    }
    for cfg in (a, b):
        assert cfg["gym_task"] == "HOPE-PingPong-Stage2-Physical-AgibotA3-v0"
        assert cfg["actor_obs_contract"] == "hope_pingpong_normal114"
        assert cfg["motion_manifest"].endswith(
            "stage1_core_two_motion_20260801/manifest.tsv"
        )
        assert set(cfg["overrides"]) == expected
    assert set(a["overrides"].values()) == {0.10}
    assert set(b["overrides"].values()) == {0.20}


def test_stage2_impact_calibration_freezes_difficulty_and_adds_direct_gradient() -> None:
    shared_keys = {
        "commands.racket_target.table_workspace_fixed_level",
        "commands.racket_target.planner_perturb_curriculum_source",
        "commands.racket_target.planner_perturb_fixed_scale",
        "commands.racket_target.impact_inverse_command_start_blend",
        "rewards.physical_capability_curriculum.params.fixed_level",
        "rewards.physical_outcome_events.params.impact_inverse_quality_floor",
        "rewards.imitation.weight",
        "rewards.planner_racket_task_space_crossfade.params.ability_scaled_stds",
        "rewards.planner_racket_task_space_crossfade.params.position_std",
        "rewards.planner_racket_task_space_crossfade.params.velocity_std",
        "rewards.planner_racket_task_space_crossfade.params.normal_std_rad",
        "rewards.near_impact_planner_velocity_progress.weight",
        "rewards.near_impact_planner_velocity_progress.params.minimum_health_multiplier",
        "rewards.near_impact_planner_velocity_progress.params.final_minimum_health_multiplier",
        "rewards.racket_velocity_projection.weight",
        "rewards.racket_velocity_projection.params.min_speed_ratio",
        "rewards.racket_velocity_projection.params.speed_std",
        "rewards.racket_velocity_projection.params.lateral_std",
        "rewards.racket_velocity_projection.params.minimum_health_multiplier",
        "rewards.racket_velocity_projection.params.final_minimum_health_multiplier",
    }
    for cfg in (_load(CALIB_A_PATH), _load(CALIB_B_PATH)):
        overrides = cfg["overrides"]
        assert cfg["actor_obs_contract"] == "hope_pingpong_normal114"
        assert cfg["motion"]["wrap_teleport"] is False
        assert shared_keys <= set(overrides)
        assert overrides["commands.racket_target.table_workspace_fixed_level"] == 0.0
        assert overrides["commands.racket_target.planner_perturb_curriculum_source"] == "fixed"
        assert overrides["commands.racket_target.planner_perturb_fixed_scale"] == 0.0
        assert overrides["rewards.physical_capability_curriculum.params.fixed_level"] == 0.0
        assert overrides["rewards.imitation.weight"] > 0.0
        assert overrides["rewards.near_impact_planner_velocity_progress.weight"] > 0.0
        assert overrides["rewards.racket_velocity_projection.weight"] > 0.0


def test_stage2_impact_calibration_b_only_adds_physical_outcome_emphasis() -> None:
    a = _load(CALIB_A_PATH)["overrides"]
    b = _load(CALIB_B_PATH)["overrides"]
    b_only = set(b) - set(a)
    assert all(a[key] == b[key] for key in a)
    assert b_only == {
        "rewards.physical_outcome_events.weight",
        "rewards.physical_outcome_events.params.contact_scale",
        "rewards.physical_outcome_events.params.net_cross_scale",
        "rewards.physical_outcome_events.params.opponent_bounce_scale",
        "rewards.physical_outcome_events.params.contact_quality_scale",
    }
    assert b["rewards.physical_outcome_events.params.net_cross_scale"] > 3.0
    assert b["rewards.physical_outcome_events.params.opponent_bounce_scale"] > 6.0


def test_physical_curriculum_fixed_level_preserves_metrics_but_freezes_distribution() -> None:
    env_source = ENV_CFG.read_text(encoding="utf-8")
    source = REWARDS.read_text(encoding="utf-8")
    assert '"fixed_level": -1.0' in env_source
    update_pos = source.index("self._maybe_update(")
    freeze_pos = source.index("if float(fixed_level) >= 0.0:", update_pos)
    publish_pos = source.index("self._publish(target)", freeze_pos)
    assert update_pos < freeze_pos < publish_pos
    assert 'target._ability_curriculum_level.fill_(float(fixed_level))' in source


def test_stage2_impact_credit_v2_covers_outcome_horizon_and_releases_only_strike() -> None:
    task = _load(IMPACT_CREDIT_V2_PATH)
    algo = _load(IMPACT_CREDIT_ALGO_PATH)
    overrides = task["overrides"]
    assert task["actor_obs_contract"] == "hope_pingpong_normal114"
    assert overrides["commands.racket_target.table_workspace_fixed_level"] == 0.0
    assert overrides["commands.racket_target.planner_perturb_fixed_scale"] == 0.0
    assert overrides["rewards.physical_capability_curriculum.params.fixed_level"] == 0.0
    assert overrides["rewards.imitation.weight"] > 0.0
    assert overrides["rewards.imitation.params.strike_scale"] == 0.10
    assert overrides["rewards.racket_velocity_projection.weight"] >= 5.0
    assert overrides["rewards.near_impact_planner_velocity_progress.weight"] >= 4.0
    assert algo["runner"]["num_steps_per_env"] == 64
    assert algo["algorithm"]["schedule"] == "fixed"
    assert algo["algorithm"]["learning_rate"] <= 1.0e-5
    assert algo["algorithm"]["num_learning_epochs"] == 2


def test_stage2_contact_alignment_v3_has_one_reward_difference() -> None:
    control = _load(CONTACT_ALIGNMENT_CONTROL_V3_PATH)
    reward = _load(CONTACT_ALIGNMENT_REWARD_V3_PATH)
    shared = control["overrides"]
    assert reward["actor_obs_contract"] == "hope_pingpong_normal114"
    assert all(reward["overrides"][key] == value for key, value in shared.items())
    reward_only = set(reward["overrides"]) - set(shared)
    assert reward_only == {
        "rewards.physical_contact_planner_alignment.weight",
        "rewards.physical_contact_planner_alignment.params.position_std",
        "rewards.physical_contact_planner_alignment.params.velocity_std",
        "rewards.physical_contact_planner_alignment.params.direction_std_deg",
        "rewards.physical_contact_planner_alignment.params.normal_std_deg",
        "rewards.physical_contact_planner_alignment.params.timing_std_s",
        "rewards.physical_contact_planner_alignment.params.component_floor",
    }
    assert reward["overrides"]["rewards.physical_contact_planner_alignment.weight"] > 0.0


def test_stage2_closed_loop_command_v4_only_changes_the_refresh_freeze_gate() -> None:
    oracle = _load(CLOSED_LOOP_ORACLE_V4_PATH)
    deploy = _load(CLOSED_LOOP_DEPLOY_V4_PATH)
    for cfg in (oracle, deploy):
        assert cfg["actor_obs_contract"] == "hope_pingpong_normal114"
        assert cfg["motion"]["wrap_teleport"] is False
        overrides = cfg["overrides"]
        assert overrides[
            "commands.physical_shadow.post_bounce_command_refresh_enabled"
        ] is True
        assert overrides[
            "commands.physical_shadow.command_refresh_solution_blend"
        ] == 1.0
        assert overrides[
            "commands.racket_target.planner_perturb_fixed_scale"
        ] == 0.0
    differing = {
        key
        for key in oracle["overrides"]
        if oracle["overrides"][key] != deploy["overrides"][key]
    }
    assert differing == {
        "commands.physical_shadow.command_refresh_freeze_tts_s"
    }
    assert oracle["overrides"][next(iter(differing))] < deploy["overrides"][
        next(iter(differing))
    ]


def test_stage2_closed_loop_command_training_ab_only_adds_planner_error() -> None:
    clean = _load(CLOSED_LOOP_CLEAN_TRAIN_V4_PATH)
    robust = _load(CLOSED_LOOP_ROBUST_TRAIN_V4_PATH)
    assert clean["actor_obs_contract"] == robust["actor_obs_contract"] == (
        "hope_pingpong_normal114"
    )
    differing = {
        key
        for key in clean["overrides"]
        if clean["overrides"][key] != robust["overrides"][key]
    }
    assert differing == {
        "commands.racket_target.planner_perturb_fixed_scale"
    }
    assert clean["overrides"][next(iter(differing))] == 0.0
    assert 0.0 < robust["overrides"][next(iter(differing))] < 1.0
    assert clean["overrides"][
        "commands.racket_target.impact_inverse_command_start_blend"
    ] == 0.30
    assert clean["overrides"][
        "commands.physical_shadow.command_refresh_freeze_tts_s"
    ] == 0.25


def test_stage2_command_executor_v5_decouples_route_motion_and_planner() -> None:
    for path in (COMMAND_EXECUTOR_CORE_V5_PATH, COMMAND_EXECUTOR_DIVERSE_V5_PATH):
        cfg = _load(path)
        overrides = cfg["overrides"]
        assert cfg["actor_obs_contract"] == "hope_pingpong_normal114"
        assert cfg["motion"]["wrap_teleport"] is False
        assert overrides["commands.physical_shadow.route_geometry_mode"] == (
            "independent"
        )
        assert overrides[
            "commands.physical_shadow.post_bounce_command_refresh_enabled"
        ] is True
        assert overrides[
            "commands.physical_shadow.command_refresh_continuous_post_bounce"
        ] is True
        assert overrides[
            "commands.physical_shadow.command_refresh_freeze_tts_s"
        ] == 0.20
        assert overrides[
            "commands.racket_target.table_workspace_motion_seed_blend_start"
        ] == 0.0
        assert overrides[
            "commands.racket_target.table_workspace_motion_seed_end_level"
        ] == 0.0
        assert overrides[
            "commands.racket_target.impact_inverse_command_curriculum_enabled"
        ] is False
        assert overrides[
            "commands.racket_target.planner_perturb_fixed_scale"
        ] == 0.0
        assert overrides[
            "rewards.physical_contact_planner_alignment.weight"
        ] > 0.0
        assert overrides[
            "rewards.physical_outcome_events.params.impact_inverse_quality_floor"
        ] == 1.0


def test_stage2_command_executor_v5_revises_from_latest_post_bounce_state() -> None:
    source = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
        / "physical_ball_shadow_command.py"
    ).read_text(encoding="utf-8")
    update_start = source.index("def _update_active_flights(")
    update_end = source.index("def _publish_metrics(", update_start)
    update_body = source[update_start:update_end]
    assert "command_refresh_continuous_post_bounce" in update_body
    assert "PhysicalShadowPhase.INCOMING_POST_BOUNCE" in update_body
    assert "self._refresh_post_bounce_commands(" in update_body
    assert "HOPEPlanner" not in source


def test_stage2_command_curriculum_v6_expands_physical_route_from_ability() -> None:
    for path in (
        COMMAND_CURRICULUM_OUTCOME_V6_PATH,
        COMMAND_CURRICULUM_ALIGNED_V6_PATH,
    ):
        cfg = _load(path)
        overrides = cfg["overrides"]
        assert cfg["actor_obs_contract"] == "hope_pingpong_normal114"
        assert cfg["motion_manifest"].endswith(
            "stage1_core_two_motion_20260801/manifest.tsv"
        )
        assert overrides[
            "commands.physical_shadow.route_ability_curriculum_enabled"
        ] is True
        assert overrides[
            "commands.physical_shadow.route_geometry_mode"
        ] == "independent"
        assert overrides[
            "commands.racket_target.table_workspace_fixed_level"
        ] == -1.0
        assert overrides[
            "rewards.physical_capability_curriculum.params.fixed_level"
        ] == -1.0
        assert overrides[
            "commands.racket_target.table_workspace_motion_seed_blend_start"
        ] == 0.0
        assert overrides[
            "commands.racket_target.planner_perturb_fixed_scale"
        ] == 0.0

        range_pairs = (
            ("pre_bounce_time_range", "route_easy_pre_bounce_time_range"),
            ("post_bounce_time_range", "route_easy_post_bounce_time_range"),
            ("bounce_dx_range", "route_easy_bounce_dx_range"),
            ("bounce_y_jitter_range", "route_easy_bounce_y_jitter_range"),
        )
        for full_name, easy_name in range_pairs:
            full = overrides[f"commands.physical_shadow.{full_name}"]
            easy = overrides[f"commands.physical_shadow.{easy_name}"]
            assert full[0] <= easy[0] <= easy[1] <= full[1]


def test_stage2_command_curriculum_v6_ab_only_changes_alignment_gate() -> None:
    outcome = _load(COMMAND_CURRICULUM_OUTCOME_V6_PATH)["overrides"]
    aligned = _load(COMMAND_CURRICULUM_ALIGNED_V6_PATH)["overrides"]
    assert set(outcome) == set(aligned)
    differing = {
        key for key in outcome if outcome[key] != aligned[key]
    }
    assert differing == {
        "rewards.physical_capability_curriculum.params.bootstrap_aligned_contact_threshold",
        "rewards.physical_capability_curriculum.params.aligned_contact_threshold",
        "rewards.physical_capability_curriculum.params.aligned_contact_regress_ratio",
    }
    assert outcome[
        "rewards.physical_capability_curriculum.params.aligned_contact_threshold"
    ] == 0.0
    assert aligned[
        "rewards.physical_capability_curriculum.params.aligned_contact_threshold"
    ] > 0.0


def test_stage2_command_curriculum_v6_gates_on_real_aligned_contact() -> None:
    source = REWARDS.read_text(encoding="utf-8")
    curriculum_start = source.index("class physical_capability_curriculum")
    curriculum = source[curriculum_start:]
    assert '"aligned_contact"' in curriculum
    assert "aligned_contact = (" in curriculum
    assert "shadow.contact_planner_position_error" in curriculum
    assert "shadow.contact_planner_velocity_error" in curriculum
    assert "shadow.contact_planner_velocity_direction_error_deg" in curriculum
    assert "shadow.contact_planner_normal_error_deg" in curriculum
    assert "torch.abs(shadow.contact_time_to_strike)" in curriculum
    assert 'self._ema["aligned_contact"]' in curriculum

    route_source = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
        / "physical_ball_shadow_command.py"
    ).read_text(encoding="utf-8")
    assert "def _route_sampling_ranges(" in route_source
    assert "target._ability_curriculum_level" in route_source
    assert "interpolate_bounds(" in route_source
    assert 'self.metrics["route_curriculum_level"]' in route_source

    env_source = ENV_CFG.read_text(encoding="utf-8")
    for key in (
        "bootstrap_contact_threshold",
        "full_contact_threshold_level",
        "aligned_contact_threshold",
        "bootstrap_aligned_contact_threshold",
        "full_aligned_contact_threshold_level",
        "aligned_contact_regress_ratio",
        "aligned_position_error_max",
        "aligned_velocity_error_max",
        "aligned_velocity_direction_error_max_deg",
        "aligned_normal_error_max_deg",
        "aligned_timing_error_max_s",
    ):
        assert f'"{key}"' in env_source


def test_stage2_command_precision_v7_only_changes_physical_credit_ratio() -> None:
    baseline = _load(COMMAND_CURRICULUM_OUTCOME_V6_PATH)
    credit = _load(COMMAND_PRECISION_CREDIT_V7_PATH)
    outcome = _load(COMMAND_PRECISION_OUTCOME_V7_PATH)

    for cfg in (credit, outcome):
        assert cfg["actor_obs_contract"] == baseline["actor_obs_contract"]
        assert cfg["motion_manifest"] == baseline["motion_manifest"]
        assert cfg["gym_task"] == baseline["gym_task"]
        assert cfg["overrides"][
            "commands.physical_shadow.route_geometry_mode"
        ] == "independent"
        assert cfg["overrides"][
            "commands.racket_target.planner_perturb_fixed_scale"
        ] == 0.0

    baseline_overrides = baseline["overrides"]
    credit_overrides = credit["overrides"]
    outcome_overrides = outcome["overrides"]
    shared_keys = set(baseline_overrides) & set(credit_overrides)
    assert {
        key
        for key in shared_keys
        if baseline_overrides[key] != credit_overrides[key]
    } == {"rewards.physical_contact_planner_alignment.weight"}
    assert credit_overrides[
        "rewards.physical_contact_planner_alignment.weight"
    ] == 6.0

    assert set(outcome_overrides) - set(credit_overrides) == {
        "rewards.physical_outcome_events.params.contact_scale",
        "rewards.physical_outcome_events.params.contact_quality_scale",
    }
    assert all(
        outcome_overrides[key] == credit_overrides[key]
        for key in credit_overrides
    )


def test_stage2_safety_credit_v8_isolates_delayed_failure_ablation() -> None:
    soft = _load(SAFETY_CREDIT_SOFT_V8_PATH)
    deferred = _load(SAFETY_CREDIT_DEFERRED_V8_PATH)
    assert soft["defaults"][0] == (
        "HOPEPingPongStage2CommandPrecisionOutcome114V7B"
    )
    assert deferred["defaults"][0] == (
        "HOPEPingPongStage2SafetyCreditSoft114V8A"
    )
    soft_overrides = soft["overrides"]
    assert soft_overrides[
        "rewards.physical_outcome_events.params.minimum_health_multiplier"
    ] == 0.0
    assert soft_overrides[
        "rewards.physical_contact_planner_alignment.params.minimum_health_multiplier"
    ] == 0.0
    assert soft_overrides[
        "rewards.physical_capability_curriculum.params.safety_floor"
    ] == 0.74
    for prefix in (
        "rewards.planner_racket_task_space_crossfade.params",
        "rewards.prestrike_racket_progress.params",
        "rewards.near_impact_planner_velocity_progress.params",
        "rewards.blade_direction.params",
    ):
        assert soft_overrides[f"{prefix}.final_minimum_health_multiplier"] < (
            soft_overrides[f"{prefix}.minimum_health_multiplier"]
        )
    assert deferred["overrides"] == {
        "rewards.physical_recovery_settlement.params.failure_cost": 0.50
    }


def test_stage2_terminal_failure_credit_is_settled_before_reset() -> None:
    terminal = _load(SAFETY_CREDIT_TERMINAL_V9_PATH)
    assert terminal["defaults"][0] == (
        "HOPEPingPongStage2SafetyCreditSoft114V8A"
    )
    assert terminal["overrides"] == {
        "rewards.physical_recovery_settlement.params.failure_cost": 0.50
    }

    source = REWARDS.read_text(encoding="utf-8")
    start = source.index("class physical_outcome_recovery_settlement")
    end = source.index("class physical_capability_curriculum", start)
    term = source[start:end]
    terminal_pos = term.index("env.termination_manager.terminated")
    failure_pos = term.index("failed = nonterminal_failed | terminal_failed")
    clear_pos = term.index("self._clear(torch.where(success | failed)[0])")
    assert terminal_pos < failure_pos < clear_pos
    assert 'target.metrics["physical_recovery_terminal_failure_event"]' in term


def test_stage2_v10_separates_hard_reset_debit_from_recovery_timeout() -> None:
    hard_debit = _load(SAFETY_CREDIT_HARD_DEBIT_V10_PATH)
    assert hard_debit["defaults"][0] == (
        "HOPEPingPongStage2SafetyCreditSoft114V8A"
    )
    assert hard_debit["overrides"] == {
        "rewards.physical_recovery_settlement.params.failure_cost": 0.20,
        "rewards.physical_recovery_settlement.params.terminal_failure_cost": 4.0,
    }

    env_source = ENV_CFG.read_text(encoding="utf-8")
    assert '"terminal_failure_cost": -1.0' in env_source
    source = REWARDS.read_text(encoding="utf-8")
    start = source.index("class physical_outcome_recovery_settlement")
    end = source.index("class physical_capability_curriculum", start)
    term = source[start:end]
    assert "nonterminal_failed = (" in term
    assert "& (~terminal_failed)" in term
    assert "- nonterminal_failed.float() * values * float(failure_cost)" in term
    assert "- terminal_failed.float() * values * hard_cost" in term


def test_physical_timing_override_is_reinitialized_on_target_resample() -> None:
    source = COMMANDS.read_text(encoding="utf-8")
    sample_start = source.index("def _sample_targets(")
    sample_end = source.index("def _assign_swing_side(", sample_start)
    sample_body = source[sample_start:sample_end]
    assert "self.physical_command_override_active[env_ids] = False" in sample_body
    assert "sampled_tts = self._motion_time_to_strike(env_ids)" in sample_body
    assert "self.true_time_to_strike[env_ids] = sampled_tts" in sample_body


def test_physical_timing_override_advances_once_per_control_step() -> None:
    source = COMMANDS.read_text(encoding="utf-8")
    timing_start = source.index("def _compute_strike_timing(")
    timing_end = source.index("def _compute_racket_state(", timing_start)
    timing_body = source[timing_start:timing_end]
    update_start = source.index("def _update_command(", timing_end)
    update_end = source.index("def _set_debug_vis_impl(", update_start)
    update_body = source[update_start:update_end]
    decrement = "self.physical_command_override_tts[active_override] -= float("
    assert decrement not in timing_body
    assert update_body.count(decrement) == 1


def test_physical_contact_alignment_uses_real_contact_and_joint_quality() -> None:
    source = REWARDS.read_text(encoding="utf-8")
    start = source.index("def physical_contact_planner_alignment(")
    end = source.index("class physical_outcome_recovery_settlement", start)
    term = source[start:end]
    assert "shadow.contact_event.float()" in term
    assert "shadow.contact_planner_position_error" in term
    assert "shadow.contact_planner_velocity_error" in term
    assert "shadow.contact_planner_velocity_direction_error_deg" in term
    assert "shadow.contact_planner_normal_error_deg" in term
    assert "shadow.contact_time_to_strike.abs()" in term
    assert "face_center_quality(" in term
    assert "target.impact_health_score" in term


def test_stage2_ball_route_is_independent_of_motion_box() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    assert 'command.strike_position_mode = "table_workspace"' in source
    assert "command.table_workspace_motion_seed_blend_start = 1.0" in source
    assert "command.table_workspace_motion_seed_end_level = 0.90" in source
    assert 'command.incoming_trajectory_mode = "one_bounce"' in source
    assert 'route_geometry_mode="target_hidden"' in source
    assert "command.impact_inverse_command_curriculum_enabled = True" in source
    assert "command.impact_inverse_command_start_blend = 0.30" in source
    assert "command.impact_inverse_command_curriculum_exponent = 1.5" in source
    assert "command.impact_inverse_command_curriculum_start_level = 0.0" in source
    assert "command.one_bounce_speed_curriculum_start_level = 0.60" in source
    assert "command.planner_perturb_curriculum_start_level = 0.75" in source


def test_stage2_curriculum_uses_physical_events_and_external_level() -> None:
    env_source = ENV_CFG.read_text(encoding="utf-8")
    reward_source = REWARDS.read_text(encoding="utf-8")
    assert "command.ability_curriculum_external_update = True" in env_source
    assert "shadow.contact_event" in reward_source
    assert "shadow.net_cross_event" in reward_source
    assert "shadow.opponent_bounce_event" in reward_source
    assert "shadow.contact_outgoing_velocity_error" in reward_source
    assert "impact_inverse_command_blend" in reward_source
    assert "shadow.contact_face_radial_error" in reward_source
    assert "physical_contact_face_quality" in reward_source
    assert '"raw_contact"' in reward_source
    assert "center_contact_radius" in reward_source
    assert "shadow.net_cross_event & self._contact" in reward_source
    assert "target._ability_curriculum_level.fill_(level)" in reward_source
    assert "contact_only_until_level" in reward_source
    assert "net_full_threshold_level" in reward_source
    assert "net_only_until_level" in reward_source
    assert "contact_regress_ratio" in reward_source
    assert "net_regress_ratio" in reward_source
    assert "bounce_regress_ratio" in reward_source
    assert "self._best_post_route_recovery" in reward_source
    assert "torch.ones(len(finished_ids)" in reward_source
    assert "self.rewards.physical_outcome_events.weight = 3.0" in env_source
    assert "self.rewards.physical_recovery_settlement.weight = 2.0" in env_source
    assert "self.rewards.physical_capability_curriculum.weight = 1.0" in env_source
    assert '"minimum_health_multiplier": 0.05' in env_source
    assert '"center_contact_radius": 0.061' in env_source
    assert '"contact_quality_scale": 1.5' in env_source
    assert '"impact_inverse_quality_floor": 0.30' in env_source
    assert '"contact_only_until_level": 0.0' in env_source
    assert '"net_full_threshold_level": 0.50' in env_source
    assert "physical_contact_quality_command_scale" in reward_source


def test_physical_stage_preserves_stage1_fixed_joint_dynamics() -> None:
    source = PHYSICAL_ENV_CFG.read_text(encoding="utf-8")
    assert "sim_utils.UsdFileCfg(" in source
    assert "AGIBOT_A3_PHYSICAL_USD_PATH" in source
    assert "wrist+mount-offset fallback" in source


def test_stage2_keeps_physical_recovery_as_deferred_settlement() -> None:
    source = REWARDS.read_text(encoding="utf-8")
    assert "shadow.outgoing_landing_event" in source
    assert 'target.metrics["recovery_functional_ready_score"]' in source
    assert "self._ready_steps >= int(required_ready_steps)" in source
    assert "target.target_just_resampled" in source


def test_stage2_b_only_changes_outcome_emphasis() -> None:
    overrides = _load(B_PATH)["overrides"]
    assert set(overrides) == {
        "rewards.physical_outcome_events.weight",
        "rewards.physical_outcome_events.params.contact_quality_scale",
        "rewards.physical_recovery_settlement.weight",
    }
    assert overrides["rewards.physical_outcome_events.weight"] > 3.0
    assert overrides["rewards.physical_recovery_settlement.weight"] > 2.0
    assert (
        overrides[
            "rewards.physical_outcome_events.params.contact_quality_scale"
        ]
        > 0.0
    )


def test_evaluator_can_freeze_workspace_and_ability_for_causal_audit() -> None:
    source = EVALUATE.read_text(encoding="utf-8")
    assert '"--fixed-workspace-level"' in source
    assert '"--fixed-ability-level"' in source
    assert "table_workspace_fixed_level" in source
    assert "physical_capability_curriculum = None" in source
    assert "cmd._ability_curriculum_level.fill_" in source
