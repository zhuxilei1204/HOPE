from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "cfg/contracts/hope_stage1_planner_executor_v1.yaml"
TASK = ROOT / "cfg/task/HOPEPingPongStage1PlannerExecutor114.yaml"
ALGO = ROOT / "cfg/algo/ppo_stage1_planner_executor.yaml"
ENV_SOURCE = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "hope_stage1_planner_executor_env_cfg.py"
)
REWARD_SOURCE = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "hope_rewards.py"
)
REGISTRY = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "__init__.py"
)
LAUNCHER = ROOT / "scripts/launch_stage1_planner_executor.sh"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_task_preserves_114d_deployment_contract() -> None:
    task = _load(TASK)
    assert task["gym_task"] == (
        "HOPE-PingPong-Stage1-PlannerExecutor-AgibotA3-v0"
    )
    assert task["actor_obs_contract"] == "hope_pingpong_normal114"
    assert "stage1_core_two_motion" in task["motion_manifest"]
    assert task["actor_obs"] == {
        "racket_target_normal_w": True,
        "stability_feedback": False,
    }


def test_stage1_has_clean_command_and_no_physical_outcome_objective() -> None:
    source = ENV_SOURCE.read_text(encoding="utf-8")
    required = (
        'command.racket_velocity_mode = "impact_inverse_landing"',
        'command.incoming_trajectory_mode = "one_bounce"',
        'command.planner_command_mode = "v4_wire_compatible"',
        "command.planner_perturb_fixed_scale = 0.0",
        'command.post_contact_ready_trigger = "targeted_attempt"',
        "self.rewards.exact_impact_planner_task_space_alignment.weight = 3.0",
        "self.rewards.racket_position.weight = 1.20",
        "self.rewards.racket_velocity.weight = 0.60",
        "self.rewards.blade_direction.weight = 0.40",
        "self.rewards.safe_recovered_planner_command.weight = 6.0",
        "self.rewards.command_cycle_failure.weight = -1.0",
        "self.rewards.termination_penalty.weight = -12.0",
        "self.rewards.healthy_trunk_support.weight = 0.08",
        "self.rewards.post_strike_base_ang_vel.weight = 0.15",
        '"ability_scaled_std": True',
        '"include_hold": False',
        "self.rewards.recovery_peak_ang_vel_excess.weight = -0.25",
        "self.rewards.table_no_touch.weight = -1.50",
        '"ability_scaled": True',
        '"ability_start_scale": 0.08',
        '"ability_attempt_start": 0.20',
        '"ability_attempt_full": 0.45',
    )
    for snippet in required:
        assert snippet in source
    forbidden_assignments = (
        "self.rewards.health_gated_net_cross.weight =",
        "self.rewards.health_gated_opponent_bounce.weight =",
        "self.rewards.safe_terminal_outcome.weight =",
        "self.rewards.capability_gated_safe_terminal_outcome.weight =",
    )
    for snippet in forbidden_assignments:
        assert snippet not in source


def test_sparse_command_rewards_use_impulse_accounting() -> None:
    source = REWARD_SOURCE.read_text(encoding="utf-8")
    assert "def stage1_safe_recovered_planner_command(" in source
    assert "def stage1_command_cycle_failure(" in source
    assert "impulse: bool = False" in source
    assert "return _event_manager_value(env, value, impulse=impulse)" in source
    assert (
        "value = cmd.post_contact_ready_peak_ang_vel_excess_increment"
        in source
    )


def test_contract_separates_stage1_and_stage2_truth() -> None:
    contract = _load(CONTRACT)
    assert contract["task"]["rigid_ball"] is False
    assert contract["stage_boundary"][
        "stage2_must_preserve_stage1_as_auxiliary_contract"
    ] is True
    assert "net_cross" in contract["stage_boundary"][
        "stage1_does_not_optimize"
    ]
    assert "safe_recovered_planner_command" in contract[
        "required_nonzero_rewards"
    ]
    assert contract["learning_bootstrap"][
        "high_value_requires_joint_command_and_safe_recovery"
    ] is True
    assert contract["learning_bootstrap"][
        "recovery_gradient_constrains_right_arm_directly"
    ] is False
    assert contract["learning_bootstrap"][
        "ordinary_motion_hold_earns_post_strike_damping"
    ] is False
    assert contract["event_accounting"][
        "recovery_peak_increment_is_impulse"
    ] is True
    assert "health_gated_net_cross" in contract[
        "forbidden_reward_terms"
    ]


def test_registry_launcher_and_scratch_ppo_are_isolated() -> None:
    registry = REGISTRY.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    algo = _load(ALGO)
    assert "HOPE-PingPong-Stage1-PlannerExecutor-AgibotA3-v0" in registry
    assert "task=HOPEPingPongStage1PlannerExecutor114" in launcher
    assert "algo=ppo_stage1_planner_executor" in launcher
    assert algo["policy"]["init_noise_std"] == 0.45
    assert algo["algorithm"]["learning_rate"] == 0.0005
