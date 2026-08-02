from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = ROOT / (
    "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/"
    "agibot_a3/hope_stage2_reward_v11_env_cfg.py"
)
TASK_CFG = ROOT / "cfg/task/HOPEPingPongStage2RewardV11.yaml"
ALGO_CFG = ROOT / "cfg/algo/ppo_stage2_reward_v11.yaml"
TRAIN_SCRIPT = ROOT / "scripts/train.py"
LAUNCHER = ROOT / "scripts/launch_stage2_reward_v11_member.sh"
RUNNER = ROOT / (
    "source/whole_body_tracking/whole_body_tracking/utils/"
    "my_on_policy_runner.py"
)
PHYSICAL_REWARDS = ROOT / (
    "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
    "physical_stage2.py"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _weight_assignments() -> dict[str, float]:
    tree = ast.parse(ENV_CFG.read_text(encoding="utf-8"))
    result: dict[str, float] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and target.attr == "weight"
            and isinstance(target.value, ast.Attribute)
        ):
            continue
        name = target.value.attr
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            result[name] = float(value.value)
        elif (
            isinstance(value, ast.UnaryOp)
            and isinstance(value.op, ast.USub)
            and isinstance(value.operand, ast.Constant)
        ):
            result[name] = -float(value.operand.value)
    return result


def test_v11_is_isolated_and_uses_same_frame_snapshot() -> None:
    task = _load(TASK_CFG)
    source = ENV_CFG.read_text(encoding="utf-8")
    assert task["gym_task"] == "HOPE-PingPong-Stage2-RewardV11-AgibotA3-v0"
    assert task["actor_obs_contract"] == "hope_pingpong_normal114"
    assert "pre_reward_command_snapshot_enabled: bool = True" in source
    assert task["warm_start_requires_critic"] is True
    assert task["critic_warmup_iterations"] >= 3
    assert task["init_at_random_ep_len"] is False
    zero = source.index("_zero_all_reward_terms(self.rewards)")
    first_active = source.index("self.rewards.imitation.weight", zero)
    assert zero < first_active


def test_v11_reward_whitelist_has_no_per_frame_ready_payoff() -> None:
    weights = _weight_assignments()
    forbidden = {
        "no_command_ready_stability",
        "functional_no_command_ready",
        "active_ready_sustained_bonus",
        "next_swing_ready_bonus",
        "recovery_health",
    }
    assert forbidden.isdisjoint(weights)
    assert weights["no_command_instability"] < 0.0
    assert weights["durable_recovery_success"] == 0.0
    assert weights["durable_recovery_failure"] < 0.0
    assert weights["physical_recovery_settlement"] > 0.0
    assert weights["physical_contact_planner_alignment"] > weights[
        "physical_recovery_settlement"
    ]
    source = ENV_CFG.read_text(encoding="utf-8")
    assert '"include_no_command_ready"\n        ] = False' in source


def test_v11_recovery_gate_is_directional_and_ability_driven() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    assert 'post_contact_ready_progress_error_mode = "bounded_interval"' in source
    assert "post_contact_ready_curriculum_enabled = True" in source
    assert "post_contact_ready_curriculum_torso_x_max" in source
    assert "post_contact_ready_durable_use_effective_gate = True" in source
    assert "post_contact_ready_curriculum_min_targeted_attempt_ema" in source
    assert "post_contact_ready_curriculum_min_return_success_ema" in source


def test_v11_one_shot_terms_are_impulses() -> None:
    source = ENV_CFG.read_text(encoding="utf-8")
    assert source.count('"impulse": True') >= 5
    assert '"require_durable_recovery": True' in source
    assert '"require_safe_settlement": True' in source
    for term in (
        "physical_outcome_events",
        "physical_contact_planner_alignment",
        "physical_recovery_settlement",
        "durable_recovery_success",
        "durable_recovery_failure",
        "safe_strike_inactivity",
    ):
        assert term in source


def test_v11_physical_value_waits_for_durable_ready_resolution() -> None:
    source = PHYSICAL_REWARDS.read_text(encoding="utf-8")
    assert "require_durable_recovery: bool = False" in source
    assert "success = resolved_active & self._durable_success" in source
    assert "self._durable_failure" in source
    assert "self._durable_success[ids] = False" in source
    assert "success &= self._safe_settlement" in source
    assert "self._unsafe_settlement[ids] = False" in source


def test_v11_uses_one_joint_task_space_objective() -> None:
    weights = _weight_assignments()
    assert weights["planner_racket_task_space_crossfade"] > 0.0
    for duplicate in (
        "racket_position",
        "racket_velocity",
        "racket_velocity_projection",
        "blade_direction",
        "near_impact_planner_velocity_progress",
    ):
        assert duplicate not in weights


def test_v11_rollout_covers_fixed_recovery_credit_horizon() -> None:
    algo = _load(ALGO_CFG)
    assert algo["runner"]["num_steps_per_env"] >= 128
    assert algo["algorithm"]["gamma"] >= 0.995
    assert algo["algorithm"]["lam"] >= 0.98
    assert algo["algorithm"]["learning_rate"] <= 2.0e-5


def test_v11_warm_start_cannot_use_a_random_critic() -> None:
    train_source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    runner_source = RUNNER.read_text(encoding="utf-8")
    assert "warm_start_requires_critic" in train_source
    assert "checkpoint_actor_only=false" in train_source
    assert "configure_critic_only_warmup" in train_source
    assert 'cfg.task.get("init_at_random_ep_len", True)' in train_source
    assert 'parameter.requires_grad_(False)' in runner_source
    assert 'losses["critic_only_warmup"]' in runner_source


def test_v11_launcher_keeps_the_audited_training_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for expected in (
        "task=HOPEPingPongStage2RewardV11",
        "algo=ppo_stage2_reward_v11",
        "checkpoint_actor_only=false",
        "checkpoint_load_optimizer=false",
        "critic_warmup_iterations=3",
        "actor_anchor_coefficient=0.0",
        "action_noise_std_global=0.08",
    ):
        assert expected in source
