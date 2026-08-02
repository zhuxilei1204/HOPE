from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "hope_stage1_plane020_escrow_env_cfg.py"
)
REGISTRY = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "__init__.py"
)
SOFT = ROOT / "cfg/task/HOPEPingPongStage1Plane020EscrowSoft114.yaml"
STRICT = ROOT / "cfg/task/HOPEPingPongStage1Plane020EscrowStrict114.yaml"
BALANCED = ROOT / "cfg/task/HOPEPingPongStage1Plane020EscrowBalanced114.yaml"
GUARD = ROOT / "cfg/task/HOPEPingPongStage1Plane020EscrowGuard114.yaml"
GUARD_MISS = ROOT / "cfg/task/HOPEPingPongStage1Plane020EscrowGuardMiss114.yaml"
IMPULSE = ROOT / "cfg/task/HOPEPingPongStage1Plane020EscrowImpulse114.yaml"
IMPULSE_MISS = (
    ROOT / "cfg/task/HOPEPingPongStage1Plane020EscrowImpulseMiss114.yaml"
)
IMPULSE_BALANCED = (
    ROOT
    / "cfg/task/HOPEPingPongStage1Plane020EscrowImpulseMissBalanced114.yaml"
)
IMPULSE_FOREHAND = (
    ROOT
    / "cfg/task/HOPEPingPongStage1Plane020EscrowImpulseMissForehand114.yaml"
)
COMMAND = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "hope_commands.py"
)
REWARDS = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp"
    / "hope_rewards.py"
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_escrow_task_is_registered_and_keeps_114d_contract() -> None:
    source = REGISTRY.read_text(encoding="utf-8")
    assert "hope_stage1_plane020_escrow_env_cfg" in source
    assert "HOPE-PingPong-Stage1-Plane020-Escrow-AgibotA3-v0" in source
    for path in (SOFT, STRICT, BALANCED):
        task = _load(path)
        assert task["gym_task"] == (
            "HOPE-PingPong-Stage1-Plane020-Escrow-AgibotA3-v0"
        )
        assert task["actor_obs_contract"] == "hope_pingpong_normal114"
        assert task["overrides"][
            "commands.motion.clip_sampling_weights"
        ] == [0.40, 0.60]


def test_soft_and_strict_differ_only_in_immediate_return_credit() -> None:
    soft = _load(SOFT)
    strict = _load(STRICT)
    assert soft["motion_manifest"] == strict["motion_manifest"]
    assert soft["actor_obs"] == strict["actor_obs"]
    assert soft["domain_rand"] == strict["domain_rand"]
    assert set(strict["overrides"]) - set(soft["overrides"]) == {
        "rewards.health_gated_net_cross.weight",
        "rewards.health_gated_opponent_bounce.weight",
    }
    assert strict["overrides"]["rewards.health_gated_net_cross.weight"] == 0.0
    assert strict["overrides"][
        "rewards.health_gated_opponent_bounce.weight"
    ] == 0.0


def test_escrow_has_delayed_safe_outcome_without_rigid_pose_anchor() -> None:
    source = ENV.read_text(encoding="utf-8")
    required = (
        "command.post_contact_ready_enabled = True",
        'command.post_contact_ready_trigger = "contact"',
        "command.post_contact_ready_durable_min_delay_s = 0.35",
        "command.post_contact_ready_durable_deadline_s = 0.70",
        "command.post_contact_ready_durable_required_consecutive_steps = 5",
        "command.post_contact_ready_min_arm_score = 0.0",
        "self.rewards.health_gated_ball_contact.weight = 2.50",
        "self.rewards.safe_terminal_outcome.weight = 8.00",
        "self.rewards.unsafe_terminal_recovery.weight = -0.50",
    )
    for snippet in required:
        assert snippet in source
    assert "anchor_pos" not in source
    assert "anchor_ori" not in source


def test_balanced_escrow_preserves_safe_contact_credit() -> None:
    strict = _load(STRICT)
    balanced = _load(BALANCED)
    shared = {
        "commands.motion.clip_sampling_weights",
        "rewards.health_gated_net_cross.weight",
        "rewards.health_gated_opponent_bounce.weight",
    }
    for key in shared:
        assert balanced["overrides"][key] == strict["overrides"][key]
    assert balanced["overrides"][
        "rewards.safe_terminal_outcome.params.tier_multipliers"
    ] == [0.80, 1.10, 1.50]


def test_capability_guard_is_ability_driven_and_reversible() -> None:
    env_source = ENV.read_text(encoding="utf-8")
    command_source = COMMAND.read_text(encoding="utf-8")
    reward_source = REWARDS.read_text(encoding="utf-8")
    registry = REGISTRY.read_text(encoding="utf-8")
    assert "HOPE-PingPong-Stage1-Plane020-EscrowGuarded-AgibotA3-v0" in registry
    assert "command.safe_outcome_capability_gate_enabled = True" in env_source
    assert "def safe_outcome_capability_gate" in command_source
    assert "return torch.min(components)" in command_source
    assert "def capability_gated_safe_terminal_outcome" in reward_source
    assert "capability * float(net_extra_value)" in reward_source
    assert "capability * float(bounce_extra_value)" in reward_source
    preservation_thresholds = (
        "command.safe_outcome_gate_contact_low = 0.50",
        "command.safe_outcome_gate_contact_high = 0.60",
        "command.safe_outcome_gate_forehand_low = 0.58",
        "command.safe_outcome_gate_forehand_high = 0.68",
        "command.safe_outcome_gate_backhand_low = 0.38",
        "command.safe_outcome_gate_backhand_high = 0.50",
        "command.safe_outcome_gate_safety_low = 0.89",
        "command.safe_outcome_gate_safety_high = 0.92",
        "command.safe_outcome_gate_recovery_low = 0.74",
        "command.safe_outcome_gate_recovery_high = 0.79",
        "command.safe_outcome_gate_settlement_low = 0.60",
        "command.safe_outcome_gate_settlement_high = 0.67",
    )
    for snippet in preservation_thresholds:
        assert snippet in env_source


def test_guard_miss_ab_changes_only_targeted_miss_cost() -> None:
    guard = _load(GUARD)
    guard_miss = _load(GUARD_MISS)
    assert guard["gym_task"] == guard_miss["gym_task"]
    assert guard["motion_manifest"] == guard_miss["motion_manifest"]
    assert guard["actor_obs"] == guard_miss["actor_obs"]
    assert guard["domain_rand"] == guard_miss["domain_rand"]
    assert guard["overrides"] == {
        "commands.motion.clip_sampling_weights": [0.40, 0.60]
    }
    assert guard_miss["overrides"] == {
        "commands.motion.clip_sampling_weights": [0.40, 0.60],
        "rewards.targeted_contact_miss.weight": -1.50,
    }
    assert "def targeted_contact_miss" in REWARDS.read_text(encoding="utf-8")


def test_impulse_escrow_uses_transition_reward_units() -> None:
    env_source = ENV.read_text(encoding="utf-8")
    reward_source = REWARDS.read_text(encoding="utf-8")
    registry = REGISTRY.read_text(encoding="utf-8")
    assert (
        "HOPE-PingPong-Stage1-Plane020-EscrowImpulse-AgibotA3-v0"
        in registry
    )
    required = (
        "class HOPEStage1Plane020EscrowImpulseEnvCfg",
        "self.rewards.health_gated_ball_contact.weight = 0.0",
        "self.rewards.targeted_strike_attempt.weight = 0.0",
        "self.rewards.strike_inactivity.weight = 0.0",
        '"contact_value": 0.50',
        '"net_extra_value": 1.50',
        '"bounce_extra_value": 3.00',
        '"impulse": True',
        "self.rewards.unsafe_terminal_recovery.weight = -1.0",
        "self.rewards.safe_strike_inactivity.weight = -0.25",
    )
    for snippet in required:
        assert snippet in env_source
    assert "def _event_manager_value" in reward_source
    assert "return value / step_dt" in reward_source
    assert "impulse: bool = False" in reward_source


def test_impulse_miss_ab_changes_only_targeted_miss_cost() -> None:
    impulse = _load(IMPULSE)
    impulse_miss = _load(IMPULSE_MISS)
    assert impulse["gym_task"] == impulse_miss["gym_task"]
    assert impulse["motion_manifest"] == impulse_miss["motion_manifest"]
    assert impulse["actor_obs"] == impulse_miss["actor_obs"]
    assert impulse["domain_rand"] == impulse_miss["domain_rand"]
    assert impulse["overrides"] == {
        "commands.motion.clip_sampling_weights": [0.40, 0.60]
    }
    assert impulse_miss["overrides"] == {
        "commands.motion.clip_sampling_weights": [0.40, 0.60],
        "rewards.targeted_contact_miss.weight": -0.50,
    }


def test_impulse_sampling_screen_changes_only_clip_mix() -> None:
    backhand = _load(IMPULSE_MISS)
    balanced = _load(IMPULSE_BALANCED)
    forehand = _load(IMPULSE_FOREHAND)
    for candidate in (balanced, forehand):
        assert candidate["gym_task"] == backhand["gym_task"]
        assert candidate["motion_manifest"] == backhand["motion_manifest"]
        assert candidate["actor_obs"] == backhand["actor_obs"]
        assert candidate["domain_rand"] == backhand["domain_rand"]
        assert candidate["overrides"][
            "rewards.targeted_contact_miss.weight"
        ] == -0.50
    assert balanced["overrides"][
        "commands.motion.clip_sampling_weights"
    ] == [0.50, 0.50]
    assert forehand["overrides"][
        "commands.motion.clip_sampling_weights"
    ] == [0.60, 0.40]
