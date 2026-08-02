from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "cfg" / "task" / "HOPEPingPongClosedLoopV3ScratchSingleCycleC8MultiSkill114.yaml"
COMMANDS = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "commands.py"
)
HOPE_COMMANDS = COMMANDS.with_name("hope_commands.py")
TERMINATIONS = COMMANDS.with_name("terminations.py")
ENV_CFG = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "config"
    / "agibot_a3"
    / "hope_env_cfg.py"
)


def _overrides() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))["overrides"]


def test_c8_keeps_a_continuous_pool_and_expands_recovery_deadline() -> None:
    overrides = _overrides()
    probabilities = overrides[
        "commands.racket_target.single_cycle_probabilities"
    ]
    assert all(right <= left for left, right in zip(probabilities, probabilities[1:]))
    assert overrides[
        "commands.racket_target.single_cycle_min_continuous_fraction"
    ] >= 0.40
    assert overrides[
        "commands.racket_target.single_cycle_deadline_steps"
    ] == 100
    assert overrides[
        "commands.racket_target.single_cycle_deadline_steps_by_level"
    ] == [6, 15, 30, 60, 100]
    assert overrides[
        "commands.racket_target.single_cycle_min_recovery_steps"
    ] == 4
    assert overrides[
        "commands.racket_target.single_cycle_continuous_only_curriculum"
    ] is True
    assert overrides[
        "terminations.single_cycle_curriculum_timeout.params"
    ] == {"command_name": "racket_target", "enabled": True}


def test_c8_forces_prestrike_and_uses_clean_timeout() -> None:
    overrides = _overrides()
    motion_source = COMMANDS.read_text(encoding="utf-8")
    command_source = HOPE_COMMANDS.read_text(encoding="utf-8")
    termination_source = TERMINATIONS.read_text(encoding="utf-8")
    env_source = ENV_CFG.read_text(encoding="utf-8")

    assert "self._move_motion_start_to_prestrike(single_cycle_ids)" in motion_source
    assert overrides["commands.motion.single_cycle_poststrike_steps"] == 40
    assert "bootstrap_end = torch.minimum(" in motion_source
    assert "settlement_resolved" in command_source
    assert "active & settlement_resolved" in command_source
    assert "self.single_cycle_timeout_latch |= resolved" in command_source
    assert "return command.single_cycle_timeout_latch" in termination_source
    assert "single_cycle_curriculum_timeout = DoneTerm(" in env_source
    single_cycle_cfg = env_source.split("single_cycle_curriculum_timeout = DoneTerm(", 1)[1]
    assert "time_out=True" in single_cycle_cfg.split(")\n", 1)[0]


def test_c8_does_not_mix_runtime_pruning_into_behavior_ab() -> None:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    assert "runtime" not in cfg
    overrides = cfg["overrides"]
    assert overrides["commands.motion.batched_metric_reset_logging"] is True
    assert overrides["commands.racket_target.batched_metric_reset_logging"] is True
