from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "cfg/contracts/hope_stage1_plane020_merged_v1.yaml"
TASK = ROOT / "cfg/task/HOPEPingPongStage1Plane020Merged114.yaml"
BH_REPLAY_TASK = (
    ROOT / "cfg/task/HOPEPingPongStage1Plane020Merged114BHReplay.yaml"
)
ENV_SOURCE = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
    / "hope_stage1_plane020_merged_env_cfg.py"
)
LAUNCHER = ROOT / "scripts/launch_stage1_plane020_member.sh"
AUDITOR = ROOT / "scripts/audit_stage1_plane020_merged.py"
RUNNER = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py"
)
ACTOR_ANCHOR = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils/actor_anchor.py"
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _audit_module():
    spec = importlib.util.spec_from_file_location("plane020_merged_audit", AUDITOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runner_module():
    """Load runner glue without importing the Isaac-dependent package root."""
    names = (
        "whole_body_tracking",
        "whole_body_tracking.utils",
        "whole_body_tracking.utils.actor_anchor",
    )
    saved = {name: sys.modules.get(name) for name in names}
    try:
        package = ModuleType("whole_body_tracking")
        package.__path__ = []
        utils = ModuleType("whole_body_tracking.utils")
        utils.__path__ = []
        sys.modules["whole_body_tracking"] = package
        sys.modules["whole_body_tracking.utils"] = utils
        anchor_spec = importlib.util.spec_from_file_location(
            "whole_body_tracking.utils.actor_anchor", ACTOR_ANCHOR
        )
        assert anchor_spec and anchor_spec.loader
        anchor_module = importlib.util.module_from_spec(anchor_spec)
        sys.modules[anchor_spec.name] = anchor_module
        anchor_spec.loader.exec_module(anchor_module)

        runner_spec = importlib.util.spec_from_file_location(
            "plane020_test_runner", RUNNER
        )
        assert runner_spec and runner_spec.loader
        runner_module = importlib.util.module_from_spec(runner_spec)
        runner_spec.loader.exec_module(runner_module)
        return runner_module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_task_uses_114d_two_motion_isolated_gym_task() -> None:
    task = _load(TASK)
    assert task["gym_task"] == "HOPE-PingPong-Stage1-Plane020-Merged-AgibotA3-v0"
    assert task["actor_obs_contract"] == "hope_pingpong_normal114"
    assert "stage1_core_two_motion" in task["motion_manifest"]
    assert task["actor_obs"] == {
        "racket_target_normal_w": True,
        "stability_feedback": False,
    }


def test_backhand_replay_is_an_isolated_sampling_ab() -> None:
    control = _load(TASK)
    replay = _load(BH_REPLAY_TASK)
    assert replay["gym_task"] == control["gym_task"]
    assert replay["motion_manifest"] == control["motion_manifest"]
    assert replay["actor_obs"] == control["actor_obs"]
    assert replay["domain_rand"] == control["domain_rand"]
    assert replay["overrides"] == {
        "commands.motion.clip_sampling_weights": [0.40, 0.60]
    }


def test_environment_contract_decouples_motion_targets_and_keeps_footwork() -> None:
    source = ENV_SOURCE.read_text(encoding="utf-8")
    required = (
        'command.strike_position_mode = "table_workspace"',
        'command.planner_hit_plane_mode = "fixed_x_hit"',
        "command.planner_hit_plane_x = 0.20",
        "command.table_workspace_motion_seed_blend_start = 0.0",
        'command.station_mode = "dynamic_from_motion"',
        "command.ability_curriculum_enabled = True",
        "command.ability_curriculum_min_resolved_events = 4096",
        "command.ability_curriculum_required_advance_checks = 2",
        "self.rewards.phase_action_rate_legs.weight = -0.007",
        "command.impact_health_reward_power = 2.0",
        "command.ability_curriculum_require_healthy_impact = True",
        "self.rewards.exact_impact_planner_task_space_alignment.weight = 4.0",
        "self.rewards.health_gated_backhand_soft_ball_contact.weight = 1.20",
        "self.rewards.termination_penalty.weight = -60.0",
    )
    for snippet in required:
        assert snippet in source
    assert "self.rewards.phase_lower_body_motion_prior.weight" not in source


def test_reward_contract_has_no_idle_or_lower_body_prior_optimum() -> None:
    contract = _load(CONTRACT)
    required = set(contract["required_nonzero_rewards"])
    forbidden = set(contract["forbidden_reward_terms"])
    assert not (required & forbidden)
    assert "targeted_strike_attempt" in required
    assert "health_gated_ball_contact" in required
    assert "health_gated_backhand_soft_ball_contact" in required
    assert "exact_impact_planner_task_space_alignment" in required
    assert "prestrike_station_progress" in required
    assert "phase_lower_body_motion_prior" in forbidden
    budget = _audit_module().reward_budget(contract)
    assert budget["active_over_idle"] >= contract["reward_budget"][
        "required_active_over_idle_ratio"
    ]


def test_launcher_supports_safe_transfer_and_scratch_ab() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'INIT_MODE="${5:-scratch}"' in source
    assert "checkpoint_actor_only=false" in source
    assert "checkpoint_load_optimizer=false" in source
    assert "optimizer_learning_rate_after_load=0.0001" in source
    assert "critic_warmup_iterations=25" in source
    assert "transfer-guarded" in source
    assert "actor_step_trust_region_rms=0.01" in source
    assert "actor_step_trust_region_p99=0.05" in source
    assert "transfer-guarded-slow" in source
    assert "transfer-frozen-audit" in source
    assert "critic_warmup_iterations=1000000" in source
    assert "optimizer_learning_rate_after_load=0.000003" in source
    assert "actor_step_trust_region_rms=0.005" in source
    assert "actor_step_trust_region_p99=0.025" in source
    assert 'TASK="${8:-HOPEPingPongStage1Plane020Merged114}"' in source
    assert "ppo_stage1_plane020_merged" in source


def test_actor_step_trust_region_bounds_behavior_delta() -> None:
    runner_class = _runner_module().HOPEOnPolicyRunner

    class Policy(torch.nn.Module):
        is_recurrent = False

        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Linear(2, 1, bias=False)
            self.log_std = torch.nn.Parameter(torch.zeros(1))

        def act_inference(self, observations):
            return self.actor(observations)

    policy = Policy()
    storage = SimpleNamespace(observations=torch.ones(2, 8, 2))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)

    def unsafe_update():
        with torch.no_grad():
            policy.actor.weight.add_(1.0)
        return {"surrogate": 0.0}

    runner = object.__new__(runner_class)
    runner.alg = SimpleNamespace(
        policy=policy,
        storage=storage,
        optimizer=optimizer,
        update=unsafe_update,
    )
    before = policy.act_inference(storage.observations.reshape(-1, 2)).detach()
    runner.configure_actor_step_trust_region(
        max_action_rms=0.02,
        max_action_p99=0.10,
        max_samples=16,
    )
    losses = runner.alg.update()
    after = policy.act_inference(storage.observations.reshape(-1, 2)).detach()
    drift = torch.abs(after - before)

    assert losses["actor_step_guarded"] == 1.0
    assert losses["actor_step_scale"] < 1.0
    assert float(torch.sqrt(torch.mean(torch.square(drift)))) <= 0.02001
    assert float(torch.quantile(drift.reshape(-1), 0.99)) <= 0.10001
