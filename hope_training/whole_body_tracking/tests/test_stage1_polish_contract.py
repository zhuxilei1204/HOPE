from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "cfg/task"
BASE_PATH = TASK_DIR / "HOPEPingPongStage1SlewRobustB114.yaml"
P1_PATH = TASK_DIR / "HOPEPingPongStage1PolishP1Conservative114.yaml"
P2_PATH = TASK_DIR / "HOPEPingPongStage1PolishP2Strong114.yaml"
LAUNCHER = ROOT / "scripts/launch_stage1_polish_member.sh"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_polish_preserves_stage1_task_distribution_contract() -> None:
    base = _load(BASE_PATH)
    for candidate in (_load(P1_PATH), _load(P2_PATH)):
        assert candidate["gym_task"] == base["gym_task"]
        assert candidate["experiment_name"] == base["experiment_name"]
        assert candidate["actor_obs_contract"] == base["actor_obs_contract"]
        assert candidate["motion_manifest"] == base["motion_manifest"]
        assert candidate["env"] == base["env"]
        assert candidate["motion"] == base["motion"]
        assert candidate["actor_obs"] == base["actor_obs"]
        assert candidate["domain_rand"] == base["domain_rand"]


def test_polish_keeps_racket_wrist_exploration_and_reduces_audited_joints() -> None:
    base = _load(BASE_PATH)["action_noise_std_overrides"]
    p1 = _load(P1_PATH)["action_noise_std_overrides"]
    p2 = _load(P2_PATH)["action_noise_std_overrides"]
    preserved = (
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        "right_elbow_joint",
    )
    audited = (
        "waist_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
    )
    for name in preserved:
        assert p1[name] == base[name]
        assert p2[name] == base[name]
    for name in audited:
        assert p2[name] < p1[name] < base[name]


def test_polish_smoothing_is_recovery_weighted_and_strike_arm_free() -> None:
    for path in (P1_PATH, P2_PATH):
        overrides = _load(path)["overrides"]
        assert overrides["rewards.joint_target_slew.params.strike_scale"] <= 0.01
        assert overrides["rewards.joint_target_slew.params.right_arm_scale"] <= 0.05
        assert (
            overrides["rewards.joint_target_slew.params.recovery_scale"]
            > overrides["rewards.joint_target_slew.params.pre_strike_scale"]
        )
        assert (
            overrides["rewards.joint_target_slew.params.hold_scale"]
            > overrides["rewards.joint_target_slew.params.recovery_scale"]
        )
        assert overrides["rewards.phase_action_rate_upper.params.joint_names"] == [
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
        ]
        assert overrides["rewards.phase_action_rate_upper.params.strike_scale"] == 0.0
        assert overrides["rewards.phase_action_rate_legs.params.strike_scale"] <= 0.001
        assert overrides["rewards.phase_action_rate_waist.params.strike_scale"] <= 0.001


def test_operational_base_declares_overridable_acceleration_cost_mode() -> None:
    source = (
        ROOT
        / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3"
        / "hope_stage1_operational_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert '"acceleration_cost_mode": "bounded_squared"' in source


def test_polish_launcher_resumes_full_b1500_with_low_learning_rate() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "stage1slewB_from1250_500/model_1500.pt" in text
    assert "checkpoint_actor_only=false" in text
    assert "checkpoint_load_optimizer=true" in text
    assert "optimizer_learning_rate_after_load=0.00001" in text
    assert 'algo.algorithm.entropy_coef="${ENTROPY}"' in text
