from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.planner_snapshot import (
    instantiate_with_supported_kwargs,
    load_planner_api,
)


_REPO = Path(__file__).resolve().parents[3]
_V4_PACKAGE = (
    _REPO
    / "analysis/external_alignment_packages_20260731/package1"
    / "real_robot_planner_rl_alignment_20260731"
    / "01_planner_V4_A4_plane_recorded/snapshot/hope_planner_code"
)


def test_loads_local_and_recorded_v4_as_distinct_packages() -> None:
    local = load_planner_api(_REPO)
    recorded = load_planner_api(_REPO, _V4_PACKAGE)

    assert local.package_dir.name == "hope_planner"
    assert recorded.package_dir.name == "hope_planner_code"
    assert local.HOPEPlanner.__module__ == "hope_planner.planner"
    assert recorded.HOPEPlanner.__module__ == "hope_planner_code.planner"
    assert recorded.CommandStabilityGate.__module__.startswith("hope_planner_code")
    assert local.source_sha256 != recorded.source_sha256


def test_rejects_non_package_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="__init__.py"):
        load_planner_api(_REPO, tmp_path)


def test_instantiate_filters_fields_for_versioned_types() -> None:
    class SmallConfig:
        def __init__(self, used: int):
            self.used = used

    config = instantiate_with_supported_kwargs(
        SmallConfig, {"used": 3, "newer_only": 7}
    )

    assert config.used == 3


def test_recorded_v4_gate_matches_yaml_publication_contract() -> None:
    recorded = load_planner_api(_REPO, _V4_PACKAGE)
    with (_V4_PACKAGE.parent / "hope_planner.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        params = yaml.safe_load(stream)["hope_planner"]["ros__parameters"]
    config = instantiate_with_supported_kwargs(
        recorded.CommandStabilityConfig,
        {
            "initial_consecutive": params["revision_gate_initial_consecutive"],
            "initial_position_tolerance_m": params[
                "revision_gate_initial_position_tolerance_m"
            ],
            "max_position_jump_m": params[
                "revision_gate_max_position_jump_m"
            ],
            "max_velocity_jump_mps": params[
                "revision_gate_max_velocity_jump_mps"
            ],
            "freeze_time_to_strike_s": params[
                "revision_gate_freeze_tts_s"
            ],
            "max_strike_time_jump_s": params[
                "revision_gate_max_strike_time_jump_s"
            ],
        },
    )
    gate = recorded.CommandStabilityGate(config)
    position = np.array([0.20, -0.80, 0.90])
    velocity = np.array([1.0, 0.0, 0.5])

    assert not gate.consider(position, velocity, 0.30, candidate_time_s=1.00)
    assert gate.last_reason == "awaiting_convergence"
    assert not gate.consider(position, velocity, 0.28, candidate_time_s=1.02)
    assert gate.last_reason == "awaiting_convergence"
    assert gate.consider(position, velocity, 0.26, candidate_time_s=1.04)
    assert gate.last_reason == "accepted_initial"
    assert not gate.consider(position, velocity, 0.08, candidate_time_s=1.22)
    assert gate.last_reason == "late_freeze"
