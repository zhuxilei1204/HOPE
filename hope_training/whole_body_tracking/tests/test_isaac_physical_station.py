from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.isaac_physical_eval import (
    _CLOSED_LOOP_V2_PHYSICAL_TASK,
    _default_planner_yaml,
    _dynamic_station_for_observation,
    _rollout_state,
    _sample_one_bounce_serve,
    _validate_physical_task_contract,
)


@dataclass
class _Target:
    pos_w: np.ndarray
    time_to_strike: float


def test_dynamic_station_uses_requested_asymmetric_clip() -> None:
    target = _Target(np.array([1.5, 0.8, 1.0]), 0.4)
    row = {"motion_racket_offset_xy": np.array([0.6, 0.5])}

    station = _dynamic_station_for_observation(
        np.array([0.0, 0.0]),
        target,
        "swing",
        row,
        (0.0, 0.0),
        (-0.10, 0.16),
        0.12,
    )

    np.testing.assert_allclose(station, np.array([0.0, 0.16]))


def test_dynamic_station_returns_home_outside_active_window() -> None:
    fixed = np.array([0.2, -0.3])
    target = _Target(np.array([1.5, 0.8, 1.0]), -0.13)
    row = {"motion_racket_offset_xy": np.array([0.6, 0.5])}

    after_window = _dynamic_station_for_observation(
        fixed, target, "follow_through", row, (0.0, 0.0), (-0.10, 0.16), 0.12
    )
    recovery = _dynamic_station_for_observation(
        fixed, target, "recovery", row, (0.0, 0.0), (-0.10, 0.16), 0.12
    )

    np.testing.assert_allclose(after_window, fixed)
    np.testing.assert_allclose(recovery, fixed)


def test_closed_loop_v2_physical_contract_rejects_leaked_termination() -> None:
    expected = [
        "time_out",
        "base_too_low",
        "base_tilted",
        "table_touch",
        "persistent_action_overflow",
    ]
    _validate_physical_task_contract(_CLOSED_LOOP_V2_PHYSICAL_TASK, expected)
    with pytest.raises(RuntimeError, match="termination mismatch"):
        _validate_physical_task_contract(
            _CLOSED_LOOP_V2_PHYSICAL_TASK,
            [*expected, "anchor_pos"],
        )


def test_default_planner_prefers_current_workspace_config(tmp_path) -> None:
    current = tmp_path / "hope_ws/src/hope_planner/config/hope_planner.yaml"
    packaged = (
        tmp_path
        / "deploy_artifacts/B17996_deploy_ready_no_command_20260726/config"
        / "hope_planner.yaml"
    )
    current.parent.mkdir(parents=True)
    packaged.parent.mkdir(parents=True)
    current.write_text("current: true\n", encoding="utf-8")
    packaged.write_text("packaged: true\n", encoding="utf-8")

    assert _default_planner_yaml(tmp_path) == current


def _fake_planner_bridge(planner_drag: float = 0.18):
    simulation_physics = SimpleNamespace(
        g=np.array([0.0, 0.0, -9.81]),
        k=0.1261,
        radius=0.02,
        C_h=0.631,
        C_v=0.9215,
    )
    return SimpleNamespace(
        simulation_physics=simulation_physics,
        physics=SimpleNamespace(
            g=np.array([0.0, 0.0, -9.81]),
            k=planner_drag,
            radius=0.02,
            C_h=0.80,
            C_v=0.95,
        ),
        table=SimpleNamespace(length=2.74, width=1.525, net_x=1.37),
        config=SimpleNamespace(x_hit=0.20),
        hysteresis_y=0.04,
        split_y=-0.7625,
    )


def test_drag_rollout_is_reversible_for_route_construction() -> None:
    physics = _fake_planner_bridge().simulation_physics
    p0 = np.array([1.9, -1.1, 0.55])
    v0 = np.array([-3.2, 0.25, 0.8])
    p1, v1 = _rollout_state(p0, v0, 0.45, physics)
    p2, v2 = _rollout_state(p1, v1, -0.45, physics)

    np.testing.assert_allclose(p2, p0, atol=2.0e-9)
    np.testing.assert_allclose(v2, v0, atol=2.0e-9)


def test_physical_serve_uses_station_relative_training_workspace() -> None:
    table_to_world = np.array([0.50, 0.7625, 0.76])
    row = {
        "ranges": (
            (0.685, 0.715),
            (-0.550, -0.200),
            (1.160, 1.340),
        )
    }
    planner = _fake_planner_bridge()

    for seed in range(16):
        strike_table, origin_table, _ = _sample_one_bounce_serve(
            np.random.default_rng(seed),
            1,
            row,
            table_to_world,
            planner,
        )
        strike_world = strike_table + table_to_world
        assert strike_world[0] == pytest.approx(0.70)
        assert -0.550 <= strike_world[1] <= -0.200
        assert 1.160 <= strike_world[2] <= 1.340
        assert planner.table.net_x + 0.10 <= origin_table[0]


def test_physical_serve_is_independent_of_planner_physics_overrides() -> None:
    table_to_world = np.array([0.50, 0.7625, 0.76])
    row = {
        "ranges": (
            (0.685, 0.715),
            (-0.550, -0.200),
            (1.160, 1.340),
        )
    }
    route_a = _sample_one_bounce_serve(
        np.random.default_rng(7),
        1,
        row,
        table_to_world,
        _fake_planner_bridge(planner_drag=0.05),
    )
    route_b = _sample_one_bounce_serve(
        np.random.default_rng(7),
        1,
        row,
        table_to_world,
        _fake_planner_bridge(planner_drag=0.50),
    )

    for value_a, value_b in zip(route_a, route_b, strict=True):
        np.testing.assert_allclose(value_a, value_b)
