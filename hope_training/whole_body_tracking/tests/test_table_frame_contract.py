from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(relative_path: str) -> dict:
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def test_simulator_translation_matches_nominal_robot_placement() -> None:
    table_frame = _load_yaml("configs/table_frame.yaml")
    base_xy_table = np.asarray(
        table_frame["nominal_robot"]["base_xy_table"], dtype=np.float64
    )
    translation = np.asarray(
        table_frame["simulation"]["table_to_world_translation_xyz"], dtype=np.float64
    )

    assert table_frame["canonical"]["table_origin"] == "near_side_left_surface_corner"
    assert np.allclose(translation[:2], -base_xy_table)
    assert np.isclose(translation[2], 0.76)
    assert table_frame["simulation"]["table_to_world_quaternion_wxyz"] == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]


def test_planner_hit_plane_is_identical_in_runtime_and_world_config() -> None:
    runtime = _load_yaml("hope_ws/src/hope_planner/config/hope_planner.yaml")
    world = _load_yaml("hope_ws/src/hope_bringup/config/hope_world_frame.yaml")

    assert np.isclose(
        runtime["hope_planner"]["ros__parameters"]["x_hit"],
        world["hope_world"]["planner"]["x_hit"],
    )


def test_ball_and_world_configs_share_table_geometry() -> None:
    ball_physics = _load_yaml("configs/ball_physics.yaml")
    world = _load_yaml("hope_ws/src/hope_bringup/config/hope_world_frame.yaml")
    world_table = world["hope_world"]["table_m"]

    assert np.isclose(ball_physics["table"]["length"], world_table["length"])
    assert np.isclose(ball_physics["table"]["width"], world_table["width"])
    assert np.isclose(ball_physics["table"]["height"], world_table["surface_height"])
    assert np.isclose(ball_physics["net"]["height"], world_table["net_height"])
