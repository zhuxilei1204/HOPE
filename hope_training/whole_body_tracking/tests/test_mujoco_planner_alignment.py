from __future__ import annotations

import math
from pathlib import Path

import pytest

import numpy as np

from scripts.mujoco_eval_onnx import (
    _planner_alignment_diagnostics,
    _racket_face_normal,
)


def test_planner_alignment_diagnostics_use_world_frame_vectors() -> None:
    result = _planner_alignment_diagnostics(
        racket_pos=[0.0, 0.0, 0.0],
        racket_vel=[1.0, 0.0, 0.0],
        racket_normal=[0.0, 0.0, 1.0],
        target_pos=[3.0, 4.0, 0.0],
        target_vel=[0.0, 1.0, 0.0],
        target_normal=[0.0, 0.0, 1.0],
    )
    assert result["racket_target_pos_error"] == pytest.approx(5.0)
    assert result["racket_target_vel_error"] == pytest.approx(math.sqrt(2.0))
    assert result["racket_target_vel_angle_deg"] == pytest.approx(90.0)
    assert result["racket_target_normal_error_deg"] == pytest.approx(0.0)


def test_planner_alignment_reports_undefined_zero_velocity_angle() -> None:
    result = _planner_alignment_diagnostics(
        racket_pos=[0.0, 0.0, 0.0],
        racket_vel=[0.0, 0.0, 0.0],
        racket_normal=[1.0, 0.0, 0.0],
        target_pos=[0.0, 0.0, 0.0],
        target_vel=[1.0, 0.0, 0.0],
        target_normal=[1.0, 0.0, 0.0],
    )
    assert result["racket_target_vel_angle_deg"] is None


def test_mujoco_racket_normal_matches_isaac_local_y_contract() -> None:
    np.testing.assert_allclose(
        _racket_face_normal(np.eye(3), "forehand"),
        [0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        _racket_face_normal(np.eye(3), "backhand"),
        [0.0, -1.0, 0.0],
    )


def test_mujoco_racket_site_preserves_urdf_frame_but_mesh_geom_does_not() -> None:
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[3]
    model_path = (
        root
        / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg"
        / "model/a3_pingpong/a3_pingpong.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_racket"
    )
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_racket_collision"
    )
    site_xmat = data.site_xmat[site_id].reshape(3, 3)
    geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)

    np.testing.assert_allclose(
        _racket_face_normal(site_xmat, "forehand"),
        [0.0, 1.0, 0.0],
        atol=1.0e-8,
    )
    assert _angle_between(
        _racket_face_normal(site_xmat, "forehand"),
        _racket_face_normal(geom_xmat, "forehand"),
    ) == pytest.approx(90.0, abs=1.0e-6)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
                    -1.0,
                    1.0,
                )
            )
        )
    )
