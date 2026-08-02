from __future__ import annotations

import numpy as np

from a3_deploy_onnx_ref_pingpong.observation import ObsTarget
from a3_deploy_onnx_ref_pingpong.station import StationCommand, StationConfig


def _target(tts: float = 0.5) -> ObsTarget:
    return ObsTarget(
        pos_w=np.array([0.20, -0.30, 1.20]),
        vel_w=np.zeros(3),
        time_to_strike=tts,
        swing_side=1.0,
    )


def test_dynamic_station_matches_training_formula_and_clip() -> None:
    station = StationCommand(
        StationConfig(
            mode="dynamic_from_racket_offset",
            racket_offset_xy=(0.66259, -0.496391),
            clip_x=(0.0, 0.0),
            clip_y=(-0.10, 0.10),
        )
    )
    actual = station.update(np.array([-0.50, 0.0]), _target(), "swing")
    np.testing.assert_allclose(actual, np.array([-0.50, 0.10]), atol=1.0e-9)


def test_dynamic_station_returns_fixed_after_post_strike_window() -> None:
    station = StationCommand(
        StationConfig(
            mode="dynamic_from_racket_offset",
            racket_offset_xy=(0.66259, -0.496391),
            clip_x=(0.0, 0.0),
            clip_y=(-0.10, 0.10),
            post_strike_window_s=0.12,
        )
    )
    fixed = np.array([-0.50, 0.0])
    np.testing.assert_allclose(
        station.update(fixed, _target(tts=-0.14), "follow_through"),
        fixed,
    )
    np.testing.assert_allclose(station.update(fixed, _target(), "recovery"), fixed)
