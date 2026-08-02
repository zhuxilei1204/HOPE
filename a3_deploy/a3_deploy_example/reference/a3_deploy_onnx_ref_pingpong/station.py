# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Deployment-side station command matching the Isaac training contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observation import ObsTarget


@dataclass
class StationConfig:
    mode: str = "fixed"
    racket_offset_xy: tuple[float, float] = (0.0, 0.0)
    clip_x: tuple[float, float] = (0.0, 0.0)
    clip_y: tuple[float, float] = (0.0, 0.0)
    blend: float = 1.0
    post_strike_window_s: float = 0.12

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "dynamic_from_racket_offset"):
            raise ValueError(f"unsupported station mode {self.mode!r}")
        if not 0.0 <= float(self.blend) <= 1.0:
            raise ValueError("station blend must be in [0, 1]")


class StationCommand:
    """Map a planner racket target to the station value in obs[101:103]."""

    def __init__(self, cfg: StationConfig) -> None:
        self.cfg = cfg

    def update(
        self,
        fixed_station_xy: np.ndarray,
        target: ObsTarget,
        phase,
    ) -> np.ndarray:
        fixed = np.asarray(fixed_station_xy, dtype=np.float64).reshape(2)
        if self.cfg.mode == "fixed":
            return fixed.copy()

        phase_value = getattr(phase, "value", str(phase))
        active = phase_value in ("swing", "follow_through")
        active &= float(target.time_to_strike) >= -float(self.cfg.post_strike_window_s)
        if not active:
            return fixed.copy()

        desired = (
            np.asarray(target.pos_w, dtype=np.float64)[:2]
            - np.asarray(self.cfg.racket_offset_xy, dtype=np.float64)
        )
        rel = desired - fixed
        rel[0] = np.clip(rel[0], *sorted(self.cfg.clip_x))
        rel[1] = np.clip(rel[1], *sorted(self.cfg.clip_y))
        return fixed + float(self.cfg.blend) * rel
