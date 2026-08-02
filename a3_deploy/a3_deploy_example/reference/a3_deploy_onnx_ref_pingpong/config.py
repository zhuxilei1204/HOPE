# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Runtime configuration loader for the reference runner.

Reads ``config/hope_pingpong_runtime.yaml`` (the clean 111-D runtime config) and
resolves the ActionAdapter, the example simulation PD gains, and the lifecycle
timing into ready-to-use arrays. All relative paths in the YAML are resolved
against the YAML file's own directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .action_adapter import ActionAdapter
from .joint_order import JOINT_NAMES, NUM_JOINTS
from .lifecycle import LifecycleConfig
from .station import StationConfig

# Index ranges of the four joint groups (used to expand example PD gains).
_GROUP_RANGES = {
    "waist": range(0, 3),
    "neck": range(3, 5),
    "arm": range(5, 19),
    "leg": range(19, 31),
}


@dataclass
class RuntimeConfig:
    control_hz: float
    onnx_path: Path
    model_xml_path: Path
    action_adapter: ActionAdapter
    sim_kp: np.ndarray
    sim_kd: np.ndarray
    lifecycle: LifecycleConfig
    station: StationConfig
    passive_neck: bool = True
    last_action_feedback_mode: str = "raw"
    config_dir: Path = field(default_factory=Path)

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.control_hz)

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeConfig":
        path = Path(path).resolve()
        cfg_dir = path.parent
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)

        norm = str(doc.get("observation_normalization", "none")).lower()
        if norm != "none":
            raise ValueError(
                f"observation_normalization must be 'none' (raw obs), got '{norm}'"
            )

        control_hz = float(doc.get("control_hz", 50.0))
        dt = 1.0 / control_hz

        onnx_path = _resolve(cfg_dir, doc["policy"]["onnx_path"])
        model_xml_path = _resolve(cfg_dir, doc["simulation"]["model_xml_path"])
        adapter_path = _resolve(cfg_dir, doc["action_adapter"]["config_path"])
        adapter = ActionAdapter.from_yaml(adapter_path)
        feedback_mode = str(doc["action_adapter"].get("last_action_feedback_mode", "raw")).lower()
        if feedback_mode not in ("raw", "effective"):
            raise ValueError(
                "action_adapter.last_action_feedback_mode must be 'raw' or 'effective', "
                f"got {feedback_mode!r}"
            )

        sim_kp, sim_kd = _expand_pd_gains(doc["simulation"]["pd_gains"])

        life_doc = doc.get("lifecycle", {})
        lifecycle = LifecycleConfig(
            dt=dt,
            follow_through_s=float(life_doc.get("follow_through_s", 0.6)),
            recovery_s=float(life_doc.get("recovery_s", 0.8)),
            ready_time_to_strike=float(life_doc.get("ready_time_to_strike", 1.0)),
            ready_reach_x=float(life_doc.get("ready_reach_x", 0.40)),
            ready_reach_y=float(life_doc.get("ready_reach_y", 0.20)),
            ready_reach_z=float(life_doc.get("ready_reach_z", -0.05)),
            recovery_blend_s=float(life_doc.get("recovery_blend_s", 0.0)),
            recovery_blend_velocity=bool(life_doc.get("recovery_blend_velocity", False)),
        )
        station_doc = doc.get("station", {})
        station = StationConfig(
            mode=str(station_doc.get("mode", "fixed")),
            racket_offset_xy=_pair(
                station_doc.get("racket_offset_xy", (0.0, 0.0)),
                "station.racket_offset_xy",
            ),
            clip_x=_pair(station_doc.get("clip_x", (0.0, 0.0)), "station.clip_x"),
            clip_y=_pair(station_doc.get("clip_y", (0.0, 0.0)), "station.clip_y"),
            blend=float(station_doc.get("blend", 1.0)),
            post_strike_window_s=float(station_doc.get("post_strike_window_s", 0.12)),
        )

        return cls(
            control_hz=control_hz,
            onnx_path=onnx_path,
            model_xml_path=model_xml_path,
            action_adapter=adapter,
            sim_kp=sim_kp,
            sim_kd=sim_kd,
            lifecycle=lifecycle,
            station=station,
            passive_neck=bool(doc.get("passive_neck", True)),
            last_action_feedback_mode=feedback_mode,
            config_dir=cfg_dir,
        )


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def _pair(value, name: str) -> tuple[float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return values


def _expand_pd_gains(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Expand example gains into length-31 kp/kd arrays.

    ``groups`` preserves the original compact runtime config. ``joints`` may then
    override exact joint names, which is useful for matching the Isaac A3 actuator
    groups more closely in the MuJoCo bridge.
    """
    kp = np.zeros(NUM_JOINTS, dtype=np.float64)
    kd = np.zeros(NUM_JOINTS, dtype=np.float64)
    groups = spec.get("groups", {})
    for name, rng in _GROUP_RANGES.items():
        if name not in groups:
            raise ValueError(f"simulation.pd_gains.groups is missing '{name}'")
        g = groups[name]
        for i in rng:
            kp[i] = float(g["kp"])
            kd[i] = float(g["kd"])
    name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
    for name, g in (spec.get("joints", {}) or {}).items():
        if name not in name_to_idx:
            raise ValueError(f"simulation.pd_gains.joints contains unknown joint '{name}'")
        i = name_to_idx[name]
        kp[i] = float(g["kp"])
        kd[i] = float(g["kd"])
    return kp, kd
