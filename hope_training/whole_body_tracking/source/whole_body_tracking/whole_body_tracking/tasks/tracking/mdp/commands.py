"""Motion-imitation command: replays a reference clip and exposes tracking targets.

:class:`MotionCommand` owns the per-env reference clock and drives the imitation reward. It supports
one or more clips concatenated on a single time axis (the HOPE default passes a forehand and a
backhand clip); each env imitates one clip ("segment") at a time, chosen uniformly per swing so all
forehand/backhand transitions appear in training. The racket-target command (``hope_commands.py``)
rides on top of this term and reads ``clip_id`` / ``time_steps`` / ``in_hold``.

Continuous-rally lifecycle: at a clip wrap the robot is NOT teleported — it must physically
carry its body from the previous swing's end into the next swing's windup (``wrap_teleport=False``).
Only a true episode reset re-initializes the robot (default stand, or reference-state-init onto the
clip frame).

NPZ schema consumed by :class:`MotionLoader` (per clip; keep small, see the motion YAML sidecars):

    fps            : scalar frames-per-second
    joint_pos      : float32 [T, 31]         joint order = canonical joint order
    joint_vel      : float32 [T, 31]
    body_pos_w     : float32 [T, B, 3]        tracked bodies only, in ``body_names`` order
    body_quat_w    : float32 [T, B, 4]        (w, x, y, z)
    body_lin_vel_w : float32 [T, B, 3]
    body_ang_vel_w : float32 [T, B, 3]

``B`` is the number of tracked bodies (``len(cfg.body_names)``) and the body axis is stored in
``body_names`` order, so no re-indexing by articulation is needed.
"""

from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.mdp.ready_recovery import (
    motion_lifecycle_fractions,
)
from whole_body_tracking.tasks.tracking.mdp.command_metrics import (
    batched_metric_means_and_clear,
)
from whole_body_tracking.utils.action_adapter_config import load_joint_order, resolve_joint_order_mapping

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """Loads one or more reference clips onto a single concatenated time axis.

    Body arrays are stored exactly as they appear in the npz (tracked bodies, in ``body_names``
    order). Passing several files concatenates them along time and records per-clip ``seg_start`` /
    ``seg_len`` so a command can step and wrap within one clip at a time.
    """

    def __init__(self, motion_file, num_bodies: int, device: str = "cpu"):
        files = [motion_file] if isinstance(motion_file, str) else list(motion_file)
        assert len(files) >= 1, "MotionLoader needs at least one motion file"
        jp, jv, bp, bq, bl, ba = [], [], [], [], [], []
        seg_lens = []
        self.fps = None
        for f in files:
            assert os.path.isfile(f), f"Invalid motion file path: {f}"
            data = np.load(f)
            if self.fps is None:
                self.fps = float(data["fps"])
            jp.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
            jv.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
            bp.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
            bq.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
            bl.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
            ba.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
            if bp[-1].shape[1] != num_bodies:
                raise ValueError(
                    f"Motion file {f} stores {bp[-1].shape[1]} bodies but the task tracks {num_bodies} "
                    "(cfg.body_names). The body axis must match, in body_names order."
                )
            seg_lens.append(jp[-1].shape[0])
        self.joint_pos = torch.cat(jp, dim=0)
        self.joint_vel = torch.cat(jv, dim=0)
        self.body_pos_w = torch.cat(bp, dim=0)
        self.body_quat_w = torch.cat(bq, dim=0)
        self.body_lin_vel_w = torch.cat(bl, dim=0)
        self.body_ang_vel_w = torch.cat(ba, dim=0)
        self.time_step_total = self.joint_pos.shape[0]
        self.num_segments = len(seg_lens)
        self.seg_len = torch.tensor(seg_lens, dtype=torch.long, device=device)
        self.seg_start = torch.zeros(self.num_segments, dtype=torch.long, device=device)
        if self.num_segments > 1:
            self.seg_start[1:] = torch.cumsum(self.seg_len, dim=0)[:-1]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_order_mapping = resolve_joint_order_mapping(
            self.robot.data.joint_names, canonical_joint_names=load_joint_order()
        )
        self.canonical_joint_names = self.joint_order_mapping.canonical
        self.canonical_joint_ids = tuple(self.joint_order_mapping.canonical_to_articulation)
        self._canonical_joint_ids_tensor = torch.tensor(
            self.canonical_joint_ids, dtype=torch.long, device=self.device
        )
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        # Articulation indices of the tracked bodies (for reading the robot's own body states).
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        body_groups = {
            "lower": [
                idx
                for idx, name in enumerate(self.cfg.body_names)
                if any(token in name.lower() for token in ("hip", "knee", "ankle"))
            ],
            "core": [
                idx
                for idx, name in enumerate(self.cfg.body_names)
                if any(token in name.lower() for token in ("pelvis", "torso"))
            ],
        }
        claimed = set(body_groups["lower"]) | set(body_groups["core"])
        body_groups["upper"] = [idx for idx in range(len(self.cfg.body_names)) if idx not in claimed]
        self._diagnostic_body_groups = {
            name: torch.tensor(indices, dtype=torch.long, device=self.device)
            for name, indices in body_groups.items()
            if indices
        }
        self._diagnostic_joint_groups = {
            "waist": torch.arange(0, 3, dtype=torch.long, device=self.device),
            "arms": torch.arange(5, 19, dtype=torch.long, device=self.device),
            "legs": torch.arange(19, 31, dtype=torch.long, device=self.device),
        }

        self.motion = MotionLoader(self.cfg.motion_file, len(self.cfg.body_names), device=self.device)
        if self.motion.joint_pos.shape[1] != len(self.canonical_joint_names):
            raise ValueError(
                f"motion joint_pos has {self.motion.joint_pos.shape[1]} columns but canonical order has "
                f"{len(self.canonical_joint_names)}"
            )
        if self.motion.joint_vel.shape[1] != len(self.canonical_joint_names):
            raise ValueError(
                f"motion joint_vel has {self.motion.joint_vel.shape[1]} columns but canonical order has "
                f"{len(self.canonical_joint_names)}"
            )
        self._clip_sampling_weights = None
        if self.cfg.clip_sampling_weights:
            if len(self.cfg.clip_sampling_weights) != self.motion.num_segments:
                raise ValueError(
                    "clip_sampling_weights length must match the number of motion clips: "
                    f"{len(self.cfg.clip_sampling_weights)} != {self.motion.num_segments}"
                )
            weights = torch.tensor(self.cfg.clip_sampling_weights, dtype=torch.float32, device=self.device)
            if bool(torch.any(weights < 0.0)) or float(weights.sum().item()) <= 0.0:
                raise ValueError("clip_sampling_weights must be non-negative and contain a positive weight")
            self._clip_sampling_weights = weights / weights.sum()
        if not 0 <= int(self.cfg.core_clip_count) <= self.motion.num_segments:
            raise ValueError(
                f"core_clip_count must be in [0, {self.motion.num_segments}], got {self.cfg.core_clip_count}"
            )

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._multiseg = self.motion.num_segments > 1
        # Which clip (swing side) each env is currently imitating: 0 = forehand, 1 = backhand.
        self.clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # "This env just started a new swing this step" — consumed by the racket-target command.
        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Age since the last true reset or wrap resample. This is separate from
        # env.episode_length_buf because rsl_rl may randomize that buffer at train start.
        self.steps_since_resample = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Pre-swing hold: while hold_counter > 0 the reference clock is frozen at the swing's first
        # frame ("waiting for the ball"); ``in_hold`` is exposed for rewards.
        self.hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # A true no-ball regression episode remains in READY for its complete lifetime. This is
        # sampled only on an environment reset, never on an intra-episode motion wrap.
        self.stand_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Distinguish a deploy-entry default stand from RSI/motion starts and intra-episode wraps.
        # The racket command uses this flag to expose the real no-command READY observation during
        # the initial hold instead of leaking the upcoming strike command.
        self.default_stand_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._resampling_from_wrap = False

        # Anchor-re-anchored reference body targets (recomputed each step in _update_command).
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # A small set of logging metrics (the training runner logs success_rate only, but the base
        # CommandTerm expects this dict to exist and be populated each step).
        self.motion_start_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.motion_start_prestrike_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.motion_start_recovery_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.motion_start_rsi_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # True only for episode resets selected by the racket command's
        # ability-driven one-swing bootstrap curriculum.  It persists across
        # the first intra-episode wrap and is replaced only on a true reset.
        self.single_cycle_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for key in (
            "error_joint_pos",
            "error_joint_vel",
            "error_body_pos",
            "error_body_rot",
            "motion_phase",
            "in_hold",
            "motion_start_warmup_prob",
            "motion_start_lifecycle_progress",
            "motion_start_effective_prestrike_fraction",
            "motion_start_effective_recovery_fraction",
            "motion_start_effective_rsi_fraction",
            "motion_start_reset",
            "motion_start_prestrike_reset",
            "motion_start_recovery_reset",
            "motion_start_rsi_reset",
            "single_cycle_reset",
            "motion_clip_id",
            "motion_supplemental_clip",
            "stand_episode",
            "default_stand_reset",
        ):
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
        for group_name in self._diagnostic_body_groups:
            for prefix in (
                "error_body_pos",
                "error_body_rot",
                "error_body_lin_vel",
                "error_body_ang_vel",
                "reference_body_lin_speed",
                "robot_body_lin_speed",
            ):
                self.metrics[f"{prefix}_{group_name}"] = torch.zeros(self.num_envs, device=self.device)

        for group_name in self._diagnostic_joint_groups:
            for prefix in ("error_joint_pos", "error_joint_vel", "reference_joint_speed"):
                self.metrics[f"{prefix}_{group_name}"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if not bool(self.cfg.batched_metric_reset_logging):
            return super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        extras = batched_metric_means_and_clear(self.metrics, env_ids)
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    # --- reference (target) state, hold-aware ------------------------------------------------- #
    @property
    def command(self) -> torch.Tensor:
        """Reference joint stream [joint_pos(31), joint_vel(31)] — critic/privileged use only."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def in_hold(self) -> torch.Tensor:
        return self.hold_counter > 0

    @property
    def joint_pos(self) -> torch.Tensor:
        # During the hold the reference is the default stand pose (a frozen, settled "ready"), not the
        # clip's first-frame windup transient.
        jp = self.motion.joint_pos[self.time_steps]
        dq = self.robot.data.default_joint_pos.index_select(-1, self._canonical_joint_ids_tensor)
        return torch.where(self.in_hold[:, None], dq, jp)

    @property
    def joint_vel(self) -> torch.Tensor:
        jv = self.motion.joint_vel[self.time_steps]
        return torch.where(self.in_hold[:, None], torch.zeros_like(jv), jv)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        v = self.motion.body_lin_vel_w[self.time_steps]
        return torch.where(self.in_hold[:, None, None], torch.zeros_like(v), v)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        v = self.motion.body_ang_vel_w[self.time_steps]
        return torch.where(self.in_hold[:, None, None], torch.zeros_like(v), v)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    # --- robot state (tracked bodies) --------------------------------------------------------- #
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos.index_select(-1, self._canonical_joint_ids_tensor)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel.index_select(-1, self._canonical_joint_ids_tensor)

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)
        for group_name, indices in self._diagnostic_body_groups.items():
            ref_pos = self.body_pos_relative_w[:, indices]
            robot_pos = self.robot_body_pos_w[:, indices]
            ref_quat = self.body_quat_relative_w[:, indices]
            robot_quat = self.robot_body_quat_w[:, indices]
            ref_lin = self.body_lin_vel_w[:, indices]
            robot_lin = self.robot_body_lin_vel_w[:, indices]
            ref_ang = self.body_ang_vel_w[:, indices]
            robot_ang = self.robot_body_ang_vel_w[:, indices]
            self.metrics[f"error_body_pos_{group_name}"] = torch.norm(
                ref_pos - robot_pos, dim=-1
            ).mean(dim=-1)
            self.metrics[f"error_body_rot_{group_name}"] = quat_error_magnitude(
                ref_quat, robot_quat
            ).mean(dim=-1)
            self.metrics[f"error_body_lin_vel_{group_name}"] = torch.norm(
                ref_lin - robot_lin, dim=-1
            ).mean(dim=-1)
            self.metrics[f"error_body_ang_vel_{group_name}"] = torch.norm(
                ref_ang - robot_ang, dim=-1
            ).mean(dim=-1)
            self.metrics[f"reference_body_lin_speed_{group_name}"] = torch.norm(
                ref_lin, dim=-1
            ).mean(dim=-1)
            self.metrics[f"robot_body_lin_speed_{group_name}"] = torch.norm(
                robot_lin, dim=-1
            ).mean(dim=-1)
        for group_name, indices in self._diagnostic_joint_groups.items():
            self.metrics[f"error_joint_pos_{group_name}"] = torch.norm(
                self.joint_pos[:, indices] - self.robot_joint_pos[:, indices], dim=-1
            )
            self.metrics[f"error_joint_vel_{group_name}"] = torch.norm(
                self.joint_vel[:, indices] - self.robot_joint_vel[:, indices], dim=-1
            )
            self.metrics[f"reference_joint_speed_{group_name}"] = torch.sqrt(
                torch.mean(torch.square(self.joint_vel[:, indices]), dim=-1)
            )
        self.metrics["in_hold"] = self.in_hold.float()
        if self._multiseg:
            seg_start = self.motion.seg_start[self.clip_id]
            seg_len = self.motion.seg_len[self.clip_id].clamp(min=2)
            self.metrics["motion_phase"] = (self.time_steps - seg_start).float() / (seg_len - 1).float()
        else:
            self.metrics["motion_phase"] = self.time_steps.float() / max(self.motion.time_step_total - 1, 1)
        self.metrics["motion_start_warmup_prob"] = torch.full_like(
            self.metrics["motion_start_warmup_prob"], self._motion_start_warmup_prob()
        )
        prestrike_fraction, recovery_fraction, rsi_fraction, lifecycle_progress = (
            self._motion_start_lifecycle_fractions()
        )
        self.metrics["motion_start_lifecycle_progress"] = torch.full_like(
            self.metrics["motion_start_lifecycle_progress"], lifecycle_progress
        )
        self.metrics["motion_start_effective_prestrike_fraction"] = torch.full_like(
            self.metrics["motion_start_effective_prestrike_fraction"],
            prestrike_fraction,
        )
        self.metrics["motion_start_effective_recovery_fraction"] = torch.full_like(
            self.metrics["motion_start_effective_recovery_fraction"],
            recovery_fraction,
        )
        self.metrics["motion_start_effective_rsi_fraction"] = torch.full_like(
            self.metrics["motion_start_effective_rsi_fraction"], rsi_fraction
        )
        self.metrics["motion_start_reset"] = self.motion_start_reset.float()
        self.metrics["motion_start_prestrike_reset"] = (
            self.motion_start_prestrike_reset.float()
        )
        self.metrics["motion_start_recovery_reset"] = (
            self.motion_start_recovery_reset.float()
        )
        self.metrics["motion_start_rsi_reset"] = self.motion_start_rsi_reset.float()
        self.metrics["single_cycle_reset"] = self.single_cycle_reset.float()
        self.metrics["motion_clip_id"] = self.clip_id.float()
        self.metrics["stand_episode"] = self.stand_episode.float()
        self.metrics["default_stand_reset"] = self.default_stand_reset.float()
        core_count = int(self.cfg.core_clip_count)
        if core_count > 0:
            self.metrics["motion_supplemental_clip"] = (self.clip_id >= core_count).float()
        else:
            self.metrics["motion_supplemental_clip"].zero_()

    def _sample_clip_and_start(self, env_ids: torch.Tensor, at_segment_start: bool):
        """Pick a clip per env and a start frame, optionally using configured clip probabilities."""
        n = len(env_ids)
        if self._multiseg:
            if self._clip_sampling_weights is None:
                new_clip = torch.randint(0, self.motion.num_segments, (n,), device=self.device)
            else:
                new_clip = torch.multinomial(self._clip_sampling_weights, n, replacement=True)
        else:
            new_clip = torch.zeros(n, dtype=torch.long, device=self.device)
        self.clip_id[env_ids] = new_clip
        seg_start = self.motion.seg_start[new_clip]
        if at_segment_start:
            self.time_steps[env_ids] = seg_start
        else:
            # Reference-state-init: start at a uniformly random frame inside the chosen segment.
            seg_len = self.motion.seg_len[new_clip]
            frac = sample_uniform(0.0, 1.0, (n,), device=self.device)
            self.time_steps[env_ids] = seg_start + (frac * (seg_len - 1).float()).long()

    def _move_motion_start_to_prestrike(self, env_ids: torch.Tensor) -> None:
        """Move seeded scratch resets to a short, audited pre-impact horizon."""
        if len(env_ids) == 0:
            return
        if not bool(self.cfg.motion_start_warmup_prestrike_enabled):
            return
        phases = tuple(
            float(value) for value in self.cfg.motion_start_warmup_phase_per_clip
        )
        if not phases:
            return
        if len(phases) != self.motion.num_segments:
            raise ValueError(
                "motion_start_warmup_phase_per_clip length must match motion clips: "
                f"{len(phases)} != {self.motion.num_segments}"
            )
        lo, hi = (
            int(value)
            for value in self.cfg.motion_start_warmup_prestrike_steps_range
        )
        if lo < 1 or hi < lo:
            raise ValueError(
                "motion_start_warmup_prestrike_steps_range must satisfy 1 <= lo <= hi"
            )
        clip = self.clip_id[env_ids]
        seg_start = self.motion.seg_start[clip]
        seg_len = self.motion.seg_len[clip].clamp(min=2)
        phase = torch.tensor(
            phases, dtype=torch.float32, device=self.device
        )[clip]
        strike_offset = (phase * (seg_len - 1).float()).round().long()
        lead = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)
        start_offset = torch.clamp(strike_offset - lead, min=0)
        self.time_steps[env_ids] = seg_start + start_offset
        self.motion_start_prestrike_reset[env_ids] = True

    def _move_motion_start_to_recovery(self, env_ids: torch.Tensor) -> None:
        """Move seeded scratch resets to an audited post-impact recovery frame."""
        if len(env_ids) == 0:
            return
        phases = tuple(
            float(value) for value in self.cfg.motion_start_warmup_phase_per_clip
        )
        if not phases:
            raise ValueError(
                "recovery motion starts require motion_start_warmup_phase_per_clip"
            )
        if len(phases) != self.motion.num_segments:
            raise ValueError(
                "motion_start_warmup_phase_per_clip length must match motion clips: "
                f"{len(phases)} != {self.motion.num_segments}"
            )
        lo, hi = (
            int(value)
            for value in self.cfg.motion_start_warmup_recovery_steps_range
        )
        if lo < 1 or hi < lo:
            raise ValueError(
                "motion_start_warmup_recovery_steps_range must satisfy 1 <= lo <= hi"
            )
        clip = self.clip_id[env_ids]
        seg_start = self.motion.seg_start[clip]
        seg_len = self.motion.seg_len[clip].clamp(min=2)
        phase = torch.tensor(
            phases, dtype=torch.float32, device=self.device
        )[clip]
        strike_offset = (phase * (seg_len - 1).float()).round().long()
        lag = torch.randint(lo, hi + 1, (len(env_ids),), device=self.device)
        start_offset = torch.minimum(strike_offset + lag, seg_len - 1)
        self.time_steps[env_ids] = seg_start + start_offset
        self.motion_start_recovery_reset[env_ids] = True

    @staticmethod
    def _linear_progress(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0 if value >= hi else 0.0
        return min(max((value - lo) / (hi - lo), 0.0), 1.0)

    def _motion_start_warmup_prob(self) -> float:
        """Return the current true-reset probability for starting from a clip's first frame."""
        if not bool(self.cfg.motion_start_warmup_enabled):
            return 0.0
        start_prob = min(max(float(self.cfg.motion_start_warmup_start_prob), 0.0), 1.0)
        min_prob = min(max(float(self.cfg.motion_start_warmup_min_prob), 0.0), start_prob)

        contact = 0.0
        recovery = 0.0
        try:
            command = self._env.command_manager.get_term(str(self.cfg.motion_start_warmup_command_name))
            contact_t = getattr(command, "_ability_contact_ema", None)
            recovery_t = getattr(command, "_ability_recovery_ema", None)
            if contact_t is not None:
                contact = float(contact_t.item())
            if recovery_t is not None:
                recovery = float(recovery_t.item())
        except Exception:
            return start_prob

        contact_progress = self._linear_progress(
            contact,
            float(self.cfg.motion_start_warmup_contact_low),
            float(self.cfg.motion_start_warmup_contact_high),
        )
        recovery_progress = self._linear_progress(
            recovery,
            float(self.cfg.motion_start_warmup_recovery_low),
            float(self.cfg.motion_start_warmup_recovery_high),
        )
        progress = min(contact_progress, recovery_progress)
        return min_prob + (start_prob - min_prob) * (1.0 - progress)

    def _single_cycle_bootstrap_probability(self) -> float:
        """Read the cached continuous-pool curriculum probability."""
        try:
            command = self._env.command_manager.get_term(
                str(self.cfg.motion_start_warmup_command_name)
            )
            probability_fn = getattr(
                command, "single_cycle_curriculum_probability", None
            )
            if callable(probability_fn):
                return min(max(float(probability_fn()), 0.0), 1.0)
        except Exception:
            pass
        return 0.0

    def _motion_start_lifecycle_fractions(
        self,
    ) -> tuple[float, float, float, float]:
        """Return ability-gated phase probabilities within a motion reset."""
        targeted_attempt = 0.0
        try:
            command = self._env.command_manager.get_term(
                str(self.cfg.motion_start_warmup_command_name)
            )
            targeted_attempt_t = getattr(
                command, "_ability_targeted_attempt_ema", None
            )
            if targeted_attempt_t is not None:
                targeted_attempt = float(targeted_attempt_t.item())
        except Exception:
            pass

        recovery_enabled = bool(self.cfg.motion_start_warmup_recovery_enabled)
        recovery_start = (
            float(self.cfg.motion_start_warmup_recovery_fraction)
            if recovery_enabled
            else 0.0
        )
        recovery_end = (
            float(self.cfg.motion_start_warmup_lifecycle_recovery_fraction_end)
            if recovery_enabled
            else 0.0
        )
        return motion_lifecycle_fractions(
            enabled=bool(self.cfg.motion_start_warmup_lifecycle_curriculum_enabled),
            targeted_attempt_ema=targeted_attempt,
            targeted_attempt_low=float(
                self.cfg.motion_start_warmup_lifecycle_targeted_attempt_low
            ),
            targeted_attempt_high=float(
                self.cfg.motion_start_warmup_lifecycle_targeted_attempt_high
            ),
            prestrike_start=float(
                self.cfg.motion_start_warmup_prestrike_fraction
            ),
            prestrike_end=float(
                self.cfg.motion_start_warmup_lifecycle_prestrike_fraction_end
            ),
            recovery_start=recovery_start,
            recovery_end=recovery_end,
        )

    def _write_motion_state_to_sim(self, env_ids: torch.Tensor, randomize: bool) -> None:
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        if randomize:
            range_list = [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
            ranges = torch.tensor(range_list, device=self.device)
            rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_pos[env_ids] += rand[:, 0:3]
            root_ori[env_ids] = quat_mul(quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5]), root_ori[env_ids])

            range_list = [self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
            ranges = torch.tensor(range_list, device=self.device)
            rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_lin_vel[env_ids] += rand[:, :3]
            root_ang_vel[env_ids] += rand[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        if randomize:
            joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        limits = self.robot.data.soft_joint_pos_limits.index_select(1, self._canonical_joint_ids_tensor)[env_ids]
        joint_pos[env_ids] = torch.clip(joint_pos[env_ids], limits[:, :, 0], limits[:, :, 1])
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids],
            joint_vel[env_ids],
            joint_ids=self.canonical_joint_ids,
            env_ids=env_ids,
        )
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _write_default_stand_state_to_sim(self, env_ids: torch.Tensor) -> None:
        """Reset into deploy-ready stand with optional active-balance perturbations."""
        if len(env_ids) == 0:
            return

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._env.scene.env_origins[env_ids]

        pose_ranges = torch.tensor(
            [
                self.cfg.stand_start_pose_range.get(key, (0.0, 0.0))
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            dtype=root_state.dtype,
            device=self.device,
        )
        pose_noise = sample_uniform(
            pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_state[:, :3] += pose_noise[:, :3]
        root_state[:, 3:7] = quat_mul(
            quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]),
            root_state[:, 3:7],
        )

        velocity_ranges = torch.tensor(
            [
                self.cfg.stand_start_velocity_range.get(key, (0.0, 0.0))
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            dtype=root_state.dtype,
            device=self.device,
        )
        root_state[:, 7:] = sample_uniform(
            velocity_ranges[:, 0],
            velocity_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos += sample_uniform(
            *self.cfg.stand_start_joint_position_range,
            joint_pos.shape,
            joint_pos.device,
        )
        joint_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, joint_limits[:, :, 0], joint_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(
            joint_pos,
            torch.zeros_like(self.robot.data.default_joint_vel[env_ids]),
            env_ids=env_ids,
        )

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.steps_since_resample[env_ids] = 0
        self.motion_start_reset[env_ids] = False
        self.motion_start_prestrike_reset[env_ids] = False
        self.motion_start_recovery_reset[env_ids] = False
        self.motion_start_rsi_reset[env_ids] = False
        self.default_stand_reset[env_ids] = False

        # Pre-swing hold (freeze the reference at the swing's first frame for U[lo, hi] control steps).
        lo, hi = self.cfg.hold_steps_range
        self.hold_counter[env_ids] = torch.randint(int(lo), int(hi) + 1, (len(env_ids),), device=self.device)

        # Intra-episode clip WRAP: pick the next swing side, start at its first frame, and do NOT
        # teleport — the policy physically carries its body between swings (deploy case).
        if self._resampling_from_wrap and not self.cfg.wrap_teleport:
            self._sample_clip_and_start(env_ids, at_segment_start=True)
            return

        # This is a true environment reset.  Select the bootstrap population
        # before the generic stand/RSI branches so these samples always begin
        # from an audited pre-strike frame and contain one useful swing.
        self.single_cycle_reset[env_ids] = False
        single_cycle_probability = self._single_cycle_bootstrap_probability()
        single_cycle_mask = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        if single_cycle_probability > 0.0:
            single_cycle_mask = (
                torch.rand(len(env_ids), device=self.device)
                < single_cycle_probability
            )
        single_cycle_ids = env_ids[single_cycle_mask]
        if len(single_cycle_ids) > 0:
            self.single_cycle_reset[single_cycle_ids] = True
            self.stand_episode[single_cycle_ids] = False
            self.hold_counter[single_cycle_ids] = 0
            self.motion_start_reset[single_cycle_ids] = True
            self._sample_clip_and_start(single_cycle_ids, at_segment_start=True)
            self._move_motion_start_to_prestrike(single_cycle_ids)
            self._write_motion_state_to_sim(single_cycle_ids, randomize=False)

        env_ids = env_ids[~single_cycle_mask]
        if len(env_ids) == 0:
            return

        # Dedicated no-ball episodes test active balance over a long deploy-style READY interval.
        # They are distinct from ``stand_start_prob``, which only selects the reset state for an
        # otherwise normal hitting episode.
        stand_episode_prob = min(max(float(self.cfg.stand_episode_prob), 0.0), 1.0)
        self.stand_episode[env_ids] = torch.rand(len(env_ids), device=self.device) < stand_episode_prob
        stand_episode_ids = env_ids[self.stand_episode[env_ids]]
        active_episode_ids = env_ids[~self.stand_episode[env_ids]]
        if len(stand_episode_ids) > 0:
            self._sample_clip_and_start(stand_episode_ids, at_segment_start=True)
            self._write_default_stand_state_to_sim(stand_episode_ids)
            self.default_stand_reset[stand_episode_ids] = True
            self.hold_counter[stand_episode_ids] = max(
                int(self.cfg.stand_episode_hold_steps),
                int(self.cfg.stand_start_min_hold),
            )
        if len(active_episode_ids) == 0:
            return
        env_ids = active_episode_ids

        if bool(self.cfg.motion_start_warmup_enabled):
            prob = self._motion_start_warmup_prob()
            motion_start_mask = torch.rand(len(env_ids), device=self.device) < prob
            motion_start_ids = env_ids[motion_start_mask]
            stand_ids = env_ids[~motion_start_mask]

            if len(stand_ids) > 0:
                self._sample_clip_and_start(stand_ids, at_segment_start=True)
                self._write_default_stand_state_to_sim(stand_ids)
                self.default_stand_reset[stand_ids] = True
                self.hold_counter[stand_ids] = torch.clamp(
                    self.hold_counter[stand_ids], min=int(self.cfg.stand_start_min_hold)
                )

            if len(motion_start_ids) > 0:
                self.hold_counter[motion_start_ids] = 0
                self.motion_start_reset[motion_start_ids] = True
                if bool(self.cfg.motion_start_warmup_prestrike_enabled):
                    prestrike_fraction, recovery_fraction, _, _ = (
                        self._motion_start_lifecycle_fractions()
                    )
                    draw = torch.rand(len(motion_start_ids), device=self.device)
                    prestrike_mask = draw < prestrike_fraction
                    recovery_mask = (
                        (draw >= prestrike_fraction)
                        & (draw < prestrike_fraction + recovery_fraction)
                    )
                    prestrike_ids = motion_start_ids[prestrike_mask]
                    recovery_ids = motion_start_ids[recovery_mask]
                    rsi_ids = motion_start_ids[~(prestrike_mask | recovery_mask)]
                    if len(prestrike_ids) > 0:
                        self._sample_clip_and_start(prestrike_ids, at_segment_start=True)
                        self._move_motion_start_to_prestrike(prestrike_ids)
                        self._write_motion_state_to_sim(prestrike_ids, randomize=False)
                    if len(recovery_ids) > 0:
                        self._sample_clip_and_start(recovery_ids, at_segment_start=True)
                        self._move_motion_start_to_recovery(recovery_ids)
                        self._write_motion_state_to_sim(recovery_ids, randomize=False)
                    if len(rsi_ids) > 0:
                        self._sample_clip_and_start(rsi_ids, at_segment_start=False)
                        self.motion_start_rsi_reset[rsi_ids] = True
                        self._write_motion_state_to_sim(rsi_ids, randomize=True)
                else:
                    self._sample_clip_and_start(motion_start_ids, at_segment_start=True)
                    self._write_motion_state_to_sim(motion_start_ids, randomize=False)
            return

        # TRUE episode reset: DEFAULT STAND (deploy entry) or reference-state-init (RSI) onto the clip.
        u = torch.rand(len(env_ids), device=self.device)
        stand_mask = u < float(self.cfg.stand_start_prob)
        stand_ids = env_ids[stand_mask]
        rsi_ids = env_ids[~stand_mask]

        if len(stand_ids) > 0:
            self._sample_clip_and_start(stand_ids, at_segment_start=True)
            self._write_default_stand_state_to_sim(stand_ids)
            self.default_stand_reset[stand_ids] = True
            # Give stand starts time to settle before the clip advances.
            self.hold_counter[stand_ids] = torch.clamp(
                self.hold_counter[stand_ids], min=int(self.cfg.stand_start_min_hold)
            )

        if len(rsi_ids) == 0:
            return
        self._sample_clip_and_start(rsi_ids, at_segment_start=False)
        # RSI drops the robot onto a RANDOM MID-CLIP frame with matching (noised) velocities. A
        # pre-swing hold would freeze the reference at that dynamic mid-swing frame while the robot
        # keeps its initialized momentum — a corrupted, unlearnable target. Holds only make sense
        # when the reference sits at a swing's first frame (stand starts and wraps), so RSI resets
        # advance the clock immediately.
        self.hold_counter[rsi_ids] = 0
        self._write_motion_state_to_sim(rsi_ids, randomize=True)

    def _update_command(self):
        held = self.in_hold
        self.hold_counter = torch.clamp(self.hold_counter - 1, min=0)
        self.time_steps += (~held).long()
        self.steps_since_resample += (~held).long()

        if self._multiseg:
            seg_end = self.motion.seg_start[self.clip_id] + self.motion.seg_len[self.clip_id]
            poststrike_steps = int(self.cfg.single_cycle_poststrike_steps)
            if poststrike_steps > 0 and bool(torch.any(self.single_cycle_reset)):
                phases = tuple(
                    float(value)
                    for value in self.cfg.motion_start_warmup_phase_per_clip
                )
                if len(phases) != self.motion.num_segments:
                    raise ValueError(
                        "single-cycle early wrap requires one "
                        "motion_start_warmup_phase_per_clip value per clip"
                    )
                phase = torch.tensor(
                    phases, dtype=torch.float32, device=self.device
                )[self.clip_id]
                seg_start = self.motion.seg_start[self.clip_id]
                seg_len = self.motion.seg_len[self.clip_id].clamp(min=2)
                strike_step = seg_start + (
                    phase * (seg_len - 1).float()
                ).round().long()
                bootstrap_end = torch.minimum(
                    seg_end,
                    strike_step + poststrike_steps + 1,
                )
                seg_end = torch.where(
                    self.single_cycle_reset,
                    bootstrap_end,
                    seg_end,
                )
            env_ids = torch.where(self.time_steps >= seg_end)[0]
        else:
            env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]

        self.just_resampled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if len(env_ids) > 0:
            self.just_resampled[env_ids] = True
            self._resampling_from_wrap = True
            try:
                self._resample_command(env_ids)
            finally:
                self._resampling_from_wrap = False

        # Re-anchor the reference body targets onto the robot's current xy + yaw so imitation is
        # invariant to where the robot actually is (only the anchor's z uses the reference height).
        n = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, n, 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, n, 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, n, 1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, n, 1)

        delta_pos = robot_anchor_pos.clone()
        delta_pos[..., 2] = anchor_pos[..., 2]
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(delta_ori, self.body_pos_w - anchor_pos)

    # --- debug visualization ------------------------------------------------------------------ #
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_body_visualizers"):
                self.goal_body_visualizers = [
                    VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name))
                    for name in self.cfg.body_names
                ]
            for vis in self.goal_body_visualizers:
                vis.set_visibility(True)
        elif hasattr(self, "goal_body_visualizers"):
            for vis in self.goal_body_visualizers:
                vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        for i in range(len(self.cfg.body_names)):
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for :class:`MotionCommand`."""

    class_type: type = MotionCommand

    asset_name: str = MISSING
    motion_file: str = MISSING  # a single path, or a list of clip paths (concatenated on time)
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    # Reference-state-init noise (applied only on true resets, RSI branch).
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.1, 0.1)

    # Fraction of true episode resets that start from the robot's DEFAULT STAND (deploy entry pose)
    # instead of reference-state-init onto the clip frame.
    stand_start_prob: float = 0.25
    stand_start_min_hold: int = 25
    stand_start_pose_range: dict[str, tuple[float, float]] = {}
    stand_start_velocity_range: dict[str, tuple[float, float]] = {}
    stand_start_joint_position_range: tuple[float, float] = (0.0, 0.0)

    # Fraction of true resets assigned to a complete no-ball READY regression episode. These
    # episodes remain in the frozen hold for ``stand_episode_hold_steps`` and are never sampled on
    # intra-episode clip wraps.
    stand_episode_prob: float = 0.0
    stand_episode_hold_steps: int = 100000

    # Scratch-training reset curriculum. When enabled, true episode resets use only two modes:
    # DEFAULT STAND and a seeded MOTION start. The motion-start probability fades out from measured
    # contact/recovery ability, so the final reset distribution returns to deploy-style default ready.
    motion_start_warmup_enabled: bool = False
    motion_start_warmup_command_name: str = "racket_target"
    motion_start_warmup_start_prob: float = 0.0
    motion_start_warmup_min_prob: float = 0.0
    motion_start_warmup_contact_low: float = 0.03
    motion_start_warmup_contact_high: float = 0.35
    motion_start_warmup_recovery_low: float = 0.60
    motion_start_warmup_recovery_high: float = 0.78
    # When phases are supplied, seeded starts use a short pre-impact horizon
    # instead of the first frame. Empty preserves the historical first-frame start.
    motion_start_warmup_prestrike_enabled: bool = False
    motion_start_warmup_prestrike_fraction: float = 1.0
    motion_start_warmup_phase_per_clip: tuple[float, ...] = ()
    motion_start_warmup_prestrike_steps_range: tuple[int, int] = (12, 24)
    # Optional explicit post-impact starts expose the braking and settling part
    # of each clip before full cycles are discovered. Fractions are conditional
    # on selecting a motion-seeded reset; the remainder remains unbiased RSI.
    motion_start_warmup_recovery_enabled: bool = False
    motion_start_warmup_recovery_fraction: float = 0.0
    motion_start_warmup_recovery_steps_range: tuple[int, int] = (8, 30)
    # Optional measured-capability curriculum for the conditional phase mix.
    # Disabled preserves the fixed fractions above exactly.
    motion_start_warmup_lifecycle_curriculum_enabled: bool = False
    motion_start_warmup_lifecycle_targeted_attempt_low: float = 0.01
    motion_start_warmup_lifecycle_targeted_attempt_high: float = 0.05
    motion_start_warmup_lifecycle_prestrike_fraction_end: float = 0.50
    motion_start_warmup_lifecycle_recovery_fraction_end: float = 0.15

    # Pre-swing hold: freeze the reference at the swing's first frame for U[lo, hi] control steps.
    hold_steps_range: tuple[int, int] = (0, 0)

    # Optional per-clip sampling probabilities. Empty preserves uniform sampling. ``core_clip_count``
    # only labels clips for diagnostics; clips at indices >= this value are reported as supplemental.
    clip_sampling_weights: tuple[float, ...] = ()
    core_clip_count: int = 0

    # Runtime-only logging optimization. It preserves every metric mean and
    # reset, but batches device-to-host synchronization during episode reset.
    batched_metric_reset_logging: bool = False
    # For selected one-swing bootstrap episodes, wrap this many motion frames
    # after impact instead of requiring the complete long clip. Zero preserves
    # the natural segment end for every existing task.
    single_cycle_poststrike_steps: int = 0

    # Teleport the robot onto the new clip frame at intra-episode wraps. MUST be False for the
    # continuous-rally lifecycle (the policy physically transitions swing -> swing).
    wrap_teleport: bool = False

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
