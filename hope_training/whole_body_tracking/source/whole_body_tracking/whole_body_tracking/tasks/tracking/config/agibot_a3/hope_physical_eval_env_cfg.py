"""Physical-ball evaluation scene for a trained HOPE A3 policy.

This is deliberately an evaluation-only hybrid:

* the robot, action adapter, observations, commands, and terminations come from
  :class:`HOPEPingPongEnvCfg`;
* the regulation table, net, visual mesh, and rigid ping-pong ball come from the
  physical table-tennis task;
* the table-frame assets are translated into the robot-centred HOPE simulation
  frame using ``configs/table_frame.yaml``.

The evaluator drives the existing racket-target command tensors from a real
``HOPEPlanner`` lifecycle.  Nothing in this config changes the training task.
"""

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.table_tennis.mdp as table_mdp
import whole_body_tracking.tasks.tracking.mdp as tracking_mdp
from whole_body_tracking.robots.agibot_a3 import (
    A3_RACKET_BODY,
    A3_WRIST_BODY,
    AGIBOT_A3_PHYSICAL_USD_PATH,
)
from whole_body_tracking.tasks.table_tennis.ball import BallAerodynamicsCfg
from whole_body_tracking.tasks.table_tennis import geometry
from whole_body_tracking.tasks.table_tennis.geometry import BounceMaterials
from whole_body_tracking.tasks.table_tennis.table_tennis_env_cfg import (
    build_ball_cfg,
    build_center_line_cfg,
    build_net_cfg,
    build_net_post_cfg,
    build_table_top_cfg,
    build_table_usd_visual_cfg,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import MySceneCfg

from .hope_env_cfg import CommandsCfg, HOPEPingPongEnvCfg, _load_table_frame_config
from .hope_closed_loop_v2_env_cfg import (
    HOPEClosedLoopV2EnvCfg,
    HOPEClosedLoopV2SafeQualityCycleEnvCfg,
)
from .hope_closed_loop_v3_scratch_env_cfg import (
    HOPEClosedLoopV3ScratchMultiSkillEnvCfg,
)


_TABLE_FRAME = _load_table_frame_config()
_TABLE_TO_WORLD = tuple(float(v) for v in _TABLE_FRAME["translation"])
_MATERIALS = BounceMaterials()


def _translated(cfg):
    """Deep-copy an asset config and translate its environment-local pose."""
    out = copy.deepcopy(cfg)
    pos = tuple(float(v) for v in out.init_state.pos)
    out.init_state.pos = tuple(pos[i] + _TABLE_TO_WORLD[i] for i in range(3))
    return out


@configclass
class HOPEPhysicalEvalSceneCfg(MySceneCfg):
    """HOPE robot-centred scene augmented with physical table-tennis assets."""

    table = _translated(build_table_top_cfg(_MATERIALS, visible=False))
    net = _translated(build_net_cfg(_MATERIALS, visible=False))
    net_post_left = _translated(
        build_net_post_cfg(
            "{ENV_REGEX_NS}/NetPostLeft", geometry.NET_OVERHANG, visible=False
        )
    )
    net_post_right = _translated(
        build_net_post_cfg(
            "{ENV_REGEX_NS}/NetPostRight",
            -geometry.TABLE_WIDTH - geometry.NET_OVERHANG,
            visible=False,
        )
    )
    center_line = _translated(build_center_line_cfg(visible=False))
    table_visual = _translated(build_table_usd_visual_cfg())
    ball = _translated(build_ball_cfg(_MATERIALS))


@configclass
class HOPEPhysicalShadowSceneCfg(MySceneCfg):
    """Headless training scene with only collision-relevant ball assets."""

    table = _translated(build_table_top_cfg(_MATERIALS, visible=False))
    net = _translated(build_net_cfg(_MATERIALS, visible=False))
    ball = _translated(build_ball_cfg(_MATERIALS))


def _configure_physical_eval(cfg) -> None:
    """Add rigid-ball simulation settings without changing task semantics."""
    source_spawn = cfg.scene.robot.spawn
    # Preserve the Stage-1 articulation dynamics. Keeping every decorative
    # fixed link as a separate PhysX body introduces many fixed constraints and
    # tiny-inertia racket marker bodies, which destabilizes the policy before a
    # ball can be reached. With merging enabled the racket command and contact
    # capture intentionally use the existing wrist+mount-offset fallback.
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=AGIBOT_A3_PHYSICAL_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=copy.deepcopy(source_spawn.rigid_props),
        articulation_props=copy.deepcopy(source_spawn.articulation_props),
    )
    cfg.decimation = 8
    cfg.sim.dt = 0.0025
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physx.enable_ccd = True
    cfg.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

    mats = BounceMaterials()
    cfg.events.racket_material = EventTerm(
        func=table_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=f"({A3_RACKET_BODY}|{A3_WRIST_BODY})"
            ),
            "static_friction_range": (
                mats.paddle_static_friction,
                mats.paddle_static_friction,
            ),
            "dynamic_friction_range": (
                mats.paddle_dynamic_friction,
                mats.paddle_dynamic_friction,
            ),
            "restitution_range": (
                mats.paddle_restitution,
                mats.paddle_restitution,
            ),
            "num_buckets": 1,
        },
    )

    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (-1.35, -3.15, 1.95)
    cfg.viewer.lookat = (1.25, 0.0, 0.95)


@configclass
class HOPEPhysicalEvalEnvCfg(HOPEPingPongEnvCfg):
    """Evaluation-only HOPE task with a rigid ball and regulation table."""

    scene: HOPEPhysicalEvalSceneCfg = HOPEPhysicalEvalSceneCfg(num_envs=1, env_spacing=6.0)
    ball_aerodynamics: BallAerodynamicsCfg = BallAerodynamicsCfg.from_physics_config(enabled=True)

    def __post_init__(self):
        super().__post_init__()
        _configure_physical_eval(self)


@configclass
class HOPEClosedLoopV2PhysicalEvalEnvCfg(HOPEClosedLoopV2EnvCfg):
    """Rigid-ball evaluator that preserves closed-loop-v2 task semantics."""

    scene: HOPEPhysicalEvalSceneCfg = HOPEPhysicalEvalSceneCfg(
        num_envs=1, env_spacing=6.0
    )
    ball_aerodynamics: BallAerodynamicsCfg = BallAerodynamicsCfg.from_physics_config(
        enabled=True
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_physical_eval(self)


@configclass
class HOPEClosedLoopV3ScratchPhysicalEvalEnvCfg(
    HOPEClosedLoopV3ScratchMultiSkillEnvCfg
):
    """Rigid-ball evaluator preserving the complete V3 scratch task contract."""

    scene: HOPEPhysicalEvalSceneCfg = HOPEPhysicalEvalSceneCfg(
        num_envs=1, env_spacing=6.0
    )
    ball_aerodynamics: BallAerodynamicsCfg = BallAerodynamicsCfg.from_physics_config(
        enabled=True
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_physical_eval(self)


@configclass
class HOPEPhysicalShadowCommandsCfg(CommandsCfg):
    """Original HOPE commands plus a reward-free rigid-ball observer."""

    physical_shadow = tracking_mdp.PhysicalBallShadowCommandCfg(
        ball_asset_name="ball",
        target_command_name="racket_target",
        debug_vis=False,
    )


@configclass
class HOPEClosedLoopV2PhysicalShadowEnvCfg(
    HOPEClosedLoopV2SafeQualityCycleEnvCfg
):
    """SafeQualityCycle with a hidden physical ball and unchanged policy task."""

    scene: HOPEPhysicalShadowSceneCfg = HOPEPhysicalShadowSceneCfg(
        num_envs=256, env_spacing=6.0
    )
    commands: HOPEPhysicalShadowCommandsCfg = HOPEPhysicalShadowCommandsCfg()
    ball_aerodynamics: BallAerodynamicsCfg = (
        BallAerodynamicsCfg.from_physics_config(enabled=True)
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_physical_eval(self)


@configclass
class HOPEClosedLoopV2PhysicalSceneControlEnvCfg(
    HOPEClosedLoopV2SafeQualityCycleEnvCfg
):
    """Physical-scene overhead control without the shadow command term."""

    scene: HOPEPhysicalShadowSceneCfg = HOPEPhysicalShadowSceneCfg(
        num_envs=256, env_spacing=6.0
    )
    ball_aerodynamics: BallAerodynamicsCfg = (
        BallAerodynamicsCfg.from_physics_config(enabled=True)
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_physical_eval(self)
