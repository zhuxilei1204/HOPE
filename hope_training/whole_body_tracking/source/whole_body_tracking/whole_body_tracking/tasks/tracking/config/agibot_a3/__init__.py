import gymnasium as gym

from . import (
    agents,
    hope_actuator_robust_env_cfg,
    hope_closed_loop_v2_env_cfg,
    hope_closed_loop_v3_scratch_env_cfg,
    hope_env_cfg,
    hope_physical_eval_env_cfg,
    hope_stage1_command_tracking_env_cfg,
    hope_stage1_operational_env_cfg,
    hope_stage1_planner_executor_env_cfg,
    hope_stage1_plane020_escrow_env_cfg,
    hope_stage1_plane020_merged_env_cfg,
    hope_stage2_physical_env_cfg,
    hope_stage2_reward_v11_env_cfg,
)

gym.register(
    id="HOPE-PingPong-ActuatorRobust-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_actuator_robust_env_cfg.HOPEActuatorRobustEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

##
# Register the single public HOPE task.
##
gym.register(
    id="HOPE-PingPong-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_env_cfg.HOPEPingPongEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg",
    },
)

gym.register(
    id="HOPE-PingPong-Stage1-CommandTracking-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_stage1_command_tracking_env_cfg
            .HOPEStage1CommandTrackingEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-Stage1-Operational-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_stage1_operational_env_cfg.HOPEStage1OperationalEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-Stage1-Plane020-Merged-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_stage1_plane020_merged_env_cfg
            .HOPEStage1Plane020MergedEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-Stage1-Plane020-Escrow-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_stage1_plane020_escrow_env_cfg
            .HOPEStage1Plane020EscrowEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-Stage1-Plane020-EscrowGuarded-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_stage1_plane020_escrow_env_cfg
            .HOPEStage1Plane020EscrowGuardedEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-Stage1-Plane020-EscrowImpulse-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_stage1_plane020_escrow_env_cfg
            .HOPEStage1Plane020EscrowImpulseEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-Stage1-PlannerExecutor-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_stage1_planner_executor_env_cfg
            .HOPEStage1PlannerExecutorEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-Stage2-Physical-AgibotA3-v0",
    entry_point=(
        "whole_body_tracking.tasks.table_tennis.table_tennis_env:"
        "TableTennisEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_stage2_physical_env_cfg.HOPEStage2PhysicalEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-Stage2-RewardV11-AgibotA3-v0",
    entry_point=(
        "whole_body_tracking.tasks.table_tennis.table_tennis_env:"
        "TableTennisEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_stage2_reward_v11_env_cfg.HOPEStage2RewardV11EnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-PhysicalEval-AgibotA3-v0",
    entry_point="whole_body_tracking.tasks.table_tennis.table_tennis_env:TableTennisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_physical_eval_env_cfg.HOPEPhysicalEvalEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg",
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-PhysicalEval-AgibotA3-v0",
    entry_point="whole_body_tracking.tasks.table_tennis.table_tennis_env:TableTennisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_physical_eval_env_cfg.HOPEClosedLoopV2PhysicalEvalEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg",
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV3-ScratchMultiSkill-PhysicalEval-AgibotA3-v0",
    entry_point="whole_body_tracking.tasks.table_tennis.table_tennis_env:TableTennisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_physical_eval_env_cfg
            .HOPEClosedLoopV3ScratchPhysicalEvalEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-PhysicalShadow-AgibotA3-v0",
    entry_point="whole_body_tracking.tasks.table_tennis.table_tennis_env:TableTennisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_physical_eval_env_cfg
            .HOPEClosedLoopV2PhysicalShadowEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-PhysicalSceneControl-AgibotA3-v0",
    entry_point="whole_body_tracking.tasks.table_tennis.table_tennis_env:TableTennisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_physical_eval_env_cfg
            .HOPEClosedLoopV2PhysicalSceneControlEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_closed_loop_v2_env_cfg.HOPEClosedLoopV2EnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg",
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV3-ScratchMultiSkill-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_closed_loop_v3_scratch_env_cfg
            .HOPEClosedLoopV3ScratchMultiSkillEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-Impact-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hope_closed_loop_v2_env_cfg.HOPEClosedLoopV2ImpactEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg",
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-DurableCycle-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_closed_loop_v2_env_cfg.HOPEClosedLoopV2DurableCycleEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-SafeQualityCycle-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_closed_loop_v2_env_cfg
            .HOPEClosedLoopV2SafeQualityCycleEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-SafeFaceQualityCycle-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_closed_loop_v2_env_cfg
            .HOPEClosedLoopV2SafeFaceQualityCycleEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-ImpactRecovery-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_closed_loop_v2_env_cfg.HOPEClosedLoopV2ImpactRecoveryEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)

gym.register(
    id="HOPE-PingPong-ClosedLoopV2-ImpactConstraint-AgibotA3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            hope_closed_loop_v2_env_cfg.HOPEClosedLoopV2ImpactConstraintEnvCfg
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.ppo:HOPEPingPongPPORunnerCfg"
        ),
    },
)
