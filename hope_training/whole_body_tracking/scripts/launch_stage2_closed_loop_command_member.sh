#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage2_closed_loop_command_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT] [ANCHOR_COEFFICIENT]}"
TASK="${2:?usage: launch_stage2_closed_loop_command_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT] [ANCHOR_COEFFICIENT]}"
SEED="${3:?usage: launch_stage2_closed_loop_command_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT] [ANCHOR_COEFFICIENT]}"
RUN_NAME="${4:?usage: launch_stage2_closed_loop_command_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT] [ANCHOR_COEFFICIENT]}"
ITERATIONS="${5:-100}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
DEFAULT_CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong_stage2_physical114/2026-08-01_20-55-40_stage2phys_cleanA_restart_fromP1_1500/model_200.pt"
CHECKPOINT="${6:-${DEFAULT_CHECKPOINT}}"
ANCHOR_COEFFICIENT="${7:-10.0}"

case "${TASK}" in
  HOPEPingPongStage2ClosedLoopCommandOracle114V4|HOPEPingPongStage2ClosedLoopCommandDeploy114V4|HOPEPingPongStage2ClosedLoopCommandCleanTrain114V4|HOPEPingPongStage2ClosedLoopCommandRobustTrain114V4|HOPEPingPongStage2CommandExecutorCore114V5|HOPEPingPongStage2CommandExecutorDiverse114V5|HOPEPingPongStage2CommandCurriculumOutcome114V6A|HOPEPingPongStage2CommandCurriculumAligned114V6B|HOPEPingPongStage2CommandPrecisionCredit114V7A|HOPEPingPongStage2CommandPrecisionOutcome114V7B|HOPEPingPongStage2SafetyCreditSoft114V8A|HOPEPingPongStage2SafetyCreditDeferred114V8B|HOPEPingPongStage2SafetyCreditTerminal114V9|HOPEPingPongStage2SafetyCreditHardDebit114V10)
    ;;
  *)
    echo "unsupported task ${TASK}" >&2
    exit 2
    ;;
esac

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "${ROOT}"

export PYTHONPATH="${ROOT}/source/whole_body_tracking${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
python scripts/train.py \
  task="${TASK}" \
  algo=ppo_stage2_impact_credit \
  device=cuda:0 \
  num_envs=256 \
  max_iterations="${ITERATIONS}" \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  checkpoint_path="${CHECKPOINT}" \
  checkpoint_actor_only=true \
  checkpoint_load_optimizer=false \
  actor_anchor_coefficient="${ANCHOR_COEFFICIENT}" \
  actor_anchor_checkpoint_path="${CHECKPOINT}" \
  action_noise_std_global=0.08 \
  headless=true
