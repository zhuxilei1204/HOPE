#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: eval_stage2_closed_loop_command_member.sh GPU TASK TAG [CHECKPOINT] [STEPS] [SEED] [WORKSPACE_LEVEL] [ABILITY_LEVEL]}"
TASK="${2:?usage: eval_stage2_closed_loop_command_member.sh GPU TASK TAG [CHECKPOINT] [STEPS] [SEED] [WORKSPACE_LEVEL] [ABILITY_LEVEL]}"
TAG="${3:?usage: eval_stage2_closed_loop_command_member.sh GPU TASK TAG [CHECKPOINT] [STEPS] [SEED] [WORKSPACE_LEVEL] [ABILITY_LEVEL]}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
CHECKPOINT="${4:-${ROOT}/logs/rsl_rl/hope_pingpong_stage2_physical114/2026-08-01_20-55-40_stage2phys_cleanA_restart_fromP1_1500/model_200.pt}"
STEPS="${5:-1800}"
SEED="${6:-8830}"
WORKSPACE_LEVEL="${7:-task}"
ABILITY_LEVEL="${8:-0.0}"
OUTPUT_DIR="${ROOT}/analysis/stage2_closed_loop_command_20260802"

case "${TASK}" in
  HOPEPingPongStage2ContactAlignmentControl114V3|HOPEPingPongStage2ClosedLoopCommandOracle114V4|HOPEPingPongStage2ClosedLoopCommandDeploy114V4|HOPEPingPongStage2CommandExecutorCore114V5|HOPEPingPongStage2CommandExecutorDiverse114V5|HOPEPingPongStage2CommandCurriculumOutcome114V6A|HOPEPingPongStage2CommandCurriculumAligned114V6B|HOPEPingPongStage2CommandPrecisionCredit114V7A|HOPEPingPongStage2CommandPrecisionOutcome114V7B|HOPEPingPongStage2SafetyCreditSoft114V8A|HOPEPingPongStage2SafetyCreditDeferred114V8B|HOPEPingPongStage2SafetyCreditTerminal114V9|HOPEPingPongStage2SafetyCreditHardDebit114V10)
    ;;
  *)
    echo "unsupported task ${TASK}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "${ROOT}"

export PYTHONPATH="${ROOT}/source/whole_body_tracking${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="/tmp/hope_mpl_${TAG}"
export CUDA_VISIBLE_DEVICES="${GPU}"
EVAL_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --task-yaml "${TASK}"
  --num-envs 256
  --num-steps "${STEPS}"
  --seed "${SEED}"
  --device cuda:0
  --experiment-name hope_pingpong_stage2_physical114
  --fixed-ability-level "${ABILITY_LEVEL}"
  --json-out "${OUTPUT_DIR}/${TAG}_task.json"
  --physical-shadow-json-out "${OUTPUT_DIR}/${TAG}_physical.json"
)
if [[ "${WORKSPACE_LEVEL}" != "task" ]]; then
  EVAL_ARGS+=(--fixed-workspace-level "${WORKSPACE_LEVEL}")
fi
python scripts/evaluate.py "${EVAL_ARGS[@]}" \
  >"${OUTPUT_DIR}/${TAG}.log" 2>&1
