#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage2_precision_long.sh GPU CHECKPOINT RUN_NAME [ITERATIONS] [ANCHOR_COEFFICIENT] [SEED]}"
CHECKPOINT="${2:?usage: launch_stage2_precision_long.sh GPU CHECKPOINT RUN_NAME [ITERATIONS] [ANCHOR_COEFFICIENT] [SEED]}"
RUN_NAME="${3:?usage: launch_stage2_precision_long.sh GPU CHECKPOINT RUN_NAME [ITERATIONS] [ANCHOR_COEFFICIENT] [SEED]}"
ITERATIONS="${4:-3000}"
ANCHOR_COEFFICIENT="${5:-2.0}"
SEED="${6:-8862}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "checkpoint does not exist or is empty: ${CHECKPOINT}" >&2
  exit 2
fi

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "${ROOT}"

export PYTHONPATH="${ROOT}/source/whole_body_tracking${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
python scripts/train.py \
  task=HOPEPingPongStage2CommandPrecisionOutcome114V7B \
  algo=ppo_stage2_impact_credit \
  device=cuda:0 \
  num_envs=256 \
  max_iterations="${ITERATIONS}" \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  checkpoint_path="${CHECKPOINT}" \
  checkpoint_actor_only=false \
  checkpoint_load_optimizer=true \
  actor_anchor_coefficient="${ANCHOR_COEFFICIENT}" \
  actor_anchor_checkpoint_path="${CHECKPOINT}" \
  action_noise_std_global=0.08 \
  headless=true
