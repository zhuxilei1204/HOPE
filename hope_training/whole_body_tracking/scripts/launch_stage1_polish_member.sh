#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage1_polish_member.sh GPU TASK SEED RUN_NAME ENTROPY [ITERATIONS] [CHECKPOINT]}"
TASK="${2:?usage: launch_stage1_polish_member.sh GPU TASK SEED RUN_NAME ENTROPY [ITERATIONS] [CHECKPOINT]}"
SEED="${3:?usage: launch_stage1_polish_member.sh GPU TASK SEED RUN_NAME ENTROPY [ITERATIONS] [CHECKPOINT]}"
RUN_NAME="${4:?usage: launch_stage1_polish_member.sh GPU TASK SEED RUN_NAME ENTROPY [ITERATIONS] [CHECKPOINT]}"
ENTROPY="${5:?usage: launch_stage1_polish_member.sh GPU TASK SEED RUN_NAME ENTROPY [ITERATIONS] [CHECKPOINT]}"
ITERATIONS="${6:-250}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
DEFAULT_CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong_stage1_operational114/2026-08-01_17-54-35_stage1slewB_from1250_500/model_1500.pt"
CHECKPOINT="${7:-${DEFAULT_CHECKPOINT}}"

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "${ROOT}"

export PYTHONPATH="${ROOT}/source/whole_body_tracking${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/train.py \
  task="${TASK}" \
  algo=ppo_stage1_command_tracking \
  algo.algorithm.entropy_coef="${ENTROPY}" \
  device="cuda:${GPU}" \
  num_envs=4096 \
  max_iterations="${ITERATIONS}" \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  checkpoint_path="${CHECKPOINT}" \
  checkpoint_actor_only=false \
  checkpoint_load_optimizer=true \
  optimizer_learning_rate_after_load=0.00001 \
  headless=true
