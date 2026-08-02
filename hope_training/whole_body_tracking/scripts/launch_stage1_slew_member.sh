#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage1_slew_member.sh GPU TASK SEED RUN_NAME [ITERATIONS]}"
TASK="${2:?usage: launch_stage1_slew_member.sh GPU TASK SEED RUN_NAME [ITERATIONS]}"
SEED="${3:?usage: launch_stage1_slew_member.sh GPU TASK SEED RUN_NAME [ITERATIONS]}"
RUN_NAME="${4:?usage: launch_stage1_slew_member.sh GPU TASK SEED RUN_NAME [ITERATIONS]}"
ITERATIONS="${5:-500}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong_stage1_operational114/2026-08-01_16-38-15_stage1op_seed821_screen1500/model_1250.pt"

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "${ROOT}"

export PYTHONPATH="${ROOT}/source/whole_body_tracking${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/train.py \
  task="${TASK}" \
  algo=ppo_stage1_command_tracking \
  device="cuda:${GPU}" \
  num_envs=4096 \
  max_iterations="${ITERATIONS}" \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  checkpoint_path="${CHECKPOINT}" \
  checkpoint_actor_only=false \
  checkpoint_load_optimizer=true \
  optimizer_learning_rate_after_load=0.00003 \
  headless=true
