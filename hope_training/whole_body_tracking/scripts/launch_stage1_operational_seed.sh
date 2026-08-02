#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage1_operational_seed.sh GPU SEED [MAX_ITERATIONS]}"
SEED="${2:?usage: launch_stage1_operational_seed.sh GPU SEED [MAX_ITERATIONS]}"
MAX_ITERATIONS="${3:-1500}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "${ROOT}"

export PYTHONPATH="${ROOT}/source/whole_body_tracking${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/train.py \
  task=HOPEPingPongStage1Operational114 \
  algo=ppo_stage1_command_tracking \
  device="cuda:${GPU}" \
  num_envs=4096 \
  max_iterations="${MAX_ITERATIONS}" \
  seed="${SEED}" \
  run_name="stage1op_seed${SEED}_screen${MAX_ITERATIONS}" \
  headless=true
