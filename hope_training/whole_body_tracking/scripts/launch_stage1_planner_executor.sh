#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage1_planner_executor.sh GPU SEED RUN_NAME [ITERATIONS] [NUM_ENVS]}"
SEED="${2:?usage: launch_stage1_planner_executor.sh GPU SEED RUN_NAME [ITERATIONS] [NUM_ENVS]}"
RUN_NAME="${3:?usage: launch_stage1_planner_executor.sh GPU SEED RUN_NAME [ITERATIONS] [NUM_ENVS]}"
ITERATIONS="${4:-300}"
NUM_ENVS="${5:-4096}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "${ROOT}"
export PYTHONPATH="${ROOT}/source/whole_body_tracking${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/train.py \
  task=HOPEPingPongStage1PlannerExecutor114 \
  algo=ppo_stage1_planner_executor \
  device="cuda:${GPU}" \
  num_envs="${NUM_ENVS}" \
  max_iterations="${ITERATIONS}" \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  headless=true
