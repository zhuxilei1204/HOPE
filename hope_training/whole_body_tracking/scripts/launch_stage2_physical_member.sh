#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage2_physical_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT] [ANCHOR] [NOISE]}"
TASK="${2:?usage: launch_stage2_physical_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT] [ANCHOR] [NOISE]}"
SEED="${3:?usage: launch_stage2_physical_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT] [ANCHOR] [NOISE]}"
RUN_NAME="${4:?usage: launch_stage2_physical_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT] [ANCHOR] [NOISE]}"
ITERATIONS="${5:-250}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
DEFAULT_CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong_stage1_operational114/2026-08-01_18-31-02_stage1polishP1_fromB1500_250/model_1749.pt"
CHECKPOINT="${6:-${DEFAULT_CHECKPOINT}}"
ANCHOR="${7:-10.0}"
NOISE="${8:-0.10}"

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "${ROOT}"

export PYTHONPATH="${ROOT}/source/whole_body_tracking${PYTHONPATH:+:${PYTHONPATH}}"
# Isaac/PhysX creates graphics and peer contexts on every visible GPU even
# when PyTorch is assigned to one device. Isolate each process, then address
# its single visible physical GPU as cuda:0.
export CUDA_VISIBLE_DEVICES="${GPU}"
python scripts/train.py \
  task="${TASK}" \
  algo=ppo_stage2_physical \
  device="cuda:0" \
  num_envs=256 \
  max_iterations="${ITERATIONS}" \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  checkpoint_path="${CHECKPOINT}" \
  checkpoint_actor_only=true \
  checkpoint_load_optimizer=false \
  actor_anchor_coefficient="${ANCHOR}" \
  actor_anchor_checkpoint_path="${CHECKPOINT}" \
  action_noise_std_global="${NOISE}" \
  headless=true
