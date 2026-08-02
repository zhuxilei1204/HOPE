#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage2_reward_v11_member.sh GPU CHECKPOINT RUN_NAME [ITERATIONS] [SEED] [ENVS]}"
CHECKPOINT="${2:?usage: launch_stage2_reward_v11_member.sh GPU CHECKPOINT RUN_NAME [ITERATIONS] [SEED] [ENVS]}"
RUN_NAME="${3:?usage: launch_stage2_reward_v11_member.sh GPU CHECKPOINT RUN_NAME [ITERATIONS] [SEED] [ENVS]}"
ITERATIONS="${4:-50}"
SEED="${5:-1301}"
ENVS="${6:-256}"
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
  task=HOPEPingPongStage2RewardV11 \
  algo=ppo_stage2_reward_v11 \
  device=cuda:0 \
  num_envs="${ENVS}" \
  max_iterations="${ITERATIONS}" \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  checkpoint_path="${CHECKPOINT}" \
  checkpoint_actor_only=false \
  checkpoint_load_optimizer=false \
  critic_warmup_iterations=3 \
  actor_anchor_coefficient=0.0 \
  action_noise_std_global=0.08 \
  headless=true
