#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage2_contact_alignment_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT]}"
TASK="${2:?usage: launch_stage2_contact_alignment_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT]}"
SEED="${3:?usage: launch_stage2_contact_alignment_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT]}"
RUN_NAME="${4:?usage: launch_stage2_contact_alignment_member.sh GPU TASK SEED RUN_NAME [ITERATIONS] [CHECKPOINT]}"
ITERATIONS="${5:-100}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
DEFAULT_CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong_stage2_physical114/2026-08-01_20-55-40_stage2phys_cleanA_restart_fromP1_1500/model_200.pt"
CHECKPOINT="${6:-${DEFAULT_CHECKPOINT}}"

case "${TASK}" in
  HOPEPingPongStage2ContactAlignmentControl114V3|HOPEPingPongStage2ContactAlignmentReward114V3)
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
  actor_anchor_coefficient=10.0 \
  actor_anchor_checkpoint_path="${CHECKPOINT}" \
  action_noise_std_global=0.08 \
  headless=true
