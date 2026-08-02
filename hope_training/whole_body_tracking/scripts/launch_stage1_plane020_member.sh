#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: launch_stage1_plane020_member.sh GPU SEED RUN_NAME [ITERATIONS] [scratch|transfer|transfer-resume|transfer-guarded|transfer-guarded-slow|transfer-frozen-audit|actor-only] [NUM_ENVS] [CHECKPOINT] [TASK]}"
SEED="${2:?usage: launch_stage1_plane020_member.sh GPU SEED RUN_NAME [ITERATIONS] [scratch|transfer|transfer-resume|transfer-guarded|transfer-guarded-slow|transfer-frozen-audit|actor-only] [NUM_ENVS] [CHECKPOINT] [TASK]}"
RUN_NAME="${3:?usage: launch_stage1_plane020_member.sh GPU SEED RUN_NAME [ITERATIONS] [scratch|transfer|transfer-resume|transfer-guarded|transfer-guarded-slow|transfer-frozen-audit|actor-only] [NUM_ENVS] [CHECKPOINT] [TASK]}"
ITERATIONS="${4:-300}"
INIT_MODE="${5:-scratch}"
NUM_ENVS="${6:-4096}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
DEFAULT_CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong_stage1_operational114/2026-08-01_18-31-02_stage1polishP1_fromB1500_250/model_1749.pt"
CHECKPOINT="${7:-${DEFAULT_CHECKPOINT}}"
TASK="${8:-HOPEPingPongStage1Plane020Merged114}"

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "${ROOT}"
export PYTHONPATH="${ROOT}/source/whole_body_tracking${PYTHONPATH:+:${PYTHONPATH}}"

EXTRA_ARGS=()
case "${INIT_MODE}" in
  scratch)
    ;;
  transfer)
    EXTRA_ARGS+=(
      checkpoint_path="${CHECKPOINT}"
      checkpoint_actor_only=false
      checkpoint_load_optimizer=false
      optimizer_learning_rate_after_load=0.0001
      critic_warmup_iterations=25
      algo.algorithm.entropy_coef=0.003
      algo.algorithm.desired_kl=0.004
    )
    ;;
  transfer-resume)
    # Preserve Adam moments from the parent policy and make the first actor
    # updates deliberately small. A fresh optimizer caused a one-update jump
    # from valid actions to persistent overflow after critic warmup.
    EXTRA_ARGS+=(
      checkpoint_path="${CHECKPOINT}"
      checkpoint_actor_only=false
      checkpoint_load_optimizer=true
      optimizer_learning_rate_after_load=0.00001
      critic_warmup_iterations=25
      algo.algorithm.entropy_coef=0.003
      algo.algorithm.desired_kl=0.004
      algo.algorithm.num_learning_epochs=2
      algo.algorithm.clip_param=0.1
      algo.algorithm.max_grad_norm=0.5
    )
    ;;
  transfer-guarded)
    EXTRA_ARGS+=(
      checkpoint_path="${CHECKPOINT}"
      checkpoint_actor_only=false
      checkpoint_load_optimizer=false
      optimizer_learning_rate_after_load=0.00001
      critic_warmup_iterations=25
      actor_step_trust_region_rms=0.01
      actor_step_trust_region_p99=0.05
      actor_step_trust_region_max_samples=4096
      algo.algorithm.entropy_coef=0.003
      algo.algorithm.desired_kl=0.004
      algo.algorithm.num_learning_epochs=2
      algo.algorithm.clip_param=0.1
      algo.algorithm.max_grad_norm=0.5
    )
    ;;
  transfer-guarded-slow)
    # Long-horizon capability-preserving fine-tune. The tighter per-update
    # action trust region controls local drift; persistent side/safety gates
    # still select checkpoints because local bounds do not prevent cumulative
    # policy drift.
    EXTRA_ARGS+=(
      checkpoint_path="${CHECKPOINT}"
      checkpoint_actor_only=false
      checkpoint_load_optimizer=false
      optimizer_learning_rate_after_load=0.000003
      critic_warmup_iterations=10
      actor_step_trust_region_rms=0.005
      actor_step_trust_region_p99=0.025
      actor_step_trust_region_max_samples=4096
      algo.algorithm.entropy_coef=0.003
      algo.algorithm.desired_kl=0.002
      algo.algorithm.num_learning_epochs=2
      algo.algorithm.clip_param=0.1
      algo.algorithm.max_grad_norm=0.5
    )
    ;;
  transfer-frozen-audit)
    # Collect rollout/event statistics without changing actor parameters.
    # The critic may adapt, but critic warmup exceeds any intended audit run.
    EXTRA_ARGS+=(
      checkpoint_path="${CHECKPOINT}"
      checkpoint_actor_only=false
      checkpoint_load_optimizer=false
      optimizer_learning_rate_after_load=0.000003
      critic_warmup_iterations=1000000
      algo.algorithm.entropy_coef=0.003
      algo.algorithm.desired_kl=0.002
      algo.algorithm.num_learning_epochs=2
      algo.algorithm.clip_param=0.1
      algo.algorithm.max_grad_norm=0.5
    )
    ;;
  actor-only)
    EXTRA_ARGS+=(
      checkpoint_path="${CHECKPOINT}"
      checkpoint_actor_only=true
      checkpoint_load_optimizer=false
      critic_warmup_iterations=50
      action_noise_std_global=0.20
      algo.algorithm.learning_rate=0.0001
      algo.algorithm.entropy_coef=0.003
    )
    ;;
  *)
    echo "unknown init mode: ${INIT_MODE}" >&2
    exit 2
    ;;
esac

python scripts/train.py \
  task="${TASK}" \
  algo=ppo_stage1_plane020_merged \
  device="cuda:${GPU}" \
  num_envs="${NUM_ENVS}" \
  max_iterations="${ITERATIONS}" \
  seed="${SEED}" \
  run_name="${RUN_NAME}" \
  headless=true \
  "${EXTRA_ARGS[@]}"
