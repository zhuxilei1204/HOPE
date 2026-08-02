#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
WINDOW="${TMUX_WINDOW:-clv3V6Cto3000}"
DEVICE="${DEVICE:-cuda:1}"
NUM_ENVS="${NUM_ENVS:-4096}"
ADDITIONAL_ITERATIONS="${ADDITIONAL_ITERATIONS:-1500}"
SEED="${SEED:-738}"
TAG="${TAG:-20260801}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong_closed_loop_v3_scratch_multiskill114/2026-08-01_02-16-50_CLV3_SCRATCH_V6C_ADAPTIVE_MULTI114_4096x10000_SEED738_20260801/model_1500.pt"
RUN_NAME="${RUN_NAME:-CLV3_V6C_CONTINUE_1500_TO3000_MULTI114_${NUM_ENVS}x${ADDITIONAL_ITERATIONS}_SEED${SEED}_${TAG}}"
OUT_DIR="${OUT_DIR:-${REPO}/analysis/closed_loop_v3_scratch_20260731/${RUN_NAME}}"

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${WINDOW}"; then
  echo "tmux window already exists: ${SESSION}:${WINDOW}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cat > "${OUT_DIR}/experiment_contract.txt" <<EOF
initialization=resume_full_policy_critic_optimizer
checkpoint_path=${CHECKPOINT}
checkpoint_iteration=1500
task=HOPEPingPongClosedLoopV3ScratchAdaptiveMotionV6CMultiSkill114
role=unchanged_V6C_lineage_extended_to_iteration_3000
actor_observation_dim=114
motion_manifest=five_contact_aware_multiskill_clips
additional_iterations=${ADDITIONAL_ITERATIONS}
seed=${SEED}
device=${DEVICE}
promotion_gate=staged_64env_survival_plus_retained_hit_ability
EOF

sha256sum \
  "${CHECKPOINT}" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_closed_loop_v3_scratch_env_cfg.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/ready_recovery.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py" \
  "${ROOT}/cfg/task/HOPEPingPongClosedLoopV3ScratchAdaptiveMotionV6CMultiSkill114.yaml" \
  "${ROOT}/cfg/algo/ppo_closed_loop_v3_scratch.yaml" \
  > "${OUT_DIR}/source_sha256.txt"

tmux new-window -d -t "${SESSION}:" -n "${WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongClosedLoopV3ScratchAdaptiveMotionV6CMultiSkill114 \
algo=ppo_closed_loop_v3_scratch \
device=${DEVICE} \
num_envs=${NUM_ENVS} \
max_iterations=${ADDITIONAL_ITERATIONS} \
seed=${SEED} \
checkpoint_path='${CHECKPOINT}' \
checkpoint_actor_only=false \
checkpoint_load_optimizer=true \
run_name=${RUN_NAME} \
headless=true > '${OUT_DIR}/train.log' 2>&1"

printf 'started %s:%s\n' "${SESSION}" "${WINDOW}"
printf 'log: %s\n' "${OUT_DIR}/train.log"
