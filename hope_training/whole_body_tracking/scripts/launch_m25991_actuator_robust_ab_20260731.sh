#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong/2026-07-30_13-41-29_M24492_DeployStationStepV1_4096x1500_20260730/model_25991.pt"
OUT_DIR="${REPO}/analysis/m25991_actuator_robust_ab_20260731"
ITERATIONS="${ITERATIONS:-500}"
NUM_ENVS="${NUM_ENVS:-4096}"
CONTROL_WINDOW="M25991ActCtrl"
ROBUST_WINDOW="M25991ActRob"

mkdir -p "${OUT_DIR}/control" "${OUT_DIR}/actuator_robust"

for window in "${CONTROL_WINDOW}" "${ROBUST_WINDOW}"; do
  if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${window}"; then
    echo "tmux window already exists: ${SESSION}:${window}" >&2
    exit 1
  fi
done

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

COMMON=(
  headless=true
  "num_envs=${NUM_ENVS}"
  "max_iterations=${ITERATIONS}"
  seed=0
  "checkpoint_path=${CHECKPOINT}"
  checkpoint_actor_only=false
  checkpoint_load_optimizer=true
  optimizer_learning_rate_after_load=null
  actor_anchor_coefficient=0.0
  algo.runner.save_interval=100
)

tmux new-window -d -t "${SESSION}:" -n "${CONTROL_WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongM25991ReplayControlV1 \
${COMMON[*]} \
device=cuda:0 \
run_name=M25991_ACTUATOR_AB_CONTROL_4096x${ITERATIONS}_20260731 \
> '${OUT_DIR}/control/train.log' 2>&1"

# Avoid simultaneous cold-start conversion of shared simulation assets.
sleep 20

tmux new-window -d -t "${SESSION}:" -n "${ROBUST_WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongM25991ActuatorRobustLightV1 \
${COMMON[*]} \
device=cuda:1 \
run_name=M25991_ACTUATOR_AB_ROBUST_LIGHT_4096x${ITERATIONS}_20260731 \
> '${OUT_DIR}/actuator_robust/train.log' 2>&1"

printf 'started model25991 actuator causal A/B:\n'
printf '  %s:%s (GPU0, unchanged replay control)\n' "${SESSION}" "${CONTROL_WINDOW}"
printf '  %s:%s (GPU1, actuator robust light)\n' "${SESSION}" "${ROBUST_WINDOW}"
printf '  checkpoint: %s\n' "${CHECKPOINT}"
printf '  rollout: %s envs x %s iterations per branch\n' "${NUM_ENVS}" "${ITERATIONS}"
printf '  logs: %s\n' "${OUT_DIR}"
