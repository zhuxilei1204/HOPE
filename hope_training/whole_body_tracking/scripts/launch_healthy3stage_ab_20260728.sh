#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721"
TRAIN_ROOT="${ROOT}/hope_training/whole_body_tracking"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${TRAIN_ROOT}/source/whole_body_tracking"
ANALYSIS="${ROOT}/analysis/healthy3stage_ab_20260728"
WARM_CKPT="${TRAIN_ROOT}/logs/rsl_rl/hope_pingpong/2026-07-28_15-58-50_M22750_F2_feedback122_newcolsfree_common3000_stableppo_20260728/model_22750.pt"

mkdir -p "${ANALYSIS}"

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}"
fi

start_window() {
  local window="$1"
  local log="$2"
  shift 2
  if tmux list-windows -t "${SESSION}" -F '#W' | grep -Fxq "${window}"; then
    echo "tmux window already exists: ${SESSION}:${window}" >&2
    return 1
  fi
  tmux new-window -d -t "${SESSION}" -n "${window}" -c "${TRAIN_ROOT}"
  local command quoted
  printf -v command 'export PYTHONPATH=%q; ' "${PYTHONPATH_VALUE}"
  local argv=(
    "${PY}" scripts/train.py
    task=HOPEPingPongHealthy3Stage122 \
    algo=ppo_healthy3stage \
    headless=true \
    num_envs=4096 \
    max_iterations=12000 \
    seed=20260728 \
    "$@"
  )
  printf -v quoted '%q ' "${argv[@]}"
  command+="${quoted}2>&1 | tee "
  printf -v quoted '%q' "${log}"
  command+="${quoted}"
  tmux send-keys -t "${SESSION}:${window}" "${command}" C-m
}

start_window \
  H3Scratch122 \
  "${ANALYSIS}/scratch122_console.log" \
  device=cuda:0 \
  run_name=Scratch122_Healthy3Stage_v1_12000_20260728

# Avoid Hydra/Kit startup-file collisions while preserving parallel training.
sleep 15

start_window \
  H3Warm22750 \
  "${ANALYSIS}/warm22750_console.log" \
  device=cuda:1 \
  run_name=Warm22750_Healthy3Stage_v1_12000_20260728 \
  checkpoint_path="${WARM_CKPT}" \
  checkpoint_actor_only=true

echo "started ${SESSION}:H3Scratch122 and ${SESSION}:H3Warm22750"
