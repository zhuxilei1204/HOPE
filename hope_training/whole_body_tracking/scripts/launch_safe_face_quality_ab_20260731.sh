#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong_closed_loop_v2_safe_quality_cycle/2026-07-31_07-34-57_CLV2_SAFEQUALITYCYCLEV2_R2C23993_256x100_SMOKE_20260731/model_99.pt"
OUT_DIR="${REPO}/analysis/closed_loop_v2_20260731/safe_face_quality_ab_256x100"
ITERATIONS="${ITERATIONS:-100}"

mkdir -p "${OUT_DIR}/control" "${OUT_DIR}/face_quality"

for window in FQCtrl100 FQFace100; do
  if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${window}"; then
    echo "tmux window already exists: ${SESSION}:${window}" >&2
    exit 1
  fi
done

COMMON=(
  algo=ppo_closed_loop_v2_durable
  headless=true
  num_envs=256
  "max_iterations=${ITERATIONS}"
  seed=20260731
  checkpoint_actor_only=true
  checkpoint_load_optimizer=false
  actor_anchor_coefficient=0.0
  algo.runner.save_interval=25
  "checkpoint_path=${CHECKPOINT}"
)

tmux new-window -d -t "${SESSION}:" -n FQCtrl100 \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongClosedLoopV2SafeQualityCycleFH \
${COMMON[*]} \
device=cuda:0 \
run_name=CLV2_FACE_AB_CONTROL_V2M99_256x100_20260731 \
> '${OUT_DIR}/control/train.log' 2>&1"

# Avoid simultaneous cold-start conversion of shared simulation assets.
sleep 20

tmux new-window -d -t "${SESSION}:" -n FQFace100 \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongClosedLoopV2SafeFaceQualityCycleFH \
${COMMON[*]} \
device=cuda:1 \
run_name=CLV2_FACE_AB_QUALITY_V2M99_256x100_20260731 \
> '${OUT_DIR}/face_quality/train.log' 2>&1"

printf 'started exact safe-face-quality A/B:\n'
printf '  %s:FQCtrl100 (GPU0, legacy SafeQuality control)\n' "${SESSION}"
printf '  %s:FQFace100 (GPU1, face-quality outcome settlement)\n' "${SESSION}"
