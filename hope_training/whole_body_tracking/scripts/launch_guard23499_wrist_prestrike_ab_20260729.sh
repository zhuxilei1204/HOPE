#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong/2026-07-29_12-16-40_CapabilityGuard122V3_gate23000_plus500_20260729/model_23499.pt"
OUT_DIR="${REPO}/analysis/wrist_prestrike_capability_plan_20260729/train1500"
ITERATIONS="${ITERATIONS:-1500}"

mkdir -p "${OUT_DIR}"

for window in WPrepTiming WPrepExplore; do
  if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${window}"; then
    echo "tmux window already exists: ${SESSION}:${window}" >&2
    exit 1
  fi
done

COMMON=(
  headless=true
  num_envs=4096
  "max_iterations=${ITERATIONS}"
  seed=20260729
  "checkpoint_path=${CHECKPOINT}"
  checkpoint_actor_only=false
  checkpoint_load_optimizer=false
  actor_anchor_coefficient=10.0
  "actor_anchor_checkpoint_path=${CHECKPOINT}"
  actor_anchor_first_layer_input_exempt_start=114
  actor_anchor_exempt_coefficient=1.0
)

tmux new-window -d -t "${SESSION}:" -n WPrepTiming \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongGuard23499WristPrepTimingV1 \
algo=ppo_protected22750_v2 \
${COMMON[*]} \
device=cuda:0 \
run_name=Guard23499_WristPrepTimingV1_1500_20260729 \
> '${OUT_DIR}/w1_timing_console.log' 2>&1"

# Isaac asset conversion uses a shared temporary cache. Stagger cold starts.
sleep 20

tmux new-window -d -t "${SESSION}:" -n WPrepExplore \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongGuard23499WristPrepExploreV1 \
algo=ppo_protected22750_v2 \
${COMMON[*]} \
device=cuda:1 \
run_name=Guard23499_WristPrepExploreV1_1500_20260729 \
> '${OUT_DIR}/w2_explore_console.log' 2>&1"

printf 'started model_23499 wrist pre-strike experiments:\n'
printf '  %s:WPrepTiming (GPU0, timing + pre-strike blade alignment)\n' "${SESSION}"
printf '  %s:WPrepExplore (GPU1, W1 + selective wrist exploration)\n' "${SESSION}"
printf '  logs: %s\n' "${OUT_DIR}"
