#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong/2026-07-29_12-16-40_CapabilityGuard122V3_gate23000_plus500_20260729/model_23499.pt"
OUT_DIR="${REPO}/analysis/guard23499_one_sided_hit_20260729/train1500"
ITERATIONS="${ITERATIONS:-1500}"

mkdir -p "${OUT_DIR}"

for window in G23499SHi G23499CHi; do
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
  checkpoint_load_optimizer=true
  actor_anchor_coefficient=10.0
  "actor_anchor_checkpoint_path=${CHECKPOINT}"
  actor_anchor_first_layer_input_exempt_start=114
  actor_anchor_exempt_coefficient=1.0
)

tmux new-window -d -t "${SESSION}:" -n G23499SHi \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongGuard23499OneSidedStrikeHitV1 \
algo=ppo_protected22750_v2 \
${COMMON[*]} \
device=cuda:0 \
run_name=Guard23499_OneSidedStrikeHitV1_1500_20260729 \
> '${OUT_DIR}/strike_hit_console.log' 2>&1"

# Isaac asset conversion uses a shared temporary cache. Stagger cold starts.
sleep 20

tmux new-window -d -t "${SESSION}:" -n G23499CHi \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongGuard23499OneSidedClosedHitV1 \
algo=ppo_protected22750_v2 \
${COMMON[*]} \
device=cuda:1 \
run_name=Guard23499_OneSidedClosedHitV1_1500_20260729 \
> '${OUT_DIR}/closed_hit_console.log' 2>&1"

printf 'started model_23499 one-sided strike/hit experiments:\n'
printf '  %s:G23499SHi (GPU0, strike posture + healthy hit)\n' "${SESSION}"
printf '  %s:G23499CHi (GPU1, strike/recovery posture + healthy hit)\n' "${SESSION}"

