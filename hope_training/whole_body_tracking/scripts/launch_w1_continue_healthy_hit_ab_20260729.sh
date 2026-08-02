#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
W1_CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong/2026-07-29_15-38-49_Guard23499_WristPrepTimingV1_1500_20260729/model_24998.pt"
ANCHOR_CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong/2026-07-29_12-16-40_CapabilityGuard122V3_gate23000_plus500_20260729/model_23499.pt"
OUT_DIR="${REPO}/analysis/w1_continue_healthy_hit_ab_20260729/train1500"
ITERATIONS="${ITERATIONS:-1500}"

mkdir -p "${OUT_DIR}"

for checkpoint in "${W1_CHECKPOINT}" "${ANCHOR_CHECKPOINT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

for window in W1C1Cont W1H2Health; do
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
  "checkpoint_path=${W1_CHECKPOINT}"
  checkpoint_actor_only=false
  checkpoint_load_optimizer=true
  actor_anchor_coefficient=10.0
  "actor_anchor_checkpoint_path=${ANCHOR_CHECKPOINT}"
  actor_anchor_first_layer_input_exempt_start=114
  actor_anchor_exempt_coefficient=1.0
)

tmux new-window -d -t "${SESSION}:" -n W1C1Cont \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongGuard23499WristPrepTimingV1 \
algo=ppo_protected22750_v2 \
${COMMON[*]} \
device=cuda:0 \
run_name=W1TimingContinue_from24998_1500_20260729 \
> '${OUT_DIR}/c1_w1_continue_console.log' 2>&1"

# Isaac asset conversion uses a shared temporary cache. Stagger cold starts.
sleep 20

tmux new-window -d -t "${SESSION}:" -n W1H2Health \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongW1HealthyHitBackleanV2 \
algo=ppo_protected22750_v2 \
${COMMON[*]} \
device=cuda:1 \
run_name=W1HealthyHitBackleanV2_from24998_1500_20260729 \
> '${OUT_DIR}/h2_healthy_hit_console.log' 2>&1"

printf 'started W1 continuation/healthy-hit experiments:\n'
printf '  %s:W1C1Cont (GPU0, exact W1 continuation)\n' "${SESSION}"
printf '  %s:W1H2Health (GPU1, healthy-hit/backlean V2)\n' "${SESSION}"
printf '  logs: %s\n' "${OUT_DIR}"
