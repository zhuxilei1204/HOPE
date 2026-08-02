#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-validation}"
SESSION="${TMUX_SESSION:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong/2026-07-28_15-58-50_M22750_F2_feedback122_newcolsfree_common3000_stableppo_20260728/model_22750.pt"

case "${MODE}" in
  validation)
    ITERATIONS=1500
    PREFIX="V3"
    OUT_DIR="${REPO}/analysis/capability_guard_v3_20260729/validation"
    ;;
  long)
    ITERATIONS=3000
    PREFIX="L3"
    OUT_DIR="${REPO}/analysis/capability_guard_v3_20260729/long"
    ;;
  *)
    echo "usage: $0 [validation|long]" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"

for window in "${PREFIX}CapGuard" "${PREFIX}CapHit"; do
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

tmux new-window -d -t "${SESSION}:" -n "${PREFIX}CapGuard" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongCapabilityGuard122V3 \
algo=ppo_protected22750_v2 \
${COMMON[*]} \
device=cuda:0 \
run_name=CapabilityGuard122V3_${MODE}_${ITERATIONS}_20260729 \
> '${OUT_DIR}/capability_guard_console.log' 2>&1"

sleep 15

tmux new-window -d -t "${SESSION}:" -n "${PREFIX}CapHit" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongCapabilityGuardHit122V3 \
algo=ppo_protected22750_v2 \
${COMMON[*]} \
device=cuda:1 \
run_name=CapabilityGuardHit122V3_${MODE}_${ITERATIONS}_20260729 \
> '${OUT_DIR}/capability_guard_hit_console.log' 2>&1"

printf 'started capability-guard %s windows:\n' "${MODE}"
printf '  %s:%sCapGuard (GPU0)\n' "${SESSION}" "${PREFIX}"
printf '  %s:%sCapHit (GPU1)\n' "${SESSION}" "${PREFIX}"
