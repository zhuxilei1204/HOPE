#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
R2C_CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong/2026-07-30_02-52-21_M22494_FR2C_StrikeEventAbilityGuard_4096x1500_20260730/model_23993.pt"
IMPACT_CHECKPOINT="${ROOT}/logs/rsl_rl/hope_pingpong_closed_loop_v2_impact/2026-07-31_04-08-47_CLV2I_R2C23993_FH_1536x300_INIT_20260731/model_125.pt"
OUT_DIR="${REPO}/analysis/closed_loop_v2_20260731/durable_cycle_v1_init_smoke_256x100"
ITERATIONS="${ITERATIONS:-100}"

mkdir -p "${OUT_DIR}/r2c23993" "${OUT_DIR}/impact_v1_model125"

for window in DCR2CSmk DCImpSmk; do
  if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${window}"; then
    echo "tmux window already exists: ${SESSION}:${window}" >&2
    exit 1
  fi
done

COMMON=(
  task=HOPEPingPongClosedLoopV2DurableCycleFH
  algo=ppo_closed_loop_v2_durable
  headless=true
  num_envs=256
  "max_iterations=${ITERATIONS}"
  seed=20260731
  checkpoint_actor_only=true
  checkpoint_load_optimizer=false
  actor_anchor_coefficient=0.0
  algo.runner.save_interval=25
)

tmux new-window -d -t "${SESSION}:" -n DCR2CSmk \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
${COMMON[*]} \
checkpoint_path='${R2C_CHECKPOINT}' \
device=cuda:0 \
run_name=CLV2_DURABLECYCLEV1_R2C23993_256x100_SMOKE_20260731 \
> '${OUT_DIR}/r2c23993/train.log' 2>&1"

# Avoid simultaneous cold-start asset conversion.
sleep 20

tmux new-window -d -t "${SESSION}:" -n DCImpSmk \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
${COMMON[*]} \
checkpoint_path='${IMPACT_CHECKPOINT}' \
device=cuda:1 \
run_name=CLV2_DURABLECYCLEV1_IMPACTV1M125_256x100_SMOKE_20260731 \
> '${OUT_DIR}/impact_v1_model125/train.log' 2>&1"

printf 'started durable-cycle-v1 initialization smoke controls:\n'
printf '  %s:DCR2CSmk (GPU0, R2C23993 actor-only)\n' "${SESSION}"
printf '  %s:DCImpSmk (GPU1, ImpactV1 model_125 actor-only)\n' "${SESSION}"
