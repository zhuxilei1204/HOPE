#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
WINDOW="${TMUX_WINDOW:-clv3ScratchSmoke}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
OUT_DIR="${REPO}/analysis/closed_loop_v3_scratch_20260731/smoke_discovery_v2_256x300"

if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${WINDOW}"; then
  echo "tmux window already exists: ${SESSION}:${WINDOW}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cat > "${OUT_DIR}/experiment_contract.txt" <<EOF
initialization=fresh_actor_critic_optimizer
checkpoint_path=null
task=HOPEPingPongClosedLoopV3ScratchMultiSkill114
algo=ppo_closed_loop_v3_scratch
motion_manifest=${REPO}/hope_training/motions/supplemental_lower_body_contact_aware_isaacfk_20260728/manifest_train_ready_with_task_boxes.tsv
num_envs=256
iterations=300
seed=731
device=cuda:0
promotion_gate=contact_discovery,ppo_health,fall_overflow_trend
EOF

tmux new-window -d -t "${SESSION}:" -n "${WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongClosedLoopV3ScratchMultiSkill114 \
algo=ppo_closed_loop_v3_scratch \
device=cuda:0 \
num_envs=256 \
max_iterations=300 \
seed=731 \
checkpoint_path=null \
checkpoint_actor_only=false \
checkpoint_load_optimizer=false \
run_name=CLV3_SCRATCH_DISCOVERY_V2_SMOKE_256x300_20260731 \
headless=true > '${OUT_DIR}/train.log' 2>&1"

printf 'started %s:%s\n' "${SESSION}" "${WINDOW}"
printf 'log: %s\n' "${OUT_DIR}/train.log"
