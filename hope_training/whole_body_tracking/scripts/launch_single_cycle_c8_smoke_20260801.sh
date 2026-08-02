#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
WINDOW="${WINDOW:-clv3C8SingleSmoke}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
SEED="${SEED:-743}"
TAG="${TAG:-20260801_SMOKE}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
TASK="HOPEPingPongClosedLoopV3ScratchSingleCycleC8MultiSkill114"
RUN_NAME="CLV3_C8_SINGLE_CYCLE_${NUM_ENVS}x${MAX_ITERATIONS}_SEED${SEED}_${TAG}"
OUT_DIR="${REPO}/analysis/closed_loop_v3_single_cycle_c8_20260801/${RUN_NAME}"

if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${WINDOW}"; then
  echo "tmux window already exists: ${SESSION}:${WINDOW}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
"${PY}" -m pytest -q \
  "${ROOT}/tests/test_single_cycle_curriculum.py" \
  "${ROOT}/tests/test_single_cycle_c8_contract.py" \
  > "${OUT_DIR}/preflight_tests.txt"

cat > "${OUT_DIR}/experiment_contract.txt" <<EOF
initialization=fresh_actor_critic_optimizer
task=${TASK}
purpose=state_machine_and_accounting_smoke_not_model_selection
single_cycle_probability=ability_driven_from_continuous_pool
minimum_continuous_fraction=0.40
single_cycle_start=audited_prestrike
single_cycle_end=settled_no_command_ready_clean_timeout
escrow_order=reward_settlement_before_timeout_reset
batched_metric_reset_logging=enabled_after_exact_same_seed_validation
reward_pruning=disabled
num_envs=${NUM_ENVS}
iterations=${MAX_ITERATIONS}
seed=${SEED}
device=${DEVICE}
EOF

sha256sum \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/single_cycle_curriculum.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py" \
  "${ROOT}/cfg/task/HOPEPingPongClosedLoopV3ScratchSingleCycleC8MultiSkill114.yaml" \
  > "${OUT_DIR}/source_sha256.txt"

tmux new-window -d -t "${SESSION}:" -n "${WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=${TASK} \
algo=ppo_closed_loop_v3_scratch \
device=${DEVICE} \
num_envs=${NUM_ENVS} \
max_iterations=${MAX_ITERATIONS} \
seed=${SEED} \
checkpoint_path=null \
checkpoint_actor_only=false \
checkpoint_load_optimizer=false \
run_name=${RUN_NAME} \
headless=true > '${OUT_DIR}/train.log' 2>&1"

printf 'started %s:%s on %s\n' "${SESSION}" "${WINDOW}" "${DEVICE}"
printf 'log: %s\n' "${OUT_DIR}/train.log"
