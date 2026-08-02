#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
CHECKPOINT="/mnt/ssd/zxl/HOPE_deploy_models_20260730/03_hope_Bdeploy_model19495_rollback/checkpoint/model_19495.pt"
OUT_DIR="${REPO}/analysis/b19495_effective_feedback_adapter_ab_20260731"
ITERATIONS="${ITERATIONS:-300}"
NUM_ENVS="${NUM_ENVS:-4096}"
LR="${LR:-2.0e-5}"
ANCHOR="${ANCHOR:-2.0}"
FREE_EXEMPT_COEFFICIENT="${FREE_EXEMPT_COEFFICIENT:-0.0}"
WEAK_EXEMPT_COEFFICIENT="${WEAK_EXEMPT_COEFFICIENT:-0.1}"
FREE_WINDOW="B19495FbAdaptFree"
WEAK_WINDOW="B19495FbAdaptWeak"

mkdir -p "${OUT_DIR}/free" "${OUT_DIR}/weak"

for window in "${FREE_WINDOW}" "${WEAK_WINDOW}"; do
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
  task=HOPEPingPongB19495ActionFeedbackEffectiveMigrationV1
  algo=ppo
  headless=true
  "num_envs=${NUM_ENVS}"
  "max_iterations=${ITERATIONS}"
  seed=0
  "checkpoint_path=${CHECKPOINT}"
  checkpoint_actor_only=false
  checkpoint_load_optimizer=true
  "optimizer_learning_rate_after_load=${LR}"
  "actor_anchor_coefficient=${ANCHOR}"
  "actor_anchor_checkpoint_path=${CHECKPOINT}"
  actor_anchor_first_layer_input_exempt_start=65
  actor_anchor_first_layer_input_exempt_end=96
  algo.runner.save_interval=100
)

cat > "${OUT_DIR}/experiment_contract.txt" <<EOF
source_checkpoint=${CHECKPOINT}
source_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
motion_manifest=/mnt/ssd/zxl/HOPE_deploy_models_20260730/03_hope_Bdeploy_model19495_rollback/motions/manifest.tsv
task=HOPEPingPongB19495ActionFeedbackEffectiveMigrationV1
last_action_feedback=effective
last_action_observation_slice=[65,96)
num_envs=${NUM_ENVS}
iterations=${ITERATIONS}
learning_rate_after_load=${LR}
actor_anchor_coefficient=${ANCHOR}
free_exempt_coefficient=${FREE_EXEMPT_COEFFICIENT}
weak_exempt_coefficient=${WEAK_EXEMPT_COEFFICIENT}
shared_variables=checkpoint,optimizer,seed,motions,rewards,lifecycle,ppo,learning_rate,feedback_mode,anchor
isolated_variable=actor_first_layer_last_action_anchor_coefficient
provisional_operational_limits_enabled=false
EOF

tmux new-window -d -t "${SESSION}:" -n "${FREE_WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
${COMMON[*]} \
actor_anchor_exempt_coefficient=${FREE_EXEMPT_COEFFICIENT} \
device=cuda:0 \
run_name=B19495_EFFECTIVE_FEEDBACK_LAST_ACTION_FREE_4096x${ITERATIONS}_20260731 \
> '${OUT_DIR}/free/train.log' 2>&1"

sleep 20

tmux new-window -d -t "${SESSION}:" -n "${WEAK_WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
${COMMON[*]} \
actor_anchor_exempt_coefficient=${WEAK_EXEMPT_COEFFICIENT} \
device=cuda:1 \
run_name=B19495_EFFECTIVE_FEEDBACK_LAST_ACTION_WEAK_4096x${ITERATIONS}_20260731 \
> '${OUT_DIR}/weak/train.log' 2>&1"

printf 'started B19495 effective-feedback last-action adapter A/B:\n'
printf '  %s:%s (GPU0, [65,96) anchor=%s)\n' "${SESSION}" "${FREE_WINDOW}" "${FREE_EXEMPT_COEFFICIENT}"
printf '  %s:%s (GPU1, [65,96) anchor=%s)\n' "${SESSION}" "${WEAK_WINDOW}" "${WEAK_EXEMPT_COEFFICIENT}"
printf '  checkpoint: %s\n' "${CHECKPOINT}"
printf '  rollout: %s envs x %s iterations per branch\n' "${NUM_ENVS}" "${ITERATIONS}"
printf '  logs: %s\n' "${OUT_DIR}"
