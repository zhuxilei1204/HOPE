#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
CHECKPOINT="/mnt/ssd/zxl/HOPE_deploy_models_20260730/03_hope_Bdeploy_model19495_rollback/checkpoint/model_19495.pt"
OUT_DIR="${REPO}/analysis/b19495_action_feedback_migration_ab_20260731"
ITERATIONS="${ITERATIONS:-500}"
NUM_ENVS="${NUM_ENVS:-4096}"
LR="${LR:-2.0e-5}"
ANCHOR="${ANCHOR:-2.0}"
RAW_WINDOW="B19495FbRaw"
EFFECTIVE_WINDOW="B19495FbEff"

mkdir -p "${OUT_DIR}/raw_control" "${OUT_DIR}/effective_migration"

for window in "${RAW_WINDOW}" "${EFFECTIVE_WINDOW}"; do
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
  algo.runner.save_interval=100
)

cat > "${OUT_DIR}/experiment_contract.txt" <<EOF
source_checkpoint=${CHECKPOINT}
source_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
motion_manifest=/mnt/ssd/zxl/HOPE_deploy_models_20260730/03_hope_Bdeploy_model19495_rollback/motions/manifest.tsv
num_envs=${NUM_ENVS}
iterations=${ITERATIONS}
learning_rate_after_load=${LR}
actor_anchor_coefficient=${ANCHOR}
shared_variables=checkpoint,optimizer,seed,motions,rewards,lifecycle,ppo,learning_rate,anchor
isolated_variable=actions.joint_pos.feedback_mode(raw/effective)
provisional_operational_limits_enabled=false
EOF

tmux new-window -d -t "${SESSION}:" -n "${RAW_WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongB19495ActionFeedbackRawControlV1 \
${COMMON[*]} \
device=cuda:0 \
run_name=B19495_ACTION_FEEDBACK_RAW_CONTROL_4096x${ITERATIONS}_20260731 \
> '${OUT_DIR}/raw_control/train.log' 2>&1"

sleep 20

tmux new-window -d -t "${SESSION}:" -n "${EFFECTIVE_WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongB19495ActionFeedbackEffectiveMigrationV1 \
${COMMON[*]} \
device=cuda:1 \
run_name=B19495_ACTION_FEEDBACK_EFFECTIVE_MIGRATION_4096x${ITERATIONS}_20260731 \
> '${OUT_DIR}/effective_migration/train.log' 2>&1"

printf 'started B19495 four-motion action-feedback A/B:\n'
printf '  %s:%s (GPU0, legacy raw control)\n' "${SESSION}" "${RAW_WINDOW}"
printf '  %s:%s (GPU1, effective-feedback migration)\n' "${SESSION}" "${EFFECTIVE_WINDOW}"
printf '  checkpoint: %s\n' "${CHECKPOINT}"
printf '  rollout: %s envs x %s iterations per branch\n' "${NUM_ENVS}" "${ITERATIONS}"
printf '  logs: %s\n' "${OUT_DIR}"

