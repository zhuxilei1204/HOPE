#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID="${1:?usage: wait_and_eval_stage2_precision.sh TRAIN_PID GPU TASK TAG RUN_DIR}"
GPU="${2:?usage: wait_and_eval_stage2_precision.sh TRAIN_PID GPU TASK TAG RUN_DIR}"
TASK="${3:?usage: wait_and_eval_stage2_precision.sh TRAIN_PID GPU TASK TAG RUN_DIR}"
TAG="${4:?usage: wait_and_eval_stage2_precision.sh TRAIN_PID GPU TASK TAG RUN_DIR}"
RUN_DIR="${5:?usage: wait_and_eval_stage2_precision.sh TRAIN_PID GPU TASK TAG RUN_DIR}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 15
done

for checkpoint_index in 200 299; do
  checkpoint="${RUN_DIR}/model_${checkpoint_index}.pt"
  for _ in $(seq 1 40); do
    [[ -s "${checkpoint}" ]] && break
    sleep 3
  done
  if [[ ! -s "${checkpoint}" ]]; then
    echo "missing checkpoint after training exit: ${checkpoint}" >&2
    exit 3
  fi

  "${ROOT}/scripts/eval_stage2_closed_loop_command_member.sh" \
    "${GPU}" "${TASK}" "${TAG}_model${checkpoint_index}_core" \
    "${checkpoint}" 900 8830 0.0 0.0
  "${ROOT}/scripts/eval_stage2_closed_loop_command_member.sh" \
    "${GPU}" "${TASK}" "${TAG}_model${checkpoint_index}_full" \
    "${checkpoint}" 900 8830 1.0 1.0
done

echo "completed fixed Stage-2 evaluations for ${TAG}"
