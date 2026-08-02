#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
ANALYSIS_ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/analysis/closed_loop_v3_phase_c7_20260801"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
TAG="${TAG:-20260801_SCREEN1500_R1}"
WINDOW="${COMPARE_WINDOW:-50}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MILESTONES="${MILESTONES:-600 1000 1500}"

A_RUN="CLV3_C7A_PHASE_IMMEDIATE_MULTI114_4096x1500_SEED742_${TAG}"
B_RUN="CLV3_C7B_PHASE_ESCROW_MULTI114_4096x1500_SEED742_${TAG}"
A_LOG="${ANALYSIS_ROOT}/${A_RUN}/train.log"
B_LOG="${ANALYSIS_ROOT}/${B_RUN}/train.log"
STATUS_LOG="${ANALYSIS_ROOT}/monitor_${TAG}.log"

latest_iteration() {
  local log="$1"
  local line
  line="$(grep -a 'Learning iteration' "${log}" 2>/dev/null | tail -n 1 || true)"
  if [[ "${line}" =~ iteration[[:space:]]+([0-9]+)/([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '%s\n' "-1"
  fi
}

mkdir -p "${ANALYSIS_ROOT}"
printf '[%s] monitor started: tag=%s milestones=%s\n' \
  "$(date --iso-8601=seconds)" "${TAG}" "${MILESTONES}" >> "${STATUS_LOG}"

for milestone in ${MILESTONES}; do
  while true; do
    a_iter="$(latest_iteration "${A_LOG}")"
    b_iter="$(latest_iteration "${B_LOG}")"
    if (( a_iter >= milestone && b_iter >= milestone )); then
      break
    fi

    if grep -aEq 'Traceback|CUDA out of memory|FloatingPointError' "${A_LOG}" "${B_LOG}" 2>/dev/null; then
      printf '[%s] runtime error detected before milestone %s (A=%s B=%s)\n' \
        "$(date --iso-8601=seconds)" "${milestone}" "${a_iter}" "${b_iter}" >> "${STATUS_LOG}"
      exit 1
    fi
    sleep "${POLL_SECONDS}"
  done

  out="${ANALYSIS_ROOT}/MILESTONE_${milestone}_${TAG}.md"
  "${PY}" "${ROOT}/scripts/compare_training_logs.py" \
    "${A_LOG}" "${B_LOG}" \
    --window "${WINDOW}" \
    --labels C7A_R1 C7B_R1 \
    --markdown-out "${out}" > /dev/null
  printf '[%s] captured milestone %s: A=%s B=%s output=%s\n' \
    "$(date --iso-8601=seconds)" "${milestone}" "${a_iter}" "${b_iter}" "${out}" >> "${STATUS_LOG}"
done

printf '[%s] monitor complete\n' "$(date --iso-8601=seconds)" >> "${STATUS_LOG}"
