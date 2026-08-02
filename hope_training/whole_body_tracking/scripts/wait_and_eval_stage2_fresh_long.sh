#!/usr/bin/env bash
set -euo pipefail

PID_S1="${1:?usage: wait_and_eval_stage2_fresh_long.sh PID_S1 PID_S2 RUN_DIR_S1 RUN_DIR_S2 FINAL_MODEL}"
PID_S2="${2:?usage: wait_and_eval_stage2_fresh_long.sh PID_S1 PID_S2 RUN_DIR_S1 RUN_DIR_S2 FINAL_MODEL}"
RUN_DIR_S1="${3:?usage: wait_and_eval_stage2_fresh_long.sh PID_S1 PID_S2 RUN_DIR_S1 RUN_DIR_S2 FINAL_MODEL}"
RUN_DIR_S2="${4:?usage: wait_and_eval_stage2_fresh_long.sh PID_S1 PID_S2 RUN_DIR_S1 RUN_DIR_S2 FINAL_MODEL}"
FINAL_MODEL="${5:?usage: wait_and_eval_stage2_fresh_long.sh PID_S1 PID_S2 RUN_DIR_S1 RUN_DIR_S2 FINAL_MODEL}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
TASK="HOPEPingPongStage2CommandPrecisionOutcome114V7B"

while kill -0 "${PID_S1}" 2>/dev/null || kill -0 "${PID_S2}" 2>/dev/null; do
  sleep 30
done

for member in s1 s2; do
  if [[ "${member}" == "s1" ]]; then
    run_dir="${RUN_DIR_S1}"
  else
    run_dir="${RUN_DIR_S2}"
  fi
  checkpoint="${run_dir}/${FINAL_MODEL}"
  for _ in $(seq 1 40); do
    [[ -s "${checkpoint}" ]] && break
    sleep 3
  done
  if [[ ! -s "${checkpoint}" ]]; then
    echo "missing final checkpoint: ${checkpoint}" >&2
    exit 3
  fi

  "${ROOT}/scripts/eval_stage2_closed_loop_command_member.sh" \
    0 "${TASK}" "v7b_fresh_long_${member}_${FINAL_MODEL%.pt}_core" \
    "${checkpoint}" 900 8830 0.0 0.0
  "${ROOT}/scripts/eval_stage2_closed_loop_command_member.sh" \
    0 "${TASK}" "v7b_fresh_long_${member}_${FINAL_MODEL%.pt}_full" \
    "${checkpoint}" 900 8830 1.0 1.0
done

echo "completed sequential Stage-2 fresh-optimizer long-run evaluations"
