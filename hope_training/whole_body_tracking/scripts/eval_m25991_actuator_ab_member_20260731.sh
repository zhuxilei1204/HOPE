#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 LABEL DEVICE CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

LABEL=$1
DEVICE=$2
CHECKPOINT=$3
OUTPUT_DIR=$4

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
WORK="${WORK:-${REPO}/hope_training/whole_body_tracking}"
TASK="${TASK:-HOPEPingPongM25991ReplayControlV1}"
MANIFEST="${MANIFEST:-${REPO}/hope_training/motions/model19495_forehand_active_ready_20260729/manifest.tsv}"
SNAPSHOT="${SNAPSHOT:-${REPO}/analysis/external_alignment_packages_20260731/package1/real_robot_planner_rl_alignment_20260731/01_planner_V4_A4_plane_recorded/snapshot}"
PLANNER="${PLANNER:-${SNAPSHOT}/hope_planner.yaml}"
PLANNER_CODE="${PLANNER_CODE:-${SNAPSHOT}/hope_planner_code}"
NUM_SERVES="${NUM_SERVES:-40}"
ISAAC_SERVES="${ISAAC_SERVES:-10}"
CONDA_SH="${CONDA_SH:-/home/zxl/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-zxl-pace}"

for path in "${CHECKPOINT}" "${MANIFEST}" "${PLANNER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required file: ${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${PLANNER_CODE}" ]]; then
  echo "missing planner code snapshot: ${PLANNER_CODE}" >&2
  exit 2
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${WORK}"
source setup_train_env.sh

mkdir -p "${OUTPUT_DIR}"/{exported,isaac,mujoco,planner_alignment,action_feasibility}

{
  printf 'label=%s\n' "${LABEL}"
  printf 'checkpoint=%s\n' "${CHECKPOINT}"
  printf 'checkpoint_sha256=%s\n' "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
  printf 'task=%s\n' "${TASK}"
  printf 'manifest=%s\n' "${MANIFEST}"
  printf 'planner=%s\n' "${PLANNER}"
  printf 'planner_code=%s\n' "${PLANNER_CODE}"
  printf 'seed=17\nisaac_serves=%s\nmujoco_serves=%s\n' \
    "${ISAAC_SERVES}" "${NUM_SERVES}"
  printf 'side_mode=forehand\nstation_mode=dynamic-from-manifest\n'
  printf 'dynamic_station_clip_x=0.0,0.0\n'
  printf 'dynamic_station_clip_y=-0.10,0.10\n'
  printf 'incoming_trajectory=one-bounce\nplanner_x_hit=0.20\n'
} > "${OUTPUT_DIR}/eval_contract.txt"

isaac_py scripts/export_onnx.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}/exported" \
  --device "${DEVICE}" \
  --task-yaml "${TASK}" \
  --motion-manifest "${MANIFEST}" \
  --actor-obs-contract auto \
  > "${OUTPUT_DIR}/export.log" 2>&1

isaac_py scripts/isaac_physical_eval.py \
  --checkpoint "${CHECKPOINT}" \
  --task-yaml "${TASK}" \
  --motion-manifest "${MANIFEST}" \
  --planner-yaml "${PLANNER}" \
  --planner-code-dir "${PLANNER_CODE}" \
  --device "${DEVICE}" \
  --seed 17 \
  --num-serves "${ISAAC_SERVES}" \
  --side-mode forehand \
  --max-trial-seconds 2.6 \
  --min-rest-seconds 1.0 \
  --max-rest-seconds 2.2 \
  --planner-x-hit 0.20 \
  --trace-csv "${OUTPUT_DIR}/isaac/trace.csv" \
  --trials-csv "${OUTPUT_DIR}/isaac/trials.csv" \
  --json-out "${OUTPUT_DIR}/isaac/result.json" \
  > "${OUTPUT_DIR}/isaac/eval.log" 2>&1

jq -e '.attempts > 0' "${OUTPUT_DIR}/isaac/result.json" >/dev/null

MUJOCO_GL=egl python scripts/mujoco_eval_onnx.py \
  --onnx "${OUTPUT_DIR}/exported/hope_pingpong.onnx" \
  --serve-manifest "${MANIFEST}" \
  --station-manifest "${MANIFEST}" \
  --num-serves "${NUM_SERVES}" \
  --side-mode forehand \
  --incoming-trajectory one-bounce \
  --planner-mode real-hope-planner \
  --real-planner-yaml "${PLANNER}" \
  --real-planner-code-dir "${PLANNER_CODE}" \
  --real-planner-x-hit 0.20 \
  --real-planner-no-command-fallback none \
  --eval-mode continuous \
  --seed 17 \
  --max-trial-seconds 2.6 \
  --min-rest-seconds 1.0 \
  --max-rest-seconds 2.2 \
  --station-mode dynamic-from-manifest \
  --dynamic-station-clip-x 0.0 0.0 \
  --dynamic-station-clip-y -0.10 0.10 \
  --dynamic-station-blend 1.0 \
  --dynamic-station-post-window 0.12 \
  --detailed \
  --trace-csv "${OUTPUT_DIR}/mujoco/trace.csv" \
  --trace-serves "${NUM_SERVES}" \
  --joint-action-diag-csv "${OUTPUT_DIR}/mujoco/joint_action.csv" \
  --contact-diag-csv "${OUTPUT_DIR}/mujoco/contact.csv" \
  --json-out "${OUTPUT_DIR}/mujoco/result.json" \
  > "${OUTPUT_DIR}/mujoco/eval.log" 2>&1

jq -e '.counts.attempts > 0' "${OUTPUT_DIR}/mujoco/result.json" >/dev/null

python scripts/summarize_planner_policy_alignment.py \
  "${OUTPUT_DIR}/mujoco/contact.csv" \
  --json-out "${OUTPUT_DIR}/planner_alignment/alignment.json" \
  --markdown-out "${OUTPUT_DIR}/planner_alignment/alignment.md" \
  > "${OUTPUT_DIR}/planner_alignment/alignment.log" 2>&1

python scripts/analyze_action_feasibility.py \
  "${OUTPUT_DIR}/mujoco/joint_action.csv" \
  --scenario full-task \
  --json-out "${OUTPUT_DIR}/action_feasibility/summary.json" \
  --markdown-out "${OUTPUT_DIR}/action_feasibility/summary.md" \
  > "${OUTPUT_DIR}/action_feasibility/analyze.log" 2>&1

sha256sum \
  "${CHECKPOINT}" \
  "${OUTPUT_DIR}/exported/hope_pingpong.onnx" \
  "${OUTPUT_DIR}/exported/hope_pingpong.onnx.data" \
  > "${OUTPUT_DIR}/SHA256SUMS"
