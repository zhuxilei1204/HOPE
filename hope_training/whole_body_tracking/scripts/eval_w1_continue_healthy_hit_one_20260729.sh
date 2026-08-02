#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 LABEL DEVICE TASK CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

LABEL=$1
DEVICE=$2
TASK=$3
CHECKPOINT=$4
OUTPUT_DIR=$5

ROOT=/mnt/ssd/zxl/HOPE_latest_20260721
WORK=$ROOT/hope_training/whole_body_tracking
MANIFEST=$ROOT/hope_training/motions/user_four_motion_manual_hits_a3fk_yaw_leg_stabilized_20260724/manifest.tsv
PLANNER=$ROOT/hope_ws/src/hope_planner/config/hope_planner.yaml

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "$WORK"
source setup_train_env.sh

mkdir -p "$OUTPUT_DIR"/{exported,isaac,mujoco,planner_alignment}

isaac_py scripts/export_onnx.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/exported" \
  --device "$DEVICE" \
  --task-yaml "$TASK" \
  --actor-obs-contract auto \
  > "$OUTPUT_DIR/export.log" 2>&1

isaac_py scripts/isaac_physical_eval.py \
  --checkpoint "$CHECKPOINT" \
  --task-yaml "$TASK" \
  --motion-manifest "$MANIFEST" \
  --planner-yaml "$PLANNER" \
  --device "$DEVICE" \
  --seed 17 \
  --num-serves 40 \
  --max-trial-seconds 2.6 \
  --min-rest-seconds 1.0 \
  --max-rest-seconds 2.2 \
  --planner-x-hit 0.20 \
  --video "$OUTPUT_DIR/isaac/${LABEL}_isaac_continuous40.mp4" \
  --trace-csv "$OUTPUT_DIR/isaac/trace.csv" \
  --trials-csv "$OUTPUT_DIR/isaac/trials.csv" \
  --json-out "$OUTPUT_DIR/isaac/result.json" \
  > "$OUTPUT_DIR/isaac/eval.log" 2>&1

MUJOCO_GL=egl python scripts/mujoco_eval_onnx.py \
  --onnx "$OUTPUT_DIR/exported/hope_pingpong.onnx" \
  --serve-manifest "$MANIFEST" \
  --num-serves 40 \
  --side-mode mixed \
  --incoming-trajectory one-bounce \
  --planner-mode real-hope-planner \
  --real-planner-x-hit 0.20 \
  --real-planner-no-command-fallback none \
  --eval-mode continuous \
  --seed 17 \
  --max-trial-seconds 2.6 \
  --min-rest-seconds 1.0 \
  --max-rest-seconds 2.2 \
  --detailed \
  --record-video "$OUTPUT_DIR/mujoco/${LABEL}_mujoco_continuous40.mp4" \
  --record-serves 12 \
  --trace-csv "$OUTPUT_DIR/mujoco/trace.csv" \
  --trace-serves 40 \
  --contact-diag-csv "$OUTPUT_DIR/mujoco/contact.csv" \
  --json-out "$OUTPUT_DIR/mujoco/result.json" \
  --video-width 640 \
  --video-height 360 \
  > "$OUTPUT_DIR/mujoco/eval.log" 2>&1

python scripts/summarize_planner_policy_alignment.py \
  "$OUTPUT_DIR/mujoco/contact.csv" \
  --json-out "$OUTPUT_DIR/planner_alignment/alignment.json" \
  --markdown-out "$OUTPUT_DIR/planner_alignment/alignment.md" \
  > "$OUTPUT_DIR/planner_alignment/alignment.log" 2>&1

printf '%s\n' "$CHECKPOINT" > "$OUTPUT_DIR/checkpoint.txt"
printf '%s\n' "$TASK" > "$OUTPUT_DIR/task.txt"
