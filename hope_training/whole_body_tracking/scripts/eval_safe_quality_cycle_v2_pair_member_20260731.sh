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

ROOT=/mnt/ssd/zxl/HOPE_latest_20260721
WORK=$ROOT/hope_training/whole_body_tracking
TASK=HOPEPingPongClosedLoopV2SafeQualityCycleFH
PHYSICAL_TASK=HOPE-PingPong-ClosedLoopV2-PhysicalEval-AgibotA3-v0
EXPERIMENT=hope_pingpong_closed_loop_v2_safe_quality_cycle
MANIFEST=$ROOT/hope_training/motions/user_four_motion_manual_hits_a3fk_yaw_leg_stabilized_20260724/manifest.tsv
PLANNER_SNAPSHOT=$ROOT/analysis/external_alignment_packages_20260731/package1/real_robot_planner_rl_alignment_20260731/01_planner_V4_A4_plane_recorded/snapshot
PLANNER=$PLANNER_SNAPSHOT/hope_planner.yaml
PLANNER_CODE=$PLANNER_SNAPSHOT/hope_planner_code
SEED=${SEED:-17}
NUM_SERVES=${NUM_SERVES:-20}
RECORD_SERVES=${RECORD_SERVES:-$NUM_SERVES}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi

source /home/zxl/miniconda3/etc/profile.d/conda.sh
conda activate zxl-pace
cd "$WORK"
source setup_train_env.sh

mkdir -p "$OUTPUT_DIR"/{exported,isaac,mujoco,planner_alignment}
rm -f "$OUTPUT_DIR/isaac/result.json" "$OUTPUT_DIR/mujoco/result.json"

{
  printf 'label=%s\n' "$LABEL"
  printf 'checkpoint=%s\n' "$CHECKPOINT"
  printf 'checkpoint_sha256=%s\n' "$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
  printf 'task=%s\n' "$TASK"
  printf 'physical_task=%s\n' "$PHYSICAL_TASK"
  printf 'experiment=%s\n' "$EXPERIMENT"
  printf 'manifest=%s\n' "$MANIFEST"
  printf 'manifest_sha256=%s\n' "$(sha256sum "$MANIFEST" | awk '{print $1}')"
  printf 'planner=%s\n' "$PLANNER"
  printf 'planner_sha256=%s\n' "$(sha256sum "$PLANNER" | awk '{print $1}')"
  printf 'planner_code=%s\n' "$PLANNER_CODE"
  printf 'side_mode=forehand\n'
  printf 'station_mode=fixed\n'
  printf 'incoming_trajectory=one-bounce\n'
  printf 'planner_mode=real-hope-planner\n'
  printf 'planner_x_hit=0.20\n'
  printf 'seed=%s\n' "$SEED"
  printf 'num_serves=%s\n' "$NUM_SERVES"
  printf 'min_rest_seconds=1.0\n'
  printf 'max_rest_seconds=2.2\n'
} > "$OUTPUT_DIR/eval_contract.txt"

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
  --physical-task "$PHYSICAL_TASK" \
  --experiment-name "$EXPERIMENT" \
  --motion-manifest "$MANIFEST" \
  --planner-yaml "$PLANNER" \
  --planner-code-dir "$PLANNER_CODE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --num-serves "$NUM_SERVES" \
  --side-mode forehand \
  --max-trial-seconds 2.6 \
  --min-rest-seconds 1.0 \
  --max-rest-seconds 2.2 \
  --planner-x-hit 0.20 \
  --video "$OUTPUT_DIR/isaac/${LABEL}_isaac_physical_forehand${NUM_SERVES}.mp4" \
  --trace-csv "$OUTPUT_DIR/isaac/trace.csv" \
  --trials-csv "$OUTPUT_DIR/isaac/trials.csv" \
  --json-out "$OUTPUT_DIR/isaac/result.json" \
  > "$OUTPUT_DIR/isaac/eval.log" 2>&1

python - "$OUTPUT_DIR/isaac/result.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Isaac physical eval did not produce {path}")
result = json.loads(path.read_text(encoding="utf-8"))
if int(result.get("attempts", 0)) <= 0:
    raise SystemExit(f"Isaac physical eval result has no attempts: {path}")
PY

MUJOCO_GL=egl python scripts/mujoco_eval_onnx.py \
  --onnx "$OUTPUT_DIR/exported/hope_pingpong.onnx" \
  --serve-manifest "$MANIFEST" \
  --num-serves "$NUM_SERVES" \
  --side-mode forehand \
  --station-mode fixed \
  --incoming-trajectory one-bounce \
  --planner-mode real-hope-planner \
  --real-planner-yaml "$PLANNER" \
  --real-planner-code-dir "$PLANNER_CODE" \
  --real-planner-x-hit 0.20 \
  --real-planner-no-command-fallback none \
  --eval-mode continuous \
  --seed "$SEED" \
  --max-trial-seconds 2.6 \
  --min-rest-seconds 1.0 \
  --max-rest-seconds 2.2 \
  --detailed \
  --record-video "$OUTPUT_DIR/mujoco/${LABEL}_mujoco_real_planner_forehand${NUM_SERVES}.mp4" \
  --record-serves "$RECORD_SERVES" \
  --trace-csv "$OUTPUT_DIR/mujoco/trace.csv" \
  --trace-serves "$NUM_SERVES" \
  --joint-action-diag-csv "$OUTPUT_DIR/mujoco/joint_action.csv" \
  --contact-diag-csv "$OUTPUT_DIR/mujoco/contact.csv" \
  --json-out "$OUTPUT_DIR/mujoco/result.json" \
  --video-width 640 \
  --video-height 360 \
  > "$OUTPUT_DIR/mujoco/eval.log" 2>&1

python - "$OUTPUT_DIR/mujoco/result.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"MuJoCo eval did not produce {path}")
result = json.loads(path.read_text(encoding="utf-8"))
attempts = int(result.get("counts", {}).get("attempts", 0))
if attempts <= 0:
    raise SystemExit(f"MuJoCo eval result has no attempts: {path}")
PY

python scripts/summarize_planner_policy_alignment.py \
  "$OUTPUT_DIR/mujoco/contact.csv" \
  --json-out "$OUTPUT_DIR/planner_alignment/alignment.json" \
  --markdown-out "$OUTPUT_DIR/planner_alignment/alignment.md" \
  > "$OUTPUT_DIR/planner_alignment/alignment.log" 2>&1

sha256sum \
  "$OUTPUT_DIR/exported/hope_pingpong.onnx" \
  "$OUTPUT_DIR/exported/hope_pingpong.onnx.data" \
  > "$OUTPUT_DIR/exported/sha256.txt"
