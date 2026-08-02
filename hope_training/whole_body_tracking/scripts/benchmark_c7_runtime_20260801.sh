#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
OUT_ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/analysis/c7_runtime_benchmark_20260801"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-4096}"
ITERATIONS="${ITERATIONS:-40}"
SEED="${SEED:-744}"
ALGO="ppo_closed_loop_v3_scratch"

run_case() {
  local label="$1"
  local task="$2"
  local out_dir="${OUT_ROOT}/${label}_${NUM_ENVS}x${ITERATIONS}_SEED${SEED}"
  mkdir -p "${out_dir}"
  printf 'starting %s on %s\n' "${label}" "${DEVICE}"
  env PYTHONPATH="${ROOT}/source/whole_body_tracking" "${PY}" "${ROOT}/scripts/train.py" \
    task="${task}" \
    algo="${ALGO}" \
    device="${DEVICE}" \
    num_envs="${NUM_ENVS}" \
    max_iterations="${ITERATIONS}" \
    seed="${SEED}" \
    checkpoint_path=null \
    checkpoint_actor_only=false \
    checkpoint_load_optimizer=false \
    run_name="${label}_${NUM_ENVS}x${ITERATIONS}_SEED${SEED}" \
    headless=true > "${out_dir}/train.log" 2>&1
}

mkdir -p "${OUT_ROOT}"
nvidia-smi --query-gpu=index,name,driver_version,pstate,clocks.current.sm,memory.total \
  --format=csv > "${OUT_ROOT}/hardware_before.csv"

# Same device, seed, environment count, and PPO configuration. Only command
# metric reset logging differs between these two runs.
run_case \
  "C7_RUNTIME_BASELINE" \
  "HOPEPingPongClosedLoopV3ScratchPhaseImmediateC7AMultiSkill114"
run_case \
  "C7_RUNTIME_BATCHED_METRICS" \
  "HOPEPingPongClosedLoopV3ScratchPhaseImmediateC7ABatchedMetricsMultiSkill114"

printf 'runtime benchmark complete: %s\n' "${OUT_ROOT}"
