#!/usr/bin/env bash
# Run the 2026-07-24 collision-recovery checkpoint through the clean 111-D
# HOPE ping-pong reference deploy path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${HERE}/.." && pwd)"
REPO_ROOT="$(cd "${EXAMPLE_DIR}/../.." && pwd)"
REF_DIR="${EXAMPLE_DIR}/reference"

PYTHON_BIN="${PYTHON:-/home/zxl/miniconda3/envs/zxl-pace/bin/python}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/analysis/parallel_collision_recovery_20260724/eval_collision_model20994/exported_model20994/hope_pingpong.onnx}"
CONFIG_PATH="${CONFIG_PATH:-${EXAMPLE_DIR}/config/hope_pingpong_runtime.yaml}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON=/path/to/python, or use the zxl-pace environment." >&2
  exit 2
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ONNX model not found: ${MODEL_PATH}" >&2
  exit 2
fi

if [[ ! -f "${MODEL_PATH}.data" ]]; then
  echo "ONNX external data file not found: ${MODEL_PATH}.data" >&2
  echo "Keep hope_pingpong.onnx.data next to hope_pingpong.onnx." >&2
  exit 2
fi

export PYTHONPATH="${REF_DIR}:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" -m a3_deploy_onnx_ref_pingpong \
  --config "${CONFIG_PATH}" \
  --backend mujoco \
  --onnx "${MODEL_PATH}" \
  "$@"
