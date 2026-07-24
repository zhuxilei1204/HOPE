#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
deploy_root="$repo_root/agibot/code_deployment/a3_deploy_example"
runtime_dir="$deploy_root/dist/codex_x86_64"
cfg="$deploy_root/src/a3/a3_deploy_onnx_ref/config/hope_pingpong_body_drive.yaml"
onnx="$repo_root/analysis/parallel_collision_recovery_20260724/eval_collision_model20994/exported_model20994/hope_pingpong.onnx"
ort_root="${ORT_ROOT:-/tmp/hope_codex_deps/onnxruntime-linux-x64-1.19.2}"
sysroot="${HOPE_CODEX_SYSROOT:-/tmp/hope_codex_deps/sysroot}"

if [ -f /opt/ros/jazzy/setup.bash ]; then
  # shellcheck source=/dev/null
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
elif [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck source=/dev/null
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

export LD_LIBRARY_PATH="$ort_root/lib:$sysroot/usr/lib/x86_64-linux-gnu:$runtime_dir:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$sysroot/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

exec "$runtime_dir/hope_pingpong_body_drive" \
  --runtime-cfg "$cfg" \
  --onnx "$onnx" \
  --duration "${1:-5}" \
  --publish-commands \
  "${@:2}"
