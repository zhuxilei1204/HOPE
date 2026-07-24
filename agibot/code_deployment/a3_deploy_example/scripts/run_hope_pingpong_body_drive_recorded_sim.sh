#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_hope_pingpong_body_drive_recorded_sim.sh [duration_seconds]
  run_hope_pingpong_body_drive_recorded_sim.sh --duration 5 --output-dir /tmp/hope_record
  run_hope_pingpong_body_drive_recorded_sim.sh --duration 5 --convert
  run_hope_pingpong_body_drive_recorded_sim.sh --duration 8 --reset-height 1.3
  run_hope_pingpong_body_drive_recorded_sim.sh --duration 5 -- --log-every 25

Runs the current HOPE pingpong ONNX body-drive deploy loop against the headless
MuJoCo body-drive simulator while recording raw /body_drive topics with AimRT.

Outputs:
  <session>/raw   AimRT raw MCAP bag and the generated recorder config
  <session>/logs  MuJoCo, recorder, deploy, and optional conversion logs
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
deploy_root="$repo_root/agibot/code_deployment/a3_deploy_example"
runtime_dir="$deploy_root/dist/codex_x86_64"
sim_bin="$repo_root/agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/build/codex_x86_64/install/bin"
deploy_script="$deploy_root/scripts/run_hope_pingpong_body_drive_sim.sh"
reset_script="$deploy_root/scripts/publish_hope_pingpong_mujoco_reset.py"
onnx="$repo_root/analysis/parallel_collision_recovery_20260724/eval_collision_model20994/exported_model20994/hope_pingpong.onnx"
adapter="$repo_root/a3_deploy/a3_deploy_example/config/action_adapter.yaml"
recorder_template="${RECORDER_TEMPLATE:-$runtime_dir/config/a3_body_drive_debug_record.iceoryx_ros2_sim.yaml}"
if [[ ! -f "${recorder_template}" ]]; then
  recorder_template="$runtime_dir/config/a3_body_drive_debug_record.iceoryx.yaml"
fi
recorder_bin="$runtime_dir/a3_body_drive_debug_record"
recorder_converter="$runtime_dir/tools/a3_body_drive_debug_convert.py"
ort_root="${ORT_ROOT:-/tmp/hope_codex_deps/onnxruntime-linux-x64-1.19.2}"
sysroot="${HOPE_CODEX_SYSROOT:-/tmp/hope_codex_deps/sysroot}"

duration="5"
duration_set=0
output_dir=""
convert=0
reset_sim=1
reset_height="${HOPE_MUJOCO_RESET_HEIGHT:-1.3}"
deploy_args=()
original_args=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --duration" >&2
        exit 64
      fi
      duration="$2"
      duration_set=1
      shift 2
      ;;
    --duration=*)
      duration="${1#*=}"
      duration_set=1
      shift
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --output-dir" >&2
        exit 64
      fi
      output_dir="$2"
      shift 2
      ;;
    --output-dir=*)
      output_dir="${1#*=}"
      shift
      ;;
    --convert)
      convert=1
      shift
      ;;
    --no-convert)
      convert=0
      shift
      ;;
    --no-reset)
      reset_sim=0
      shift
      ;;
    --reset-height)
      if [[ $# -lt 2 ]]; then
        echo "missing value for --reset-height" >&2
        exit 64
      fi
      reset_height="$2"
      shift 2
      ;;
    --reset-height=*)
      reset_height="${1#*=}"
      shift
      ;;
    --)
      shift
      deploy_args+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "unknown argument: $1" >&2
      echo "pass deploy-only arguments after --" >&2
      exit 64
      ;;
    *)
      if [[ "${duration_set}" -ne 0 ]]; then
        echo "unexpected positional argument: $1" >&2
        exit 64
      fi
      duration="$1"
      duration_set=1
      shift
      ;;
  esac
done

if [[ ! "${duration}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "duration must be a positive number of seconds: ${duration}" >&2
  exit 64
fi
if [[ ! "${reset_height}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
  echo "reset height must be a number: ${reset_height}" >&2
  exit 64
fi

require_file() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "${path}" ]]; then
    echo "missing ${desc}: ${path}" >&2
    exit 66
  fi
}

require_exec() {
  local path="$1"
  local desc="$2"
  if [[ ! -x "${path}" ]]; then
    echo "missing executable ${desc}: ${path}" >&2
    exit 66
  fi
}

require_exec "${deploy_script}" "HOPE body-drive deploy script"
require_exec "${reset_script}" "MuJoCo reset publisher"
require_exec "${recorder_bin}" "A3 body-drive recorder"
require_exec "${sim_bin}/start_a3_pingpong_iceoryx.sh" "MuJoCo body-drive simulator launcher"
require_file "${recorder_template}" "recorder template"
require_file "${onnx}" "HOPE pingpong ONNX model"
require_file "${adapter}" "HOPE action adapter config"
require_file "${ort_root}/lib/libonnxruntime.so" "ONNX Runtime shared library"

if [[ -x /home/zxl/miniconda3/envs/zxl-pace/bin/python ]]; then
  default_python="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
else
  default_python="python3"
fi
python_bin="${PYTHON_BIN:-${default_python}}"

if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required to manage MuJoCo/recorder process groups" >&2
  exit 69
fi

if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
  set -u
elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi

for setup in \
    "${sim_bin}/../share/ros2_plugin_proto/local_setup.bash" \
    "${sim_bin}/../share/aimrt_msgs/local_setup.bash" \
    "${sim_bin}/../share/joint_msgs/local_setup.bash" \
    "${sim_bin}/../share/mujoco_sim_msgs/local_setup.bash"; do
  if [[ -f "${setup}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${setup}"
    set -u
  fi
done

export LD_LIBRARY_PATH="${ort_root}/lib:${sysroot}/usr/lib/x86_64-linux-gnu:${runtime_dir}:${sim_bin}:${sim_bin}/../lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${sysroot}/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

stamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${output_dir}" ]]; then
  session_dir="${HOPE_BODY_DRIVE_RECORD_ROOT:-${runtime_dir}/bags/hope_pingpong_body_drive}/${stamp}"
else
  session_dir="${output_dir}"
fi
session_dir="$("${python_bin}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${session_dir}")"
raw_dir="${session_dir}/raw"
log_dir="${session_dir}/logs"
mkdir -p "${raw_dir}" "${log_dir}"

recorder_cfg="${raw_dir}/a3_body_drive_debug_record.iceoryx.yaml"
"${python_bin}" - "${recorder_template}" "${recorder_cfg}" "${raw_dir}" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
bag = Path(sys.argv[3]).resolve()
text = src.read_text(encoding="utf-8").replace("__BAG_PATH__", str(bag))
dst.write_text(text, encoding="utf-8")
PY

"${python_bin}" - "${session_dir}" "${raw_dir}" "${log_dir}" "${recorder_cfg}" "${onnx}" "${duration}" "${runtime_dir}" "${sim_bin}" "${reset_sim}" "${reset_height}" "${original_args[@]}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

session = Path(sys.argv[1])
raw = Path(sys.argv[2])
logs = Path(sys.argv[3])
cfg = Path(sys.argv[4])
onnx = Path(sys.argv[5])
duration = sys.argv[6]
runtime = Path(sys.argv[7])
sim = Path(sys.argv[8])
reset_enabled = bool(int(sys.argv[9]))
reset_height = float(sys.argv[10])
args = sys.argv[11:]

metadata = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "duration_seconds": duration,
    "onnx": str(onnx),
    "transport": "iceoryx+ros2",
    "recorded_topics": [
        "/body_drive/waist_joint_state",
        "/body_drive/leg_joint_state",
        "/body_drive/arm_joint_state",
        "/body_drive/neck_joint_state",
        "/body_drive/waist_joint_command",
        "/body_drive/leg_joint_command",
        "/body_drive/arm_joint_command",
        "/body_drive/neck_joint_command",
        "/body_drive/pelvis_imu/data",
        "/body_drive/torso_imu/data",
        "/sim/a3/pelvis_pose",
    ],
    "session_dir": str(session),
    "raw_dir": str(raw),
    "log_dir": str(logs),
    "recorder_config": str(cfg),
    "runtime_dir": str(runtime),
    "mujoco_sim_bin": str(sim),
    "mujoco_reset": {
        "enabled": reset_enabled,
        "mode": "keyframe",
        "keyframe_id": 0,
        "height": reset_height,
        "pose": "a3_pingpong.xml keyframe stand",
        "joints": "a3_pingpong.xml keyframe stand qpos[7:38]",
    },
    "script_args": args,
}
session.joinpath("metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
session.joinpath("README.txt").write_text(
    "\n".join(
        [
            "HOPE pingpong body-drive MuJoCo recording",
            f"duration_seconds: {duration}",
            f"transport: iceoryx+ros2",
            f"raw: {raw}",
            f"logs: {logs}",
            f"recorder_config: {cfg}",
            f"onnx: {onnx}",
            "",
            "This session records the same /body_drive joint state, IMU, and joint",
            "command topics used by the hal_ethercat body-drive interface.",
            "",
            "Optional conversion:",
            f"  {runtime}/run_a3_body_drive_debug_convert.sh \"{raw}\"",
            "",
        ]
    ),
    encoding="utf-8",
)
PY

sim_pgid=""
deploy_pgid=""
recorder_pgid=""
preexisting_roudi="$(pgrep -x iox-roudi || true)"

stop_group() {
  local label="$1"
  local pgid="${2:-}"
  local grace_s="${3:-2}"
  if [[ -z "${pgid}" ]]; then
    return
  fi
  local target=""
  if kill -0 -- "-${pgid}" 2>/dev/null; then
    target="-${pgid}"
  elif kill -0 "${pgid}" 2>/dev/null; then
    target="${pgid}"
  else
    return
  fi
  echo "[hope-recorded-sim] stopping ${label} (pgid=${pgid})"
  if [[ "${target}" == -* ]]; then
    kill -INT -- "${target}" 2>/dev/null || true
  else
    kill -INT "${target}" 2>/dev/null || true
  fi
  sleep "${grace_s}"
  if [[ "${target}" == -* ]]; then
    if kill -0 -- "${target}" 2>/dev/null; then
      kill -TERM -- "${target}" 2>/dev/null || true
      sleep 1
    fi
    if kill -0 -- "${target}" 2>/dev/null; then
      kill -KILL -- "${target}" 2>/dev/null || true
    fi
  else
    if kill -0 "${target}" 2>/dev/null; then
      kill -TERM "${target}" 2>/dev/null || true
      sleep 1
    fi
    if kill -0 "${target}" 2>/dev/null; then
      kill -KILL "${target}" 2>/dev/null || true
    fi
  fi
  wait "${pgid}" 2>/dev/null || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  stop_group "recorder" "${recorder_pgid}" 5
  recorder_pgid=""
  stop_group "deploy" "${deploy_pgid}" 3
  deploy_pgid=""
  stop_group "MuJoCo sim" "${sim_pgid}" 2
  sim_pgid=""
  if [[ -z "${preexisting_roudi}" ]]; then
    pkill -x iox-roudi >/dev/null 2>&1 || true
  fi
  exit "${rc}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_deploy_prepare() {
  local log_file="$1"
  local timeout_s="${2:-12}"
  local start_s="${SECONDS}"
  while (( SECONDS - start_s < timeout_s )); do
    if [[ -f "${log_file}" ]] && rg -q 'prepare tick=0|mode: policy inference' "${log_file}"; then
      return 0
    fi
    sleep 0.1
  done
  return 1
}

echo "[hope-recorded-sim] session=${session_dir}"
echo "[hope-recorded-sim] starting headless MuJoCo body-drive simulator"
setsid bash -c '
  set -euo pipefail
  cd "$1"
  export A3_MUJOCO_HEADLESS=1
  export AIMRT_MUJOCO_SIM_HEADLESS=1
  export LD_LIBRARY_PATH="$2"
  exec ./start_a3_pingpong_iceoryx.sh
' bash "${sim_bin}" "${LD_LIBRARY_PATH}" >"${log_dir}/mujoco_sim.log" 2>&1 &
sim_pgid="$!"

sleep 2
if ! kill -0 -- "-${sim_pgid}" 2>/dev/null; then
  echo "[hope-recorded-sim] MuJoCo simulator exited during startup" >&2
  tail -n 80 "${log_dir}/mujoco_sim.log" >&2 || true
  exit 1
fi

if [[ "${reset_sim}" -ne 0 ]]; then
  deploy_duration="$("${python_bin}" -c 'import sys; print(f"{float(sys.argv[1]) + 3.0:g}")' "${duration}")"
  echo "[hope-recorded-sim] starting deploy prepare loop for ${deploy_duration}s"
  setsid "${deploy_script}" "${deploy_duration}" "${deploy_args[@]}" >"${log_dir}/deploy.log" 2>&1 &
  deploy_pgid="$!"

  if ! wait_for_deploy_prepare "${log_dir}/deploy.log" 12; then
    echo "[hope-recorded-sim] deploy did not enter prepare before reset" >&2
    tail -n 120 "${log_dir}/deploy.log" >&2 || true
    exit 1
  fi

  echo "[hope-recorded-sim] resetting MuJoCo to a3_pingpong.xml keyframe stand"
  set +e
  python3 "${reset_script}" \
    --mode keyframe \
    --keyframe-id 0 \
    --repeat 20 \
    --rate-hz 50 \
    --wait-subscriber-s 3 \
    >"${log_dir}/reset.log" 2>&1
  reset_rc=$?
  set -e
  if [[ "${reset_rc}" -ne 0 ]]; then
    echo "[hope-recorded-sim] MuJoCo reset failed with rc=${reset_rc}" >&2
    tail -n 120 "${log_dir}/reset.log" >&2 || true
    exit "${reset_rc}"
  fi
  sleep 0.2

  echo "[hope-recorded-sim] starting AimRT raw recorder for ${duration}s"
  setsid bash -c '
    set -euo pipefail
    cd "$1"
    export LD_LIBRARY_PATH="$2"
    exec ./a3_body_drive_debug_record --cfg_file_path "$3"
  ' bash "${runtime_dir}" "${LD_LIBRARY_PATH}" "${recorder_cfg}" >"${log_dir}/recorder.log" 2>&1 &
  recorder_pgid="$!"

  sleep 1
  if ! kill -0 -- "-${recorder_pgid}" 2>/dev/null; then
    echo "[hope-recorded-sim] recorder exited during startup" >&2
    tail -n 80 "${log_dir}/recorder.log" >&2 || true
    exit 1
  fi

  record_sleep="$("${python_bin}" -c 'import sys; print(f"{max(float(sys.argv[1]) - 1.0, 0.0):g}")' "${duration}")"
  sleep "${record_sleep}"
  stop_group "recorder" "${recorder_pgid}" 5
  recorder_pgid=""

  set +e
  wait "${deploy_pgid}"
  deploy_rc=$?
  set -e
  deploy_pgid=""
  if [[ "${deploy_rc}" -ne 0 ]]; then
    echo "[hope-recorded-sim] deploy failed with rc=${deploy_rc}" >&2
    tail -n 120 "${log_dir}/deploy.log" >&2 || true
    exit "${deploy_rc}"
  fi
else
  echo "[hope-recorded-sim] starting AimRT raw recorder"
  setsid bash -c '
    set -euo pipefail
    cd "$1"
    export LD_LIBRARY_PATH="$2"
    exec ./a3_body_drive_debug_record --cfg_file_path "$3"
  ' bash "${runtime_dir}" "${LD_LIBRARY_PATH}" "${recorder_cfg}" >"${log_dir}/recorder.log" 2>&1 &
  recorder_pgid="$!"

  sleep 1
  if ! kill -0 -- "-${recorder_pgid}" 2>/dev/null; then
    echo "[hope-recorded-sim] recorder exited during startup" >&2
    tail -n 80 "${log_dir}/recorder.log" >&2 || true
    exit 1
  fi

  echo "[hope-recorded-sim] running deploy loop for ${duration}s"
  set +e
  "${deploy_script}" "${duration}" "${deploy_args[@]}" >"${log_dir}/deploy.log" 2>&1
  deploy_rc=$?
  set -e

  if [[ "${deploy_rc}" -ne 0 ]]; then
    echo "[hope-recorded-sim] deploy failed with rc=${deploy_rc}" >&2
    tail -n 120 "${log_dir}/deploy.log" >&2 || true
    exit "${deploy_rc}"
  fi

  stop_group "recorder" "${recorder_pgid}" 5
  recorder_pgid=""
fi

stop_group "MuJoCo sim" "${sim_pgid}" 2
sim_pgid=""
if [[ -z "${preexisting_roudi}" ]]; then
  pkill -x iox-roudi >/dev/null 2>&1 || true
fi

if [[ "${convert}" -ne 0 ]]; then
  if [[ ! -f "${recorder_converter}" ]]; then
    echo "[hope-recorded-sim] missing converter: ${recorder_converter}" >&2
    exit 66
  fi
  echo "[hope-recorded-sim] converting raw bag"
  set +e
  "${python_bin}" "${recorder_converter}" "${raw_dir}" >"${log_dir}/convert.log" 2>&1
  convert_rc=$?
  set -e
  if [[ "${convert_rc}" -ne 0 ]]; then
    echo "[hope-recorded-sim] conversion failed with rc=${convert_rc}" >&2
    tail -n 120 "${log_dir}/convert.log" >&2 || true
    exit "${convert_rc}"
  fi
fi

bag_count="$(find "${raw_dir}" -type f -name '*.mcap' | wc -l | tr -d ' ')"
echo "[hope-recorded-sim] complete"
echo "[hope-recorded-sim] raw=${raw_dir}"
echo "[hope-recorded-sim] logs=${log_dir}"
echo "[hope-recorded-sim] mcap_files=${bag_count}"
echo "[hope-recorded-sim] convert later with:"
echo "  ${runtime_dir}/run_a3_body_drive_debug_convert.sh \"${raw_dir}\""
