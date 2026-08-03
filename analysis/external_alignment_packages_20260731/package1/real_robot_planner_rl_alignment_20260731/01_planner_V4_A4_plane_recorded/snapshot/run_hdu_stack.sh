#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="${A3_HOPE_WS:-/agibot/hope_726_ws}"

if [[ -f /agibot/software/v0/entry/env/env.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source /agibot/software/v0/entry/env/env.sh
  set -u
fi

if [[ ! -f "$WS/install/setup.bash" ]]; then
  echo "missing 726 ROS workspace: $WS/install/setup.bash" >&2
  exit 1
fi

set +u
# shellcheck source=/dev/null
source "$WS/install/setup.bash"
set -u

exec 9>/tmp/hope_726_hdu_stack.lock
if ! flock -n 9; then
  echo "another HOPE 726 HDU stack is already running" >&2
  exit 1
fi

if [[ -t 0 && -z "${A3_PLANNER_VERSION:-}" && "${A3_PLANNER_MENU:-1}" != "0" ]]; then
  flock -u 9
  set +e
  "$ROOT/planner_version.sh" menu
  menu_status=$?
  set -e
  if [[ "$menu_status" -eq 10 ]]; then
    exit 0
  elif [[ "$menu_status" -ne 0 ]]; then
    exit "$menu_status"
  fi
  if ! flock -n 9; then
    echo "another HOPE 726 HDU stack started while the planner menu was open" >&2
    exit 1
  fi
fi

MOTIVE_HOST="${A3_MOTIVE_HOST:-192.168.100.111}"
MOTIVE_INTERFACE_IP="${A3_MOTIVE_INTERFACE_IP:-192.168.100.252}"
ROBOT_OBJECT="${A3_ROBOT_OBJECT:-FDU_a3}"
BALL_OBJECT="${A3_BALL_OBJECT:-A}"
BALL_SOURCE="${A3_BALL_SOURCE:-unlabeled}"
BALL_MIN_SIZE_M="${A3_BALL_MIN_SIZE_M:-0.030}"
export A3_BALL_MIN_SIZE_M="$BALL_MIN_SIZE_M"
BASE_TOPIC="${A3_BASE_TOPIC:-/FDU_a3/pose}"
RACKET_OBJECT="${A3_RACKET_OBJECT:-FDU_pai}"
RACKET_TOPIC="${A3_RACKET_TOPIC:-/FDU_pai/pose}"
STARTUP_TIMEOUT_S="${A3_STARTUP_TIMEOUT_S:-30}"

children=()
input_pid=""
cleanup() {
  trap - INT TERM EXIT
  if [[ -n "$input_pid" ]]; then
    kill "$input_pid" 2>/dev/null || true
    wait "$input_pid" 2>/dev/null || true
  fi
  if ((${#children[@]})); then
    for pid in "${children[@]}"; do
      kill -TERM -- "-$pid" 2>/dev/null || true
    done
    wait "${children[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

setsid python3 -u "$ROOT/scripts/hdu_record_control_server.py" \
  --port "${A3_HDU_RECORD_PORT:-17243}" \
  --recorder "$ROOT/record_hdu_planner.sh" \
  --monitor-host "${A3_MONITOR_HOST:-192.168.47.249}" \
  --monitor-port "${A3_MONITOR_PORT:-17650}" \
  --mdu-marker-host "${A3_MDU_HOST:-10.42.10.12}" \
  --mdu-marker-port "${A3_MDU_MARKER_PORT:-17244}" \
  --record-root "${A3_HDU_RECORD_ROOT:-$ROOT/records_hdu}" 9>&- &
children+=("$!")

setsid ros2 launch hope_bringup optitrack_hope_bridge.launch.py \
  hostname:="$MOTIVE_HOST" \
  interface_ip:="$MOTIVE_INTERFACE_IP" \
  robot_object:="$ROBOT_OBJECT" \
  ball_object:="$BALL_OBJECT" \
  ball_source:="$BALL_SOURCE" \
  ball_min_size_m:="$BALL_MIN_SIZE_M" \
  robot_pose_topic:="$BASE_TOPIC" \
  racket_object:="$RACKET_OBJECT" \
  racket_pose_topic:="$RACKET_TOPIC" \
  start_world:=false 9>&- &
optitrack_pid="$!"
children+=("$optitrack_pid")

setsid "$ROOT/run_hdu_planner.sh" 9>&- &
children+=("$!")

echo "waiting for $BASE_TOPIC from Motive $MOTIVE_HOST"
base_ready=false
deadline=$((SECONDS + STARTUP_TIMEOUT_S))
while ((SECONDS < deadline)); do
  if timeout 3 ros2 topic echo "$BASE_TOPIC" --once \
      --qos-reliability best_effort >/dev/null 2>&1; then
    base_ready=true
    break
  fi
  if ! kill -0 "$optitrack_pid" 2>/dev/null; then
    echo "OptiTrack launch exited before the first base sample" >&2
    exit 1
  fi
done

if [[ "$base_ready" != true ]]; then
  echo "no base sample on $BASE_TOPIC after ${STARTUP_TIMEOUT_S}s" >&2
  exit 1
fi

echo "base pose ready; starting HDU -> MDU planner UDP bridge"
setsid "$ROOT/run_hdu_planner_bridge.sh" 9>&- &
children+=("$!")

if [[ -t 0 ]]; then
  stack_pid="$$"
  echo "HDU stack is running. Type q then Enter, or press Ctrl-C, to stop cleanly."
  (
    while IFS= read -r command; do
      case "$command" in
        q|Q|quit|exit)
          echo "stopping HDU stack"
          kill -TERM "$stack_pid"
          exit 0
          ;;
        *)
          echo "running: type q then Enter to stop"
          ;;
      esac
    done
  ) &
  input_pid="$!"
fi

set +e
wait -n "${children[@]}"
status=$?
set -e
echo "HOPE 726 HDU component exited (status=$status); stopping stack" >&2
exit "$status"
