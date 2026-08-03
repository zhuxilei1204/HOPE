#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CALIBRATION_ENV="${A3_PELVIS_CALIBRATION_ENV:-$ROOT/config/fdu_pelvis_rotation.env}"
if [[ -f "$CALIBRATION_ENV" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$CALIBRATION_ENV"
  set -u
  echo "[hdu-bridge] loaded pelvis rotation: $CALIBRATION_ENV" >&2
fi

if [[ -f /agibot/software/v0/entry/env/env.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source /agibot/software/v0/entry/env/env.sh
  set -u
fi

if [[ -f /agibot/hope_726_ws/install/setup.bash ]]; then
  set +u
  # shellcheck source=/dev/null
  source /agibot/hope_726_ws/install/setup.bash
  set -u
fi

exec python3 "$ROOT/scripts/hdu_planner_udp_bridge.py" \
  --mdu-host "${A3_MDU_HOST:-10.42.10.12}" \
  --base-topic "${A3_BASE_TOPIC:-/FDU_a3/pose}" \
  --ball-topic "${A3_BALL_TOPIC:-/ball/point}" \
  --marker-topic "${A3_MARKER_TOPIC:-/optitrack/markerMetadata}" \
  --ball-min-size-m "${A3_BALL_MIN_SIZE_M:-0.015}" \
  --ball-max-size-m "${A3_BALL_MAX_SIZE_M:-0.050}" \
  --racket-topic "${A3_RACKET_TOPIC:-/FDU_pai/pose}" \
  --command-topic "${A3_COMMAND_TOPIC:-/racket/command}" \
  --max-base-age "${A3_MAX_BASE_AGE:-1.00}" \
  --tracking-warning-age "${A3_TRACKING_WARNING_AGE:-0.10}" \
  --max-command-age "${A3_MAX_COMMAND_AGE:-0.35}" \
  --monitor-host "${A3_MONITOR_HOST:-192.168.47.249}" \
  --monitor-port "${A3_MONITOR_PORT:-17650}" \
  --joint-listen-host "${A3_JOINT_LISTEN_HOST:-0.0.0.0}" \
  --joint-listen-port "${A3_JOINT_LISTEN_PORT:-17242}" \
  --mdu-status-listen-host "${A3_MDU_STATUS_LISTEN_HOST:-0.0.0.0}" \
  --mdu-status-listen-port "${A3_MDU_STATUS_LISTEN_PORT:-17241}" \
  --bias-body-x "${A3_BIAS_X:-0.017662585}" \
  --bias-body-y "${A3_BIAS_Y:--0.003902495}" \
  --bias-body-z "${A3_BIAS_Z:-0.152126430}" \
  --fdu-to-pelvis-qx "${A3_FDU_TO_PELVIS_QX:-0.00382893327}" \
  --fdu-to-pelvis-qy "${A3_FDU_TO_PELVIS_QY:-0.05297072783}" \
  --fdu-to-pelvis-qz "${A3_FDU_TO_PELVIS_QZ:--0.11294189240}" \
  --fdu-to-pelvis-qw "${A3_FDU_TO_PELVIS_QW:-0.99218121843}" \
  "$@"
