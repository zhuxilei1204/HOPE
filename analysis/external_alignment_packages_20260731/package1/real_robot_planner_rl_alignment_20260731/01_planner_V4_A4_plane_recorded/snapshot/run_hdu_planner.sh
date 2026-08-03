#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERSION="$($ROOT/planner_version.sh resolve)"
VERSION_ROOT="$ROOT/planner_versions/$VERSION"
CONFIG="$VERSION_ROOT/config/hope_planner.yaml"
PYTHON_ROOT="$VERSION_ROOT/python"

echo "planner_version=$VERSION"
echo "planner_config=$CONFIG"
echo "planner_python=$PYTHON_ROOT/hope_planner"

export PYTHONPATH="$PYTHON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m hope_planner.node --ros-args \
  -r __node:=hope_planner \
  --params-file "$CONFIG" \
  "$@"
