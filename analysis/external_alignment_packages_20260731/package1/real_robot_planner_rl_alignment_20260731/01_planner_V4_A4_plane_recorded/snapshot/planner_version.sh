#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERSIONS_DIR="$ROOT/planner_versions"
ACTIVE_FILE="$ROOT/config/active_planner_version"
REGISTRY_FILE="$ROOT/config/planner_versions.tsv"
DEFAULT_VERSION="v3_mac_validated"
STACK_LOCK="${A3_HDU_STACK_LOCK:-/tmp/hope_726_hdu_stack.lock}"

usage() {
  cat <<'EOF'
usage:
  ./planner_version.sh menu
  ./planner_version.sh list
  ./planner_version.sh current
  ./planner_version.sh resolve
  ./planner_version.sh select <number|version_id>
  ./planner_version.sh check [number|version_id]
  ./planner_version.sh path [number|version_id]
  ./planner_version.sh edit [number|version_id]
  ./planner_version.sh clone <source_number|source_id> <new_version_id>

Environment override for one stack launch:
  A3_PLANNER_VERSION=2 ./run_hdu_stack.sh
EOF
}

registry_rows() {
  [[ -f "$REGISTRY_FILE" ]] || {
    echo "missing planner registry: $REGISTRY_FILE" >&2
    return 2
  }
  awk 'NF >= 2 && $1 !~ /^#/ { print $1, $2 }' "$REGISTRY_FILE"
}

validate_version() {
  local version="$1"
  if [[ ! "$version" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "invalid planner version id: $version" >&2
    return 2
  fi
  local dir="$VERSIONS_DIR/$version"
  for required in \
    "$dir/VERSION.md" \
    "$dir/config/hope_planner.yaml" \
    "$dir/python/hope_planner/__init__.py" \
    "$dir/python/hope_planner/node.py"; do
    if [[ ! -f "$required" ]]; then
      echo "incomplete planner version $version: missing $required" >&2
      return 2
    fi
  done
}

version_from_ref() {
  local ref="$1" version=""
  if [[ "$ref" =~ ^[0-9]+$ ]]; then
    version="$(registry_rows | awk -v wanted="$ref" '$1 == wanted { print $2; exit }')"
  else
    version="$(registry_rows | awk -v wanted="$ref" '$2 == wanted { print $2; exit }')"
  fi
  if [[ -z "$version" ]]; then
    echo "planner version is not registered: $ref" >&2
    return 2
  fi
  validate_version "$version"
  printf '%s\n' "$version"
}

number_for_version() {
  local version="$1"
  registry_rows | awk -v wanted="$version" '$2 == wanted { print $1; exit }'
}

persistent_version() {
  local version="$DEFAULT_VERSION"
  if [[ -s "$ACTIVE_FILE" ]]; then
    IFS= read -r version <"$ACTIVE_FILE"
  fi
  version_from_ref "$version"
}

resolved_version() {
  if [[ -n "${A3_PLANNER_VERSION:-}" ]]; then
    version_from_ref "$A3_PLANNER_VERSION"
  else
    persistent_version
  fi
}

version_summary() {
  local version="$1"
  awk 'NF && $0 !~ /^#/ { print; exit }' "$VERSIONS_DIR/$version/VERSION.md"
}

list_versions() {
  local active number version marker summary
  active="$(persistent_version)"
  while read -r number version; do
    validate_version "$version"
    marker=" "
    [[ "$version" == "$active" ]] && marker="*"
    summary="$(version_summary "$version")"
    printf '%s %2s  %-24s %s\n' "$marker" "$number" "$version" "$summary"
  done < <(registry_rows)
  if [[ -n "${A3_PLANNER_VERSION:-}" ]]; then
    printf 'environment override: %s\n' "$(resolved_version)"
  fi
}

select_version() (
  local version
  version="$(version_from_ref "$1")"

  exec 8>"$STACK_LOCK"
  if ! flock -n 8; then
    echo "refusing to switch planner version while run_hdu_stack.sh is active" >&2
    echo "type q in the stack terminal (or Ctrl-C), then retry" >&2
    return 3
  fi

  local tmp
  tmp="$(mktemp "$ROOT/config/.active_planner_version.XXXXXX")"
  printf '%s\n' "$version" >"$tmp"
  chmod 0644 "$tmp"
  mv -f "$tmp" "$ACTIVE_FILE"
  echo "active planner version: $version"
  echo "takes effect on the next run_hdu_stack.sh start"
)

check_one() {
  local version
  version="$(version_from_ref "$1")"
  local dir="$VERSIONS_DIR/$version"
  PYTHONPATH="$dir/python${PYTHONPATH:+:$PYTHONPATH}" \
    python3 - "$version" "$dir/config/hope_planner.yaml" <<'PY'
import hashlib
import math
import pathlib
import sys

import numpy as np
import yaml

from hope_planner.racket_target_planner import RacketTargetPlanner

version = sys.argv[1]
config_path = pathlib.Path(sys.argv[2])
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
if "hope_planner" not in config or "ros__parameters" not in config["hope_planner"]:
    raise SystemExit(f"{version}: invalid ROS parameter file")

normal = RacketTargetPlanner._opponent_facing_normal(
    np.array([2.0, 0.2, 0.1], dtype=float)
)
if not np.all(np.isfinite(normal)) or not math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1e-9):
    raise SystemExit(f"{version}: invalid racket normal {normal}")
if normal[0] <= 0.0:
    raise SystemExit(f"{version}: racket normal does not face +X: {normal}")

digest = hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]
print(f"{version}: PASS config_sha256={digest} sample_normal={normal.tolist()}")
PY
}

edit_version() (
  local version editor
  version="$(version_from_ref "${1:-$(persistent_version)}")"
  exec 8>"$STACK_LOCK"
  if ! flock -n 8; then
    echo "stop run_hdu_stack.sh before editing planner parameters" >&2
    return 3
  fi

  editor="${EDITOR:-}"
  if [[ -z "$editor" ]]; then
    if command -v nano >/dev/null 2>&1; then editor="nano"; else editor="vi"; fi
  fi
  echo "editing $version"
  "$editor" "$VERSIONS_DIR/$version/config/hope_planner.yaml"
  check_one "$version"
)

clone_version() (
  local source new_version next_number tmp
  source="$(version_from_ref "$1")"
  new_version="$2"
  if [[ ! "$new_version" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "invalid new version id: $new_version" >&2
    return 2
  fi
  if [[ -e "$VERSIONS_DIR/$new_version" ]]; then
    echo "planner version already exists: $new_version" >&2
    return 2
  fi

  exec 8>"$STACK_LOCK"
  if ! flock -n 8; then
    echo "stop run_hdu_stack.sh before creating a planner version" >&2
    return 3
  fi

  next_number="$(registry_rows | awk 'BEGIN { max=0 } $1+0 > max { max=$1+0 } END { print max+1 }')"
  cp -a "$VERSIONS_DIR/$source" "$VERSIONS_DIR/$new_version"
  {
    printf '# %s\n\n' "$new_version"
    printf '从 `%s` 克隆。请修改 config/hope_planner.yaml 或 python/hope_planner，\n' "$source"
    printf '运行 `./planner_version.sh check %s` 后再选择启动。\n' "$new_version"
  } >"$VERSIONS_DIR/$new_version/VERSION.md"

  tmp="$(mktemp "$ROOT/config/.planner_versions.XXXXXX")"
  cp -a "$REGISTRY_FILE" "$tmp"
  printf '%s %s\n' "$next_number" "$new_version" >>"$tmp"
  chmod 0644 "$tmp"
  mv -f "$tmp" "$REGISTRY_FILE"
  echo "created planner version $next_number: $new_version (from $source)"
  echo "next: ./planner_version.sh edit $new_version"
)

interactive_menu() {
  [[ -t 0 ]] || {
    echo "planner menu requires an interactive terminal" >&2
    return 2
  }

  local active active_number choice source new_version
  while true; do
    active="$(persistent_version)"
    active_number="$(number_for_version "$active")"
    echo
    echo "========== HOPE Planner Versions =========="
    list_versions
    echo "  e  edit parameters of current version"
    echo "  n  clone a new version"
    echo "  q  exit without starting stack"
    printf 'Select version [%s = %s]: ' "$active_number" "$active"
    IFS= read -r choice
    choice="${choice:-$active_number}"
    case "$choice" in
      q|Q|quit|exit)
        echo "stack start cancelled"
        return 10
        ;;
      e|E)
        edit_version "$active"
        ;;
      n|N)
        printf 'Clone from version [%s]: ' "$active_number"
        IFS= read -r source
        source="${source:-$active_number}"
        printf 'New version id (example: v5_my_test): '
        IFS= read -r new_version
        clone_version "$source" "$new_version"
        ;;
      *)
        select_version "$choice"
        return 0
        ;;
    esac
  done
}

case "${1:-}" in
  menu)
    interactive_menu
    ;;
  list)
    list_versions
    ;;
  current)
    persistent_version
    ;;
  resolve)
    resolved_version
    ;;
  select)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    select_version "$2"
    ;;
  check)
    if [[ $# -eq 2 ]]; then
      check_one "$2"
    elif [[ $# -eq 1 ]]; then
      while read -r _ version; do check_one "$version"; done < <(registry_rows)
    else
      usage >&2
      exit 2
    fi
    ;;
  path)
    version="$(version_from_ref "${2:-$(persistent_version)}")"
    printf '%s\n' "$VERSIONS_DIR/$version"
    ;;
  edit)
    [[ $# -le 2 ]] || { usage >&2; exit 2; }
    edit_version "${2:-$(persistent_version)}"
    ;;
  clone)
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    clone_version "$2" "$3"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
