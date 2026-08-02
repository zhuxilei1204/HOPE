#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"
SEED="${SEED:-742}"
TAG="${TAG:-20260801}"
START_STAGGER_S="${START_STAGGER_S:-70}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
ANALYSIS_ROOT="${REPO}/analysis/closed_loop_v3_phase_c7_20260801"

TASK_A="HOPEPingPongClosedLoopV3ScratchPhaseImmediateC7AMultiSkill114"
TASK_B="HOPEPingPongClosedLoopV3ScratchPhaseEscrowC7BMultiSkill114"
WINDOW_A="${WINDOW_A:-clv3C7AImmediate}"
WINDOW_B="${WINDOW_B:-clv3C7BEscrow}"
RUN_NAME_A="${RUN_NAME_A:-CLV3_C7A_PHASE_IMMEDIATE_MULTI114_${NUM_ENVS}x${MAX_ITERATIONS}_SEED${SEED}_${TAG}}"
RUN_NAME_B="${RUN_NAME_B:-CLV3_C7B_PHASE_ESCROW_MULTI114_${NUM_ENVS}x${MAX_ITERATIONS}_SEED${SEED}_${TAG}}"

mkdir -p "${ANALYSIS_ROOT}"
"${PY}" "${ROOT}/scripts/audit_closed_loop_v3_phase_c7.py" \
  > "${ANALYSIS_ROOT}/preflight_audit.json"

for window in "${WINDOW_A}" "${WINDOW_B}"; do
  if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${window}"; then
    echo "tmux window already exists: ${SESSION}:${window}" >&2
    exit 1
  fi
done

start_run() {
  local role="$1"
  local task="$2"
  local device="$3"
  local window="$4"
  local run_name="$5"
  local out_dir="${ANALYSIS_ROOT}/${run_name}"

  mkdir -p "${out_dir}"
  cat > "${out_dir}/experiment_contract.txt" <<EOF
initialization=fresh_actor_critic_optimizer
checkpoint_path=null
task=${task}
ab_role=${role}
paired_seed=${SEED}
actor_observation_dim=114
motion_manifest=five_contact_aware_multiskill_clips
ball_route=table_workspace_independent_of_motion
route_curriculum=station_relocation_arrival_settle_contact_safety
incoming_speed_curriculum=healthy_contact_recovery_safety
planner_noise=fixed_0.10_not_coupled_to_curricula
lifecycle=relocation_ready_then_strike_then_outcome_conditioned_recovery_ready
outcome_tier=highest_actual_contact_net_bounce
num_envs=${NUM_ENVS}
iterations=${MAX_ITERATIONS}
seed=${SEED}
device=${device}
promotion_gate=resolved_config_audit,healthy_contact,net_cross,cycle_ready,fall,relocation_and_speed_level
EOF

  sha256sum \
    "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/closed_loop_v2.py" \
    "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py" \
    "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py" \
    "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py" \
    "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_closed_loop_v3_scratch_env_cfg.py" \
    "${ROOT}/cfg/task/HOPEPingPongClosedLoopV3ScratchPhaseCurriculumC7CommonMultiSkill114.yaml" \
    "${ROOT}/cfg/task/HOPEPingPongClosedLoopV3ScratchPhaseImmediateC7AMultiSkill114.yaml" \
    "${ROOT}/cfg/task/HOPEPingPongClosedLoopV3ScratchPhaseEscrowC7BMultiSkill114.yaml" \
    "${ROOT}/cfg/algo/ppo_closed_loop_v3_scratch.yaml" \
    > "${out_dir}/source_sha256.txt"

  tmux new-window -d -t "${SESSION}:" -n "${window}" \
    "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=${task} \
algo=ppo_closed_loop_v3_scratch \
device=${device} \
num_envs=${NUM_ENVS} \
max_iterations=${MAX_ITERATIONS} \
seed=${SEED} \
checkpoint_path=null \
checkpoint_actor_only=false \
checkpoint_load_optimizer=false \
run_name=${run_name} \
headless=true > '${out_dir}/train.log' 2>&1"

  printf 'started %s:%s (%s)\n' "${SESSION}" "${window}" "${device}"
  printf 'log: %s\n' "${out_dir}/train.log"
}

start_run "A_immediate_physical_payoff" "${TASK_A}" "cuda:0" "${WINDOW_A}" "${RUN_NAME_A}"
if (( START_STAGGER_S > 0 )); then
  # Both jobs import the same URDF into a temporary USD. Stagger only scene
  # creation; training remains concurrent after the second process starts.
  sleep "${START_STAGGER_S}"
fi
start_run "B_escrow_until_reusable_ready" "${TASK_B}" "cuda:1" "${WINDOW_B}" "${RUN_NAME_B}"
