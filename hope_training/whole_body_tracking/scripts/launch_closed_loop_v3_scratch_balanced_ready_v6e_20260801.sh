#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
WINDOW="${TMUX_WINDOW:-clv3V6EBalReady10}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
SEED="${SEED:-738}"
TAG="${TAG:-20260801}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
RUN_NAME="${RUN_NAME:-CLV3_SCRATCH_V6E_BALANCED_READY10_MULTI114_${NUM_ENVS}x${MAX_ITERATIONS}_SEED${SEED}_${TAG}}"
OUT_DIR="${OUT_DIR:-${REPO}/analysis/closed_loop_v3_scratch_20260731/${RUN_NAME}}"

if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${WINDOW}"; then
  echo "tmux window already exists: ${SESSION}:${WINDOW}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cat > "${OUT_DIR}/experiment_contract.txt" <<EOF
initialization=fresh_actor_critic_optimizer
checkpoint_path=null
task=HOPEPingPongClosedLoopV3ScratchBalancedReadyV6EMultiSkill114
ab_role=V6C_plus_ready_balance_reward_with_unchanged_no_ball_distribution
paired_control=V6C_seed738
actor_observation_dim=114
motion_manifest=five_contact_aware_multiskill_clips
motion_clip_identity_weights=0.25,0.25,0.20,0.15,0.15
ball_route=independent_table_workspace_one_bounce
planner_command=v4_wire_compatible_with_joint_perturbation
ready_credit=signed_potential_plus_no_command_additive_balance_weight_1.2
stand_episode_probability=0.10
impact_health_floor_progress=targeted_attempt_ema_at_0.05
motion_seed_low_ability=prestrike_0.60,recovery_0.00,rsi_0.40
motion_seed_high_ability=prestrike_0.50,recovery_0.15,rsi_0.35
motion_lifecycle_transition=targeted_attempt_ema_0.01_to_0.05
motion_seed_probability=ability_gated_0.60_to_0.20
num_envs=${NUM_ENVS}
iterations=${MAX_ITERATIONS}
seed=${SEED}
device=${DEVICE}
promotion_gate=staged_64env_survival_plus_retained_hit_ability
EOF

sha256sum \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_closed_loop_v3_scratch_env_cfg.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/ready_recovery.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py" \
  "${ROOT}/cfg/task/HOPEPingPongClosedLoopV3ScratchAdaptiveMotionV6CMultiSkill114.yaml" \
  "${ROOT}/cfg/task/HOPEPingPongClosedLoopV3ScratchBalancedReadyV6EMultiSkill114.yaml" \
  "${ROOT}/cfg/algo/ppo_closed_loop_v3_scratch.yaml" \
  > "${OUT_DIR}/source_sha256.txt"

tmux new-window -d -t "${SESSION}:" -n "${WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongClosedLoopV3ScratchBalancedReadyV6EMultiSkill114 \
algo=ppo_closed_loop_v3_scratch \
device=${DEVICE} \
num_envs=${NUM_ENVS} \
max_iterations=${MAX_ITERATIONS} \
seed=${SEED} \
checkpoint_path=null \
checkpoint_actor_only=false \
checkpoint_load_optimizer=false \
run_name=${RUN_NAME} \
headless=true > '${OUT_DIR}/train.log' 2>&1"

printf 'started %s:%s\n' "${SESSION}" "${WINDOW}"
printf 'log: %s\n' "${OUT_DIR}/train.log"
