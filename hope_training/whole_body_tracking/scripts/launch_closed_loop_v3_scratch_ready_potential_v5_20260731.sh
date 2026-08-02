#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-zxl}"
WINDOW="${TMUX_WINDOW:-clv3ScratchReadyV5}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
SEED="${SEED:-737}"
ROOT="/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking"
REPO="/mnt/ssd/zxl/HOPE_latest_20260721"
PY="/home/zxl/miniconda3/envs/zxl-pace/bin/python"
PYTHONPATH_VALUE="${ROOT}/source/whole_body_tracking"
OUT_DIR="${REPO}/analysis/closed_loop_v3_scratch_20260731/main_ready_potential_v5_4096x10000_seed737"
RUN_NAME="CLV3_SCRATCH_READY_POTENTIAL_V5_MULTI114_4096x10000_SEED737_20260731"

if tmux list-windows -t "${SESSION}" -F '#{window_name}' | grep -Fxq "${WINDOW}"; then
  echo "tmux window already exists: ${SESSION}:${WINDOW}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cat > "${OUT_DIR}/experiment_contract.txt" <<EOF
initialization=fresh_actor_critic_optimizer
checkpoint_path=null
failed_parent_diagnostic=ready_contract_v4_model_1000.pt_not_loaded
task=HOPEPingPongClosedLoopV3ScratchReadyPotentialV5MultiSkill114
algo=ppo_closed_loop_v3_scratch
motion_manifest=${REPO}/hope_training/motions/supplemental_lower_body_contact_aware_isaacfk_20260728/manifest_train_ready_with_task_boxes.tsv
actor_observation_dim=114
action_feedback=effective
planner_command=v4_wire_compatible_with_joint_perturbation
ball_route=independent_table_workspace_one_bounce
reset_distribution=ability_gated_default_ready_prestrike_random_rsi_and_complete_no_ball
stand_episode_joint_contract=always_deploy_ready_no_command
default_stand_hold_joint_contract=always_deploy_ready_no_command
ready_reward=bounded_instability_cost_plus_signed_score_potential_plus_one_shot_sustained_event
ready_potential_entry_payoff=zero
ready_potential_static_payoff=zero
ready_potential_deterioration=signed_negative
recovery_reward=signed_backlean_error_reduction_only
dense_ready_region_reward=disabled
num_envs=${NUM_ENVS}
iterations=${MAX_ITERATIONS}
seed=${SEED}
device=${DEVICE}
promotion_gate=no_command_physical_survival,contact_discovery,healthy_contact,impact_speed,both_side_contact,post_contact_ready,fall_overflow_trend
EOF

sha256sum \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_closed_loop_v3_scratch_env_cfg.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/ready_recovery.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/closed_loop_v2.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py" \
  "${ROOT}/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py" \
  "${ROOT}/cfg/task/HOPEPingPongClosedLoopV3ScratchMultiSkill114.yaml" \
  "${ROOT}/cfg/task/HOPEPingPongClosedLoopV3ScratchReadyPotentialV5MultiSkill114.yaml" \
  "${ROOT}/cfg/algo/ppo_closed_loop_v3_scratch.yaml" \
  > "${OUT_DIR}/source_sha256.txt"

tmux new-window -d -t "${SESSION}:" -n "${WINDOW}" \
  "cd '${ROOT}' && exec env PYTHONPATH='${PYTHONPATH_VALUE}' '${PY}' scripts/train.py \
task=HOPEPingPongClosedLoopV3ScratchReadyPotentialV5MultiSkill114 \
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
