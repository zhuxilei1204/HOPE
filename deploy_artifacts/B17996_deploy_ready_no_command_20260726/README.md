# B17996 Deploy-Ready/No-Command Baseline

This bundle is the validated B+deploy baseline, not the currently running `+3000`
continuation experiment.

## Source Checkpoint

`/mnt/ssd/zxl/HOPE_latest_20260721/hope_training/whole_body_tracking/logs/rsl_rl/hope_pingpong/2026-07-25_18-09-45_normal114_deploy_ready_no_command_fromB14997_noopt_4096x3000_20260725/model_17996.pt`

The checkpoint files are copied into this bundle under:

- `checkpoints/model_15000.pt`
- `checkpoints/model_15500.pt`
- `checkpoints/model_16000.pt`
- `checkpoints/model_16500.pt`
- `checkpoints/model_17000.pt`
- `checkpoints/model_17500.pt`
- `checkpoints/model_17996.pt`

Use `checkpoints/model_17996.pt` as the validated B17996 baseline checkpoint.
The earlier checkpoint files are included only for rollback / comparison.

Training configuration snapshots are included under:

- `train_params/env.yaml`
- `train_params/agent.yaml`

Git diff snapshots from the training run are included under:

- `git_diff/HOPE_latest_20260721.diff`
- `git_diff/PACE-ICRA2026.diff`

## Exported Policy

- `models/hope_pingpong.onnx`
- `models/hope_pingpong.onnx.data`
- `models/policy_manifest.json`

Contract:

- observation: `float32[1,114]`
- action: `float32[1,31]`
- control rate: `50 Hz`
- observation normalization: `none`
- actor observation contract: `hope_pingpong_normal114`

## Runtime Files

- `config/hope_pingpong_runtime.yaml`
- `config/action_adapter.yaml`
- `config/joint_order_agibot_a3.yaml`
- `config/hope_planner.yaml`

## Reference Deploy Code

Use the runner under:

`/mnt/ssd/zxl/HOPE_latest_20260721/a3_deploy/a3_deploy_example/reference/a3_deploy_onnx_ref_pingpong`

The runner supports both 111-D and 114-D ONNX policies by inspecting the ONNX input
dimension.  For this model it must build the 114-D observation, including
`racket_target_normal_w`. If planner normal is not available, the current reference
runner falls back to normalizing the target velocity.

## Last Validated MuJoCo Gate

Evaluation file:

`/mnt/ssd/zxl/HOPE_latest_20260721/analysis/Bdeploy_recheck_after_S2S3_20260725/B17996_currentcode_realplanner_continuous40.json`

Result:

- success: `0.325`
- contact: `0.475`
- net: `0.325`
- opponent bounce: `0.350`
- fall: `0.000`

## Integrity

`checksums.sha256` contains SHA-256 checksums for the policy, config, checkpoint,
training-parameter, and git-diff files in this bundle.
