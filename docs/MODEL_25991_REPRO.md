# model_25991 Reproduction Package

This note documents the lightweight reproduction package committed for
`model_25991`.

## Checkpoint

The baseline checkpoint used by the 25991 replay/evaluation scripts is committed
at:

```text
hope_training/whole_body_tracking/logs/rsl_rl/hope_pingpong/2026-07-30_13-41-29_M24492_DeployStationStepV1_4096x1500_20260730/model_25991.pt
```

SHA256:

```text
402ef9118db5fa5bedb10580c3df398e7ed3fcab98123f388dcc6c1b3dfcadfb
```

## Planner Snapshot

The recorded planner snapshot required by `eval_m25991_actuator_ab_member_20260731.sh`
is committed at:

```text
analysis/external_alignment_packages_20260731/package1/real_robot_planner_rl_alignment_20260731/01_planner_V4_A4_plane_recorded/snapshot/
```

Only source/config files are included; `__pycache__` files are excluded.

## Experiment Evidence

The committed analysis subset includes text-readable diagnostics for:

- `analysis/deploy_station_step_v1_20260730/eval_model_25991/`
- `analysis/deployment_jitter_audit_20260730/stand_traces/model25991_ready20.json`
- `analysis/footwork_station_ab_20260730/final_eval_20260730/baseline_model25991/`
- `analysis/m25991_actuator_robust_ab_20260731/`
- `analysis/real_session_replay_model25991_20260731/`

The included files are small `*.md`, `*.json`, `*.txt`, short `*.log`, manifest,
checksum, and contract files. They are intended to capture experiment setup,
diagnosis, outcome summaries, and reproducibility metadata.

## Excluded

The following raw artifacts remain local and are intentionally not committed:

- Videos: `*.mp4`
- Raw or large CSV traces, including joint-action traces
- Exported model binaries: `*.onnx`, `*.onnx.data`
- Large raw train logs
- Recorded replay blobs such as `*.npz`
- Follow-up A/B checkpoints such as `model_26490.pt`

## Entry Points

The two primary scripts are:

```text
hope_training/whole_body_tracking/scripts/launch_m25991_actuator_robust_ab_20260731.sh
hope_training/whole_body_tracking/scripts/eval_m25991_actuator_ab_member_20260731.sh
```

Both scripts now derive paths from the Git repository root by default. Runtime
environment details can still be overridden through environment variables such
as `REPO`, `ROOT`, `PY`, `CONDA_SH`, `CONDA_ENV`, `CHECKPOINT`, `SNAPSHOT`,
`MANIFEST`, `NUM_SERVES`, and `ISAAC_SERVES`.
