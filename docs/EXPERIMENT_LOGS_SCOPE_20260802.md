# Experiment Logs Scope - 2026-08-02

This note documents what was intentionally committed as experiment evidence on
2026-08-02.

## Included

The committed log set focuses on curated, text-readable experiment evidence under:

```text
hope_training/whole_body_tracking/analysis/
```

That directory contains README files, design notes, diagnosis reports, JSON
result snapshots, small stdout/stderr/time artifacts, and short diagnostic logs
for the current stage-1 / stage-2 planner-executor and physical-evaluation
experiments.

The main included experiment groups are:

- `stage1_plane020_merged_20260802`
- `stage1_plane020_escrow_20260802`
- `stage1_planner_executor_20260802`
- `stage2_closed_loop_command_20260802`
- `stage2_impact_calibration_20260802`
- `stage2_physical_20260801`
- smaller closed-loop-v2 and deployment-station diagnostic logs

To keep the Git repository usable, only concise logs were committed. Large raw
training logs and trajectory dumps remain local.

## Excluded

The following were intentionally not committed:

- Checkpoints: `*.pt`, `*.pth`, `*.ckpt`
- Exported model binaries: `*.onnx`, `*.onnx.data`
- Videos: `*.mp4`
- ROS / recording bags: `*.mcap`
- Large CSV trajectories and raw replay dumps
- Long raw train/eval logs above 1 MB
- Raw motion-pipeline data under top-level `data/`
- Large historical experiment dumps under top-level `analysis/`
- Full TensorBoard/checkpoint history under `hope_training/whole_body_tracking/logs/`

The excluded directories contain many large files and several blobs above the
GitHub 100 MB hard limit. They should be archived through Git LFS, GitHub
Releases, object storage, or a separate shared drive if full-fidelity history is
required.

## Size Guard

Before committing, staged files were checked so that no file exceeded 95 MB and
no checkpoint/model/video/MCAP files were included. The final committed
experiment evidence is intentionally summary-heavy rather than a full raw log
archive.
