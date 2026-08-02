"""Runtime-only helpers for command metric reset logging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def batched_metric_means_and_clear(
    metrics: Mapping[str, torch.Tensor],
    env_ids: Sequence[int] | slice,
) -> dict[str, float]:
    """Match Isaac CommandTerm.reset with one device-to-host synchronization."""
    if not metrics:
        return {}
    names: list[str] = []
    device_means: list[torch.Tensor] = []
    # Keep Isaac's mean-then-clear ordering. This matters if two diagnostic
    # names intentionally alias the same tensor; only the final host sync is
    # batched.
    for name, value in metrics.items():
        names.append(name)
        device_means.append(torch.mean(value[env_ids]).to(dtype=torch.float32))
        value[env_ids] = 0.0
    means = torch.stack(device_means).detach().cpu().tolist()
    return {name: float(mean) for name, mean in zip(names, means)}
