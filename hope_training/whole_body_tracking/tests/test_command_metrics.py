from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "command_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("command_metrics", MODULE)
assert SPEC and SPEC.loader
command_metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = command_metrics
SPEC.loader.exec_module(command_metrics)


def test_batched_means_match_individual_means_and_clear_selected_envs() -> None:
    metrics = {
        "a": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "b": torch.tensor([8.0, 6.0, 4.0, 2.0]),
    }
    env_ids = torch.tensor([1, 3])
    expected = {
        name: float(value[env_ids].mean().item())
        for name, value in metrics.items()
    }

    actual = command_metrics.batched_metric_means_and_clear(metrics, env_ids)

    assert actual == expected
    assert torch.equal(metrics["a"], torch.tensor([1.0, 0.0, 3.0, 0.0]))
    assert torch.equal(metrics["b"], torch.tensor([8.0, 0.0, 4.0, 0.0]))


def test_batched_means_support_full_slice_and_preserve_alias_ordering() -> None:
    metrics = {"a": torch.tensor([1.0, 3.0])}
    assert command_metrics.batched_metric_means_and_clear(metrics, slice(None)) == {
        "a": 2.0
    }
    assert torch.count_nonzero(metrics["a"]) == 0

    shared = torch.tensor([2.0, 4.0])
    aliased = {"first": shared, "second": shared}
    assert command_metrics.batched_metric_means_and_clear(
        aliased, slice(None)
    ) == {"first": 3.0, "second": 0.0}
