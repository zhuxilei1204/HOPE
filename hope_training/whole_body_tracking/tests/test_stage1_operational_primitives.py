from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/whole_body_tracking/whole_body_tracking/utils/operational_action.py"
)
SPEC = importlib.util.spec_from_file_location("operational_action", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bounded_squared_excess_is_zero_inside_and_capped() -> None:
    values = torch.tensor([[0.0, 0.03, 0.06, 0.30]])
    result = MODULE.bounded_squared_excess(values, scale=0.03, maximum=4.0)
    torch.testing.assert_close(result, torch.tensor([[0.0, 1.0, 4.0, 4.0]]))


def test_robust_log_excess_retains_order_before_final_cap() -> None:
    values = torch.tensor([[0.0, 0.5, 2.0, 10.0, 100.0]])
    result = MODULE.robust_log_excess(values, scale=0.5, maximum=4.0)
    assert result[0, 0] == 0.0
    assert result[0, 1] < result[0, 2] < result[0, 3] < result[0, 4]
    torch.testing.assert_close(result[0, 4], torch.tensor(4.0))


def test_grouped_joint_cost_does_not_dilute_waist() -> None:
    values = torch.zeros((1, 31))
    values[:, 0:3] = 1.0
    values[:, 19:31] = 2.0
    result = MODULE.grouped_joint_cost(
        values,
        waist_scale=1.0,
        other_upper_scale=0.2,
        right_arm_scale=0.1,
        leg_scale=0.5,
    )
    torch.testing.assert_close(result, torch.tensor([2.0]))
