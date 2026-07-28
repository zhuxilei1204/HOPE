from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(__file__))
_PATH = os.path.join(
    _ROOT,
    "source",
    "whole_body_tracking",
    "whole_body_tracking",
    "tasks",
    "tracking",
    "mdp",
    "table_workspace.py",
)
_SPEC = importlib.util.spec_from_file_location("hope_table_workspace", _PATH)
workspace = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = workspace
_SPEC.loader.exec_module(workspace)

interpolate_bounds = workspace.interpolate_bounds
table_side_lateral_bounds = workspace.table_side_lateral_bounds
validate_table_workspace = workspace.validate_table_workspace


def test_side_ranges_cover_regulation_table_width() -> None:
    width = 1.525
    margin = 0.02
    overlap = 0.12
    forehand = table_side_lateral_bounds(width, margin, overlap, 1.0)
    backhand = table_side_lateral_bounds(width, margin, overlap, -1.0)

    assert math.isclose(forehand[0], -(width / 2.0 - margin))
    assert math.isclose(backhand[1], width / 2.0 - margin)
    assert forehand[1] == overlap
    assert backhand[0] == -overlap
    assert forehand[1] >= backhand[0]


def test_bounds_expand_from_core_to_full() -> None:
    core = (-0.55, -0.20)
    full = (-0.7425, 0.12)
    assert interpolate_bounds(core, full, 0.0) == core
    assert interpolate_bounds(core, full, 1.0) == full
    halfway = interpolate_bounds(core, full, 0.5)
    assert halfway == pytest.approx((-0.64625, -0.04))


def test_workspace_validation_rejects_core_outside_table() -> None:
    with pytest.raises(ValueError, match="contained"):
        validate_table_workspace(
            1.525,
            0.02,
            0.12,
            (-0.80, -0.20),
            (-0.15, 0.18),
        )
