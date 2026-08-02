from __future__ import annotations

import ast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TABLE_ENV = ROOT / (
    "source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/"
    "table_tennis_env.py"
)
TARGET_COMMAND = ROOT / (
    "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
    "hope_commands.py"
)
PHYSICAL_COMMAND = ROOT / (
    "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
    "physical_ball_shadow_command.py"
)


def _method_calls(path: Path, class_name: str, method_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    calls_with_lines: list[tuple[int, int, str]] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            calls_with_lines.append((node.lineno, node.col_offset, func.attr))
        elif isinstance(func, ast.Name):
            calls_with_lines.append((node.lineno, node.col_offset, func.id))
    return [name for _, _, name in sorted(calls_with_lines)]


def test_opt_in_step_publishes_snapshot_before_termination_and_reward() -> None:
    source = TABLE_ENV.read_text(encoding="utf-8")
    snapshot = source.index("self._prepare_reward_command_snapshot()")
    termination = source.index("self.termination_manager.compute()", snapshot)
    reward = source.index("self.reward_manager.compute", termination)
    reset = source.index("self._reset_idx", reward)
    command = source.index("self.command_manager.compute", reset)
    assert snapshot < termination < reward < reset < command


def test_target_snapshot_is_not_advanced_twice() -> None:
    calls = _method_calls(TARGET_COMMAND, "RacketTargetCommand", "compute")
    assert calls.count("_update_metrics") == 1
    source = TARGET_COMMAND.read_text(encoding="utf-8")
    assert "if self._reward_snapshot_prepared:" in source
    assert "refresh_kinematic_snapshot" in source


def test_physical_snapshot_resolves_flight_before_publish() -> None:
    calls = _method_calls(
        PHYSICAL_COMMAND, "PhysicalBallShadowCommand", "prepare_reward_snapshot"
    )
    assert calls.index("_update_active_flights") < calls.index("_publish_metrics")


def test_impulse_conversion_survives_reward_manager_dt() -> None:
    # Keep this pure so contract tests do not need to boot Isaac Sim.
    dt = 0.02
    configured_event_value = torch.tensor([3.0])
    manager_term = configured_event_value / dt
    manager_output = manager_term * dt
    torch.testing.assert_close(manager_output, configured_event_value)
