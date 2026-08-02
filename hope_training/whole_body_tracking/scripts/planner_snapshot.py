"""Load a HOPE planner package from an explicit, versioned source tree."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import pathlib
import sys
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PlannerAPI:
    PlannerConfig: type
    HOPEPlanner: type
    CommandStabilityConfig: type | None
    CommandStabilityGate: type | None
    load_ball_physics: object
    load_paddle_params: object
    load_table_params: object
    select_swing_side: object
    package_dir: pathlib.Path
    source_sha256: str


def instantiate_with_supported_kwargs(cls: type, values: Mapping[str, object]):
    """Instantiate a versioned planner type without passing unknown fields."""
    parameters = inspect.signature(cls).parameters
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = (
        dict(values)
        if accepts_arbitrary
        else {name: value for name, value in values.items() if name in parameters}
    )
    return cls(**kwargs)


def _source_tree_sha256(package_dir: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def load_planner_api(
    repo_root: str | pathlib.Path,
    package_dir: str | pathlib.Path | None = None,
) -> PlannerAPI:
    """Import the planner API without silently replacing the requested version."""
    root = pathlib.Path(repo_root).expanduser().resolve()
    requested = (
        pathlib.Path(package_dir).expanduser().resolve()
        if package_dir is not None
        else root / "hope_ws/src/hope_planner/hope_planner"
    )
    if not requested.is_dir() or not (requested / "__init__.py").is_file():
        raise FileNotFoundError(
            "planner code directory must be a Python package containing "
            f"__init__.py: {requested}"
        )
    package_name = requested.name
    if not package_name.isidentifier():
        raise ValueError(
            f"planner package directory must be a valid identifier: {requested}"
        )

    loaded = sys.modules.get(package_name)
    if loaded is not None:
        loaded_file = pathlib.Path(getattr(loaded, "__file__", "")).resolve()
        if loaded_file.parent != requested:
            raise RuntimeError(
                f"planner package {package_name!r} is already loaded from "
                f"{loaded_file.parent}, requested {requested}"
            )

    parent = str(requested.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    constants = importlib.import_module(f"{package_name}.constants")
    planner = importlib.import_module(f"{package_name}.planner")
    side_selection = importlib.import_module(f"{package_name}.side_selection")
    try:
        stability_gate = importlib.import_module(
            f"{package_name}.command_stability_gate"
        )
    except ModuleNotFoundError as exc:
        if exc.name != f"{package_name}.command_stability_gate":
            raise
        stability_gate = None
    return PlannerAPI(
        PlannerConfig=constants.PlannerConfig,
        HOPEPlanner=planner.HOPEPlanner,
        CommandStabilityConfig=(
            stability_gate.CommandStabilityConfig
            if stability_gate is not None
            else None
        ),
        CommandStabilityGate=(
            stability_gate.CommandStabilityGate
            if stability_gate is not None
            else None
        ),
        load_ball_physics=constants.load_ball_physics,
        load_paddle_params=constants.load_paddle_params,
        load_table_params=constants.load_table_params,
        select_swing_side=side_selection.select_swing_side,
        package_dir=requested,
        source_sha256=_source_tree_sha256(requested),
    )
