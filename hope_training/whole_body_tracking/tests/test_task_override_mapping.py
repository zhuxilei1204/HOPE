from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts/train.py"
SPEC = importlib.util.spec_from_file_location("hope_train_override_test", TRAIN_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def _fixture() -> SimpleNamespace:
    return SimpleNamespace(
        rewards=SimpleNamespace(
            term=SimpleNamespace(
                weight=-1.0,
                params={"strike_scale": 0.1, "nested": {"gain": 1.0}},
            )
        )
    )


def test_set_dotted_traverses_attribute_and_mapping_nodes() -> None:
    cfg = _fixture()
    applied: list[str] = []
    TRAIN._set_dotted(
        cfg,
        "rewards.term.params.strike_scale",
        0.0,
        applied,
        "test",
    )
    TRAIN._set_dotted(
        cfg,
        "rewards.term.params.nested.gain",
        2.0,
        applied,
        "test",
    )
    assert cfg.rewards.term.params["strike_scale"] == 0.0
    assert cfg.rewards.term.params["nested"]["gain"] == 2.0
    assert applied == [
        "rewards.term.params.strike_scale = 0.0",
        "rewards.term.params.nested.gain = 2.0",
    ]


def test_set_dotted_does_not_create_unknown_mapping_keys() -> None:
    cfg = _fixture()
    applied: list[str] = []
    TRAIN._set_dotted(
        cfg,
        "rewards.term.params.unknown",
        3.0,
        applied,
        "test",
    )
    assert "unknown" not in cfg.rewards.term.params
    assert applied == []
