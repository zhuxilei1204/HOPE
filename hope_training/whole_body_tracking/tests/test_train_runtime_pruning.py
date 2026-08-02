from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train.py"
SPEC = importlib.util.spec_from_file_location("hope_train_runtime", SCRIPT)
assert SPEC and SPEC.loader
train_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train_module
SPEC.loader.exec_module(train_module)


class _StatefulReward:
    def __call__(self):
        return None

    def reset(self, env_ids):
        return None


def _term(weight: float, func) -> SimpleNamespace:
    return SimpleNamespace(weight=weight, func=func)


def test_prunes_only_stateless_zero_weight_rewards() -> None:
    active = _term(1.5, lambda: None)
    zero_stateless = _term(0.0, lambda: None)
    zero_stateful = _term(0.0, _StatefulReward())
    rewards = SimpleNamespace(
        active=active,
        zero_stateless=zero_stateless,
        zero_stateful=zero_stateful,
        absent=None,
    )
    env_cfg = SimpleNamespace(rewards=rewards)

    pruned, stateful = train_module._prune_stateless_zero_weight_reward_terms(env_cfg)

    assert pruned == ("zero_stateless",)
    assert stateful == ("zero_stateful",)
    assert rewards.active is active
    assert rewards.zero_stateless is None
    assert rewards.zero_stateful is zero_stateful
    assert rewards.absent is None


def test_pruning_is_idempotent_and_handles_missing_reward_cfg() -> None:
    rewards = SimpleNamespace(zero=_term(0.0, lambda: None))
    env_cfg = SimpleNamespace(rewards=rewards)

    first = train_module._prune_stateless_zero_weight_reward_terms(env_cfg)
    second = train_module._prune_stateless_zero_weight_reward_terms(env_cfg)

    assert first == (("zero",), ())
    assert second == ((), ())
    assert train_module._prune_stateless_zero_weight_reward_terms(SimpleNamespace()) == ((), ())
