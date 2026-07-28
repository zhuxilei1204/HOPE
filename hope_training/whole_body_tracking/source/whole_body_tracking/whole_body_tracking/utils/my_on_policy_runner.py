"""Minimal rsl_rl PPO runner glue for HOPE training.

The base ``rsl_rl.runners.OnPolicyRunner`` already writes periodic (every ``save_interval``) and final
local checkpoints. This subclass only replaces the logging writer with a local, offline no-op sink so
training pulls in **no** Weights & Biases / TensorBoard / external logging service, and adds no gate,
lineage, receipt, or ONNX-export coupling (export is a separate script). Per-iteration console
progress from rsl_rl is preserved; the only shipped machine-readable metric is ``success_rate`` from
``scripts/evaluate.py``.
"""

from __future__ import annotations

import os

import torch
from rsl_rl.runners import OnPolicyRunner

from whole_body_tracking.utils.actor_anchor import ActorParameterAnchor


class _LocalNullWriter:
    """A local, offline stand-in for rsl_rl's summary writer.

    Implements the small surface rsl_rl calls on ``self.writer`` (``add_scalar``, ``log_config``,
    ``save_model``, ``save_file``, ``stop``/``flush``) as no-ops, and returns a no-op for anything
    else, so training never depends on TensorBoard or Weights & Biases. Checkpoints are still written
    locally by the runner's ``save()``.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def log_config(self, *args, **kwargs) -> None:
        pass

    def save_model(self, *args, **kwargs) -> None:
        pass

    def save_file(self, *args, **kwargs) -> None:
        pass

    def stop(self, *args, **kwargs) -> None:
        pass

    def flush(self, *args, **kwargs) -> None:
        pass

    def close(self, *args, **kwargs) -> None:
        pass

    def __getattr__(self, _name):
        # Any other writer method (across rsl_rl versions) becomes a no-op.
        def _noop(*args, **kwargs):
            return None

        return _noop


class HOPEOnPolicyRunner(OnPolicyRunner):
    """rsl_rl OnPolicyRunner with local-only, offline logging (no W&B / TensorBoard)."""

    def configure_actor_anchor(
        self,
        checkpoint_path: str,
        coefficient: float,
        first_layer_input_exempt_start: int | None = None,
    ) -> None:
        """Anchor the actor to the actor stored in ``checkpoint_path``."""
        path = os.path.abspath(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise ValueError(f"checkpoint has no model_state_dict: {path}")
        reference_actor_state = {
            name.removeprefix("actor."): tensor
            for name, tensor in model_state.items()
            if name.startswith("actor.")
        }
        actor = getattr(self.alg.policy, "actor", None)
        if actor is None:
            raise ValueError("actor anchoring requires the policy to expose an actor module")
        self.actor_anchor = ActorParameterAnchor(
            actor,
            reference_actor_state,
            coefficient,
            first_layer_input_exempt_start=first_layer_input_exempt_start,
        )

        original_update = self.alg.update

        def anchored_update():
            losses = original_update()
            losses.update(self.actor_anchor.metrics())
            return losses

        self.alg.update = anchored_update
        print(
            f"[train.py] actor anchor enabled: coefficient={float(coefficient):.6g}, "
            f"first_layer_input_exempt_start={first_layer_input_exempt_start}, "
            f"reference={path}",
            flush=True,
        )

    def _prepare_logging_writer(self) -> None:
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = "local"
            self.writer = _LocalNullWriter(log_dir=self.log_dir)
