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

    def load_actor_only(self, checkpoint_path: str) -> None:
        """Load only actor-network weights and keep a fresh critic/optimizer/iteration."""
        path = os.path.abspath(checkpoint_path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise ValueError(f"checkpoint has no model_state_dict: {path}")
        actor_state = {
            name.removeprefix("actor."): tensor
            for name, tensor in model_state.items()
            if name.startswith("actor.")
        }
        if not actor_state:
            raise ValueError(f"checkpoint has no actor parameters: {path}")
        actor = getattr(self.alg.policy, "actor", None)
        if actor is None:
            raise ValueError("actor-only loading requires the policy to expose an actor module")
        actor.load_state_dict(actor_state, strict=True)
        print(
            "[train.py] actor-only warm start loaded; critic, exploration state, "
            f"optimizer, and iteration remain fresh: {path}",
            flush=True,
        )

    def configure_critic_only_warmup(self, num_updates: int) -> None:
        """Fit the critic to a changed task before allowing actor drift.

        A task-changing fine-tune must not combine a pretrained actor with a
        random or badly scaled value function. During these initial updates the
        rollout policy is unchanged; only critic parameters receive gradients.
        """
        remaining = int(num_updates)
        if remaining < 1:
            raise ValueError("critic-only warmup must contain at least one update")
        policy = self.alg.policy
        actor = getattr(policy, "actor", None)
        if actor is None:
            raise ValueError("critic-only warmup requires an actor module")
        actor_parameters = list(actor.parameters())
        for name in ("std", "log_std"):
            parameter = getattr(policy, name, None)
            if isinstance(parameter, torch.nn.Parameter):
                actor_parameters.append(parameter)
        original_update = self.alg.update
        state = {"remaining": remaining}

        def critic_only_update():
            frozen = state["remaining"] > 0
            if frozen:
                for parameter in actor_parameters:
                    parameter.requires_grad_(False)
            try:
                losses = dict(original_update())
            finally:
                if frozen:
                    for parameter in actor_parameters:
                        parameter.requires_grad_(True)
            losses["critic_only_warmup"] = float(frozen)
            if frozen:
                state["remaining"] -= 1
            return losses

        self.alg.update = critic_only_update
        print(
            "[train.py] critic-only task adaptation enabled for "
            f"{remaining} update(s); actor mean and action std are frozen",
            flush=True,
        )

    def configure_actor_step_trust_region(
        self,
        max_action_rms: float,
        max_action_p99: float,
        max_samples: int = 4096,
    ) -> None:
        """Bound each PPO actor update by its behavior on the current rollout.

        PPO's clipped objective does not bound the final policy after all mini-
        batch epochs.  A task-changing warm start can therefore destroy a valid
        actor in one update even with a small learning rate.  This wrapper keeps
        the critic update intact and line-searches only the actor/noise parameter
        delta until deterministic action drift on rollout observations is inside
        the configured trust region.
        """
        rms_limit = float(max_action_rms)
        p99_limit = float(max_action_p99)
        sample_limit = int(max_samples)
        if rms_limit <= 0.0 or p99_limit <= 0.0:
            raise ValueError("actor step trust-region limits must be positive")
        if sample_limit < 1:
            raise ValueError("actor step trust-region max_samples must be positive")
        policy = self.alg.policy
        if getattr(policy, "is_recurrent", False):
            raise ValueError("actor step trust region currently requires a feed-forward actor")
        actor = getattr(policy, "actor", None)
        if actor is None:
            raise ValueError("actor step trust region requires an actor module")

        guarded_parameters = list(actor.parameters())
        for name in ("std", "log_std"):
            parameter = getattr(policy, name, None)
            if isinstance(parameter, torch.nn.Parameter) and all(
                parameter is not existing for existing in guarded_parameters
            ):
                guarded_parameters.append(parameter)

        original_update = self.alg.update

        @torch.no_grad()
        def action_drift(observations: torch.Tensor, reference: torch.Tensor):
            current = policy.act_inference(observations)
            absolute = torch.abs(current - reference)
            rms = torch.sqrt(torch.mean(torch.square(absolute)))
            p99 = torch.quantile(absolute.reshape(-1), 0.99)
            return rms, p99

        def guarded_update():
            observations = self.alg.storage.observations.reshape(
                -1, *self.alg.storage.observations.shape[2:]
            )
            if observations.shape[0] > sample_limit:
                stride = (observations.shape[0] + sample_limit - 1) // sample_limit
                observations = observations[::stride][:sample_limit]
            observations = observations.detach().clone()
            with torch.no_grad():
                reference_actions = policy.act_inference(observations).detach().clone()
                before = [parameter.detach().clone() for parameter in guarded_parameters]

            losses = dict(original_update())
            with torch.no_grad():
                after = [parameter.detach().clone() for parameter in guarded_parameters]
                raw_rms, raw_p99 = action_drift(observations, reference_actions)
                scale = min(
                    1.0,
                    rms_limit / max(float(raw_rms.item()), 1.0e-12),
                    p99_limit / max(float(raw_p99.item()), 1.0e-12),
                )
                if scale < 1.0:
                    # Account for network non-linearity rather than assuming
                    # action drift scales exactly with parameter interpolation.
                    low = 0.0
                    high = scale
                    for _ in range(10):
                        candidate = 0.5 * (low + high)
                        for parameter, start, end in zip(
                            guarded_parameters, before, after, strict=True
                        ):
                            parameter.copy_(start + candidate * (end - start))
                        candidate_rms, candidate_p99 = action_drift(
                            observations, reference_actions
                        )
                        if (
                            float(candidate_rms.item()) <= rms_limit
                            and float(candidate_p99.item()) <= p99_limit
                        ):
                            low = candidate
                        else:
                            high = candidate
                    scale = low
                    for parameter, start, end in zip(
                        guarded_parameters, before, after, strict=True
                    ):
                        parameter.copy_(start + scale * (end - start))

                    # Keep Adam momentum consistent with the accepted actor
                    # step. Critic optimizer state is intentionally untouched.
                    for parameter in guarded_parameters:
                        state = self.alg.optimizer.state.get(parameter, {})
                        if "exp_avg" in state:
                            state["exp_avg"].mul_(scale)
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"].mul_(scale * scale)
                        if "max_exp_avg_sq" in state:
                            state["max_exp_avg_sq"].mul_(scale * scale)

                applied_rms, applied_p99 = action_drift(
                    observations, reference_actions
                )
            losses.update(
                {
                    "actor_step_raw_action_rms": float(raw_rms.item()),
                    "actor_step_raw_action_p99": float(raw_p99.item()),
                    "actor_step_applied_action_rms": float(applied_rms.item()),
                    "actor_step_applied_action_p99": float(applied_p99.item()),
                    "actor_step_scale": float(scale),
                    "actor_step_guarded": float(scale < 0.999999),
                }
            )
            return losses

        self.alg.update = guarded_update
        print(
            "[train.py] actor step trust region enabled: "
            f"rms<={rms_limit:.6g}, p99<={p99_limit:.6g}, "
            f"samples<={sample_limit}",
            flush=True,
        )

    @torch.no_grad()
    def override_action_noise_std(self, std_by_index: dict[int, float]) -> None:
        """Override selected raw-action exploration standard deviations."""
        if not std_by_index:
            return
        policy = self.alg.policy
        noise_type = getattr(policy, "noise_std_type", None)
        parameter = getattr(policy, "log_std" if noise_type == "log" else "std", None)
        if not isinstance(parameter, torch.Tensor):
            raise ValueError(
                "action-noise overrides require a policy with per-action std or log_std"
            )
        applied = []
        for index, value in sorted(std_by_index.items()):
            index = int(index)
            value = float(value)
            if not 0 <= index < parameter.numel():
                raise ValueError(
                    f"action-noise override index {index} is outside [0, {parameter.numel()})"
                )
            if value <= 0.0:
                raise ValueError(
                    f"action-noise std must be positive at index {index}, got {value}"
                )
            before = (
                float(parameter[index].exp().item())
                if noise_type == "log"
                else float(parameter[index].item())
            )
            parameter[index] = torch.log(parameter.new_tensor(value)) if noise_type == "log" else value
            applied.append((index, before, value))
        print(
            "[train.py] action-noise std overrides applied: "
            + ", ".join(f"{index}:{before:.6g}->{after:.6g}" for index, before, after in applied),
            flush=True,
        )

    def configure_actor_anchor(
        self,
        checkpoint_path: str,
        coefficient: float,
        first_layer_input_exempt_start: int | None = None,
        first_layer_input_exempt_end: int | None = None,
        exempt_coefficient: float = 0.0,
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
            first_layer_input_exempt_end=first_layer_input_exempt_end,
            exempt_coefficient=exempt_coefficient,
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
            f"first_layer_input_exempt_end={first_layer_input_exempt_end}, "
            f"exempt_coefficient={float(exempt_coefficient):.6g}, "
            f"reference={path}",
            flush=True,
        )

    def _prepare_logging_writer(self) -> None:
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = "local"
            self.writer = _LocalNullWriter(log_dir=self.log_dir)
