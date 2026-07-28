"""Actor-parameter anchoring for deployment-preserving PPO fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ActorParameterAnchor:
    """Add an L2 pull toward a frozen reference actor through gradient hooks.

    The hook is equivalent to adding ``0.5 * coefficient * ||theta-theta_ref||^2``
    to the PPO loss, but it leaves the shared rsl_rl PPO implementation and
    checkpoint format unchanged.
    """

    def __init__(
        self,
        actor: nn.Module,
        reference_state: Mapping[str, torch.Tensor],
        coefficient: float,
    ) -> None:
        self.actor = actor
        self.coefficient = float(coefficient)
        if self.coefficient <= 0.0:
            raise ValueError(f"actor anchor coefficient must be positive, got {coefficient}")

        actor_params = dict(actor.named_parameters())
        missing = sorted(set(actor_params) - set(reference_state))
        if missing:
            raise ValueError(f"reference actor state is missing parameters: {missing}")

        self.reference: dict[str, torch.Tensor] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, parameter in actor_params.items():
            reference = reference_state[name].detach().to(
                device=parameter.device, dtype=parameter.dtype
            )
            if reference.shape != parameter.shape:
                raise ValueError(
                    f"reference actor parameter shape mismatch for {name}: "
                    f"{tuple(reference.shape)} != {tuple(parameter.shape)}"
                )
            self.reference[name] = reference.clone()
            self.handles.append(
                parameter.register_hook(
                    lambda gradient, p=parameter, ref=self.reference[name]: (
                        gradient + self.coefficient * (p.detach() - ref)
                    )
                )
            )

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        square_error = torch.zeros((), device=next(self.actor.parameters()).device)
        square_reference = torch.zeros_like(square_error)
        count = 0
        for name, parameter in self.actor.named_parameters():
            reference = self.reference[name]
            square_error += torch.sum(torch.square(parameter - reference))
            square_reference += torch.sum(torch.square(reference))
            count += parameter.numel()
        rms = torch.sqrt(square_error / max(count, 1))
        relative_l2 = torch.sqrt(square_error / square_reference.clamp_min(1.0e-12))
        return {
            "actor_anchor_rms": float(rms.item()),
            "actor_anchor_relative_l2": float(relative_l2.item()),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

