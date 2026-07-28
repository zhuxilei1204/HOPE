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
        first_layer_input_exempt_start: int | None = None,
    ) -> None:
        self.actor = actor
        self.coefficient = float(coefficient)
        if self.coefficient <= 0.0:
            raise ValueError(f"actor anchor coefficient must be positive, got {coefficient}")

        actor_params = dict(actor.named_parameters())
        missing = sorted(set(actor_params) - set(reference_state))
        if missing:
            raise ValueError(f"reference actor state is missing parameters: {missing}")

        first_linear_weight_name: str | None = None
        if first_layer_input_exempt_start is not None:
            for module_name, module in actor.named_modules():
                if isinstance(module, nn.Linear):
                    first_linear_weight_name = (
                        f"{module_name}.weight" if module_name else "weight"
                    )
                    if not 0 <= first_layer_input_exempt_start <= module.in_features:
                        raise ValueError(
                            "first-layer input anchor exemption must be within "
                            f"[0, {module.in_features}], got {first_layer_input_exempt_start}"
                        )
                    break
            if first_linear_weight_name is None:
                raise ValueError(
                    "first-layer input anchor exemption requires an actor with nn.Linear"
                )

        self.reference: dict[str, torch.Tensor] = {}
        self.anchor_masks: dict[str, torch.Tensor] = {}
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
            mask = torch.ones_like(parameter)
            if name == first_linear_weight_name:
                mask[:, first_layer_input_exempt_start:] = 0.0
            self.anchor_masks[name] = mask
            self.handles.append(
                parameter.register_hook(
                    lambda gradient,
                    p=parameter,
                    ref=self.reference[name],
                    anchor_mask=self.anchor_masks[name]: (
                        gradient + self.coefficient * (p.detach() - ref) * anchor_mask
                    )
                )
            )

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        square_error = torch.zeros((), device=next(self.actor.parameters()).device)
        square_reference = torch.zeros_like(square_error)
        exempt_square_error = torch.zeros_like(square_error)
        count = torch.zeros_like(square_error)
        exempt_count = torch.zeros_like(square_error)
        for name, parameter in self.actor.named_parameters():
            reference = self.reference[name]
            anchor_mask = self.anchor_masks[name]
            exempt_mask = 1.0 - anchor_mask
            square_error += torch.sum(torch.square(parameter - reference) * anchor_mask)
            square_reference += torch.sum(torch.square(reference) * anchor_mask)
            exempt_square_error += torch.sum(
                torch.square(parameter - reference) * exempt_mask
            )
            count += torch.sum(anchor_mask)
            exempt_count += torch.sum(exempt_mask)
        rms = torch.sqrt(square_error / count.clamp_min(1.0))
        relative_l2 = torch.sqrt(square_error / square_reference.clamp_min(1.0e-12))
        exempt_rms = torch.sqrt(exempt_square_error / exempt_count.clamp_min(1.0))
        return {
            "actor_anchor_rms": float(rms.item()),
            "actor_anchor_relative_l2": float(relative_l2.item()),
            "actor_anchor_exempt_rms": float(exempt_rms.item()),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
