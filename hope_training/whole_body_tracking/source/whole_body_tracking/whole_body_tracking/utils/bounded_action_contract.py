"""Pure-Torch primitives for the bounded/effective HOPE action contract.

This module is intentionally independent of Isaac Lab. It defines the numeric
contract shared by a future PPO policy, simulator action term, ONNX exporter,
and deploy decoder. Importing it does not activate the new contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def stable_tanh_log_abs_det_jacobian(latent: torch.Tensor) -> torch.Tensor:
    """Return ``log(1 - tanh(latent)^2)`` without saturation underflow."""
    return 2.0 * (
        math.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent)
    )


def atanh_clamped(
    bounded_action: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Numerically invert tanh while keeping the result finite."""
    clipped = torch.clamp(bounded_action, -1.0 + epsilon, 1.0 - epsilon)
    return torch.atanh(clipped)


def squashed_gaussian_log_prob_from_latent(
    latent: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Log density of ``tanh(N(mean, std))`` evaluated at its latent sample."""
    normal = torch.distributions.Normal(mean, std)
    base_log_prob = normal.log_prob(latent)
    correction = stable_tanh_log_abs_det_jacobian(latent)
    return (base_log_prob - correction).sum(dim=-1)


def squashed_gaussian_log_prob(
    bounded_action: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Log density of a bounded action under a tanh-squashed Gaussian."""
    latent = atanh_clamped(bounded_action, epsilon=epsilon)
    return squashed_gaussian_log_prob_from_latent(latent, mean, std)


@dataclass(frozen=True)
class EncodedJointTarget:
    bounded_action: torch.Tensor
    latent: torch.Tensor
    feasible: torch.Tensor


class BoundedJointActionCodec(nn.Module):
    """Map policy actions to operational joint targets and effective feedback."""

    def __init__(
        self,
        *,
        operational_lower: torch.Tensor,
        operational_upper: torch.Tensor,
        mechanical_lower: torch.Tensor,
        mechanical_upper: torch.Tensor,
        default_q: torch.Tensor,
        action_scale: torch.Tensor,
        passive_mask: torch.Tensor | None = None,
        inverse_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        values = {
            "operational_lower": operational_lower,
            "operational_upper": operational_upper,
            "mechanical_lower": mechanical_lower,
            "mechanical_upper": mechanical_upper,
            "default_q": default_q,
            "action_scale": action_scale,
        }
        flattened = {
            name: torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            for name, value in values.items()
        }
        size = flattened["default_q"].numel()
        if any(value.numel() != size for value in flattened.values()):
            raise ValueError("all action-contract vectors must have the same size")
        if torch.any(flattened["operational_lower"] >= flattened["operational_upper"]):
            raise ValueError("operational lower bounds must be strictly below upper bounds")
        if torch.any(
            flattened["operational_lower"] <= flattened["mechanical_lower"]
        ) or torch.any(
            flattened["operational_upper"] >= flattened["mechanical_upper"]
        ):
            raise ValueError("operational bounds must be strictly inside mechanical bounds")
        if torch.any(flattened["action_scale"] == 0.0):
            raise ValueError("action_scale must be nonzero")
        if torch.any(
            flattened["default_q"] < flattened["operational_lower"]
        ) or torch.any(
            flattened["default_q"] > flattened["operational_upper"]
        ):
            raise ValueError("default_q must lie inside operational bounds")
        if passive_mask is None:
            passive = torch.zeros(size, dtype=torch.bool)
        else:
            passive = torch.as_tensor(passive_mask, dtype=torch.bool).reshape(-1)
            if passive.numel() != size:
                raise ValueError("passive_mask size does not match the joint vectors")
        for name, value in flattened.items():
            self.register_buffer(name, value)
        self.register_buffer("passive_mask", passive)
        self.inverse_epsilon = float(inverse_epsilon)

    @property
    def q_mid(self) -> torch.Tensor:
        return 0.5 * (self.operational_lower + self.operational_upper)

    @property
    def q_half(self) -> torch.Tensor:
        return 0.5 * (self.operational_upper - self.operational_lower)

    def decode_bounded(
        self, bounded_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode ``[-1, 1]`` action and report emergency mechanical clipping."""
        bounded = torch.clamp(bounded_action, -1.0, 1.0)
        q_operational = self.q_mid + self.q_half * bounded
        q_final = torch.clamp(
            q_operational, self.mechanical_lower, self.mechanical_upper
        )
        emergency_clip = torch.ne(q_final, q_operational)
        if torch.any(self.passive_mask):
            q_final = torch.where(self.passive_mask, self.default_q, q_final)
            emergency_clip = torch.where(
                self.passive_mask, False, emergency_clip
            )
        return q_final, emergency_clip

    def decode_latent(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bounded = torch.tanh(latent)
        q_final, emergency_clip = self.decode_bounded(bounded)
        return bounded, q_final, emergency_clip

    def effective_feedback(self, q_des_final: torch.Tensor) -> torch.Tensor:
        feedback = (q_des_final - self.default_q) / self.action_scale
        if torch.any(self.passive_mask):
            feedback = torch.where(self.passive_mask, 0.0, feedback)
        return feedback

    def encode_joint_target(self, q_des_final: torch.Tensor) -> EncodedJointTarget:
        """Encode a distillation target without silently projecting infeasible joints."""
        bounded = (q_des_final - self.q_mid) / self.q_half
        feasible_per_joint = (bounded >= -1.0) & (bounded <= 1.0)
        if torch.any(self.passive_mask):
            passive_ok = torch.isclose(
                q_des_final,
                self.default_q,
                atol=1.0e-6,
                rtol=0.0,
            )
            feasible_per_joint = torch.where(
                self.passive_mask, passive_ok, feasible_per_joint
            )
            bounded = torch.where(self.passive_mask, 0.0, bounded)
        feasible = torch.all(feasible_per_joint, dim=-1)
        latent = atanh_clamped(bounded, epsilon=self.inverse_epsilon)
        return EncodedJointTarget(
            bounded_action=bounded,
            latent=latent,
            feasible=feasible,
        )


class SquashedGaussianSampler(nn.Module):
    """Small policy-distribution primitive with corrected bounded-action density."""

    def __init__(self, minimum_std: float = 1.0e-5) -> None:
        super().__init__()
        self.minimum_std = float(minimum_std)

    def forward(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std = torch.clamp(std, min=self.minimum_std)
        latent = mean + std * noise
        action = torch.tanh(latent)
        log_prob = squashed_gaussian_log_prob_from_latent(
            latent, mean, std
        )
        return action, latent, log_prob
