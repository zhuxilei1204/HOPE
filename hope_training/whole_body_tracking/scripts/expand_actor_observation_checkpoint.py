#!/usr/bin/env python3
"""Expand only the actor's first observation layer with zero-initialized columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--old-dim", type=int, default=114)
    parser.add_argument("--new-dim", type=int, default=122)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    key = "actor.0.weight"
    old_weight = state[key]
    if tuple(old_weight.shape[1:]) != (args.old_dim,):
        raise ValueError(
            f"{key} has shape {tuple(old_weight.shape)}; expected (*, {args.old_dim})"
        )
    if args.new_dim <= args.old_dim:
        raise ValueError("--new-dim must be larger than --old-dim")

    new_weight = old_weight.new_zeros(old_weight.shape[0], args.new_dim)
    new_weight[:, : args.old_dim] = old_weight
    state[key] = new_weight
    checkpoint["optimizer_state_dict"] = {}
    infos = dict(checkpoint.get("infos") or {})
    infos["actor_observation_migration"] = {
        "source": str(source),
        "old_dim": int(args.old_dim),
        "new_dim": int(args.new_dim),
        "new_columns": "zero",
    }
    checkpoint["infos"] = infos

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    probe = torch.randn(32, args.old_dim, dtype=old_weight.dtype)
    old_out = probe @ old_weight.T
    expanded_probe = torch.cat(
        (probe, torch.randn(32, args.new_dim - args.old_dim, dtype=probe.dtype)),
        dim=-1,
    )
    new_out = expanded_probe @ new_weight.T
    max_error = float(torch.max(torch.abs(old_out - new_out)))
    print(f"wrote {output}")
    print(f"{key}: {tuple(old_weight.shape)} -> {tuple(new_weight.shape)}")
    print(f"first-layer equivalence max_abs_error={max_error:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
