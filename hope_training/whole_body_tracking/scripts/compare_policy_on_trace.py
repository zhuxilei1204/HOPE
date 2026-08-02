#!/usr/bin/env python3
"""Replay one MuJoCo observation trace through multiple ONNX actors.

This counterfactual comparison holds observations fixed, so action differences
come from policy weights rather than from the policies visiting different
MuJoCo states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort


GROUPS = {
    "waist": np.arange(0, 3),
    "right_arm": np.arange(12, 19),
    "legs": np.arange(19, 31),
}


def _model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("model label must not be empty")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"ONNX file not found: {resolved}")
    return label, resolved


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {"n": 0, "mean": None, "rms": None, "max": None}
    return {
        "n": int(flat.size),
        "mean": float(np.mean(flat)),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "max": float(np.max(flat)),
    }


def _run(path: Path, observations: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.shape[-1] != observations.shape[1]:
        raise ValueError(
            f"{path}: expected observation dim {input_meta.shape[-1]}, "
            f"trace has {observations.shape[1]}"
        )
    output = session.run(
        [output_meta.name], {input_meta.name: observations.astype(np.float32)}
    )[0]
    if output.shape != (len(observations), 31):
        raise ValueError(f"{path}: unexpected actor output shape {output.shape}")
    return np.asarray(output, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_csv")
    parser.add_argument("--model", action="append", type=_model_arg, required=True)
    parser.add_argument("--near-impact-window", type=float, default=0.15)
    parser.add_argument("--recorded-label", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    with Path(args.trace_csv).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    obs_fields = [name for name in fieldnames if name.startswith("obs__")]
    if not rows or not obs_fields:
        raise ValueError("trace contains no rows or obs__ columns")
    observations = np.asarray(
        [[float(row[name]) for name in obs_fields] for row in rows],
        dtype=np.float32,
    )
    outputs = {label: _run(path, observations) for label, path in args.model}

    side = np.asarray([row.get("side", "") for row in rows])
    command_valid = np.asarray(
        [row.get("command_valid", "0").lower() in {"1", "1.0", "true"} for row in rows]
    )
    time_to_strike = np.asarray(
        [float(row.get("time_to_strike") or "nan") for row in rows]
    )
    near = command_valid & np.isfinite(time_to_strike) & (
        np.abs(time_to_strike) <= float(args.near_impact_window)
    )
    masks = {
        "all": near,
        "forehand": near & (side == "forehand"),
        "backhand": near & (side == "backhand"),
    }

    result: dict = {
        "trace_csv": str(Path(args.trace_csv).resolve()),
        "observation_dim": int(observations.shape[1]),
        "rows": len(rows),
        "near_impact_window_s": float(args.near_impact_window),
        "models": {},
        "counterfactual_deltas_from_first": {},
    }
    for label, actions in outputs.items():
        model_summary = {}
        for mask_name, mask in masks.items():
            group_summary = {"rows": int(np.sum(mask)), "action_groups": {}}
            for group_name, indices in GROUPS.items():
                values = np.abs(actions[mask][:, indices])
                group_summary["action_groups"][group_name] = _stats(values)
            model_summary[mask_name] = group_summary
        result["models"][label] = model_summary

    first_label = args.model[0][0]
    first = outputs[first_label]
    for label, actions in outputs.items():
        if label == first_label:
            continue
        delta_summary = {}
        delta = actions - first
        for mask_name, mask in masks.items():
            side_summary = {"rows": int(np.sum(mask)), "action_groups": {}}
            for group_name, indices in GROUPS.items():
                values = delta[mask][:, indices]
                side_summary["action_groups"][group_name] = {
                    "signed": _stats(values),
                    "absolute": _stats(np.abs(values)),
                }
            delta_summary[mask_name] = side_summary
        result["counterfactual_deltas_from_first"][label] = delta_summary

    if args.recorded_label:
        raw_fields = [name for name in fieldnames if name.startswith("raw_policy__")]
        if len(raw_fields) != 31:
            raise ValueError(f"expected 31 raw_policy__ columns, got {len(raw_fields)}")
        if args.recorded_label not in outputs:
            raise ValueError("--recorded-label must match one --model label")
        recorded = np.asarray(
            [[float(row[name]) for name in raw_fields] for row in rows], dtype=np.float64
        )
        error = outputs[args.recorded_label] - recorded
        result["recorded_replay_error"] = {
            "label": args.recorded_label,
            "max_abs": float(np.max(np.abs(error))),
            "rms": float(math.sqrt(np.mean(np.square(error)))),
        }

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
