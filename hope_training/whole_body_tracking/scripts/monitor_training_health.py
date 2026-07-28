#!/usr/bin/env python3
"""Monitor live rsl_rl TensorBoard runs for training regressions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import time

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCALARS = {
    "noise_std": "Policy/mean_noise_std",
    "learning_rate": "Loss/learning_rate",
    "actor_anchor_rms": "Loss/actor_anchor_rms",
    "contact_ema": "Metrics/racket_target/ability_contact_ema",
    "net_ema": "Metrics/racket_target/ability_net_ema",
    "success_ema": "Metrics/racket_target/ability_success_ema",
    "recovery_ema": "Metrics/racket_target/ability_recovery_ema",
    "safety_ema": "Metrics/racket_target/ability_safety_ema",
    "workspace_level": "Metrics/racket_target/workspace_level",
    "recovery_ready": "Metrics/racket_target/recovery_ready_score",
    "base_backward_velocity": "Metrics/racket_target/base_backward_velocity",
    "action_clamp_fraction": "Metrics/racket_target/action_clamp_fraction",
}


def _parse_run(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=/absolute/run/path")
    label, raw_path = value.split("=", 1)
    path = pathlib.Path(raw_path).expanduser().resolve()
    if not label or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return label, path


def _check_agent_config(run: pathlib.Path) -> list[str]:
    path = run / "params" / "agent.yaml"
    if not path.is_file():
        return ["agent_config_missing"]
    cfg = yaml.safe_load(path.read_text())
    algo = cfg["algorithm"]
    expected = {
        "learning_rate": 3.0e-5,
        "desired_kl": 0.0012,
        "entropy_coef": 0.0,
        "num_learning_epochs": 2,
    }
    failures = []
    for key, expected_value in expected.items():
        actual = algo.get(key)
        if actual != expected_value:
            failures.append(f"config_{key}={actual}_expected={expected_value}")
    return failures


def _read_scalars(run: pathlib.Path) -> tuple[int, dict[str, float]]:
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", ()))
    values: dict[str, float] = {}
    steps = []
    for name, tag in SCALARS.items():
        if tag not in available:
            continue
        event = accumulator.Scalars(tag)[-1]
        values[name] = float(event.value)
        steps.append(int(event.step))
    if not steps:
        raise RuntimeError("no monitored TensorBoard scalars found")
    return max(steps), values


def _classify(step: int, start_step: int, values: dict[str, float]) -> tuple[str, list[str]]:
    critical = []
    warnings = []
    if values.get("noise_std", 0.0) > 0.65:
        critical.append("exploration_noise_high")
    if values.get("learning_rate", 0.0) > 1.0e-4:
        critical.append("adaptive_learning_rate_high")
    if values.get("actor_anchor_rms", 0.0) > 0.002:
        critical.append("actor_anchor_drift")

    updates = step - start_step
    if updates >= 250:
        if values.get("contact_ema", 1.0) < 0.45:
            warnings.append("contact_ema_low")
        if values.get("success_ema", 1.0) < 0.20:
            warnings.append("success_ema_low")
        if values.get("recovery_ema", 1.0) < 0.70:
            warnings.append("recovery_ema_low")
    if values.get("safety_ema", 1.0) < 0.95:
        warnings.append("safety_ema_low")
    if values.get("action_clamp_fraction", 0.0) > 0.20:
        warnings.append("action_clamp_high")

    if critical:
        return "CRITICAL", critical + warnings
    if warnings:
        return "WARN", warnings
    return "HEALTHY", []


def _emit(record: dict, output: pathlib.Path | None) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run)
    parser.add_argument("--start-step", type=int, default=22750)
    parser.add_argument("--end-step", type=int, default=25750)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config_failures = {label: _check_agent_config(path) for label, path in args.run}
    last_steps = {label: -1 for label, _path in args.run}
    stagnant_since = {label: time.monotonic() for label, _path in args.run}

    while True:
        complete = True
        for label, path in args.run:
            try:
                step, values = _read_scalars(path)
                status, reasons = _classify(step, args.start_step, values)
                reasons = config_failures[label] + reasons
                if config_failures[label]:
                    status = "CRITICAL"
                if step != last_steps[label]:
                    last_steps[label] = step
                    stagnant_since[label] = time.monotonic()
                elif time.monotonic() - stagnant_since[label] > 600.0:
                    status = "CRITICAL"
                    reasons.append("training_stalled_over_600s")
                record = {
                    "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "run": label,
                    "step": step,
                    "status": status,
                    "reasons": reasons,
                    **{key: round(value, 6) for key, value in values.items()},
                }
                _emit(record, args.output)
                complete &= step >= args.end_step
            except Exception as exc:
                complete = False
                _emit(
                    {
                        "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                        "run": label,
                        "status": "READ_ERROR",
                        "reason": str(exc),
                    },
                    args.output,
                )
        if args.once or complete:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
