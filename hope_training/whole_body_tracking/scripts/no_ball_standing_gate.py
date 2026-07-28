"""No-ball standing regression gate for exported HOPE ONNX policies.

Runs the MuJoCo deploy-style stand diagnostic in two no-ball modes:

* ready-only: no planner command is ever submitted, matching deploy idle/ready.
* fake-cycle: synthetic command cycles exercise ready -> swing -> recovery without a ball.

The script exits 0 only if the policy stays above the configured base-height and
tilt thresholds for the requested durations.  It is intentionally separate from
return-success eval so a policy that can hit once but cannot stay deploy-ready is
caught before long MuJoCo/real-planner evaluations.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from types import SimpleNamespace

from mujoco_stand_diagnostic import run as run_stand_diagnostic


def _diagnostic_args(args: argparse.Namespace, *, mode: str, seconds: float) -> SimpleNamespace:
    return SimpleNamespace(
        onnx=args.onnx,
        model_xml=args.model_xml,
        runtime_config=args.runtime_config,
        reference_dir=args.reference_dir,
        serve_manifest=args.serve_manifest,
        mode=mode,
        controller="policy",
        side_mode=args.side_mode,
        seconds=float(seconds),
        cycle_period_s=float(args.cycle_period_s),
        low_height=float(args.low_height),
        tilt_threshold=float(args.tilt_threshold),
        kp_scale=float(args.kp_scale),
        kd_scale=float(args.kd_scale),
        last_action_feedback_mode=args.last_action_feedback_mode,
        near_edge_x=None if args.near_edge_x is None else float(args.near_edge_x),
        seed=int(args.seed),
        stop_on_fall=True,
        view=False,
        record_video=None,
        video_width=640,
        video_height=480,
        video_fps=50,
        mujoco_gl=args.mujoco_gl,
        trace_csv=None,
        trace_joints=False,
        json_out=None,
    )


def _passed(result: dict, requested_seconds: float) -> bool:
    if result.get("fell_low") or result.get("fell_tilt"):
        return False
    return float(result.get("seconds_simulated", 0.0)) + 1.0e-6 >= float(requested_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--model-xml", default=None)
    parser.add_argument("--runtime-config", default=None)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--serve-manifest", default=None)
    parser.add_argument("--side-mode", choices=["mixed", "forehand", "backhand"], default="mixed")
    parser.add_argument("--seconds-ready", type=float, default=60.0)
    parser.add_argument("--seconds-fake-cycle", type=float, default=30.0)
    parser.add_argument("--skip-fake-cycle", action="store_true")
    parser.add_argument("--cycle-period-s", type=float, default=2.2)
    parser.add_argument("--low-height", type=float, default=0.55)
    parser.add_argument("--tilt-threshold", type=float, default=0.85)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument(
        "--last-action-feedback-mode",
        choices=["auto", "raw", "effective"],
        default="auto",
        help="Actor last-action feedback contract. auto prefers ONNX metadata, then runtime config.",
    )
    parser.add_argument(
        "--near-edge-x",
        type=float,
        default=None,
        help="Optional eval-only override; default uses configs/table_frame.yaml.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = {}

    ready = run_stand_diagnostic(_diagnostic_args(args, mode="ready-only", seconds=args.seconds_ready))
    results["ready_only"] = ready
    passed = _passed(ready, args.seconds_ready)

    if not args.skip_fake_cycle:
        fake = run_stand_diagnostic(_diagnostic_args(args, mode="fake-cycle", seconds=args.seconds_fake_cycle))
        results["fake_cycle"] = fake
        passed = passed and _passed(fake, args.seconds_fake_cycle)

    out = {
        "passed": bool(passed),
        "thresholds": {
            "low_height": float(args.low_height),
            "tilt_threshold": float(args.tilt_threshold),
            "seconds_ready": float(args.seconds_ready),
            "seconds_fake_cycle": None if args.skip_fake_cycle else float(args.seconds_fake_cycle),
        },
        "results": results,
    }
    print(json.dumps(out))
    if args.json_out:
        path = pathlib.Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
