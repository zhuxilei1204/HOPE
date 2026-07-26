"""Compare real Motive ball tracks, planner rollout, and MuJoCo rollout.

The planner is more than a crossing-point predictor: internally it has a
no-spin trajectory model.  This script uses the same estimator state that the
planner would have online, then rolls the ball forward from that state and
compares future positions against the measured trajectory.

Three trajectories are compared from identical initial states:

* ``real``: measured Motive ball positions, linearly interpolated in time;
* ``planner``: HOPE planner analytic no-spin model with planner YAML overrides;
* ``shared_analytic``: same analytic model but with ``configs/ball_physics.yaml``
  values, which is useful when the Python ``mujoco`` package is unavailable;
* ``mujoco``: an optional minimal MuJoCo ball+table scene using the shared
  ``configs/ball_physics.yaml`` contact and drag values.

The MuJoCo comparison requires the ``mujoco`` Python package.  If it is not
installed, the script still writes the real/planner/shared-analytic comparison
and records that MuJoCo was unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_planner_on_segments import (  # noqa: E402
    REPO_ROOT,
    _crossings,
    _diagnose_geometry,
    _load_segment,
    _make_config,
    _manifest_metadata,
    _next_crossing,
    _planner_frame_issue,
    _segment_files,
)

PLANNER_SRC = REPO_ROOT / "hope_ws" / "src" / "hope_planner"
if str(PLANNER_SRC) not in sys.path:
    sys.path.insert(0, str(PLANNER_SRC))

from hope_planner.ball_state_estimator import BallStateEstimator  # noqa: E402
from hope_planner.constants import BallPhysics, load_ball_physics  # noqa: E402


@dataclass(frozen=True)
class RolloutCase:
    file: str
    sample_i: int
    t: np.ndarray
    pos: np.ndarray
    p_est: np.ndarray
    v_est: np.ndarray
    t_est: float
    actual_t: float
    actual_p: np.ndarray
    horizon_s: float


def _clone_physics(src: Any) -> BallPhysics:
    return BallPhysics(
        k=float(src.k),
        C_h=float(src.C_h),
        C_v=float(src.C_v),
        g=np.asarray(src.g, dtype=float).copy(),
        radius=float(src.radius),
        mass=float(src.mass),
    )


def _is_on_table(p: np.ndarray, physics: BallPhysics, table: Any) -> bool:
    r = physics.radius
    y_hi = table.y_max
    return bool(
        -r <= p[0] <= table.length + r
        and y_hi - table.width - r <= p[1] <= y_hi + r
    )


def _flight_acceleration(v: np.ndarray, physics: BallPhysics) -> np.ndarray:
    speed = float(np.linalg.norm(v))
    return -physics.k * speed * v + physics.g


def _apply_diagonal_bounce(v: np.ndarray, physics: BallPhysics) -> np.ndarray:
    return np.array([physics.C_h * v[0], physics.C_h * v[1], -physics.C_v * v[2]], dtype=float)


def _analytic_rollout(
    p0: np.ndarray,
    v0: np.ndarray,
    offsets: np.ndarray,
    physics: BallPhysics,
    cfg: Any,
    table: Any,
) -> np.ndarray:
    """Roll out the same diagonal no-spin ball model used by the planner."""
    offsets = np.asarray(offsets, dtype=float)
    out = np.empty((len(offsets), 3), dtype=float)
    if len(offsets) == 0:
        return out

    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    t = 0.0
    dt = float(cfg.dt_integrate)
    contact_z = physics.radius

    for j, target_t in enumerate(offsets):
        while t + 1e-12 < target_t:
            h = min(dt, target_t - t)
            a = _flight_acceleration(v, physics)
            v_new = v + a * h
            p_new = p + v * h + 0.5 * a * h * h

            if p_new[2] < contact_z and v_new[2] < 0.0:
                if _is_on_table(p_new, physics, table):
                    dz = p[2] - p_new[2]
                    frac = (p[2] - contact_z) / dz if dz > 1e-9 else 0.5
                    frac = float(np.clip(frac, 0.0, 1.0))
                    p_bounce = p + frac * (p_new - p)
                    p_bounce[2] = contact_z
                    v_at_bounce = v + a * (frac * h)
                    v_post = _apply_diagonal_bounce(v_at_bounce, physics)
                    rem = (1.0 - frac) * h
                    a_post = _flight_acceleration(v_post, physics)
                    p_new = p_bounce + v_post * rem + 0.5 * a_post * rem * rem
                    v_new = v_post + a_post * rem
                else:
                    p_new[2] = max(p_new[2], contact_z)

            p, v = p_new, v_new
            t += h
        out[j] = p
    return out


def _interp_real(t: np.ndarray, pos: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query_t, t, pos[:, axis]) for axis in range(3)])


def _collect_cases(
    files: list[Path],
    cfg: Any,
    eval_period_s: float,
    min_horizon_s: float,
    max_horizon_s: float,
) -> list[RolloutCase]:
    cases: list[RolloutCase] = []
    for path in files:
        t, pos = _load_segment(path)
        if len(t) < max(cfg.fit_window, 6):
            continue
        crossings = _crossings(t, pos, cfg.x_hit)
        if not crossings:
            continue

        est = BallStateEstimator(cfg)
        next_eval_t = -np.inf
        for i, (ti, pi) in enumerate(zip(t, pos)):
            ti = float(ti)
            est.push(ti, pi)
            if eval_period_s > 0.0 and ti < next_eval_t:
                continue
            next_eval_t = ti + eval_period_s
            if not est.ready:
                continue
            p_est, v_est, t_est = est.estimate()
            if v_est[0] >= 0.0:
                continue
            actual = _next_crossing(crossings, ti, min_horizon_s)
            if actual is None:
                continue
            horizon = float(actual["t"] - t_est)
            if horizon < min_horizon_s or horizon > max_horizon_s:
                continue
            cases.append(
                RolloutCase(
                    file=path.name,
                    sample_i=int(i),
                    t=t,
                    pos=pos,
                    p_est=p_est,
                    v_est=v_est,
                    t_est=float(t_est),
                    actual_t=float(actual["t"]),
                    actual_p=np.asarray(actual["p"], dtype=float),
                    horizon_s=horizon,
                )
            )
    return cases


def _stratified_sample(cases: list[RolloutCase], max_cases: int, seed: int) -> list[RolloutCase]:
    if max_cases <= 0 or len(cases) <= max_cases:
        return cases
    rng = np.random.default_rng(seed)
    horizons = np.asarray([c.horizon_s for c in cases])
    edges = np.array([0.0, 0.1, 0.2, 0.3, 0.5, np.inf])
    chosen: list[int] = []
    per_bin = max(1, max_cases // (len(edges) - 1))
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((horizons >= lo) & (horizons < hi))
        if len(idx) <= per_bin:
            chosen.extend(idx.tolist())
        else:
            chosen.extend(rng.choice(idx, size=per_bin, replace=False).tolist())
    if len(chosen) < max_cases:
        rest = np.setdiff1d(np.arange(len(cases)), np.asarray(chosen, dtype=int), assume_unique=False)
        add = min(max_cases - len(chosen), len(rest))
        if add:
            chosen.extend(rng.choice(rest, size=add, replace=False).tolist())
    chosen = sorted(set(chosen))[:max_cases]
    return [cases[i] for i in chosen]


def _dampratio_from_restitution(e: float) -> float:
    e = float(min(max(e, 1e-3), 0.999))
    le = math.log(e)
    return -le / math.sqrt(math.pi * math.pi + le * le)


class MinimalMujocoBallTable:
    """Small MuJoCo model in the HOPE table frame, used only for trajectory checks."""

    def __init__(
        self,
        physics_yaml: Path,
        table: Any,
        timestep: float,
        *,
        drag_k: float | None = None,
        table_restitution: float | None = None,
        table_friction: float | None = None,
        solref_time_s: float = 0.03,
    ) -> None:
        try:
            import yaml
            import mujoco
        except Exception as exc:  # pragma: no cover - exercised when mujoco is absent
            raise RuntimeError(f"mujoco unavailable: {exc}") from exc

        self.mj = mujoco
        data = yaml.safe_load(physics_yaml.read_text(encoding="utf-8"))
        contact = data.get("contact", {})
        ball = data.get("ball", {})
        drag = data.get("drag", {})
        table_cfg = data.get("table", {})
        gravity = float(data.get("gravity", 9.81))

        self.radius = float(ball.get("radius", 0.020))
        self.mass = float(ball.get("mass", 0.0027))
        self.drag_k = float(drag_k if drag_k is not None else drag.get("k", 0.1261))
        self.velocity_clip = float(drag.get("velocity_clip", 50.0))
        thickness = float(table_cfg.get("thickness", 0.05))
        e_table = float(
            table_restitution
            if table_restitution is not None
            else contact.get("table", {}).get("restitution", 0.9215)
        )
        mu_table = float(
            table_friction
            if table_friction is not None
            else contact.get("table", {}).get("dynamic_friction", 0.40)
        )

        y_center = float(table.y_max - table.width / 2.0)
        xml = f"""
<mujoco model="hope_ball_table">
  <option timestep="{float(timestep):.9f}" gravity="0 0 {-gravity:.9f}"/>
  <worldbody>
    <geom name="table_geom" type="box"
          pos="{table.length / 2.0:.9f} {y_center:.9f} {-thickness / 2.0:.9f}"
          size="{table.length / 2.0:.9f} {table.width / 2.0:.9f} {thickness / 2.0:.9f}"
          contype="1" conaffinity="1"/>
    <body name="ball" pos="1 0 1">
      <freejoint name="ball_free_joint"/>
      <geom name="ball_geom" type="sphere" size="{self.radius:.9f}" mass="{self.mass:.9f}"
            contype="1" conaffinity="1"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="ball_geom" geom2="table_geom"
          solref="{float(solref_time_s):.9f} {_dampratio_from_restitution(e_table):.9f}"
          friction="{mu_table:.9f} {mu_table:.9f} 0.005 0.0001 0.0001"
          condim="3"/>
  </contact>
</mujoco>
"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free_joint")
        self.qadr = int(self.model.jnt_qposadr[jid])
        self.vadr = int(self.model.jnt_dofadr[jid])
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.dt = float(self.model.opt.timestep)

    def rollout(self, p0: np.ndarray, v0: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        d = self.data
        d.qpos[self.qadr:self.qadr + 3] = np.asarray(p0, dtype=np.float64)
        d.qpos[self.qadr + 3:self.qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[self.vadr:self.vadr + 3] = np.asarray(v0, dtype=np.float64)
        d.qvel[self.vadr + 3:self.vadr + 6] = 0.0
        d.xfrc_applied[:] = 0.0
        self.mj.mj_forward(self.model, d)

        out = np.empty((len(offsets), 3), dtype=float)
        sim_t = 0.0
        prev_t = 0.0
        prev_p = d.qpos[self.qadr:self.qadr + 3].copy()
        cur_p = prev_p.copy()
        for j, target_t in enumerate(offsets):
            while sim_t + 1e-12 < target_t:
                prev_t = sim_t
                prev_p = d.qpos[self.qadr:self.qadr + 3].copy()
                v = d.qvel[self.vadr:self.vadr + 3]
                speed = min(float(np.linalg.norm(v)), self.velocity_clip)
                d.xfrc_applied[self.bid, :3] = -self.mass * self.drag_k * speed * v
                self.mj.mj_step(self.model, d)
                d.xfrc_applied[self.bid, :3] = 0.0
                sim_t += self.dt
                cur_p = d.qpos[self.qadr:self.qadr + 3].copy()
            if sim_t <= prev_t + 1e-12:
                out[j] = cur_p
            else:
                alpha = float(np.clip((target_t - prev_t) / (sim_t - prev_t), 0.0, 1.0))
                out[j] = prev_p + alpha * (cur_p - prev_p)
        return out


def _summary(values: list[float], scale: float = 1e3) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "median_mm": float(np.median(arr) * scale),
        "p90_mm": float(np.percentile(arr, 90) * scale),
        "mean_mm": float(np.mean(arr) * scale),
        "max_mm": float(np.max(arr) * scale),
    }


def _add_pair_metrics(store: dict[str, list[float]], a: np.ndarray, b: np.ndarray) -> None:
    diff = a - b
    store["point_xyz"].extend(np.linalg.norm(diff, axis=1).tolist())
    store["point_yz"].extend(np.linalg.norm(diff[:, 1:3], axis=1).tolist())
    store["endpoint_xyz"].append(float(np.linalg.norm(diff[-1])))
    store["endpoint_yz"].append(float(np.linalg.norm(diff[-1, 1:3])))


def compare(args: argparse.Namespace) -> dict[str, Any]:
    planner_physics, cfg, table = _make_config(args)
    shared_physics = load_ball_physics(args.physics_path)

    files = _segment_files(Path(args.segments))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no segment CSVs found under {args.segments}")

    geometry = _diagnose_geometry(files, table, planner_physics)
    frame_issue = _planner_frame_issue(geometry)
    if frame_issue and not args.allow_bad_geometry:
        raise SystemExit(
            f"{frame_issue} Use --allow-bad-geometry only for intentional diagnostics."
        )

    min_h = args.min_horizon_ms * 1e-3
    max_h = args.max_horizon_ms * 1e-3
    cfg.max_predict_time = max(float(cfg.max_predict_time), max_h)
    cases_full = _collect_cases(files, cfg, args.eval_period_s, min_h, max_h)
    cases = _stratified_sample(cases_full, args.max_cases, args.seed)
    if not cases:
        raise SystemExit("no comparable incoming-ball rollout cases")

    mujoco_runner = None
    mujoco_error = None
    if not args.no_mujoco:
        try:
            mujoco_drag_k = args.mujoco_drag_k
            mujoco_restitution = args.mujoco_table_restitution
            if args.mujoco_physics_source == "planner-yaml":
                if mujoco_drag_k is None:
                    mujoco_drag_k = float(planner_physics.k)
                if mujoco_restitution is None:
                    mujoco_restitution = float(planner_physics.C_v)
            mujoco_runner = MinimalMujocoBallTable(
                Path(args.physics_path),
                table,
                args.mujoco_timestep_s,
                drag_k=mujoco_drag_k,
                table_restitution=mujoco_restitution,
                table_friction=args.mujoco_table_friction,
                solref_time_s=args.mujoco_solref_time_s,
            )
        except RuntimeError as exc:
            mujoco_error = str(exc)
            if args.require_mujoco:
                raise SystemExit(mujoco_error)

    metrics: dict[str, dict[str, list[float]]] = {
        "planner_vs_real": {"point_xyz": [], "point_yz": [], "endpoint_xyz": [], "endpoint_yz": []},
        "shared_analytic_vs_real": {"point_xyz": [], "point_yz": [], "endpoint_xyz": [], "endpoint_yz": []},
        "planner_vs_shared_analytic": {"point_xyz": [], "point_yz": [], "endpoint_xyz": [], "endpoint_yz": []},
    }
    if mujoco_runner is not None:
        metrics["mujoco_vs_real"] = {"point_xyz": [], "point_yz": [], "endpoint_xyz": [], "endpoint_yz": []}
        metrics["planner_vs_mujoco"] = {"point_xyz": [], "point_yz": [], "endpoint_xyz": [], "endpoint_yz": []}
        metrics["shared_analytic_vs_mujoco"] = {
            "point_xyz": [], "point_yz": [], "endpoint_xyz": [], "endpoint_yz": []
        }

    row_samples = []
    for case in cases:
        offsets = np.arange(0.0, case.horizon_s + 0.5 * args.sample_dt_s, args.sample_dt_s)
        if offsets[-1] < case.horizon_s:
            offsets = np.append(offsets, case.horizon_s)
        else:
            offsets[-1] = case.horizon_s
        real = _interp_real(case.t, case.pos, case.t_est + offsets)
        planner = _analytic_rollout(case.p_est, case.v_est, offsets, planner_physics, cfg, table)
        shared = _analytic_rollout(case.p_est, case.v_est, offsets, _clone_physics(shared_physics), cfg, table)

        _add_pair_metrics(metrics["planner_vs_real"], planner, real)
        _add_pair_metrics(metrics["shared_analytic_vs_real"], shared, real)
        _add_pair_metrics(metrics["planner_vs_shared_analytic"], planner, shared)

        mujoco = None
        if mujoco_runner is not None:
            mujoco = mujoco_runner.rollout(case.p_est, case.v_est, offsets)
            _add_pair_metrics(metrics["mujoco_vs_real"], mujoco, real)
            _add_pair_metrics(metrics["planner_vs_mujoco"], planner, mujoco)
            _add_pair_metrics(metrics["shared_analytic_vs_mujoco"], shared, mujoco)

        if len(row_samples) < args.keep_rows:
            row = {
                "file": case.file,
                "sample_i": case.sample_i,
                "horizon_s": case.horizon_s,
                "p_est": [float(v) for v in case.p_est],
                "v_est": [float(v) for v in case.v_est],
                "real_endpoint": [float(v) for v in real[-1]],
                "planner_endpoint": [float(v) for v in planner[-1]],
                "shared_analytic_endpoint": [float(v) for v in shared[-1]],
            }
            if mujoco is not None:
                row["mujoco_endpoint"] = [float(v) for v in mujoco[-1]]
            row_samples.append(row)

    summaries = {
        name: {metric_name: _summary(values) for metric_name, values in metric.items()}
        for name, metric in metrics.items()
    }
    result = {
        "segments": str(Path(args.segments).resolve()),
        "planner_yaml": str(Path(args.planner_yaml).resolve()),
        "physics_path": str(Path(args.physics_path).resolve()),
        "geometry": geometry,
        "segments_manifest": _manifest_metadata(Path(args.segments)),
        "planner_frame_geometry_ok": frame_issue is None,
        "config": {
            "x_hit": float(cfg.x_hit),
            "fit_window": int(cfg.fit_window),
            "min_ready_samples": int(cfg.min_ready_samples),
            "bounce_center_z_max": float(cfg.bounce_center_z_max),
            "eval_period_s": float(args.eval_period_s),
            "sample_dt_s": float(args.sample_dt_s),
            "horizon_range_ms": [float(args.min_horizon_ms), float(args.max_horizon_ms)],
        },
        "physics": {
            "planner": {
                "drag_k": float(planner_physics.k),
                "table_C_h": float(planner_physics.C_h),
                "table_C_v": float(planner_physics.C_v),
            },
            "shared": {
                "drag_k": float(shared_physics.k),
                "table_C_h": float(shared_physics.C_h),
                "table_C_v": float(shared_physics.C_v),
            },
        },
        "mujoco": {
            "available": mujoco_runner is not None,
            "error": mujoco_error,
            "timestep_s": float(args.mujoco_timestep_s),
            "physics_source": args.mujoco_physics_source,
            "drag_k": (None if mujoco_runner is None else float(mujoco_runner.drag_k)),
            "table_restitution": (
                float(args.mujoco_table_restitution)
                if args.mujoco_table_restitution is not None
                else (
                    float(planner_physics.C_v)
                    if args.mujoco_physics_source == "planner-yaml"
                    else float(shared_physics.C_v)
                )
            ),
            "table_friction": args.mujoco_table_friction,
            "solref_time_s": float(args.mujoco_solref_time_s),
            "note": (
                "MuJoCo rollout is a minimal ball/table contact simulation. "
                "It uses no spin. The planner diagonal tangential coefficient C_h "
                "is not exactly equivalent to MuJoCo Coulomb friction."
            ),
        },
        "num_cases_full": int(len(cases_full)),
        "num_cases_used": int(len(cases)),
        "summary": summaries,
        "sample_rows": row_samples,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    printable = {
        "num_cases_full": result["num_cases_full"],
        "num_cases_used": result["num_cases_used"],
        "physics": result["physics"],
        "mujoco": result["mujoco"],
        "summary": result["summary"],
    }
    print(json.dumps(printable, indent=1))
    print(f"-> {out_path}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segments", help="directory of canonical planner-frame t,x,y,z segment CSVs")
    parser.add_argument(
        "--planner-yaml",
        default=str(REPO_ROOT / "hope_ws" / "src" / "hope_planner" / "config" / "hope_planner.yaml"),
    )
    parser.add_argument("--physics-path", default=str(REPO_ROOT / "configs" / "ball_physics.yaml"))
    parser.add_argument("--out-json", default=str(REPO_ROOT / "analysis" / "real_planner_mujoco_compare.json"))
    parser.add_argument("--x-hit", type=float, default=None)
    parser.add_argument("--table-y-max", type=float, default=None)
    parser.add_argument("--fit-window", type=int, default=None)
    parser.add_argument("--min-ready-samples", type=int, default=None)
    parser.add_argument("--min-horizon-ms", type=float, default=50.0)
    parser.add_argument("--max-horizon-ms", type=float, default=500.0)
    parser.add_argument("--eval-period-s", type=float, default=0.02)
    parser.add_argument("--sample-dt-s", type=float, default=0.02)
    parser.add_argument("--mujoco-timestep-s", type=float, default=0.001)
    parser.add_argument("--mujoco-physics-source", choices=("shared", "planner-yaml"), default="shared")
    parser.add_argument("--mujoco-drag-k", type=float, default=None)
    parser.add_argument("--mujoco-table-restitution", type=float, default=None)
    parser.add_argument("--mujoco-table-friction", type=float, default=None)
    parser.add_argument("--mujoco-solref-time-s", type=float, default=0.03)
    parser.add_argument("--max-cases", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-rows", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-mujoco", action="store_true", help="skip MuJoCo even if the package is installed")
    parser.add_argument("--require-mujoco", action="store_true", help="fail if the MuJoCo Python package is missing")
    parser.add_argument(
        "--allow-bad-geometry",
        action="store_true",
        help="do not fail when near-table points fall outside planner table bounds",
    )
    return parser


def main() -> None:
    compare(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
