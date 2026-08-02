"""Replay HOPE's moving-racket impact model on rigid MuJoCo contacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class ContactSample:
    label: str
    trial: int
    success: bool
    miss_reason: str
    ball_pre_pos: np.ndarray
    ball_pre_vel: np.ndarray
    ball_out_vel: np.ndarray
    target_pos: np.ndarray
    contact_pos: np.ndarray
    racket_site_pos: np.ndarray
    racket_tick_vel: np.ndarray
    racket_site_vel: np.ndarray
    racket_point_vel: np.ndarray
    racket_normal: np.ndarray
    contact_normal: np.ndarray
    angular_velocity: np.ndarray


_VARIANTS = {
    "control_tick_site__racket_frame": ("racket_tick_vel", "racket_normal"),
    "exact_site__racket_frame": ("racket_site_vel", "racket_normal"),
    "exact_point__racket_frame": ("racket_point_vel", "racket_normal"),
    "exact_site__contact_normal": ("racket_site_vel", "contact_normal"),
    "exact_point__contact_normal": ("racket_point_vel", "contact_normal"),
}

_PARAMETER_SETS = {
    "training_current": {"restitution": 0.654, "tangent_retain": 0.85},
    "planner_v4_contact": {"restitution": 0.654, "tangent_retain": 0.48},
}


def _vector(row: dict[str, str], prefix: str) -> np.ndarray | None:
    try:
        value = np.asarray(
            [float(row[f"{prefix}_{axis}"]) for axis in "xyz"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.all(np.isfinite(value)) else None


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-9:
        raise ValueError("normal vector has zero magnitude")
    return np.asarray(value, dtype=np.float64) / norm


def load_contact_csv(label: str, path: str | pathlib.Path) -> list[ContactSample]:
    samples = []
    with pathlib.Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("contact_kind") != "real":
                continue
            values = {
                name: _vector(row, prefix)
                for name, prefix in (
                    ("ball_pre_pos", "ball_pre_pos"),
                    ("ball_pre_vel", "ball_pre_vel"),
                    ("ball_out_vel", "ball_separation_vel"),
                    ("target_pos", "target_pos"),
                    ("contact_pos", "contact_pos"),
                    ("racket_site_pos", "racket_site_pos_exact_pre"),
                    ("racket_tick_vel", "racket_pre_vel"),
                    ("racket_site_vel", "racket_site_vel_exact_pre"),
                    ("racket_point_vel", "racket_point_vel_exact_pre"),
                    ("racket_normal", "racket_normal"),
                    ("contact_normal", "mujoco_contact_normal"),
                    ("angular_velocity", "racket_ang_vel_exact_pre"),
                )
            }
            if any(value is None for value in values.values()):
                continue
            samples.append(
                ContactSample(
                    label=label,
                    trial=int(row["trial"]),
                    success=bool(int(row.get("success", "0") or 0)),
                    miss_reason=row.get("miss_reason", ""),
                    **values,
                )
            )
    return samples


def moving_racket_impact(
    ball_in: np.ndarray,
    racket_velocity: np.ndarray,
    normal: np.ndarray,
    *,
    restitution: float,
    tangent_retain: float,
) -> np.ndarray:
    normal = _unit(normal)
    relative_in = np.asarray(ball_in) - np.asarray(racket_velocity)
    relative_normal = float(np.dot(relative_in, normal))
    relative_tangent = relative_in - relative_normal * normal
    relative_out = (
        float(tangent_retain) * relative_tangent
        - float(restitution) * relative_normal * normal
    )
    return np.asarray(racket_velocity) + relative_out


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1.0e-9 or b_norm <= 1.0e-9:
        return float("nan")
    cosine = float(np.dot(a, b) / (a_norm * b_norm))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _unsigned_normal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between unoriented planes; ``n`` and ``-n`` are equivalent."""
    cosine = abs(float(np.dot(_unit(a), _unit(b))))
    return math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0))))


def _plane_offset_geometry(
    sample: ContactSample,
    position: np.ndarray,
) -> tuple[float, float]:
    offset = position - sample.racket_site_pos
    normal = _unit(sample.racket_normal)
    signed_normal_offset = float(np.dot(offset, normal))
    tangent_offset = offset - signed_normal_offset * normal
    return float(np.linalg.norm(tangent_offset)), abs(signed_normal_offset)


def _ball_center_geometry(sample: ContactSample) -> tuple[float, float]:
    return _plane_offset_geometry(sample, sample.ball_pre_pos)


def _contact_point_geometry(sample: ContactSample) -> tuple[float, float]:
    return _plane_offset_geometry(sample, sample.contact_pos)


def _tangent_component(value: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = _unit(normal)
    return value - float(np.dot(value, normal)) * normal


def target_execution_decomposition(samples: list[ContactSample]) -> dict:
    """Split physical center error into planner-target and actor-execution terms."""
    if not samples:
        return {"sample_count": 0}
    planner_tangent = []
    actor_tangent = []
    total_tangent = []
    cross_terms = []
    normal_offsets = []
    for sample in samples:
        normal = sample.racket_normal
        planner = _tangent_component(
            sample.ball_pre_pos - sample.target_pos,
            normal,
        )
        actor = _tangent_component(
            sample.target_pos - sample.racket_site_pos,
            normal,
        )
        total = _tangent_component(
            sample.ball_pre_pos - sample.racket_site_pos,
            normal,
        )
        planner_tangent.append(float(np.linalg.norm(planner)))
        actor_tangent.append(float(np.linalg.norm(actor)))
        total_tangent.append(float(np.linalg.norm(total)))
        cross_terms.append(float(2.0 * np.dot(planner, actor)))
        normal_offsets.append(
            abs(
                float(
                    np.dot(
                        sample.ball_pre_pos - sample.racket_site_pos,
                        _unit(normal),
                    )
                )
            )
        )
    return {
        "sample_count": len(samples),
        "success_fraction": float(np.mean([sample.success for sample in samples])),
        "planner_ball_tangent_error_mean_m": float(np.mean(planner_tangent)),
        "actor_target_tangent_error_mean_m": float(np.mean(actor_tangent)),
        "ball_racket_tangent_error_mean_m": float(np.mean(total_tangent)),
        "planner_actor_cross_term_mean_m2": float(np.mean(cross_terms)),
        "ball_racket_normal_offset_mean_m": float(np.mean(normal_offsets)),
    }


def _target_execution_buckets(samples: list[ContactSample]) -> dict:
    usable_radius = 0.061
    return {
        "all": target_execution_decomposition(samples),
        "usable_center_radius_le_0p061m": target_execution_decomposition(
            [
                sample
                for sample in samples
                if _ball_center_geometry(sample)[0] <= usable_radius
            ]
        ),
        "rim_radius_gt_0p061m": target_execution_decomposition(
            [
                sample
                for sample in samples
                if _ball_center_geometry(sample)[0] > usable_radius
            ]
        ),
        "success": target_execution_decomposition(
            [sample for sample in samples if sample.success]
        ),
        "failure": target_execution_decomposition(
            [sample for sample in samples if not sample.success]
        ),
    }


def evaluate_parameters(
    samples: list[ContactSample],
    *,
    velocity_field: str,
    normal_field: str,
    restitution: float,
    tangent_retain: float,
) -> dict:
    errors = []
    angles = []
    vx_errors = []
    speed_ratios = []
    closing = []
    for sample in samples:
        racket_velocity = getattr(sample, velocity_field)
        normal = _unit(getattr(sample, normal_field))
        relative_in = sample.ball_pre_vel - racket_velocity
        closing.append(float(-np.dot(relative_in, normal)))
        predicted = moving_racket_impact(
            sample.ball_pre_vel,
            racket_velocity,
            normal,
            restitution=restitution,
            tangent_retain=tangent_retain,
        )
        error = float(np.linalg.norm(predicted - sample.ball_out_vel))
        errors.append(error)
        angles.append(_angle_deg(predicted, sample.ball_out_vel))
        vx_errors.append(float(predicted[0] - sample.ball_out_vel[0]))
        speed_ratios.append(
            float(np.linalg.norm(predicted))
            / max(float(np.linalg.norm(sample.ball_out_vel)), 1.0e-9)
        )
    error_array = np.asarray(errors)
    angle_array = np.asarray(angles)
    closing_array = np.asarray(closing)
    return {
        "sample_count": len(samples),
        "restitution": float(restitution),
        "tangent_retain": float(tangent_retain),
        "vector_error_mean_mps": float(np.mean(error_array)),
        "vector_error_rmse_mps": float(np.sqrt(np.mean(error_array**2))),
        "vector_error_median_mps": float(np.median(error_array)),
        "vector_error_p90_mps": float(np.quantile(error_array, 0.90)),
        "direction_error_mean_deg": float(np.nanmean(angle_array)),
        "direction_error_median_deg": float(np.nanmedian(angle_array)),
        "vx_bias_mps": float(np.mean(vx_errors)),
        "speed_ratio_mean": float(np.mean(speed_ratios)),
        "closing_speed_mean_mps": float(np.mean(closing_array)),
        "closing_fraction": float(np.mean(closing_array > 0.0)),
    }


def fit_parameters(
    samples: list[ContactSample],
    *,
    velocity_field: str,
    normal_field: str,
) -> dict[str, float]:
    design_rows = []
    targets = []
    for sample in samples:
        racket_velocity = getattr(sample, velocity_field)
        normal = _unit(getattr(sample, normal_field))
        relative_in = sample.ball_pre_vel - racket_velocity
        relative_normal = float(np.dot(relative_in, normal))
        relative_tangent = relative_in - relative_normal * normal
        design_rows.append(
            np.stack((relative_tangent, -relative_normal * normal), axis=1)
        )
        targets.append(sample.ball_out_vel - racket_velocity)
    design = np.concatenate(design_rows, axis=0)
    target = np.concatenate(targets, axis=0)
    result = lsq_linear(design, target, bounds=([0.0, 0.0], [1.5, 1.5]))
    return {
        "tangent_retain": float(result.x[0]),
        "restitution": float(result.x[1]),
        "optimizer_cost": float(result.cost),
        "optimizer_success": bool(result.success),
    }


def _sample_diagnostics(samples: list[ContactSample]) -> dict:
    unsigned_normal_angles = [
        _unsigned_normal_angle_deg(sample.racket_normal, sample.contact_normal)
        for sample in samples
    ]
    ball_radial_offsets = [
        _ball_center_geometry(sample)[0] for sample in samples
    ]
    ball_normal_offsets = [
        _ball_center_geometry(sample)[1] for sample in samples
    ]
    contact_radial_offsets = [
        _contact_point_geometry(sample)[0] for sample in samples
    ]
    angular_point_speed = [
        float(np.linalg.norm(sample.racket_point_vel - sample.racket_site_vel))
        for sample in samples
    ]
    return {
        "sample_count": len(samples),
        "racket_vs_contact_normal_unsigned_angle_mean_deg": float(
            np.mean(unsigned_normal_angles)
        ),
        "racket_vs_contact_normal_unsigned_angle_median_deg": float(
            np.median(unsigned_normal_angles)
        ),
        "edge_normal_fraction_15deg": float(
            np.mean(np.asarray(unsigned_normal_angles) > 15.0)
        ),
        "ball_center_radial_offset_mean_m": float(
            np.mean(ball_radial_offsets)
        ),
        "ball_center_radial_offset_p90_m": float(
            np.quantile(ball_radial_offsets, 0.90)
        ),
        "ball_center_normal_offset_mean_m": float(
            np.mean(ball_normal_offsets)
        ),
        "contact_point_radial_offset_mean_m": float(
            np.mean(contact_radial_offsets)
        ),
        "angular_contact_point_speed_mean_mps": float(
            np.mean(angular_point_speed)
        ),
        "angular_contact_point_speed_p90_mps": float(
            np.quantile(angular_point_speed, 0.90)
        ),
    }


def _bucket_evaluation(samples: list[ContactSample]) -> dict:
    usable_radius = 0.061
    buckets = {
        "face_normal_le_15deg": [
            sample
            for sample in samples
            if _unsigned_normal_angle_deg(
                sample.racket_normal, sample.contact_normal
            )
            <= 15.0
        ],
        "edge_normal_gt_15deg": [
            sample
            for sample in samples
            if _unsigned_normal_angle_deg(
                sample.racket_normal, sample.contact_normal
            )
            > 15.0
        ],
        "usable_center_radius_le_0p061m": [
            sample
            for sample in samples
            if _ball_center_geometry(sample)[0] <= usable_radius
        ],
        "rim_radius_gt_0p061m": [
            sample
            for sample in samples
            if _ball_center_geometry(sample)[0] > usable_radius
        ],
    }
    result = {}
    for name, bucket in buckets.items():
        if not bucket:
            result[name] = {"sample_count": 0}
            continue
        result[name] = {
            "sample_count": len(bucket),
            "success_fraction": float(
                np.mean([sample.success for sample in bucket])
            ),
            "edge_normal_fraction_15deg": float(
                np.mean(
                    [
                        _unsigned_normal_angle_deg(
                            sample.racket_normal, sample.contact_normal
                        )
                        > 15.0
                        for sample in bucket
                    ]
                )
            ),
            "training_current_racket_frame": evaluate_parameters(
                bucket,
                velocity_field="racket_site_vel",
                normal_field="racket_normal",
                **_PARAMETER_SETS["training_current"],
            ),
            "training_current_contact_normal": evaluate_parameters(
                bucket,
                velocity_field="racket_site_vel",
                normal_field="contact_normal",
                **_PARAMETER_SETS["training_current"],
            ),
        }
    return result


def _radius_scan(samples: list[ContactSample]) -> list[dict]:
    result = []
    for threshold in (0.045, 0.050, 0.055, 0.060, 0.061, 0.065, 0.070):
        inside = [
            sample
            for sample in samples
            if _ball_center_geometry(sample)[0] <= threshold
        ]
        outside = [
            sample
            for sample in samples
            if _ball_center_geometry(sample)[0] > threshold
        ]
        result.append(
            {
                "radius_m": threshold,
                "inside_count": len(inside),
                "inside_success_fraction": (
                    float(np.mean([sample.success for sample in inside]))
                    if inside
                    else None
                ),
                "inside_edge_normal_fraction_15deg": (
                    float(
                        np.mean(
                            [
                                _unsigned_normal_angle_deg(
                                    sample.racket_normal,
                                    sample.contact_normal,
                                )
                                > 15.0
                                for sample in inside
                            ]
                        )
                    )
                    if inside
                    else None
                ),
                "outside_count": len(outside),
                "outside_success_fraction": (
                    float(np.mean([sample.success for sample in outside]))
                    if outside
                    else None
                ),
            }
        )
    return result


def replay(samples: list[ContactSample]) -> dict:
    labels = sorted({sample.label for sample in samples})
    result = {
        "schema_version": 1,
        "sample_diagnostics": _sample_diagnostics(samples),
        "by_label": {
            label: _sample_diagnostics(
                [sample for sample in samples if sample.label == label]
            )
            for label in labels
        },
        "contact_quality_buckets": {
            "combined": _bucket_evaluation(samples),
            **{
                label: _bucket_evaluation(
                    [sample for sample in samples if sample.label == label]
                )
                for label in labels
            },
        },
        "target_execution_decomposition": {
            "combined": _target_execution_buckets(samples),
            **{
                label: _target_execution_buckets(
                    [sample for sample in samples if sample.label == label]
                )
                for label in labels
            },
        },
        "usable_radius_scan": {
            "combined": _radius_scan(samples),
            **{
                label: _radius_scan(
                    [sample for sample in samples if sample.label == label]
                )
                for label in labels
            },
        },
        "variants": {},
    }
    for variant, (velocity_field, normal_field) in _VARIANTS.items():
        fitted = fit_parameters(
            samples,
            velocity_field=velocity_field,
            normal_field=normal_field,
        )
        evaluations = {
            name: evaluate_parameters(
                samples,
                velocity_field=velocity_field,
                normal_field=normal_field,
                **params,
            )
            for name, params in _PARAMETER_SETS.items()
        }
        evaluations["fitted_combined"] = evaluate_parameters(
            samples,
            velocity_field=velocity_field,
            normal_field=normal_field,
            restitution=fitted["restitution"],
            tangent_retain=fitted["tangent_retain"],
        )
        cross_policy = {}
        if len(labels) > 1:
            for train_label in labels:
                train_samples = [
                    sample
                    for sample in samples
                    if sample.label == train_label
                ]
                test_samples = [
                    sample
                    for sample in samples
                    if sample.label != train_label
                ]
                params = fit_parameters(
                    train_samples,
                    velocity_field=velocity_field,
                    normal_field=normal_field,
                )
                cross_policy[f"fit_{train_label}__test_other"] = {
                    **params,
                    "evaluation": evaluate_parameters(
                        test_samples,
                        velocity_field=velocity_field,
                        normal_field=normal_field,
                        restitution=params["restitution"],
                        tangent_retain=params["tangent_retain"],
                    ),
                }
        result["variants"][variant] = {
            "velocity_field": velocity_field,
            "normal_field": normal_field,
            "fit": fitted,
            "evaluations": evaluations,
            "cross_policy": cross_policy,
        }
    return result


def _markdown(result: dict, inputs: list[str]) -> str:
    lines = [
        "# Rigid-contact impact-model replay",
        "",
        "Inputs:",
        *[f"- `{value}`" for value in inputs],
        "",
        "Only first real contacts with a recorded separation velocity are used.",
        "",
        "## State diagnostics",
        "",
        "| Samples | Unsigned normal angle | Edge-normal fraction | Ball-center radial mean/p90 | Angular point-speed mean/p90 |",
        "|---:|---:|---:|---:|---:|",
    ]
    diag = result["sample_diagnostics"]
    lines.append(
        f"| {diag['sample_count']} | "
        f"{diag['racket_vs_contact_normal_unsigned_angle_mean_deg']:.2f} deg | "
        f"{diag['edge_normal_fraction_15deg']:.1%} | "
        f"{diag['ball_center_radial_offset_mean_m']:.3f}/"
        f"{diag['ball_center_radial_offset_p90_m']:.3f} m | "
        f"{diag['angular_contact_point_speed_mean_mps']:.3f}/"
        f"{diag['angular_contact_point_speed_p90_mps']:.3f} m/s |"
    )
    lines.extend(
        [
            "",
            "## Contact-quality buckets",
            "",
            "The usable-center threshold is racket radius 0.081 m minus ball radius 0.020 m.",
            "",
            "| Policy | Bucket | Samples | Success | Edge normals | Racket-frame error | Contact-normal error |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, buckets in result["contact_quality_buckets"].items():
        for bucket_name, values in buckets.items():
            if values["sample_count"] == 0:
                lines.append(
                    f"| `{label}` | `{bucket_name}` | 0 | - | - | - | - |"
                )
                continue
            racket_metrics = values["training_current_racket_frame"]
            contact_metrics = values["training_current_contact_normal"]
            lines.append(
                f"| `{label}` | `{bucket_name}` | {values['sample_count']} | "
                f"{values['success_fraction']:.1%} | "
                f"{values['edge_normal_fraction_15deg']:.1%} | "
                f"{racket_metrics['vector_error_mean_mps']:.3f} m/s, "
                f"{racket_metrics['direction_error_mean_deg']:.2f} deg | "
                f"{contact_metrics['vector_error_mean_mps']:.3f} m/s, "
                f"{contact_metrics['direction_error_mean_deg']:.2f} deg |"
            )
    lines.extend(
        [
            "",
            "## Planner-target and actor-execution decomposition",
            "",
            "All position errors are projected into the achieved racket tangent plane.",
            "The cross term is from `||ball-target + target-racket||^2`;",
            "a negative value means the two errors partially cancel.",
            "",
            "| Policy | Bucket | Samples | Success | Planner-ball | Actor-target | Ball-racket | Cross term |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, buckets in result["target_execution_decomposition"].items():
        for bucket_name, values in buckets.items():
            if values["sample_count"] == 0:
                lines.append(
                    f"| `{label}` | `{bucket_name}` | 0 | - | - | - | - | - |"
                )
                continue
            lines.append(
                f"| `{label}` | `{bucket_name}` | {values['sample_count']} | "
                f"{values['success_fraction']:.1%} | "
                f"{values['planner_ball_tangent_error_mean_m']:.4f} m | "
                f"{values['actor_target_tangent_error_mean_m']:.4f} m | "
                f"{values['ball_racket_tangent_error_mean_m']:.4f} m | "
                f"{values['planner_actor_cross_term_mean_m2']:+.5f} m^2 |"
            )
    lines.extend(
        [
            "",
            "## Model comparison",
            "",
            "| Variant | Parameters | Mean error | RMSE | Direction error | Closing fraction |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for variant, values in result["variants"].items():
        for name, metrics in values["evaluations"].items():
            lines.append(
                f"| `{variant}` | `{name}` "
                f"(e={metrics['restitution']:.3f}, "
                f"retain={metrics['tangent_retain']:.3f}) | "
                f"{metrics['vector_error_mean_mps']:.3f} m/s | "
                f"{metrics['vector_error_rmse_mps']:.3f} m/s | "
                f"{metrics['direction_error_mean_deg']:.2f} deg | "
                f"{metrics['closing_fraction']:.1%} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=CSV",
    )
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args()

    samples = []
    for specification in args.input:
        if "=" not in specification:
            raise ValueError(f"--input must be LABEL=CSV, got {specification!r}")
        label, path = specification.split("=", 1)
        samples.extend(load_contact_csv(label, path))
    if not samples:
        raise ValueError("no complete first-real-contact samples were loaded")

    result = replay(samples)
    result["inputs"] = list(args.input)
    json_path = pathlib.Path(args.json_out)
    markdown_path = pathlib.Path(args.markdown_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _markdown(result, list(args.input)),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
