#!/usr/bin/env python3
"""Convert A3 monolithic ONNX policies to RKNN models.

The script intentionally does not edit the runtime YAML by default. It prints
the generated `onnx.rknn_*_model_path` values after conversion so callers can
decide when to switch `onnx.backend` to `rknn`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

GEAR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GEAR_ROOT.parent
DEFAULT_RUNTIME_CFG = (
    GEAR_ROOT
    / "src/a3/a3_deploy_onnx_ref/config/a3_runtime_config.yaml"
)
DEFAULT_OUT_DIR = GEAR_ROOT / "assets/a3_runtime/rknn_models"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    onnx_key: str
    rknn_key: str
    expected_input_dim: int
    expected_action_dim: int = 29
    expected_input_name: str = "obs_dict"
    expected_output_name: str = "action"


MODEL_SPECS = {
    "a3": ModelSpec(
        label="a3",
        onnx_key="model_path",
        rknn_key="rknn_model_path",
        expected_input_dim=1570,
    ),
    "smpl": ModelSpec(
        label="smpl",
        onnx_key="smpl_model_path",
        rknn_key="smpl_rknn_model_path",
        expected_input_dim=1770,
    ),
    "a3_fast": ModelSpec(
        label="a3_fast",
        onnx_key="a3_fast_model_path",
        rknn_key="a3_fast_rknn_model_path",
        expected_input_dim=1570,
    ),
    "hope": ModelSpec(
        label="hope",
        onnx_key="hope_model_path",
        rknn_key="hope_rknn_model_path",
        expected_input_dim=111,
        expected_action_dim=31,
        expected_input_name="observation",
        expected_output_name="raw_action",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_relative_to(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def resolve_path(raw: str, runtime_cfg: Path) -> Path:
    p = Path(raw).expanduser()
    candidates = (
        [p]
        if p.is_absolute()
        else [REPO_ROOT / p, GEAR_ROOT / p, runtime_cfg.parent / p]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"{raw} does not exist; tried: {tried}")


def load_runtime_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if isinstance(cfg.get("onnx"), dict):
        mode = str(cfg["onnx"].get("mode") or "monolithic").lower().replace("-", "_")
        if mode in ("encoder_decoder", "encoderdecoder", "split"):
            raise ValueError("RKNN conversion supports monolithic A3 policies only")
    elif cfg.get("contract") != "hope_pingpong":
        raise ValueError(
            "runtime config is missing top-level onnx mapping and is not a "
            "hope_pingpong runtime config"
        )
    return cfg


def probe_onnx_schema(path: Path) -> dict:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required for schema validation: pip install onnx") from exc

    model = onnx.load(str(path))
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if len(inputs) != 1:
        raise ValueError(f"{path} must have exactly 1 input, got {len(inputs)}")
    if len(outputs) != 1:
        raise ValueError(f"{path} must have exactly 1 output, got {len(outputs)}")

    def tensor_shape(value_info) -> list[int | str]:
        dims: list[int | str] = []
        for dim in value_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param") and dim.dim_param:
                dims.append(str(dim.dim_param))
            else:
                dims.append("?")
        return dims

    return {
        "input_name": inputs[0].name,
        "input_shape": tensor_shape(inputs[0]),
        "output_name": outputs[0].name,
        "output_shape": tensor_shape(outputs[0]),
        "opset": [
            {"domain": op.domain or "ai.onnx", "version": int(op.version)}
            for op in model.opset_import
        ],
        "ir_version": int(model.ir_version),
    }


def validate_schema(spec: ModelSpec, schema: dict, path: Path) -> None:
    if schema["input_name"] != spec.expected_input_name:
        raise ValueError(
            f"{path} input name must be {spec.expected_input_name}, "
            f"got {schema['input_name']}"
        )
    if schema["output_name"] != spec.expected_output_name:
        raise ValueError(
            f"{path} output name must be {spec.expected_output_name}, "
            f"got {schema['output_name']}"
        )
    if not _batched_vector_shape_matches(
        schema["input_shape"], spec.expected_input_dim
    ):
        raise ValueError(
            f"{path} input shape must be [1, {spec.expected_input_dim}] "
            f"or dynamic batch x {spec.expected_input_dim}, "
            f"got {schema['input_shape']}"
        )
    if not _batched_vector_shape_matches(
        schema["output_shape"], spec.expected_action_dim
    ):
        raise ValueError(
            f"{path} output shape must be [1, {spec.expected_action_dim}] "
            f"or dynamic batch x {spec.expected_action_dim}, "
            f"got {schema['output_shape']}"
        )


def _batched_vector_shape_matches(shape: list[int | str], width: int) -> bool:
    if len(shape) != 2:
        return False
    batch_dim, feature_dim = shape
    if feature_dim != width:
        return False
    return batch_dim == 1 or isinstance(batch_dim, str)


def ensure_onnx_mapping_compat() -> None:
    """Provide onnx.mapping for RKNN Toolkit2 when running with ONNX >= 1.20."""
    import onnx

    if hasattr(onnx, "mapping"):
        return
    try:
        from onnx import _mapping
    except ImportError as exc:
        raise RuntimeError(
            "RKNN Toolkit2 2.3.2 needs onnx.mapping; install onnx<=1.19 "
            "or use an ONNX build with onnx._mapping"
        ) from exc

    compat = types.SimpleNamespace(
        TENSOR_TYPE_TO_NP_TYPE={
            key: value.np_dtype for key, value in _mapping.TENSOR_TYPE_MAP.items()
        }
    )
    onnx.mapping = compat
    sys.modules["onnx.mapping"] = compat


def convert_one(
    spec: ModelSpec,
    onnx_path: Path,
    out_dir: Path,
    target_platform: str,
    overwrite: bool,
    verbose: bool,
) -> dict:
    ensure_onnx_mapping_compat()
    from rknn.api import RKNN

    schema = probe_onnx_schema(onnx_path)
    validate_schema(spec, schema, onnx_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    rknn_path = out_dir / f"{onnx_path.stem}.rknn"
    manifest_path = out_dir / f"{onnx_path.stem}.rknn.json"
    if rknn_path.exists() and not overwrite:
        raise FileExistsError(f"{rknn_path} exists; pass --overwrite to replace it")

    rknn = RKNN(verbose=verbose)
    try:
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with ret={ret}")
        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=[schema["input_name"]],
            input_size_list=[[1, spec.expected_input_dim]],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with ret={ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with ret={ret}")
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with ret={ret}")
    finally:
        rknn.release()

    manifest = {
        "label": spec.label,
        "target_platform": target_platform,
        "do_quantization": False,
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": sha256_file(onnx_path),
        "rknn": str(rknn_path),
        "rknn_sha256": sha256_file(rknn_path),
        "schema": schema,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def selected_specs(names: Iterable[str]) -> list[ModelSpec]:
    specs = []
    for name in names:
        try:
            specs.append(MODEL_SPECS[name])
        except KeyError as exc:
            raise ValueError(f"unknown model '{name}'") from exc
    return specs


def model_path_from_cfg(cfg: dict, spec: ModelSpec) -> str | None:
    if spec.label == "hope" and cfg.get("contract") == "hope_pingpong":
        policy = cfg.get("policy") or {}
        if isinstance(policy, dict):
            raw = policy.get("onnx_path")
            if raw:
                return str(raw)
    onnx = cfg.get("onnx") or {}
    if isinstance(onnx, dict):
        raw = onnx.get(spec.onnx_key)
        if raw:
            return str(raw)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-cfg", type=Path, default=DEFAULT_RUNTIME_CFG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="Direct ONNX path override; requires exactly one selected --models entry.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["a3", "smpl", "a3_fast"],
        choices=sorted(MODEL_SPECS),
    )
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg_path = args.runtime_cfg.resolve()
    cfg = load_runtime_cfg(cfg_path)
    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = (GEAR_ROOT / out_dir).resolve()

    manifests = []
    specs = selected_specs(args.models)
    if args.onnx_path is not None and len(specs) != 1:
        parser.error("--onnx-path requires exactly one selected model")
    for spec in specs:
        if args.onnx_path is not None:
            onnx_path = args.onnx_path.expanduser().resolve()
            if not onnx_path.exists():
                raise FileNotFoundError(f"{onnx_path} does not exist")
        else:
            raw = model_path_from_cfg(cfg, spec)
            if not raw:
                if spec.label == "hope" and cfg.get("contract") == "hope_pingpong":
                    print("[skip] policy.onnx_path is not configured")
                else:
                    print(f"[skip] onnx.{spec.onnx_key} is not configured")
                continue
            onnx_path = resolve_path(str(raw), cfg_path)
        print(f"[convert] {spec.label}: {onnx_path}")
        manifest = convert_one(
            spec=spec,
            onnx_path=onnx_path,
            out_dir=out_dir,
            target_platform=args.target_platform,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        manifests.append((spec, manifest))
        print(f"[write] {manifest['rknn']}")
        print(f"[write] {manifest['rknn']}.json")

    if not manifests:
        print("no models converted", file=sys.stderr)
        return 2

    print("\n# Add these under `onnx:` when switching to RKNN:")
    print("backend: rknn")
    print("rknn_core_mask: auto")
    for spec, manifest in manifests:
        rknn_path = Path(manifest["rknn"])
        rel = path_relative_to(rknn_path, GEAR_ROOT)
        value = rel if rel is not None else str(rknn_path)
        print(f"{spec.rknn_key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
