"""Optimisation helpers: paths, yaml, image lists."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from common.config import PROJECT_ROOT

OPT_CFG = PROJECT_ROOT / "config" / "experiments" / "optimisation.yaml"


def ensure_src_on_path() -> None:
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "common").is_dir() and (p / "labeling").is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        p = p.parent


def load_opt_cfg() -> dict[str, Any]:
    import yaml

    if not OPT_CFG.is_file():
        raise SystemExit(f"Missing config: {OPT_CFG}")
    return yaml.safe_load(OPT_CFG.read_text(encoding="utf-8")) or {}


def defaults(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = cfg if cfg is not None else load_opt_cfg()
    return dict(raw.get("defaults") or {})


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.png"))
    return files


def train_image_dir(train_pack: Path) -> Path:
    nested = train_pack / "train" / "images"
    if nested.is_dir():
        return nested
    return train_pack / "images"


def artifact_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def locked_weights_path(cfg: dict[str, Any] | None = None) -> Path:
    return project_path(defaults(cfg)["weights"])


def weights_variant(weights: Path, cfg: dict[str, Any] | None = None) -> str:
    """Subdir name: prototype (locked) vs pruned variants vs other checkpoint stem."""
    locked = locked_weights_path(cfg).resolve()
    w = (weights if weights.is_absolute() else project_path(weights)).resolve()
    if w == locked:
        return "prototype"
    name = w.name.lower()
    if "unstructured" in name:
        return "pruned_unstructured"
    if "structured" in name:
        return "pruned_structured"
    if "pruned" in name:
        return "pruned"
    return w.stem.replace(".", "_")


def optimisation_runs_dir(step: str, weights: Path, cfg: dict[str, Any] | None = None) -> Path:
    return project_path(defaults(cfg)["runs_dir"]) / step / weights_variant(weights, cfg)


def optimisation_artifacts_dir(step: str, weights: Path, cfg: dict[str, Any] | None = None) -> Path:
    return project_path(defaults(cfg)["artifacts_dir"]) / step / weights_variant(weights, cfg)


PLATFORMS = ("raspberry", "android", "jetson")

# Track A1 export formats per deploy target
PLATFORM_A1_FORMATS: dict[str, tuple[str, ...]] = {
    "raspberry": ("openvino",),
    "android": ("tflite",),
    "jetson": ("onnx",),
}

# Track A2 quant cells per deploy target (jetson: ONNX from A1 only)
PLATFORM_A2_CELLS: dict[str, tuple[str, ...]] = {
    "raspberry": ("ov_fp32", "ov_fp16", "ov_int8"),
    "android": ("tflite_int8",),
    "jetson": (),
}

# Track A3 QAT format (jetson has no INT8 QAT path)
PLATFORM_A3_FORMAT: dict[str, str] = {
    "android": "tflite",
    "raspberry": "openvino",
}
