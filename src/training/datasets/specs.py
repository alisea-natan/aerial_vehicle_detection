"""Dataset pack variant specs (config/datasets/variants.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from common.config import AUTOLABEL_LABELS_DIR, LABELS_DIR, PROJECT_ROOT, TRAIN_IMGSZ

VARIANTS_PATH = PROJECT_ROOT / "config" / "datasets" / "variants.yaml"


@dataclass(frozen=True)
class VariantSpec:
    id: str
    description: str
    dataset_action: str  # build | reuse
    reuse_from: str | None
    tiling_mode: str  # auto | off | fixed
    tile_size: int | None
    overlap: float | None
    overlap_override: float | None
    min_visible_ratio: float
    keep_negative_tiles: bool
    negative_ratio: float
    train_augmentation: str  # applied in dataset_round; tiles on disk are unaugmented
    sampling: str  # full | strided | clip_balanced
    stride: int
    log_multi_tile_bboxes: bool
    imgsz: int
    val_fraction: float
    seed: int
    frame_step_only: bool
    labels_root: Path
    out_dir: Path
    raw: dict[str, Any]


def load_variants_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or VARIANTS_PATH
    if not cfg_path.is_file():
        raise SystemExit(f"Missing dataset variants config: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def list_variant_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    data = cfg or load_variants_config()
    return list((data.get("variants") or {}).keys())


def resolve_labels_root(
    raw: str | Path | None = None,
    *,
    from_autolabel: bool = False,
) -> Path:
    """Resolve label tree (train|eval/<clip>/*.txt). Default = CVAT ``labels/``."""
    if from_autolabel and raw is None:
        return AUTOLABEL_LABELS_DIR
    if raw is None:
        return LABELS_DIR
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_variant(
    variant_id: str,
    cfg: dict[str, Any] | None = None,
    *,
    labels_root: Path | str | None = None,
    from_autolabel: bool = False,
) -> VariantSpec:
    data = cfg or load_variants_config()
    variants = data.get("variants") or {}
    if variant_id not in variants:
        known = ", ".join(variants) or "(none)"
        raise SystemExit(f"Unknown dataset variant {variant_id!r}. Known: {known}")

    defaults = data.get("defaults") or {}
    raw = dict(variants[variant_id])
    tiling = dict(raw.get("tiling") or {})
    sampling = dict(raw.get("sampling") or {})
    metrics = dict(raw.get("metrics") or {})

    out_root = Path(str(defaults.get("out_root") or "data/datasets"))
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    labels_override = labels_root if labels_root is not None else defaults.get("labels_root")
    labels_path = resolve_labels_root(
        labels_override,
        from_autolabel=from_autolabel and labels_root is None and labels_override is None,
    )

    raw_mode = tiling.get("mode")
    if raw_mode is False:
        mode = "off"
    elif raw_mode is True:
        mode = "on"
    elif raw_mode is None:
        mode = "auto"
    else:
        mode = str(raw_mode).lower()
    tile_size = tiling.get("tile_size", None)
    if tile_size is not None:
        tile_size = int(tile_size)
    overlap = tiling.get("overlap", None)
    if overlap is not None:
        overlap = float(overlap)
    overlap_override = tiling.get("overlap_override", None)
    if overlap_override is not None:
        overlap_override = float(overlap_override)

    # Prefer explicit train_augmentation; fall back to legacy augmentation.set
    aug = raw.get("train_augmentation")
    if aug is None:
        aug = (raw.get("augmentation") or {}).get("set")
    if aug is None:
        aug = defaults.get("train_augmentation") or "poc"

    return VariantSpec(
        id=variant_id,
        description=str(raw.get("description") or ""),
        dataset_action=str(raw.get("dataset") or "build").lower(),
        reuse_from=(str(raw["reuse_from"]) if raw.get("reuse_from") else None),
        tiling_mode=mode,
        tile_size=tile_size,
        overlap=overlap,
        overlap_override=overlap_override,
        min_visible_ratio=float(
            tiling.get("min_visible_ratio", defaults.get("min_visible_ratio", 0.3))
        ),
        keep_negative_tiles=bool(
            tiling.get(
                "keep_negative_tiles",
                defaults.get("keep_negative_tiles", False),
            )
        ),
        negative_ratio=float(tiling.get("negative_ratio") or 0.0),
        train_augmentation=str(aug),
        sampling=str(sampling.get("strategy") or "full").lower(),
        stride=max(1, int(sampling.get("stride") or 1)),
        log_multi_tile_bboxes=bool(metrics.get("log_multi_tile_bboxes", False)),
        imgsz=int(raw.get("imgsz", defaults.get("imgsz") or TRAIN_IMGSZ)),
        val_fraction=float(defaults.get("val_fraction") or 0.15),
        seed=int(defaults.get("seed") or 42),
        frame_step_only=bool(
            raw.get("frame_step_only", defaults.get("frame_step_only", True))
        ),
        labels_root=labels_path,
        out_dir=out_root / variant_id,
        raw=raw,
    )
