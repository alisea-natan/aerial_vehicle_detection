#!/usr/bin/env python3
"""Build dataset packs under data/datasets/<variant_id>/.

Does not train — Round 1 is ``src/training/experiments/run_dataset_round.py``.

Default: CVAT ``labels/``, frame_step, imgsz 1024 (config/datasets/variants.yaml).

  python src/training/datasets/generate_variant.py --list
  python src/training/datasets/generate_variant.py --variant baseline_1
  python src/training/datasets/generate_variant.py --all
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path


def _ensure_src_on_path() -> None:
    p = _Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "common").is_dir() and (p / "labeling").is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        p = p.parent


_ensure_src_on_path()

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sahi.slicing import slice_image

from common.config import (
    FRAMES_DIR,
    PROJECT_ROOT,
    TrainGroupTiling,
    build_split_map,
    effective_slice_size,
    load_tiling_payload,
    resolve_train_group_tiling,
)
from training.datasets.specs import (
    VARIANTS_PATH,
    VariantSpec,
    list_variant_ids,
    load_variants_config,
    resolve_labels_root,
    resolve_variant,
)
from training.train import (
    SLICE_OUT_EXT,
    clear_dir,
    coco_bbox_to_yolo_line,
    iter_labeled_frames,
    load_image_size,
    parse_yolo_labels,
    split_train_frames_for_val,
    write_full_frame_sample,
    yolo_boxes_to_coco,
    _frame_index,
)


@dataclass(frozen=True)
class SlicePlan:
    group: str
    tile_size: int | None
    overlap: float
    uses_tiling: bool
    train_imgsz: int


def _plan_for_clip(clip_name: str, spec: VariantSpec, payload: dict) -> SlicePlan:
    base = resolve_train_group_tiling(clip_name, payload=payload)
    if spec.tiling_mode == "off":
        return SlicePlan(
            group=base.group,
            tile_size=None,
            overlap=0.0,
            uses_tiling=False,
            train_imgsz=spec.imgsz,
        )
    if spec.tiling_mode == "fixed":
        if spec.tile_size is None:
            raise SystemExit(f"{spec.id}: fixed tiling requires tile_size")
        overlap = 0.0 if spec.overlap is None else float(spec.overlap)
        return SlicePlan(
            group=base.group,
            tile_size=int(spec.tile_size),
            overlap=overlap,
            uses_tiling=True,
            train_imgsz=spec.imgsz,
        )
    # auto — train_groups, optional overlap override
    overlap = float(base.overlap)
    if spec.overlap_override is not None and base.uses_tiling:
        overlap = float(spec.overlap_override)
    return SlicePlan(
        group=base.group,
        tile_size=base.tile_size,
        overlap=overlap,
        uses_tiling=base.uses_tiling,
        train_imgsz=int(base.train_imgsz),
    )


def _filter_strided(
    frames: list[tuple[str, Path, Path]],
    stride: int,
) -> list[tuple[str, Path, Path]]:
    if stride <= 1:
        return frames
    out: list[tuple[str, Path, Path]] = []
    for clip_name, image_path, label_path in frames:
        idx = _frame_index(image_path.stem)
        if idx is None:
            continue
        if (idx - 1) % stride == 0:
            out.append((clip_name, image_path, label_path))
    return out


def _tile_origins(frame_w: int, frame_h: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    stride = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, max(1, frame_w - tile + 1), stride))
    ys = list(range(0, max(1, frame_h - tile + 1), stride))
    if not xs or xs[-1] != max(0, frame_w - tile):
        xs.append(max(0, frame_w - tile))
    if not ys or ys[-1] != max(0, frame_h - tile):
        ys.append(max(0, frame_h - tile))
    # unique preserve order
    seen: set[tuple[int, int]] = set()
    origins: list[tuple[int, int]] = []
    for y in ys:
        for x in xs:
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            origins.append(key)
    return origins


def _visible_ratio(
    box: tuple[float, float, float, float],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float:
    bx, by, bw, bh = box
    if bw <= 0 or bh <= 0:
        return 0.0
    ix1 = max(bx, float(x1))
    iy1 = max(by, float(y1))
    ix2 = min(bx + bw, float(x2))
    iy2 = min(by + bh, float(y2))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return (iw * ih) / (bw * bh)


def count_multi_tile_bboxes(
    frames: list[tuple[str, Path, Path]],
    spec: VariantSpec,
    payload: dict,
) -> dict:
    """Count GT boxes that land in 2+ tiles (overlap sensitivity)."""
    n_boxes = 0
    n_multi = 0
    per_clip: dict[str, dict[str, int]] = {}
    for clip_name, image_path, label_path in frames:
        plan = _plan_for_clip(clip_name, spec, payload)
        if not plan.uses_tiling or plan.tile_size is None:
            continue
        img_w, img_h = load_image_size(image_path)
        tile = min(int(plan.tile_size), img_w, img_h)
        boxes = parse_yolo_labels(label_path, img_w, img_h)
        if not boxes:
            continue
        origins = _tile_origins(img_w, img_h, tile, plan.overlap)
        clip_boxes = 0
        clip_multi = 0
        for _cls, xmin, ymin, bw, bh in boxes:
            hits = 0
            for x0, y0 in origins:
                ratio = _visible_ratio((xmin, ymin, bw, bh), x0, y0, x0 + tile, y0 + tile)
                if ratio >= spec.min_visible_ratio:
                    hits += 1
            clip_boxes += 1
            n_boxes += 1
            if hits >= 2:
                clip_multi += 1
                n_multi += 1
        stats = per_clip.setdefault(clip_name, {"boxes": 0, "multi_tile_boxes": 0})
        stats["boxes"] += clip_boxes
        stats["multi_tile_boxes"] += clip_multi
    return {
        "n_boxes_checked": n_boxes,
        "n_boxes_in_2plus_tiles": n_multi,
        "fraction_multi_tile": (n_multi / n_boxes) if n_boxes else 0.0,
        "per_clip": per_clip,
        "min_visible_ratio": spec.min_visible_ratio,
        "note": "Boxes with visible area ratio >= min_visible_ratio in 2+ tiles",
    }


def _apply_negative_quota(
    images_dir: Path,
    labels_dir: Path,
    *,
    negative_ratio: float,
    seed: int,
) -> dict:
    """Keep empty tiles up to negative_ratio * n_positive; delete the rest."""
    labeled: list[str] = []
    empty: list[str] = []
    for image_path in sorted(images_dir.glob(f"*{SLICE_OUT_EXT}")):
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
        if any(line.strip() for line in text.splitlines()):
            labeled.append(stem)
        else:
            empty.append(stem)

    n_pos = len(labeled)
    n_keep = int(math.floor(n_pos * max(0.0, negative_ratio))) if n_pos else 0
    rng = random.Random(seed)
    rng.shuffle(empty)
    keep = set(empty[:n_keep])
    removed = 0
    for stem in empty:
        if stem in keep:
            continue
        img = images_dir / f"{stem}{SLICE_OUT_EXT}"
        lab = labels_dir / f"{stem}.txt"
        if img.exists():
            img.unlink()
        if lab.exists():
            lab.unlink()
        removed += 1
    return {
        "positive_tiles": n_pos,
        "empty_before": len(empty),
        "empty_kept": len(keep),
        "empty_removed": removed,
        "negative_ratio": negative_ratio,
    }


def slice_variant_frames(
    frames: list[tuple[str, Path, Path]],
    images_dir: Path,
    labels_dir: Path,
    *,
    spec: VariantSpec,
    progress_label: str,
) -> dict:
    clear_dir(images_dir)
    clear_dir(labels_dir)
    payload = load_tiling_payload()

    slices_total = 0
    slices_kept = 0
    slices_with_labels = 0
    slices_dropped = 0
    empty_candidates = 0
    clip_plan: dict[str, dict] = {}

    # When keep_negative_tiles: keep all empties during slice, then quota-trim.
    keep_all_empty = bool(spec.keep_negative_tiles)

    for source_i, (clip_name, image_path, label_path) in enumerate(frames, start=1):
        img_w, img_h = load_image_size(image_path)
        plan = _plan_for_clip(clip_name, spec, payload)
        clip_plan[clip_name] = {
            "group": plan.group,
            "tile_size": plan.tile_size,
            "overlap": plan.overlap,
            "uses_tiling": plan.uses_tiling,
        }

        if not plan.uses_tiling:
            slices_total += 1
            kept, has_labels = write_full_frame_sample(
                clip_name=clip_name,
                image_path=image_path,
                label_path=label_path,
                images_dir=images_dir,
                labels_dir=labels_dir,
                img_w=img_w,
                img_h=img_h,
            )
            if kept:
                slices_kept += 1
                if has_labels:
                    slices_with_labels += 1
                else:
                    empty_candidates += 1
            else:
                slices_dropped += 1
            continue

        tiling = TrainGroupTiling(
            group=plan.group,
            tile_size=plan.tile_size,
            overlap=plan.overlap,
            train_imgsz=plan.train_imgsz,
            uses_tiling=True,
        )
        slice_size = effective_slice_size(tiling, img_w, img_h)
        coco_annotations = yolo_boxes_to_coco(parse_yolo_labels(label_path, img_w, img_h))
        output_stem = f"{clip_name}__{image_path.stem}"
        result = slice_image(
            str(image_path),
            coco_annotation_list=coco_annotations or None,
            output_file_name=output_stem,
            output_dir=str(images_dir),
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=plan.overlap,
            overlap_width_ratio=plan.overlap,
            auto_slice_resolution=False,
            min_area_ratio=spec.min_visible_ratio,
            out_ext=SLICE_OUT_EXT,
            verbose=False,
        )
        for sliced in result.sliced_image_list:
            slices_total += 1
            slice_name = Path(sliced.coco_image.file_name).stem
            image_file = images_dir / sliced.coco_image.file_name
            label_lines: list[str] = []
            for ann in sliced.coco_image.annotations:
                line = coco_bbox_to_yolo_line(
                    ann.bbox,
                    sliced.coco_image.width,
                    sliced.coco_image.height,
                    cls_id=ann.category_id,
                )
                if line:
                    label_lines.append(line)

            has_labels = bool(label_lines)
            keep_slice = has_labels or keep_all_empty
            if not keep_slice:
                slices_dropped += 1
                if image_file.exists():
                    image_file.unlink()
                continue

            slices_kept += 1
            if has_labels:
                slices_with_labels += 1
            else:
                empty_candidates += 1
            (labels_dir / f"{slice_name}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )

        if source_i % 50 == 0:
            print(
                f"  {progress_label}: {source_i}/{len(frames)} frames → "
                f"{slices_kept} kept ({slices_dropped} dropped)"
            )

    neg_info = None
    if keep_all_empty and spec.negative_ratio > 0:
        neg_info = _apply_negative_quota(
            images_dir,
            labels_dir,
            negative_ratio=spec.negative_ratio,
            seed=spec.seed,
        )
        # refresh kept counts from disk
        slices_kept = len(list(images_dir.glob(f"*{SLICE_OUT_EXT}")))
        slices_with_labels = 0
        for lab in labels_dir.glob("*.txt"):
            if any(line.strip() for line in lab.read_text(encoding="utf-8").splitlines()):
                slices_with_labels += 1

    return {
        "source_frames": len(frames),
        "slices_total": slices_total,
        "slices_kept": slices_kept,
        "slices_with_labels": slices_with_labels,
        "slices_dropped": slices_dropped,
        "empty_candidates_before_quota": empty_candidates,
        "negative_quota": neg_info,
        "clip_plan": clip_plan,
        "min_visible_ratio": spec.min_visible_ratio,
    }


def write_data_yaml(dataset_dir: Path, *, imgsz: int, train_stats: dict, val_stats: dict) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_dir / "data.yaml"
    payload = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "vehicle"},
        "imgsz": imgsz,
        "val_source": "holdout_from_train_videos",
        "clip_plan": {
            **(train_stats.get("clip_plan") or {}),
            **(val_stats.get("clip_plan") or {}),
        },
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def build_variant(spec: VariantSpec, *, recreate: bool = True) -> Path:
    out = spec.out_dir
    if spec.dataset_action == "reuse":
        src_id = spec.reuse_from or "baseline_1"
        src = out.parent / src_id
        if not (src / "data.yaml").is_file():
            raise SystemExit(
                f"{spec.id} reuses {src_id}, but {src} is missing. "
                f"Build it first: python src/training/datasets/generate_variant.py --variant {src_id}"
            )
        if out.exists() or out.is_symlink():
            if out.is_symlink() or out.is_file():
                out.unlink()
            else:
                shutil.rmtree(out)
        # Symlink pack so variant_5 shares tiles with baseline_1
        out.symlink_to(src.resolve(), target_is_directory=True)
        manifest = {
            "variant_id": spec.id,
            "description": spec.description,
            "dataset_action": "reuse",
            "reuse_from": src_id,
            "dataset_dir": str(out.relative_to(PROJECT_ROOT)),
            "train_augmentation": spec.train_augmentation,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "note": "Online aug only — tiles identical to reuse_from",
        }
        # Write manifest beside symlink target? Prefer a small sidecar next to link name.
        side = out.parent / f"{spec.id}.manifest.json"
        side.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Reused {src_id} → {out} (symlink); manifest {side.name}")
        return src / "data.yaml"

    if out.exists() and not recreate:
        yaml_path = out / "data.yaml"
        if yaml_path.is_file():
            print(f"Using existing {out}")
            return yaml_path
        raise SystemExit(f"{out} exists but has no data.yaml; pass recreate")

    split_map = build_split_map()
    frames = iter_labeled_frames(
        "train",
        split_map,
        frame_step_only=spec.frame_step_only,
        labels_root=spec.labels_root,
    )
    if not frames:
        raise SystemExit(f"No train labels under {spec.labels_root / 'train'}")

    if spec.sampling == "strided":
        before = len(frames)
        frames = _filter_strided(frames, spec.stride)
        print(f"Strided sampling stride={spec.stride}: {before} → {len(frames)} frames")
    elif spec.sampling != "full":
        raise SystemExit(f"Unknown sampling strategy: {spec.sampling}")

    train_frames, val_frames = split_train_frames_for_val(
        frames,
        val_fraction=spec.val_fraction,
        seed=spec.seed,
    )

    print(
        f"Building {spec.id}: train_frames={len(train_frames)}, val_frames={len(val_frames)}, "
        f"tiling={spec.tiling_mode}, frame_step_only={spec.frame_step_only}, "
        f"labels={spec.labels_root}, aug={spec.train_augmentation} (dataset has no online aug)"
    )
    if out.exists():
        shutil.rmtree(out)

    train_stats = slice_variant_frames(
        train_frames,
        out / "train" / "images",
        out / "train" / "labels",
        spec=spec,
        progress_label="train",
    )
    val_stats = slice_variant_frames(
        val_frames,
        out / "val" / "images",
        out / "val" / "labels",
        spec=spec,
        progress_label="val",
    )
    # Val: never keep negatives quota games beyond what slicing did with keep_all_empty
    # (already applied). OK.

    yaml_path = write_data_yaml(out, imgsz=spec.imgsz, train_stats=train_stats, val_stats=val_stats)

    multi_tile = None
    if spec.log_multi_tile_bboxes:
        payload = load_tiling_payload()
        multi_tile = count_multi_tile_bboxes(train_frames + val_frames, spec, payload)
        print(
            f"  multi-tile boxes: {multi_tile['n_boxes_in_2plus_tiles']}/"
            f"{multi_tile['n_boxes_checked']} "
            f"({100 * multi_tile['fraction_multi_tile']:.1f}%)"
        )

    manifest = {
        "variant_id": spec.id,
        "description": spec.description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(out.relative_to(PROJECT_ROOT)),
        "labels_root": str(spec.labels_root.relative_to(PROJECT_ROOT))
        if spec.labels_root.is_relative_to(PROJECT_ROOT)
        else str(spec.labels_root),
        "frame_step_only": spec.frame_step_only,
        "tiling": {
            "mode": spec.tiling_mode,
            "tile_size": spec.tile_size,
            "overlap": spec.overlap,
            "overlap_override": spec.overlap_override,
            "min_visible_ratio": spec.min_visible_ratio,
            "keep_negative_tiles": spec.keep_negative_tiles,
            "negative_ratio": spec.negative_ratio,
        },
        "train_augmentation": spec.train_augmentation,
        "train_augmentation_note": "Applied in dataset_round at train time; tiles on disk are unaugmented",
        "sampling": {
            "strategy": spec.sampling,
            "stride": spec.stride if spec.sampling == "strided" else None,
        },
        "imgsz": spec.imgsz,
        "val_fraction": spec.val_fraction,
        "seed": spec.seed,
        "train": train_stats,
        "val": val_stats,
        "multi_tile_bbox_stats": multi_tile,
        "config_source": str(VARIANTS_PATH.relative_to(PROJECT_ROOT)),
    }
    (out / "variant_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Done {spec.id}: train={train_stats['slices_kept']} "
        f"val={val_stats['slices_kept']} → {yaml_path}"
    )
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List variant ids and exit.")
    parser.add_argument("--variant", action="append", default=None, help="Variant id (repeatable).")
    parser.add_argument("--all", action="store_true", help="Build all variants (reuse last).")
    parser.add_argument(
        "--from-autolabel",
        action="store_true",
        help="Use outputs/autolabel/labels instead of CVAT labels/.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=None,
        help="Override label tree (must contain train/<clip>/*.txt). Default: labels/.",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Skip rebuild when data.yaml already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_variants_config()
    ids = list_variant_ids(cfg)
    labels_root = None
    if args.labels_root is not None:
        labels_root = resolve_labels_root(args.labels_root)
    elif args.from_autolabel:
        labels_root = resolve_labels_root(from_autolabel=True)
    if args.list:
        for vid in ids:
            spec = resolve_variant(vid, cfg, labels_root=labels_root)
            print(
                f"{vid}: {spec.description} [{spec.dataset_action}] "
                f"aug={spec.train_augmentation} frame_step={spec.frame_step_only}"
            )
        return

    selected = list(args.variant or [])
    if args.all:
        selected = ids
    if not selected:
        raise SystemExit("Pass --variant ID, --all, or --list")

    # Build order: dependencies first (baseline_1 before variant_5)
    ordered: list[str] = []
    for vid in ids:
        if vid in selected:
            ordered.append(vid)
    for vid in selected:
        if vid not in ordered:
            ordered.append(vid)

    recreate = not args.no_recreate
    for vid in ordered:
        spec = resolve_variant(vid, cfg, labels_root=labels_root)
        print(f"\n=== {vid} ===")
        build_variant(spec, recreate=recreate)


if __name__ == "__main__":
    main()
