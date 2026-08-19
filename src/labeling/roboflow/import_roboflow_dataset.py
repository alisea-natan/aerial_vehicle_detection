#!/usr/bin/env python3
"""Import Roboflow YOLOv8 export → Ultralytics training dataset (outputs/dataset/).

Roboflow images are already 1280×1280 (resize + offline augs). This script copies
them as full-frame samples (no SAHI re-tiling) and optionally adds object-centric
zoom crops for small vehicles so they appear larger during training.

Usage:
  python src/labeling/roboflow/import_roboflow_dataset.py --export-dir path/to/yolov8_export
  python src/labeling/roboflow/import_roboflow_dataset.py --export-dir path/to/yolov8_export --no-zoom-aug

Then train without rebuilding the dataset:
  python src/training/train.py
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

def _ensure_src_on_path() -> None:
    """Allow `python src/<pkg>/….py` without PYTHONPATH."""
    p = _Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "common").is_dir() and (p / "labeling").is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        p = p.parent

_ensure_src_on_path()

from common.config import PROJECT_ROOT

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import yaml


DATASET_DIR = PROJECT_ROOT / "outputs" / "dataset"
CLASS_NAME = "vehicle"
OUT_EXT = ".jpg"
IMGSZ = 1280

# Offline object-zoom: crop around small boxes and resize back to frame size.
ZOOM_FACTORS = (1.5, 2.0)
SMALL_BOX_AREA_RATIO = 0.0012  # normalized w*h; below → eligible for zoom aug
ZOOM_PADDING = 2.5  # crop spans max(box_w, box_h) * padding around object center

CLIP_ALIASES: dict[str, str] = {
    "8457857": "8457857-uhd_3840_2160_24fps",
    "266987": "266987",
    "13722965": "13722965_2160_3840_30fps",
    "5382494-uhd_3840_2160_24fps": "5382494-uhd_3840_2160_24fps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Roboflow export for YOLO11s training.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Unpacked Roboflow YOLOv8 export folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Ultralytics dataset root (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "--no-zoom-aug",
        action="store_true",
        help="Skip offline object-centric zoom augmentation on train split.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMGSZ,
        help=f"Recorded imgsz in data.yaml (default {IMGSZ}; images stay at export resolution).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output-dir before import.",
    )
    return parser.parse_args()


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def infer_clip_name(image_stem: str) -> str:
    """Map Roboflow filename stem → project clip id."""
    if image_stem.startswith("5382494-uhd_3840_2160_24fps"):
        return CLIP_ALIASES["5382494-uhd_3840_2160_24fps"]
    m = re.match(r"^(8457857)_", image_stem)
    if m:
        return CLIP_ALIASES["8457857"]
    m = re.match(r"^(266987)_", image_stem)
    if m:
        return CLIP_ALIASES["266987"]
    m = re.match(r"^(13722965)_", image_stem)
    if m:
        return CLIP_ALIASES["13722965"]
    if "__" in image_stem:
        return image_stem.split("__", 1)[0]
    return image_stem.split("_jpg", 1)[0]


def parse_yolo_label_lines(text: str) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        if bw <= 0 or bh <= 0:
            continue
        boxes.append((cls_id, xc, yc, bw, bh))
    return boxes


def format_yolo_line(cls_id: int, xc: float, yc: float, bw: float, bh: float) -> str:
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def boxes_to_lines(
    boxes: list[tuple[int, float, float, float, float]],
    crop_w: int,
    crop_h: int,
) -> list[str]:
    lines: list[str] = []
    for cls_id, xc, yc, bw, bh in boxes:
        px_cx = xc * crop_w
        px_cy = yc * crop_h
        px_w = bw * crop_w
        px_h = bh * crop_h
        if px_w <= 1 or px_h <= 1:
            continue
        line = format_yolo_line(cls_id, px_cx / crop_w, px_cy / crop_h, px_w / crop_w, px_h / crop_h)
        lines.append(line)
    return lines


def smallest_box_area(boxes: list[tuple[int, float, float, float, float]]) -> float:
    if not boxes:
        return 1.0
    return min(bw * bh for _, _, _, bw, bh in boxes)


def object_zoom_crop(
    image,
    boxes: list[tuple[int, float, float, float, float]],
    *,
    zoom: float,
) -> tuple[object, list[tuple[int, float, float, float, float]]] | tuple[None, None]:
    """Crop around the smallest object, zoom in, resize back to original size."""
    if not boxes or zoom <= 1.0:
        return None, None

    img_h, img_w = image.shape[:2]
    _, focus_xc, focus_yc, focus_bw, focus_bh = min(boxes, key=lambda b: b[3] * b[4])

    obj_w = focus_bw * img_w
    obj_h = focus_bh * img_h
    crop_size = max(obj_w, obj_h) * ZOOM_PADDING / zoom
    crop_size = max(crop_size, 32.0)
    crop_size = min(crop_size, float(min(img_w, img_h)))

    cx_px = focus_xc * img_w
    cy_px = focus_yc * img_h
    x1 = int(round(cx_px - crop_size / 2))
    y1 = int(round(cy_px - crop_size / 2))
    x2 = int(round(cx_px + crop_size / 2))
    y2 = int(round(cy_px + crop_size / 2))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None, None

    crop = image[y1:y2, x1:x2]
    out = cv2.resize(crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    remapped: list[tuple[int, float, float, float, float]] = []
    crop_w = x2 - x1
    crop_h = y2 - y1
    for cls_id, xc, yc, bw, bh in boxes:
        px_xc = xc * img_w
        px_yc = yc * img_h
        px_w = bw * img_w
        px_h = bh * img_h
        bx1 = px_xc - px_w / 2
        by1 = px_yc - px_h / 2
        bx2 = px_xc + px_w / 2
        by2 = px_yc + px_h / 2

        ix1 = max(bx1, float(x1))
        iy1 = max(by1, float(y1))
        ix2 = min(bx2, float(x2))
        iy2 = min(by2, float(y2))
        if ix2 - ix1 <= 1 or iy2 - iy1 <= 1:
            continue

        scale_x = img_w / crop_w
        scale_y = img_h / crop_h
        nx1 = (ix1 - x1) * scale_x
        ny1 = (iy1 - y1) * scale_y
        nx2 = (ix2 - x1) * scale_x
        ny2 = (iy2 - y1) * scale_y
        nw = nx2 - nx1
        nh = ny2 - ny1
        nxc = (nx1 + nx2) / 2 / img_w
        nyc = (ny1 + ny2) / 2 / img_h
        remapped.append((cls_id, nxc, nyc, nw / img_w, nh / img_h))

    if not remapped:
        return None, None
    return out, remapped


def copy_labeled_pair(
    image_path: Path,
    label_path: Path,
    images_dir: Path,
    labels_dir: Path,
    *,
    out_stem: str,
) -> tuple[bool, str | None, list[tuple[int, float, float, float, float]]]:
    """Copy one image+label; return (ok, clip_name, parsed_boxes)."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False, None, []

    text = label_path.read_text(encoding="utf-8")
    boxes = parse_yolo_label_lines(text)
    if not boxes:
        return False, None, []

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [skip] unreadable image: {image_path.name}")
        return False, None, []

    out_image = images_dir / f"{out_stem}{OUT_EXT}"
    out_label = labels_dir / f"{out_stem}.txt"
    if not cv2.imwrite(str(out_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Failed to write {out_image}")

    lines = boxes_to_lines(boxes, image.shape[1], image.shape[0])
    out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
    clip = infer_clip_name(image_path.stem)
    return True, clip, boxes


def import_split(
    export_dir: Path,
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    *,
    dest_prefix: str,
) -> dict:
    src_images = export_dir / split_name / "images"
    src_labels = export_dir / split_name / "labels"
    if not src_images.is_dir():
        raise SystemExit(f"Missing {src_images}")

    clear_dir(images_dir)
    clear_dir(labels_dir)

    copied = 0
    skipped_empty = 0
    clip_counts: dict[str, int] = {}

    for image_path in sorted(src_images.glob("*.jpg")):
        label_path = src_labels / f"{image_path.stem}.txt"
        out_stem = f"{dest_prefix}__{image_path.stem}"
        ok, clip, _ = copy_labeled_pair(
            image_path, label_path, images_dir, labels_dir, out_stem=out_stem
        )
        if ok:
            copied += 1
            clip_counts[clip or "unknown"] = clip_counts.get(clip or "unknown", 0) + 1
        else:
            skipped_empty += 1

    return {
        "split": split_name,
        "copied": copied,
        "skipped_empty": skipped_empty,
        "clip_counts": clip_counts,
    }


def add_object_zoom_augs(
    images_dir: Path,
    labels_dir: Path,
    *,
    zoom_factors: tuple[float, ...] = ZOOM_FACTORS,
) -> dict:
    """Add zoomed copies for images that contain at least one small vehicle."""
    created = 0
    eligible = 0
    for image_path in sorted(images_dir.glob(f"*{OUT_EXT}")):
        if "_zoom" in image_path.stem:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        boxes = parse_yolo_label_lines(label_path.read_text(encoding="utf-8"))
        if smallest_box_area(boxes) >= SMALL_BOX_AREA_RATIO:
            continue
        eligible += 1

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        for zoom in zoom_factors:
            zoom_tag = str(zoom).replace(".", "")
            out_stem = f"{image_path.stem}_zoom{zoom_tag}"
            if (images_dir / f"{out_stem}{OUT_EXT}").exists():
                continue
            cropped, remapped = object_zoom_crop(image, boxes, zoom=zoom)
            if cropped is None or not remapped:
                continue
            out_image = images_dir / f"{out_stem}{OUT_EXT}"
            out_label = labels_dir / f"{out_stem}.txt"
            cv2.imwrite(str(out_image), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            lines = boxes_to_lines(remapped, cropped.shape[1], cropped.shape[0])
            out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            created += 1

    return {"eligible_source_images": eligible, "zoom_copies_created": created, "zoom_factors": list(zoom_factors)}


def write_dataset_yaml(output_dir: Path, *, imgsz: int, train_stats: dict, val_stats: dict) -> Path:
    clip_group = {
        **train_stats.get("clip_counts", {}),
        **val_stats.get("clip_counts", {}),
    }
    # Mark all roboflow imports as full-frame A_close-style samples.
    clip_meta = {clip: False for clip in clip_group}  # uses_tiling=False

    payload = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: CLASS_NAME},
        "nc": 1,
        "imgsz": imgsz,
        "val_source": "roboflow_test_split",
        "clip_group": {k: "roboflow" for k in clip_group},
        "clip_uses_tiling": clip_meta,
    }
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not export_dir.is_dir():
        raise SystemExit(f"Roboflow export not found: {export_dir}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"Importing Roboflow export: {export_dir}")
    print(f"Writing Ultralytics dataset: {output_dir}")

    train_stats = import_split(
        export_dir,
        "train",
        output_dir / "train" / "images",
        output_dir / "train" / "labels",
        dest_prefix="rf_train",
    )
    print(
        f"Train: {train_stats['copied']} labeled images "
        f"({train_stats['skipped_empty']} empty skipped)"
    )

    # Roboflow export has test/ but no valid/ — use test as val.
    val_stats = import_split(
        export_dir,
        "test",
        output_dir / "val" / "images",
        output_dir / "val" / "labels",
        dest_prefix="rf_val",
    )
    print(
        f"Val (from Roboflow test): {val_stats['copied']} labeled images "
        f"({val_stats['skipped_empty']} empty skipped)"
    )

    zoom_stats: dict = {"enabled": False}
    if not args.no_zoom_aug:
        print("Adding object-centric zoom augmentations on train split...")
        zoom_stats = add_object_zoom_augs(
            output_dir / "train" / "images",
            output_dir / "train" / "labels",
        )
        zoom_stats["enabled"] = True
        print(
            f"Zoom aug: {zoom_stats['zoom_copies_created']} copies from "
            f"{zoom_stats['eligible_source_images']} small-object images "
            f"(factors={zoom_stats['zoom_factors']})"
        )

    yaml_path = write_dataset_yaml(output_dir, imgsz=args.imgsz, train_stats=train_stats, val_stats=val_stats)
    manifest = {
        "source": str(export_dir),
        "imgsz": args.imgsz,
        "val_source": "roboflow_test_split",
        "train": train_stats,
        "val": val_stats,
        "zoom_aug": zoom_stats,
    }
    (output_dir / "prepare_stats.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    n_train = len(list((output_dir / "train" / "images").glob(f"*{OUT_EXT}")))
    n_val = len(list((output_dir / "val" / "images").glob(f"*{OUT_EXT}")))
    print(f"\nDataset ready: {yaml_path}")
    print(f"  train images: {n_train}")
    print(f"  val images:   {n_val}")
    print("Train with: python src/training/train.py")


if __name__ == "__main__":
    main()
