#!/usr/bin/env python3
"""Fine-tune YOLOv8n on SAHI-sliced train pseudo-labels; validate on matching eval tiles."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import torch
import yaml
from sahi.slicing import slice_image
from sahi.utils.coco import CocoAnnotation
from ultralytics import YOLO

from config import FRAMES_DIR, LABELS_DIR, PROJECT_ROOT, build_split_map

DATASET_DIR = PROJECT_ROOT / "outputs" / "dataset"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"

# --- training config ---
MODEL_NAME = "yolov8n.pt"
EPOCHS = 15  # PoC; increase for production runs
SLICE_SIZE = 512  # fixed train/eval tile size (see README — known mismatch vs autolabel)
OVERLAP_RATIO = 0.2
SLICE_OUT_EXT = ".jpg"  # SAHI defaults to PNG — huge on 4K; JPEG saves ~5–10× disk
MAX_EMPTY_SLICES_PER_FRAME = 2  # keep a few hard-negative tiles per frame
MIN_AREA_RATIO = 0.1  # SAHI: keep box if >=10% of original area visible in tile
BATCH_SIZE = 16
WORKERS = 4
CLASS_NAME = "vehicle"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sliced train dataset and fine-tune YOLOv8n on pseudo-labels.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build outputs/dataset (SAHI train + val slices).",
    )
    parser.add_argument(
        "--recreate-dataset",
        action="store_true",
        help="Force rebuild dataset even if outputs/dataset already exists.",
    )
    return parser.parse_args()


def iter_labeled_frames(split: str, split_map: dict[str, str]) -> list[tuple[str, Path, Path]]:
    """Return (clip_name, image_path, label_path) for frames that have a label file."""
    pairs: list[tuple[str, Path, Path]] = []
    label_root = LABELS_DIR / split
    if not label_root.is_dir():
        return pairs

    for clip_dir in sorted(label_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        clip_name = clip_dir.name
        if split_map.get(clip_name) != split:
            continue
        frame_dir = FRAMES_DIR / clip_name
        if not frame_dir.is_dir():
            print(f"  skip {clip_name}: frames dir missing")
            continue

        for label_path in sorted(clip_dir.glob("*.txt")):
            image_path = frame_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                continue
            pairs.append((clip_name, image_path, label_path))
    return pairs


def load_image_size(image_path: Path) -> tuple[int, int]:
    metadata_path = image_path.parent / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        return int(meta["width"]), int(meta["height"])

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    boxes: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        xmin = (xc - bw / 2) * img_w
        ymin = (yc - bh / 2) * img_h
        box_w = bw * img_w
        box_h = bh * img_h
        if box_w <= 0 or box_h <= 0:
            continue
        boxes.append((cls_id, xmin, ymin, box_w, box_h))
    return boxes


def yolo_boxes_to_coco(boxes: list[tuple[int, float, float, float, float]]) -> list[CocoAnnotation]:
    annotations: list[CocoAnnotation] = []
    for cls_id, xmin, ymin, box_w, box_h in boxes:
        annotations.append(
            CocoAnnotation.from_coco_bbox(
                bbox=[int(round(xmin)), int(round(ymin)), int(round(box_w)), int(round(box_h))],
                category_id=cls_id,
                category_name=CLASS_NAME,
            )
        )
    return annotations


def coco_bbox_to_yolo_line(bbox: list[float], slice_w: int, slice_h: int, cls_id: int = 0) -> str | None:
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return None
    xc = (x + w / 2) / slice_w
    yc = (y + h / 2) / slice_h
    nw = w / slice_w
    nh = h / slice_h
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and nw > 0 and nh > 0):
        return None
    return f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def slice_frames_to_dataset(
    frames: list[tuple[str, Path, Path]],
    images_dir: Path,
    labels_dir: Path,
    *,
    max_empty_slices_per_frame: int,
    progress_label: str,
) -> dict:
    """Slice labeled frames into fixed tiles; write YOLO images + labels."""
    clear_dir(images_dir)
    clear_dir(labels_dir)

    slices_total = 0
    slices_with_labels = 0
    slices_kept = 0
    slices_dropped = 0
    source_frames = 0

    for clip_name, image_path, label_path in frames:
        source_frames += 1
        img_w, img_h = load_image_size(image_path)
        coco_annotations = yolo_boxes_to_coco(parse_yolo_labels(label_path, img_w, img_h))
        output_stem = f"{clip_name}__{image_path.stem}"
        empty_kept_for_frame = 0

        result = slice_image(
            str(image_path),
            coco_annotation_list=coco_annotations or None,
            output_file_name=output_stem,
            output_dir=str(images_dir),
            slice_height=SLICE_SIZE,
            slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            auto_slice_resolution=False,
            min_area_ratio=MIN_AREA_RATIO,
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

            keep_slice = bool(label_lines)
            if not keep_slice and max_empty_slices_per_frame > 0:
                if empty_kept_for_frame < max_empty_slices_per_frame:
                    keep_slice = True
                    empty_kept_for_frame += 1

            if not keep_slice:
                slices_dropped += 1
                if image_file.exists():
                    image_file.unlink()
                continue

            slices_kept += 1
            label_out = labels_dir / f"{slice_name}.txt"
            label_out.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            if label_lines:
                slices_with_labels += 1

        if source_frames % 50 == 0:
            print(
                f"  {progress_label}: sliced {source_frames}/{len(frames)} frames -> "
                f"{slices_kept} kept ({slices_dropped} empty tiles dropped)"
            )

    return {
        "source_frames": source_frames,
        "slices_total": slices_total,
        "slices_kept": slices_kept,
        "slices_dropped": slices_dropped,
        "slices_with_labels": slices_with_labels,
        "slice_size": SLICE_SIZE,
        "slice_out_ext": SLICE_OUT_EXT,
        "overlap_ratio": OVERLAP_RATIO,
        "max_empty_slices_per_frame": max_empty_slices_per_frame,
    }


def prepare_sliced_train(split_map: dict[str, str]) -> dict:
    frames = iter_labeled_frames("train", split_map)
    if not frames:
        raise SystemExit("No train pseudo-labels found under labels/train/.")

    stats = slice_frames_to_dataset(
        frames,
        DATASET_DIR / "train" / "images",
        DATASET_DIR / "train" / "labels",
        max_empty_slices_per_frame=MAX_EMPTY_SLICES_PER_FRAME,
        progress_label="train",
    )
    print(
        f"Train slices: {stats['slices_kept']} tiles kept from {stats['source_frames']} frames "
        f"({stats['slices_with_labels']} with labels, {stats['slices_dropped']} empty dropped, "
        f"format={SLICE_OUT_EXT})"
    )
    return stats


def prepare_sliced_val(split_map: dict[str, str]) -> dict:
    """Val on the same 512×512 tiles as train (labeled tiles only)."""
    frames = iter_labeled_frames("eval", split_map)
    if not frames:
        raise SystemExit("No eval pseudo-labels found under labels/eval/.")

    stats = slice_frames_to_dataset(
        frames,
        DATASET_DIR / "val" / "images",
        DATASET_DIR / "val" / "labels",
        max_empty_slices_per_frame=0,
        progress_label="val",
    )
    print(
        f"Val slices: {stats['slices_kept']} labeled tiles from {stats['source_frames']} eval frames "
        f"({stats['slices_dropped']} empty dropped)"
    )
    return stats


def write_dataset_yaml() -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = DATASET_DIR / "data.yaml"
    payload = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: CLASS_NAME},
        "nc": 1,
        "slice_size": SLICE_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def prepare_dataset(split_map: dict[str, str], *, recreate: bool) -> Path:
    if recreate or not (DATASET_DIR / "train" / "images").exists():
        print("Preparing SAHI-sliced train set...")
        train_stats = prepare_sliced_train(split_map)
        print("Preparing SAHI-sliced val set (same tile size as train)...")
        val_stats = prepare_sliced_val(split_map)
        yaml_path = write_dataset_yaml()
        manifest = {
            "train": train_stats,
            "val": val_stats,
        }
        (DATASET_DIR / "prepare_stats.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(f"Dataset ready: {yaml_path}")
        return yaml_path

    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        write_dataset_yaml()
    print(f"Using existing dataset at {DATASET_DIR}")
    return yaml_path


def train_model(yaml_path: Path) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL_NAME)

    print(
        f"Training {MODEL_NAME} on {DEVICE}: "
        f"{EPOCHS} epochs, imgsz={SLICE_SIZE} (tiled, no per-epoch val)"
    )
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=SLICE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=str(RUNS_DIR),
        name="yolov8n_vehicle",
        exist_ok=True,
        pretrained=True,
        val=False,  # skip per-epoch val for speed; use evaluate.py for metrics
    )

    best_weights = RUNS_DIR / "yolov8n_vehicle" / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = RUNS_DIR / "yolov8n_vehicle" / "weights" / "last.pt"

    deliverable = PROJECT_ROOT / "checkpoints" / "yolov8n_vehicle_best.pt"
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, deliverable)
    print(f"Git-tracked copy: {deliverable}")
    return best_weights


def main() -> None:
    args = parse_args()
    split_map = build_split_map()
    if not split_map:
        raise SystemExit("No clips found in data/train or data/eval.")

    train_frames = iter_labeled_frames("train", split_map)
    eval_frames = iter_labeled_frames("eval", split_map)
    print(f"Labeled frames: train={len(train_frames)}, eval={len(eval_frames)}")
    if not train_frames:
        raise SystemExit("No train labels. Run autolabel_yworld.py on train clips first.")

    yaml_path = prepare_dataset(split_map, recreate=args.recreate_dataset)

    if args.prepare_only:
        print("Dataset prepared. Run without --prepare-only to train.")
        return

    weights = train_model(yaml_path)
    print(f"Done. Best weights: {weights}")


if __name__ == "__main__":
    main()
