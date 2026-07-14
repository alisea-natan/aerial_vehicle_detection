#!/usr/bin/env python3
"""Fine-tune YOLOv8n on SAHI-sliced train pseudo-labels; validate on matching eval tiles.

Per-clip scale_coeff in config/clip_tiling.json sets crop size so objects land near
~64 px after resize to TRAIN_IMGSZ (see config.slice_size_from_scale).
"""

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

from config import (
    FRAMES_DIR,
    LABELS_DIR,
    PROJECT_ROOT,
    TRAIN_IMGSZ,
    TRAIN_OVERLAP_RATIO,
    build_split_map,
    load_clip_tile_config,
    resolve_scale_coeff,
    slice_size_from_scale,
)

DATASET_DIR = PROJECT_ROOT / "outputs" / "dataset"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"

# --- training config (tuned for Apple Silicon MPS) ---
MODEL_NAME = "yolov8n.pt"
EPOCHS = 15  # PoC; increase for production runs
SLICE_OUT_EXT = ".jpg"  # SAHI defaults to PNG — huge on 4K; JPEG saves ~5–10× disk
MAX_EMPTY_SLICES_PER_FRAME = 2  # keep a few hard-negative tiles per frame
MIN_AREA_RATIO = 0.1  # SAHI: keep box if >=10% of original area visible in tile
CLASS_NAME = "vehicle"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# MPS: few dataloader workers avoid fork/hang; disk cache cuts JPEG decode cost.
# Batch is chosen from unified memory when --batch is omitted.
DEFAULT_WORKERS_MPS = 2
DEFAULT_WORKERS_CPU = 4


def system_ram_gb() -> float:
    try:
        import os

        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024**3)
    except Exception:
        return 16.0


def default_batch_size(imgsz: int) -> int:
    """Pick a stable MPS-friendly batch for yolov8n at the given imgsz."""
    ram = system_ram_gb()
    if imgsz >= 1280:
        if ram >= 32:
            return 12
        if ram >= 16:
            return 8
        return 4
    if imgsz >= 960:
        return 16 if ram >= 16 else 8
    return 32 if ram >= 16 else 16


def default_workers() -> int:
    return DEFAULT_WORKERS_MPS if DEVICE == "mps" else DEFAULT_WORKERS_CPU


def configure_mps_runtime() -> None:
    """Env + allocator knobs that reduce MPS crashes / stalls."""
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if DEVICE != "mps":
        return
    # Leave headroom for macOS + dataloader; avoids sudden process kills.
    try:
        torch.mps.set_per_process_memory_fraction(0.85)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def patch_ultralytics_mps_unique() -> None:
    """Avoid MPS bug: unique(return_counts=True) can overflow → negative dim in loss.

    https://github.com/ultralytics/ultralytics/issues/12999
    """
    if DEVICE != "mps":
        return

    import ultralytics.utils.loss as yolo_loss
    from ultralytics.utils.tal import xywh2xyxy

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        batch_idx = targets[:, 0].long()
        # Critical: run unique on CPU — MPS return_counts is unreliable.
        _, counts = batch_idx.detach().cpu().unique(return_counts=True)
        counts = counts.to(device=self.device, dtype=torch.int32)
        max_count = int(counts.max().item())
        if max_count < 0:
            # Defensive: should not happen after CPU unique; keep training alive.
            max_count = int(batch_idx.detach().cpu().bincount().max().item())
        out = torch.zeros(batch_size, max_count, ne - 1, device=self.device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    yolo_loss.v8DetectionLoss.preprocess = preprocess
    print("Applied MPS unique() workaround for Ultralytics detection loss")


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
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TRAIN_IMGSZ,
        help=f"YOLO imgsz (default {TRAIN_IMGSZ} from config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Training epochs (default {EPOCHS}).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (default: auto from RAM / imgsz, e.g. 8 on 16GB M1 @1280).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Dataloader workers (default: {DEFAULT_WORKERS_MPS} on MPS, {DEFAULT_WORKERS_CPU} else).",
    )
    parser.add_argument(
        "--cache",
        choices=("disk", "ram", "false"),
        default="disk" if DEVICE == "mps" else "false",
        help="Ultralytics image cache: disk (fast+safe on Mac), ram, or false.",
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
    clip_scales: dict[str, float],
    imgsz: int,
    max_empty_slices_per_frame: int,
    progress_label: str,
) -> dict:
    """Slice labeled frames with per-clip scale_coeff → slice size; write YOLO images + labels."""
    clear_dir(images_dir)
    clear_dir(labels_dir)

    slices_total = 0
    slices_with_labels = 0
    slices_kept = 0
    slices_dropped = 0
    source_frames = 0
    clip_slice_sizes: dict[str, int] = {}

    for clip_name, image_path, label_path in frames:
        source_frames += 1
        img_w, img_h = load_image_size(image_path)
        scale = clip_scales.get(clip_name, 1.0)
        slice_size = slice_size_from_scale(scale, img_w, img_h, imgsz=imgsz)
        clip_slice_sizes[clip_name] = slice_size

        coco_annotations = yolo_boxes_to_coco(parse_yolo_labels(label_path, img_w, img_h))
        output_stem = f"{clip_name}__{image_path.stem}"
        empty_kept_for_frame = 0

        result = slice_image(
            str(image_path),
            coco_annotation_list=coco_annotations or None,
            output_file_name=output_stem,
            output_dir=str(images_dir),
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=TRAIN_OVERLAP_RATIO,
            overlap_width_ratio=TRAIN_OVERLAP_RATIO,
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
        "imgsz": imgsz,
        "overlap_ratio": TRAIN_OVERLAP_RATIO,
        "slice_out_ext": SLICE_OUT_EXT,
        "max_empty_slices_per_frame": max_empty_slices_per_frame,
        "clip_scale_coeff": {k: clip_scales[k] for k in sorted(clip_slice_sizes)},
        "clip_slice_size": clip_slice_sizes,
    }


def build_clip_scales(split_map: dict[str, str]) -> dict[str, float]:
    tile_config = load_clip_tile_config()
    scales: dict[str, float] = {}
    for clip_name, split in split_map.items():
        if split not in ("train", "eval"):
            continue
        scales[clip_name] = resolve_scale_coeff(tile_config.get(clip_name))
    return scales


def prepare_sliced_train(split_map: dict[str, str], *, imgsz: int) -> dict:
    frames = iter_labeled_frames("train", split_map)
    if not frames:
        raise SystemExit("No train pseudo-labels found under labels/train/.")

    clip_scales = build_clip_scales(split_map)
    print("Per-clip train slice sizes (from scale_coeff):")
    seen = set()
    for clip_name, _, _ in frames:
        if clip_name in seen:
            continue
        seen.add(clip_name)
        meta = json.loads((FRAMES_DIR / clip_name / "metadata.json").read_text(encoding="utf-8"))
        w, h = int(meta["width"]), int(meta["height"])
        scale = clip_scales[clip_name]
        sl = slice_size_from_scale(scale, w, h, imgsz=imgsz)
        print(f"  {clip_name}: scale_coeff={scale:.2f} → slice={sl} (frame {w}x{h})")

    stats = slice_frames_to_dataset(
        frames,
        DATASET_DIR / "train" / "images",
        DATASET_DIR / "train" / "labels",
        clip_scales=clip_scales,
        imgsz=imgsz,
        max_empty_slices_per_frame=MAX_EMPTY_SLICES_PER_FRAME,
        progress_label="train",
    )
    print(
        f"Train slices: {stats['slices_kept']} tiles kept from {stats['source_frames']} frames "
        f"({stats['slices_with_labels']} with labels, {stats['slices_dropped']} empty dropped, "
        f"format={SLICE_OUT_EXT})"
    )
    return stats


def prepare_sliced_val(split_map: dict[str, str], *, imgsz: int) -> dict:
    """Val tiles use the same per-clip scale_coeff as train/eval inference."""
    frames = iter_labeled_frames("eval", split_map)
    if not frames:
        raise SystemExit("No eval pseudo-labels found under labels/eval/.")

    clip_scales = build_clip_scales(split_map)
    stats = slice_frames_to_dataset(
        frames,
        DATASET_DIR / "val" / "images",
        DATASET_DIR / "val" / "labels",
        clip_scales=clip_scales,
        imgsz=imgsz,
        max_empty_slices_per_frame=0,
        progress_label="val",
    )
    print(
        f"Val slices: {stats['slices_kept']} labeled tiles from {stats['source_frames']} eval frames "
        f"({stats['slices_dropped']} empty dropped)"
    )
    return stats


def write_dataset_yaml(imgsz: int, train_stats: dict, val_stats: dict) -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = DATASET_DIR / "data.yaml"
    payload = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: CLASS_NAME},
        "nc": 1,
        "imgsz": imgsz,
        "overlap_ratio": TRAIN_OVERLAP_RATIO,
        "clip_scale_coeff": {
            **train_stats.get("clip_scale_coeff", {}),
            **val_stats.get("clip_scale_coeff", {}),
        },
        "clip_slice_size": {
            **train_stats.get("clip_slice_size", {}),
            **val_stats.get("clip_slice_size", {}),
        },
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def prepare_dataset(split_map: dict[str, str], *, recreate: bool, imgsz: int) -> Path:
    if recreate or not (DATASET_DIR / "train" / "images").exists():
        print(f"Preparing SAHI-sliced train set (imgsz={imgsz}, per-clip scale_coeff)...")
        train_stats = prepare_sliced_train(split_map, imgsz=imgsz)
        print("Preparing SAHI-sliced val set (same per-clip scale as eval)...")
        val_stats = prepare_sliced_val(split_map, imgsz=imgsz)
        yaml_path = write_dataset_yaml(imgsz, train_stats, val_stats)
        manifest = {
            "imgsz": imgsz,
            "overlap_ratio": TRAIN_OVERLAP_RATIO,
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
        raise SystemExit(
            f"Dataset images exist but {yaml_path} is missing. "
            "Rebuild with: python src/train.py --recreate-dataset"
        )
    print(f"Using existing dataset at {DATASET_DIR}")
    return yaml_path


def train_model(
    yaml_path: Path,
    *,
    imgsz: int,
    epochs: int,
    batch: int,
    workers: int,
    cache: str | bool,
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    configure_mps_runtime()
    patch_ultralytics_mps_unique()

    cache_arg: str | bool
    if cache == "false":
        cache_arg = False
    elif cache == "ram":
        cache_arg = True
    else:
        cache_arg = "disk"

    batch_try = max(1, int(batch))
    last_error: Exception | None = None

    for attempt in range(4):
        if DEVICE == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        model = YOLO(MODEL_NAME)
        print(
            f"Training {MODEL_NAME} on {DEVICE}: "
            f"{epochs} epochs, imgsz={imgsz}, batch={batch_try}, workers={workers}, "
            f"cache={cache_arg} (attempt {attempt + 1}/4, no per-epoch val)"
        )
        try:
            model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_try,
                workers=workers,
                device=DEVICE,
                project=str(RUNS_DIR),
                name="yolov8n_vehicle",
                exist_ok=True,
                pretrained=True,
                val=False,  # metrics via evaluate.py
                amp=True,
                cache=cache_arg,
                plots=False,
                save_period=-1,
                # Slightly lighter aug → fewer MPS edge cases + faster steps
                close_mosaic=5,
                mixup=0.0,
                copy_paste=0.0,
            )
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
            msg = str(exc).lower()
            oom = (
                "out of memory" in msg
                or "oom" in msg
                or "memory" in msg
                or "mps backend out of memory" in msg
            )
            unique_bug = "non-negative" in msg or "negative dimension" in msg
            if (oom or unique_bug) and batch_try > 1:
                new_batch = max(1, batch_try // 2)
                print(
                    f"MPS/train fault at batch={batch_try} ({exc}); "
                    f"retrying with batch={new_batch}"
                )
                batch_try = new_batch
                continue
            raise

    if last_error is not None:
        raise last_error

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

    batch = args.batch if args.batch is not None else default_batch_size(args.imgsz)
    workers = args.workers if args.workers is not None else default_workers()
    print(
        f"Device={DEVICE}, RAM≈{system_ram_gb():.0f}GB → "
        f"batch={batch}, workers={workers}, cache={args.cache}, imgsz={args.imgsz}"
    )

    yaml_path = prepare_dataset(
        split_map,
        recreate=args.recreate_dataset,
        imgsz=args.imgsz,
    )

    if args.prepare_only:
        print("Dataset prepared. Run without --prepare-only to train.")
        return

    weights = train_model(
        yaml_path,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        workers=workers,
        cache=args.cache,
    )
    print(f"Done. Best weights: {weights}")


if __name__ == "__main__":
    main()
