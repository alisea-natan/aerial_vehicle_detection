#!/usr/bin/env python3
"""Staged fine-tune of YOLO11s on group-tiled train pseudo-labels.

Stage 1 freezes the COCO backbone (layers 0–10 → freeze=11) and warms up the head.
Stage 2 unfreezes all layers and continues from the Stage 1 weights at a lower LR.

Train + in-training val both come from data/train videos (frame holdout).
Eval clips are reserved for evaluate.py only — never written into outputs/dataset.
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

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import torch
import yaml
from sahi.slicing import slice_image
from sahi.utils.coco import CocoAnnotation
from ultralytics import YOLO

from common.config import (
    FRAMES_DIR,
    LABELS_DIR,
    PROJECT_ROOT,
    TRAIN_IMGSZ,
    build_split_map,
    clip_skip_reason,
    effective_slice_size,
    is_clip_skipped,
    load_clip_tile_config,
    load_tiling_payload,
    resolve_clip_tile_config,
    resolve_train_group_tiling,
)
from common.aug_config import train_aug_kwargs

DATASET_DIR = PROJECT_ROOT / "outputs" / "dataset"
BASELINE_DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "baseline_v1"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"

# --- training config (tuned for Apple Silicon MPS) ---
MODEL_NAME = "yolo11s.pt"  # Ultralytics asset name for YOLO11s
# YOLO11s backbone = model.0..model.10 (Conv…SPPF…C2PSA). freeze=N → indices 0..N-1.
FREEZE_BACKBONE = 11
WARMUP_EPOCHS = 5  # Stage 1: head-only
EPOCHS = 20  # Stage 2: full fine-tune (backbone + head)
STAGE1_LR0 = 0.01  # head adapts quickly on frozen features
STAGE2_LR0 = STAGE1_LR0 / 10  # gentle backbone adaptation
PATIENCE = 7  # early stop on val fitness (mAP composite) with no improvement
# Fast PoC schedule (--prototype): still runs Stage 2 with backbone unfrozen.
PROTOTYPE_WARMUP_EPOCHS = 2
PROTOTYPE_EPOCHS = 5
PROTOTYPE_PATIENCE = 3
SLICE_OUT_EXT = ".jpg"  # SAHI defaults to PNG — huge on 4K; JPEG saves ~5–10× disk
# MAX_EMPTY_SLICES_PER_FRAME = 2  # keep a few hard-negative tiles per frame
MAX_EMPTY_SLICES_PER_FRAME = 0  # temporary: drop all empty tiles (empty_kept=0)
# Train-group balance targets (train split only; val untouched).
A_CLOSE_TARGET_SHARE = 0.30  # oversample A_close toward ~25–35% of final train
A_CLOSE_SHARE_MIN = 0.25
A_CLOSE_SHARE_MAX = 0.35
BALANCE_GROUP_ORDER = ("A_close", "B_medium", "C_far")
ROTATION_AUG_ANGLES = (90, 180, 270)  # CW degrees; aerial top-down safe

MIN_AREA_RATIO = 0.1  # SAHI: keep box if >=10% of original area visible in tile
CLASS_NAME = "vehicle"
DEVICE = (
    "mps"
    if torch.backends.mps.is_built() and torch.backends.mps.is_available()
    else "cpu"
)
STAGE1_RUN_NAME = "yolo11s_vehicle_stage1"
STAGE2_RUN_NAME = "yolo11s_vehicle"
# Hold out this fraction of *train* frames for Ultralytics val (stratified by clip).
# Eval videos stay out of outputs/dataset entirely.
VAL_FRACTION = 0.15
VAL_SPLIT_SEED = 42

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
    """Pick a stable MPS-friendly batch for yolo11s at the given imgsz.

    Smaller than the old yolov8n defaults — 11s is ~3× params / more activations.
    """
    ram = system_ram_gb()
    if imgsz >= 1280:
        if ram >= 32:
            return 8
        if ram >= 16:
            return 4
        return 2
    if imgsz >= 960:
        return 8 if ram >= 16 else 4
    if imgsz >= 640:
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
        description="Prepare sliced train dataset and fine-tune YOLO11s on pseudo-labels.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build the dataset (train + holdout-val from train videos).",
    )
    parser.add_argument(
        "--recreate-dataset",
        action="store_true",
        help="Force rebuild dataset even if it already exists.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=f"Dataset output root (default {DATASET_DIR.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Baseline prepare preset: frame_step frames only, no group balance/rotations, "
            f"imgsz={TRAIN_IMGSZ}, out={BASELINE_DATASET_DIR.relative_to(PROJECT_ROOT)} "
            "(unless --dataset-dir / --imgsz override)."
        ),
    )
    parser.add_argument(
        "--frame-step-only",
        action="store_true",
        help="Keep only frames on each clip's frame_step from clip_tiling.json.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Skip train-group downsample / rotation oversample (no prepare-time augs).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=VAL_FRACTION,
        help=(
            f"Fraction of train frames held out for in-training val "
            f"(default {VAL_FRACTION}; stratified by clip). Eval clips are never used here."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TRAIN_IMGSZ,
        help=f"YOLO model imgsz (default {TRAIN_IMGSZ}). Crops still use train_groups tile_size.",
    )
    parser.add_argument(
        "--prototype",
        action="store_true",
        help=(
            "Fast PoC schedule: short Stage 1 (head) + Stage 2 (full model incl. backbone). "
            f"Sets --warmup-epochs {PROTOTYPE_WARMUP_EPOCHS}, --epochs {PROTOTYPE_EPOCHS}, "
            f"--patience {PROTOTYPE_PATIENCE} unless those flags are also passed."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Stage 2 (full fine-tune, backbone unfrozen) epochs (default {EPOCHS}).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=None,
        help=f"Stage 1 head-only epochs with frozen backbone (default {WARMUP_EPOCHS}).",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=FREEZE_BACKBONE,
        help=(
            f"Stage 1: freeze first N model layers (default {FREEZE_BACKBONE} = full "
            "YOLO11s backbone 0–10). Stage 2 always unfreezes (freeze=0)."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=f"Early-stop patience on val fitness (default {PATIENCE}; 0 disables).",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=STAGE1_LR0,
        help=f"Stage 1 initial LR (default {STAGE1_LR0}); Stage 2 uses lr0/10.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (default: auto from RAM / imgsz, e.g. 4 on 16GB M1 @1280 for 11s).",
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


def _frame_index(stem: str) -> int | None:
    """Parse leading frame index from stems like '000123'."""
    digits = []
    for ch in stem:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def iter_labeled_frames(
    split: str,
    split_map: dict[str, str],
    *,
    frame_step_only: bool = False,
    labels_root: Path | None = None,
) -> list[tuple[str, Path, Path]]:
    """Return (clip_name, image_path, label_path) for frames that have a label file."""
    pairs: list[tuple[str, Path, Path]] = []
    root = (labels_root or LABELS_DIR).resolve()
    label_root = root / split
    if not label_root.is_dir():
        return pairs

    tile_config = load_clip_tile_config() if frame_step_only else {}

    for clip_dir in sorted(label_root.iterdir()):
        if not clip_dir.is_dir():
            continue
        clip_name = clip_dir.name
        if split_map.get(clip_name) != split:
            continue
        if is_clip_skipped(clip_name):
            print(f"  skip {clip_name}: {clip_skip_reason(clip_name)}")
            continue
        frame_dir = FRAMES_DIR / clip_name
        if not frame_dir.is_dir():
            print(f"  skip {clip_name}: frames dir missing")
            continue

        step = 1
        if frame_step_only:
            cfg = resolve_clip_tile_config(clip_name, tile_config)
            raw_step = cfg.get("frame_step")
            step = max(1, int(raw_step)) if raw_step is not None else 1
            print(f"  {clip_name}: frame_step={step} (labels_root={root.relative_to(PROJECT_ROOT) if root.is_relative_to(PROJECT_ROOT) else root})")

        kept = 0
        for label_path in sorted(clip_dir.glob("*.txt")):
            if frame_step_only and step > 1:
                idx = _frame_index(label_path.stem)
                if idx is None or (idx - 1) % step != 0:
                    continue
            image_path = frame_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                continue
            pairs.append((clip_name, image_path, label_path))
            kept += 1
        if frame_step_only:
            print(f"    kept {kept} labeled frames on step")
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


def write_full_frame_sample(
    *,
    clip_name: str,
    image_path: Path,
    label_path: Path,
    images_dir: Path,
    labels_dir: Path,
    img_w: int,
    img_h: int,
) -> tuple[bool, bool]:
    """Copy one full frame + YOLO labels (A_close: no tiling). Returns (kept, has_labels)."""
    boxes = parse_yolo_labels(label_path, img_w, img_h)
    label_lines: list[str] = []
    for cls_id, xmin, ymin, box_w, box_h in boxes:
        line = coco_bbox_to_yolo_line([xmin, ymin, box_w, box_h], img_w, img_h, cls_id=cls_id)
        if line:
            label_lines.append(line)

    if not label_lines:
        return False, False

    stem = f"{clip_name}__{image_path.stem}"
    out_image = images_dir / f"{stem}{SLICE_OUT_EXT}"
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    if not cv2.imwrite(str(out_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Failed to write image: {out_image}")
    (labels_dir / f"{stem}.txt").write_text(
        "\n".join(label_lines) + "\n",
        encoding="utf-8",
    )
    return True, True


def slice_frames_to_dataset(
    frames: list[tuple[str, Path, Path]],
    images_dir: Path,
    labels_dir: Path,
    *,
    imgsz: int,
    max_empty_slices_per_frame: int,
    progress_label: str,
) -> dict:
    """Write YOLO images+labels using train_groups tile_size (or full frame if null)."""
    clear_dir(images_dir)
    clear_dir(labels_dir)

    tiling_payload = load_tiling_payload()
    slices_total = 0
    slices_with_labels = 0
    slices_kept = 0
    slices_dropped = 0
    source_frames = 0
    clip_slice_size: dict[str, int] = {}
    clip_overlap: dict[str, float] = {}
    clip_group: dict[str, str] = {}
    clip_train_imgsz: dict[str, int] = {}
    clip_uses_tiling: dict[str, bool] = {}
    group_produced: dict[str, int] = {}
    group_labeled: dict[str, int] = {}
    group_empty_kept: dict[str, int] = {}
    group_empty_removed: dict[str, int] = {}

    for clip_name, image_path, label_path in frames:
        source_frames += 1
        img_w, img_h = load_image_size(image_path)
        tiling = resolve_train_group_tiling(clip_name, payload=tiling_payload)
        group_name = tiling.group
        clip_group[clip_name] = group_name
        clip_train_imgsz[clip_name] = tiling.train_imgsz
        clip_uses_tiling[clip_name] = tiling.uses_tiling
        clip_overlap[clip_name] = tiling.overlap

        if not tiling.uses_tiling:
            # Full-frame sample: one image per labeled frame (no SAHI).
            clip_slice_size[clip_name] = min(img_w, img_h)
            slices_total += 1
            group_produced[group_name] = group_produced.get(group_name, 0) + 1
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
                    group_labeled[group_name] = group_labeled.get(group_name, 0) + 1
                else:
                    group_empty_kept[group_name] = group_empty_kept.get(group_name, 0) + 1
            else:
                slices_dropped += 1
                group_empty_removed[group_name] = group_empty_removed.get(group_name, 0) + 1
        else:
            slice_size = effective_slice_size(tiling, img_w, img_h)
            clip_slice_size[clip_name] = slice_size
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
                overlap_height_ratio=tiling.overlap,
                overlap_width_ratio=tiling.overlap,
                auto_slice_resolution=False,
                min_area_ratio=MIN_AREA_RATIO,
                out_ext=SLICE_OUT_EXT,
                verbose=False,
            )

            for sliced in result.sliced_image_list:
                slices_total += 1
                group_produced[group_name] = group_produced.get(group_name, 0) + 1
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
                    group_empty_removed[group_name] = group_empty_removed.get(group_name, 0) + 1
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
                    group_labeled[group_name] = group_labeled.get(group_name, 0) + 1
                else:
                    group_empty_kept[group_name] = group_empty_kept.get(group_name, 0) + 1

        if source_frames % 50 == 0:
            print(
                f"  {progress_label}: processed {source_frames}/{len(frames)} frames -> "
                f"{slices_kept} kept ({slices_dropped} empty dropped)"
            )

    empty_kept = slices_kept - slices_with_labels
    group_stats = {
        g: {
            "labeled_kept": int(group_labeled.get(g, 0)),
            "empty_kept": int(group_empty_kept.get(g, 0)),
            "empty_removed": int(group_empty_removed.get(g, 0)),
            "produced": int(group_produced.get(g, 0)),
        }
        for g in BALANCE_GROUP_ORDER
    }
    # Include any unexpected group keys.
    for g in sorted(set(group_produced) | set(group_labeled) | set(group_empty_kept)):
        if g not in group_stats:
            group_stats[g] = {
                "labeled_kept": int(group_labeled.get(g, 0)),
                "empty_kept": int(group_empty_kept.get(g, 0)),
                "empty_removed": int(group_empty_removed.get(g, 0)),
                "produced": int(group_produced.get(g, 0)),
            }

    print(f"\n[{progress_label}] per-group tile counts (after tiling / empty filter):")
    for g in BALANCE_GROUP_ORDER:
        gs = group_stats.get(g, {})
        print(
            f"  {g}: labeled_kept={gs.get('labeled_kept', 0)}, "
            f"empty_kept={gs.get('empty_kept', 0)}, "
            f"empty_removed={gs.get('empty_removed', 0)}"
        )

    return {
        "source_frames": source_frames,
        "slices_total": slices_total,
        "slices_kept": slices_kept,
        "slices_dropped": slices_dropped,
        "slices_with_labels": slices_with_labels,
        "empty_kept": empty_kept,
        "empty_removed": slices_dropped,
        "group_stats": group_stats,
        "imgsz": imgsz,
        "slice_out_ext": SLICE_OUT_EXT,
        "max_empty_slices_per_frame": max_empty_slices_per_frame,
        "clip_slice_size": clip_slice_size,
        "clip_overlap": clip_overlap,
        "clip_group": clip_group,
        "clip_train_imgsz": clip_train_imgsz,
        "clip_uses_tiling": clip_uses_tiling,
    }


def print_group_tiling_plan(frames: list[tuple[str, Path, Path]]) -> None:
    payload = load_tiling_payload()
    print("Per-clip train tiling (from train_groups):")
    seen: set[str] = set()
    for clip_name, _, _ in frames:
        if clip_name in seen:
            continue
        seen.add(clip_name)
        meta = json.loads((FRAMES_DIR / clip_name / "metadata.json").read_text(encoding="utf-8"))
        w, h = int(meta["width"]), int(meta["height"])
        tiling = resolve_train_group_tiling(clip_name, payload=payload)
        tile_desc = "full-frame" if not tiling.uses_tiling else f"tile={tiling.tile_size}"
        print(
            f"  {clip_name}: group={tiling.group}, {tile_desc}, "
            f"overlap={tiling.overlap}, preferred_imgsz={tiling.train_imgsz} "
            f"(frame {w}x{h})"
        )


def _clip_from_sample_stem(stem: str) -> str:
    return stem.split("__", 1)[0]


def _parse_tile_stem(stem: str) -> dict | None:
    """Parse SAHI stem `clip__frame_x1_y1_x2_y2` or full-frame `clip__frame`."""
    if "__" not in stem:
        return None
    clip, rest = stem.split("__", 1)
    parts = rest.split("_")
    if len(parts) >= 5 and all(p.isdigit() for p in parts[-4:]):
        x1, y1, x2, y2 = map(int, parts[-4:])
        frame = "_".join(parts[:-4])
        return {
            "clip": clip,
            "frame": frame,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "tiled": True,
        }
    if parts and parts[0].isdigit():
        return {"clip": clip, "frame": parts[0], "x1": 0, "y1": 0, "x2": 0, "y2": 0, "tiled": False}
    return {"clip": clip, "frame": rest, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "tiled": False}


def _label_nonempty(label_path: Path) -> bool:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return True
    return False


def scan_split_group_counts(
    images_dir: Path,
    labels_dir: Path,
    *,
    clip_group: dict[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Count labeled/empty samples currently on disk, by train_groups name."""
    payload = load_tiling_payload()
    out: dict[str, dict[str, int]] = {
        g: {"labeled_kept": 0, "empty_kept": 0, "total": 0} for g in BALANCE_GROUP_ORDER
    }
    if not images_dir.is_dir():
        return out
    for image_path in sorted(images_dir.glob(f"*{SLICE_OUT_EXT}")):
        stem = image_path.stem
        # Skip rotation augments when attributing to source group (still A_close).
        base_stem = stem
        for angle in ROTATION_AUG_ANGLES:
            suffix = f"_rot{angle}"
            if stem.endswith(suffix):
                base_stem = stem[: -len(suffix)]
                break
        clip = _clip_from_sample_stem(base_stem)
        if clip_group and clip in clip_group:
            group = clip_group[clip]
        else:
            group = resolve_train_group_tiling(clip, payload=payload).group
        bucket = out.setdefault(group, {"labeled_kept": 0, "empty_kept": 0, "total": 0})
        label_path = labels_dir / f"{stem}.txt"
        bucket["total"] += 1
        if _label_nonempty(label_path):
            bucket["labeled_kept"] += 1
        else:
            bucket["empty_kept"] += 1
    return out


def _delete_sample(images_dir: Path, labels_dir: Path, stem: str) -> None:
    img = images_dir / f"{stem}{SLICE_OUT_EXT}"
    lab = labels_dir / f"{stem}.txt"
    if img.exists():
        img.unlink()
    if lab.exists():
        lab.unlink()


def downsample_b_medium_spatial(
    images_dir: Path,
    labels_dir: Path,
    *,
    clip_group: dict[str, str],
    target_count: int,
) -> dict:
    """Keep every N-th B_medium tile in grid order; never drop a frame's sole labeled tile."""
    b_clips = {c for c, g in clip_group.items() if g == "B_medium"}
    samples: list[dict] = []
    for image_path in sorted(images_dir.glob(f"*{SLICE_OUT_EXT}")):
        stem = image_path.stem
        clip = _clip_from_sample_stem(stem)
        if clip not in b_clips:
            continue
        meta = _parse_tile_stem(stem)
        if meta is None:
            continue
        labeled = _label_nonempty(labels_dir / f"{stem}.txt")
        samples.append(
            {
                "stem": stem,
                "clip": clip,
                "frame": meta["frame"],
                "x1": meta["x1"],
                "y1": meta["y1"],
                "labeled": labeled,
            }
        )

    before = len(samples)
    if before == 0 or target_count <= 0 or before <= target_count:
        print(
            f"B_medium downsample: keep all {before} tiles "
            f"(target={target_count})"
        )
        return {"before": before, "after": before, "removed": 0, "stride": 1}

    stride = max(1, int(round(before / target_count)))
    by_frame: dict[tuple[str, str], list[dict]] = {}
    for s in samples:
        by_frame.setdefault((s["clip"], s["frame"]), []).append(s)

    keep_stems: set[str] = set()
    for (_clip, _frame), tiles in by_frame.items():
        labeled_tiles = [t for t in tiles if t["labeled"]]
        # Spatial order: row-major by (y1, x1), not "first N" file order.
        labeled_tiles.sort(key=lambda t: (t["y1"], t["x1"], t["stem"]))
        empty_tiles = [t for t in tiles if not t["labeled"]]
        empty_tiles.sort(key=lambda t: (t["y1"], t["x1"], t["stem"]))

        if len(labeled_tiles) == 1:
            keep_stems.add(labeled_tiles[0]["stem"])
            selected_labeled = labeled_tiles
        elif len(labeled_tiles) > 1:
            selected_labeled = labeled_tiles[::stride]
            if not selected_labeled:
                selected_labeled = [labeled_tiles[0]]
            # Always retain at least one labeled tile per frame.
            if labeled_tiles[0]["stem"] not in {t["stem"] for t in selected_labeled}:
                selected_labeled = [labeled_tiles[0]] + selected_labeled
            for t in selected_labeled:
                keep_stems.add(t["stem"])
        else:
            selected_labeled = []

        # Keep empties only when the frame has no labeled tile (rare with empty_kept=0).
        if empty_tiles and not selected_labeled:
            for t in empty_tiles[::stride]:
                keep_stems.add(t["stem"])

    # If still above target, drop additional labeled tiles that are not sole-per-frame.
    labeled_kept = [
        s for s in samples if s["stem"] in keep_stems and s["labeled"]
    ]
    if len(keep_stems) > target_count and len(labeled_kept) > target_count:
        sole_frames = {
            (s["clip"], s["frame"])
            for s in samples
            if s["labeled"]
            and sum(
                1
                for o in samples
                if o["clip"] == s["clip"] and o["frame"] == s["frame"] and o["labeled"]
            )
            == 1
        }
        removable = [
            s
            for s in labeled_kept
            if (s["clip"], s["frame"]) not in sole_frames
        ]
        removable.sort(key=lambda t: (t["y1"], t["x1"], t["stem"]))
        # Second-pass stride over removable set.
        need_drop = len(keep_stems) - target_count
        # Prefer dropping empties first.
        empty_kept = [s for s in samples if s["stem"] in keep_stems and not s["labeled"]]
        for s in empty_kept:
            if need_drop <= 0:
                break
            keep_stems.discard(s["stem"])
            need_drop -= 1
        if need_drop > 0 and removable:
            drop_stride = max(1, len(removable) // need_drop) if need_drop else 1
            # Drop every drop_stride-th in grid order (spatial), not a prefix.
            dropped = 0
            for i, s in enumerate(removable):
                if dropped >= need_drop:
                    break
                if i % drop_stride == 0 and s["stem"] in keep_stems:
                    # Don't drop if it would leave frame with zero labeled keeps.
                    frame_key = (s["clip"], s["frame"])
                    still = [
                        o
                        for o in labeled_kept
                        if o["stem"] in keep_stems
                        and (o["clip"], o["frame"]) == frame_key
                        and o["stem"] != s["stem"]
                    ]
                    if still:
                        keep_stems.discard(s["stem"])
                        dropped += 1

    removed = 0
    for s in samples:
        if s["stem"] not in keep_stems:
            _delete_sample(images_dir, labels_dir, s["stem"])
            removed += 1

    after = before - removed
    print(
        f"B_medium spatial downsample: {before} → {after} "
        f"(removed={removed}, stride={stride}, target≈{target_count})"
    )
    return {"before": before, "after": after, "removed": removed, "stride": stride}


def rotate_yolo_labels_cw(lines: list[str], angle: int) -> list[str]:
    """Remap YOLO-normalized boxes for CW rotation by 90/180/270."""
    out: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = parts[0]
        xc, yc, bw, bh = map(float, parts[1:5])
        if angle == 90:
            nxc, nyc, nw, nh = yc, 1.0 - xc, bh, bw
        elif angle == 180:
            nxc, nyc, nw, nh = 1.0 - xc, 1.0 - yc, bw, bh
        elif angle == 270:
            nxc, nyc, nw, nh = 1.0 - yc, xc, bh, bw
        else:
            raise ValueError(f"unsupported rotation angle: {angle}")
        nxc = min(1.0, max(0.0, nxc))
        nyc = min(1.0, max(0.0, nyc))
        out.append(f"{cls_id} {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}")
    return out


def _cv2_rotate_cw(image, angle: int):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotation angle: {angle}")


def oversample_a_close_rotations(
    images_dir: Path,
    labels_dir: Path,
    *,
    clip_group: dict[str, str],
    target_share: float = A_CLOSE_TARGET_SHARE,
) -> dict:
    """Physically write 90/180/270 rotated copies of A_close images until share≈target."""
    a_clips = {c for c, g in clip_group.items() if g == "A_close"}
    originals: list[str] = []
    for image_path in sorted(images_dir.glob(f"*{SLICE_OUT_EXT}")):
        stem = image_path.stem
        if any(stem.endswith(f"_rot{a}") for a in ROTATION_AUG_ANGLES):
            continue
        clip = _clip_from_sample_stem(stem)
        if clip not in a_clips:
            continue
        if not _label_nonempty(labels_dir / f"{stem}.txt"):
            continue
        originals.append(stem)

    n_a = len(originals)
    # Current totals including other groups.
    n_total = len(list(images_dir.glob(f"*{SLICE_OUT_EXT}")))
    n_other = n_total - n_a
    if n_a == 0:
        print("A_close oversample: no labeled originals found; skipped")
        return {"originals": 0, "added": 0, "final_a": 0, "share": 0.0}

    # a_final = share/(1-share) * n_other  ; need a_final >= n_a
    share = min(A_CLOSE_SHARE_MAX, max(A_CLOSE_SHARE_MIN, target_share))
    target_a = int(round((share / (1.0 - share)) * n_other))
    target_a = max(target_a, n_a)
    # Cap at originals * (1 + len(angles))
    max_a = n_a * (1 + len(ROTATION_AUG_ANGLES))
    target_a = min(target_a, max_a)
    need = max(0, target_a - n_a)

    print(
        f"A_close oversample: originals={n_a}, other={n_other}, "
        f"target_share={share:.0%} → target_a={target_a}, need={need} rotated copies"
    )
    if need == 0:
        return {
            "originals": n_a,
            "added": 0,
            "final_a": n_a,
            "share": n_a / max(1, n_total),
        }

    # Cycle originals; for each, emit unused angles until need filled.
    added = 0
    # Pass 1..3: apply angle k to every original until need met (covers source diversity).
    for angle in ROTATION_AUG_ANGLES:
        if added >= need:
            break
        for stem in originals:
            if added >= need:
                break
            out_stem = f"{stem}_rot{angle}"
            out_img = images_dir / f"{out_stem}{SLICE_OUT_EXT}"
            out_lab = labels_dir / f"{out_stem}.txt"
            if out_img.exists():
                continue
            src_img = images_dir / f"{stem}{SLICE_OUT_EXT}"
            src_lab = labels_dir / f"{stem}.txt"
            image = cv2.imread(str(src_img))
            if image is None:
                continue
            rotated = _cv2_rotate_cw(image, angle)
            lines = [
                ln for ln in src_lab.read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
            new_lines = rotate_yolo_labels_cw(lines, angle)
            if not new_lines:
                continue
            if not cv2.imwrite(str(out_img), rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                continue
            out_lab.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            added += 1

    final_total = len(list(images_dir.glob(f"*{SLICE_OUT_EXT}")))
    final_a = n_a + added
    share_final = final_a / max(1, final_total)
    print(
        f"A_close oversample done: added={added} rotations, "
        f"A_close={final_a} ({share_final:.1%} of train)"
    )
    return {
        "originals": n_a,
        "added": added,
        "final_a": final_a,
        "share": share_final,
        "target_a": target_a,
    }


def balance_train_groups(
    images_dir: Path,
    labels_dir: Path,
    *,
    clip_group: dict[str, str],
) -> dict:
    """Downsample B_medium spatially, oversample A_close with 90° rotations. Train only."""
    print("\n=== Train group balancing ===")
    before = scan_split_group_counts(images_dir, labels_dir, clip_group=clip_group)
    total_before = sum(v["total"] for v in before.values())
    print("Before balance:")
    for g in BALANCE_GROUP_ORDER:
        gs = before.get(g, {})
        pct = 100.0 * gs.get("total", 0) / total_before if total_before else 0.0
        print(
            f"  {g}: total={gs.get('total', 0)} ({pct:.1f}%), "
            f"labeled_kept={gs.get('labeled_kept', 0)}, "
            f"empty_kept={gs.get('empty_kept', 0)}"
        )

    n_a = before.get("A_close", {}).get("total", 0)
    n_b = before.get("B_medium", {}).get("total", 0)
    n_c = before.get("C_far", {}).get("total", 0)
    # Aim B ≈ C (absolute), so far/medium are comparable before A oversample.
    target_b = max(n_c, 1) if n_b > 0 else 0
    # If C is much smaller than frame share suggests, don't crush B below ~A*2.
    target_b = max(target_b, min(n_b, max(n_a * 2, n_c)))

    b_info = downsample_b_medium_spatial(
        images_dir,
        labels_dir,
        clip_group=clip_group,
        target_count=target_b,
    )
    a_info = oversample_a_close_rotations(
        images_dir,
        labels_dir,
        clip_group=clip_group,
        target_share=A_CLOSE_TARGET_SHARE,
    )

    after = scan_split_group_counts(images_dir, labels_dir, clip_group=clip_group)
    total_after = sum(v["total"] for v in after.values())
    print("After balance:")
    for g in BALANCE_GROUP_ORDER:
        gs = after.get(g, {})
        pct = 100.0 * gs.get("total", 0) / total_after if total_after else 0.0
        print(
            f"  {g}: total={gs.get('total', 0)} ({pct:.1f}%), "
            f"labeled_kept={gs.get('labeled_kept', 0)}, "
            f"empty_kept={gs.get('empty_kept', 0)}"
        )

    return {
        "before": before,
        "after": after,
        "b_medium": b_info,
        "a_close": a_info,
        "total_before": total_before,
        "total_after": total_after,
    }


def refresh_train_stats_from_disk(train_stats: dict, dataset_dir: Path) -> dict:
    """Update aggregate / per-group counters after on-disk balancing."""
    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"
    clip_group = train_stats.get("clip_group") or {}
    counts = scan_split_group_counts(images_dir, labels_dir, clip_group=clip_group)
    labeled = sum(v["labeled_kept"] for v in counts.values())
    empty_kept = sum(v["empty_kept"] for v in counts.values())
    total = labeled + empty_kept
    train_stats = dict(train_stats)
    train_stats["slices_kept"] = total
    train_stats["slices_with_labels"] = labeled
    train_stats["empty_kept"] = empty_kept
    train_stats["group_stats"] = {
        g: {
            "labeled_kept": counts.get(g, {}).get("labeled_kept", 0),
            "empty_kept": counts.get(g, {}).get("empty_kept", 0),
            "empty_removed": (train_stats.get("group_stats") or {}).get(g, {}).get(
                "empty_removed", 0
            ),
            "produced": (train_stats.get("group_stats") or {}).get(g, {}).get("produced", 0),
            "total": counts.get(g, {}).get("total", 0),
        }
        for g in BALANCE_GROUP_ORDER
    }
    return train_stats


def split_train_frames_for_val(
    frames: list[tuple[str, Path, Path]],
    *,
    val_fraction: float,
    seed: int = VAL_SPLIT_SEED,
) -> tuple[list[tuple[str, Path, Path]], list[tuple[str, Path, Path]]]:
    """Hold out a stratified fraction of train frames for val; never uses eval clips."""
    if not 0.0 < val_fraction < 1.0:
        raise SystemExit(f"--val-fraction must be in (0, 1), got {val_fraction}")

    by_clip: dict[str, list[tuple[str, Path, Path]]] = {}
    for item in frames:
        by_clip.setdefault(item[0], []).append(item)

    rng = random.Random(seed)
    train_frames: list[tuple[str, Path, Path]] = []
    val_frames: list[tuple[str, Path, Path]] = []
    for clip_name, clip_frames in sorted(by_clip.items()):
        ordered = list(clip_frames)
        rng.shuffle(ordered)
        n_val = max(1, int(round(len(ordered) * val_fraction))) if len(ordered) > 1 else 0
        if len(ordered) <= 1:
            train_frames.extend(ordered)
            print(f"  val split {clip_name}: {len(ordered)} frames → all train (too few for holdout)")
            continue
        n_val = min(n_val, len(ordered) - 1)
        val_part = ordered[:n_val]
        train_part = ordered[n_val:]
        val_frames.extend(val_part)
        train_frames.extend(train_part)
        print(
            f"  val split {clip_name}: {len(ordered)} frames → "
            f"train={len(train_part)}, val={len(val_part)}"
        )

    if not train_frames:
        raise SystemExit("Train holdout left zero train frames.")
    if not val_frames:
        raise SystemExit(
            "Train holdout left zero val frames. Increase --val-fraction or add more labels."
        )
    return train_frames, val_frames


def prepare_sliced_val_from_train_holdout(
    frames: list[tuple[str, Path, Path]],
    *,
    imgsz: int,
    dataset_dir: Path,
) -> dict:
    """Build Ultralytics val set from held-out train frames (not eval clips)."""
    if not frames:
        raise SystemExit("No holdout frames for val.")

    print_group_tiling_plan(frames)
    stats = slice_frames_to_dataset(
        frames,
        dataset_dir / "val" / "images",
        dataset_dir / "val" / "labels",
        imgsz=imgsz,
        max_empty_slices_per_frame=0,
        progress_label="val",
    )
    print(
        f"Val samples: {stats['slices_kept']} labeled from {stats['source_frames']} "
        f"held-out train frames ({stats['slices_dropped']} empty dropped)"
    )
    return stats


def write_dataset_yaml(
    imgsz: int,
    train_stats: dict,
    val_stats: dict,
    *,
    dataset_dir: Path,
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = dataset_dir / "data.yaml"
    payload = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: CLASS_NAME},
        "nc": 1,
        "imgsz": imgsz,
        "val_source": "holdout_from_train_videos",
        "clip_slice_size": {
            **train_stats.get("clip_slice_size", {}),
            **val_stats.get("clip_slice_size", {}),
        },
        "clip_overlap": {
            **train_stats.get("clip_overlap", {}),
            **val_stats.get("clip_overlap", {}),
        },
        "clip_group": {
            **train_stats.get("clip_group", {}),
            **val_stats.get("clip_group", {}),
        },
        "clip_train_imgsz": {
            **train_stats.get("clip_train_imgsz", {}),
            **val_stats.get("clip_train_imgsz", {}),
        },
        "clip_uses_tiling": {
            **train_stats.get("clip_uses_tiling", {}),
            **val_stats.get("clip_uses_tiling", {}),
        },
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def summarize_dataset_size(train_stats: dict, val_stats: dict, *, dataset_dir: Path) -> None:
    """Print final train vs val counts + empty-label accounting + on-disk size."""
    train_images = dataset_dir / "train" / "images"
    val_images = dataset_dir / "val" / "images"
    n_train = len(list(train_images.glob(f"*{SLICE_OUT_EXT}"))) if train_images.is_dir() else 0
    n_val = len(list(val_images.glob(f"*{SLICE_OUT_EXT}"))) if val_images.is_dir() else 0

    def _dir_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024 * 1024)

    def _empty_line(label: str, stats: dict) -> str:
        total = stats.get("slices_total", "?")
        labeled = stats.get("slices_with_labels", "?")
        empty_kept = stats.get("empty_kept")
        empty_removed = stats.get("empty_removed", stats.get("slices_dropped", "?"))
        if empty_kept is None and isinstance(stats.get("slices_kept"), int) and isinstance(labeled, int):
            empty_kept = stats["slices_kept"] - labeled
        return (
            f"{label}: produced={total}, labeled_kept={labeled}, "
            f"empty_kept={empty_kept}, empty_removed={empty_removed}"
        )

    train_mb = _dir_mb(dataset_dir / "train")
    val_mb = _dir_mb(dataset_dir / "val")
    total_mb = _dir_mb(dataset_dir)

    print("\n=== Dataset size (train videos only; eval clips excluded) ===")
    print(
        f"Train: {n_train} images "
        f"(from {train_stats.get('source_frames', '?')} frames) — {train_mb:.1f} MB"
    )
    print(f"  {_empty_line('empty labels', train_stats)}")
    print(
        f"Val:   {n_val} images "
        f"(from {val_stats.get('source_frames', '?')} held-out train frames) — {val_mb:.1f} MB"
    )
    print(f"  {_empty_line('empty labels', val_stats)}")
    print(f"Total under dataset/: {total_mb:.1f} MB")
    print("Eval clips are not in this dataset — use evaluate.py for external metrics.")

    def _print_groups(label: str, stats: dict) -> None:
        groups = stats.get("group_stats") or {}
        if not groups:
            return
        total = sum(
            int(g.get("total", g.get("labeled_kept", 0) + g.get("empty_kept", 0)))
            for g in groups.values()
        )
        print(f"{label} by group:")
        for name in BALANCE_GROUP_ORDER:
            gs = groups.get(name) or {}
            n = int(gs.get("total", gs.get("labeled_kept", 0) + gs.get("empty_kept", 0)))
            pct = 100.0 * n / total if total else 0.0
            print(
                f"  {name}: total={n} ({pct:.1f}%), "
                f"labeled_kept={gs.get('labeled_kept', 0)}, "
                f"empty_kept={gs.get('empty_kept', 0)}, "
                f"empty_removed={gs.get('empty_removed', 0)}"
            )

    _print_groups("Train", train_stats)
    _print_groups("Val", val_stats)

    clip_group = {
        **(train_stats.get("clip_group") or {}),
        **(val_stats.get("clip_group") or {}),
    }
    clip_slice = {
        **(train_stats.get("clip_slice_size") or {}),
        **(val_stats.get("clip_slice_size") or {}),
    }
    clip_tiling = {
        **(train_stats.get("clip_uses_tiling") or {}),
        **(val_stats.get("clip_uses_tiling") or {}),
    }
    if clip_group:
        print("Per-clip crop:")
        for name in sorted(clip_group):
            group = clip_group[name]
            uses = clip_tiling.get(name, True)
            tile = "full-frame" if not uses else f"tile={clip_slice.get(name, '?')}"
            print(f"  {name}: group={group}, {tile}")


def prepare_dataset(
    split_map: dict[str, str],
    *,
    recreate: bool,
    imgsz: int,
    val_fraction: float,
    dataset_dir: Path | None = None,
    frame_step_only: bool = False,
    balance: bool = True,
    max_empty_slices_per_frame: int | None = None,
    labels_root: Path | None = None,
) -> Path:
    dataset_dir = (dataset_dir or DATASET_DIR).resolve()
    labels_root_resolved = (labels_root or LABELS_DIR).resolve()
    empty_cap = (
        MAX_EMPTY_SLICES_PER_FRAME
        if max_empty_slices_per_frame is None
        else int(max_empty_slices_per_frame)
    )
    train_images = dataset_dir / "train" / "images"
    if recreate or not train_images.exists():
        all_train_frames = iter_labeled_frames(
            "train",
            split_map,
            frame_step_only=frame_step_only,
            labels_root=labels_root_resolved,
        )
        if not all_train_frames:
            raise SystemExit(
                f"No train labels found under {labels_root_resolved / 'train'}/."
            )

        labels_disp = (
            labels_root_resolved.relative_to(PROJECT_ROOT)
            if labels_root_resolved.is_relative_to(PROJECT_ROOT)
            else labels_root_resolved
        )
        print(
            f"Dataset dir: {dataset_dir.relative_to(PROJECT_ROOT) if dataset_dir.is_relative_to(PROJECT_ROOT) else dataset_dir}"
        )
        print(f"Labels root: {labels_disp}")
        print(
            f"Prepare flags: frame_step_only={frame_step_only}, balance={balance}, "
            f"imgsz={imgsz}, empty_slices_cap={empty_cap}"
        )
        print(
            f"Splitting train videos into train/val "
            f"(val_fraction={val_fraction}, seed={VAL_SPLIT_SEED}); "
            "eval clips stay out of dataset..."
        )
        train_frames, val_frames = split_train_frames_for_val(
            all_train_frames,
            val_fraction=val_fraction,
        )
        print(
            f"Frame split: train={len(train_frames)}, val={len(val_frames)} "
            f"(from {len(all_train_frames)} labeled train frames)"
        )

        print(f"Preparing train set (model imgsz={imgsz}, crops from train_groups)...")
        print_group_tiling_plan(train_frames)
        train_stats = slice_frames_to_dataset(
            train_frames,
            dataset_dir / "train" / "images",
            dataset_dir / "train" / "labels",
            imgsz=imgsz,
            max_empty_slices_per_frame=empty_cap,
            progress_label="train",
        )
        print(
            f"Train samples: {train_stats['slices_kept']} kept from "
            f"{train_stats['source_frames']} frames "
            f"({train_stats['slices_with_labels']} with labels, "
            f"{train_stats['slices_dropped']} empty dropped, format={SLICE_OUT_EXT})"
        )

        if balance:
            balance_info = balance_train_groups(
                dataset_dir / "train" / "images",
                dataset_dir / "train" / "labels",
                clip_group=train_stats.get("clip_group") or {},
            )
            train_stats = refresh_train_stats_from_disk(train_stats, dataset_dir)
            train_stats["balance"] = balance_info
        else:
            balance_info = {"skipped": True, "reason": "no_balance"}
            train_stats["balance"] = balance_info
            print("Skipping train-group balance / rotation oversample.")

        print("Preparing val set from held-out train frames (no balance/augs)...")
        val_stats = prepare_sliced_val_from_train_holdout(
            val_frames,
            imgsz=imgsz,
            dataset_dir=dataset_dir,
        )
        yaml_path = write_dataset_yaml(
            imgsz,
            train_stats,
            val_stats,
            dataset_dir=dataset_dir,
        )
        manifest = {
            "dataset_dir": str(dataset_dir.relative_to(PROJECT_ROOT))
            if dataset_dir.is_relative_to(PROJECT_ROOT)
            else str(dataset_dir),
            "imgsz": imgsz,
            "val_fraction": val_fraction,
            "val_seed": VAL_SPLIT_SEED,
            "val_source": "holdout_from_train_videos",
            "frame_step_only": frame_step_only,
            "balance": balance,
            "empty_slices_per_frame": empty_cap,
            "labels_root": str(labels_disp),
            "augmentation": "none_at_prepare",
            "tiling_source": "config/clip_tiling.json train_groups",
            "train": train_stats,
            "val": val_stats,
            "balance_info": balance_info,
        }
        (dataset_dir / "prepare_stats.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(f"Dataset ready: {yaml_path}")
        summarize_dataset_size(train_stats, val_stats, dataset_dir=dataset_dir)
        return yaml_path

    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(
            f"Dataset images exist but {yaml_path} is missing. "
            "Rebuild with: python src/training/train.py --recreate-dataset"
        )
    print(f"Using existing dataset at {dataset_dir}")
    stats_path = dataset_dir / "prepare_stats.json"
    if stats_path.exists():
        manifest = json.loads(stats_path.read_text(encoding="utf-8"))
        summarize_dataset_size(
            manifest.get("train") or {},
            manifest.get("val") or {},
            dataset_dir=dataset_dir,
        )
    return yaml_path


def resolve_cache_arg(cache: str | bool) -> str | bool:
    if cache == "false":
        return False
    if cache == "ram":
        return True
    return "disk"


def attach_component_loss_logger(model: YOLO, run_label: str) -> None:
    """Print box/cls/dfl losses each epoch (Ultralytics also writes them to results.csv)."""

    def on_fit_epoch_end(trainer) -> None:
        loss_names = tuple(getattr(trainer, "loss_names", ("box_loss", "cls_loss", "dfl_loss")))
        tloss = getattr(trainer, "tloss", None)
        if tloss is None:
            return
        values = tloss.detach().float().cpu().tolist()
        if not isinstance(values, list):
            values = [float(values)]
        parts = [
            f"{name}={value:.5f}"
            for name, value in zip(loss_names, values, strict=False)
        ]
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        epochs = int(getattr(trainer.args, "epochs", 0))
        print(f"[{run_label}] epoch {epoch}/{epochs} train losses: " + ", ".join(parts))

        # Val component losses when validation ran this epoch
        metrics = getattr(trainer, "metrics", None) or {}
        val_parts = []
        for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss"):
            if key in metrics:
                val_parts.append(f"{key.split('/', 1)[1]}={float(metrics[key]):.5f}")
        if val_parts:
            print(f"[{run_label}] epoch {epoch}/{epochs} val losses: " + ", ".join(val_parts))

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def summarize_results_csv(run_dir: Path, run_label: str) -> None:
    """Echo per-epoch box/cls/dfl columns from Ultralytics results.csv."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"[{run_label}] no results.csv at {csv_path}")
        return

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return

    headers = [h.strip() for h in lines[0].split(",")]
    wanted = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    idxs = {name: headers.index(name) for name in wanted if name in headers}
    print(f"[{run_label}] per-epoch loss/metric summary ({csv_path}):")
    for line in lines[1:]:
        cols = [c.strip() for c in line.split(",")]
        epoch = cols[0] if cols else "?"
        bits = [f"epoch={epoch}"]
        for name, idx in idxs.items():
            if idx < len(cols) and cols[idx] != "":
                short = name.replace("train/", "tr/").replace("val/", "va/").replace("metrics/", "")
                bits.append(f"{short}={cols[idx]}")
        print("  " + " | ".join(bits))


def resolve_best_weights(run_dir: Path) -> Path:
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        return best_weights
    last_weights = run_dir / "weights" / "last.pt"
    if last_weights.exists():
        return last_weights
    raise FileNotFoundError(f"No weights found under {run_dir / 'weights'}")


def train_one_stage(
    *,
    weights: str | Path,
    yaml_path: Path,
    run_name: str,
    imgsz: int,
    epochs: int,
    batch: int,
    workers: int,
    cache_arg: str | bool,
    freeze: int,
    lr0: float,
    patience: int,
    close_mosaic: int,
    aug: dict[str, float] | None = None,
) -> tuple[Path, int]:
    """Run one Ultralytics train() stage with MPS OOM batch backoff.

    Returns (best_or_last_weights, final_batch_size).
    Loads `weights` then starts a fresh optimizer schedule (not resume=True), so
    Stage 2 can change freeze/LR without restoring Stage 1 training state.
    """
    if epochs <= 0:
        raise ValueError(f"{run_name}: epochs must be > 0, got {epochs}")

    aug_kwargs = dict(train_aug_kwargs() if aug is None else aug)
    batch_try = max(1, int(batch))
    last_error: Exception | None = None
    run_dir = RUNS_DIR / run_name

    for attempt in range(4):
        if DEVICE == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        model = YOLO(str(weights))
        attach_component_loss_logger(model, run_name)
        freeze_desc = (
            f"freeze first {freeze} layers (backbone)"
            if freeze > 0
            else "all layers trainable"
        )
        print(
            f"[{run_name}] {weights} on {DEVICE}: {epochs} epochs, imgsz={imgsz}, "
            f"batch={batch_try}, workers={workers}, cache={cache_arg}, "
            f"lr0={lr0}, patience={patience}, {freeze_desc}, "
            f"aug(hsv=({aug_kwargs['hsv_h']},{aug_kwargs['hsv_s']},{aug_kwargs['hsv_v']}), "
            f"deg={aug_kwargs['degrees']}, fliplr={aug_kwargs['fliplr']}, "
            f"flipud={aug_kwargs['flipud']}, mosaic={aug_kwargs['mosaic']}) "
            f"(attempt {attempt + 1}/4)"
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
                name=run_name,
                exist_ok=True,
                pretrained=True,
                freeze=freeze,
                lr0=lr0,
                val=True,
                patience=patience,
                amp=True,
                cache=cache_arg,
                plots=False,
                save_period=-1,
                resume=False,
                close_mosaic=close_mosaic,
                **aug_kwargs,
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
                    f"[{run_name}] MPS/train fault at batch={batch_try} ({exc}); "
                    f"retrying with batch={new_batch}"
                )
                batch_try = new_batch
                continue
            raise

    if last_error is not None:
        raise last_error

    summarize_results_csv(run_dir, run_name)
    return resolve_best_weights(run_dir), batch_try


def train_model(
    yaml_path: Path,
    *,
    imgsz: int,
    warmup_epochs: int,
    epochs: int,
    batch: int,
    workers: int,
    cache: str | bool,
    freeze: int,
    lr0: float,
    patience: int,
    aug: dict[str, float] | None = None,
    stage1_run_name: str | None = None,
    stage2_run_name: str | None = None,
    deliverable_name: str = "yolo11s_vehicle_best.pt",
) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    configure_mps_runtime()
    patch_ultralytics_mps_unique()

    cache_arg = resolve_cache_arg(cache)
    stage2_lr0 = lr0 / 10.0
    # Ultralytics patience=0 still early-stops in some versions; use a large cap to disable.
    patience_arg = patience if patience > 0 else 10**9
    aug_kwargs = dict(train_aug_kwargs() if aug is None else aug)
    s1_name = stage1_run_name or STAGE1_RUN_NAME
    s2_name = stage2_run_name or STAGE2_RUN_NAME

    print(
        f"Staged fine-tune: Stage1 head-only ({warmup_epochs} ep, freeze={freeze}, "
        f"lr0={lr0}) → Stage2 full ({epochs} ep, freeze=0, lr0={stage2_lr0}), "
        f"val=True, patience={patience if patience > 0 else 'off'}, "
        f"mosaic={aug_kwargs.get('mosaic', 0.0)}"
    )

    # --- Stage 1: freeze backbone, adapt Detect head to single-class "vehicle" ---
    stage1_weights, batch = train_one_stage(
        weights=MODEL_NAME,
        yaml_path=yaml_path,
        run_name=s1_name,
        imgsz=imgsz,
        epochs=warmup_epochs,
        batch=batch,
        workers=workers,
        cache_arg=cache_arg,
        freeze=freeze,
        lr0=lr0,
        patience=patience_arg,
        close_mosaic=max(0, min(3, warmup_epochs // 2)),
        aug=aug_kwargs,
    )
    # Continue from end-of-warmup weights (trajectory into Stage 2), not necessarily best.
    stage1_last = RUNS_DIR / s1_name / "weights" / "last.pt"
    stage2_init = stage1_last if stage1_last.exists() else stage1_weights
    print(f"Stage 1 done. Continuing from {stage2_init}")

    # --- Stage 2: unfreeze all, lower LR, early-stop on val ---
    best_weights, _ = train_one_stage(
        weights=stage2_init,
        yaml_path=yaml_path,
        run_name=s2_name,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        cache_arg=cache_arg,
        freeze=0,
        lr0=stage2_lr0,
        patience=patience_arg,
        close_mosaic=max(0, min(5, epochs // 3)),
        aug=aug_kwargs,
    )

    deliverable = PROJECT_ROOT / "checkpoints" / deliverable_name
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, deliverable)
    print(f"Git-tracked copy: {deliverable}")
    return best_weights


def dataset_already_prepared(dataset_dir: Path | None = None) -> bool:
    root = (dataset_dir or DATASET_DIR).resolve()
    yaml_path = root / "data.yaml"
    train_images = root / "train" / "images"
    return yaml_path.exists() and train_images.is_dir() and any(train_images.glob(f"*{SLICE_OUT_EXT}"))


def main() -> None:
    args = parse_args()
    split_map = build_split_map()

    if args.prototype:
        print(
            "Prototype (fast) schedule: Stage1 head-only → Stage2 full "
            f"(backbone unfrozen); "
            f"warmup={PROTOTYPE_WARMUP_EPOCHS}, epochs={PROTOTYPE_EPOCHS}, "
            f"patience={PROTOTYPE_PATIENCE}"
        )
    warmup_epochs = (
        args.warmup_epochs
        if args.warmup_epochs is not None
        else (PROTOTYPE_WARMUP_EPOCHS if args.prototype else WARMUP_EPOCHS)
    )
    epochs = (
        args.epochs
        if args.epochs is not None
        else (PROTOTYPE_EPOCHS if args.prototype else EPOCHS)
    )
    patience = (
        args.patience
        if args.patience is not None
        else (PROTOTYPE_PATIENCE if args.prototype else PATIENCE)
    )

    dataset_dir = args.dataset_dir
    frame_step_only = bool(args.frame_step_only)
    balance = not bool(args.no_balance)
    imgsz = args.imgsz
    if args.baseline:
        dataset_dir = dataset_dir or BASELINE_DATASET_DIR
        frame_step_only = True
        balance = False
        if imgsz == TRAIN_IMGSZ or args.imgsz == TRAIN_IMGSZ:
            imgsz = TRAIN_IMGSZ
        print(
            "Baseline preset: frame_step_only=True, balance=False, "
            f"imgsz={imgsz}, dataset_dir={dataset_dir}"
        )
    dataset_dir = (dataset_dir or DATASET_DIR).resolve()

    prepared = dataset_already_prepared(dataset_dir) and not args.recreate_dataset

    if prepared:
        yaml_meta = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
        val_source = yaml_meta.get("val_source", "unknown")
        print(f"Using prepared dataset at {dataset_dir} (val_source={val_source})")
    else:
        if not split_map:
            raise SystemExit("No clips found in data/train or data/eval.")

        train_frames = iter_labeled_frames(
            "train",
            split_map,
            frame_step_only=frame_step_only,
        )
        eval_frames = iter_labeled_frames(
            "eval",
            split_map,
            frame_step_only=frame_step_only,
        )
        print(f"Labeled frames: train={len(train_frames)}, eval={len(eval_frames)}")
        if not train_frames:
            raise SystemExit(
                "No train labels under labels/train/. "
                "Sync CVAT: python src/labeling/cvat_pull.py --verify --sync-labels"
            )

    batch = args.batch if args.batch is not None else default_batch_size(imgsz)
    workers = args.workers if args.workers is not None else default_workers()
    print(
        f"Device={DEVICE}, RAM≈{system_ram_gb():.0f}GB → "
        f"batch={batch}, workers={workers}, cache={args.cache}, imgsz={imgsz}"
    )

    yaml_path = prepare_dataset(
        split_map,
        recreate=args.recreate_dataset,
        imgsz=imgsz,
        val_fraction=args.val_fraction,
        dataset_dir=dataset_dir,
        frame_step_only=frame_step_only,
        balance=balance,
    )

    if args.prepare_only:
        print("Dataset prepared. Run without --prepare-only to train.")
        return

    if warmup_epochs < 1:
        raise SystemExit("--warmup-epochs must be >= 1")
    if epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.freeze < 0:
        raise SystemExit("--freeze must be >= 0")

    weights = train_model(
        yaml_path,
        imgsz=imgsz,
        warmup_epochs=warmup_epochs,
        epochs=epochs,
        batch=batch,
        workers=workers,
        cache=args.cache,
        freeze=args.freeze,
        lr0=args.lr0,
        patience=patience,
    )
    print(f"Done. Best weights: {weights}")


if __name__ == "__main__":
    main()

