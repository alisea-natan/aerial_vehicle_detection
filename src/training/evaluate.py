#!/usr/bin/env python3
"""Evaluate fine-tuned detector on prepared eval packs (autolabel / manual) per distance band."""
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
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from sahi.slicing import slice_image
from ultralytics import YOLO

from training.model_load import load_ultralytics_model, predict_device

from common.config import (
    CLIP_TILING_CONFIG_PATH,
    FRAMES_DIR,
    LABELS_DIR,
    POC_CHECKPOINT,
    PROJECT_ROOT,
    PROTOTYPE_CHECKPOINT,
    TRAIN_IMGSZ,
    TRAIN_OVERLAP_RATIO,
    build_split_map,
    effective_slice_size,
    is_clip_skipped,
    clip_skip_reason,
    load_clip_tile_config,
    load_tiling_payload,
    resolve_clip_tile_config,
    resolve_train_group_tiling,
)
from common.detect import device

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_DIR = OUTPUTS_DIR / "dataset"
DATASET_YAML = DATASET_DIR / "data.yaml"
PREPARE_STATS = DATASET_DIR / "prepare_stats.json"
DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
EVAL_PACKS = {
    "autolabel": DATASETS_ROOT / "eval_autolabel",
    "manual": DATASETS_ROOT / "eval_manual",
}

DEFAULT_WEIGHTS = PROTOTYPE_CHECKPOINT
POC_WEIGHTS = POC_CHECKPOINT
VEHICLE_CLASS = "vehicle"
PRED_CONF_HIGH = 0.5
PRED_COLOR_HIGH = (0, 200, 0)  # BGR green — conf > PRED_CONF_HIGH
PRED_COLOR_LOW = (0, 140, 255)  # BGR orange — conf <= PRED_CONF_HIGH
# Ultralytics YOLO predict default; same for all clips (not autolabel per-tile conf).
EVAL_CONF_THRESHOLD = 0.25
EVAL_VIDEOS_DIR = OUTPUTS_DIR / "eval_videos"

TILE_NMS_IOU = 0.5
PRED_NMS_IOU = 0.7
SOFT_NMS_SIGMA = 0.5
# Inner box almost fully inside a larger one: IoU is low, so NMS keeps both.
NESTED_COVER = 0.8
PREDICT_BATCH_SIZE = 8

# One eval clip per band; whole clip inherits probe distance band (no per-frame distance).
# A = <200 m, B = >200 m. JSON keys stay 0-200m / 200-400m for Round 1 compatibility.
EVAL_BAND_A = "0-200m"
EVAL_BAND_B = "200-400m"
BAND_LABELS = {
    EVAL_BAND_A: "A (<200m)",
    EVAL_BAND_B: "B (>200m)",
}
DEFAULT_BAND_CLIPS = {
    EVAL_BAND_A: "13722965_2160_3840_30fps",
    EVAL_BAND_B: "266987",
}

CLIP_EVAL_BAND = {clip: band for band, clip in DEFAULT_BAND_CLIPS.items()}

IOU_MATCH = 0.5
# COCO-style mAP@0.5:0.95 thresholds
IOU_THRESHOLDS_50_95 = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


@dataclass(frozen=True)
class TrainSliceConfig:
    """Model imgsz + per-clip crop from train_groups (tile_size / full-frame)."""

    imgsz: int
    overlap_ratio: float
    clip_slice_size: dict[str, int]
    clip_overlap: dict[str, float]
    clip_train_imgsz: dict[str, int]
    clip_uses_tiling: dict[str, bool]
    clip_group: dict[str, str]
    source: str

    def tiling_for(
        self,
        clip_name: str,
        frame_w: int,
        frame_h: int,
        *,
        predict_imgsz: int | None = None,
    ) -> tuple[int, float, int, bool]:
        """Return (slice_size, overlap, predict_imgsz, uses_tiling).

        Crop comes from train_groups; YOLO11s letterbox is TRAIN_IMGSZ unless overridden.
        """
        group = resolve_train_group_tiling(clip_name)
        slice_size = effective_slice_size(group, frame_w, frame_h)
        imgsz = TRAIN_IMGSZ if predict_imgsz is None else int(predict_imgsz)
        return slice_size, group.overlap, imgsz, group.uses_tiling


def load_train_slice_config() -> TrainSliceConfig:
    """Read imgsz / per-clip slices written by train.py (data.yaml or prepare_stats.json)."""
    if DATASET_YAML.exists():
        payload = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8")) or {}
        return _slice_config_from_payload(payload, source=str(DATASET_YAML))

    if PREPARE_STATS.exists():
        stats = json.loads(PREPARE_STATS.read_text(encoding="utf-8"))
        train = stats.get("train", {})
        val = stats.get("val", {})
        payload = {
            "imgsz": stats.get("imgsz", train.get("imgsz", TRAIN_IMGSZ)),
            "overlap_ratio": stats.get("overlap_ratio", TRAIN_OVERLAP_RATIO),
            "clip_slice_size": {
                **(train.get("clip_slice_size") or {}),
                **(val.get("clip_slice_size") or {}),
            },
            "clip_overlap": {
                **(train.get("clip_overlap") or {}),
                **(val.get("clip_overlap") or {}),
            },
            "clip_train_imgsz": {
                **(train.get("clip_train_imgsz") or {}),
                **(val.get("clip_train_imgsz") or {}),
            },
            "clip_uses_tiling": {
                **(train.get("clip_uses_tiling") or {}),
                **(val.get("clip_uses_tiling") or {}),
            },
            "clip_group": {
                **(train.get("clip_group") or {}),
                **(val.get("clip_group") or {}),
            },
        }
        return _slice_config_from_payload(payload, source=str(PREPARE_STATS))

    # Fall back to train_groups directly (before dataset prepare).
    payload = load_tiling_payload()
    groups = payload.get("train_groups") or {}
    if groups:
        return TrainSliceConfig(
            imgsz=int(payload.get("train_imgsz", TRAIN_IMGSZ)),
            overlap_ratio=float(TRAIN_OVERLAP_RATIO),
            clip_slice_size={},
            clip_overlap={},
            clip_train_imgsz={},
            clip_uses_tiling={},
            clip_group={},
            source=str(CLIP_TILING_CONFIG_PATH),
        )

    raise SystemExit(
        f"Train dataset metadata not found ({DATASET_YAML} or {PREPARE_STATS}).\n"
        "Run train.py first so eval can match training tiles:\n"
        "  python src/training/train.py --prepare-only\n"
        "  python src/training/train.py --recreate-dataset"
    )


def _slice_config_from_payload(payload: dict, *, source: str) -> TrainSliceConfig:
    imgsz = payload.get("imgsz", payload.get("slice_size", TRAIN_IMGSZ))
    overlap_ratio = payload.get("overlap_ratio", TRAIN_OVERLAP_RATIO)
    clip_slice_size = payload.get("clip_slice_size") or {}
    clip_overlap = payload.get("clip_overlap") or {}
    clip_train_imgsz = payload.get("clip_train_imgsz") or {}
    clip_uses_tiling = payload.get("clip_uses_tiling") or {}
    clip_group = payload.get("clip_group") or {}

    # Older dataset.json payloads may only have clip_slice_size / clip_scale_coeff.
    if not clip_uses_tiling and clip_slice_size:
        clip_uses_tiling = {k: True for k in clip_slice_size}
    if not clip_overlap and payload.get("clip_scale_coeff"):
        clip_overlap = {k: float(overlap_ratio) for k in clip_slice_size}

    return TrainSliceConfig(
        imgsz=int(imgsz),
        overlap_ratio=float(overlap_ratio),
        clip_slice_size={str(k): int(v) for k, v in clip_slice_size.items()},
        clip_overlap={str(k): float(v) for k, v in clip_overlap.items()},
        clip_train_imgsz={str(k): int(v) for k, v in clip_train_imgsz.items()},
        clip_uses_tiling={str(k): bool(v) for k, v in clip_uses_tiling.items()},
        clip_group={str(k): str(v) for k, v in clip_group.items()},
        source=source,
    )


def verify_materialized_tile_size(slice_cfg: TrainSliceConfig) -> None:
    """Sanity-check on-disk train tiles match per-clip slice metadata when available."""
    images_dir = DATASET_DIR / "train" / "images"
    if not images_dir.is_dir() or not slice_cfg.clip_slice_size:
        return
    for clip_name, expected in slice_cfg.clip_slice_size.items():
        if slice_cfg.clip_uses_tiling and not slice_cfg.clip_uses_tiling.get(clip_name, True):
            continue  # full-frame samples are larger than short-side metadata
        sample = next(images_dir.glob(f"{clip_name}__*.jpg"), None)
        if sample is None:
            continue
        image = cv2.imread(str(sample))
        if image is None:
            continue
        height, width = image.shape[:2]
        # Edge tiles can be smaller than slice_size near frame borders.
        if width > expected or height > expected:
            raise SystemExit(
                f"Dataset tile size mismatch: {sample.name} is {width}×{height}, "
                f"but metadata says slice≤{expected}×{expected} for {clip_name} "
                f"({slice_cfg.source}).\n"
                "Rebuild dataset: python src/training/train.py --recreate-dataset"
            )
        # One successful check is enough.
        return


@dataclass
class Box:
    xyxy: list[float]
    confidence: float = 1.0
    frame: str = ""
    clip: str = ""


@dataclass
class BandStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_frames: int = 0
    fps: float = 30.0
    first_tp_frame: int | None = None
    ap50_pairs: list[tuple[float, int]] = field(default_factory=list)  # (score, is_tp) @0.5
    # IoU thresh → (score, is_tp) pairs for mAP@0.5:0.95
    ap_pairs_by_iou: dict[float, list[tuple[float, int]]] = field(default_factory=dict)

    def detection_rate(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    def false_alarms_per_min(self) -> float | None:
        if self.n_frames <= 0:
            return None
        duration_min = self.n_frames / (self.fps * 60.0)
        return self.fp / duration_min if duration_min > 0 else None

    def time_to_first_detection_s(self) -> float | None:
        if self.first_tp_frame is None:
            return None
        return self.first_tp_frame / self.fps

    def _n_gt(self) -> int:
        return self.tp + self.fn

    def map50(self) -> float | None:
        return average_precision(self.ap50_pairs, self._n_gt())

    def map50_95(self) -> float | None:
        n_gt = self._n_gt()
        if n_gt == 0:
            return None
        aps: list[float] = []
        for thresh in IOU_THRESHOLDS_50_95:
            pairs = self.ap_pairs_by_iou.get(thresh) or []
            ap = average_precision(pairs, n_gt)
            if ap is None:
                return None
            aps.append(ap)
        return sum(aps) / len(aps) if aps else None


def average_precision(
    pairs: list[tuple[float, int]],
    n_gt: int,
) -> float | None:
    """11-point interpolated AP from (score, is_tp) pairs."""
    if not pairs or n_gt <= 0:
        return None
    preds = sorted(pairs, key=lambda item: item[0], reverse=True)
    tp_cum = 0
    fp_cum = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _, is_tp in preds:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)
    ap = 0.0
    for t in [i / 10 for i in range(0, 11)]:
        p_max = max((p for r, p in zip(recalls, precisions) if r >= t), default=0.0)
        ap += p_max / 11.0
    return ap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate detector on prepared eval packs "
            "(data/datasets/eval_autolabel or eval_manual), "
            "or --live on full videos."
        ),
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Fine-tuned YOLO weights (.pt).")
    parser.add_argument(
        "--gt",
        choices=("autolabel", "manual", "both"),
        default="autolabel",
        help="Prepared pack: autolabel, manual, or both.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override prepared pack root (images/ + labels/ + data.yaml).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Legacy: score full eval videos from data/frames + labels/eval (not prepared packs).",
    )
    parser.add_argument(
        "--clips",
        nargs="*",
        default=None,
        help="Eval clip names (default: band clips / pack clips).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"Override infer confidence (default {EVAL_CONF_THRESHOLD}, Ultralytics YOLO default).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TRAIN_IMGSZ,
        help=f"YOLO11s predict imgsz (default {TRAIN_IMGSZ}; same as train). "
        "Ignores leftover pack metadata.",
    )
    parser.add_argument("--iou", type=float, default=IOU_MATCH, help="IoU threshold for TP match.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for metrics_table.md and eval_metrics.json "
            "(default outputs/eval_autolabel or outputs/eval_manual)."
        ),
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help=(
            "Directory for overlay videos "
            f"(default: {EVAL_VIDEOS_DIR.relative_to(PROJECT_ROOT)}; "
            "one mp4 per eval clip). Ignored with --no-video."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing prediction overlay videos.",
    )
    return parser.parse_args()


def load_metadata(clip_dir: Path) -> dict:
    return json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))


def parse_yolo_label_file(label_path: Path, img_w: int, img_h: int) -> list[Box]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = x1 + bw * img_w
        y2 = y1 + bh * img_h
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(Box(xyxy=[x1, y1, x2, y2]))
    return boxes


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_area(xyxy: list[float]) -> float:
    return max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])


def _box_intersection(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def suppress_nested_boxes(boxes: list[Box], cover: float = NESTED_COVER) -> list[Box]:
    """Drop a smaller box that mostly sits inside a larger one (same car, part vs whole).

    Standard IoU NMS does not: inner/outer IoU ≈ area_inner / area_outer, often 0.1–0.3.
    """
    if len(boxes) < 2:
        return boxes
    kept: list[Box] = []
    for i, inner in enumerate(boxes):
        area_i = _box_area(inner.xyxy)
        if area_i <= 0:
            continue
        nested = False
        for j, outer in enumerate(boxes):
            if i == j:
                continue
            area_o = _box_area(outer.xyxy)
            if area_o <= area_i:
                continue
            if _box_intersection(inner.xyxy, outer.xyxy) / area_i >= cover:
                nested = True
                break
        if not nested:
            kept.append(inner)
    return kept


def resolve_clip_eval_band(clip_name: str, tile_cfg: dict) -> str | None:
    """Map clip to eval band from probe config — constant for the whole video."""
    if clip_name in CLIP_EVAL_BAND:
        return CLIP_EVAL_BAND[clip_name]

    probe_band = str(tile_cfg.get("distance_band", "")).lower()
    distance_m = tile_cfg.get("distance_m")
    if probe_band == "<200m":
        return "0-200m"
    if distance_m is not None and distance_m < 400:
        return "200-400m"
    if probe_band in (">200m", ">400m"):
        return "200-400m"
    return None


def shift_box_to_full_frame(
    xyxy: list[float],
    offset_x: int,
    offset_y: int,
    img_w: int,
    img_h: int,
) -> list[float]:
    x1, y1, x2, y2 = xyxy
    return [
        max(0.0, x1 + offset_x),
        max(0.0, y1 + offset_y),
        min(float(img_w), x2 + offset_x),
        min(float(img_h), y2 + offset_y),
    ]


def nms_boxes(boxes: list[Box], iou_thresh: float) -> list[Box]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[Box] = []
    for candidate in ordered:
        if all(box_iou(candidate.xyxy, kept_box.xyxy) < iou_thresh for kept_box in kept):
            kept.append(candidate)
    return kept


def soft_nms_boxes(
    boxes: list[Box],
    iou_thresh: float,
    *,
    sigma: float = SOFT_NMS_SIGMA,
    score_thresh: float = 0.001,
    method: str = "gaussian",
) -> list[Box]:
    """Bodla et al. Soft-NMS: decay overlapping scores instead of dropping boxes.

    Gaussian (default): ``s *= exp(-(iou²)/σ)`` for every other box.
    Linear: ``s *= (1 - iou)`` when IoU ≥ ``iou_thresh``.
    Boxes whose decayed score falls below ``score_thresh`` are removed.
    """
    if not boxes:
        return []
    remaining = [Box(xyxy=list(b.xyxy), confidence=float(b.confidence)) for b in boxes]
    kept: list[Box] = []
    method = method.lower()
    while remaining:
        remaining.sort(key=lambda b: b.confidence, reverse=True)
        best = remaining.pop(0)
        if best.confidence < score_thresh:
            break
        kept.append(best)
        nxt: list[Box] = []
        for other in remaining:
            iou = box_iou(best.xyxy, other.xyxy)
            if method == "linear":
                scale = (1.0 - iou) if iou >= iou_thresh else 1.0
            else:
                scale = math.exp(-(iou * iou) / sigma) if iou > 0 else 1.0
            new_score = other.confidence * scale
            if new_score >= score_thresh:
                nxt.append(Box(xyxy=other.xyxy, confidence=new_score))
        remaining = nxt
    return kept


def split_predict_kw(predict_kw: dict | None) -> tuple[dict, str | None, dict]:
    """Split Ultralytics predict kwargs from our NMS postprocess keys.

    When ``nms`` is set (hard / off / soft), Ultralytics NMS is disabled
    (``iou=1.0``) so all three Group C modes start from the same raw boxes.
    """
    extra = dict(predict_kw or {})
    nms_mode = extra.pop("nms", None)
    nms_iou = float(extra.pop("iou", PRED_NMS_IOU)) if nms_mode else float(extra.get("iou", PRED_NMS_IOU))
    opts = {
        "iou": nms_iou,
        "sigma": float(extra.pop("soft_nms_sigma", SOFT_NMS_SIGMA)),
        "method": str(extra.pop("soft_nms_method", "gaussian")),
    }
    if nms_mode:
        extra["iou"] = 1.0
        extra.setdefault("max_det", 300)
    return extra, (str(nms_mode).lower() if nms_mode is not None else None), opts


def apply_pred_nms(
    boxes: list[Box],
    nms_mode: str | None,
    *,
    iou: float,
    score_thresh: float,
    sigma: float = SOFT_NMS_SIGMA,
    method: str = "gaussian",
) -> list[Box]:
    if nms_mode in (None, "off", "none"):
        return boxes
    if nms_mode in ("hard", "on"):
        return nms_boxes(boxes, iou)
    if nms_mode in ("soft", "soft_nms"):
        return soft_nms_boxes(
            boxes,
            iou,
            sigma=sigma,
            score_thresh=score_thresh,
            method=method,
        )
    raise ValueError(f"Unknown predict nms={nms_mode!r} (use hard, off, soft)")


def predict_frame_tiled(
    yolo_model: YOLO,
    image_path: Path,
    slice_cfg: TrainSliceConfig,
    *,
    clip_name: str,
    img_w: int,
    img_h: int,
    conf: float,
    dev: str,
    predict_imgsz: int | None = None,
) -> list[Box]:
    """Infer with train_groups tiling (or full-frame); map boxes to full frame."""
    slice_size, overlap, predict_imgsz, uses_tiling = slice_cfg.tiling_for(
        clip_name, img_w, img_h, predict_imgsz=predict_imgsz
    )

    if not uses_tiling:
        results = yolo_model.predict(
            str(image_path),
            conf=conf,
            device=dev,
            imgsz=predict_imgsz,
            verbose=False,
        )
        merged: list[Box] = []
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                xyxy = result.boxes.xyxy[i].cpu().tolist()
                score = float(result.boxes.conf[i].cpu().item())
                x1, y1, x2, y2 = xyxy
                x1 = float(max(0, min(img_w, x1)))
                y1 = float(max(0, min(img_h, y1)))
                x2 = float(max(0, min(img_w, x2)))
                y2 = float(max(0, min(img_h, y2)))
                if x2 - x1 > 1 and y2 - y1 > 1:
                    merged.append(Box(xyxy=[x1, y1, x2, y2], confidence=score))
        return suppress_nested_boxes(nms_boxes(merged, TILE_NMS_IOU))

    slice_result = slice_image(
        str(image_path),
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        auto_slice_resolution=False,
        verbose=False,
    )
    slices = slice_result.sliced_image_list
    merged = []

    for start in range(0, len(slices), PREDICT_BATCH_SIZE):
        batch_slices = slices[start : start + PREDICT_BATCH_SIZE]
        images = [sliced.image for sliced in batch_slices]
        results = yolo_model.predict(
            images,
            conf=conf,
            device=dev,
            imgsz=predict_imgsz,
            verbose=False,
        )
        for sliced, result in zip(batch_slices, results):
            ox, oy = sliced.starting_pixel
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for i in range(len(result.boxes)):
                xyxy = result.boxes.xyxy[i].cpu().tolist()
                score = float(result.boxes.conf[i].cpu().item())
                full_xyxy = shift_box_to_full_frame(xyxy, ox, oy, img_w, img_h)
                if full_xyxy[2] - full_xyxy[0] > 1 and full_xyxy[3] - full_xyxy[1] > 1:
                    merged.append(Box(xyxy=full_xyxy, confidence=score))

    return suppress_nested_boxes(nms_boxes(merged, TILE_NMS_IOU))


def match_frame(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_thresh: float,
) -> tuple[int, int, int, list[tuple[float, int]]]:
    """Return tp, fp, fn and (score, is_tp) pairs for AP at a single IoU threshold."""
    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    ap_pairs: list[tuple[float, int]] = []

    ordered_preds = sorted(pred_boxes, key=lambda b: b.confidence, reverse=True)
    for pred in ordered_preds:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = box_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        is_tp = int(best_iou >= iou_thresh and best_j >= 0)
        ap_pairs.append((pred.confidence, is_tp))
        if is_tp:
            tp += 1
            gt_matched[best_j] = True
        else:
            fp += 1

    fn = sum(1 for matched in gt_matched if not matched)
    return tp, fp, fn, ap_pairs


def accumulate_frame_metrics(
    stats: BandStats,
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_thresh: float,
) -> tuple[int, int, int]:
    """Update band stats (counts + AP@0.5 + AP@0.5:0.95) for one sample; return tp, fp, fn."""
    tp, fp, fn, ap_pairs = match_frame(gt_boxes, pred_boxes, iou_thresh)
    stats.tp += tp
    stats.fp += fp
    stats.fn += fn
    stats.ap50_pairs.extend(ap_pairs)
    for thresh in IOU_THRESHOLDS_50_95:
        if abs(thresh - iou_thresh) < 1e-9:
            pairs = ap_pairs
        else:
            _, _, _, pairs = match_frame(gt_boxes, pred_boxes, thresh)
        stats.ap_pairs_by_iou.setdefault(thresh, []).extend(pairs)
    return tp, fp, fn


def iter_all_frames(clip_name: str) -> list[tuple[Path, int]]:
    """All extracted frames for a clip, sorted by index."""
    frame_dir = FRAMES_DIR / clip_name
    if not frame_dir.is_dir():
        return []

    pairs: list[tuple[Path, int]] = []
    for image_path in sorted(frame_dir.glob("*.jpg")):
        pairs.append((image_path, int(image_path.stem)))
    return pairs


def labeled_frame_paths(clip_name: str) -> dict[str, Path]:
    """Map frame stem -> label path for frames used in metrics."""
    label_dir = LABELS_DIR / "eval" / clip_name
    if not label_dir.is_dir():
        return {}

    mapping: dict[str, Path] = {}
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = FRAMES_DIR / clip_name / f"{label_path.stem}.jpg"
        if image_path.exists():
            mapping[label_path.stem] = label_path
    return mapping


def pred_box_color(confidence: float) -> tuple[int, int, int]:
    return PRED_COLOR_HIGH if confidence > PRED_CONF_HIGH else PRED_COLOR_LOW


def resolve_eval_conf(conf_override: float | None) -> float:
    return float(conf_override) if conf_override is not None else EVAL_CONF_THRESHOLD


def draw_vehicle_predictions(
    image: np.ndarray,
    boxes: list[Box],
    *,
    clip_band: str,
    probe_distance_m: float | None,
) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    box_t = 3 if min(w, h) >= 1080 else 2
    font = 0.8 if min(w, h) >= 1080 else 0.6
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy)
        color = pred_box_color(box.confidence)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_t)
        label = f"{VEHICLE_CLASS} {box.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            color,
            box_t,
        )
    band = BAND_LABELS.get(clip_band, clip_band)
    dist = f" (~{probe_distance_m:.0f}m probe)" if probe_distance_m is not None else ""
    cv2.putText(
        canvas,
        f"{band}{dist}",
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        font,
        (255, 255, 255),
        box_t,
    )
    return canvas


def save_prediction_video(
    preds_by_frame: dict[str, list[Box]],
    frame_paths: list[Path],
    metadata: dict,
    out_path: Path,
    *,
    clip_band: str,
    probe_distance_m: float | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(metadata["width"])
    height = int(metadata["height"])
    fps = float(metadata.get("fps", 30.0))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        writer.write(
            draw_vehicle_predictions(
                image,
                preds_by_frame.get(frame_path.stem, []),
                clip_band=clip_band,
                probe_distance_m=probe_distance_m,
            )
        )
    writer.release()


def evaluate_clip(
    clip_name: str,
    tile_config: dict[str, dict],
    slice_cfg: TrainSliceConfig,
    *,
    yolo_model: YOLO,
    conf_override: float | None,
    iou_thresh: float,
    band_stats: dict[str, BandStats],
    video_dir: Path | None = None,
    predict_imgsz: int | None = None,
) -> dict:
    tile_cfg = resolve_clip_tile_config(clip_name, tile_config)
    clip_band = resolve_clip_eval_band(clip_name, tile_cfg)
    if clip_band is None:
        raise SystemExit(
            f"Clip {clip_name!r} has no eval distance band. "
            f"Probe band={tile_cfg.get('distance_band')!r}, distance_m={tile_cfg.get('distance_m')}."
        )

    conf = resolve_eval_conf(conf_override)
    ckpt_hint = str(getattr(yolo_model, "ckpt_path", "") or "")
    dev = predict_device(ckpt_hint) if ckpt_hint else device()

    meta = load_metadata(FRAMES_DIR / clip_name)
    img_w, img_h = int(meta["width"]), int(meta["height"])
    fps = float(meta.get("fps", 30.0))
    probe_distance_m = tile_cfg.get("distance_m")
    if probe_distance_m is not None:
        probe_distance_m = float(probe_distance_m)

    slice_size, overlap, predict_imgsz, uses_tiling = slice_cfg.tiling_for(
        clip_name, img_w, img_h, predict_imgsz=predict_imgsz
    )
    group = slice_cfg.clip_group.get(clip_name) or resolve_train_group_tiling(clip_name).group
    tile_desc = "full-frame" if not uses_tiling else f"tile={slice_size}"
    print(
        f"  tiles: group={group}, {tile_desc}, predict_imgsz={predict_imgsz}, "
        f"overlap={overlap}"
    )

    labeled_frames = labeled_frame_paths(clip_name)
    all_frames = iter_all_frames(clip_name)
    if not all_frames:
        raise SystemExit(f"No frames found for clip {clip_name!r} under data/frames/.")

    preds_by_frame: dict[str, list[Box]] = {}
    clip_tp = clip_fp = clip_fn = 0
    metrics_frames = 0
    stats = band_stats[clip_band]
    total_frames = len(all_frames)

    t_infer0 = time.perf_counter()
    for frame_no, (image_path, frame_idx) in enumerate(all_frames, start=1):
        preds = predict_frame_tiled(
            yolo_model,
            image_path,
            slice_cfg,
            clip_name=clip_name,
            img_w=img_w,
            img_h=img_h,
            conf=conf,
            dev=dev,
            predict_imgsz=predict_imgsz,
        )
        preds_by_frame[image_path.stem] = preds

        if frame_no % 25 == 0 or frame_no == total_frames:
            print(f"  {clip_name}: {frame_no}/{total_frames} frames")

        label_path = labeled_frames.get(image_path.stem)
        if label_path is None:
            continue

        metrics_frames += 1
        gt_all = parse_yolo_label_file(label_path, img_w, img_h)
        if not gt_all and not preds:
            continue

        stats.n_frames += 1
        stats.fps = fps

        tp, fp, fn = accumulate_frame_metrics(stats, gt_all, preds, iou_thresh)
        clip_tp += tp
        clip_fp += fp
        clip_fn += fn

        if tp > 0 and stats.first_tp_frame is None:
            stats.first_tp_frame = frame_idx

    infer_sec = time.perf_counter() - t_infer0
    frames_per_sec = (total_frames / infer_sec) if infer_sec > 0 else None

    video_path: Path | None = None
    video_sec = 0.0
    if video_dir is not None:
        t_vid0 = time.perf_counter()
        video_path = video_dir / f"{clip_name}_predictions.mp4"
        save_prediction_video(
            preds_by_frame,
            [path for path, _ in all_frames],
            meta,
            video_path,
            clip_band=clip_band,
            probe_distance_m=probe_distance_m,
        )
        video_sec = time.perf_counter() - t_vid0

    clip_elapsed_sec = round(infer_sec + video_sec, 2)
    fps_bit = f", {frames_per_sec:.2f} frame/s" if frames_per_sec is not None else ""
    vid_bit = f", video {video_sec:.1f}s" if video_dir is not None else ""
    print(f"  timing: {clip_elapsed_sec:.1f}s total (infer {infer_sec:.1f}s{fps_bit}{vid_bit})")

    return {
        "clip": clip_name,
        "eval_band": clip_band,
        "probe_distance_m": probe_distance_m,
        "probe_distance_band": tile_cfg.get("distance_band"),
        "frames": metrics_frames,
        "video_frames": len(all_frames),
        "conf": conf,
        "timing": {
            "elapsed_sec": clip_elapsed_sec,
            "infer_sec": round(infer_sec, 2),
            "video_sec": round(video_sec, 2),
            "frames_per_sec": round(frames_per_sec, 3) if frames_per_sec is not None else None,
        },
        "eval_mode": {
            "group": group,
            "slice_size": slice_size if uses_tiling else None,
            "full_frame": not uses_tiling,
            "predict_imgsz": predict_imgsz,
            "imgsz": slice_cfg.imgsz,
            "overlap_ratio": overlap,
            "tile_nms_iou": TILE_NMS_IOU,
            "predict_batch_size": PREDICT_BATCH_SIZE,
            "train_metadata": slice_cfg.source,
        },
        "tp": clip_tp,
        "fp": clip_fp,
        "fn": clip_fn,
        "prediction_video": str(video_path) if video_path else None,
    }


def format_metric(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "-"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def build_markdown_table(band_stats: dict[str, BandStats], meta: dict) -> str:
    eval_mode = meta.get("eval_mode", {})
    assumptions = meta.get("assumptions") or {}
    gt_line = assumptions.get(
        "gt_source",
        "pseudo-labels from `labels/eval/` (class 0 = vehicle).",
    )
    infer_line = assumptions.get(
        "inference",
        (
            f"per-clip `train_groups` tiling (or full-frame) → "
            f"model imgsz={eval_mode.get('imgsz')} (from `{eval_mode.get('train_metadata', 'train dataset')}`), "
            f"tile NMS IoU {eval_mode.get('tile_nms_iou', TILE_NMS_IOU)}, boxes mapped to full frame."
        ),
    )
    band_line = assumptions.get(
        "band_assignment",
        "one eval clip per band; distance band is fixed per whole video from probe "
        "(`clip_tiling.json`), not recomputed per frame or per car.",
    )
    lines = [
        "# Eval metrics by distance band",
        "",
        f"Model: `{meta['weights']}`",
        "",
        "## Assumptions",
        "",
        f"- GT: {gt_line}",
        f"- Inference: {infer_line}",
        f"- Bands: {band_line}",
        f"- Match: IoU ≥ {meta['iou']} → TP; unmatched GT → FN; unmatched pred → FP.",
        f"- False alarms/min: `FP / (N_frames / fps / 60)`.",
        f"- Time to first detection: first TP frame index / fps (seconds from clip start).",
        f"- Eval clips: {', '.join(meta['clips'])}",
        "",
    ]
    timing = meta.get("timing") or {}
    if timing:
        lines.extend(
            [
                "## Timing",
                "",
                f"- Wall time: **{timing.get('elapsed_human', timing.get('elapsed_sec'))}** "
                f"({timing.get('elapsed_sec')}s).",
            ]
        )
        for item in timing.get("clips", []):
            fps = item.get("frames_per_sec")
            fps_s = f", {fps:.2f} frame/s" if fps is not None else ""
            vid = item.get("video_sec") or 0
            vid_s = f", video {vid}s" if vid else ""
            lines.append(
                f"- `{item.get('clip')}`: {item.get('elapsed_sec')}s "
                f"(infer {item.get('infer_sec')}s{fps_s}{vid_s})"
            )
        lines.extend(["", f"| Metric | {BAND_LABELS[EVAL_BAND_A]} | {BAND_LABELS[EVAL_BAND_B]} |"])
    else:
        lines.extend(
            [
                f"| Metric | {BAND_LABELS[EVAL_BAND_A]} | {BAND_LABELS[EVAL_BAND_B]} |",
            ]
        )
    lines.extend(
        [
            "|--------|---------|-----------|",
            f"| Detection rate TP/(TP+FN) | {format_metric(band_stats[EVAL_BAND_A].detection_rate())} | "
            f"{format_metric(band_stats[EVAL_BAND_B].detection_rate())} |",
            f"| Precision TP/(TP+FP) | {format_metric(band_stats[EVAL_BAND_A].precision())} | "
            f"{format_metric(band_stats[EVAL_BAND_B].precision())} |",
            f"| False alarms / min | "
            f"{format_metric(band_stats[EVAL_BAND_A].false_alarms_per_min(), pct=False)} | "
            f"{format_metric(band_stats[EVAL_BAND_B].false_alarms_per_min(), pct=False)} |",
            f"| Time to first detection (s) | "
            f"{format_metric(band_stats[EVAL_BAND_A].time_to_first_detection_s(), pct=False)} | "
            f"{format_metric(band_stats[EVAL_BAND_B].time_to_first_detection_s(), pct=False)} |",
            f"| mAP@0.5 | {format_metric(band_stats[EVAL_BAND_A].map50())} | "
            f"{format_metric(band_stats[EVAL_BAND_B].map50())} |",
            f"| mAP@0.5:0.95 | {format_metric(band_stats[EVAL_BAND_A].map50_95())} | "
            f"{format_metric(band_stats[EVAL_BAND_B].map50_95())} |",
            "",
            "## Per-band counts",
            "",
        ]
    )
    for band, stats in band_stats.items():
        lines.append(
            f"**{band}**: TP={stats.tp}, FP={stats.fp}, FN={stats.fn}, "
            f"frames_with_activity={stats.n_frames}, fps={stats.fps}"
        )
    return "\n".join(lines) + "\n"


def _clip_from_sample_stem(stem: str) -> str:
    return stem.split("__", 1)[0]


def _frame_idx_from_sample_stem(stem: str) -> int | None:
    """Best-effort frame index from `clip__000001` or `clip__000001_x_y_w_h`."""
    if "__" not in stem:
        return None
    rest = stem.split("__", 1)[1]
    parts = rest.split("_")
    if len(parts) >= 5 and all(p.isdigit() for p in parts[-4:]):
        frame = "_".join(parts[:-4])
    else:
        frame = parts[0] if parts else rest
    try:
        return int(frame)
    except ValueError:
        return None


def predict_prepared_image(
    yolo_model: YOLO,
    image_path: Path,
    *,
    img_w: int,
    img_h: int,
    predict_imgsz: int,
    conf: float,
    dev: str,
    predict_kw: dict | None = None,
) -> list[Box]:
    """Predict on an already-cropped pack image (tile or full-frame sample)."""
    extra, nms_mode, nms_opts = split_predict_kw(predict_kw)
    imgsz = int(extra.pop("imgsz", predict_imgsz))
    results = yolo_model.predict(
        str(image_path),
        conf=conf,
        device=dev,
        imgsz=imgsz,
        verbose=False,
        **extra,
    )
    merged: list[Box] = []
    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            xyxy = result.boxes.xyxy[i].cpu().tolist()
            score = float(result.boxes.conf[i].cpu().item())
            x1 = float(max(0, min(img_w, xyxy[0])))
            y1 = float(max(0, min(img_h, xyxy[1])))
            x2 = float(max(0, min(img_w, xyxy[2])))
            y2 = float(max(0, min(img_h, xyxy[3])))
            if x2 - x1 > 1 and y2 - y1 > 1:
                merged.append(Box(xyxy=[x1, y1, x2, y2], confidence=score))
    boxes = apply_pred_nms(
        merged,
        nms_mode,
        iou=float(nms_opts["iou"]),
        score_thresh=conf,
        sigma=float(nms_opts["sigma"]),
        method=str(nms_opts["method"]),
    )
    if nms_mode not in ("off", "none"):
        boxes = suppress_nested_boxes(boxes)
    return boxes


def _clip_fps(clip_name: str) -> float:
    meta_path = FRAMES_DIR / clip_name / "metadata.json"
    if meta_path.exists():
        try:
            return float(json.loads(meta_path.read_text(encoding="utf-8")).get("fps", 30.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 30.0


def _write_eval_outputs(
    *,
    out_dir: Path,
    band_stats: dict[str, BandStats],
    payload: dict,
    clip_results: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_metrics.json"
    md_path = out_dir / "metrics_table.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown_table(band_stats, payload), encoding="utf-8")

    print(f"\n{'Metric':<32} {'0-200 m':>10} {'200-400 m':>10}")
    print("-" * 54)
    rows = [
        ("Detection rate", "detection_rate", True),
        ("Precision", "precision", True),
        ("False alarms/min", "false_alarms_per_min", False),
        ("Time to first det (s)", "time_to_first_detection_s", False),
        ("mAP@0.5", "map50", True),
        ("mAP@0.5:0.95", "map50_95", True),
    ]
    for label, key, pct in rows:
        getter = {
            "detection_rate": lambda s: s.detection_rate(),
            "precision": lambda s: s.precision(),
            "false_alarms_per_min": lambda s: s.false_alarms_per_min(),
            "time_to_first_detection_s": lambda s: s.time_to_first_detection_s(),
            "map50": lambda s: s.map50(),
            "map50_95": lambda s: s.map50_95(),
        }[key]
        a, b = getter(band_stats["0-200m"]), getter(band_stats["200-400m"])
        print(f"{label:<32} {format_metric(a, pct):>10} {format_metric(b, pct):>10}")

    timing = payload.get("timing") or {}
    print(f"\nEval wall time: {timing.get('elapsed_human')} ({timing.get('elapsed_sec')}s)")
    for item in timing.get("clips", []):
        fps = item.get("frames_per_sec")
        fps_s = f", {fps:.2f} frame/s" if fps is not None else ""
        vid = item.get("video_sec") or 0
        vid_s = f", video {vid:.1f}s" if vid else ""
        print(
            f"  {item['clip']}: {item.get('elapsed_sec', 0):.1f}s "
            f"(infer {item.get('infer_sec', 0):.1f}s{fps_s}{vid_s})"
        )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    for result in clip_results:
        video = result.get("prediction_video")
        if video:
            print(f"Saved: {video}")


def evaluate_prepared_pack(
    dataset_dir: Path,
    *,
    weights: Path,
    gt_name: str,
    conf_override: float | None,
    iou_thresh: float,
    clip_filter: list[str] | None,
    output_dir: Path,
    tile_config: dict[str, dict],
    predict_kw: dict | None = None,
    video_dir: Path | None = None,
) -> None:
    """Score a frozen eval pack (images already tiled / cropped)."""
    yaml_path = dataset_dir / "data.yaml"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not yaml_path.exists() or not images_dir.is_dir():
        raise SystemExit(
            f"Prepared pack incomplete: {dataset_dir}\n"
            "Build with: python src/training/prepare_eval.py"
            + (" --from-autolabel" if "autolabel" in gt_name else "")
        )

    payload_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    bands_map: dict[str, str] = {
        str(k): str(v) for k, v in (payload_yaml.get("bands") or {}).items()
    }
    default_imgsz = TRAIN_IMGSZ
    if predict_kw and predict_kw.get("imgsz") is not None:
        default_imgsz = int(predict_kw["imgsz"])
    pack_clips = [str(c) for c in (payload_yaml.get("clips") or [])]
    if clip_filter:
        clip_names = clip_filter
    elif pack_clips:
        clip_names = pack_clips
    else:
        clip_names = list(DEFAULT_BAND_CLIPS.values())

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"No images in {images_dir}")

    clip_set = set(clip_names)
    by_clip: dict[str, list[Path]] = {c: [] for c in clip_names}
    for path in image_paths:
        clip = _clip_from_sample_stem(path.stem)
        if clip in clip_set:
            by_clip.setdefault(clip, []).append(path)

    missing = [c for c in clip_names if not by_clip.get(c)]
    if missing:
        raise SystemExit(f"Pack {dataset_dir} has no images for clips: {missing}")

    print(
        f"Eval pack [{gt_name}]: {dataset_dir} "
        f"({len(image_paths)} images, clips={clip_names}, predict_imgsz={default_imgsz})"
    )
    yolo_model = load_ultralytics_model(weights)
    dev = predict_device(str(weights))
    if dev != device():
        print(f"Eval on {dev} (Apple Silicon fallback for this architecture)")
    band_stats = {band: BandStats() for band in DEFAULT_BAND_CLIPS}
    clip_results: list[dict] = []
    run_t0 = time.perf_counter()

    for clip_name in clip_names:
        paths = by_clip[clip_name]
        tile_cfg = resolve_clip_tile_config(clip_name, tile_config) if clip_name in tile_config else {}
        clip_band = bands_map.get(clip_name) or resolve_clip_eval_band(clip_name, tile_cfg)
        if clip_band not in band_stats:
            raise SystemExit(
                f"Clip {clip_name!r} band {clip_band!r} not in {list(band_stats)}"
            )
        conf = resolve_eval_conf(conf_override)
        predict_imgsz = default_imgsz
        fps = _clip_fps(clip_name)
        stats = band_stats[clip_band]
        clip_tp = clip_fp = clip_fn = 0
        print(
            f"Evaluating {clip_name} ({clip_band}, {len(paths)} samples, "
            f"predict_imgsz={predict_imgsz}, conf={conf})..."
        )
        t_infer0 = time.perf_counter()
        for i, image_path in enumerate(paths, start=1):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            preds = predict_prepared_image(
                yolo_model,
                image_path,
                img_w=img_w,
                img_h=img_h,
                predict_imgsz=predict_imgsz,
                conf=conf,
                dev=dev,
                predict_kw=predict_kw,
            )
            label_path = labels_dir / f"{image_path.stem}.txt"
            gt_all = parse_yolo_label_file(label_path, img_w, img_h)
            if not gt_all and not preds:
                continue
            stats.n_frames += 1
            stats.fps = fps
            tp, fp, fn = accumulate_frame_metrics(stats, gt_all, preds, iou_thresh)
            clip_tp += tp
            clip_fp += fp
            clip_fn += fn
            if tp > 0 and stats.first_tp_frame is None:
                frame_idx = _frame_idx_from_sample_stem(image_path.stem)
                stats.first_tp_frame = frame_idx if frame_idx is not None else (i - 1)
            if i % 25 == 0 or i == len(paths):
                print(f"  {clip_name}: {i}/{len(paths)} samples")
        infer_sec = time.perf_counter() - t_infer0
        frames_per_sec = (len(paths) / infer_sec) if infer_sec > 0 else None
        print(
            f"  timing: {infer_sec:.1f}s"
            + (f", {frames_per_sec:.2f} sample/s" if frames_per_sec else "")
        )
        clip_results.append(
            {
                "clip": clip_name,
                "eval_band": clip_band,
                "samples": len(paths),
                "conf": conf,
                "tp": clip_tp,
                "fp": clip_fp,
                "fn": clip_fn,
                "prediction_video": None,
                "timing": {
                    "elapsed_sec": round(infer_sec, 2),
                    "infer_sec": round(infer_sec, 2),
                    "video_sec": 0.0,
                    "frames_per_sec": round(frames_per_sec, 3) if frames_per_sec else None,
                },
                "eval_mode": {
                    "prepared_pack": True,
                    "predict_imgsz": predict_imgsz,
                    "imgsz": default_imgsz,
                },
            }
        )

    if video_dir is not None:
        print(f"Overlay videos → {video_dir} ({len(clip_names)} clips)")
        slice_cfg = load_train_slice_config()
        dummy_stats = {band: BandStats() for band in DEFAULT_BAND_CLIPS}
        by_id = {r["clip"]: r for r in clip_results}
        for clip_name in clip_names:
            if is_clip_skipped(clip_name, tile_config):
                print(f"  skip overlay {clip_name}: {clip_skip_reason(clip_name, tile_config)}")
                continue
            row = evaluate_clip(
                clip_name,
                tile_config,
                slice_cfg,
                yolo_model=yolo_model,
                conf_override=conf_override,
                iou_thresh=iou_thresh,
                band_stats=dummy_stats,
                video_dir=video_dir,
                predict_imgsz=default_imgsz,
            )
            target = by_id.get(clip_name)
            if target is None:
                continue
            target["prediction_video"] = row.get("prediction_video")
            timing = target.setdefault("timing", {})
            timing["video_sec"] = (row.get("timing") or {}).get("video_sec") or 0.0

    run_elapsed_sec = round(time.perf_counter() - run_t0, 2)
    source = payload_yaml.get("source") or gt_name
    labels_root = payload_yaml.get("labels_root") or ""
    payload = {
        "weights": str(weights),
        "iou": iou_thresh,
        "clips": clip_names,
        "dataset_dir": str(dataset_dir),
        "gt": gt_name,
        "timing": {
            "elapsed_sec": run_elapsed_sec,
            "elapsed_human": (
                f"{int(run_elapsed_sec // 60)}m {run_elapsed_sec % 60:.1f}s"
                if run_elapsed_sec >= 60
                else f"{run_elapsed_sec:.1f}s"
            ),
            "clips": [
                {"clip": r["clip"], **(r.get("timing") or {})} for r in clip_results
            ],
        },
        "eval_mode": {
            "mode": "prepared_pack",
            "imgsz": default_imgsz,
            "clip_uses_tiling": payload_yaml.get("clip_uses_tiling") or {},
            "clip_group": payload_yaml.get("clip_group") or {},
            "train_metadata": str(yaml_path),
            "predict_kw": dict(predict_kw or {}),
            "band_labels": dict(BAND_LABELS),
        },
        "assumptions": {
            "gt_source": (
                f"prepared pack `{dataset_dir.name}/` "
                f"(source={source}"
                + (f", labels_root={labels_root}" if labels_root else "")
                + ")"
            ),
            "band_assignment": "from pack data.yaml `bands` (clip → distance band)",
            "eval_clips_by_band": bands_map,
            "false_alarms_formula": "FP / (N_pack_samples / fps / 60)",
            "inference": (
                "direct YOLO predict on pack images (already tiled/cropped); "
                f"YOLO11s predict_imgsz={default_imgsz} (train default {TRAIN_IMGSZ})"
            ),
        },
        "bands": {
            band: {
                "tp": s.tp,
                "fp": s.fp,
                "fn": s.fn,
                "n_frames": s.n_frames,
                "fps": s.fps,
                "detection_rate": s.detection_rate(),
                "precision": s.precision(),
                "false_alarms_per_min": s.false_alarms_per_min(),
                "time_to_first_detection_s": s.time_to_first_detection_s(),
                "map50": s.map50(),
                "map50_95": s.map50_95(),
            }
            for band, s in band_stats.items()
        },
        "clips_detail": clip_results,
    }
    del yolo_model
    _write_eval_outputs(
        out_dir=output_dir,
        band_stats=band_stats,
        payload=payload,
        clip_results=clip_results,
    )


def evaluate_live(
    *,
    weights: Path,
    conf_override: float | None,
    iou_thresh: float,
    clip_names: list[str],
    output_dir: Path,
    video_dir: Path | None,
    predict_imgsz: int = TRAIN_IMGSZ,
) -> None:
    """Legacy: full-frame video eval from data/frames + labels/eval."""
    split_map = build_split_map()
    tile_config = load_clip_tile_config()
    slice_cfg = load_train_slice_config()
    verify_materialized_tile_size(slice_cfg)
    print(
        f"Eval (live): predict_imgsz={predict_imgsz}, "
        f"per-clip train_groups tiling (from {slice_cfg.source})"
    )
    if slice_cfg.clip_group:
        for name in sorted(slice_cfg.clip_group):
            group = slice_cfg.clip_group[name]
            uses = slice_cfg.clip_uses_tiling.get(name, True)
            size = slice_cfg.clip_slice_size.get(name, "?")
            ov = slice_cfg.clip_overlap.get(name, "?")
            tile_desc = "full-frame" if not uses else f"tile={size}"
            print(f"  {name}: group={group}, {tile_desc}, overlap={ov}, predict_imgsz={predict_imgsz}")

    for clip in clip_names:
        if split_map.get(clip) != "eval":
            raise SystemExit(f"Clip {clip!r} is not in data/eval.")
        if clip not in tile_config:
            raise SystemExit(
                f"Clip {clip!r} missing from config/clip_tiling.json. Run data/preprocess_clips.py."
            )
        if is_clip_skipped(clip, tile_config):
            raise SystemExit(
                f"Clip {clip!r} is skipped: {clip_skip_reason(clip, tile_config)}"
            )

    band_stats = {band: BandStats() for band in DEFAULT_BAND_CLIPS}
    yolo_model = load_ultralytics_model(weights)
    clip_results = []
    run_t0 = time.perf_counter()
    for clip_name in clip_names:
        print(f"Evaluating {clip_name}...")
        clip_results.append(
            evaluate_clip(
                clip_name,
                tile_config,
                slice_cfg,
                yolo_model=yolo_model,
                conf_override=conf_override,
                iou_thresh=iou_thresh,
                band_stats=band_stats,
                video_dir=video_dir,
                predict_imgsz=predict_imgsz,
            )
        )
    run_elapsed_sec = round(time.perf_counter() - run_t0, 2)
    payload = {
        "weights": str(weights),
        "iou": iou_thresh,
        "clips": clip_names,
        "timing": {
            "elapsed_sec": run_elapsed_sec,
            "elapsed_human": (
                f"{int(run_elapsed_sec // 60)}m {run_elapsed_sec % 60:.1f}s"
                if run_elapsed_sec >= 60
                else f"{run_elapsed_sec:.1f}s"
            ),
            "clips": [
                {"clip": r["clip"], **(r.get("timing") or {})} for r in clip_results
            ],
        },
        "eval_mode": {
            "mode": "live",
            "imgsz": predict_imgsz,
            "overlap_ratio": slice_cfg.overlap_ratio,
            "clip_slice_size": slice_cfg.clip_slice_size,
            "clip_overlap": slice_cfg.clip_overlap,
            "clip_uses_tiling": slice_cfg.clip_uses_tiling,
            "clip_group": slice_cfg.clip_group,
            "tile_nms_iou": TILE_NMS_IOU,
            "predict_batch_size": PREDICT_BATCH_SIZE,
            "train_metadata": slice_cfg.source,
        },
        "prediction_videos": [
            r["prediction_video"] for r in clip_results if r.get("prediction_video")
        ],
        "assumptions": {
            "gt_source": "labels/eval (live frames; not a prepared pack)",
            "band_assignment": "whole clip from probe distance_band in clip_tiling.json",
            "eval_clips_by_band": DEFAULT_BAND_CLIPS,
            "false_alarms_formula": "FP / (N_frames / fps / 60)",
            "inference": (
                f"per-clip train_groups tile_size (or full-frame), "
                f"YOLO11s predict_imgsz={predict_imgsz}, "
                "merged to full frame"
            ),
        },
        "bands": {
            band: {
                "tp": s.tp,
                "fp": s.fp,
                "fn": s.fn,
                "n_frames": s.n_frames,
                "fps": s.fps,
                "detection_rate": s.detection_rate(),
                "precision": s.precision(),
                "false_alarms_per_min": s.false_alarms_per_min(),
                "time_to_first_detection_s": s.time_to_first_detection_s(),
                "map50": s.map50(),
                "map50_95": s.map50_95(),
            }
            for band, s in band_stats.items()
        },
        "clips_detail": clip_results,
    }
    _write_eval_outputs(
        out_dir=output_dir,
        band_stats=band_stats,
        payload=payload,
        clip_results=clip_results,
    )


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first: python src/training/train.py"
        )

    if args.live:
        clip_names = args.clips or list(DEFAULT_BAND_CLIPS.values())
        out_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / "eval_live"
        video_dir = None if args.no_video else Path(args.video_dir) if args.video_dir else EVAL_VIDEOS_DIR
        evaluate_live(
            weights=weights,
            conf_override=args.conf,
            iou_thresh=args.iou,
            clip_names=clip_names,
            output_dir=out_dir,
            video_dir=video_dir,
            predict_imgsz=args.imgsz,
        )
        return

    tile_config = load_clip_tile_config()
    video_dir = None if args.no_video else Path(args.video_dir) if args.video_dir else EVAL_VIDEOS_DIR
    if args.dataset:
        gt_name = args.gt if args.gt != "both" else "custom"
        out_dir = (
            Path(args.output_dir)
            if args.output_dir
            else OUTPUTS_DIR / f"eval_{Path(args.dataset).name}"
        )
        evaluate_prepared_pack(
            Path(args.dataset),
            weights=weights,
            gt_name=gt_name,
            conf_override=args.conf,
            iou_thresh=args.iou,
            clip_filter=args.clips,
            output_dir=out_dir,
            tile_config=tile_config,
            predict_kw={"imgsz": args.imgsz},
            video_dir=video_dir,
        )
        return

    targets = ("autolabel", "manual") if args.gt == "both" else (args.gt,)
    for gt_name in targets:
        pack = EVAL_PACKS[gt_name]
        if args.output_dir and len(targets) == 1:
            out_dir = Path(args.output_dir)
        else:
            out_dir = OUTPUTS_DIR / f"eval_{gt_name}"
        evaluate_prepared_pack(
            pack,
            weights=weights,
            gt_name=gt_name,
            conf_override=args.conf,
            iou_thresh=args.iou,
            clip_filter=args.clips,
            output_dir=out_dir,
            tile_config=tile_config,
            predict_kw={"imgsz": args.imgsz},
            video_dir=video_dir,
        )


if __name__ == "__main__":
    main()
