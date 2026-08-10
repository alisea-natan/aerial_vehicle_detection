#!/usr/bin/env python3
"""Evaluate fine-tuned detector on eval clips; report metrics per distance band (0-200 m, 200-400 m)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from sahi.slicing import slice_image
from ultralytics import YOLO

from config import (
    CLIP_TILING_CONFIG_PATH,
    FRAMES_DIR,
    LABELS_DIR,
    PROJECT_ROOT,
    TRAIN_IMGSZ,
    TRAIN_OVERLAP_RATIO,
    build_split_map,
    effective_slice_size,
    load_clip_tile_config,
    load_tiling_payload,
    resolve_clip_tile_config,
    resolve_train_group_tiling,
)
from detect import device

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_DIR = OUTPUTS_DIR / "dataset"
DATASET_YAML = DATASET_DIR / "data.yaml"
PREPARE_STATS = DATASET_DIR / "prepare_stats.json"
DEFAULT_WEIGHTS = PROJECT_ROOT / "outputs" / "runs" / "yolo11s_vehicle" / "weights" / "best.pt"
VEHICLE_CLASS = "vehicle"
PRED_COLOR = (0, 200, 0)  # BGR green

TILE_NMS_IOU = 0.5
PREDICT_BATCH_SIZE = 8

# One eval clip per band; whole clip inherits probe distance band (no per-frame distance).
DEFAULT_BAND_CLIPS = {
    "0-200m": "13722965_2160_3840_30fps",
    "200-400m": "266987",
}

CLIP_EVAL_BAND = {clip: band for band, clip in DEFAULT_BAND_CLIPS.items()}

IOU_MATCH = 0.5


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

    def tiling_for(self, clip_name: str, frame_w: int, frame_h: int) -> tuple[int, float, int, bool]:
        """Return (slice_size, overlap, predict_imgsz, uses_tiling) from train_groups."""
        group = resolve_train_group_tiling(clip_name)
        slice_size = effective_slice_size(group, frame_w, frame_h)
        return slice_size, group.overlap, group.train_imgsz, group.uses_tiling


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
        "  python src/train.py --prepare-only\n"
        "  python src/train.py --recreate-dataset"
    )


def _slice_config_from_payload(payload: dict, *, source: str) -> TrainSliceConfig:
    imgsz = payload.get("imgsz", payload.get("slice_size", TRAIN_IMGSZ))
    overlap_ratio = payload.get("overlap_ratio", TRAIN_OVERLAP_RATIO)
    clip_slice_size = payload.get("clip_slice_size") or {}
    clip_overlap = payload.get("clip_overlap") or {}
    clip_train_imgsz = payload.get("clip_train_imgsz") or {}
    clip_uses_tiling = payload.get("clip_uses_tiling") or {}
    clip_group = payload.get("clip_group") or {}

    # Legacy scale_coeff datasets: synthesize uses_tiling=True for known slice sizes.
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
                "Rebuild dataset: python src/train.py --recreate-dataset"
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
    ap50_pairs: list[tuple[float, int]] = field(default_factory=list)  # (score, is_tp)

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

    def map50(self) -> float | None:
        if not self.ap50_pairs:
            return None
        preds = sorted(self.ap50_pairs, key=lambda item: item[0], reverse=True)
        tp_cum = 0
        fp_cum = 0
        precisions: list[float] = []
        recalls: list[float] = []
        n_gt = self.tp + self.fn
        if n_gt == 0:
            return None
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
    parser = argparse.ArgumentParser(description="Evaluate detector metrics per distance band.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Fine-tuned YOLO weights (.pt).")
    parser.add_argument(
        "--clips",
        nargs="*",
        default=None,
        help="Eval clip names (default: one <200m + one 200-400m clip).",
    )
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold.")
    parser.add_argument("--iou", type=float, default=IOU_MATCH, help="IoU threshold for TP match.")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUTS_DIR),
        help="Directory for metrics_table.md and eval_metrics.json.",
    )
    parser.add_argument(
        "--video-dir",
        default=str(OUTPUTS_DIR / "eval_videos"),
        help="Directory for per-clip prediction overlay videos.",
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
) -> list[Box]:
    """Infer with train_groups tiling (or full-frame); map boxes to full frame."""
    slice_size, overlap, predict_imgsz, uses_tiling = slice_cfg.tiling_for(
        clip_name, img_w, img_h
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
        return nms_boxes(merged, TILE_NMS_IOU)

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

    return nms_boxes(merged, TILE_NMS_IOU)


def match_frame(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_thresh: float,
) -> tuple[int, int, int, list[tuple[float, int]]]:
    """Return tp, fp, fn and (score, is_tp) pairs for AP."""
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


def draw_vehicle_predictions(
    image: np.ndarray,
    boxes: list[Box],
    *,
    clip_band: str,
    probe_distance_m: float | None,
) -> np.ndarray:
    canvas = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PRED_COLOR, 2)
        label = f"{VEHICLE_CLASS} {box.confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            PRED_COLOR,
            2,
        )
    if boxes and probe_distance_m is not None:
        cv2.putText(
            canvas,
            f"clip {clip_band} (~{probe_distance_m:.0f}m probe)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            PRED_COLOR,
            2,
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
) -> dict:
    tile_cfg = resolve_clip_tile_config(clip_name, tile_config)
    clip_band = resolve_clip_eval_band(clip_name, tile_cfg)
    if clip_band is None:
        raise SystemExit(
            f"Clip {clip_name!r} has no eval distance band. "
            f"Probe band={tile_cfg.get('distance_band')!r}, distance_m={tile_cfg.get('distance_m')}."
        )

    conf = conf_override if conf_override is not None else tile_cfg["label_confidence_threshold"]
    dev = device()

    meta = load_metadata(FRAMES_DIR / clip_name)
    img_w, img_h = int(meta["width"]), int(meta["height"])
    fps = float(meta.get("fps", 30.0))
    probe_distance_m = tile_cfg.get("distance_m")
    if probe_distance_m is not None:
        probe_distance_m = float(probe_distance_m)

    slice_size, overlap, predict_imgsz, uses_tiling = slice_cfg.tiling_for(
        clip_name, img_w, img_h
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

        tp, fp, fn, ap_pairs = match_frame(gt_all, preds, iou_thresh)
        stats.tp += tp
        stats.fp += fp
        stats.fn += fn
        stats.ap50_pairs.extend(ap_pairs)
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
    lines = [
        "# Eval metrics by distance band",
        "",
        f"Generated: {meta['generated_at']}",
        f"Model: `{meta['weights']}`",
        "",
        "## Assumptions",
        "",
        f"- GT: pseudo-labels from `labels/eval/` (class 0 = vehicle).",
        f"- Inference: per-clip `train_groups` tiling (or full-frame) → "
        f"model imgsz={eval_mode.get('imgsz')} (from `{eval_mode.get('train_metadata', 'train dataset')}`), "
        f"tile NMS IoU {eval_mode.get('tile_nms_iou', TILE_NMS_IOU)}, boxes mapped to full frame.",
        f"- Bands: one eval clip per band; distance band is fixed per whole video from probe "
        f"(`clip_tiling.json`), not recomputed per frame or per car.",
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
        lines.extend(["", "| Metric | 0-200 m | 200-400 m |"])
    else:
        lines.extend(
            [
                "| Metric | 0-200 m | 200-400 m |",
            ]
        )
    lines.extend(
        [
            "|--------|---------|-----------|",
            f"| Detection rate TP/(TP+FN) | {format_metric(band_stats['0-200m'].detection_rate())} | "
            f"{format_metric(band_stats['200-400m'].detection_rate())} |",
            f"| Precision TP/(TP+FP) | {format_metric(band_stats['0-200m'].precision())} | "
            f"{format_metric(band_stats['200-400m'].precision())} |",
            f"| False alarms / min | "
            f"{format_metric(band_stats['0-200m'].false_alarms_per_min(), pct=False)} | "
            f"{format_metric(band_stats['200-400m'].false_alarms_per_min(), pct=False)} |",
            f"| Time to first detection (s) | "
            f"{format_metric(band_stats['0-200m'].time_to_first_detection_s(), pct=False)} | "
            f"{format_metric(band_stats['200-400m'].time_to_first_detection_s(), pct=False)} |",
            f"| mAP@0.5 (bonus) | {format_metric(band_stats['0-200m'].map50())} | "
            f"{format_metric(band_stats['200-400m'].map50())} |",
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


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"Weights not found: {weights}\n"
            "Train first: python src/train.py"
        )

    split_map = build_split_map()
    tile_config = load_clip_tile_config()
    slice_cfg = load_train_slice_config()
    verify_materialized_tile_size(slice_cfg)
    print(
        f"Eval: default_imgsz={slice_cfg.imgsz}, "
        f"per-clip train_groups tiling (from {slice_cfg.source})"
    )
    if slice_cfg.clip_group:
        for name in sorted(slice_cfg.clip_group):
            group = slice_cfg.clip_group[name]
            uses = slice_cfg.clip_uses_tiling.get(name, True)
            size = slice_cfg.clip_slice_size.get(name, "?")
            ov = slice_cfg.clip_overlap.get(name, "?")
            p_imgsz = slice_cfg.clip_train_imgsz.get(name, slice_cfg.imgsz)
            tile_desc = "full-frame" if not uses else f"tile={size}"
            print(f"  {name}: group={group}, {tile_desc}, overlap={ov}, predict_imgsz={p_imgsz}")

    if args.clips:
        clip_names = args.clips
    else:
        clip_names = list(DEFAULT_BAND_CLIPS.values())

    for clip in clip_names:
        if split_map.get(clip) != "eval":
            raise SystemExit(f"Clip {clip!r} is not in data/eval.")
        if clip not in tile_config:
            raise SystemExit(f"Clip {clip!r} missing from config/clip_tiling.json. Run probe_clips.py.")

    band_stats = {band: BandStats() for band in DEFAULT_BAND_CLIPS}
    video_dir = None if args.no_video else Path(args.video_dir)
    yolo_model = YOLO(str(weights))
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
                conf_override=args.conf,
                iou_thresh=args.iou,
                band_stats=band_stats,
                video_dir=video_dir,
            )
        )
    run_elapsed_sec = round(time.perf_counter() - run_t0, 2)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "generated_at": generated_at,
        "weights": str(weights),
        "iou": args.iou,
        "clips": clip_names,
        "timing": {
            "elapsed_sec": run_elapsed_sec,
            "elapsed_human": (
                f"{int(run_elapsed_sec // 60)}m {run_elapsed_sec % 60:.1f}s"
                if run_elapsed_sec >= 60
                else f"{run_elapsed_sec:.1f}s"
            ),
            "clips": [
                {
                    "clip": r["clip"],
                    **(r.get("timing") or {}),
                }
                for r in clip_results
            ],
        },
        "eval_mode": {
            "imgsz": slice_cfg.imgsz,
            "overlap_ratio": slice_cfg.overlap_ratio,
            "clip_slice_size": slice_cfg.clip_slice_size,
            "clip_overlap": slice_cfg.clip_overlap,
            "clip_train_imgsz": slice_cfg.clip_train_imgsz,
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
            "gt_source": "labels/eval pseudo-labels",
            "band_assignment": "whole clip from probe distance_band in clip_tiling.json",
            "eval_clips_by_band": DEFAULT_BAND_CLIPS,
            "false_alarms_formula": "FP / (N_frames / fps / 60)",
            "inference": (
                f"per-clip train_groups tile_size (or full-frame), "
                f"YOLO predict_imgsz from group (default {slice_cfg.imgsz}), "
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
            }
            for band, s in band_stats.items()
        },
        "clips_detail": clip_results,
    }

    out_dir = Path(args.output_dir)
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
    ]
    for label, key, pct in rows:
        getter = {
            "detection_rate": lambda s: s.detection_rate(),
            "precision": lambda s: s.precision(),
            "false_alarms_per_min": lambda s: s.false_alarms_per_min(),
            "time_to_first_detection_s": lambda s: s.time_to_first_detection_s(),
            "map50": lambda s: s.map50(),
        }[key]
        a, b = getter(band_stats["0-200m"]), getter(band_stats["200-400m"])
        print(f"{label:<32} {format_metric(a, pct):>10} {format_metric(b, pct):>10}")

    print(f"\nEval wall time: {payload['timing']['elapsed_human']} ({run_elapsed_sec:.1f}s)")
    for item in payload["timing"]["clips"]:
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


if __name__ == "__main__":
    main()
