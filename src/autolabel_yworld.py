#!/usr/bin/env python3
"""Pseudo-label frames with YOLO-World (step 3: needs config from probe_clips.py)."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from config import PROJECT_ROOT, build_split_map, iter_autolabel_clips

DEBUG_DIR_REL = Path("debug")
PLOT_CACHE_DIR = PROJECT_ROOT / DEBUG_DIR_REL / ".matplotlib_cache"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR))

FRAMES_DIR = Path("data/frames")
LABELS_DIR = Path("labels")

import cv2
import numpy as np
from sahi import AutoDetectionModel

from config import (
    CLIP_TILING_CONFIG_PATH,
    RAW_CONFIDENCE_THRESHOLD,
    load_clip_tile_config,
    resolve_clip_tile_config,
)
from detect import MODEL_NAME, build_yolo_world, compute_slice_size, detect_frame_sahi, device

# --- final-run YOLO-World config ---
VEHICLE_CLASSES = ["car", "truck", "pickup", "bus", "van", "motorcycle"]
# VEHICLE_CLASSES = ["car roof", "top-down view of a car", "truck from above", "vehicle from above", "bus roof", "motorcycle from above"]
TRACK_IOU_THRESHOLD = 0.3  # lightweight IoU tracking after SAHI
DEVICE = device()

MIN_TRACK_FRAMES = 3
MAX_FILL_GAP_FRAMES = 2
MAX_THRESHOLD_DIP_FRAMES = 2
MAX_THRESHOLD_SPIKE_FRAMES = 2
DEDUPE_COORD_TOLERANCE_PX = 2.0

# Set to a small integer for a quick smoke run, or None for final pre-labeling.
LIMIT_FRAMES: int | None = None
WRITE_DEBUG_VIDEO = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pseudo-label vehicle frames with YOLO-World.",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Process a single clip (video stem / folder name under data/frames).",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Experimental: CLAHE contrast boost before YOLO-World (see image_enhance.py).",
    )
    return parser.parse_args()


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
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


def boxes_are_duplicate(box_a, box_b, tol_px: float = DEDUPE_COORD_TOLERANCE_PX) -> bool:
    if box_iou(box_a, box_b) >= 0.99:
        return True
    return all(abs(a - b) <= tol_px for a, b in zip(box_a, box_b))


def copy_detection(det: dict, **extra) -> dict:
    out = {
        "xyxy": list(det["xyxy"]),
        "confidence": det["confidence"],
        "subclass_name": det.get("subclass_name", "vehicle"),
        "track_id": det["track_id"],
    }
    out.update(extra)
    return out


def duplicate_detection(det: dict) -> dict:
    return copy_detection(det, filled_gap=True)


def dedupe_frame_detections(detections: list[dict]) -> tuple[list[dict], int]:
    if len(detections) <= 1:
        return detections, 0
    ordered = sorted(detections, key=lambda det: det["confidence"], reverse=True)
    kept: list[dict] = []
    removed = 0
    for det in ordered:
        if any(boxes_are_duplicate(det["xyxy"], other["xyxy"]) for other in kept):
            removed += 1
            continue
        kept.append(det)
    return kept, removed


def dedupe_overlapping_labels(labels_by_frame: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], int]:
    deduped: dict[str, list[dict]] = {}
    removed = 0
    for frame_name, detections in labels_by_frame.items():
        kept, frame_removed = dedupe_frame_detections(detections)
        deduped[frame_name] = kept
        removed += frame_removed
    return deduped, removed


def dedupe_detections_by_frame(detections_by_frame: dict[str, list[dict]], frame_order: list[str]) -> int:
    removed = 0
    for frame_name in frame_order:
        kept, frame_removed = dedupe_frame_detections(detections_by_frame.get(frame_name, []))
        detections_by_frame[frame_name] = kept
        removed += frame_removed
    return removed


def identify_stable_tracks(
    detections_by_frame: dict[str, list[dict]],
    frame_order: list[str],
    label_confidence_threshold: float,
    min_track_frames: int = MIN_TRACK_FRAMES,
) -> tuple[set[int], dict]:
    track_label_frames: dict[int, int] = {}
    for frame_name in frame_order:
        seen_tracks = set()
        for det in detections_by_frame[frame_name]:
            if det["confidence"] < label_confidence_threshold:
                continue
            track_id = det.get("track_id", -1)
            if track_id is None or track_id < 0 or track_id in seen_tracks:
                continue
            seen_tracks.add(track_id)
            track_label_frames[track_id] = track_label_frames.get(track_id, 0) + 1
    stable_tracks = {
        track_id for track_id, frame_count in track_label_frames.items()
        if frame_count >= min_track_frames
    }
    return stable_tracks, {
        "tracks_total": len(track_label_frames),
        "tracks_stable": len(stable_tracks),
        "min_track_frames": min_track_frames,
    }


def close_short_false_runs(flags: list[bool], max_gap: int) -> list[bool]:
    result = flags[:]
    i = 0
    while i < len(result):
        if result[i]:
            i += 1
            continue
        j = i
        while j < len(result) and not result[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and result[i - 1] and j < len(result) and result[j]:
            for k in range(i, j):
                result[k] = True
        i = max(j, i + 1)
    return result


def open_short_true_runs(flags: list[bool], max_spike: int) -> list[bool]:
    result = flags[:]
    i = 0
    while i < len(result):
        if not result[i]:
            i += 1
            continue
        j = i
        while j < len(result) and result[j]:
            j += 1
        run_len = j - i
        before_false = i == 0 or not result[i - 1]
        after_false = j == len(result) or not result[j]
        if run_len <= max_spike and before_false and after_false:
            for k in range(i, j):
                result[k] = False
        i = max(j, i + 1)
    return result


def smooth_threshold_labels(
    detections_by_frame: dict[str, list[dict]],
    frame_order: list[str],
    stable_tracks: set[int],
    label_confidence_threshold: float,
    max_dip_frames: int = MAX_THRESHOLD_DIP_FRAMES,
    max_spike_frames: int = MAX_THRESHOLD_SPIKE_FRAMES,
) -> tuple[dict[str, list[dict]], dict]:
    track_timeline: dict[int, dict[int, dict]] = {}
    for frame_idx, frame_name in enumerate(frame_order):
        for det in detections_by_frame[frame_name]:
            track_id = det.get("track_id", -1)
            if track_id is None or track_id < 0 or track_id not in stable_tracks:
                continue
            current = track_timeline.setdefault(track_id, {}).get(frame_idx)
            if current is None or det["confidence"] > current["confidence"]:
                track_timeline[track_id][frame_idx] = det

    labels_by_frame = {frame_name: [] for frame_name in frame_order}
    before = 0
    after = 0
    dips_filled = 0
    spikes_removed = 0

    for frame_dets in track_timeline.values():
        frame_indices = sorted(frame_dets)
        if not frame_indices:
            continue
        start_idx, end_idx = frame_indices[0], frame_indices[-1]
        span_indices = list(range(start_idx, end_idx + 1))
        above = []
        for frame_idx in span_indices:
            det = frame_dets.get(frame_idx)
            above.append(det is not None and det["confidence"] >= label_confidence_threshold)

        before += sum(above)
        smoothed = close_short_false_runs(above, max_dip_frames)
        smoothed = open_short_true_runs(smoothed, max_spike_frames)
        after += sum(smoothed)

        for frame_idx, keep, was_above in zip(span_indices, smoothed, above):
            if not keep:
                if was_above:
                    spikes_removed += 1
                continue
            det = frame_dets.get(frame_idx)
            if det is None:
                continue
            out_det = copy_detection(det, filled_dip=True) if not was_above else det
            if not was_above:
                dips_filled += 1
            labels_by_frame[frame_order[frame_idx]].append(out_det)

    return labels_by_frame, {
        "labels_before_threshold_smooth": before,
        "labels_after_threshold_smooth": after,
        "labels_dips_filled": dips_filled,
        "labels_spikes_removed": spikes_removed,
        "max_threshold_dip_frames": max_dip_frames,
        "max_threshold_spike_frames": max_spike_frames,
    }


def filter_stable_tracked_labels(
    detections_by_frame: dict[str, list[dict]],
    frame_order: list[str],
    label_confidence_threshold: float,
    min_track_frames: int = MIN_TRACK_FRAMES,
    max_dip_frames: int = MAX_THRESHOLD_DIP_FRAMES,
    max_spike_frames: int = MAX_THRESHOLD_SPIKE_FRAMES,
) -> tuple[dict[str, list[dict]], dict]:
    stable_tracks, track_stats = identify_stable_tracks(
        detections_by_frame, frame_order, label_confidence_threshold, min_track_frames,
    )
    labels_by_frame, smooth_stats = smooth_threshold_labels(
        detections_by_frame, frame_order, stable_tracks, label_confidence_threshold,
        max_dip_frames, max_spike_frames,
    )
    before_tracking = sum(
        1 for frame_name in frame_order for det in detections_by_frame[frame_name]
        if det["confidence"] >= label_confidence_threshold
    )
    after_tracking = sum(len(dets) for dets in labels_by_frame.values())
    track_stats.update(smooth_stats)
    track_stats.update({
        "labels_before_tracking": before_tracking,
        "labels_after_tracking": after_tracking,
        "labels_dropped_by_tracking": before_tracking - after_tracking,
    })
    return labels_by_frame, track_stats


def fill_track_gaps(
    labels_by_frame: dict[str, list[dict]],
    frame_order: list[str],
    max_gap_frames: int = MAX_FILL_GAP_FRAMES,
) -> tuple[dict[str, list[dict]], int]:
    track_by_frame: dict[int, dict[int, dict]] = {}
    for frame_idx, frame_name in enumerate(frame_order):
        for det in labels_by_frame[frame_name]:
            track_id = det.get("track_id", -1)
            if track_id is None or track_id < 0:
                continue
            track_by_frame.setdefault(track_id, {})[frame_idx] = det

    filled = {frame_name: list(detections) for frame_name, detections in labels_by_frame.items()}
    fill_count = 0
    for frame_dets in track_by_frame.values():
        labeled_indices = sorted(frame_dets)
        for left, right in zip(labeled_indices, labeled_indices[1:]):
            gap = right - left - 1
            if gap < 1 or gap > max_gap_frames:
                continue
            src_det = frame_dets[left]
            for missing_idx in range(left + 1, right):
                frame_name = frame_order[missing_idx]
                present_ids = {det.get("track_id") for det in filled[frame_name]}
                if src_det["track_id"] in present_ids:
                    continue
                filled[frame_name].append(duplicate_detection(src_det))
                fill_count += 1
    return filled, fill_count


def assign_track_ids_iou(
    detections_by_frame: dict[str, list[dict]],
    frame_order: list[str],
) -> None:
    """Greedy IoU tracking for SAHI detections on train clips."""
    next_track_id = 0
    prev_tracks: list[tuple[int, list[float]]] = []

    for frame_name in frame_order:
        detections = detections_by_frame[frame_name]
        used_prev = set()

        for det in detections:
            best_iou = TRACK_IOU_THRESHOLD
            best_idx = -1
            for idx, (track_id, prev_box) in enumerate(prev_tracks):
                if idx in used_prev:
                    continue
                iou = box_iou(det["xyxy"], prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0:
                det["track_id"] = prev_tracks[best_idx][0]
                used_prev.add(best_idx)
            else:
                det["track_id"] = next_track_id
                next_track_id += 1

        prev_tracks = [(det["track_id"], det["xyxy"]) for det in detections]


def reset_ultralytics_tracker(ultra_model) -> None:
    """Drop predictor so the next model.track() re-inits ByteTrack for a new clip."""
    ultra_model.predictor = None


def parse_track_result(result) -> list[dict]:
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    track_ids = result.boxes.id
    if track_ids is None:
        ids = [-1] * len(boxes)
    else:
        ids = track_ids.cpu().numpy().astype(int)

    for i in range(len(boxes)):
        detections.append({
            "xyxy": [float(v) for v in boxes[i]],
            "confidence": float(confs[i]),
            "subclass_name": result.names[int(cls_ids[i])],
            "track_id": int(ids[i]),
        })
    return detections


def xyxy_to_yolo(box, width: int, height: int) -> str | None:
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), width)
    x2 = min(max(x2, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    y2 = min(max(y2, 0.0), height)
    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def bbox_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def format_duration(seconds: float) -> str:
    minutes, sec = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def save_labels(detections_by_frame: dict[str, list[dict]], out_dir: Path, width: int, height: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_name, detections in detections_by_frame.items():
        lines = [line for det in detections if (line := xyxy_to_yolo(det["xyxy"], width, height))]
        (out_dir / f"{frame_name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
    return len(detections_by_frame)


def cache_meta(tile_cfg: dict, slice_size: list[int] | None = None) -> dict:
    meta = {
        "model_name": MODEL_NAME,
        "vehicle_classes": VEHICLE_CLASSES,
        "model_confidence": RAW_CONFIDENCE_THRESHOLD,
        "target_tiles": tile_cfg["target_tiles"],
        "overlap_ratio": tile_cfg["overlap_ratio"],
        "distance_band": tile_cfg["distance_band"],
        "note": tile_cfg["note"],
    }
    if tile_cfg["uses_sahi"]:
        meta.update({
            "detection_mode": "sahi",
            "tracker": "iou",
            "slice_size": slice_size,
        })
    else:
        meta.update({
            "detection_mode": "model.track",
            "tracker": "bytetrack",
        })
    return meta


def run_eval_fullframe_track(
    ultra_model,
    frame_paths: list[Path],
    t0: float,
    *,
    enhance: bool = False,
) -> tuple[dict[str, list[dict]], list[float]]:
    from image_enhance import inference_source

    reset_ultralytics_tracker(ultra_model)
    raw_detections_by_frame = {}
    all_confidences = []

    for i, frame_path in enumerate(frame_paths):
        result = ultra_model.track(
            inference_source(frame_path, enhance=enhance),
            persist=True,
            conf=RAW_CONFIDENCE_THRESHOLD,
            verbose=False,
            device=DEVICE,
        )[0]
        detections = parse_track_result(result)
        raw_detections_by_frame[frame_path.stem] = detections
        all_confidences.extend(det["confidence"] for det in detections)

        if (i + 1) % 50 == 0 or i + 1 == len(frame_paths):
            elapsed = time.perf_counter() - t0
            remaining = elapsed / (i + 1) * (len(frame_paths) - i - 1)
            print(f"  {i + 1}/{len(frame_paths)} frames, ~{remaining / 60:.1f} min left")

    return raw_detections_by_frame, all_confidences


def run_fixed_sahi_detect(
    model: AutoDetectionModel,
    frame_paths: list[Path],
    width: int,
    height: int,
    tile_cfg: dict,
    t0: float,
    *,
    enhance: bool = False,
) -> tuple[dict[str, list[dict]], list[float], dict]:
    target_tiles = tile_cfg["target_tiles"]
    overlap_ratio = tile_cfg["overlap_ratio"]
    slice_h, slice_w = compute_slice_size(width, height, target_tiles)
    raw_detections_by_frame = {}
    all_confidences = []

    for i, frame_path in enumerate(frame_paths):
        detections = detect_frame_sahi(
            model, frame_path, slice_h, slice_w, overlap_ratio, enhance=enhance
        )
        raw_detections_by_frame[frame_path.stem] = detections
        all_confidences.extend(det["confidence"] for det in detections)

        if (i + 1) % 50 == 0 or i + 1 == len(frame_paths):
            elapsed = time.perf_counter() - t0
            remaining = elapsed / (i + 1) * (len(frame_paths) - i - 1)
            print(f"  {i + 1}/{len(frame_paths)} frames, ~{remaining / 60:.1f} min left")

    frame_order = [frame_path.stem for frame_path in frame_paths]
    assign_track_ids_iou(raw_detections_by_frame, frame_order)
    tiling_stats = {
        "target_tiles": target_tiles,
        "overlap_ratio": overlap_ratio,
        "distance_band": tile_cfg["distance_band"],
        "slice_size": [slice_w, slice_h],
    }
    return raw_detections_by_frame, all_confidences, tiling_stats


def save_raw_detections_cache(
    detections_by_frame: dict[str, list[dict]],
    cache_dir: Path,
    meta: dict,
) -> int:
    """Write fresh raw detections. Always overwrites old cache for this clip."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old_file in cache_dir.glob("*.json"):
        old_file.unlink()

    for frame_name, detections in detections_by_frame.items():
        payload = {"meta": meta, "detections": detections}
        (cache_dir / f"{frame_name}.json").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return len(detections_by_frame)


def save_confidence_histogram(
    all_confidences: list[float],
    clip_name: str,
    out_path: Path,
    label_threshold: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if all_confidences:
        plt.hist(all_confidences, bins=30, range=(0, 1), edgecolor="black")
    plt.axvline(
        label_threshold,
        color="red",
        linestyle="--",
        label=f"threshold={label_threshold}",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title(f"{clip_name}: confidence distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_boxes(image: np.ndarray, detections: list[dict], label_threshold: float) -> np.ndarray:
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        kept = det["confidence"] >= label_threshold
        color = (0, 255, 0) if kept else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{det['confidence']:.2f}",
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return canvas


def save_debug_video(
    detections_by_frame: dict[str, list[dict]],
    frame_paths: list[Path],
    metadata: dict,
    out_path: Path,
    label_threshold: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = metadata["width"]
    height = metadata["height"]
    fps = metadata.get("fps", 30)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))
        writer.write(draw_boxes(image, detections_by_frame[frame_path.stem], label_threshold))
    writer.release()


def summarize(all_detections: list[dict], label_threshold: float) -> dict:
    kept = [det for det in all_detections if det["confidence"] >= label_threshold]
    dropped = [det for det in all_detections if det["confidence"] < label_threshold]
    kept_areas = [bbox_area(det["xyxy"]) for det in kept]
    dropped_areas = [bbox_area(det["xyxy"]) for det in dropped]
    total = len(all_detections)
    return {
        "total_detections": total,
        "kept": len(kept),
        "dropped": len(dropped),
        "pct_dropped": round((len(dropped) / total * 100), 2) if total else 0.0,
        "median_bbox_area_kept": float(np.median(kept_areas)) if kept_areas else 0.0,
        "median_bbox_area_dropped": float(np.median(dropped_areas)) if dropped_areas else 0.0,
    }


def process_clip(
    model: AutoDetectionModel,
    split: str,
    clip_dir: Path,
    root: Path,
    tile_config: dict[str, dict],
    *,
    enhance: bool = False,
) -> dict:
    clip_name = clip_dir.name
    metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
    width, height = metadata["width"], metadata["height"]

    frame_paths = sorted(clip_dir.glob("*.jpg"))
    if LIMIT_FRAMES is not None:
        frame_paths = frame_paths[:LIMIT_FRAMES]

    tile_cfg = resolve_clip_tile_config(clip_name, tile_config)
    label_threshold = tile_cfg["label_confidence_threshold"]
    enhance_note = ", CLAHE enhance ON" if enhance else ""

    if tile_cfg["uses_sahi"]:
        detection_mode = "sahi"
        slice_h, slice_w = compute_slice_size(width, height, tile_cfg["target_tiles"])
        print(
            f"\n{split}/{clip_name}: {width}x{height}, "
            f"SAHI {tile_cfg['target_tiles']} tiles (overlap={tile_cfg['overlap_ratio']}, "
            f"band={tile_cfg['distance_band']}, label_conf>={label_threshold}) + IoU track, "
            f"{len(frame_paths)} frames{enhance_note}"
        )
        if tile_cfg["note"]:
            print(f"  note: {tile_cfg['note']}")
    else:
        detection_mode = "model.track"
        slice_h = slice_w = None
        print(
            f"\n{split}/{clip_name}: {width}x{height}, "
            f"full-frame model.track + ByteTrack (band={tile_cfg['distance_band']}, "
            f"label_conf>={label_threshold}), "
            f"{len(frame_paths)} frames{enhance_note}"
        )
        if tile_cfg["note"]:
            print(f"  note: {tile_cfg['note']}")

    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()

    if tile_cfg["uses_sahi"]:
        raw_detections_by_frame, all_confidences, tiling_stats = run_fixed_sahi_detect(
            model, frame_paths, width, height, tile_cfg, t0, enhance=enhance
        )
        print(
            f"  tiling: {tiling_stats['target_tiles']} tiles, "
            f"slice={tiling_stats['slice_size']}, band={tiling_stats['distance_band']}"
        )
    else:
        tiling_stats = {
            "target_tiles": 1,
            "overlap_ratio": 0.0,
            "distance_band": tile_cfg["distance_band"],
            "slice_size": None,
        }
        raw_detections_by_frame, all_confidences = run_eval_fullframe_track(
            model.model, frame_paths, t0, enhance=enhance
        )

    frame_order = [frame_path.stem for frame_path in frame_paths]
    raw_deduped = dedupe_detections_by_frame(raw_detections_by_frame, frame_order)
    kept_detections_by_frame, track_stats = filter_stable_tracked_labels(
        raw_detections_by_frame,
        frame_order,
        label_threshold,
    )
    kept_detections_by_frame, filled_gaps = fill_track_gaps(
        kept_detections_by_frame,
        frame_order,
    )
    kept_detections_by_frame, label_deduped = dedupe_overlapping_labels(kept_detections_by_frame)
    track_stats["raw_labels_deduped"] = raw_deduped
    track_stats["labels_deduped"] = label_deduped
    track_stats["labels_filled_gaps"] = filled_gaps
    track_stats["labels_after_gap_fill"] = sum(len(dets) for dets in kept_detections_by_frame.values())
    print(
        f"  tracking ({detection_mode}): "
        f"{track_stats['labels_after_tracking']}/{track_stats['labels_before_tracking']} labels kept, "
        f"{track_stats['tracks_stable']}/{track_stats['tracks_total']} stable tracks "
        f"(>={MIN_TRACK_FRAMES} frames), "
        f"+{track_stats['labels_dips_filled']} dips, "
        f"-{track_stats['labels_spikes_removed']} spikes, "
        f"+{filled_gaps} gap-filled, "
        f"-{label_deduped} deduped"
    )

    labels_dir = root / LABELS_DIR / split / clip_name
    cache_dir = root / DEBUG_DIR_REL / clip_name / "cache"

    label_files = save_labels(kept_detections_by_frame, labels_dir, width, height)
    cache_files = save_raw_detections_cache(
        raw_detections_by_frame,
        cache_dir,
        cache_meta(
            tile_cfg,
            tiling_stats.get("slice_size") if tile_cfg["uses_sahi"] else None,
        ),
    )
    print(f"  labels: {labels_dir} ({label_files} files)")
    print(f"  raw cache: {cache_dir} ({cache_files} files)")
    save_confidence_histogram(
        all_confidences,
        clip_name,
        root / DEBUG_DIR_REL / clip_name / "confidence_hist.png",
        label_threshold,
    )
    if WRITE_DEBUG_VIDEO:
        save_debug_video(
            raw_detections_by_frame,
            frame_paths,
            metadata,
            root / DEBUG_DIR_REL / clip_name / "labels_debug.mp4",
            label_threshold,
        )

    flat = [det for detections in raw_detections_by_frame.values() for det in detections]
    elapsed_sec = time.perf_counter() - t0
    finished_at = datetime.now().isoformat(timespec="seconds")
    stats = summarize(flat, label_threshold)
    stats.update({
        "label_confidence_threshold": label_threshold,
        "frame_size": [width, height],
        "detection_mode": detection_mode,
        "tile_config": tile_cfg,
        "slice_size": tiling_stats.get("slice_size"),
        "tiling_stats": tiling_stats,
        "frames_processed": len(frame_paths),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed_sec, 2),
        "elapsed_human": format_duration(elapsed_sec),
        "seconds_per_frame": round(elapsed_sec / len(frame_paths), 3) if frame_paths else 0.0,
        "enhance_clahe": bool(enhance),
        **track_stats,
    })
    print(f"  kept={stats['kept']}/{stats['total_detections']} ({stats['pct_dropped']}% dropped)")
    print(f"  finished in {stats['elapsed_human']} ({stats['seconds_per_frame']} sec/frame)")
    return stats


def main() -> None:
    args = parse_args()
    clip_filter = args.clip
    if clip_filter and clip_filter.endswith(".mp4"):
        clip_filter = Path(clip_filter).stem

    root = PROJECT_ROOT
    split_map = build_split_map(root)
    clips = iter_autolabel_clips(split_map, clip_filter=clip_filter)
    if not clips:
        if clip_filter:
            raise SystemExit(
                f"No frames found for clip {clip_filter!r}. "
                "Expected data/frames/{clip}/ with metadata.json and .jpg frames."
            )
        raise SystemExit("No clips found. Expected data/frames/* plus matching .mp4 stems in data/train or data/eval.")

    tile_config = load_clip_tile_config()
    if not tile_config:
        raise SystemExit(
            f"No clip tiling config at {CLIP_TILING_CONFIG_PATH}. "
            "Run: python src/probe_clips.py"
        )

    print(
        f"Loading {MODEL_NAME} on {DEVICE}, "
        f"raw_confidence={RAW_CONFIDENCE_THRESHOLD}, "
        f"per-clip label_confidence_threshold from config"
    )
    model, _ = build_yolo_world(VEHICLE_CLASSES)

    run_started_at = datetime.now().isoformat(timespec="seconds")
    run_t0 = time.perf_counter()
    clip_stats = {}
    for split, clip_dir in clips:
        key = f"{split}/{clip_dir.name}"
        clip_stats[key] = process_clip(
            model, split, clip_dir, root, tile_config, enhance=args.enhance
        )

    run_elapsed_sec = time.perf_counter() - run_t0
    run_finished_at = datetime.now().isoformat(timespec="seconds")
    stats_path = root / DEBUG_DIR_REL / "label_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "run_started_at": run_started_at,
                "run_finished_at": run_finished_at,
                "run_elapsed_seconds": round(run_elapsed_sec, 2),
                "run_elapsed_human": format_duration(run_elapsed_sec),
                "raw_confidence_threshold": RAW_CONFIDENCE_THRESHOLD,
                "min_track_frames": MIN_TRACK_FRAMES,
                "max_fill_gap_frames": MAX_FILL_GAP_FRAMES,
                "max_threshold_dip_frames": MAX_THRESHOLD_DIP_FRAMES,
                "max_threshold_spike_frames": MAX_THRESHOLD_SPIKE_FRAMES,
                "clip_tiling_config_path": str(CLIP_TILING_CONFIG_PATH),
                "clip_tile_config": tile_config,
                "clips": clip_stats,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nDone in {format_duration(run_elapsed_sec)}. Stats: {stats_path}")


if __name__ == "__main__":
    main()
