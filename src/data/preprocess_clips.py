#!/usr/bin/env python3
"""Preprocess clips: tiles, car size, distance drift, frame_step → clip_tiling.json.

Samples 3 frames from start / middle / end (9 total). Uses YOLO-World **car**
detections only (person→car alias; ignores truck/bus/bike) to estimate:

  - min SAHI tiles (+1 headroom → target_tiles)
  - object size in full-frame pixels (median/mean car long-side)
  - distance per segment and whether it changes along the video
  - suggested label frame_step = floor(size × fraction / speed_px_per_frame)
  - suggested train_groups band from car size

Replaces the old middle-only probe_clips.py flow. Backward-compatible CLI:
  python src/data/preprocess_clips.py
  python src/probe_clips.py          # thin wrapper → this script
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
import statistics
import time
from pathlib import Path

from common.config import (
    CLIP_TILING_CONFIG_PATH,
    DEBUG_DIR,
    FALLBACK_LABEL_THRESHOLD,
    FALLBACK_TILES,
    PROBE_CLASS_ALIASES,
    PROBE_MAX_LABEL_THRESHOLD,
    RAW_CONFIDENCE_THRESHOLD,
    TILE_CANDIDATES,
    TILE_HEADROOM_STEPS,
    build_split_map,
    iter_frame_clip_dirs,
    label_threshold_for_tiles,
    load_clip_tile_config,
    load_tiling_payload,
    merge_probe_results,
    next_tile_candidate,
    overlap_for_tiles,
    probe_detection_class,
    probe_model_classes,
    probe_result_to_config_entry,
    reject_if_clip_skipped,
    save_clip_tile_config,
)
from common.detect import (
    MODEL_NAME,
    build_yolo_world,
    car_detection_record,
    compute_slice_size,
    detect_frame_probe,
    distance_band,
    pick_largest_car,
    resolve_camera_model,
)

SEGMENTS = ("start", "middle", "end")
FRAMES_PER_SEGMENT = 3
FRAME_STEP_FRACTION = 0.5
TRACK_IOU = 0.3
# Relative size change start↔end (or band mismatch) ⇒ distance_varies
SIZE_CHANGE_RATIO = 0.30
# Suggested train group from median car long-side (full-frame px)
TRAIN_GROUP_FAR_MAX = 32.0
TRAIN_GROUP_MED_MAX = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess clips: start/middle/end car probe → tiles, size, "
            "distance change, frame_step → clip_tiling.json."
        ),
    )
    parser.add_argument("--clip", default=None, help="Single clip (video stem).")
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="Also process clips marked skip=true in clip_tiling.json.",
    )
    parser.add_argument(
        "--per-segment",
        type=int,
        default=FRAMES_PER_SEGMENT,
        help=f"Frames per start/middle/end (default {FRAMES_PER_SEGMENT}).",
    )
    parser.add_argument(
        "--step-fraction",
        type=float,
        default=FRAME_STEP_FRACTION,
        help=f"frame_step = size×fraction/speed (default {FRAME_STEP_FRACTION}).",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Experimental: CLAHE contrast boost before YOLO-World (see src/common/image_enhance.py).",
    )
    parser.add_argument(
        "--report",
        default=str(DEBUG_DIR / "preprocess_probe.json"),
        help="Detailed preprocess report JSON.",
    )
    parser.add_argument(
        "--config",
        default=str(CLIP_TILING_CONFIG_PATH),
        help="Clip tiling config used by autolabel.",
    )
    return parser.parse_args()


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if not paths:
        return []
    if len(paths) <= count:
        return list(paths)
    if count == 1:
        return [paths[len(paths) // 2]]
    out: list[Path] = []
    for i in range(count):
        idx = int(round(i * (len(paths) - 1) / (count - 1)))
        out.append(paths[idx])
    # de-dupe while preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        if p.name in seen:
            continue
        seen.add(p.name)
        uniq.append(p)
    return uniq


def select_segment_frames(
    frame_paths: list[Path],
    per_segment: int,
) -> list[tuple[str, Path]]:
    """Return (segment, path) for 3 evenly spaced frames in start/middle/end thirds."""
    n = len(frame_paths)
    if n == 0:
        return []
    if n <= per_segment * 3:
        # Too short: spread whatever we have across segments by index thirds.
        selected = evenly_spaced(frame_paths, min(n, per_segment * 3))
        out: list[tuple[str, Path]] = []
        for i, path in enumerate(selected):
            if len(selected) == 1:
                seg = "middle"
            else:
                seg = SEGMENTS[min(2, i * 3 // len(selected))]
            out.append((seg, path))
        return out

    boundaries = [0, n // 3, (2 * n) // 3, n]
    picked: list[tuple[str, Path]] = []
    for seg, a, b in zip(SEGMENTS, boundaries[:-1], boundaries[1:]):
        chunk = frame_paths[a:b]
        for path in evenly_spaced(chunk, per_segment):
            picked.append((seg, path))
    return picked


def summarize_cars(cars: list[dict], focal_px: float) -> dict:
    records = []
    for det in cars:
        record = car_detection_record(det, focal_px)
        if record:
            records.append(record)
    primary = pick_largest_car(records)
    return {"car_count": len(records), "cars": records, "primary_car": primary}


def detect_cars_on_frame(
    model,
    ultra_model,
    frame_path: Path,
    width: int,
    height: int,
    target_tiles: int,
    label_threshold: float,
    detection_class: str,
    device: str,
    focal_px: float,
    *,
    enhance: bool = False,
) -> dict:
    cars = detect_frame_probe(
        model,
        ultra_model,
        frame_path,
        width,
        height,
        target_tiles,
        label_threshold,
        detection_class,
        device,
        enhance=enhance,
    )
    summary = summarize_cars(cars, focal_px)
    return {"frame": frame_path.stem, **summary}


def match_speed_px(
    prev_cars: list[dict],
    next_cars: list[dict],
) -> list[float]:
    """Centroid shifts (px) for greedy IoU matches between two frames."""
    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(prev_cars):
        for j, b in enumerate(next_cars):
            iou = box_iou(a["bbox"], b["bbox"])
            if iou >= TRACK_IOU:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    speeds: list[float] = []
    for _, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        ax1, ay1, ax2, ay2 = prev_cars[i]["bbox"]
        bx1, by1, bx2, by2 = next_cars[j]["bbox"]
        acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        speeds.append(((bcx - acx) ** 2 + (bcy - acy) ** 2) ** 0.5)
    return speeds


def suggested_train_group(size_med: float | None) -> str | None:
    if size_med is None:
        return None
    if size_med < TRAIN_GROUP_FAR_MAX:
        return "C_far"
    if size_med < TRAIN_GROUP_MED_MAX:
        return "B_medium"
    return "A_close"


def median_or_none(vals: list[float]) -> float | None:
    return float(statistics.median(vals)) if vals else None


def mean_or_none(vals: list[float]) -> float | None:
    return float(statistics.mean(vals)) if vals else None


def preprocess_clip(
    model,
    clip_dir: Path,
    per_segment: int,
    step_fraction: float,
    split: str | None,
    detection_class: str,
    device: str,
    *,
    enhance: bool = False,
) -> dict:
    clip_name = clip_dir.name
    metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
    width, height = int(metadata["width"]), int(metadata["height"])
    fps = float(metadata.get("fps") or 0) or None
    camera = resolve_camera_model(clip_name, width, height)
    focal_px = camera["focal_px"]

    all_frames = sorted(clip_dir.glob("*.jpg"))
    segment_frames = select_segment_frames(all_frames, per_segment)
    frame_paths = [p for _, p in segment_frames]
    stem_to_segment = {p.stem: seg for seg, p in segment_frames}
    ultra_model = model.model

    # --- Tile ladder on all sampled frames ---
    attempts: list[dict] = []
    min_tiles: int | None = None
    hit_label_threshold: float | None = None
    ladder_hits: list[dict] = []

    for target_tiles in TILE_CANDIDATES:
        label_threshold = label_threshold_for_tiles(target_tiles)
        if target_tiles == TILE_CANDIDATES[0]:
            alias_note = ", ".join(
                f"{alias}→{target}"
                for alias, target in sorted(PROBE_CLASS_ALIASES.items())
                if target == detection_class
            ) or "none"
            print(
                f"  probe classes: {detection_class!r} (aliases: {alias_note}); "
                f"frames: {[p.stem for p in frame_paths]}"
            )
        print(f"  trying tiles={target_tiles}, label_threshold>={label_threshold}")

        t0 = time.perf_counter()
        frame_hits: list[dict] = []
        for frame_path in frame_paths:
            hit = detect_cars_on_frame(
                model,
                ultra_model,
                frame_path,
                width,
                height,
                target_tiles,
                label_threshold,
                detection_class,
                device,
                focal_px,
                enhance=enhance,
            )
            if hit["car_count"] > 0:
                hit["segment"] = stem_to_segment.get(hit["frame"], "?")
                frame_hits.append(hit)

        elapsed_sec = round(time.perf_counter() - t0, 2)
        slice_size = (
            None if target_tiles <= 1 else list(compute_slice_size(width, height, target_tiles))
        )
        attempts.append(
            {
                "target_tiles": target_tiles,
                "label_threshold": label_threshold,
                "overlap_ratio": 0.0 if target_tiles <= 1 else overlap_for_tiles(target_tiles),
                "slice_size": slice_size,
                "elapsed_sec": elapsed_sec,
                "frames_with_cars": len(frame_hits),
                "total_cars": sum(h["car_count"] for h in frame_hits),
                "segments_with_cars": sorted({h["segment"] for h in frame_hits}),
            }
        )

        if frame_hits and min_tiles is None:
            min_tiles = target_tiles
            hit_label_threshold = label_threshold
            ladder_hits = frame_hits
            target_with_headroom = next_tile_candidate(target_tiles)
            print(
                f"  tiles={target_tiles}: car HIT on {len(frame_hits)}/{len(frame_paths)} frames "
                f"({sorted({h['segment'] for h in frame_hits})}) in {elapsed_sec}s — "
                f"label target_tiles={target_with_headroom} (+{TILE_HEADROOM_STEPS} headroom)"
            )
            break
        print(f"  tiles={target_tiles}: no cars in {elapsed_sec}s")

    target_tiles = next_tile_candidate(min_tiles) if min_tiles is not None else None
    label_tiles = int(target_tiles) if target_tiles is not None else FALLBACK_TILES
    label_threshold = (
        label_threshold_for_tiles(label_tiles)
        if min_tiles is not None
        else FALLBACK_LABEL_THRESHOLD
    )

    # --- Detailed pass at labeling tile count (better for small/far cars) ---
    detail_hits: list[dict] = []
    if min_tiles is not None:
        # Reuse ladder hits if label tiles == min tiles; else re-detect at headroom level.
        if label_tiles == min_tiles:
            detail_hits = ladder_hits
        else:
            print(f"  detail pass at target_tiles={label_tiles}")
            t0 = time.perf_counter()
            for frame_path in frame_paths:
                hit = detect_cars_on_frame(
                    model,
                    ultra_model,
                    frame_path,
                    width,
                    height,
                    label_tiles,
                    label_threshold,
                    detection_class,
                    device,
                    focal_px,
                    enhance=enhance,
                )
                if hit["car_count"] > 0:
                    hit["segment"] = stem_to_segment.get(hit["frame"], "?")
                    detail_hits.append(hit)
            print(f"  detail pass: {len(detail_hits)} frames with cars in {time.perf_counter()-t0:.1f}s")
    else:
        detail_hits = []

    # Per-segment aggregates (car only)
    by_segment: dict[str, dict] = {}
    all_sizes: list[float] = []
    all_dists: list[float] = []
    for seg in SEGMENTS:
        seg_hits = [h for h in detail_hits if h.get("segment") == seg]
        sizes = [c["bbox_long_side_px"] for h in seg_hits for c in h["cars"]]
        dists = [c["distance_m"] for h in seg_hits for c in h["cars"]]
        all_sizes.extend(sizes)
        all_dists.extend(dists)
        size_med = median_or_none(sizes)
        dist_med = median_or_none(dists)
        by_segment[seg] = {
            "frames_with_cars": len(seg_hits),
            "car_boxes": len(sizes),
            "object_size_px_median": round(size_med, 1) if size_med is not None else None,
            "object_size_px_mean": round(mean_or_none(sizes), 1) if sizes else None,
            "distance_m_median": round(dist_med, 1) if dist_med is not None else None,
            "distance_band": distance_band(dist_med),
            "frames": [h["frame"] for h in seg_hits],
        }

    size_med = median_or_none(all_sizes)
    size_mean = mean_or_none(all_sizes)
    dist_med = median_or_none(all_dists)
    bands = {
        by_segment[s]["distance_band"]
        for s in SEGMENTS
        if by_segment[s]["distance_band"] is not None
    }
    start_size = by_segment["start"]["object_size_px_median"]
    end_size = by_segment["end"]["object_size_px_median"]
    size_ratio_change = None
    if start_size and end_size and min(start_size, end_size) > 0:
        size_ratio_change = abs(start_size - end_size) / max(start_size, end_size)
    distance_varies = (len(bands) > 1) or (
        size_ratio_change is not None and size_ratio_change >= SIZE_CHANGE_RATIO
    )

    # Primary car / clip-level distance: largest car among detail hits (stable vs old probe)
    all_primaries = [h["primary_car"] for h in detail_hits if h.get("primary_car")]
    primary_car = pick_largest_car(all_primaries) if all_primaries else None
    distance_m = primary_car["distance_m"] if primary_car else dist_med
    hit_frame = None
    if primary_car:
        for h in detail_hits:
            if h.get("primary_car") is primary_car or (
                h.get("primary_car")
                and h["primary_car"]["bbox_area_px"] == primary_car["bbox_area_px"]
            ):
                hit_frame = h["frame"]
                break
    if hit_frame is None and detail_hits:
        hit_frame = max(detail_hits, key=lambda h: h["car_count"])["frame"]

    # --- frame_step from consecutive-frame car motion near probe frames ---
    speeds: list[float] = []
    if min_tiles is not None and detail_hits:
        stem_to_path = {p.stem: p for p in all_frames}
        stems_sorted = [p.stem for p in all_frames]
        stem_index = {s: i for i, s in enumerate(stems_sorted)}
        for hit in detail_hits:
            idx = stem_index.get(hit["frame"])
            if idx is None or idx + 1 >= len(stems_sorted):
                continue
            next_stem = stems_sorted[idx + 1]
            next_path = stem_to_path[next_stem]
            next_hit = detect_cars_on_frame(
                model,
                ultra_model,
                next_path,
                width,
                height,
                label_tiles,
                label_threshold,
                detection_class,
                device,
                focal_px,
                enhance=enhance,
            )
            if next_hit["car_count"] == 0:
                continue
            speeds.extend(match_speed_px(hit["cars"], next_hit["cars"]))

    speed_med = median_or_none(speeds)
    frame_step = None
    if size_med is not None and speed_med is not None and speed_med > 1e-6:
        frame_step = max(1, int((size_med * step_fraction) / speed_med))

    train_group = suggested_train_group(size_med)

    total_elapsed = round(sum(item["elapsed_sec"] for item in attempts), 2)
    return {
        "clip": clip_name,
        "split": split,
        "detection_class": detection_class,
        "label_threshold": hit_label_threshold if min_tiles is not None else None,
        "resolution": [width, height],
        "fps": fps,
        "frames_probed": [p.stem for p in frame_paths],
        "frames_probed_by_segment": {
            seg: [p.stem for s, p in segment_frames if s == seg] for seg in SEGMENTS
        },
        "camera": camera,
        "min_tiles": min_tiles,
        "target_tiles": target_tiles,
        "hit_frame": hit_frame,
        "distance_m": round(distance_m, 1) if distance_m is not None else None,
        "distance_source": "largest_car" if primary_car else ("median_car" if dist_med else None),
        "distance_band": distance_band(distance_m),
        "distance_varies": distance_varies,
        "distance_by_segment": by_segment,
        "size_ratio_change_start_end": (
            round(size_ratio_change, 3) if size_ratio_change is not None else None
        ),
        "object_size_px_median": round(size_med, 1) if size_med is not None else None,
        "object_size_px_mean": round(size_mean, 1) if size_mean is not None else None,
        "object_size_n_boxes": len(all_sizes),
        "speed_px_per_frame_median": round(speed_med, 2) if speed_med is not None else None,
        "speed_samples": len(speeds),
        "frame_step": frame_step,
        "frame_step_fraction": step_fraction,
        "suggested_train_group": train_group,
        "primary_car": primary_car,
        "attempts": attempts,
        "total_probe_sec": total_elapsed,
    }


def main() -> None:
    args = parse_args()
    clip_filter = args.clip
    if clip_filter and clip_filter.endswith(".mp4"):
        clip_filter = Path(clip_filter).stem

    split_map = build_split_map()
    if clip_filter:
        reject_if_clip_skipped(clip_filter, allow_skipped=args.include_skipped)
    clip_dirs = iter_frame_clip_dirs(
        clip_filter, include_skipped=args.include_skipped
    )
    if not clip_dirs:
        raise SystemExit("No frame folders found under data/frames/.")

    config_path = Path(args.config)
    tiling_payload = load_tiling_payload(config_path)
    existing_config = load_clip_tile_config(config_path)
    detection_class = probe_detection_class(tiling_payload)
    probe_classes = probe_model_classes(detection_class)
    print(f"Loading {MODEL_NAME} on probe classes {probe_classes}")
    model, device = build_yolo_world(probe_classes)

    results: list[dict] = []
    run_t0 = time.perf_counter()
    for clip_dir in clip_dirs:
        split = split_map.get(clip_dir.name)
        split_label = split or "unknown"
        enhance_note = ", CLAHE enhance ON" if args.enhance else ""
        print(
            f"\n{split_label}/{clip_dir.name}: preprocess "
            f"{args.per_segment}×{len(SEGMENTS)} frames (start/middle/end){enhance_note}"
        )
        results.append(
            preprocess_clip(
                model,
                clip_dir,
                args.per_segment,
                args.step_fraction,
                split,
                detection_class,
                device,
                enhance=args.enhance,
            )
        )

    run_elapsed = round(time.perf_counter() - run_t0, 2)
    merged_config = merge_probe_results(existing_config, results)
    config_path = save_clip_tile_config(
        merged_config,
        config_path,
        source="preprocess_clips.py",
        extra_meta={
            "model_name": MODEL_NAME,
            "raw_confidence_threshold": RAW_CONFIDENCE_THRESHOLD,
            "probe_max_label_threshold": PROBE_MAX_LABEL_THRESHOLD,
            "fallback_tiles": FALLBACK_TILES,
            "fallback_label_threshold": FALLBACK_LABEL_THRESHOLD,
            "adaptive_label_threshold": "probe_max / target_tiles, floor=raw_confidence",
            "detection_class": detection_class,
            "probe_class_aliases": PROBE_CLASS_ALIASES,
            "probe_model_classes": probe_classes,
            "tile_candidates": list(TILE_CANDIDATES),
            "tile_headroom_steps": TILE_HEADROOM_STEPS,
            "frames_per_segment": args.per_segment,
            "frames_per_clip": args.per_segment * len(SEGMENTS),
            "frame_step_fraction": args.step_fraction,
            "last_probe_elapsed_sec": run_elapsed,
            "preprocess_note": (
                "car-only size/distance from start/middle/end samples; "
                "frame_step=floor(size×fraction/speed); "
                "label_box_stats.py remains optional post-label QA"
            ),
        },
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "detection_class": detection_class,
                "probe_class_aliases": PROBE_CLASS_ALIASES,
                "probe_model_classes": probe_classes,
                "frames_per_segment": args.per_segment,
                "frame_step_fraction": args.step_fraction,
                "run_elapsed_sec": run_elapsed,
                "enhance_clahe": bool(args.enhance),
                "clips": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"\n{'clip':<36} {'tiles':>5} {'size':>6} {'dist':>6} {'vary':>4} "
        f"{'step':>5} {'group':>9} {'sec':>6}"
    )
    print("-" * 90)
    for item in results:
        entry = probe_result_to_config_entry(item)
        tiles = str(entry["target_tiles"])
        size = (
            f"{item['object_size_px_median']:.0f}"
            if item.get("object_size_px_median") is not None
            else "-"
        )
        dist = f"{item['distance_m']:.0f}" if item.get("distance_m") is not None else "-"
        vary = "yes" if item.get("distance_varies") else "no"
        step = str(item["frame_step"]) if item.get("frame_step") is not None else "-"
        group = item.get("suggested_train_group") or "-"
        print(
            f"{item['clip']:<36} {tiles:>5} {size:>6} {dist:>6} {vary:>4} "
            f"{step:>5} {group:>9} {item['total_probe_sec']:>5.1f}s"
        )

    print(f"\nConfig: {config_path}")
    print(f"Report: {report_path} (total {run_elapsed}s)")


if __name__ == "__main__":
    main()
