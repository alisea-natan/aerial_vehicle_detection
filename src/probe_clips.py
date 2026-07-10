#!/usr/bin/env python3
"""Probe minimum SAHI tiles on middle frames; write config/clip_tiling.json."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import (
    CLIP_TILING_CONFIG_PATH,
    FALLBACK_LABEL_THRESHOLD,
    FALLBACK_TILES,
    PROBE_CLASS_ALIASES,
    PROBE_MAX_LABEL_THRESHOLD,
    RAW_CONFIDENCE_THRESHOLD,
    DEBUG_DIR,
    build_split_map,
    iter_frame_clip_dirs,
    label_threshold_for_tiles,
    load_clip_tile_config,
    load_tiling_payload,
    merge_probe_results,
    overlap_for_tiles,
    probe_detection_class,
    probe_model_classes,
    save_clip_tile_config,
)
from detect import (
    MODEL_NAME,
    build_yolo_world,
    car_detection_record,
    compute_slice_size,
    detect_frame_probe,
    distance_band,
    pick_largest_car,
    resolve_camera_model,
)

PROBE_FRAMES = 5
TILE_CANDIDATES = (1, 2, 3, 4, 6, 8, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find minimum tiles for car detection; estimate distance; update clip config.",
    )
    parser.add_argument("--clip", default=None, help="Probe a single clip (video stem).")
    parser.add_argument("--frames", type=int, default=PROBE_FRAMES, help="Middle frames to probe per clip.")
    parser.add_argument(
        "--report",
        default=str(DEBUG_DIR / "tile_probe.json"),
        help="Detailed probe report JSON path.",
    )
    parser.add_argument(
        "--config",
        default=str(CLIP_TILING_CONFIG_PATH),
        help="Clip tiling config used by autolabel_yworld.py.",
    )
    return parser.parse_args()


def summarize_cars(cars: list[dict], focal_px: float) -> dict:
    records = []
    for det in cars:
        record = car_detection_record(det, focal_px)
        if record:
            records.append(record)

    primary_car = pick_largest_car(records)
    return {
        "car_count": len(records),
        "primary_car": primary_car,
    }


def select_probe_frames(frame_paths: list[Path], count: int) -> list[Path]:
    if len(frame_paths) <= count:
        return frame_paths
    center = len(frame_paths) // 2
    half = count // 2
    start = max(0, center - half)
    end = start + count
    if end > len(frame_paths):
        end = len(frame_paths)
        start = end - count
    return frame_paths[start:end]


def probe_clip(
    model,
    clip_dir: Path,
    frame_count: int,
    split: str | None,
    detection_class: str,
    device: str,
) -> dict:
    clip_name = clip_dir.name
    metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
    width, height = int(metadata["width"]), int(metadata["height"])
    camera = resolve_camera_model(clip_name, width, height)
    focal_px = camera["focal_px"]

    all_frames = sorted(clip_dir.glob("*.jpg"))
    frame_paths = select_probe_frames(all_frames, frame_count)
    ultra_model = model.model

    attempts = []
    min_tiles: int | None = None
    hit_summary: dict | None = None
    hit_frame: str | None = None
    hit_label_threshold: float | None = None

    for target_tiles in TILE_CANDIDATES:
        label_threshold = label_threshold_for_tiles(target_tiles)
        if target_tiles == TILE_CANDIDATES[0]:
            alias_note = ", ".join(
                f"{alias}→{target}"
                for alias, target in sorted(PROBE_CLASS_ALIASES.items())
                if target == detection_class
            ) or "none"
            print(
                f"  probe classes: {detection_class!r} (aliases: {alias_note}), "
                f"adaptive threshold (tiles=1 -> {PROBE_MAX_LABEL_THRESHOLD})"
            )
        print(f"  trying tiles={target_tiles}, label_threshold>={label_threshold}")

        t0 = time.perf_counter()
        frame_hits = []

        for frame_path in frame_paths:
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
            )
            summary = summarize_cars(cars, focal_px)
            if summary["car_count"] > 0:
                frame_hits.append({
                    "frame": frame_path.stem,
                    **summary,
                })

        elapsed_sec = round(time.perf_counter() - t0, 2)
        slice_size = None if target_tiles <= 1 else list(compute_slice_size(width, height, target_tiles))
        attempt = {
            "target_tiles": target_tiles,
            "label_threshold": label_threshold,
            "overlap_ratio": 0.0 if target_tiles <= 1 else overlap_for_tiles(target_tiles),
            "slice_size": slice_size,
            "elapsed_sec": elapsed_sec,
            "frames_with_cars": len(frame_hits),
            "total_cars": sum(hit["car_count"] for hit in frame_hits),
        }
        attempts.append(attempt)

        if frame_hits and min_tiles is None:
            min_tiles = target_tiles
            hit_label_threshold = label_threshold
            best_hit = max(frame_hits, key=lambda hit: hit["car_count"])
            hit_summary = best_hit
            hit_frame = best_hit["frame"]
            print(
                f"  tiles={target_tiles}: car HIT on {len(frame_hits)}/{len(frame_paths)} frames "
                f"in {elapsed_sec}s — stopping"
            )
            break

        print(f"  tiles={target_tiles}: no cars in {elapsed_sec}s")

    primary_car = hit_summary["primary_car"] if hit_summary else None
    distance_m = primary_car["distance_m"] if primary_car else None
    distance_source = "largest_car" if primary_car else None
    if min_tiles is not None and hit_label_threshold is None:
        hit_label_threshold = label_threshold_for_tiles(min_tiles)

    total_elapsed = round(sum(item["elapsed_sec"] for item in attempts), 2)
    return {
        "clip": clip_name,
        "split": split,
        "detection_class": detection_class,
        "label_threshold": hit_label_threshold,
        "resolution": [width, height],
        "frames_probed": [path.stem for path in frame_paths],
        "camera": camera,
        "min_tiles": min_tiles,
        "hit_frame": hit_frame,
        "distance_m": distance_m,
        "distance_source": distance_source,
        "distance_band": distance_band(distance_m),
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
    clip_dirs = iter_frame_clip_dirs(clip_filter)
    if not clip_dirs:
        raise SystemExit("No frame folders found under data/frames/.")

    config_path = Path(args.config)
    tiling_payload = load_tiling_payload(config_path)
    existing_config = load_clip_tile_config(config_path)
    detection_class = probe_detection_class(tiling_payload)
    probe_classes = probe_model_classes(detection_class)
    print(f"Loading {MODEL_NAME} on probe classes {probe_classes}")
    model, device = build_yolo_world(probe_classes)

    results = []
    run_t0 = time.perf_counter()
    for clip_dir in clip_dirs:
        split = split_map.get(clip_dir.name)
        split_label = split or "unknown"
        print(f"\n{split_label}/{clip_dir.name}: probing {args.frames} middle frames")
        results.append(probe_clip(model, clip_dir, args.frames, split, detection_class, device))

    run_elapsed = round(time.perf_counter() - run_t0, 2)
    merged_config = merge_probe_results(existing_config, results)
    config_path = save_clip_tile_config(
        merged_config,
        config_path,
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
            "frames_per_clip": args.frames,
            "last_probe_elapsed_sec": run_elapsed,
        },
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
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
                "frames_per_clip": args.frames,
                "run_elapsed_sec": run_elapsed,
                "clips": results,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'clip':<40} {'tiles':>5} {'dist_m':>8} {'band':>8} {'probe_s':>8}")
    print("-" * 75)
    for item in results:
        dist = f"{item['distance_m']:.0f}" if item["distance_m"] is not None else "-"
        tiles = str(item["min_tiles"]) if item["min_tiles"] is not None else "-"
        band = item["distance_band"] or "-"
        print(
            f"{item['clip']:<40} {tiles:>5} {dist:>8} {band:>8} "
            f"{item['total_probe_sec']:>7.1f}s"
        )
    print(f"\nConfig: {config_path}")
    print(f"Report: {report_path} (total {run_elapsed}s)")


if __name__ == "__main__":
    main()
