#!/usr/bin/env python3
"""Sample car tiles at labeling SAHI slice size (QA).

  python src/labeling/sample_tiles.py
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
import re
import shutil
from pathlib import Path

import cv2
from sahi.slicing import slice_image
from sahi.utils.coco import CocoAnnotation


from common.config import (
    CLIP_TILING_CONFIG_PATH,
    DEBUG_DIR,
    FRAMES_DIR,
    LABELS_DIR,
    build_split_map,
    clip_skip_reason,
    is_clip_skipped,
    load_clip_tile_config,
    resolve_clip_tile_config,
)
from common.detect import compute_slice_size

SAMPLES_DIR = DEBUG_DIR / "train_tile_samples"
CLASS_NAME = "vehicle"
MIN_AREA_RATIO = 0.1
DEFAULT_PER_CLIP = 2
DEFAULT_FRAME_CANDIDATES = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a few car-containing tiles per video at the labeling slice size "
            "from clip_tiling.json; report average tile size."
        ),
    )
    parser.add_argument("--clip", default=None, help="Only this clip (video stem).")
    parser.add_argument(
        "--per-clip",
        type=int,
        default=DEFAULT_PER_CLIP,
        help=f"Max sample tiles to keep per clip (default {DEFAULT_PER_CLIP}).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAME_CANDIDATES,
        help="Max labeled frames to scan per clip before giving up.",
    )
    parser.add_argument(
        "--out",
        default=str(SAMPLES_DIR),
        help="Output directory (default debug/train_tile_samples).",
    )
    parser.add_argument(
        "--config",
        default=str(CLIP_TILING_CONFIG_PATH),
        help="Clip tiling config path.",
    )
    return parser.parse_args()


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


def draw_boxes(image, annotations: list) -> None:
    for ann in annotations:
        x, y, w, h = ann.bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def hit_frame_from_note(note: str) -> str | None:
    match = re.search(r"probed frame (\d+)", note or "")
    return match.group(1) if match else None


def iter_clip_label_frames(
    clip_name: str,
    split: str | None,
    hit_frame: str | None,
    max_frames: int,
) -> list[tuple[Path, Path]]:
    """Prefer probe hit frame, then middle labeled frames."""
    if split not in ("train", "eval"):
        return []
    label_dir = LABELS_DIR / split / clip_name
    frame_dir = FRAMES_DIR / clip_name
    if not label_dir.is_dir() or not frame_dir.is_dir():
        return []

    labeled = []
    for label_path in sorted(label_dir.glob("*.txt")):
        if label_path.stat().st_size == 0:
            continue
        image_path = frame_dir / f"{label_path.stem}.jpg"
        if image_path.exists():
            labeled.append((image_path, label_path))
    if not labeled:
        return []

    ordered: list[tuple[Path, Path]] = []
    if hit_frame:
        for pair in labeled:
            if pair[0].stem == hit_frame:
                ordered.append(pair)
                break
    remaining = [p for p in labeled if not ordered or p[0] != ordered[0][0]]
    if remaining:
        center = len(remaining) // 2
        half = max_frames // 2
        start = max(0, center - half)
        end = min(len(remaining), start + max_frames)
        start = max(0, end - max_frames)
        ordered.extend(remaining[start:end])
    return ordered[:max_frames]


def sample_clip(
    clip_name: str,
    tile_cfg: dict,
    split: str | None,
    out_dir: Path,
    per_clip: int,
    max_frames: int,
) -> dict:
    meta_path = FRAMES_DIR / clip_name / "metadata.json"
    if not meta_path.exists():
        return {"clip": clip_name, "error": "missing frames/metadata"}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    width, height = int(meta["width"]), int(meta["height"])
    target_tiles = int(tile_cfg["target_tiles"])
    overlap_ratio = float(tile_cfg["overlap_ratio"])
    probe_min = tile_cfg.get("probe_min_tiles")

    if target_tiles <= 1:
        slice_h = height
        slice_w = width
        uses_sahi = False
    else:
        slice_h, slice_w = compute_slice_size(width, height, target_tiles)
        uses_sahi = True

    clip_out = out_dir / clip_name
    if clip_out.exists():
        shutil.rmtree(clip_out)
    clip_out.mkdir(parents=True, exist_ok=True)

    hit_frame = hit_frame_from_note(tile_cfg.get("note", ""))
    frame_pairs = iter_clip_label_frames(clip_name, split, hit_frame, max_frames)
    saved: list[dict] = []
    frames_scanned = 0

    for image_path, label_path in frame_pairs:
        if len(saved) >= per_clip:
            break
        frames_scanned += 1
        boxes = parse_yolo_labels(label_path, width, height)
        if not boxes:
            continue

        if not uses_sahi:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            for cls_id, xmin, ymin, box_w, box_h in boxes:
                x1, y1 = int(xmin), int(ymin)
                x2, y2 = int(xmin + box_w), int(ymin + box_h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            out_name = f"{image_path.stem}__full.jpg"
            out_path = clip_out / out_name
            cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved.append({
                "file": out_name,
                "frame": image_path.stem,
                "vehicles": len(boxes),
                "slice_wh": [slice_w, slice_h],
            })
            continue

        coco = yolo_boxes_to_coco(boxes)
        result = slice_image(
            str(image_path),
            coco_annotation_list=coco,
            output_file_name=f"{clip_name}__{image_path.stem}",
            output_dir=str(clip_out),
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            auto_slice_resolution=False,
            min_area_ratio=MIN_AREA_RATIO,
            out_ext=".jpg",
            verbose=False,
        )

        for sliced in result.sliced_image_list:
            if len(saved) >= per_clip:
                break
            anns = sliced.coco_image.annotations
            if not anns:
                image_file = clip_out / sliced.coco_image.file_name
                if image_file.exists():
                    image_file.unlink()
                continue

            image_file = clip_out / sliced.coco_image.file_name
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            draw_boxes(image, anns)
            cv2.imwrite(str(image_file), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved.append({
                "file": image_file.name,
                "frame": image_path.stem,
                "vehicles": len(anns),
                "slice_wh": [sliced.coco_image.width, sliced.coco_image.height],
            })

        # Drop leftover empty SAHI outputs from this frame.
        keep_names = {item["file"] for item in saved}
        for path in clip_out.glob(f"{clip_name}__{image_path.stem}_*"):
            if path.name not in keep_names:
                path.unlink(missing_ok=True)

    return {
        "clip": clip_name,
        "split": split,
        "resolution": [width, height],
        "target_tiles": target_tiles,
        "probe_min_tiles": probe_min,
        "overlap_ratio": overlap_ratio,
        "slice_size": [slice_w, slice_h],
        "distance_band": tile_cfg.get("distance_band"),
        "distance_m": tile_cfg.get("distance_m"),
        "frames_scanned": frames_scanned,
        "samples_saved": len(saved),
        "samples": saved,
        "note": tile_cfg.get("note", ""),
    }


def main() -> None:
    args = parse_args()
    clip_filter = args.clip
    if clip_filter and clip_filter.endswith(".mp4"):
        clip_filter = Path(clip_filter).stem

    config_path = Path(args.config)
    tile_config = load_clip_tile_config(config_path)
    if not tile_config:
        raise SystemExit(f"No clips in {config_path}. Run: python src/probe_clips.py")

    split_map = build_split_map()
    out_dir = Path(args.out)
    if out_dir.exists():
        # Only clear clip subdirs we will rewrite; keep manifest rewrite at end.
        pass
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for clip_name in sorted(tile_config):
        if clip_filter and clip_name != clip_filter:
            continue
        if is_clip_skipped(clip_name, tile_config):
            print(f"skip {clip_name}: {clip_skip_reason(clip_name, tile_config)}")
            continue
        split = split_map.get(clip_name) or tile_config[clip_name].get("split")
        if split not in ("train", "eval"):
            print(f"skip {clip_name}: not in train/eval")
            continue
        tile_cfg = resolve_clip_tile_config(clip_name, tile_config)
        # Preserve probe_min_tiles / note from raw config for reporting.
        raw = tile_config[clip_name]
        tile_cfg["probe_min_tiles"] = raw.get("probe_min_tiles")
        tile_cfg["note"] = raw.get("note", tile_cfg.get("note", ""))
        print(f"{split}/{clip_name}: sampling tiles={tile_cfg['target_tiles']} …")
        result = sample_clip(
            clip_name,
            tile_cfg,
            split,
            out_dir,
            args.per_clip,
            args.frames,
        )
        results.append(result)
        if result.get("error"):
            print(f"  error: {result['error']}")
        else:
            sw, sh = result["slice_size"]
            print(
                f"  slice={sw}x{sh}, saved {result['samples_saved']}/{args.per_clip} "
                f"(scanned {result['frames_scanned']} frames)"
            )

    sized = [r for r in results if "slice_size" in r and not r.get("error")]
    if sized:
        areas = [r["slice_size"][0] * r["slice_size"][1] for r in sized]
        sides = [r["slice_size"][0] for r in sized]  # square tiles from compute_slice_size
        mean_side = sum(sides) / len(sides)
        mean_area = sum(areas) / len(areas)
        # Suggest a round shared tile size near the mean (power-of-two-ish YOLO sizes).
        suggested = int(round(mean_side / 64) * 64)
        suggested = max(512, suggested)
    else:
        mean_side = mean_area = suggested = None

    manifest = {
        "source_config": str(config_path),
        "per_clip": args.per_clip,
        "clips": results,
        "summary": {
            "clips": len(sized),
            "slice_sides_px": [r["slice_size"][0] for r in sized],
            "mean_slice_side_px": round(mean_side, 1) if mean_side is not None else None,
            "mean_slice_area_px": round(mean_area, 1) if mean_area is not None else None,
            "suggested_shared_slice_px": suggested,
            "note": (
                "slice_size comes from compute_slice_size(resolution, target_tiles). "
                "If sides cluster, a single train imgsz near suggested_shared_slice_px is reasonable."
            ),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'clip':<40} {'tiles':>5} {'slice':>11} {'band':>8} {'n':>3}")
    print("-" * 75)
    for r in sized:
        sw, sh = r["slice_size"]
        print(
            f"{r['clip']:<40} {r['target_tiles']:>5} {f'{sw}x{sh}':>11} "
            f"{(r.get('distance_band') or '-'):>8} {r['samples_saved']:>3}"
        )
    if mean_side is not None:
        print(
            f"\nMean slice side: {mean_side:.0f}px "
            f"(suggested shared ≈ {suggested}px). Manifest: {manifest_path}"
        )
    else:
        print(f"\nNo samples written. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
