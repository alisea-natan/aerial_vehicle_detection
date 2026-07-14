#!/usr/bin/env python3
"""Per-clip bbox size stats from pseudo-labels (train / eval separately).

Reports vehicle long-side in full-frame pixels and after the per-clip
scale_coeff crop→imgsz path. By default also writes scale_coeff into
config/clip_tiling.json (scale ≈ target_px / ff_p50, clamped).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CLIP_TILING_CONFIG_PATH,
    DEBUG_DIR,
    FRAMES_DIR,
    LABELS_DIR,
    SCALE_COEFF_MAX,
    SCALE_COEFF_MIN,
    TARGET_OBJECT_LONG_PX,
    TRAIN_IMGSZ,
    build_split_map,
    load_tiling_payload,
    scale_coeff_from_median,
    slice_size_from_scale,
)

SWEET_MIN = 32.0
SWEET_MAX = 96.0
OUT_DEFAULT = DEBUG_DIR / "label_box_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize labeled vehicle box sizes per video (train vs eval) "
            "and write scale_coeff into clip_tiling.json."
        ),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TRAIN_IMGSZ,
        help=f"YOLO imgsz used with scale_coeff (default {TRAIN_IMGSZ}).",
    )
    parser.add_argument(
        "--target-px",
        type=float,
        default=TARGET_OBJECT_LONG_PX,
        help=f"Target median long-side after imgsz (default {TARGET_OBJECT_LONG_PX}).",
    )
    parser.add_argument(
        "--sweet-min",
        type=float,
        default=SWEET_MIN,
        help=f"Lower end of desired long-side after imgsz (default {SWEET_MIN}).",
    )
    parser.add_argument(
        "--sweet-max",
        type=float,
        default=SWEET_MAX,
        help=f"Upper end of desired long-side after imgsz (default {SWEET_MAX}).",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_DEFAULT),
        help="JSON report path (default debug/label_box_stats.json).",
    )
    parser.add_argument(
        "--config",
        default=str(CLIP_TILING_CONFIG_PATH),
        help="clip_tiling.json to update with scale_coeff.",
    )
    parser.add_argument(
        "--no-write-scale-coeff",
        action="store_true",
        help="Only print/report stats; do not update clip_tiling.json.",
    )
    parser.add_argument("--clip", default=None, help="Only this clip stem.")
    return parser.parse_args()


def load_frame_size(clip_name: str) -> tuple[int, int] | None:
    meta_path = FRAMES_DIR / clip_name / "metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return int(meta["width"]), int(meta["height"])


def iter_boxes(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    boxes: list[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        xc, yc, bw, bh = map(float, parts[1:5])
        box_w = bw * img_w
        box_h = bh * img_h
        if box_w <= 0 or box_h <= 0:
            continue
        long_side = max(box_w, box_h)
        short_side = min(box_w, box_h)
        boxes.append({
            "w": box_w,
            "h": box_h,
            "long": long_side,
            "short": short_side,
            "area": box_w * box_h,
        })
    return boxes


def percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def side_stats(vals: list[float]) -> dict | None:
    if not vals:
        return None
    sorted_vals = sorted(vals)
    return {
        "mean": round(sum(vals) / len(vals), 1),
        "p10": round(percentile(sorted_vals, 10), 1),
        "p25": round(percentile(sorted_vals, 25), 1),
        "p50": round(percentile(sorted_vals, 50), 1),
        "p75": round(percentile(sorted_vals, 75), 1),
        "p90": round(percentile(sorted_vals, 90), 1),
        "min": round(sorted_vals[0], 1),
        "max": round(sorted_vals[-1], 1),
    }


def summarize_sizes(
    long_full: list[float],
    long_imgsz: list[float],
    *,
    sweet_min: float,
    sweet_max: float,
) -> dict:
    if not long_full:
        return {
            "n_boxes": 0,
            "fullframe_long_px": None,
            "at_imgsz_long_px": None,
            "pct_in_sweet_spot": None,
        }
    in_sweet = sum(1 for v in long_imgsz if sweet_min <= v <= sweet_max)
    return {
        "n_boxes": len(long_full),
        "fullframe_long_px": side_stats(long_full),
        "at_imgsz_long_px": side_stats(long_imgsz),
        "pct_in_sweet_spot": round(100.0 * in_sweet / len(long_imgsz), 1),
    }


def collect_clip_long_sides(split: str, clip_name: str) -> dict | None:
    label_dir = LABELS_DIR / split / clip_name
    if not label_dir.is_dir():
        return None
    size = load_frame_size(clip_name)
    if size is None:
        return {
            "clip": clip_name,
            "split": split,
            "error": "missing frames/metadata.json",
            "long_sides": [],
        }
    img_w, img_h = size
    long_sides: list[float] = []
    n_frames = 0
    n_labeled_frames = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        n_frames += 1
        boxes = iter_boxes(label_path, img_w, img_h)
        if not boxes:
            continue
        n_labeled_frames += 1
        long_sides.extend(box["long"] for box in boxes)
    return {
        "clip": clip_name,
        "split": split,
        "resolution": [img_w, img_h],
        "n_label_files": n_frames,
        "n_frames_with_boxes": n_labeled_frames,
        "long_sides": long_sides,
    }


def enrich_clip(
    raw: dict,
    *,
    imgsz: int,
    target_px: float,
    sweet_min: float,
    sweet_max: float,
) -> dict:
    if raw.get("error"):
        return raw
    long_sides: list[float] = raw.get("long_sides") or []
    img_w, img_h = raw["resolution"]
    ff = side_stats(long_sides)
    if not ff:
        return {
            **{k: v for k, v in raw.items() if k != "long_sides"},
            "n_boxes": 0,
            "fullframe_long_px": None,
            "at_imgsz_long_px": None,
            "pct_in_sweet_spot": None,
            "scale_coeff": None,
            "slice_size": None,
        }

    ff_p50 = float(ff["p50"])
    scale_coeff = scale_coeff_from_median(ff_p50, target_px=target_px)
    slice_size = slice_size_from_scale(scale_coeff, img_w, img_h, imgsz=imgsz)
    resize_scale = imgsz / slice_size
    long_imgsz = [v * resize_scale for v in long_sides]
    stats = summarize_sizes(long_sides, long_imgsz, sweet_min=sweet_min, sweet_max=sweet_max)

    raw_ideal = target_px / ff_p50
    note = (
        f"ff_p50={ff_p50:.1f} → {target_px:g}/{ff_p50:.1f}={raw_ideal:.2f}; "
        f"clamp[{SCALE_COEFF_MIN:g},{SCALE_COEFF_MAX:g}] → {scale_coeff}; "
        f"slice={slice_size} @ imgsz={imgsz}"
    )
    if raw_ideal < SCALE_COEFF_MIN or raw_ideal > SCALE_COEFF_MAX:
        note += " (clamped)"
    eff = imgsz / slice_size
    if abs(eff - scale_coeff) > 0.05:
        note += f"; frame-capped effective_scale≈{eff:.2f}"

    return {
        **{k: v for k, v in raw.items() if k != "long_sides"},
        **stats,
        "scale_coeff": scale_coeff,
        "slice_size": slice_size,
        "scale_coeff_note": note,
        "target_px": target_px,
        "imgsz": imgsz,
    }


def write_scale_coeffs(
    clips: list[dict],
    config_path: Path,
    *,
    imgsz: int,
    target_px: float,
) -> list[dict]:
    payload = load_tiling_payload(config_path)
    if not payload.get("clips"):
        payload["clips"] = {}

    written: list[dict] = []
    for clip in clips:
        name = clip["clip"]
        coeff = clip.get("scale_coeff")
        if coeff is None or clip.get("error") or not clip.get("n_boxes"):
            continue
        entry = payload["clips"].setdefault(name, {})
        prev = entry.get("scale_coeff")
        entry["scale_coeff"] = float(coeff)
        entry["scale_coeff_note"] = clip["scale_coeff_note"]
        written.append({
            "clip": name,
            "scale_coeff": float(coeff),
            "slice_size": clip.get("slice_size"),
            "previous": prev,
        })

    payload["train_imgsz"] = int(imgsz)
    payload["target_object_long_px"] = float(target_px)
    payload["scale_coeff_source"] = "label_box_stats.py"
    payload["scale_coeff_updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["scale_coeff_formula"] = (
        f"clamp({target_px:g} / ff_p50, {SCALE_COEFF_MIN:g}, {SCALE_COEFF_MAX:g}); "
        f"slice = floor32(min(imgsz / scale_coeff, frame_short_side))"
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return written


def fmt_row(clip: dict) -> str:
    if clip.get("error"):
        return f"{clip['clip']:<40} {'err':>7}"
    if not clip.get("n_boxes"):
        return (
            f"{clip['clip']:<40} {'0':>7} {'-':>7} {'-':>7} {'-':>5} "
            f"{'-':>7} {'-':>7} {'-':>6}"
        )
    ff = clip["fullframe_long_px"]
    im = clip["at_imgsz_long_px"]
    return (
        f"{clip['clip']:<40} {clip['n_boxes']:>7} "
        f"{ff['p50']:>7.0f} {ff['p90']:>7.0f} "
        f"{clip['scale_coeff']:>5.2f} {clip['slice_size']:>7} "
        f"{im['p50']:>7.0f} {clip['pct_in_sweet_spot']:>5.0f}%"
    )


def print_split_table(
    split: str,
    clips: list[dict],
    *,
    imgsz: int,
    target_px: float,
    sweet_min: float,
    sweet_max: float,
) -> None:
    print(
        f"\n=== {split} ===  (imgsz={imgsz}, target≈{target_px:g}px, "
        f"sweet={sweet_min:.0f}–{sweet_max:.0f}px)"
    )
    print(
        f"{'clip':<40} {'n_box':>7} {'ff_p50':>7} {'ff_p90':>7} "
        f"{'coeff':>5} {'slice':>7} {'im_p50':>7} {'sweet':>6}"
    )
    print("-" * 98)
    for clip in clips:
        print(fmt_row(clip))
    print("coeff/slice from ff_p50; im_p50 = fullframe × (imgsz/slice)")


def main() -> None:
    args = parse_args()
    clip_filter = args.clip
    if clip_filter and clip_filter.endswith(".mp4"):
        clip_filter = Path(clip_filter).stem

    split_map = build_split_map()
    if not split_map:
        raise SystemExit("No clips in data/train or data/eval.")

    by_split: dict[str, list[dict]] = {"train": [], "eval": []}
    all_clips: list[dict] = []
    for clip_name, split in sorted(split_map.items(), key=lambda kv: (kv[1], kv[0])):
        if clip_filter and clip_name != clip_filter:
            continue
        if split not in by_split:
            continue
        raw = collect_clip_long_sides(split, clip_name)
        if raw is None:
            print(f"skip {split}/{clip_name}: no labels/{split}/{clip_name}")
            continue
        clip = enrich_clip(
            raw,
            imgsz=args.imgsz,
            target_px=args.target_px,
            sweet_min=args.sweet_min,
            sweet_max=args.sweet_max,
        )
        by_split[split].append(clip)
        all_clips.append(clip)

    for split in ("train", "eval"):
        if by_split[split]:
            print_split_table(
                split,
                by_split[split],
                imgsz=args.imgsz,
                target_px=args.target_px,
                sweet_min=args.sweet_min,
                sweet_max=args.sweet_max,
            )
        else:
            print(f"\n=== {split} ===  (no clips)")

    # Drop long_sides from JSON payload (already removed in enrich).
    payload = {
        "imgsz": args.imgsz,
        "target_px": args.target_px,
        "sweet_spot_px": [args.sweet_min, args.sweet_max],
        "scale_coeff_formula": (
            f"clamp({args.target_px:g} / ff_p50, {SCALE_COEFF_MIN:g}, {SCALE_COEFF_MAX:g})"
        ),
        "note": (
            "fullframe_long_px: max(w,h) in original frame. "
            "scale_coeff/slice chosen so median ≈ target_px after imgsz "
            "(frame short-side may cap slice). "
            "at_imgsz_long_px uses that per-clip slice."
        ),
        "train": by_split["train"],
        "eval": by_split["eval"],
    }

    for split in ("train", "eval"):
        longs_ff: list[float] = []
        longs_im: list[float] = []
        for clip in by_split[split]:
            if clip.get("error") or not clip.get("n_boxes"):
                continue
            size = load_frame_size(clip["clip"])
            label_dir = LABELS_DIR / split / clip["clip"]
            if size is None or not label_dir.is_dir():
                continue
            img_w, img_h = size
            resize_scale = args.imgsz / clip["slice_size"]
            for label_path in label_dir.glob("*.txt"):
                for box in iter_boxes(label_path, img_w, img_h):
                    longs_ff.append(box["long"])
                    longs_im.append(box["long"] * resize_scale)
        payload[f"{split}_aggregate"] = summarize_sizes(
            longs_ff,
            longs_im,
            sweet_min=args.sweet_min,
            sweet_max=args.sweet_max,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")

    for split in ("train", "eval"):
        agg = payload[f"{split}_aggregate"]
        if not agg.get("n_boxes"):
            continue
        im = agg["at_imgsz_long_px"]
        print(
            f"{split} aggregate: n={agg['n_boxes']}, "
            f"at_imgsz p50={im['p50']:.0f} p90={im['p90']:.0f}, "
            f"in sweet spot={agg['pct_in_sweet_spot']:.0f}%"
        )

    if args.no_write_scale_coeff:
        print("Skipped writing scale_coeff (--no-write-scale-coeff).")
        return

    config_path = Path(args.config)
    written = write_scale_coeffs(
        all_clips,
        config_path,
        imgsz=args.imgsz,
        target_px=args.target_px,
    )
    print(f"\nUpdated scale_coeff in {config_path} ({len(written)} clips):")
    for item in written:
        prev = item["previous"]
        prev_s = f"{prev}" if prev is not None else "-"
        print(
            f"  {item['clip']}: {prev_s} → {item['scale_coeff']} "
            f"(slice={item['slice_size']})"
        )


if __name__ == "__main__":
    main()
