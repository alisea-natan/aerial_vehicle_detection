#!/usr/bin/env python3
"""Build a fixed eval pack for all tests (eval clips only).

Same tiling as train (`train_groups` + `frame_step`), but never mixes train videos.
Autolabel and manual (CVAT) packs go to **separate folders**.

Manual / CVAT GT → data/datasets/eval_manual/:
  python src/training/prepare_eval.py

YOLO-World → data/datasets/eval_autolabel/:
  python src/training/prepare_eval.py --from-autolabel

Scale-adapted packs (copy native pack, pad only oversized clips so letterbox shrinks cars):
  python src/training/prepare_eval.py --from-autolabel              # native first
  python src/training/prepare_eval.py --from-autolabel --scale-adapt  # → *_adapted/
  → same stems as native; only close-band images change

Clips default to the two distance-band eval videos used by evaluate.py.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path


def _ensure_src_on_path() -> None:
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
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import yaml

from common.config import (
    LABELS_DIR,
    PROJECT_ROOT,
    TRAIN_IMGSZ,
    build_split_map,
    load_clip_tile_config,
)
from training.evaluate import DEFAULT_BAND_CLIPS, resolve_clip_eval_band
from training.train import (
    print_group_tiling_plan,
    slice_frames_to_dataset,
    iter_labeled_frames,
)

AUTOLABEL_LABELS = PROJECT_ROOT / "outputs" / "autolabel" / "labels"
DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
OUT_MANUAL = DATASETS_ROOT / "eval_manual"
OUT_AUTOLABEL = DATASETS_ROOT / "eval_autolabel"
OUT_MANUAL_ADAPTED = DATASETS_ROOT / "eval_manual_adapted"
OUT_AUTOLABEL_ADAPTED = DATASETS_ROOT / "eval_autolabel_adapted"
DEFAULT_CLIPS = tuple(DEFAULT_BAND_CLIPS.values())
DEFAULT_REF_DATASET = DATASETS_ROOT / "baseline_v0"
# Eval packs: after frame_step, cap samples per clip (long/fast clips otherwise flood).
DEFAULT_MAX_FRAMES_PER_CLIP = 64
# Only shrink oversized eval cars toward train; never enlarge tiny ones.
SCALE_ADAPT_MAX = 1.0
SCALE_ADAPT_MIN = 0.25
SCALE_ADAPT_NOOP = 0.98  # skip rewrite when nearly matched
# YOLO letterbox fill (Ultralytics default)
LETTERBOX_PAD_VALUE = (114, 114, 114)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output root (default: eval_manual / eval_autolabel, or "
            "*_adapted when --scale-adapt)."
        ),
    )
    parser.add_argument(
        "--from-autolabel",
        action="store_true",
        help="Use outputs/autolabel/labels → data/datasets/eval_autolabel[/_adapted]/.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=None,
        help="Override label tree (must contain eval/<clip>/*.txt).",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=None,
        help="Eval clip stem (repeatable). Default: band clips from evaluate.py.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TRAIN_IMGSZ,
        help=f"Model imgsz metadata (default {TRAIN_IMGSZ}).",
    )
    parser.add_argument(
        "--full-frames",
        action="store_true",
        help="Keep every labeled frame (default: frame_step only, same as train packs).",
    )
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=DEFAULT_MAX_FRAMES_PER_CLIP,
        help=(
            f"After frame_step, evenly subsample each clip to at most this many "
            f"labeled frames (default {DEFAULT_MAX_FRAMES_PER_CLIP}). "
            "Stops long videos with a small frame_step from flooding the eval pack; "
            "clips already under the cap are unchanged. 0 = no cap."
        ),
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Reuse existing pack if present.",
    )
    parser.add_argument(
        "--scale-adapt",
        action="store_true",
        help=(
            "After tiling, pad each oversized eval clip onto a larger canvas so that "
            "after YOLO letterbox to imgsz, median vehicle size (network pixels) matches "
            "the train pack for that distance band. Whole-image resize is NOT used "
            "(letterbox would cancel it). Writes eval_*_adapted/ by default."
        ),
    )
    parser.add_argument(
        "--ref-dataset",
        type=Path,
        default=DEFAULT_REF_DATASET,
        help=(
            "Train pack used as scale reference when --scale-adapt "
            f"(default {DEFAULT_REF_DATASET.relative_to(PROJECT_ROOT)})."
        ),
    )
    parser.add_argument(
        "--ref-split",
        default="train",
        help="Split under --ref-dataset for scale stats (default train).",
    )
    return parser.parse_args()


def _clip_from_stem(stem: str) -> str:
    return stem.split("__", 1)[0]


def _iter_network_shorts(
    image_path: Path,
    label_path: Path,
    imgsz: int,
) -> list[float]:
    """Vehicle short-sides in pixels after letterbox to imgsz (what the detector sees)."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    h, w = image.shape[:2]
    scale = float(imgsz) / float(max(h, w))
    shorts: list[float] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        bw, bh = float(parts[3]) * w * scale, float(parts[4]) * h * scale
        if bw > 0 and bh > 0:
            shorts.append(min(bw, bh))
    return shorts


def median_network_short_by_clip(
    images_dir: Path,
    labels_dir: Path,
    imgsz: int,
) -> dict[str, float]:
    by_clip: dict[str, list[float]] = defaultdict(list)
    for label_path in labels_dir.glob("*.txt"):
        clip = _clip_from_stem(label_path.stem)
        image_path = images_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            image_path = images_dir / f"{label_path.stem}.png"
        by_clip[clip].extend(_iter_network_shorts(image_path, label_path, imgsz))
    return {clip: float(median(vals)) for clip, vals in by_clip.items() if vals}


def train_band_network_medians(
    ref_dataset: Path,
    ref_split: str,
    imgsz: int,
) -> tuple[dict[str, float], dict[str, float], Path]:
    """Return (band → median network-px short, clip → median, labels_dir used)."""
    labels_dir = ref_dataset / ref_split / "labels"
    images_dir = ref_dataset / ref_split / "images"
    if not labels_dir.is_dir():
        labels_dir = ref_dataset / "labels"
        images_dir = ref_dataset / "images"
    if not labels_dir.is_dir() or not images_dir.is_dir():
        raise SystemExit(
            f"Scale reference images/labels not found under {ref_dataset} "
            f"(tried {ref_split}/ and flat layout)."
        )

    clip_medians = median_network_short_by_clip(images_dir, labels_dir, imgsz)
    if not clip_medians:
        raise SystemExit(f"No labeled boxes in reference pack: {labels_dir}")

    tile_config = load_clip_tile_config()
    band_values: dict[str, list[float]] = defaultdict(list)
    for clip, med in clip_medians.items():
        tile_cfg = tile_config.get(clip) or {}
        band = resolve_clip_eval_band(clip, tile_cfg)
        if band is None:
            continue
        band_values[band].append(med)

    band_medians = {band: float(median(vals)) for band, vals in band_values.items() if vals}
    if not band_medians:
        global_med = float(median(list(clip_medians.values())))
        band_medians = {band: global_med for band in DEFAULT_BAND_CLIPS}
    return band_medians, clip_medians, labels_dir


def _remap_yolo_labels_for_pad(
    src_lab: Path,
    out_lab: Path,
    *,
    src_w: int,
    src_h: int,
    canvas_w: int,
    canvas_h: int,
    offset_x: int,
    offset_y: int,
) -> None:
    """Rewrite YOLO-normalized labels after pasting the image onto a larger canvas."""
    if not src_lab.exists():
        return
    lines_out: list[str] = []
    for line in src_lab.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = parts[0]
        xc, yc, bw, bh = map(float, parts[1:5])
        # Absolute on source, then on canvas, then re-normalize.
        abs_xc = xc * src_w + offset_x
        abs_yc = yc * src_h + offset_y
        abs_bw = bw * src_w
        abs_bh = bh * src_h
        lines_out.append(
            f"{cls_id} {abs_xc / canvas_w:.6f} {abs_yc / canvas_h:.6f} "
            f"{abs_bw / canvas_w:.6f} {abs_bh / canvas_h:.6f}"
        )
    out_lab.write_text(("\n".join(lines_out) + "\n") if lines_out else "", encoding="utf-8")


def pad_image_to_shrink_network_objects(
    image: np.ndarray,
    shrink: float,
) -> tuple[np.ndarray, int, int]:
    """Pad image onto a larger canvas so letterbox shrinks objects by `shrink` (< 1).

    Letterbox scale = imgsz / max_side. Inflating max_side by 1/shrink makes
    network object size ≈ shrink × original (without blurring the content).
    Returns (canvas, offset_x, offset_y).
    """
    h, w = image.shape[:2]
    # New canvas ≈ source / shrink (same aspect) → max_side grows by 1/shrink.
    canvas_w = max(w, int(round(w / shrink)))
    canvas_h = max(h, int(round(h / shrink)))
    canvas = np.full((canvas_h, canvas_w, 3), LETTERBOX_PAD_VALUE, dtype=image.dtype)
    ox = (canvas_w - w) // 2
    oy = (canvas_h - h) // 2
    canvas[oy : oy + h, ox : ox + w] = image
    return canvas, ox, oy


def scale_adapt_pack(
    src_dir: Path,
    out_dir: Path,
    *,
    ref_dataset: Path,
    ref_split: str,
    band_by_clip: dict[str, str | None],
    imgsz: int,
) -> dict:
    """Pad oversized eval clips so letterboxed cars match train network size."""
    src_images = src_dir / "images"
    src_labels = src_dir / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise SystemExit(f"Source pack incomplete for scale-adapt: {src_dir}")

    band_medians, ref_clip_medians, ref_labels = train_band_network_medians(
        ref_dataset, ref_split, imgsz
    )
    eval_clip_medians = median_network_short_by_clip(src_images, src_labels, imgsz)

    if out_dir.resolve() != src_dir.resolve():
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        images_dir = out_dir / "images"
        labels_dir = out_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
    else:
        images_dir = src_images
        labels_dir = src_labels

    per_clip: dict[str, dict] = {}
    for clip, eval_med in sorted(eval_clip_medians.items()):
        band = band_by_clip.get(clip)
        if not band or band not in band_medians:
            target = float(median(list(band_medians.values())))
            band_used = "global_train"
        else:
            target = band_medians[band]
            band_used = band
        raw_scale = target / eval_med if eval_med > 0 else 1.0
        scale = max(SCALE_ADAPT_MIN, min(SCALE_ADAPT_MAX, raw_scale))
        per_clip[clip] = {
            "band": band_used,
            "eval_median_network_px": round(eval_med, 3),
            "train_band_median_network_px": round(target, 3),
            "shrink": round(scale, 6),
            "applied": scale < SCALE_ADAPT_NOOP,
        }

    image_paths = sorted(src_images.glob("*.jpg")) + sorted(src_images.glob("*.png"))
    n_padded = 0
    n_copied = 0
    for image_path in image_paths:
        clip = _clip_from_stem(image_path.stem)
        info = per_clip.get(clip)
        shrink = float(info["shrink"]) if info else 1.0
        apply = bool(info and info["applied"])
        out_img = images_dir / image_path.name
        out_lab = labels_dir / f"{image_path.stem}.txt"
        src_lab = src_labels / f"{image_path.stem}.txt"

        if not apply:
            if out_img.resolve() != image_path.resolve():
                shutil.copy2(image_path, out_img)
            if src_lab.exists() and out_lab.resolve() != src_lab.resolve():
                shutil.copy2(src_lab, out_lab)
            n_copied += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read {image_path}")
        src_h, src_w = image.shape[:2]
        canvas, ox, oy = pad_image_to_shrink_network_objects(image, shrink)
        if not cv2.imwrite(str(out_img), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"Failed to write {out_img}")
        _remap_yolo_labels_for_pad(
            src_lab,
            out_lab,
            src_w=src_w,
            src_h=src_h,
            canvas_w=canvas.shape[1],
            canvas_h=canvas.shape[0],
            offset_x=ox,
            offset_y=oy,
        )
        n_padded += 1

    ref_disp = (
        ref_dataset.relative_to(PROJECT_ROOT)
        if ref_dataset.is_relative_to(PROJECT_ROOT)
        else ref_dataset
    )
    meta = {
        "scale_adapted": True,
        "method": "letterbox_aware_pad",
        "ref_dataset": str(ref_disp),
        "ref_split": ref_split,
        "ref_labels": str(
            ref_labels.relative_to(PROJECT_ROOT)
            if ref_labels.is_relative_to(PROJECT_ROOT)
            else ref_labels
        ),
        "imgsz": imgsz,
        "metric": "median_vehicle_short_side_after_letterbox_to_imgsz",
        "rule": (
            f"per-clip shrink = clamp(train_band_net_px / eval_clip_net_px, "
            f"{SCALE_ADAPT_MIN}, {SCALE_ADAPT_MAX}); only shrink; "
            "pad content onto larger gray canvas (114) so YOLO letterbox makes "
            "cars smaller in-network (isotropic resize would be canceled by letterbox); "
            "YOLO labels remapped to the padded canvas"
        ),
        "train_band_medians_network_px": {
            k: round(v, 3) for k, v in sorted(band_medians.items())
        },
        "train_clip_medians_network_px": {
            k: round(v, 3) for k, v in sorted(ref_clip_medians.items())
        },
        "per_clip": per_clip,
        "n_images_padded": n_padded,
        "n_images_unscaled": n_copied,
    }
    print("Scale-adapt (pad → smaller cars after letterbox):")
    for clip, info in per_clip.items():
        flag = (
            f"shrink={info['shrink']:.3f} (pad canvas)"
            if info["applied"]
            else "noop (already ≤ train)"
        )
        print(
            f"  {clip} [{info['band']}]: "
            f"eval_net={info['eval_median_network_px']:.1f}px → "
            f"train_net={info['train_band_median_network_px']:.1f}px ({flag})"
        )
    print(f"  padded={n_padded}, unscaled_copy={n_copied}")
    return meta


def thin_frames_per_clip(
    frames: list[tuple[str, Path, Path]],
    max_per_clip: int,
) -> tuple[list[tuple[str, Path, Path]], dict[str, dict[str, int]]]:
    """Evenly subsample each clip to ≤ max_per_clip (eval coverage, not dense train)."""
    if max_per_clip <= 0:
        return frames, {}

    by_clip: dict[str, list[tuple[str, Path, Path]]] = defaultdict(list)
    for item in frames:
        by_clip[item[0]].append(item)

    out: list[tuple[str, Path, Path]] = []
    stats: dict[str, dict[str, int]] = {}
    for clip in sorted(by_clip):
        items = by_clip[clip]
        before = len(items)
        if before <= max_per_clip:
            kept = items
        else:
            # Inclusive endpoints; unique indices across the clip timeline.
            idxs = sorted(
                {
                    int(round(i * (before - 1) / (max_per_clip - 1)))
                    for i in range(max_per_clip)
                }
            )
            kept = [items[i] for i in idxs]
        out.extend(kept)
        stats[clip] = {"before": before, "after": len(kept), "cap": max_per_clip}
    return out, stats


def main() -> None:
    args = parse_args()

    from_autolabel = bool(args.from_autolabel)
    scale_adapt = bool(args.scale_adapt)
    labels_root = args.labels_root
    if labels_root is None and from_autolabel:
        labels_root = AUTOLABEL_LABELS
    if labels_root is None:
        labels_root = LABELS_DIR
    if not labels_root.is_absolute():
        labels_root = PROJECT_ROOT / labels_root

    out_dir = args.out_dir
    if out_dir is None:
        if scale_adapt:
            out_dir = OUT_AUTOLABEL_ADAPTED if from_autolabel else OUT_MANUAL_ADAPTED
        else:
            out_dir = OUT_AUTOLABEL if from_autolabel else OUT_MANUAL
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    if args.no_recreate and (out_dir / "data.yaml").is_file() and any(images_dir.glob("*.jpg")):
        print(f"Reusing existing eval pack at {out_dir}")
        return

    # Scale-adapt copies the native pack, then pads only oversized clips.
    # Unchanged clips (e.g. mid-band) stay byte-identical → same scores there.
    if scale_adapt:
        native_dir = OUT_AUTOLABEL if from_autolabel else OUT_MANUAL
        if not (native_dir / "data.yaml").is_file() or not any(
            (native_dir / "images").glob("*.jpg")
        ):
            raise SystemExit(
                f"Scale-adapt needs the native eval pack at {native_dir}.\n"
                "Build it first (same frames), then adapt:\n"
                "  python src/training/prepare_eval.py"
                + (" --from-autolabel" if from_autolabel else "")
                + "\n  python src/training/prepare_eval.py"
                + (" --from-autolabel" if from_autolabel else "")
                + " --scale-adapt"
            )

        native_yaml = yaml.safe_load((native_dir / "data.yaml").read_text(encoding="utf-8")) or {}
        band_by_clip = {
            str(k): (str(v) if v is not None else None)
            for k, v in (native_yaml.get("bands") or {}).items()
        }
        if not band_by_clip:
            band_by_clip = {clip: band for band, clip in DEFAULT_BAND_CLIPS.items()}

        ref_dataset = args.ref_dataset
        if not ref_dataset.is_absolute():
            ref_dataset = PROJECT_ROOT / ref_dataset

        native_disp = (
            native_dir.relative_to(PROJECT_ROOT)
            if native_dir.is_relative_to(PROJECT_ROOT)
            else native_dir
        )
        out_disp = (
            out_dir.relative_to(PROJECT_ROOT)
            if out_dir.is_relative_to(PROJECT_ROOT)
            else out_dir
        )
        print(
            f"Scale-adapt from native pack → {out_disp}\n"
            f"  source={native_disp} (same stems; pad only oversized clips)\n"
            f"  imgsz={native_yaml.get('imgsz', args.imgsz)}"
        )
        scale_meta = scale_adapt_pack(
            native_dir,
            out_dir,
            ref_dataset=ref_dataset,
            ref_split=str(args.ref_split),
            band_by_clip=band_by_clip,
            imgsz=int(native_yaml.get("imgsz", args.imgsz)),
        )

        payload = {
            **{k: v for k, v in native_yaml.items() if k not in ("scale_adapt", "role")},
            "role": "fixed_eval_scale_adapted",
            "scale_adapted": True,
            "adapted_from": str(native_disp),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scale_adapt": scale_meta,
        }
        yaml_path = out_dir / "data.yaml"
        yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        (out_dir / "prepare_stats.json").write_text(
            json.dumps({"meta": payload, "adapted_from": str(native_disp)}, indent=2) + "\n",
            encoding="utf-8",
        )
        n_img = len(list((out_dir / "images").glob("*.jpg")))
        print(
            f"\nFixed eval saved: {yaml_path}\n"
            f"  samples={n_img} (identical stems to native; only padded clips differ)\n"
            "  Scale-adapted diagnostic pack — use: "
            f"python src/training/evaluate.py --gt "
            f"{'autolabel_adapted' if from_autolabel else 'manual_adapted'}"
        )
        return

    split_map = build_split_map()
    if not split_map:
        raise SystemExit("No clips in data/train or data/eval.")

    want_clips = set(args.clip) if args.clip else set(DEFAULT_CLIPS)
    for clip in sorted(want_clips):
        if split_map.get(clip) != "eval":
            raise SystemExit(
                f"Clip {clip!r} is not under data/eval/. "
                "Fixed eval pack must use eval videos only."
            )

    frame_step_only = not bool(args.full_frames)
    all_eval = iter_labeled_frames(
        "eval",
        split_map,
        frame_step_only=frame_step_only,
        labels_root=labels_root,
    )
    frames = [(c, img, lab) for c, img, lab in all_eval if c in want_clips]
    if not frames:
        raise SystemExit(
            f"No eval frames for clips {sorted(want_clips)} under {labels_root / 'eval'}.\n"
            "CVAT: python src/labeling/cvat/cvat_pull.py --verify --sync-labels\n"
            "Autolabel: python src/labeling/autolabel.py"
        )

    missing = want_clips - {c for c, _, _ in frames}
    if missing:
        print(f"Warning: no frames for clips: {sorted(missing)}")

    max_per_clip = int(args.max_frames_per_clip)
    frames, thin_stats = thin_frames_per_clip(frames, max_per_clip)
    if thin_stats:
        print("Per-clip frame cap (after frame_step):")
        for clip, st in thin_stats.items():
            if st["before"] != st["after"]:
                print(
                    f"  {clip}: {st['before']} → {st['after']} "
                    f"(cap={st['cap']}, even subsample)"
                )
            else:
                print(f"  {clip}: {st['before']} (under cap={st['cap']})")

    labels_disp = (
        labels_root.relative_to(PROJECT_ROOT)
        if labels_root.is_relative_to(PROJECT_ROOT)
        else labels_root
    )
    out_disp = (
        out_dir.relative_to(PROJECT_ROOT)
        if out_dir.is_relative_to(PROJECT_ROOT)
        else out_dir
    )
    print(
        f"Building fixed eval pack → {out_disp}\n"
        f"  labels_root={labels_disp}\n"
        f"  clips={sorted(want_clips)}\n"
        f"  frames={len(frames)}, frame_step_only={frame_step_only}, "
        f"max_frames_per_clip={max_per_clip or 'none'}, "
        f"imgsz={args.imgsz}"
    )
    print_group_tiling_plan(frames)

    stats = slice_frames_to_dataset(
        frames,
        images_dir,
        labels_dir,
        imgsz=args.imgsz,
        max_empty_slices_per_frame=0,
        progress_label="eval",
    )
    print(
        f"Eval samples: {stats['slices_kept']} kept from {stats['source_frames']} frames "
        f"({stats['slices_with_labels']} with labels, {stats['slices_dropped']} empty dropped)"
    )
    if stats.get("clip_group"):
        print("Per-clip crop:")
        for name in sorted(stats["clip_group"]):
            group = stats["clip_group"][name]
            uses = (stats.get("clip_uses_tiling") or {}).get(name, True)
            tile = (
                "full-frame"
                if not uses
                else f"tile={(stats.get('clip_slice_size') or {}).get(name, '?')}"
            )
            print(f"  {name}: group={group}, {tile}")

    band_by_clip = {clip: band for band, clip in DEFAULT_BAND_CLIPS.items()}
    payload = {
        "path": ".",
        "train": "images",  # unused; pack is eval-only
        "val": "images",
        "nc": 1,
        "names": ["vehicle"],
        "imgsz": int(args.imgsz),
        "role": "fixed_eval",
        "val_source": "eval_clips_only",
        "labels_root": str(labels_disp),
        "clips": sorted(want_clips),
        "bands": {c: band_by_clip.get(c) for c in sorted(want_clips)},
        "frame_step_only": frame_step_only,
        "max_frames_per_clip": max_per_clip if max_per_clip > 0 else None,
        "frames_per_clip_after_cap": thin_stats or None,
        "from_autolabel": from_autolabel,
        "source": "autolabel" if from_autolabel else "manual_cvat",
        "scale_adapted": False,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clip_slice_size": stats.get("clip_slice_size") or {},
        "clip_overlap": stats.get("clip_overlap") or {},
        "clip_train_imgsz": stats.get("clip_train_imgsz") or {},
        "clip_uses_tiling": stats.get("clip_uses_tiling") or {},
        "clip_group": stats.get("clip_group") or {},
    }

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (out_dir / "prepare_stats.json").write_text(
        json.dumps({"eval": stats, "meta": payload}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"\nFixed eval saved: {yaml_path}\n"
        f"  samples={stats['slices_kept']} "
        f"(labeled={stats['slices_with_labels']}, empty_dropped={stats['slices_dropped']})\n"
        "  Primary eval pack — use: "
        f"python src/training/evaluate.py --gt "
        f"{'autolabel' if from_autolabel else 'manual'}"
    )


if __name__ == "__main__":
    main()
