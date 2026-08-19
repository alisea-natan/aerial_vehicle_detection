#!/usr/bin/env python3
"""Build a fixed eval pack for all tests (eval clips only).

Same tiling as train (`train_groups` + `frame_step`), but never mixes train videos.
Autolabel and manual (CVAT) packs go to **separate folders**.

Manual / CVAT GT → data/datasets/eval_manual/:
  python src/training/prepare_eval.py

YOLO-World → data/datasets/eval_autolabel/:
  python src/training/prepare_eval.py --from-autolabel

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
from collections import defaultdict
from pathlib import Path

import yaml

from common.config import (
    LABELS_DIR,
    PROJECT_ROOT,
    TRAIN_IMGSZ,
    build_split_map,
)
from training.evaluate import DEFAULT_BAND_CLIPS
from training.train import (
    print_group_tiling_plan,
    slice_frames_to_dataset,
    iter_labeled_frames,
)

AUTOLABEL_LABELS = PROJECT_ROOT / "outputs" / "autolabel" / "labels"
DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
OUT_MANUAL = DATASETS_ROOT / "eval_manual"
OUT_AUTOLABEL = DATASETS_ROOT / "eval_autolabel"
DEFAULT_CLIPS = tuple(DEFAULT_BAND_CLIPS.values())
# Eval packs: after frame_step, cap samples per clip (long/fast clips otherwise flood).
DEFAULT_MAX_FRAMES_PER_CLIP = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (default: eval_manual / eval_autolabel).",
    )
    parser.add_argument(
        "--from-autolabel",
        action="store_true",
        help="Use outputs/autolabel/labels → data/datasets/eval_autolabel/.",
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
    return parser.parse_args()


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
    labels_root = args.labels_root
    if labels_root is None and from_autolabel:
        labels_root = AUTOLABEL_LABELS
    if labels_root is None:
        labels_root = LABELS_DIR
    if not labels_root.is_absolute():
        labels_root = PROJECT_ROOT / labels_root

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = OUT_AUTOLABEL if from_autolabel else OUT_MANUAL
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    if args.no_recreate and (out_dir / "data.yaml").is_file() and any(images_dir.glob("*.jpg")):
        print(f"Reusing existing eval pack at {out_dir}")
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
        "train": "images",
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
        "  Use: python src/training/evaluate.py --gt "
        f"{'autolabel' if from_autolabel else 'manual'}"
    )


if __name__ == "__main__":
    main()
