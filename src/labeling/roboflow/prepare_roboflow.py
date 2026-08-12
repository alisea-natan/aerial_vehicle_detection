#!/usr/bin/env python3
"""Build labelling/roboflow/roboflow/ packs from frames + labels/ (CVAT via cvat_pull).

  python src/labeling/roboflow/prepare_roboflow.py --clean
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
import shutil
from pathlib import Path


from common.config import (
    LABELS_DIR,
    PROJECT_ROOT,
    SPLITS,
    clip_skip_reason,
    is_clip_skipped,
)

FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
OUT_DIR = PROJECT_ROOT / "labelling" / "roboflow" / "roboflow"
CLASS_NAMES = ["vehicle"]

DATA_YAML_TEMPLATE = """\
train: images
val: images

nc: {nc}
names: {names}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare roboflow/ clip folders.")
    parser.add_argument("--clean", action="store_true", help="Remove out/ first.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinks.")
    return parser.parse_args()


def find_cvat_label_dir(clip: str) -> Path | None:
    for split in SPLITS:
        path = LABELS_DIR / split / clip
        if path.is_dir() and any(path.glob("*.txt")):
            return path
    return None


def prepare_clip(
    clip: str,
    frames_root: Path,
    out_root: Path,
    *,
    copy_images: bool,
) -> tuple[int, int]:
    """Returns (n_frames, n_labels)."""
    frame_dir = frames_root / clip
    jpg_paths = sorted(frame_dir.glob("*.jpg"))
    if not jpg_paths:
        print(f"[skip] {clip}: no .jpg under {frame_dir}")
        return 0, 0

    clip_out = out_root / clip
    images_dir = clip_out / "images"
    labels_dir = clip_out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)

    if labels_dir.is_dir():
        shutil.rmtree(labels_dir)

    for jpg in jpg_paths:
        dest = images_dir / jpg.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if copy_images:
            shutil.copy2(jpg, dest)
        else:
            dest.symlink_to(jpg.resolve())

    cvat_dir = find_cvat_label_dir(clip)
    n_labels = 0
    if cvat_dir is not None:
        labels_dir.mkdir(parents=True, exist_ok=True)
        frame_stems = {jpg.stem for jpg in jpg_paths}
        for src_txt in sorted(cvat_dir.glob("*.txt")):
            if src_txt.name.lower() == "classes.txt":
                continue
            if src_txt.stem not in frame_stems:
                continue
            shutil.copy2(src_txt, labels_dir / src_txt.name)
            n_labels += 1

    (clip_out / "data.yaml").write_text(
        DATA_YAML_TEMPLATE.format(nc=len(CLASS_NAMES), names=CLASS_NAMES),
        encoding="utf-8",
    )

    if cvat_dir is None:
        print(f"[ok] {clip}: {len(jpg_paths)} images, no CVAT labels in labels/")
    else:
        print(f"[ok] {clip}: {len(jpg_paths)} images, {n_labels} labels from {cvat_dir}")
    return len(jpg_paths), n_labels


def main() -> None:
    args = parse_args()
    frames_root = FRAMES_DIR
    out_root = OUT_DIR

    if not frames_root.is_dir():
        raise SystemExit(f"Frames dir not found: {frames_root}")

    clips = []
    for p in sorted(frames_root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if is_clip_skipped(p.name):
            print(f"[skip] {p.name}: {clip_skip_reason(p.name)}")
            continue
        clips.append(p.name)
    if not clips:
        raise SystemExit(f"No clips under {frames_root}")

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(clips)} clips → {out_root}")

    total_frames = 0
    total_labels = 0
    for clip in clips:
        n_f, n_l = prepare_clip(
            clip, frames_root, out_root, copy_images=args.copy_images
        )
        total_frames += n_f
        total_labels += n_l

    print(f"Done: {total_frames} frames, {total_labels} labels → {out_root}")


if __name__ == "__main__":
    main()
