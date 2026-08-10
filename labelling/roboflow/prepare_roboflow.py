#!/usr/bin/env python3
"""Build labelling/roboflow/roboflow/ from data/frames + CVAT label_man.

For every clip under data/frames/ (except EXCLUDE):
  labelling/roboflow/roboflow/<clip>/images/*.jpg   → symlink to data/frames/...
  labelling/roboflow/roboflow/<clip>/data.yaml
  labelling/roboflow/roboflow/<clip>/labels/*.txt   → ONLY if label_man has that clip;
                                   copy only existing label_man files (no empty placeholders)

Examples:
  python labelling/roboflow/prepare_roboflow.py --clean
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
LABEL_MAN_DIR = PROJECT_ROOT / "labelling" / "cvat" / "label_man"
OUT_DIR = HERE / "roboflow"
EXCLUDE = {"8968356-hd_1920_1080_30fps"}
MANUAL_SUBDIRS = ("obj_Train_data", "obj_Test_data")
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


def find_manual_label_dir(clip_dir: Path) -> Path | None:
    for name in MANUAL_SUBDIRS:
        candidate = clip_dir / name
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def prepare_clip(
    clip: str,
    frames_root: Path,
    man_root: Path,
    out_root: Path,
    *,
    copy_images: bool,
) -> tuple[int, int]:
    """Returns (n_frames, n_labels_from_man)."""
    frame_dir = frames_root / clip
    jpg_paths = sorted(frame_dir.glob("*.jpg"))
    if not jpg_paths:
        print(f"[skip] {clip}: no .jpg under {frame_dir}")
        return 0, 0

    clip_out = out_root / clip
    images_dir = clip_out / "images"
    labels_dir = clip_out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Drop stale labels/ from older runs
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

    man_dir = find_manual_label_dir(man_root / clip) if (man_root / clip).is_dir() else None
    n_labels = 0
    if man_dir is not None:
        labels_dir.mkdir(parents=True, exist_ok=True)
        frame_stems = {jpg.stem for jpg in jpg_paths}
        for src_txt in sorted(man_dir.glob("*.txt")):
            if src_txt.stem not in frame_stems:
                continue
            shutil.copy2(src_txt, labels_dir / src_txt.name)
            n_labels += 1

    (clip_out / "data.yaml").write_text(
        DATA_YAML_TEMPLATE.format(nc=len(CLASS_NAMES), names=CLASS_NAMES),
        encoding="utf-8",
    )

    if man_dir is None:
        print(f"[ok] {clip}: {len(jpg_paths)} images, no labels/ (no label_man)")
    else:
        print(f"[ok] {clip}: {len(jpg_paths)} images, {n_labels} labels from label_man")
    return len(jpg_paths), n_labels


def main() -> None:
    args = parse_args()
    frames_root = FRAMES_DIR
    man_root = LABEL_MAN_DIR
    out_root = OUT_DIR

    if not frames_root.is_dir():
        raise SystemExit(f"Frames dir not found: {frames_root}")

    clips = sorted(
        p.name
        for p in frames_root.iterdir()
        if p.is_dir() and p.name not in EXCLUDE and not p.name.startswith(".")
    )
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
            clip, frames_root, man_root, out_root, copy_images=args.copy_images
        )
        total_frames += n_f
        total_labels += n_l

    print(f"Done: {total_frames} images, {total_labels} labels (label_man only)")


if __name__ == "__main__":
    main()
