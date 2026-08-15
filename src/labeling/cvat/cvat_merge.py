#!/usr/bin/env python3
"""Optional: build data/dataset/ YOLO layout from CVAT pulls + local frames.

Prefer ``cvat/cvat_pull.py --sync-labels`` for train.py (writes labels/).
Use this when you want a flat Ultralytics dataset under data/dataset/.

  python src/labeling/cvat/cvat_merge.py
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

from common.config import PROJECT_ROOT

import shutil
from pathlib import Path


RAW = PROJECT_ROOT / "data" / "cvat"
FRAMES = PROJECT_ROOT / "data" / "frames"
DATASET = PROJECT_ROOT / "data" / "dataset"


def _write_data_yaml(root: Path) -> None:
    (root / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: vehicle",
                "",
            ]
        ),
        encoding="utf-8",
    )


def merge() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"Missing {RAW} — run: python src/labeling/cvat/cvat_pull.py")

    label_dirs = sorted(RAW.glob("*/*/labels_raw"))
    if not label_dirs:
        raise SystemExit(
            f"No */*/labels_raw under {RAW}. "
            "Run: python src/labeling/cvat/cvat_pull.py --verify"
        )

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split_dir in ("images/train", "images/val", "labels/train", "labels/val"):
        (DATASET / split_dir).mkdir(parents=True, exist_ok=True)

    for labels_raw in label_dirs:
        video_name = labels_raw.parent.name
        split = labels_raw.parent.parent.name
        if split not in ("train", "eval"):
            print(f"[WARN] skip unexpected split path: {labels_raw}")
            continue
        # Ultralytics val folder name
        yolo_split = "train" if split == "train" else "val"
        frames_dir = FRAMES / video_name
        if not frames_dir.is_dir():
            print(f"[WARN] {video_name}: no local frames, labels only")

        n_lbl = n_img = 0
        for label in sorted(labels_raw.glob("*.txt")):
            if label.name.lower() == "classes.txt":
                continue
            dest_lbl = DATASET / "labels" / yolo_split / f"{video_name}__{label.name}"
            shutil.copy2(label, dest_lbl)
            n_lbl += 1
            for ext in (".jpg", ".png"):
                src_img = frames_dir / f"{label.stem}{ext}"
                if src_img.is_file():
                    dest_img = DATASET / "images" / yolo_split / f"{video_name}__{label.stem}{ext}"
                    shutil.copy2(src_img, dest_img)
                    n_img += 1
                    break
        print(f"{split}/{video_name}: {n_lbl} labels, {n_img} images → {yolo_split}")

    _write_data_yaml(DATASET)
    print(f"Merge done → {DATASET}")


if __name__ == "__main__":
    merge()
