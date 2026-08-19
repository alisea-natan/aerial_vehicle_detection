#!/usr/bin/env python3
"""Build labelling/roboflow/roboflow_upload/ test split (YOLOv8) from labels/ + frames.

Videos / steps:
  266987              → every 9th frame
  13722965_…          → every 18th frame  (video_id prefix: 13722965)

  python src/labeling/roboflow/prepare_roboflow_test.py
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

from common.config import LABELS_DIR, PROJECT_ROOT, SPLITS

import shutil
from pathlib import Path


FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
OUT_DIR = PROJECT_ROOT / "labelling" / "roboflow" / "roboflow_upload"
PREPARED_DIR = PROJECT_ROOT / "labelling" / "roboflow" / "roboflow"

# (frames folder name, short video_id for renamed files, step)
CLIPS = (
    ("266987", "266987", 9),
    ("13722965_2160_3840_30fps", "13722965", 18),
)


def find_cvat_label_dir(clip: str) -> Path:
    for split in SPLITS:
        path = LABELS_DIR / split / clip
        if path.is_dir() and any(path.glob("*.txt")):
            return path
    raise SystemExit(f"No CVAT labels under labels/{{train,eval}}/{clip}")


def label_has_boxes(path: Path) -> bool:
    text = path.read_text(encoding="utf-8").strip()
    return bool(text)


def main() -> None:
    images_out = OUT_DIR / "test" / "images"
    labels_out = OUT_DIR / "test" / "labels"
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    src_yaml = PREPARED_DIR / "266987" / "data.yaml"
    if not src_yaml.is_file():
        src_yaml = PREPARED_DIR / CLIPS[1][0] / "data.yaml"
    if not src_yaml.is_file():
        raise SystemExit(
            "Need data.yaml under labelling/roboflow/roboflow/<clip>/ "
            "(run prepare_roboflow.py)"
        )

    names_nc = []
    for line in src_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("nc:") or line.startswith("names:"):
            names_nc.append(line)
    if len(names_nc) < 2:
        raise SystemExit(f"Could not read nc/names from {src_yaml}")

    (OUT_DIR / "data.yaml").write_text(
        "test: test/images\n\n" + "\n".join(names_nc) + "\n",
        encoding="utf-8",
    )

    total = 0
    total_boxes = 0
    total_null = 0

    for folder, video_id, step in CLIPS:
        frame_dir = FRAMES_DIR / folder
        man_dir = find_cvat_label_dir(folder)
        jpgs = sorted(frame_dir.glob("*.jpg"))
        if not jpgs:
            raise SystemExit(f"No frames in {frame_dir}")

        sampled = jpgs[::step]
        n_boxes = 0
        n_null = 0

        for jpg in sampled:
            src_txt = man_dir / f"{jpg.stem}.txt"
            if not src_txt.is_file():
                raise SystemExit(f"Missing CVAT label for frame: {src_txt}")

            stem = f"{video_id}_{jpg.stem}"
            shutil.copy2(jpg.resolve(), images_out / f"{stem}.jpg")
            shutil.copy2(src_txt, labels_out / f"{stem}.txt")

            if label_has_boxes(src_txt):
                n_boxes += 1
            else:
                n_null += 1

        total += len(sampled)
        total_boxes += n_boxes
        total_null += n_null
        print(f"{folder}: {len(sampled)} frames (step={step}), boxes={n_boxes}, empty={n_null}")

    print(f"Done: {total} test images → {OUT_DIR} (with boxes={total_boxes}, empty={total_null})")


if __name__ == "__main__":
    main()
