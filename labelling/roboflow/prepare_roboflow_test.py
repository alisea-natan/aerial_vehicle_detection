#!/usr/bin/env python3
"""Build labelling/roboflow/roboflow_upload/ test split (YOLOv8) from label_man + frames.

Videos / steps:
  266987              → every 9th frame
  13722965_…          → every 18th frame  (video_id prefix: 13722965)

  python labelling/roboflow/prepare_roboflow_test.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
LABEL_MAN_DIR = PROJECT_ROOT / "labelling" / "cvat" / "label_man"
OUT_DIR = HERE / "roboflow_upload"
PREPARED_DIR = HERE / "roboflow"
MANUAL_SUBDIRS = ("obj_Train_data", "obj_Test_data")

# (frames folder name, short video_id for renamed files, step)
CLIPS = (
    ("266987", "266987", 9),
    ("13722965_2160_3840_30fps", "13722965", 18),
)


def find_manual_label_dir(clip_dir: Path) -> Path:
    for name in MANUAL_SUBDIRS:
        candidate = clip_dir / name
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    raise SystemExit(f"No label_man labels under {clip_dir}")


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

    # classes from either clip's prepared data.yaml (identical)
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
        man_dir = find_manual_label_dir(LABEL_MAN_DIR / folder)
        jpgs = sorted(frame_dir.glob("*.jpg"))
        if not jpgs:
            raise SystemExit(f"No frames in {frame_dir}")

        sampled = jpgs[::step]
        n_boxes = 0
        n_null = 0

        for jpg in sampled:
            src_txt = man_dir / f"{jpg.stem}.txt"
            if not src_txt.is_file():
                raise SystemExit(f"Missing label_man file for every frame: {src_txt}")

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
        print(
            f"{video_id}: copied {len(sampled)} "
            f"(step={step}, source_frames={len(jpgs)}) "
            f"— with boxes: {n_boxes}, null: {n_null}"
        )

    print(f"test/ total: {total} (with boxes: {total_boxes}, null: {total_null})")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
