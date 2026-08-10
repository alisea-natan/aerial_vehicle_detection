#!/usr/bin/env python3
"""Build flat roboflow_ui_upload_8457857/ for Roboflow UI upload (train).

  python labelling/roboflow/prepare_roboflow_ui_8457857.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

CLIP_DIR = "8457857-uhd_3840_2160_24fps"
VIDEO_ID = "8457857"
STEP = 5
FRAMES_DIR = PROJECT_ROOT / "data" / "frames" / CLIP_DIR
LABEL_MAN_DIR = (
    PROJECT_ROOT / "labelling" / "cvat" / "label_man" / CLIP_DIR / "obj_Train_data"
)
# Project export — keep class index order as-is
PROJECT_YAML = HERE / "Vehicle_roboflow" / "data.yaml"
OUT_DIR = HERE / "roboflow_ui_upload_8457857"


def classes_from_yaml(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # names: ['Vehicle']  or  names:\n  - Vehicle
    m = re.search(r"names:\s*\[([^\]]*)\]", text)
    if m:
        return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
    names: list[str] = []
    in_names = False
    for line in text.splitlines():
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if re.match(r"\s*-\s*", line):
                names.append(re.sub(r"^\s*-\s*", "", line).strip().strip("'\""))
            elif line.strip() and not line.startswith(" "):
                break
    if not names:
        raise SystemExit(f"Could not parse names from {path}")
    return names


def main() -> None:
    if not FRAMES_DIR.is_dir():
        raise SystemExit(f"Missing frames: {FRAMES_DIR}")
    if not LABEL_MAN_DIR.is_dir():
        raise SystemExit(f"Missing label_man: {LABEL_MAN_DIR}")
    if not PROJECT_YAML.is_file():
        raise SystemExit(f"Missing project yaml: {PROJECT_YAML}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    classes = classes_from_yaml(PROJECT_YAML)
    (OUT_DIR / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    jpgs = sorted(FRAMES_DIR.glob("*.jpg"))
    sampled = jpgs[::STEP]
    n_boxes = 0
    n_null = 0

    for jpg in sampled:
        src_txt = LABEL_MAN_DIR / f"{jpg.stem}.txt"
        if not src_txt.is_file():
            raise SystemExit(f"Missing label_man for frame: {src_txt}")

        stem = f"{VIDEO_ID}_{jpg.stem}"
        shutil.copy2(jpg.resolve(), OUT_DIR / f"{stem}.jpg")
        shutil.copy2(src_txt, OUT_DIR / f"{stem}.txt")

        if src_txt.read_text(encoding="utf-8").strip():
            n_boxes += 1
        else:
            n_null += 1

    print(f"classes.txt: {classes}")
    print(
        f"{VIDEO_ID}: copied {len(sampled)} "
        f"(step={STEP}, source_frames={len(jpgs)}) "
        f"— with boxes: {n_boxes}, null: {n_null}"
    )
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
