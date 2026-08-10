#!/usr/bin/env python3
"""Attach missing YOLO labels to images already in Roboflow.

Skips empty .txt (API rejects them as Unrecognized annotation format).
Uses image_id + overwrite — does not re-upload images.

  python labelling/roboflow/retry_roboflow_annotations.py
"""

from __future__ import annotations

import time
from pathlib import Path

import roboflow

HERE = Path(__file__).resolve().parent
DATASET = HERE / "roboflow_upload"
PROJECT = "vehicle-cige6"
api_key = "6fDavmzCe6KEWpM0GZxg"
LABELMAP = {0: "vehicle"}


def iter_project_images(project):
    offset = 0
    limit = 100
    while True:
        page = project.search(
            offset=offset,
            limit=limit,
            fields=["id", "name", "annotations", "labels", "split"],
        )
        if not page:
            break
        yield from page
        if len(page) < limit:
            break
        offset += limit


def main() -> None:
    labels_dir = DATASET / "test" / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"Missing {labels_dir}")

    local = {}
    for txt in sorted(labels_dir.glob("*.txt")):
        if not txt.read_text(encoding="utf-8").strip():
            continue  # null — skip; API cannot accept empty YOLO
        local[txt.stem + ".jpg"] = txt

    print(f"local non-empty labels: {len(local)}")

    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace().project(PROJECT)

    remote = {}
    for img in iter_project_images(project):
        name = img.get("name")
        if name:
            remote[name] = img
    print(f"images in project: {len(remote)}")

    ok = fail = skip_missing = 0
    for name, txt in local.items():
        img = remote.get(name)
        if img is None:
            skip_missing += 1
            print(f"[MISS] {name} not in project")
            continue

        image_id = img["id"]
        try:
            project.save_annotation(
                annotation_path=str(txt),
                annotation_labelmap=LABELMAP,
                image_id=image_id,
                annotation_overwrite=True,
                num_retry_uploads=2,
            )
            ok += 1
            print(f"[OK] {name} -> {image_id}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {name} ({image_id}): {e}")
        time.sleep(0.15)  # ease rate limits / SDK races

    print(
        f"Done: ok={ok} fail={fail} "
        f"not_in_project={skip_missing} "
        f"empty_txt_skipped={sum(1 for p in labels_dir.glob('*.txt') if not p.read_text().strip())}"
    )


if __name__ == "__main__":
    main()
