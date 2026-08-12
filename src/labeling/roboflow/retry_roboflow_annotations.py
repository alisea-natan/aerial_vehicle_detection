#!/usr/bin/env python3
"""Attach missing YOLO labels to images already in Roboflow.

  # ROBOFLOW_API_KEY in .env (see .env.example)
  python src/labeling/roboflow/retry_roboflow_annotations.py
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

import os
import time

import roboflow


DATASET = PROJECT_ROOT / "labelling" / "roboflow" / "roboflow_upload"
PROJECT = "vehicle-cige6"
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
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in .env (see .env.example)")

    labels_dir = DATASET / "test" / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"Missing {labels_dir}")

    local = {}
    for txt in sorted(labels_dir.glob("*.txt")):
        if not txt.read_text(encoding="utf-8").strip():
            continue
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
        time.sleep(0.15)

    print(f"Done: ok={ok} fail={fail} not_in_project={skip_missing}")


if __name__ == "__main__":
    main()
