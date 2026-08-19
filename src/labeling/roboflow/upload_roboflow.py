#!/usr/bin/env python3
"""Upload labelling/roboflow/roboflow_upload/ to Roboflow.

  # ROBOFLOW_API_KEY in .env (see .env.example)
  python src/labeling/roboflow/upload_roboflow.py
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

import roboflow


DATASET = PROJECT_ROOT / "labelling" / "roboflow" / "roboflow_upload"
PROJECT = "vehicle-cige6"


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in .env (see .env.example)")

    if not (DATASET / "test" / "images").is_dir():
        raise SystemExit(
            f"Missing {DATASET}/test/images — run prepare_roboflow_test.py"
        )
    if not (DATASET / "data.yaml").is_file():
        raise SystemExit(f"Missing {DATASET}/data.yaml")

    print(f"roboflow {roboflow.__version__}")
    print(f"upload {DATASET}")

    rf = roboflow.Roboflow(api_key=api_key)
    project = rf.workspace().project(PROJECT)
    project.upload(str(DATASET), num_retry_uploads=3)
    print("Done.")


if __name__ == "__main__":
    main()
