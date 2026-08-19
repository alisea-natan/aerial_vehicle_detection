#!/usr/bin/env python3
"""Deploy local detector weights to Roboflow (versionless model).

  # ROBOFLOW_API_KEY in .env (see .env.example)
  python src/labeling/roboflow/deploy_roboflow.py
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

from common.config import POC_CHECKPOINT, PROTOTYPE_CHECKPOINT, PROJECT_ROOT

import os

import roboflow


PROJECT = "vehicle-cige6"
MODEL_NAME = "yolo11s-vehicle"
MODEL_DIR = PROJECT_ROOT / "outputs" / "runs" / "yolo11s_vehicle"
WEIGHTS_REL = "weights/best.pt"
CKPT = PROTOTYPE_CHECKPOINT
POC_CKPT = POC_CHECKPOINT


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY in .env (see .env.example)")

    weights = MODEL_DIR / WEIGHTS_REL
    if CKPT.is_file():
        model_dir = CKPT.parent
        filename = CKPT.name
    elif weights.is_file():
        model_dir = MODEL_DIR
        filename = WEIGHTS_REL
    elif POC_CKPT.is_file():
        model_dir = POC_CKPT.parent
        filename = POC_CKPT.name
    else:
        raise SystemExit(f"Missing weights: {CKPT}, {weights}, or {POC_CKPT}")

    print(f"roboflow {roboflow.__version__}")
    print(f"versionless deploy → project={PROJECT} name={MODEL_NAME}")
    print(f"path={model_dir / filename}")

    rf = roboflow.Roboflow(api_key=api_key)
    rf.workspace().deploy_model(
        model_type="yolov8n",
        model_path=str(model_dir),
        project_ids=[PROJECT],
        model_name=MODEL_NAME,
        filename=filename,
    )
    print("Deploy submitted.")


if __name__ == "__main__":
    main()
