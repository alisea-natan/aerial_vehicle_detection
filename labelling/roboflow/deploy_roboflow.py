#!/usr/bin/env python3
"""Upload yolov8n_vehicle_best.pt as a versionless model on Roboflow.

  python labelling/roboflow/deploy_roboflow.py
"""

from __future__ import annotations

from pathlib import Path

import roboflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT = "vehicle-cige6"
MODEL_NAME = "yolov8n-vehicle"
api_key = "6fDavmzCe6KEWpM0GZxg"

MODEL_DIR = PROJECT_ROOT / "outputs" / "runs" / "yolov8n_vehicle"
WEIGHTS_REL = "weights/best.pt"
CKPT = PROJECT_ROOT / "checkpoints" / "yolov8n_vehicle_best.pt"


def main() -> None:
    weights = MODEL_DIR / WEIGHTS_REL
    if weights.is_file():
        model_dir = MODEL_DIR
        filename = WEIGHTS_REL
    elif CKPT.is_file():
        model_dir = CKPT.parent
        filename = CKPT.name
    else:
        raise SystemExit(f"Missing weights: {weights} and {CKPT}")

    print(f"roboflow {roboflow.__version__}")
    print(f"versionless deploy → project={PROJECT} name={MODEL_NAME}")
    print(f"model_type=yolov8n  path={model_dir / filename}")

    rf = roboflow.Roboflow(api_key=api_key)
    rf.workspace().deploy_model(
        model_type="yolov8n",
        model_path=str(model_dir),
        project_ids=[PROJECT],
        model_name=MODEL_NAME,
        filename=filename,
    )
    print("Deploy submitted. Check Models / Deploy in Roboflow (may take a few minutes).")


if __name__ == "__main__":
    main()
