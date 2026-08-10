#!/usr/bin/env python3
"""Upload labelling/roboflow/roboflow_upload/ (YOLOv8 test split) to Roboflow.

  python labelling/roboflow/upload_roboflow.py
"""

from __future__ import annotations

from pathlib import Path

import roboflow

HERE = Path(__file__).resolve().parent
DATASET = HERE / "roboflow_upload"
PROJECT = "vehicle-cige6"
api_key = "6fDavmzCe6KEWpM0GZxg"


def main() -> None:
    if not (DATASET / "test" / "images").is_dir():
        raise SystemExit(
            f"Missing {DATASET}/test/images — run prepare_roboflow_test.py"
        )
    if not (DATASET / "data.yaml").is_file():
        raise SystemExit(f"Missing {DATASET}/data.yaml")

    print(f"roboflow {roboflow.__version__}")
    print(f"upload {DATASET}")

    rf = roboflow.Roboflow(api_key=api_key)
    rf.workspace().upload_dataset(
        str(DATASET),
        PROJECT,
        dataset_format="yolov8",
        project_type="object-detection",
        num_workers=10,
        num_retries=2,
    )


if __name__ == "__main__":
    main()
