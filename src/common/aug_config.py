#!/usr/bin/env python3
"""Ultralytics online augmentation presets for aerial vehicle training.

Used when training on subsampled frames (``frame_step``): color jitter + flips +
rotations compensate for fewer unique viewpoints. Mosaic stays off here and is
enabled only in the local experiment runner under ``experiments/``.
"""

from __future__ import annotations

# HSV jitter (Ultralytics fraction of full range).
HSV_H = 0.015
HSV_S = 0.70
HSV_V = 0.40

# Top-down aerial: any yaw is valid.
DEGREES = 180.0
FLIPLR = 0.5
FLIPUD = 0.5

# Random zoom/pan — simulates altitude / distance variation (see guide §11).
SCALE = 0.30
TRANSLATE = 0.10

# Main train: no mosaic (sparse frame_step tiles glue poorly).
MOSAIC = 0.0
MIXUP = 0.0
COPY_PASTE = 0.0

# Local A/B probe (experiments/train_aug_probe.py) — not committed.
EXPERIMENT_MOSAIC = 1.0
EXPERIMENT_WARMUP_EPOCHS = 1
EXPERIMENT_EPOCHS = 3
EXPERIMENT_PATIENCE = 0  # no early stop — always finish the short run


def train_aug_kwargs(*, mosaic: float | None = None) -> dict[str, float]:
    """Kwargs passed into ``YOLO.train(...)`` for color / geometry aug."""
    return {
        "hsv_h": HSV_H,
        "hsv_s": HSV_S,
        "hsv_v": HSV_V,
        "degrees": DEGREES,
        "fliplr": FLIPLR,
        "flipud": FLIPUD,
        "scale": SCALE,
        "translate": TRANSLATE,
        "mosaic": MOSAIC if mosaic is None else float(mosaic),
        "mixup": MIXUP,
        "copy_paste": COPY_PASTE,
    }


def experiment_aug_kwargs() -> dict[str, float]:
    return train_aug_kwargs(mosaic=EXPERIMENT_MOSAIC)
