#!/usr/bin/env python3
"""Optional aerial-frame enhancement (CLAHE) for YOLO-World experiments.

Ultralytics already normalizes pixels internally. This module does **not** replace
that — it only boosts local contrast (LAB-L channel) before inference.

Default pipeline stays off. Enable with ``--enhance`` on preprocess / autolabel.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Mild defaults — strong CLAHE invents edges / noise on drone footage.
DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID = (8, 8)


def apply_clahe_bgr(
    image: np.ndarray,
    *,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid: tuple[int, int] = DEFAULT_TILE_GRID,
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on the L channel."""
    if image is None or image.size == 0:
        raise ValueError("apply_clahe_bgr: empty image")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid)
    merged = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def load_bgr(
    path: Path | str,
    *,
    enhance: bool = False,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid: tuple[int, int] = DEFAULT_TILE_GRID,
) -> np.ndarray:
    """Read a BGR JPEG/PNG; optionally apply CLAHE."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if enhance:
        return apply_clahe_bgr(image, clip_limit=clip_limit, tile_grid=tile_grid)
    return image


def inference_source(
    path: Path | str,
    *,
    enhance: bool = False,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid: tuple[int, int] = DEFAULT_TILE_GRID,
) -> str | np.ndarray:
    """Path for default inference, or CLAHE ndarray when ``enhance`` is on."""
    if not enhance:
        return str(path)
    return load_bgr(path, enhance=True, clip_limit=clip_limit, tile_grid=tile_grid)
