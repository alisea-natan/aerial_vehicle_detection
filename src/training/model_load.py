"""Load Ultralytics detectors (YOLO / RT-DETR) and architecture defaults."""

from __future__ import annotations

import re
from pathlib import Path


def load_ultralytics_model(weights: str | Path):
    """YOLO() for YOLO*; RTDETR() when the stem looks like RT-DETR."""
    name = Path(str(weights)).name.lower()
    if "rtdetr" in name:
        from ultralytics import RTDETR

        return RTDETR(str(weights))
    from ultralytics import YOLO

    return YOLO(str(weights))


# YOLO26 / RT-DETR: GatherND abort on MPS val+predict.
# Train stays on MPS; skip in-train val and run predict/eval on CPU for those families.
MPS_FRAGILE_FAMILIES = frozenset({"yolo26", "rtdetr"})


def model_family(model: str) -> str:
    blob = str(model).lower()
    name = Path(str(model)).name.lower()
    if "rtdetr" in blob:
        return "rtdetr"
    if "yolo26" in blob:
        return "yolo26"
    if name.startswith("yolov8") or name.startswith("yolo8") or "/yolov8" in blob:
        return "yolov8"
    if "yolo11" in blob:
        return "yolo11"
    return "yolo"


def mps_available() -> bool:
    import torch

    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def skip_in_train_val(model: str, device: str) -> bool:
    if device != "mps":
        return False
    return model_family(model) in MPS_FRAGILE_FAMILIES


def predict_device(model_or_weights: str, train_device: str | None = None) -> str:
    """Device for predict/eval. Fragile families leave MPS to avoid a Metal abort."""
    dev = train_device
    if dev is None:
        dev = "mps" if mps_available() else "cpu"
    if skip_in_train_val(model_or_weights, dev):
        return "cpu"
    return dev


_SIZE_PT = re.compile(r"(yolo(?:v8|11|26))([nslmx])(\.pt)$", re.I)


def model_size_letter(model: str) -> str | None:
    """n/s/m/l/x from an Ultralytics YOLO checkpoint name, else None."""
    match = _SIZE_PT.search(Path(str(model)).name)
    return match.group(2).lower() if match else None


def with_model_size(model: str, letter: str) -> str:
    """Swap compound scale on a YOLO .pt name (``yolo11s.pt`` → ``yolo11m.pt``)."""
    letter = str(letter).lower().strip()
    if letter not in {"n", "s", "m", "l", "x"}:
        raise SystemExit(f"Unknown YOLO size letter {letter!r}")
    name = Path(str(model)).name
    match = _SIZE_PT.search(name)
    if not match:
        raise SystemExit(f"Cannot set size {letter!r} on {model!r}")
    return f"{match.group(1)}{letter}{match.group(3)}"


def default_freeze(model: str) -> int:
    """Stage-1 backbone freeze count. 0 = do not freeze (RT-DETR)."""
    family = model_family(model)
    if family == "rtdetr":
        return 0
    if family == "yolov8":
        return 10
    return 11  # YOLO11 / YOLO26 backbone 0–10
