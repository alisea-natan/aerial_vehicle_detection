"""Load Ultralytics detectors (YOLO / RT-DETR) and architecture defaults."""

from __future__ import annotations

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
# P2 head at imgsz=1024: DFL decode needs >65536 output channels; MPS refuses.
# Train stays on MPS; skip in-train val and run predict/eval on CPU.
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


def is_p2_architecture(model: str) -> bool:
    """True for P2 yaml names and Group E run paths (``head_p2``, ``*-p2.yaml``)."""
    return "-p2" in Path(str(model)).as_posix().lower().replace("_", "-")


def mps_available() -> bool:
    import torch

    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def skip_in_train_val(model: str, device: str) -> bool:
    if device != "mps":
        return False
    return model_family(model) in MPS_FRAGILE_FAMILIES or is_p2_architecture(model)


def predict_device(model_or_weights: str, train_device: str | None = None) -> str:
    """Device for predict/eval. Fragile families leave MPS to avoid a Metal abort."""
    dev = train_device
    if dev is None:
        dev = "mps" if mps_available() else "cpu"
    if skip_in_train_val(model_or_weights, dev):
        return "cpu"
    return dev


def default_freeze(model: str) -> int:
    """Stage-1 backbone freeze count. 0 = do not freeze (RT-DETR)."""
    family = model_family(model)
    if family == "rtdetr":
        return 0
    if family == "yolov8":
        return 10
    return 11  # YOLO11 / YOLO26 backbone 0–10


def default_lr0(model: str, base_lr0: float) -> float:
    """RT-DETR uses AdamW-scale LR; YOLO keeps the round default (0.01)."""
    if model_family(model) == "rtdetr":
        return 0.0001
    return float(base_lr0)


def yaml_architecture(model: str) -> str:
    """Scratch-init yaml for a .pt checkpoint (same compound scale)."""
    name = Path(model).name
    if name.endswith(".pt"):
        return name[:-3] + ".yaml"
    return name


def p2_architecture(model: str) -> str:
    """P2 small-object head yaml (Ultralytics does not publish official *-p2.pt).

    ``YOLO('yolov8s-p2.yaml')`` builds P2/4–P5/32; ``train(pretrained=True)``
    copies matching COCO layers from the s-scale checkpoint.
    """
    family = model_family(model)
    if family == "yolo26":
        return "yolo26s-p2.yaml"
    if family == "yolov8":
        return "yolov8s-p2.yaml"
    if family == "yolo11":
        return str(Path("config") / "models" / "yolo11s-p2.yaml")
    raise SystemExit(f"No P2 architecture for {model!r} (family={family})")
