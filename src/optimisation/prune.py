"""Structured and unstructured prune of a YOLO .pt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.model_load import load_ultralytics_model
from training.train import release_torch_memory

PRUNE_METHODS: dict[str, dict[str, Any]] = {
    "structured": {
        "stem": "yolo11s_pruned_structured",
        "legacy_stem": "yolo11s_pruned",
        "note": "Removes whole channels — smaller dense ops on Pi / mobile.",
    },
    "unstructured": {
        "stem": "yolo11s_pruned_unstructured",
        "legacy_stem": None,
        "note": "Zeros individual weights — same shapes; speed needs sparse HW.",
    },
}


def pruned_checkpoint_stem(method: str) -> str:
    return str(PRUNE_METHODS[method]["stem"])


def recovered_checkpoint_stem(method: str) -> str:
    return f"{pruned_checkpoint_stem(method)}_recovered"


def unstructured_prune_pt(
    weights: Path,
    dest: Path,
    *,
    ratio: float,
) -> Path:
    """Zero smallest-magnitude weights (same layer shapes — sparse speed needs HW support)."""
    import torch.nn as nn
    import torch.nn.utils.prune as prune

    model = load_ultralytics_model(weights)
    net = model.model
    to_prune: list[tuple[nn.Module, str]] = []
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            to_prune.append((module, "weight"))
    if not to_prune:
        raise SystemExit("No Conv2d layers to prune")
    prune.global_unstructured(
        to_prune,
        pruning_method=prune.L1Unstructured,
        amount=float(ratio),
    )
    for module, param_name in to_prune:
        prune.remove(module, param_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dest))
    del model
    release_torch_memory()
    return dest


def structured_prune_pt(
    weights: Path,
    dest: Path,
    *,
    imgsz: int,
    ratio: float,
) -> Path:
    """Remove whole channels (smaller dense model — helps on Pi / mobile runtimes)."""
    try:
        import torch_pruning as tp
    except ImportError as exc:
        raise SystemExit("pip install torch-pruning") from exc
    from ultralytics.nn.modules.head import Detect

    model = load_ultralytics_model(weights)
    net = model.model
    net.eval()
    example = torch.randn(1, 3, imgsz, imgsz)
    ignored = [m for m in net.modules() if isinstance(m, Detect)]
    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = tp.pruner.MetaPruner(
        net,
        example,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=float(ratio),
        ignored_layers=ignored,
        round_to=8,
    )
    pruner.step()
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dest))
    del model
    release_torch_memory()
    return dest
