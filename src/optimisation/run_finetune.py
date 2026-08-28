#!/usr/bin/env python3
"""Track B2 — fine-tune after run_prune.py (optional lab).

See OPTIMISATION.md Track B.

  python src/optimisation/run_finetune.py --method structured
  python src/optimisation/run_finetune.py --methods structured,unstructured
  python src/optimisation/run_finetune.py --weights checkpoints/yolo11s_pruned_structured.pt
"""
from __future__ import annotations

from pathlib import Path as _Path
import sys


def _ensure_src_on_path() -> None:
    p = _Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "common").is_dir() and (p / "labeling").is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        p = p.parent


_ensure_src_on_path()

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

from optimisation.common import defaults, load_opt_cfg, project_path
from optimisation.pipeline import make_step_timelog, score_cell, write_summary
from optimisation.prune import PRUNE_METHODS, pruned_checkpoint_stem, recovered_checkpoint_stem
from training.experiments.common import resolve_dataset_yaml, run_train_job
from training.train import default_batch_size


def publish_checkpoint(src: Path, filename: str) -> Path:
    if not src.is_file():
        raise SystemExit(f"Missing weights to publish: {src}")
    dest = project_path("checkpoints") / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Checkpoint → {dest}")
    return dest


def resolve_pruned_weights(method: str, explicit: Path | None, artifacts: Path) -> Path:
    if explicit is not None:
        w = explicit if explicit.is_absolute() else project_path(explicit)
        if not w.is_file():
            raise SystemExit(f"Missing pruned weights: {w}")
        return w
    stem = pruned_checkpoint_stem(method)
    for candidate in (
        project_path("checkpoints") / f"{stem}.pt",
        artifacts / f"{stem}.pt",
    ):
        if candidate.is_file():
            return candidate
    legacy = PRUNE_METHODS[method].get("legacy_stem")
    if legacy:
        legacy_path = project_path("checkpoints") / f"{legacy}.pt"
        if legacy_path.is_file():
            return legacy_path
    raise SystemExit(
        f"No pruned weights for {method}. Run run_prune.py first "
        f"(expected checkpoints/{stem}.pt)."
    )


def parse_args() -> argparse.Namespace:
    d = defaults()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=None, help="Single pruned .pt (infers one method).")
    p.add_argument("--method", type=str, default=None, help="structured or unstructured")
    p.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods (default: structured,unstructured if no --weights)",
    )
    p.add_argument("--teacher", type=Path, default=project_path(d["weights"]))
    p.add_argument("--imgsz", type=int, default=int(d["imgsz"]))
    p.add_argument("--epochs", type=int, default=None, help="Default: finetune_epochs in yaml.")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=project_path(d["runs_dir"]) / "finetune",
    )
    return p.parse_args()


def finetune_one(
    method: str,
    *,
    pruned: Path,
    args: argparse.Namespace,
    d: dict[str, Any],
    locked: dict[str, Any],
    teacher: Path,
    artifacts: Path,
    runs_dir: Path,
    epochs: int,
) -> dict[str, Any]:
    stem = pruned_checkpoint_stem(method)
    method_runs = runs_dir / method
    method_runs.mkdir(parents=True, exist_ok=True)
    pack_id = Path(str(d["train_pack"])).name
    yaml_path = resolve_dataset_yaml(project_path(d["datasets_root"]), pack_id)
    print(f"\n=== {method} fine-tune {epochs} ep ← {pruned}")
    t0 = time.perf_counter()
    recover = run_train_job(
        yaml_path=yaml_path,
        model=str(pruned),
        imgsz=int(d["imgsz"]),
        epochs=epochs,
        warmup_epochs=0,
        freeze=0,
        lr0=float(d["lr0"]),
        patience=int(d["patience"]),
        train_augmentation=str(d["train_augmentation"]),
        runs_dir=method_runs,
        run_name="finetune",
        deliverable_name=f"{recovered_checkpoint_stem(method)}.pt",
        batch=default_batch_size(int(d["imgsz"])),
        staged=False,
        train_backbone=True,
        pretrained=True,
        write_checkpoint=True,
    )
    finetune_sec = round(time.perf_counter() - t0, 2)
    recovered = publish_checkpoint(
        Path(str(recover["best_weights"])),
        f"{recovered_checkpoint_stem(method)}.pt",
    )
    # legacy alias for structured (Aug-20 run)
    if method == "structured":
        publish_checkpoint(recovered, "yolo11s_pruned_recovered.pt")

    row = score_cell(
        cell_id=method,
        weights=recovered,
        teacher=teacher,
        d=d,
        locked=locked,
        runs_dir=method_runs,
        skip_eval=args.skip_eval,
        extra={
            "prune_method": method,
            "pruned_from": str(pruned),
            "finetune_epochs": epochs,
            "stage": "finetuned",
            "timelog": {"finetune_sec": finetune_sec},
        },
    )
    passed = (row.get("gates") or {}).get("pass")
    if passed:
        print(
            f"  Research export: python src/optimisation/run_export.py --weights {recovered}\n"
            f"  Research quantize: python src/optimisation/run_quantize.py --weights {recovered}"
        )
    return row


def infer_method_from_weights(path: Path) -> str:
    name = path.name.lower()
    for method in PRUNE_METHODS:
        if method in name:
            return method
    if "pruned" in name and "structured" not in name and "unstructured" not in name:
        return "structured"
    raise SystemExit(f"Cannot infer prune method from weights name: {path.name}")


def main() -> None:
    step_t0 = time.perf_counter()
    args = parse_args()
    cfg = load_opt_cfg()
    d = {**defaults(cfg), "imgsz": args.imgsz}
    locked = dict(cfg.get("locked") or {})
    teacher = args.teacher if args.teacher.is_absolute() else project_path(args.teacher)
    epochs = int(args.epochs if args.epochs is not None else d.get("finetune_epochs", d["prune_recover_epochs"]))
    artifacts = project_path(d["artifacts_dir"]) / "prune"
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else project_path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.weights is not None:
        methods = [args.method or infer_method_from_weights(args.weights)]
        weight_map = {methods[0]: args.weights}
    else:
        methods = [
            m.strip()
            for m in (args.methods or args.method or "structured,unstructured").split(",")
            if m.strip()
        ]
        weight_map = {m: None for m in methods}

    unknown = set(methods) - set(PRUNE_METHODS)
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Choose from {list(PRUNE_METHODS)}")

    rows = []
    for method in methods:
        pruned = resolve_pruned_weights(method, weight_map.get(method), artifacts)
        rows.append(
            finetune_one(
                method,
                pruned=pruned,
                args=args,
                d=d,
                locked=locked,
                teacher=teacher,
                artifacts=artifacts,
                runs_dir=runs_dir,
                epochs=epochs,
            )
        )

    write_summary(
        runs_dir,
        "Optimisation Track B2 — fine-tune",
        rows,
        step_timelog=make_step_timelog(step_t0),
    )


if __name__ == "__main__":
    main()
