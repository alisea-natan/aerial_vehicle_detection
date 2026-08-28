#!/usr/bin/env python3
"""Track B1 — structured + unstructured prune (optional lab).

See OPTIMISATION.md Track B. Fine-tune = run_finetune.py next.

  python src/optimisation/run_prune.py
  python src/optimisation/run_prune.py --methods structured
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
from typing import Any, Callable

from optimisation.common import defaults, load_opt_cfg, project_path
from optimisation.pipeline import make_step_timelog, score_cell, write_summary
from optimisation.prune import (
    PRUNE_METHODS,
    pruned_checkpoint_stem,
    structured_prune_pt,
    unstructured_prune_pt,
)


def publish_checkpoint(src: Path, filename: str) -> Path:
    if not src.is_file():
        raise SystemExit(f"Missing weights to publish: {src}")
    dest = project_path("checkpoints") / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Checkpoint → {dest}")
    return dest


def parse_args() -> argparse.Namespace:
    d = defaults()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=project_path(d["weights"]))
    p.add_argument("--imgsz", type=int, default=int(d["imgsz"]))
    p.add_argument("--ratio", type=float, default=float(d["prune_ratio"]))
    p.add_argument(
        "--methods",
        default="structured,unstructured",
        help="Comma-separated: structured, unstructured",
    )
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=project_path(d["runs_dir"]) / "prune",
    )
    return p.parse_args()


def run_method(
    method: str,
    *,
    weights: Path,
    args: argparse.Namespace,
    d: dict[str, Any],
    locked: dict[str, Any],
    artifacts: Path,
    runs_dir: Path,
) -> dict[str, Any]:
    stem = pruned_checkpoint_stem(method)
    method_runs = runs_dir / method
    method_runs.mkdir(parents=True, exist_ok=True)
    pruned = artifacts / f"{stem}.pt"
    print(f"\n=== {method} prune ratio={args.ratio} → {pruned}")
    t0 = time.perf_counter()
    if method == "structured":
        structured_prune_pt(weights, pruned, imgsz=int(d["imgsz"]), ratio=args.ratio)
    else:
        unstructured_prune_pt(weights, pruned, ratio=args.ratio)
    prune_sec = round(time.perf_counter() - t0, 2)

    published = publish_checkpoint(pruned, f"{stem}.pt")
    row = score_cell(
        cell_id=method,
        weights=published,
        teacher=weights,
        d=d,
        locked=locked,
        runs_dir=method_runs,
        skip_eval=args.skip_eval,
        extra={
            "prune_method": method,
            "prune_ratio": args.ratio,
            "stage": "pruned",
            "timelog": {"prune_sec": prune_sec},
        },
    )
    print(f"Next: python src/optimisation/run_finetune.py --method {method}")
    return row


def main() -> None:
    step_t0 = time.perf_counter()
    args = parse_args()
    cfg = load_opt_cfg()
    d = {**defaults(cfg), "imgsz": args.imgsz}
    locked = dict(cfg.get("locked") or {})
    weights = args.weights if args.weights.is_absolute() else project_path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Missing locked weights: {weights}")
    artifacts = project_path(d["artifacts_dir"]) / "prune"
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else project_path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - set(PRUNE_METHODS)
    if unknown:
        raise SystemExit(f"Unknown --methods: {unknown}. Choose from {list(PRUNE_METHODS)}")

    rows = [
        run_method(
            method,
            weights=weights,
            args=args,
            d=d,
            locked=locked,
            artifacts=artifacts,
            runs_dir=runs_dir,
        )
        for method in methods
    ]
    write_summary(
        runs_dir,
        "Optimisation Track B1 — prune",
        rows,
        step_timelog=make_step_timelog(step_t0),
    )


if __name__ == "__main__":
    main()
