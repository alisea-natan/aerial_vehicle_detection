#!/usr/bin/env python3
"""Step A1 — export ONNX + OpenVINO + TFLite (Track A, compatibility gate).

See OPTIMISATION.md — production path step 1 after baseline (EVALUATION.md §6).

Unsupported ops / dynamic shapes can fail here. Then stop and go back to PoC.

  python src/optimisation/run_export.py
  python src/optimisation/run_export.py --platform android
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
import json
import time
from pathlib import Path

from optimisation.common import (
    PLATFORMS,
    PLATFORM_A1_FORMATS,
    defaults,
    load_opt_cfg,
    locked_weights_path,
    optimisation_artifacts_dir,
    optimisation_runs_dir,
    project_path,
    weights_variant,
)
from optimisation.pipeline import export_artifact, make_step_timelog, score_cell, write_summary


def parse_args() -> argparse.Namespace:
    d = defaults()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=project_path(d["weights"]))
    p.add_argument("--imgsz", type=int, default=int(d["imgsz"]))
    p.add_argument("--skip-eval", action="store_true", help="Bench + parity only (no full holdout).")
    p.add_argument(
        "--platform",
        choices=PLATFORMS,
        default=None,
        help="Export one deploy target only (raspberry=OpenVINO, android=TFLite, jetson=ONNX).",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Default: outputs/.../export/<prototype|pruned|…> from --weights",
    )
    return p.parse_args()


def main() -> None:
    step_t0 = time.perf_counter()
    args = parse_args()
    cfg = load_opt_cfg()
    d = defaults(cfg)
    locked = dict(cfg.get("locked") or {})
    weights = args.weights if args.weights.is_absolute() else project_path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Missing locked weights: {weights}")
    d = {**d, "imgsz": args.imgsz}
    variant = weights_variant(weights, cfg)
    is_prototype = variant == "prototype"
    artifacts = optimisation_artifacts_dir("export", weights, cfg)
    runs_dir = (
        args.runs_dir
        if args.runs_dir is not None
        else optimisation_runs_dir("export", weights, cfg)
    )
    runs_dir = runs_dir if runs_dir.is_absolute() else project_path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Variant: {variant} → runs {runs_dir} · artifacts {artifacts}")

    rows = []
    rows.append(
        score_cell(
            cell_id="pt",
            weights=weights,
            teacher=locked_weights_path(cfg),
            d=d,
            locked=locked,
            runs_dir=runs_dir,
            skip_eval=is_prototype,
            use_locked_quality=is_prototype,
            extra={"format": "pytorch", "export_ok": True},
        )
    )

    fail = False
    formats = list(PLATFORM_A1_FORMATS[args.platform]) if args.platform else ["onnx", "openvino", "tflite"]
    for fmt in formats:
        exported = export_artifact(
            weights,
            fmt=fmt,
            imgsz=int(d["imgsz"]),
            dest_dir=artifacts / fmt,
        )
        (runs_dir / f"{fmt}_export.json").write_text(
            json.dumps(exported, indent=2) + "\n", encoding="utf-8"
        )
        if not exported.get("export_ok"):
            fail = True
            rows.append({"id": fmt, **exported})
            print(f"EXPORT FAIL {fmt}: {exported.get('error')}")
            print(exported.get("hint"))
            continue
        skip_eval = args.skip_eval
        rows.append(
            score_cell(
                cell_id=fmt,
                weights=Path(str(exported["path"])),
                teacher=locked_weights_path(cfg),
                d=d,
                locked=locked,
                runs_dir=runs_dir,
                skip_eval=skip_eval,
                extra=exported,
            )
        )

    write_summary(
        runs_dir,
        "Optimisation Track A1 — export",
        rows,
        step_timelog=make_step_timelog(step_t0),
    )
    if fail:
        raise SystemExit(
            "Export failed. Do not quantize or deploy pruned weights from a failed export. "
            "Go back to PoC.md / EVALUATION.md."
        )


if __name__ == "__main__":
    main()
