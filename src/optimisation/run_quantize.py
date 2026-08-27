#!/usr/bin/env python3
"""Step A2 — quantisation PTQ (Track A). QAT = curriculum manual step.

See OPTIMISATION.md — FP16 + INT8 PTQ; Δ vs ov_fp32 in summary.

Compare each cell to FP32 baseline — what changed in quality, speed, size.
Needs step 2 export to have succeeded. INT8 calib = train pack, not eval.

  python src/optimisation/run_quantize.py
  python src/optimisation/run_quantize.py --platform raspberry
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
    PLATFORM_A2_CELLS,
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
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--platform",
        choices=PLATFORMS,
        default=None,
        help="Quantise one deploy target only (jetson: no A2 cells — use A1 ONNX).",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Default: outputs/.../quantize/<prototype|pruned_…> from --weights",
    )
    return p.parse_args()


def main() -> None:
    step_t0 = time.perf_counter()
    args = parse_args()
    cfg = load_opt_cfg()
    d = {**defaults(cfg), "imgsz": args.imgsz}
    locked = dict(cfg.get("locked") or {})
    weights = args.weights if args.weights.is_absolute() else project_path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Missing weights: {weights}")
    calib = project_path(d["train_pack"]) / "data.yaml"
    if not calib.is_file():
        raise SystemExit(
            f"Missing train pack yaml {calib} (INT8 calib). "
            "python src/training/datasets/generate_variant.py --variant strided_clip_balanced"
        )
    artifacts = optimisation_artifacts_dir("quantize", weights, cfg)
    runs_dir = (
        args.runs_dir
        if args.runs_dir is not None
        else optimisation_runs_dir("quantize", weights, cfg)
    )
    runs_dir = runs_dir if runs_dir.is_absolute() else project_path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Variant: {weights_variant(weights, cfg)} → runs {runs_dir} · artifacts {artifacts}")
    teacher = locked_weights_path(cfg)

    cells = [
        ("ov_fp32", "openvino", {"half": False, "int8": False}),
        ("ov_fp16", "openvino", {"half": True, "int8": False}),
        ("ov_int8", "openvino", {"half": False, "int8": True}),
        ("tflite_int8", "tflite", {"half": False, "int8": True}),
    ]
    if args.platform:
        allowed = set(PLATFORM_A2_CELLS[args.platform])
        if not allowed:
            raise SystemExit(
                f"Track A2 has no quant cells for {args.platform!r}. "
                "Use run_export.py --platform jetson for ONNX."
            )
        cells = [c for c in cells if c[0] in allowed]
    rows = []
    for cell_id, fmt, kw in cells:
        exported = export_artifact(
            weights,
            fmt=fmt,
            imgsz=int(d["imgsz"]),
            dest_dir=artifacts / cell_id,
            half=bool(kw["half"]),
            int8=bool(kw["int8"]),
            calib_yaml=calib if kw["int8"] else None,
        )
        (runs_dir / f"{cell_id}_export.json").write_text(
            json.dumps(exported, indent=2) + "\n", encoding="utf-8"
        )
        if not exported.get("export_ok"):
            rows.append({"id": cell_id, **exported})
            print(f"EXPORT FAIL {cell_id}: {exported.get('error')}")
            continue
        rows.append(
            score_cell(
                cell_id=cell_id,
                weights=Path(str(exported["path"])),
                teacher=teacher,
                d=d,
                locked=locked,
                runs_dir=runs_dir,
                skip_eval=args.skip_eval,
                extra=exported,
            )
        )

    write_summary(
        runs_dir,
        "Optimisation Track A2 — quantise PTQ",
        rows,
        step_timelog=make_step_timelog(step_t0),
        baseline_id="ov_fp32",
    )
    passed = [r for r in rows if (r.get("gates") or {}).get("pass")]
    if passed:
        order = {"ov_int8": 0, "tflite_int8": 1, "ov_fp16": 2, "ov_fp32": 3}
        winner = min(passed, key=lambda r: order.get(str(r.get("id")), 9))
        print(f"Ship candidate (smallest quantised that passes gates): {winner.get('id')}")
        print(f"  {winner.get('weights')}")
    else:
        print("No cell passed gates. Keep FP32 export or revisit PoC.")


if __name__ == "__main__":
    main()
