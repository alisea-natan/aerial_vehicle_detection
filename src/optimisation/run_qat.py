#!/usr/bin/env python3
"""Track A3 — QAT recovery when A2 INT8 PTQ fails gates.

Ultralytics has no native TFLite/OpenVINO QAT export. This script:
  1. Short fine-tune on the train pack (quantisation-aware recovery).
  2. INT8 re-export with calib on train split (more images than val).

Skip when A2 PTQ already passes for that format (OpenVINO on this model).

  python src/optimisation/run_qat.py                    # TFLite (Android)
  python src/optimisation/run_qat.py --platform android
  python src/optimisation/run_qat.py --platform raspberry --force  # curriculum only
  python src/optimisation/run_qat.py --skip-finetune --weights checkpoints/yolo11s_prototype_qat.pt
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
import shutil
import time
from pathlib import Path
from typing import Any

from optimisation.common import (
    PLATFORM_A3_FORMAT,
    defaults,
    load_opt_cfg,
    locked_weights_path,
    optimisation_artifacts_dir,
    optimisation_runs_dir,
    project_path,
    weights_variant,
)
from optimisation.pipeline import export_artifact, make_step_timelog, score_cell, write_summary
from training.experiments.common import resolve_dataset_yaml, run_train_job
from training.train import default_batch_size

QAT_CHECKPOINT = "yolo11s_prototype_qat.pt"

FORMAT_CELLS = {
    "tflite": ("tflite_int8_qat", "tflite"),
    "openvino": ("ov_int8_qat", "openvino"),
}


def parse_args() -> argparse.Namespace:
    d = defaults()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--platform",
        choices=tuple(PLATFORM_A3_FORMAT),
        default=None,
        help="Deploy target (android or raspberry). Overrides --format.",
    )
    p.add_argument(
        "--format",
        choices=("tflite", "openvino", "all"),
        default="tflite",
        help="Target INT8 format (ignored when --platform is set).",
    )
    p.add_argument("--weights", type=Path, default=project_path(d["weights"]))
    p.add_argument("--imgsz", type=int, default=int(d["imgsz"]))
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="QAT recovery fine-tune epochs (default: qat_epochs in yaml).",
    )
    p.add_argument(
        "--skip-finetune",
        action="store_true",
        help=f"Export only — expects recovered weights at checkpoints/{QAT_CHECKPOINT}.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Run OpenVINO QAT even though A2 ov_int8 PTQ passes (curriculum).",
    )
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Default: outputs/.../quantize_qat/<prototype|…>",
    )
    return p.parse_args()


def publish_checkpoint(src: Path, filename: str) -> Path:
    if not src.is_file():
        raise SystemExit(f"Missing weights to publish: {src}")
    dest = project_path("checkpoints") / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Checkpoint → {dest}")
    return dest


def qat_finetune(
    weights: Path,
    *,
    d: dict[str, Any],
    runs_dir: Path,
    epochs: int,
) -> Path:
    pack_id = Path(str(d["train_pack"])).name
    yaml_path = resolve_dataset_yaml(project_path(d["datasets_root"]), pack_id)
    lr0 = float(d.get("qat_lr0", d["lr0"]))
    print(f"\n=== QAT recovery fine-tune {epochs} ep ← {weights} (lr0={lr0})")
    t0 = time.perf_counter()
    recover = run_train_job(
        yaml_path=yaml_path,
        model=str(weights),
        imgsz=int(d["imgsz"]),
        epochs=epochs,
        warmup_epochs=0,
        freeze=0,
        lr0=lr0,
        patience=int(d["patience"]),
        train_augmentation=str(d["train_augmentation"]),
        runs_dir=runs_dir,
        run_name="qat_finetune",
        deliverable_name=QAT_CHECKPOINT,
        batch=default_batch_size(int(d["imgsz"])),
        staged=False,
        train_backbone=True,
        pretrained=True,
        write_checkpoint=True,
    )
    finetune_sec = round(time.perf_counter() - t0, 2)
    recovered = publish_checkpoint(Path(str(recover["best_weights"])), QAT_CHECKPOINT)
    (runs_dir / "qat_finetune.json").write_text(
        json.dumps({"finetune_sec": finetune_sec, "weights": str(recovered)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return recovered


def run_one_format(
    fmt: str,
    *,
    weights: Path,
    args: argparse.Namespace,
    d: dict[str, Any],
    locked: dict[str, Any],
    teacher: Path,
    artifacts: Path,
    runs_dir: Path,
    calib: Path,
    calib_split: str,
    finetune_sec: float | None,
) -> dict[str, Any]:
    cell_id, export_fmt = FORMAT_CELLS[fmt]
    exported = export_artifact(
        weights,
        fmt=export_fmt,
        imgsz=int(d["imgsz"]),
        dest_dir=artifacts / cell_id,
        half=False,
        int8=True,
        calib_yaml=calib,
        calib_split=calib_split,
    )
    (runs_dir / f"{cell_id}_export.json").write_text(
        json.dumps(exported, indent=2) + "\n", encoding="utf-8"
    )
    if not exported.get("export_ok"):
        return {"id": cell_id, **exported}
    extra: dict[str, Any] = dict(exported)
    if finetune_sec is not None:
        extra["timelog"] = {"finetune_sec": finetune_sec}
    return score_cell(
        cell_id=cell_id,
        weights=Path(str(exported["path"])),
        teacher=teacher,
        d=d,
        locked=locked,
        runs_dir=runs_dir,
        skip_eval=args.skip_eval,
        extra=extra,
    )


def main() -> None:
    step_t0 = time.perf_counter()
    args = parse_args()
    cfg = load_opt_cfg()
    d = {**defaults(cfg), "imgsz": args.imgsz}
    locked = dict(cfg.get("locked") or {})
    teacher = locked_weights_path(cfg)
    base_weights = args.weights if args.weights.is_absolute() else project_path(args.weights)
    if not base_weights.is_file():
        raise SystemExit(f"Missing weights: {base_weights}")

    formats = (
        [PLATFORM_A3_FORMAT[args.platform]]
        if args.platform
        else (["tflite", "openvino"] if args.format == "all" else [args.format])
    )
    if "openvino" in formats and not args.force:
        print("OpenVINO: A2 ov_int8 PTQ passes gates — skip A3 (use --force for curriculum).")
        formats = [f for f in formats if f != "openvino"]
    if not formats:
        raise SystemExit("Nothing to run.")

    calib = project_path(d["train_pack"]) / "data.yaml"
    if not calib.is_file():
        raise SystemExit(
            f"Missing train pack yaml {calib}. "
            "python src/training/datasets/generate_variant.py --variant strided_clip_balanced"
        )
    calib_split = str(d.get("qat_calib_split") or "train")
    epochs = int(args.epochs if args.epochs is not None else d.get("qat_epochs", 3))

    variant = weights_variant(base_weights, cfg)
    artifacts = optimisation_artifacts_dir("quantize_qat", base_weights, cfg)
    runs_dir = (
        args.runs_dir
        if args.runs_dir is not None
        else optimisation_runs_dir("quantize_qat", base_weights, cfg)
    )
    runs_dir = runs_dir if runs_dir.is_absolute() else project_path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Variant: {variant} → runs {runs_dir} · artifacts {artifacts}")
    print(f"INT8 calib: {calib} split={calib_split}")

    finetune_sec: float | None = None
    if args.skip_finetune:
        weights = project_path("checkpoints") / QAT_CHECKPOINT
        if not weights.is_file():
            raise SystemExit(f"--skip-finetune requires checkpoints/{QAT_CHECKPOINT}")
    else:
        weights = qat_finetune(
            base_weights,
            d=d,
            runs_dir=runs_dir,
            epochs=epochs,
        )
        payload = json.loads((runs_dir / "qat_finetune.json").read_text(encoding="utf-8"))
        finetune_sec = payload.get("finetune_sec")

    rows = []
    for fmt in formats:
        rows.append(
            run_one_format(
                fmt,
                weights=weights,
                args=args,
                d=d,
                locked=locked,
                teacher=teacher,
                artifacts=artifacts,
                runs_dir=runs_dir,
                calib=calib,
                calib_split=calib_split,
                finetune_sec=finetune_sec,
            )
        )

    write_summary(
        runs_dir,
        "Optimisation Track A3 — quantise QAT",
        rows,
        step_timelog=make_step_timelog(step_t0),
        baseline_id=None,
    )
    passed = [r for r in rows if (r.get("gates") or {}).get("pass")]
    if passed:
        for row in passed:
            print(f"Pass: {row.get('id')} → {row.get('weights')}")
        print("Next: Track A4 device bench with passing artifact.")
    else:
        print("A3 did not pass gates. Android: bench A1 FP32 TFLite on device, or tune qat_epochs / qat_lr0.")


if __name__ == "__main__":
    main()
