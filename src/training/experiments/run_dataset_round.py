#!/usr/bin/env python3
"""Round 1 — dataset ablation (CVAT labels, PoC YOLO11s schedule).

No flags: full round from scratch (all packs).
``--resume``: skip finished, continue incomplete.

Eval is always the same two packs (eval clips only, CVAT):
  data/datasets/eval_manual
  data/datasets/eval_manual_adapted

  python src/training/experiments/run_dataset_round.py
  python src/training/experiments/run_dataset_round.py --resume
  python src/training/experiments/run_dataset_round.py --variant baseline_1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path as _Path


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
import time
import traceback
from pathlib import Path

from common.config import PROJECT_ROOT
from training.datasets.specs import resolve_labels_root, resolve_variant
from training.experiments.common import (
    abs_runs_dir,
    attach_variant_timing,
    ensure_dataset_pack,
    is_round_worker,
    load_result_json,
    load_yaml,
    parse_eval_targets,
    release_torch_memory,
    run_eval_targets,
    result_json_path,
    run_train_job,
    supervise_round_variants,
    timing_stamp,
    utcnow,
    variant_log_path,
    variant_progress,
    write_round_summary,
    write_round_timing,
)

ROUND_CFG = PROJECT_ROOT / "config" / "experiments" / "dataset_round.yaml"
SUMMARY_TITLE = "Dataset round — YOLO11s --prototype (CVAT labels)"


def parse_args() -> argparse.Namespace:
    cfg = load_yaml(ROUND_CFG)
    d = cfg.get("defaults") or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--variant", action="append", default=None, help="Dataset pack id.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every id (default when no --variant is given).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip finished variants; continue incomplete train/eval only.",
    )
    parser.add_argument("--model", default=str(d.get("model") or "yolo11s.pt"))
    parser.add_argument("--imgsz", type=int, default=int(d.get("imgsz") or 1024))
    parser.add_argument("--epochs", type=int, default=int(d.get("epochs") or 5))
    parser.add_argument("--warmup-epochs", type=int, default=int(d.get("warmup_epochs") or 2))
    parser.add_argument("--freeze", type=int, default=int(d.get("freeze") or 11))
    parser.add_argument("--lr0", type=float, default=float(d.get("lr0") or 0.01))
    parser.add_argument("--patience", type=int, default=int(d.get("patience") or 3))
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(str(d.get("runs_dir") or "outputs/experiments/dataset_round")),
    )
    parser.add_argument(
        "--rebuild-packs",
        action="store_true",
        help="Rebuild train packs even if data.yaml already exists.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Train only; do not score eval packs.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=None,
        help="Override label tree when building packs (default: labels/).",
    )
    return parser.parse_args()


def _save_result(runs_dir: Path, vid: str, payload: dict) -> Path:
    path = abs_runs_dir(runs_dir) / f"ds_{vid}_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    cfg = load_yaml(ROUND_CFG)
    defaults = cfg.get("defaults") or {}
    runs = list(cfg.get("runs") or [])
    datasets_root = Path(str(defaults.get("datasets_root") or "data/datasets"))
    eval_targets = parse_eval_targets(defaults)

    labels_root = None
    if args.labels_root is not None:
        labels_root = resolve_labels_root(args.labels_root)
    elif defaults.get("labels_root"):
        labels_root = resolve_labels_root(defaults.get("labels_root"))

    if args.list:
        print(
            f"Protocol: {args.model} imgsz={args.imgsz} "
            f"warmup={args.warmup_epochs} epochs={args.epochs} freeze={args.freeze} "
            f"patience={args.patience}"
        )
        print("Eval packs:")
        for gt_name, pack in eval_targets:
            print(f"  {gt_name}: {pack}")
        for vid in runs:
            spec = resolve_variant(vid, labels_root=labels_root)
            status, _ = variant_progress(
                runs_dir=args.runs_dir,
                variant_id=vid,
                run_name=f"ds_{vid}",
                eval_targets=eval_targets,
                skip_eval=args.skip_eval,
            )
            print(
                f"{vid}: {spec.description} (train_aug={spec.train_augmentation}) [{status}]"
            )
        return

    selected = list(args.variant or [])
    if args.all or not selected:
        selected = runs
    if not selected:
        raise SystemExit("No runs listed in dataset_round.yaml")

    if not args.dry_run and not args.list and not is_round_worker():
        skip: set[str] = set()
        if args.resume:
            for vid in selected:
                status, _ = variant_progress(
                    runs_dir=args.runs_dir,
                    variant_id=vid,
                    run_name=f"ds_{vid}",
                    eval_targets=eval_targets,
                    skip_eval=args.skip_eval,
                )
                if status == "complete":
                    skip.add(vid)
        extra = ["--runs-dir", str(args.runs_dir)]
        if args.resume:
            extra.append("--resume")
        if args.skip_eval:
            extra.append("--skip-eval")
        if args.rebuild_packs:
            extra.append("--rebuild-packs")
        if args.batch is not None:
            extra.extend(["--batch", str(args.batch)])
        extra.extend(
            [
                "--model",
                args.model,
                "--imgsz",
                str(args.imgsz),
                "--epochs",
                str(args.epochs),
                "--warmup-epochs",
                str(args.warmup_epochs),
                "--freeze",
                str(args.freeze),
                "--lr0",
                str(args.lr0),
                "--patience",
                str(args.patience),
            ]
        )
        if args.labels_root is not None:
            extra.extend(["--labels-root", str(args.labels_root)])
        supervise_round_variants(
            script=Path(__file__).resolve(),
            variant_ids=selected,
            extra_args=extra,
            skip_ids=skip,
            runs_dir=args.runs_dir,
            log_prefix="ds_",
            session_label="dataset",
        )
        rows = []
        for vid in selected:
            row = load_result_json(result_json_path(args.runs_dir, f"ds_{vid}"))
            if row:
                rows.append(row)
        if rows:
            write_round_summary(args.runs_dir, rows, title=SUMMARY_TITLE)
        return

    results: list[dict] = []
    session_started = utcnow()
    session_t0 = time.perf_counter()
    session_variants: list[str] = []

    def _flush_timing(*, partial: bool = False) -> None:
        if args.dry_run:
            return
        session = None
        if session_variants or partial:
            session = timing_stamp(started_at=session_started, t0=session_t0)
            session["variants"] = list(session_variants)
            session["partial"] = partial
        write_round_timing(
            args.runs_dir,
            round_name="dataset",
            variant_rows=results,
            session=session,
        )

    def _flush_summary() -> None:
        if args.dry_run or not results:
            return
        _flush_timing()
        write_round_summary(args.runs_dir, results, title=SUMMARY_TITLE)

    try:
        for vid in selected:
            if vid not in runs:
                print(f"[warn] {vid} not listed in dataset_round.yaml runs; continuing")
            spec = resolve_variant(vid, labels_root=labels_root)
            run_name = f"ds_{vid}"
            print(f"\n=== dataset_round / {vid} ===")

            status, existing = variant_progress(
                runs_dir=args.runs_dir,
                variant_id=vid,
                run_name=run_name,
                eval_targets=eval_targets,
                skip_eval=args.skip_eval,
            )
            if args.dry_run:
                print(f"[dry-run] {vid} status={status} resume={args.resume}")
                if existing:
                    results.append(existing)
                continue

            if args.resume and status == "complete":
                print(f"Skip {vid}: already finished")
                results.append(existing or {})
                _flush_summary()
                continue

            yaml_path = ensure_dataset_pack(
                vid,
                datasets_root=datasets_root,
                labels_root=labels_root,
                rebuild=args.rebuild_packs,
                dry_run=False,
            )

            try:
                variant_started = utcnow()
                variant_t0 = time.perf_counter()
                train_result: dict
                if args.resume and status == "train_done":
                    print(f"Resume {vid}: weights exist, running remaining evals")
                    train_result = dict(existing or {})
                else:
                    train_result = run_train_job(
                        yaml_path=yaml_path,
                        model=args.model,
                        imgsz=args.imgsz,
                        epochs=args.epochs,
                        warmup_epochs=args.warmup_epochs,
                        freeze=args.freeze,
                        lr0=args.lr0,
                        patience=args.patience,
                        train_augmentation=spec.train_augmentation,
                        runs_dir=args.runs_dir,
                        run_name=run_name,
                        deliverable_name=f"dataset_round_{vid}_best.pt",
                        batch=args.batch,
                        dry_run=False,
                        extra_plan={
                            "round": "dataset",
                            "dataset_variant": vid,
                            "description": spec.description,
                            "eval_targets": [{"gt": g, "pack": str(p)} for g, p in eval_targets],
                        },
                    )

                if not args.skip_eval:
                    weights = train_result.get("best_weights")
                    evals = run_eval_targets(
                        weights=Path(str(weights)) if weights else None,
                        targets=eval_targets,
                        runs_dir=args.runs_dir,
                        variant_id=vid,
                        dry_run=False,
                    )
                    train_result["evals"] = evals
                train_result = attach_variant_timing(
                    train_result, started_at=variant_started, t0=variant_t0
                )
                session_variants.append(vid)
                if not args.skip_eval:
                    _save_result(args.runs_dir, vid, train_result)

                results.append(train_result)
                _flush_summary()
                release_torch_memory()
                print(f"Memory released after {vid}")
            except KeyboardInterrupt:
                raise
            except Exception:
                err_text = traceback.format_exc()
                err_path = abs_runs_dir(args.runs_dir) / f"{run_name}_error.txt"
                err_path.write_text(err_text, encoding="utf-8")
                log_path = variant_log_path(args.runs_dir, run_name)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n===== exception =====\n")
                    fh.write(err_text)
                print(f"{vid}: skipped. Detail → {err_path}")
                raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nInterrupted — finished variants are on disk; continue with --resume")
        _flush_timing(partial=True)
        if not args.dry_run and results:
            write_round_summary(args.runs_dir, results, title=SUMMARY_TITLE)
        release_torch_memory()
        raise

    _flush_summary()


if __name__ == "__main__":
    main()
