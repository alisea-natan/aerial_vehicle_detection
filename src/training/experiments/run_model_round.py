#!/usr/bin/env python3
"""Round 2 — model & optimization.

Default: Group A (architecture + coarse LR), 9 runs.
Later groups B–E need ``--winner`` from Group A.

  python src/training/experiments/run_model_round.py
  python src/training/experiments/run_model_round.py --resume
  python src/training/experiments/run_model_round.py --pick-winner
  python src/training/experiments/run_model_round.py --group B --winner yolo11s_lr_x2
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
from typing import Any

from common.config import PROJECT_ROOT
from training.evaluate import BAND_LABELS, EVAL_BAND_A, EVAL_BAND_B
from training.experiments.common import (
    abs_runs_dir,
    attach_variant_timing,
    collect_existing_evals,
    is_round_worker,
    load_result_json,
    load_yaml,
    parse_eval_targets,
    release_torch_memory,
    resolve_dataset_yaml,
    result_json_path,
    run_eval_targets,
    run_train_job,
    supervise_round_variants,
    timing_stamp,
    utcnow,
    variant_log_path,
    variant_progress,
    write_round_summary,
    write_round_timing,
)
from training.model_load import (
    default_freeze,
    default_lr0,
    model_family,
    p2_architecture,
    yaml_architecture,
)

ROUND_CFG = PROJECT_ROOT / "config" / "experiments" / "model_round.yaml"


def parse_args() -> argparse.Namespace:
    cfg = load_yaml(ROUND_CFG)
    d = cfg.get("defaults") or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--group",
        default="A",
        help="Round-2 group: A (default), B, C, D, or E.",
    )
    parser.add_argument("--variant", action="append", default=None, help="Run these ids only.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dataset",
        default=str(d.get("dataset") or "variant_6_strided"),
        help="Fixed dataset pack (Round 1 winner).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(str(d.get("runs_dir") or "outputs/experiments/model_round")),
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip finished variants; continue incomplete train/eval only.",
    )
    parser.add_argument(
        "--winner",
        default=None,
        help="Group A variant id — required for groups B–E.",
    )
    parser.add_argument(
        "--pick-winner",
        action="store_true",
        help="Rank finished Group A YOLO runs (native mAP@0.5 A+B) and print the rule.",
    )
    parser.add_argument(
        "--tie-candidate",
        action="append",
        default=None,
        help="With --group C, extra Group A ids for the NMS / no-NMS / Soft-NMS eval.",
    )
    return parser.parse_args()


def _abs_model(model: str) -> str:
    path = Path(model)
    if path.suffix in {".yaml", ".yml"} and not path.is_absolute():
        cand = PROJECT_ROOT / path
        if cand.is_file():
            return str(cand)
    return model


def _variant_lr0(raw: dict[str, Any], defaults: dict[str, Any]) -> float:
    model = str(raw.get("model") or defaults.get("model") or "yolo11s.pt")
    if raw.get("lr0") is not None:
        return float(raw["lr0"])
    base = default_lr0(model, float(defaults.get("lr0") or 0.01))
    return base * float(raw.get("lr_mult", 1.0))


def _materialize_group_a(raw: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    model = str(raw["model"])
    staged = bool(raw.get("staged", True))
    freeze = int(raw.get("freeze", default_freeze(model)))
    return {
        **raw,
        "model": model,
        "imgsz": int(raw.get("imgsz", defaults.get("imgsz") or 1024)),
        "epochs": int(raw.get("epochs", defaults.get("epochs") or 15)),
        "warmup_epochs": int(raw.get("warmup_epochs", defaults.get("warmup_epochs") or 5)),
        "freeze": freeze,
        "lr0": _variant_lr0(raw, defaults),
        "patience": int(raw.get("patience", defaults.get("patience") or 7)),
        "train_augmentation": str(
            raw.get("train_augmentation", defaults.get("train_augmentation") or "poc")
        ),
        "pretrained": True,
        "train_backbone": True,
        "staged": staged,
        "eval_only": False,
        "predict_kw": dict(raw.get("predict_kw") or {}),
    }


def _load_winner_row(runs_dir: Path, winner_id: str) -> dict[str, Any]:
    result = load_result_json(result_json_path(runs_dir, f"md_{winner_id}"))
    if not result:
        raise SystemExit(
            f"No result for winner {winner_id!r}. Finish Group A first "
            f"(python src/training/experiments/run_model_round.py --resume)."
        )
    return result


def expand_group_b(winner_id: str, winner: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict]:
    defaults = cfg.get("defaults") or {}
    mults = [float(x) for x in (cfg.get("fine_lr_mults") or [0.8, 1.0, 1.25])]
    base_lr = float(winner.get("lr0") or defaults.get("lr0") or 0.01)
    out: dict[str, dict] = {}
    for mult in mults:
        tag = str(mult).replace(".", "p")
        vid = f"{winner_id}_lr_{tag}"
        out[vid] = {
            "group": "B",
            "description": f"Fine LR ×{mult} around {winner_id} (lr0={base_lr * mult:.6g})",
            "model": winner["model"],
            "imgsz": int(winner.get("imgsz") or defaults.get("imgsz") or 1024),
            "epochs": int(defaults.get("epochs") or 15),
            "warmup_epochs": int(defaults.get("warmup_epochs") or 5),
            "freeze": int(winner.get("freeze") or default_freeze(str(winner["model"]))),
            "lr0": base_lr * mult,
            "patience": int(defaults.get("patience") or 7),
            "train_augmentation": str(winner.get("train_augmentation") or "poc"),
            "pretrained": True,
            "train_backbone": True,
            "staged": True,
            "eval_only": False,
            "predict_kw": {},
        }
    return out


def expand_group_c(
    winner_id: str,
    winner: dict[str, Any],
    extra_ids: list[str],
    runs_dir: Path,
) -> dict[str, dict]:
    """Eval-only: hard NMS vs no NMS vs Gaussian Soft-NMS on frozen weights."""
    sources = [(winner_id, winner)]
    for vid in extra_ids:
        if vid == winner_id:
            continue
        sources.append((vid, _load_winner_row(runs_dir, vid)))

    out: dict[str, dict] = {}
    for src_id, row in sources:
        weights = row.get("best_weights")
        specs = [
            ("nms_on", {"nms": "hard", "iou": 0.7}, "hard NMS (iou=0.7)"),
            ("nms_off", {"nms": "off"}, "no NMS"),
            (
                "nms_soft",
                {"nms": "soft", "iou": 0.7, "soft_nms_sigma": 0.5, "soft_nms_method": "gaussian"},
                "Soft-NMS Gaussian σ=0.5",
            ),
        ]
        for suffix, predict_kw, desc in specs:
            vid = f"{src_id}_{suffix}"
            out[vid] = {
                "group": "C",
                "description": f"{src_id}: {desc}",
                "model": row.get("model") or winner["model"],
                "eval_only": True,
                "source_weights": weights,
                "predict_kw": predict_kw,
                "pretrained": True,
                "staged": True,
            }
    return out


def expand_group_d(winner_id: str, winner: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict]:
    defaults = cfg.get("defaults") or {}
    model = str(winner["model"])
    imgsz = int(winner.get("imgsz") or defaults.get("imgsz") or 1024)
    freeze = int(winner.get("freeze") or default_freeze(model))
    lr0 = float(winner.get("lr0") or defaults.get("lr0") or 0.01)
    patience = int(defaults.get("patience") or 7)
    aug = str(winner.get("train_augmentation") or "poc")
    full_warm = int(defaults.get("full_warmup_epochs") or 5)
    full_ep = int(defaults.get("full_epochs") or 20)
    shared = {
        "group": "D",
        "imgsz": imgsz,
        "patience": patience,
        "train_augmentation": aug,
        "lr0": lr0,
        "eval_only": False,
        "predict_kw": {},
    }
    return {
        f"{winner_id}_bb_full": {
            **shared,
            "description": "COCO-pretrained, full fine-tune (staged)",
            "model": model,
            "warmup_epochs": full_warm,
            "epochs": full_ep,
            "freeze": freeze,
            "pretrained": True,
            "staged": True,
            "train_backbone": True,
        },
        f"{winner_id}_bb_frozen": {
            **shared,
            "description": "COCO-pretrained, frozen backbone, head only",
            "model": model,
            "warmup_epochs": 0,
            "epochs": full_warm + full_ep,
            "freeze": freeze,
            "pretrained": True,
            "staged": False,
            "train_backbone": False,
        },
        f"{winner_id}_bb_scratch": {
            **shared,
            "description": "From-scratch (yaml, random init)",
            "model": yaml_architecture(model),
            "warmup_epochs": 0,
            "epochs": full_warm + full_ep,
            "freeze": 0,
            "pretrained": False,
            "staged": False,
            "train_backbone": True,
        },
    }


def expand_group_e(winner_id: str, winner: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict]:
    defaults = cfg.get("defaults") or {}
    model = str(winner["model"])
    imgsz = int(winner.get("imgsz") or defaults.get("imgsz") or 1024)
    freeze = int(winner.get("freeze") or default_freeze(model))
    lr0 = float(winner.get("lr0") or defaults.get("lr0") or 0.01)
    patience = int(defaults.get("patience") or 7)
    aug = str(winner.get("train_augmentation") or "poc")
    full_warm = int(defaults.get("full_warmup_epochs") or 5)
    full_ep = int(defaults.get("full_epochs") or 20)
    shared = {
        "group": "E",
        "imgsz": imgsz,
        "warmup_epochs": full_warm,
        "epochs": full_ep,
        "freeze": freeze,
        "lr0": lr0,
        "patience": patience,
        "train_augmentation": aug,
        "pretrained": True,
        "train_backbone": True,
        "staged": True,
        "eval_only": False,
        "predict_kw": {},
    }
    return {
        f"{winner_id}_head_std": {
            **shared,
            "description": "Standard head (winner architecture, full budget)",
            "model": model,
        },
        f"{winner_id}_head_p2": {
            **shared,
            "description": "P2 small-object head",
            "model": p2_architecture(model),
        },
    }


def resolve_group_variants(
    *,
    group: str,
    cfg: dict[str, Any],
    winner_id: str | None,
    runs_dir: Path,
    tie_candidates: list[str],
) -> dict[str, dict[str, Any]]:
    group = group.upper()
    defaults = cfg.get("defaults") or {}
    static = dict(cfg.get("variants") or {})
    if group == "A":
        out = {}
        for vid, raw in static.items():
            if str(raw.get("group") or "A").upper() != "A":
                continue
            out[vid] = _materialize_group_a(dict(raw), defaults)
        order = list((cfg.get("groups") or {}).get("A", {}).get("runs") or out.keys())
        return {vid: out[vid] for vid in order if vid in out}
    if not winner_id:
        raise SystemExit(f"Group {group} requires --winner <Group A variant id>")
    winner = _load_winner_row(runs_dir, winner_id)
    if group == "B":
        return expand_group_b(winner_id, winner, cfg)
    if group == "C":
        return expand_group_c(winner_id, winner, tie_candidates, runs_dir)
    if group == "D":
        return expand_group_d(winner_id, winner, cfg)
    if group == "E":
        return expand_group_e(winner_id, winner, cfg)
    raise SystemExit(f"Unknown group {group!r}. Use A, B, C, D, or E.")


def _native_bands(row: dict[str, Any]) -> dict[str, Any]:
    evals = dict(row.get("evals") or {})
    path = evals.get("manual") or evals.get("eval_manual")
    if not path:
        return {}
    metrics = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    return dict(metrics.get("bands") or {})


def _score(bands: dict[str, Any], metric: str = "map50") -> float | None:
    a = (bands.get(EVAL_BAND_A) or {}).get(metric)
    b = (bands.get(EVAL_BAND_B) or {}).get(metric)
    if a is None or b is None:
        return None
    return (float(a) + float(b)) / 2.0


def pick_winner(runs_dir: Path, cfg: dict[str, Any]) -> None:
    defaults = cfg.get("defaults") or {}
    delta = float(defaults.get("winner_delta") or 0.015)
    metric = str(defaults.get("decision_metric") or "map50")
    variants = resolve_group_variants(
        group="A", cfg=cfg, winner_id=None, runs_dir=runs_dir, tie_candidates=[]
    )
    ranked: list[tuple[str, float, dict, str]] = []
    missing = []
    for vid, raw in variants.items():
        status, existing = variant_progress(
            runs_dir=runs_dir,
            variant_id=vid,
            run_name=f"md_{vid}",
            eval_targets=parse_eval_targets(defaults),
            skip_eval=False,
        )
        if status != "complete" or not existing:
            missing.append(vid)
            continue
        bands = _native_bands(existing)
        score = _score(bands, metric)
        if score is None:
            missing.append(vid)
            continue
        ranked.append((vid, score, bands, model_family(str(raw["model"]))))

    if missing:
        print("Incomplete Group A (need --resume):", ", ".join(missing))
    if not ranked:
        raise SystemExit("No finished Group A evals to rank.")

    ranked.sort(key=lambda r: r[1], reverse=True)
    a_lbl = BAND_LABELS[EVAL_BAND_A]
    b_lbl = BAND_LABELS[EVAL_BAND_B]
    metric_label = "mAP@0.5" if metric == "map50" else metric
    print(f"Native eval_manual — mean {metric_label} of {a_lbl} and {b_lbl}:")
    for vid, score, bands, _family in ranked:
        a = (bands.get(EVAL_BAND_A) or {}).get(metric)
        b = (bands.get(EVAL_BAND_B) or {}).get(metric)
        print(f"  {vid}: mean={100 * score:.1f}%  A={100 * float(a):.1f}%  B={100 * float(b):.1f}%")

    if len(ranked) < 1:
        raise SystemExit("No Group A runs finished.")
    top_id, top_score, _, _ = ranked[0]
    print(f"\nTop: {top_id} ({100 * top_score:.1f}%)")
    if len(ranked) >= 2:
        second_id, second_score, _, _ = ranked[1]
        gap = top_score - second_score
        print(f"Second:   {second_id} ({100 * second_score:.1f}%)  gap={100 * gap:.1f} pts")
        if gap >= delta:
            print(
                f"Gap ≥ {100 * delta:.1f} pts → take {top_id} into Groups B, C, D, E. "
                f"Skip Group C for the second model."
            )
            print(f"  python src/training/experiments/run_model_round.py --group B --winner {top_id}")
            print(f"  python src/training/experiments/run_model_round.py --group C --winner {top_id}")
        else:
            print(
                f"Gap < {100 * delta:.1f} pts (noise) → run Group C on both, then pick one for D/E."
            )
            print(
                f"  python src/training/experiments/run_model_round.py --group C "
                f"--winner {top_id} --tie-candidate {second_id}"
            )
    else:
        print(f"  python src/training/experiments/run_model_round.py --group B --winner {top_id}")


def collect_all_model_results(
    runs_dir: Path,
    eval_targets: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(abs_runs_dir(runs_dir).glob("md_*_result.json")):
        data = load_result_json(path)
        if not data:
            continue
        vid = str(data.get("model_variant") or path.stem.replace("md_", "").replace("_result", ""))
        evals = collect_existing_evals(runs_dir, vid, eval_targets)
        if evals:
            data["evals"] = evals
        rows.append(data)
    return rows


def _write_eval_only_result(runs_dir: Path, run_name: str, payload: dict[str, Any]) -> Path:
    path = result_json_path(runs_dir, run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Result → {path}")
    return path


def main() -> None:
    args = parse_args()
    cfg = load_yaml(ROUND_CFG)
    defaults = dict(cfg.get("defaults") or {})
    eval_targets = parse_eval_targets(defaults)
    group = str(args.group or "A").upper()

    if args.pick_winner:
        pick_winner(args.runs_dir, cfg)
        return

    variants = resolve_group_variants(
        group=group,
        cfg=cfg,
        winner_id=args.winner,
        runs_dir=args.runs_dir,
        tie_candidates=list(args.tie_candidate or []),
    )
    if args.list:
        print(f"Group {group}  dataset={args.dataset}")
        for vid, raw in variants.items():
            extra = " eval-only" if raw.get("eval_only") else f" model={raw.get('model')}"
            print(f"  {vid}: {raw.get('description', '')}{extra}")
        return

    selected = list(args.variant or [])
    if not selected:
        selected = list(variants.keys())
    unknown = [v for v in selected if v not in variants]
    if unknown:
        raise SystemExit(f"Unknown variant(s) {unknown}. Known: {', '.join(variants)}")

    if args.dry_run:
        yaml_path = PROJECT_ROOT / Path(str(defaults.get("datasets_root") or "data/datasets")) / args.dataset / "data.yaml"
    else:
        yaml_path = resolve_dataset_yaml(
            Path(str(defaults.get("datasets_root") or "data/datasets")),
            args.dataset,
        )

    if not args.dry_run and not is_round_worker():
        skip: set[str] = set()
        if args.resume:
            for vid in selected:
                status, _ = variant_progress(
                    runs_dir=args.runs_dir,
                    variant_id=vid,
                    run_name=f"md_{vid}",
                    eval_targets=eval_targets,
                    skip_eval=args.skip_eval,
                )
                if status == "complete":
                    skip.add(vid)
        extra = [
            "--group",
            group,
            "--dataset",
            args.dataset,
            "--runs-dir",
            str(args.runs_dir),
        ]
        if args.resume:
            extra.append("--resume")
        if args.skip_eval:
            extra.append("--skip-eval")
        if args.batch is not None:
            extra.extend(["--batch", str(args.batch)])
        if args.winner:
            extra.extend(["--winner", args.winner])
        for cand in args.tie_candidate or []:
            extra.extend(["--tie-candidate", cand])
        supervise_round_variants(
            script=Path(__file__).resolve(),
            variant_ids=selected,
            extra_args=extra,
            skip_ids=skip,
            runs_dir=args.runs_dir,
            log_prefix="md_",
            session_label=f"model_{group}",
        )
        all_rows = collect_all_model_results(args.runs_dir, eval_targets)
        if all_rows:
            write_round_summary(
                args.runs_dir,
                all_rows,
                title=f"Model round — dataset={args.dataset}",
            )
        return

    results: list[dict] = []
    session_started = utcnow()
    session_t0 = time.perf_counter()
    session_variants: list[str] = []

    def _flush_timing(*, partial: bool = False) -> None:
        session = None
        if session_variants or partial:
            session = timing_stamp(started_at=session_started, t0=session_t0)
            session["group"] = group
            session["variants"] = list(session_variants)
            session["partial"] = partial
        write_round_timing(
            args.runs_dir,
            round_name="model",
            variant_rows=results,
            session=session,
        )

    def _flush_summary() -> None:
        _flush_timing()
        all_rows = collect_all_model_results(args.runs_dir, eval_targets) or results
        write_round_summary(
            args.runs_dir,
            all_rows,
            title=f"Model round — dataset={args.dataset}",
        )

    try:
        for vid in selected:
            raw = variants[vid]
            run_name = f"md_{vid}"
            print(f"\n=== model_round / {group} / {vid} (dataset={args.dataset}) ===")
            status, existing = variant_progress(
                runs_dir=args.runs_dir,
                variant_id=vid,
                run_name=run_name,
                eval_targets=eval_targets,
                skip_eval=args.skip_eval,
            )
            if args.dry_run:
                print(f"[dry-run] {vid} status={status} {raw.get('description')}")
                continue
            if args.resume and status == "complete":
                print(f"Skip {vid}: already finished")
                results.append(existing or {})
                continue

            try:
                predict_kw = dict(raw.get("predict_kw") or {})
                variant_started = utcnow()
                variant_t0 = time.perf_counter()
                if raw.get("eval_only"):
                    weights = Path(str(raw.get("source_weights") or ""))
                    if not weights.is_file():
                        raise SystemExit(f"{vid}: missing source weights {weights}")
                    train_result = {
                        "round": "model",
                        "group": group,
                        "model_variant": vid,
                        "dataset": args.dataset,
                        "description": str(raw.get("description") or ""),
                        "eval_only": True,
                        "predict_kw": predict_kw,
                        "best_weights": str(weights),
                    }
                    if not args.skip_eval:
                        evals = run_eval_targets(
                            weights=weights,
                            targets=eval_targets,
                            runs_dir=args.runs_dir,
                            variant_id=vid,
                            dry_run=False,
                            predict_kw=predict_kw,
                        )
                        train_result["evals"] = evals
                elif args.resume and status == "train_done":
                    print(f"Resume {vid}: weights exist, running remaining evals")
                    train_result = dict(existing or {})
                    if not args.skip_eval:
                        weights = Path(str(train_result.get("best_weights") or ""))
                        evals = run_eval_targets(
                            weights=weights if weights.is_file() else None,
                            targets=eval_targets,
                            runs_dir=args.runs_dir,
                            variant_id=vid,
                            dry_run=False,
                            predict_kw=predict_kw,
                        )
                        train_result["evals"] = evals
                else:
                    model = _abs_model(str(raw["model"]))
                    train_result = run_train_job(
                        yaml_path=yaml_path,
                        model=model,
                        imgsz=int(raw["imgsz"]),
                        epochs=int(raw["epochs"]),
                        warmup_epochs=int(raw["warmup_epochs"]),
                        freeze=int(raw["freeze"]),
                        lr0=float(raw["lr0"]),
                        patience=int(raw["patience"]),
                        train_augmentation=str(raw["train_augmentation"]),
                        runs_dir=args.runs_dir,
                        run_name=run_name,
                        deliverable_name=f"model_round_{vid}_best.pt",
                        batch=args.batch,
                        dry_run=False,
                        pretrained=bool(raw.get("pretrained", True)),
                        staged=bool(raw.get("staged", True)),
                        train_backbone=bool(raw.get("train_backbone", True)),
                        extra_plan={
                            "round": "model",
                            "group": group,
                            "model_variant": vid,
                            "dataset": args.dataset,
                            "description": str(raw.get("description") or ""),
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
                            predict_kw=predict_kw,
                        )
                        train_result["evals"] = evals
                train_result = attach_variant_timing(
                    train_result, started_at=variant_started, t0=variant_t0
                )
                session_variants.append(vid)
                _write_eval_only_result(args.runs_dir, run_name, train_result)
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
        if not args.dry_run:
            _flush_timing(partial=True)
            all_rows = collect_all_model_results(args.runs_dir, eval_targets) or results
            if all_rows:
                write_round_summary(
                    args.runs_dir,
                    all_rows,
                    title=f"Model round — dataset={args.dataset}",
                )
        release_torch_memory()
        raise

    if not args.dry_run:
        _flush_summary()


if __name__ == "__main__":
    main()
