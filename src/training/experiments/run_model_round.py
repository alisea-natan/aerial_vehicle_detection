#!/usr/bin/env python3
"""Round 3 — protocol → family → size → epoch curve.

  python src/training/experiments/run_model_round.py --all
  python src/training/experiments/run_model_round.py --all --resume
  python src/training/experiments/run_model_round.py --pick-winner
  python src/training/experiments/run_model_round.py --group F --winner proto_frozen_e20
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

from common.config import PROJECT_ROOT, TRAIN_IMGSZ
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
    model_family,
    model_size_letter,
    with_model_size,
)

ROUND_CFG = PROJECT_ROOT / "config" / "experiments" / "model_round.yaml"
NEXT_GROUP = {"P": "F", "F": "S", "S": "E"}
GROUP_ORDER = ("P", "F", "S", "E")
FAMILY_S_WEIGHTS = {"yolov8": "yolov8s.pt", "yolo11": "yolo11s.pt"}


def parse_args() -> argparse.Namespace:
    cfg = load_yaml(ROUND_CFG)
    d = cfg.get("defaults") or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--group",
        default="P",
        help="P protocol (default), F family, S size, E epoch curve. Ignored with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run P→F→S→E. After each group, pick the winner (mean native mAP@0.5) and expand the next.",
    )
    parser.add_argument("--variant", action="append", default=None, help="Run these ids only.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dataset",
        default=str(d.get("dataset") or "auto"),
        help="Fixed dataset pack (Round 2 winner).",
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
        help="Previous-group variant id — required for groups F, S, E.",
    )
    parser.add_argument(
        "--pick-winner",
        action="store_true",
        help="Rank finished runs in --group (native mAP@0.5 A+B) and print the next command.",
    )
    return parser.parse_args()


def _abs_model(model: str) -> str:
    path = Path(model)
    if path.suffix in {".yaml", ".yml"} and not path.is_absolute():
        cand = PROJECT_ROOT / path
        if cand.is_file():
            return str(cand)
    return model


def _materialize_static(raw: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    model = str(raw["model"])
    staged = bool(raw.get("staged", True))
    train_backbone = bool(raw.get("train_backbone", True))
    return {
        **raw,
        "model": model,
        "imgsz": int(raw.get("imgsz", defaults.get("imgsz") or TRAIN_IMGSZ)),
        "epochs": int(raw.get("epochs", defaults.get("epochs") or 20)),
        "warmup_epochs": int(raw.get("warmup_epochs", defaults.get("warmup_epochs") or 0)),
        "freeze": int(raw.get("freeze", default_freeze(model))),
        "lr0": float(raw.get("lr0", defaults.get("lr0") or 0.01)),
        "patience": int(raw.get("patience", defaults.get("patience") or 7)),
        "train_augmentation": str(
            raw.get("train_augmentation", defaults.get("train_augmentation") or "poc")
        ),
        "pretrained": bool(raw.get("pretrained", True)),
        "train_backbone": train_backbone,
        "staged": staged,
        "eval_only": False,
        "predict_kw": dict(raw.get("predict_kw") or {}),
    }


def _load_winner_row(runs_dir: Path, winner_id: str) -> dict[str, Any]:
    result = load_result_json(result_json_path(runs_dir, f"md_{winner_id}"))
    if not result:
        raise SystemExit(
            f"No result for winner {winner_id!r}. Finish the previous group first "
            f"(python src/training/experiments/run_model_round.py --resume)."
        )
    return result


def _protocol_from_winner(winner: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    model = str(winner.get("model") or defaults.get("model") or "yolo11s.pt")
    return {
        "imgsz": int(winner.get("imgsz") or defaults.get("imgsz") or TRAIN_IMGSZ),
        "epochs": int(winner.get("epochs") or 20),
        "warmup_epochs": int(winner.get("warmup_epochs") or 0),
        "freeze": int(winner.get("freeze") or default_freeze(model)),
        "lr0": float(winner.get("lr0") or defaults.get("lr0") or 0.01),
        "patience": int(winner.get("patience") or defaults.get("patience") or 7),
        "train_augmentation": str(winner.get("train_augmentation") or "poc"),
        "pretrained": bool(winner.get("pretrained", True)),
        "staged": bool(winner.get("staged", True)),
        "train_backbone": bool(winner.get("train_backbone", True)),
        "eval_only": False,
        "predict_kw": {},
    }


def expand_group_f(winner_id: str, winner: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict]:
    """YOLOv8s vs YOLO11s; skip the family that is already the winner (usually 11s)."""
    proto = _protocol_from_winner(winner, cfg.get("defaults") or {})
    winner_family = model_family(str(winner.get("model") or ""))
    out: dict[str, dict] = {}
    for family, weights in FAMILY_S_WEIGHTS.items():
        vid = f"fam_{family}s"
        if family == winner_family:
            print(f"Group F: skip {vid} — already scored as {winner_id}")
            continue
        out[vid] = {
            **proto,
            "group": "F",
            "description": f"{weights} with protocol from {winner_id}",
            "model": weights,
            "freeze": default_freeze(weights),
        }
    if not out:
        print(f"Group F: skip — {winner_id} already covers both families.")
    return out


def expand_group_s(winner_id: str, winner: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict]:
    proto = _protocol_from_winner(winner, cfg.get("defaults") or {})
    model = str(winner["model"])
    letters = [str(x).lower() for x in (cfg.get("size_letters") or ["n", "s", "m"])]
    current = model_size_letter(model) or "s"
    out: dict[str, dict] = {}
    for letter in letters:
        vid = f"size_{letter}"
        if letter == current:
            print(f"Group S: skip {vid} — already scored as {winner_id} ({model})")
            continue
        sized = with_model_size(model, letter)
        out[vid] = {
            **proto,
            "group": "S",
            "description": f"{sized} ({letter}) with protocol from {winner_id}",
            "model": sized,
            "freeze": default_freeze(sized),
        }
    if not out:
        print(f"Group S: skip — {winner_id} already covers {letters}.")
    return out


def expand_group_e(winner_id: str, winner: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict]:
    """Epoch curve. Frozen: total head epochs. Staged: Stage-2 lengths, same warmup."""
    proto = _protocol_from_winner(winner, cfg.get("defaults") or {})
    staged = bool(proto["train_backbone"]) and bool(proto["staged"])
    if staged:
        lengths = [int(x) for x in (cfg.get("epoch_curve_unfreeze") or [5, 10, 15])]
        current = int(proto["epochs"])
        kind = "unfreeze"
    else:
        lengths = [int(x) for x in (cfg.get("epoch_curve_frozen") or [5, 10, 20])]
        current = int(proto["epochs"])
        kind = "frozen"
    out: dict[str, dict] = {}
    for n_ep in lengths:
        vid = f"ep_{kind}_{n_ep}"
        if n_ep == current:
            print(f"Group E: skip {vid} — already scored as {winner_id} ({n_ep} ep)")
            continue
        row = {
            **proto,
            "group": "E",
            "description": f"{winner.get('model')} {kind} {n_ep} ep (from {winner_id})",
            "model": winner["model"],
            "epochs": n_ep,
            "patience": 0,
        }
        out[vid] = row
    if not out:
        print(f"Group E: skip — {winner_id} already covers {kind} {lengths}.")
    return out


def resolve_group_variants(
    *,
    group: str,
    cfg: dict[str, Any],
    winner_id: str | None,
    runs_dir: Path,
) -> dict[str, dict[str, Any]]:
    group = group.upper()
    defaults = cfg.get("defaults") or {}
    static = dict(cfg.get("variants") or {})
    if group == "P":
        out = {}
        for vid, raw in static.items():
            if str(raw.get("group") or "P").upper() != "P":
                continue
            out[vid] = _materialize_static(dict(raw), defaults)
        order = list((cfg.get("groups") or {}).get("P", {}).get("runs") or out.keys())
        return {vid: out[vid] for vid in order if vid in out}
    if group in {"A", "B", "C", "D"}:
        raise SystemExit(
            f"Group {group} was the old LR/NMS/P2 grid. Use P → F → S → E "
            f"(see config/experiments/model_round.yaml)."
        )
    if not winner_id:
        raise SystemExit(f"Group {group} requires --winner <previous group id>")
    winner = _load_winner_row(runs_dir, winner_id)
    if group == "F":
        return expand_group_f(winner_id, winner, cfg)
    if group == "S":
        return expand_group_s(winner_id, winner, cfg)
    if group == "E":
        return expand_group_e(winner_id, winner, cfg)
    raise SystemExit(f"Unknown group {group!r}. Use P, F, S, or E.")


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


def pick_winner(
    runs_dir: Path,
    cfg: dict[str, Any],
    *,
    group: str,
    winner_id: str | None,
    require_complete: bool = True,
) -> dict[str, Any]:
    defaults = cfg.get("defaults") or {}
    delta = float(defaults.get("winner_delta") or 0.015)
    metric = str(defaults.get("decision_metric") or "map50")
    group = group.upper()
    eval_targets = parse_eval_targets(defaults)

    def _rank_id(vid: str, model: str | None = None) -> tuple[str, float, dict, str] | None:
        status, existing = variant_progress(
            runs_dir=runs_dir,
            variant_id=vid,
            run_name=f"md_{vid}",
            eval_targets=eval_targets,
            skip_eval=False,
        )
        if status != "complete" or not existing:
            return None
        bands = _native_bands(existing)
        score = _score(bands, metric)
        if score is None:
            return None
        fam = model_family(str(model or existing.get("model") or vid))
        return (vid, score, bands, fam)

    ranked: list[tuple[str, float, dict, str]] = []
    missing: list[str] = []

    if group == "P":
        variants = resolve_group_variants(
            group="P", cfg=cfg, winner_id=None, runs_dir=runs_dir
        )
        for vid, raw in variants.items():
            row = _rank_id(vid, str(raw.get("model") or ""))
            if row is None:
                missing.append(vid)
            else:
                ranked.append(row)
    else:
        if not winner_id:
            raise SystemExit(f"--pick-winner --group {group} needs --winner <id from previous group>")
        carry = _rank_id(winner_id)
        if carry is None:
            raise SystemExit(f"Winner {winner_id!r} is not finished; cannot rank Group {group}.")
        ranked.append(carry)
        variants = resolve_group_variants(
            group=group, cfg=cfg, winner_id=winner_id, runs_dir=runs_dir
        )
        for vid, raw in variants.items():
            row = _rank_id(vid, str(raw.get("model") or ""))
            if row is None:
                missing.append(vid)
            else:
                ranked.append(row)

    if missing:
        print(f"Incomplete Group {group} (need --resume):", ", ".join(missing))
        if require_complete:
            raise SystemExit(f"Incomplete Group {group}: {', '.join(missing)}")
    if not ranked:
        raise SystemExit(f"No finished Group {group} evals to rank.")

    ranked.sort(key=lambda r: r[1], reverse=True)
    a_lbl = BAND_LABELS[EVAL_BAND_A]
    b_lbl = BAND_LABELS[EVAL_BAND_B]
    metric_label = "mAP@0.5" if metric == "map50" else metric
    print(f"Native eval_manual — mean {metric_label} of {a_lbl} and {b_lbl}:")
    for vid, score, bands, _family in ranked:
        a = (bands.get(EVAL_BAND_A) or {}).get(metric)
        b = (bands.get(EVAL_BAND_B) or {}).get(metric)
        print(f"  {vid}: mean={100 * score:.1f}%  A={100 * float(a):.1f}%  B={100 * float(b):.1f}%")

    top_id, top_score, _, _ = ranked[0]
    print(f"\nTop: {top_id} ({100 * top_score:.1f}%)")
    tie = False
    second_id = None
    gap = None
    if len(ranked) >= 2:
        second_id, second_score, _, _ = ranked[1]
        gap = top_score - second_score
        print(f"Second: {second_id} ({100 * second_score:.1f}%)  gap={100 * gap:.1f} pts")
        if gap < delta:
            tie = True
            print(f"Gap < {100 * delta:.1f} pts — treat as a tie; inspect P/FA before locking.")
    nxt = NEXT_GROUP.get(group)
    if nxt:
        print(
            f"\nNext: python src/training/experiments/run_model_round.py "
            f"--group {nxt} --winner {top_id}"
        )
    else:
        print("\nEpoch curve is the last group. Lock the top id as the deliverable candidate.")
    return {
        "id": top_id,
        "mean_map50": top_score,
        "tie": tie,
        "second": second_id,
        "gap": gap,
        "ranked": [{"id": vid, "mean_map50": score} for vid, score, _, _ in ranked],
    }


def winners_path(runs_dir: Path) -> Path:
    return abs_runs_dir(runs_dir) / "winners.json"


def write_winners(runs_dir: Path, chain: dict[str, Any], *, dataset: str) -> Path:
    path = winners_path(runs_dir)
    payload = {
        "dataset": dataset,
        "decision_metric": "map50",
        "rule": "native mAP@0.5, mean of bands A and B",
        "updated_at": utcnow().isoformat(timespec="seconds"),
        "groups": chain,
        "final": (chain.get("E") or {}).get("id"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Winners → {path}")
    return path


def _predict_kw(raw: dict[str, Any]) -> dict[str, Any]:
    kw = dict(raw.get("predict_kw") or {})
    if kw.get("imgsz") is None and raw.get("imgsz") is not None:
        kw["imgsz"] = int(raw["imgsz"])
    return kw


def supervise_group(
    *,
    args: argparse.Namespace,
    group: str,
    winner_id: str | None,
    selected: list[str],
    eval_targets: list[tuple[str, Path]],
) -> None:
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
    if winner_id:
        extra.extend(["--winner", winner_id])
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


def run_all_groups(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    eval_targets = parse_eval_targets(cfg.get("defaults") or {})
    prev: str | None = None
    chain: dict[str, Any] = {}
    for group in GROUP_ORDER:
        print(f"\n######## Round 3 / Group {group} ########")
        variants = resolve_group_variants(
            group=group,
            cfg=cfg,
            winner_id=prev,
            runs_dir=args.runs_dir,
        )
        selected = list(variants.keys())
        if not selected:
            print(f"Group {group}: nothing new to train; keep {prev}")
        elif args.dry_run:
            for vid, raw in variants.items():
                print(f"  [dry-run] {vid}: {raw.get('description')}")
            print("[dry-run] F/S/E expand from the P winner after it is scored.")
            return
        else:
            supervise_group(
                args=args,
                group=group,
                winner_id=prev,
                selected=selected,
                eval_targets=eval_targets,
            )
        picked = pick_winner(
            args.runs_dir,
            cfg,
            group=group,
            winner_id=prev,
            require_complete=True,
        )
        prev = str(picked["id"])
        chain[group] = picked
        write_winners(args.runs_dir, chain, dataset=args.dataset)
    final = (chain.get("E") or chain.get("S") or chain.get("F") or chain.get("P") or {}).get("id")
    print(f"\nRound 3 complete. Deliverable candidate: {final}")
    print("Chain: " + " → ".join(f"{g}={chain[g]['id']}" for g in GROUP_ORDER if g in chain))


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
    group = str(args.group or "P").upper()

    if args.all and args.list:
        print(f"dataset={args.dataset}  chain={' → '.join(GROUP_ORDER)}")
        variants = resolve_group_variants(
            group="P", cfg=cfg, winner_id=None, runs_dir=args.runs_dir
        )
        print("Group P:")
        for vid, raw in variants.items():
            print(f"  {vid}: {raw.get('description', '')} model={raw.get('model')}")
        print("Groups F, S, E expand from the previous winner (mean native mAP@0.5).")
        return

    if args.pick_winner:
        pick_winner(
            args.runs_dir,
            cfg,
            group=group,
            winner_id=args.winner,
        )
        return

    if args.all:
        if is_round_worker():
            raise SystemExit("--all is the parent driver; workers must pass --group / --variant")
        run_all_groups(args, cfg)
        return

    variants = resolve_group_variants(
        group=group,
        cfg=cfg,
        winner_id=args.winner,
        runs_dir=args.runs_dir,
    )
    if args.list:
        print(f"Group {group}  dataset={args.dataset}")
        if not variants:
            print("  (nothing new to train)")
            return
        for vid, raw in variants.items():
            extra = " eval-only" if raw.get("eval_only") else f" model={raw.get('model')}"
            print(f"  {vid}: {raw.get('description', '')}{extra}")
        return

    if not variants:
        print(f"Group {group}: nothing new to train. Previous winner {args.winner} stands.")
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
        supervise_group(
            args=args,
            group=group,
            winner_id=args.winner,
            selected=selected,
            eval_targets=eval_targets,
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
                predict_kw = _predict_kw(raw)
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
