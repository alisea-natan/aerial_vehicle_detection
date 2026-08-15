"""Shared helpers for experiment rounds (dataset / model)."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from common.aug_config import aug_preset
from common.config import PROJECT_ROOT, load_clip_tile_config
from training.datasets.generate_variant import build_variant
from training.datasets.specs import resolve_variant
from training.evaluate import (
    BAND_LABELS,
    EVAL_BAND_A,
    EVAL_BAND_B,
    EVAL_PACKS,
    evaluate_prepared_pack,
)
from training.train import (
    DEVICE,
    default_batch_size,
    default_workers,
    release_torch_memory,
    train_model,
)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing config: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_dataset_yaml(datasets_root: Path, dataset_id: str) -> Path:
    root = datasets_root if datasets_root.is_absolute() else PROJECT_ROOT / datasets_root
    yaml_path = root / dataset_id / "data.yaml"
    if yaml_path.is_file():
        return yaml_path
    side = root / f"{dataset_id}.manifest.json"
    if side.is_file():
        meta = json.loads(side.read_text(encoding="utf-8"))
        reuse = meta.get("reuse_from")
        if reuse:
            cand = root / str(reuse) / "data.yaml"
            if cand.is_file():
                return cand
    raise SystemExit(
        f"Missing dataset pack {dataset_id!r} under {root}.\n"
        f"Build: python src/training/datasets/generate_variant.py --variant {dataset_id}"
    )


def ensure_dataset_pack(
    dataset_id: str,
    *,
    datasets_root: Path,
    labels_root: Path | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
) -> Path:
    """Build a variant pack if missing (reuse deps first). Returns data.yaml."""
    spec = resolve_variant(dataset_id, labels_root=labels_root)
    if spec.dataset_action == "reuse" and spec.reuse_from:
        ensure_dataset_pack(
            spec.reuse_from,
            datasets_root=datasets_root,
            labels_root=labels_root,
            rebuild=rebuild,
            dry_run=dry_run,
        )
    root = datasets_root if datasets_root.is_absolute() else PROJECT_ROOT / datasets_root
    yaml_path = root / dataset_id / "data.yaml"
    if yaml_path.is_file() and not rebuild:
        return yaml_path
    if dry_run:
        print(f"[dry-run] would build pack {dataset_id} → {yaml_path}")
        return yaml_path
    print(f"Building pack {dataset_id} …")
    build_variant(spec, recreate=True)
    return resolve_dataset_yaml(datasets_root, dataset_id)


def resolve_eval_pack(eval_pack: str | None, eval_gt: str) -> Path:
    if eval_pack:
        path = Path(eval_pack)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return EVAL_PACKS.get(eval_gt, EVAL_PACKS["manual"])


def run_eval_job(
    *,
    weights: Path,
    eval_pack: Path,
    eval_gt: str,
    output_dir: Path,
    dry_run: bool = False,
    predict_kw: dict[str, Any] | None = None,
) -> Path:
    output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    if dry_run:
        print(f"[dry-run] would eval {weights} on {eval_pack} → {output_dir}")
        return output_dir / "eval_metrics.json"
    if not eval_pack.is_dir() or not (eval_pack / "data.yaml").is_file():
        raise SystemExit(
            f"Missing eval pack {eval_pack}.\n"
            "Build: python src/training/prepare_eval.py"
            + (" --scale-adapt" if "adapted" in eval_gt else "")
        )
    evaluate_prepared_pack(
        eval_pack,
        weights=weights,
        gt_name=eval_gt,
        conf_override=None,
        iou_thresh=0.5,
        clip_filter=None,
        output_dir=output_dir,
        tile_config=load_clip_tile_config(),
        predict_kw=predict_kw,
    )
    release_torch_memory()
    return output_dir / "eval_metrics.json"


def parse_eval_targets(defaults: dict[str, Any]) -> list[tuple[str, Path]]:
    """[(gt_name, pack_dir), ...] — native + adapted by default."""
    raw = defaults.get("eval_targets")
    if raw:
        out: list[tuple[str, Path]] = []
        for item in raw:
            row = dict(item or {})
            gt = str(row.get("gt") or "manual")
            out.append((gt, resolve_eval_pack(row.get("pack"), gt)))
        return out
    gt = str(defaults.get("eval_gt") or "manual")
    return [(gt, resolve_eval_pack(defaults.get("eval_pack"), gt))]


def run_eval_targets(
    *,
    weights: Path | None,
    targets: list[tuple[str, Path]],
    runs_dir: Path,
    variant_id: str,
    dry_run: bool = False,
    predict_kw: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Eval one weights file on each pack. Returns gt → eval_metrics.json path."""
    root = runs_dir if runs_dir.is_absolute() else PROJECT_ROOT / runs_dir
    evals: dict[str, str] = {}
    for gt_name, pack in targets:
        eval_out = root / variant_id / f"eval_{gt_name}"
        eval_json = run_eval_job(
            weights=weights if weights else Path("dry-run.pt"),
            eval_pack=pack,
            eval_gt=gt_name,
            output_dir=eval_out,
            dry_run=dry_run or not weights,
            predict_kw=predict_kw,
        )
        evals[gt_name] = str(eval_json)
    release_torch_memory()
    return evals


def abs_runs_dir(runs_dir: Path) -> Path:
    return runs_dir if runs_dir.is_absolute() else PROJECT_ROOT / runs_dir


ROUND_WORKER_ENV = "VD_ROUND_WORKER"
_MPS_STDERR_NOISE = (
    "MPSNDArray",
    "failed assertion",
    "MetalPerformanceShaders",
    "Abort trap",
    "zsh: abort",
)


def is_round_worker() -> bool:
    return os.environ.get(ROUND_WORKER_ENV) == "1"


def round_logs_dir(runs_dir: Path) -> Path:
    path = abs_runs_dir(runs_dir) / "logs"
    (path / "sessions").mkdir(parents=True, exist_ok=True)
    return path


def variant_log_path(runs_dir: Path, run_name: str) -> Path:
    return round_logs_dir(runs_dir) / f"{run_name}.log"


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if text and not text.endswith("\n"):
            fh.write("\n")


class _TerminalSink:
    """Write to a stream, dropping Metal abort noise."""

    def __init__(self, stream) -> None:
        self.stream = stream

    def write(self, line: str) -> None:
        if any(tok in line for tok in _MPS_STDERR_NOISE):
            return
        self.stream.write(line)
        self.stream.flush()


def supervise_round_variants(
    *,
    script: Path,
    variant_ids: list[str],
    extra_args: list[str],
    runs_dir: Path,
    log_prefix: str,
    session_label: str,
    skip_ids: set[str] | None = None,
) -> None:
    """One child process per variant. Full stdout/stderr → logs/; Metal noise hidden in the terminal."""
    import subprocess

    skip_ids = skip_ids or set()
    logs = round_logs_dir(runs_dir)
    stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
    session_path = logs / "sessions" / f"{session_label}_{stamp}.log"
    latest_session = logs / "sessions" / f"{session_label}_latest.log"
    header = (
        f"===== session {session_label} {utcnow().isoformat(timespec='seconds')} =====\n"
        f"script={script} variants={variant_ids}\n"
    )
    _append_log(session_path, header)
    latest_session.write_text(header, encoding="utf-8")
    print(f"Session log → {session_path}")

    def _session(line: str) -> None:
        _append_log(session_path, line)
        _append_log(latest_session, line)

    for vid in variant_ids:
        run_name = f"{log_prefix}{vid}"
        if vid in skip_ids:
            msg = f"Skip {vid}: already finished"
            print(msg)
            _session(msg)
            continue
        cmd = [sys.executable, "-u", str(script), "--variant", vid, *extra_args]
        env = os.environ.copy()
        env[ROUND_WORKER_ENV] = "1"
        banner = f"\n=== isolate {vid} ===\n"
        print(banner, end="")
        _session(banner.rstrip())
        var_log = variant_log_path(runs_dir, run_name)
        start_line = f"\n===== {utcnow().isoformat(timespec='seconds')} start {vid} =====\n"
        _append_log(var_log, start_line)
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        term = _TerminalSink(sys.stdout)
        try:
            with var_log.open("a", encoding="utf-8") as log_fh:
                for line in proc.stdout:
                    log_fh.write(line)
                    log_fh.flush()
                    term.write(line)
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            raise
        end_line = f"===== {utcnow().isoformat(timespec='seconds')} exit {rc} {vid} =====\n"
        _append_log(var_log, end_line)
        _session(f"{vid}: exit {rc}  log={var_log}")
        print(f"Run log → {var_log}")
        if rc != 0:
            print(f"{vid}: stopped on this Mac (exit {rc}). Continuing.")


def result_json_path(runs_dir: Path, run_name: str) -> Path:
    return abs_runs_dir(runs_dir) / f"{run_name}_result.json"


def trained_weights_path(runs_dir: Path, run_name: str) -> Path:
    return abs_runs_dir(runs_dir) / run_name / "weights" / "best.pt"


def eval_metrics_path(runs_dir: Path, variant_id: str, gt_name: str) -> Path:
    return abs_runs_dir(runs_dir) / variant_id / f"eval_{gt_name}" / "eval_metrics.json"


def load_result_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


ROUND_TIMING_NAME = "round_timing.json"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    sec = max(0.0, float(seconds))
    if sec < 60:
        return f"{sec:.1f}s"
    total = int(round(sec))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m" if secs == 0 else f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def timing_stamp(*, started_at: datetime, t0: float) -> dict[str, Any]:
    elapsed = time.perf_counter() - t0
    return {
        "started_at": started_at.isoformat(),
        "finished_at": utcnow().isoformat(),
        "elapsed_sec": round(elapsed, 2),
        "elapsed_human": format_duration(elapsed),
    }


def eval_elapsed_sec(row: dict[str, Any]) -> float | None:
    total = 0.0
    found = False
    for path in dict(row.get("evals") or {}).values():
        p = Path(str(path))
        if not p.is_file():
            continue
        try:
            metrics = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sec = (metrics.get("timing") or {}).get("elapsed_sec")
        if sec is not None:
            total += float(sec)
            found = True
    return round(total, 2) if found else None


def attach_variant_timing(row: dict[str, Any], *, started_at: datetime, t0: float) -> dict[str, Any]:
    """Stamp wall time for one train+eval variant (overwrites finished_at)."""
    out = {**row, **timing_stamp(started_at=started_at, t0=t0)}
    eval_sec = eval_elapsed_sec(out)
    if eval_sec is not None:
        out["eval_elapsed_sec"] = eval_sec
        out["eval_elapsed_human"] = format_duration(eval_sec)
    return out


def load_round_timing(runs_dir: Path) -> dict[str, Any]:
    path = abs_runs_dir(runs_dir) / ROUND_TIMING_NAME
    if not path.is_file():
        return {"sessions": [], "variants": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"sessions": [], "variants": {}}
    if not isinstance(data, dict):
        return {"sessions": [], "variants": {}}
    data.setdefault("sessions", [])
    data.setdefault("variants", {})
    return data


def write_round_timing(
    runs_dir: Path,
    *,
    round_name: str,
    variant_rows: list[dict[str, Any]],
    session: dict[str, Any] | None = None,
    note: str | None = None,
) -> Path:
    """Merge session + per-variant wall times into ``round_timing.json``."""
    runs_dir = abs_runs_dir(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    data = load_round_timing(runs_dir)
    data["round"] = round_name
    data["updated_at"] = utcnow().isoformat(timespec="seconds")
    if note:
        data["note"] = note
    if session:
        start = session.get("started_at")
        sessions = [s for s in list(data.get("sessions") or []) if s.get("started_at") != start]
        sessions.append(session)
        data["sessions"] = sessions
    variants = dict(data.get("variants") or {})
    for row in variant_rows:
        vid = str(
            row.get("dataset_variant") or row.get("model_variant") or row.get("run_name") or ""
        )
        if not vid or row.get("elapsed_sec") is None:
            continue
        eval_sec = row.get("eval_elapsed_sec")
        if eval_sec is None:
            eval_sec = eval_elapsed_sec(row)
        entry: dict[str, Any] = {
            "started_at": row.get("started_at"),
            "finished_at": row.get("finished_at"),
            "elapsed_sec": row.get("elapsed_sec"),
            "elapsed_human": row.get("elapsed_human") or format_duration(row.get("elapsed_sec")),
        }
        if row.get("train_elapsed_sec") is not None:
            entry["train_elapsed_sec"] = row.get("train_elapsed_sec")
            entry["train_elapsed_human"] = row.get("train_elapsed_human") or format_duration(
                row.get("train_elapsed_sec")
            )
        if eval_sec is not None:
            entry["eval_elapsed_sec"] = eval_sec
            entry["eval_elapsed_human"] = format_duration(eval_sec)
        if row.get("group"):
            entry["group"] = row.get("group")
        variants[vid] = entry
    data["variants"] = variants
    compute = sum(float(v.get("elapsed_sec") or 0) for v in variants.values())
    sessions_sec = sum(float(s.get("elapsed_sec") or 0) for s in data.get("sessions") or [])
    data["totals"] = {
        "compute_sec": round(compute, 2),
        "compute_human": format_duration(compute),
        "sessions_sec": round(sessions_sec, 2),
        "sessions_human": format_duration(sessions_sec),
        "n_sessions": len(data.get("sessions") or []),
        "n_variants": len(variants),
    }
    path = runs_dir / ROUND_TIMING_NAME
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        f"Round timing → {path}  compute={data['totals']['compute_human']}  "
        f"sessions={data['totals']['sessions_human']}"
    )
    return path


def _timing_markdown(timing: dict[str, Any]) -> list[str]:
    totals = timing.get("totals") or {}
    variants = timing.get("variants") or {}
    if not totals and not variants:
        return []
    lines = [
        "## Wall time",
        "",
        f"Compute (sum of variants): **{totals.get('compute_human', '—')}**. "
        f"Sessions: **{totals.get('sessions_human', '—')}** "
        f"({totals.get('n_sessions', 0)} session(s)).",
        "",
    ]
    note = timing.get("note")
    if note:
        lines.extend([str(note), ""])
    if variants:
        lines.extend(
            [
                "| variant | wall | train | eval |",
                "| ------- | ---: | ----: | ---: |",
            ]
        )
        for vid, entry in variants.items():
            wall = entry.get("elapsed_human") or format_duration(entry.get("elapsed_sec"))
            train = entry.get("train_elapsed_human") or format_duration(
                entry.get("train_elapsed_sec")
            )
            ev = entry.get("eval_elapsed_human") or format_duration(entry.get("eval_elapsed_sec"))
            lines.append(f"| `{vid}` | {wall} | {train} | {ev} |")
        lines.append("")
    return lines


def resolve_trained_weights(runs_dir: Path, run_name: str, result: dict[str, Any] | None = None) -> Path | None:
    if result:
        raw = result.get("best_weights")
        if raw:
            path = Path(str(raw))
            if path.is_file():
                return path
    path = trained_weights_path(runs_dir, run_name)
    return path if path.is_file() else None


def collect_existing_evals(
    runs_dir: Path,
    variant_id: str,
    targets: list[tuple[str, Path]],
) -> dict[str, str] | None:
    evals: dict[str, str] = {}
    for gt_name, _pack in targets:
        path = eval_metrics_path(runs_dir, variant_id, gt_name)
        if not path.is_file():
            return None
        evals[gt_name] = str(path)
    return evals


def variant_progress(
    *,
    runs_dir: Path,
    variant_id: str,
    run_name: str,
    eval_targets: list[tuple[str, Path]],
    skip_eval: bool,
) -> tuple[str, dict[str, Any] | None]:
    """Return (complete | train_done | missing, result_or_none).

    Train counts as done only after ``*_result.json`` was written (full train
    finished). A leftover ``best.pt`` from a killed run is treated as missing
    so that variant is retrained.
    """
    result = load_result_json(result_json_path(runs_dir, run_name))
    if result is None:
        return "missing", None
    weights = resolve_trained_weights(runs_dir, run_name, result)
    if weights is None:
        return "missing", result
    row = dict(result)
    row["best_weights"] = str(weights)
    if skip_eval:
        return "complete", row
    evals = collect_existing_evals(runs_dir, variant_id, eval_targets)
    if evals is None:
        return "train_done", row
    row["evals"] = evals
    return "complete", row


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _load_eval_bands(eval_json: Path) -> dict[str, Any]:
    if not eval_json.is_file():
        return {}
    metrics = json.loads(eval_json.read_text(encoding="utf-8"))
    return dict(metrics.get("bands") or {})


def _summary_table_row(vid: str, bands: dict[str, Any]) -> str:
    close = bands.get(EVAL_BAND_A) or bands.get("0-200m") or {}
    mid = bands.get(EVAL_BAND_B) or bands.get("200-400m") or {}
    return (
        "| "
        + " | ".join(
            [
                f"`{vid}`",
                _fmt_pct(close.get("map50")),
                _fmt_pct(mid.get("map50")),
                _fmt_pct(close.get("map50_95")),
                _fmt_pct(mid.get("map50_95")),
                _fmt_pct(close.get("detection_rate")),
                _fmt_pct(mid.get("detection_rate")),
            ]
        )
        + " |"
    )


def write_round_summary(
    runs_dir: Path,
    results: list[dict[str, Any]],
    *,
    title: str,
) -> Path:
    """Write comparison tables (one per eval pack) plus a combined JSON dump."""
    runs_dir = runs_dir if runs_dir.is_absolute() else PROJECT_ROOT / runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    a_lbl = BAND_LABELS[EVAL_BAND_A]
    b_lbl = BAND_LABELS[EVAL_BAND_B]
    header = (
        f"| variant | mAP@0.5 {a_lbl} | mAP@0.5 {b_lbl} | "
        f"mAP50-95 {a_lbl} | mAP50-95 {b_lbl} | det {a_lbl} | det {b_lbl} |"
    )
    sep = "| ------- | -------------: | -------------: | ---------: | ---------: | --------: | --------: |"

    eval_names: list[str] = []
    for row in results:
        evals = dict(row.get("evals") or {})
        if not evals and row.get("eval_metrics"):
            evals = {"manual": str(row["eval_metrics"])}
        for name in evals:
            if name not in eval_names:
                eval_names.append(name)
    if not eval_names:
        eval_names = ["manual"]

    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]
    payload: list[dict[str, Any]] = []
    for gt_name in eval_names:
        lines.extend([f"## Eval `{gt_name}`", "", header, sep])
        for row in results:
            vid = str(
                row.get("dataset_variant")
                or row.get("model_variant")
                or row.get("run_name")
                or ""
            )
            evals = dict(row.get("evals") or {})
            if not evals and row.get("eval_metrics"):
                evals = {"manual": str(row["eval_metrics"])}
            bands = _load_eval_bands(Path(str(evals.get(gt_name) or "")))
            lines.append(_summary_table_row(vid, bands))
        lines.append("")

    for row in results:
        vid = str(
            row.get("dataset_variant") or row.get("model_variant") or row.get("run_name") or ""
        )
        evals = dict(row.get("evals") or {})
        by_gt: dict[str, Any] = {}
        for gt_name, path in evals.items():
            by_gt[gt_name] = {"path": path, "bands": _load_eval_bands(Path(path))}
        payload.append(
            {
                "id": vid,
                "best_weights": row.get("best_weights"),
                "evals": by_gt,
                "result": row,
            }
        )

    lines.extend(_timing_markdown(load_round_timing(runs_dir)))

    md_path = runs_dir / "summary.md"
    json_path = runs_dir / "summary.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Round summary → {md_path}")
    return md_path


def run_train_job(
    *,
    yaml_path: Path,
    model: str,
    imgsz: int,
    epochs: int,
    warmup_epochs: int,
    freeze: int,
    lr0: float,
    patience: int,
    train_augmentation: str,
    runs_dir: Path,
    run_name: str,
    deliverable_name: str,
    batch: int | None = None,
    dry_run: bool = False,
    extra_plan: dict[str, Any] | None = None,
    pretrained: bool = True,
    staged: bool = True,
    train_backbone: bool = True,
) -> dict[str, Any]:
    runs_dir = runs_dir if runs_dir.is_absolute() else PROJECT_ROOT / runs_dir
    batch_n = batch if batch is not None else default_batch_size(imgsz)
    workers = default_workers()
    aug = aug_preset(train_augmentation)

    plan: dict[str, Any] = {
        "dataset_yaml": str(yaml_path),
        "model": model,
        "imgsz": imgsz,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "freeze": freeze,
        "lr0": lr0,
        "patience": patience,
        "batch": batch_n,
        "workers": workers,
        "device": DEVICE,
        "train_augmentation": train_augmentation,
        "augmentation": aug,
        "run_name": run_name,
        "runs_dir": str(runs_dir),
        "stage1_run_name": f"{run_name}_stage1",
        "stage2_run_name": run_name,
        "pretrained": pretrained,
        "staged": staged,
        "train_backbone": train_backbone,
        "note": (
            "head-only; backbone frozen (Group D ablation)"
            if not train_backbone
            else (
                "single-stage, backbone trainable"
                if not staged
                else "Stage 2 unfreezes the backbone (freeze=0)"
            )
        ),
    }
    if extra_plan:
        plan = {**extra_plan, **plan}
    print(json.dumps(plan, indent=2))
    if dry_run:
        print("[dry-run] skip train")
        return plan

    import training.train as train_mod

    prev_runs = train_mod.RUNS_DIR
    prev_model = train_mod.MODEL_NAME
    train_mod.RUNS_DIR = runs_dir
    train_mod.MODEL_NAME = model
    runs_dir.mkdir(parents=True, exist_ok=True)
    train_started = utcnow()
    train_t0 = time.perf_counter()
    try:
        weights = train_model(
            yaml_path,
            imgsz=imgsz,
            warmup_epochs=warmup_epochs,
            epochs=epochs,
            batch=batch_n,
            workers=workers,
            cache="disk" if DEVICE == "mps" else "false",
            freeze=freeze,
            lr0=lr0,
            patience=patience,
            stage1_run_name=f"{run_name}_stage1",
            stage2_run_name=run_name,
            deliverable_name=deliverable_name,
            write_checkpoint=False,
            aug=aug,
            pretrained=pretrained,
            staged=staged,
            train_backbone=train_backbone,
        )
    finally:
        train_mod.RUNS_DIR = prev_runs
        train_mod.MODEL_NAME = prev_model
        release_torch_memory()

    train_stamp = timing_stamp(started_at=train_started, t0=train_t0)
    result = {
        **plan,
        "best_weights": str(weights),
        "train_started_at": train_stamp["started_at"],
        "train_finished_at": train_stamp["finished_at"],
        "train_elapsed_sec": train_stamp["elapsed_sec"],
        "train_elapsed_human": train_stamp["elapsed_human"],
        "finished_at": train_stamp["finished_at"],
    }
    out_json = runs_dir / f"{run_name}_result.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Result → {out_json}")
    return result
