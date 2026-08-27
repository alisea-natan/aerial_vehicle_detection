"""Export, tile bench, parity vs locked .pt, eval reuse, markdown report."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2

from common.config import load_clip_tile_config
from optimisation.common import (
    artifact_bytes,
    list_images,
    project_path,
    train_image_dir,
)
from training.evaluate import (
    EVAL_BAND_A,
    EVAL_BAND_B,
    EVAL_CONF_THRESHOLD,
    Box,
    box_iou,
    evaluate_prepared_pack,
    predict_prepared_image,
)
from training.model_load import load_ultralytics_model, predict_device
from training.train import release_torch_memory

LOCKED_MEAN = 0.877


def make_step_timelog(t0: float) -> dict[str, Any]:
    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "elapsed_sec": elapsed,
        "elapsed_human": (
            f"{int(elapsed // 60)}m {elapsed % 60:.1f}s"
            if elapsed >= 60
            else f"{elapsed:.1f}s"
        ),
    }


def export_artifact(
    weights: Path,
    *,
    fmt: str,
    imgsz: int,
    dest_dir: Path,
    half: bool = False,
    int8: bool = False,
    calib_yaml: Path | None = None,
    calib_split: str | None = None,
) -> dict[str, Any]:
    """Ultralytics export. On failure return export_ok=False (then back to PoC)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    kw: dict[str, Any] = {"format": fmt, "imgsz": imgsz, "exist_ok": True}
    if fmt == "onnx":
        kw["simplify"] = True
    if half:
        kw["half"] = True
    if int8:
        kw["int8"] = True
        if calib_yaml is None:
            raise SystemExit("INT8 export needs calib_yaml (train pack data.yaml)")
        kw["data"] = str(calib_yaml)
        if calib_split:
            kw["split"] = calib_split
    t0 = time.perf_counter()
    try:
        model = load_ultralytics_model(weights)
        raw = model.export(**kw)
    except Exception as exc:
        return {
            "export_ok": False,
            "format": fmt,
            "half": half,
            "int8": int8,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "hint": (
                "Unsupported ops / dynamic shapes / opset. "
                "Stop INT8 and prune. Go back to PoC.md / EVALUATION.md "
                "and lock an exportable family, then return here."
            ),
        }
    src = Path(str(raw))
    if not src.exists():
        return {
            "export_ok": False,
            "format": fmt,
            "error": f"export returned missing path: {src}",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "hint": "Export claimed success but wrote nothing. Treat as fail — back to PoC.",
        }
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        dest = dest_dir / src.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
    return {
        "export_ok": True,
        "format": fmt,
        "half": half,
        "int8": int8,
        "path": str(dest),
        "mb": round(artifact_bytes(dest) / (1024 * 1024), 3),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def _sync(device: str) -> None:
    if device != "mps":
        return
    import torch

    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _rss_mb() -> float | None:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS: bytes; Linux: kilobytes
        if rss > 10**9:
            return round(rss / (1024 * 1024), 1)
        return round(rss / 1024, 1)
    except Exception:
        return None


def bench_tiles(
    weights: Path,
    images: list[Path],
    *,
    imgsz: int,
    conf: float,
    warmup: int,
    n: int,
) -> dict[str, Any]:
    """ms / tile, batch 1. Pack images are already tiles — not full frames."""
    if not images:
        raise SystemExit("No tile images for bench")
    model = load_ultralytics_model(weights)
    dev = predict_device(str(weights))
    cycle = images * (1 + (warmup + n) // max(1, len(images)))
    times: list[float] = []
    cold_ms: float | None = None
    for i, image_path in enumerate(cycle[: warmup + n]):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        predict_prepared_image(
            model,
            image_path,
            img_w=w,
            img_h=h,
            predict_imgsz=imgsz,
            conf=conf,
            dev=dev,
            predict_kw={"imgsz": imgsz},
        )
        _sync(dev)
        ms = (time.perf_counter() - t0) * 1000.0
        if cold_ms is None:
            cold_ms = ms
        if i >= warmup:
            times.append(ms)
    del model
    release_torch_memory()
    times.sort()
    if not times:
        raise SystemExit("Bench produced no timings")

    def pct(p: float) -> float:
        k = min(len(times) - 1, max(0, int(round((p / 100.0) * (len(times) - 1)))))
        return times[k]

    p50 = pct(50)
    p95 = pct(95)
    return {
        "device": dev,
        "n": len(times),
        "warmup": warmup,
        "cold_ms": round(cold_ms or 0.0, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "mean_ms": round(sum(times) / len(times), 2),
        "tile_fps": round(1000.0 / p50, 2) if p50 > 0 else None,
        "rss_mb": _rss_mb(),
        "unit": "ms per tile (batch 1, pack image, not full frame)",
    }


def _xyxy(box: Box) -> list[float]:
    return list(box.xyxy)


def _predict_boxes(model, image_path: Path, imgsz: int, conf: float, dev: str) -> list[Box]:
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    return predict_prepared_image(
        model,
        image_path,
        img_w=w,
        img_h=h,
        predict_imgsz=imgsz,
        conf=conf,
        dev=dev,
        predict_kw={"imgsz": imgsz},
    )


def parity_vs_teacher(
    teacher_weights: Path,
    student_weights: Path,
    images: list[Path],
    *,
    imgsz: int,
    conf: float,
    n: int,
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """Does the export still fire the same boxes as locked FP32? (train tiles, not eval)."""
    sample = images[:n]
    if not sample:
        raise SystemExit("No train tiles for parity")
    teacher = load_ultralytics_model(teacher_weights)
    t_dev = predict_device(str(teacher_weights))
    student = load_ultralytics_model(student_weights)
    s_dev = predict_device(str(student_weights))
    matched = 0
    teacher_n = 0
    student_n = 0
    ious: list[float] = []
    conf_err: list[float] = []
    for path in sample:
        tb = _predict_boxes(teacher, path, imgsz, conf, t_dev)
        sb = _predict_boxes(student, path, imgsz, conf, s_dev)
        teacher_n += len(tb)
        student_n += len(sb)
        used = [False] * len(sb)
        for tbox in tb:
            best_j = -1
            best_iou = 0.0
            for j, sbox in enumerate(sb):
                if used[j]:
                    continue
                iou = box_iou(_xyxy(tbox), _xyxy(sbox))
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thresh:
                used[best_j] = True
                matched += 1
                ious.append(best_iou)
                conf_err.append(abs(tbox.confidence - sb[best_j].confidence))
    del teacher, student
    release_torch_memory()
    agreement = (matched / teacher_n) if teacher_n else None
    return {
        "n_images": len(sample),
        "teacher_boxes": teacher_n,
        "student_boxes": student_n,
        "matched": matched,
        "agreement_iou50": None if agreement is None else round(agreement, 4),
        "mean_iou_matched": round(sum(ious) / len(ious), 4) if ious else None,
        "conf_mae": round(sum(conf_err) / len(conf_err), 4) if conf_err else None,
        "delta_count": student_n - teacher_n,
    }


def run_eval(
    weights: Path,
    *,
    eval_pack: Path,
    output_dir: Path,
    imgsz: int,
) -> Path:
    evaluate_prepared_pack(
        eval_pack,
        weights=weights,
        gt_name="manual",
        conf_override=None,
        iou_thresh=0.5,
        clip_filter=None,
        output_dir=output_dir,
        tile_config=load_clip_tile_config(),
        predict_kw={"imgsz": imgsz},
        video_dir=None,
    )
    release_torch_memory()
    return output_dir / "eval_metrics.json"


def load_eval_bands(eval_json: Path) -> dict[str, Any]:
    if not eval_json.is_file():
        return {}
    return dict(json.loads(eval_json.read_text(encoding="utf-8")).get("bands") or {})


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def quality_from_bands(bands: dict[str, Any]) -> dict[str, Any]:
    a = bands.get(EVAL_BAND_A) or {}
    b = bands.get(EVAL_BAND_B) or {}
    ma, mb = a.get("map50"), b.get("map50")
    mean = None
    if ma is not None and mb is not None:
        mean = (float(ma) + float(mb)) / 2.0
    return {
        "map50_a": ma,
        "map50_b": mb,
        "map50_mean": mean,
        "det_a": a.get("detection_rate"),
        "det_b": b.get("detection_rate"),
        "precision_a": a.get("precision"),
        "precision_b": b.get("precision"),
        "fa_per_min_a": a.get("false_alarms_per_min"),
        "fa_per_min_b": b.get("false_alarms_per_min"),
        "ttff_a": a.get("time_to_first_detection_s"),
        "ttff_b": b.get("time_to_first_detection_s"),
        "tp_a": a.get("tp"),
        "fp_a": a.get("fp"),
        "fn_a": a.get("fn"),
        "tp_b": b.get("tp"),
        "fp_b": b.get("fp"),
        "fn_b": b.get("fn"),
    }


def locked_quality(locked: dict[str, Any]) -> dict[str, Any]:
    return {
        "map50_a": locked.get("map50_a"),
        "map50_b": locked.get("map50_b"),
        "map50_mean": locked.get("map50_mean", LOCKED_MEAN),
        "det_a": locked.get("det_a"),
        "det_b": locked.get("det_b"),
        "precision_a": locked.get("precision_a"),
        "precision_b": locked.get("precision_b"),
        "fa_per_min_a": locked.get("fa_per_min_a"),
        "fa_per_min_b": locked.get("fa_per_min_b"),
        "ttff_a": None,
        "ttff_b": None,
        "tp_a": None,
        "fp_a": None,
        "fn_a": None,
        "tp_b": None,
        "fp_b": None,
        "fn_b": None,
    }


def pass_gates(
    quality: dict[str, Any],
    *,
    locked_mean: float,
    winner_delta: float,
    min_det_b: float,
) -> dict[str, Any]:
    mean = quality.get("map50_mean")
    det_b = quality.get("det_b")
    mean_ok = mean is not None and float(mean) + 1e-12 >= locked_mean - winner_delta
    det_ok = det_b is not None and float(det_b) + 1e-12 >= min_det_b
    return {
        "mean_ok": mean_ok,
        "det_b_ok": det_ok,
        "pass": bool(mean_ok and det_ok),
        "bar_map50": round(locked_mean - winner_delta, 4),
        "bar_det_b": min_det_b,
    }


def score_cell(
    *,
    cell_id: str,
    weights: Path | None,
    teacher: Path,
    d: dict[str, Any],
    locked: dict[str, Any],
    runs_dir: Path,
    skip_eval: bool = False,
    use_locked_quality: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_pack = project_path(d["eval_pack"])
    train_pack = project_path(d["train_pack"])
    imgsz = int(d["imgsz"])
    conf = float(d.get("conf") or EVAL_CONF_THRESHOLD)
    bench_images = list_images(eval_pack / "images")
    parity_images = list_images(train_image_dir(train_pack))
    row: dict[str, Any] = {"id": cell_id, **(extra or {})}
    t_cell = time.perf_counter()
    timelog: dict[str, Any] = {}
    if extra and extra.get("elapsed_sec") is not None:
        timelog["export_sec"] = extra["elapsed_sec"]
    if weights is None or not (weights.exists()):
        row["error"] = "missing artifact"
        timelog["score_sec"] = round(time.perf_counter() - t_cell, 2)
        row["timelog"] = timelog
        return row
    row["weights"] = str(weights)
    row["mb"] = round(artifact_bytes(weights) / (1024 * 1024), 3)
    row["bench"] = bench_tiles(
        weights,
        bench_images,
        imgsz=imgsz,
        conf=conf,
        warmup=int(d["bench_warmup"]),
        n=int(d["bench_n"]),
    )
    if use_locked_quality:
        row["quality"] = locked_quality(locked)
        row["parity"] = {
            "agreement_iou50": 1.0,
            "conf_mae": 0.0,
            "note": "teacher (locked .pt)",
        }
    else:
        row["parity"] = parity_vs_teacher(
            teacher,
            weights,
            parity_images,
            imgsz=imgsz,
            conf=conf,
            n=int(d["parity_n"]),
        )
        if skip_eval:
            row["quality"] = {}
        else:
            eval_json = run_eval(
                weights,
                eval_pack=eval_pack,
                output_dir=runs_dir / cell_id / "eval_manual",
                imgsz=imgsz,
            )
            row["eval_metrics"] = str(eval_json)
            eval_payload = json.loads(eval_json.read_text(encoding="utf-8"))
            row["quality"] = quality_from_bands(load_eval_bands(eval_json))
            eval_timing = eval_payload.get("timing") or {}
            timelog["eval_sec"] = eval_timing.get("elapsed_sec")
            timelog["eval_human"] = eval_timing.get("elapsed_human")
            timelog["eval_clips"] = eval_timing.get("clips")
    row["gates"] = pass_gates(
        row.get("quality") or {},
        locked_mean=float(locked.get("map50_mean") or LOCKED_MEAN),
        winner_delta=float(d["winner_delta"]),
        min_det_b=float(d["min_det_b"]),
    )
    timelog["score_sec"] = round(time.perf_counter() - t_cell, 2)
    if extra and extra.get("timelog"):
        timelog.update({k: v for k, v in extra["timelog"].items() if v is not None})
    row["timelog"] = timelog
    return row


def write_summary(
    runs_dir: Path,
    title: str,
    rows: list[dict[str, Any]],
    *,
    step_timelog: dict[str, Any] | None = None,
    baseline_id: str | None = None,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / "summary.json"
    md_path = runs_dir / "summary.md"
    payload: dict[str, Any] = {"rows": rows}
    if step_timelog:
        payload["step_timelog"] = step_timelog
        tl_path = runs_dir / "timelog.json"
        tl_path.write_text(json.dumps(step_timelog, indent=2) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {title}",
        "",
        "Decision = mean A+B mAP@0.5 (find a car). Rest = how it behaves in operation.",
        "",
        "## Decision",
        "",
        "| cell | mean mAP@0.5 | A | B | pass? |",
        "| ---- | -----------: | -: | -: | ----- |",
    ]
    for row in rows:
        q = row.get("quality") or {}
        g = row.get("gates") or {}
        passed = "yes" if g.get("pass") else ("—" if not q else "no")
        lines.append(
            f"| `{row.get('id')}` | {_pct(q.get('map50_mean'))} | "
            f"{_pct(q.get('map50_a'))} | {_pct(q.get('map50_b'))} | {passed} |"
        )
    lines.extend(
        [
            "",
            "## Real-world (holdout @ conf 0.25)",
            "",
            "| cell | Det A / B | P A / B | FA/min A / B | TTFF A / B s | TP/FP/FN A | TP/FP/FN B |",
            "| ---- | --------- | ------- | ------------ | ------------ | ---------- | ---------- |",
        ]
    )
    for row in rows:
        q = row.get("quality") or {}
        lines.append(
            "| `{id}` | {da} / {db} | {pa} / {pb} | {fa} / {fb} | {ta} / {tb} | "
            "{tpa}/{fpa}/{fna} | {tpb}/{fpb}/{fnb} |".format(
                id=row.get("id"),
                da=_pct(q.get("det_a")),
                db=_pct(q.get("det_b")),
                pa=_pct(q.get("precision_a")),
                pb=_pct(q.get("precision_b")),
                fa=_num(q.get("fa_per_min_a"), 0),
                fb=_num(q.get("fa_per_min_b"), 0),
                ta=_num(q.get("ttff_a"), 2),
                tb=_num(q.get("ttff_b"), 2),
                tpa=q.get("tp_a") if q.get("tp_a") is not None else "—",
                fpa=q.get("fp_a") if q.get("fp_a") is not None else "—",
                fna=q.get("fn_a") if q.get("fn_a") is not None else "—",
                tpb=q.get("tp_b") if q.get("tp_b") is not None else "—",
                fpb=q.get("fp_b") if q.get("fp_b") is not None else "—",
                fnb=q.get("fn_b") if q.get("fn_b") is not None else "—",
            )
        )
    lines.extend(
        [
            "",
            "## Speed / size (this Mac, **ms per tile**, batch 1)",
            "",
            "| cell | device | cold ms | p50 ms | p95 ms | tile FPS | RSS MB | artifact MB |",
            "| ---- | ------ | ------: | -----: | -----: | -------: | -----: | ----------: |",
        ]
    )
    for row in rows:
        b = row.get("bench") or {}
        lines.append(
            f"| `{row.get('id')}` | {b.get('device', '—')} | {_num(b.get('cold_ms'))} | "
            f"{_num(b.get('p50_ms'))} | {_num(b.get('p95_ms'))} | {_num(b.get('tile_fps'))} | "
            f"{_num(b.get('rss_mb'))} | {_num(row.get('mb'), 2)} |"
        )
    lines.extend(
        [
            "",
            "## Parity vs locked FP32 (train tiles, not eval)",
            "",
            "| cell | agreement@0.5 | mean IoU | conf MAE | Δ count |",
            "| ---- | ------------: | -------: | -------: | ------: |",
        ]
    )
    for row in rows:
        p = row.get("parity") or {}
        lines.append(
            f"| `{row.get('id')}` | {_num(p.get('agreement_iou50'), 3)} | "
            f"{_num(p.get('mean_iou_matched'), 3)} | {_num(p.get('conf_mae'), 3)} | "
            f"{p.get('delta_count', '—')} |"
        )
    if baseline_id:
        baseline = next((r for r in rows if r.get("id") == baseline_id), None)
        if baseline:
            bq = baseline.get("quality") or {}
            bb = baseline.get("bench") or {}
            bp = baseline.get("parity") or {}
            b_mean = bq.get("map50_mean")
            b_p50 = (bb.get("p50_ms") if bb else None)
            b_mb = baseline.get("mb")
            b_agree = bp.get("agreement_iou50")
            lines.extend(
                [
                    "",
                    f"## vs `{baseline_id}` baseline (what quantisation changed)",
                    "",
                    "| cell | Δ mean mAP@0.5 | Δ p50 ms | Δ artifact MB | Δ parity agreement |",
                    "| ---- | -------------: | -------: | ------------: | -----------------: |",
                ]
            )
            for row in rows:
                if row.get("id") == baseline_id:
                    lines.append(
                        f"| `{row.get('id')}` | — (baseline) | — | — | — |"
                    )
                    continue
                q = row.get("quality") or {}
                bench = row.get("bench") or {}
                par = row.get("parity") or {}
                mean = q.get("map50_mean")
                p50 = bench.get("p50_ms")
                mb = row.get("mb")
                agree = par.get("agreement_iou50")
                d_mean = (
                    f"{(mean - b_mean) * 100:+.1f}pp"
                    if mean is not None and b_mean is not None
                    else "—"
                )
                d_p50 = (
                    f"{p50 - b_p50:+.0f}"
                    if p50 is not None and b_p50 is not None
                    else "—"
                )
                d_mb = (
                    f"{mb - b_mb:+.1f}"
                    if mb is not None and b_mb is not None
                    else "—"
                )
                d_agree = (
                    f"{agree - b_agree:+.3f}"
                    if agree is not None and b_agree is not None
                    else "—"
                )
                lines.append(
                    f"| `{row.get('id')}` | {d_mean} | {d_p50} | {d_mb} | {d_agree} |"
                )
    failed = [r for r in rows if r.get("export_ok") is False]
    if failed:
        lines.extend(["", "## Export failed — stop", ""])
        for row in failed:
            lines.append(f"- `{row.get('id')}`: {row.get('error')}")
            if row.get("hint"):
                lines.append(f"  {row['hint']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary → {md_path}")
    return md_path
