#!/usr/bin/env python3
"""Compare Android bench JSON (tile_log detections) to eval clip labels.

Aligns on video decode index: bench ``video_frame_index`` (0-based) ↔ label stem
``{index+1:06d}`` under ``labels/eval/{clip}/`` (same as ``extract_frames.py``).

  python src/optimisation/android/compare_bench_to_labels.py \\
    --bench src/optimisation/android/logs/vehicle_bench_latest.json

Requires bench JSON exported after APK update (``detections[]`` per ``tile_log`` entry).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

from pathlib import Path

from common.config import FRAMES_DIR, LABELS_DIR, PROJECT_ROOT, load_clip_tile_config, resolve_clip_tile_config
from training.evaluate import (
    BAND_LABELS,
    BandStats,
    Box,
    CLIP_EVAL_BAND,
    DEFAULT_BAND_CLIPS,
    EVAL_BAND_A,
    EVAL_BAND_B,
    IOU_MATCH,
    accumulate_frame_metrics,
    format_metric,
    load_metadata,
    nms_boxes,
    parse_yolo_label_file,
)
from training.train import _frame_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", type=Path, required=True, help="Shared vehicle_bench_*.json from APK.")
    p.add_argument(
        "--clip",
        default=None,
        help="Eval clip stem (default: infer from bench ``video`` field).",
    )
    p.add_argument(
        "--labels-root",
        type=Path,
        default=LABELS_DIR,
        help=f"Label tree with eval/{{clip}}/ (default: {LABELS_DIR.relative_to(PROJECT_ROOT)}).",
    )
    p.add_argument(
        "--match-iou",
        type=float,
        default=IOU_MATCH,
        help=f"GT match IoU (default {IOU_MATCH}, same as evaluate.py).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Score threshold for preds (default: bench JSON ``conf``).",
    )
    p.add_argument(
        "--merge-iou",
        type=float,
        default=None,
        help="Tile merge NMS IoU (default: bench JSON ``iou`` or 0.45).",
    )
    p.add_argument(
        "--no-frame-step",
        action="store_true",
        help="Score every labeled frame, not only frame_step subsample.",
    )
    p.add_argument("--json-out", type=Path, default=None, help="Write eval-style metrics JSON.")
    return p.parse_args()


def resolve_clip(video_name: str, clip_override: str | None) -> str:
    if clip_override:
        return clip_override
    stem = Path(video_name).stem
    if stem in CLIP_EVAL_BAND:
        return stem
    for clip in DEFAULT_BAND_CLIPS.values():
        if stem == clip or stem.startswith(clip.split("_")[0]):
            return clip
    return stem


def labeled_eval_frames(
    clip: str,
    labels_root: Path,
    *,
    frame_step_only: bool,
) -> dict[int, Path]:
    """Map 1-based extracted frame index → label path."""
    label_dir = labels_root / "eval" / clip
    if not label_dir.is_dir():
        raise SystemExit(f"No labels at {label_dir}")

    step = 1
    if frame_step_only:
        cfg = resolve_clip_tile_config(clip, load_clip_tile_config())
        raw = cfg.get("frame_step")
        step = max(1, int(raw)) if raw is not None else 1

    out: dict[int, Path] = {}
    for label_path in sorted(label_dir.glob("*.txt")):
        idx = _frame_index(label_path.stem)
        if idx is None:
            continue
        if frame_step_only and step > 1 and (idx - 1) % step != 0:
            continue
        image_path = FRAMES_DIR / clip / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        out[idx] = label_path
    return out


def preds_by_video_frame(
    tile_log: list[dict],
    *,
    conf: float,
    merge_iou: float,
) -> tuple[dict[int, list[Box]], set[int]]:
    """Merge per-tile detections into full-frame preds keyed by video_frame_index."""
    raw: dict[int, list[Box]] = defaultdict(list)
    scored: set[int] = set()

    for entry in tile_log:
        if entry.get("warmup"):
            continue
        vf = int(entry["video_frame_index"])
        scored.add(vf)
        for det in entry.get("detections") or []:
            if not det.get("above_conf", float(det.get("score", 0)) >= conf):
                continue
            raw[vf].append(
                Box(
                    xyxy=[
                        float(det["x1"]),
                        float(det["y1"]),
                        float(det["x2"]),
                        float(det["y2"]),
                    ],
                    confidence=float(det["score"]),
                ),
            )

    merged = {vf: nms_boxes(boxes, merge_iou) for vf, boxes in raw.items()}
    return merged, scored


def _preds_in_frame(preds: dict[int, list[Box]], img_w: int, img_h: int) -> int:
    """Count preds with a corner inside [0,w]×[0,h] — catches normalized-coord decode bugs."""
    n = 0
    for boxes in preds.values():
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy
            if x2 > 0 and x1 < img_w and y2 > 0 and y1 < img_h:
                n += 1
    return n


def compare(
    bench: dict,
    *,
    clip: str,
    labels_root: Path,
    match_iou: float,
    conf: float,
    merge_iou: float,
    frame_step_only: bool,
) -> dict:
    tile_log = bench.get("tile_log") or []
    if not tile_log:
        raise SystemExit("Bench JSON has empty tile_log.")
    if not any("detections" in entry for entry in tile_log):
        raise SystemExit(
            "Bench JSON has no per-tile detections[] — re-run bench with updated APK and Share JSON.",
        )

    band = CLIP_EVAL_BAND.get(clip)
    if band is None:
        raise SystemExit(f"Unknown eval clip {clip!r} — pass --clip explicitly.")

    meta = load_metadata(FRAMES_DIR / clip)
    img_w, img_h = int(meta["width"]), int(meta["height"])
    fps = float(meta.get("fps", 30.0))

    labeled = labeled_eval_frames(clip, labels_root, frame_step_only=frame_step_only)
    if not labeled:
        raise SystemExit(f"No labeled eval frames for {clip!r} under {labels_root / 'eval'}")

    preds, scored_vfs = preds_by_video_frame(tile_log, conf=conf, merge_iou=merge_iou)

    total_preds = sum(len(v) for v in preds.values())
    in_frame = _preds_in_frame(preds, img_w, img_h)
    if total_preds > 0 and in_frame == 0:
        raise SystemExit(
            "All logged boxes fall outside the frame — likely stale bench JSON from before "
            "the TFLite xywh normalize fix. Rebuild APK, re-run bench, Share JSON again.",
        )

    stats = BandStats()
    frame_rows: list[dict] = []
    n_scored = 0

    for frame_idx in sorted(labeled):
        vf = frame_idx - 1
        if vf not in scored_vfs:
            continue
        label_path = labeled[frame_idx]
        gt = parse_yolo_label_file(label_path, img_w, img_h)
        pred = preds.get(vf, [])
        if not gt and not pred:
            continue

        stats.n_frames += 1
        stats.fps = fps
        tp, fp, fn = accumulate_frame_metrics(stats, gt, pred, match_iou)
        n_scored += 1
        if tp > 0 and stats.first_tp_frame is None:
            stats.first_tp_frame = frame_idx

        frame_rows.append(
            {
                "video_frame_index": vf,
                "label_frame_index": frame_idx,
                "label_stem": label_path.stem,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "n_gt": len(gt),
                "n_pred": len(pred),
            },
        )

    if n_scored == 0:
        raise SystemExit(
            f"No overlap between bench frames and labeled eval frames for {clip!r}. "
            "Check video matches clip and frame_step settings.",
        )

    other_band = EVAL_BAND_B if band == EVAL_BAND_A else EVAL_BAND_A
    band_stats = {band: stats, other_band: BandStats()}

    return {
        "source": "android_bench_json",
        "bench_json": str(bench.get("timestamp", "")),
        "clip": clip,
        "eval_band": band,
        "video": bench.get("video"),
        "device": bench.get("device"),
        "delegate": bench.get("delegate"),
        "model": bench.get("model"),
        "conf": conf,
        "match_iou": match_iou,
        "merge_iou": merge_iou,
        "frame_step_only": frame_step_only,
        "bbox_coord_space": bench.get("bbox_coord_space", "video_frame_xyxy"),
        "labeled_frames": len(labeled),
        "scored_frames": n_scored,
        "bands": {
            band: {
                "tp": stats.tp,
                "fp": stats.fp,
                "fn": stats.fn,
                "n_frames": stats.n_frames,
                "detection_rate": stats.detection_rate(),
                "precision": stats.precision(),
                "false_alarms_per_min": stats.false_alarms_per_min(),
                "time_to_first_detection_s": stats.time_to_first_detection_s(),
                "map50": stats.map50(),
                "map50_95": stats.map50_95(),
            },
        },
        "frames": frame_rows,
    }


def print_summary(result: dict) -> None:
    band = result["eval_band"]
    stats = result["bands"][band]
    a_lbl = BAND_LABELS.get(band, band)
    b_lbl = BAND_LABELS.get(EVAL_BAND_B if band == EVAL_BAND_A else EVAL_BAND_A, "-")

    print(f"Clip: {result['clip']} ({a_lbl})")
    print(f"Video: {result.get('video')}  device={result.get('device')}  delegate={result.get('delegate')}")
    print(
        f"Scored {result['scored_frames']}/{result['labeled_frames']} labeled frames "
        f"(frame_step_only={result['frame_step_only']})",
    )
    print(f"conf={result['conf']}  match_iou={result['match_iou']}  merge_iou={result['merge_iou']}")
    print()
    print(f"{'Metric':<28} {a_lbl:>12} {b_lbl:>12}")
    print("-" * 54)
    rows = [
        ("Detection rate", stats["detection_rate"], None),
        ("Precision", stats["precision"], None),
        ("False alarms/min", stats["false_alarms_per_min"], False),
        ("Time to first det (s)", stats["time_to_first_detection_s"], False),
        ("mAP@0.5", stats["map50"], None),
        ("mAP@0.5:0.95", stats["map50_95"], None),
    ]
    for label, value, pct in rows:
        use_pct = pct if pct is not None else True
        print(f"{label:<28} {format_metric(value, pct=use_pct):>12} {'-':>12}")
    print()
    print(f"TP={stats['tp']} FP={stats['fp']} FN={stats['fn']}  frames={stats['n_frames']}")


def main() -> None:
    args = parse_args()
    bench_path = args.bench if args.bench.is_absolute() else PROJECT_ROOT / args.bench
    if not bench_path.is_file():
        raise SystemExit(f"Missing bench JSON: {bench_path}")

    labels_root = args.labels_root if args.labels_root.is_absolute() else PROJECT_ROOT / args.labels_root
    bench = json.loads(bench_path.read_text(encoding="utf-8"))

    conf = float(args.conf if args.conf is not None else bench.get("conf", 0.25))
    merge_iou = float(args.merge_iou if args.merge_iou is not None else bench.get("iou", 0.45))
    clip = resolve_clip(str(bench.get("video", "")), args.clip)

    result = compare(
        bench,
        clip=clip,
        labels_root=labels_root,
        match_iou=float(args.match_iou),
        conf=conf,
        merge_iou=merge_iou,
        frame_step_only=not args.no_frame_step,
    )
    result["bench_path"] = str(bench_path)

    print_summary(result)

    if args.json_out:
        out = args.json_out if args.json_out.is_absolute() else PROJECT_ROOT / args.json_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
