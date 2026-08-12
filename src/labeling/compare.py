#!/usr/bin/env python3
"""Compare CVAT manual labels (labelling/cvat/label_man/) with labels/.

  python src/labeling/compare.py
  python src/labeling/compare.py --iou 0.5 --clip 266987
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

def _ensure_src_on_path() -> None:
    """Allow `python src/<pkg>/….py` without PYTHONPATH."""
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
import statistics
from dataclasses import dataclass, field
from pathlib import Path


from common.config import PROJECT_ROOT, LABELS_DIR, build_split_map

LABEL_MAN_DIR = PROJECT_ROOT / "labelling" / "cvat" / "label_man"
DEFAULT_OUT = PROJECT_ROOT / "debug" / "compare_labels_report.txt"
DEFAULT_IOU = 0.5
MANUAL_SUBDIRS = ("obj_Train_data", "obj_Test_data")


@dataclass
class FrameStats:
    stem: str
    n_manual: int
    n_auto: int
    match_ious: list[float]  # one IoU per matched pair (greedy, any IoU > 0)
    tp: int
    fp: int
    fn: int

    @property
    def n_matched(self) -> int:
        return len(self.match_ious)

    @property
    def mean_iou(self) -> float | None:
        return statistics.mean(self.match_ious) if self.match_ious else None


@dataclass
class ClipStats:
    clip: str
    split: str
    manual_dir: Path
    auto_dir: Path
    iou_thresh: float
    frames: list[FrameStats] = field(default_factory=list)
    only_manual: list[str] = field(default_factory=list)
    only_auto: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def total_manual(self) -> int:
        return sum(f.n_manual for f in self.frames)

    @property
    def total_auto(self) -> int:
        return sum(f.n_auto for f in self.frames)

    @property
    def total_tp(self) -> int:
        return sum(f.tp for f in self.frames)

    @property
    def total_fp(self) -> int:
        return sum(f.fp for f in self.frames)

    @property
    def total_fn(self) -> int:
        return sum(f.fn for f in self.frames)

    @property
    def all_match_ious(self) -> list[float]:
        out: list[float] = []
        for f in self.frames:
            out.extend(f.match_ious)
        return out

    def precision(self) -> float | None:
        denom = self.total_tp + self.total_fp
        return self.total_tp / denom if denom else None

    def recall(self) -> float | None:
        denom = self.total_tp + self.total_fn
        return self.total_tp / denom if denom else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare label_man (CVAT) vs labels/ (autolabel) — IoU + bbox counts.",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Compare one clip only (default: every folder under label_man/).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"IoU threshold for TP match (default {DEFAULT_IOU}).",
    )
    parser.add_argument(
        "--man-dir",
        default=str(LABEL_MAN_DIR),
        help="Root of CVAT exports (default label_man/).",
    )
    parser.add_argument(
        "--labels-dir",
        default=str(LABELS_DIR),
        help="Root of autolabels (default labels/).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Report .txt path (default {DEFAULT_OUT}).",
    )
    return parser.parse_args()


def yolo_line_to_xyxy(parts: list[str]) -> list[float] | None:
    """Normalized YOLO → [x1, y1, x2, y2] in the same normalized space."""
    if len(parts) < 5:
        return None
    try:
        xc, yc, bw, bh = map(float, parts[1:5])
    except ValueError:
        return None
    if bw <= 0 or bh <= 0:
        return None
    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_yolo_boxes(path: Path) -> list[list[float]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        box = yolo_line_to_xyxy(parts)
        if box is not None:
            boxes.append(box)
    return boxes


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    iou_thresh: float,
) -> tuple[list[float], int, int, int]:
    """Greedy best-IoU matching. Returns (pair_ious, tp, fp, fn).

    pair_ious: IoU of every greedy pair with IoU > 0 (for mean IoU).
    TP = pairs with IoU ≥ thresh; FN = |GT| − TP; FP = |pred| − TP.
    """
    gt_used = [False] * len(gt_boxes)
    pred_used = [False] * len(pred_boxes)
    pair_ious: list[float] = []

    pairs: list[tuple[float, int, int]] = []
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou = box_iou(gt, pred)
            if iou > 0:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    for iou, i, j in pairs:
        if gt_used[i] or pred_used[j]:
            continue
        gt_used[i] = True
        pred_used[j] = True
        pair_ious.append(iou)

    tp = sum(1 for iou in pair_ious if iou >= iou_thresh)
    fn = len(gt_boxes) - tp
    fp = len(pred_boxes) - tp
    return pair_ious, tp, fp, fn


def find_manual_label_dir(clip_dir: Path) -> Path | None:
    for name in MANUAL_SUBDIRS:
        candidate = clip_dir / name
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def find_auto_label_dir(clip: str, labels_root: Path, split_map: dict[str, str]) -> Path | None:
    split = split_map.get(clip)
    if split:
        path = labels_root / split / clip
        if path.is_dir():
            return path
    for split in ("train", "eval"):
        path = labels_root / split / clip
        if path.is_dir():
            return path
    return None


def compare_clip(
    clip: str,
    man_root: Path,
    labels_root: Path,
    split_map: dict[str, str],
    iou_thresh: float,
) -> ClipStats | None:
    clip_dir = man_root / clip
    manual_dir = find_manual_label_dir(clip_dir)
    if manual_dir is None:
        print(f"[skip] {clip}: no obj_Train_data / obj_Test_data under {clip_dir}")
        return None

    auto_dir = find_auto_label_dir(clip, labels_root, split_map)
    if auto_dir is None:
        print(f"[skip] {clip}: no autolabels under {labels_root}/{{train,eval}}/{clip}")
        return None

    split = split_map.get(clip, auto_dir.parent.name)
    man_stems = {p.stem for p in manual_dir.glob("*.txt")}
    auto_stems = {p.stem for p in auto_dir.glob("*.txt")}
    common = sorted(man_stems & auto_stems)

    stats = ClipStats(
        clip=clip,
        split=split,
        manual_dir=manual_dir,
        auto_dir=auto_dir,
        iou_thresh=iou_thresh,
        only_manual=sorted(man_stems - auto_stems),
        only_auto=sorted(auto_stems - man_stems),
    )

    for stem in common:
        gt = parse_yolo_boxes(manual_dir / f"{stem}.txt")
        pred = parse_yolo_boxes(auto_dir / f"{stem}.txt")
        ious, tp, fp, fn = match_boxes(gt, pred, iou_thresh)
        stats.frames.append(
            FrameStats(
                stem=stem,
                n_manual=len(gt),
                n_auto=len(pred),
                match_ious=ious,
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )
    return stats


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def fmt_float(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def format_clip_report(stats: ClipStats) -> list[str]:
    ious = stats.all_match_ious
    mean_iou = statistics.mean(ious) if ious else None
    median_iou = statistics.median(ious) if ious else None
    strong = [u for u in ious if u >= stats.iou_thresh]
    mean_strong = statistics.mean(strong) if strong else None

    lines = [
        "=" * 72,
        f"CLIP: {stats.clip}  (split={stats.split})",
        f"  manual: {stats.manual_dir}",
        f"  auto:   {stats.auto_dir}",
        f"  IoU threshold (TP): {stats.iou_thresh}",
        "-" * 72,
        "SUMMARY",
        f"  frames compared:     {stats.n_frames}",
        f"  frames only manual:  {len(stats.only_manual)}",
        f"  frames only auto:    {len(stats.only_auto)}",
        f"  total bboxes manual: {stats.total_manual}",
        f"  total bboxes auto:   {stats.total_auto}",
        f"  delta (auto-manual): {stats.total_auto - stats.total_manual}",
        f"  matched pairs (IoU>0): {len(ious)}",
        f"  mean IoU (matched):  {fmt_float(mean_iou)}",
        f"  median IoU (matched):{fmt_float(median_iou)}",
        f"  mean IoU (TP≥thresh):{fmt_float(mean_strong)}",
        f"  TP / FP / FN:        {stats.total_tp} / {stats.total_fp} / {stats.total_fn}",
        f"  precision (auto):    {fmt_pct(stats.precision())}",
        f"  recall (vs manual):  {fmt_pct(stats.recall())}",
        "-" * 72,
        "PER-FRAME  (stem | n_man | n_auto | delta | matched | mean_iou | TP FP FN)",
    ]

    for f in stats.frames:
        delta = f.n_auto - f.n_manual
        lines.append(
            f"  {f.stem} | {f.n_manual:4d} | {f.n_auto:4d} | {delta:+5d} | "
            f"{f.n_matched:4d} | {fmt_float(f.mean_iou):>7} | "
            f"{f.tp:3d} {f.fp:3d} {f.fn:3d}"
        )

    if stats.only_manual:
        preview = ", ".join(stats.only_manual[:10])
        more = f" … (+{len(stats.only_manual) - 10})" if len(stats.only_manual) > 10 else ""
        lines.append(f"  [only in manual] {preview}{more}")
    if stats.only_auto:
        preview = ", ".join(stats.only_auto[:10])
        more = f" … (+{len(stats.only_auto) - 10})" if len(stats.only_auto) > 10 else ""
        lines.append(f"  [only in auto]   {preview}{more}")

    lines.append("")
    return lines


def format_report(all_stats: list[ClipStats], iou_thresh: float) -> str:
    lines = [
        "Manual (label_man / CVAT) vs Autolabel (labels/) comparison",
        f"IoU match threshold: {iou_thresh}",
        f"Clips: {len(all_stats)}",
        "",
    ]
    for stats in all_stats:
        lines.extend(format_clip_report(stats))

    if len(all_stats) > 1:
        lines.append("=" * 72)
        lines.append("OVERALL (all clips)")
        tot_man = sum(s.total_manual for s in all_stats)
        tot_auto = sum(s.total_auto for s in all_stats)
        tot_tp = sum(s.total_tp for s in all_stats)
        tot_fp = sum(s.total_fp for s in all_stats)
        tot_fn = sum(s.total_fn for s in all_stats)
        all_ious: list[float] = []
        for s in all_stats:
            all_ious.extend(s.all_match_ious)
        mean_iou = statistics.mean(all_ious) if all_ious else None
        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else None
        rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else None
        lines.extend(
            [
                f"  frames:              {sum(s.n_frames for s in all_stats)}",
                f"  total bboxes manual: {tot_man}",
                f"  total bboxes auto:   {tot_auto}",
                f"  mean IoU (matched):  {fmt_float(mean_iou)}",
                f"  TP / FP / FN:        {tot_tp} / {tot_fp} / {tot_fn}",
                f"  precision:           {fmt_pct(prec)}",
                f"  recall:              {fmt_pct(rec)}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    man_root = Path(args.man_dir)
    labels_root = Path(args.labels_dir)
    out_path = Path(args.out)

    if not man_root.is_dir():
        raise SystemExit(f"Manual label root not found: {man_root}")

    split_map = build_split_map()
    clip_names = sorted(
        d.name
        for d in man_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if args.clip:
        clip_filter = Path(args.clip).stem
        clip_names = [c for c in clip_names if c == clip_filter]
        if not clip_names:
            raise SystemExit(f"Clip {clip_filter!r} not found under {man_root}")

    all_stats: list[ClipStats] = []
    for clip in clip_names:
        stats = compare_clip(clip, man_root, labels_root, split_map, args.iou)
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        raise SystemExit("No clips compared — check label_man/ and labels/ paths.")

    report = format_report(all_stats, args.iou)
    print(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
