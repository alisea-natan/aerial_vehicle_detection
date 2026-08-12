#!/usr/bin/env python3
"""Compare CVAT labels (labels/, from cvat_pull --sync-labels) vs YOLO-World autolabel.

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


from common.config import PROJECT_ROOT, LABELS_DIR, SPLITS, build_split_map

# CVAT extract (source of truth for human GT). Autolabel never writes here.
CVAT_LABELS_DIR = LABELS_DIR
AUTOLABEL_LABELS_DIR = PROJECT_ROOT / "outputs" / "autolabel" / "labels"
DEFAULT_OUT = PROJECT_ROOT / "debug" / "compare_autolabel_vs_cvat.txt"
DEFAULT_IOU = 0.5


@dataclass
class FrameStats:
    stem: str
    n_manual: int
    n_auto: int
    match_ious: list[float]
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
        description=(
            "Compare CVAT labels (labels/, from cvat_pull) vs "
            "YOLO-World autolabel (outputs/autolabel/labels/)."
        ),
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Compare one clip only (default: every clip under labels/).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"IoU threshold for TP match (default {DEFAULT_IOU}).",
    )
    parser.add_argument(
        "--cvat-dir",
        default=str(CVAT_LABELS_DIR),
        help="CVAT extract root (default labels/).",
    )
    parser.add_argument(
        "--autolabel-dir",
        default=str(AUTOLABEL_LABELS_DIR),
        help="Autolabel root (default outputs/autolabel/labels/).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Report .txt path (default {DEFAULT_OUT.name}).",
    )
    return parser.parse_args()


def yolo_line_to_xyxy(parts: list[str]) -> list[float] | None:
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
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
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


def find_split_clip_dir(root: Path, clip: str, split_map: dict[str, str]) -> Path | None:
    split = split_map.get(clip)
    if split:
        path = root / split / clip
        if path.is_dir():
            return path
    for split in SPLITS:
        path = root / split / clip
        if path.is_dir():
            return path
    return None


def list_cvat_clips(cvat_root: Path) -> list[tuple[str, str, Path]]:
    """Return (clip, split, dir) for every labels/{split}/{clip}/ with .txt files."""
    found: list[tuple[str, str, Path]] = []
    for split in SPLITS:
        split_dir = cvat_root / split
        if not split_dir.is_dir():
            continue
        for clip_dir in sorted(split_dir.iterdir()):
            if not clip_dir.is_dir() or clip_dir.name.startswith("."):
                continue
            if not any(clip_dir.glob("*.txt")):
                continue
            found.append((clip_dir.name, split, clip_dir))
    return found


def compare_clip(
    clip: str,
    split: str,
    cvat_dir: Path,
    auto_dir: Path,
    iou_thresh: float,
) -> ClipStats:
    man_stems = {p.stem for p in cvat_dir.glob("*.txt") if p.name.lower() != "classes.txt"}
    auto_stems = {p.stem for p in auto_dir.glob("*.txt") if p.name.lower() != "classes.txt"}
    common = sorted(man_stems & auto_stems)

    stats = ClipStats(
        clip=clip,
        split=split,
        manual_dir=cvat_dir,
        auto_dir=auto_dir,
        iou_thresh=iou_thresh,
        only_manual=sorted(man_stems - auto_stems),
        only_auto=sorted(auto_stems - man_stems),
    )

    for stem in common:
        gt = parse_yolo_boxes(cvat_dir / f"{stem}.txt")
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
        f"  cvat:      {stats.manual_dir}",
        f"  autolabel: {stats.auto_dir}",
        f"  IoU threshold (TP): {stats.iou_thresh}",
        "-" * 72,
        "SUMMARY",
        f"  frames compared:     {stats.n_frames}",
        f"  frames only cvat:    {len(stats.only_manual)}",
        f"  frames only auto:    {len(stats.only_auto)}",
        f"  total bboxes cvat:   {stats.total_manual}",
        f"  total bboxes auto:   {stats.total_auto}",
        f"  delta (auto-cvat):   {stats.total_auto - stats.total_manual}",
        f"  matched pairs (IoU>0): {len(ious)}",
        f"  mean IoU (matched):  {fmt_float(mean_iou)}",
        f"  median IoU (matched):{fmt_float(median_iou)}",
        f"  mean IoU (TP≥thresh):{fmt_float(mean_strong)}",
        f"  TP / FP / FN:        {stats.total_tp} / {stats.total_fp} / {stats.total_fn}",
        f"  precision (auto):    {fmt_pct(stats.precision())}",
        f"  recall (vs cvat):    {fmt_pct(stats.recall())}",
        "-" * 72,
        "PER-FRAME  (stem | n_cvat | n_auto | delta | matched | mean_iou | TP FP FN)",
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
        lines.append(f"  [only in cvat] {preview}{more}")
    if stats.only_auto:
        preview = ", ".join(stats.only_auto[:10])
        more = f" … (+{len(stats.only_auto) - 10})" if len(stats.only_auto) > 10 else ""
        lines.append(f"  [only in auto] {preview}{more}")

    lines.append("")
    return lines


def format_report(all_stats: list[ClipStats], iou_thresh: float) -> str:
    lines = [
        "CVAT (labels/, from cvat_pull) vs YOLO-World autolabel (outputs/autolabel/labels/)",
        f"IoU match threshold: {iou_thresh}",
        f"Clips: {len(all_stats)}",
        "",
    ]
    for stats in all_stats:
        lines.extend(format_clip_report(stats))

    if len(all_stats) > 1:
        tot_man = sum(s.total_manual for s in all_stats)
        tot_auto = sum(s.total_auto for s in all_stats)
        tot_tp = sum(s.total_tp for s in all_stats)
        tot_fp = sum(s.total_fp for s in all_stats)
        tot_fn = sum(s.total_fn for s in all_stats)
        lines.extend(["=" * 72, "OVERALL (all clips)"])
        all_ious: list[float] = []
        for s in all_stats:
            all_ious.extend(s.all_match_ious)
        mean_iou = statistics.mean(all_ious) if all_ious else None
        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else None
        rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else None
        lines.extend(
            [
                f"  frames:              {sum(s.n_frames for s in all_stats)}",
                f"  total bboxes cvat:   {tot_man}",
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
    cvat_root = Path(args.cvat_dir)
    auto_root = Path(args.autolabel_dir)
    out_path = Path(args.out)

    if not cvat_root.is_dir():
        raise SystemExit(
            f"CVAT labels not found: {cvat_root}\n"
            "Pull with: python src/labeling/cvat_pull.py --verify --sync-labels"
        )
    if not auto_root.is_dir():
        raise SystemExit(
            f"Autolabel labels not found: {auto_root}\n"
            "Run: python src/labeling/autolabel.py"
        )

    split_map = build_split_map()
    clips = list_cvat_clips(cvat_root)
    if args.clip:
        clip_filter = Path(args.clip).stem
        clips = [c for c in clips if c[0] == clip_filter]
        if not clips:
            raise SystemExit(f"Clip {clip_filter!r} not found under {cvat_root}")

    all_stats: list[ClipStats] = []
    for clip, split, cvat_dir in clips:
        auto_dir = find_split_clip_dir(auto_root, clip, split_map)
        if auto_dir is None:
            print(f"[skip] {clip}: no autolabels under {auto_root}/{{train,eval}}/{clip}")
            continue
        all_stats.append(
            compare_clip(clip, split, cvat_dir, auto_dir, args.iou)
        )

    if not all_stats:
        raise SystemExit(
            "No clips compared — need both labels/{split}/{clip}/ "
            "and outputs/autolabel/labels/{split}/{clip}/."
        )

    report = format_report(all_stats, args.iou)
    print(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
