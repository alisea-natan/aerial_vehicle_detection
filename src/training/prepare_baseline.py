#!/usr/bin/env python3
"""Build and save a versioned baseline dataset (no prepare-time augs).

Default (human / CVAT GT → baseline_v1):
  labels/ + data/frames/
  train_groups tiling from config/clip_tiling.json (~1024)
  only frames on each clip's frame_step
  no group balance / rotation oversample

Optional pseudo baseline from YOLO-World autolabel (→ baseline_v0):
  python src/training/prepare_baseline.py --from-autolabel

Datasets live under data/datasets/ locally only.
"""
from __future__ import annotations

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
from pathlib import Path

import yaml

from common.config import LABELS_DIR, PROJECT_ROOT, TRAIN_IMGSZ, build_split_map
from training.train import BASELINE_DATASET_DIR, prepare_dataset

AUTOLABEL_LABELS = PROJECT_ROOT / "outputs" / "autolabel" / "labels"
DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"


def _load_baseline_params() -> dict:
    path = PROJECT_ROOT / "params.yaml"
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(payload.get("baseline") or {})


def parse_args() -> argparse.Namespace:
    cfg = _load_baseline_params()
    default_name = str(cfg.get("name") or "baseline_v1")
    default_out = Path(str(cfg.get("out_dir") or BASELINE_DATASET_DIR))
    if not default_out.is_absolute():
        default_out = PROJECT_ROOT / default_out
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=None, help="Dataset version name (folder under data/datasets/).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dataset root (default data/datasets/<name>).",
    )
    parser.add_argument(
        "--from-autolabel",
        action="store_true",
        help=(
            "Use outputs/autolabel/labels (YOLO-World pseudo-labels) instead of labels/. "
            "Defaults --name to baseline_v0. Not real GT — for smoke / comparison only."
        ),
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=None,
        help="Override label tree root (must contain train|eval/<clip>/*.txt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(cfg.get("imgsz") or TRAIN_IMGSZ),
        help=f"Model imgsz metadata (default {cfg.get('imgsz') or TRAIN_IMGSZ}).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=float(cfg.get("val_fraction") or 0.15),
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Reuse existing dataset if present (default: always rebuild).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_baseline_params()

    from_autolabel = bool(args.from_autolabel)
    name = args.name
    if name is None:
        name = "baseline_v0" if from_autolabel else str(cfg.get("name") or "baseline_v1")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = DATASETS_ROOT / name
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    labels_root = args.labels_root
    if labels_root is None and from_autolabel:
        labels_root = AUTOLABEL_LABELS
    if labels_root is None:
        labels_root = LABELS_DIR
    if not labels_root.is_absolute():
        labels_root = PROJECT_ROOT / labels_root

    if from_autolabel and not (labels_root / "train").is_dir():
        raise SystemExit(
            f"No autolabel train labels at {labels_root / 'train'}.\n"
            "Run: python src/labeling/autolabel.py   # writes outputs/autolabel/labels/"
        )

    split_map = build_split_map()
    if not split_map:
        raise SystemExit("No clips in data/train or data/eval.")

    recreate = not bool(args.no_recreate)
    print(
        f"Building {'autolabel pseudo' if from_autolabel else 'GT'} baseline "
        f"name={name!r} → {out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir}"
    )
    yaml_path = prepare_dataset(
        split_map,
        recreate=recreate,
        imgsz=args.imgsz,
        val_fraction=args.val_fraction,
        dataset_dir=out_dir,
        frame_step_only=True,
        balance=False,
        max_empty_slices_per_frame=0,
        labels_root=labels_root,
    )
    print(f"\nBaseline saved: {yaml_path}")
    print("Local only under data/datasets/ (gitignored).")
    print("Optional local DVC cache:")
    rel = out_dir.relative_to(PROJECT_ROOT) if out_dir.is_relative_to(PROJECT_ROOT) else out_dir
    print(f"  dvc add {rel} && dvc push")


if __name__ == "__main__":
    main()
