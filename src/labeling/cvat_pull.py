#!/usr/bin/env python3
"""Pull YOLO labels from CVAT (frames already local — labels-only by default).

Primary usage — credentials from repo ``.env`` (see ``.env.example``):

  cp .env.example .env   # set CVAT_HOST, CVAT_USER, CVAT_PASS, optional CVAT_PROJECT
  python src/labeling/cvat_pull.py --project aerial_vehicles --verify --sync-labels

  # control run on one task (with images) after listing
  python src/labeling/cvat_pull.py --project aerial_vehicles --list
  python src/labeling/cvat_pull.py --project aerial_vehicles --task 7 --with-images --verify

Task.name must match local clip stem under data/frames/ (same as upload).
Split (train|eval) is taken from data/train vs data/eval video folders.

Optional: --config config/cvat_tasks.json still works as a manual override map.
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
import json
import os
import shutil
import zipfile
from pathlib import Path


from common.config import (
    PROJECT_ROOT,
    build_split_map,
    clip_skip_reason,
    is_clip_skipped,
)

FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
LABELS_DIR = PROJECT_ROOT / "labels"
RAW_BASE = PROJECT_ROOT / "data" / "cvat"
TASKS_CONFIG = PROJECT_ROOT / "config" / "cvat_tasks.json"
EXPORT_FORMAT = "YOLO 1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CVAT YOLO labels (default: include_images=False).",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--project",
        default=os.environ.get("CVAT_PROJECT", "").strip() or None,
        help="CVAT project name — export all its tasks (preferred). "
        "Or set CVAT_PROJECT.",
    )
    src.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="CVAT project id (alternative to --project name).",
    )
    src.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional manual task map JSON (legacy). Prefer --project.",
    )
    parser.add_argument(
        "--task",
        type=int,
        action="append",
        default=None,
        help="Only these CVAT task id(s). Repeatable. Default: all in project.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered tasks and exit (no export).",
    )
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="Include images in the zip (control run to compare filenames).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Unzip and compare label stems to local data/frames/{video}/.",
    )
    parser.add_argument(
        "--sync-labels",
        action="store_true",
        help="Copy matched .txt into labels/{split}/{video_name}/ (train.py path).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_BASE,
        help=f"Output root (default {RAW_BASE.relative_to(PROJECT_ROOT)}).",
    )
    return parser.parse_args()


def _credentials() -> tuple[str, str, str]:
    host = os.environ.get("CVAT_HOST", "").strip()
    user = os.environ.get("CVAT_USER", "").strip()
    password = os.environ.get("CVAT_PASS", "").strip()
    if not host or not user or not password:
        raise SystemExit(
            "Set CVAT_HOST, CVAT_USER, and CVAT_PASS in .env (see .env.example) "
            "or export them in the shell."
        )
    return host, user, password


def load_task_map_from_config(path: Path) -> dict[int, dict]:
    if not path.is_file():
        raise SystemExit(f"Missing config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("tasks", payload)
    out: dict[int, dict] = {}
    for key, meta in raw.items():
        if str(key).startswith("_"):
            continue
        task_id = int(key)
        video = str(meta["video_name"]).strip()
        split = str(meta["split"]).strip()
        if split not in ("train", "eval"):
            raise SystemExit(f"task {task_id}: split must be train|eval, got {split!r}")
        out[task_id] = {"video_name": video, "split": split}
    if not out:
        raise SystemExit(f"No tasks in {path}")
    return out


def resolve_project(client, *, name: str | None, project_id: int | None):
    if project_id is not None:
        return client.projects.retrieve(project_id)

    assert name
    # Exact name filter when API supports it; fall back to scan.
    matches = client.projects.list(name=name)
    exact = [p for p in matches if p.name == name]
    if not exact:
        # Some servers ignore name=; scan all.
        exact = [p for p in client.projects.list() if p.name == name]
    if not exact:
        known = ", ".join(sorted({p.name for p in client.projects.list()})[:30]) or "(none)"
        raise SystemExit(f"No CVAT project named {name!r}. Known (sample): {known}")
    if len(exact) > 1:
        ids = ", ".join(str(p.id) for p in exact)
        raise SystemExit(
            f"Multiple projects named {name!r} (ids {ids}). Use --project-id."
        )
    return exact[0]


def discover_task_map(
    client,
    *,
    project_name: str | None,
    project_id: int | None,
) -> dict[int, dict]:
    """Build task_id → {video_name, split} from CVAT project + local data/ folders."""
    project = resolve_project(client, name=project_name, project_id=project_id)
    split_map = build_split_map(PROJECT_ROOT)
    tasks = project.get_tasks()
    if not tasks:
        raise SystemExit(f"Project {project.name!r} (id={project.id}) has no tasks.")

    out: dict[int, dict] = {}
    skipped: list[str] = []
    unknown_split: list[str] = []

    print(f"Project {project.name!r} (id={project.id}): {len(tasks)} task(s)")
    for task in sorted(tasks, key=lambda t: t.id):
        video_name = (task.name or "").strip()
        if not video_name:
            print(f"  [skip] task {task.id}: empty name")
            continue
        if is_clip_skipped(video_name):
            skipped.append(f"{task.id}:{video_name}")
            print(f"  [skip] task {task.id} {video_name}: {clip_skip_reason(video_name)}")
            continue
        split = split_map.get(video_name)
        if split is None:
            unknown_split.append(f"{task.id}:{video_name}")
            print(
                f"  [skip] task {task.id} {video_name}: "
                "no matching .mp4 in data/train or data/eval "
                "(name must equal video stem)"
            )
            continue
        frames_dir = FRAMES_DIR / video_name
        if not frames_dir.is_dir():
            print(
                f"  [warn] task {task.id} {video_name}: "
                f"no local frames at {frames_dir.relative_to(PROJECT_ROOT)} "
                "(export ok; --verify/--sync-labels will need frames)"
            )
        out[task.id] = {"video_name": video_name, "split": split}
        print(f"  task {task.id}: {video_name} [{split}]")

    if not out:
        raise SystemExit(
            "No exportable tasks after filters.\n"
            f"  skipped(config)={skipped or '-'}\n"
            f"  unknown_split={unknown_split or '-'}\n"
            "Ensure CVAT task names == data/frames/<stem> == data/{{train,eval}}/<stem>.mp4"
        )
    return out


# CVAT YOLO 1.1 export uses train and/or test folders (casing varies by version).
_YOLO_OBJ_DIR_NAMES = (
    "obj_train_data",
    "obj_Train_data",
    "obj_test_data",
    "obj_Test_data",
)


def _find_obj_dirs(extract_root: Path) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for name in _YOLO_OBJ_DIR_NAMES:
        candidates = []
        direct = extract_root / name
        if direct.is_dir():
            candidates.append(direct)
        candidates.extend(p for p in extract_root.rglob(name) if p.is_dir())
        for candidate in candidates:
            key = str(candidate.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(candidate)
    return found


def unzip_labels(zip_path: Path, labels_raw_dir: Path) -> Path:
    """Extract YOLO zip; copy .txt into labels_raw_dir; return primary obj_* path."""
    extract_to = zip_path.with_suffix("")
    if extract_to.exists():
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)

    obj_dirs = _find_obj_dirs(extract_to)
    if not obj_dirs:
        raise SystemExit(
            f"No obj_train_data / obj_Test_data in {zip_path} "
            f"(looked for: {', '.join(_YOLO_OBJ_DIR_NAMES)})"
        )

    if labels_raw_dir.exists():
        shutil.rmtree(labels_raw_dir)
    labels_raw_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for obj_dir in obj_dirs:
        for label in obj_dir.glob("*.txt"):
            if label.name.lower() == "classes.txt":
                continue
            shutil.copy2(label, labels_raw_dir / label.name)
            n += 1
    print(f"  unpacked {n} label files → {labels_raw_dir.relative_to(PROJECT_ROOT)}")
    return obj_dirs[0]


def verify_against_frames(
    video_name: str,
    labels_raw_dir: Path,
    obj_dir: Path | None = None,
) -> dict:
    frames_dir = FRAMES_DIR / video_name
    if not frames_dir.is_dir():
        print(f"  [VERIFY FAIL] missing local frames: {frames_dir}")
        return {"ok": False, "error": "missing_frames"}

    frame_stems = {p.stem for p in frames_dir.glob("*.jpg")} | {
        p.stem for p in frames_dir.glob("*.png")
    }
    label_stems = {
        p.stem
        for p in labels_raw_dir.glob("*.txt")
        if p.name.lower() != "classes.txt"
    }

    export_image_stems: set[str] = set()
    if obj_dir is not None:
        export_image_stems = {p.stem for p in obj_dir.glob("*.jpg")} | {
            p.stem for p in obj_dir.glob("*.png")
        }

    labels_without_frame = sorted(label_stems - frame_stems)
    frames_without_label = sorted(frame_stems - label_stems)

    report = {
        "ok": not labels_without_frame,
        "video_name": video_name,
        "n_local_frames": len(frame_stems),
        "n_label_files": len(label_stems),
        "n_export_images": len(export_image_stems) if export_image_stems else None,
        "labels_without_local_frame": labels_without_frame[:50],
        "n_labels_without_local_frame": len(labels_without_frame),
        "frames_without_label": frames_without_label[:50],
        "n_frames_without_label": len(frames_without_label),
        "export_images_vs_frames_match": (
            None if not export_image_stems else export_image_stems == frame_stems
        ),
        "export_only_images": sorted(export_image_stems - frame_stems)[:20]
        if export_image_stems
        else [],
        "local_only_images": sorted(frame_stems - export_image_stems)[:20]
        if export_image_stems
        else [],
    }

    print(
        f"  [verify] frames={report['n_local_frames']} "
        f"labels={report['n_label_files']} "
        f"labels_wo_frame={report['n_labels_without_local_frame']} "
        f"frames_wo_label={report['n_frames_without_label']}"
    )
    if export_image_stems:
        match = report["export_images_vs_frames_match"]
        print(f"  [verify] export images vs local frames 1:1 → {match}")
    if labels_without_frame:
        print(f"  [VERIFY WARN] labels with no local frame: {labels_without_frame[:10]}")
    return report


def sync_labels_to_pipeline(split: str, video_name: str, labels_raw_dir: Path) -> int:
    frames_dir = FRAMES_DIR / video_name
    if not frames_dir.is_dir():
        raise SystemExit(f"Cannot sync: missing {frames_dir}")

    frame_stems = {p.stem for p in frames_dir.glob("*.jpg")} | {
        p.stem for p in frames_dir.glob("*.png")
    }
    dest = LABELS_DIR / split / video_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    n = skipped = 0
    for src in sorted(labels_raw_dir.glob("*.txt")):
        if src.name.lower() == "classes.txt":
            continue
        if src.stem not in frame_stems:
            skipped += 1
            continue
        shutil.copy2(src, dest / src.name)
        n += 1
    print(
        f"  synced {n} labels → {dest.relative_to(PROJECT_ROOT)}"
        + (f" (skipped {skipped} without local frame)" if skipped else "")
    )
    return n


def pull_labels(
    task_map: dict[int, dict],
    *,
    task_filter: list[int] | None,
    include_images: bool,
    verify: bool,
    sync_labels: bool,
    out_root: Path,
    client,
) -> None:
    selected = (
        {tid: task_map[tid] for tid in task_filter}
        if task_filter
        else dict(task_map)
    )
    missing = [tid for tid in (task_filter or []) if tid not in task_map]
    if missing:
        raise SystemExit(f"Task id(s) not in discovered/config map: {missing}")

    mode = "WITH images (control)" if include_images else "labels-only"
    print(f"CVAT export ({mode}) → {out_root.relative_to(PROJECT_ROOT)}")

    verify_reports: list[dict] = []
    for task_id, meta in sorted(selected.items()):
        video_name = meta["video_name"]
        split = meta["split"]
        task = client.tasks.retrieve(task_id)

        jobs = task.get_jobs()
        not_done = [j.id for j in jobs if getattr(j, "state", None) != "completed"]
        if not_done:
            print(f"[WARN] task {task_id}: jobs not completed: {not_done}")

        clip_root = out_root / split / video_name
        clip_root.mkdir(parents=True, exist_ok=True)
        export_path = clip_root / f"{video_name}_labels.zip"
        labels_raw = clip_root / "labels_raw"

        print(
            f"Exporting task {task_id} ({video_name}, {split}) → "
            f"{export_path.relative_to(PROJECT_ROOT)}"
        )
        if export_path.exists():
            export_path.unlink()
        task.export_dataset(
            format_name=EXPORT_FORMAT,
            filename=str(export_path),
            include_images=include_images,
        )

        obj_dir = unzip_labels(export_path, labels_raw)

        if verify or include_images:
            report = verify_against_frames(
                video_name,
                labels_raw,
                obj_dir=obj_dir if include_images else None,
            )
            verify_reports.append(report)

        if sync_labels:
            sync_labels_to_pipeline(split, video_name, labels_raw)

    if verify_reports:
        report_path = out_root / "verify_report.json"
        report_path.write_text(
            json.dumps(verify_reports, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Verify report: {report_path.relative_to(PROJECT_ROOT)}")

    print("Done.")


def main() -> None:
    args = parse_args()
    try:
        from cvat_sdk import make_client
    except ImportError as exc:
        raise SystemExit("cvat-sdk is not installed. Run: pip install cvat-sdk") from exc

    host, user, password = _credentials()
    with make_client(host=host, credentials=(user, password)) as client:
        if args.config is not None:
            task_map = load_task_map_from_config(args.config)
            print(f"Using manual map: {args.config}")
        elif args.project or args.project_id is not None:
            task_map = discover_task_map(
                client,
                project_name=args.project,
                project_id=args.project_id,
            )
        elif TASKS_CONFIG.is_file():
            print(f"[info] falling back to {TASKS_CONFIG.relative_to(PROJECT_ROOT)}")
            task_map = load_task_map_from_config(TASKS_CONFIG)
        else:
            raise SystemExit(
                "Pass --project NAME (or --project-id / --config).\n"
                "Example: python src/labeling/cvat_pull.py --project aerial_vehicles --list"
            )

        if args.list:
            print(f"{'id':>6}  {'split':<5}  video_name")
            for tid, meta in sorted(task_map.items()):
                print(f"{tid:>6}  {meta['split']:<5}  {meta['video_name']}")
            return

        pull_labels(
            task_map,
            task_filter=args.task,
            include_images=args.with_images,
            verify=args.verify,
            sync_labels=args.sync_labels,
            out_root=args.out.resolve(),
            client=client,
        )


if __name__ == "__main__":
    main()
