#!/usr/bin/env python3
"""Extract every frame from videos and save clip metadata (step 1 of pipeline)."""

import argparse
import json
from pathlib import Path

import cv2

SOURCE_DIRS = ("data/train", "data/eval")
FRAMES_DIR = Path("data/frames")


def iter_videos(root: Path):
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() == ".mp4":
            yield path


def extract_clip(video_path: Path, output_dir: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for {video_path}: {fps}")

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        saved_idx += 1
        cv2.imwrite(str(output_dir / f"{saved_idx:06d}.jpg"), frame)

    cap.release()

    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from train/eval videos.")
    parser.add_argument(
        "--clip",
        default=None,
        help="Extract a single clip (video stem). Default: all videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip_filter = args.clip
    if clip_filter and clip_filter.endswith(".mp4"):
        clip_filter = Path(clip_filter).stem

    project_root = Path(__file__).resolve().parent.parent
    frames_root = project_root / FRAMES_DIR
    found = False

    for source in SOURCE_DIRS:
        source_dir = project_root / source
        if not source_dir.is_dir():
            print(f"Skipping missing directory: {source_dir}")
            continue

        for video_path in iter_videos(source_dir):
            clip_name = video_path.stem
            if clip_filter and clip_name != clip_filter:
                continue
            found = True
            output_dir = frames_root / clip_name
            print(f"{video_path.relative_to(project_root)} -> {output_dir.relative_to(project_root)}/")
            extract_clip(video_path, output_dir)

    if clip_filter and not found:
        raise SystemExit(f"Video not found for clip {clip_filter!r} in data/train or data/eval.")


if __name__ == "__main__":
    main()
