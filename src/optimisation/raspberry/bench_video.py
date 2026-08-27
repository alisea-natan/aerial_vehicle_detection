#!/usr/bin/env python3
"""Raspberry Pi — ms per frame/tile from a video file (OpenVINO via Ultralytics).

  python bench_video.py --model ./yolo11s_prototype_best_int8_openvino_model --video clip.mp4
  python bench_video.py ... --json-out latest.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="OpenVINO model dir or .xml")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--json-out", type=Path, default=None, help="Write metrics JSON locally")
    return p.parse_args()


def artifact_mb(model: Path) -> float:
    if model.is_file():
        return model.stat().st_size / (1024 * 1024)
    if model.is_dir():
        total = sum(p.stat().st_size for p in model.rglob("*") if p.is_file())
        return total / (1024 * 1024)
    return 0.0


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Missing video: {args.video}")
    model = YOLO(str(args.model), task="detect")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    times: list[float] = []
    cold_ms: float | None = None
    n_done = 0
    idx = 0
    while n_done < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride != 0:
            idx += 1
            continue
        idx += 1
        t0 = time.perf_counter()
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False, device="cpu")
        ms = (time.perf_counter() - t0) * 1000.0
        if cold_ms is None:
            cold_ms = ms
        if n_done >= args.warmup:
            times.append(ms)
        n_done += 1
    cap.release()

    if not times:
        raise SystemExit("No timings collected")
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(round(0.95 * (len(times) - 1)))]
    mean_ms = statistics.mean(times)
    print(f"Device: cpu (OpenVINO)  frames: {len(times)}  warmup: {args.warmup}")
    print(f"cold_ms={cold_ms:.1f}  p50_ms={p50:.1f}  p95_ms={p95:.1f}  mean_ms={mean_ms:.1f}")
    print(f"tile_fps@p50={1000.0 / p50:.2f}" if p50 > 0 else "")
    print(f"artifact_mb={artifact_mb(args.model):.1f}")

    if args.json_out:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": platform.node(),
            "platform": platform.platform(),
            "model": str(args.model),
            "runtime": "openvino",
            "video": str(args.video),
            "cold_ms": cold_ms,
            "p50_ms": p50,
            "p95_ms": p95,
            "mean_ms": mean_ms,
            "tile_fps": 1000.0 / p50 if p50 > 0 else 0.0,
            "frames": len(times),
            "warmup": args.warmup,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "artifact_mb": artifact_mb(args.model),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
