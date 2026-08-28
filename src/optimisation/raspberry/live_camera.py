#!/usr/bin/env python3
"""Raspberry Pi — live OpenVINO detect from camera (Picamera2).

  python live_camera.py --model ./yolo11s_prototype_best_int8_openvino_model

Run bench_video.py first for p50/p95 numbers. This script is for live preview + on-screen ms.
Press q to quit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise SystemExit("Install picamera2 (Pi only): pip install picamera2") from exc

    model = YOLO(str(args.model), task="detect")
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={"FrameRate": args.fps},
    )
    picam.configure(config)
    picam.start()
    time.sleep(1.0)

    print("Live preview — q to quit")
    try:
        while True:
            frame = picam.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            t0 = time.perf_counter()
            results = model.predict(
                frame_bgr,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
                device="cpu",
            )
            ms = (time.perf_counter() - t0) * 1000.0
            annotated = results[0].plot()
            cv2.putText(
                annotated,
                f"{ms:.0f} ms  conf>={args.conf}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("vehicle_detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        picam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
