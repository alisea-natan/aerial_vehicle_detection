# Raspberry Pi — Track A4

Prerequisites and ship artifact: **[OPTIMISATION.md](../../../OPTIMISATION.md)** Track A1 → A2 (OpenVINO INT8 `ov_int8`).

Pi **5**, aarch64, Raspberry Pi OS Bookworm. Inference defaults: `imgsz=1280`, `conf=0.25` (`config/experiments/optimisation.yaml`). Do not re-run mAP on the Pi unless you choose to — quality gates are on Mac holdout.

---

## 1. Copy model + scripts to the Pi

On Mac (from repo root):

```bash
scp -r checkpoints/opt/quantize/prototype/ov_int8/yolo11s_prototype_best_int8_openvino_model \
  pi@<pi-host>:~/vehicle_detection/model/

scp -r src/optimisation/raspberry pi@<pi-host>:~/vehicle_detection/
```

Optional test clip (pack-sized frames):

```bash
scp path/to/tile_clip.mp4 pi@<pi-host>:~/vehicle_detection/
```

---

## 2. Pi environment

On the Pi:

```bash
cd ~/vehicle_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r raspberry/requirements-raspberry.txt
```

---

## 3. Bench — ms per tile (video)

```bash
cd ~/vehicle_detection/raspberry
python bench_video.py \
  --model ../model/yolo11s_prototype_best_int8_openvino_model \
  --video ../tile_clip.mp4
```

Stdout: `cold_ms`, `p50_ms`, `p95_ms`, `tile_fps@p50`, `artifact_mb`. Copy into **[OPTIMISATION.md § A4 — Cross-target comparison](../../../OPTIMISATION.md#a4-device-validation)** (Pi row).

Optional JSON on the Pi: add `--json-out latest.json`.

---

## 4. Live video (camera)

Picamera2 (CSI) on Pi OS:

```bash
python live_camera.py \
  --model ../model/yolo11s_prototype_best_int8_openvino_model
```

Overlay: per-frame ms and boxes. **q** to quit. Sanity check only — formal metrics from step 3.

---

## Files

| file | role |
| ---- | ---- |
| `bench_video.py` | ms/tile from video → stdout + optional JSON |
| `live_camera.py` | Picamera2 live preview |
| `requirements-raspberry.txt` | Pi venv pins |
| `README.md` | this file |

---

## Troubleshooting

- **Import openvino fails** — aarch64 wheels; 64-bit Pi OS only.
- **Very slow first frame** — cold start; bench uses 20-frame warmup by default.
- **USB webcam** — `live_camera.py` is Picamera2; USB needs OpenCV `VideoCapture(0)` (not included).
