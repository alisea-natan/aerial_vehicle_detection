# Android — Track A4

Track A for TFLite: **[OPTIMISATION.md](../../../OPTIMISATION.md)**.

**Ship model:** A1 TFLite FP32 @ **1280** — `python src/optimisation/run_export.py --platform android` → `checkpoints/opt/export/prototype/tflite/*.tflite`

**On-phone test:** [android_tflite_benchmark](https://github.com/alisea-natan/android_tflite_benchmark) (`vehicle-bench.apk`). Copy the `.tflite` + any eval video to the phone.

---

## Run and collect stats

1. Install APK; pick **1280** `.tflite` and video in the app (`pack_tile` mode for tile clips).
2. Pick delegate (**CPU** or **GPU**; optional **NNAPI** if load works). **Run** — **Stop** when done (≥100 post-warmup tiles for A4, per OPTIMISATION.md).
3. **Share JSON** or copy summary → paste p50 / tile_fps into **OPTIMISATION.md § A4** (cite CPU and GPU as separate rows).

Warmup (default 20) excludes first N tile inferences from p50, not a sample cap.

---



## Compare bench JSON to labels (Mac)

After **Share JSON** from the app (needs `detections[]` per `tile_log` entry — rebuild APK if your export only has det counts).

```bash
# from repo root; video on device should be an eval clip (e.g. data/eval/266987.mp4)
python src/optimisation/android/compare_bench_to_labels.py \
  --bench vehicle_bench.json
```

**Prerequisites:** `labels/eval/{clip}/` and `data/frames/{clip}/` present (same as Mac holdout eval).

**Clip:** inferred from bench `video` (`266987.mp4` → band B). Override with `--clip 13722965_2160_3840_30fps`.

**Scoring:** `labels/eval/{clip}/` with each clip’s `frame_step` from `clip_tiling.json` (same subsample as `eval_manual`). Only frames the bench ran (non-warmup) and that have labels. Match IoU **0.5** (evaluate.py default); tile merge NMS uses bench JSON `iou` (default 0.45).

**Output:** Det, P, FA/min, [mAP@0.5](mailto:mAP@0.5) / @0.5:0.95 for the clip’s distance band (A or B).

```bash
# save metrics JSON
python src/optimisation/android/compare_bench_to_labels.py \
  --bench vehicle_bench_*.json \
  --json-out outputs/android_bench_eval.json

# every labeled frame (skip frame_step filter)
python src/optimisation/android/compare_bench_to_labels.py \
  --bench vehicle_bench_*.json --no-frame-step
```

Old bench exports without `detections[]` will fail with a clear message — re-run bench and Share JSON again.

---

The bench app is a **homemade smoke tool** — **CPU / GPU** (TFLite GpuDelegate) and optional NNAPI. On this mt6886, NNAPI ≈ CPU; GPU is the speed path (~10×). For a production mobile stack, a **640 imgsz** model and a **per-chip NPU runtime** (separate from this repo) would still be the next step; this project’s artifact stays **1280 FP32** for quality parity with Pi/Jetson.