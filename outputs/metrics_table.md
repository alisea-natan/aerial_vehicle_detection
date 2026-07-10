# Eval metrics by distance band

Generated: 2026-07-09T23:05:55+00:00
Model: `/Users/alisa/Cursor/vehicle_detection/outputs/runs/yolov8n_vehicle/weights/best.pt`

## Assumptions

- GT: pseudo-labels from `labels/eval/` (class 0 = vehicle).
- Inference: 512×512 tiles, overlap 0.2 (same as train), tile NMS IoU 0.5, boxes mapped to full frame.
- Bands: one eval clip per band; distance band is fixed per whole video from probe (`clip_tiling.json`), not recomputed per frame or per car.
- Match: IoU ≥ 0.5 → TP; unmatched GT → FN; unmatched pred → FP.
- False alarms/min: `FP / (N_frames / fps / 60)`.
- Time to first detection: first TP frame index / fps (seconds from clip start).
- Eval clips: 13722965_2160_3840_30fps, 266987

| Metric | 0-200 m | 200-400 m |
|--------|---------|-----------|
| Detection rate TP/(TP+FN) | 12.1% | 19.4% |
| Precision TP/(TP+FP) | 7.7% | 58.6% |
| False alarms / min | 6137.25 | 311.53 |
| Time to first detection (s) | 0.03 | 2.20 |
| mAP@0.5 (bonus) | 3.3% | 12.7% |

## Per-band counts

**0-200m**: TP=262, FP=3130, FN=1903, frames_with_activity=918, fps=30.0
**200-400m**: TP=153, FP=108, FN=634, frames_with_activity=624, fps=29.99901319035558
