# Eval metrics by distance band

Generated: 2026-07-14T05:28:15+00:00
Model: `/Users/alisa/Cursor/vehicle_detection/outputs/runs/yolov8n_vehicle/weights/best.pt`

## Assumptions

- GT: pseudo-labels from `labels/eval/` (class 0 = vehicle).
- Inference: per-clip slice from `scale_coeff` → imgsz=1280, overlap 0.2 (from `/Users/alisa/Cursor/vehicle_detection/outputs/dataset/data.yaml`), tile NMS IoU 0.5, boxes mapped to full frame.
- Bands: one eval clip per band; distance band is fixed per whole video from probe (`clip_tiling.json`), not recomputed per frame or per car.
- Match: IoU ≥ 0.5 → TP; unmatched GT → FN; unmatched pred → FP.
- False alarms/min: `FP / (N_frames / fps / 60)`.
- Time to first detection: first TP frame index / fps (seconds from clip start).
- Eval clips: 13722965_2160_3840_30fps, 266987

## Timing

- Wall time: **7m 17.3s** (437.29s).
- `13722965_2160_3840_30fps`: 200.62s (infer 147.34s, 6.23 frame/s, video 53.28s)
- `266987`: 236.62s (infer 150.33s, 6.07 frame/s, video 86.29s)

| Metric | 0-200 m | 200-400 m |
|--------|---------|-----------|
| Detection rate TP/(TP+FN) | 65.6% | 42.4% |
| Precision TP/(TP+FP) | 64.5% | 84.3% |
| False alarms / min | 2119.61 | 202.31 |
| Time to first detection (s) | 0.03 | 0.03 |
| mAP@0.5 (bonus) | 51.3% | 42.5% |

## Per-band counts

**0-200m**: TP=1965, FP=1081, FN=1031, frames_with_activity=918, fps=30.0
**200-400m**: TP=366, FP=68, FN=497, frames_with_activity=605, fps=29.99901319035558
