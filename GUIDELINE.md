# Guideline — labeling, datasets, experiments, metrics

Companion to **[README.md](README.md)** (main PoC: config → autolabel → YOLO11s → eval).

This document covers extras and deeper notes:

1. Preprocess details (`frame_step`, tiles, distance)  
2. Manual labeling (CVAT / Roboflow) & QA vs autolabel  
3. Dataset packs (local only)  
4. Ablation rounds (datasets → models)  
5. Extra training / eval notes & metric history  

**Vehicle definition** (what is / is not a vehicle) lives in **[README.md](README.md)** — keep that as the source of truth.

## Where things are saved

| Artifact | Path | How |
| -------- | ---- | --- |
| Source videos | `data/train/*.mp4`, `data/eval/*.mp4` | Git LFS / local |
| Frames | `data/frames/{clip}/*.jpg` | `extract_frames.py` |
| Preprocess config | `config/clip_tiling.json` | `preprocess_clips.py` |
| Preprocess report | `debug/preprocess_probe.json` | same |
| **CVAT GT (human)** | `labels/{train\|eval}/{clip}/*.txt` | `cvat_pull.py --sync-labels` |
| CVAT raw export | `data/cvat/` | `cvat_pull.py` (before sync) |
| **YOLO-World autolabel** | `outputs/autolabel/labels/{train\|eval}/{clip}/*.txt` | `autolabel.py` |
| Autolabel stats | `outputs/autolabel/debug/label_stats.json` | same |
| Compare report | `debug/compare_autolabel_vs_cvat.txt` | `compare.py` (default: autolabel ↔ `labels/`) |
| Dataset packs | `data/datasets/{name}/` | `prepare_baseline.py` / `generate_variant.py` |
| Fixed eval packs | `data/datasets/eval_manual/`, `eval_autolabel/` | `prepare_eval.py` |
| Experiment runs | `outputs/experiments/` | `run_*_round.py` |
| Train runs | `outputs/runs/…/weights/` | `train.py` |
| Checkpoints | `checkpoints/` | copy deliverable `best.pt` |
| Eval metrics | `outputs/eval_autolabel/`, `outputs/eval_manual/` | `evaluate.py` |
| Roboflow packs (optional) | `labelling/roboflow/` | prepare_* scripts |

**Do not confuse:** `labels/` = CVAT only. Autolabel never writes there; it always goes under `outputs/autolabel/`.

---

## 1. Preprocess: tiles, size, distance, `frame_step`

`python src/data/preprocess_clips.py` (alias: `probe_clips.py`) writes `config/clip_tiling.json`.

### How it works

1. Sample **3 frames from start, middle, end** of each clip.  
2. Try tile candidates `1 → 2 → 3 → 4 → 6 → 8 → 12`. First **car** hit (`person` → car during probe only) = `probe_min_tiles`.  
3. `target_tiles` = next candidate (+1 headroom). Threshold / overlap follow `target_tiles`.  
4. Detail pass: car-only sizes & distances per segment; `distance_varies` if bands differ or start↔end size changes ≥30%.  
5. Motion: match cars on consecutive frames → `speed_px_per_frame`.  
6. `frame_step = floor(object_size_px_median × 0.5 / speed)` (min 1). Suggests `train_groups` band from size.

Fallback if no car: **12 tiles** @ threshold 0.1.

### Distance

```
distance_m = 4.5 m × focal_px / bbox_long_side_px
```

Assumed passenger-car length 4.5 m; optional `calibration/{clip}.json`.

### Tiles & confidence

| Tiles | Typical band | Overlap |
| ----- | ------------ | ------- |
| 1     | <200 m       | 0       |
| 2–7   | >200 m       | 0.10    |
| ≥8    | >400 m       | 0.05    |

```
tiles = 1   →  0.50
tiles > 1   →  max(0.05, 0.50 / tiles)
```

### Probe-only alias

During **preprocess only**, `person → car` for tile search / size / distance. Autolabel uses full vehicle prompts separately.

---

## 2. Why YOLO-World

YOLO-World (`yolov8x-worldv2.pt`) is **not** training GT:

1. **Preprocess probe** — tiles / size / distance / `frame_step`.  
2. **Demo autolabel** — `frame_step` frames only → `outputs/autolabel/`.

Treat demo boxes as noisy; replace with CVAT for real training.

```bash
python src/data/preprocess_clips.py
python src/labeling/autolabel.py --clip 266987
```

---

## 3. Manual labeling & QA

### Layout

```
labels/{train|eval}/{clip}/{frame_stem}.txt     ← CVAT only (cvat_pull --sync-labels)
outputs/autolabel/labels/{train|eval}/{clip}/   ← YOLO-World (autolabel.py)
data/frames/{clip}/{frame_stem}.jpg             ← must match stems
```

Empty `.txt` = no vehicles. Use preprocess `frame_step` as annotation density guide.

```bash
python src/labeling/cvat_pull.py --verify --sync-labels
```

CVAT GT for the pipeline lives only in `labels/` (via `cvat_pull --sync-labels`). Optional Roboflow packs under `labelling/roboflow/`. Old `label_man/` dumps were removed.

### Unusable clip: `8968356-hd_1920_1080_30fps`

Preprocess probes it, then sets `"skip": true` when median size &lt; 32 px (~17 px here). Rationale (PoC time): the clip is out-of-distribution vs both train and eval scales — see README. Autolabel / train / eval honor the flag (`--include-skipped` to force).

### Autolabel vs CVAT (`labels/`) — IoU ≥ 0.5

```bash
python src/labeling/compare.py
# → debug/compare_autolabel_vs_cvat.txt
```

Default: **YOLO-World** `outputs/autolabel/labels/` vs **CVAT extract** `labels/`.

**Results (2026-08-12)** — IoU ≥ 0.5 on frames present in **both** sets (`frame_step` → fewer auto frames):

| Clip | Split | Frames both | CVAT | Auto | Precision | Recall | Mean IoU |
| ---- | ----- | ----------: | ---: | ---: | --------: | -----: | -------: |
| `13722965…` | eval | 49 | 121 | 78 | 84.6% | 54.5% | 0.90 |
| `266987` | eval | 304 | 274 | 257 | 86.4% | 81.0% | 0.81 |
| `3405804…` | train | 67 | 1374 | 774 | 54.1% | 30.5% | 0.78 |
| `5382494…` | train | 54 | 347 | 94 | 100.0% | 27.1% | 0.91 |
| `8457857…` | train | 203 | 4110 | 5017 | 53.4% | 65.2% | 0.70 |
| **Overall** | | **677** | **6226** | **6220** | **55.9%** | **55.9%** | **0.72** |

Takeaway: eval precision ~85%; several train clips under-recall or over-label. Full dump: `debug/compare_autolabel_vs_cvat.txt`.

### Mean object size (historical, 2026-07-13)

Long side in full-frame pixels (from an older label-box dump; sizes also live in `clip_tiling.json` after preprocess as `object_size_px_median`):

| Split | Video | Mean (px) | Median |
| ----- | ----- | --------: | -----: |
| train | `3405804…` | 152.2 | 137.2 |
| train | `5382494…` | 64.8 | 53.1 |
| train | `8457857…` | 161.3 | 107.6 |
| train | `8968356…` | 22.2 | 17.2 |
| eval | `13722965…` | 347.7 | 368.8 |
| eval | `266987` | 135.0 | 133.2 |

### Suggested `frame_step`

Stored per clip in `clip_tiling.json` after preprocess. Historical full-autolabel estimates (fraction 0.5):

| Clip | step @0.5 (approx) |
| ---- | -----------------: |
| `3405804…` | 18 |
| `5382494…` | 9 |
| `8457857…` | 5 |
| `8968356…` | 2 |
| `13722965…` | 19 |
| `266987` | 3 |

Recompute: `python src/data/preprocess_clips.py`.

---

## 4. Dataset packs (local only)

Under `data/datasets/` — local only (only README / `.gitignore` tracked).

### Helpers (`prepare_baseline.py`)

| Pack | Source | Command |
| ---- | ------ | ------- |
| `baseline_v1` | `labels/` (CVAT) | `python src/training/prepare_baseline.py` |
| `baseline_v0` | autolabel | `python src/training/prepare_baseline.py --from-autolabel` |
| `eval_manual` | CVAT `labels/eval/` | `python src/training/prepare_eval.py` |
| `eval_autolabel` | YOLO-World | `python src/training/prepare_eval.py --from-autolabel` |
| `eval_manual_adapted` | CVAT, pad→smaller cars | `python src/training/prepare_eval.py --scale-adapt` |
| `eval_autolabel_adapted` | YOLO-World, pad→smaller cars | `python src/training/prepare_eval.py --from-autolabel --scale-adapt` |

Both use the same `train_groups` tiling / `frame_step`, but only `data/eval/` videos (default: the two band clips). Keep GT sources in **separate folders** — never mix. Build once; reuse for every experiment. Eval packs also cap each clip at **64** frames after `frame_step` (`--max-frames-per-clip`; `0` disables) so long/fast clips do not flood the pack.

`--scale-adapt` **copies** the native pack, then pads only oversized clips (mid-band stays byte-identical). Build native first, then adapt. Primary metrics stay on the native pack; evaluate with `--gt autolabel_adapted` / `manual_adapted`.

Uses `frame_step`, `train_groups` (~1024), no prepare-time balance/rotations.

### Ablation packs (`config/datasets/variants.yaml`)

AUTO from `train_groups`: B_medium tile **1024** / overlap **0.2**; A_close = full frame.

```bash
python src/training/datasets/generate_variant.py --list
python src/training/datasets/generate_variant.py --all
```

| id | Change vs `baseline_1` |
| -- | ---------------------- |
| `baseline_1` | AUTO tiling, full sampling, drop empty tiles |
| `variant_2_no_tiling` | no tiling |
| `variant_3_tiling_fixed` | tile 640 / overlap 0.1 |
| `variant_4_low_overlap` | AUTO tiles, overlap 0.1 (+ multi-tile bbox stats) |
| `variant_5_aug` | reuse `baseline_1` tiles; mosaic_hsv at **train** |
| `variant_6_strided` | stride 5 |
| `variant_7_negatives` | keep empty tiles @ 15% of positives |

See `data/datasets/README.md`, `config/datasets/README.md`.

---

## 5. Ablation rounds (experiments)

Two rounds — **datasets first**, then **models**. Specs under `config/experiments/`.

| Round | Config | Runner | Question |
| ----- | ------ | ------ | -------- |
| 1 Datasets | `dataset_round.yaml` | `run_dataset_round.py` | Best pack with fixed yolo11n? |
| 2 Models | `model_round.yaml` | `run_model_round.py` | Best model/hparams on fixed pack? |

```bash
python src/training/datasets/generate_variant.py --all

# Round 1
python src/training/experiments/run_dataset_round.py --dry-run --variant baseline_1
python src/training/experiments/run_dataset_round.py --variant baseline_1

# Round 2 (default dataset = baseline_1)
python src/training/experiments/run_model_round.py --dry-run --variant model_n_640
python src/training/experiments/run_model_round.py --variant model_n_640
```

Fast defaults: **yolo11n**, imgsz **640**, short schedule. Runs → `outputs/experiments/`.

Model-round placeholders: `model_n_640`, `model_n_1024`, `model_s_640`, `model_n_640_aug`, `model_n_640_long`.

---

## 6. Training details

Main train (`train.py`): **YOLO11s**, staged head-then-full fine-tune.

| Setting | Value |
| ------- | ----- |
| Base | `yolo11s.pt` |
| Epochs | Stage1 ~5 (freeze) + Stage2 ~20; early-stop patience |
| Crops | `train_groups` in `clip_tiling.json` |
| Aug (main) | HSV + flips + degrees=180; **mosaic off** |
| Deliverable | `checkpoints/yolo11s_vehicle_best.pt` |

### Train groups

| Group | Videos | tile_size | overlap | train_imgsz |
| ----- | ------ | --------: | ------: | ----------: |
| `C_far` | *(none)* | 768 | 0.20 | 1024 |
| `B_medium` | `5382494` | 1024 | 0.20 | 1024 |
| `A_close` | close clips | null (full frame) | — | 1024 |

| Stage | Input |
| ----- | ----- |
| Train | group tile → letterbox `--imgsz` |
| Val | holdout from **train** videos (~15%) |
| Eval | **eval** clips only |

```bash
python -u src/training/train.py --recreate-dataset
python src/training/train.py --prepare-only
```

Apple Silicon: MPS `unique()` workaround, `cache=disk`, auto batch.

### Wall time (historical MPS)

| Stage | Wall time |
| ----- | --------- |
| Old full-frame autolabel (6 clips) | ~1 h |
| Train 15 ep @1280 (legacy) | ~12–13 h |
| Eval 2 clips + videos | ~7 m |

---

## 7. Evaluation & historical metrics

Evaluate on **prepared packs**, separately for each GT source (never mix):

```bash
python src/training/prepare_eval.py --from-autolabel   # data/datasets/eval_autolabel/
python src/training/prepare_eval.py                   # data/datasets/eval_manual/ (CVAT)
python src/training/prepare_eval.py --from-autolabel --scale-adapt  # scale-matched diagnostic

python src/training/evaluate.py --gt autolabel           # → outputs/eval_autolabel/
python src/training/evaluate.py --gt manual              # → outputs/eval_manual/
python src/training/evaluate.py --gt autolabel_adapted   # → outputs/eval_autolabel_adapted/
python src/training/evaluate.py --gt both                # autolabel + manual, separate dirs
```

Pack images are already tiled/cropped; predict directly, IoU 0.5 vs pack labels. Optional legacy full-video path: `--live`.

| Band | Clip |
| ---- | ---- |
| 0–200 m | `13722965_2160_3840_30fps` |
| 200–400 m | `266987` |

### Historical results (2026-07-14, yolov8n, pseudo-GT)

| Metric | 0–200 m | 200–400 m |
| ------ | ------- | --------- |
| Detection rate | 65.6% | 42.4% |
| Precision | 64.5% | 84.3% |
| mAP@0.5 | 51.3% | 42.5% |

**Caveat:** those numbers used YOLO-World as GT. After CVAT labels, build `eval_manual` and `--gt manual`.

Reports: `outputs/metrics_table.md`, `outputs/eval_metrics.json`, videos in `outputs/eval_videos/`.

---

## 8. Ideas for improvement

- CLAHE A/B on preprocess/autolabel (`--enhance`)  
- Stronger tile NMS / conf on close band (cut FPs)  
- Confirmed-only or confidence-weighted train tiles  
- Live drone: smaller imgsz / fewer tiles / faster model than tiled @1280  

---

## Script index (extended)

| Script | Role |
| ------ | ---- |
| `src/data/extract_frames.py` | Videos → `data/frames/` |
| `src/data/preprocess_clips.py` | Probe → `clip_tiling.json` |
| `src/labeling/cvat_pull.py` | CVAT → `labels/` |
| `src/labeling/autolabel.py` | Demo YOLO-World |
| `src/labeling/compare.py` | Autolabel vs manual IoU report |
| `src/training/train.py` / `evaluate.py` | Main train / eval |
| `src/training/prepare_baseline.py` | `baseline_v0` / `v1` packs |
| `src/training/prepare_eval.py` | `eval_manual` / `eval_autolabel` packs |
| `src/training/datasets/generate_variant.py` | Ablation packs |
| `src/training/experiments/run_dataset_round.py` | Round 1 |
| `src/training/experiments/run_model_round.py` | Round 2 |
