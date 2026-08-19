# Guideline — labeling, datasets, experiments, metrics

Companion to **[README.md](README.md)** (PoC pipeline). This document is the **manual-label** path: CVAT GT in `labels/`, dataset ablations, eval packs, experiment rounds.

**Vehicle definition** (what is / is not a vehicle) lives in **[README.md](README.md)**.

## Where things are saved


| Artifact                  | Path                                                 | How                                                         |
| ------------------------- | ---------------------------------------------------- | ----------------------------------------------------------- |
| Source videos             | `data/train/*.mp4`, `data/eval/*.mp4`                | Git LFS / local                                             |
| Frames                    | `data/frames/{clip}/*.jpg`                           | `extract_frames.py`                                         |
| Preprocess config         | `config/clip_tiling.json`                            | `preprocess_clips.py`                                       |
| Preprocess report         | `debug/preprocess_probe.json`                        | same                                                        |
| **CVAT GT (human)**       | `labels/{train|eval}/{clip}/*.txt`                   | `cvat/cvat_pull.py --sync-labels`                           |
| CVAT raw export           | `data/cvat/`                                         | `cvat/cvat_pull.py` (before sync)                           |
| Dataset packs             | `data/datasets/{name}/`                              | `prepare_baseline.py` / `generate_variant.py` (blobs local) |
| Pack settings             | `config/datasets/variants.yaml`                      | git-tracked                                                 |
| Round settings            | `config/experiments/*.yaml`                          | git-tracked                                                 |
| Eval packs                | `data/datasets/eval_manual/`, `eval_manual_adapted/` | `prepare_eval.py`                                           |
| Experiment runs + metrics | `outputs/experiments/`                               | `run_*_round.py` (local)                                    |
| Train runs                | `outputs/runs/…/weights/`                            | `train.py`                                                  |
| Checkpoints               | `checkpoints/`                                       | copy deliverable `best.pt`                                  |
| Roboflow packs (optional) | `labelling/roboflow/`                                | prepare_* scripts                                           |


`labels/` is CVAT only.

---



## 1. Preprocess: tiles, size, distance, `frame_step`

Same probe as the PoC path — full write-up in **[README.md](README.md)** §1 (`python src/data/preprocess_clips.py` → `config/clip_tiling.json`).

Range is **not** GPS altitude. A **4.5 m** standard car plus assumed 24 mm / 1-inch camera converts bbox long-side to metres (`distance_m = 4.5 × focal_px / bbox_long_side_px`), then bins **<200 m / 200–400 m / ≥400 m**. Eval columns 0–200 m and 200–400 m are those bins on clips `13722965…` and `266987`.

---



## 2. Manual labeling

Commands and tool layout: **[src/labeling/README.md](src/labeling/README.md)** (`cvat/` and `roboflow/` subfolders).

### Layout

```
labels/{train|eval}/{clip}/{frame_stem}.txt     ← CVAT (cvat_pull --sync-labels)
data/frames/{clip}/{frame_stem}.jpg             ← must match stems
```

Empty `.txt` = no vehicles. Use preprocess `frame_step` as annotation density guide.

```bash
python src/labeling/cvat/cvat_pull.py --verify --sync-labels
```

CVAT GT for this path lives only in `labels/`. Optional Roboflow packs under `labelling/roboflow/`.

### Unusable clip: `8968356-hd_1920_1080_30fps`

Preprocess probes it, then sets `"skip": true` when median size < 32 px (~17 px here). Rationale (PoC time): the clip is out-of-distribution vs both train and eval scales — see README. Train / eval honor the flag (`--include-skipped` to force).

### Mean object size (historical, 2026-07-13)

Long side in full-frame pixels (from an older label-box dump; sizes also live in `clip_tiling.json` after preprocess as `object_size_px_median`):


| Split | Video       | Mean (px) | Median |
| ----- | ----------- | --------- | ------ |
| train | `3405804…`  | 152.2     | 137.2  |
| train | `5382494…`  | 64.8      | 53.1   |
| train | `8457857…`  | 161.3     | 107.6  |
| train | `8968356…`  | 22.2      | 17.2   |
| eval  | `13722965…` | 347.7     | 368.8  |
| eval  | `266987`    | 135.0     | 133.2  |




### Suggested `frame_step`

Stored per clip in `clip_tiling.json` after preprocess:


| Clip        | step @0.5 (approx) |
| ----------- | ------------------ |
| `3405804…`  | 18                 |
| `5382494…`  | 9                  |
| `8457857…`  | 5                  |
| `8968356…`  | 2                  |
| `13722965…` | 19                 |
| `266987`    | 3                  |


Recompute: `python src/data/preprocess_clips.py`.

---



## 3. Dataset packs (local blobs; settings in git)

Under `data/datasets/` — **blobs local only** (only README / `.gitignore` tracked). Specs: `config/datasets/variants.yaml` (git).

All packs below use **CVAT** `labels/`.

### Helpers


| Pack                  | Source                 | Command                                             |
| --------------------- | ---------------------- | --------------------------------------------------- |
| `baseline_v1`         | `labels/` (CVAT)       | `python src/training/prepare_baseline.py`           |
| `eval_manual`         | CVAT `labels/eval/`    | `python src/training/prepare_eval.py`               |
| `eval_manual_adapted` | CVAT, pad→smaller cars | `python src/training/prepare_eval.py --scale-adapt` |


Same `train_groups` tiling / `frame_step`, **eval clips only**. Build once; reuse for every experiment. Eval packs cap each clip at **64** frames after `frame_step` (`--max-frames-per-clip`; `0` disables).

`--scale-adapt` **copies** the native pack, then pads only oversized clips (mid-band stays byte-identical). Build native first, then adapt.

### Ablation packs (`config/datasets/variants.yaml`)

Defaults: **CVAT** `labels/`, `frame_step` only, imgsz **1024**. AUTO tiling from `train_groups`: B_medium tile **1024** / overlap **0.2**; A_close = full frame.

```bash
python src/training/datasets/generate_variant.py --list
python src/training/datasets/generate_variant.py --all
```


| id                       | Change vs `baseline_1`                            |
| ------------------------ | ------------------------------------------------- |
| `baseline_1`             | AUTO tiling, frame_step, drop empty tiles         |
| `variant_2_no_tiling`    | no tiling (full frame → letterbox)                |
| `variant_3_tiling_fixed` | tile 640 / overlap 0.1                            |
| `variant_4_low_overlap`  | AUTO tiles, overlap 0.1 (+ multi-tile bbox stats) |
| `variant_5_aug`          | reuse `baseline_1` tiles; mosaic_hsv at **train** |
| `variant_6_strided`      | every 5th frame_step frame (**Round 2 train pack**) |
| `variant_7_negatives`    | keep empty tiles @ 15% of positives               |


See `data/datasets/README.md`, `config/datasets/README.md`.

---



## 4. Dataset round (commands)

Fixed train protocol = README PoC schedule on **manual** labels: YOLO11s, imgsz 1024, Stage 1 = 2 epochs (backbone frozen), Stage 2 = 5 epochs (backbone unfrozen), patience 3. Only the train pack changes. Every run is scored on **both** eval packs. Rank packs by **native mAP@0.5** (find the car, IoU ≥ 0.5), mean of A and B. mAP@0.5:0.95 is a box-quality diagnostic, not the winner rule.


| Setting        | Value                                                               |
| -------------- | ------------------------------------------------------------------- |
| Model          | `yolo11s.pt`                                                        |
| imgsz          | 1024                                                                |
| Stage 1        | 2 epochs, backbone frozen (`freeze=11`)                             |
| Stage 2        | 5 epochs, backbone **unfrozen**                                     |
| patience / lr0 | 3 / 0.01                                                            |
| Aug            | PoC (HSV + flips + degrees=180, mosaic off), except `variant_5_aug` |
| Labels         | `labels/` (CVAT)                                                    |
| Eval           | `eval_manual` + `eval_manual_adapted`                               |




### 1. Eval packs (once)

```bash
python src/training/prepare_eval.py                 # → data/datasets/eval_manual/
python src/training/prepare_eval.py --scale-adapt   # → data/datasets/eval_manual_adapted/
```



### 2. Train packs

```bash
python src/training/datasets/generate_variant.py --list
python src/training/datasets/generate_variant.py --all
```

The round runner also builds a pack if it is missing.

### 3. Train + eval all variants

Default (no flags) = **full round from scratch** (retrains every pack). `--resume` skips finished ones and continues incomplete train/eval.

```bash
python src/training/experiments/run_dataset_round.py
python src/training/experiments/run_dataset_round.py --resume
python src/training/experiments/run_dataset_round.py --variant baseline_1
```

After each variant: MPS/CPU cache is cleared; `summary.md` is rewritten.

### Eval packs (same for every train variant)

Training uses the ablation pack (`baseline_1`, `variant_2_…`). **Eval does not** — every weights file is scored on the same two **eval-clip** packs (CVAT `labels/eval/`, not train videos):


| Pack          | Path                                 | What                                                                |
| ------------- | ------------------------------------ | ------------------------------------------------------------------- |
| Native        | `data/datasets/eval_manual/`         | Clips `13722965_2160_3840_30fps` (0–200 m) and `266987` (200–400 m) |
| Scale-adapted | `data/datasets/eval_manual_adapted/` | Same stems; close clip padded so cars look smaller                  |


In-training val is a 15% holdout from **train** videos (inside each pack). That is not the reported eval.

### Where metrics land

All under `outputs/experiments/dataset_round/` (local, not git):


| Path                                         | What                                        |
| -------------------------------------------- | ------------------------------------------- |
| `ds_<id>/weights/best.pt`                    | Stage 2 weights for that pack               |
| `ds_<id>_stage1/`                            | Stage 1 run                                 |
| `<id>/eval_manual/eval_metrics.json`         | Native-scale eval                           |
| `<id>/eval_manual/metrics_table.md`          | Same, markdown                              |
| `<id>/eval_manual_adapted/eval_metrics.json` | Scale-adapted eval                          |
| `<id>/eval_manual_adapted/metrics_table.md`  | Same, markdown                              |
| `ds_<id>_result.json`                        | Train plan + paths to both evals            |
| `round_timing.json`                          | Wall time per variant and per session       |
| `logs/ds_<id>.log`                           | Full stdout/stderr for that pack            |
| `logs/sessions/dataset_*.log`                | Session index (what ran, exit codes)        |
| `summary.md`                                 | Comparison table (both evals, all variants) |
| `summary.json`                               | Same numbers as JSON                        |


Runs do **not** overwrite `outputs/runs/yolo11s_vehicle/` or `checkpoints/yolo11s_vehicle_best.pt`.

### Round 1 results (2026-08-14)

YOLO11s `--prototype` on the seven CVAT packs above. Same eval packs for every run (`eval_manual` / `eval_manual_adapted`, 49 close + 43 mid frames). Source: `outputs/experiments/dataset_round/summary.md`.

**0–200 m** (`13722965_2160_3840_30fps`). Native vs scale-adapted (pad close cars toward train size).


| variant                  | mAP@0.5 | mAP@0.5:0.95 | Det   | P     | FA/min | mAP@0.5 adapted | Det adapted | P adapted |
| ------------------------ | ------------------------- | ------------------------------ | ----- | ----- | ------ | --------------------------------- | ----------- | --------- |
| `baseline_1`             | 62.5%                     | 25.5%                          | 99.2% | 37.9% | 7237   | 86.5%                             | 97.5%       | 49.4%     |
| `variant_2_no_tiling`    | 84.1%                     | 26.1%                          | 97.5% | 56.5% | 3343   | 88.6%                             | 94.2%       | 79.2%     |
| `variant_3_tiling_fixed` | 78.8%                     | 45.3%                          | 90.1% | 71.7% | 1580   | —                                 | 0.0%        | —         |
| `variant_4_low_overlap`  | 63.6%                     | 23.6%                          | 90.1% | 58.6% | 2829   | 86.2%                             | 96.7%       | 59.1%     |
| `variant_5_aug`          | 74.5%                     | 44.2%                          | 91.7% | 39.1% | 6355   | 78.7%                             | 91.7%       | 57.5%     |
| `variant_6_strided`      | 84.6%                     | 33.8%                          | 91.7% | 66.5% | 2057   | 75.3%                             | 97.5%       | 27.9%     |
| `variant_7_negatives`    | 78.3%                     | 25.6%                          | 98.3% | 41.5% | 6171   | 76.8%                             | 95.9%       | 43.4%     |


**200–400 m** (`266987`). Scale-adapt does not change this clip (numbers match native).


| variant                  | mAP@0.5 | mAP@0.5:0.95 | Det   | P     | FA/min |
| ------------------------ | ------------------------- | ------------------------------ | ----- | ----- | ------ |
| `baseline_1`             | 89.0%                     | 21.1%                          | 93.0% | 85.5% | 377    |
| `variant_2_no_tiling`    | 78.9%                     | 17.2%                          | 87.7% | 82.0% | 460    |
| `variant_3_tiling_fixed` | —                         | —                              | 0.0%  | —     | 0      |
| `variant_4_low_overlap`  | 79.6%                     | 17.1%                          | 89.5% | 85.0% | 377    |
| `variant_5_aug`          | 67.2%                     | 43.5%                          | 78.9% | 80.4% | 460    |
| `variant_6_strided`      | 90.1%                     | 27.7%                          | 96.5% | 46.2% | 2679   |
| `variant_7_negatives`    | 79.9%                     | 14.9%                          | 87.7% | 87.7% | 293    |


`variant_3_tiling_fixed` (tile 640) produced no mid-band detections; adapted close-band also collapsed.

**Wall time (reconstructed):** compute **2h 48m** (sum of the seven packs). Two sessions — 13 Aug ~2h 2m (`baseline_1`…`variant_4`), 14 Aug ~46m (`variant_5`…`variant_7`). Live timer was not on for Round 1; numbers come from run-dir mtimes + Ultralytics `results.csv`. Going forward each round writes `round_timing.json` and a **Wall time** section in `summary.md`.

**Round 2 train pack:** `variant_6_strided` (every 5th `frame_step` frame, AUTO tiling). Primary metric is **native mAP@0.5** (find the car, IoU ≥ 0.5), mean of A and B — not mAP@0.5:0.95 (box tightness). Highest on both bands: **84.6%** A, **90.1%** B. mAP@0.5:0.95 stays in the tables as a diagnostic. Watch B precision (46.2%) and FA/min (2679) in later rounds.

### Round 1 — practical conclusions

Decision metric = **native mAP@0.5**, mean of A and B. Adapted A says “is the close-clip failure just scale?”

| Pack | What we tested | Takeaway |
| ---- | -------------- | -------- |
| `baseline_1` | AUTO tiles, every `frame_step` frame, drop empties | Solid B (89.0%). Native A only 62.5% jumps to 86.5% when the close clip is padded — the model can do the cars, not the native scale. High A FA (7237/min). Default pack, not the winner. |
| `variant_2_no_tiling` | Full frame → letterbox | Best **large-object** pack after stride: native A 84.1%, P 56.5%. B drops to 78.9% (far cars shrink in the letterbox). Use if the product is close-range only. |
| `variant_3_tiling_fixed` | Tile 640 / overlap 0.1 on every clip | **Do not use.** B Det 0%. Adapted A also dies. 640 px eats mid-band cars. The high A 0.5:0.95 (45.3%) is tight boxes on the few large cars that still fit — not a win. |
| `variant_4_low_overlap` | AUTO tiles, overlap 0.2 → 0.1 | Same story as baseline, a bit worse (A 63.6 / B 79.6). Overlap is not the lever on this data. |
| `variant_5_aug` | Same tiles as baseline; mosaic on at train | Wins **box tightness** (0.5:0.95 44.2 / 43.5) and loses **find-the-car** (B 67.2%). Mosaic is the wrong knob if the product question is “did we see the car?”. |
| `variant_6_strided` | Every 5th `frame_step` frame | **Winner.** Native 84.6% A / 90.1% B. Sparse frames cut near-duplicates and still cover the clip. Cost: B P 46.2%, FA 2679/min — the operator sees a lot of extra boxes. Train pack for Round 2. |
| `variant_7_negatives` | Keep empty tiles @ 15% of positives | Best B precision (87.7%) and lowest B FA (293). Mean mAP@0.5 loses to stride (~79 vs ~87). Take this if spam hurts more than a few missed cars. |

**What to keep from Round 1:** AUTO tiling (not 640, not “no tile” as the only pack). Stride the labels. Leave mosaic off for the mAP@0.5 product. Revisit negatives only if FA/min becomes the blocker.

---



## 5. Round 2 — model & optimization

Fixed **train pack** = Round 1 winner **`variant_6_strided`**. Vary architecture / LR / NMS / backbone / P2 head. Eval bands:


| Band           | Clip                       |
| -------------- | -------------------------- |
| **A** (<200 m) | `13722965_2160_3840_30fps` |
| **B** (>200 m) | `266987`                   |


Same two eval packs as Round 1 (`eval_manual` + `eval_manual_adapted`). Rank on **native mAP@0.5** (mean of A and B). mAP@0.5:0.95 is reported but not the winner rule.

Config: `config/experiments/model_round.yaml`. Artifacts: `outputs/experiments/model_round/` (`round_timing.json`, `logs/md_<id>.log`, `logs/sessions/model_<group>_*.log`, Wall time in `summary.md`).


| Group | What                                                                | Runs |
| ----- | ------------------------------------------------------------------- | ---- |
| **A** | YOLOv8s / YOLO11s / YOLO26s × LR {×0.5, ×1, ×2}                     | 9    |
| **B** | Fine LR around the winner: ×[0.8, 1.0, 1.25]                        | 3    |
| **C** | Hard NMS vs no NMS vs Soft-NMS (Gaussian); **eval only**            | 3–6  |
| **D** | Pretrained full / frozen-head / from-scratch                        | 3    |
| **E** | Standard head vs P2 small-object head                               | 2    |


Group A/B/E: COCO-pretrained **backbone is always trained** (Stage 2 unfreezes). The only exceptions are Group D: frozen-head and from-scratch.

**Winner rule (after A):** if top-2 YOLO mean mAP@0.5 differ by **≥ 1.5 pts**, take the top one into B, C, D, E (skip C on the second model). If the gap is smaller, run Group C on both, then pick one for D/E. Group C is eval-only: same raw boxes, then hard NMS (`iou=0.7`), no NMS, or Gaussian Soft-NMS (`σ=0.5`).

```bash
# Group A (default — 9 runs)
python src/training/experiments/run_model_round.py
python src/training/experiments/run_model_round.py --resume
python src/training/experiments/run_model_round.py --list
python src/training/experiments/run_model_round.py --pick-winner

# After A (replace WINNER with the printed id)
python src/training/experiments/run_model_round.py --group B --winner WINNER
python src/training/experiments/run_model_round.py --group C --winner WINNER
python src/training/experiments/run_model_round.py --group C --winner WINNER --tie-candidate OTHER
python src/training/experiments/run_model_round.py --group D --winner WINNER
python src/training/experiments/run_model_round.py --group E --winner WINNER
```

`--dataset` overrides the yaml pack. YOLO26 P2 uses Ultralytics `yolo26s-p2.yaml`; YOLOv8 uses `yolov8s-p2.yaml`; YOLO11 uses `config/models/yolo11s-p2.yaml`.

Apple Silicon: YOLO26 and P2 skip in-train val (including last-epoch / final val) and eval packs on CPU. YOLO26: Metal GatherND. P2 at imgsz=1024: MPS DFL ``output channels > 65536``. Variants run in child processes. Resume with `--resume`. Logs: `outputs/experiments/model_round/logs/`.

### Round 2 results (2026-08-15)

Fixed pack `variant_6_strided`. Winner after A: **`yolov8s_lr_default`** (mean native mAP@0.5 **72.8%**, gap to YOLO11s 5.1 pts ≥ 1.5 → B/C/D/E on v8 only). Source: `outputs/experiments/model_round/summary.md`. Compute **2h 38m**.

Within an architecture the three coarse LRs often printed **identical** evals (11s all three; 26s all three; v8 ×0.5 = ×2). Weights files differ; Ultralytics `optimizer=auto` ignored `lr0` on several runs. Treat those as one architecture sample, not three independent LR points.

**Native `eval_manual`** (decision tables). Adapted A is in the notes when it flips the story.

**Group A — architecture × coarse LR**

| id | mean | A mAP@0.5 | B mAP@0.5 | Det A / B | P A / B | FA/min A / B |
| -- | ---: | --------: | --------: | --------- | ------- | ------------ |
| `yolov8s_lr_default` | **72.8%** | 62.4% | **83.2%** | 95.0 / 91.2 | 39.7 / 75.4 | 6429 / 712 |
| `yolov8s_lr_x0.5` / `_x2` | 61.2% | 49.5% | 72.8% | 82.6 / 84.2 | 59.9 / 69.6 | 2461 / 879 |
| `yolo11s_lr_*` (all 3) | 67.7% | **77.1%** | 58.3% | 85.1 / 63.2 | 77.4 / 80.0 | 1102 / 377 |
| `yolo26s_lr_*` (all 3) | 52.4% | 16.1% | **88.8%** | 39.7 / 93.0 | 43.2 / 85.5 | 2314 / 377 |

Adapted A: v8 default **89.0%**, 11s 78.6%, 26s **88.5%**. YOLO26’s native-A collapse is mostly scale (close cars too big); padded, it matches v8 on A.

**Group B — fine LR ×[0.8, 1.0, 1.25] around `yolov8s_lr_default`**

| id | mean | A | B |
| -- | ---: | -: | -: |
| `…_lr_0p8` / `_1p0` / `_1p25` | 72.8% | 62.4% | 83.2% |

Same numbers as Group A default (P, FA, Det included).

**Group C — eval only, same `yolov8s_lr_default` weights**

| id | mean | A | B | Det A / B | P A / B | FA/min A / B |
| -- | ---: | -: | -: | --------- | ------- | ------------ |
| `…_nms_on` (hard, iou=0.7) | 72.8% | 62.4% | 83.2% | 95.0 / 91.2 | 40.1 / 75.4 | 6318 / 712 |
| `…_nms_soft` (Gaussian σ=0.5) | 72.8% | 62.4% | 83.1% | 92.6 / 93.0 | 47.9 / 62.4 | 4482 / 1339 |
| `…_nms_off` | 30.6% | 23.1% | 38.1% | 95.9 / 98.2 | 7.4 / 10.2 | 53376 / 20678 |

**Group D — backbone (longer budget: warmup 5 + 20)**

| id | mean | A | B | Det A / B | P A / B | FA/min A / B |
| -- | ---: | -: | -: | --------- | ------- | ------------ |
| `…_bb_frozen` | **77.4%** | 65.6% | **89.3%** | 87.6 / 93.0 | 48.0 / 88.3 | 4224 / 293 |
| `…_bb_full` | 32.5% | 48.1% | 16.9% | 82.6 / 45.6 | 31.4 / 36.6 | 8008 / 1884 |
| `…_bb_scratch` | 10.4% | 4.5% | 16.3% | 8.3 / 26.3 | 18.9 / 44.1 | 1580 / 795 |

Adapted A: frozen **90.2%** (best close-clip number in the round). Full 33.5%. Scratch 9.1%.

**Group E — head (same longer budget as D full)**

| id | mean | A | B | Det A / B |
| -- | ---: | -: | -: | --------- |
| `…_head_std` | 32.5% | 48.1% | 16.9% | 82.6 / 45.6 |
| `…_head_p2` | 2.3% | 4.5% | 0.0% | 7.4 / 0.0 |

`head_std` matches `bb_full` (same recipe). P2 trained at imgsz 1024 with in-train val off on MPS; eval on CPU.

### Round 2 — practical conclusions

| Run | What we tested | Takeaway |
| --- | -------------- | -------- |
| `yolov8s_lr_default` | v8, lr0=0.01, staged 5+15 | **Group A winner.** Mean 72.8% because B is strong (83.2%). Native A 62.4% is the weak side (adapted A 89% — again scale, not “v8 cannot see cars”). High A FA (6429). Carry this id into B–E. |
| `yolov8s_lr_x0.5` / `_x2` | v8, lr0=0.005 / 0.02 | Both **61.2%**, worse than default. Coarse LR *did* move v8, and the middle value won. Do not ship ×0.5/×2. |
| `yolo11s_lr_*` | 11s × three LRs | All three identical (67.7%). **Best native A** (77.1%) and cleanest FA (1102 / 377). Loses the round on B (58.3%). Pick 11s only if the product is &lt;200 m. |
| `yolo26s_lr_*` | 26s × three LRs | All three identical (52.4%). Best raw B (88.8%) and **dead native A** (16.1%, Det 40%). Adapted A 88.5% — NMS-free head hates huge close boxes, not the far band. Not a dual-band model. |
| `…_lr_0p8` / `_1p0` / `_1p25` | Fine LR around 0.01 | **No change** vs A default. Another LR sweep around 0.01 is wasted on this budget. |
| `…_nms_on` | Hard NMS iou=0.7 | Same as the A checkpoint. Keep this as the default decode. |
| `…_nms_soft` | Gaussian Soft-NMS | Same mAP@0.5 as hard. A P up (48% vs 40%), B P down, B FA up (1339). Not a reason to switch; optional if close-clip spam is the complaint. |
| `…_nms_off` | No NMS | Det looks great (96–98%) because everything is kept; mAP collapses (30.6%) and FA explodes (53k / 21k). Never ship. |
| `…_bb_frozen` | COCO backbone frozen, head only | **Best model in the whole round** on the decision metric: mean **77.4%** (A 65.6 / B 89.3), B P 88.3%, B FA 293. Longer unfreeze is not “more better” — leaving ImageNet/COCO features alone beat Group A. Strong candidate to ship. |
| `…_bb_full` | Same v8, staged 5+20 unfreeze | Mean **32.5%**, B 16.9%. Extra epochs on the backbone *overwrote* the useful COCO features. Same recipe as E `head_std`. Do not use the long unfreeze. |
| `…_bb_scratch` | Random yaml init | Mean 10.4%. This dataset is too small to train a detector from scratch. Always start from COCO. |
| `…_head_std` | Standard head, D-full budget | Duplicate of `bb_full` (32.5%). Confirms the long unfreeze, not the head, is the problem. |
| `…_head_p2` | P2/4–P5/32 yaml, imgsz 1024 | **Failure** (mean 2.3%, B Det 0%). Extra small-object head did not learn on this pack/budget; MPS also cannot val P2 at 1024 (DFL &gt;65536 channels — eval is CPU-only). Do not take P2 forward. |

**What to keep from Round 2**

1. **Architecture:** YOLOv8s for both bands. YOLO11s if A is the only product band. YOLO26 only as a far-band specialist.
2. **Optimization:** `lr0=0.01` is enough; fine LR and Soft-NMS did not move mAP@0.5. Keep **hard NMS**.
3. **Backbone:** prefer **frozen COCO + trainable head** (`bb_frozen`) over a long Stage-2 unfreeze. Group A’s shorter unfreeze (5+15) was OK; 5+20 full was harmful.
4. **Head:** standard. P2 is out.
5. **Honest gap vs Round 1:** Round 1 stride + YOLO11s `--prototype` posted 84.6 / 90.1. Round 2 v8 on the same pack is 62.4 / 83.2 (A default) or 65.6 / 89.3 (frozen). Different train length and architecture — do not treat R1 11s and R2 v8 as one curve. If the next step is “best so far”, compare **R1 `variant_6_strided` + 11s prototype** vs **R2 `bb_frozen`** on the same protocol before locking a deliverable.

---



## 6. Training details

Main train (`train.py`): **YOLO11s**, staged head-then-full fine-tune.


| Setting     | Value                                                            |
| ----------- | ---------------------------------------------------------------- |
| Base        | `yolo11s.pt`                                                     |
| Epochs      | Stage1 5 (freeze) + Stage2 20; `--prototype` = 2 + 5, patience 3 |
| Crops       | `train_groups` in `clip_tiling.json`                             |
| Aug (main)  | HSV + flips + degrees=180; **mosaic off**                        |
| Deliverable | `checkpoints/yolo11s_vehicle_best.pt`                            |




### Train groups


| Group      | Videos      | tile_size         | overlap | train_imgsz |
| ---------- | ----------- | ----------------- | ------- | ----------- |
| `C_far`    | *(none)*    | 768               | 0.20    | 1024        |
| `B_medium` | `5382494`   | 1024              | 0.20    | 1024        |
| `A_close`  | close clips | null (full frame) | —       | 1024        |



| Stage | Input                                |
| ----- | ------------------------------------ |
| Train | group tile → letterbox `--imgsz`     |
| Val   | holdout from **train** videos (~15%) |
| Eval  | **eval** clips only                  |


```bash
python -u src/training/train.py --dataset-dir data/datasets/baseline_v1 --prototype
```

The per-batch Ultralytics line (`17/20 … 1/14 5.7s/it`) is a tqdm heartbeat, not the epoch metric. Mute it for a session with `TQDM_DISABLE=1` and/or `YOLO_VERBOSE=False` (set before Python starts). Full stdout is already in `outputs/experiments/…/logs/md_<id>.log` — `tail -f` that file instead of scrolling the bar. What the columns mean and when to keep the bar on: [unified guide §26](guides/unified_cv_ds_engineer_guide.md#26-робочі-звички-логи-прогрес-бари-як-читати-тренування).

Apple Silicon: MPS `unique()` workaround, `cache=disk`, auto batch.

---



## 7. Evaluation

Score a prepared pack (already tiled/cropped). Matching uses **IoU 0.5** vs pack labels.

**How to read the numbers** (same language as the unified guide §1):

**mAP@0.5** is mean precision where a pred counts as a hit only if IoU with GT ≥ 0.5. The box can be coarse — it just has to cover the object more than halfway.

**mAP@0.5:0.95** is the same, averaged over 10 IoU thresholds: 0.50, 0.55, …, 0.95. To hold up at 0.75–0.95 the box has to almost match the label.

That is why Round 1 **mosaic** (`variant_5_aug`) won 0.5:0.95 (tighter boxes) and **stride** (`variant_6_strided`) won 0.5 (more coarse hits, worse geometry, many FP). For the product question “did we find the car at all?” use **0.5**. For box quality use **0.5:0.95**. Rounds rank on **native mAP@0.5** (mean of A and B). 0.5:0.95 stays in the tables as a diagnostic.

Other columns are at **one** confidence and IoU=0.5 (not averaged like mAP):

| Column | In practice |
| ------ | ----------- |
| **Det** | Share of GT cars we caught. Ignores extra boxes. Misses, not spam. |
| **P** | Share of drawn boxes that are really a car. Low P = operator drowning in FP. Stride on B: mAP@0.5 90% but P 46%. |
| **FA/min** | Extra boxes per minute of video. Baseline B ≈ 377; stride B ≈ 2679. Same “found it”, different human load. |
| **Time to first det** | Seconds until the first true box. “How soon did we notice?”, not “how tight is the box”. |
| **Native vs adapted** | Native = clip as-is (product). Adapted = pad the close clip so cars look smaller, like train; mid clip unchanged. Separates “can’t do large” from “wrong scale vs train”. |

**A** (&lt;200 m) = `13722965…`, **B** (&gt;200 m) = `266987`. Distance is not GPS — see [README.md](README.md) §1. One pooled mAP would hide a far-band failure.

```bash
python src/training/prepare_eval.py                 # data/datasets/eval_manual/
python src/training/prepare_eval.py --scale-adapt   # data/datasets/eval_manual_adapted/

python src/training/evaluate.py --gt manual              # → outputs/eval_manual/
python src/training/evaluate.py --gt manual_adapted      # → outputs/eval_manual_adapted/
```

The dataset round writes the same metrics under `outputs/experiments/dataset_round/<id>/eval_*` so runs do not clobber each other.

---



## Script index (extended)


| Script                                          | Role                                          |
| ----------------------------------------------- | --------------------------------------------- |
| `src/data/extract_frames.py`                    | Videos → `data/frames/`                       |
| `src/data/preprocess_clips.py`                  | Probe → `clip_tiling.json`                    |
| `src/labeling/cvat/cvat_pull.py`                | CVAT → `labels/`                              |
| `src/labeling/roboflow/`                        | Optional Roboflow prepare / upload / import   |
| `src/training/train.py` / `evaluate.py`         | Main train / eval                             |
| `src/training/prepare_baseline.py`              | `baseline_v1` pack from `labels/`             |
| `src/training/prepare_eval.py`                  | `eval_manual` / `eval_manual_adapted`         |
| `src/training/datasets/generate_variant.py`     | Ablation packs from `labels/`                 |
| `src/training/experiments/run_dataset_round.py` | Round 1 (YOLO11s `--prototype` on each pack)  |
| `src/training/experiments/run_model_round.py`   | Round 2 (Group A default; B–E via `--winner`) |


---



## 8. Winners vs the README PoC run

Three checkpoints, same two eval clips (`13722965…` = A &lt;200 m, `266987` = B &gt;200 m). They are **not** one ranking: the PoC is scored on **autolabel** GT, the rounds on **CVAT manual** GT. Same frames, different “what is a car” — do not subtract 12% from 85% and call it a model gain.

| | README PoC | Round 1 winner | Round 2 Group A winner | Round 2 best (all groups) |
| --- | --- | --- | --- | --- |
| Id | `yolo11s_vehicle` | `variant_6_strided` | `yolov8s_lr_default` | `yolov8s_lr_default_bb_frozen` |
| Weights | `checkpoints/yolo11s_vehicle_best.pt` | `outputs/experiments/dataset_round/ds_variant_6_strided/weights/best.pt` | `…/model_round/md_yolov8s_lr_default/weights/best.pt` | `…/md_yolov8s_lr_default_bb_frozen/weights/best.pt` |
| Labels | YOLO-World autolabel | CVAT `labels/` | CVAT | CVAT |
| Train pack | `baseline_v0` (every `frame_step`, AUTO tiles) | `variant_6_strided` (every 5th `frame_step`) | same as R1 | same as R1 |
| Model | YOLO11s | YOLO11s | YOLOv8s | YOLOv8s |
| Schedule | `--prototype` 2+5, patience 3 | same 2+5 | staged 5+15 | head only, 5+20, backbone **frozen** |
| Eval GT | `eval_autolabel` | `eval_manual` | `eval_manual` | `eval_manual` |
| Date | 2026-08-12 | 2026-08-14 | 2026-08-15 | 2026-08-15 |

Sources: [README.md](README.md) §3 (PoC eval tables); this file §4 / §5.

### Native mAP@0.5 (decision metric)

| Run | Eval GT | A | B | mean A+B | A adapted |
| --- | --- | ---: | ---: | ---: | ---: |
| PoC YOLO11s | autolabel | 12.2% | 48.2% | 30.2% | 42.5% |
| R1 `variant_6_strided` + 11s | manual | **84.6%** | **90.1%** | **87.4%** | 75.3% |
| R2 `yolov8s_lr_default` | manual | 62.4% | 83.2% | 72.8% | 89.0% |
| R2 `bb_frozen` | manual | 65.6% | 89.3% | 77.4% | **90.2%** |

### Rest of the native table

| Run | Det A / B | P A / B | FA/min A / B | mAP@0.5:0.95 A / B |
| --- | --- | --- | --- | --- |
| PoC (autolabel GT) | 50.0 / 67.9 | 21.7 / 66.7 | 5180 / 900 | 6.9 / 10.7 |
| R1 stride + 11s | 91.7 / 96.5 | 66.5 / 46.2 | 2057 / 2679 | 33.8 / 27.7 |
| R2 v8 default | 95.0 / 91.2 | 39.7 / 75.4 | 6429 / 712 | 14.7 / 18.3 |
| R2 v8 frozen | 87.6 / 93.0 | 48.0 / 88.3 | 4224 / 293 | 17.7 / 16.4 |

### How to read this

**PoC vs Round 1 is mostly labels, not architecture.** Same YOLO11s `--prototype`, same clips. Autolabel GT is noisy (missed cars + junk boxes), so PoC mAP@0.5 is 12% / 48%. Swap in CVAT eval and a strided CVAT train pack and the same recipe posts 85% / 90%. The +70 pt jump on A is “we stopped scoring against a bad teacher,” plus fewer near-duplicate train frames. Holdout val in the README (mAP50 0.75 on autolabel train-video split) is a third number — do not mix it with eval-clip mAP.

**Round 1 vs Round 2 is not a fair bake-off.** R1 winner is 11s, 2+5 epochs. R2 A winner is v8, 5+15. R2 `bb_frozen` is v8, head-only, 25 epochs. On **manual** GT, R1 stride+11s still has the highest native mean (87.4%). R2 frozen is the best *v8* (77.4%) and the cleanest B (P 88.3%, FA 293 vs stride’s 46% / 2679). R2 default v8 is worse on A than R1 11s (62% vs 85%) and noisier on A (FA 6429).

**Adapted A** (pad the close clip): PoC 12% → 42% (scale was part of the PoC miss). R2 v8 62% → 89%. R1 stride *drops* 85% → 75% — that pack already liked native-scale close cars; shrinking them hurts. So “close cars are too big” is a v8/PoC problem, not a stride+11s problem.

### Practical pick

| If you need… | Take |
| --- | --- |
| Highest find-the-car on **manual** GT, both bands | Round 1: `variant_6_strided` + YOLO11s `--prototype` |
| Best **precision / FA** on B, still dual-band | Round 2: `yolov8s_lr_default_bb_frozen` |
| The documented PoC deliverable (autolabel path) | README: `checkpoints/yolo11s_vehicle_best.pt` — expect ~12% / 48% vs autolabel eval, not the R1 numbers |
| A locked product checkpoint | Re-run the two manual winners on **one** schedule (same epochs, same 11s or v8) before copying to `checkpoints/` |

Do not replace the PoC `best.pt` with an R2 file and quote README tables — those tables are autolabel-eval. Do not quote R1 87% mean next to PoC 30% mean as “the model got 3× better” without saying the GT changed.


