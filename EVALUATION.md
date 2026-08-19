# Evaluation — manual labels, pack ablations, experiment rounds

**Read [PoC.md](PoC.md) first.** That doc covers preprocess, autolabel PoC, train/eval mechanics, metrics, and postprocess. Task spec (class `0 = vehicle`): **[README.md](README.md)**. This doc adds only the **manual-label path**: CVAT GT in `labels/`, ablation packs, three experiment rounds, and lock-in.

Shared with PoC.md (not repeated here): `clip_tiling.json` / distance bins / `frame_step`, `train_groups` tiling, eval bands A/B, `--prototype` 2+5 schedule, aug, nested-box NMS drop, metric columns, `prepare_eval` cap-64 sampling.

All round tables = **eval** on `eval_manual` (eval clips A/B), not in-train val. **Decision metric:** mean A+B mAP@0.5.

## 1. Manual labeling (CVAT)

Commands and tool layout: **[src/labeling/README.md](src/labeling/README.md)**.

### Layout

```
labels/{train|eval}/{clip}/{frame_stem}.txt     ← CVAT (cvat_pull --sync-labels)
data/frames/{clip}/{frame_stem}.jpg             ← must match stems
```

Empty `.txt` = no vehicles. Use PoC.md `frame_step` as annotation density guide.

```bash
python src/labeling/cvat/cvat_pull.py --verify --sync-labels
```

### Autolabel vs CVAT (IoU ≥ 0.5)

We transitioned away from autolabels for training because reviews proved they missed too many targets. Manual labeling is essential for this phase, and evaluation comparisons will reflect this difference. Ground truth (GT) for this stage uses manual data exclusively.

```bash
python src/labeling/cvat/compare.py
# → debug/compare_autolabel_vs_cvat.txt
```
CVAT manual labels persist across each frame of video since tracking simplifies labeling for this specific task. Frame steps derived from the PoC configuration will be used for these videos unless an experiment explicitly states otherwise. For the comparison, we use only the frames that are present in the autolabel.

| Clip        | Split | Frames both | CVAT boxes | Auto boxes | Precision | Recall | Mean IoU | TP / FP / FN       | CVAT size med | AUTO size med |
| ----------- | ----- | ----------- | ---------- | ---------- | --------- | ------ | -------- | ------------------ | ------------- | -------------- |
| `3405804…`  | train | 200         | 4112       | 8324       | 36.1%     | 73.1%  | 0.807    | 3006 / 5318 / 1106 | 129.5         | 88.9           |
| `5382494…`  | train | 161         | 1036       | 1621       | 49.9%     | 78.1%  | 0.894    | 809 / 812 / 227    | 43.9          | 52.8           |
| `8457857…`  | train | 203         | 4110       | 7265       | 50.5%     | 89.3%  | 0.754    | 3671 / 3594 / 439  | 61.3          | 50.9           |
| `13722965…` | eval  | 86          | 214        | 291        | 67.4%     | 91.6%  | 0.886    | 196 / 95 / 18      | 384.0         | 298.5          |
| `266987`    | eval  | 456         | 408        | 437        | 87.9%     | 94.1%  | 0.806    | 384 / 53 / 24      | 111.8         | 141.8          |
| `8968356…`  | train | —           | —          | —          | —         | —      | —        | —                  | 17.6          | 17.6           |
| **all**     |       | 1106        | 9880       | 17938      | 45.0%     | 81.6%  | 0.792    | 8066 / 9872 / 1814 |               |                |


The comparison reveals that clip `8457857` disproportionately drives the results, motivating us to better balance clip selection across diverse visual conditions (e.g., angle and distance) rather than relying on unique labels alone.
The size comparison also shows significant variations between manual and automated box dimensions for 4 out of the 6 clips.
`8968356` skipped ([PoC.md](PoC.md)).

---

## 2. Experiment design

Three rounds in order. Each locks one choice into yaml for the next. One factor at a time.

```
Round 1  imgsz        →  lock defaults.imgsz in dataset_round.yaml
Round 2  train pack   →  lock defaults.dataset in model_round.yaml
Round 3  model setup  →  lock recipe in §6
```

Base train for Rounds 1–2: YOLO11s, PoC `--prototype` (2 ep freeze + 5 ep unfreeze). Round 3 relaxes protocol on the locked pack + imgsz.

### Round 1 — imgsz

Sweep Ultralytics letterbox **imgsz** (640 / 768 / 1024 / 1280) on the reference train pack `auto` — AUTO tiling, every `frame_step` frame. Pack blobs stay the same on disk; only train and eval predict resolution changes. YOLO11s `--prototype` 2+5 throughout. 

Winner → `dataset_round.yaml` `defaults.imgsz`. 

Config: `imgsz_round.yaml` · `run_imgsz_round.py`.

### Round 2 — dataset pack

Ablation over **how** CVAT labels become train tiles — tiling vs full frame, temporal stride, empty tiles, clip balancing (motivated by `8457857` box count, autolabel precision on train). 

**imgsz** locked to Round 1 winner; YOLO11s `--prototype` 2+5 on every cell. 

Winner → `model_round.yaml` `defaults.dataset`.

Cells (`config/datasets/variants.yaml`):

| id                      | Change vs `auto`                                    |
| ----------------------- | --------------------------------------------------- |
| `auto`                  | AUTO tiling, every `frame_step` frame, drop empty   |
| `fullframe`             | no tiling (full frame → letterbox)                  |
| `strided`               | every 5th `frame_step` frame                        |
| `fullframe_strided`     | no tiling + stride 5                                |
| `strided_negatives`     | stride 5 + empty tiles @ 15%                        |
| `strided_clip_balanced` | stride 5, cap each clip to median clip frame count  |

Config: `dataset_round.yaml` · Runner: `run_dataset_round.py`

### Round 3 — model

On the locked Round 2 pack and Round 1 imgsz, tune **train setup** in four steps — protocol, then family, then size letter, then epochs — one factor per group. Each group picks **winner** and carries it forward:


| Group | Varies | Fixed from prior |
| ----- | ------ | ---------------- |
| **P** protocol | frozen 20 ep / 2+5 / 5+15 | pack, imgsz |
| **F** family | YOLOv8s vs YOLO11s | winner P protocol |
| **S** size | n / s / m | winner F family + protocol |
| **E** epochs | Stage-2 5 / 10 / 15 ep | winner S; warmup copied from P |

Config: `model_round.yaml` · Runner: `run_model_round.py --all`

### Out of scope (cancelled factors)

- **Learning rate** — Fixed `lr0=0.01` (Stage 2: `0.001`). Best run (**2+5**) converged cleanly to **87.7%** mean A+B; longer Stage 2 (10/15 ep) at the same LR just drifted — an epoch budget issue, not an LR issue. Also `optimizer=auto` adjusts LR internally, making a clean grid search unreliable without disabling it first. Skipped in favor of higher-impact experiments.

- **NMS variants** — Default hard NMS, IoU 0.7; not varied. Our failure mode is partial/edge cars (nested boxes), fixed in postprocessing — not dense overlap, which is what soft-NMS/IoU tuning targets. Traffic here is spatially sparse, so this wasn't expected to help.

- **Neck (P2)** — Stock YOLO11s, no P2 head. P2 helps with sub-small objects; scale was already handled via imgsz 1280 + ~1024px tiling, so vehicles stay large in-frame (Band A: ~130–380px wide, well above COCO-small). Full-frame 1280 run confirms this — Band A dropped (58%) because cars got smaller, not because of a missing P2 head.

- **Mosaic** — Off (`mosaic=0`). Frames are continuous aerial video; mosaic stitches unrelated crops and breaks motion/road geometry. Frame diversity is already handled by `frame_step` subsampling.

### Run (once per machine, then R1 → R2 → R3)

**Prereq** — CVAT labels on disk, eval pack, Round 1 reference pack (`auto` = AUTO tiling, every `frame_step` frame; same recipe family as `baseline_v1` but built from `variants.yaml`, not `prepare_baseline.py`):

```bash
python src/labeling/cvat/cvat_pull.py --verify --sync-labels
python src/training/prepare_eval.py
python src/training/datasets/generate_variant.py --variant auto
```

If you change `config/datasets/variants.yaml`, rebuild affected packs (`--variant fullframe …`) or pass `--rebuild-packs` to Round 2.

**Round 1** — imgsz sweep on pack `auto`; results → `outputs/experiments/imgsz_round/summary.md`

```bash
python src/training/experiments/run_imgsz_round.py
python src/training/experiments/run_imgsz_round.py --resume          # continue interrupted
python src/training/experiments/run_imgsz_round.py --variant 1280    # one cell
```

Lock winner into `config/experiments/dataset_round.yaml` → `defaults.imgsz`.

**Round 2** — pack ablation at locked imgsz; builds each pack, trains, evals `eval_manual`

```bash
python src/training/experiments/run_dataset_round.py
python src/training/experiments/run_dataset_round.py --resume
python src/training/experiments/run_dataset_round.py --rebuild-packs  # after variants.yaml edit
python src/training/experiments/run_dataset_round.py --variant strided_clip_balanced
```

Lock winner into `config/experiments/model_round.yaml` → `defaults.dataset`.

**Round 3** — protocol → family → size → epochs on locked pack + imgsz

```bash
python src/training/experiments/run_model_round.py --all
python src/training/experiments/run_model_round.py --all --resume
python src/training/experiments/run_model_round.py --pick-winner     # rank only
```

Fill §5 from `outputs/experiments/model_round/summary.md`; copy final weights → `checkpoints/yolo11s_prototype_best.pt` (see §6).

---

## 3. Round 1 — results (imgsz)

| imgsz | A mAP@0.5 | B mAP@0.5 | mean A+B | A Det | B Det | A P | B P | A FA/min | B FA/min |
| ----- | --------- | --------- | -------- | ----- | ----- | --- | --- | -------- | -------- |
| 640   | 75.5%     | 79.9%     | 77.7%    | 92.9% | 85.7% | 58.3% | 90.6% | 2925     | 209      |
| 768   | 65.9%     | 90.7%     | 78.3%    | 98.0% | 94.6% | 57.1% | 96.4% | 3240     | 84       |
| 1024  | 46.9%     | 90.9%     | 68.9%    | 85.7% | 94.6% | 52.2% | 96.4% | 3465     | 84       |
| 1280  | 82.8%     | 78.0%     | **80.4%** | 99.0% | 85.7% | 61.0% | 85.7% | 2790     | 335      |

**Winner:** `1280` (80.4% mean; +2.1 pts vs 768)

**Wall:** 1h 25m 51s

**Notes:** 768 / 1024 win B only (A drops). 640 runner-up on mean but weaker A than 1280.

---

## 4. Round 2 — results (dataset pack)

Fixed imgsz **1280** (Round 1).

**Winner:** `strided_clip_balanced` (87.7% mean A+B; 85.6% A / 89.7% B)

**Wall:** 1h 10m 12s (6 cells)

### Band A — 0–200 m (`13722965…`)


| variant                 | mAP@0.5 | mAP@0.5:0.95 | Det   | P     | FA/min |
| ----------------------- | ------- | ------------ | ----- | ----- | ------ |
| `auto`                  | 82.8%   | 21.7%        | 99.0% | 61.0% | 2790   |
| `fullframe`             | 58.0%   | 15.2%        | 77.6% | 44.2% | 4320   |
| `strided`               | 9.1%    | 4.8%         | 6.1%  | 66.7% | 135    |
| `fullframe_strided`     | 86.3%   | 51.0%        | 92.9% | 67.4% | 1980   |
| `strided_negatives`     | 90.9%   | 32.7%        | 99.0% | 75.2% | 1440   |
| `strided_clip_balanced` | 85.6%   | 50.1%        | 98.0% | 64.9% | 2340   |

### Band B — 200–400 m (`266987`)


| variant                 | mAP@0.5 | mAP@0.5:0.95 | Det   | P     | FA/min |
| ----------------------- | ------- | ------------ | ----- | ----- | ------ |
| `auto`                  | 78.0%   | 18.3%        | 85.7% | 85.7% | 335    |
| `fullframe`             | 90.0%   | 19.3%        | 91.1% | 82.3% | 460    |
| `strided`               | —       | —            | 0.0%  | —     | 0      |
| `fullframe_strided`     | 76.1%   | 21.2%        | 83.9% | 88.7% | 251    |
| `strided_negatives`     | 1.3%    | 0.3%         | 1.8%  | 14.3% | 251    |
| `strided_clip_balanced` | 89.7%   | 24.8%        | 92.9% | 98.1% | 42     |

**Notes:**

- **`fullframe` vs `auto`:** Dropping tiles helps band B (`5382494` trains on whole frames; eval B is full-frame too → 90.0% B) but hurts band A — close cars in A clips shrink after letterbox to 1280, so mAP falls (58.0% vs 82.8% A).
- **`strided` on B:** Stride 5 leaves only ~171 train images (~6 min train). Eval clip `266987` is never in train; the model must generalize from tiled `5382494` only. `8457857` still contributes many A_close frames, so the run underfits and emits **no detections** on `266987` (0% det → mAP undefined).
- **`fullframe_strided` vs `strided`:** Same stride, but full-frame crops align with full-frame eval on B → 76.1% B vs 0% for tiled strided.
- **`strided_negatives`:** Extra empty tiles improve band A (90.9%) via hard negatives; band B unchanged (~1.3%) — negatives do not add B-medium signal or fix clip imbalance.
- **Winner `strided_clip_balanced`:** Same stride and tiling as `strided`, but caps `8457857` to the median clip length so B_medium tiles from `5382494` are actually learned; best mean A+B (87.7%).

---

## 5. Round 3 — results (model)

Fixed: pack **`strided_clip_balanced`** (Round 2) · imgsz **1280** (Round 1).

**Deliverable:** `proto_short_2p5` (YOLO11s, 2+5) — **87.7%** mean A+B (85.6% A / 89.7% B)

**Wall:** 1h 22m 2s (8 cells)

### Group P — protocol


| run | mean | A mAP@0.5 | B mAP@0.5 | Det A / B | P A / B | FA/min A / B |
| --- | ---- | --------- | --------- | --------- | ------- | ------------ |
| frozen 20 ep (head) | 80.6% | 85.0% | 76.2% | 99.0% / 80.4% | 56.1% / 72.6% | 3420 / 712 |
| **winner P** (2+5) | **87.7%** | **85.6%** | **89.7%** | 98.0% / 92.9% | 64.9% / 98.1% | 2340 / 42 |
| 5+15 | 76.8% | 62.9% | 90.7% | 98.0% / 92.9% | 49.5% / 94.5% | 4410 / 126 |

**Winner P:** `proto_short_2p5` — same mean as Round 2 winner on this pack; 5+15 trades A for B without beating dual-band mean.

### Group F — family


| run | mean | A mAP@0.5 | B mAP@0.5 | Det A / B | P A / B | FA/min A / B |
| --- | ---- | --------- | --------- | --------- | ------- | ------------ |
| **winner F** (11s) | **87.7%** | **85.6%** | **89.7%** | 98.0% / 92.9% | 64.9% / 98.1% | 2340 / 42 |
| YOLOv8s | 45.4% | 14.3% | 76.5% | 89.8% / 87.5% | 17.0% / 80.3% | 19395 / 502 |

**Winner F:** YOLO11s

### Group S — size


| run | mean | A mAP@0.5 | B mAP@0.5 | Det A / B | P A / B | FA/min A / B |
| --- | ---- | --------- | --------- | --------- | ------- | ------------ |
| n | 22.0% | 35.0% | 9.1% | 48.0% / 1.8% | 71.2% / 100.0% | 855 / 0 |
| **winner S** (s) | **87.7%** | **85.6%** | **89.7%** | 98.0% / 92.9% | 64.9% / 98.1% | 2340 / 42 |
| m | 43.7% | 7.3% | 80.2% | 22.4% / 87.5% | 10.1% / 59.0% | 8775 / 1423 |

**Winner S:** s

### Group E — Stage-2 epochs


| run | epochs | mean | A mAP@0.5 | B mAP@0.5 | Det A / B | P A / B | FA/min A / B |
| --- | ------ | ---- | --------- | --------- | --------- | ------- | ------------ |
| **winner E** (5 ep) | 5 | **87.7%** | **85.6%** | **89.7%** | 98.0% / 92.9% | 64.9% / 98.1% | 2340 / 42 |
| Stage-2 10 ep | 10 | 56.3% | 80.4% | 32.2% | 93.9% / 33.9% | 48.7% / 33.3% | 4365 / 1591 |
| Stage-2 15 ep | 15 | 56.8% | 43.9% | 69.6% | 59.2% / 71.4% | 35.2% / 71.4% | 4815 / 670 |

**Winner E:** 5 ep (Stage 2) · Chain: P (2+5) → F (11s) → S (s) → E (5 ep) → **`proto_short_2p5`**

**Notes:** Deliverable metrics match Round 2 `strided_clip_balanced` exactly — R3 did not beat the 2+5 recipe on this pack. Longer Stage-2 (10 / 15 ep) hurt mean (B collapses at 10 ep). `n` / `m` and YOLOv8s unusable on A.

### Takeaways

**Small-data regime.** Deliverable pack has **152 train slices**, ~6 min train per cell at imgsz 1280. Every R3 comparison is under heavy data scarcity — capacity and train length matter as much as architecture.

**YOLO11s vs YOLOv8s (F).** v8s keeps high detection on both bands but **A mAP 14.3%** and **19k FA/min** on A — lots of confident wrong boxes. Likely a backbone / neck mismatch for this domain at 1280 on manual labels: v8s overfits clip texture or fires on background at high recall. YOLO11s was used throughout PoC → R1–R3; same 2+5 schedule gives balanced A+B. Not proof 11 is universally better, but **on this label budget v8s is not a drop-in substitute**.

**Size n / s / m (S).** **`n`** — under capacity (48% det A). **`m`** — too many params for ~152 images and a 2+5 schedule: A mAP **7.3%**, det **22%**, FA explodes — classic overfit / unstable head on a tiny pack. **`s`** sits in the middle and matches what the data can support. More data or longer regularized training might reopen `m`; here it hurts.

**Protocol & epochs (P, E).** **2+5** wins; **frozen 20 ep** (head only) is close on mean (80.6%) but misses backbone adapt for B. **5+15** and **Stage-2 10 / 15 ep** do not help — with so few train tiles, extra epochs drift (10 ep: B mAP **32%**; 15 ep: A mAP **44%**). Short unfreeze + early stop (`patience=3`) looks right for this prototype scale.

**R3 vs R2.** No gain over Round 2 on the same pack — sensible: pack + imgsz + 2+5 were co-tuned in R1–R2; R3 confirms the recipe rather than replacing it. Further gains likely need **more labeled diversity** (clips / bands), not another epoch or model size on the current set.

---

## 6. Lock-in

Deliverable: **`proto_short_2p5`** — YOLO11s on `strided_clip_balanced`, imgsz 1280, 2 ep frozen head + 5 ep full model.

### Locked recipe


| Field | Value |
| ----- | ----- |
| **Labels** | CVAT YOLO txt under `labels/{train,eval}/{clip}/` (class `0 = vehicle`) |
| **Train pack** | `strided_clip_balanced` — stride 5, AUTO tiling, clip cap to median frame count (`config/datasets/variants.yaml`) |
| **Train images** | 152 train / 26 val slices (640 letterbox target in pack yaml; override at train time) |
| **Eval pack** | `data/datasets/eval_manual/` — clips `13722965…` (band A), `266987` (band B), cap 64 frames/clip |
| **Model** | YOLO11s (`yolo11s.pt`) |
| **Letterbox imgsz** | **1280** (train + predict) |
| **Schedule** | 2 ep frozen head + 5 ep full model (`freeze=11`, `patience=3`, `lr0=0.01`) |
| **Aug** | PoC set (HSV, flips, degrees=180; no mosaic/mixup) |
| **Seed** | 42 |
| **Postprocess** | Nested-box drop (≥80% area inside larger box), IoU match @0.5, conf **0.25** |
| **Weights** | `outputs/experiments/model_round/md_proto_short_2p5/weights/best.pt` → copy to `checkpoints/yolo11s_prototype_best.pt` |

**Local checkpoints** (`checkpoints/` — not committed): `yolo11s_poc_best.pt` (PoC); `yolo11s_prototype_best.pt` (deliverable copy):

```bash
cp outputs/experiments/model_round/md_proto_short_2p5/weights/best.pt \
   checkpoints/yolo11s_prototype_best.pt
```

### Reproduce (from clean labels on disk)

```bash
# 1. Labels + eval pack
python src/labeling/cvat/cvat_pull.py --verify --sync-labels
python src/training/prepare_eval.py

# 2. Train pack
python src/training/datasets/generate_variant.py --variant strided_clip_balanced

# 3. Train (proto_short_2p5 recipe)
python -u src/training/train.py \
  --dataset-dir data/datasets/strided_clip_balanced \
  --imgsz 1280 \
  --warmup-epochs 2 \
  --epochs 5 \
  --patience 3

# 4. Score on manual eval clips
python src/training/evaluate.py \
  --gt manual \
  --weights checkpoints/yolo11s_prototype_best.pt \
  --imgsz 1280 \
  --no-video \
  --output-dir outputs/eval_manual_final
```

Or re-run Round 3 cell only: `python src/training/experiments/run_model_round.py --variant proto_short_2p5`

### Final eval (manual GT, eval clips)


| Band | Clip | mAP@0.5 | mAP@0.5:0.95 | Det | P | FA/min |
| ---- | ---- | ------- | ------------ | --- | - | ------ |
| A (<200 m) | `13722965…` | 85.6% | 50.1% | 98.0% | 64.9% | 2340 |
| B (>200 m) | `266987` | 89.7% | 24.8% | 92.9% | 98.1% | 42 |
| **mean A+B** | | **87.7%** | | | | |

Source: `outputs/experiments/model_round/proto_short_2p5/eval_manual/eval_metrics.json`

### PoC vs prototype (same eval clips, different GT)


| | **PoC** | **Prototype (locked)** |
| --- | --- | --- |
| Labels | autolabel (`outputs/autolabel/labels/`) | CVAT `labels/` |
| Train pack | `baseline_v0` (autolabel, full stride) | `strided_clip_balanced` |
| Train command | `prepare_baseline.py --from-autolabel` + `train.py --prototype` | `generate_variant.py` + `train.py --imgsz 1280 …` |
| Eval pack | `eval_autolabel` | `eval_manual` |
| Eval GT | autolabel | manual CVAT |
| imgsz | 640 | **1280** |
| Weights | `checkpoints/yolo11s_poc_best.pt` | `checkpoints/yolo11s_prototype_best.pt` |

**mAP is not directly comparable** — GT differs (§1: autolabel recall 81.6% but precision 45% on shared frames). Use this table for operational tradeoffs on the same two distance bands, not as a pure model gain.


| Metric | PoC A / B | Prototype A / B |
| ------ | --------- | --------------- |
| mAP@0.5 | 50.0% / 69.0% | 85.6% / 89.7% |
| mAP@0.5:0.95 | 22.9% / 15.0% | 50.1% / 24.8% |
| Detection rate | 65.7% / 74.2% | 98.0% / 92.9% |
| Precision | 59.0% / 89.1% | 64.9% / 98.1% |
| FA/min | 3130 / 257 | 2340 / 42 |
| mean mAP@0.5 | 59.5% | **87.7%** |

PoC numbers: PoC.md eval run (autolabel GT, imgsz 640, 2+5, conf 0.25).

**Re-score PoC on same command shape (reference only):**

```bash
python src/training/prepare_eval.py --from-autolabel
python src/training/evaluate.py --gt autolabel \
  --weights checkpoints/yolo11s_poc_best.pt --imgsz 640 --no-video
```
