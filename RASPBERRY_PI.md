# Raspberry Pi 5 — prune, compress, live camera

**Read [EVALUATION.md](EVALUATION.md) first.** Locked prototype: YOLO11s, pack `strided_clip_balanced`, letterbox **1280**, 2+5, **87.7%** mean A+B mAP@0.5. Task spec: **[README.md](README.md)** — class `0 = vehicle`, **find a car**, not bbox tightness (`mAP@0.5`, not `mAP@0.5:0.95`).

This doc is the **edge path**: take that locked recipe and make it run on a **Raspberry Pi 5** with a **live camera**. One factor per round. Fill result tables from `outputs/experiments/pi_*/summary.md` after each round.

No runners yet — commands below are the intended shape (`config/experiments/pi_*.yaml` + `src/training/experiments/run_pi_*.py`), same pattern as Rounds 1–3.

## Why this is not “just prune the locked model”

Pi 5 is **CPU + VideoCore** (BCM2712, 4× Cortex-A76). No CUDA, no TensorRT.

- **Unstructured prune** (zero individual weights) shrinks the file; on Pi it **does not** speed up dense GEMM/conv. Keep it only as a control cell.
- **Structured prune** (drop whole channels/filters) *can* raise FPS because the graph is smaller. Needs a **short recover fine-tune** on the locked pack or mAP collapses.
- Locked **imgsz 1280 + YOLO11s** is likely the bigger latency cost than 30–50% prune. Round 3 already showed **YOLO11n is unusable** (22% mean mAP@0.5) — do not “just switch to n”. Prune **s**, do not replace it with n.
- Desktop / MPS FPS is **not** a Pi number. Quality eval can stay on the train machine; **FPS / RSS / thermal** only count on the Pi.

If Round 0 (baseline) already misses the FPS floor by 5–10×, prune-only cannot close it — Round 1 (infer imgsz) must run before you interpret prune results.

---

## Target

| | Spec |
| --- | --- |
| **Device** | Raspberry Pi 5 (note 4 GB vs 8 GB — RSS budget changes) |
| **Camera** | CSI (Picamera2 / libcamera) or USB; live stream, not offline clips |
| **Quality holdout** | Same as prototype: `eval_manual`, bands A/B, conf **0.25**, IoU match **0.5** |
| **Decision (quality)** | mean A+B **mAP@0.5**. Det / P / FA/min = diagnostics. Ignore `mAP@0.5:0.95` for winner. |
| **Decision (speed)** | **Pi FPS** after warmup (see SLA). RSS peak, model MB, end-to-end camera→box ms. |
| **Winner rule** | Fastest cell that stays within **`winner_delta` of locked mean mAP@0.5** *and* meets FPS floor. If none meet both — lock the cell on the Pareto front (highest mAP among those ≥ FPS floor) and write why. Do not pick max mAP if it is 2 FPS. |

### SLA (lock before Round 1 — change the numbers, not the rule)

| Knob | Proposed | Meaning |
| --- | --- | --- |
| **FPS floor** | **10** processed frames/s | Live overlay; below this, skip frames (Round 5) instead of pretending it is live |
| **winner_delta** | **0.015** (1.5 pts) vs 87.7% | Same delta as model round. Loosen only if Round 0 proves 1280 cannot hit FPS |
| **RAM** | leave **1.5 GB** for OS + camera | Abort a cell if RSS blows the rest |
| **Camera** | grab **latest** frame, drop queue | Stale-frame queue is not latency, it is lag |

Live pipeline (fixed across prune rounds; only Round 5 may change skip/conf):

```
camera → drop stale frames → letterbox imgsz → infer → NMS + nested-box drop → overlay
```

Do **not** SAHI-tile on device unless a round explicitly tests it (tiles × N inferences).

Domain note: quality tables use **drone eval clips**. If the Pi camera is a different view (not nadir), those numbers are a proxy — collect a small Pi-cam holdout before calling it production. Same class spec; new images.

---

## Experiment design

Four rounds in order after a one-shot baseline. Each locks one choice for the next.

```
Round 0  baseline on Pi          →  numbers only (no lock)
Round 1  infer imgsz             →  lock defaults.imgsz
Round 2  structured prune        →  lock sparsity + recover
Round 3  quantize + runtime      →  lock INT8/NCNN (or whatever wins)
Round 4  live-stream knobs       →  skip / conf  (FPS already in SLA)
```

Train / prune / recover: Mac or GPU. **Eval quality:** `eval_manual` (same script as prototype). **Eval speed:** copy artifact to Pi, bench there.

Shared with EVALUATION.md (not repeated): pack `strided_clip_balanced`, nested-box NMS, metric columns, bands A/B.

---

## Round 0 — baseline (locked model, no change)

One cell. Proves whether prune is even the right next lever.

| id | Weights | imgsz | Runtime | Device |
| -- | ------- | ----- | ------- | ------ |
| `locked_pt` | `checkpoints/yolo11s_prototype_best.pt` | 1280 | PyTorch | Pi 5 CPU |

```bash
# Quality (train machine — same as EVALUATION.md §6)
python src/training/evaluate.py \
  --gt manual \
  --weights checkpoints/yolo11s_prototype_best.pt \
  --imgsz 1280 \
  --no-video \
  --output-dir outputs/eval_manual_final

# Speed (on the Pi, after warmup 20 frames; report median + p95 of 100)
# TODO: python src/training/experiments/bench_pi.py --weights ... --imgsz 1280 --device cpu
```

| | A mAP@0.5 | B mAP@0.5 | mean | Det A / B | FPS (Pi) | p95 ms | RSS MB | weights MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `locked_pt` | 85.6% | 89.7% | 87.7% | 98.0 / 92.9 | *fill* | *fill* | *fill* | *fill* |

**Go / no-go:** if Pi FPS ≥ 10, skip Round 1 imgsz sweep (keep 1280). If FPS is ~1–3, **do not start prune at 1280** — Round 1 first.

---

## Round 1 — infer imgsz

**Varies:** letterbox at **predict** time. Train weights stay locked (no retrain). Same `eval_manual`.

Cells: `640` / `768` / `1024` / `1280` (1280 = Round 0 reference).

Round 1 of EVALUATION.md picked 1280 for **quality**. Here 1280 may lose on the **constrained** rule. One factor: imgsz only.

Config (intended): `config/experiments/pi_imgsz_round.yaml` · `run_pi_imgsz_round.py`

| imgsz | A mAP@0.5 | B mAP@0.5 | mean | Det A / B | FPS (Pi) | p95 ms | pass SLA? |
| ----- | --------- | --------- | ---- | --------- | -------- | ------ | --------- |
| 640   | | | | | | | |
| 768   | | | | | | | |
| 1024  | | | | | | | |
| 1280  | | | | | | | |

**Winner →** `pi_prune_round.yaml` `defaults.imgsz`.

Expect band A (close / large cars) to survive a smaller letterbox better than B (far / small). If B Det collapses, you cannot buy FPS that way — then prune + 1280, or accept skip-frames in Round 4.

---

## Round 2 — structured prune + recover

Fixed: Round 1 imgsz, locked pack, YOLO11s graph.

**Unstructured = control only** (size ↓, FPS ≈ same). Decision cells = **structured channel prune**.

Library: [torch-pruning](https://github.com/VainF/Torch-Pruning) DepGraph (YOLO-safe); not `torch.nn.utils.prune` magnitude masks.

### Group A — method (sparsity fixed, e.g. 40%)

| id | Method | Recover |
| -- | ------ | ------- |
| `unstruct_40` | unstructured magnitude | none (control) |
| `struct_40_zero` | structured channels | none |
| `struct_40_ft` | structured channels | 5 ep full model, same pack / aug / lr0 as lock-in |

One factor: method (+ whether recover is on). Do not change sparsity here.

### Group B — sparsity on winning method (recover on if A says so)

| id | Target FLOPs / channels removed |
| -- | ------------------------------ |
| `p00` | 0% (locked, reference) |
| `p20` | 20% |
| `p40` | 40% |
| `p60` | 60% |

Recover recipe if Group A chose fine-tune: **5 ep**, `freeze=0`, pack `strided_clip_balanced`, seed 42 — same data as lock-in, not a new pack. Do not extend to 15 ep (EVALUATION.md Group E: longer Stage-2 hurt).

Optional later (not this round): distillation from locked teacher. Only if 40–60% structured + 5 ep cannot hold `winner_delta`.

Config (intended): `config/experiments/pi_prune_round.yaml` · `run_pi_prune_round.py`

```bash
# python src/training/experiments/run_pi_prune_round.py --all
# python src/training/experiments/run_pi_prune_round.py --pick-winner
```

### Group A — results

| run | mean mAP@0.5 | A / B | Det A / B | FPS (Pi) | weights MB | params |
| --- | ------------ | ----- | --------- | -------- | ---------- | ------ |
| `unstruct_40` | | | | | | |
| `struct_40_zero` | | | | | | |
| `struct_40_ft` | | | | | | |

**Winner A:**

### Group B — results

| run | mean mAP@0.5 | A / B | Det A / B | FPS (Pi) | pass delta? | pass FPS? |
| --- | ------------ | ----- | --------- | -------- | ----------- | --------- |
| `p00` | | | | | | |
| `p20` | | | | | | |
| `p40` | | | | | | |
| `p60` | | | | | | |

**Winner B →** sparsity + recover locked for Round 3. Artifact: `checkpoints/yolo11s_pi_pruned.pt` (or the DepGraph `.pt`).

---

## Round 3 — quantize + runtime

Fixed: pruned weights + Round 1 imgsz.

**Varies:** numeric format **or** runtime — **not both in one cell**. Two groups.

Pi-relevant exports (Ultralytics): **NCNN** (ARM NEON, usual winner), **TFLite** (+ XNNPACK), **ONNX Runtime**. Skip TensorRT / Hailo unless that hat is on the board.

### Group Q — precision (runtime fixed: NCNN, or ONNX if NCNN export fails)

| id | Precision | Calib |
| -- | --------- | ----- |
| `fp32` | FP32 | — |
| `fp16` | FP16 | — (often little gain on Pi CPU) |
| `int8` | INT8 | 256 images from **train** pack (not eval) |

INT8 calib on eval = leakage into the speed/quality story. Use `strided_clip_balanced` train tiles only.

### Group R — runtime (precision = winner Q)

| id | Format |
| -- | ------ |
| `ncnn` | NCNN |
| `tflite` | TFLite |
| `onnx` | ONNX Runtime |

After export: **numeric parity** on a 32-image smoke set (max abs box/conf drift vs FP32). Then full `eval_manual` + Pi bench.

Config (intended): `config/experiments/pi_export_round.yaml` · `run_pi_export_round.py`

```bash
# Shape (fill after Group Q/R lock):
# yolo export model=checkpoints/yolo11s_pi_pruned.pt format=ncnn imgsz=... int8=...
```

### Group Q — results

| run | mean mAP@0.5 | A / B | Det A / B | FPS (Pi) | weights MB |
| --- | ------------ | ----- | --------- | -------- | ---------- |
| `fp32` | | | | | |
| `fp16` | | | | | |
| `int8` | | | | | |

**Winner Q:**

### Group R — results

| run | mean mAP@0.5 | A / B | Det A / B | FPS (Pi) | p95 ms |
| --- | ------------ | ----- | --------- | -------- | ------ |
| `ncnn` | | | | | |
| `tflite` | | | | | |
| `onnx` | | | | | |

**Winner R →** deployable artifact under `checkpoints/pi/` (not git).

INT8 often hits **small objects (band B)** first. If mean mAP holds but B Det dies, reject INT8 even if FPS wins.

---

## Round 4 — live camera (not another train)

Fixed: Round 3 artifact + imgsz. **No more architecture.**

| id | Change |
| -- | ------ |
| `skip1` | every frame |
| `skip2` | infer every 2nd captured frame (display last box) |
| `skip3` | every 3rd |
| `conf15` | conf 0.15 (recall) |
| `conf25` | conf 0.25 (locked eval) |
| `conf40` | conf 0.40 (fewer FA) |

One factor per cell (skip **or** conf). Camera resolution is a separate one-off: match letterbox (e.g. capture 1280×720 if imgsz is 640 — do not capture 4K then downscale if the CSI pipe is the bottleneck).

**Quality on live:** no GT on the Pi stream. Score skip/conf on `eval_manual` (temporal skip ≈ dropping frames). Then a **15 min watch** on the real camera: count obvious FN/FA by hand, log thermal (`vcgencmd measure_temp`) and whether FPS stays at SLA once warm.

| run | mean mAP@0.5 (eval) | Det A / B | camera e2e FPS | notes |
| --- | ------------------- | --------- | -------------- | ----- |
| `skip1` | | | | |
| `skip2` | | | | |
| `skip3` | | | | |
| `conf15` | | | | |
| `conf25` | | | | |
| `conf40` | | | | |

**Winner →** § Lock-in live recipe.

---

## Lock-in

Fill after rounds. Prototype row is the quality ceiling, not the deployable.

| Field | Prototype (EVALUATION.md) | Pi live (this doc) |
| --- | --- | --- |
| **Weights** | `checkpoints/yolo11s_prototype_best.pt` | *fill* (`checkpoints/pi/…`) |
| **imgsz** | 1280 | *fill* (R1) |
| **Prune** | none | *fill* (R2 method + %) |
| **Recover** | — | *fill* |
| **Precision / runtime** | FP32 PyTorch | *fill* (R3) |
| **conf** | 0.25 | *fill* (R4) |
| **Frame skip** | n/a | *fill* |
| **mean A+B mAP@0.5** | 87.7% | *fill* |
| **Det A / B** | 98.0% / 92.9% | *fill* |
| **FPS on Pi 5** | — | *fill* |

### Reproduce (quality)

```bash
python src/training/evaluate.py \
  --gt manual \
  --weights checkpoints/pi/<locked_artifact> \
  --imgsz <R1> \
  --no-video \
  --output-dir outputs/eval_pi_final
```

### Reproduce (Pi bench + camera)

```bash
# TODO: python src/training/experiments/bench_pi.py --weights checkpoints/pi/<locked> --imgsz <R1>
# TODO: python src/training/experiments/pi_camera.py --weights checkpoints/pi/<locked> --imgsz <R1> --skip <R4>
```

---

## Order of work

1. Flash Pi 5, confirm RAM, camera (`rpicam-hello` / Picamera2 still). Install the **same** Ultralytics pin as `requirements.txt` or only the runtime needed for the Round 3 winner (NCNN/TFLite) — do not mix torch versions between export machine and Pi.
2. Round 0 bench on Pi with locked `.pt`.
3. Round 1 → lock imgsz in yaml.
4. Round 2 prune on train machine → copy `.pt` → Pi FPS.
5. Round 3 export/quant → Pi only for speed; `eval_manual` for mAP.
6. Round 4 on the desk with the real camera.

Do not start Round 2 until Round 0 numbers exist. Do not tune conf to rescue a prune that killed band-B Det.