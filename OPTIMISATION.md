# Optimisation — prune, compress, live camera

Locked prototype: YOLO11s, pack `strided_clip_balanced`, letterbox **1280**, 2+5, **87.7%** mean A+B mAP@0.5. Task spec: **[README.md](README.md)** — class `0 = vehicle`, **find a car**, not bbox tightness (`mAP@0.5`, not `mAP@0.5:0.95`).

Two deploy targets, **one shared recipe** then **two device tracks**. Do not mix FPS across devices into one winner.

| Track | Device | Required runtime |
| ----- | ------ | ---------------- |
| Shared (S) | train machine | quality + prune only |
| **Pi (P)** | Raspberry Pi 5 | **OpenVINO** (plus NCNN/TFLite as cells) |
| **Win (W)** | NVIDIA GPU (Windows target) | **TensorRT** (OpenVINO = Intel CPU fallback on the same PC) |

**TensorRT work lives on Colab or Kaggle** (T4/L4-class GPU): export, FP16/INT8 build, `eval_manual` quality. A **Jetson emulator** can reproduce TensorRT API / pipeline interactions (load engine, infer, NMS wiring). It is **not** a speed test and **not** real-life — no SLA FPS, no thermal, no live camera. Colab/Kaggle FPS is also not Jetson or Windows FPS; engines are GPU-arch specific, so a T4 `.engine` does not ship to Jetson.

No runners yet — commands are the intended shape (`config/experiments/opt_*.yaml` + `src/training/experiments/run_opt_*.py`), same pattern as EVALUATION.md Rounds 1–3. Fill tables from `outputs/experiments/opt_*/summary.md`.

```
S0 quality baseline → S1 infer imgsz → S2 structured prune
        ↓                                      ↓
   P0 Pi speed                            W0 Win speed
   P1 OpenVINO ± INT8                     W1 TensorRT FP16/INT8
   P2 live CSI                            W2 OpenVINO CPU (fallback)
                                          W3 live camera
```

---

# Shared — any device

Quality holdout, prune, and imgsz are **the same** for Pi and Windows. Speed columns stay empty here; fill them in the device parts.

## Why this is not “just prune the locked model”

- **Unstructured prune** (zero individual weights) shrinks the file; on dense conv it **does not** raise FPS. Control cell only.
- **Structured prune** (drop whole channels/filters) *can* raise FPS. Needs a **short recover fine-tune** on the locked pack or mAP collapses.
- Locked **imgsz 1280 + YOLO11s** is often a bigger latency cost than 30–50% prune. Round 3 already showed **YOLO11n is unusable** (22% mean mAP@0.5) — prune **s**, do not switch to n.
- Quality eval (`eval_manual`) can stay on the train machine. **FPS / RSS / thermal** only count on the **target device**. Mac MPS / CUDA laptop numbers are not Pi numbers and are not the Windows TensorRT numbers either.

If a device Round 0 already misses that device’s FPS floor by 5–10×, prune-only cannot close it — use shared S1 (imgsz) or accept frame skip on that device.

## Decision rules (both tracks)

| | Spec |
| --- | --- |
| **Quality holdout** | `eval_manual`, bands A/B, conf **0.25**, IoU match **0.5** |
| **Decision (quality)** | mean A+B **mAP@0.5**. Det / P / FA/min = diagnostics. Ignore `mAP@0.5:0.95` for winner. |
| **Decision (speed)** | device FPS after warmup (see that device’s SLA). RSS peak, artifact MB, camera→box ms. |
| **Winner rule** | Fastest cell **on that device** that stays within **`winner_delta`** of locked mean mAP@0.5 *and* meets **that device’s** FPS floor. If none meet both — Pareto (highest mAP among cells ≥ FPS floor) and write why. Two lock-ins, not one. |

`winner_delta` proposed **0.015** (same as model round). Change the number before S1; do not change the rule after seeing results.

INT8 calibration: **256 images from `strided_clip_balanced` train**, never eval. Calib on eval = leakage.

Live pipeline (device parts may change skip/conf only):

```
camera → drop stale frames → letterbox imgsz → infer → NMS + nested-box drop → overlay
```

Do **not** SAHI-tile on device unless a round explicitly tests it.

Domain note: quality tables use **drone eval clips**. If a live camera is a different view (not nadir), those numbers are a proxy — collect a small camera holdout before calling it production.

Shared with EVALUATION.md (not repeated): pack `strided_clip_balanced`, nested-box NMS, metric columns, bands A/B.

Train / prune / recover: Mac or GPU. Copy artifacts to Pi / Windows for speed.

---

## Round S0 — quality baseline (locked model)

Already measured. Copy here so device tables have a ceiling.

| id | Weights | imgsz | A mAP@0.5 | B mAP@0.5 | mean | Det A / B |
| -- | ------- | ----- | --------- | --------- | ---- | --------- |
| `locked_pt` | `checkpoints/yolo11s_prototype_best.pt` | 1280 | 85.6% | 89.7% | **87.7%** | 98.0 / 92.9 |

```bash
python src/training/evaluate.py \
  --gt manual \
  --weights checkpoints/yolo11s_prototype_best.pt \
  --imgsz 1280 \
  --no-video \
  --output-dir outputs/eval_manual_final
```

---

## Round S1 — infer imgsz (quality only)

**Varies:** letterbox at **predict** time. Weights stay locked (no retrain). Same `eval_manual`.

Cells: `640` / `768` / `1024` / `1280`.

EVALUATION.md Round 1 picked 1280 for unconstrained quality. Here a smaller imgsz may be required so a device can hit its FPS floor. One factor: imgsz.

Config (intended): `config/experiments/opt_imgsz_round.yaml`

| imgsz | A mAP@0.5 | B mAP@0.5 | mean | Det A / B | pass delta? |
| ----- | --------- | --------- | ---- | --------- | ----------- |
| 640   | | | | | |
| 768   | | | | | |
| 1024  | | | | | |
| 1280  | | | | | |

**Winner S1 →** default infer imgsz for prune and both device tracks.

A device may **override downward** later if this imgsz still misses that device’s SLA (one factor, recorded in that device’s lock-in). Do not raise imgsz above S1 on a device.

Expect band B (far / small) to die first when imgsz drops.

---

## Round S2 — structured prune + recover

Fixed: S1 imgsz, locked pack, YOLO11s graph.

Library: [torch-pruning](https://github.com/VainF/Torch-Pruning) DepGraph (YOLO-safe); not `torch.nn.utils.prune` magnitude masks.

### Group A — method (sparsity fixed, e.g. 40%)

| id | Method | Recover |
| -- | ------ | ------- |
| `unstruct_40` | unstructured magnitude | none (control) |
| `struct_40_zero` | structured channels | none |
| `struct_40_ft` | structured channels | 5 ep full model, same pack / aug / lr0 as lock-in |

### Group B — sparsity on winning method (recover on if A says so)

| id | Target FLOPs / channels removed |
| -- | ------------------------------ |
| `p00` | 0% (locked, reference) |
| `p20` | 20% |
| `p40` | 40% |
| `p60` | 60% |

Recover if A chose fine-tune: **5 ep**, `freeze=0`, pack `strided_clip_balanced`, seed 42. Do not extend to 15 ep (EVALUATION.md Group E: longer Stage-2 hurt). Distillation from locked teacher — only if 40–60% + 5 ep cannot hold `winner_delta`.

Config (intended): `config/experiments/opt_prune_round.yaml`

```bash
# python src/training/experiments/run_opt_prune_round.py --all
# python src/training/experiments/run_opt_prune_round.py --pick-winner
```

### Group A — results

| run | mean mAP@0.5 | A / B | Det A / B | weights MB | params |
| --- | ------------ | ----- | --------- | ---------- | ------ |
| `unstruct_40` | | | | | |
| `struct_40_zero` | | | | | |
| `struct_40_ft` | | | | | |

**Winner A:**

### Group B — results

| run | mean mAP@0.5 | A / B | Det A / B | pass delta? |
| --- | ------------ | ----- | --------- | ----------- |
| `p00` | | | | |
| `p20` | | | | |
| `p40` | | | | |
| `p60` | | | | |

**Winner S2 →** `checkpoints/yolo11s_opt_pruned.pt`. Both device tracks start from this file (or `locked_pt` if prune loses on delta).

---

# Raspberry Pi 5 — OpenVINO

Pi 5 is **CPU + VideoCore** (BCM2712, 4× Cortex-A76). No CUDA, **no TensorRT**.

**OpenVINO on Pi 5 is ARM CPU**, not the Intel iGPU product. Export may fail or be slow; that is a result, not a surprise. Keep **NCNN** and **TFLite** as runtime cells so the round is interpretable if OpenVINO loses or does not export.

## Pi SLA (lock before P0 — change numbers, not the rule)

| Knob | Proposed | Meaning |
| --- | --- | --- |
| **FPS floor** | **10** | live overlay; else skip frames in P2 |
| **RAM** | leave **1.5 GB** for OS + camera | abort if RSS blows the rest |
| **Camera** | CSI (Picamera2 / libcamera) or USB; grab **latest**, drop queue | |

Note 4 GB vs 8 GB RAM — RSS budget changes.

## Round P0 — speed baseline on Pi

Same weights as S0 / S2. PyTorch CPU first (proves the gap), then OpenVINO if export already exists.

| id | Weights | imgsz | Runtime | FPS | p95 ms | RSS MB |
| -- | ------- | ----- | ------- | --- | ------ | ------ |
| `locked_pt` | prototype `.pt` | S1 | PyTorch CPU | | | |
| `pruned_pt` | S2 `.pt` | S1 | PyTorch CPU | | | |

```bash
# TODO: python src/training/experiments/bench_opt.py --device pi --weights ... --imgsz ...
```

**Go / no-go:** if PyTorch already ≥ 10 FPS, skip heroic prune/INT8. If ~1–3 FPS at S1 imgsz, do not interpret P1 until imgsz override is decided (one step: drop imgsz, re-score `eval_manual`).

## Round P1 — OpenVINO + precision (runtime as second group)

Fixed: S2 weights, S1 imgsz (or Pi override). **Precision or runtime — not both in one cell.**

### Group Q — precision (runtime = OpenVINO)

| id | Precision | Calib |
| -- | --------- | ----- |
| `ov_fp32` | FP32 | — |
| `ov_fp16` | FP16 | — (often little gain on Pi CPU) |
| `ov_int8` | INT8 | 256 train tiles |

### Group R — runtime (precision = winner Q)

| id | Format |
| -- | ------ |
| `openvino` | **OpenVINO** (required cell) |
| `ncnn` | NCNN (ARM NEON control) |
| `tflite` | TFLite + XNNPACK |

After export: numeric parity on 32-image smoke vs FP32 `.pt`, then `eval_manual` + Pi bench.

```bash
# yolo export model=checkpoints/yolo11s_opt_pruned.pt format=openvino imgsz=... int8=...
```

### Group Q — results

| run | mean mAP@0.5 | A / B | Det A / B | FPS (Pi) | artifact MB |
| --- | ------------ | ----- | --------- | -------- | ----------- |
| `ov_fp32` | | | | | |
| `ov_fp16` | | | | | |
| `ov_int8` | | | | | |

**Winner Q:**

### Group R — results

| run | mean mAP@0.5 | A / B | Det A / B | FPS (Pi) | p95 ms |
| --- | ------------ | ----- | --------- | -------- | ------ |
| `openvino` | | | | | |
| `ncnn` | | | | | |
| `tflite` | | | | | |

**Winner P1 →** `checkpoints/opt/pi/` (not git). INT8 often kills **band B** first — reject if Det B dies even if FPS wins.

## Round P2 — live CSI (not another train)

Fixed: P1 artifact. One factor per cell (skip **or** conf).

| id | Change |
| -- | ------ |
| `skip1` | every frame |
| `skip2` | every 2nd captured frame |
| `skip3` | every 3rd |
| `conf15` | conf 0.15 |
| `conf25` | conf 0.25 (locked eval) |
| `conf40` | conf 0.40 |

Score skip/conf on `eval_manual` (temporal skip ≈ dropping frames). Then **15 min** on the real camera: hand-count obvious FN/FA, `vcgencmd measure_temp`, FPS once warm.

| run | mean mAP@0.5 (eval) | Det A / B | camera e2e FPS | notes |
| --- | ------------------- | --------- | -------------- | ----- |
| `skip1` | | | | |
| `skip2` | | | | |
| `skip3` | | | | |
| `conf15` | | | | |
| `conf25` | | | | |
| `conf40` | | | | |

## Pi lock-in

| Field | Prototype | Pi live |
| --- | --- | --- |
| **Weights / IR** | `yolo11s_prototype_best.pt` | *fill* `checkpoints/opt/pi/` |
| **imgsz** | 1280 | *fill* (S1 or Pi override) |
| **Prune** | none | *fill* (S2) |
| **Runtime** | FP32 PyTorch | OpenVINO *or fill if NCNN/TFLite won* |
| **Precision** | FP32 | *fill* |
| **conf / skip** | 0.25 / n/a | *fill* |
| **mean A+B mAP@0.5** | 87.7% | *fill* |
| **Det A / B** | 98.0% / 92.9% | *fill* |
| **FPS on Pi 5** | — | *fill* |

```bash
python src/training/evaluate.py \
  --gt manual \
  --weights checkpoints/opt/pi/<locked> \
  --imgsz <S1> \
  --no-video \
  --output-dir outputs/eval_opt_pi_final
# TODO: bench_opt.py --device pi · pi_camera.py
```

---

# NVIDIA Windows — TensorRT (+ OpenVINO CPU)

**Build TensorRT on Colab or Kaggle**, not on the laptop. Windows NVIDIA is the intended surrounding; Jetson emulator is for interaction checks only (see top of this doc). **TensorRT** is the deploy path (FP16, then INT8). Build the engine **on the GPU that will run it** for any speed/SLA table — engines are not portable across GPU arch / TensorRT versions. Colab T4 numbers do not lock the Windows (or Jetson) FPS floor.

**OpenVINO** on this machine is the **Intel CPU (or iGPU) fallback**, not a second GPU compiler. Do not compare OpenVINO CPU FPS to TensorRT GPU FPS as a quality winner; it answers “does a no-GPU fallback still find cars?”.

Pin and record before W0: GPU name, driver, **CUDA**, **cuDNN**, **TensorRT**, Ultralytics pin. Mismatch is a common silent numeric skew.

## Windows SLA (lock before W0)

| Knob | Proposed | Meaning |
| --- | --- | --- |
| **FPS floor** | **30** | live overlay on GPU; raise if the camera is 60 fps and you need every frame |
| **VRAM** | leave headroom for desktop + capture | abort if OOM |
| **Camera** | USB / capture card; grab **latest**, drop queue | |

Fill GPU model here: ________

## Round W0 — speed baseline on Windows

| id | Weights | imgsz | Runtime | FPS | p95 ms | VRAM MB |
| -- | ------- | ----- | ------- | --- | ------ | ------- |
| `locked_pt_cuda` | prototype `.pt` | S1 | PyTorch CUDA | | | |
| `pruned_pt_cuda` | S2 `.pt` | S1 | PyTorch CUDA | | | |
| `trt_fp16_probe` | S2 → TensorRT FP16 | S1 | **TensorRT** | | | |

```bash
# yolo export model=checkpoints/yolo11s_opt_pruned.pt format=engine imgsz=... half=True device=0
# TODO: python src/training/experiments/bench_opt.py --device win --backend tensorrt ...
```

If TensorRT FP16 already ≥ 30 FPS at S1 with mAP inside delta, W1 INT8 is optional (still run if you need headroom or a smaller engine).

## Round W1 — TensorRT precision

Fixed: S2 weights, S1 imgsz. Runtime = TensorRT. One factor: precision.

| id | Precision | Calib |
| -- | --------- | ----- |
| `trt_fp32` | FP32 engine | — (control; often slower than FP16) |
| `trt_fp16` | FP16 | — |
| `trt_int8` | INT8 | 256 train tiles |

Parity smoke vs `.pt`, then `eval_manual` + GPU bench. INT8: watch **band B**.

| run | mean mAP@0.5 | A / B | Det A / B | FPS (GPU) | engine MB |
| --- | ------------ | ----- | --------- | --------- | --------- |
| `trt_fp32` | | | | | |
| `trt_fp16` | | | | | |
| `trt_int8` | | | | | |

**Winner W1 →** `checkpoints/opt/win/*.engine`

## Round W2 — OpenVINO CPU fallback

Fixed: same S2 weights / imgsz as W1. Runtime = **OpenVINO** on Intel CPU (or iGPU if present). Precision: winner of a small Q-style sweep (`ov_fp32` / `ov_fp16` / `ov_int8`) — one factor.

This is **not** a TensorRT competitor. Pass = mean mAP@0.5 within `winner_delta` (or recorded drop) and some usable CPU FPS (propose **10**, same as Pi floor). Fail = “no CPU fallback”.

```bash
# yolo export model=checkpoints/yolo11s_opt_pruned.pt format=openvino imgsz=...
```

| run | mean mAP@0.5 | A / B | Det A / B | FPS (CPU) | pass fallback SLA? |
| --- | ------------ | ----- | --------- | --------- | ------------------ |
| `ov_fp32` | | | | | |
| `ov_fp16` | | | | | |
| `ov_int8` | | | | | |

**Winner W2 →** `checkpoints/opt/win/openvino/` (optional artifact next to the engine).

## Round W3 — live camera (not another train)

Fixed: W1 TensorRT engine (primary). Same skip/conf cells as Pi P2. Capture API = DirectShow / Media Foundation / whatever the box has — one-off, not an ablation.

Score skip/conf on `eval_manual`, then 15 min on the real camera: FN/FA by eye, GPU temp, FPS once warm.

| run | mean mAP@0.5 (eval) | Det A / B | camera e2e FPS | notes |
| --- | ------------------- | --------- | -------------- | ----- |
| `skip1` | | | | |
| `skip2` | | | | |
| `skip3` | | | | |
| `conf15` | | | | |
| `conf25` | | | | |
| `conf40` | | | | |

## Windows lock-in

| Field | Prototype | Win live (TensorRT) | Win fallback (OpenVINO) |
| --- | --- | --- | --- |
| **Artifact** | `.pt` | *fill* `.engine` | *fill* OpenVINO IR |
| **imgsz** | 1280 | *fill* | same unless noted |
| **Prune** | none | *fill* (S2) | same |
| **Precision** | FP32 | *fill* (W1) | *fill* (W2) |
| **conf / skip** | 0.25 / n/a | *fill* | *fill* |
| **mean A+B mAP@0.5** | 87.7% | *fill* | *fill* |
| **Det A / B** | 98.0% / 92.9% | *fill* | *fill* |
| **FPS** | — | *fill* GPU | *fill* CPU |

```bash
python src/training/evaluate.py \
  --gt manual \
  --weights checkpoints/opt/win/<locked.engine> \
  --imgsz <S1> \
  --no-video \
  --output-dir outputs/eval_opt_win_final
# TODO: bench_opt.py --device win --backend tensorrt|openvino
```

---

## Order of work

1. **S0–S2** on the train machine (quality + prune). Do not start device INT8 before S2 exists (or an explicit decision to skip prune).
2. **Pi:** flash, camera (`rpicam-hello`), same Ultralytics pin or only the OpenVINO runtime. P0 → P1 → P2.
3. **Windows:** install CUDA / cuDNN / TensorRT matching the export pin. W0 → W1 → W2 (fallback) → W3.
4. Do not tune conf to rescue a prune that killed band-B Det.
5. Do not copy a TensorRT engine to the Pi. Do not treat OpenVINO-on-Pi FPS as comparable to OpenVINO-on-Intel-CPU.
