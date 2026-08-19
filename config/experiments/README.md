# Experiments — two rounds

Settings here are git-tracked. Dataset **blobs** under `data/datasets/` and run artifacts under `outputs/experiments/` are local only.

Labels: **CVAT** `labels/`. Eval: **native + scale-adapted** manual packs.

Eval bands (metrics): **A (&lt;200 m)** = `13722965…`, **B (&gt;200 m)** = `266987`. Decision metric: **mAP@0.5** per band (find the car). mAP@0.5:0.95 is diagnostic only.

| Round | Config | Code | Question |
| ----- | ------ | ---- | -------- |
| **1. Datasets** (done) | [`dataset_round.yaml`](dataset_round.yaml) | `src/training/experiments/run_dataset_round.py` | Which **train pack** works best with YOLO11s `--prototype`? |
| **2. Models** | [`model_round.yaml`](model_round.yaml) | `src/training/experiments/run_model_round.py` | Which **architecture / LR / head** on the Round 1 pack? |

Set `defaults.dataset` in `model_round.yaml` to the Round 1 winner (**`variant_6_strided`** — highest native mAP@0.5 on both bands).

### Round 2 groups

| Group | Command |
| ----- | ------- |
| A — 3 YOLO × 3 LR (9 runs) | `python src/training/experiments/run_model_round.py` |
| A resume | `python src/training/experiments/run_model_round.py --resume` |
| Rank YOLO winners | `python src/training/experiments/run_model_round.py --pick-winner` |
| B — fine LR | `--group B --winner <id>` |
| C — hard NMS / no NMS / Soft-NMS (eval only) | `--group C --winner <id>` |
| D — backbone | `--group D --winner <id>` |
| E — P2 head | `--group E --winner <id>` |

Winner rule: if top-2 mean mAP@0.5 (A+B, native) differ by ≥ 1.5 pts, skip C on the second model; otherwise C on both, then one model into D/E. Group C compares hard NMS, no NMS, and Gaussian Soft-NMS on frozen weights.

Every train run except Group D frozen-head **fine-tunes the backbone**. Group D scratch is the only run without a COCO-pretrained backbone.

Wall time: `outputs/experiments/{dataset_round,model_round}/round_timing.json` (per variant + per `--resume` session). `summary.md` has a **Wall time** section.

Logs: `outputs/experiments/{dataset_round,model_round}/logs/` — one `ds_<id>.log` / `md_<id>.log` per run, plus `logs/sessions/` for the round or group (A–E).

Apple Silicon: YOLO26 and P2 train without in-train val (including last-epoch / final val) and eval on CPU (GatherND / DFL channel limit). Variants run in child processes so a Metal abort cannot stop the rest of the round.

```bash
python src/training/experiments/run_model_round.py --list
python src/training/experiments/run_model_round.py --resume
```
