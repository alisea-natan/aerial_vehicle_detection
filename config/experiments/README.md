# Experiments — two rounds

Settings here are git-tracked. Dataset **blobs** under `data/datasets/` and run artifacts under `outputs/experiments/` are local only.

Labels: **CVAT** `labels/`. Eval: **manual** pack (`eval_manual/`).

Eval bands (metrics): **A (&lt;200 m)** = `13722965…`, **B (&gt;200 m)** = `266987`. Decision metric: **mAP@0.5** per band (find the car). mAP@0.5:0.95 is diagnostic only.

| Round | Config | Code | Question |
| ----- | ------ | ---- | -------- |
| **1. Datasets** | [`dataset_round.yaml`](dataset_round.yaml) | `src/training/experiments/run_dataset_round.py` | Which **train pack** (YOLO11s `--prototype`, 2+5)? Tiling × stride, negatives, clip balance, imgsz 768/1280. |
| **2. Train setup** | [`model_round.yaml`](model_round.yaml) | `src/training/experiments/run_model_round.py` | Protocol → family → size → epochs on `strided_imgsz768`. Writes `outputs/experiments/model_round/`. |

### Round 2 groups (one factor each)

| Group | What | Command |
| ----- | ---- | ------- |
| **P** protocol | 11s frozen-20 / 2+5 / 5+15 | `python src/training/experiments/run_model_round.py --all` |
| Rank + expand | mean native mAP@0.5 | automatic with `--all`; or `--pick-winner` per group |
| **F** family | v8s vs 11s, same protocol | (from P winner) |
| **S** size | n / s / m | (from F winner) |
| **E** epochs | 5 / 10 / 20 (frozen) or Stage-2 5 / 10 / 15 | (from S winner) |

No LR grid. No P2 / YOLO26 / NMS-off.

```bash
python src/training/experiments/run_model_round.py --all
python src/training/experiments/run_model_round.py --all --resume
python src/training/experiments/run_model_round.py --all --list
```
