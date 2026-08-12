# Dataset versions (local only)

Prepared Ultralytics packs live here. **Do not commit blobs or `*.dvc`** — only this README + `.gitignore` are tracked.

```
data/datasets/
  README.md
  baseline_v0/ / baseline_v1/     # prepare_baseline.py helpers
  eval_manual/                    # prepare_eval.py (CVAT / human)
  eval_autolabel/                 # prepare_eval.py --from-autolabel
  baseline_1/                     # ablation reference pack
  variant_2_no_tiling/ …
  variant_5_aug -> baseline_1     # symlink; train aug only in Round 1
```

## Build packs (dataset layer)

```bash
# Shared eval (once per label source; all tests reuse the same pack)
python src/training/prepare_eval.py --from-autolabel   # → eval_autolabel/
python src/training/prepare_eval.py                   # → eval_manual/

# Train packs
python src/training/prepare_baseline.py --from-autolabel
```

Ablation specs: `config/datasets/variants.yaml`

```bash
python src/training/datasets/generate_variant.py --list
python src/training/datasets/generate_variant.py --all
```

## Train (experiment rounds — separate)

| Round | Config | Command |
| ----- | ------ | ------- |
| 1 Datasets | `config/experiments/dataset_round.yaml` | `python src/training/experiments/run_dataset_round.py --dry-run --variant baseline_1` |
| 2 Models | `config/experiments/model_round.yaml` | `python src/training/experiments/run_model_round.py --dry-run --variant model_n_640` |

See `config/experiments/README.md`.
