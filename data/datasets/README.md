# Dataset versions (local only)

Prepared Ultralytics layouts live here. **Do not push dataset blobs or `*.dvc` to GitLab** — only this folder layout is tracked.

```
data/datasets/
  README.md              # this file (in git)
  baseline_v0/           # optional: from YOLO-World autolabel (pseudo-GT)
  baseline_v1/           # human / CVAT GT baseline (frame_step, ~1024 tiling, no aug)
  <experiment_name>/     # further local experiment packs
```

Build:

```bash
# GT baseline (default)
python src/training/prepare_baseline.py
# → data/datasets/baseline_v1/

# Theoretical / smoke baseline from autolabel demo
python src/labeling/autolabel.py
python src/training/prepare_baseline.py --from-autolabel
# → data/datasets/baseline_v0/
```

Optional local DVC (cache in `.dvc-storage/`, not GitLab):

```bash
dvc add data/datasets/baseline_v1
dvc push
```
