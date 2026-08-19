# Labeling guide

Vehicle definition (what is / is not a vehicle) → **[README.md](../../README.md)**.

Pipeline GT is always **`labels/{train|eval}/{clip}/*.txt`** (YOLO txt, empty file = no vehicles). Frames must match stems in `data/frames/{clip}/`. Use preprocess `frame_step` as annotation density.

```
src/labeling/
  README.md          ← this file
  autolabel.py       ← YOLO-World demo (PoC only; never writes labels/)
  cvat/              ← CVAT import / QA
  roboflow/          ← Roboflow prepare / upload / import
```

Downloaded exports you unpack locally go under `labelling/` (gitignored packs). Do not treat those copies as GT.

## CVAT

Credentials: copy `.env.example` → `.env` (`CVAT_HOST`, `CVAT_USER`, `CVAT_PASS`, optional `CVAT_PROJECT`). Task names must match clip stems under `data/frames/`. Split comes from `data/train` vs `data/eval`.

```bash
python src/labeling/cvat/cvat_pull.py --project aerial_vehicles --list
python src/labeling/cvat/cvat_pull.py --project aerial_vehicles --verify --sync-labels
# → labels/{train|eval}/{clip}/*.txt

python src/labeling/cvat/cvat_pull.py --project aerial_vehicles --task 7 --with-images --verify
python src/labeling/cvat/cvat_merge.py    # optional flat pack under data/dataset/
python src/labeling/cvat/compare.py       # CVAT labels/ vs outputs/autolabel/
```

## Roboflow

Optional. Upload packs are generated under `labelling/roboflow/` (local, deletable). Import a downloaded YOLOv8 export with `--export-dir`.

```bash
python src/labeling/roboflow/prepare_roboflow.py --clean
python src/labeling/roboflow/upload_roboflow.py
python src/labeling/roboflow/import_roboflow_dataset.py --export-dir path/to/yolov8_export
```

Helpers: `prepare_roboflow_test.py`, `prepare_roboflow_ui_8457857.py`, `retry_roboflow_annotations.py`, `deploy_roboflow.py`.

## Autolabel (PoC bootstrap only)

Not training GT. Writes `outputs/autolabel/` only.

```bash
python src/labeling/autolabel.py
python src/labeling/autolabel.py --clip 266987
```
