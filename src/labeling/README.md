# Labeling guide

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

### RF-DETR Nano (trial)

Trained on Roboflow: **RF-DETR Object Detection (Nano)**, imgsz **1280×1280**, **3.15 credits**.

Split: **441** train / **22** valid / **153** test (eval). Train clips `8457857`, `5382494`. Test = both eval videos.

Roboflow **Valid Set (External)**:

| mAP@50 | Precision | Recall | F1 |
| -----: | --------: | -----: | -: |
| 97.7% | 98.4% | 95.6% | 97.0% |

Precision by split: valid **98%**, test (eval) **81%**.

## Autolabel (PoC bootstrap only)

Not training GT. Writes `outputs/autolabel/` only. YOLO-World **imgsz 1280**. Per-frame conf + dedupe on `frame_step` subsample — no tracking (see [PoC.md](../../PoC.md) §2).

```bash
python src/labeling/autolabel.py
python src/labeling/autolabel.py --clip 266987
```
