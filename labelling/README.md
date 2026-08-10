# Manual labeling helpers (CVAT + Roboflow)

```
labelling/
├── cvat/
│   ├── label_man/           # CVAT YOLO exports (2 eval + 1 train)
│   ├── coco_export/         # COCO / CVAT track exports
│   └── compare_labels.py    # manual vs autolabel IoU report
└── roboflow/
    ├── Vehicle_roboflow/            # 1-frame Roboflow export (8968356)
    ├── roboflow/                    # prepared per-clip folders (symlinks + labels)
    ├── roboflow_upload/             # YOLOv8 test split for API upload
    ├── roboflow_ui_upload_8457857/  # flat UI upload pack (train)
    ├── prepare_roboflow.py
    ├── prepare_roboflow_test.py
    ├── prepare_roboflow_ui_8457857.py
    ├── upload_roboflow.py
    ├── retry_roboflow_annotations.py
    └── deploy_roboflow.py
```

## CVAT

```bash
python labelling/cvat/compare_labels.py
# → debug/compare_labels_report.txt
```

## Roboflow

```bash
python labelling/roboflow/prepare_roboflow.py --clean
python labelling/roboflow/prepare_roboflow_test.py
python labelling/roboflow/prepare_roboflow_ui_8457857.py
python labelling/roboflow/upload_roboflow.py
python labelling/roboflow/retry_roboflow_annotations.py
python labelling/roboflow/deploy_roboflow.py
```

Manual labels live under `labelling/cvat/label_man/`. Pipeline frames stay in `data/frames/`; autolabels in `labels/`.
