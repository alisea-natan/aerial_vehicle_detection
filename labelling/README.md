# Annotation artifacts only (CVAT / Roboflow exports). Code lives in `src/labeling/`.

```
labels/{train|eval}/{clip}/*.txt   ← CVAT GT via: python src/labeling/cvat_pull.py --sync-labels
outputs/autolabel/labels/…         ← YOLO-World (compare against labels/)
labelling/roboflow/                ← optional upload packs (gitignored)
```

```bash
python src/labeling/cvat_pull.py --verify --sync-labels
python src/labeling/compare.py   # autolabel vs labels/ (CVAT)
```
