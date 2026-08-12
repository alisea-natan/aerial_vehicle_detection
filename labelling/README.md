# Annotation artifacts only (CVAT / Roboflow exports). Code lives in src/labeling/.
#
# labelling/cvat/label_man/     — manual YOLO exports from CVAT
# labelling/roboflow/           — upload packs / exports (gitignored packs)
#
#   python src/labeling/cvat_pull.py --verify --sync-labels
#   python src/labeling/compare.py
#   python src/labeling/roboflow/prepare_roboflow.py --clean
