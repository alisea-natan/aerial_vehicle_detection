# Інструкція: MLflow + DVC + CVAT -> baseline YOLOv11s

## 0. Архітектура рішення (що з чим звʼязано)

```
CVAT (jobs, розмітка)
   |  export dataset (YOLO format)
   v
data/  <- DVC версіонує (не git, файли важкі)
   |
   v
train.py (ultralytics YOLOv11s)
   |  логує метрики/артефакти
   v
MLflow tracking server (self-hosted, локально)
   |
   v
mlruns/ + mlflow.db  <- теж можна тримати під DVC, або окремо на диску

git репозиторій тримає:
- код (train.py, конфіги, скрипти експорту з CVAT)
- .dvc-файли (покажчики на версії датасету, НЕ самі дані)
- dvc.yaml / params.yaml (пайплайн + гіперпараметри)
- НЕ тримає: сирі зображення, ваги моделей, mlruns/ (це все в DVC remote або .gitignore)
```

Принцип: git = код + вказівники, DVC = важкі файли (датасет, ваги), MLflow = метрики + порівняння прогонів. Кожен run в MLflow тегується git commit hash + DVC data hash -> повна відтворюваність.

---

## 1. Встановлення

```bash
pip install mlflow dvc cvat-sdk ultralytics
pip install "dvc[s3]"
```

---

## 2. Ініціалізація git + DVC (одноразово)

```bash
cd aerial_vehicle_detection

git init
dvc init
git add .dvc .dvcignore
git commit -m "chore: init DVC"

dvc remote add -d storage /mnt/dvc-storage/aerial_dataset

git add .dvc/config
git commit -m "chore: configure DVC remote"
```

Варіант з S3-сумісним сховищем (MinIO):
```bash
dvc remote add -d storage s3://your-bucket/aerial_dataset
dvc remote modify storage endpointurl https://your-minio-host:9000
```

Важливо для DefTech-контексту: тримай DVC remote локально або на власному сервері, не на публічному хмарному S3 без шифрування.

---

## 3. Витягування датасету з CVAT (job-level export)

```python
# scripts/pull_from_cvat.py
import os
from pathlib import Path
from cvat_sdk import make_client

CVAT_HOST = "https://your-cvat-instance.local"
CVAT_USER = os.environ["CVAT_USER"]
CVAT_PASS = os.environ["CVAT_PASS"]

TASK_IDS = [12, 13, 14]
EXPORT_FORMAT = "YOLO 1.1"
OUTPUT_DIR = Path("data/raw_export")

def pull_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with make_client(host=CVAT_HOST, credentials=(CVAT_USER, CVAT_PASS)) as client:
        for task_id in TASK_IDS:
            task = client.tasks.retrieve(task_id)
            jobs = task.get_jobs()
            not_done = [j.id for j in jobs if j.state != "completed"]
            if not_done:
                print(f"[WARN] task {task_id}: jobs ще не completed: {not_done}")

            export_path = OUTPUT_DIR / f"task_{task_id}.zip"
            print(f"Exporting task {task_id} -> {export_path}")
            task.export_dataset(
                format_name=EXPORT_FORMAT,
                filename=str(export_path),
                include_images=True,
            )
    print("Done.")

if __name__ == "__main__":
    pull_dataset()
```

```bash
export CVAT_USER=your_login
export CVAT_PASS=your_password
python scripts/pull_from_cvat.py
```

Скрипт злиття декількох task-експортів в єдину YOLO-структуру:

```python
# scripts/merge_cvat_exports.py
import zipfile
import shutil
from pathlib import Path

RAW = Path("data/raw_export")
DATASET = Path("data/dataset")

def merge():
    for split_dir in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DATASET / split_dir).mkdir(parents=True, exist_ok=True)

    for zip_path in RAW.glob("task_*.zip"):
        extract_to = RAW / zip_path.stem
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_to)

        for img in (extract_to / "obj_train_data").glob("*.jpg"):
            shutil.copy(img, DATASET / "images/train" / img.name)
        for label in (extract_to / "obj_train_data").glob("*.txt"):
            shutil.copy(label, DATASET / "labels/train" / label.name)

    print("Merge done ->", DATASET)

if __name__ == "__main__":
    merge()
```

Порада: якщо train/val split закладений на рівні CVAT task_id (окремі таски = окремі відеозаписи, video-level split проти temporal leakage) - розводь по train/val саме за task_id, а не рандомно по кадрах.

---

## 4. Версіонування датасету через DVC

```bash
dvc add data/dataset
git add data/dataset.dvc data/.gitignore
git commit -m "data: v1 dataset from CVAT tasks 12-14"

dvc push
git push
```

Кожного разу коли розмітка змінюється:

```bash
python scripts/pull_from_cvat.py
python scripts/merge_cvat_exports.py

dvc add data/dataset
git add data/dataset.dvc
git commit -m "data: v2 dataset - added tasks 15-16, fixed B_medium mislabels"
dvc push
git push
```

DVC хеш у `data/dataset.dvc` унікально ідентифікує версію файлів - це і є "версія датасету" для прив'язки до MLflow run.

Відкат до старої версії датасету:

```bash
git checkout <commit_hash_старого_датасету> -- data/dataset.dvc
dvc checkout
```

---

## 5. MLflow сервер (self-hosted, локально)

```bash
mkdir -p mlflow_server
cd mlflow_server

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000
```

Тримай запущеним в окремому терміналі/tmux/systemd-сервісі. UI на http://localhost:5000.

sqlite достатньо для маленького проєкту. Для більшого масштабу - Postgres (`postgresql://user:pass@host/db`).

---

## 6. Інтеграція MLflow з ultralytics (YOLOv11s baseline)

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="aerial_vehicle_detection"
```

```python
# scripts/train_baseline.py
import subprocess
import mlflow
from ultralytics import YOLO

def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

def get_dvc_data_hash():
    with open("data/dataset.dvc") as f:
        for line in f:
            if "md5" in line:
                return line.strip()
    return "unknown"

def main():
    run_name = "exp10_yolov11s_baseline"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dvc_data_hash", get_dvc_data_hash())
        mlflow.set_tag("stage", "baseline")

        model = YOLO("yolo11s.pt")

        results = model.train(
            data="data/dataset/data.yaml",
            epochs=100,
            imgsz=640,
            batch=16,
            lr0=0.01,
            seed=42,
            project="runs/detect",
            name=run_name,
        )

        # окремі метрики по distance band рахуються окремим eval-скриптом:
        # mlflow.log_metric("mAP50_C_far", value)

if __name__ == "__main__":
    main()
```

```bash
python scripts/train_baseline.py
```

`data/dataset/data.yaml`:

```yaml
path: data/dataset
train: images/train
val: images/val
names:
  0: drone
  1: bird
```

---

## 7. Що логується автоматично vs вручну

Автоматично через ultralytics MLflow integration:
- Всі гіперпараметри (imgsz, batch, lr0, epochs, ...)
- Криві loss/metrics по епохах (box_loss, cls_loss, mAP50, mAP50-95, precision, recall)
- Фінальні ваги (best.pt) як artifact
- Графіки (confusion matrix, PR-curve)

Треба додати вручну:

```python
mlflow.log_metric("mAP50_A_close", map_close)
mlflow.log_metric("mAP50_B_medium", map_medium)
mlflow.log_metric("mAP50_C_far", map_far)
mlflow.log_param("tiling_enabled", False)
mlflow.log_param("tile_size", "none")
mlflow.log_param("augmentation_set", "default")

mlflow.log_artifacts("runs/detect/exp10_yolov11s_baseline/val_predictions_sample")

mlflow.set_tag("notes", "baseline без tiling, для порівняння з exp-tiled")
```

---

## 8. Повний git-workflow під один цикл експерименту

```bash
# 1. якщо змінилась розмітка/датасет
python scripts/pull_from_cvat.py
python scripts/merge_cvat_exports.py
dvc add data/dataset
git add data/dataset.dvc
git commit -m "data: v3 - added new tiling-oriented annotations"
dvc push

# 2. зміна конфігу/гіперпараметрів - окремий commit ДО тренування
git add scripts/train_baseline.py params.yaml
git commit -m "exp: yolov11s baseline lr0=0.01 seed=42"
git push

# 3. тренування - MLflow автоматично прив'яже run до поточного git commit
python scripts/train_baseline.py

# 4. переглянути результат в MLflow UI, порівняти прогони side-by-side
```

Git branching під ряди експериментів:

```bash
git checkout -b exp/dataset-ablation      # ряд 1: датасет
# кожен варіант аугментації/тайлінгу - окремий commit або гілка

git checkout main
git checkout -b exp/model-ablation        # ряд 2: модель, від найкращого датасету з ряду 1
```

Кожен run в MLflow має чіткий git commit -> можна `git checkout <hash>` і відтворити рівно той код+конфіг, який дав конкретний mAP.

---

## 9. Структура репозиторію (підсумково)

```
aerial_vehicle_detection/
├── .dvc/
│   └── config
├── .git/
├── .gitignore              # містить data/dataset/ (реальні файли - DVC)
├── data/
│   ├── dataset.dvc           # покажчик версії (у git)
│   ├── raw_export/           # сирі zip з CVAT
│   └── dataset/               # фактичні файли (НЕ в git, тягнеться через dvc pull)
│       ├── data.yaml
│       ├── images/{train,val}
│       └── labels/{train,val}
├── scripts/
│   ├── pull_from_cvat.py
│   ├── merge_cvat_exports.py
│   └── train_baseline.py
├── params.yaml
├── requirements.txt
└── README.md
```

---

## 10. Для когось нового (або тебе через місяць), хто клонує репо

```bash
git clone <repo_url>
cd aerial_vehicle_detection

pip install -r requirements.txt

dvc pull

export MLFLOW_TRACKING_URI="http://localhost:5000"
python scripts/train_baseline.py
```

git+DVC+MLflow разом дають повну відтворюваність: будь-який run можна відкотити до точного коду + точних даних + точних гіперпараметрів, а MLflow UI дає порівняння прогонів "своїми очима" (криві, sample-прогнози, метрики по бендах) саме так, як треба для обох рядів експериментів.
