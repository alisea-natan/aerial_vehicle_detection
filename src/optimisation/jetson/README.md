# Jetson — Track A4 (Colab mock)

Prerequisites and ship artifact: **[OPTIMISATION.md](../../../OPTIMISATION.md)** Track A1 FP32 ONNX. TensorRT 11 (Colab) has no `--fp16` flag — AutoCast mixed precision on the GPU, then build.

Upload `checkpoints/opt/export/prototype/onnx/yolo11s_prototype_best.onnx`. Defaults: `imgsz=1280`.

---

## 1. Colab mock — ms per tile

Open `jetson_colab_mock.ipynb` in [Google Colab](https://colab.research.google.com/) with **GPU** (T4).

1. Upload `yolo11s_prototype_best.onnx`.
2. Run all — AutoCast mixed FP16 + TensorRT engine + bench (dummy; optional pack JPEGs).
3. Results are in **OPTIMISATION.md** § A4 (pack-tile row).

---

## 2. Live video


| stage       | live video                                                                    |
| ----------- | ----------------------------------------------------------------------------- |
| Colab mock  | skip — ms bench only                                                          |
| Real Jetson | TRT `.engine` + camera (mirror `../raspberry/live_camera.py` with `device=0`) |

---

## 3. Real Jetson (when available)

1. Copy A1 ONNX to device (rebuild `.engine` if CUDA/TRT versions differ).
2. Same notebook cells (AutoCast + engine build). JetPack 10 still uses `trtexec --fp16`; JetPack / TRT 11 does not.
3. Bench with the same warmup loop.
4. Update the Jetson row in **OPTIMISATION.md** § A4 (add a real-device note in `source` if needed).

[jetson-containers](https://github.com/dusty-nv/jetson-containers) is a common base image.

---

## Files


| file                      | role                             |
| ------------------------- | -------------------------------- |
| `jetson_colab_mock.ipynb` | Colab GPU: A1 ONNX → AutoCast → TRT bench |


Cross-target comparison: **OPTIMISATION.md § A4**.
