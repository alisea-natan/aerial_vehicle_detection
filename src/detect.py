"""YOLO-World inference, SAHI tiling, and bbox → distance (m)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from config import PROJECT_ROOT, PROBE_CLASS_ALIASES, RAW_CONFIDENCE_THRESHOLD, overlap_for_tiles
from sahi.predict import get_sliced_prediction

if TYPE_CHECKING:
    from sahi import AutoDetectionModel

MODEL_NAME = "yolov8x-worldv2.pt"
MIN_SLICE_PX = 512

CALIBRATION_DIR = PROJECT_ROOT / "calibration"
CAR_LENGTH_M = 4.5

SENSOR_1INCH = {"sensor_width_mm": 13.2, "sensor_height_mm": 8.8}
SENSOR_123 = {"sensor_width_mm": 6.17, "sensor_height_mm": 4.55}
DEFAULT_FOCAL_LENGTH_MM = 24.0


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def compute_slice_size(width: int, height: int, target_tiles: int) -> tuple[int, int]:
    slice_size = max(MIN_SLICE_PX, int(math.isqrt(width * height // target_tiles)))
    return slice_size, slice_size


def parse_ultralytics_result(result) -> list[dict]:
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    for i in range(len(boxes)):
        subclass_name = result.names[int(cls_ids[i])].lower()
        detections.append({
            "xyxy": [float(v) for v in boxes[i]],
            "confidence": float(confs[i]),
            "subclass_name": subclass_name,
        })
    return detections


def parse_sahi_result(result, *, default_subclass: str = "") -> list[dict]:
    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        category = getattr(pred, "category", None)
        subclass_name = (getattr(category, "name", "") or default_subclass).lower()
        detections.append({
            "xyxy": [float(bbox.minx), float(bbox.miny), float(bbox.maxx), float(bbox.maxy)],
            "confidence": float(pred.score.value),
            "subclass_name": subclass_name,
        })
    return detections


def build_yolo_world(classes: list[str]) -> tuple["AutoDetectionModel", str]:
    from sahi import AutoDetectionModel

    dev = device()
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_NAME,
        confidence_threshold=RAW_CONFIDENCE_THRESHOLD,
        device=dev,
    )
    model.model.set_classes(classes)
    return model, dev


def filter_probe_detections(
    detections: list[dict],
    primary_class: str,
    label_threshold: float,
    aliases: dict[str, str] | None = None,
) -> list[dict]:
    """Keep primary class and aliases (e.g. person counted as car); normalize tag to primary."""
    aliases = aliases if aliases is not None else PROBE_CLASS_ALIASES
    primary = primary_class.lower()
    accepted = {primary, *aliases.keys()}
    kept: list[dict] = []
    for det in detections:
        name = det["subclass_name"].lower()
        if name not in accepted or det["confidence"] < label_threshold:
            continue
        normalized = aliases.get(name, name)
        if normalized != primary:
            continue
        out = dict(det)
        out["subclass_name"] = primary
        if name != primary:
            out["probe_matched_as"] = name
        kept.append(out)
    return kept


def filter_detections(
    detections: list[dict],
    detection_class: str,
    label_threshold: float,
) -> list[dict]:
    target = detection_class.lower()
    return [
        det for det in detections
        if det["subclass_name"] == target and det["confidence"] >= label_threshold
    ]


def detect_frame_sahi(
    model: "AutoDetectionModel",
    image_path: Path,
    slice_h: int,
    slice_w: int,
    overlap_ratio: float,
    *,
    default_subclass: str = "vehicle",
    enhance: bool = False,
) -> list[dict]:
    from image_enhance import inference_source

    source = inference_source(image_path, enhance=enhance)
    result = get_sliced_prediction(
        source,
        model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        force_postprocess_type=True,
        verbose=0,
    )
    return parse_sahi_result(result, default_subclass=default_subclass)


def detect_frame_probe(
    model: "AutoDetectionModel",
    ultra_model,
    frame_path: Path,
    width: int,
    height: int,
    target_tiles: int,
    label_threshold: float,
    detection_class: str,
    device_name: str,
    *,
    enhance: bool = False,
) -> list[dict]:
    from image_enhance import inference_source

    source = inference_source(frame_path, enhance=enhance)
    if target_tiles <= 1:
        result = ultra_model.predict(
            source,
            conf=RAW_CONFIDENCE_THRESHOLD,
            verbose=False,
            device=device_name,
        )[0]
        return filter_probe_detections(
            parse_ultralytics_result(result), detection_class, label_threshold
        )

    overlap = overlap_for_tiles(target_tiles)
    slice_h, slice_w = compute_slice_size(width, height, target_tiles)
    result = get_sliced_prediction(
        source,
        model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        force_postprocess_type=True,
        verbose=0,
    )
    return filter_probe_detections(
        parse_sahi_result(result), detection_class, label_threshold
    )


# --- camera model + distance ---


def resolution_tier(width: int, height: int, clip_name: str) -> str:
    name = clip_name.lower()
    if "uhd" in name or re.search(r"_3840_2160_|_2160_3840_", name):
        return "4k"
    if "hd" in name or re.search(r"_1920_1080_", name):
        return "1080p"
    long_edge = max(width, height)
    if long_edge >= 3000:
        return "4k"
    if long_edge >= 1900:
        return "1080p"
    return "sd"


def default_sensor_for_tier(tier: str) -> dict[str, float]:
    if tier == "sd":
        return SENSOR_123.copy()
    return SENSOR_1INCH.copy()


def fov_deg_from_sensor(sensor_mm: float, focal_mm: float) -> float:
    return math.degrees(2.0 * math.atan(sensor_mm / (2.0 * focal_mm)))


def camera_from_physics(
    focal_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    image_width: int,
    image_height: int,
    source: str,
) -> dict:
    focal_px = focal_mm / sensor_height_mm * image_height
    return {
        "source": source,
        "focal_length_mm": focal_mm,
        "sensor_width_mm": sensor_width_mm,
        "sensor_height_mm": sensor_height_mm,
        "vertical_fov_deg": round(fov_deg_from_sensor(sensor_height_mm, focal_mm), 2),
        "horizontal_fov_deg": round(fov_deg_from_sensor(sensor_width_mm, focal_mm), 2),
        "focal_px": round(focal_px, 2),
        "image_width": image_width,
        "image_height": image_height,
        "resolution_tier": None,
    }


def camera_from_vertical_fov(vertical_fov_deg: float, image_width: int, image_height: int, source: str) -> dict:
    half_fov_rad = math.radians(vertical_fov_deg / 2.0)
    focal_px = image_height / (2.0 * math.tan(half_fov_rad))
    aspect = image_width / image_height
    sensor_height_mm = 2.0 * DEFAULT_FOCAL_LENGTH_MM * math.tan(half_fov_rad)
    sensor_width_mm = sensor_height_mm * aspect
    return {
        "source": source,
        "focal_length_mm": DEFAULT_FOCAL_LENGTH_MM,
        "sensor_width_mm": round(sensor_width_mm, 3),
        "sensor_height_mm": round(sensor_height_mm, 3),
        "vertical_fov_deg": round(vertical_fov_deg, 2),
        "horizontal_fov_deg": round(fov_deg_from_sensor(sensor_width_mm, DEFAULT_FOCAL_LENGTH_MM), 2),
        "focal_px": round(focal_px, 2),
        "image_width": image_width,
        "image_height": image_height,
        "resolution_tier": None,
    }


def load_calibration_override(clip_name: str) -> dict | None:
    calib_path = CALIBRATION_DIR / f"{clip_name}.json"
    if not calib_path.exists():
        return None
    return json.loads(calib_path.read_text(encoding="utf-8"))


def resolve_camera_model(
    clip_name: str,
    width: int,
    height: int,
    vertical_fov_override: float | None = None,
) -> dict:
    tier = resolution_tier(width, height, clip_name)
    calib = load_calibration_override(clip_name)

    if vertical_fov_override is not None:
        camera = camera_from_vertical_fov(vertical_fov_override, width, height, "cli_override")
        camera["resolution_tier"] = tier
        return camera

    if calib and "vertical_fov_deg" in calib and "focal_length_mm" not in calib:
        camera = camera_from_vertical_fov(float(calib["vertical_fov_deg"]), width, height, "calibration_fov")
        camera["resolution_tier"] = tier
        return camera

    sensor = default_sensor_for_tier(tier)
    focal_mm = DEFAULT_FOCAL_LENGTH_MM
    source = f"inferred_{tier}_24mm"

    if calib:
        if "focal_length_mm" in calib:
            focal_mm = float(calib["focal_length_mm"])
            source = "calibration_focal"
        if "sensor_width_mm" in calib and "sensor_height_mm" in calib:
            sensor = {
                "sensor_width_mm": float(calib["sensor_width_mm"]),
                "sensor_height_mm": float(calib["sensor_height_mm"]),
            }
            source = "calibration_full" if "focal_length_mm" in calib else "calibration_sensor"

    camera = camera_from_physics(
        focal_mm,
        sensor["sensor_width_mm"],
        sensor["sensor_height_mm"],
        width,
        height,
        source,
    )
    camera["resolution_tier"] = tier
    return camera


def bbox_long_side_px(xyxy) -> float:
    x1, y1, x2, y2 = xyxy
    return max(max(0.0, x2 - x1), max(0.0, y2 - y1))


def bbox_area(xyxy) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def vehicle_distance_m(bbox_long_side_px: float, focal_px: float) -> float | None:
    if bbox_long_side_px <= 0:
        return None
    return CAR_LENGTH_M * focal_px / bbox_long_side_px


def car_detection_record(det: dict, focal_px: float) -> dict | None:
    x1, y1, x2, y2 = (float(v) for v in det["xyxy"])
    xyxy = [x1, y1, x2, y2]
    long_side_px = bbox_long_side_px(xyxy)
    distance_m = vehicle_distance_m(long_side_px, focal_px)
    if distance_m is None:
        return None
    return {
        "bbox": xyxy,
        "confidence": round(float(det["confidence"]), 4),
        "bbox_long_side_px": round(long_side_px, 1),
        "bbox_area_px": round(bbox_area(xyxy), 1),
        "distance_m": round(distance_m, 1),
    }


def pick_largest_car(cars: list[dict]) -> dict | None:
    if not cars:
        return None
    return max(cars, key=lambda car: car["bbox_area_px"])


def distance_band(distance_m: float | None) -> str | None:
    if distance_m is None:
        return None
    if distance_m < 200:
        return "<200m"
    if distance_m < 400:
        return ">200m"
    return ">400m"
