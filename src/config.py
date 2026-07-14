"""Project paths, clip discovery, and per-clip tiling config (clip_tiling.json)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
LABELS_DIR = PROJECT_ROOT / "labels"
DEBUG_DIR = PROJECT_ROOT / "debug"
CLIP_TILING_CONFIG_PATH = PROJECT_ROOT / "config" / "clip_tiling.json"

SPLITS = ("train", "eval")

DEFAULT_DETECTION_CLASS = "car"
# Aerial YOLO-World often tags top-down cars as person; treat as car during tile probe.
PROBE_CLASS_ALIASES: dict[str, str] = {"person": "car"}
RAW_CONFIDENCE_THRESHOLD = 0.05
PROBE_MAX_LABEL_THRESHOLD = 0.5
DEFAULT_OVERLAP = 0.10
FAR_OVERLAP = 0.05
FALLBACK_TILES = 12
FALLBACK_LABEL_THRESHOLD = 0.1
# Probe tries these in order; first hit = min tiles for the largest/nearest car.
TILE_CANDIDATES = (1, 2, 3, 4, 6, 8, 12)
# Final labeling uses one step above that hit so smaller/farther cars get more resolution.
TILE_HEADROOM_STEPS = 1

# Train/eval: global YOLO imgsz; per-clip scale_coeff sets crop so
# at_imgsz ≈ fullframe_long × scale_coeff  (target ~64 px median).
TRAIN_IMGSZ = 1280
TRAIN_OVERLAP_RATIO = 0.2
TARGET_OBJECT_LONG_PX = 64.0
DEFAULT_SCALE_COEFF = 1.0
SCALE_COEFF_MIN = 0.4
SCALE_COEFF_MAX = 3.0
MIN_TRAIN_SLICE_PX = 256


def build_split_map(root: Path | None = None) -> dict[str, str]:
    root = root or PROJECT_ROOT
    mapping: dict[str, str] = {}
    for split in SPLITS:
        split_dir = root / "data" / split
        if not split_dir.is_dir():
            continue
        for video_path in split_dir.iterdir():
            if video_path.suffix.lower() == ".mp4":
                mapping[video_path.stem] = split
    return mapping


def iter_frame_clip_dirs(clip_filter: str | None = None) -> list[Path]:
    """Every data/frames/* folder with metadata.json and at least one .jpg."""
    clips: list[Path] = []
    if not FRAMES_DIR.is_dir():
        return clips

    for clip_dir in sorted(FRAMES_DIR.iterdir()):
        if not clip_dir.is_dir():
            continue
        if not (clip_dir / "metadata.json").exists():
            continue
        if not any(clip_dir.glob("*.jpg")):
            continue
        if clip_filter and clip_dir.name != clip_filter:
            continue
        clips.append(clip_dir)
    return clips


def iter_autolabel_clips(
    split_map: dict[str, str],
    clip_filter: str | None = None,
) -> list[tuple[str, Path]]:
    clips: list[tuple[str, Path]] = []
    if not FRAMES_DIR.is_dir():
        return clips

    for clip_dir in sorted(FRAMES_DIR.iterdir()):
        if not clip_dir.is_dir():
            continue
        if not (clip_dir / "metadata.json").exists():
            continue
        if not any(clip_dir.glob("*.jpg")):
            continue

        clip_name = clip_dir.name
        if clip_filter and clip_name != clip_filter:
            continue

        split = split_map.get(clip_name)
        if split is None:
            if clip_filter:
                raise SystemExit(
                    f"Clip {clip_name!r} has frames but no matching .mp4 in data/train or data/eval."
                )
            print(f"Skipping {clip_name}: not in data/train or data/eval")
            continue
        clips.append((split, clip_dir))
    return clips


def probe_detection_class(payload: dict | None = None) -> str:
    data = payload or {}
    return str(data.get("detection_class", DEFAULT_DETECTION_CLASS)).lower()


def probe_model_classes(primary_class: str | None = None) -> list[str]:
    """YOLO-World class list for probe; includes aliases (e.g. person → car)."""
    primary = (primary_class or DEFAULT_DETECTION_CLASS).lower()
    classes = [primary]
    for alias, target in PROBE_CLASS_ALIASES.items():
        if target == primary and alias not in classes:
            classes.append(alias)
    return classes


def load_tiling_payload(path: Path | None = None) -> dict:
    config_path = path or CLIP_TILING_CONFIG_PATH
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_clip_tile_config(path: Path | None = None) -> dict[str, dict]:
    return load_tiling_payload(path).get("clips", {})


def label_threshold_for_tiles(target_tiles: int) -> float:
    if target_tiles <= 1:
        return PROBE_MAX_LABEL_THRESHOLD
    scaled = PROBE_MAX_LABEL_THRESHOLD / target_tiles
    return max(RAW_CONFIDENCE_THRESHOLD, round(scaled, 4))


def overlap_for_tiles(target_tiles: int) -> float:
    if target_tiles <= 1:
        return 0.0
    return FAR_OVERLAP if target_tiles >= 8 else DEFAULT_OVERLAP


def next_tile_candidate(tiles: int, steps: int = TILE_HEADROOM_STEPS) -> int:
    """Move `steps` levels up the TILE_CANDIDATES ladder (clamped at the last)."""
    if steps <= 0:
        return int(tiles)
    try:
        idx = TILE_CANDIDATES.index(int(tiles))
    except ValueError:
        greater = [t for t in TILE_CANDIDATES if t > tiles]
        return greater[0] if greater else TILE_CANDIDATES[-1]
    return TILE_CANDIDATES[min(idx + steps, len(TILE_CANDIDATES) - 1)]


def clamp_scale_coeff(scale: float) -> float:
    return float(min(SCALE_COEFF_MAX, max(SCALE_COEFF_MIN, scale)))


def scale_coeff_from_median(
    ff_p50: float,
    *,
    target_px: float = TARGET_OBJECT_LONG_PX,
) -> float:
    """Choose scale so median full-frame long-side ≈ target_px after imgsz resize."""
    if ff_p50 is None or ff_p50 <= 0:
        return DEFAULT_SCALE_COEFF
    return round(clamp_scale_coeff(float(target_px) / float(ff_p50)), 2)


def align_slice_size(size: int, multiple: int = 32, *, max_size: int | None = None) -> int:
    """Floor to multiple of `multiple`, capped by frame short side."""
    size = int(size)
    if max_size is not None:
        size = min(size, int(max_size))
    size = max(MIN_TRAIN_SLICE_PX, size)
    aligned = (size // multiple) * multiple
    return max(multiple, aligned)


def slice_size_from_scale(
    scale_coeff: float,
    frame_w: int,
    frame_h: int,
    *,
    imgsz: int = TRAIN_IMGSZ,
) -> int:
    """Train/eval crop side: slice ≈ imgsz / scale_coeff (capped to frame).

    After YOLO resizes the crop to imgsz, object long-side scales by ≈ scale_coeff.
    """
    coeff = clamp_scale_coeff(float(scale_coeff))
    raw = imgsz / coeff
    return align_slice_size(int(round(raw)), max_size=min(frame_w, frame_h))


def resolve_scale_coeff(clip_cfg: dict | None, default: float = DEFAULT_SCALE_COEFF) -> float:
    if not clip_cfg or "scale_coeff" not in clip_cfg:
        return float(default)
    return clamp_scale_coeff(float(clip_cfg["scale_coeff"]))


def save_clip_tile_config(
    clips: dict[str, dict],
    path: Path | None = None,
    *,
    source: str = "probe_clips.py",
    extra_meta: dict | None = None,
) -> Path:
    config_path = path or CLIP_TILING_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": source,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clips": clips,
    }
    if extra_meta:
        payload.update(extra_meta)

    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


def probe_result_to_config_entry(result: dict) -> dict:
    no_hit = result.get("min_tiles") is None
    probe_min_tiles = result.get("min_tiles")
    if probe_min_tiles is None:
        target_tiles = FALLBACK_TILES
        note = f"probe found no cars; fallback {FALLBACK_TILES} tiles @ {FALLBACK_LABEL_THRESHOLD}"
    else:
        probe_min_tiles = int(probe_min_tiles)
        target_tiles = next_tile_candidate(probe_min_tiles)
        hit_frame = result.get("hit_frame") or "?"
        distance_source = result.get("distance_source") or "unknown"
        if target_tiles == probe_min_tiles:
            note = (
                f"probed frame {hit_frame}; distance from {distance_source}; "
                f"probe_min_tiles={probe_min_tiles} (already max ladder step)"
            )
        else:
            note = (
                f"probed frame {hit_frame}; distance from {distance_source}; "
                f"probe_min_tiles={probe_min_tiles} → target_tiles={target_tiles} "
                f"(+{TILE_HEADROOM_STEPS} headroom)"
            )

    target_tiles = int(target_tiles)
    # Threshold/overlap follow the final labeling tile count, not the raw probe hit.
    if no_hit:
        label_threshold = FALLBACK_LABEL_THRESHOLD
    else:
        label_threshold = label_threshold_for_tiles(target_tiles)

    distance_band = result.get("distance_band")
    if not distance_band:
        distance_band = ">400m" if target_tiles >= 8 else ">200m"

    entry = {
        "target_tiles": target_tiles,
        "overlap_ratio": overlap_for_tiles(target_tiles),
        "label_confidence_threshold": label_threshold,
        "detection_class": result.get("detection_class", DEFAULT_DETECTION_CLASS),
        "distance_band": distance_band,
        "distance_m": result.get("distance_m"),
        "split": result.get("split"),
        "note": note,
    }
    if probe_min_tiles is not None:
        entry["probe_min_tiles"] = int(probe_min_tiles)
    return entry


def merge_probe_results(
    existing: dict[str, dict],
    probe_results: list[dict],
) -> dict[str, dict]:
    merged = dict(existing)
    for result in probe_results:
        clip_name = result["clip"]
        entry = probe_result_to_config_entry(result)
        # Preserve scale_coeff from label_box_stats.py across re-probes.
        old = existing.get(clip_name, {})
        if "scale_coeff" in old:
            entry["scale_coeff"] = old["scale_coeff"]
            if "scale_coeff_note" in old:
                entry["scale_coeff_note"] = old["scale_coeff_note"]
        merged[clip_name] = entry
    return merged


def resolve_clip_tile_config(
    clip_name: str,
    config: dict[str, dict] | None = None,
) -> dict:
    clips = config if config is not None else load_clip_tile_config()
    if clip_name not in clips:
        known = ", ".join(sorted(clips)) or "(empty)"
        raise SystemExit(
            f"No tile config for clip {clip_name!r}. "
            f"Run: python src/probe_clips.py --clip {clip_name}\n"
            f"Known clips in config: {known}"
        )

    cfg = clips[clip_name]
    target_tiles = int(cfg["target_tiles"])
    overlap_ratio = float(cfg.get("overlap_ratio", overlap_for_tiles(target_tiles)))
    if "label_confidence_threshold" not in cfg:
        raise SystemExit(
            f"Clip {clip_name!r} missing label_confidence_threshold in config. "
            f"Run: python src/probe_clips.py --clip {clip_name}"
        )
    return {
        "target_tiles": target_tiles,
        "overlap_ratio": overlap_ratio,
        "label_confidence_threshold": float(cfg["label_confidence_threshold"]),
        "detection_class": str(cfg.get("detection_class", DEFAULT_DETECTION_CLASS)),
        "distance_band": str(cfg.get("distance_band", "")),
        "distance_m": cfg.get("distance_m"),
        "note": str(cfg.get("note", "")),
        "uses_sahi": target_tiles > 1,
        "scale_coeff": resolve_scale_coeff(cfg),
    }
