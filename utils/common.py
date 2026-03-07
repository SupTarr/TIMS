#!/usr/bin/env python3
"""
Shared utilities for CCTV tile image processing.

Provides common filename parsing, frame grouping, and tile selection
used by cluster_by_location.py, generate_cluster_preview.py, and classifier.py.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    # Constants
    "DENSITY_BASE_PATH",
    "RAW_TRAIN_PATH",
    "TRAIN_BY_LOCATION_PATH",
    "ROI_CONFIG_PATH",
    "CLUSTER_PREVIEW_PATH",
    "CLUSTER_CSV_PATH",
    "DENSITY_OUTPUT_PATH",
    "TIMS_FINAL_BASE_PATH",
    "TIMS_FINAL_IMAGES_PATH",
    "TIMS_FINAL_LABELS_PATH",
    "LANE_SEG_WEIGHTS_PATH",
    "CCTV_PATTERN",
    "CCTV_PATTERN_LOOSE",
    "CLASS_NAMES",
    "IMAGE_EXTENSIONS",
    "IR_SATURATION_THRESHOLD",
    "LANE_VEHICLE_CLASSES",
    # Functions
    "detect_frame_modality",
    "detect_modality",
    "discover_locations",
    "filter_vehicles_in_roi",
    "group_tiles_by_frame",
    "load_road_roi",
    "parse_filename",
    "parse_yolo_labels",
    "pick_representative",
    "pick_representatives",
    "save_road_roi",
    "setup_logging",
    "time_period",
]

# ──────────────────────────────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────────────────────────────
import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure standard root logger format for all utils."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Overwrite any existing root logger config
    )


PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

DENSITY_BASE_PATH = (
    PROJECT_ROOT_PATH / "gdrive" / "YOLOv10" / "data_train" / "TIMS_density_dataset"
)

RAW_TRAIN_PATH = DENSITY_BASE_PATH / "raw" / "train"
TRAIN_BY_LOCATION_PATH = DENSITY_BASE_PATH / "raw" / "train_by_location"

ROI_CONFIG_PATH = TRAIN_BY_LOCATION_PATH / "road_roi.json"
CLUSTER_PREVIEW_PATH = TRAIN_BY_LOCATION_PATH / "cluster_preview.png"
CLUSTER_CSV_PATH = TRAIN_BY_LOCATION_PATH / "cluster_mapping.csv"

DENSITY_OUTPUT_PATH = DENSITY_BASE_PATH / "train"

TIMS_FINAL_BASE_PATH = (
    PROJECT_ROOT_PATH
    / "gdrive"
    / "YOLOv10"
    / "data_train"
    / "TIMS_dataset_final"
    / "train_original"
)

TIMS_FINAL_IMAGES_PATH = TIMS_FINAL_BASE_PATH / "images"

TIMS_FINAL_LABELS_PATH = TIMS_FINAL_BASE_PATH / "labels"

LANE_SEG_WEIGHTS_PATH = (
    PROJECT_ROOT_PATH
    / "gdrive"
    / "YOLOv10"
    / "weights"
    / "pre-final"
    / "best_lane_seg_capstone"
    / "weights"
    / "best.pt"
)

CCTV_PATTERN = re.compile(
    r"^(?P<hexhash>[0-9a-fA-F]{8})-(?P<timestamp>\d{6})_100_(?P<tile>\d+)\.jpe?g$"
)

CCTV_PATTERN_LOOSE = re.compile(r"^[0-9a-fA-F]{8}-\d{6}_\d+_\d+")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

LANE_VEHICLE_CLASSES = {0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13}


# ──────────────────────────────────────────────────────────────────────
# Filename parsing & frame grouping
# ──────────────────────────────────────────────────────────────────────
def parse_filename(filename: str):
    """Return (hex_hash, timestamp, tile_num) or None if not CCTV pattern."""
    m = CCTV_PATTERN.match(filename)
    if not m:
        return None
    return m.group("hexhash"), m.group("timestamp"), int(m.group("tile"))


def group_tiles_by_frame(src_dir: Path) -> dict[str, list[dict]]:
    """
    Group all image files by their frame key (``{hex}_{timestamp}``).

    Using both the camera hash and the timestamp prevents frames from
    different cameras that happen to share the same capture time from
    being merged into a single group.

    Returns ``{hex_ts: [{"path": Path, "hex": str, "ts": str, "tile": int}, ...]}``
    """
    frames = defaultdict(list)
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        parsed = parse_filename(f.name)
        if parsed is None:
            continue
        hex_hash, ts, tile = parsed
        frame_key = f"{hex_hash}_{ts}"
        frames[frame_key].append({"path": f, "hex": hex_hash, "ts": ts, "tile": tile})
    for key in frames:
        frames[key].sort(key=lambda x: x["tile"])
    return dict(frames)


def pick_representative(tiles: list[dict]) -> dict:
    """Pick single median tile (for preview grid)."""
    return tiles[len(tiles) // 2]


def pick_representatives(tiles: list[dict], n: int = 3) -> list[dict]:
    """
    Pick n tiles evenly spread across the tile range for robust embeddings.
    Falls back to all tiles if fewer than n available.
    """
    if len(tiles) <= n:
        return tiles
    indices = np.linspace(0, len(tiles) - 1, n, dtype=int)
    return [tiles[i] for i in indices]


def time_period(ts: str, img_path: Optional[Path] = None) -> str:
    """
    Classify a frame into ``"day"`` / ``"night"`` / ``"IR"``.

    If *img_path* is provided the image's actual pixel saturation is used
    to distinguish RGB from IR (cameras switch based on ambient light, not
    the clock).  The hour is then only used to separate day from night
    among the RGB frames.

    Without *img_path* the old heuristic (hour-only) is used as fallback.
    """
    hour = int(ts[:2])

    if img_path is not None:
        modality = detect_modality(img_path)
        if modality == "IR":
            return "IR"
        return "day" if 6 <= hour < 18 else "night"

    if hour < 6 or hour >= 22:
        return "IR"
    if hour >= 18:
        return "night"
    return "day"


IR_SATURATION_THRESHOLD = 25


def detect_modality(img_path: Path) -> str:
    """
    Detect whether an image is RGB (color) or IR (grayscale / infrared)
    by measuring the mean saturation in HSV space.  More reliable than
    timestamp-based classification because cameras switch to IR based on
    ambient light, not clock time.
    """
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        return "unknown"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = hsv[:, :, 1].mean()
    return "IR" if mean_sat < IR_SATURATION_THRESHOLD else "RGB"


def detect_frame_modality(tiles: list[dict]) -> str:
    """Detect modality from the representative tile of a frame."""
    rep = pick_representative(tiles)
    return detect_modality(rep["path"])


# ──────────────────────────────────────────────────────────────────────
# Location discovery
# ──────────────────────────────────────────────────────────────────────
def discover_locations(base_dir: Optional[Path] = None) -> list[tuple[int, Path]]:
    """Find all location_* folders sorted by numeric id."""
    base_dir = base_dir or TRAIN_BY_LOCATION_PATH
    locs = []
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith("location_"):
            try:
                loc_id = int(d.name.split("_", 1)[1])
                locs.append((loc_id, d))
            except ValueError:
                continue
    locs.sort(key=lambda x: x[0])
    return locs


# ──────────────────────────────────────────────────────────────────────
# ROI config I/O
# ──────────────────────────────────────────────────────────────────────
def load_road_roi(config_path: Optional[Path] = None) -> dict[str, dict]:
    """
    Load road ROI config from JSON.

    Returns a dict mapping location name (e.g. ``"location_0"``) to a dict
    with keys:
      - ``"polygon"``: ``np.ndarray`` of shape (N, 2), dtype ``int32``
      - ``"image_size"``: ``[width, height]``
      - ``"num_lanes"``: ``int`` (number of lanes, 0 if not set)
      - ``"cars_per_lane"``: ``int`` (max cars per lane, 0 if not set)
    """
    config_path = config_path or ROI_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"ROI config not found: {config_path}")
    data = json.loads(config_path.read_text())
    roi_map: dict[str, dict] = {}
    for loc_name, entry in data.items():
        polygon = entry.get("polygon", [])
        if polygon:
            roi_map[loc_name] = {
                "polygon": np.array(polygon, dtype=np.int32),
                "image_size": entry.get("image_size", [0, 0]),
                "num_lanes": entry.get("num_lanes", 0),
                "cars_per_lane": entry.get("cars_per_lane", 0),
            }
    return roi_map


class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalars/arrays so roi_data round-trips cleanly."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_road_roi(
    roi_data: dict[str, dict], config_path: Optional[Path] = None
) -> Path:
    """
    Save road ROI config to JSON.

    *roi_data* maps location name → ``{"polygon": [[x,y], ...], "image_size": [w, h]}``.
    Returns the path written.
    """
    config_path = config_path or ROI_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(roi_data, indent=2, cls=_NumpyEncoder) + "\n")
    return config_path


# ──────────────────────────────────────────────────────────────────────
# YOLO label parsing (shared)
# ──────────────────────────────────────────────────────────────────────
CLASS_NAMES: dict[int, str] = {
    0: "full_truck",
    1: "full_trailer",
    2: "semi_trailer",
    3: "modified_car",
    4: "pedestrian",
    5: "bicycle",
    6: "motorcycle",
    7: "car",
    8: "car_7",
    9: "small_bus",
    10: "medium_bus",
    11: "large_bus",
    12: "pickup",
    13: "truck",
}


def parse_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Parse a YOLO label file into a list of detections.

    Each detection is ``(class_id, cx, cy, w, h)`` with **normalised**
    coordinates (0-1).  Returns an empty list for missing or empty files.
    """
    if not label_path.exists():
        return []
    content = label_path.read_text().strip()
    if not content:
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in content.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


# ──────────────────────────────────────────────────────────────────────
# ROI vehicle filtering (shared)
# ──────────────────────────────────────────────────────────────────────
def filter_vehicles_in_roi(
    boxes: list[tuple[int, float, float, float, float]],
    roi_polygon: np.ndarray,
    img_w: int,
    img_h: int,
) -> list[tuple[int, float, float, float, float]]:
    """
    Return only boxes whose centre falls inside *roi_polygon*.

    *boxes* use normalised coords; the polygon uses pixel coords.
    """
    import cv2

    if len(roi_polygon) < 3 or not boxes:
        return list(boxes)
    roi_contour = roi_polygon.reshape(-1, 1, 2).astype(np.float32)
    return [
        b
        for b in boxes
        if cv2.pointPolygonTest(
            roi_contour, (float(b[1] * img_w), float(b[2] * img_h)), False
        )
        >= 0
    ]
