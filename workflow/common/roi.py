#!/usr/bin/env python3
"""
Road ROI config I/O, location discovery, and vehicle filtering.
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .paths import ROI_CONFIG_PATH, TRAIN_BY_LOCATION_PATH


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
# ROI vehicle filtering
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
    if len(roi_polygon) < 3 or not boxes:
        return []
    roi_contour = roi_polygon.reshape(-1, 1, 2).astype(np.float32)
    return [
        b
        for b in boxes
        if cv2.pointPolygonTest(
            roi_contour, (float(b[1] * img_w), float(b[2] * img_h)), False
        )
        >= 0
    ]
