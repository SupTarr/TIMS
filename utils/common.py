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

# ──────────────────────────────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = (
    Path(__file__).resolve().parent.parent
    / "gdrive"
    / "YOLOv10"
    / "data_train"
    / "TIMS_density_dataset"
)

TRAIN_BY_LOCATION = BASE_DIR / "raw" / "train_by_location"

ROI_CONFIG = TRAIN_BY_LOCATION / "road_roi.json"

CCTV_PATTERN = re.compile(
    r"^(?P<hexhash>[0-9a-fA-F]{8})-(?P<timestamp>\d{6})_100_(?P<tile>\d+)\.jpe?g$"
)

CCTV_PATTERN_LOOSE = re.compile(r"^[0-9a-fA-F]{8}-\d{6}_\d+_\d+")


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
    Group all image files by their frame key (timestamp).
    Returns {timestamp: [{"path": Path, "hex": str, "ts": str, "tile": int}, ...]}
    """
    frames = defaultdict(list)
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        parsed = parse_filename(f.name)
        if parsed is None:
            continue
        hex_hash, ts, tile = parsed
        frames[ts].append({"path": f, "hex": hex_hash, "ts": ts, "tile": tile})
    for ts in frames:
        frames[ts].sort(key=lambda x: x["tile"])
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


def time_period(ts: str) -> str:
    """Classify timestamp into day/night/IR label."""
    hour = int(ts[:2])
    if hour < 6 or hour >= 22:
        return "IR"
    if hour >= 18:
        return "night"
    return "day"


# ──────────────────────────────────────────────────────────────────────
# Location discovery
# ──────────────────────────────────────────────────────────────────────
def discover_locations(base_dir: Optional[Path] = None) -> list[tuple[int, Path]]:
    """Find all location_* folders sorted by numeric id."""
    base_dir = base_dir or TRAIN_BY_LOCATION
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
def load_road_roi(config_path: Optional[Path] = None) -> dict[str, np.ndarray]:
    """
    Load road ROI polygons from JSON config.

    Returns a dict mapping location name (e.g. ``"location_0"``) to an
    ``np.ndarray`` of shape (N, 2) with dtype ``int32``, matching the
    ``ROI_POLYGON`` convention used elsewhere in the project.
    """
    config_path = config_path or ROI_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"ROI config not found: {config_path}")
    data = json.loads(config_path.read_text())
    roi_map: dict[str, np.ndarray] = {}
    for loc_name, entry in data.items():
        polygon = entry.get("polygon", [])
        if polygon:
            roi_map[loc_name] = np.array(polygon, dtype=np.int32)
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
    config_path = config_path or ROI_CONFIG
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(roi_data, indent=2, cls=_NumpyEncoder) + "\n")
    return config_path
