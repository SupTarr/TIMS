#!/usr/bin/env python3
"""
Shared utilities for CCTV tile image processing.

Provides common filename parsing, frame grouping, and tile selection
used by cluster_by_location.py, generate_cluster_preview.py, and classifier.py.
"""

import re
from collections import defaultdict
from pathlib import Path

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
