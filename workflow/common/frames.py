#!/usr/bin/env python3
"""
Frame grouping, tile selection, and modality detection utilities.
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .parsing import parse_filename

# ──────────────────────────────────────────────────────────────────────
# Frame grouping
# ──────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────
# Tile selection
# ──────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────
# Time period / modality detection
# ──────────────────────────────────────────────────────────────────────

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


def time_period(ts: str, img_path: Optional[Path] = None) -> str:
    """
    Classify a frame into ``"day"`` / ``"night"`` / ``"IR"``.

    If *img_path* is provided the image's actual pixel saturation is used
    to distinguish RGB from IR (cameras switch based on ambient light, not
    the clock). The hour is then only used to separate day from night
    among the RGB frames.

    Without *img_path* the old heuristic (hour-only) is used as fallback.
    """
    hour = int(ts[:2])

    if img_path is not None:
        modality = detect_modality(img_path)
        if modality == "IR":
            return "IR"
        return "day" if 6 <= hour < 18 else "night"

    return "day" if 6 <= hour < 18 else "night"
