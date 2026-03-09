#!/usr/bin/env python3
"""
Filename parsing and YOLO label parsing utilities.
"""

from pathlib import Path

from .paths import CCTV_PATTERN

# ──────────────────────────────────────────────────────────────────────
# Class name mapping
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


# ──────────────────────────────────────────────────────────────────────
# Filename parsing
# ──────────────────────────────────────────────────────────────────────
def parse_filename(filename: str):
    """Return (hex_hash, timestamp, tile_num) or None if not CCTV pattern."""
    m = CCTV_PATTERN.match(filename)
    if not m:
        return None
    return m.group("hexhash"), m.group("timestamp"), int(m.group("tile"))


# ──────────────────────────────────────────────────────────────────────
# YOLO label parsing
# ──────────────────────────────────────────────────────────────────────
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
