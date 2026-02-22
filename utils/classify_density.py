#!/usr/bin/env python3
"""
Classify images by traffic density based on vehicle-ROI intersection.

For each location in raw/train_by_location/, computes the percentage P of
the road ROI area covered by detected vehicle bounding boxes, then copies
images into train/{light,medium,high,full}/ folders.

Density thresholds:
  - light:  P < 40%
  - medium: 40% ≤ P < 65%
  - high:   65% ≤ P < 90%
  - full:   P ≥ 90%

Usage:
    python classify_density.py                # run classification
    python classify_density.py --dry-run      # stats only, no copy
    python classify_density.py --verbose       # per-image logging
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from common import BASE_DIR, ROI_CONFIG, TRAIN_BY_LOCATION, discover_locations

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / "train"

# Max dimension for downscaled mask computation (saves memory + time)
MAX_MASK_DIM = 640


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def classify_percentage(p: float) -> str:
    """Classify a percentage into a density category."""
    if p < 40:
        return "light"
    elif p < 65:
        return "medium"
    elif p < 90:
        return "high"
    else:
        return "full"


def load_roi_config(config_path: Path = ROI_CONFIG) -> dict:
    """Load raw road_roi.json with polygon and image_size."""
    if not config_path.exists():
        raise FileNotFoundError(f"ROI config not found: {config_path}")
    return json.loads(config_path.read_text())


def parse_yolo_labels(label_path: Path) -> list[tuple[float, float, float, float]]:
    """
    Parse YOLO label file into list of (cx, cy, w, h) normalised coords.
    Returns empty list for empty / missing files.
    """
    if not label_path.exists():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cx, cy, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            boxes.append((cx, cy, w, h))
    return boxes


def compute_density_percentage(
    boxes: list[tuple[float, float, float, float]],
    roi_polygon: np.ndarray,
    img_w: int,
    img_h: int,
) -> float:
    """
    Compute the percentage of ROI area covered by vehicle bounding boxes.

    Uses downscaled mask rasterisation for efficiency.
    Overlapping bboxes are counted only once (union within ROI).

    Returns P in [0, 100].
    """
    if len(boxes) == 0:
        return 0.0

    # Compute scale factor for downscaled masks
    scale = min(MAX_MASK_DIM / max(img_w, img_h), 1.0)
    sw, sh = int(img_w * scale), int(img_h * scale)

    # Scale ROI polygon
    scaled_roi = (roi_polygon * scale).astype(np.int32)

    # Create ROI mask
    roi_mask = np.zeros((sh, sw), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [scaled_roi], 1)
    roi_area = np.count_nonzero(roi_mask)

    if roi_area == 0:
        return 0.0

    # Create bbox union mask
    bbox_mask = np.zeros((sh, sw), dtype=np.uint8)
    for cx, cy, w, h in boxes:
        # Convert normalised YOLO to pixel coords (scaled)
        x1 = int((cx - w / 2) * img_w * scale)
        y1 = int((cy - h / 2) * img_h * scale)
        x2 = int((cx + w / 2) * img_w * scale)
        y2 = int((cy + h / 2) * img_h * scale)
        # Clamp to mask bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(sw, x2), min(sh, y2)
        if x2 > x1 and y2 > y1:
            bbox_mask[y1:y2, x1:x2] = 1

    # Intersection: pixels that are both in ROI and in any bbox
    intersection = np.count_nonzero(roi_mask & bbox_mask)

    return (intersection / roi_area) * 100.0


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
DENSITY_CLASSES = ("light", "medium", "high", "full")


def classify_density(dry_run: bool = False, verbose: bool = False) -> None:
    """Iterate over every location, classify each image, copy to output."""
    roi_config = load_roi_config()

    locations = discover_locations()
    if not locations:
        logger.error("No location folders found in %s", TRAIN_BY_LOCATION)
        sys.exit(1)

    logger.info("Found %d locations", len(locations))

    # Prepare output directories
    if not dry_run:
        for cls in DENSITY_CLASSES:
            (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    # Accumulate statistics
    stats: dict[str, int] = {cls: 0 for cls in DENSITY_CLASSES}
    location_stats: dict[str, dict[str, int]] = {}

    for loc_id, loc_dir in locations:
        loc_name = f"location_{loc_id}"

        if loc_name not in roi_config:
            logger.warning("No ROI config for %s — skipping", loc_name)
            continue

        entry = roi_config[loc_name]
        roi_polygon = np.array(entry["polygon"], dtype=np.int32)
        img_w, img_h = entry["image_size"]

        images_dir = loc_dir / "images"
        labels_dir = loc_dir / "labels"

        if not images_dir.is_dir():
            logger.warning("No images/ dir in %s — skipping", loc_name)
            continue

        loc_counts: dict[str, int] = {cls: 0 for cls in DENSITY_CLASSES}

        image_files = sorted(images_dir.glob("*.jpg"))

        for img_path in image_files:
            label_path = labels_dir / (img_path.stem + ".txt")
            boxes = parse_yolo_labels(label_path)
            p = compute_density_percentage(boxes, roi_polygon, img_w, img_h)
            density = classify_percentage(p)

            if verbose:
                logger.info(
                    "  %s: %d boxes, P=%.1f%% → %s",
                    img_path.name,
                    len(boxes),
                    p,
                    density,
                )

            if not dry_run:
                dst = OUTPUT_DIR / density / img_path.name
                shutil.copy2(str(img_path), str(dst))

            stats[density] += 1
            loc_counts[density] += 1

        location_stats[loc_name] = loc_counts
        total_loc = sum(loc_counts.values())
        logger.info(
            "%s: %d images — light=%d  medium=%d  high=%d  full=%d",
            loc_name,
            total_loc,
            loc_counts["light"],
            loc_counts["medium"],
            loc_counts["high"],
            loc_counts["full"],
        )

    # ── Summary ──────────────────────────────────────────────────────
    total = sum(stats.values())
    prefix = "DRY RUN — " if dry_run else ""
    print(f"\n{'=' * 60}")
    print(f"{prefix}Classification Summary")
    print(f"{'=' * 60}")
    for cls in DENSITY_CLASSES:
        pct = (stats[cls] / total * 100) if total > 0 else 0
        print(f"  {cls:>8s}: {stats[cls]:5d} images ({pct:5.1f}%)")
    print(f"  {'TOTAL':>8s}: {total:5d} images")
    print(f"{'=' * 60}")

    if not dry_run:
        print(f"\nImages copied to: {OUTPUT_DIR}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify images by traffic density (vehicle-ROI coverage)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print classification stats without copying files.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log per-image density classification.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    classify_density(dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
