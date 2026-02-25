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
    python classify_density.py --histogram     # density distribution analysis
"""

import argparse
import csv
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

    scale = min(MAX_MASK_DIM / max(img_w, img_h), 1.0)
    sw, sh = int(img_w * scale), int(img_h * scale)

    scaled_roi = (roi_polygon * scale).astype(np.int32)
    roi_mask = np.zeros((sh, sw), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [scaled_roi], 1)
    roi_area = np.count_nonzero(roi_mask)

    if roi_area == 0:
        return 0.0

    bbox_mask = np.zeros((sh, sw), dtype=np.uint8)
    for cx, cy, w, h in boxes:
        x1 = int((cx - w / 2) * img_w * scale)
        y1 = int((cy - h / 2) * img_h * scale)
        x2 = int((cx + w / 2) * img_w * scale)
        y2 = int((cy + h / 2) * img_h * scale)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(sw, x2), min(sh, y2)
        if x2 > x1 and y2 > y1:
            bbox_mask[y1:y2, x1:x2] = 1

    intersection = np.count_nonzero(roi_mask & bbox_mask)
    return (intersection / roi_area) * 100.0


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
DENSITY_CLASSES = ("light", "medium", "high", "full")


def classify_density(
    dry_run: bool = False, verbose: bool = False, histogram: bool = False
) -> None:
    """Iterate over every location, classify each image, copy to output."""
    roi_config = load_roi_config()

    locations = discover_locations()
    if not locations:
        logger.error("No location folders found in %s", TRAIN_BY_LOCATION)
        sys.exit(1)

    logger.info("Found %d locations", len(locations))
    if not dry_run and not histogram:
        for cls in DENSITY_CLASSES:
            (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {cls: 0 for cls in DENSITY_CLASSES}
    location_stats: dict[str, dict[str, int]] = {}
    all_percentages: list[dict] = []

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

            all_percentages.append(
                {
                    "location": loc_name,
                    "image": img_path.name,
                    "num_boxes": len(boxes),
                    "density_pct": round(p, 2),
                    "class": density,
                }
            )

            if verbose:
                logger.info(
                    "  %s: %d boxes, P=%.1f%% → %s",
                    img_path.name,
                    len(boxes),
                    p,
                    density,
                )

            if not dry_run and not histogram:
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

    if histogram:
        _print_histogram(all_percentages)
        _describe_distribution(all_percentages)
        _export_csv(all_percentages)
        return

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
# Histogram / distribution helpers
# ──────────────────────────────────────────────────────────────────────
def _print_histogram(records: list[dict], bin_width: int = 5) -> None:
    """Print an ASCII histogram of density percentages."""
    pcts = [r["density_pct"] for r in records]
    if not pcts:
        print("No data.")
        return

    max_pct = max(pcts)
    bins: list[tuple[float, float]] = []
    lo = 0.0
    while lo <= max(max_pct, 100):
        bins.append((lo, lo + bin_width))
        lo += bin_width

    counts = [0] * len(bins)
    for p in pcts:
        idx = min(int(p // bin_width), len(counts) - 1)
        counts[idx] += 1

    bar_max = max(counts) if counts else 1
    bar_width = 50

    print(f"\n{'=' * 70}")
    print("Density Percentage Histogram")
    print(f"{'=' * 70}")
    print(f"  Total images: {len(pcts)}")
    print(
        f"  Min: {min(pcts):.1f}%  Max: {max(pcts):.1f}%  "
        f"Mean: {sum(pcts)/len(pcts):.1f}%  "
        f"Median: {sorted(pcts)[len(pcts)//2]:.1f}%"
    )
    print()

    for (lo, hi), cnt in zip(bins, counts):
        if cnt == 0 and lo > max_pct + bin_width:
            continue
        bar_len = int(cnt / bar_max * bar_width) if bar_max > 0 else 0
        bar = "█" * bar_len
        pct_of_total = cnt / len(pcts) * 100 if pcts else 0
        print(f"  [{lo:5.0f}-{hi:5.0f}%) {cnt:5d} ({pct_of_total:5.1f}%) {bar}")

    print(
        f"\n  Current thresholds: "
        f"light<40% | medium<65% | high<90% | full≥90%"
    )
    print()

    sorted_pcts = sorted(pcts)
    for q in (10, 25, 50, 75, 90, 95, 99):
        idx = min(int(len(sorted_pcts) * q / 100), len(sorted_pcts) - 1)
        print(f"  P{q:02d}: {sorted_pcts[idx]:6.1f}%")


def _describe_distribution(records: list[dict]) -> None:
    """Print a human-readable narrative description of the density distribution."""
    pcts = [r["density_pct"] for r in records]
    if not pcts:
        return

    n = len(pcts)
    mean = sum(pcts) / n
    sorted_pcts = sorted(pcts)
    median = sorted_pcts[n // 2]
    std = (sum((x - mean) ** 2 for x in pcts) / n) ** 0.5
    q1 = sorted_pcts[n // 4]
    q3 = sorted_pcts[3 * n // 4]
    iqr = q3 - q1

    # Class breakdown
    class_counts: dict[str, int] = {}
    for r in records:
        cls = r["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    dominant_class = max(class_counts, key=class_counts.get)  # type: ignore[arg-type]
    dominant_pct = class_counts[dominant_class] / n * 100

    # Per-location variation
    loc_means: dict[str, list[float]] = {}
    for r in records:
        loc_means.setdefault(r["location"], []).append(r["density_pct"])
    loc_avg = {k: sum(v) / len(v) for k, v in loc_means.items()}
    busiest = max(loc_avg, key=loc_avg.get)  # type: ignore[arg-type]
    quietest = min(loc_avg, key=loc_avg.get)  # type: ignore[arg-type]

    # Skewness description
    if mean > median * 1.15:
        skew_desc = "right-skewed (long tail towards high density)"
    elif median > mean * 1.15:
        skew_desc = "left-skewed (long tail towards low density)"
    else:
        skew_desc = "approximately symmetric"

    # Spread description
    if std < 5:
        spread_desc = "very tight"
    elif std < 15:
        spread_desc = "moderate"
    elif std < 30:
        spread_desc = "wide"
    else:
        spread_desc = "very wide"

    print(f"\n{'=' * 70}")
    print("Distribution Description")
    print(f"{'=' * 70}")
    print(f"\n  The dataset contains {n} images across {len(loc_means)} locations.")
    print(
        f"  Density percentages range from {sorted_pcts[0]:.1f}% to "
        f"{sorted_pcts[-1]:.1f}% with a mean of {mean:.1f}% "
        f"(median {median:.1f}%, std {std:.1f}%)."
    )
    print(f"  The distribution is {skew_desc} with a {spread_desc} spread.")
    print(f"  The interquartile range (Q1–Q3) is {q1:.1f}%–{q3:.1f}% (IQR={iqr:.1f}%).")
    print(
        f"\n  The dominant class is '{dominant_class}' with "
        f"{class_counts[dominant_class]} images ({dominant_pct:.1f}% of total)."
    )
    print(f"  Class breakdown:")
    for cls in DENSITY_CLASSES:
        cnt = class_counts.get(cls, 0)
        pct = cnt / n * 100
        print(f"    {cls:>8s}: {cnt:5d} ({pct:5.1f}%)")
    print(
        f"\n  Busiest location:  {busiest} (avg density {loc_avg[busiest]:.1f}%)"
    )
    print(
        f"  Quietest location: {quietest} (avg density {loc_avg[quietest]:.1f}%)"
    )
    print()


def _export_csv(records: list[dict]) -> None:
    """Write all per-image density data to a CSV alongside the output dir."""
    csv_path = OUTPUT_DIR / "density_distribution.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["location", "image", "num_boxes", "density_pct", "class"]
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV exported to: {csv_path}")


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
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Print density distribution histogram, description and export CSV (no copy).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    classify_density(
        dry_run=args.dry_run, verbose=args.verbose, histogram=args.histogram
    )


if __name__ == "__main__":
    main()
