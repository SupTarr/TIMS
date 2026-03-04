#!/usr/bin/env python3
"""
Classify images by traffic density using weighted vehicle counts.

For each location in raw/train_by_location/, counts detected vehicles
inside the road ROI, weights each by class (e.g. car=1, truck=2), and
computes:

    density_ratio = total_weight / (num_lanes x cars_per_lane)

Images are then copied into train/{light,medium,high,full}/ folders.

Density thresholds (on ratio):
  - light:  ratio < 0.4
  - medium: 0.4 ≤ ratio < 0.65
  - high:   0.65 ≤ ratio < 0.9
  - full:   ratio ≥ 0.9

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

from common import (
    DENSITY_OUTPUT_PATH as OUTPUT_DIR,
    ROI_CONFIG_PATH,
    TRAIN_BY_LOCATION_PATH,
    discover_locations,
)

logger = logging.getLogger(__name__)

CLASS_WEIGHTS: dict[int, float] = {
    0: 2.5,  # 10_full_truck
    1: 3.0,  # 11_full_trailer
    2: 3.0,  # 12_semi_trailer
    3: 1.0,  # 13_modified_car
    4: 0.0,  # 14_pedestrian  (excluded)
    5: 0.3,  # 1_bicycle
    6: 0.5,  # 2_motorcycle
    7: 1.0,  # 3_car
    8: 1.0,  # 4_car_7
    9: 1.5,  # 5_small_bus
    10: 2.0,  # 6_medium_bus
    11: 2.5,  # 7_large_bus
    12: 1.5,  # 8_pickup
    13: 2.0,  # 9_truck
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def classify_ratio(r: float) -> str:
    """Classify a density ratio into a density category."""
    if r < 0.4:
        return "light"
    elif r < 0.65:
        return "medium"
    elif r < 0.9:
        return "high"
    else:
        return "full"


def load_roi_config(config_path: Path = ROI_CONFIG_PATH) -> dict:
    """Load raw road_roi.json with polygon, image_size, num_lanes, cars_per_lane."""
    if not config_path.exists():
        raise FileNotFoundError(f"ROI config not found: {config_path}")
    return json.loads(config_path.read_text())


def parse_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Parse YOLO label file into list of (class_id, cx, cy, w, h).
    Returns empty list for empty / missing files.
    """
    if not label_path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            cx, cy, w, h = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
            boxes.append((cls_id, cx, cy, w, h))
    return boxes


def compute_density_ratio(
    boxes: list[tuple[int, float, float, float, float]],
    roi_polygon: np.ndarray,
    img_w: int,
    img_h: int,
    num_lanes: int,
    cars_per_lane: int,
) -> tuple[float, float]:
    """
    Compute density ratio = total_weight / (num_lanes x cars_per_lane).

    Only vehicles whose bbox centre falls inside the ROI polygon are counted.
    Each vehicle class has a weight factor defined in CLASS_WEIGHTS.

    Returns (density_ratio, total_weight).
    """
    capacity = num_lanes * cars_per_lane
    if capacity == 0 or len(boxes) == 0:
        return 0.0, 0.0

    roi_contour = roi_polygon.reshape(-1, 1, 2).astype(np.float32)

    total_weight = 0.0
    for cls_id, cx, cy, _w, _h in boxes:
        px, py = cx * img_w, cy * img_h
        total_weight += CLASS_WEIGHTS.get(cls_id, 0.0)

    return total_weight / capacity, total_weight


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
        logger.error("No location folders found in %s", TRAIN_BY_LOCATION_PATH)
        sys.exit(1)

    logger.info("Found %d locations", len(locations))
    if not dry_run and not histogram:
        for cls in DENSITY_CLASSES:
            (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {cls: 0 for cls in DENSITY_CLASSES}
    location_stats: dict[str, dict[str, int]] = {}
    all_records: list[dict] = []

    for loc_id, loc_dir in locations:
        loc_name = f"location_{loc_id}"

        if loc_name not in roi_config:
            logger.warning("No ROI config for %s — skipping", loc_name)
            continue

        entry = roi_config[loc_name]
        roi_polygon = np.array(entry["polygon"], dtype=np.int32)
        img_w, img_h = entry["image_size"]

        num_lanes = entry.get("num_lanes")
        cars_per_lane = entry.get("cars_per_lane")
        if num_lanes is None or cars_per_lane is None:
            logger.warning(
                "%s missing num_lanes/cars_per_lane in road_roi.json — skipping",
                loc_name,
            )
            continue

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
            ratio, total_w = compute_density_ratio(
                boxes, roi_polygon, img_w, img_h, num_lanes, cars_per_lane
            )
            density = classify_ratio(ratio)

            all_records.append(
                {
                    "location": loc_name,
                    "image": img_path.name,
                    "num_boxes": len(boxes),
                    "total_weight": round(total_w, 2),
                    "density_ratio": round(ratio, 4),
                    "class": density,
                }
            )

            if verbose:
                logger.info(
                    "  %s: %d boxes, weight=%.1f, ratio=%.3f → %s",
                    img_path.name,
                    len(boxes),
                    total_w,
                    ratio,
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
        _print_histogram(all_records)
        _describe_distribution(all_records)
        _export_csv(all_records)
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
def _print_histogram(records: list[dict], bin_width: float = 0.05) -> None:
    """Print an ASCII histogram of density ratios."""
    ratios = [r["density_ratio"] for r in records]
    if not ratios:
        print("No data.")
        return

    max_ratio = max(ratios)
    bins: list[tuple[float, float]] = []
    lo = 0.0
    while lo <= max(max_ratio, 1.5):
        bins.append((lo, round(lo + bin_width, 4)))
        lo = round(lo + bin_width, 4)

    counts = [0] * len(bins)
    for r in ratios:
        idx = min(int(r / bin_width), len(counts) - 1)
        counts[idx] += 1

    bar_max = max(counts) if counts else 1
    bar_width = 50

    print(f"\n{'=' * 70}")
    print("Density Ratio Histogram")
    print(f"{'=' * 70}")
    print(f"  Total images: {len(ratios)}")
    print(
        f"  Min: {min(ratios):.3f}  Max: {max(ratios):.3f}  "
        f"Mean: {sum(ratios)/len(ratios):.3f}  "
        f"Median: {sorted(ratios)[len(ratios)//2]:.3f}"
    )
    print()

    for (lo, hi), cnt in zip(bins, counts):
        if cnt == 0 and lo > max_ratio + bin_width:
            continue
        bar_len = int(cnt / bar_max * bar_width) if bar_max > 0 else 0
        bar = "█" * bar_len
        pct_of_total = cnt / len(ratios) * 100 if ratios else 0
        print(f"  [{lo:5.2f}-{hi:5.2f}) {cnt:5d} ({pct_of_total:5.1f}%) {bar}")

    print(f"\n  Current thresholds: " f"light<0.4 | medium<0.65 | high<0.9 | full≥0.9")
    print()

    sorted_ratios = sorted(ratios)
    for q in (10, 25, 50, 75, 90, 95, 99):
        idx = min(int(len(sorted_ratios) * q / 100), len(sorted_ratios) - 1)
        print(f"  P{q:02d}: {sorted_ratios[idx]:6.3f}")


def _describe_distribution(records: list[dict]) -> None:
    """Print a human-readable narrative description of the density distribution."""
    ratios = [r["density_ratio"] for r in records]
    if not ratios:
        return

    n = len(ratios)
    mean = sum(ratios) / n
    sorted_ratios = sorted(ratios)
    median = sorted_ratios[n // 2]
    std = (sum((x - mean) ** 2 for x in ratios) / n) ** 0.5
    q1 = sorted_ratios[n // 4]
    q3 = sorted_ratios[3 * n // 4]
    iqr = q3 - q1

    class_counts: dict[str, int] = {}
    for r in records:
        cls = r["class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    dominant_class = max(class_counts, key=class_counts.get)
    dominant_pct = class_counts[dominant_class] / n * 100

    loc_means: dict[str, list[float]] = {}
    for r in records:
        loc_means.setdefault(r["location"], []).append(r["density_ratio"])
    loc_avg = {k: sum(v) / len(v) for k, v in loc_means.items()}
    busiest = max(loc_avg, key=loc_avg.get)
    quietest = min(loc_avg, key=loc_avg.get)

    if mean > median * 1.15:
        skew_desc = "right-skewed (long tail towards high density)"
    elif median > mean * 1.15:
        skew_desc = "left-skewed (long tail towards low density)"
    else:
        skew_desc = "approximately symmetric"

    if std < 0.05:
        spread_desc = "very tight"
    elif std < 0.15:
        spread_desc = "moderate"
    elif std < 0.30:
        spread_desc = "wide"
    else:
        spread_desc = "very wide"

    print(f"\n{'=' * 70}")
    print("Distribution Description")
    print(f"{'=' * 70}")
    print(f"\n  The dataset contains {n} images across {len(loc_means)} locations.")
    print(
        f"  Density ratios range from {sorted_ratios[0]:.3f} to "
        f"{sorted_ratios[-1]:.3f} with a mean of {mean:.3f} "
        f"(median {median:.3f}, std {std:.3f})."
    )
    print(f"  The distribution is {skew_desc} with a {spread_desc} spread.")
    print(f"  The interquartile range (Q1-Q3) is {q1:.3f}-{q3:.3f} (IQR={iqr:.3f}).")
    print(
        f"\n  The dominant class is '{dominant_class}' with "
        f"{class_counts[dominant_class]} images ({dominant_pct:.1f}% of total)."
    )
    print(f"  Class breakdown:")
    for cls in DENSITY_CLASSES:
        cnt = class_counts.get(cls, 0)
        pct = cnt / n * 100
        print(f"    {cls:>8s}: {cnt:5d} ({pct:5.1f}%)")
    print(f"\n  Busiest location:  {busiest} (avg ratio {loc_avg[busiest]:.3f})")
    print(f"  Quietest location: {quietest} (avg ratio {loc_avg[quietest]:.3f})")
    print()


def _export_csv(records: list[dict]) -> None:
    """Write all per-image density data to a CSV alongside the output dir."""
    csv_path = OUTPUT_DIR / "density_distribution.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "location",
                "image",
                "num_boxes",
                "total_weight",
                "density_ratio",
                "class",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV exported to: {csv_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify images by traffic density (weighted vehicle count / lane capacity)."
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
