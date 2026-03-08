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
import logging
import shutil
import sys

import numpy as np

from common import (
    DENSITY_OUTPUT_PATH as OUTPUT_DIR,
    IMAGE_EXTENSIONS,
    TRAIN_BY_LOCATION_PATH,
    CCTV_PATTERN,
    discover_locations,
    filter_vehicles_in_roi,
    load_road_roi,
    parse_yolo_labels,
    setup_logging,
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

    inside = filter_vehicles_in_roi(boxes, roi_polygon, img_w, img_h)
    total_weight = sum(CLASS_WEIGHTS.get(b[0], 0.0) for b in inside)

    return total_weight / capacity, total_weight


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
DENSITY_CLASSES = ("light", "medium", "high", "full")


def classify_density(
    dry_run: bool = False, verbose: bool = False, histogram: bool = False
) -> None:
    """Iterate over every location, classify each image, copy to output."""
    roi_map = load_road_roi()

    locations = discover_locations()
    if not locations:
        logger.error("No location folders found in %s", TRAIN_BY_LOCATION_PATH)
        sys.exit(1)

    logger.info("Found %d locations", len(locations))
    if not dry_run and not histogram:
        for cls in DENSITY_CLASSES:
            cls_dir = OUTPUT_DIR / cls
            if cls_dir.exists():
                shutil.rmtree(cls_dir)
            cls_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {cls: 0 for cls in DENSITY_CLASSES}
    location_stats: dict[str, dict[str, int]] = {}
    all_records: list[dict] = []

    for loc_id, loc_dir in locations:
        loc_name = f"location_{loc_id}"

        if loc_name not in roi_map:
            logger.warning("No ROI config for %s — skipping", loc_name)
            continue

        entry = roi_map[loc_name]
        roi_polygon = entry["polygon"]
        img_w, img_h = entry["image_size"]

        num_lanes = entry.get("num_lanes")
        cars_per_lane = entry.get("cars_per_lane")
        if not num_lanes or not cars_per_lane:
            logger.warning(
                "%s missing or has zero num_lanes/cars_per_lane — skipping", loc_name
            )
            continue

        images_dir = loc_dir / "images"
        labels_dir = loc_dir / "labels"

        if not images_dir.is_dir():
            logger.warning("No images/ dir in %s — skipping", loc_name)
            continue

        loc_counts: dict[str, int] = {cls: 0 for cls in DENSITY_CLASSES}

        image_files = sorted(
            f
            for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

        for img_path in image_files:
            if not CCTV_PATTERN.match(img_path.name):
                logger.warning("  Skipping non-CCTV file: %s", img_path.name)
                continue

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
    logger.info("")
    logger.info("=" * 60)
    logger.info("%sClassification Summary", prefix)
    logger.info("=" * 60)
    for cls in DENSITY_CLASSES:
        pct = (stats[cls] / total * 100) if total > 0 else 0
        logger.info("  %8s: %5d images (%5.1f%%)", cls, stats[cls], pct)
    logger.info("  %8s: %5d images", "TOTAL", total)
    logger.info("=" * 60)

    if not dry_run:
        logger.info("Images copied to: %s", OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────
# Histogram / distribution helpers
# ──────────────────────────────────────────────────────────────────────
def _print_histogram(records: list[dict], bin_width: float = 0.05) -> None:
    """Log an ASCII histogram of density ratios."""
    ratios = [r["density_ratio"] for r in records]
    if not ratios:
        logger.info("No data.")
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

    logger.info("")
    logger.info("=" * 70)
    logger.info("Density Ratio Histogram")
    logger.info("=" * 70)
    logger.info("  Total images: %d", len(ratios))
    logger.info(
        "  Min: %.3f  Max: %.3f  Mean: %.3f  Median: %.3f",
        min(ratios),
        max(ratios),
        sum(ratios) / len(ratios),
        sorted(ratios)[len(ratios) // 2],
    )
    logger.info("")

    for (lo, hi), cnt in zip(bins, counts):
        if cnt == 0 and lo > max_ratio + bin_width:
            continue
        bar_len = int(cnt / bar_max * bar_width) if bar_max > 0 else 0
        bar = "█" * bar_len
        pct_of_total = cnt / len(ratios) * 100 if ratios else 0
        logger.info("  [%5.2f-%5.2f) %5d (%5.1f%%) %s", lo, hi, cnt, pct_of_total, bar)

    logger.info("")
    logger.info("  Current thresholds: light<0.4 | medium<0.65 | high<0.9 | full≥0.9")
    logger.info("")

    sorted_ratios = sorted(ratios)
    for q in (10, 25, 50, 75, 90, 95, 99):
        idx = min(int(len(sorted_ratios) * q / 100), len(sorted_ratios) - 1)
        logger.info("  P%02d: %6.3f", q, sorted_ratios[idx])


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

    logger.info("")
    logger.info("=" * 70)
    logger.info("Distribution Description")
    logger.info("=" * 70)
    logger.info("")
    logger.info(
        "  The dataset contains %d images across %d locations.", n, len(loc_means)
    )
    logger.info(
        "  Density ratios range from %.3f to %.3f with a mean of %.3f "
        "(median %.3f, std %.3f).",
        sorted_ratios[0],
        sorted_ratios[-1],
        mean,
        median,
        std,
    )
    logger.info("  The distribution is %s with a %s spread.", skew_desc, spread_desc)
    logger.info(
        "  The interquartile range (Q1-Q3) is %.3f-%.3f (IQR=%.3f).", q1, q3, iqr
    )
    logger.info(
        "  The dominant class is '%s' with %d images (%.1f%% of total).",
        dominant_class,
        class_counts[dominant_class],
        dominant_pct,
    )
    logger.info("  Class breakdown:")
    for cls in DENSITY_CLASSES:
        cnt = class_counts.get(cls, 0)
        pct = cnt / n * 100
        logger.info("    %8s: %5d (%5.1f%%)", cls, cnt, pct)
    logger.info("  Busiest location:  %s (avg ratio %.3f)", busiest, loc_avg[busiest])
    logger.info("  Quietest location: %s (avg ratio %.3f)", quietest, loc_avg[quietest])
    logger.info("")


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
    logger.info("CSV exported to: %s", csv_path)


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
    setup_logging(args.verbose)

    classify_density(
        dry_run=args.dry_run, verbose=args.verbose, histogram=args.histogram
    )


if __name__ == "__main__":
    main()
