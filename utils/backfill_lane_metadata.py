#!/usr/bin/env python3
"""
Backfill num_lanes and cars_per_lane into an existing road_roi.json.

Reads each location's polygon + labels, reuses the KDE-based estimation
from generate_road_roi.py, and writes the enriched config back in place.

Usage:
    python backfill_lane_metadata.py                  # estimate + write all
    python backfill_lane_metadata.py --dry-run         # print only, no write
    python backfill_lane_metadata.py --location 3      # single location
    python backfill_lane_metadata.py --manual-override  # prompt to accept/edit
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from common import (
    ROI_CONFIG_PATH,
    TRAIN_BY_LOCATION_PATH,
    discover_locations,
    load_road_roi,
    save_road_roi,
    setup_logging,
)
from generate_road_roi import (
    estimate_cars_per_lane,
    estimate_num_lanes,
    load_label_data,
    prompt_positive_int,
)

logger = logging.getLogger(__name__)


def backfill(
    roi_path: Path = ROI_CONFIG_PATH,
    location_id: int | None = None,
    dry_run: bool = False,
    manual_override: bool = False,
) -> None:
    """Estimate and write num_lanes / cars_per_lane for existing ROI entries."""

    roi_map = load_road_roi(roi_path)
    locations = discover_locations(TRAIN_BY_LOCATION_PATH)

    if location_id is not None:
        locations = [(lid, p) for lid, p in locations if lid == location_id]
        if not locations:
            logger.error("location_%d not found", location_id)
            sys.exit(1)

    results: list[dict] = []

    for loc_id, loc_dir in locations:
        loc_name = f"location_{loc_id}"
        entry = roi_map.get(loc_name)
        if entry is None:
            logger.warning("%s has no ROI entry — skipping", loc_name)
            continue

        polygon: np.ndarray = entry["polygon"]
        img_w, img_h = entry["image_size"]

        labels_dir = loc_dir / "labels"
        if not labels_dir.exists() or not any(labels_dir.glob("*.txt")):
            logger.warning("%s has no labels — skipping", loc_name)
            continue

        label_data = load_label_data(labels_dir, img_w, img_h)
        logger.info(
            "%s: %d detections, polygon %d vertices, image %dx%d",
            loc_name,
            len(label_data),
            len(polygon),
            img_w,
            img_h,
        )

        num_lanes = estimate_num_lanes(polygon, label_data)
        cars_per_lane = estimate_cars_per_lane(polygon, label_data, num_lanes)

        if manual_override:
            logger.info("")
            logger.info("--- %s ---", loc_name)
            logger.info(
                "  Auto-estimated:  num_lanes=%d  cars_per_lane=%d",
                num_lanes,
                cars_per_lane,
            )
            num_lanes = prompt_positive_int(
                f"  num_lanes [{num_lanes}]: ", default=num_lanes
            )
            cars_per_lane = prompt_positive_int(
                f"  cars_per_lane [{cars_per_lane}]: ", default=cars_per_lane
            )

        entry["num_lanes"] = int(num_lanes)
        entry["cars_per_lane"] = int(cars_per_lane)

        results.append(
            {
                "location": loc_name,
                "num_lanes": num_lanes,
                "cars_per_lane": cars_per_lane,
                "detections": len(label_data),
            }
        )

    if results:
        hdr = f"{'Location':<14} {'Lanes':>5} {'Cars/Lane':>9} {'Detections':>10}"
        sep = "-" * len(hdr)
        logger.info("")
        logger.info(sep)
        logger.info(hdr)
        logger.info(sep)
        for r in results:
            logger.info(
                "%-14s %5d %9d %10d",
                r["location"],
                r["num_lanes"],
                r["cars_per_lane"],
                r["detections"],
            )
        logger.info(sep)
    else:
        logger.warning("No locations processed.")
        return

    if dry_run:
        logger.info("[dry-run] No changes written.")
    else:
        save_road_roi(roi_map, roi_path)
        logger.info("Updated %s (%d locations)", roi_path, len(results))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Backfill num_lanes / cars_per_lane into road_roi.json"
    )
    parser.add_argument(
        "--location",
        type=int,
        default=None,
        help="Re-estimate a single location (e.g. --location 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print estimates without modifying road_roi.json",
    )
    parser.add_argument(
        "--manual-override",
        action="store_true",
        help="Prompt to accept or edit each estimated value",
    )
    args = parser.parse_args()
    backfill(
        location_id=args.location,
        dry_run=args.dry_run,
        manual_override=args.manual_override,
    )


if __name__ == "__main__":
    main()
