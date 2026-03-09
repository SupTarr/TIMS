#!/usr/bin/env python3
"""
Semi-automatic road ROI annotation for each camera location.

For every location_N folder, this script:
1. Auto-suggests a road ROI polygon from YOLO detection-label heatmaps
   (optionally refined with a lane-segmentation model).
2. Opens an interactive OpenCV window for manual adjustment.
3. Prompts for number of lanes and cars-per-lane (auto-suggested).
4. Saves all polygons + lane metadata to a single road_roi.json config.

Usage:
  python generate_road_roi.py                # annotate all locations
  python generate_road_roi.py --location 3   # re-annotate one location
  python generate_road_roi.py --no-auto      # skip auto-suggestion
  python generate_road_roi.py --no-lane-seg  # skip lane-seg refinement
  python generate_road_roi.py --preview-only # view existing ROIs (read-only)

Controls (interactive window):
  Left-click    Add vertex / drag existing vertex
  Right-click   Delete nearest vertex
  r             Reset to auto-suggested polygon
  c             Clear all vertices
  n / Enter     Accept and move to next location
  s             Skip location (no ROI saved)
  q             Save progress and quit
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .common import (
    LANE_SEG_WEIGHTS_PATH,
    ROI_CONFIG_PATH,
    discover_locations,
    save_road_roi,
    setup_logging,
)
from .utils.bev_transform import compute_bev_config
from .utils.lane_estimation import estimate_cars_per_lane, estimate_num_lanes
from .utils.roi_annotator import (
    ROIAnnotator,
    pick_annotation_image,
    preview_existing,
    prompt_positive_int,
)
from .utils.roi_heatmap import autosuggest_from_labels, load_label_data
from .utils.roi_lane_seg import load_lane_seg_model, refine_polygon_with_lane_seg

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Semi-automatic road ROI annotation per camera location"
    )
    parser.add_argument(
        "--location",
        type=int,
        default=None,
        help="Annotate only this location ID (e.g. --location 3)",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Skip auto-suggestion, start with blank polygon",
    )
    parser.add_argument(
        "--no-lane-seg", action="store_true", help="Skip lane-segmentation refinement"
    )
    parser.add_argument(
        "--preview-only", action="store_true", help="View existing ROIs without editing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output JSON path (default: {ROI_CONFIG_PATH})",
    )
    parser.add_argument(
        "--recompute-bev",
        action="store_true",
        help="Batch-recompute BEV config for all existing ROI entries (no GUI)",
    )
    args = parser.parse_args()

    config_path = Path(args.output) if args.output else ROI_CONFIG_PATH

    if args.preview_only:
        preview_existing(config_path)
        return

    if args.recompute_bev:
        _batch_recompute_bev(config_path)
        return

    locations = discover_locations()
    if args.location is not None:
        locations = [(lid, ld) for lid, ld in locations if lid == args.location]
        if not locations:
            logger.error("location_%d not found", args.location)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Road ROI Annotation Tool")
    logger.info("=" * 60)
    logger.info("Locations: %s", [f"location_{lid}" for lid, _ in locations])
    logger.info("Auto-suggest: %s", "OFF" if args.no_auto else "ON (labels heatmap)")
    logger.info("Lane-seg refinement: %s", "OFF" if args.no_lane_seg else "ON")
    logger.info("Output: %s", config_path)
    logger.info("")

    existing: dict[str, dict] = {}
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        logger.info("Loaded existing config with %d locations", len(existing))

    lane_model = None
    if not args.no_auto and not args.no_lane_seg:
        lane_model = load_lane_seg_model(LANE_SEG_WEIGHTS_PATH)

    for loc_id, loc_dir in locations:
        loc_name = f"location_{loc_id}"
        logger.info("─" * 40)
        logger.info("Processing %s", loc_name)
        logger.info("─" * 40)

        img_path = pick_annotation_image(loc_dir)
        if img_path is None:
            logger.warning("  No images found in %s, skipping", loc_name)
            continue

        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            logger.warning("  Failed to read %s, skipping", img_path.name)
            continue

        logger.info("  Image: %s (%dx%d)", img_path.name, img_w, img_h)

        if args.no_auto:
            suggested_poly = np.empty((0, 2), dtype=np.int32)
        else:
            suggested_poly = autosuggest_from_labels(loc_dir, img_w, img_h)
            if lane_model is not None:
                suggested_poly = refine_polygon_with_lane_seg(
                    lane_model, loc_dir, suggested_poly, img_w, img_h
                )

        if len(suggested_poly) > 0:
            logger.info("  Auto-suggested polygon: %d vertices", len(suggested_poly))
        else:
            logger.info("  No auto-suggestion available — draw manually")

        annotator = ROIAnnotator(cv_img, suggested_poly, title=loc_name)
        result = annotator.run()

        if annotator.quit:
            logger.info("Quitting early — saving progress...")
            break

        if result is None or len(result) < 3:
            logger.info("  %s: skipped (invalid polygon or user skipped)", loc_name)
            continue

        result_poly = np.array(result, dtype=np.int32)
        labels_dir = loc_dir / "labels"
        label_data = np.empty((0, 5))
        if labels_dir.exists() and not args.no_auto:
            label_data = load_label_data(labels_dir, img_w, img_h)

        auto_lanes = estimate_num_lanes(result_poly, label_data)
        auto_cpl = estimate_cars_per_lane(result_poly, label_data, auto_lanes)

        prev = existing.get(loc_name, {})
        default_lanes = prev.get("num_lanes", auto_lanes)
        default_cpl = prev.get("cars_per_lane", auto_cpl)

        logger.info("  Auto-estimated: %d lanes, %d cars/lane", auto_lanes, auto_cpl)
        num_lanes = prompt_positive_int(
            f"  Number of lanes [{default_lanes}]: ", default_lanes
        )
        cars_per_lane = prompt_positive_int(
            f"  Cars per lane [{default_cpl}]: ", default_cpl
        )

        max_cars = num_lanes * cars_per_lane
        logger.info(
            "  → %d lanes × %d cars/lane = %d max cars in ROI",
            num_lanes,
            cars_per_lane,
            max_cars,
        )

        entry_data: dict = {
            "polygon": result,
            "image_size": [img_w, img_h],
            "num_lanes": num_lanes,
            "cars_per_lane": cars_per_lane,
        }

        # Compute BEV homography config
        bev_cfg = compute_bev_config(result_poly, num_lanes)
        if bev_cfg:
            entry_data.update(bev_cfg)
            logger.info(
                "  BEV: road %.1f×%.1f m, %.3f m/px",
                bev_cfg["road_width_m"],
                bev_cfg["road_length_m"],
                bev_cfg["meters_per_pixel"],
            )

        existing[loc_name] = entry_data
        logger.info("  %s: saved %d vertices + lane metadata", loc_name, len(result))

    out = save_road_roi(existing, config_path)
    n_defined = sum(1 for v in existing.values() if v.get("polygon"))
    logger.info("Saved %d ROI polygon(s) to %s", n_defined, out)
    logger.info("Done!")


def _batch_recompute_bev(config_path: Path) -> None:
    """Recompute BEV config for all existing ROI entries without opening GUI."""
    if not config_path.exists():
        logger.error("ROI config not found: %s", config_path)
        sys.exit(1)

    data: dict = json.loads(config_path.read_text())
    updated = 0
    for loc_name, entry in data.items():
        polygon = entry.get("polygon", [])
        num_lanes = entry.get("num_lanes", 0)
        if len(polygon) < 4 or num_lanes < 1:
            logger.warning("%s: skipped (polygon=%d verts, lanes=%d)",
                           loc_name, len(polygon), num_lanes)
            continue
        poly_arr = np.array(polygon, dtype=np.float32)
        bev_cfg = compute_bev_config(poly_arr, num_lanes)
        if bev_cfg:
            entry.update(bev_cfg)
            updated += 1
            logger.info("%s: road %.1f×%.1f m, %.3f m/px",
                        loc_name,
                        bev_cfg["road_width_m"],
                        bev_cfg["road_length_m"],
                        bev_cfg["meters_per_pixel"])

    save_road_roi(data, config_path)
    logger.info("Recomputed BEV for %d / %d locations → %s",
                updated, len(data), config_path)


if __name__ == "__main__":
    main()
