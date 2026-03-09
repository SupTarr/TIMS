#!/usr/bin/env python3
"""
Lane-segmentation refinement for road ROI polygons.

Loads a YOLO lane-segmentation model, runs inference on sample images,
and combines the resulting mask with the label-based polygon to produce
a refined ROI boundary.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from ..common import IMAGE_EXTENSIONS, parse_filename, time_period
from .roi_heatmap import APPROX_EPSILON_FRAC

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Lane-segmentation model
# ──────────────────────────────────────────────────────────────────────
def load_lane_seg_model(weights: Path):
    """Load YOLO lane segmentation model. Returns None on failure."""
    if not weights.exists():
        logger.warning(f"Lane-seg weights not found: {weights}")
        return None
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        logger.info(f"Loaded lane-seg model from {weights}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load lane-seg model: {e}")
        return None


def lane_seg_mask(model, images: list[Path], img_w: int, img_h: int) -> np.ndarray:
    """
    Run lane-seg inference on a set of images, union the masks,
    and return a binary (H, W) uint8 mask.
    """
    union_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for img_path in images:
        try:
            results = model(str(img_path), verbose=False)
            if results and results[0].masks is not None:
                for m in results[0].masks.data:
                    seg = m.cpu().numpy()
                    seg_resized = cv2.resize(seg.astype(np.float32), (img_w, img_h))
                    union_mask[seg_resized > 0.5] = 255
        except Exception as e:
            logger.debug(f"Lane-seg failed on {img_path.name}: {e}")
    return union_mask


def refine_polygon_with_lane_seg(
    model, loc_dir: Path, base_polygon: np.ndarray, img_w: int, img_h: int
) -> np.ndarray:
    """Combine label-based polygon with lane-seg mask."""
    images_dir = loc_dir / "images"
    if not images_dir.is_dir():
        images_dir = loc_dir
    all_imgs = sorted(
        f
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_imgs:
        return base_polygon

    day_imgs = []
    for p in all_imgs:
        parsed = parse_filename(p.name)
        if parsed and time_period(parsed[1], p) == "day":
            day_imgs.append(p)

    sample = (day_imgs or all_imgs)[:5]

    seg_mask = lane_seg_mask(model, sample, img_w, img_h)
    if seg_mask.max() == 0:
        logger.info(
            f"  {loc_dir.name}: lane-seg produced empty mask, using labels only"
        )
        return base_polygon

    combined = seg_mask.copy()
    if len(base_polygon) >= 3:
        cv2.fillPoly(combined, [base_polygon], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return base_polygon

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (img_w * img_h * 0.01):
        logger.info("  Mask is too small (likely noise), falling back to labels only")
        return base_polygon

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, APPROX_EPSILON_FRAC * peri, True)
    refined = approx.reshape(-1, 2).astype(np.int32)
    logger.info(f"  {loc_dir.name}: refined polygon has {len(refined)} vertices")
    return refined
