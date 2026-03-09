#!/usr/bin/env python3
"""
Label-based heatmap auto-suggestion for road ROI polygons.

Reads YOLO detection labels, builds a 2-D KDE density map, and extracts
the largest contour as a simplified polygon suitable for road ROI annotation.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from sklearn.neighbors import KernelDensity

from ..common import parse_yolo_labels

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

KDE_BANDWIDTH = 80
HEATMAP_THRESHOLD = 0.25
APPROX_EPSILON_FRAC = 0.02


# ──────────────────────────────────────────────────────────────────────
# Label data loading
# ──────────────────────────────────────────────────────────────────────
def load_label_data(labels_dir: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Read YOLO label files and return full detection data in pixel coords.
    Returns an (N, 5) array: [class_id, cx_px, cy_px, w_px, h_px].
    """
    rows: list[list[float]] = []
    for txt in labels_dir.glob("*.txt"):
        boxes = parse_yolo_labels(txt)
        for cls_id, cx_norm, cy_norm, w_norm, h_norm in boxes:
            rows.append(
                [
                    float(cls_id),
                    cx_norm * img_w,
                    cy_norm * img_h,
                    w_norm * img_w,
                    h_norm * img_h,
                ]
            )
    if not rows:
        return np.empty((0, 5))
    return np.array(rows)


def load_label_centroids(labels_dir: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Read YOLO label files and convert normalised bbox centres to pixel coords.
    Returns an (N, 2) array of [x, y] pixel positions.
    """
    data = load_label_data(labels_dir, img_w, img_h)
    if len(data) == 0:
        return np.empty((0, 2))
    return data[:, 1:3]


# ──────────────────────────────────────────────────────────────────────
# Heatmap → polygon
# ──────────────────────────────────────────────────────────────────────
def build_heatmap(
    centroids: np.ndarray, img_w: int, img_h: int, bandwidth: float = KDE_BANDWIDTH
) -> np.ndarray:
    """Build a 2-D KDE density map from pixel centroids. Returns a (H, W) float array."""
    if len(centroids) < 3:
        return np.zeros((img_h, img_w), dtype=np.float32)

    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(centroids)

    step = 8
    xs = np.arange(0, img_w, step)
    ys = np.arange(0, img_h, step)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    log_dens = kde.score_samples(grid)
    dens = np.exp(log_dens).reshape(len(ys), len(xs)).astype(np.float32)
    heatmap = cv2.resize(dens, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    return heatmap


def heatmap_to_polygon(
    heatmap: np.ndarray, threshold_frac: float = HEATMAP_THRESHOLD
) -> np.ndarray:
    """Threshold heatmap and extract the largest contour as a simplified polygon."""
    if heatmap.max() == 0:
        return np.empty((0, 2), dtype=np.int32)
    thresh_val = heatmap.max() * threshold_frac
    mask = (heatmap >= thresh_val).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.int32)

    largest = max(contours, key=cv2.contourArea)
    img_h, img_w = heatmap.shape
    if cv2.contourArea(largest) < (img_w * img_h * 0.01):
        return np.empty((0, 2), dtype=np.int32)

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, APPROX_EPSILON_FRAC * peri, True)
    return approx.reshape(-1, 2).astype(np.int32)


# ──────────────────────────────────────────────────────────────────────
# Full auto-suggestion pipeline
# ──────────────────────────────────────────────────────────────────────
def autosuggest_from_labels(loc_dir: Path, img_w: int, img_h: int) -> np.ndarray:
    """Full auto-suggestion pipeline: labels → heatmap → polygon."""
    labels_dir = loc_dir / "labels"
    if not labels_dir.exists():
        logger.warning(f"No labels dir in {loc_dir.name}")
        return np.empty((0, 2), dtype=np.int32)

    centroids = load_label_centroids(labels_dir, img_w, img_h)
    logger.info(f"  {loc_dir.name}: {len(centroids)} detection centroids")

    heatmap = build_heatmap(centroids, img_w, img_h)
    polygon = heatmap_to_polygon(heatmap)
    logger.info(f"  {loc_dir.name}: auto-polygon has {len(polygon)} vertices")
    return polygon
