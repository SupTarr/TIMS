#!/usr/bin/env python3
"""
Illumination-invariant structural feature extraction for CCTV images.

Extracts edge orientation histograms, spatial edge density, and resolution
fingerprints — features that capture scene geometry and layout consistently
across day, night, and IR imagery for the same camera location.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from ..common import pick_representatives

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

EDGE_HIST_BINS = 36


# ──────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────
def extract_structural_features(img_path: Path) -> np.ndarray:
    """
    Extract illumination-invariant structural features from an image:
    1. Edge orientation histogram (Sobel gradients -> orientation bins)
    2. Spatial edge density (4x4 grid, compute edge density per cell)
    3. Resolution fingerprint (width, height, aspect ratio)

    These features capture the *geometry and layout* of the scene, which
    is consistent across day/night/IR for the same camera location.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning("Failed to read image: %s (returning zero vector)", img_path)
        return np.zeros(EDGE_HIST_BINS + 16 + 3)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx)

    threshold = np.percentile(magnitude, 70)
    mask = magnitude > threshold
    if mask.sum() > 0:
        edge_hist, _ = np.histogram(
            orientation[mask],
            bins=EDGE_HIST_BINS,
            range=(-np.pi, np.pi),
            weights=magnitude[mask],
        )
        edge_hist = edge_hist / (edge_hist.sum() + 1e-8)
    else:
        edge_hist = np.zeros(EDGE_HIST_BINS)

    edges = cv2.Canny(gray, 50, 150)
    grid_h, grid_w = 4, 4
    cell_h, cell_w = h // grid_h, w // grid_w
    spatial_density = np.zeros(grid_h * grid_w)
    for r in range(grid_h):
        for c in range(grid_w):
            cell = edges[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            spatial_density[r * grid_w + c] = cell.mean() / 255.0

    max_dim = max(w, h)
    res_feat = np.array([w / max_dim, h / max_dim, w / (h + 1e-8)])

    return np.concatenate([edge_hist, spatial_density, res_feat])


def extract_structural_batch(
    frames: dict[str, list[dict]], ts_list: list[str], tiles_per_frame: int
) -> np.ndarray:
    """Extract structural features for each frame (averaged over representative tiles)."""
    all_features = []
    total = len(ts_list)
    for i, ts in enumerate(ts_list):
        reps = pick_representatives(frames[ts], tiles_per_frame)
        feats = []
        for r in reps:
            feat = extract_structural_features(r["path"])
            if np.any(feat):
                feats.append(feat)

        if feats:
            avg_feat = np.mean(feats, axis=0)
        else:
            avg_feat = np.zeros(EDGE_HIST_BINS + 16 + 3)

        all_features.append(avg_feat)
        if (i + 1) % 20 == 0 or i + 1 == total:
            logger.info("  Structural features: %d/%d", i + 1, total)
    return np.array(all_features)
