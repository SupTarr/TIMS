#!/usr/bin/env python3
"""
CLIP embedding extraction with CLAHE preprocessing for CCTV images.

Provides device auto-selection, CLAHE contrast enhancement (important for
IR and low-light imagery), and batched CLIP embedding extraction with
multi-tile averaging per frame.
"""

import logging

import cv2
import numpy as np
import torch
from PIL import Image

from ..common import pick_representatives

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

CLIP_MODEL = "ViT-B/32"
TILES_PER_FRAME = 3


# ──────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────
def select_device(preferred: str) -> str:
    """Auto-select best available device."""
    if preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────
# CLAHE preprocessing
# ──────────────────────────────────────────────────────────────────────
def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    enhance contrast in IR/low-light images. Works on the L channel of LAB
    color space to preserve hue in color images.
    """
    arr = np.array(pil_img)
    if len(arr.shape) == 2 or (len(arr.shape) == 3 and arr.shape[2] == 1):
        gray = arr if len(arr.shape) == 2 else arr[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


# ──────────────────────────────────────────────────────────────────────
# CLIP embedding extraction (with CLAHE + multi-tile averaging)
# ──────────────────────────────────────────────────────────────────────
def extract_clip_embeddings(
    frames: dict[str, list[dict]],
    ts_list: list[str],
    model,
    preprocess,
    device: str,
    batch_size: int = 32,
    tiles_per_frame: int = TILES_PER_FRAME,
) -> np.ndarray:
    """
    Extract CLIP embeddings with improvements for IR accuracy:
    - CLAHE preprocessing to enhance IR/low-light contrast
    - Multi-tile averaging per frame for robustness
    """
    tile_items = []
    for ts_idx, ts in enumerate(ts_list):
        reps = pick_representatives(frames[ts], tiles_per_frame)
        for rep in reps:
            tile_items.append((ts_idx, rep))

    total_tiles = len(tile_items)
    embed_dim = model.visual.output_dim
    all_features = np.zeros((total_tiles, embed_dim), dtype=np.float32)
    skipped_global: set[int] = set()

    for i in range(0, total_tiles, batch_size):
        batch = tile_items[i : i + batch_size]
        images_orig = []

        skipped = []
        for batch_idx, (_, item) in enumerate(batch):
            try:
                img = Image.open(item["path"]).convert("RGB")
            except Exception as e:
                logger.warning("Failed to open %s: %s (skipping tile)", item["path"], e)
                skipped.append(batch_idx)
                images_orig.append(preprocess(Image.new("RGB", (224, 224))))
                continue
            img = apply_clahe(img)
            images_orig.append(preprocess(img))

        orig_tensor = torch.stack(images_orig).to(device)

        with torch.no_grad():
            feat_orig = model.encode_image(orig_tensor).cpu().numpy()

        for s in skipped:
            feat_orig[s] = 0.0
            skipped_global.add(i + s)
        all_features[i : i + len(batch)] = feat_orig

        done = min(i + batch_size, total_tiles)
        logger.info("  CLIP: %d/%d tiles", done, total_tiles)

    n_frames = len(ts_list)
    frame_embeddings = np.zeros((n_frames, embed_dim), dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.int32)

    for idx, (ts_idx, _) in enumerate(tile_items):
        if idx in skipped_global:
            continue
        frame_embeddings[ts_idx] += all_features[idx]
        counts[ts_idx] += 1

    for j in range(n_frames):
        if counts[j] > 0:
            frame_embeddings[j] /= counts[j]

    return frame_embeddings
