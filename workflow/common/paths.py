#!/usr/bin/env python3
"""
Path constants and shared configuration for the TIMS workflow.
"""

import re
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Project root
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent

# ──────────────────────────────────────────────────────────────────────
# Dataset paths
# ──────────────────────────────────────────────────────────────────────

DENSITY_BASE_PATH = (
    PROJECT_ROOT_PATH / "gdrive" / "YOLOv10" / "data_train" / "TIMS_density_dataset"
)

RAW_TRAIN_PATH = DENSITY_BASE_PATH / "raw" / "train"
TRAIN_BY_LOCATION_PATH = DENSITY_BASE_PATH / "raw" / "train_by_location"

ROI_CONFIG_PATH = TRAIN_BY_LOCATION_PATH / "road_roi.json"
CLUSTER_PREVIEW_PATH = TRAIN_BY_LOCATION_PATH / "cluster_preview.png"
CLUSTER_CSV_PATH = TRAIN_BY_LOCATION_PATH / "cluster_mapping.csv"

DENSITY_OUTPUT_PATH = DENSITY_BASE_PATH / "train"

TIMS_FINAL_BASE_PATH = (
    PROJECT_ROOT_PATH
    / "gdrive"
    / "YOLOv10"
    / "data_train"
    / "TIMS_dataset_final"
    / "train_original"
)

TIMS_FINAL_IMAGES_PATH = TIMS_FINAL_BASE_PATH / "images"
TIMS_FINAL_LABELS_PATH = TIMS_FINAL_BASE_PATH / "labels"

LANE_SEG_WEIGHTS_PATH = (
    PROJECT_ROOT_PATH
    / "gdrive"
    / "YOLOv10"
    / "weights"
    / "pre-final"
    / "best_lane_seg_capstone"
    / "weights"
    / "best.pt"
)

# ──────────────────────────────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────────────────────────────

CCTV_PATTERN = re.compile(
    r"^(?P<hexhash>[0-9a-fA-F]{8})-(?P<timestamp>\d{6})_100_(?P<tile>\d+)\.(?:jpe?g|png|bmp|tiff)$",
    re.IGNORECASE,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

LANE_VEHICLE_CLASSES = {0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13}

# ──────────────────────────────────────────────────────────────────────
# Clustering / validation config (shared between cluster_by_location
# and validate_location so neither depends on the other)
# ──────────────────────────────────────────────────────────────────────

PCA_COMPONENTS = 50
STRUCTURAL_WEIGHT = 0.3
