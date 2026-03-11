#!/usr/bin/env python3
"""
Shared utilities for CCTV tile image processing.

Re-exports everything from sub-modules so existing ``from .common import X``
statements continue to work without changes.
"""

from .frames import (
    IR_SATURATION_THRESHOLD,
    detect_frame_modality,
    detect_modality,
    group_tiles_by_frame,
    pick_representative,
    pick_representatives,
    time_period,
)
from .logging import setup_logging
from .parsing import CLASS_NAMES, parse_filename, parse_yolo_labels
from .paths import (
    CCTV_PATTERN,
    CLUSTER_CSV_PATH,
    CLUSTER_PREVIEW_PATH,
    DENSITY_BASE_PATH,
    DENSITY_OUTPUT_PATH,
    DENSITY_TEST_OUTPUT_PATH,
    DENSITY_VALID_OUTPUT_PATH,
    IMAGE_EXTENSIONS,
    LANE_SEG_WEIGHTS_PATH,
    LANE_VEHICLE_CLASSES,
    PCA_COMPONENTS,
    PROJECT_ROOT_PATH,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    RAW_VALID_PATH,
    ROI_CONFIG_PATH,
    STRUCTURAL_WEIGHT,
    TEST_BY_LOCATION_PATH,
    TIMS_FINAL_BASE_PATH,
    TIMS_FINAL_IMAGES_PATH,
    TIMS_FINAL_LABELS_PATH,
    TIMS_FINAL_TEST_BASE_PATH,
    TIMS_FINAL_TEST_LABELS_PATH,
    TIMS_FINAL_VALID_BASE_PATH,
    TIMS_FINAL_VALID_LABELS_PATH,
    TRAIN_BY_LOCATION_PATH,
    VALID_BY_LOCATION_PATH,
)
from .roi import (
    discover_locations,
    filter_vehicles_in_roi,
    load_road_roi,
    save_road_roi,
)

__all__ = [
    # paths
    "DENSITY_BASE_PATH",
    "RAW_TRAIN_PATH",
    "TRAIN_BY_LOCATION_PATH",
    "ROI_CONFIG_PATH",
    "CLUSTER_PREVIEW_PATH",
    "CLUSTER_CSV_PATH",
    "DENSITY_OUTPUT_PATH",
    "DENSITY_TEST_OUTPUT_PATH",
    "DENSITY_VALID_OUTPUT_PATH",
    "RAW_TEST_PATH",
    "RAW_VALID_PATH",
    "TEST_BY_LOCATION_PATH",
    "VALID_BY_LOCATION_PATH",
    "TIMS_FINAL_BASE_PATH",
    "TIMS_FINAL_IMAGES_PATH",
    "TIMS_FINAL_LABELS_PATH",
    "TIMS_FINAL_TEST_BASE_PATH",
    "TIMS_FINAL_TEST_LABELS_PATH",
    "TIMS_FINAL_VALID_BASE_PATH",
    "TIMS_FINAL_VALID_LABELS_PATH",
    "LANE_SEG_WEIGHTS_PATH",
    "PROJECT_ROOT_PATH",
    "PCA_COMPONENTS",
    "STRUCTURAL_WEIGHT",
    # parsing
    "CCTV_PATTERN",
    "CLASS_NAMES",
    "IMAGE_EXTENSIONS",
    "LANE_VEHICLE_CLASSES",
    "parse_filename",
    "parse_yolo_labels",
    # frames
    "IR_SATURATION_THRESHOLD",
    "detect_frame_modality",
    "detect_modality",
    "group_tiles_by_frame",
    "pick_representative",
    "pick_representatives",
    "time_period",
    # roi
    "discover_locations",
    "filter_vehicles_in_roi",
    "load_road_roi",
    "save_road_roi",
    # logging
    "setup_logging",
]
