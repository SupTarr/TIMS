#!/usr/bin/env python3
"""
Perspective-aware lane count and cars-per-lane estimation.

Provides:
  estimate_num_lanes           — KDE peak counting (delegated to lane_kde).
  estimate_num_lanes_gmm       — GMM-BIC model selection (delegated to lane_gmm).
  estimate_num_lanes_consensus — Multi-method consensus (KDE + GMM + geometry).
  estimate_cars_per_lane       — Class-aware bumper-to-bumper capacity estimate.

Helper geometry functions (_get_road_axes, _cross_road_width_px,
_polygon_cross_extent_at_depth) are also exported for re-use.
"""

import logging
from collections import Counter

import cv2
import numpy as np

from ..common import LANE_VEHICLE_CLASSES
from .lane_gmm import GMM_MIN_VEHICLES
from .lane_gmm import _gmm_lane_count
from .lane_kde import _kde_lane_count

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MIN_LANE_WIDTH_PX = 80
GAP_FACTOR = 0.3
DEFAULT_NUM_LANES = 2
DEFAULT_CARS_PER_LANE = 5

MIN_VEHICLES_FOR_LANE_ESTIMATE = 2
MIN_VEHICLES_FOR_CPL_ESTIMATE = 2

LANE_TO_CAR_WIDTH_RATIO = 2.05

CLASS_GAP_FACTORS: dict[int, float] = {
    0: 0.45,   # full_truck
    1: 0.50,   # full_trailer
    2: 0.50,   # semi_trailer
    3: 0.25,   # modified_car
    7: 0.25,   # car
    8: 0.25,   # car_7
    9: 0.35,   # small_bus
    10: 0.40,  # medium_bus
    11: 0.45,  # large_bus
    12: 0.30,  # pickup
    13: 0.40,  # truck
}


# ──────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────
def _get_road_axes(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute principal (road direction) and cross-road unit vectors from the
    minimum-area bounding rectangle of the polygon.

    Returns (road_axis, cross_axis, road_length_px).
    """
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    edge0 = box[1] - box[0]
    edge1 = box[2] - box[1]
    len0 = float(np.linalg.norm(edge0))
    len1 = float(np.linalg.norm(edge1))

    if len0 >= len1:
        road_axis = edge0 / (len0 + 1e-9)
        road_length = len0
    else:
        road_axis = edge1 / (len1 + 1e-9)
        road_length = len1

    cross_axis = np.array([-road_axis[1], road_axis[0]])
    return road_axis, cross_axis, road_length


def _cross_road_width_px(polygon: np.ndarray) -> float:
    """
    Return the cross-road extent of the ROI polygon in pixels
    (the shorter dimension of the minAreaRect).
    """
    rect = cv2.minAreaRect(polygon)
    w, h = rect[1]
    return float(min(w, h))


def _polygon_cross_extent_at_depth(
    polygon: np.ndarray, road_axis: np.ndarray, cross_axis: np.ndarray, depth: float
) -> tuple[float, float]:
    """
    Find the cross-road extent (min, max) of *polygon* at a given *depth*
    along *road_axis*.
    """
    n = len(polygon)
    cross_positions: list[float] = []

    for i in range(n):
        p0 = polygon[i].astype(float)
        p1 = polygon[(i + 1) % n].astype(float)

        d0 = float(p0 @ road_axis)
        d1 = float(p1 @ road_axis)

        if abs(d1 - d0) < 1e-9:
            continue
        if not (min(d0, d1) - 1e-6 <= depth <= max(d0, d1) + 1e-6):
            continue

        t = (depth - d0) / (d1 - d0)
        t = max(0.0, min(1.0, t))
        intersection = p0 + t * (p1 - p0)
        cross_positions.append(float(intersection @ cross_axis))

    if len(cross_positions) < 2:
        return 0.0, 0.0

    return min(cross_positions), max(cross_positions)


# ──────────────────────────────────────────────────────────────────────
# Internal: shared vehicle filtering & perspective normalisation
# ──────────────────────────────────────────────────────────────────────
def _filter_vehicles_in_polygon(
    polygon: np.ndarray, label_data: np.ndarray
) -> np.ndarray:
    """Return rows of *label_data* whose centroid lies inside *polygon*."""
    mask_cls = np.isin(label_data[:, 0].astype(int), list(LANE_VEHICLE_CLASSES))
    vehicles = label_data[mask_cls]
    if len(vehicles) == 0:
        return vehicles

    poly_contour = polygon.reshape(-1, 1, 2).astype(np.float32)
    inside_mask = np.array(
        [
            cv2.pointPolygonTest(poly_contour, (float(cx), float(cy)), False) >= 0
            for cx, cy in vehicles[:, 1:3]
        ]
    )
    return vehicles[inside_mask]


def _perspective_normalise(
    polygon: np.ndarray,
    inside: np.ndarray,
    road_axis: np.ndarray,
    cross_axis: np.ndarray,
) -> np.ndarray:
    """
    Return perspective-normalised cross-road positions in [0, 1].
    """
    norm_positions: list[float] = []
    for cx, cy in inside[:, 1:3]:
        pt = np.array([cx, cy])
        depth = float(pt @ road_axis)
        c_min, c_max = _polygon_cross_extent_at_depth(
            polygon, road_axis, cross_axis, depth
        )
        local_width = c_max - c_min
        if local_width < 1e-6:
            continue
        c_val = float(pt @ cross_axis)
        norm_positions.append((c_val - c_min) / local_width)
    return np.array(norm_positions) if norm_positions else np.array([])


def _adaptive_geo_lane_estimate(
    polygon: np.ndarray, cross_axis: np.ndarray, inside: np.ndarray | None = None
) -> int:
    """
    Geometry-based lane estimate that adapts to camera perspective.
    """
    cross_proj = polygon.astype(float) @ cross_axis
    cross_width_px = float(cross_proj.max() - cross_proj.min())

    if inside is not None and len(inside) >= MIN_VEHICLES_FOR_LANE_ESTIMATE:
        road_axis, _, _ = _get_road_axes(polygon)
        veh_cross_widths = inside[:, 3] * abs(cross_axis[0]) + inside[:, 4] * abs(
            cross_axis[1]
        )

        median_car_width = float(np.median(veh_cross_widths))
        if median_car_width > 5.0:
            adaptive_lane_width = median_car_width * LANE_TO_CAR_WIDTH_RATIO
            estimate = max(1, round(cross_width_px / adaptive_lane_width))
            logger.info(
                f"  Adaptive geo estimate: {estimate} lanes "
                f"(cross={cross_width_px:.0f}px, car_w={median_car_width:.0f}px, "
                f"lane_w={adaptive_lane_width:.0f}px)"
            )
            return estimate

    estimate = max(1, round(cross_width_px / MIN_LANE_WIDTH_PX))
    return estimate


# ──────────────────────────────────────────────────────────────────────
# Public: KDE-based lane estimation (delegates to lane_kde)
# ──────────────────────────────────────────────────────────────────────
def estimate_num_lanes(polygon: np.ndarray, label_data: np.ndarray) -> int:
    """
    Estimate number of lanes via KDE peak counting on perspective-normalised
    cross-road projections.
    """
    from .lane_kde import estimate_num_lanes as _kde_estimate

    return _kde_estimate(
        polygon,
        label_data,
        _get_road_axes=_get_road_axes,
        _filter_vehicles_in_polygon=_filter_vehicles_in_polygon,
        _perspective_normalise=_perspective_normalise,
        _adaptive_geo_lane_estimate=_adaptive_geo_lane_estimate,
        MIN_VEHICLES_FOR_LANE_ESTIMATE=MIN_VEHICLES_FOR_LANE_ESTIMATE,
        DEFAULT_NUM_LANES=DEFAULT_NUM_LANES,
    )


# ──────────────────────────────────────────────────────────────────────
# Public: GMM-based lane estimation (delegates to lane_gmm)
# ──────────────────────────────────────────────────────────────────────
def estimate_num_lanes_gmm(polygon: np.ndarray, label_data: np.ndarray) -> int:
    """
    Estimate number of lanes by fitting Gaussian Mixture Models with
    k = 1 … K_max and selecting the best k via BIC.
    """
    from .lane_gmm import estimate_num_lanes_gmm as _gmm_estimate

    return _gmm_estimate(
        polygon,
        label_data,
        _get_road_axes=_get_road_axes,
        _filter_vehicles_in_polygon=_filter_vehicles_in_polygon,
        _perspective_normalise=_perspective_normalise,
        _adaptive_geo_lane_estimate=_adaptive_geo_lane_estimate,
    )


# ──────────────────────────────────────────────────────────────────────
# Public: consensus lane estimation (KDE + GMM + geometry)
# ──────────────────────────────────────────────────────────────────────
def estimate_num_lanes_consensus(polygon: np.ndarray, label_data: np.ndarray) -> int:
    """
    Multi-method consensus lane estimation.

    Runs KDE, GMM, and adaptive geometry independently, then takes the
    **median** of all three estimates.
    """
    if len(polygon) < 3:
        return DEFAULT_NUM_LANES

    road_axis, cross_axis, _ = _get_road_axes(polygon)
    inside = (
        _filter_vehicles_in_polygon(polygon, label_data)
        if len(label_data) > 0
        else np.empty((0, 5))
    )

    geo = _adaptive_geo_lane_estimate(
        polygon, cross_axis, inside if len(inside) else None
    )
    estimates = [geo]
    labels = ["geo"]

    projections = np.array([])
    if len(inside) >= MIN_VEHICLES_FOR_LANE_ESTIMATE:
        projections = _perspective_normalise(polygon, inside, road_axis, cross_axis)

    if len(projections) >= MIN_VEHICLES_FOR_LANE_ESTIMATE:
        kde_est = _kde_lane_count(projections, len(inside))
        estimates.append(kde_est)
        labels.append("kde")

    if len(projections) >= GMM_MIN_VEHICLES:
        gmm_est = _gmm_lane_count(projections, len(inside))
        estimates.append(gmm_est)
        labels.append("gmm")

    result = int(np.median(estimates))
    result = max(result, 1)

    detail = ", ".join(f"{l}={e}" for l, e in zip(labels, estimates))
    logger.info(f"  Lane estimate (consensus): {result} " f"(median of [{detail}])")
    return result


# ──────────────────────────────────────────────────────────────────────
# Public: class-aware cars-per-lane estimation
# ──────────────────────────────────────────────────────────────────────
def estimate_cars_per_lane(
    polygon: np.ndarray, label_data: np.ndarray, num_lanes: int
) -> int:
    """
    Class-aware bumper-to-bumper capacity estimate.
    """
    if len(polygon) < 3 or num_lanes < 1:
        return DEFAULT_CARS_PER_LANE

    road_axis, _, _ = _get_road_axes(polygon)
    vertex_proj = polygon.astype(float) @ road_axis
    road_length = float(vertex_proj.max() - vertex_proj.min())

    if len(label_data) == 0:
        cars = max(1, int(road_length / (MIN_LANE_WIDTH_PX * (1 + GAP_FACTOR))))
        logger.info(
            f"  Cars/lane (geo fallback): {cars} "
            f"(road_len={road_length:.0f}px, no label data)"
        )
        return cars

    inside = _filter_vehicles_in_polygon(polygon, label_data)
    if len(inside) < MIN_VEHICLES_FOR_CPL_ESTIMATE:
        cars = max(1, int(road_length / (MIN_LANE_WIDTH_PX * (1 + GAP_FACTOR))))
        logger.info(
            f"  Cars/lane (geo fallback): {cars} "
            f"(only {len(inside)} vehicles in ROI)"
        )
        return cars

    classes = inside[:, 0].astype(int)
    w_px = inside[:, 3]
    h_px = inside[:, 4]
    car_lengths = w_px * abs(road_axis[0]) + h_px * abs(road_axis[1])

    class_counts = Counter(classes.tolist())
    total_count = sum(class_counts.values())

    weighted_length = 0.0
    weighted_gap = 0.0
    for cls_id, count in class_counts.items():
        cls_mask = classes == cls_id
        cls_median_len = float(np.median(car_lengths[cls_mask]))
        fraction = count / total_count
        weighted_length += cls_median_len * fraction
        weighted_gap += CLASS_GAP_FACTORS.get(cls_id, GAP_FACTOR) * fraction

    weighted_length = float(np.clip(weighted_length, 5.0, road_length))
    if weighted_length < 1:
        return DEFAULT_CARS_PER_LANE

    cars = int(road_length / (weighted_length * (1 + weighted_gap)))
    cars = max(cars, 1)
    logger.info(
        f"  Cars/lane estimate: {cars} "
        f"(road_len={road_length:.0f}px, weighted_car={weighted_length:.0f}px, "
        f"weighted_gap={weighted_gap:.2f}, classes={dict(class_counts)})"
    )
    return cars
