#!/usr/bin/env python3
"""
Perspective-aware lane count and cars-per-lane estimation.

Provides:
  estimate_num_lanes        — KDE peak counting on perspective-normalised
                              cross-road projections.
  estimate_cars_per_lane    — Geometric bumper-to-bumper capacity estimate.

Helper geometry functions (_get_road_axes, _cross_road_width_px,
_polygon_cross_extent_at_depth) are also exported for re-use.
"""

import logging

import cv2
import numpy as np
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity

from common import LANE_VEHICLE_CLASSES

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
    Used for adaptive KDE bandwidth scaling.
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

    Slices the polygon perpendicular to *road_axis* at *depth* and returns
    ``(cross_min, cross_max)`` measured along *cross_axis*.
    Returns ``(0.0, 0.0)`` when fewer than two edge intersections are found.
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
# Public estimation functions
# ──────────────────────────────────────────────────────────────────────
def estimate_num_lanes(polygon: np.ndarray, label_data: np.ndarray) -> int:
    """
    Estimate number of lanes by projecting vehicle centroids within the ROI
    onto the cross-road axis and counting peaks in the 1-D KDE.

    Cross-road positions are **perspective-normalised**: each centroid's
    lateral position is expressed as a fraction [0, 1] of the polygon
    width at that depth along the road axis.  This prevents lanes from
    merging in the KDE when the ROI is a perspective trapezoid (road
    narrows toward the vanishing point).

    Fallback behaviour:
    - Geometry-based estimate (cross-width ÷ MIN_LANE_WIDTH_PX) when too
      few vehicles are detected or normalisation fails.
    - Safe find_peaks distance clamped to [1, 499].

    Parameters
    ----------
    polygon    : (V, 2) int32 array of ROI vertices
    label_data : (N, 5) array [class_id, cx, cy, w, h] in pixel coords

    Returns
    -------
    Estimated lane count (>= 1).
    """
    if len(polygon) < 3:
        return DEFAULT_NUM_LANES

    cross_width_px = _cross_road_width_px(polygon)
    geo_lane_estimate = max(1, round(cross_width_px / MIN_LANE_WIDTH_PX))

    if len(label_data) == 0:
        logger.info(
            f"  Lane estimate (geo fallback): {geo_lane_estimate} "
            f"(cross_width={cross_width_px:.0f}px, no label data)"
        )
        return geo_lane_estimate

    mask_cls = np.isin(label_data[:, 0].astype(int), list(LANE_VEHICLE_CLASSES))
    vehicles = label_data[mask_cls]

    poly_contour = polygon.reshape(-1, 1, 2).astype(np.float32)
    inside_mask = np.array(
        [
            cv2.pointPolygonTest(poly_contour, (float(cx), float(cy)), False) >= 0
            for cx, cy in vehicles[:, 1:3]
        ]
    )
    inside = vehicles[inside_mask]

    if len(inside) < MIN_VEHICLES_FOR_LANE_ESTIMATE:
        logger.info(
            f"  Lane estimate (geo fallback): {geo_lane_estimate} "
            f"(only {len(inside)} vehicles in ROI, need {MIN_VEHICLES_FOR_LANE_ESTIMATE})"
        )
        return geo_lane_estimate

    road_axis, cross_axis, _ = _get_road_axes(polygon)
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

    if len(norm_positions) < MIN_VEHICLES_FOR_LANE_ESTIMATE:
        logger.info(
            f"  Lane estimate (geo fallback): {geo_lane_estimate} "
            f"(only {len(norm_positions)} normalisable vehicles in ROI)"
        )
        return geo_lane_estimate

    projections = np.array(norm_positions)
    p_min, p_max = float(projections.min()), float(projections.max())
    spread = p_max - p_min

    adaptive_bw = float(np.clip(0.25 / max(geo_lane_estimate, 1), 0.03, 0.25))
    logger.info(
        f"  Lane KDE bandwidth: {adaptive_bw:.3f} "
        f"(normalised, perspective-corrected)"
    )

    kde = KernelDensity(bandwidth=adaptive_bw, kernel="gaussian")
    kde.fit(projections.reshape(-1, 1))

    x_eval = np.linspace(p_min, p_max, 500).reshape(-1, 1)
    density = np.exp(kde.score_samples(x_eval))

    step_size = (spread / 500) + 1e-9
    min_norm_dist = 0.5 / max(geo_lane_estimate, 1)
    peak_distance = int(np.clip(min_norm_dist / step_size, 1, 499))
    peaks, _ = find_peaks(density, distance=peak_distance)

    n_lanes = max(len(peaks), 1)
    logger.info(
        f"  Lane estimate (KDE): {n_lanes} peaks "
        f"({len(inside)} vehicles in ROI, spread={spread:.3f} normalised)"
    )
    return n_lanes


def estimate_cars_per_lane(
    polygon: np.ndarray, label_data: np.ndarray, num_lanes: int
) -> int:
    """
    Geometric estimate: how many cars fit bumper-to-bumper (with gap) in one lane.

    Road length is measured by projecting the ROI polygon vertices onto the
    road axis (more accurate than minAreaRect for non-rectangular trapezoid ROIs).
    Car lengths are the support-function projection of each AABB onto the road
    axis: ``w*|ax| + h*|ay|``.

    Note: pixel-space perspective largely cancels out because both road_length
    and car_lengths shrink together at greater depth — the ratio is stable
    for uniformly distributed detections.

    Parameters
    ----------
    polygon    : (V, 2) int32 array of ROI vertices
    label_data : (N, 5) array [class_id, cx, cy, w, h] in pixel coords
    num_lanes  : number of lanes (>= 1)

    Returns
    -------
    Estimated cars-per-lane (>= 1).
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

    mask_cls = np.isin(label_data[:, 0].astype(int), list(LANE_VEHICLE_CLASSES))
    vehicles = label_data[mask_cls]
    if len(vehicles) < MIN_VEHICLES_FOR_CPL_ESTIMATE:
        cars = max(1, int(road_length / (MIN_LANE_WIDTH_PX * (1 + GAP_FACTOR))))
        logger.info(
            f"  Cars/lane (geo fallback): {cars} "
            f"(only {len(vehicles)} vehicles total)"
        )
        return cars

    poly_contour = polygon.reshape(-1, 1, 2).astype(np.float32)
    inside_mask = np.array(
        [
            cv2.pointPolygonTest(poly_contour, (float(cx), float(cy)), False) >= 0
            for cx, cy in vehicles[:, 1:3]
        ]
    )
    inside = vehicles[inside_mask]
    if len(inside) < MIN_VEHICLES_FOR_CPL_ESTIMATE:
        cars = max(1, int(road_length / (MIN_LANE_WIDTH_PX * (1 + GAP_FACTOR))))
        logger.info(
            f"  Cars/lane (geo fallback): {cars} "
            f"(only {len(inside)} vehicles in ROI)"
        )
        return cars

    w_px = inside[:, 3]
    h_px = inside[:, 4]
    car_lengths = w_px * abs(road_axis[0]) + h_px * abs(road_axis[1])

    median_length = float(np.median(car_lengths))
    median_length = float(np.clip(median_length, 5.0, road_length / max(num_lanes, 1)))

    if median_length < 1:
        return DEFAULT_CARS_PER_LANE

    cars = int(road_length / (median_length * (1 + GAP_FACTOR)))
    cars = max(cars, 1)
    logger.info(
        f"  Cars/lane estimate: {cars} "
        f"(road_len={road_length:.0f}px, median_car={median_length:.0f}px)"
    )
    return cars
