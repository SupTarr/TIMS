#!/usr/bin/env python3
"""
KDE-based lane count estimation.

Uses Kernel Density Estimation on perspective-normalised cross-road
projections to count peaks (lanes) with adaptive Silverman bandwidth.
"""

import logging

import numpy as np
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)


def _kde_lane_count(projections: np.ndarray, n_inside: int) -> int:
    """Run KDE peak counting on normalised projections. Returns lane count."""
    p_min, p_max = float(projections.min()), float(projections.max())
    spread = p_max - p_min

    n_pts = len(projections)
    std = float(np.std(projections))
    iqr = float(np.percentile(projections, 75) - np.percentile(projections, 25))
    a = min(std, iqr / 1.34) if iqr > 1e-6 else std
    silverman_bw = 0.75 * a * n_pts ** (-1 / 5) if a > 1e-6 else 0.10
    adaptive_bw = float(np.clip(silverman_bw, 0.03, 0.20))
    logger.info(
        f"  Lane KDE bandwidth: {adaptive_bw:.3f} "
        f"(Silverman ROT, n={n_pts}, std={std:.3f}, IQR={iqr:.3f})"
    )

    kde = KernelDensity(bandwidth=adaptive_bw, kernel="gaussian")
    kde.fit(projections.reshape(-1, 1))

    x_eval = np.linspace(p_min, p_max, 500).reshape(-1, 1)
    density = np.exp(kde.score_samples(x_eval))

    step_size = (spread / 500) + 1e-9
    min_norm_dist = max(1.5 * adaptive_bw, 0.10)
    peak_distance = int(np.clip(min_norm_dist / step_size, 1, 499))
    peaks, _ = find_peaks(density, distance=peak_distance)

    n_lanes = max(len(peaks), 1)
    logger.info(
        f"  Lane estimate (KDE): {n_lanes} peaks "
        f"({n_inside} vehicles in ROI, spread={spread:.3f} normalised)"
    )
    return n_lanes


def estimate_num_lanes(
    polygon: np.ndarray,
    label_data: np.ndarray,
    *,
    _get_road_axes,
    _filter_vehicles_in_polygon,
    _perspective_normalise,
    _adaptive_geo_lane_estimate,
    MIN_VEHICLES_FOR_LANE_ESTIMATE: int,
    DEFAULT_NUM_LANES: int,
) -> int:
    """
    Estimate number of lanes via KDE peak counting on perspective-normalised
    cross-road projections.

    Uses adaptive geometry fallback (vehicle-width-scaled) when too few
    vehicles are detected.

    Note: geometry helpers are injected to avoid circular imports.
    """
    if len(polygon) < 3:
        return DEFAULT_NUM_LANES

    road_axis, cross_axis, _ = _get_road_axes(polygon)

    if len(label_data) == 0:
        geo = _adaptive_geo_lane_estimate(polygon, cross_axis)
        logger.info(f"  Lane estimate (geo fallback): {geo} (no label data)")
        return geo

    inside = _filter_vehicles_in_polygon(polygon, label_data)

    if len(inside) < MIN_VEHICLES_FOR_LANE_ESTIMATE:
        geo = _adaptive_geo_lane_estimate(polygon, cross_axis, inside)
        logger.info(
            f"  Lane estimate (geo fallback): {geo} "
            f"(only {len(inside)} vehicles in ROI)"
        )
        return geo

    projections = _perspective_normalise(polygon, inside, road_axis, cross_axis)

    if len(projections) < MIN_VEHICLES_FOR_LANE_ESTIMATE:
        geo = _adaptive_geo_lane_estimate(polygon, cross_axis, inside)
        logger.info(
            f"  Lane estimate (geo fallback): {geo} "
            f"(only {len(projections)} normalisable vehicles)"
        )
        return geo

    return _kde_lane_count(projections, len(inside))
