#!/usr/bin/env python3
"""
GMM-based lane count estimation.

Uses Gaussian Mixture Models with BIC-based model selection to estimate
the number of lanes from perspective-normalised cross-road projections.
"""

import logging

import numpy as np
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

GMM_MAX_LANES = 6
GMM_MIN_VEHICLES = 6


def _gmm_lane_count(projections: np.ndarray, n_inside: int) -> int:
    """Fit GMMs with k=1..K_max and select best k via BIC."""
    X = projections.reshape(-1, 1)
    max_k = min(GMM_MAX_LANES, len(projections) // 2)
    max_k = max(max_k, 1)

    best_bic = np.inf
    best_k = 1
    bic_values: list[float] = []

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", n_init=3, random_state=42
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_values.append(bic)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    gmm_best = GaussianMixture(
        n_components=best_k, covariance_type="full", n_init=3, random_state=42
    )
    gmm_best.fit(X)
    centres = sorted(gmm_best.means_.flatten().tolist())

    logger.info(
        f"  Lane estimate (GMM-BIC): {best_k} lanes "
        f"({n_inside} vehicles, BIC values: "
        f"{', '.join(f'k={i+1}:{b:.1f}' for i, b in enumerate(bic_values))})"
    )
    logger.info(f"  GMM lane centres (normalised): {[f'{c:.3f}' for c in centres]}")
    return best_k


def estimate_num_lanes_gmm(
    polygon: np.ndarray,
    label_data: np.ndarray,
    *,
    _get_road_axes,
    _filter_vehicles_in_polygon,
    _perspective_normalise,
    _adaptive_geo_lane_estimate,
) -> int:
    """
    Estimate number of lanes by fitting Gaussian Mixture Models with
    k = 1 … GMM_MAX_LANES components and selecting the best k via BIC.

    Note: geometry helpers are injected to avoid circular imports.
    """
    from .lane_estimation import DEFAULT_NUM_LANES

    if len(polygon) < 3:
        return DEFAULT_NUM_LANES

    road_axis, cross_axis, _ = _get_road_axes(polygon)

    if len(label_data) == 0:
        geo = _adaptive_geo_lane_estimate(polygon, cross_axis)
        logger.info(f"  GMM lane estimate (geo fallback): {geo} (no label data)")
        return geo

    inside = _filter_vehicles_in_polygon(polygon, label_data)

    if len(inside) < GMM_MIN_VEHICLES:
        geo = _adaptive_geo_lane_estimate(polygon, cross_axis, inside)
        logger.info(
            f"  GMM lane estimate (geo fallback): {geo} "
            f"(only {len(inside)} vehicles, need {GMM_MIN_VEHICLES})"
        )
        return geo

    projections = _perspective_normalise(polygon, inside, road_axis, cross_axis)

    if len(projections) < GMM_MIN_VEHICLES:
        geo = _adaptive_geo_lane_estimate(polygon, cross_axis, inside)
        logger.info(
            f"  GMM lane estimate (geo fallback): {geo} "
            f"(only {len(projections)} normalisable vehicles)"
        )
        return geo

    return _gmm_lane_count(projections, len(inside))
