#!/usr/bin/env python3
"""
Bird's-Eye View (BEV) homography utilities.

Provides perspective transform from a camera-view ROI quadrilateral to a
top-down rectangular view where 1 pixel = constant real-world distance.

Main entry points:
  order_quadrilateral    — Sort polygon vertices into [TL, TR, BR, BL].
  reduce_to_quad         — Approximate an N-vertex polygon to 4 corners.
  compute_homography     — Build the 3×3 perspective matrix.
  transform_point        — Warp a single (x, y) through the matrix.
  compute_bev_scale      — Calibrate meters-per-pixel using lane width.
  compute_bev_config     — One-call helper that builds everything from a polygon.

Physical vehicle dimensions (metres, along-road length) are also exported
so that density and lane-estimation modules can convert pixel boxes to
real-world occupancy.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Physical vehicle lengths (metres, along-road)
# Keyed by YOLO class id — same numbering as common/parsing.py
# ──────────────────────────────────────────────────────────────────────
VEHICLE_LENGTHS_M: dict[int, float] = {
    0: 10.0,  # 10_full_truck
    1: 18.0,  # 11_full_trailer
    2: 16.5,  # 12_semi_trailer
    3: 4.5,  # 13_modified_car
    4: 0.0,  # 14_pedestrian  (excluded)
    5: 1.8,  # 1_bicycle
    6: 2.2,  # 2_motorcycle
    7: 4.5,  # 3_car
    8: 4.8,  # 4_car_7
    9: 5.5,  # 5_small_bus
    10: 9.0,  # 6_medium_bus
    11: 12.0,  # 7_large_bus
    12: 5.3,  # 8_pickup
    13: 7.5,  # 9_truck
}

VEHICLE_WIDTHS_M: dict[int, float] = {
    0: 2.5,  # full_truck
    1: 2.5,  # full_trailer
    2: 2.5,  # semi_trailer
    3: 1.8,  # modified_car
    4: 0.0,  # pedestrian
    5: 0.6,  # bicycle
    6: 0.8,  # motorcycle
    7: 1.8,  # car
    8: 1.8,  # car_7
    9: 2.3,  # small_bus
    10: 2.5,  # medium_bus
    11: 2.5,  # large_bus
    12: 1.9,  # pickup
    13: 2.5,  # truck
}

STANDARD_LANE_WIDTH_M = 3.5
BEV_PPM_DEFAULT = 20.0


# ──────────────────────────────────────────────────────────────────────
# Vertex ordering
# ──────────────────────────────────────────────────────────────────────
def order_quadrilateral(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as [top-left, top-right, bottom-right, bottom-left].

    Sorts vertices by angle from their centroid to obtain a consistent
    clockwise winding, then rotates so the point with the smallest
    (x + y) — i.e. closest to the top-left corner — comes first.

    This avoids the classic sum/diff heuristic which can assign the
    same physical vertex to multiple corners when the quadrilateral
    is oriented diagonally.
    """
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    cx, cy = pts.mean(axis=0)

    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    cw_order = np.argsort(-angles)
    pts = pts[cw_order]

    sums = pts.sum(axis=1)
    start = int(np.argmin(sums))
    pts = np.roll(pts, -start, axis=0)

    return pts


def reduce_to_quad(polygon: np.ndarray) -> np.ndarray:
    """
    Approximate an N-vertex polygon (N ≥ 4) to exactly 4 vertices.

    Uses ``cv2.approxPolyDP`` with increasing epsilon until 4 vertices
    remain. If the polygon already has 4 vertices, returns it as-is.
    Falls back to the 4 extreme points of the convex hull if approxPolyDP
    cannot reduce to exactly 4.
    """
    polygon = np.asarray(polygon, dtype=np.float32)
    if len(polygon) == 4:
        return polygon.reshape(4, 2)

    contour = polygon.reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, closed=True)
    for factor in np.arange(0.01, 0.25, 0.005):
        approx = cv2.approxPolyDP(contour, factor * perimeter, closed=True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    hull = cv2.convexHull(contour).reshape(-1, 2)
    s = hull.sum(axis=1)
    d = np.diff(hull, axis=1).ravel()
    indices = [
        int(np.argmin(s)),
        int(np.argmin(d)),
        int(np.argmax(s)),
        int(np.argmax(d)),
    ]

    seen: set[int] = set()
    unique: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    if len(unique) < 4:
        for i in range(len(hull)):
            if i not in seen:
                unique.append(i)
                seen.add(i)
            if len(unique) == 4:
                break
    return hull[unique[:4]]


# ──────────────────────────────────────────────────────────────────────
# Homography computation
# ──────────────────────────────────────────────────────────────────────
def compute_homography(
    src_quad: np.ndarray, dst_width: float, dst_height: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Compute the 3×3 perspective transform from *src_quad* to a rectangle.

    Parameters
    ----------
    src_quad : (4, 2) array
        Source quadrilateral in pixel coords, ordered [TL, TR, BR, BL].
    dst_width, dst_height : float
        Desired output rectangle dimensions (pixels).

    Returns
    -------
    M : (3, 3) float64 — forward homography  (camera → BEV)
    M_inv : (3, 3) float64 — inverse homography (BEV → camera)
    bev_size : (int, int) — (width, height) of BEV image
    """
    src = np.array(src_quad, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    bev_size = (int(round(dst_width)), int(round(dst_height)))
    return M, M_inv, bev_size


def transform_point(pt: tuple[float, float], M: np.ndarray) -> tuple[float, float]:
    """Apply a 3×3 homography *M* to a single 2-D point."""
    x, y = float(pt[0]), float(pt[1])
    denom = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    if abs(denom) < 1e-12:
        return (0.0, 0.0)
    tx = (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / denom
    ty = (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / denom
    return (float(tx), float(ty))


def transform_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply homography *M* to an (N, 2) array of points. Returns (N, 2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, M)
    return warped.reshape(-1, 2)


# ──────────────────────────────────────────────────────────────────────
# Scale calibration
# ──────────────────────────────────────────────────────────────────────
def compute_bev_scale(
    src_quad: np.ndarray, num_lanes: int, lane_width_m: float = STANDARD_LANE_WIDTH_M
) -> tuple[float, float, float]:
    """
    Calibrate the BEV rectangle dimensions and metres-per-pixel.

    Uses the cross-road extent of *src_quad* as ``num_lanes × lane_width_m``
    and derives the along-road extent proportionally.

    Returns (bev_width_px, bev_height_px, meters_per_pixel).
    """
    ordered = order_quadrilateral(src_quad)
    top_width = float(np.linalg.norm(ordered[1] - ordered[0]))
    bot_width = float(np.linalg.norm(ordered[2] - ordered[3]))
    avg_cross_px = (top_width + bot_width) / 2.0

    left_height = float(np.linalg.norm(ordered[3] - ordered[0]))
    right_height = float(np.linalg.norm(ordered[2] - ordered[1]))
    avg_along_px = (left_height + right_height) / 2.0

    if avg_along_px < 1.0:
        logger.warning(
            "BEV quad along-road extent is near-zero (%.1f px) — "
            "the quad may be degenerate",
            avg_along_px,
        )

    total_cross_m = max(num_lanes, 1) * lane_width_m
    mpp = total_cross_m / avg_cross_px if avg_cross_px > 0 else 0.05
    road_length_m = avg_along_px * mpp

    ppm = BEV_PPM_DEFAULT
    bev_w = total_cross_m * ppm
    bev_h = road_length_m * ppm

    bev_w = max(bev_w, 50)
    bev_h = max(bev_h, 50)

    actual_mpp = total_cross_m / bev_w
    return bev_w, bev_h, actual_mpp


# ──────────────────────────────────────────────────────────────────────
# High-level helper
# ──────────────────────────────────────────────────────────────────────
def compute_bev_config(
    polygon: np.ndarray, num_lanes: int, lane_width_m: float = STANDARD_LANE_WIDTH_M
) -> dict:
    """
    One-call helper: compute all BEV parameters from a polygon + lane count.

    Returns a dict with:
      - ``"bev_quad"``:  ordered [TL, TR, BR, BL] as list of [x, y]
      - ``"bev_matrix"``: 3×3 forward homography as nested list
      - ``"bev_matrix_inv"``: 3×3 inverse homography
      - ``"bev_size"``: [width, height] of BEV rectangle (pixels)
      - ``"meters_per_pixel"``: float calibration factor
      - ``"road_length_m"``: estimated road length in metres
      - ``"road_width_m"``: = num_lanes × lane_width_m

    Returns an empty dict if the polygon has < 4 vertices.
    """
    poly = np.asarray(polygon, dtype=np.float32)
    if len(poly) < 4:
        logger.warning("Polygon has < 4 vertices — cannot compute BEV config")
        return {}

    quad = reduce_to_quad(poly) if len(poly) > 4 else poly.reshape(4, 2)
    ordered = order_quadrilateral(quad)

    bev_w, bev_h, mpp = compute_bev_scale(ordered, num_lanes, lane_width_m)
    M, M_inv, bev_size = compute_homography(ordered, bev_w, bev_h)

    road_width_m = max(num_lanes, 1) * lane_width_m
    road_length_m = bev_h * mpp

    logger.info(
        "  BEV config: %dx%d px, %.3f m/px, road %.1f×%.1f m",
        bev_size[0],
        bev_size[1],
        mpp,
        road_width_m,
        road_length_m,
    )

    return {
        "bev_quad": ordered.tolist(),
        "bev_matrix": M.tolist(),
        "bev_matrix_inv": M_inv.tolist(),
        "bev_size": list(bev_size),
        "meters_per_pixel": float(mpp),
        "road_length_m": float(road_length_m),
        "road_width_m": float(road_width_m),
    }
