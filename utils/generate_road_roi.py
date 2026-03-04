#!/usr/bin/env python3
"""
Semi-automatic road ROI annotation for each camera location.

For every location_N folder, this script:
  1. Auto-suggests a road ROI polygon from YOLO detection-label heatmaps
     (optionally refined with a lane-segmentation model).
  2. Opens an interactive OpenCV window for manual adjustment.
  3. Prompts for number of lanes and cars-per-lane (auto-suggested).
  4. Saves all polygons + lane metadata to a single road_roi.json config.

Usage:
    python generate_road_roi.py                   # annotate all locations
    python generate_road_roi.py --location 3      # re-annotate one location
    python generate_road_roi.py --no-auto          # skip auto-suggestion
    python generate_road_roi.py --no-lane-seg      # skip lane-seg refinement
    python generate_road_roi.py --preview-only     # view existing ROIs (read-only)

Controls (interactive window):
    Left-click          Add vertex / drag existing vertex
    Right-click         Delete nearest vertex
    r                   Reset to auto-suggested polygon
    c                   Clear all vertices
    n / Enter           Accept and move to next location
    s                   Skip location (no ROI saved)
    q                   Save progress and quit
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity

from common import (
    LANE_SEG_WEIGHTS_PATH,
    ROI_CONFIG_PATH,
    discover_locations,
    save_road_roi,
    time_period,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

KDE_BANDWIDTH = 80
HEATMAP_THRESHOLD = 0.25
APPROX_EPSILON_FRAC = 0.02

VERTEX_RADIUS = 8
VERTEX_GRAB_RADIUS = 15
OVERLAY_ALPHA = 0.30
POLY_COLOR = (0, 255, 0)
POLY_EDGE_COLOR = (0, 200, 0)
VERTEX_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

WINDOW_NAME = "Road ROI Annotation"

LANE_KDE_BANDWIDTH = 35
MIN_LANE_WIDTH_PX = 80
GAP_FACTOR = 0.3
DEFAULT_NUM_LANES = 2
DEFAULT_CARS_PER_LANE = 5
LANE_VEHICLE_CLASSES = {0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13}


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
        content = txt.read_text().strip()
        if not content:
            continue
        for line in content.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            rows.append([cls, cx, cy, w, h])
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


def build_heatmap(
    centroids: np.ndarray, img_w: int, img_h: int, bandwidth: float = KDE_BANDWIDTH
) -> np.ndarray:
    """Build a 2-D KDE density map from pixel centroids.  Returns a (H, W) float array."""
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
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, APPROX_EPSILON_FRAC * peri, True)
    return approx.reshape(-1, 2).astype(np.int32)


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


# ──────────────────────────────────────────────────────────────────────
# Step B — Refine with lane-segmentation model (optional)
# ──────────────────────────────────────────────────────────────────────
def load_lane_seg_model(weights: Path):
    """Load YOLO lane segmentation model.  Returns None on failure."""
    if not weights.exists():
        logger.warning(f"Lane-seg weights not found: {weights}")
        return None
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        logger.info(f"Loaded lane-seg model from {weights}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load lane-seg model: {e}")
        return None


def lane_seg_mask(model, images: list[Path], img_w: int, img_h: int) -> np.ndarray:
    """
    Run lane-seg inference on a set of images, union the masks,
    and return a binary (H, W) uint8 mask.
    """
    union_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for img_path in images:
        try:
            results = model(str(img_path), verbose=False)
            if results and results[0].masks is not None:
                for m in results[0].masks.data:
                    seg = m.cpu().numpy()
                    seg_resized = cv2.resize(seg.astype(np.float32), (img_w, img_h))
                    union_mask[seg_resized > 0.5] = 255
        except Exception as e:
            logger.debug(f"Lane-seg failed on {img_path.name}: {e}")
    return union_mask


def refine_polygon_with_lane_seg(
    model, loc_dir: Path, base_polygon: np.ndarray, img_w: int, img_h: int
) -> np.ndarray:
    """Combine label-based polygon with lane-seg mask."""
    images_dir = loc_dir / "images"
    all_imgs = sorted(images_dir.glob("*.jpg"))
    if not all_imgs:
        return base_polygon

    day_imgs = [p for p in all_imgs if time_period(p.name.split("-")[1][:6]) == "day"]
    sample = (day_imgs or all_imgs)[:5]

    seg_mask = lane_seg_mask(model, sample, img_w, img_h)
    if seg_mask.max() == 0:
        logger.info(
            f"  {loc_dir.name}: lane-seg produced empty mask, using labels only"
        )
        return base_polygon

    combined = seg_mask.copy()
    if len(base_polygon) >= 3:
        cv2.fillPoly(combined, [base_polygon], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return base_polygon
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, APPROX_EPSILON_FRAC * peri, True)
    refined = approx.reshape(-1, 2).astype(np.int32)
    logger.info(f"  {loc_dir.name}: refined polygon has {len(refined)} vertices")
    return refined


# ──────────────────────────────────────────────────────────────────────
# Step B2 — Auto-estimate number of lanes and cars per lane
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


def estimate_num_lanes(polygon: np.ndarray, label_data: np.ndarray) -> int:
    """
    Estimate number of lanes by projecting vehicle centroids within the ROI
    onto the cross-road axis and counting peaks in the 1-D KDE.

    Parameters
    ----------
    polygon : (V, 2) int32 array of ROI vertices
    label_data : (N, 5) array [class_id, cx, cy, w, h] in pixel coords

    Returns
    -------
    Estimated lane count (>= 1).
    """
    if len(polygon) < 3 or len(label_data) < 5:
        return DEFAULT_NUM_LANES

    mask_cls = np.isin(label_data[:, 0].astype(int), list(LANE_VEHICLE_CLASSES))
    vehicles = label_data[mask_cls]
    if len(vehicles) < 5:
        return DEFAULT_NUM_LANES

    poly_contour = polygon.reshape(-1, 1, 2).astype(np.float32)
    inside_mask = np.array(
        [
            cv2.pointPolygonTest(poly_contour, (float(cx), float(cy)), False) >= 0
            for cx, cy in vehicles[:, 1:3]
        ]
    )
    inside = vehicles[inside_mask]
    if len(inside) < 5:
        return DEFAULT_NUM_LANES

    _, cross_axis, _ = _get_road_axes(polygon)
    projections = inside[:, 1:3] @ cross_axis

    proj_col = projections.reshape(-1, 1)
    kde = KernelDensity(bandwidth=LANE_KDE_BANDWIDTH, kernel="gaussian")
    kde.fit(proj_col)

    p_min, p_max = float(projections.min()), float(projections.max())
    x_eval = np.linspace(p_min, p_max, 500).reshape(-1, 1)
    density = np.exp(kde.score_samples(x_eval))

    step_size = (p_max - p_min) / 500 + 1e-9
    peaks, _ = find_peaks(density, distance=MIN_LANE_WIDTH_PX / step_size)
    n_lanes = max(len(peaks), 1)
    logger.info(
        f"  Lane estimate: {n_lanes} peaks in cross-road KDE "
        f"({len(inside)} vehicles in ROI)"
    )
    return n_lanes


def estimate_cars_per_lane(
    polygon: np.ndarray, label_data: np.ndarray, num_lanes: int
) -> int:
    """
    Geometric estimate: how many cars fit bumper-to-bumper (with gap) in one lane.

    Uses the road-direction extent of the ROI and median vehicle length
    along that axis.
    """
    if len(polygon) < 3 or len(label_data) < 3 or num_lanes < 1:
        return DEFAULT_CARS_PER_LANE

    road_axis, _, road_length = _get_road_axes(polygon)

    mask_cls = np.isin(label_data[:, 0].astype(int), list(LANE_VEHICLE_CLASSES))
    vehicles = label_data[mask_cls]
    if len(vehicles) < 3:
        return DEFAULT_CARS_PER_LANE

    poly_contour = polygon.reshape(-1, 1, 2).astype(np.float32)
    inside_mask = np.array(
        [
            cv2.pointPolygonTest(poly_contour, (float(cx), float(cy)), False) >= 0
            for cx, cy in vehicles[:, 1:3]
        ]
    )
    inside = vehicles[inside_mask]
    if len(inside) < 3:
        return DEFAULT_CARS_PER_LANE

    w_px = inside[:, 3]
    h_px = inside[:, 4]
    car_lengths = np.maximum(
        np.abs(w_px * road_axis[0]) + np.abs(h_px * road_axis[1]),
        np.abs(w_px * road_axis[1]) + np.abs(h_px * road_axis[0]),
    )
    median_length = float(np.median(car_lengths))
    if median_length < 1:
        return DEFAULT_CARS_PER_LANE

    cars = int(road_length / (median_length * (1 + GAP_FACTOR)))
    cars = max(cars, 1)
    logger.info(
        f"  Cars/lane estimate: {cars} "
        f"(road_len={road_length:.0f}px, median_car={median_length:.0f}px)"
    )
    return cars


# ──────────────────────────────────────────────────────────────────────
# Console prompt helpers
# ──────────────────────────────────────────────────────────────────────
def prompt_positive_int(prompt_text: str, default: int) -> int:
    """Prompt user for a positive integer with a default value."""
    while True:
        raw = input(prompt_text).strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if val >= 1:
                return val
            logger.info("  Please enter a positive integer.")
        except ValueError:
            logger.warning("  Please enter a valid integer.")


# ──────────────────────────────────────────────────────────────────────
# Step C — Interactive OpenCV annotation GUI
# ──────────────────────────────────────────────────────────────────────
class ROIAnnotator:
    """Interactive polygon editor using OpenCV highgui."""

    def __init__(self, image: np.ndarray, polygon: np.ndarray, title: str = ""):
        self.base_image = image.copy()
        self.polygon = list(map(list, polygon)) if len(polygon) > 0 else []
        self.auto_polygon = [list(p) for p in polygon] if len(polygon) > 0 else []
        self.title = title
        self.dragging_idx: Optional[int] = None
        self.accepted = False
        self.skipped = False
        self.quit = False

    def _find_nearest_vertex(self, x: int, y: int) -> Optional[int]:
        """Return index of nearest vertex within grab radius, or None."""
        for i, (vx, vy) in enumerate(self.polygon):
            if (vx - x) ** 2 + (vy - y) ** 2 <= VERTEX_GRAB_RADIUS**2:
                return i
        return None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._find_nearest_vertex(x, y)
            if idx is not None:
                self.dragging_idx = idx
            else:
                self.polygon.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx is not None:
                self.polygon[self.dragging_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self._find_nearest_vertex(x, y)
            if idx is not None and len(self.polygon) > 0:
                self.polygon.pop(idx)

    def _render(self) -> np.ndarray:
        vis = self.base_image.copy()
        poly = np.array(self.polygon, dtype=np.int32)

        if len(poly) >= 3:
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly], POLY_COLOR)
            cv2.addWeighted(overlay, OVERLAY_ALPHA, vis, 1 - OVERLAY_ALPHA, 0, vis)
            cv2.polylines(vis, [poly], True, POLY_EDGE_COLOR, 2, cv2.LINE_AA)

        for i, (vx, vy) in enumerate(self.polygon):
            cv2.circle(vis, (vx, vy), VERTEX_RADIUS, VERTEX_COLOR, -1, cv2.LINE_AA)
            cv2.putText(
                vis,
                str(i),
                (vx + 10, vy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

        h = vis.shape[0]
        lines = [
            f"{self.title}  |  {len(self.polygon)} vertices",
            "L-click: add/drag  |  R-click: delete  |  r: reset  |  c: clear",
            "n/Enter: accept  |  s: skip  |  q: save & quit",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(
                vis,
                txt,
                (10, h - 10 - (len(lines) - 1 - i) * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return vis

    def run(self) -> Optional[list[list[int]]]:
        """
        Show interactive window; returns polygon vertices or None if skipped.
        Sets self.quit if user pressed 'q'.
        """
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_NAME,
            min(self.base_image.shape[1], 1600),
            min(self.base_image.shape[0], 900),
        )
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        while True:
            vis = self._render()
            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("n") or key == 13:
                self.accepted = True
                break
            elif key == ord("s"):
                self.skipped = True
                break
            elif key == ord("q"):
                self.quit = True
                break
            elif key == ord("r"):
                self.polygon = [list(p) for p in self.auto_polygon]
            elif key == ord("c"):
                self.polygon = []
            elif key == 27:
                self.quit = True
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)

        if self.skipped:
            return None
        return self.polygon if self.polygon else None


# ──────────────────────────────────────────────────────────────────────
# Preview-only mode
# ──────────────────────────────────────────────────────────────────────
def preview_existing(config_path: Path):
    """Display existing ROI polygons for each location in sequence."""
    if not config_path.exists():
        logger.warning("No ROI config found at %s", config_path)
        return
    data = json.loads(config_path.read_text())
    locations = discover_locations()

    for loc_id, loc_dir in locations:
        loc_name = f"location_{loc_id}"
        entry = data.get(loc_name)
        if not entry or not entry.get("polygon"):
            logger.info("%s: no ROI defined, skipping", loc_name)
            continue

        images_dir = loc_dir / "images"
        sample_img = next(images_dir.glob("*.jpg"), None)
        if sample_img is None:
            continue
        img = cv2.imread(str(sample_img))
        poly = np.array(entry["polygon"], dtype=np.int32)

        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], POLY_COLOR)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, img, 1 - OVERLAY_ALPHA, 0, img)
        cv2.polylines(img, [poly], True, POLY_EDGE_COLOR, 2, cv2.LINE_AA)

        num_lanes = entry.get("num_lanes", 0)
        cars_per_lane = entry.get("cars_per_lane", 0)
        max_cars = num_lanes * cars_per_lane if num_lanes and cars_per_lane else 0

        info_text = f"{loc_name} ({len(poly)} verts"
        if num_lanes:
            info_text += (
                f" | {num_lanes} lanes | {cars_per_lane} cars/lane | max {max_cars}"
            )
        info_text += ")"

        cv2.putText(
            img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(img.shape[1], 1600), min(img.shape[0], 900))
        cv2.imshow(WINDOW_NAME, img)
        logger.info("Showing %s — press any key for next, 'q' to quit", loc_name)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────
# Pick a good representative image for annotation
# ──────────────────────────────────────────────────────────────────────
def pick_annotation_image(loc_dir: Path) -> Optional[Path]:
    """Pick a random daytime image (if available) for clearest annotation."""
    images_dir = loc_dir / "images"
    all_imgs = sorted(images_dir.glob("*.jpg"))
    if not all_imgs:
        return None

    for period_pref in ("day", "night", "IR"):
        candidates = [
            p for p in all_imgs if time_period(p.name.split("-")[1][:6]) == period_pref
        ]
        if candidates:
            return random.choice(candidates)
    return random.choice(all_imgs)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Semi-automatic road ROI annotation per camera location"
    )
    parser.add_argument(
        "--location",
        type=int,
        default=None,
        help="Annotate only this location ID (e.g. --location 3)",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Skip auto-suggestion, start with blank polygon",
    )
    parser.add_argument(
        "--no-lane-seg", action="store_true", help="Skip lane-segmentation refinement"
    )
    parser.add_argument(
        "--preview-only", action="store_true", help="View existing ROIs without editing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output JSON path (default: {ROI_CONFIG_PATH})",
    )
    args = parser.parse_args()

    config_path = Path(args.output) if args.output else ROI_CONFIG_PATH

    if args.preview_only:
        preview_existing(config_path)
        return

    locations = discover_locations()
    if args.location is not None:
        locations = [(lid, ld) for lid, ld in locations if lid == args.location]
        if not locations:
            logger.error("location_%d not found", args.location)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Road ROI Annotation Tool")
    logger.info("=" * 60)
    logger.info("Locations: %s", [f"location_{lid}" for lid, _ in locations])
    logger.info("Auto-suggest: %s", "OFF" if args.no_auto else "ON (labels heatmap)")
    logger.info("Lane-seg refinement: %s", "OFF" if args.no_lane_seg else "ON")
    logger.info("Output: %s", config_path)
    logger.info("")

    existing: dict[str, dict] = {}
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        logger.info("Loaded existing config with %d locations", len(existing))

    lane_model = None
    if not args.no_auto and not args.no_lane_seg:
        lane_model = load_lane_seg_model(LANE_SEG_WEIGHTS_PATH)

    for loc_id, loc_dir in locations:
        loc_name = f"location_{loc_id}"
        logger.info("─" * 40)
        logger.info("Processing %s", loc_name)
        logger.info("─" * 40)

        img_path = pick_annotation_image(loc_dir)
        if img_path is None:
            logger.warning("  No images found in %s, skipping", loc_name)
            continue

        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            logger.warning("  Failed to read %s, skipping", img_path.name)
            continue

        logger.info("  Image: %s (%dx%d)", img_path.name, img_w, img_h)

        if args.no_auto:
            suggested_poly = np.empty((0, 2), dtype=np.int32)
        else:
            suggested_poly = autosuggest_from_labels(loc_dir, img_w, img_h)
            if lane_model is not None and len(suggested_poly) >= 3:
                suggested_poly = refine_polygon_with_lane_seg(
                    lane_model, loc_dir, suggested_poly, img_w, img_h
                )

        if len(suggested_poly) > 0:
            logger.info("  Auto-suggested polygon: %d vertices", len(suggested_poly))
        else:
            logger.info("  No auto-suggestion available — draw manually")

        annotator = ROIAnnotator(cv_img, suggested_poly, title=loc_name)
        result = annotator.run()

        if annotator.quit:
            logger.info("Quitting early — saving progress...")
            break

        if result is None:
            logger.info("  %s: skipped", loc_name)
            continue

        result_poly = np.array(result, dtype=np.int32)
        labels_dir = loc_dir / "labels"
        label_data = np.empty((0, 5))
        if labels_dir.exists() and not args.no_auto:
            label_data = load_label_data(labels_dir, img_w, img_h)

        auto_lanes = estimate_num_lanes(result_poly, label_data)
        auto_cpl = estimate_cars_per_lane(result_poly, label_data, auto_lanes)

        prev = existing.get(loc_name, {})
        default_lanes = prev.get("num_lanes", auto_lanes)
        default_cpl = prev.get("cars_per_lane", auto_cpl)

        logger.info("  Auto-estimated: %d lanes, %d cars/lane", auto_lanes, auto_cpl)
        num_lanes = prompt_positive_int(
            f"  Number of lanes [{default_lanes}]: ", default_lanes
        )
        cars_per_lane = prompt_positive_int(
            f"  Cars per lane [{default_cpl}]: ", default_cpl
        )

        max_cars = num_lanes * cars_per_lane
        logger.info(
            "  \u2192 %d lanes \u00d7 %d cars/lane = %d max cars in ROI",
            num_lanes,
            cars_per_lane,
            max_cars,
        )

        existing[loc_name] = {
            "polygon": result,
            "image_size": [img_w, img_h],
            "num_lanes": num_lanes,
            "cars_per_lane": cars_per_lane,
        }
        logger.info("  %s: saved %d vertices + lane metadata", loc_name, len(result))

    out = save_road_roi(existing, config_path)
    n_defined = sum(1 for v in existing.values() if v.get("polygon"))
    logger.info("Saved %d ROI polygon(s) to %s", n_defined, out)
    logger.info("Done!")


if __name__ == "__main__":
    main()
