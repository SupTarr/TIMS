#!/usr/bin/env python3
"""
Semi-automatic road ROI annotation for each camera location.

For every location_N folder, this script:
  1. Auto-suggests a road ROI polygon from YOLO detection-label heatmaps
     (optionally refined with a lane-segmentation model).
  2. Opens an interactive OpenCV window for manual adjustment.
  3. Saves all polygons to a single road_roi.json config.

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
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from sklearn.neighbors import KernelDensity

from common import BASE_DIR, ROI_CONFIG, discover_locations, save_road_roi, time_period

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
LANE_SEG_WEIGHTS = (
    BASE_DIR.parent.parent
    / "weights"
    / "pre-final"
    / "best_lane_seg_capstone"
    / "weights"
    / "best.pt"
)

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


# ──────────────────────────────────────────────────────────────────────
# Step A — Auto-suggest ROI via detection-label heatmap
# ──────────────────────────────────────────────────────────────
def load_label_centroids(labels_dir: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Read YOLO label files and convert normalised bbox centres to pixel coords.
    Returns an (N, 2) array of [x, y] pixel positions.
    """
    centroids = []
    for txt in labels_dir.glob("*.txt"):
        content = txt.read_text().strip()
        if not content:
            continue
        for line in content.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy = float(parts[1]), float(parts[2])
            centroids.append([cx * img_w, cy * img_h])
    if not centroids:
        return np.empty((0, 2))
    return np.array(centroids)


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
        print(f"No ROI config found at {config_path}")
        return
    data = json.loads(config_path.read_text())
    locations = discover_locations()

    for loc_id, loc_dir in locations:
        loc_name = f"location_{loc_id}"
        entry = data.get(loc_name)
        if not entry or not entry.get("polygon"):
            print(f"{loc_name}: no ROI defined, skipping")
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
        cv2.putText(
            img,
            f"{loc_name} ({len(poly)} vertices)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(img.shape[1], 1600), min(img.shape[0], 900))
        cv2.imshow(WINDOW_NAME, img)
        print(f"Showing {loc_name} — press any key for next, 'q' to quit")
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────
# Pick a good representative image for annotation
# ──────────────────────────────────────────────────────────────────────
def pick_annotation_image(loc_dir: Path) -> Optional[Path]:
    """Pick a daytime image (if available) for clearest annotation."""
    images_dir = loc_dir / "images"
    all_imgs = sorted(images_dir.glob("*.jpg"))
    if not all_imgs:
        return None

    for period_pref in ("day", "night", "IR"):
        candidates = [
            p for p in all_imgs if time_period(p.name.split("-")[1][:6]) == period_pref
        ]
        if candidates:
            return candidates[len(candidates) // 2]
    return all_imgs[len(all_imgs) // 2]


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
        help=f"Output JSON path (default: {ROI_CONFIG})",
    )
    args = parser.parse_args()

    config_path = Path(args.output) if args.output else ROI_CONFIG

    if args.preview_only:
        preview_existing(config_path)
        return

    locations = discover_locations()
    if args.location is not None:
        locations = [(lid, ld) for lid, ld in locations if lid == args.location]
        if not locations:
            print(f"ERROR: location_{args.location} not found")
            sys.exit(1)

    print("=" * 60)
    print("Road ROI Annotation Tool")
    print("=" * 60)
    print(f"Locations: {[f'location_{lid}' for lid, _ in locations]}")
    print(f"Auto-suggest: {'OFF' if args.no_auto else 'ON (labels heatmap)'}")
    print(f"Lane-seg refinement: {'OFF' if args.no_lane_seg else 'ON'}")
    print(f"Output: {config_path}")
    print()

    existing: dict[str, dict] = {}
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        print(f"Loaded existing config with {len(existing)} locations")

    lane_model = None
    if not args.no_auto and not args.no_lane_seg:
        lane_model = load_lane_seg_model(LANE_SEG_WEIGHTS)

    for loc_id, loc_dir in locations:
        loc_name = f"location_{loc_id}"
        print(f"\n{'─' * 40}")
        print(f"Processing {loc_name}")
        print(f"{'─' * 40}")

        img_path = pick_annotation_image(loc_dir)
        if img_path is None:
            print(f"  No images found in {loc_name}, skipping")
            continue

        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            print(f"  Failed to read {img_path.name}, skipping")
            continue

        print(f"  Image: {img_path.name} ({img_w}x{img_h})")

        if args.no_auto:
            suggested_poly = np.empty((0, 2), dtype=np.int32)
        else:
            suggested_poly = autosuggest_from_labels(loc_dir, img_w, img_h)
            if lane_model is not None and len(suggested_poly) >= 3:
                suggested_poly = refine_polygon_with_lane_seg(
                    lane_model, loc_dir, suggested_poly, img_w, img_h
                )

        if len(suggested_poly) > 0:
            print(f"  Auto-suggested polygon: {len(suggested_poly)} vertices")
        else:
            print("  No auto-suggestion available — draw manually")

        annotator = ROIAnnotator(cv_img, suggested_poly, title=loc_name)
        result = annotator.run()

        if annotator.quit:
            print("\nQuitting early — saving progress...")
            break

        if result is None:
            print(f"  {loc_name}: skipped")
            continue

        existing[loc_name] = {"polygon": result, "image_size": [img_w, img_h]}
        print(f"  {loc_name}: saved {len(result)} vertices")

    out = save_road_roi(existing, config_path)
    n_defined = sum(1 for v in existing.values() if v.get("polygon"))
    print(f"\nSaved {n_defined} ROI polygon(s) to {out}")
    print("Done!")


if __name__ == "__main__":
    main()
