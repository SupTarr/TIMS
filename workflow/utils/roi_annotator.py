#!/usr/bin/env python3
"""
Interactive OpenCV annotation GUI for road ROI polygons.

Provides the ROIAnnotator class (polygon editor with drag, add, delete),
a preview-only mode for viewing existing ROIs, and helpers for picking
representative annotation images and prompting for integer input.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..common import IMAGE_EXTENSIONS, discover_locations, parse_filename, time_period

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# GUI constants
# ──────────────────────────────────────────────────────────────────────

VERTEX_RADIUS = 8
VERTEX_GRAB_RADIUS = 15
OVERLAY_ALPHA = 0.30
POLY_COLOR = (0, 255, 0)
POLY_EDGE_COLOR = (0, 200, 0)
VERTEX_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

WINDOW_NAME = "Road ROI Annotation"


# ──────────────────────────────────────────────────────────────────────
# Console prompt helpers
# ──────────────────────────────────────────────────────────────────────
def prompt_positive_int(prompt_text: str, default: int) -> int:
    """Prompt user for a positive integer with a default value."""
    while True:
        raw = input(prompt_text).strip()
        if not raw:
            return max(default, 1)
        try:
            val = int(raw)
            if val >= 1:
                return val
            logger.info("  Please enter a positive integer.")
        except ValueError:
            logger.warning("  Please enter a valid integer.")


# ──────────────────────────────────────────────────────────────────────
# Interactive OpenCV annotation GUI
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
            f"{self.title} | {len(self.polygon)} vertices",
            "L-click: add/drag | R-click: delete | r: reset | c: clear",
            "n/Enter: accept | s: skip | q: save & quit",
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
        if not images_dir.is_dir():
            images_dir = loc_dir
        sample_img = next(
            (
                f
                for f in images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ),
            None,
        )
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
    if not images_dir.is_dir():
        images_dir = loc_dir
    all_imgs = sorted(
        f
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_imgs:
        return None

    shuffled_imgs = list(all_imgs)
    random.shuffle(shuffled_imgs)

    for period_pref in ("day", "night", "IR"):
        for p in shuffled_imgs:
            parsed = parse_filename(p.name)
            if parsed and time_period(parsed[1], p) == period_pref:
                return p

    return random.choice(all_imgs)
