#!/usr/bin/env python3
"""
Generate cluster_preview.png from existing location folders.

Reads from train_by_location/location_0 .. location_N and produces a
preview grid with:
  - Road ROI polygon overlay (green)
  - YOLO bounding boxes with class labels

Usage:
    python generate_cluster_preview.py
    python generate_cluster_preview.py --samples 8   # more columns
    python generate_cluster_preview.py --no-roi       # skip ROI overlay
    python generate_cluster_preview.py --no-boxes     # skip bounding boxes
"""

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .common import (
    TRAIN_BY_LOCATION_PATH as BASE_DIR,
    CLASS_NAMES,
    ROI_CONFIG_PATH,
    discover_locations,
    group_tiles_by_frame,
    load_road_roi,
    parse_yolo_labels,
    pick_representative,
    setup_logging,
    time_period,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

PREVIEW_SAMPLES = 5
OVERLAY_ALPHA = 0.30
ROI_COLOR = (0, 200, 0)

_PALETTE = [
    (255, 0, 0),
    (0, 0, 255),
    (255, 165, 0),
    (0, 200, 200),
    (255, 0, 255),
    (128, 0, 128),
    (0, 128, 0),
    (30, 144, 255),
    (255, 215, 0),
    (0, 255, 127),
    (220, 20, 60),
    (75, 0, 130),
    (244, 164, 96),
    (0, 206, 209),
]


def class_color(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[cls_id % len(_PALETTE)]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def pick_sample_frames(frames: dict[str, list[dict]], n: int) -> list[str]:
    """Pick n timestamps evenly spaced across sorted timestamps."""
    ts_sorted = sorted(frames.keys())
    if len(ts_sorted) <= n:
        return ts_sorted
    indices = np.linspace(0, len(ts_sorted) - 1, n, dtype=int)
    return [ts_sorted[i] for i in indices]


def pick_frames_by_vehicle_count(frames: dict[str, list[dict]], n: int) -> list[tuple[str, int]]:
    """Pick n timestamps sorted by bounding-box count (most vehicles first).

    Returns list of (timestamp, box_count) tuples in descending order.
    """
    counts: list[tuple[str, int]] = []
    for ts, tiles in frames.items():
        rep = pick_representative(tiles)
        lbl_path = label_path_for_image(rep["path"])
        boxes = parse_yolo_labels(lbl_path)
        counts.append((ts, len(boxes)))
    counts.sort(key=lambda x: x[1], reverse=True)
    return counts[:n]


def overlay_roi(img_rgb: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Draw filled ROI polygon overlay on an RGB image."""
    vis = img_rgb.copy()
    if len(polygon) < 3:
        return vis
    overlay = vis.copy()
    cv2.fillPoly(overlay, [polygon], ROI_COLOR)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, vis, 1 - OVERLAY_ALPHA, 0, vis)
    cv2.polylines(vis, [polygon], True, ROI_COLOR, 2, cv2.LINE_AA)
    for pt in polygon:
        cv2.circle(vis, tuple(pt), 5, (255, 0, 0), -1, cv2.LINE_AA)
    return vis


def draw_boxes(img_rgb: np.ndarray, boxes: list, thickness: int = 2) -> np.ndarray:
    """Draw YOLO bounding boxes with class labels on an RGB image."""
    vis = img_rgb.copy()
    img_h, img_w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(img_w, img_h) / 1500)
    txt_thick = max(1, thickness - 1)

    for cls_id, cx, cy, w, h in boxes:
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        color = class_color(cls_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = CLASS_NAMES.get(cls_id, str(cls_id))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, txt_thick)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 1, y1 - 2),
            font,
            font_scale,
            (255, 255, 255),
            txt_thick,
            cv2.LINE_AA,
        )
    return vis


def label_path_for_image(img_path: Path) -> Path:
    """Derive the YOLO label .txt path from an image path (images/ → labels/)."""
    if img_path.parent.name == "images":
        labels_dir = img_path.parent.parent / "labels"
    else:
        labels_dir = img_path.parent / "labels"
    return labels_dir / img_path.with_suffix(".txt").name


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate cluster_preview.png from existing location folders"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=PREVIEW_SAMPLES,
        help="Number of sample images per location row (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: train_by_location/cluster_preview.png)",
    )
    parser.add_argument(
        "--no-roi", action="store_true", help="Skip drawing ROI polygon overlay"
    )
    parser.add_argument(
        "--no-boxes", action="store_true", help="Skip drawing YOLO bounding boxes"
    )
    args = parser.parse_args()

    save_path = Path(args.output) if args.output else BASE_DIR / "cluster_preview.png"
    n_samples = args.samples
    draw_roi_flag = not args.no_roi
    draw_boxes_flag = not args.no_boxes

    logger.info("=" * 60)
    logger.info("Generate Cluster Preview from Location Folders")
    logger.info("=" * 60)
    logger.info("Base dir: %s", BASE_DIR)
    logger.info("Draw ROI: %s  |  Draw boxes: %s", draw_roi_flag, draw_boxes_flag)

    if not BASE_DIR.exists():
        logger.error("Base directory not found: %s", BASE_DIR)
        return

    roi_map: dict[str, np.ndarray] = {}
    if draw_roi_flag:
        try:
            roi_map = load_road_roi(ROI_CONFIG_PATH)
            logger.info(
                "Loaded %d ROI polygon(s) from %s", len(roi_map), ROI_CONFIG_PATH
            )
        except FileNotFoundError:
            logger.warning(
                "ROI config not found (%s), skipping ROI overlay", ROI_CONFIG_PATH
            )

    locations = discover_locations(BASE_DIR)
    n_clusters = len(locations)
    logger.info(
        "Found %d locations: %s",
        n_clusters,
        [f"location_{lid}" for lid, _ in locations],
    )

    if n_clusters == 0:
        logger.info("No location folders found. Nothing to do.")
        return

    loc_data = {}
    for loc_id, loc_dir in locations:
        src_dir = loc_dir / "images"
        if not src_dir.is_dir():
            src_dir = loc_dir

        frames = group_tiles_by_frame(src_dir)
        n_frames = len(frames)
        n_tiles = sum(len(v) for v in frames.values())
        loc_data[loc_id] = {"frames": frames, "n_frames": n_frames, "n_tiles": n_tiles}
        logger.info("  location_%d: %d frames (%d tiles)", loc_id, n_frames, n_tiles)

    logger.info("Generating preview grid (%d rows x %d cols)...", n_clusters, n_samples)

    fig, axes = plt.subplots(
        n_clusters, n_samples, figsize=(4 * n_samples, 3 * n_clusters), squeeze=False
    )
    fig.suptitle(
        "Cluster Preview - ROI + Detections (rows = locations)", fontsize=14, y=1.01
    )

    for row_idx, (loc_id, _) in enumerate(locations):
        data = loc_data[loc_id]
        frames = data["frames"]
        n_frames = data["n_frames"]

        loc_name = f"location_{loc_id}"
        roi_entry = roi_map.get(loc_name, {})
        polygon = roi_entry.get("polygon", np.empty((0, 2), dtype=np.int32))
        has_roi = len(polygon) >= 3

        num_lanes = roi_entry.get("num_lanes", "?")
        cars_per_lane = roi_entry.get("cars_per_lane", "?")

        sampled = pick_frames_by_vehicle_count(frames, n_samples)

        for col in range(n_samples):
            ax = axes[row_idx][col]
            ax.axis("off")
            if col < len(sampled):
                ts, precount = sampled[col]
                rep = pick_representative(frames[ts])
                boxes = []
                try:
                    img = np.array(Image.open(rep["path"]).convert("RGB"))

                    if draw_roi_flag and has_roi:
                        img = overlay_roi(img, polygon)

                    lbl_path = label_path_for_image(rep["path"])
                    boxes = parse_yolo_labels(lbl_path)
                    if draw_boxes_flag and boxes:
                        img = draw_boxes(img, boxes)

                    ax.imshow(img)
                    if col == 0:
                        lane_text = f"{num_lanes} lanes / {cars_per_lane} cars·lane"
                        ax.text(
                            0.02,
                            0.98,
                            lane_text,
                            transform=ax.transAxes,
                            fontsize=7,
                            color="white",
                            va="top",
                            ha="left",
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
                        )
                except Exception:
                    ax.text(
                        0.5,
                        0.5,
                        "ERR",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                period = time_period(rep["ts"], rep["path"])
                n_boxes = len(boxes)
                box_info = f" [{n_boxes} obj]" if n_boxes else ""
                rank_label = f"#{col + 1}"
                ax.set_title(f"{rank_label} loc_{loc_id} | {ts} ({period}){box_info}", fontsize=7)
            else:
                ax.set_visible(False)

        if has_roi:
            roi_status = f"\nROI: {len(polygon)} pts\n{num_lanes} lanes\n{cars_per_lane} cars/lane"
        else:
            roi_status = f"\n{num_lanes} lanes\n{cars_per_lane} cars/lane"

        axes[row_idx][0].set_ylabel(
            f"location_{loc_id}\n({n_frames} frames){roi_status}",
            fontsize=12,
            rotation=0,
            labelpad=15,
            ha="right",
            va="center",
        )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Preview saved to %s", save_path)
    logger.info("File size: %.1f KB", save_path.stat().st_size / 1024)
    logger.info("Done!")


if __name__ == "__main__":
    setup_logging()
    main()
