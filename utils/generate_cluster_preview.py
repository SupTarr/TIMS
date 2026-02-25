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
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import (
    BASE_DIR as _TIMS_BASE,
    TRAIN_BY_LOCATION,
    ROI_CONFIG,
    group_tiles_by_frame,
    load_road_roi,
    pick_representative,
    time_period,
    discover_locations,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = _TIMS_BASE / "raw" / "train_by_location"

PREVIEW_SAMPLES = 5
OVERLAY_ALPHA = 0.30
ROI_COLOR = (0, 200, 0)

CLASS_NAMES = {
    0: "full_truck",
    1: "full_trailer",
    2: "semi_trailer",
    3: "modified_car",
    4: "pedestrian",
    5: "bicycle",
    6: "motorcycle",
    7: "car",
    8: "car_7",
    9: "small_bus",
    10: "medium_bus",
    11: "large_bus",
    12: "pickup",
    13: "truck",
}

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


def load_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Read a YOLO label file.
    Returns list of (class_id, cx, cy, w, h) with normalised coords.
    """
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


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
    labels_dir = img_path.parent.parent / "labels"
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
        "--no-roi",
        action="store_true",
        help="Skip drawing ROI polygon overlay",
    )
    parser.add_argument(
        "--no-boxes",
        action="store_true",
        help="Skip drawing YOLO bounding boxes",
    )
    args = parser.parse_args()

    save_path = Path(args.output) if args.output else BASE_DIR / "cluster_preview.png"
    n_samples = args.samples
    draw_roi_flag = not args.no_roi
    draw_boxes_flag = not args.no_boxes

    print("=" * 60)
    print("Generate Cluster Preview from Location Folders")
    print("=" * 60)
    print(f"Base dir: {BASE_DIR}")
    print(f"Draw ROI: {draw_roi_flag}  |  Draw boxes: {draw_boxes_flag}")

    if not BASE_DIR.exists():
        print(f"ERROR: Base directory not found: {BASE_DIR}")
        return

    roi_map: dict[str, np.ndarray] = {}
    if draw_roi_flag:
        try:
            roi_map = load_road_roi(ROI_CONFIG)
            print(f"Loaded {len(roi_map)} ROI polygon(s) from {ROI_CONFIG}")
        except FileNotFoundError:
            print(f"WARNING: ROI config not found ({ROI_CONFIG}), skipping ROI overlay")

    locations = discover_locations(BASE_DIR)
    n_clusters = len(locations)
    print(
        f"Found {n_clusters} locations: {[f'location_{lid}' for lid, _ in locations]}"
    )

    if n_clusters == 0:
        print("No location folders found. Nothing to do.")
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
        print(f"  location_{loc_id}: {n_frames} frames ({n_tiles} tiles)")

    print(f"\nGenerating preview grid ({n_clusters} rows x {n_samples} cols)...")

    fig, axes = plt.subplots(
        n_clusters, n_samples, figsize=(4 * n_samples, 3 * n_clusters), squeeze=False
    )
    fig.suptitle(
        "Cluster Preview – ROI + Detections (rows = locations)",
        fontsize=14,
        y=1.01,
    )

    for row_idx, (loc_id, _) in enumerate(locations):
        data = loc_data[loc_id]
        frames = data["frames"]
        n_frames = data["n_frames"]

        loc_name = f"location_{loc_id}"
        polygon = roi_map.get(loc_name, np.empty((0, 2), dtype=np.int32))
        has_roi = len(polygon) >= 3

        sampled_ts = pick_sample_frames(frames, n_samples)

        for col in range(n_samples):
            ax = axes[row_idx][col]
            ax.axis("off")
            if col < len(sampled_ts):
                ts = sampled_ts[col]
                rep = pick_representative(frames[ts])
                try:
                    img = np.array(Image.open(rep["path"]).convert("RGB"))

                    if draw_roi_flag and has_roi:
                        img = overlay_roi(img, polygon)

                    if draw_boxes_flag:
                        lbl_path = label_path_for_image(rep["path"])
                        boxes = load_yolo_labels(lbl_path)
                        if boxes:
                            img = draw_boxes(img, boxes)

                    ax.imshow(img)
                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        "ERR",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                period = time_period(ts)
                n_boxes = (
                    len(load_yolo_labels(label_path_for_image(rep["path"])))
                    if draw_boxes_flag
                    else 0
                )
                box_info = f" [{n_boxes} obj]" if n_boxes else ""
                ax.set_title(f"loc_{loc_id} | {ts} ({period}){box_info}", fontsize=7)
            else:
                ax.set_visible(False)

        roi_status = f", ROI {len(polygon)} pts" if has_roi else ""
        axes[row_idx][0].set_ylabel(
            f"location_{loc_id}\n({n_frames} frames{roi_status})",
            fontsize=9,
            rotation=0,
            labelpad=100,
            va="center",
        )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPreview saved to {save_path}")
    print(f"File size: {save_path.stat().st_size / 1024:.1f} KB")
    print("Done!")


if __name__ == "__main__":
    main()
