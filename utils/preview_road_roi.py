#!/usr/bin/env python3
"""
Generate a montage image showing all road ROI overlays across locations.

Reads road_roi.json and produces roi_preview.png, similar to
cluster_preview.png but with ROI polygons drawn on each tile.

Usage:
    python preview_road_roi.py
    python preview_road_roi.py --output /tmp/my_preview.png
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
    ROI_CONFIG,
    TRAIN_BY_LOCATION,
    discover_locations,
    group_tiles_by_frame,
    load_road_roi,
    pick_representative,
    time_period,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
PREVIEW_SAMPLES = 3  # images per location row
OVERLAY_ALPHA = 0.30
POLY_COLOR_RGB = (0, 200, 0)  # green for matplotlib (RGB)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def pick_sample_frames(frames: dict[str, list[dict]], n: int) -> list[str]:
    ts_sorted = sorted(frames.keys())
    if len(ts_sorted) <= n:
        return ts_sorted
    indices = np.linspace(0, len(ts_sorted) - 1, n, dtype=int)
    return [ts_sorted[i] for i in indices]


def overlay_roi(img_rgb: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Draw filled ROI overlay on an RGB image (in-place safe)."""
    vis = img_rgb.copy()
    if len(polygon) < 3:
        return vis
    # matplotlib uses RGB; cv2 fillPoly expects BGR but we work in RGB here
    overlay = vis.copy()
    cv2.fillPoly(overlay, [polygon], POLY_COLOR_RGB)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, vis, 1 - OVERLAY_ALPHA, 0, vis)
    cv2.polylines(vis, [polygon], True, POLY_COLOR_RGB, 2, cv2.LINE_AA)
    # Draw vertex dots
    for pt in polygon:
        cv2.circle(vis, tuple(pt), 5, (255, 0, 0), -1, cv2.LINE_AA)
    return vis


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate ROI preview montage from road_roi.json"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to road_roi.json (default: {ROI_CONFIG})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: train_by_location/roi_preview.png)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=PREVIEW_SAMPLES,
        help="Number of sample images per location (default: 3)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else ROI_CONFIG
    save_path = (
        Path(args.output) if args.output else TRAIN_BY_LOCATION / "roi_preview.png"
    )
    n_samples = args.samples

    print("=" * 60)
    print("Road ROI Preview Generator")
    print("=" * 60)

    # Load ROI config
    try:
        roi_map = load_road_roi(config_path)
    except FileNotFoundError:
        print(f"ERROR: ROI config not found: {config_path}")
        print("Run generate_road_roi.py first.")
        return

    print(f"Loaded {len(roi_map)} ROI polygon(s) from {config_path}")

    # Discover locations
    locations = discover_locations()
    n_locs = len(locations)
    if n_locs == 0:
        print("No location folders found.")
        return

    print(f"Found {n_locs} locations")

    # Build grid: one row per location, n_samples columns
    fig, axes = plt.subplots(
        n_locs, n_samples, figsize=(4 * n_samples, 3 * n_locs), squeeze=False
    )
    fig.suptitle("Road ROI Preview (green overlay)", fontsize=14, y=1.01)

    for row, (loc_id, loc_dir) in enumerate(locations):
        loc_name = f"location_{loc_id}"
        polygon = roi_map.get(loc_name, np.empty((0, 2), dtype=np.int32))
        has_roi = len(polygon) >= 3

        # Group frames from images/ subdir
        images_dir = loc_dir / "images"
        frames = group_tiles_by_frame(images_dir)
        sampled_ts = pick_sample_frames(frames, n_samples)

        for col in range(n_samples):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(sampled_ts):
                ts = sampled_ts[col]
                rep = pick_representative(frames[ts])
                try:
                    img = np.array(Image.open(rep["path"]).convert("RGB"))
                    if has_roi:
                        img = overlay_roi(img, polygon)
                    ax.imshow(img)
                except Exception:
                    ax.text(
                        0.5,
                        0.5,
                        "ERR",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                period = time_period(ts)
                ax.set_title(f"{ts} ({period})", fontsize=7)
            else:
                ax.set_visible(False)

        status = f"{len(polygon)} pts" if has_roi else "NO ROI"
        axes[row][0].set_ylabel(
            f"{loc_name}\n({status})", fontsize=9, rotation=0, labelpad=80, va="center"
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
