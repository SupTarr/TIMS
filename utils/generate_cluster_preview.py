#!/usr/bin/env python3
"""
Generate cluster_preview.png from existing location folders.

Reads from train_by_location/location_0 .. location_N and produces a
preview grid matching the style of cluster_by_location.py's generate_preview().

No heavy dependencies (no CLIP, torch, sklearn).
Requires: matplotlib, Pillow, numpy.

Usage:
    python generate_cluster_preview.py
    python generate_cluster_preview.py --samples 8   # more columns
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import (
    BASE_DIR as _TIMS_BASE,
    group_tiles_by_frame,
    pick_representative,
    time_period,
    discover_locations,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = _TIMS_BASE / "raw" / "train_by_location"

PREVIEW_SAMPLES = 5


# ──────────────────────────────────────────────────────────────────────
# Helpers (location folder discovery + sampling)
# ──────────────────────────────────────────────────────────────────────
def pick_sample_frames(frames: dict[str, list[dict]], n: int) -> list[str]:
    """Pick n timestamps evenly spaced across sorted timestamps."""
    ts_sorted = sorted(frames.keys())
    if len(ts_sorted) <= n:
        return ts_sorted
    indices = np.linspace(0, len(ts_sorted) - 1, n, dtype=int)
    return [ts_sorted[i] for i in indices]


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
    args = parser.parse_args()

    save_path = Path(args.output) if args.output else BASE_DIR / "cluster_preview.png"
    n_samples = args.samples

    print("=" * 60)
    print("Generate Cluster Preview from Location Folders")
    print("=" * 60)
    print(f"Base dir: {BASE_DIR}")

    if not BASE_DIR.exists():
        print(f"ERROR: Base directory not found: {BASE_DIR}")
        return

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
        frames = group_tiles_by_frame(loc_dir)
        n_frames = len(frames)
        n_tiles = sum(len(v) for v in frames.values())
        loc_data[loc_id] = {"frames": frames, "n_frames": n_frames, "n_tiles": n_tiles}
        print(f"  location_{loc_id}: {n_frames} frames ({n_tiles} tiles)")

    print(f"\nGenerating preview grid ({n_clusters} rows x {n_samples} cols)...")

    fig, axes = plt.subplots(
        n_clusters, n_samples, figsize=(3 * n_samples, 3 * n_clusters), squeeze=False
    )
    fig.suptitle(
        "Cluster Preview (rows = locations) - CLIP + Structural", fontsize=14, y=1.01
    )

    for row_idx, (loc_id, _) in enumerate(locations):
        data = loc_data[loc_id]
        frames = data["frames"]
        n_frames = data["n_frames"]

        sampled_ts = pick_sample_frames(frames, n_samples)

        for col in range(n_samples):
            ax = axes[row_idx][col]
            ax.axis("off")
            if col < len(sampled_ts):
                ts = sampled_ts[col]
                rep = pick_representative(frames[ts])
                try:
                    img = Image.open(rep["path"]).convert("RGB")
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
                ax.set_title(f"loc_{loc_id} | {ts} ({period})", fontsize=7)
            else:
                ax.set_visible(False)

        axes[row_idx][0].set_ylabel(
            f"location_{loc_id}\n({n_frames} frames)",
            fontsize=9,
            rotation=0,
            labelpad=90,
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
