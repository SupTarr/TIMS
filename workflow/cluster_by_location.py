#!/usr/bin/env python3
"""
Cluster CCTV tile images by camera location using CLIP + structural features.

Optimized for mixed day/night/IR imagery:
- CLAHE contrast enhancement for low-light & IR images
- Multi-tile sampling per frame (not just one representative)
- Structural features (edge histograms, resolution) fused with CLIP
- Agglomerative clustering with cosine affinity for better grouping
- Grayscale-normalized structural features to bridge day/night gap

Reads images from raw/train/ (READ-ONLY — never modified).
All writes go to raw/train_by_location/.

Usage:
    python cluster_by_location.py                  # auto-detect best K
    python cluster_by_location.py --n-clusters 5   # force 5 clusters
    python cluster_by_location.py --device cpu     # force CPU
    python cluster_by_location.py --batch-size 16  # smaller batches
    python cluster_by_location.py --tiles-per-frame 5  # more tiles sampled
    python cluster_by_location.py --no-structural  # CLIP-only (skip edge features)
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import clip
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from .utils.clip_features import (
    CLIP_MODEL,
    TILES_PER_FRAME,
    extract_clip_embeddings,
    select_device,
)
from .common import (
    CLUSTER_CSV_PATH,
    CLUSTER_PREVIEW_PATH,
    RAW_TRAIN_PATH as SRC_DIR,
    TRAIN_BY_LOCATION_PATH as DST_DIR,
    detect_frame_modality,
    group_tiles_by_frame,
    pick_representative,
    setup_logging,
)
from .utils.structural_features import EDGE_HIST_BINS, extract_structural_batch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

PCA_COMPONENTS = 50
K_RANGE = range(2, 16)
PREVIEW_SAMPLES = 5
STRUCTURAL_WEIGHT = 0.3


# ──────────────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────────────
def find_best_k(embeddings: np.ndarray, k_range: range) -> int:
    """Sweep Agglomerative clustering over k_range, return K with best silhouette."""
    best_k, best_score = 2, -1.0
    logger.info("  Silhouette sweep K=%d..%d:", k_range.start, k_range.stop - 1)
    for k in k_range:
        if k >= len(embeddings):
            break
        agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = agg.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric="cosine")
        marker = ""
        if score > best_score:
            best_score = score
            best_k = k
            marker = " <-- best so far"
        logger.info("  K=%2d silhouette=%.4f%s", k, score, marker)
    logger.info("  => Best K = %d (silhouette = %.4f)", best_k, best_score)
    return best_k


def cluster_frames(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster using Agglomerative with cosine distance (better for mixed features)."""
    agg = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    return agg.fit_predict(embeddings)


# ──────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────
def generate_preview(
    frames: dict[str, list[dict]],
    ts_to_cluster: dict[str, int],
    n_clusters: int,
    save_path: Path,
    n_samples: int = PREVIEW_SAMPLES,
):
    """Save a grid image: rows = clusters, cols = sample tiles."""
    fig, axes = plt.subplots(
        n_clusters, n_samples, figsize=(3 * n_samples, 3 * n_clusters), squeeze=False
    )
    fig.suptitle(
        "Cluster Preview (rows = locations) - CLIP + Structural", fontsize=14, y=1.01
    )

    for cluster_id in range(n_clusters):
        cluster_ts = [ts for ts, cid in ts_to_cluster.items() if cid == cluster_id]
        sampled = cluster_ts[:n_samples]

        for col in range(n_samples):
            ax = axes[cluster_id][col]
            ax.axis("off")
            if col < len(sampled):
                ts = sampled[col]
                rep = pick_representative(frames[ts])
                img = Image.open(rep["path"]).convert("RGB")
                ax.imshow(img)
                modality = detect_frame_modality(frames[ts])
                ax.set_title(f"loc_{cluster_id} | {ts} ({modality})", fontsize=7)
            else:
                ax.set_visible(False)

        axes[cluster_id][0].set_ylabel(
            f"location_{cluster_id}\n({len(cluster_ts)} frames)",
            fontsize=9,
            rotation=0,
            labelpad=90,
            va="center",
        )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Preview saved to %s", save_path)


def copy_to_location_folders(
    frames: dict[str, list[dict]], ts_to_cluster: dict[str, int], dst_dir: Path
):
    """Copy all tiles into per-location subdirectories. Source is never modified."""
    total_copied = 0
    for ts, cluster_id in ts_to_cluster.items():
        loc_dir = dst_dir / f"location_{cluster_id}"
        loc_dir.mkdir(parents=True, exist_ok=True)
        for tile in frames[ts]:
            shutil.copy2(tile["path"], loc_dir / tile["path"].name)
            total_copied += 1
    return total_copied


def write_csv(
    frames: dict[str, list[dict]], ts_to_cluster: dict[str, int], csv_path: Path
):
    """Write cluster_mapping.csv with full per-tile metadata."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "hex_hash",
                "timestamp",
                "tile_num",
                "cluster_id",
                "hour",
                "is_night",
                "modality",
            ]
        )
        for ts in sorted(ts_to_cluster.keys()):
            cid = ts_to_cluster[ts]
            actual_ts = frames[ts][0]["ts"]
            hour = int(actual_ts[:2])
            modality = detect_frame_modality(frames[ts])
            is_night = modality == "IR" or hour >= 18 or hour < 6
            for tile in frames[ts]:
                writer.writerow(
                    [
                        tile["path"].name,
                        tile["hex"],
                        tile["ts"],
                        tile["tile"],
                        cid,
                        hour,
                        is_night,
                        modality,
                    ]
                )
    logger.info("  Metadata saved to %s", csv_path)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Cluster CCTV images by camera location (CLIP + structural, IR-optimized)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Force number of clusters (default: auto-detect)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--pca",
        type=int,
        default=PCA_COMPONENTS,
        help="PCA components for CLIP embeddings (0 to disable)",
    )
    parser.add_argument(
        "--tiles-per-frame",
        type=int,
        default=TILES_PER_FRAME,
        help="Number of tiles to sample per frame for averaging",
    )
    parser.add_argument(
        "--structural-weight",
        type=float,
        default=STRUCTURAL_WEIGHT,
        help="Weight for structural features vs CLIP (0.0-1.0)",
    )
    parser.add_argument(
        "--no-structural",
        action="store_true",
        help="Disable structural features (CLIP-only mode)",
    )

    args = parser.parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("CCTV Image Clustering by Location")
    logger.info("(CLIP + Structural Features, IR-Optimized)")
    logger.info("=" * 60)
    logger.info("Source (READ-ONLY): %s", SRC_DIR)
    logger.info("Output: %s", DST_DIR)
    logger.info("Tiles/frame: %d", args.tiles_per_frame)
    logger.info(
        "Structural weight: %s",
        "disabled" if args.no_structural else args.structural_weight,
    )
    logger.info("CLAHE: enabled")

    if not SRC_DIR.exists():
        logger.error("Source directory not found: %s", SRC_DIR)
        sys.exit(1)

    src_count_before = len([f for f in SRC_DIR.iterdir() if f.is_file()])
    logger.info("Source file count (before): %d", src_count_before)

    logger.info("[1/8] Grouping tiles by frame (timestamp)...")
    frames = group_tiles_by_frame(SRC_DIR)
    n_frames = len(frames)
    n_tiles = sum(len(v) for v in frames.values())
    logger.info("  %d tiles across %d frames", n_tiles, n_frames)

    ts_list = sorted(frames.keys())
    device = select_device(args.device)
    logger.info("[2/8] Loading CLIP (%s) on %s...", CLIP_MODEL, device)
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    logger.info(
        "[3/8] Extracting CLIP embeddings (CLAHE + %d tiles/frame)...",
        args.tiles_per_frame,
    )
    clip_embeddings = extract_clip_embeddings(
        frames,
        ts_list,
        model,
        preprocess,
        device,
        args.batch_size,
        args.tiles_per_frame,
    )
    clip_embeddings = normalize(clip_embeddings)
    logger.info("  CLIP shape: %s", clip_embeddings.shape)

    n_components = min(args.pca, clip_embeddings.shape[0], clip_embeddings.shape[1])
    if n_components > 0 and n_components < clip_embeddings.shape[1]:
        logger.info(
            "[4/8] PCA reduction %d -> %d dims...",
            clip_embeddings.shape[1],
            n_components,
        )
        pca = PCA(n_components=n_components, random_state=42)
        clip_reduced = pca.fit_transform(clip_embeddings)
        variance = sum(pca.explained_variance_ratio_) * 100
        logger.info("  Explained variance: %.1f%%", variance)
    else:
        logger.info("[4/8] PCA skipped")
        clip_reduced = clip_embeddings

    clip_reduced = normalize(clip_reduced)

    if not args.no_structural:
        logger.info(
            "[5/8] Extracting structural features (edge histograms + spatial density)..."
        )
        struct_features = extract_structural_batch(
            frames, ts_list, args.tiles_per_frame
        )
        struct_features = normalize(struct_features)
        logger.info("  Structural shape: %s", struct_features.shape)

        w_clip = 1.0 - args.structural_weight
        w_struct = args.structural_weight
        embeddings = np.hstack([clip_reduced * w_clip, struct_features * w_struct])
        embeddings = normalize(embeddings)
        logger.info(
            "  Fused shape: %s (CLIP x%.1f + Struct x%.1f)",
            embeddings.shape,
            w_clip,
            w_struct,
        )
    else:
        logger.info("[5/8] Structural features skipped (--no-structural)")
        embeddings = clip_reduced

    if args.n_clusters is not None:
        n_clusters = args.n_clusters
        logger.info("[6/8] Clustering with K=%d (user-specified)...", n_clusters)
    else:
        logger.info("[6/8] Auto-detecting optimal K (Agglomerative + cosine)...")
        n_clusters = find_best_k(embeddings, K_RANGE)

    logger.info("[7/8] Final clustering (Agglomerative, cosine, K=%d)...", n_clusters)
    labels = cluster_frames(embeddings, n_clusters)

    ts_to_cluster = {}
    for i, ts in enumerate(ts_list):
        ts_to_cluster[ts] = int(labels[i])

    logger.info("  Cluster distribution:")
    for cid in range(n_clusters):
        cluster_timestamps = [ts for ts, c in ts_to_cluster.items() if c == cid]
        tile_count = sum(len(frames[ts]) for ts in cluster_timestamps)
        rgb_count = sum(
            1 for ts in cluster_timestamps if detect_frame_modality(frames[ts]) == "RGB"
        )
        ir_count = len(cluster_timestamps) - rgb_count
        logger.info(
            "  location_%d: %d frames (%d tiles) [RGB=%d, IR=%d]",
            cid,
            len(cluster_timestamps),
            tile_count,
            rgb_count,
            ir_count,
        )

    logger.info("[8/8] Generating outputs...")

    if DST_DIR.exists():
        logger.info("  Clearing previous location folders at %s...", DST_DIR)
        for child in sorted(DST_DIR.iterdir()):
            if child.is_dir() and child.name.startswith("location_"):
                shutil.rmtree(child)
            elif child.is_file() and child.suffix.lower() not in {".json", ".csv"}:
                child.unlink()

    total_copied = copy_to_location_folders(frames, ts_to_cluster, DST_DIR)
    logger.info("  Copied %d tiles into %d location folders", total_copied, n_clusters)

    generate_preview(frames, ts_to_cluster, n_clusters, CLUSTER_PREVIEW_PATH)
    write_csv(frames, ts_to_cluster, CLUSTER_CSV_PATH)

    src_count_after = len([f for f in SRC_DIR.iterdir() if f.is_file()])
    logger.info("=" * 60)
    logger.info("Source file count (before): %d", src_count_before)
    logger.info("Source file count (after):  %d", src_count_after)
    if src_count_before == src_count_after:
        logger.info("✓ Source directory UNCHANGED — safe.")
    else:
        logger.warning("✗ Source file count changed! Investigate immediately.")
    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
