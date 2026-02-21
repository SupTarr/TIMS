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
    python cluster_by_location.py                        # auto-detect best K
    python cluster_by_location.py --n-clusters 5         # force 5 clusters
    python cluster_by_location.py --device cpu            # force CPU
    python cluster_by_location.py --batch-size 16         # smaller batches
    python cluster_by_location.py --tiles-per-frame 5     # more tiles sampled
    python cluster_by_location.py --no-structural         # CLIP-only (skip edge features)
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import clip
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from common import (
    BASE_DIR,
    CCTV_PATTERN,
    group_tiles_by_frame,
    parse_filename,
    pick_representative,
    pick_representatives,
    time_period,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
SRC_DIR = BASE_DIR / "raw" / "train"
DST_DIR = BASE_DIR / "raw" / "train_by_location"
PREVIEW_PATH = DST_DIR / "cluster_preview.png"
CSV_PATH = DST_DIR / "cluster_mapping.csv"

CLIP_MODEL = "ViT-B/32"
PCA_COMPONENTS = 50
K_RANGE = range(2, 16)
PREVIEW_SAMPLES = 5
TILES_PER_FRAME = 3  # number of tiles to average per frame
STRUCTURAL_WEIGHT = 0.3  # blend weight for structural features (0..1)
EDGE_HIST_BINS = 36  # bins for Canny edge orientation histogram


def select_device(preferred: str) -> str:
    """Auto-select best available device."""
    if preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def apply_clahe(pil_img: Image.Image) -> Image.Image:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    enhance contrast in IR/low-light images. Works on the L channel of LAB
    color space to preserve hue in color images.
    """
    arr = np.array(pil_img)

    # Grayscale image
    if len(arr.shape) == 2 or (len(arr.shape) == 3 and arr.shape[2] == 1):
        gray = arr if len(arr.shape) == 2 else arr[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

    # Color image: enhance L channel in LAB space
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)


# ──────────────────────────────────────────────────────────────────────
# Structural feature extraction (illumination-invariant)
# ──────────────────────────────────────────────────────────────────────
def extract_structural_features(img_path: Path) -> np.ndarray:
    """
    Extract illumination-invariant structural features from an image:
      1. Edge orientation histogram (Sobel gradients -> orientation bins)
      2. Spatial edge density (4x4 grid, compute edge density per cell)
      3. Resolution fingerprint (width, height, aspect ratio)

    These features capture the *geometry and layout* of the scene, which
    is consistent across day/night/IR for the same camera location.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning("Failed to read image: %s (returning zero vector)", img_path)
        return np.zeros(EDGE_HIST_BINS + 16 + 3)

    # Convert to grayscale for structural analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Apply CLAHE to normalize brightness
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- 1. Edge orientation histogram ---
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx)  # -pi to pi

    # Only consider significant edges (top 30% by magnitude)
    threshold = np.percentile(magnitude, 70)
    mask = magnitude > threshold
    if mask.sum() > 0:
        edge_hist, _ = np.histogram(
            orientation[mask],
            bins=EDGE_HIST_BINS,
            range=(-np.pi, np.pi),
            weights=magnitude[mask],
        )
        edge_hist = edge_hist / (edge_hist.sum() + 1e-8)
    else:
        edge_hist = np.zeros(EDGE_HIST_BINS)

    # --- 2. Spatial edge density (4x4 grid) ---
    edges = cv2.Canny(gray, 50, 150)
    grid_h, grid_w = 4, 4
    cell_h, cell_w = h // grid_h, w // grid_w
    spatial_density = np.zeros(grid_h * grid_w)
    for r in range(grid_h):
        for c in range(grid_w):
            cell = edges[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            spatial_density[r * grid_w + c] = cell.mean() / 255.0

    # --- 3. Resolution fingerprint ---
    max_dim = max(w, h)
    res_feat = np.array([w / max_dim, h / max_dim, w / (h + 1e-8)])

    return np.concatenate([edge_hist, spatial_density, res_feat])


def extract_structural_batch(
    frames: dict[str, list[dict]],
    ts_list: list[str],
    tiles_per_frame: int,
) -> np.ndarray:
    """Extract structural features for each frame (averaged over representative tiles)."""
    all_features = []
    total = len(ts_list)
    for i, ts in enumerate(ts_list):
        reps = pick_representatives(frames[ts], tiles_per_frame)
        feats = [extract_structural_features(r["path"]) for r in reps]
        avg_feat = np.mean(feats, axis=0)
        all_features.append(avg_feat)
        if (i + 1) % 20 == 0 or i + 1 == total:
            print(f"  Structural features: {i + 1}/{total}", end="\r")
    print()
    return np.array(all_features)


# ──────────────────────────────────────────────────────────────────────
# CLIP embedding extraction (with CLAHE + multi-tile averaging)
# ──────────────────────────────────────────────────────────────────────
def extract_clip_embeddings(
    frames: dict[str, list[dict]],
    ts_list: list[str],
    model,
    preprocess,
    device: str,
    batch_size: int = 32,
    tiles_per_frame: int = TILES_PER_FRAME,
) -> np.ndarray:
    """
    Extract CLIP embeddings with improvements for IR accuracy:
      - CLAHE preprocessing to enhance IR/low-light contrast
      - Multi-tile averaging per frame for robustness
      - Horizontal flip augmentation for viewpoint consistency
    """
    # Collect all tiles to process (multiple per frame)
    tile_items = []  # (ts_index, tile_dict)
    for ts_idx, ts in enumerate(ts_list):
        reps = pick_representatives(frames[ts], tiles_per_frame)
        for rep in reps:
            tile_items.append((ts_idx, rep))

    total_tiles = len(tile_items)
    embed_dim = model.visual.output_dim  # derive from model (e.g. 512 for ViT-B/32)
    all_features = np.zeros((total_tiles, embed_dim), dtype=np.float32)

    for i in range(0, total_tiles, batch_size):
        batch = tile_items[i : i + batch_size]
        images_orig = []
        images_flip = []

        skipped = []
        for batch_idx, (_, item) in enumerate(batch):
            try:
                img = Image.open(item["path"]).convert("RGB")
            except Exception as e:
                logger.warning("Failed to open %s: %s (skipping tile)", item["path"], e)
                skipped.append(batch_idx)
                # Append placeholder tensors (will be zeroed out below)
                images_orig.append(preprocess(Image.new("RGB", (224, 224))))
                images_flip.append(preprocess(Image.new("RGB", (224, 224))))
                continue
            img = apply_clahe(img)  # CLAHE enhancement

            images_orig.append(preprocess(img))
            images_flip.append(preprocess(img.transpose(Image.FLIP_LEFT_RIGHT)))

        # Encode original + flipped, average for viewpoint robustness
        orig_tensor = torch.stack(images_orig).to(device)
        flip_tensor = torch.stack(images_flip).to(device)

        with torch.no_grad():
            feat_orig = model.encode_image(orig_tensor).cpu().numpy()
            feat_flip = model.encode_image(flip_tensor).cpu().numpy()

        # Average original and flipped embeddings
        feat_avg = (feat_orig + feat_flip) / 2.0
        # Zero out skipped (corrupt) tiles
        for s in skipped:
            feat_avg[s] = 0.0
        all_features[i : i + len(batch)] = feat_avg

        done = min(i + batch_size, total_tiles)
        print(f"  CLIP: {done}/{total_tiles} tiles (x2 with flip aug)", end="\r")

    print()

    # Average embeddings per frame
    n_frames = len(ts_list)
    frame_embeddings = np.zeros((n_frames, embed_dim), dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.int32)

    for idx, (ts_idx, _) in enumerate(tile_items):
        frame_embeddings[ts_idx] += all_features[idx]
        counts[ts_idx] += 1

    for j in range(n_frames):
        if counts[j] > 0:
            frame_embeddings[j] /= counts[j]

    return frame_embeddings


# ──────────────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────────────
def find_best_k(embeddings: np.ndarray, k_range: range) -> int:
    """Sweep Agglomerative clustering over k_range, return K with best silhouette."""
    best_k, best_score = 2, -1.0
    print(f"  Silhouette sweep K={k_range.start}..{k_range.stop - 1}:")
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
        print(f"    K={k:2d}  silhouette={score:.4f}{marker}")
    print(f"  => Best K = {best_k} (silhouette = {best_score:.4f})")
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
        n_clusters,
        n_samples,
        figsize=(3 * n_samples, 3 * n_clusters),
        squeeze=False,
    )
    fig.suptitle(
        "Cluster Preview (rows = locations) - CLIP + Structural",
        fontsize=14,
        y=1.01,
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
                period = time_period(ts)
                ax.set_title(f"loc_{cluster_id} | {ts} ({period})", fontsize=7)
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
    print(f"  Preview saved to {save_path}")


def copy_to_location_folders(
    frames: dict[str, list[dict]],
    ts_to_cluster: dict[str, int],
    dst_dir: Path,
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
    frames: dict[str, list[dict]],
    ts_to_cluster: dict[str, int],
    csv_path: Path,
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
            ]
        )
        for ts in sorted(ts_to_cluster.keys()):
            cid = ts_to_cluster[ts]
            hour = int(ts[:2])
            is_night = hour < 6 or hour >= 18
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
                    ]
                )
    print(f"  Metadata saved to {csv_path}")


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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
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

    # ── Banner ──
    print("=" * 60)
    print("CCTV Image Clustering by Location")
    print("(CLIP + Structural Features, IR-Optimized)")
    print("=" * 60)
    print(f"\nSource (READ-ONLY): {SRC_DIR}")
    print(f"Output:             {DST_DIR}")
    print(f"Tiles/frame:        {args.tiles_per_frame}")
    print(
        f"Structural weight:  {'disabled' if args.no_structural else args.structural_weight}"
    )
    print(f"CLAHE:              enabled")
    print(f"Flip augmentation:  enabled")

    if not SRC_DIR.exists():
        print(f"\nERROR: Source directory not found: {SRC_DIR}")
        sys.exit(1)

    src_count_before = len([f for f in SRC_DIR.iterdir() if f.is_file()])
    print(f"\nSource file count (before): {src_count_before}")

    print("\n[1/8] Grouping tiles by frame (timestamp)...")
    frames = group_tiles_by_frame(SRC_DIR)
    n_frames = len(frames)
    n_tiles = sum(len(v) for v in frames.values())
    print(f"  {n_tiles} tiles across {n_frames} frames")

    ts_list = sorted(frames.keys())
    device = select_device(args.device)
    print(f"\n[2/8] Loading CLIP ({CLIP_MODEL}) on {device}...")
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    print(
        f"\n[3/8] Extracting CLIP embeddings (CLAHE + {args.tiles_per_frame} tiles/frame + flip)..."
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
    print(f"  CLIP shape: {clip_embeddings.shape}")

    if args.pca > 0 and args.pca < clip_embeddings.shape[1]:
        print(f"\n[4/8] PCA reduction {clip_embeddings.shape[1]} -> {args.pca} dims...")
        pca = PCA(n_components=args.pca, random_state=42)
        clip_reduced = pca.fit_transform(clip_embeddings)
        variance = sum(pca.explained_variance_ratio_) * 100
        print(f"  Explained variance: {variance:.1f}%")
    else:
        print(f"\n[4/8] PCA skipped")
        clip_reduced = clip_embeddings

    clip_reduced = normalize(clip_reduced)

    if not args.no_structural:
        print(
            f"\n[5/8] Extracting structural features (edge histograms + spatial density)..."
        )
        struct_features = extract_structural_batch(
            frames, ts_list, args.tiles_per_frame
        )
        struct_features = normalize(struct_features)
        print(f"  Structural shape: {struct_features.shape}")

        w_clip = 1.0 - args.structural_weight
        w_struct = args.structural_weight
        embeddings = np.hstack(
            [
                clip_reduced * w_clip,
                struct_features * w_struct,
            ]
        )
        embeddings = normalize(embeddings)
        print(
            f"  Fused shape: {embeddings.shape} (CLIP x{w_clip:.1f} + Struct x{w_struct:.1f})"
        )
    else:
        print(f"\n[5/8] Structural features skipped (--no-structural)")
        embeddings = clip_reduced
    
    if args.n_clusters is not None:
        n_clusters = args.n_clusters
        print(f"\n[6/8] Clustering with K={n_clusters} (user-specified)...")
    else:
        print(f"\n[6/8] Auto-detecting optimal K (Agglomerative + cosine)...")
        n_clusters = find_best_k(embeddings, K_RANGE)

    print(f"\n[7/8] Final clustering (Agglomerative, cosine, K={n_clusters})...")
    labels = cluster_frames(embeddings, n_clusters)

    ts_to_cluster = {}
    for i, ts in enumerate(ts_list):
        ts_to_cluster[ts] = int(labels[i])

    print(f"\n  Cluster distribution:")
    for cid in range(n_clusters):
        cluster_timestamps = [ts for ts, c in ts_to_cluster.items() if c == cid]
        tile_count = sum(len(frames[ts]) for ts in cluster_timestamps)
        day_count = sum(1 for ts in cluster_timestamps if 6 <= int(ts[:2]) < 18)
        night_count = len(cluster_timestamps) - day_count
        print(
            f"    location_{cid}: {len(cluster_timestamps)} frames "
            f"({tile_count} tiles) [day={day_count}, night={night_count}]"
        )

    print(f"\n[8/8] Generating outputs...")

    if DST_DIR.exists():
        print(f"  Clearing previous output at {DST_DIR}...")
        shutil.rmtree(DST_DIR)

    total_copied = copy_to_location_folders(frames, ts_to_cluster, DST_DIR)
    print(f"  Copied {total_copied} tiles into {n_clusters} location folders")

    generate_preview(frames, ts_to_cluster, n_clusters, PREVIEW_PATH)
    write_csv(frames, ts_to_cluster, CSV_PATH)

    src_count_after = len([f for f in SRC_DIR.iterdir() if f.is_file()])
    print(f"\n{'=' * 60}")
    print(f"Source file count (before): {src_count_before}")
    print(f"Source file count (after):  {src_count_after}")
    if src_count_before == src_count_after:
        print("✓ Source directory UNCHANGED — safe.")
    else:
        print("✗ WARNING: Source file count changed! Investigate immediately.")
    print(f"{'=' * 60}")
    print("Done!")


if __name__ == "__main__":
    main()
