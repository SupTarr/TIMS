#!/usr/bin/env python3
"""
Validate that images in each location_* folder truly belong to that location.

Uses the same combined CLIP + structural feature pipeline as
cluster_by_location.py to compute per-frame embeddings, then checks
cosine similarity against each location's centroid.  Frames below the
similarity threshold are flagged as potential mismatches.

Output is a diagnostic CSV report — no images are moved or deleted.

Usage:
    python validate_location.py                        # default threshold 0.70
    python validate_location.py --threshold 0.80       # stricter
    python validate_location.py --no-structural        # CLIP-only check
    python validate_location.py --device cpu
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import clip
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from cluster_by_location import (
    CLIP_MODEL,
    EDGE_HIST_BINS,
    PCA_COMPONENTS,
    STRUCTURAL_WEIGHT,
    TILES_PER_FRAME,
    extract_clip_embeddings,
    extract_structural_batch,
    select_device,
)
from common import (
    TRAIN_BY_LOCATION_PATH,
    discover_locations,
    group_tiles_by_frame,
    setup_logging,
)

logger = logging.getLogger(__name__)

REPORT_PATH = TRAIN_BY_LOCATION_PATH / "validation_report.csv"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _frames_from_location(loc_dir: Path) -> dict[str, list[dict]]:
    """Group tiles by frame inside a location's images/ subfolder."""
    images_dir = loc_dir / "images"
    if not images_dir.exists():
        images_dir = loc_dir
    return group_tiles_by_frame(images_dir)


# ──────────────────────────────────────────────────────────────────────
# Core validation
# ──────────────────────────────────────────────────────────────────────
def validate(
    base_dir: Path,
    threshold: float,
    device_pref: str,
    batch_size: int,
    tiles_per_frame: int,
    pca_components: int,
    structural_weight: float,
    use_structural: bool,
) -> list[dict]:
    """
    Validate location consistency across all location_* folders.

    Returns a list of per-frame dicts ready for CSV output.
    """
    locations = discover_locations(base_dir)
    if not locations:
        logger.error("No location_* folders found in %s", base_dir)
        sys.exit(1)

    logger.info("Found %d locations in %s", len(locations), base_dir)

    loc_frames: dict[int, dict[str, list[dict]]] = {}
    loc_ts_lists: dict[int, list[str]] = {}

    for loc_id, loc_dir in locations:
        frames = _frames_from_location(loc_dir)
        if not frames:
            logger.warning("  location_%d: no images found — skipping", loc_id)
            continue
        loc_frames[loc_id] = frames
        loc_ts_lists[loc_id] = sorted(frames.keys())
        n_tiles = sum(len(v) for v in frames.values())
        logger.info("  location_%d: %d frames (%d tiles)", loc_id, len(frames), n_tiles)

    if not loc_frames:
        logger.error("No images found in any location folder")
        sys.exit(1)

    all_ts: list[str] = []
    all_frames: dict[str, list[dict]] = {}
    frame_loc_id: list[int] = []

    for loc_id in sorted(loc_frames):
        for ts in loc_ts_lists[loc_id]:
            all_ts.append(ts)
            all_frames[ts] = loc_frames[loc_id][ts]
            frame_loc_id.append(loc_id)

    n_total = len(all_ts)
    logger.info("Total: %d frames across %d locations", n_total, len(loc_frames))

    device = select_device(device_pref)
    logger.info("Loading CLIP (%s) on %s …", CLIP_MODEL, device)
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    logger.info("Extracting CLIP embeddings …")
    clip_emb = extract_clip_embeddings(
        all_frames, all_ts, model, preprocess, device, batch_size, tiles_per_frame
    )
    clip_emb = normalize(clip_emb)

    if pca_components > 0 and pca_components < clip_emb.shape[1]:
        logger.info("PCA %d → %d dims …", clip_emb.shape[1], pca_components)
        pca = PCA(n_components=pca_components, random_state=42)
        clip_emb = normalize(pca.fit_transform(clip_emb))

    if use_structural:
        logger.info("Extracting structural features …")
        struct_feat = extract_structural_batch(all_frames, all_ts, tiles_per_frame)
        struct_feat = normalize(struct_feat)

        w_clip = 1.0 - structural_weight
        w_struct = structural_weight
        embeddings = np.hstack([clip_emb * w_clip, struct_feat * w_struct])
        embeddings = normalize(embeddings)
        logger.info(
            "Fused shape: %s (CLIP×%.1f + Struct×%.1f)",
            embeddings.shape,
            w_clip,
            w_struct,
        )
    else:
        embeddings = clip_emb

    loc_ids_sorted = sorted(loc_frames.keys())
    centroids: dict[int, np.ndarray] = {}
    for loc_id in loc_ids_sorted:
        indices = [i for i, lid in enumerate(frame_loc_id) if lid == loc_id]
        centroids[loc_id] = embeddings[indices].mean(axis=0, keepdims=True)
    centroid_matrix = np.vstack([centroids[lid] for lid in loc_ids_sorted])
    centroid_matrix = normalize(centroid_matrix)

    sims = cosine_similarity(embeddings, centroid_matrix)

    results: list[dict] = []
    for i, ts in enumerate(all_ts):
        assigned_loc = frame_loc_id[i]
        assigned_idx = loc_ids_sorted.index(assigned_loc)
        sim_to_assigned = float(sims[i, assigned_idx])

        nearest_idx = int(sims[i].argmax())
        nearest_loc = loc_ids_sorted[nearest_idx]
        nearest_sim = float(sims[i, nearest_idx])

        is_mismatch = sim_to_assigned < threshold

        for tile in all_frames[ts]:
            results.append(
                {
                    "filename": tile["path"].name,
                    "assigned_location": f"location_{assigned_loc}",
                    "cosine_similarity": round(sim_to_assigned, 4),
                    "is_mismatch": is_mismatch,
                    "nearest_location": f"location_{nearest_loc}",
                    "nearest_similarity": round(nearest_sim, 4),
                }
            )

    return results


# ──────────────────────────────────────────────────────────────────────
# Report writing & summary
# ──────────────────────────────────────────────────────────────────────
def write_report(results: list[dict], report_path: Path) -> None:
    """Write validation_report.csv."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "assigned_location",
        "cosine_similarity",
        "is_mismatch",
        "nearest_location",
        "nearest_similarity",
    ]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Report saved to %s", report_path)


def print_summary(results: list[dict]) -> None:
    """Print a human-readable summary to the terminal."""
    from collections import Counter

    total = len(results)
    mismatched = [r for r in results if r["is_mismatch"]]
    n_mismatch = len(mismatched)

    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total images checked : %d", total)
    logger.info(
        "Mismatched images    : %d (%.1f%%)",
        n_mismatch,
        100 * n_mismatch / max(total, 1),
    )
    logger.info("Clean images         : %d", total - n_mismatch)

    loc_counts: dict[str, dict] = {}
    for r in results:
        loc = r["assigned_location"]
        if loc not in loc_counts:
            loc_counts[loc] = {"total": 0, "mismatch": 0}
        loc_counts[loc]["total"] += 1
        if r["is_mismatch"]:
            loc_counts[loc]["mismatch"] += 1

    logger.info("-" * 60)
    logger.info("%-15s  %8s  %10s  %s", "Location", "Total", "Mismatched", "Rate")
    logger.info("-" * 60)
    for loc in sorted(loc_counts):
        t = loc_counts[loc]["total"]
        m = loc_counts[loc]["mismatch"]
        rate = 100 * m / max(t, 1)
        flag = "  ⚠" if m > 0 else ""
        logger.info("%-15s  %8d  %10d  %5.1f%%%s", loc, t, m, rate, flag)
    logger.info("-" * 60)

    if mismatched:
        remap = Counter(
            (r["assigned_location"], r["nearest_location"])
            for r in mismatched
            if r["assigned_location"] != r["nearest_location"]
        )
        if remap:
            logger.info("")
            logger.info("Suggested reclassifications:")
            for (src, dst), count in remap.most_common():
                logger.info("  %s → %s : %d images", src, dst, count)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Validate location consistency of clustered CCTV images"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold below which a frame is flagged (default: 0.70)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tiles-per-frame", type=int, default=TILES_PER_FRAME)
    parser.add_argument(
        "--pca", type=int, default=PCA_COMPONENTS, help="PCA components (0 to disable)"
    )
    parser.add_argument("--structural-weight", type=float, default=STRUCTURAL_WEIGHT)
    parser.add_argument(
        "--no-structural",
        action="store_true",
        help="Disable structural features (CLIP-only)",
    )
    parser.add_argument(
        "--base-dir", type=str, default=None, help="Override train_by_location path"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Override report CSV path"
    )
    args = parser.parse_args()
    setup_logging()

    base_dir = Path(args.base_dir) if args.base_dir else TRAIN_BY_LOCATION_PATH
    report_path = Path(args.output) if args.output else REPORT_PATH

    logger.info("=" * 60)
    logger.info("Location Consistency Validation")
    logger.info("(CLIP + Structural Features)")
    logger.info("=" * 60)
    logger.info("Base directory : %s", base_dir)
    logger.info("Threshold      : %.2f", args.threshold)
    logger.info(
        "Structural     : %s",
        "disabled" if args.no_structural else args.structural_weight,
    )
    logger.info("Report output  : %s", report_path)

    results = validate(
        base_dir=base_dir,
        threshold=args.threshold,
        device_pref=args.device,
        batch_size=args.batch_size,
        tiles_per_frame=args.tiles_per_frame,
        pca_components=args.pca,
        structural_weight=args.structural_weight,
        use_structural=not args.no_structural,
    )

    write_report(results, report_path)
    print_summary(results)


if __name__ == "__main__":
    main()
