#!/usr/bin/env python3
"""
Assign validation/test images to training location clusters using CLIP + structural features.

Computes centroids from existing train_by_location/ clusters, then assigns
each image from raw/valid/ or raw/test/ to its nearest location centroid via
cosine similarity.

Output mirrors train_by_location/ structure:
    raw/valid_by_location/location_N/images/
    raw/valid_by_location/location_N/labels/
    raw/test_by_location/location_N/images/
    raw/test_by_location/location_N/labels/

Usage:
    python -m workflow.assign_valid_to_location                      # assign valid
    python -m workflow.assign_valid_to_location --split test          # assign test
    python -m workflow.assign_valid_to_location --split both          # assign both
    python -m workflow.assign_valid_to_location --dry-run             # preview only
    python -m workflow.assign_valid_to_location --device cpu
    python -m workflow.assign_valid_to_location --no-structural
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import clip
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .common import (
    PCA_COMPONENTS,
    RAW_TEST_PATH,
    RAW_VALID_PATH,
    STRUCTURAL_WEIGHT,
    TEST_BY_LOCATION_PATH,
    TIMS_FINAL_TEST_LABELS_PATH,
    TIMS_FINAL_VALID_LABELS_PATH,
    TRAIN_BY_LOCATION_PATH,
    VALID_BY_LOCATION_PATH,
    discover_locations,
    group_tiles_by_frame,
    setup_logging,
)
from .utils.clip_features import (
    CLIP_MODEL,
    TILES_PER_FRAME,
    extract_clip_embeddings,
    select_device,
)
from .utils.structural_features import extract_structural_batch

logger = logging.getLogger(__name__)

SPLIT_CONFIG: dict[str, tuple[Path, Path, Path]] = {
    "valid": (RAW_VALID_PATH, VALID_BY_LOCATION_PATH, TIMS_FINAL_VALID_LABELS_PATH),
    "test": (RAW_TEST_PATH, TEST_BY_LOCATION_PATH, TIMS_FINAL_TEST_LABELS_PATH),
}


# ──────────────────────────────────────────────────────────────────────
# Core assignment
# ──────────────────────────────────────────────────────────────────────
def _frames_from_location(loc_dir: Path) -> dict[str, list[dict]]:
    """Group tiles by frame inside a location's images/ subfolder."""
    images_dir = loc_dir / "images"
    if not images_dir.exists():
        images_dir = loc_dir
    return group_tiles_by_frame(images_dir)


def assign(
    split: str = "valid",
    dry_run: bool = False,
    device_pref: str = "auto",
    batch_size: int = 32,
    tiles_per_frame: int = TILES_PER_FRAME,
    pca_components: int = PCA_COMPONENTS,
    structural_weight: float = STRUCTURAL_WEIGHT,
    use_structural: bool = True,
) -> None:
    """Assign images from one or more splits to training location clusters."""
    splits = ["valid", "test"] if split == "both" else [split]

    for split_name in splits:
        cfg = SPLIT_CONFIG[split_name]
        _assign_split(
            split_name=split_name,
            raw_source=cfg[0],
            output_dir=cfg[1],
            labels_source=cfg[2],
            dry_run=dry_run,
            device_pref=device_pref,
            batch_size=batch_size,
            tiles_per_frame=tiles_per_frame,
            pca_components=pca_components,
            structural_weight=structural_weight,
            use_structural=use_structural,
        )


def _assign_split(
    split_name: str,
    raw_source: Path,
    output_dir: Path,
    labels_source: Path,
    dry_run: bool = False,
    device_pref: str = "auto",
    batch_size: int = 32,
    tiles_per_frame: int = TILES_PER_FRAME,
    pca_components: int = PCA_COMPONENTS,
    structural_weight: float = STRUCTURAL_WEIGHT,
    use_structural: bool = True,
) -> None:
    """Assign images from a single split to training location clusters."""
    assert TRAIN_BY_LOCATION_PATH.is_dir(), f"Not found: {TRAIN_BY_LOCATION_PATH}"
    assert raw_source.is_dir(), f"Not found: {raw_source}"

    train_locations = discover_locations(TRAIN_BY_LOCATION_PATH)
    if not train_locations:
        logger.error("No location folders found in %s", TRAIN_BY_LOCATION_PATH)
        sys.exit(1)

    logger.info("Found %d training locations", len(train_locations))

    train_loc_frames: dict[int, dict[str, list[dict]]] = {}
    train_loc_keys: dict[int, list[str]] = {}

    for loc_id, loc_dir in train_locations:
        frames = _frames_from_location(loc_dir)
        if not frames:
            logger.warning("  location_%d: no images — skipping", loc_id)
            continue
        train_loc_frames[loc_id] = frames
        train_loc_keys[loc_id] = sorted(frames.keys())
        n_tiles = sum(len(v) for v in frames.values())
        logger.info("  location_%d: %d frames (%d tiles)", loc_id, len(frames), n_tiles)

    if not train_loc_frames:
        logger.error("No images found in any training location folder")
        sys.exit(1)

    all_train_keys: list[str] = []
    all_train_frames: dict[str, list[dict]] = {}
    frame_loc_id: list[int] = []

    for loc_id in sorted(train_loc_frames):
        for fk in train_loc_keys[loc_id]:
            key = f"train_{loc_id}_{fk}"
            all_train_keys.append(key)
            all_train_frames[key] = train_loc_frames[loc_id][fk]
            frame_loc_id.append(loc_id)

    target_frames = group_tiles_by_frame(raw_source)
    if not target_frames:
        logger.error("No %s images found in %s", split_name, raw_source)
        sys.exit(1)

    target_keys = sorted(target_frames.keys())
    n_target_tiles = sum(len(v) for v in target_frames.values())
    logger.info(
        "[%s] %d frames (%d tiles)", split_name, len(target_keys), n_target_tiles
    )

    all_target_keys: list[str] = [f"{split_name}_{vk}" for vk in target_keys]
    all_target_frames: dict[str, list[dict]] = {
        f"{split_name}_{vk}": target_frames[vk] for vk in target_keys
    }

    device = select_device(device_pref)
    logger.info("Loading CLIP (%s) on %s …", CLIP_MODEL, device)
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    combined_frames = {**all_train_frames, **all_target_frames}
    combined_keys = all_train_keys + all_target_keys

    logger.info("Extracting CLIP embeddings for %d frames …", len(combined_keys))
    clip_emb = extract_clip_embeddings(
        combined_frames,
        combined_keys,
        model,
        preprocess,
        device,
        batch_size,
        tiles_per_frame,
    )
    clip_emb = normalize(clip_emb)

    n_train = len(all_train_keys)
    train_clip = clip_emb[:n_train]
    target_clip = clip_emb[n_train:]

    n_components = min(pca_components, train_clip.shape[0], train_clip.shape[1])
    if n_components > 0 and n_components < train_clip.shape[1]:
        logger.info(
            "PCA %d → %d dims (fitted on training) …", train_clip.shape[1], n_components
        )
        pca = PCA(n_components=n_components, random_state=42)
        train_clip = normalize(pca.fit_transform(train_clip))
        target_clip = normalize(pca.transform(target_clip))

    if use_structural:
        logger.info("Extracting structural features (training) …")
        train_struct = extract_structural_batch(
            all_train_frames, all_train_keys, tiles_per_frame
        )
        train_struct = normalize(train_struct)

        logger.info("Extracting structural features (%s) …", split_name)
        target_struct = extract_structural_batch(
            all_target_frames, all_target_keys, tiles_per_frame
        )
        target_struct = normalize(target_struct)

        w_clip = 1.0 - structural_weight
        w_struct = structural_weight
        train_emb = normalize(np.hstack([train_clip * w_clip, train_struct * w_struct]))
        target_emb = normalize(
            np.hstack([target_clip * w_clip, target_struct * w_struct])
        )
        logger.info(
            "Fused shape: train=%s, %s=%s (CLIP×%.1f + Struct×%.1f)",
            train_emb.shape,
            split_name,
            target_emb.shape,
            w_clip,
            w_struct,
        )
    else:
        train_emb = train_clip
        target_emb = target_clip

    loc_ids_sorted = sorted(train_loc_frames.keys())
    centroids: list[np.ndarray] = []

    for loc_id in loc_ids_sorted:
        mask = [i for i, lid in enumerate(frame_loc_id) if lid == loc_id]
        centroid = train_emb[mask].mean(axis=0)
        centroids.append(centroid)

    centroid_matrix = normalize(np.array(centroids))
    logger.info("Computed %d location centroids", len(centroids))

    similarities = cosine_similarity(target_emb, centroid_matrix)
    assignments = similarities.argmax(axis=1)
    best_sims = similarities.max(axis=1)

    assignment_map: dict[str, int] = {}
    for i, vk in enumerate(target_keys):
        assignment_map[vk] = loc_ids_sorted[assignments[i]]

    loc_counts: dict[int, int] = {lid: 0 for lid in loc_ids_sorted}
    for lid in assignment_map.values():
        loc_counts[lid] += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("[%s] Assignment Summary", split_name)
    logger.info("=" * 60)
    for lid in loc_ids_sorted:
        n_train_frames = len(train_loc_keys.get(lid, []))
        logger.info(
            "  location_%d: %d %s frames (train: %d frames) " "avg_sim=%.3f",
            lid,
            loc_counts[lid],
            split_name,
            n_train_frames,
            (
                np.mean(best_sims[assignments == loc_ids_sorted.index(lid)])
                if loc_counts[lid] > 0
                else 0.0
            ),
        )
    logger.info("  TOTAL: %d %s frames assigned", len(target_keys), split_name)
    logger.info(
        "  Overall avg cosine similarity: %.3f (min=%.3f, max=%.3f)",
        best_sims.mean(),
        best_sims.min(),
        best_sims.max(),
    )
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN — no files copied.")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_labels = labels_source.is_dir()
    if not has_labels:
        logger.warning(
            "%s labels not found at %s — skipping label copy", split_name, labels_source
        )

    total_copied = 0
    total_labels = 0
    total_missing_labels = 0

    for vk in target_keys:
        loc_id = assignment_map[vk]
        loc_dir = output_dir / f"location_{loc_id}"
        images_dir = loc_dir / "images"
        labels_dir = loc_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for tile in target_frames[vk]:
            shutil.copy2(tile["path"], images_dir / tile["path"].name)
            total_copied += 1

            if has_labels:
                label_name = tile["path"].stem + ".txt"
                src_label = labels_source / label_name
                if src_label.exists():
                    shutil.copy2(str(src_label), str(labels_dir / label_name))
                    total_labels += 1
                else:
                    total_missing_labels += 1

    logger.info(
        "Copied %d images, %d labels (%d missing labels) to %s",
        total_copied,
        total_labels,
        total_missing_labels,
        output_dir,
    )

    csv_path = output_dir / "assignment_mapping.csv"
    _write_assignment_csv(
        target_keys, target_frames, assignment_map, best_sims, csv_path
    )


def _write_assignment_csv(
    target_keys: list[str],
    target_frames: dict[str, list[dict]],
    assignment_map: dict[str, int],
    best_sims: np.ndarray,
    csv_path: Path,
) -> None:
    """Write per-tile assignment mapping to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "frame_key", "assigned_location", "cosine_similarity"]
        )
        for i, vk in enumerate(target_keys):
            loc_id = assignment_map[vk]
            sim = float(best_sims[i])
            for tile in target_frames[vk]:
                writer.writerow(
                    [tile["path"].name, vk, f"location_{loc_id}", f"{sim:.4f}"]
                )
    logger.info("Assignment CSV saved to %s", csv_path)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign validation/test images to training location clusters via CLIP + structural features."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["valid", "test", "both"],
        help="Which split to assign: valid, test, or both (default: valid).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show assignment statistics without copying files.",
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
        help="Disable structural features (CLIP-only).",
    )
    args = parser.parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("Assign Images to Training Locations")
    logger.info("(CLIP + Structural — nearest centroid)")
    logger.info("=" * 60)
    logger.info("Split              : %s", args.split)
    logger.info("Training locations : %s", TRAIN_BY_LOCATION_PATH)
    logger.info(
        "Structural         : %s",
        "disabled" if args.no_structural else args.structural_weight,
    )

    assign(
        split=args.split,
        dry_run=args.dry_run,
        device_pref=args.device,
        batch_size=args.batch_size,
        tiles_per_frame=args.tiles_per_frame,
        pca_components=args.pca,
        structural_weight=args.structural_weight,
        use_structural=not args.no_structural,
    )


if __name__ == "__main__":
    main()
