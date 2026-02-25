#!/usr/bin/env python3
"""
Sync raw/train with raw/train_by_location.

Deletes images in raw/train that are not present in any location folder
inside raw/train_by_location. This is useful if you have manually cleaned
up clusters in train_by_location and want to reflect those deletions in
the master raw/train folder.

Usage:
    python utils/sync_raw_train.py
    python utils/sync_raw_train.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    from common import BASE_DIR
except ImportError:
    BASE_DIR = (
        Path(__file__).resolve().parent.parent
        / "gdrive"
        / "YOLOv10"
        / "data_train"
        / "TIMS_density_dataset"
    )

TRAIN_BY_LOCATION = BASE_DIR / "raw" / "train_by_location"
RAW_TRAIN = BASE_DIR / "raw" / "train"


def main():
    parser = argparse.ArgumentParser(
        description="Delete images in raw/train not found in raw/train_by_location"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    logger = logging.getLogger(__name__)

    if not TRAIN_BY_LOCATION.exists():
        logger.error(f"Source directory not found: {TRAIN_BY_LOCATION}")
        return

    if not RAW_TRAIN.exists():
        logger.error(f"Target directory not found: {RAW_TRAIN}")
        return

    logger.info(f"Scanning {TRAIN_BY_LOCATION}...")

    valid_filenames = set()

    for loc_dir in TRAIN_BY_LOCATION.iterdir():
        if loc_dir.is_dir() and loc_dir.name.startswith("location_"):
            img_dir = loc_dir / "images"
            if not img_dir.is_dir():
                img_dir = loc_dir

            for img_path in img_dir.glob("*"):
                if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    valid_filenames.add(img_path.name)

    if not valid_filenames:
        logger.warning("No images found in train_by_location! Aborting to prevent deleting all files.")
        return

    logger.info(f"Found {len(valid_filenames)} valid images in train_by_location.")

    logger.info(f"Scanning {RAW_TRAIN}...")
    deleted_count = 0
    kept_count = 0

    for img_path in RAW_TRAIN.glob("*"):
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            if img_path.name not in valid_filenames:
                if args.dry_run:
                    logger.info(f"[DRY RUN] Would delete: {img_path.name}")
                else:
                    img_path.unlink()
                    logger.info(f"Deleted: {img_path.name}")
                deleted_count += 1
            else:
                kept_count += 1

    logger.info(f"\nSync complete. Deleted {deleted_count} files. Kept {kept_count} files.")

if __name__ == "__main__":
    main()
