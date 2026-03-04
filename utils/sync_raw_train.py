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

from common import RAW_TRAIN_PATH, TRAIN_BY_LOCATION_PATH


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

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    if not TRAIN_BY_LOCATION_PATH.exists():
        logger.error("Source directory not found: %s", TRAIN_BY_LOCATION_PATH)
        return

    if not RAW_TRAIN_PATH.exists():
        logger.error("Target directory not found: %s", RAW_TRAIN_PATH)
        return

    logger.info("Scanning %s...", TRAIN_BY_LOCATION_PATH)

    valid_filenames = set()

    for loc_dir in TRAIN_BY_LOCATION_PATH.iterdir():
        if loc_dir.is_dir() and loc_dir.name.startswith("location_"):
            img_dir = loc_dir / "images"
            if not img_dir.is_dir():
                img_dir = loc_dir

            for img_path in img_dir.glob("*"):
                if img_path.is_file() and img_path.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                ]:
                    valid_filenames.add(img_path.name)

    if not valid_filenames:
        logger.warning(
            "No images found in train_by_location! Aborting to prevent deleting all files."
        )
        return

    logger.info("Found %d valid images in train_by_location.", len(valid_filenames))

    logger.info("Scanning %s...", RAW_TRAIN_PATH)
    deleted_count = 0
    kept_count = 0

    for img_path in RAW_TRAIN_PATH.glob("*"):
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            if img_path.name not in valid_filenames:
                if args.dry_run:
                    logger.info("[DRY RUN] Would delete: %s", img_path.name)
                else:
                    img_path.unlink()
                    logger.info("Deleted: %s", img_path.name)
                deleted_count += 1
            else:
                kept_count += 1

    logger.info(
        "\nSync complete. Deleted %d files. Kept %d files.", deleted_count, kept_count
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    main()
