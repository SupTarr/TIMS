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

from common import (
    IMAGE_EXTENSIONS,
    RAW_TRAIN_PATH,
    TRAIN_BY_LOCATION_PATH,
    setup_logging,
)


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

    setup_logging()
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
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    valid_filenames.add(img_path.name)

    if not valid_filenames:
        logger.warning(
            "No images found in train_by_location! Aborting to prevent deleting all files."
        )
        return

    logger.info("Found %d valid images in train_by_location.", len(valid_filenames))

    logger.info("Scanning %s...", RAW_TRAIN_PATH)

    files_to_delete = []
    kept_count = 0

    for img_path in RAW_TRAIN_PATH.glob("*"):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            if img_path.name not in valid_filenames:
                files_to_delete.append(img_path)
            else:
                kept_count += 1

    deleted_count = 0

    if files_to_delete:
        if args.dry_run:
            for img_path in files_to_delete:
                logger.info("[DRY RUN] Would delete: %s", img_path.name)
            deleted_count = len(files_to_delete)
        else:
            print(
                f"\n[WARNING] You are about to delete {len(files_to_delete)} files from raw/train."
            )
            ans = input("Do you want to proceed? [y/N]: ").strip().lower()

            if ans == "y":
                for img_path in files_to_delete:
                    img_path.unlink()
                    logger.info("Deleted: %s", img_path.name)
                deleted_count = len(files_to_delete)
            else:
                logger.info("Aborted deletion. No files were deleted.")
                deleted_count = 0
    else:
        logger.info("No files need to be deleted.")

    logger.info(
        "\nSync complete. Deleted %d files. Kept %d files.", deleted_count, kept_count
    )


if __name__ == "__main__":
    main()
