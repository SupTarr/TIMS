#!/usr/bin/env python3
"""
Classify images by filename pattern (CCTV vs Google) and copy CCTV
images to the raw/train directory.

Usage:
    python classifier.py
"""

import logging
import shutil

from common import (
    CCTV_PATTERN_LOOSE,
    IMAGE_EXTENSIONS,
    RAW_TRAIN_PATH,
    TIMS_FINAL_IMAGES_PATH,
    setup_logging,
)

logger = logging.getLogger(__name__)


def classify_by_filename(filename: str) -> str:
    """
    Classify by filename pattern:
      - UUID + numeric timestamp pattern -> CCTV
      - Anything else -> Google
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    if CCTV_PATTERN_LOOSE.match(stem):
        return "CCTV"
    return "Google"


def main() -> None:
    setup_logging()

    if not TIMS_FINAL_IMAGES_PATH.is_dir():
        logger.error("Source directory not found: %s", TIMS_FINAL_IMAGES_PATH)
        return

    RAW_TRAIN_PATH.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f
        for f in TIMS_FINAL_IMAGES_PATH.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    total = len(image_files)
    logger.info("Found %d images in source directory.", total)

    moved_count = 0
    google_count = 0

    for i, img_path in enumerate(image_files, 1):
        category = classify_by_filename(img_path.name)

        if category == "CCTV":
            shutil.copy2(img_path, RAW_TRAIN_PATH / img_path.name)
            moved_count += 1
            status = "-> Copied to Raw"
        else:
            google_count += 1
            status = "-> Skipped (Google)"

        if i % 500 == 0 or i == total:
            logger.info("[%d/%d] %s: %s %s", i, total, img_path.name, category, status)

    logger.info("")
    logger.info("=" * 50)
    logger.info("Done! Total: %d", total)
    logger.info("  CCTV (copied):    %d", moved_count)
    logger.info("  Google (skipped): %d", google_count)


if __name__ == "__main__":
    main()
