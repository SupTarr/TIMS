"""
Organize train_by_location folders into /images and /labels subdirectories.

For each location_N folder:
  - Creates /images and /labels subdirectories
  - Moves .jpg files into /images
  - Copies matching .txt labels from TIMS_dataset_final/train_original/labels into /labels
"""

import logging
import shutil

from common import TIMS_FINAL_LABELS_PATH, TRAIN_BY_LOCATION_PATH, discover_locations

logger = logging.getLogger(__name__)


def organize():
    assert TRAIN_BY_LOCATION_PATH.is_dir(), f"Not found: {TRAIN_BY_LOCATION_PATH}"
    assert TIMS_FINAL_LABELS_PATH.is_dir(), f"Not found: {TIMS_FINAL_LABELS_PATH}"

    location_dirs = [(lid, ld) for lid, ld in discover_locations()]

    logger.info("Found %d location folders\n", len(location_dirs))
    total_moved = 0
    total_copied = 0
    total_missing = 0

    for _loc_id, loc_dir in location_dirs:
        images_dir = loc_dir / "images"
        labels_dir = loc_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        jpg_files = sorted(
            f for f in loc_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"
        )

        moved = 0
        copied = 0
        missing = 0

        for jpg in jpg_files:
            dest_img = images_dir / jpg.name
            shutil.move(str(jpg), str(dest_img))
            moved += 1

            label_name = jpg.stem + ".txt"
            src_label = TIMS_FINAL_LABELS_PATH / label_name
            if src_label.exists():
                shutil.copy2(str(src_label), str(labels_dir / label_name))
                copied += 1
            else:
                missing += 1
                logger.warning("  No label for %s", jpg.name)

        image_stems = {p.stem for p in images_dir.iterdir() if p.is_file()}
        orphans = 0
        for label_file in list(labels_dir.glob("*.txt")):
            if label_file.stem not in image_stems:
                label_file.unlink()
                orphans += 1

        logger.info(
            "%s: %d images moved, %d labels copied, %d missing labels, %d orphan labels removed",
            loc_dir.name,
            moved,
            copied,
            missing,
            orphans,
        )
        total_moved += moved
        total_copied += copied
        total_missing += missing

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "TOTAL: %d images moved, %d labels copied, %d missing labels",
        total_moved,
        total_copied,
        total_missing,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    organize()
