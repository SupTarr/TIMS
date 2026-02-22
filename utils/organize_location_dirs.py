"""
Organize train_by_location folders into /images and /labels subdirectories.

For each location_N folder:
  - Creates /images and /labels subdirectories
  - Moves .jpg files into /images
  - Copies matching .txt labels from TIMS_dataset_final/train_original/labels into /labels
"""

import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TRAIN_BY_LOCATION = (
    BASE
    / "gdrive"
    / "YOLOv10"
    / "data_train"
    / "TIMS_density_dataset"
    / "raw"
    / "train_by_location"
)
LABELS_SOURCE = (
    BASE
    / "gdrive"
    / "YOLOv10"
    / "data_train"
    / "TIMS_dataset_final"
    / "train_original"
    / "labels"
)


def organize():
    assert TRAIN_BY_LOCATION.is_dir(), f"Not found: {TRAIN_BY_LOCATION}"
    assert LABELS_SOURCE.is_dir(), f"Not found: {LABELS_SOURCE}"

    location_dirs = sorted(
        [
            d
            for d in TRAIN_BY_LOCATION.iterdir()
            if d.is_dir() and d.name.startswith("location_")
        ],
        key=lambda p: int(p.name.split("_")[1]),
    )

    print(f"Found {len(location_dirs)} location folders\n")
    total_moved = 0
    total_copied = 0
    total_missing = 0

    for loc_dir in location_dirs:
        images_dir = loc_dir / "images"
        labels_dir = loc_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        jpg_files = sorted(
            [f for f in loc_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"]
        )

        moved = 0
        copied = 0
        missing = 0

        for jpg in jpg_files:
            dest_img = images_dir / jpg.name
            shutil.move(str(jpg), str(dest_img))
            moved += 1

            label_name = jpg.stem + ".txt"
            src_label = LABELS_SOURCE / label_name
            if src_label.exists():
                shutil.copy2(str(src_label), str(labels_dir / label_name))
                copied += 1
            else:
                missing += 1
                print(f"  WARNING: No label for {jpg.name}")

        print(
            f"{loc_dir.name}: {moved} images moved, {copied} labels copied, {missing} missing labels"
        )
        total_moved += moved
        total_copied += copied
        total_missing += missing

    print(f"\n{'='*60}")
    print(
        f"TOTAL: {total_moved} images moved, {total_copied} labels copied, {total_missing} missing labels"
    )


if __name__ == "__main__":
    organize()
