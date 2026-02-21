import os
import shutil
import glob
from pathlib import Path

from common import BASE_DIR, CCTV_PATTERN_LOOSE

SRC_DIR = str(
    Path(__file__).resolve().parent.parent
    / "gdrive"
    / "YOLOv10"
    / "data_train"
    / "TIMS_dataset_final_qwen"
    / "train_original"
    / "images"
)
DST_DIR = str(BASE_DIR / "raw" / "train")


def classify_by_filename(filename):
    """
    Classify by filename pattern:
      - UUID + numeric timestamp pattern -> CCTV
      - Anything else -> Google
    """
    name = os.path.splitext(filename)[0]
    if CCTV_PATTERN_LOOSE.match(name):
        return "CCTV"
    return "Google"


if __name__ == "__main__":
    os.makedirs(DST_DIR, exist_ok=True)

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(SRC_DIR, ext)))
    image_files.sort()

    total = len(image_files)
    print(f"Found {total} images in source directory.")

    moved_count = 0
    google_count = 0

    for i, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        category = classify_by_filename(filename)

        if category == "CCTV":
            shutil.copy2(img_path, os.path.join(DST_DIR, filename))
            moved_count += 1
            status = "-> Copied to Raw"
        else:
            google_count += 1
            status = "-> Skipped (Google)"

        if i % 500 == 0 or i == total:
            print(f"[{i}/{total}] {filename}: {category} {status}")

    print("\n" + "=" * 50)
    print(f"Done! Total: {total}")
    print(f"  CCTV (coppied):    {moved_count}")
    print(f"  Google (skipped): {google_count}")
