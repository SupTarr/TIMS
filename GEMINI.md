# TIMS (Traffic Image Management System)

## Project Overview

TIMS is a sophisticated computer vision and data engineering pipeline designed to process raw CCTV traffic images. Its primary objective is to build, organize, and analyze a high-quality dataset for vehicle detection, traffic density classification, and perspective-aware lane capacity estimation.

The system handles challenges such as mixed day/night/infrared imagery and varied camera perspectives. It leverages:

- **CLIP (ViT-B/32):** Combined with structural feature extraction (edge orientation histograms, spatial edge density) to cluster and match images by their physical camera location.
- **YOLOv10:** Used for precise vehicle detection bounding boxes and lane segmentation.
- **Homography & BEV (Bird's-Eye View):** Perspective transforms to normalize road areas, allowing for accurate physical vehicle density calculations.
- **KDE & GMM:** Advanced statistical models (Kernel Density Estimation and Gaussian Mixture Models) to automatically estimate the number of lanes and cars per lane.

## Architecture & Key Components

The core logic resides in the `workflow/` directory:

- `workflow/cluster_by_location.py`: Clusters raw CCTV images into location-specific folders using CLIP and structural features.
- `workflow/generate_road_roi.py`: A semi-automatic tool with an OpenCV GUI to annotate and generate road Region of Interest (ROI) polygons based on YOLO detection heatmaps.
- `workflow/classify_density.py`: Classifies images into `light`, `medium`, `high`, or `full` traffic density by analyzing vehicles inside the road ROI and computing a density ratio.
- `workflow/assign_valid_to_location.py`: Maps validation and test sets back to their nearest training location centroids.
- `workflow/utils/`: Contains deep mathematical and computer vision utilities, such as:
  - `bev_transform.py`: Homography matrix calculation for Bird's-Eye View transforms.
  - `lane_estimation.py`, `lane_gmm.py`, `lane_kde.py`: Lane count and capacity estimation logic.
  - `clip_features.py`, `structural_features.py`: Image embedding and feature extraction.

## Building and Running

### Environment Setup

The project heavily relies on Python, PyTorch, OpenCV, and scikit-learn. It is recommended to use the existing `venv-yolo` environment or create a new one.

```bash
# Activate the virtual environment
source venv-yolo/bin/activate

# Install dependencies if needed
pip install -r requirements.txt
```

### Key Workflow Commands

The scripts are designed to be run as modules from the project root:

1. **Clustering Locations:**

   ```bash
   python -m workflow.cluster_by_location
   ```

2. **Generating/Annotating Road ROIs:**

   ```bash
   python -m workflow.generate_road_roi
   ```

3. **Classifying Traffic Density:**

   ```bash
   python -m workflow.classify_density
   ```

4. **Assigning Validation Data:**

   ```bash
   python -m workflow.assign_valid_to_location
   ```

## Development Conventions

- **Path Management:** All absolute and relative dataset paths (often pointing to a `gdrive` directory) are strictly managed centrally in `workflow/common/paths.py`. Avoid hardcoding paths in individual scripts.
- **Logging:** Standardized logging is configured via `workflow.common.logging.setup_logging()`. Do not use standard `print()` statements for tracking script progress.
- **Modularity:** Reusable math, geometry, and model-inference functions should be placed in `workflow/utils/` to prevent circular dependencies.
- **Data Safety:** Tools that operate on the `raw/train` directories are generally designed to be READ-ONLY on the source to prevent destructive overwrites (e.g., `cluster_by_location.py` copies files instead of moving them).
