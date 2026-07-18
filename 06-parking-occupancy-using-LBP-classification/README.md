# Parking Lot Occupancy Recognition Using LBP

## Project overview

This project solves **parking lot occupancy recognition** using a supervised
pipeline based on **Local Binary Patterns (LBP)**.

The implementation takes a full parking-lot image, extracts each parking space
as an independent ROI, preprocesses the ROI, computes an LBP-based texture
descriptor, classifies the ROI as **free** or **full**, and evaluates the
predictions against ground-truth labels.

The project is organized as a modular pipeline so that each stage is clearly
separated:

1. dataset loading
2. parking-space ROI extraction
3. preprocessing
4. LBP feature extraction
5. classifier training and prediction
6. evaluation
7. multi-configuration experiment search
8. output saving and best-configuration inspection

---

## Main idea of the pipeline

The pipeline follows this logic:

### Training side
- load cropped parking-space training samples from:
  - `data/training/free`
  - `data/training/full`
- convert them into a unified record structure
- preprocess each sample
- compute LBP descriptors
- build the training matrix `X_train` and label vector `y_train`
- train the selected classifier

### Test side
- load the parking map from `data/parking_map_python.txt`
- load full parking-lot test images from `data/test_images_zao`
- for each test image:
  - extract one ROI for each parking space using perspective transform
  - preprocess each ROI
  - compute LBP descriptors
  - build the test matrix `X_test`
  - predict labels and optional scores
  - evaluate predictions against the matching `testX.txt` file

### Experiment side
- generate combinations of:
  - preprocessing configurations
  - LBP configurations
  - classifier configurations
- run one full experiment for each combination
- compute:
  - accuracy
  - confusion counts
  - processing times
- rank all experiments
- save:
  - CSV results table
  - text summary of top results

---

## Class labels

The whole project uses the same binary label convention:

- `free = 0`
- `full = 1`

In evaluation terminology:

- `occupied_label = 1`
- `empty_label = 0`

---

## Project structure

```text
project_root/
│
├── main.py
├── parking_io.py
├── parking_training_io.py
├── geometry.py
├── roi_extraction.py
├── preprocessing.py
├── lbp_features.py
├── parking_lbp_dataset.py
├── parking_lbp_classifier.py
├── evaluation.py
├── experiment_search.py
├── results_io.py
├── inspect_best_config.py
├── debug_utils.py
│
├── data/
│   ├── parking_map_python.txt
│   ├── training/
│   │   ├── free/
│   │   └── full/
│   └── test_images_zao/
│       ├── test1.jpg
│       ├── test1.txt
│       ├── test2.jpg
│       ├── test2.txt
│       └── ...
│
└── outputs/
    ├── results/
    │   └── final_run/
    │       ├── parking_lbp_results.csv
    │       └── parking_lbp_summary.txt
    │
    └── inspection/
        └── best_config/
            ├── overlay/
            ├── roi/
            ├── processed/
            ├── lbp/
            └── report/

Default experiment grid

The current implementation uses the following experiment search space.

Preprocessing configurations
no contrast normalization, no filtering
CLAHE + Gaussian blur
LBP configurations
neighbors=8, radius=1, method="uniform", grid_shape=(2, 2)
neighbors=8, radius=1, method="uniform", grid_shape=(4, 4)
neighbors=16, radius=2, method="uniform", grid_shape=(4, 4)
Classifier configurations
knn, n_neighbors=3
linear_svm, C=1.0

So the total number of experiments is: 
2 preprocessing configs
× 3 LBP configs
× 2 classifier configs
= 12 experiments
