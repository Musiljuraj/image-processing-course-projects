# Parking Occupancy Detection

This project solves a parking-space occupancy assignment using a classical image-processing pipeline.  
Its purpose is to determine whether each parking place in a fixed parking-lot camera view is **occupied** or **empty**.

The solution works by:
1. loading the parking-space map
2. extracting each parking place as its own ROI
3. preprocessing the ROI
4. running edge detection
5. computing edge statistics
6. classifying occupancy using edge ratio
7. evaluating results against ground truth
8. searching for the best parameter combination automatically

---

## Project structure

```text
04_ParkingOccupancy/
├── main.py
├── inspect_best_config.py
├── parking_io.py
├── geometry.py
├── roi_extraction.py
├── preprocessing.py
├── edge_detection.py
├── evaluation.py
├── experiment_search.py
├── results_io.py
├── debug_utils.py
├── data/
│   ├── parking_map_python.txt
│   └── test_images_zao/
│       ├── test1.jpg
│       ├── test1.txt
│       ├── test2.jpg
│       ├── test2.txt
│       └── ...
└── outputs/

Main files
main.py: Runs the exhaustive search over parameter combinations and saves ranked results.

inspect_best_config.py: Loads the best configuration from the search results and applies it to one selected image for visual inspection.

parking_io.py: Loads the parking map, test images, and ground-truth labels.

geometry.py: Handles point ordering and perspective transform.

roi_extraction.py: Extracts parking-space ROI patches.

preprocessing.py: Converts ROIs to grayscale and applies filtering.

edge_detection.py: Runs Sobel or Canny and computes edge statistics.

evaluation.py: Classifies parking occupancy and computes TP / TN / FP / FN / accuracy.

experiment_search.py: Builds valid experiment combinations and evaluates them.

results_io.py: Saves and loads ranked experiment results.

debug_utils.py: Saves visual debug outputs.


How to use the project

1. Go to the project root
cd ~/3rls/ZAO/DU/assignments_Final/04_ParkingOccupancy

2. Run the exhaustive search
python3 main.py

This will:

test all valid parameter combinations defined in main.py
evaluate them on the whole dataset
rank the results
save:
outputs/results/results.csv
outputs/results/best_results.txt

3. Inspect the best configuration

After the exhaustive search finishes, run:

python3 inspect_best_config.py

This will:

load the best configuration from outputs/results/results.csv
apply it to one selected image
save visual/debug outputs under outputs/inspection/best_config/
