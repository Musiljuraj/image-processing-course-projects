#!/usr/bin/python3

"""
Fundamentals of Image Processing - Parking Occupancy Assignment
Final stage: automated exhaustive search over preprocessing, edge detection,
and classification settings.

Current purpose of this version:
1. locate the project folders
2. load parking-space geometry from data/parking_map_python.txt
3. load all test images from data/test_images_zao/
4. define the search space for exhaustive testing
5. run exhaustive search over all valid configurations
6. rank all results and identify the best one
7. save:
   - the full ranked results table to outputs/results/results.csv
   - a human-readable summary to outputs/results/best_results.txt

Design note:
The search space itself is intentionally defined here in main.py, exactly as
requested. The experiment_search.py module accepts that search space and builds
only the valid combinations from it.
"""

from pathlib import Path

from parking_io import load_parking_map, load_test_images
from experiment_search import run_exhaustive_search
from results_io import (
    ensure_results_directory,
    save_best_results_summary,
    save_ranked_results_csv,
)


def main():
    """
    Main orchestration function for the final exhaustive-search stage.

    Overall logic:
    1. define project paths
    2. verify expected input locations exist
    3. load parking-space map
    4. load all test images
    5. define the exhaustive-search space
    6. run exhaustive search over all valid combinations
    7. save full results table
    8. save best-results summary
    9. print the best configuration to the terminal

    The detailed pipeline work is delegated to:
    - parking_io.py
    - roi_extraction.py
    - preprocessing.py
    - edge_detection.py
    - evaluation.py
    - experiment_search.py
    - results_io.py
    """

    # determine the project root directory
    project_root = Path(__file__).resolve().parent

    # define the input-data locations
    data_dir = project_root / "data"
    map_path = data_dir / "parking_map_python.txt"
    images_dir = data_dir / "test_images_zao"
    outputs_dir = project_root / "outputs"
    results_dir = outputs_dir / "results"

    # perform basic existence checks for expected inputs
    if not map_path.exists():
        raise FileNotFoundError(f"Parking map file not found: {map_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {images_dir}")

    # load parking-space geometry
    parking_map = load_parking_map(map_path)

    # load all test images and verify matching .txt files exist
    test_cases = load_test_images(images_dir)

    if not test_cases:
        raise RuntimeError(f"No .jpg files found in {images_dir}")

    # =============================================================================
    # SEARCH SPACE
    # =============================================================================
    search_space = {
        "preprocessing": {
            # Focus only on the strongest preprocessing families found so far.
            # Your top results are dominated by:
            # - median
            # - gaussian
            #
            # Kernel sizes are centered around the strongest region from the 950-run
            # search, where k=5 and k=7 repeatedly appeared near the top.
            # - gaussian: classic smoothing baseline
            # - median: robust alternative that sometimes preserves edges better
            "filter_names": ["gaussian", "median"],

            "kernel_sizes": [5, 7],
        },
        "edge_detection": {
            # Focus only on Canny, because it clearly dominates the current top results.
            "detector_names": ["canny"],

            # Search in the neighborhood of the current best Canny region.
            # Current best: threshold1=30, threshold2=100
            #
            # These values are chosen to explore slightly lower / higher sensitivity
            # around that winning point without exploding the search space.
            "canny_threshold1_values": [20, 30, 40, 50],
            "canny_threshold2_values": [100, 120, 150],
            "canny_aperture_sizes": [3],
            "canny_l2gradient_values": [False],

            # Sobel ranges are left empty because Sobel is intentionally excluded
            # from this optimization-focused search.
            "sobel_ksizes": [],
            "sobel_thresholds": [],
        },
        "classification_evaluation": {

            "occupancy_threshold_ratios": [0.03, 0.04, 0.05],

            # Ground-truth label convention
            "occupied_label": 1,
            "empty_label": 0,
        },
    }

    verbose_search = True
    top_n_summary = 10
    # =============================================================================
    # END OF SEARCH SPACE BLOCK
    # =============================================================================

    # print basic dataset summary
    print(f"Project root: {project_root}")
    print(f"Map file: {map_path}")
    print(f"Images directory: {images_dir}")
    print(f"Loaded {len(parking_map)} parking spaces.")
    print(f"Loaded {len(test_cases)} test images.")
    print("Starting exhaustive search...")

    # run the exhaustive search and save its outputs
    search_summary = run_exhaustive_search(
        parking_map=parking_map,
        test_cases=test_cases,
        search_space=search_space,
        verbose=verbose_search,
    )

    best_result = search_summary["best_result"]

    results_dir = ensure_results_directory(results_dir)

    results_csv_path = results_dir / "results.csv"
    best_summary_path = results_dir / "best_results.txt"

    save_ranked_results_csv(search_summary, results_csv_path)
    save_best_results_summary(
        search_summary,
        best_summary_path,
        top_n=top_n_summary,
    )

    # final terminal summary focuses on ranked search results
    print("=" * 72)
    print("EXHAUSTIVE SEARCH FINISHED")
    print(f"Total tested configurations: {search_summary['total_configurations']}")
    print(f"Results CSV saved to: {results_csv_path}")
    print(f"Best-results summary saved to: {best_summary_path}")
    print("-" * 72)
    print("BEST CONFIGURATION")
    print(f"Rank: {best_result['rank']}")
    print(f"Experiment index: {best_result['experiment_index']}")
    print(f"Accuracy: {best_result['accuracy']:.6f}")
    print(f"TP: {best_result['tp']}")
    print(f"TN: {best_result['tn']}")
    print(f"FP: {best_result['fp']}")
    print(f"FN: {best_result['fn']}")
    print(f"Preprocessing config: {best_result['preprocessing_config']}")
    print(f"Edge detection config: {best_result['edge_detection_config']}")
    print(
        "Classification/evaluation config: "
        f"{best_result['classification_evaluation_config']}"
    )
    print("=" * 72)
    print("To visually inspect the best configuration on one image, run:")
    print("  python3 inspect_best_config.py")


if __name__ == "__main__":
    main()