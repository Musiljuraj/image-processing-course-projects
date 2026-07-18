from pathlib import Path

from experiment_search import run_experiment_search
from results_io import save_experiment_outputs


def main():
    project_root = Path(__file__).resolve().parent

    training_root = project_root / "data" / "training"
    map_path = project_root / "data" / "parking_map_python.txt"
    test_images_dir = project_root / "data" / "test_images_zao"

    output_dir = project_root / "outputs" / "results" / "step_09"

    preprocessing_configurations = [
        {
            "target_size": (80, 80),
            "contrast_method": "clahe",
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": (8, 8),
            "filter_name": "gaussian",
            "kernel_size": 3,
        },
        {
            "target_size": (80, 80),
            "contrast_method": "none",
            "filter_name": "none",
            "kernel_size": 3,
        },
    ]

    lbp_configurations = [
        {
            "neighbors": 8,
            "radius": 1,
            "method": "uniform",
            "grid_shape": (4, 4),
            "normalize_histogram": True,
        },
        {
            "neighbors": 8,
            "radius": 1,
            "method": "uniform",
            "grid_shape": (2, 2),
            "normalize_histogram": True,
        },
    ]

    classifier_configurations = [
        {
            "classifier_name": "knn",
            "n_neighbors": 3,
        },
        {
            "classifier_name": "linear_svm",
            "C": 1.0,
        },
    ]

    evaluation_config = {
        "occupied_label": 1,
        "empty_label": 0,
    }

    print("=== STEP 9 SMOKE TEST: experiment_search.py ===")

    print("\n[1] Running experiment search...")
    search_result = run_experiment_search(
        training_root=training_root,
        map_path=map_path,
        test_images_dir=test_images_dir,
        preprocessing_configurations=preprocessing_configurations,
        lbp_configurations=lbp_configurations,
        classifier_configurations=classifier_configurations,
        evaluation_config=evaluation_config,
        max_test_cases=1,  # keep smoke test small and fast
    )

    experiment_configurations = search_result["experiment_configurations"]
    experiment_results = search_result["experiment_results"]
    ranked_results = search_result["ranked_results"]

    print(f"  experiment configuration count : {len(experiment_configurations)}")
    print(f"  experiment result count        : {len(experiment_results)}")
    print(f"  ranked result count            : {len(ranked_results)}")

    expected_count = (
        len(preprocessing_configurations)
        * len(lbp_configurations)
        * len(classifier_configurations)
    )

    if len(experiment_configurations) != expected_count:
        raise ValueError(
            "Unexpected number of experiment configurations. "
            f"Expected {expected_count}, got {len(experiment_configurations)}"
        )

    if len(experiment_results) != expected_count:
        raise ValueError(
            "Unexpected number of experiment results. "
            f"Expected {expected_count}, got {len(experiment_results)}"
        )

    if len(ranked_results) != expected_count:
        raise ValueError(
            "Unexpected number of ranked results. "
            f"Expected {expected_count}, got {len(ranked_results)}"
        )

    print("\n[2] Inspecting first raw result...")
    first_result = experiment_results[0]

    required_keys = [
        "preprocessing_config",
        "lbp_config",
        "classifier_config",
        "evaluation_config",
        "confusion_counts",
        "accuracy",
        "num_samples",
        "processing_time_total",
        "processing_time_training",
        "processing_time_test_feature_preparation",
        "processing_time_prediction",
        "processing_time_per_image",
        "processing_time_per_roi",
        "per_image_results",
        "experiment_index",
    ]

    for key in required_keys:
        if key not in first_result:
            raise KeyError(f"Missing required experiment-result key: {key}")

    print("  first raw result required keys: OK")
    print(f"  first raw result accuracy      : {first_result['accuracy']:.6f}")
    print(f"  first raw result num_samples   : {first_result['num_samples']}")
    print(f"  first raw result total time    : {first_result['processing_time_total']:.6f}s")

    confusion_counts = first_result["confusion_counts"]
    for key in ["tp", "tn", "fp", "fn"]:
        if key not in confusion_counts:
            raise KeyError(f"Missing confusion-count key: {key}")

    print(f"  first raw result confusion     : {confusion_counts}")

    if first_result["num_samples"] <= 0:
        raise ValueError("Experiment result has non-positive num_samples.")

    if not (0.0 <= first_result["accuracy"] <= 1.0):
        raise ValueError("Experiment accuracy is outside [0, 1].")

    if first_result["processing_time_total"] <= 0:
        raise ValueError("Experiment total processing time must be positive.")

    if not first_result["per_image_results"]:
        raise ValueError("per_image_results must not be empty.")

    print("\n[3] Inspecting first ranked result...")
    best_result = ranked_results[0]
    print(f"  best accuracy                  : {best_result['accuracy']:.6f}")
    print(f"  best classifier config         : {best_result['classifier_config']}")
    print(f"  best lbp config                : {best_result['lbp_config']}")
    print(f"  best preprocessing config      : {best_result['preprocessing_config']}")

    print("\n[4] Checking ranked ordering...")
    for i in range(1, len(ranked_results)):
        prev_result = ranked_results[i - 1]
        curr_result = ranked_results[i]

        prev_key = (
            -prev_result.get("accuracy", 0.0),
            prev_result.get("processing_time_total", float("inf")),
            -prev_result.get("num_samples", 0),
        )
        curr_key = (
            -curr_result.get("accuracy", 0.0),
            curr_result.get("processing_time_total", float("inf")),
            -curr_result.get("num_samples", 0),
        )

        if prev_key > curr_key:
            raise ValueError(
                "ranked_results are not sorted according to the expected ranking rule."
            )

    print("  ranked ordering: OK")

    print("\n[5] Saving ranked results via results_io.py...")
    saved_outputs = save_experiment_outputs(
        experiment_results=ranked_results,
        output_dir=output_dir,
        csv_filename="step_09_results.csv",
        summary_filename="step_09_summary.txt",
        top_k=10,
    )

    csv_path = saved_outputs["csv_path"]
    summary_path = saved_outputs["summary_path"]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV output was not created: {csv_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary output was not created: {summary_path}")

    if csv_path.stat().st_size == 0:
        raise ValueError(f"CSV output is empty: {csv_path}")

    if summary_path.stat().st_size == 0:
        raise ValueError(f"Summary output is empty: {summary_path}")

    print(f"  csv_path                       : {csv_path}")
    print(f"  summary_path                   : {summary_path}")
    print(f"  csv size                       : {csv_path.stat().st_size} bytes")
    print(f"  summary size                   : {summary_path.stat().st_size} bytes")

    print("\n[6] Preview of first ranked result fields...")
    preview_keys = [
        "experiment_index",
        "accuracy",
        "num_samples",
        "processing_time_total",
        "classifier_config",
    ]
    for key in preview_keys:
        print(f"  {key}: {best_result.get(key)}")

    print("\nSmoke test finished successfully.")
    print("experiment_search.py is ready for final main.py integration.")


if __name__ == "__main__":
    main()