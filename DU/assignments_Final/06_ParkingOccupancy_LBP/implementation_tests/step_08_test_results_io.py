from pathlib import Path
import time

from parking_io import load_parking_map, load_test_images
from parking_training_io import load_all_training_records
from roi_extraction import extract_all_rois_from_image
from parking_lbp_dataset import (
    prepare_training_feature_records,
    prepare_test_feature_records,
    build_training_matrix_and_labels,
    build_test_matrix,
)
from parking_lbp_classifier import (
    train_classifier,
    predict_labels,
    predict_scores,
    build_prediction_records,
)
from evaluation import evaluate_one_test_case
from results_io import save_experiment_outputs


def run_one_experiment(
    training_records,
    test_case,
    parking_map,
    preprocessing_config,
    lbp_config,
    classifier_config,
    evaluation_config,
):
    """
    Run one small end-to-end experiment and return one experiment-result dict
    suitable for results_io.py.
    """
    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # 1. Prepare training features
    # -------------------------------------------------------------------------
    t_train_features_start = time.perf_counter()

    training_feature_records = prepare_training_feature_records(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    X_train, y_train, training_metadata = build_training_matrix_and_labels(
        training_feature_records=training_feature_records
    )

    t_train_features_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 2. Train classifier
    # -------------------------------------------------------------------------
    t_training_start = time.perf_counter()

    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=classifier_config,
    )

    t_training_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 3. Prepare test features
    # -------------------------------------------------------------------------
    t_test_features_start = time.perf_counter()

    test_roi_records = extract_all_rois_from_image(
        image=test_case["image"],
        parking_map=parking_map,
        image_name=test_case["name"],
    )

    test_feature_records = prepare_test_feature_records(
        test_roi_records=test_roi_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    X_test, test_metadata = build_test_matrix(test_feature_records)

    t_test_features_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 4. Predict
    # -------------------------------------------------------------------------
    t_prediction_start = time.perf_counter()

    predicted_labels = predict_labels(model=model, X_test=X_test)
    predicted_scores = predict_scores(model=model, X_test=X_test)

    prediction_records = build_prediction_records(
        feature_records=test_feature_records,
        predicted_labels=predicted_labels,
        predicted_scores=predicted_scores,
    )

    t_prediction_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 5. Evaluate
    # -------------------------------------------------------------------------
    image_evaluation = evaluate_one_test_case(
        prediction_records=prediction_records,
        txt_path=test_case["txt_path"],
        evaluation_config=evaluation_config,
    )

    t1 = time.perf_counter()

    confusion_counts = image_evaluation["confusion_counts"]

    experiment_result = {
        "preprocessing_config": preprocessing_config,
        "lbp_config": lbp_config,
        "classifier_config": classifier_config,
        "evaluation_config": evaluation_config,
        "confusion_counts": confusion_counts,
        "accuracy": image_evaluation["accuracy"],
        "num_samples": image_evaluation["num_samples"],
        "source_image_name": test_case["name"],
        "processing_time_total": t1 - t0,
        "processing_time_training": t_training_end - t_training_start,
        "processing_time_test_feature_preparation": (
            t_test_features_end - t_test_features_start
        ),
        "processing_time_prediction": t_prediction_end - t_prediction_start,
        "processing_time_per_image": t1 - t0,
        "processing_time_per_roi": (
            (t1 - t0) / image_evaluation["num_samples"]
            if image_evaluation["num_samples"] > 0
            else 0.0
        ),
        "training_sample_count": len(training_records),
        "training_feature_count": X_train.shape[1],
        "test_roi_count": len(test_roi_records),
        "training_metadata_count": len(training_metadata),
        "test_metadata_count": len(test_metadata),
    }

    return experiment_result


def main():
    project_root = Path(__file__).resolve().parent

    training_root = project_root / "data" / "training"
    map_path = project_root / "data" / "parking_map_python.txt"
    test_images_dir = project_root / "data" / "test_images_zao"

    output_dir = project_root / "outputs" / "results" / "step_08"

    evaluation_config = {
        "occupied_label": 1,
        "empty_label": 0,
    }

    print("=== STEP 8 SMOKE TEST: results_io.py ===")

    # -------------------------------------------------------------------------
    # 1. Load shared inputs once
    # -------------------------------------------------------------------------
    print("\n[1] Loading shared inputs...")
    training_records = load_all_training_records(training_root)
    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(test_images_dir)

    if not test_cases:
        raise ValueError("No test images were found.")

    test_case = test_cases[0]

    print(f"  training sample count : {len(training_records)}")
    print(f"  selected test image   : {test_case['name']}")
    print(f"  parking-space count   : {len(parking_map)}")

    # -------------------------------------------------------------------------
    # 2. Define a few small experiment configs
    # -------------------------------------------------------------------------
    experiment_configs = [
        {
            "experiment_index": 1,
            "preprocessing_config": {
                "target_size": (80, 80),
                "contrast_method": "clahe",
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": (8, 8),
                "filter_name": "gaussian",
                "kernel_size": 3,
            },
            "lbp_config": {
                "neighbors": 8,
                "radius": 1,
                "method": "uniform",
                "grid_shape": (4, 4),
                "normalize_histogram": True,
            },
            "classifier_config": {
                "classifier_name": "knn",
                "n_neighbors": 3,
            },
        },
        {
            "experiment_index": 2,
            "preprocessing_config": {
                "target_size": (80, 80),
                "contrast_method": "clahe",
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": (8, 8),
                "filter_name": "gaussian",
                "kernel_size": 3,
            },
            "lbp_config": {
                "neighbors": 8,
                "radius": 1,
                "method": "uniform",
                "grid_shape": (4, 4),
                "normalize_histogram": True,
            },
            "classifier_config": {
                "classifier_name": "linear_svm",
                "C": 1.0,
            },
        },
    ]

    # -------------------------------------------------------------------------
    # 3. Run a few real experiments
    # -------------------------------------------------------------------------
    print("\n[2] Running small experiments...")
    experiment_results = []

    for config in experiment_configs:
        print(
            f"  running experiment #{config['experiment_index']} "
            f"with classifier={config['classifier_config']}"
        )

        result = run_one_experiment(
            training_records=training_records,
            test_case=test_case,
            parking_map=parking_map,
            preprocessing_config=config["preprocessing_config"],
            lbp_config=config["lbp_config"],
            classifier_config=config["classifier_config"],
            evaluation_config=evaluation_config,
        )

        result["experiment_index"] = config["experiment_index"]
        experiment_results.append(result)

        print(
            f"    accuracy={result['accuracy']:.6f}, "
            f"num_samples={result['num_samples']}, "
            f"time_total={result['processing_time_total']:.6f}s"
        )

    if not experiment_results:
        raise ValueError("No experiment results were produced.")

    # -------------------------------------------------------------------------
    # 4. Save outputs through results_io.py
    # -------------------------------------------------------------------------
    print("\n[3] Saving experiment outputs...")
    saved_outputs = save_experiment_outputs(
        experiment_results=experiment_results,
        output_dir=output_dir,
        csv_filename="step_08_results.csv",
        summary_filename="step_08_summary.txt",
        top_k=10,
    )

    csv_path = saved_outputs["csv_path"]
    summary_path = saved_outputs["summary_path"]

    print(f"  csv_path     : {csv_path}")
    print(f"  summary_path : {summary_path}")

    # -------------------------------------------------------------------------
    # 5. Verify files exist and are non-empty
    # -------------------------------------------------------------------------
    print("\n[4] Verifying saved files...")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV output was not created: {csv_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary output was not created: {summary_path}")

    if csv_path.stat().st_size == 0:
        raise ValueError(f"CSV output is empty: {csv_path}")

    if summary_path.stat().st_size == 0:
        raise ValueError(f"Summary output is empty: {summary_path}")

    print(f"  CSV exists and is non-empty    : OK ({csv_path.stat().st_size} bytes)")
    print(f"  Summary exists and is non-empty: OK ({summary_path.stat().st_size} bytes)")

    # -------------------------------------------------------------------------
    # 6. Print a preview of the summary file
    # -------------------------------------------------------------------------
    print("\n[5] Preview of summary file:")
    summary_text = summary_path.read_text(encoding="utf-8")
    preview_lines = summary_text.splitlines()[:20]

    for line in preview_lines:
        print(f"  {line}")

    print("\nSmoke test finished successfully.")
    print("results_io.py is ready for experiment orchestration.")
    

if __name__ == "__main__":
    main()