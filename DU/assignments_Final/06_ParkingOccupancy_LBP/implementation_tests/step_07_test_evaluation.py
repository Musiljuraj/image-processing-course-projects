from pathlib import Path

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
    summarize_predictions,
)
from evaluation import (
    evaluate_one_test_case,
    evaluate_prediction_records,
)


def main():
    project_root = Path(__file__).resolve().parent

    training_root = project_root / "data" / "training"
    map_path = project_root / "data" / "parking_map_python.txt"
    test_images_dir = project_root / "data" / "test_images_zao"

    preprocessing_config = {
        "target_size": (80, 80),
        "contrast_method": "clahe",
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": (8, 8),
        "filter_name": "gaussian",
        "kernel_size": 3,
    }

    lbp_config = {
        "neighbors": 8,
        "radius": 1,
        "method": "uniform",
        "grid_shape": (4, 4),
        "normalize_histogram": True,
    }

    classifier_config = {
        "classifier_name": "linear_svm",
        "C": 1.0,
    }

    evaluation_config = {
        "occupied_label": 1,
        "empty_label": 0,
    }

    print("=== STEP 7 SMOKE TEST: evaluation.py ===")

    # -------------------------------------------------------------------------
    # 1. Build training matrix and train classifier
    # -------------------------------------------------------------------------
    print("\n[1] Loading training records...")
    training_records = load_all_training_records(training_root)
    print(f"  training sample count: {len(training_records)}")

    print("\n[2] Preparing training feature records...")
    training_feature_records = prepare_training_feature_records(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    print(f"  training feature record count: {len(training_feature_records)}")

    print("\n[3] Building X_train / y_train...")
    X_train, y_train, training_metadata = build_training_matrix_and_labels(
        training_feature_records=training_feature_records
    )

    print(f"  X_train.shape: {X_train.shape}")
    print(f"  y_train.shape: {y_train.shape}")
    print(f"  training_metadata length: {len(training_metadata)}")

    print("\n[4] Training classifier...")
    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=classifier_config,
    )
    print("  classifier training: OK")

    # -------------------------------------------------------------------------
    # 2. Load one real test case and produce prediction records
    # -------------------------------------------------------------------------
    print("\n[5] Loading parking map and test images...")
    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(test_images_dir)

    if not test_cases:
        raise ValueError("No test images were found.")

    test_case = test_cases[0]
    image_name = test_case["name"]
    image = test_case["image"]
    txt_path = test_case["txt_path"]

    print(f"  selected test image: {image_name}")
    print(f"  txt_path: {txt_path}")
    print(f"  parking spaces in map: {len(parking_map)}")

    print("\n[6] Extracting ROIs...")
    test_roi_records = extract_all_rois_from_image(
        image=image,
        parking_map=parking_map,
        image_name=image_name,
    )
    print(f"  extracted ROI count: {len(test_roi_records)}")

    print("\n[7] Preparing test feature records...")
    test_feature_records = prepare_test_feature_records(
        test_roi_records=test_roi_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )
    print(f"  test feature record count: {len(test_feature_records)}")

    print("\n[8] Building X_test...")
    X_test, test_metadata = build_test_matrix(test_feature_records)
    print(f"  X_test.shape: {X_test.shape}")
    print(f"  test_metadata length: {len(test_metadata)}")

    print("\n[9] Predicting labels...")
    predicted_labels = predict_labels(model=model, X_test=X_test)
    predicted_scores = predict_scores(model=model, X_test=X_test)

    prediction_summary = summarize_predictions(predicted_labels)
    print("  prediction summary:")
    print(f"    total_count   : {prediction_summary['total_count']}")
    print(f"    free_count    : {prediction_summary['free_count']}")
    print(f"    full_count    : {prediction_summary['full_count']}")
    print(f"    labels_present: {prediction_summary['labels_present']}")

    prediction_records = build_prediction_records(
        feature_records=test_feature_records,
        predicted_labels=predicted_labels,
        predicted_scores=predicted_scores,
    )

    print(f"  prediction record count: {len(prediction_records)}")

    # -------------------------------------------------------------------------
    # 3. Evaluate using the new evaluation module
    # -------------------------------------------------------------------------
    print("\n[10] Evaluating via evaluate_one_test_case(...)...")
    image_evaluation = evaluate_one_test_case(
        prediction_records=prediction_records,
        txt_path=txt_path,
        evaluation_config=evaluation_config,
    )

    confusion_counts = image_evaluation["confusion_counts"]
    accuracy = image_evaluation["accuracy"]

    print("  evaluation result:")
    print(f"    num_samples: {image_evaluation['num_samples']}")
    print(f"    accuracy   : {accuracy:.6f}")
    print(f"    tp         : {confusion_counts['tp']}")
    print(f"    tn         : {confusion_counts['tn']}")
    print(f"    fp         : {confusion_counts['fp']}")
    print(f"    fn         : {confusion_counts['fn']}")

    if image_evaluation["num_samples"] != len(prediction_records):
        raise ValueError(
            "Evaluation num_samples does not match prediction record count."
        )

    evaluated_records = image_evaluation["evaluated_records"]
    if len(evaluated_records) != len(prediction_records):
        raise ValueError(
            "evaluated_records length does not match prediction record count."
        )

    # -------------------------------------------------------------------------
    # 4. Optional cross-check: evaluate the same records directly against
    #    already loaded labels, through evaluate_prediction_records(...)
    # -------------------------------------------------------------------------
    print("\n[11] Cross-checking evaluate_prediction_records(...)...")
    from parking_io import load_ground_truth_labels

    ground_truth_labels = load_ground_truth_labels(txt_path)

    image_evaluation_direct = evaluate_prediction_records(
        prediction_records=prediction_records,
        ground_truth_labels=ground_truth_labels,
        evaluation_config=evaluation_config,
    )

    if image_evaluation_direct["confusion_counts"] != confusion_counts:
        raise ValueError(
            "Direct evaluation confusion counts do not match "
            "evaluate_one_test_case(...) results."
        )

    if abs(image_evaluation_direct["accuracy"] - accuracy) > 1e-12:
        raise ValueError(
            "Direct evaluation accuracy does not match "
            "evaluate_one_test_case(...) results."
        )

    print("  direct evaluation cross-check: OK")

    # -------------------------------------------------------------------------
    # 5. Inspect one evaluated record
    # -------------------------------------------------------------------------
    first_record = evaluated_records[0]
    print("\n[12] First evaluated record (selected fields):")
    print(f"  source_image_name  : {first_record.get('source_image_name')}")
    print(f"  space_index        : {first_record.get('space_index')}")
    print(f"  predicted_label    : {first_record.get('predicted_label')}")
    print(f"  ground_truth_label : {first_record.get('ground_truth_label')}")
    print(f"  evaluation_outcome : {first_record.get('evaluation_outcome')}")
    print(f"  has_score          : {'predicted_score' in first_record}")

    print("\nSmoke test finished successfully.")
    print("evaluation.py is ready for results/orchestration stages.")


if __name__ == "__main__":
    main()