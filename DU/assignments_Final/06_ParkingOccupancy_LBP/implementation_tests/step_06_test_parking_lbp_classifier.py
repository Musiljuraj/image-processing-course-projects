from pathlib import Path

from parking_io import load_parking_map, load_test_images
from parking_training_io import load_all_training_records, summarize_training_records
from roi_extraction import extract_all_rois_from_image
from parking_lbp_dataset import (
    prepare_training_feature_records,
    prepare_test_feature_records,
    build_training_matrix_and_labels,
    build_test_matrix,
    summarize_feature_records,
)
from parking_lbp_classifier import (
    train_classifier,
    predict_labels,
    predict_scores,
    build_prediction_records,
    summarize_predictions,
)


def run_classifier_smoke_test(
    classifier_config,
    X_train,
    y_train,
    training_feature_records,
    X_test,
    test_feature_records,
):
    """
    Run one smoke-test pass for a single classifier configuration.
    """
    print("\n============================================================")
    print(f"Classifier config: {classifier_config}")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Train classifier
    # -------------------------------------------------------------------------
    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=classifier_config,
    )
    print("Model training: OK")

    # -------------------------------------------------------------------------
    # 2. Predict on training set (sanity check only)
    # -------------------------------------------------------------------------
    train_predicted_labels = predict_labels(model=model, X_test=X_train)
    train_predicted_scores = predict_scores(model=model, X_test=X_train)

    train_prediction_summary = summarize_predictions(train_predicted_labels)

    print("\nTraining prediction summary:")
    print(f"  total_count   : {train_prediction_summary['total_count']}")
    print(f"  free_count    : {train_prediction_summary['free_count']}")
    print(f"  full_count    : {train_prediction_summary['full_count']}")
    print(f"  labels_present: {train_prediction_summary['labels_present']}")

    if len(train_predicted_labels) != X_train.shape[0]:
        raise ValueError("Training prediction count does not match X_train rows.")

    if train_predicted_scores is not None and len(train_predicted_scores) != X_train.shape[0]:
        raise ValueError("Training score count does not match X_train rows.")

    training_prediction_records = build_prediction_records(
        feature_records=training_feature_records,
        predicted_labels=train_predicted_labels,
        predicted_scores=train_predicted_scores,
    )

    if len(training_prediction_records) != len(training_feature_records):
        raise ValueError(
            "Training prediction-record count does not match training feature-record count."
        )

    print("Training prediction records: OK")

    # -------------------------------------------------------------------------
    # 3. Predict on real test ROIs
    # -------------------------------------------------------------------------
    test_predicted_labels = predict_labels(model=model, X_test=X_test)
    test_predicted_scores = predict_scores(model=model, X_test=X_test)

    test_prediction_summary = summarize_predictions(test_predicted_labels)

    print("\nTest prediction summary:")
    print(f"  total_count   : {test_prediction_summary['total_count']}")
    print(f"  free_count    : {test_prediction_summary['free_count']}")
    print(f"  full_count    : {test_prediction_summary['full_count']}")
    print(f"  labels_present: {test_prediction_summary['labels_present']}")

    if len(test_predicted_labels) != X_test.shape[0]:
        raise ValueError("Test prediction count does not match X_test rows.")

    if test_predicted_scores is not None and len(test_predicted_scores) != X_test.shape[0]:
        raise ValueError("Test score count does not match X_test rows.")

    test_prediction_records = build_prediction_records(
        feature_records=test_feature_records,
        predicted_labels=test_predicted_labels,
        predicted_scores=test_predicted_scores,
    )

    if len(test_prediction_records) != len(test_feature_records):
        raise ValueError(
            "Test prediction-record count does not match test feature-record count."
        )

    print("Test prediction records: OK")

    # -------------------------------------------------------------------------
    # 4. Inspect one example prediction record
    # -------------------------------------------------------------------------
    first_record = test_prediction_records[0]
    print("\nFirst test prediction record (selected fields):")
    print(f"  source_image_name : {first_record.get('source_image_name')}")
    print(f"  space_index       : {first_record.get('space_index')}")
    print(f"  predicted_label   : {first_record.get('predicted_label')}")
    print(f"  has_score         : {'predicted_score' in first_record}")

    print("\nClassifier smoke test for this config finished successfully.")


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

    classifier_configs = [
        {
            "classifier_name": "knn",
            "n_neighbors": 3,
        },
        {
            "classifier_name": "linear_svm",
            "C": 1.0,
        },
    ]

    print("=== STEP 6 SMOKE TEST: parking_lbp_classifier ===")

    # -------------------------------------------------------------------------
    # 1. TRAINING SIDE
    # -------------------------------------------------------------------------
    print("\n[1] Loading training dataset...")
    training_records = load_all_training_records(training_root)
    training_summary = summarize_training_records(training_records)

    print("Training dataset summary:")
    print(f"  Total samples : {training_summary['total_count']}")
    print(f"  Free samples  : {training_summary['free_count']}")
    print(f"  Full samples  : {training_summary['full_count']}")
    print(f"  Classes found : {training_summary['class_names_present']}")
    print(f"  Labels found  : {training_summary['labels_present']}")

    print("\n[2] Preparing training feature records...")
    training_feature_records = prepare_training_feature_records(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    training_feature_summary = summarize_feature_records(training_feature_records)

    print("Training feature summary:")
    print(f"  Feature records count   : {training_feature_summary['total_count']}")
    print(f"  Feature lengths present : {training_feature_summary['feature_lengths_present']}")
    print(f"  Labels present          : {training_feature_summary['labels_present']}")
    print(f"  Has labels              : {training_feature_summary['has_labels']}")

    print("\n[3] Building X_train and y_train...")
    X_train, y_train, training_metadata = build_training_matrix_and_labels(
        training_feature_records=training_feature_records
    )

    print("Training matrix outputs:")
    print(f"  X_train.shape            : {X_train.shape}")
    print(f"  y_train.shape            : {y_train.shape}")
    print(f"  training_metadata length : {len(training_metadata)}")

    # -------------------------------------------------------------------------
    # 2. TEST SIDE
    # -------------------------------------------------------------------------
    print("\n[4] Loading parking map and first test image...")
    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(test_images_dir)

    if not test_cases:
        raise ValueError("No test images were found.")

    first_test_case = test_cases[0]
    test_image_name = first_test_case["name"]
    test_image = first_test_case["image"]

    print(f"  Selected test image   : {test_image_name}")
    print(f"  Parking spaces in map : {len(parking_map)}")

    print("\n[5] Extracting test ROIs...")
    test_roi_records = extract_all_rois_from_image(
        image=test_image,
        parking_map=parking_map,
        image_name=test_image_name,
    )

    print(f"  Extracted ROI count   : {len(test_roi_records)}")

    print("\n[6] Preparing test feature records...")
    test_feature_records = prepare_test_feature_records(
        test_roi_records=test_roi_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    test_feature_summary = summarize_feature_records(test_feature_records)

    print("Test feature summary:")
    print(f"  Feature records count   : {test_feature_summary['total_count']}")
    print(f"  Feature lengths present : {test_feature_summary['feature_lengths_present']}")
    print(f"  Has labels              : {test_feature_summary['has_labels']}")
    print(f"  Source image names      : {test_feature_summary['source_image_names_present']}")

    print("\n[7] Building X_test...")
    X_test, test_metadata = build_test_matrix(test_feature_records)

    print("Test matrix outputs:")
    print(f"  X_test.shape         : {X_test.shape}")
    print(f"  test_metadata length : {len(test_metadata)}")

    # -------------------------------------------------------------------------
    # 3. CROSS-CHECK TRAIN/TEST COMPATIBILITY
    # -------------------------------------------------------------------------
    print("\n[8] Checking feature compatibility...")
    training_feature_lengths = training_feature_summary["feature_lengths_present"]
    test_feature_lengths = test_feature_summary["feature_lengths_present"]

    if len(training_feature_lengths) != 1:
        raise ValueError("Training feature vectors do not all have the same length.")

    if len(test_feature_lengths) != 1:
        raise ValueError("Test feature vectors do not all have the same length.")

    training_feature_length = training_feature_lengths[0]
    test_feature_length = test_feature_lengths[0]

    print(f"  Training feature length : {training_feature_length}")
    print(f"  Test feature length     : {test_feature_length}")

    if training_feature_length != test_feature_length:
        raise ValueError(
            "Training and test feature lengths do not match."
        )

    # -------------------------------------------------------------------------
    # 4. RUN BOTH CLASSIFIERS
    # -------------------------------------------------------------------------
    print("\n[9] Running classifier smoke tests...")
    for classifier_config in classifier_configs:
        run_classifier_smoke_test(
            classifier_config=classifier_config,
            X_train=X_train,
            y_train=y_train,
            training_feature_records=training_feature_records,
            X_test=X_test,
            test_feature_records=test_feature_records,
        )

    print("\nAll Step 6 smoke tests finished successfully.")
    print("parking_lbp_classifier.py is ready for evaluation/orchestration stages.")


if __name__ == "__main__":
    main()