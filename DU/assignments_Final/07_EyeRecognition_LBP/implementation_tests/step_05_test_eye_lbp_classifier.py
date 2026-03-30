"""
step_05_test_eye_lbp_classifier.py

Simple smoke test for eye_lbp_classifier.py.

This test verifies:
- a small subset of training records can be loaded,
- classifier-ready training matrices can be prepared,
- default classifier configuration is valid,
- a classifier model can be built,
- the classifier can be trained,
- labels can be predicted on a small test subset,
- optional scores can be obtained if supported,
- predictions can be attached back to feature records,
- single-record runtime-style prediction works,
- prediction summary can be computed.

The test intentionally remains lightweight:
- only a small subset of the dataset is used,
- train/test split is simple and local,
- it checks structural correctness, not final model quality.
"""

from pathlib import Path
import numpy as np

from eye_training_io import load_all_eye_training_records
from eye_preprocessing import get_default_preprocessing_config
from lbp_features import get_default_lbp_config
from eye_lbp_dataset import (
    prepare_training_matrix_and_labels,
    prepare_training_feature_records,
)
from eye_lbp_classifier import (
    get_default_classifier_config,
    validate_classifier_config,
    build_classifier_model,
    train_classifier,
    predict_labels,
    predict_single_label,
    predict_scores,
    predict_single_score,
    build_prediction_records,
    predict_from_runtime_feature_record,
    train_and_predict,
    summarize_predictions,
)


def assert_true(condition, message):
    """
    Raise AssertionError with a readable message when the condition is false.
    """
    if not condition:
        raise AssertionError(message)


def main():
    project_root = Path(__file__).resolve().parent
    dataset_root = project_root / "input" / "training" / "mrlEyes_2018_01"

    print("=== STEP 05 SMOKE TEST: eye_lbp_classifier.py ===")
    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print()

    preprocessing_config = get_default_preprocessing_config()
    lbp_config = get_default_lbp_config()
    classifier_config = get_default_classifier_config()

    # -------------------------------------------------------------
    # 1. Load a small balanced subset of training records
    # -------------------------------------------------------------
    print("[1/8] Loading a small training subset...")
    all_records = load_all_eye_training_records(
        dataset_root,
        load_images=True,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False,
    )

    assert_true(len(all_records) > 0, "No training records were loaded.")

    close_records = [record for record in all_records if record["label"] == 0]
    open_records = [record for record in all_records if record["label"] == 1]

    assert_true(len(close_records) >= 6, "Not enough close-eye samples for smoke test.")
    assert_true(len(open_records) >= 6, "Not enough open-eye samples for smoke test.")

    selected_records = close_records[:6] + open_records[:6]

    print("[OK] Loaded training records successfully.")
    print(f"  total loaded records: {len(all_records)}")
    print(f"  selected subset size: {len(selected_records)}")
    print(f"  selected close count: {sum(r['label'] == 0 for r in selected_records)}")
    print(f"  selected open count:  {sum(r['label'] == 1 for r in selected_records)}")
    print()

    # -------------------------------------------------------------
    # 2. Prepare training matrices and feature records
    # -------------------------------------------------------------
    print("[2/8] Preparing classifier-ready matrices and feature records...")
    X_all, y_all, metadata_all, training_feature_records = prepare_training_matrix_and_labels(
        training_records=selected_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        image_key="image",
    )

    assert_true(isinstance(X_all, np.ndarray), "X_all must be a NumPy array.")
    assert_true(isinstance(y_all, np.ndarray), "y_all must be a NumPy array.")
    assert_true(X_all.ndim == 2, "X_all must be 2D.")
    assert_true(y_all.ndim == 1, "y_all must be 1D.")
    assert_true(X_all.shape[0] == len(selected_records), "X_all row count mismatch.")
    assert_true(y_all.shape[0] == len(selected_records), "y_all length mismatch.")
    assert_true(len(metadata_all) == len(selected_records), "metadata_all length mismatch.")
    assert_true(len(training_feature_records) == len(selected_records), "training_feature_records length mismatch.")

    print("[OK] Matrices and feature records prepared successfully.")
    print(f"  X_all shape: {X_all.shape}")
    print(f"  y_all shape: {y_all.shape}")
    print()

    # Simple local split:
    # first 4 close + first 4 open => training
    # remaining 2 close + remaining 2 open => test
    train_records = selected_records[:4] + selected_records[6:10]
    test_records = selected_records[4:6] + selected_records[10:12]

    X_train, y_train, train_metadata, train_feature_records = prepare_training_matrix_and_labels(
        training_records=train_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        image_key="image",
    )

    X_test, y_test, test_metadata, test_feature_records = prepare_training_matrix_and_labels(
        training_records=test_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        image_key="image",
    )

    assert_true(set(np.unique(y_train)).issubset({0, 1}), "y_train contains invalid labels.")
    assert_true(set(np.unique(y_test)).issubset({0, 1}), "y_test contains invalid labels.")
    assert_true(len(np.unique(y_train)) == 2, "y_train must contain both classes for training.")

    # -------------------------------------------------------------
    # 3. Validate classifier config and build model
    # -------------------------------------------------------------
    print("[3/8] Validating classifier configuration and building model...")
    validated_classifier_config = validate_classifier_config(classifier_config)
    model = build_classifier_model(validated_classifier_config)

    assert_true(isinstance(validated_classifier_config, dict), "Validated config must be a dictionary.")
    assert_true("classifier_name" in validated_classifier_config, "Validated config missing classifier_name.")
    assert_true(model is not None, "Model construction returned None.")
    assert_true(hasattr(model, "fit"), "Model does not provide fit(...).")
    assert_true(hasattr(model, "predict"), "Model does not provide predict(...).")

    print("[OK] Classifier configuration is valid and model was built.")
    print(f"  classifier_name: {validated_classifier_config['classifier_name']}")
    print(f"  model type:       {type(model).__name__}")
    print()

    # -------------------------------------------------------------
    # 4. Train classifier
    # -------------------------------------------------------------
    print("[4/8] Training classifier...")
    trained_model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=validated_classifier_config,
    )

    assert_true(trained_model is not None, "Training returned None.")
    assert_true(hasattr(trained_model, "predict"), "Trained model does not provide predict(...).")

    print("[OK] Classifier trained successfully.")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print()

    # -------------------------------------------------------------
    # 5. Predict labels and optional scores on test set
    # -------------------------------------------------------------
    print("[5/8] Predicting labels and optional scores...")
    predicted_labels = predict_labels(trained_model, X_test)
    predicted_scores = predict_scores(trained_model, X_test)

    assert_true(isinstance(predicted_labels, np.ndarray), "predicted_labels must be a NumPy array.")
    assert_true(predicted_labels.ndim == 1, "predicted_labels must be 1D.")
    assert_true(predicted_labels.shape[0] == X_test.shape[0], "Prediction count mismatch.")
    assert_true(set(np.unique(predicted_labels)).issubset({0, 1}), "Predicted labels outside {0,1}.")

    if predicted_scores is not None:
        assert_true(isinstance(predicted_scores, np.ndarray), "predicted_scores must be a NumPy array.")
        assert_true(predicted_scores.ndim == 1, "predicted_scores must be 1D.")
        assert_true(predicted_scores.shape[0] == X_test.shape[0], "predicted_scores count mismatch.")

    print("[OK] Prediction on test set works.")
    print(f"  X_test shape:       {X_test.shape}")
    print(f"  predicted_labels:   {predicted_labels.tolist()}")
    print(f"  predicted_scores:   {None if predicted_scores is None else predicted_scores.tolist()}")
    print()

    # -------------------------------------------------------------
    # 6. Predict single label/score and attach predictions to records
    # -------------------------------------------------------------
    print("[6/8] Testing single-sample prediction and prediction-record building...")
    single_feature_vector = test_feature_records[0]["lbp_feature_vector"]

    single_label = predict_single_label(trained_model, single_feature_vector)
    single_score = predict_single_score(trained_model, single_feature_vector)

    assert_true(single_label in (0, 1), "Single predicted label must be 0 or 1.")
    if single_score is not None:
        assert_true(isinstance(single_score, float), "Single predicted score must be float or None.")

    prediction_records = build_prediction_records(
        feature_records=test_feature_records,
        predicted_labels=predicted_labels,
        predicted_scores=predicted_scores,
    )

    assert_true(len(prediction_records) == len(test_feature_records), "prediction_records length mismatch.")
    assert_true("predicted_label" in prediction_records[0], "Missing predicted_label in prediction record.")
    assert_true("predicted_class_name" in prediction_records[0], "Missing predicted_class_name in prediction record.")

    print("[OK] Single-sample prediction and prediction-record building work.")
    print(f"  single predicted label: {single_label}")
    print(f"  single predicted score: {single_score}")
    print(f"  first prediction record predicted_class_name: {prediction_records[0]['predicted_class_name']}")
    print()

    # -------------------------------------------------------------
    # 7. Test runtime-style prediction from one runtime feature record
    # -------------------------------------------------------------
    print("[7/8] Testing runtime-style prediction from one runtime feature record...")
    runtime_prediction_record = predict_from_runtime_feature_record(
        model=trained_model,
        runtime_feature_record=test_feature_records[0],
    )

    assert_true("predicted_label" in runtime_prediction_record, "Runtime prediction record missing predicted_label.")
    assert_true(
        "predicted_class_name" in runtime_prediction_record,
        "Runtime prediction record missing predicted_class_name."
    )
    assert_true(runtime_prediction_record["predicted_label"] in (0, 1), "Runtime predicted label must be 0 or 1.")

    print("[OK] Runtime-style single-record prediction works.")
    print(f"  runtime predicted_label:      {runtime_prediction_record['predicted_label']}")
    print(f"  runtime predicted_class_name: {runtime_prediction_record['predicted_class_name']}")
    print()

    # -------------------------------------------------------------
    # 8. Test train_and_predict convenience wrapper and summary
    # -------------------------------------------------------------
    print("[8/8] Testing train_and_predict(...) and summarize_predictions(...).")
    trained_model_2, predicted_labels_2, predicted_scores_2 = train_and_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        classifier_config=validated_classifier_config,
    )

    assert_true(trained_model_2 is not None, "train_and_predict returned no model.")
    assert_true(predicted_labels_2.shape[0] == X_test.shape[0], "train_and_predict label count mismatch.")
    if predicted_scores_2 is not None:
        assert_true(predicted_scores_2.shape[0] == X_test.shape[0], "train_and_predict score count mismatch.")

    prediction_summary = summarize_predictions(
        predicted_labels=predicted_labels_2,
        ground_truth_labels=y_test,
    )

    assert_true(isinstance(prediction_summary, dict), "prediction_summary must be a dictionary.")
    assert_true(prediction_summary["predicted_count"] == X_test.shape[0], "prediction_summary predicted_count mismatch.")
    assert_true(prediction_summary["has_ground_truth"] is True, "prediction_summary has_ground_truth should be True.")
    assert_true("accuracy_percent" in prediction_summary, "prediction_summary missing accuracy_percent.")

    print("[OK] train_and_predict(...) and summarize_predictions(...) work.")
    print(f"  predicted_count:   {prediction_summary['predicted_count']}")
    print(f"  accuracy_percent:  {prediction_summary['accuracy_percent']:.2f}")
    print(f"  predicted_close:   {prediction_summary['predicted_close_count']}")
    print(f"  predicted_open:    {prediction_summary['predicted_open_count']}")
    print()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()