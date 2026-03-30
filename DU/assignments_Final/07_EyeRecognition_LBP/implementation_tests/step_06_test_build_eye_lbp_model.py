"""
step_06_test_build_eye_lbp_model.py

Simple smoke test for build_eye_lbp_model(...) in eye_lbp_classifier.py.

This test verifies:
- the full startup training function can load the dataset,
- preprocessing + LBP + matrix building are executed successfully,
- the classifier is trained successfully,
- the returned model bundle has the expected structure,
- the trained model can immediately predict on training samples,
- one runtime-style prediction record can be produced.

The test is intentionally simple:
- it uses the default configs,
- it trains on the whole dataset returned by the loader,
- it checks structural correctness, not final model quality.
"""

from pathlib import Path
import numpy as np

from eye_preprocessing import get_default_preprocessing_config
from lbp_features import get_default_lbp_config
from eye_lbp_classifier import (
    get_default_classifier_config,
    build_eye_lbp_model,
    predict_labels,
    predict_scores,
    predict_from_runtime_feature_record,
)


REQUIRED_MODEL_BUNDLE_KEYS = {
    "model",
    "preprocessing_config",
    "lbp_config",
    "classifier_config",
    "dataset_root",
    "training_sample_count",
    "feature_count",
    "X_train_shape",
    "y_train_shape",
    "class_counts",
    "training_metadata",
    "training_feature_records",
    "X_train",
    "y_train",
}


def assert_true(condition, message):
    """
    Raise AssertionError with a readable message when the condition is false.
    """
    if not condition:
        raise AssertionError(message)


def main():
    project_root = Path(__file__).resolve().parent
    dataset_root = project_root / "input" / "training" / "mrlEyes_2018_01"

    print("=== STEP 06 SMOKE TEST: build_eye_lbp_model(...) ===")
    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print()

    preprocessing_config = get_default_preprocessing_config()
    lbp_config = get_default_lbp_config()
    classifier_config = get_default_classifier_config()

    # -------------------------------------------------------------
    # 1. Basic dataset path check
    # -------------------------------------------------------------
    print("[1/5] Checking dataset path...")
    assert_true(dataset_root.exists(), f"Dataset directory does not exist: {dataset_root}")
    assert_true(dataset_root.is_dir(), f"Dataset path is not a directory: {dataset_root}")
    print("[OK] Dataset path exists.")
    print()

    # -------------------------------------------------------------
    # 2. Build startup-trained model bundle
    # -------------------------------------------------------------
    print("[2/5] Building startup-trained eye LBP model bundle...")
    model_bundle = build_eye_lbp_model(
        dataset_root=dataset_root,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        classifier_config=classifier_config,
        load_images=True,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False,
        image_key="image",
    )

    assert_true(isinstance(model_bundle, dict), "model_bundle must be a dictionary.")

    missing_keys = REQUIRED_MODEL_BUNDLE_KEYS - set(model_bundle.keys())
    assert_true(not missing_keys, f"model_bundle is missing keys: {sorted(missing_keys)}")

    assert_true(model_bundle["model"] is not None, "model_bundle['model'] is None.")
    assert_true(model_bundle["training_sample_count"] > 0, "training_sample_count must be positive.")
    assert_true(model_bundle["feature_count"] > 0, "feature_count must be positive.")
    assert_true(isinstance(model_bundle["training_feature_records"], list), "training_feature_records must be a list.")
    assert_true(isinstance(model_bundle["training_metadata"], list), "training_metadata must be a list.")

    print("[OK] Model bundle built successfully.")
    print(f"  training_sample_count: {model_bundle['training_sample_count']}")
    print(f"  feature_count:         {model_bundle['feature_count']}")
    print(f"  X_train_shape:         {model_bundle['X_train_shape']}")
    print(f"  y_train_shape:         {model_bundle['y_train_shape']}")
    print()

    # -------------------------------------------------------------
    # 3. Validate internal bundle consistency
    # -------------------------------------------------------------
    print("[3/5] Validating internal bundle consistency...")
    X_train = model_bundle["X_train"]
    y_train = model_bundle["y_train"]
    training_feature_records = model_bundle["training_feature_records"]
    training_metadata = model_bundle["training_metadata"]
    class_counts = model_bundle["class_counts"]

    assert_true(isinstance(X_train, np.ndarray), "X_train must be a NumPy array.")
    assert_true(isinstance(y_train, np.ndarray), "y_train must be a NumPy array.")
    assert_true(X_train.ndim == 2, "X_train must be 2D.")
    assert_true(y_train.ndim == 1, "y_train must be 1D.")

    assert_true(X_train.shape[0] == model_bundle["training_sample_count"], "X_train row count mismatch.")
    assert_true(X_train.shape[1] == model_bundle["feature_count"], "X_train feature count mismatch.")
    assert_true(y_train.shape[0] == model_bundle["training_sample_count"], "y_train sample count mismatch.")
    assert_true(len(training_feature_records) == model_bundle["training_sample_count"], "training_feature_records length mismatch.")
    assert_true(len(training_metadata) == model_bundle["training_sample_count"], "training_metadata length mismatch.")

    assert_true(set(np.unique(y_train)).issubset({0, 1}), "y_train contains labels outside {0,1}.")
    assert_true(class_counts["close"] + class_counts["open"] == model_bundle["training_sample_count"], "class_counts do not sum to training_sample_count.")
    assert_true(class_counts["close"] > 0, "No close-eye samples found in trained dataset.")
    assert_true(class_counts["open"] > 0, "No open-eye samples found in trained dataset.")

    print("[OK] Bundle consistency is correct.")
    print(f"  class_counts: {class_counts}")
    print()

    # -------------------------------------------------------------
    # 4. Predict on a few training samples
    # -------------------------------------------------------------
    print("[4/5] Predicting on a few training samples...")
    trained_model = model_bundle["model"]

    sample_count = min(5, X_train.shape[0])
    X_sample = X_train[:sample_count]
    y_sample = y_train[:sample_count]

    predicted_labels = predict_labels(trained_model, X_sample)
    predicted_scores = predict_scores(trained_model, X_sample)

    assert_true(isinstance(predicted_labels, np.ndarray), "predicted_labels must be a NumPy array.")
    assert_true(predicted_labels.ndim == 1, "predicted_labels must be 1D.")
    assert_true(predicted_labels.shape[0] == sample_count, "predicted_labels count mismatch.")
    assert_true(set(np.unique(predicted_labels)).issubset({0, 1}), "predicted_labels contain values outside {0,1}.")

    if predicted_scores is not None:
        assert_true(isinstance(predicted_scores, np.ndarray), "predicted_scores must be a NumPy array.")
        assert_true(predicted_scores.ndim == 1, "predicted_scores must be 1D.")
        assert_true(predicted_scores.shape[0] == sample_count, "predicted_scores count mismatch.")

    print("[OK] Immediate prediction works.")
    print(f"  sample_count:      {sample_count}")
    print(f"  true_labels:       {y_sample.tolist()}")
    print(f"  predicted_labels:  {predicted_labels.tolist()}")
    print(f"  predicted_scores:  {None if predicted_scores is None else predicted_scores.tolist()}")
    print()

    # -------------------------------------------------------------
    # 5. Test one runtime-style prediction record
    # -------------------------------------------------------------
    print("[5/5] Testing runtime-style prediction from one feature record...")
    runtime_prediction_record = predict_from_runtime_feature_record(
        model=trained_model,
        runtime_feature_record=training_feature_records[0],
    )

    assert_true(isinstance(runtime_prediction_record, dict), "runtime_prediction_record must be a dictionary.")
    assert_true("predicted_label" in runtime_prediction_record, "Missing predicted_label in runtime_prediction_record.")
    assert_true("predicted_class_name" in runtime_prediction_record, "Missing predicted_class_name in runtime_prediction_record.")
    assert_true(runtime_prediction_record["predicted_label"] in (0, 1), "predicted_label must be 0 or 1.")
    assert_true(runtime_prediction_record["predicted_class_name"] in ("close", "open"), "predicted_class_name must be 'close' or 'open'.")

    print("[OK] Runtime-style prediction record works.")
    print(f"  predicted_label:      {runtime_prediction_record['predicted_label']}")
    print(f"  predicted_class_name: {runtime_prediction_record['predicted_class_name']}")
    if "predicted_score" in runtime_prediction_record:
        print(f"  predicted_score:      {runtime_prediction_record['predicted_score']}")
    print()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()