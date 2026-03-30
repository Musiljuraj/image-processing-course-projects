"""
step_04_test_eye_lbp_dataset.py

Simple smoke test for eye_lbp_dataset.py.

This test verifies:
- a small subset of training records can be loaded,
- training feature records can be prepared end-to-end,
- feature matrix X can be built,
- label vector y can be built,
- metadata can be built,
- the convenience wrappers return aligned outputs,
- runtime matrix preparation works for a few raw eye images.

The test is intentionally lightweight:
- only a small subset of training records is used,
- it checks structural correctness and alignment, not classifier quality.
"""

from pathlib import Path
import numpy as np

from eye_training_io import load_all_eye_training_records
from lbp_features import get_default_lbp_config, get_lbp_histogram_bin_count
from eye_preprocessing import get_default_preprocessing_config
from eye_lbp_dataset import (
    prepare_training_feature_records,
    build_feature_matrix,
    build_label_vector,
    build_metadata_list,
    build_training_matrix_and_labels,
    prepare_training_matrix_and_labels,
    prepare_runtime_feature_records,
    build_runtime_matrix,
    prepare_runtime_matrix,
    summarize_feature_records,
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

    print("=== STEP 04 SMOKE TEST: eye_lbp_dataset.py ===")
    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print()

    preprocessing_config = get_default_preprocessing_config()
    lbp_config = get_default_lbp_config()

    neighbors = lbp_config["neighbors"]
    method = lbp_config["method"]
    grid_rows, grid_cols = lbp_config["grid_shape"]
    bin_count = get_lbp_histogram_bin_count(neighbors, method)
    expected_feature_length = bin_count * grid_rows * grid_cols

    # -------------------------------------------------------------
    # 1. Load a small subset of training records
    # -------------------------------------------------------------
    print("[1/7] Loading a small subset of training records...")
    all_training_records = load_all_eye_training_records(
        dataset_root,
        load_images=True,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False,
    )

    assert_true(len(all_training_records) > 0, "No training records were loaded.")

    subset_size = min(12, len(all_training_records))
    training_records = all_training_records[:subset_size]

    print(f"[OK] Loaded {len(all_training_records)} total records.")
    print(f"[OK] Using subset of {subset_size} records.")
    print()

    # -------------------------------------------------------------
    # 2. Prepare training feature records end-to-end
    # -------------------------------------------------------------
    print("[2/7] Preparing training feature records...")
    training_feature_records = prepare_training_feature_records(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        image_key="image",
    )

    assert_true(
        len(training_feature_records) == len(training_records),
        "Training feature record count mismatch."
    )

    first_feature_record = training_feature_records[0]
    assert_true("preprocessed_image" in first_feature_record, "Missing preprocessed_image.")
    assert_true("lbp_image" in first_feature_record, "Missing lbp_image.")
    assert_true("lbp_feature_vector" in first_feature_record, "Missing lbp_feature_vector.")
    assert_true("label" in first_feature_record, "Missing label.")
    assert_true(
        len(first_feature_record["lbp_feature_vector"]) == expected_feature_length,
        "Unexpected feature length in first feature record."
    )

    print("[OK] Training feature records prepared successfully.")
    print(f"  first file_name:       {first_feature_record.get('file_name')}")
    print(f"  first class_name:      {first_feature_record.get('class_name')}")
    print(f"  first feature length:  {len(first_feature_record['lbp_feature_vector'])}")
    print()

    # -------------------------------------------------------------
    # 3. Build feature matrix X
    # -------------------------------------------------------------
    print("[3/7] Building feature matrix X...")
    X = build_feature_matrix(training_feature_records)

    assert_true(isinstance(X, np.ndarray), "X must be a NumPy array.")
    assert_true(X.ndim == 2, "X must be 2D.")
    assert_true(X.shape[0] == len(training_feature_records), "X row count mismatch.")
    assert_true(X.shape[1] == expected_feature_length, "X column count mismatch.")

    print("[OK] Feature matrix built successfully.")
    print(f"  X shape: {X.shape}")
    print()

    # -------------------------------------------------------------
    # 4. Build label vector y
    # -------------------------------------------------------------
    print("[4/7] Building label vector y...")
    y = build_label_vector(training_feature_records)

    assert_true(isinstance(y, np.ndarray), "y must be a NumPy array.")
    assert_true(y.ndim == 1, "y must be 1D.")
    assert_true(len(y) == len(training_feature_records), "y length mismatch.")
    assert_true(set(np.unique(y)).issubset({0, 1}), "y contains labels outside {0, 1}.")

    print("[OK] Label vector built successfully.")
    print(f"  y shape: {y.shape}")
    print(f"  unique labels: {sorted(np.unique(y).tolist())}")
    print()

    # -------------------------------------------------------------
    # 5. Build metadata and combined training outputs
    # -------------------------------------------------------------
    print("[5/7] Building metadata and combined training outputs...")
    metadata_list = build_metadata_list(training_feature_records)
    X_train, y_train, train_metadata = build_training_matrix_and_labels(training_feature_records)

    assert_true(len(metadata_list) == len(training_feature_records), "metadata_list length mismatch.")
    assert_true(X_train.shape == X.shape, "X_train shape mismatch.")
    assert_true(y_train.shape == y.shape, "y_train shape mismatch.")
    assert_true(len(train_metadata) == len(training_feature_records), "train_metadata length mismatch.")

    print("[OK] Metadata and combined training outputs built successfully.")
    print(f"  metadata count: {len(metadata_list)}")
    print(f"  sample metadata keys: {sorted(list(metadata_list[0].keys()))[:8]}")
    print()

    # -------------------------------------------------------------
    # 6. Test full convenience wrapper for training pipeline
    # -------------------------------------------------------------
    print("[6/7] Testing full training convenience wrapper...")
    X_full, y_full, metadata_full, feature_records_full = prepare_training_matrix_and_labels(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        image_key="image",
    )

    assert_true(X_full.shape == X.shape, "X_full shape mismatch.")
    assert_true(y_full.shape == y.shape, "y_full shape mismatch.")
    assert_true(len(metadata_full) == len(metadata_list), "metadata_full length mismatch.")
    assert_true(len(feature_records_full) == len(training_feature_records), "feature_records_full length mismatch.")

    summary = summarize_feature_records(feature_records_full)
    assert_true(summary["total_count"] == len(feature_records_full), "Summary total_count mismatch.")
    assert_true(expected_feature_length in summary["feature_lengths_present"], "Expected feature length missing in summary.")
    assert_true(summary["has_labels"] is True, "Summary has_labels should be True for training data.")

    print("[OK] Full training convenience wrapper works.")
    print(f"  summary total_count: {summary['total_count']}")
    print(f"  summary feature lengths: {summary['feature_lengths_present']}")
    print(f"  summary labels present: {summary['labels_present']}")
    print()

    # -------------------------------------------------------------
    # 7. Test runtime feature-record and runtime matrix preparation
    # -------------------------------------------------------------
    print("[7/7] Testing runtime feature-record and matrix preparation...")
    runtime_eye_images = [record["image"] for record in training_records[:3]]

    runtime_metadata_input = [
        {"frame_index": 101, "eye_index": 1},
        {"frame_index": 101, "eye_index": 2},
        {"frame_index": 102, "eye_index": 1},
    ]

    runtime_feature_records = prepare_runtime_feature_records(
        runtime_eye_images=runtime_eye_images,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        metadata_list=runtime_metadata_input,
    )

    X_runtime, runtime_metadata = build_runtime_matrix(runtime_feature_records)

    assert_true(len(runtime_feature_records) == len(runtime_eye_images), "runtime_feature_records length mismatch.")
    assert_true(X_runtime.shape[0] == len(runtime_eye_images), "X_runtime row count mismatch.")
    assert_true(X_runtime.shape[1] == expected_feature_length, "X_runtime column count mismatch.")
    assert_true(len(runtime_metadata) == len(runtime_eye_images), "runtime_metadata length mismatch.")

    X_runtime_full, runtime_metadata_full, runtime_feature_records_full = prepare_runtime_matrix(
        runtime_eye_images=runtime_eye_images,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        metadata_list=runtime_metadata_input,
    )

    assert_true(X_runtime_full.shape == X_runtime.shape, "X_runtime_full shape mismatch.")
    assert_true(len(runtime_metadata_full) == len(runtime_metadata), "runtime_metadata_full length mismatch.")
    assert_true(len(runtime_feature_records_full) == len(runtime_feature_records), "runtime_feature_records_full length mismatch.")

    print("[OK] Runtime matrix preparation works.")
    print(f"  X_runtime shape: {X_runtime.shape}")
    print(f"  runtime metadata sample: {runtime_metadata[0]}")
    print()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()