"""
step_03_test_lbp_features.py

Simple smoke test for lbp_features.py.

This test verifies:
- a small subset of training records can be loaded,
- the subset can be preprocessed,
- the default LBP configuration is valid,
- one preprocessed eye image can be converted into an LBP image and descriptor,
- one preprocessed record can be extended with LBP outputs,
- multiple records can be processed consistently,
- the resulting feature lengths and histogram-bin counts are stable.

The test intentionally remains lightweight:
- only a small subset of dataset records is used,
- it checks structural correctness, not classification quality.
"""

from pathlib import Path

from eye_training_io import load_all_eye_training_records
from eye_preprocessing import preprocess_all_eye_records
from lbp_features import (
    get_default_lbp_config,
    validate_lbp_config,
    get_lbp_histogram_bin_count,
    compute_lbp_image,
    compute_spatial_lbp_descriptor,
    extract_lbp_features_from_record,
    extract_lbp_features_from_records,
    summarize_lbp_feature_records,
    print_lbp_feature_summary,
)


EXPECTED_LBP_RECORD_KEYS = {
    "lbp_image",
    "lbp_feature_vector",
    "lbp_histogram_bin_count",
    "lbp_feature_length",
    "lbp_config",
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

    print("=== STEP 03 SMOKE TEST: lbp_features.py ===")
    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print()

    # -------------------------------------------------------------
    # 1. Load a small subset of records with images
    # -------------------------------------------------------------
    print("[1/6] Loading a small subset of training records...")
    all_records = load_all_eye_training_records(
        dataset_root,
        load_images=True,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False
    )

    assert_true(len(all_records) > 0, "No training records were loaded.")

    subset_size = min(10, len(all_records))
    raw_records = all_records[:subset_size]

    print(f"[OK] Loaded {len(all_records)} total records.")
    print(f"[OK] Using subset of {subset_size} records.")
    print()

    # -------------------------------------------------------------
    # 2. Preprocess the subset
    # -------------------------------------------------------------
    print("[2/6] Preprocessing the subset...")
    preprocessed_records = preprocess_all_eye_records(raw_records)

    assert_true(len(preprocessed_records) == subset_size, "Preprocessed record count mismatch.")

    first_preprocessed_image = preprocessed_records[0]["preprocessed_image"]
    assert_true(first_preprocessed_image is not None, "First preprocessed image is None.")
    assert_true(first_preprocessed_image.ndim == 2, "Preprocessed image must be 2D grayscale.")

    print("[OK] Preprocessing completed successfully.")
    print(f"  first preprocessed image shape: {first_preprocessed_image.shape}")
    print()

    # -------------------------------------------------------------
    # 3. Validate default LBP configuration
    # -------------------------------------------------------------
    print("[3/6] Validating default LBP configuration...")
    default_lbp_config = get_default_lbp_config()
    validated_lbp_config = validate_lbp_config(default_lbp_config)

    assert_true(isinstance(validated_lbp_config, dict), "Validated LBP config must be a dictionary.")

    neighbors = validated_lbp_config["neighbors"]
    radius = validated_lbp_config["radius"]
    method = validated_lbp_config["method"]
    grid_shape = validated_lbp_config["grid_shape"]
    normalize_histogram = validated_lbp_config["normalize_histogram"]

    bin_count = get_lbp_histogram_bin_count(neighbors, method)
    grid_rows, grid_cols = grid_shape
    expected_feature_length = bin_count * grid_rows * grid_cols

    assert_true(bin_count > 0, "Histogram bin count must be positive.")
    assert_true(expected_feature_length > 0, "Expected feature length must be positive.")

    print("[OK] Default LBP configuration is valid.")
    print(f"  neighbors:              {neighbors}")
    print(f"  radius:                 {radius}")
    print(f"  method:                 {method}")
    print(f"  grid_shape:             {grid_shape}")
    print(f"  normalize_histogram:    {normalize_histogram}")
    print(f"  histogram bin count:    {bin_count}")
    print(f"  expected feature length:{expected_feature_length}")
    print()

    # -------------------------------------------------------------
    # 4. Compute LBP for one preprocessed image
    # -------------------------------------------------------------
    print("[4/6] Computing LBP image and descriptor for one preprocessed eye image...")
    lbp_image = compute_lbp_image(
        gray_image=first_preprocessed_image,
        neighbors=neighbors,
        radius=radius,
        method=method,
    )

    descriptor_lbp_image, descriptor_vector = compute_spatial_lbp_descriptor(
        gray_image=first_preprocessed_image,
        neighbors=neighbors,
        radius=radius,
        method=method,
        grid_shape=grid_shape,
        normalize_histogram=normalize_histogram,
    )

    assert_true(lbp_image is not None, "LBP image is None.")
    assert_true(lbp_image.shape == first_preprocessed_image.shape, "LBP image shape mismatch.")
    assert_true(descriptor_lbp_image.shape == first_preprocessed_image.shape, "Descriptor LBP image shape mismatch.")
    assert_true(descriptor_vector.ndim == 1, "LBP descriptor must be 1D.")
    assert_true(len(descriptor_vector) == expected_feature_length, "Unexpected descriptor length.")

    print("[OK] One preprocessed image converted successfully.")
    print(f"  input shape:            {first_preprocessed_image.shape}")
    print(f"  lbp image shape:        {lbp_image.shape}")
    print(f"  descriptor length:      {len(descriptor_vector)}")
    print()

    # -------------------------------------------------------------
    # 5. Extract LBP features from one preprocessed record
    # -------------------------------------------------------------
    print("[5/6] Extracting LBP features from one preprocessed record...")
    first_lbp_record = extract_lbp_features_from_record(
        preprocessed_record=preprocessed_records[0],
        lbp_config=validated_lbp_config,
        image_key="preprocessed_image"
    )

    missing_keys = EXPECTED_LBP_RECORD_KEYS - set(first_lbp_record.keys())
    assert_true(not missing_keys, f"LBP feature record is missing keys: {sorted(missing_keys)}")

    assert_true(first_lbp_record["lbp_image"] is not None, "lbp_image is None.")
    assert_true(first_lbp_record["lbp_feature_vector"] is not None, "lbp_feature_vector is None.")
    assert_true(first_lbp_record["lbp_image"].shape == first_preprocessed_image.shape, "Record lbp_image shape mismatch.")
    assert_true(
        first_lbp_record["lbp_feature_length"] == expected_feature_length,
        "Record feature length mismatch."
    )
    assert_true(
        len(first_lbp_record["lbp_feature_vector"]) == expected_feature_length,
        "Record vector length mismatch."
    )
    assert_true(
        first_lbp_record["lbp_histogram_bin_count"] == bin_count,
        "Record histogram bin count mismatch."
    )

    print("[OK] One record converted to LBP features successfully.")
    print(f"  file_name:              {first_lbp_record['file_name']}")
    print(f"  class_name:             {first_lbp_record['class_name']}")
    print(f"  lbp image shape:        {first_lbp_record['lbp_image'].shape}")
    print(f"  feature length:         {first_lbp_record['lbp_feature_length']}")
    print()

    # -------------------------------------------------------------
    # 6. Extract LBP features from multiple records and summarize
    # -------------------------------------------------------------
    print("[6/6] Extracting LBP features from multiple records and checking consistency...")
    lbp_feature_records = extract_lbp_features_from_records(
        preprocessed_records=preprocessed_records,
        lbp_config=validated_lbp_config,
        image_key="preprocessed_image"
    )

    assert_true(len(lbp_feature_records) == len(preprocessed_records), "LBP feature record count mismatch.")

    for index, record in enumerate(lbp_feature_records):
        assert_true(record["lbp_image"] is not None, f"Missing lbp_image in record {index}.")
        assert_true(record["lbp_feature_vector"] is not None, f"Missing lbp_feature_vector in record {index}.")
        assert_true(record["lbp_image"].ndim == 2, f"lbp_image must be 2D in record {index}.")
        assert_true(record["lbp_feature_vector"].ndim == 1, f"lbp_feature_vector must be 1D in record {index}.")
        assert_true(
            record["lbp_feature_length"] == expected_feature_length,
            f"Unexpected feature length in record {index}."
        )
        assert_true(
            len(record["lbp_feature_vector"]) == expected_feature_length,
            f"Unexpected feature vector size in record {index}."
        )
        assert_true(
            record["lbp_histogram_bin_count"] == bin_count,
            f"Unexpected histogram bin count in record {index}."
        )

    summary = summarize_lbp_feature_records(lbp_feature_records)

    assert_true(summary["total_count"] == len(lbp_feature_records), "Summary total_count mismatch.")
    assert_true(
        expected_feature_length in summary["feature_length_counts"],
        "Expected feature length is missing from summary."
    )
    assert_true(
        bin_count in summary["histogram_bin_count_counts"],
        "Expected histogram bin count is missing from summary."
    )

    print("[OK] Multiple records converted consistently.")
    print()
    print_lbp_feature_summary(summary)
    print()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()