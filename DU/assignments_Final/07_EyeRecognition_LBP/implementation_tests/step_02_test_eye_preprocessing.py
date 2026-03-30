"""
step_02_test_eye_preprocessing.py

Simple smoke test for eye_preprocessing.py.

This test verifies:
- default preprocessing configuration is valid,
- one training image can be loaded and preprocessed,
- one structured record can be preprocessed,
- multiple records can be preprocessed consistently,
- output image shapes match expectations.

The test intentionally stays lightweight:
- it loads only a small subset of records,
- it focuses on structural correctness, not model quality.
"""

from pathlib import Path

from eye_training_io import (
    load_all_eye_training_records,
    summarize_eye_training_records,
    print_eye_training_summary,
)

from eye_preprocessing import (
    get_default_preprocessing_config,
    validate_preprocessing_config,
    preprocess_one_eye_image,
    preprocess_one_eye_record,
    preprocess_all_eye_records,
    summarize_preprocessed_eye_records,
    print_preprocessed_eye_summary,
)


EXPECTED_PROCESSED_RECORD_KEYS = {
    "preprocessed_image",
    "preprocessed_image_shape",
    "preprocessing_config",
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

    print("=== STEP 02 SMOKE TEST: eye_preprocessing.py ===")
    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print()

    # -------------------------------------------------------------
    # 1. Load a small subset of training records with images
    # -------------------------------------------------------------
    print("[1/6] Loading a small subset of training records with images...")
    all_records = load_all_eye_training_records(
        dataset_root,
        load_images=True,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False
    )

    assert_true(len(all_records) > 0, "No training records were loaded.")

    subset_size = min(10, len(all_records))
    records = all_records[:subset_size]

    print(f"[OK] Loaded {len(all_records)} total records.")
    print(f"[OK] Using subset of {subset_size} records for preprocessing smoke test.")
    print()

    # -------------------------------------------------------------
    # 2. Validate default preprocessing configuration
    # -------------------------------------------------------------
    print("[2/6] Validating default preprocessing configuration...")
    default_config = get_default_preprocessing_config()
    validated_config = validate_preprocessing_config(default_config)

    assert_true(isinstance(validated_config, dict), "Validated config must be a dictionary.")
    assert_true("target_size" in validated_config, "Validated config is missing 'target_size'.")
    assert_true("crop_analysis_band" in validated_config, "Validated config is missing 'crop_analysis_band'.")
    assert_true("contrast_method" in validated_config, "Validated config is missing 'contrast_method'.")
    assert_true("filter_name" in validated_config, "Validated config is missing 'filter_name'.")

    print("[OK] Default preprocessing configuration is valid.")
    print(f"  target_size:          {validated_config['target_size']}")
    print(f"  crop_analysis_band:   {validated_config['crop_analysis_band']}")
    print(f"  analysis_top_ratio:   {validated_config['analysis_top_ratio']}")
    print(f"  analysis_bottom_ratio:{validated_config['analysis_bottom_ratio']}")
    print(f"  contrast_method:      {validated_config['contrast_method']}")
    print(f"  filter_name:          {validated_config['filter_name']}")
    print()

    # -------------------------------------------------------------
    # 3. Preprocess one raw image
    # -------------------------------------------------------------
    print("[3/6] Preprocessing one raw eye image...")
    first_record = records[0]
    raw_image = first_record["image"]

    assert_true(raw_image is not None, "First record does not contain a loaded image.")

    processed_image = preprocess_one_eye_image(
        raw_image,
        preprocessing_config=validated_config
    )

    assert_true(processed_image is not None, "Processed image is None.")
    assert_true(processed_image.size > 0, "Processed image is empty.")
    assert_true(len(processed_image.shape) == 2, "Processed image must be grayscale (2D).")

    expected_height = int(round(
        validated_config["target_size"][1]
        * (validated_config["analysis_bottom_ratio"] - validated_config["analysis_top_ratio"])
    )) if validated_config["crop_analysis_band"] else validated_config["target_size"][1]

    expected_width = validated_config["target_size"][0]

    assert_true(
        processed_image.shape[1] == expected_width,
        f"Processed image width mismatch: expected {expected_width}, got {processed_image.shape[1]}"
    )

    print("[OK] One raw eye image preprocessed successfully.")
    print(f"  original_shape:   {raw_image.shape}")
    print(f"  processed_shape:  {processed_image.shape}")
    print()

    # -------------------------------------------------------------
    # 4. Preprocess one structured record
    # -------------------------------------------------------------
    print("[4/6] Preprocessing one structured training record...")
    processed_record = preprocess_one_eye_record(
        first_record,
        preprocessing_config=validated_config,
        image_key="image"
    )

    missing_keys = EXPECTED_PROCESSED_RECORD_KEYS - set(processed_record.keys())
    assert_true(not missing_keys, f"Processed record is missing keys: {sorted(missing_keys)}")

    assert_true(processed_record["preprocessed_image"] is not None, "Processed record image is None.")
    assert_true(
        tuple(processed_record["preprocessed_image"].shape) == tuple(processed_record["preprocessed_image_shape"]),
        "preprocessed_image_shape does not match actual image shape."
    )

    print("[OK] One structured record preprocessed successfully.")
    print(f"  file_name:                {processed_record['file_name']}")
    print(f"  class_name:               {processed_record['class_name']}")
    print(f"  preprocessed_image_shape: {processed_record['preprocessed_image_shape']}")
    print()

    # -------------------------------------------------------------
    # 5. Preprocess multiple records
    # -------------------------------------------------------------
    print("[5/6] Preprocessing multiple records...")
    processed_records = preprocess_all_eye_records(
        records,
        preprocessing_config=validated_config,
        image_key="image"
    )

    assert_true(len(processed_records) == len(records), "Processed record count does not match input record count.")

    reference_shape = tuple(processed_records[0]["preprocessed_image_shape"])

    for index, record in enumerate(processed_records):
        assert_true(record["preprocessed_image"] is not None, f"Processed image missing in record {index}.")
        assert_true(
            tuple(record["preprocessed_image_shape"]) == tuple(record["preprocessed_image"].shape),
            f"Shape mismatch in record {index}."
        )
        assert_true(
            tuple(record["preprocessed_image_shape"]) == reference_shape,
            f"Inconsistent preprocessed shape in record {index}: "
            f"expected {reference_shape}, got {record['preprocessed_image_shape']}"
        )

    print("[OK] Multiple records preprocessed consistently.")
    print(f"  processed_record_count: {len(processed_records)}")
    print(f"  common_output_shape:    {reference_shape}")
    print()

    # -------------------------------------------------------------
    # 6. Print summaries
    # -------------------------------------------------------------
    print("[6/6] Printing dataset and preprocessing summaries...")
    raw_summary = summarize_eye_training_records(records)
    preprocessed_summary = summarize_preprocessed_eye_records(processed_records)

    print()
    print_eye_training_summary(raw_summary)
    print()
    print_preprocessed_eye_summary(preprocessed_summary)
    print()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()