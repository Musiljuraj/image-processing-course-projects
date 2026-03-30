"""
step_01_test_eye_training_io.py

Simple smoke test for eye_training_io.py.

This test verifies:
- dataset directory exists,
- image files can be discovered,
- one file name can be parsed correctly,
- one image can be loaded correctly,
- one structured record can be built,
- the whole dataset can be loaded into structured records,
- the summary function returns consistent counts.

The test is intentionally simple and safe:
- it loads only one image into memory for direct inspection,
- it loads the full dataset with load_images=False to keep the test light.
"""

from pathlib import Path

from eye_training_io import (
    collect_eye_image_paths,
    parse_eye_filename,
    load_one_eye_training_image,
    build_one_eye_training_record,
    load_all_eye_training_records,
    summarize_eye_training_records,
    print_eye_training_summary,
)


EXPECTED_RECORD_KEYS = {
    "file_path",
    "relative_path",
    "file_name",
    "subject_dir",
    "subject_id",
    "image_id",
    "gender",
    "glasses",
    "eye_state",
    "reflections",
    "lighting",
    "sensor_id",
    "label",
    "class_name",
    "image",
    "image_shape",
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

    print("=== STEP 01 SMOKE TEST: eye_training_io.py ===")
    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print()

    # -----------------------------------------------------------------
    # 1. Dataset root existence
    # -----------------------------------------------------------------
    print("[1/6] Checking dataset directory...")
    assert_true(dataset_root.exists(), f"Dataset directory does not exist: {dataset_root}")
    assert_true(dataset_root.is_dir(), f"Dataset path is not a directory: {dataset_root}")
    print("[OK] Dataset directory exists.")
    print()

    # -----------------------------------------------------------------
    # 2. Collect image paths
    # -----------------------------------------------------------------
    print("[2/6] Collecting eye image paths...")
    image_paths = collect_eye_image_paths(dataset_root, recursive=True)

    assert_true(len(image_paths) > 0, "No image files were found in the dataset.")
    print(f"[OK] Found {len(image_paths)} image files.")

    preview_count = min(3, len(image_paths))
    print("First discovered files:")
    for path in image_paths[:preview_count]:
        print(f"  - {path.relative_to(dataset_root)}")
    print()

    # -----------------------------------------------------------------
    # 3. Parse first filename
    # -----------------------------------------------------------------
    print("[3/6] Parsing metadata from first file name...")
    first_image_path = image_paths[0]
    parsed = parse_eye_filename(first_image_path.name)

    assert_true(parsed["label"] in (0, 1), "Parsed label must be 0 or 1.")
    assert_true(parsed["class_name"] in ("close", "open"), "Parsed class_name must be 'close' or 'open'.")
    assert_true(str(parsed["subject_id"]).startswith("s"), "Parsed subject_id should start with 's'.")

    print("[OK] File name parsed successfully.")
    print(f"  file_name:   {first_image_path.name}")
    print(f"  subject_id:  {parsed['subject_id']}")
    print(f"  image_id:    {parsed['image_id']}")
    print(f"  eye_state:   {parsed['eye_state']}")
    print(f"  label:       {parsed['label']}")
    print(f"  class_name:  {parsed['class_name']}")
    print()

    # -----------------------------------------------------------------
    # 4. Load first image
    # -----------------------------------------------------------------
    print("[4/6] Loading first image in grayscale...")
    image = load_one_eye_training_image(first_image_path, grayscale=True)

    assert_true(image is not None, "Loaded image is None.")
    assert_true(image.size > 0, "Loaded image is empty.")
    assert_true(len(image.shape) == 2, "Expected grayscale image with 2 dimensions.")

    print("[OK] First image loaded successfully.")
    print(f"  shape:  {image.shape}")
    print(f"  dtype:  {image.dtype}")
    print()

    # -----------------------------------------------------------------
    # 5. Build one structured record
    # -----------------------------------------------------------------
    print("[5/6] Building one structured training record...")
    record = build_one_eye_training_record(
        first_image_path,
        dataset_root=dataset_root,
        load_image=True,
        grayscale=True
    )

    missing_keys = EXPECTED_RECORD_KEYS - set(record.keys())
    assert_true(not missing_keys, f"Record is missing keys: {sorted(missing_keys)}")
    assert_true(record["label"] in (0, 1), "Record label must be 0 or 1.")
    assert_true(record["class_name"] in ("close", "open"), "Record class_name must be 'close' or 'open'.")
    assert_true(record["image"] is not None, "Record image should be loaded in this step.")
    assert_true(record["image_shape"] is not None, "Record image_shape should be present in this step.")

    print("[OK] One structured record built successfully.")
    print(f"  relative_path: {record['relative_path']}")
    print(f"  subject_id:    {record['subject_id']}")
    print(f"  label:         {record['label']}")
    print(f"  class_name:    {record['class_name']}")
    print(f"  image_shape:   {record['image_shape']}")
    print()

    # -----------------------------------------------------------------
    # 6. Load all records and summarize
    # -----------------------------------------------------------------
    print("[6/6] Loading all records (without image arrays) and checking summary...")
    records = load_all_eye_training_records(
        dataset_root,
        load_images=False,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False
    )

    assert_true(len(records) == len(image_paths), "Record count should match discovered image count.")

    for index, item in enumerate(records):
        assert_true(item["label"] in (0, 1), f"Invalid label in record {index}: {item['label']}")
        assert_true(item["class_name"] in ("close", "open"), f"Invalid class_name in record {index}: {item['class_name']}")
        assert_true(item["image"] is None, f"Record {index} should not contain loaded image in this step.")
        assert_true(item["file_name"], f"Record {index} has empty file_name.")

    summary = summarize_eye_training_records(records)

    assert_true(summary["total_count"] == len(records), "Summary total_count does not match record count.")
    assert_true(
        summary["class_counts"]["close"] + summary["class_counts"]["open"] == len(records),
        "Summary class counts do not add up to total record count."
    )
    assert_true(summary["subject_count"] > 0, "Summary subject_count should be greater than 0.")

    print("[OK] Full dataset loaded successfully.")
    print()
    print_eye_training_summary(summary)
    print()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()