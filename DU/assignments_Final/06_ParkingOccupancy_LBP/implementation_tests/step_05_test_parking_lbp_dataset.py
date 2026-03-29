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

    print("=== STEP 5 SMOKE TEST: parking_lbp_dataset ===")

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
    print(f"  Feature records count    : {training_feature_summary['total_count']}")
    print(f"  Feature lengths present  : {training_feature_summary['feature_lengths_present']}")
    print(f"  Class names present      : {training_feature_summary['class_names_present']}")
    print(f"  Labels present           : {training_feature_summary['labels_present']}")
    print(f"  Has labels               : {training_feature_summary['has_labels']}")

    print("\n[3] Building X_train and y_train...")
    X_train, y_train, training_metadata = build_training_matrix_and_labels(
        training_feature_records=training_feature_records
    )

    print("Training matrix outputs:")
    print(f"  X_train.shape            : {X_train.shape}")
    print(f"  y_train.shape            : {y_train.shape}")
    print(f"  training_metadata length : {len(training_metadata)}")
    print(f"  Unique labels in y_train : {sorted(set(y_train.tolist()))}")

    if X_train.shape[0] != len(training_feature_records):
        raise ValueError("X_train row count does not match training feature record count.")

    if y_train.shape[0] != len(training_feature_records):
        raise ValueError("y_train length does not match training feature record count.")

    if len(training_metadata) != len(training_feature_records):
        raise ValueError("Training metadata length does not match training feature record count.")

    # -------------------------------------------------------------------------
    # 2. TEST SIDE
    # -------------------------------------------------------------------------
    print("\n[4] Loading parking map and test images...")
    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(test_images_dir)

    if not test_cases:
        raise ValueError("No test images were found.")

    first_test_case = test_cases[0]
    test_image_name = first_test_case["name"]
    test_image = first_test_case["image"]

    print(f"  Selected test image      : {test_image_name}")
    print(f"  Parking spaces in map    : {len(parking_map)}")

    print("\n[5] Extracting test ROIs...")
    test_roi_records = extract_all_rois_from_image(
        image=test_image,
        parking_map=parking_map,
        image_name=test_image_name,
    )

    print(f"  Extracted test ROI count : {len(test_roi_records)}")

    print("\n[6] Preparing test feature records...")
    test_feature_records = prepare_test_feature_records(
        test_roi_records=test_roi_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    test_feature_summary = summarize_feature_records(test_feature_records)

    print("Test feature summary:")
    print(f"  Feature records count    : {test_feature_summary['total_count']}")
    print(f"  Feature lengths present  : {test_feature_summary['feature_lengths_present']}")
    print(f"  Source image names       : {test_feature_summary['source_image_names_present']}")
    print(f"  Has labels               : {test_feature_summary['has_labels']}")

    print("\n[7] Building X_test...")
    X_test, test_metadata = build_test_matrix(test_feature_records)

    print("Test matrix outputs:")
    print(f"  X_test.shape             : {X_test.shape}")
    print(f"  test_metadata length     : {len(test_metadata)}")

    if X_test.shape[0] != len(test_feature_records):
        raise ValueError("X_test row count does not match test feature record count.")

    if len(test_metadata) != len(test_feature_records):
        raise ValueError("Test metadata length does not match test feature record count.")

    if len(test_feature_summary["feature_lengths_present"]) != 1:
        raise ValueError("Test feature vectors do not all have the same length.")

    if len(training_feature_summary["feature_lengths_present"]) != 1:
        raise ValueError("Training feature vectors do not all have the same length.")

    training_feature_length = training_feature_summary["feature_lengths_present"][0]
    test_feature_length = test_feature_summary["feature_lengths_present"][0]

    print("\n[8] Cross-checking training/test feature compatibility...")
    print(f"  Training feature length  : {training_feature_length}")
    print(f"  Test feature length      : {test_feature_length}")

    if training_feature_length != test_feature_length:
        raise ValueError(
            "Training and test feature lengths do not match. "
            "Classifier input would be incompatible."
        )

    print("\nSmoke test finished successfully.")
    print("parking_lbp_dataset.py is ready for classifier integration.")


if __name__ == "__main__":
    main()