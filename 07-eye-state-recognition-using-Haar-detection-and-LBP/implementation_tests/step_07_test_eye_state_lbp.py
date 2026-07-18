"""
step_07_test_eye_state_lbp.py

Simple smoke test for eye_state_lbp.py.

This test verifies:
- the startup-trained LBP model bundle can be built,
- one runtime eye can be classified with classify_single_eye_lbp(...),
- one frame can be classified with classify_eye_state_lbp(...),
- two-eye frame aggregation works structurally,
- deterministic fallback behavior works when no eyes are present.

The test intentionally stays lightweight:
- it uses a very small subset of the training dataset,
- it treats dataset eye images as fake runtime frames,
- it checks structural correctness, not final model quality.
"""

from pathlib import Path
import cv2

from eye_training_io import load_all_eye_training_records
from eye_preprocessing import get_default_preprocessing_config
from lbp_features import get_default_lbp_config
from eye_lbp_classifier import (
    get_default_classifier_config,
    build_eye_lbp_model,
)
from eye_state_lbp import (
    classify_single_eye_lbp,
    classify_eye_state_lbp,
    aggregate_eye_predictions,
)


def assert_true(condition, message):
    """
    Raise AssertionError with a readable message when the condition is false.
    """
    if not condition:
        raise AssertionError(message)


def build_full_image_eye_box(gray_image):
    """
    Build one eye box covering the full image.

    This is sufficient for smoke testing because the dataset images are already
    cropped eye images.
    """
    height, width = gray_image.shape[:2]
    return (0, 0, width, height)


def build_two_eye_test_frame(left_eye_image, right_eye_image):
    """
    Build one fake frame containing two eye images side by side.

    Returns:
    - combined_frame
    - left_eye_box
    - right_eye_box
    """
    left_height, left_width = left_eye_image.shape[:2]
    right_height, right_width = right_eye_image.shape[:2]

    target_height = max(left_height, right_height)

    if left_height != target_height:
        left_eye_image = cv2.resize(
            left_eye_image,
            (left_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        left_height, left_width = left_eye_image.shape[:2]

    if right_height != target_height:
        right_eye_image = cv2.resize(
            right_eye_image,
            (right_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        right_height, right_width = right_eye_image.shape[:2]

    combined_frame = cv2.hconcat([left_eye_image, right_eye_image])

    left_eye_box = (0, 0, left_width, target_height)
    right_eye_box = (left_width, 0, right_width, target_height)

    return combined_frame, left_eye_box, right_eye_box


def main():
    project_root = Path(__file__).resolve().parent
    dataset_root = project_root / "input" / "training" / "mrlEyes_2018_01"

    print("=== STEP 07 SMOKE TEST: eye_state_lbp.py ===")
    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print()

    preprocessing_config = get_default_preprocessing_config()
    lbp_config = get_default_lbp_config()
    classifier_config = get_default_classifier_config()

    # -------------------------------------------------------------
    # 1. Build the startup-trained model bundle
    # -------------------------------------------------------------
    print("[1/5] Building startup-trained model bundle...")
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
    assert_true("model" in model_bundle, "model_bundle is missing 'model'.")
    assert_true("preprocessing_config" in model_bundle, "model_bundle is missing 'preprocessing_config'.")
    assert_true("lbp_config" in model_bundle, "model_bundle is missing 'lbp_config'.")
    assert_true(model_bundle["training_sample_count"] > 0, "training_sample_count must be positive.")

    print("[OK] Model bundle built successfully.")
    print(f"  training_sample_count: {model_bundle['training_sample_count']}")
    print(f"  feature_count:         {model_bundle['feature_count']}")
    print()

    # -------------------------------------------------------------
    # 2. Load a small sample of eye images for fake runtime testing
    # -------------------------------------------------------------
    print("[2/5] Loading a small sample of eye images...")
    all_records = load_all_eye_training_records(
        dataset_root=dataset_root,
        load_images=True,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False,
    )

    assert_true(len(all_records) > 0, "No records were loaded.")

    close_record = next((record for record in all_records if record["label"] == 0), None)
    open_record = next((record for record in all_records if record["label"] == 1), None)

    assert_true(close_record is not None, "Could not find one close-eye sample.")
    assert_true(open_record is not None, "Could not find one open-eye sample.")

    close_image = close_record["image"]
    open_image = open_record["image"]

    assert_true(close_image is not None, "close_image is None.")
    assert_true(open_image is not None, "open_image is None.")

    print("[OK] Sample runtime images loaded.")
    print(f"  close sample: {close_record['file_name']}")
    print(f"  open sample:  {open_record['file_name']}")
    print()

    # -------------------------------------------------------------
    # 3. Test classify_single_eye_lbp(...)
    # -------------------------------------------------------------
    print("[3/5] Testing classify_single_eye_lbp(...)...")
    single_eye_box = build_full_image_eye_box(open_image)

    single_eye_prediction = classify_single_eye_lbp(
        gray_frame=open_image,
        eye_box=single_eye_box,
        model_bundle=model_bundle,
        eye_index=1,
        frame_index=1,
    )

    assert_true(isinstance(single_eye_prediction, dict), "single_eye_prediction must be a dictionary.")
    assert_true(single_eye_prediction["success"] is True, "single_eye_prediction should succeed.")
    assert_true(single_eye_prediction["predicted_label"] in (0, 1), "predicted_label must be 0 or 1.")
    assert_true(single_eye_prediction["predicted_class_name"] in ("open", "close"), "predicted_class_name must be open/close.")
    assert_true(single_eye_prediction["predicted_open_score"] is not None, "predicted_open_score must not be None.")
    assert_true(
        0.0 <= float(single_eye_prediction["predicted_open_score"]) <= 1.0,
        "predicted_open_score must be in [0, 1]."
    )

    print("[OK] classify_single_eye_lbp(...) works.")
    print(f"  predicted_label:      {single_eye_prediction['predicted_label']}")
    print(f"  predicted_class_name: {single_eye_prediction['predicted_class_name']}")
    print(f"  predicted_open_score: {single_eye_prediction['predicted_open_score']:.4f}")
    print()

    # -------------------------------------------------------------
    # 4. Test classify_eye_state_lbp(...) on one-eye and two-eye frames
    # -------------------------------------------------------------
    print("[4/5] Testing classify_eye_state_lbp(...) and aggregation...")

    one_eye_face_parts = {
        "eyes": [build_full_image_eye_box(close_image)],
        "mouth": [],
    }

    one_eye_label, one_eye_details = classify_eye_state_lbp(
        gray_frame=close_image,
        face_parts=one_eye_face_parts,
        model_bundle=model_bundle,
        previous_eye_state=None,
        fallback_to_heuristic=False,
        frame_index=2,
        return_details=True,
    )

    assert_true(one_eye_label in ("open", "close"), "one_eye_label must be open/close.")
    assert_true(isinstance(one_eye_details, dict), "one_eye_details must be a dictionary.")
    assert_true("eye_prediction_records" in one_eye_details, "one_eye_details missing eye_prediction_records.")
    assert_true("aggregation" in one_eye_details, "one_eye_details missing aggregation.")
    assert_true(len(one_eye_details["eye_prediction_records"]) == 1, "Expected exactly one eye prediction record.")
    assert_true(one_eye_details["aggregation"]["valid_eye_count"] in (0, 1), "Unexpected valid_eye_count for one-eye frame.")

    two_eye_frame, left_eye_box, right_eye_box = build_two_eye_test_frame(close_image, open_image)
    two_eye_face_parts = {
        "eyes": [left_eye_box, right_eye_box],
        "mouth": [],
    }

    two_eye_label, two_eye_details = classify_eye_state_lbp(
        gray_frame=two_eye_frame,
        face_parts=two_eye_face_parts,
        model_bundle=model_bundle,
        previous_eye_state="close",
        fallback_to_heuristic=False,
        frame_index=3,
        return_details=True,
    )

    assert_true(two_eye_label in ("open", "close"), "two_eye_label must be open/close.")
    assert_true(isinstance(two_eye_details, dict), "two_eye_details must be a dictionary.")
    assert_true(len(two_eye_details["eye_prediction_records"]) == 2, "Expected exactly two eye prediction records.")
    assert_true(
        0 <= two_eye_details["aggregation"]["valid_eye_count"] <= 2,
        "Unexpected valid_eye_count for two-eye frame."
    )

    aggregated_label, aggregated_details = aggregate_eye_predictions(
        eye_prediction_records=two_eye_details["eye_prediction_records"],
        previous_eye_state="close",
        fallback_frame_label="close",
    )

    assert_true(aggregated_label in ("open", "close"), "aggregated_label must be open/close.")
    assert_true(isinstance(aggregated_details, dict), "aggregated_details must be a dictionary.")

    print("[OK] classify_eye_state_lbp(...) and aggregation work.")
    print(f"  one_eye_label:        {one_eye_label}")
    print(f"  two_eye_label:        {two_eye_label}")
    print(f"  aggregated_label:     {aggregated_label}")
    print(f"  valid_eye_count(two): {two_eye_details['aggregation']['valid_eye_count']}")
    print()

    # -------------------------------------------------------------
    # 5. Test deterministic fallback behavior with no eyes
    # -------------------------------------------------------------
    print("[5/5] Testing fallback behavior with no detected eyes...")
    no_eye_face_parts = {"eyes": [], "mouth": []}

    fallback_label, fallback_details = classify_eye_state_lbp(
        gray_frame=open_image,
        face_parts=no_eye_face_parts,
        model_bundle=model_bundle,
        previous_eye_state="open",
        fallback_to_heuristic=False,
        fallback_frame_label="close",
        frame_index=4,
        return_details=True,
    )

    assert_true(fallback_label == "open", "Fallback with previous_eye_state='open' should return 'open'.")
    assert_true(isinstance(fallback_details, dict), "fallback_details must be a dictionary.")
    assert_true(fallback_details["aggregation"]["valid_eye_count"] == 0, "Expected valid_eye_count == 0 for no-eye case.")
    assert_true(fallback_details["used_heuristic_fallback"] is False, "Heuristic fallback should not be used here.")

    print("[OK] Deterministic fallback behavior works.")
    print(f"  fallback_label: {fallback_label}")
    print()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()