from pathlib import Path

import cv2
import numpy as np

from parking_training_io import (
    load_all_training_records,
    summarize_training_records,
)
from preprocessing import preprocess_all_rois
from lbp_features import extract_lbp_features_from_records


def save_debug_image(image, output_path):
    """
    Save one image to disk and raise a clear error if saving fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise IOError(f"Could not save debug image: {output_path}")


def convert_training_records_to_roi_like_records(training_records):
    """
    Convert training records into a record structure compatible with
    preprocessing.py, which expects 'roi_image' and usually carries
    ROI-style metadata.

    We assign:
    - source_image_name = file_name
    - space_index = 1-based index within this selected subset
    - polygon = None, because training images are already cropped patches
    - roi_image = original loaded training image
    """

    roi_like_records = []

    for index, record in enumerate(training_records, start=1):
        roi_like_record = {
            "source_image_name": record["file_name"],
            "space_index": index,
            "polygon": None,
            "roi_image": record["image"],
            "file_path": record["file_path"],
            "file_name": record["file_name"],
            "class_name": record["class_name"],
            "label": record["label"],
        }
        roi_like_records.append(roi_like_record)

    return roi_like_records


def find_first_n_by_class(training_records, class_name, n):
    """
    Return the first n records with the requested class_name.
    """
    selected = []

    for record in training_records:
        if record["class_name"] == class_name:
            selected.append(record)
            if len(selected) == n:
                break

    return selected


def convert_lbp_image_to_visualization(lbp_image):
    """
    Convert an LBP-coded image into an 8-bit image suitable for visual saving.

    Why this helper exists:
    LBP code images may use integer ranges that are not directly convenient
    for visual inspection. This helper rescales the code range to 0..255.
    """

    lbp_float = lbp_image.astype(np.float32)

    min_value = float(lbp_float.min())
    max_value = float(lbp_float.max())

    if max_value == min_value:
        return np.zeros_like(lbp_float, dtype=np.uint8)

    vis = 255.0 * (lbp_float - min_value) / (max_value - min_value)
    vis = vis.astype(np.uint8)

    return vis


def main():
    project_root = Path(__file__).resolve().parent

    training_root = project_root / "data" / "training"
    output_root = project_root / "outputs" / "lbp" / "step_04"

    print("=== STEP 4 SMOKE TEST: lbp_features ===")
    print(f"Training root: {training_root}")

    # -------------------------------------------------------------------------
    # 1. Load training dataset
    # -------------------------------------------------------------------------
    training_records = load_all_training_records(training_root)
    summary = summarize_training_records(training_records)

    print("\nTraining dataset summary:")
    print(f"  Total samples : {summary['total_count']}")
    print(f"  Free samples  : {summary['free_count']}")
    print(f"  Full samples  : {summary['full_count']}")
    print(f"  Classes found : {summary['class_names_present']}")
    print(f"  Labels found  : {summary['labels_present']}")

    # -------------------------------------------------------------------------
    # 2. Select only a few samples for a small smoke test
    # -------------------------------------------------------------------------
    free_examples = find_first_n_by_class(training_records, "free", 3)
    full_examples = find_first_n_by_class(training_records, "full", 3)

    if len(free_examples) < 3:
        raise ValueError("Not enough 'free' samples for smoke test.")

    if len(full_examples) < 3:
        raise ValueError("Not enough 'full' samples for smoke test.")

    selected_training_records = free_examples + full_examples

    print(f"\nSelected sample count: {len(selected_training_records)}")
    for index, record in enumerate(selected_training_records, start=1):
        print(
            f"  {index}. "
            f"class_name={record['class_name']}, "
            f"label={record['label']}, "
            f"file_name={record['file_name']}, "
            f"image_shape={record['image'].shape}"
        )

    # -------------------------------------------------------------------------
    # 3. Convert training records into ROI-like records so they can be sent
    #    through preprocessing.py
    # -------------------------------------------------------------------------
    roi_like_records = convert_training_records_to_roi_like_records(
        selected_training_records
    )

    # -------------------------------------------------------------------------
    # 4. Preprocess images
    # -------------------------------------------------------------------------
    preprocessing_config = {
        "target_size": (80, 80),
        "contrast_method": "clahe",
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": (8, 8),
        "filter_name": "gaussian",
        "kernel_size": 3,
    }

    preprocessed_records = preprocess_all_rois(
        rois=roi_like_records,
        preprocessing_config=preprocessing_config,
    )

    print(f"\nPreprocessed record count: {len(preprocessed_records)}")

    # -------------------------------------------------------------------------
    # 5. Extract LBP features
    # -------------------------------------------------------------------------
    lbp_config = {
        "neighbors": 8,
        "radius": 1,
        "method": "uniform",
        "grid_shape": (4, 4),
        "normalize_histogram": True,
    }

    lbp_feature_records = extract_lbp_features_from_records(
        preprocessed_records=preprocessed_records,
        lbp_config=lbp_config,
    )

    print(f"LBP feature record count: {len(lbp_feature_records)}")

    # Print descriptor information
    print("\nLBP descriptor info:")
    for index, record in enumerate(lbp_feature_records, start=1):
        feature_vector = record["lbp_feature_vector"]
        lbp_image = record["lbp_image"]

        print(
            f"  {index}. "
            f"class_name={record['class_name']}, "
            f"label={record['label']}, "
            f"file_name={record['file_name']}, "
            f"lbp_shape={lbp_image.shape}, "
            f"feature_length={len(feature_vector)}, "
            f"feature_sum={feature_vector.sum():.4f}, "
            f"lbp_min={lbp_image.min()}, "
            f"lbp_max={lbp_image.max()}"
        )

    # -------------------------------------------------------------------------
    # 6. Save processed images and LBP visualizations
    # -------------------------------------------------------------------------
    for index, record in enumerate(lbp_feature_records, start=1):
        class_name = record["class_name"]
        file_stem = Path(record["file_name"]).stem

        processed_output_path = (
            output_root
            / class_name
            / f"{index:02d}_{file_stem}_processed.jpg"
        )

        lbp_output_path = (
            output_root
            / class_name
            / f"{index:02d}_{file_stem}_lbp.jpg"
        )

        save_debug_image(record["processed_image"], processed_output_path)

        lbp_vis = convert_lbp_image_to_visualization(record["lbp_image"])
        save_debug_image(lbp_vis, lbp_output_path)

    print("\nSaved processed images and LBP visualizations to:")
    print(f"  {output_root}")

    print("\nSmoke test finished successfully.")


if __name__ == "__main__":
    main()