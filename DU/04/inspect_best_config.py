#!/usr/bin/python3

"""
inspect_best_config.py

Purpose of this script:
- load the best-ranked configuration found by the exhaustive search
- apply it to one selected test image
- save debug/inspection outputs for that one image

Why this script exists:
The final exhaustive-search version is optimized for:
- testing many configurations
- computing metrics
- saving ranked results

That is excellent for selecting the best configuration, but it is not ideal for
visual inspection. This script provides the complementary workflow:
- pick the best configuration from outputs/results/results.csv
- run one image through the full pipeline
- save visual outputs and a per-space text summary

Typical outputs of this script:
- full parking-lot overlay image
- optionally raw ROI patches
- optionally grayscale ROI patches
- processed ROI patches
- edge maps
- text summary per parking space:
  - edge_count
  - roi_pixel_count
  - edge_ratio
  - predicted_label
  - ground_truth_label
  - evaluation_outcome

Design note:
This script is intentionally self-contained. It reads the best configuration
directly from results.csv, so no change to results_io.py is required.
"""

import csv
from pathlib import Path

from parking_io import load_parking_map, load_test_images, load_ground_truth_labels
from roi_extraction import extract_all_rois_from_image
from preprocessing import preprocess_all_rois
from edge_detection import detect_edges_all_records
from evaluation import classify_all_edge_records, evaluate_one_image
from debug_utils import draw_parking_map, save_overlay_image, save_processed_patches


def parse_bool_string(value):
    """
    Parse a string representation of a boolean.

    Input:
        value ... expected values such as:
                  "True", "False", "true", "false", "1", "0"

    Return:
        parsed_bool ... Python bool

    Why this helper exists:
    Values read from CSV are strings, so detector configuration reconstruction
    needs a reliable bool parser.
    """

    normalized_value = str(value).strip().lower()

    if normalized_value in ("true", "1", "yes"):
        return True

    if normalized_value in ("false", "0", "no"):
        return False

    raise ValueError(f"Cannot parse boolean value from: {value}")


def parse_optional_int(value):
    """
    Parse an optional integer from a CSV field.

    Input:
        value ... string value from CSV, may be empty

    Return:
        int value or None
    """

    if value is None or value == "":
        return None

    return int(value)


def load_best_result_from_csv(csv_path):
    """
    Load and reconstruct the best-ranked configuration from results.csv.

    Input:
        csv_path ... path to outputs/results/results.csv

    Return:
        best_result ... dictionary containing:
                        - preprocessing_config
                        - edge_detection_config
                        - classification_evaluation_config
                        - rank
                        - experiment_index
                        - accuracy
                        - tp
                        - tn
                        - fp
                        - fn
                        - num_samples
                        - num_images

    Assumption:
    The CSV rows are already saved in ranked order, so the first data row
    corresponds to the best-ranked configuration.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Results CSV is empty: {csv_path}")

    row = rows[0]

    filter_name = row["filter_name"]
    kernel_size = parse_optional_int(row["kernel_size"])
    detector_name = row["detector_name"]

    preprocessing_config = {
        "filter_name": filter_name,
        "kernel_size": kernel_size,
    }

    if detector_name == "sobel":
        edge_detection_config = {
            "detector_name": "sobel",
            "sobel": {
                "ksize": int(row["sobel_ksize"]),
                "threshold": float(row["sobel_threshold"]),
            },
        }

    elif detector_name == "canny":
        edge_detection_config = {
            "detector_name": "canny",
            "canny": {
                "threshold1": float(row["canny_threshold1"]),
                "threshold2": float(row["canny_threshold2"]),
                "aperture_size": int(row["canny_aperture_size"]),
                "l2gradient": parse_bool_string(row["canny_l2gradient"]),
            },
        }

    else:
        raise ValueError(f"Unsupported detector_name in CSV: {detector_name}")

    classification_evaluation_config = {
        "occupancy_threshold_ratio": float(row["occupancy_threshold_ratio"]),
        "occupied_label": int(row["occupied_label"]),
        "empty_label": int(row["empty_label"]),
    }

    best_result = {
        "rank": int(row["rank"]),
        "experiment_index": int(row["experiment_index"]),
        "accuracy": float(row["accuracy"]),
        "tp": int(row["tp"]),
        "tn": int(row["tn"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "num_samples": int(row["num_samples"]),
        "num_images": int(row["num_images"]),
        "preprocessing_config": preprocessing_config,
        "edge_detection_config": edge_detection_config,
        "classification_evaluation_config": classification_evaluation_config,
    }

    return best_result


def build_processed_subdir_name(preprocessing_config):
    """
    Build a compact subdirectory name describing the preprocessing configuration.

    Input:
        preprocessing_config ... dictionary containing filter_name and kernel_size

    Return:
        processed_subdir_name ... string such as:
                                  - "none"
                                  - "gaussian_k5"
                                  - "median_k3"
    """

    normalized_filter_name = preprocessing_config["filter_name"].strip().lower()

    if normalized_filter_name == "none":
        return "none"

    return f"{normalized_filter_name}_k{preprocessing_config['kernel_size']}"


def build_edge_subdir_name(edge_records):
    """
    Build a compact subdirectory name describing the edge-detection configuration.

    Input:
        edge_records ... list of edge-detection records

    Return:
        edge_subdir_name ... string such as:
                             - "canny_t1_50_t2_150_a3"
                             - "sobel_k3_t100"
    """

    if not edge_records:
        return "unknown_detector"

    used_edge_config = edge_records[0]["edge_detection_config"]
    used_detector_name = used_edge_config["detector_name"]

    if used_detector_name == "canny":
        canny_cfg = used_edge_config["canny"]
        return (
            f"canny_t1_{int(canny_cfg['threshold1'])}"
            f"_t2_{int(canny_cfg['threshold2'])}"
            f"_a{canny_cfg['aperture_size']}"
        )

    if used_detector_name == "sobel":
        sobel_cfg = used_edge_config["sobel"]
        return f"sobel_k{sobel_cfg['ksize']}_t{int(sobel_cfg['threshold'])}"

    return used_detector_name


def save_evaluation_summary_text(image_evaluation, output_path):
    """
    Save a human-readable per-space evaluation summary for one inspected image.

    Inputs:
        image_evaluation ... dictionary returned by evaluate_one_image(...)
        output_path ....... full destination path

    Why this helper exists:
    Visual outputs are useful, but for understanding errors you also want a
    compact text summary showing the per-space statistics and outcomes.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluated_records = image_evaluation["evaluated_records"]
    confusion_counts = image_evaluation["confusion_counts"]
    accuracy = image_evaluation["accuracy"]
    num_samples = image_evaluation["num_samples"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("INSPECTION SUMMARY FOR BEST CONFIGURATION\n")
        f.write("=" * 72 + "\n")
        f.write(f"Num samples: {num_samples}\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"TP: {confusion_counts['tp']}\n")
        f.write(f"TN: {confusion_counts['tn']}\n")
        f.write(f"FP: {confusion_counts['fp']}\n")
        f.write(f"FN: {confusion_counts['fn']}\n")
        f.write("\n")

        f.write("PER-SPACE DETAILS\n")
        f.write("-" * 72 + "\n")

        for record in evaluated_records:
            f.write(
                f"space={record['space_index']:02d} | "
                f"edge_count={record['edge_count']} | "
                f"roi_pixel_count={record['roi_pixel_count']} | "
                f"edge_ratio={record['edge_ratio']:.6f} | "
                f"predicted={record['predicted_label']} | "
                f"ground_truth={record['ground_truth_label']} | "
                f"outcome={record['evaluation_outcome']}\n"
            )


def main():
    """
    Main function for one-image visual inspection of the best configuration.

    Overall logic:
    1. locate the project and results folders
    2. load the best-ranked configuration from outputs/results/results.csv
    3. load the parking map and test images
    4. choose one image for inspection
    5. run the full pipeline for that image only
    6. save debug outputs
    7. save a per-space text summary
    """

    # ------------------------------------------------------------------
    # CONFIGURATION BLOCK
    # ------------------------------------------------------------------
    # Which image should be inspected.
    # This index refers to the naturally sorted list returned by load_test_images(...).
    debug_image_index = 0

    # Inspection-output configuration.
    inspect_options = {
        "save_overlay": True,
        "save_raw_rois": False,
        "save_grayscale_rois": False,
        "save_processed_rois": True,
        "save_edge_maps": True,
        "save_text_summary": True,
    }

    # Optional GUI preview of the overlay image.
    show_window = False
    # ------------------------------------------------------------------
    # END OF CONFIGURATION BLOCK
    # ------------------------------------------------------------------

    project_root = Path(__file__).resolve().parent

    data_dir = project_root / "data"
    map_path = data_dir / "parking_map_python.txt"
    images_dir = data_dir / "test_images_zao"

    outputs_dir = project_root / "outputs"
    results_dir = outputs_dir / "results"
    inspection_root_dir = outputs_dir / "inspection" / "best_config"

    results_csv_path = results_dir / "results.csv"

    if not map_path.exists():
        raise FileNotFoundError(f"Parking map file not found: {map_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {images_dir}")

    if not results_csv_path.exists():
        raise FileNotFoundError(
            f"results.csv not found. Run the exhaustive search first: {results_csv_path}"
        )

    best_result = load_best_result_from_csv(results_csv_path)

    preprocessing_config = best_result["preprocessing_config"]
    edge_detection_config = best_result["edge_detection_config"]
    classification_evaluation_config = best_result[
        "classification_evaluation_config"
    ]

    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(images_dir)

    if not test_cases:
        raise RuntimeError(f"No .jpg files found in {images_dir}")

    if not (0 <= debug_image_index < len(test_cases)):
        raise IndexError(
            f"debug_image_index={debug_image_index} is out of range for "
            f"{len(test_cases)} loaded test images."
        )

    selected_case = test_cases[debug_image_index]

    print(f"Project root: {project_root}")
    print(f"Selected image for inspection: {selected_case['name']}")
    print(f"Best rank: {best_result['rank']}")
    print(f"Best experiment index: {best_result['experiment_index']}")
    print(f"Best accuracy: {best_result['accuracy']:.6f}")
    print(f"Best preprocessing config: {preprocessing_config}")
    print(f"Best edge-detection config: {edge_detection_config}")
    print(
        "Best classification/evaluation config: "
        f"{classification_evaluation_config}"
    )

    image_output_dir = inspection_root_dir / selected_case["name"]
    image_output_dir.mkdir(parents=True, exist_ok=True)

    overlay = draw_parking_map(selected_case["image"], parking_map)

    if inspect_options["save_overlay"]:
        overlay_output_dir = image_output_dir / "overlay"
        overlay_path = overlay_output_dir / f"{selected_case['name']}_parking_map_overlay.jpg"
        save_overlay_image(overlay, overlay_path)
        print(f"Saved overlay to: {overlay_path}")

    rois = extract_all_rois_from_image(
        image=selected_case["image"],
        parking_map=parking_map,
        image_name=selected_case["name"],
    )

    if inspect_options["save_raw_rois"]:
        raw_roi_output_dir = image_output_dir / "raw_rois"
        saved_raw_roi_paths = save_processed_patches(
            records=rois,
            output_dir=raw_roi_output_dir,
            image_key="roi_image",
        )
        print(f"Saved {len(saved_raw_roi_paths)} raw ROI patches to: {raw_roi_output_dir}")

    preprocessed_rois = preprocess_all_rois(
        rois=rois,
        preprocessing_config=preprocessing_config,
    )

    processed_subdir_name = build_processed_subdir_name(preprocessing_config)

    if inspect_options["save_grayscale_rois"]:
        grayscale_output_dir = image_output_dir / "grayscale"
        saved_grayscale_paths = save_processed_patches(
            records=preprocessed_rois,
            output_dir=grayscale_output_dir,
            image_key="grayscale_image",
        )
        print(f"Saved {len(saved_grayscale_paths)} grayscale ROI patches to: {grayscale_output_dir}")

    if inspect_options["save_processed_rois"]:
        processed_output_dir = image_output_dir / "processed" / processed_subdir_name
        saved_processed_paths = save_processed_patches(
            records=preprocessed_rois,
            output_dir=processed_output_dir,
            image_key="processed_image",
        )
        print(f"Saved {len(saved_processed_paths)} processed ROI patches to: {processed_output_dir}")

    edge_records = detect_edges_all_records(
        preprocessed_records=preprocessed_rois,
        edge_detection_config=edge_detection_config,
    )

    edge_subdir_name = build_edge_subdir_name(edge_records)

    if inspect_options["save_edge_maps"]:
        edge_output_dir = image_output_dir / "edges" / edge_subdir_name
        saved_edge_paths = save_processed_patches(
            records=edge_records,
            output_dir=edge_output_dir,
            image_key="edge_image",
        )
        print(f"Saved {len(saved_edge_paths)} edge maps to: {edge_output_dir}")

    ground_truth_labels = load_ground_truth_labels(selected_case["txt_path"])

    classified_records = classify_all_edge_records(
        edge_records=edge_records,
        classification_evaluation_config=classification_evaluation_config,
    )

    image_evaluation = evaluate_one_image(
        classified_records=classified_records,
        ground_truth_labels=ground_truth_labels,
        classification_evaluation_config=classification_evaluation_config,
    )

    if inspect_options["save_text_summary"]:
        summary_path = image_output_dir / "inspection_summary.txt"
        save_evaluation_summary_text(image_evaluation, summary_path)
        print(f"Saved inspection summary to: {summary_path}")

    confusion_counts = image_evaluation["confusion_counts"]
    accuracy = image_evaluation["accuracy"]

    print("-" * 72)
    print("INSPECTION RESULT FOR SELECTED IMAGE")
    print(f"TP = {confusion_counts['tp']}")
    print(f"TN = {confusion_counts['tn']}")
    print(f"FP = {confusion_counts['fp']}")
    print(f"FN = {confusion_counts['fn']}")
    print(f"Accuracy = {accuracy:.6f}")

    preview_count = min(10, len(image_evaluation["evaluated_records"]))
    print(f"Preview of first {preview_count} evaluated records:")
    for record in image_evaluation["evaluated_records"][:preview_count]:
        print(
            f"  space {record['space_index']:02d} | "
            f"edge_count = {record['edge_count']} | "
            f"roi_pixel_count = {record['roi_pixel_count']} | "
            f"edge_ratio = {record['edge_ratio']:.6f} | "
            f"predicted = {record['predicted_label']} | "
            f"ground_truth = {record['ground_truth_label']} | "
            f"outcome = {record['evaluation_outcome']}"
        )

    if show_window:
        import cv2

        cv2.imshow("Parking map overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()