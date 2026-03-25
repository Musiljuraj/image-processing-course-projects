#!/usr/bin/python3

"""
Fundamentals of Image Processing - Parking Occupancy Assignment
Step 6: Occupancy classification and evaluation using edge-ratio thresholding.

Current purpose of this version:
1. locate the project folders
2. load parking-space geometry from data/parking_map_python.txt
3. load all test images from data/test_images_zao/
4. run the full pipeline for every test image:
   - extract ROI patches
   - preprocess ROI patches
   - detect edges
   - compute edge statistics
   - load ground truth
   - classify occupied / empty from edge ratio
   - evaluate predictions
5. accumulate TP / TN / FP / FN across the whole dataset
6. compute final accuracy
7. optionally save selected debug outputs for one chosen image

Important:
This version now performs a real dataset evaluation.
The debug outputs remain optional and are mainly intended for visual inspection
of one selected image while the full evaluation runs over all test images.

Design note:
The configuration block now controls:
- preprocessing
- edge detection
- classification/evaluation
- optional debug outputs
"""

from pathlib import Path

from parking_io import load_parking_map, load_test_images, load_ground_truth_labels
from roi_extraction import extract_all_rois_from_image
from preprocessing import preprocess_all_rois
from debug_utils import draw_parking_map, save_overlay_image, save_processed_patches
from edge_detection import detect_edges_all_records
from evaluation import (
    classify_all_edge_records,
    compute_accuracy,
    evaluate_one_image,
    initialize_confusion_counts,
    merge_confusion_counts,
)

def build_processed_subdir_name(preprocessing_config):
    """
    Build a compact subdirectory name describing the preprocessing configuration.

    Input:
        preprocessing_config ... dictionary containing filter_name and kernel_size

    Return:
        processed_subdir_name ... string such as:
                                  - "none"
                                  - "gaussian_k5"
                                  - "median_k7"

    Why this helper exists:
    The main evaluation loop should stay readable. Naming logic for output
    folders is easier to maintain if it is kept in a small helper.
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

    Why this helper exists:
    The directory name should reflect the actual validated detector config used.
    This makes saved debug outputs much easier to compare later.
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


def main():
    """
    Main orchestration function for the full evaluation stage.

    Overall logic:
    1. define project paths
    2. verify expected input locations exist
    3. load parking-space map
    4. load all test images
    5. initialize dataset-level confusion counts
    6. loop through all test images
    7. for each image:
       - optionally create selected debug outputs
       - extract ROI patches
       - preprocess ROI patches
       - detect edges and compute edge statistics
       - load ground-truth labels
       - classify occupancy using edge-ratio threshold
       - evaluate predictions for that image
       - merge the image confusion counts into dataset totals
    8. compute final dataset accuracy
    9. print a complete evaluation summary

    This function intentionally stays high-level.
    The detailed work is delegated to helper modules:
    - parking_io.py
    - roi_extraction.py
    - preprocessing.py
    - edge_detection.py
    - evaluation.py
    - debug_utils.py
    """

    # ------------------------------------------------------------------
    # CONFIGURATION BLOCK
    # ------------------------------------------------------------------
    # Select which image will be used for optional debug-output saving.
    # The full evaluation still runs across all test images.
    debug_image_index = 0

    # Preprocessing configuration.
    # Supported filter_name values:
    # - "none"
    # - "box"
    # - "gaussian"
    # - "median"
    preprocessing_config = {
        "filter_name": "gaussian",
        "kernel_size": 5,
    }

    # Edge-detection configuration.
    #
    # Supported detector_name values:
    # - "sobel"
    # - "canny"
    edge_detection_config = {
        "detector_name": "canny",
        "sobel": {
            "ksize": 3,
            "threshold": 100,
        },
        "canny": {
            "threshold1": 50,
            "threshold2": 150,
            "aperture_size": 3,
            "l2gradient": False,
        },
    }

    # Classification/evaluation configuration.
    #
    # Decision rule used later:
    #   if edge_ratio > occupancy_threshold_ratio -> occupied
    #   else -> empty
    #
    # Default label convention:
    # - occupied_label = 1
    # - empty_label = 0
    #
    # If your ground-truth files use the opposite convention, change these
    # values here instead of changing the evaluation code.
    classification_evaluation_config = {
        "occupancy_threshold_ratio": 0.08,
        "occupied_label": 1,
        "empty_label": 0,
    }

    # Debug-output configuration.
    # The full evaluation runs on all images, but debug saves are performed
    # only for the one image selected by debug_image_index.
    debug_options = {
        "save_overlay": True,
        "save_raw_rois": False,
        "save_grayscale_rois": False,
        "save_processed_rois": True,
        "save_edge_maps": True,
    }

    # Optional GUI preview of the full overlay image for the selected debug image.
    # Keep disabled on headless systems / remote terminals / WSL setups.
    show_window = False
    # ------------------------------------------------------------------
    # END OF CONFIGURATION BLOCK
    # ------------------------------------------------------------------


    # determine the project root directory
    project_root = Path(__file__).resolve().parent

    # define the input-data locations
    data_dir = project_root / "data"
    map_path = data_dir / "parking_map_python.txt"
    images_dir = data_dir / "test_images_zao"

    # define the output locations
    outputs_dir = project_root / "outputs"
    debug_dir = outputs_dir / "debug"
    rois_dir = outputs_dir / "rois"
    preprocessed_dir = outputs_dir / "preprocessed"
    edges_dir = outputs_dir / "edges"

    # create only the top-level output directory eagerly
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # perform basic existence checks for expected inputs
    if not map_path.exists():
        raise FileNotFoundError(f"Parking map file not found: {map_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {images_dir}")

    # load parking-space geometry
    parking_map = load_parking_map(map_path)

    # load all test images and verify matching .txt files exist
    test_cases = load_test_images(images_dir)

    if not test_cases:
        raise RuntimeError(f"No .jpg files found in {images_dir}")

    # validate chosen debug image index
    if not (0 <= debug_image_index < len(test_cases)):
        raise IndexError(
            f"debug_image_index={debug_image_index} is out of range for "
            f"{len(test_cases)} loaded test images."
        )

    # print overall run configuration
    print(f"Project root: {project_root}")
    print(f"Map file: {map_path}")
    print(f"Images directory: {images_dir}")
    print(f"Loaded {len(parking_map)} parking spaces.")
    print(f"Loaded {len(test_cases)} test images.")
    print(
        "Preprocessing config: "
        f"filter = {preprocessing_config['filter_name']}, "
        f"kernel_size = {preprocessing_config['kernel_size']}"
    )
    print(f"Edge detection config: {edge_detection_config}")
    print(f"Classification/evaluation config: {classification_evaluation_config}")
    print(f"Debug options: {debug_options}")

    dataset_confusion_counts = initialize_confusion_counts()
    per_image_summaries = []
    total_num_samples = 0


    # the main flow is  a full dataset loop 
    for image_index, test_case in enumerate(test_cases):
        print("-" * 72)
        print(
            f"Processing image {image_index + 1}/{len(test_cases)}: "
            f"{test_case['name']}"
        )

        is_debug_image = (image_index == debug_image_index)

        # --------------------------------------------------------------
        # PART 1: optionally build and save a full parking-lot overlay
        # --------------------------------------------------------------
        overlay = None
        if is_debug_image and (debug_options["save_overlay"] or show_window):
            overlay = draw_parking_map(test_case["image"], parking_map)

            if debug_options["save_overlay"]:
                debug_dir.mkdir(parents=True, exist_ok=True)
                overlay_path = (
                    debug_dir / f"{test_case['name']}_parking_map_overlay.jpg"
                )
                save_overlay_image(overlay, overlay_path)
                print(f"  Saved full-lot debug overlay to: {overlay_path}")

        # --------------------------------------------------------------
        # PART 2: extract all parking-space ROIs from the current image
        # --------------------------------------------------------------
        rois = extract_all_rois_from_image(
            image=test_case["image"],
            parking_map=parking_map,
            image_name=test_case["name"],
        )

        print(f"  Extracted {len(rois)} ROI patches.")

        if is_debug_image and debug_options["save_raw_rois"]:
            raw_roi_output_dir = rois_dir / test_case["name"]
            saved_raw_roi_paths = save_processed_patches(
                records=rois,
                output_dir=raw_roi_output_dir,
                image_key="roi_image",
            )
            print(f"  Saved {len(saved_raw_roi_paths)} raw ROI patches to: {raw_roi_output_dir}")

        # --------------------------------------------------------------
        # PART 3: preprocess all ROI patches
        # --------------------------------------------------------------
        preprocessed_rois = preprocess_all_rois(
            rois=rois,
            preprocessing_config=preprocessing_config,
        )

        print(f"  Preprocessed {len(preprocessed_rois)} ROI patches.")

        processed_subdir_name = build_processed_subdir_name(preprocessing_config)

        if is_debug_image and debug_options["save_grayscale_rois"]:
            grayscale_output_dir = preprocessed_dir / test_case["name"] / "grayscale"
            saved_grayscale_paths = save_processed_patches(
                records=preprocessed_rois,
                output_dir=grayscale_output_dir,
                image_key="grayscale_image",
            )
            print(
                f"  Saved {len(saved_grayscale_paths)} grayscale ROI patches to: "
                f"{grayscale_output_dir}"
            )

        if is_debug_image and debug_options["save_processed_rois"]:
            processed_output_dir = (
                preprocessed_dir / test_case["name"] / processed_subdir_name
            )
            saved_processed_paths = save_processed_patches(
                records=preprocessed_rois,
                output_dir=processed_output_dir,
                image_key="processed_image",
            )
            print(
                f"  Saved {len(saved_processed_paths)} processed ROI patches to: "
                f"{processed_output_dir}"
            )

        # --------------------------------------------------------------
        # PART 4: run edge detection on all processed ROI patches
        # --------------------------------------------------------------
        edge_records = detect_edges_all_records(
            preprocessed_records=preprocessed_rois,
            edge_detection_config=edge_detection_config,
        )

        print(f"  Detected edges for {len(edge_records)} ROI patches.")

        edge_subdir_name = build_edge_subdir_name(edge_records)

        if is_debug_image and debug_options["save_edge_maps"]:
            edge_output_dir = edges_dir / test_case["name"] / edge_subdir_name
            saved_edge_paths = save_processed_patches(
                records=edge_records,
                output_dir=edge_output_dir,
                image_key="edge_image",
            )
            print(f"  Saved {len(saved_edge_paths)} edge maps to: {edge_output_dir}")

        # --------------------------------------------------------------
        # PART 5: load ground-truth labels for the current image
        # --------------------------------------------------------------
        ground_truth_labels = load_ground_truth_labels(test_case["txt_path"])

        print(f"  Loaded {len(ground_truth_labels)} ground-truth labels.")

        # --------------------------------------------------------------
        # PART 6: classify all parking spaces using edge ratio
        # --------------------------------------------------------------
        classified_records = classify_all_edge_records(
            edge_records=edge_records,
            classification_evaluation_config=classification_evaluation_config,
        )

        print(f"  Classified {len(classified_records)} parking spaces.")

        # --------------------------------------------------------------
        # PART 7: evaluate predictions for the current image
        # --------------------------------------------------------------
        image_evaluation = evaluate_one_image(
            classified_records=classified_records,
            ground_truth_labels=ground_truth_labels,
            classification_evaluation_config=classification_evaluation_config,
        )

        image_confusion_counts = image_evaluation["confusion_counts"]
        image_accuracy = image_evaluation["accuracy"]
        num_samples = image_evaluation["num_samples"]

        total_num_samples += num_samples
        dataset_confusion_counts = merge_confusion_counts(
            dataset_confusion_counts,
            image_confusion_counts,
        )

        per_image_summary = {
            "image_name": test_case["name"],
            "num_samples": num_samples,
            "tp": image_confusion_counts["tp"],
            "tn": image_confusion_counts["tn"],
            "fp": image_confusion_counts["fp"],
            "fn": image_confusion_counts["fn"],
            "accuracy": image_accuracy,
        }
        per_image_summaries.append(per_image_summary)

        print(
            f"  Image evaluation: "
            f"TP={image_confusion_counts['tp']} | "
            f"TN={image_confusion_counts['tn']} | "
            f"FP={image_confusion_counts['fp']} | "
            f"FN={image_confusion_counts['fn']} | "
            f"accuracy={image_accuracy:.4f}"
        )

        # --------------------------------------------------------------
        # PART 8: optional detailed preview for the selected debug image
        # --------------------------------------------------------------
        if is_debug_image:
            preview_count = min(10, len(image_evaluation["evaluated_records"]))
            print(f"  Preview of first {preview_count} evaluated records:")
            for record in image_evaluation["evaluated_records"][:preview_count]:
                print(
                    f"    space {record['space_index']:02d} | "
                    f"edge_count = {record['edge_count']} | "
                    f"roi_pixel_count = {record['roi_pixel_count']} | "
                    f"edge_ratio = {record['edge_ratio']:.6f} | "
                    f"predicted = {record['predicted_label']} | "
                    f"ground_truth = {record['ground_truth_label']} | "
                    f"outcome = {record['evaluation_outcome']}"
                )

            if show_window and overlay is not None:
                import cv2

                cv2.imshow("Parking map overlay", overlay)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # final dataset-level evaluation summary added in this stage
    final_accuracy = compute_accuracy(dataset_confusion_counts)

    print("=" * 72)
    print("FINAL DATASET EVALUATION SUMMARY")
    print(f"Total evaluated parking spaces: {total_num_samples}")
    print(f"TP = {dataset_confusion_counts['tp']}")
    print(f"TN = {dataset_confusion_counts['tn']}")
    print(f"FP = {dataset_confusion_counts['fp']}")
    print(f"FN = {dataset_confusion_counts['fn']}")
    print(f"Accuracy = {final_accuracy:.6f}")

    print("=" * 72)
    print("PER-IMAGE SUMMARY")
    for summary in per_image_summaries:
        print(
            f"{summary['image_name']}: "
            f"samples={summary['num_samples']} | "
            f"TP={summary['tp']} | "
            f"TN={summary['tn']} | "
            f"FP={summary['fp']} | "
            f"FN={summary['fn']} | "
            f"accuracy={summary['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()