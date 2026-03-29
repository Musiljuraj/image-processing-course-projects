from pathlib import Path
import ast
import csv

import cv2
import numpy as np

from parking_io import load_parking_map, load_test_images
from parking_training_io import load_all_training_records
from roi_extraction import extract_all_rois_from_image
from parking_lbp_dataset import (
    prepare_training_feature_records,
    prepare_test_feature_records,
    build_training_matrix_and_labels,
    build_test_matrix,
)
from parking_lbp_classifier import (
    train_classifier,
    predict_labels,
    predict_scores,
    build_prediction_records,
)
from evaluation import evaluate_one_test_case
from debug_utils import draw_parking_map, save_overlay_image, save_processed_patches


def parse_flat_value(value):
    """
    Parse one CSV cell value back into a Python value when possible.

    Examples:
        "8"        -> 8
        "1.0"      -> 1.0
        "(4, 4)"   -> (4, 4)
        "True"     -> True
        "uniform"  -> "uniform"
    """

    if value is None:
        return None

    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if stripped == "":
        return ""

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped


def parse_prefixed_config_from_row(row, prefix):
    """
    Extract one config dictionary from a flattened CSV row.

    Example:
        prefix="lbp"
        row keys:
            lbp_neighbors=8
            lbp_radius=1
            lbp_method='uniform'

        output:
            {
                "neighbors": 8,
                "radius": 1,
                "method": "uniform",
            }
    """

    if not isinstance(row, dict):
        raise TypeError("row must be a dictionary.")

    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string.")

    prefix_with_sep = f"{prefix}_"
    config = {}

    for key, value in row.items():
        if not key.startswith(prefix_with_sep):
            continue

        short_key = key[len(prefix_with_sep):]
        config[short_key] = parse_flat_value(value)

    return config


def load_best_result_row(csv_path):
    """
    Load the first row from the ranked experiment CSV.

    Why first row:
    The final CSV is saved from ranked_results, so the first row is the
    best-ranked configuration.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Results CSV contains no data rows: {csv_path}")

    return rows[0]


def select_test_case(test_cases, test_case_name=None):
    """
    Select one test case by name, or return the first one by default.
    """

    if not isinstance(test_cases, list):
        raise TypeError("test_cases must be a list.")

    if not test_cases:
        raise ValueError("test_cases must not be empty.")

    if test_case_name is None:
        return test_cases[0]

    if not isinstance(test_case_name, str):
        raise TypeError("test_case_name must be a string or None.")

    normalized_name = test_case_name.strip()

    for test_case in test_cases:
        if test_case["name"] == normalized_name:
            return test_case

    raise ValueError(f"Could not find test case with name: {test_case_name}")


def convert_lbp_image_to_visualization(lbp_image):
    """
    Convert an LBP-coded image into an 8-bit visualization suitable for saving.
    """

    lbp_float = lbp_image.astype(np.float32)

    min_value = float(lbp_float.min())
    max_value = float(lbp_float.max())

    if max_value == min_value:
        return np.zeros_like(lbp_float, dtype=np.uint8)

    vis = 255.0 * (lbp_float - min_value) / (max_value - min_value)
    return vis.astype(np.uint8)


def save_lbp_visualizations(feature_records, output_dir):
    """
    Save one LBP visualization per feature record.

    Filenames follow the same convention as save_processed_patches(...):
        space_01.jpg
        space_02.jpg
        ...
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not feature_records:
        return []

    digits = max(2, len(str(len(feature_records))))
    saved_paths = []

    for record in feature_records:
        if "space_index" not in record:
            raise KeyError("Each feature record must contain 'space_index'.")

        if "lbp_image" not in record:
            raise KeyError("Each feature record must contain 'lbp_image'.")

        lbp_vis = convert_lbp_image_to_visualization(record["lbp_image"])

        output_path = output_dir / f"space_{record['space_index']:0{digits}d}.jpg"
        ok = cv2.imwrite(str(output_path), lbp_vis)

        if not ok:
            raise IOError(f"Could not save LBP visualization: {output_path}")

        saved_paths.append(output_path)

    return saved_paths


def save_evaluated_records_report(evaluated_records, report_path):
    """
    Save a readable per-space inspection report.

    Each line contains:
    - space index
    - predicted label
    - ground-truth label
    - evaluation outcome
    - optional score
    """

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("Best Configuration Inspection Report")
    lines.append("=" * 80)
    lines.append("")

    for record in evaluated_records:
        space_index = record.get("space_index", "<missing>")
        predicted_label = record.get("predicted_label", "<missing>")
        ground_truth_label = record.get("ground_truth_label", "<missing>")
        evaluation_outcome = record.get("evaluation_outcome", "<missing>")
        predicted_score = record.get("predicted_score", None)

        line = (
            f"space_index={space_index}, "
            f"predicted_label={predicted_label}, "
            f"ground_truth_label={ground_truth_label}, "
            f"evaluation_outcome={evaluation_outcome}"
        )

        if predicted_score is not None:
            line += f", predicted_score={predicted_score:.6f}"

        lines.append(line)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def inspect_best_configuration(
    results_csv_path,
    training_root,
    map_path,
    test_images_dir,
    output_dir,
    test_case_name=None,
):
    """
    Rerun the best-ranked configuration on one selected test image and save
    inspection outputs.

    Inputs:
        results_csv_path ... ranked CSV produced by main.py / results_io.py
        training_root .... path to data/training
        map_path ......... path to data/parking_map_python.txt
        test_images_dir ... path to data/test_images_zao
        output_dir ....... destination directory for inspection outputs
        test_case_name ... optional test image name such as "test1"

    Return:
        inspection_result . dictionary containing:
                            - best_row
                            - preprocessing_config
                            - lbp_config
                            - classifier_config
                            - evaluation_config
                            - selected_test_case_name
                            - image_evaluation
                            - saved_paths
    """

    results_csv_path = Path(results_csv_path)
    training_root = Path(training_root)
    map_path = Path(map_path)
    test_images_dir = Path(test_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. load best-ranked row and reconstruct configs
    # -------------------------------------------------------------------------
    best_row = load_best_result_row(results_csv_path)

    preprocessing_config = parse_prefixed_config_from_row(best_row, "preprocessing")
    lbp_config = parse_prefixed_config_from_row(best_row, "lbp")
    classifier_config = parse_prefixed_config_from_row(best_row, "classifier")
    evaluation_config = parse_prefixed_config_from_row(best_row, "evaluation")

    if not evaluation_config:
        evaluation_config = {
            "occupied_label": 1,
            "empty_label": 0,
        }

    # -------------------------------------------------------------------------
    # 2. load shared inputs
    # -------------------------------------------------------------------------
    training_records = load_all_training_records(training_root)
    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(test_images_dir)
    test_case = select_test_case(test_cases, test_case_name=test_case_name)

    # -------------------------------------------------------------------------
    # 3. prepare training data and train classifier
    # -------------------------------------------------------------------------
    training_feature_records = prepare_training_feature_records(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    X_train, y_train, training_metadata = build_training_matrix_and_labels(
        training_feature_records
    )

    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=classifier_config,
    )

    # -------------------------------------------------------------------------
    # 4. prepare selected test image and predict
    # -------------------------------------------------------------------------
    roi_records = extract_all_rois_from_image(
        image=test_case["image"],
        parking_map=parking_map,
        image_name=test_case["name"],
    )

    test_feature_records = prepare_test_feature_records(
        test_roi_records=roi_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    X_test, test_metadata = build_test_matrix(test_feature_records)

    predicted_labels = predict_labels(model=model, X_test=X_test)
    predicted_scores = predict_scores(model=model, X_test=X_test)

    prediction_records = build_prediction_records(
        feature_records=test_feature_records,
        predicted_labels=predicted_labels,
        predicted_scores=predicted_scores,
    )

    image_evaluation = evaluate_one_test_case(
        prediction_records=prediction_records,
        txt_path=test_case["txt_path"],
        evaluation_config=evaluation_config,
    )

    evaluated_records = image_evaluation["evaluated_records"]

    # -------------------------------------------------------------------------
    # 5. save visual outputs
    # -------------------------------------------------------------------------
    overlay_dir = output_dir / "overlay"
    roi_dir = output_dir / "roi"
    processed_dir = output_dir / "processed"
    lbp_dir = output_dir / "lbp"
    report_dir = output_dir / "report"

    overlay_image = draw_parking_map(test_case["image"], parking_map)
    overlay_path = overlay_dir / f"{test_case['name']}_overlay.jpg"
    save_overlay_image(overlay_image, overlay_path)

    roi_paths = save_processed_patches(
        records=roi_records,
        output_dir=roi_dir,
        image_key="roi_image",
    )

    processed_paths = save_processed_patches(
        records=test_feature_records,
        output_dir=processed_dir,
        image_key="processed_image",
    )

    lbp_paths = save_lbp_visualizations(
        feature_records=test_feature_records,
        output_dir=lbp_dir,
    )

    report_path = save_evaluated_records_report(
        evaluated_records=evaluated_records,
        report_path=report_dir / f"{test_case['name']}_inspection_report.txt",
    )

    inspection_result = {
        "best_row": best_row,
        "preprocessing_config": preprocessing_config,
        "lbp_config": lbp_config,
        "classifier_config": classifier_config,
        "evaluation_config": evaluation_config,
        "selected_test_case_name": test_case["name"],
        "image_evaluation": image_evaluation,
        "saved_paths": {
            "overlay_path": overlay_path,
            "roi_paths": roi_paths,
            "processed_paths": processed_paths,
            "lbp_paths": lbp_paths,
            "report_path": report_path,
        },
        "training_metadata_count": len(training_metadata),
        "test_metadata_count": len(test_metadata),
    }

    return inspection_result


def main():
    """
    Default entry point for best-configuration inspection.
    """

    project_root = Path(__file__).resolve().parent

    results_csv_path = (
        project_root
        / "outputs"
        / "results"
        / "final_run"
        / "parking_lbp_results.csv"
    )

    training_root = project_root / "data" / "training"
    map_path = project_root / "data" / "parking_map_python.txt"
    test_images_dir = project_root / "data" / "test_images_zao"
    output_dir = project_root / "outputs" / "inspection" / "best_config"

    inspection_result = inspect_best_configuration(
        results_csv_path=results_csv_path,
        training_root=training_root,
        map_path=map_path,
        test_images_dir=test_images_dir,
        output_dir=output_dir,
        test_case_name=None,   # default: first test image in natural order
    )

    image_evaluation = inspection_result["image_evaluation"]
    confusion_counts = image_evaluation["confusion_counts"]

    print("=== BEST CONFIG INSPECTION ===")
    print(f"selected_test_case_name : {inspection_result['selected_test_case_name']}")
    print(f"accuracy                : {image_evaluation['accuracy']:.6f}")
    print(f"num_samples             : {image_evaluation['num_samples']}")
    print(f"tp                      : {confusion_counts['tp']}")
    print(f"tn                      : {confusion_counts['tn']}")
    print(f"fp                      : {confusion_counts['fp']}")
    print(f"fn                      : {confusion_counts['fn']}")
    print(f"overlay_path            : {inspection_result['saved_paths']['overlay_path']}")
    print(f"report_path             : {inspection_result['saved_paths']['report_path']}")
    print("Inspection finished successfully.")


if __name__ == "__main__":
    main()