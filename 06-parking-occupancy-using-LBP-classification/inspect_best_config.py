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

# ---------------------------------------------------------------------------
# Module orientation:
# This module is a post-search inspection utility. After the main experiment
# search has ranked configurations and saved them to CSV, this module reloads
# the best-ranked configuration, reconstructs its parameter dictionaries,
# reruns that exact pipeline on one selected test image, evaluates the result,
# and saves a collection of human-readable inspection outputs. Its purpose is
# not to search for the best setup again, but to make the already-selected best
# setup understandable at the level of individual parking spaces and image
# representations.
# ---------------------------------------------------------------------------
#
# This file is a second top-level workflow that branches off after the main
# experiment search has already finished.
#
# The main experiment pipeline answers:
# - which configuration performed best overall?
#
# This inspection pipeline answers the next practical question:
# - what does that best configuration actually do on one concrete test image?
#
# So this module reconstructs the already selected winning setup and reruns the
# exact same processing chain, but now with inspection-friendly outputs:
# - one overlay showing the numbered parking-space map on the full image
# - one folder of raw ROI patches
# - one folder of processed grayscale patches
# - one folder of LBP visualizations
# - one text report describing the prediction/evaluation outcome for each space
#
# In other words, this module does not compete configurations anymore.
# It makes the chosen configuration interpretable.


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

    # The ranked CSV saved by results_io.py flattens everything into text cells.
    # This helper reverses that flattening as much as safely possible so config
    # dictionaries can later be reconstructed in their original Python-friendly
    # form.
    #
    # The conversion intentionally uses ast.literal_eval(...) rather than eval(...)
    # so that literal numbers, tuples, and booleans can be restored while keeping
    # the parsing safe and restricted to literal values.

    # CSV rows store all values as text, so this helper reverses that flattening as much
    # as is safely possible. Literal numbers, tuples, and booleans are restored, while
    # plain free-form strings remain strings.
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

    # One ranked CSV row contains multiple flattened config blocks mixed together.
    # For example, preprocessing fields, LBP fields, classifier fields, evaluation
    # fields, and metric fields all share the same row.
    #
    # This helper extracts only the subset belonging to one config block by:
    # 1. selecting keys with the requested prefix
    # 2. stripping that prefix away
    # 3. parsing each stored text cell back into a Python value
    #
    # The result is a reconstructed config dictionary that closely matches the
    # original nested config structure used during the experiment run.

    # Each saved CSV row contains multiple flattened configuration blocks mixed together.
    # The prefix convention created in results_io.py allows this helper to rebuild one
    # original config block by selecting only the keys that start with the requested
    # prefix and then stripping that prefix from the key names.
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

    # This helper defines the contract between the main experiment outputs and the
    # inspection utility:
    # - results_io.py saves ranked experiment rows in best-to-worst order
    # - therefore the first data row is the winning configuration
    #
    # The function does not reinterpret ranking. It simply trusts the already saved
    # ranked CSV and returns the first row unchanged for later reconstruction.

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    # The saved experiment CSV is already sorted by rank, so the first data row is the
    # best-ranked configuration according to the project ranking rule. This helper simply
    # loads that row and returns it unchanged.
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

    # The inspection workflow usually focuses on one concrete full-scene test
    # image. This helper makes that selection explicit.
    #
    # If no name is provided, the first test case in the already sorted list is
    # used as a deterministic default. If a name is provided, it must match one
    # of the loaded test-case records exactly.

    # The inspection utility typically needs only one test image. If the caller does not
    # specify which one, the first image in the already-sorted list is used as the
    # default. This keeps the helper convenient for quick inspection.
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

    # LBP images store descriptor codes, not ordinary viewable brightness values.
    # That means they are meaningful numerically but are not directly convenient for
    # visual inspection.
    #
    # This helper rescales the numeric LBP-code range into 0-255 so the result can
    # be saved as a normal grayscale image and inspected by eye. This is only for
    # visualization; it does not affect the actual classifier features.

    # LBP codes are numerical descriptors, not directly intended for normal image
    # display. This helper rescales the code values to the 0-255 range so the result can
    # be saved as a human-viewable image for debugging and inspection.
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

    # This helper is the LBP-specific counterpart to the more generic patch-saving
    # helpers in debug_utils.py.
    #
    # The feature records already contain one lbp_image per parking space, but that
    # lbp_image must first be converted into a human-viewable 8-bit image. After
    # that conversion, the function saves one visualization file per parking space
    # using the same stable "space_XX.jpg" naming convention as the other saved
    # inspection images.

    # This helper mirrors the generic patch-saving utilities in debug_utils.py, but it
    # first converts each numerical LBP image into a viewable 8-bit image before
    # writing it to disk.
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

    # This helper creates a lightweight text report that makes the image-level
    # evaluation easy to inspect without opening Python structures.
    #
    # Each output line corresponds to one parking space and summarizes:
    # - which space it was
    # - which label was predicted
    # - which ground-truth label it should have had
    # - whether that became TP / TN / FP / FN
    # - optionally which score-like value accompanied the prediction
    #
    # This is especially useful when cross-checking the visual outputs against the
    # numerical evaluation outcome.

    # The goal here is to create one lightweight text file that summarizes what happened
    # for each parking space in the selected test image. This is especially useful when
    # visual outputs and numeric evaluation need to be cross-checked by hand.
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

    # This is the main workflow of the inspection utility.
    #
    # It mirrors the main experiment pipeline closely, but with a different goal.
    # Instead of comparing many configurations across the whole test set, it:
    # 1. loads the already best-ranked configuration from the saved CSV
    # 2. reconstructs its nested config dictionaries
    # 3. loads the shared dataset inputs
    # 4. retrains the winning model
    # 5. reruns that model on one selected test image
    # 6. evaluates the result for that image
    # 7. saves a broad set of inspection artifacts
    #
    # So this function is effectively:
    # "replay the winning pipeline in a human-inspectable way"

    results_csv_path = Path(results_csv_path)
    training_root = Path(training_root)
    map_path = Path(map_path)
    test_images_dir = Path(test_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. load best-ranked row and reconstruct configs
    # -------------------------------------------------------------------------
    # The main experiment search stores flattened config values in a ranked CSV. This
    # stage rebuilds the original nested config dictionaries so the exact winning
    # configuration can be rerun without manual transcription.
    #
    # The CSV stores everything in one flat row, so each config block must be
    # reconstructed by prefix. The result is the exact preprocessing / LBP /
    # classifier / evaluation setup that produced the best ranked experiment.
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
    # Shared project inputs are loaded exactly the same way as in the main search, but
    # this inspection run focuses on just one selected test image instead of the whole
    # dataset.
    #
    # That symmetry is important: the inspection rerun should reflect the real
    # pipeline inputs, not an approximate or manually re-created variant.
    training_records = load_all_training_records(training_root)
    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(test_images_dir)
    test_case = select_test_case(test_cases, test_case_name=test_case_name)

    # -------------------------------------------------------------------------
    # 3. prepare training data and train classifier
    # -------------------------------------------------------------------------
    # The best configuration is retrained from scratch here so the inspection reflects a
    # real rerun of the same pipeline, not merely a replay of saved outputs.
    #
    # This stage deliberately follows the same record -> feature matrix -> classifier
    # training path used in experiment_search.py. That way, the inspected behavior
    # truly belongs to the winning configuration as it would run normally.
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
    # This mirrors the per-image processing path used in experiment_search.py:
    # scene image -> ROI records -> test feature records -> prediction records ->
    # evaluation against the matching ground-truth file.
    #
    # This is the exact test-side path of the pipeline, but limited to one image so
    # that all intermediate outputs remain understandable and easy to save.
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
    # The inspection output set is intentionally broad: one overlay for spatial context,
    # one folder of raw ROIs, one folder of processed images, one folder of LBP
    # visualizations, and one text report of per-space evaluation details.
    #
    # These outputs correspond to several levels of understanding:
    # - overlay ........ relationship between full scene and parking map
    # - roi ............ raw extracted parking-space patches
    # - processed ...... final grayscale inputs that fed LBP
    # - lbp ............ descriptor-coded images made viewable for humans
    # - report ......... per-space prediction vs. ground-truth summary
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

    # The returned inspection_result dictionary is a compact summary of everything this
    # rerun produced, including the configs used, the image-level evaluation, and the
    # paths of the saved inspection artifacts.
    #
    # This packaged return value makes the inspection workflow reusable from code:
    # callers get both the logical result (configs + evaluation) and the artifact
    # locations that were written to disk.
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

    # This is the standalone script entry point for the inspection workflow.
    # It mirrors main.py structurally, but instead of launching a whole experiment
    # search it points directly to:
    # - the already saved ranked results CSV
    # - the standard dataset locations
    # - the standard inspection-output location
    #
    # Then it runs one default inspection pass and prints a concise terminal
    # summary of what was produced.
    # This top-level entry point mirrors the structure of main.py but targets the
    # inspection workflow rather than the full experiment search.
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

    print("=== BEST CONFIGURATION INSPECTION ===")
    print(f"selected_test_case_name : {inspection_result['selected_test_case_name']}")
    print(f"accuracy                : {image_evaluation['accuracy']:.6f}")
    print(f"num_samples             : {image_evaluation['num_samples']}")
    print(f"confusion_counts        : {image_evaluation['confusion_counts']}")
    print(f"overlay_path            : {inspection_result['saved_paths']['overlay_path']}")
    print(f"roi_count               : {len(inspection_result['saved_paths']['roi_paths'])}")
    print(f"processed_count         : {len(inspection_result['saved_paths']['processed_paths'])}")
    print(f"lbp_count               : {len(inspection_result['saved_paths']['lbp_paths'])}")
    print(f"report_path             : {inspection_result['saved_paths']['report_path']}")


if __name__ == "__main__":
    main()