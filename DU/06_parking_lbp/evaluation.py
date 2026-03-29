"""
evaluation.py

Purpose of this module:
- evaluate parking-space occupancy predictions against ground truth
- keep evaluation logic separate from dataset loading, preprocessing,
  LBP feature extraction, classifier training, and experiment orchestration

Why this module exists:
At this stage of the project, earlier modules can already produce:
- extracted parking-space ROI records
- preprocessed ROI records
- LBP feature records
- classifier predictions for each parking space

The next logical stage is no longer to classify from edge statistics.
Instead, it is to compare already produced predicted labels with the
ground-truth labels stored in the corresponding testX.txt files.

This module currently provides:
- validate_evaluation_config(...)
- validate_classification_evaluation_config(...)   # compatibility alias
- validate_ground_truth_label(...)
- validate_predicted_label(...)
- initialize_confusion_counts(...)
- merge_confusion_counts(...)
- compute_accuracy(...)
- evaluate_predicted_labels(...)
- evaluate_prediction_records(...)
- evaluate_one_image(...)
- evaluate_one_test_case(...)

Expected label convention by default:
- occupied_label = 1
- empty_label = 0
"""
# ---------------------------------------------------------------------------
# Module orientation:
# This module is the final judging stage of the parking pipeline. Upstream
# modules prepare feature records and prediction records; this module compares
# those predicted labels against the ground-truth labels stored next to each
# test image. The shared terminology used here matches the rest of the project:
# a prediction record belongs to one parking space, multiple records form one
# image-level result, and multiple image-level results are later merged into a
# dataset-level experiment result.
# ---------------------------------------------------------------------------

from pathlib import Path

import numpy as np

from parking_io import load_ground_truth_labels


def validate_evaluation_config(evaluation_config):
    """
    Validate and normalize the evaluation configuration.

    Input:
        evaluation_config ... dictionary, for example:
            {
                "occupied_label": 1,
                "empty_label": 0,
            }

    Return:
        normalized_config ... validated normalized dictionary

    Why this helper exists:
    Later functions should not repeatedly re-check the same label convention.
    Instead, it is normalized once and then reused consistently.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if evaluation_config is None:
        evaluation_config = {}

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not isinstance(evaluation_config, dict):
        raise TypeError("evaluation_config must be a dictionary or None.")

    # Resolve configuration-dependent values into local variables here so the later
    # logic can use short, consistent names.
    occupied_label = evaluation_config.get("occupied_label", 1)
    # Resolve configuration-dependent values into local variables here so the later
    # logic can use short, consistent names.
    empty_label = evaluation_config.get("empty_label", 0)

    if occupied_label == empty_label:
        raise ValueError("occupied_label and empty_label must be different values.")

    normalized_config = {
        "occupied_label": occupied_label,
        "empty_label": empty_label,
    }

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return normalized_config


def validate_classification_evaluation_config(classification_evaluation_config):
    """
    Compatibility alias for the old function name used in the edge-based version.

    Input:
        classification_evaluation_config ... evaluation configuration dictionary

    Return:
        normalized_config ... validated normalized dictionary

    Why this alias exists:
    The old evaluation module used a longer function name. Keeping this alias
    makes the transition to the new classifier-based evaluation slightly easier.
    """

    # This wrapper keeps the surrounding API simple and delegates the actual work to the
    # shared helper that already implements the full logic.
    return validate_evaluation_config(classification_evaluation_config)


def _normalize_label_value(label):
    """
    Normalize NumPy scalar labels to plain Python scalar values when needed.

    Input:
        label ... plain Python value or NumPy scalar

    Return:
        normalized_label ... normalized scalar value
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if isinstance(label, np.generic):
        return label.item()

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return label


def validate_ground_truth_label(label, evaluation_config):
    """
    Validate one ground-truth label.

    Inputs:
        label ................. one label from a ground-truth .txt file
        evaluation_config ..... validated label convention

    Return:
        label ................. same value if valid

    Why this helper exists:
    Evaluation should fail clearly if the ground-truth file contains labels
    outside the configured label convention.
    """

    # Read configuration values and normalize them up front so the rest of the function
    # can rely on one stable internal convention.
    label = _normalize_label_value(label)

    occupied_label = evaluation_config["occupied_label"]
    empty_label = evaluation_config["empty_label"]

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if label not in (occupied_label, empty_label):
        raise ValueError(
            "Ground-truth label has unsupported value. "
            f"Expected one of ({empty_label}, {occupied_label}), got: {label}"
        )

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return label


def validate_predicted_label(label, evaluation_config):
    """
    Validate one predicted label.

    Inputs:
        label ................. predicted label
        evaluation_config ..... validated label convention

    Return:
        label ................. same value if valid

    Why this helper exists:
    Evaluation should fail clearly if the classifier produces labels outside
    the configured label convention.
    """

    # Read configuration values and normalize them up front so the rest of the function
    # can rely on one stable internal convention.
    label = _normalize_label_value(label)

    occupied_label = evaluation_config["occupied_label"]
    empty_label = evaluation_config["empty_label"]

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if label not in (occupied_label, empty_label):
        raise ValueError(
            "Predicted label has unsupported value. "
            f"Expected one of ({empty_label}, {occupied_label}), got: {label}"
        )

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return label


def initialize_confusion_counts():
    """
    Create an empty confusion-count dictionary.

    Return:
        confusion_counts ... dictionary with:
                             - tp
                             - tn
                             - fp
                             - fn

    Why this helper exists:
    Using one standard structure everywhere keeps per-image and dataset-level
    evaluation compatible and easy to merge.
    """

    # This wrapper keeps the surrounding API simple and delegates the actual work to the
    # shared helper that already implements the full logic.
    return {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }


def merge_confusion_counts(base_counts, additional_counts):
    """
    Merge two confusion-count dictionaries.

    Inputs:
        base_counts ......... existing totals
        additional_counts ... counts from one image or one subset

    Return:
        merged_counts ....... new merged dictionary

    Why this helper exists:
    Dataset-level evaluation is simply repeated per-image evaluation with
    accumulation of TP / TN / FP / FN.
    """

    # Package the already available values into the standard dictionary structure used
    # by the rest of the pipeline.
    merged_counts = {
        "tp": base_counts["tp"] + additional_counts["tp"],
        "tn": base_counts["tn"] + additional_counts["tn"],
        "fp": base_counts["fp"] + additional_counts["fp"],
        "fn": base_counts["fn"] + additional_counts["fn"],
    }

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return merged_counts


def compute_accuracy(confusion_counts):
    """
    Compute accuracy from confusion counts.

    Input:
        confusion_counts ... dictionary with tp, tn, fp, fn

    Return:
        accuracy ........... float in [0, 1]

    Formula:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    If there are no samples, 0.0 is returned to avoid division by zero.
    """

    # Set up the local working state first so the later processing steps can operate on
    # explicit, well-named intermediate values.
    total = (
        confusion_counts["tp"]
        + confusion_counts["tn"]
        + confusion_counts["fp"]
        + confusion_counts["fn"]
    )

    if total == 0:
        return 0.0

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return (confusion_counts["tp"] + confusion_counts["tn"]) / total


def _determine_evaluation_outcome(predicted_label, ground_truth_label, evaluation_config):
    """
    Determine TP / TN / FP / FN for one prediction-ground-truth pair.

    Inputs:
        predicted_label ..... predicted parking-space label
        ground_truth_label .. true parking-space label
        evaluation_config ... validated label convention

    Return:
        evaluation_outcome .. one of:
                              "tp", "tn", "fp", "fn"
    """

    occupied_label = evaluation_config["occupied_label"]
    empty_label = evaluation_config["empty_label"]

    # The four branches below translate the shared binary label convention into the
    # standard confusion-count terminology used by the rest of the evaluation logic.
    if predicted_label == occupied_label and ground_truth_label == occupied_label:
        return "tp"

    if predicted_label == empty_label and ground_truth_label == empty_label:
        return "tn"

    if predicted_label == occupied_label and ground_truth_label == empty_label:
        return "fp"

    if predicted_label == empty_label and ground_truth_label == occupied_label:
        return "fn"

    raise RuntimeError(
        "Unexpected label combination during evaluation. "
        f"predicted_label={predicted_label}, ground_truth_label={ground_truth_label}"
    )


def evaluate_predicted_labels(predicted_labels, ground_truth_labels, evaluation_config=None):
    """
    Evaluate bare predicted-label arrays against ground truth.

    Inputs:
        predicted_labels ..... 1D array-like of predicted labels
        ground_truth_labels .. list or array-like of true labels
        evaluation_config .... optional label-convention dictionary

    Return:
        evaluation_result .... dictionary containing:
                               - evaluated_items
                               - confusion_counts
                               - accuracy
                               - num_samples
                               - evaluation_config

    Why this function exists:
    Sometimes evaluation is needed before prediction records are built.
    This function provides a minimal label-only evaluation path.
    """

    # Read configuration values and normalize them up front so the rest of the function
    # can rely on one stable internal convention.
    normalized_config = validate_evaluation_config(evaluation_config)

    predicted_labels_array = np.asarray(predicted_labels)

    if predicted_labels_array.ndim != 1:
        raise ValueError("predicted_labels must be a 1D array-like object.")

    ground_truth_labels_list = list(ground_truth_labels)

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if len(predicted_labels_array) != len(ground_truth_labels_list):
        raise ValueError(
            "Number of predicted labels does not match number of ground-truth labels. "
            f"predicted_labels={len(predicted_labels_array)}, "
            f"ground_truth_labels={len(ground_truth_labels_list)}"
        )

    confusion_counts = initialize_confusion_counts()
    evaluated_items = []

    # Process items in a deterministic order so the produced outputs stay aligned with
    # the corresponding inputs, labels, or metadata.
    for sample_index, (predicted_label, ground_truth_label) in enumerate(
        zip(predicted_labels_array, ground_truth_labels_list),
        start=1,
    ):
        predicted_label = validate_predicted_label(
            predicted_label,
            normalized_config,
        )
        ground_truth_label = validate_ground_truth_label(
            ground_truth_label,
            normalized_config,
        )

        evaluation_outcome = _determine_evaluation_outcome(
            predicted_label=predicted_label,
            ground_truth_label=ground_truth_label,
            evaluation_config=normalized_config,
        )

        confusion_counts[evaluation_outcome] += 1

        evaluated_item = {
            "sample_index": sample_index,
            "predicted_label": predicted_label,
            "ground_truth_label": ground_truth_label,
            "evaluation_outcome": evaluation_outcome,
        }
        evaluated_items.append(evaluated_item)

    accuracy = compute_accuracy(confusion_counts)

    # Assemble the standard output dictionary here so downstream modules receive both
    # the computed values and the metadata needed for traceability.
    evaluation_result = {
        "evaluated_items": evaluated_items,
        "confusion_counts": confusion_counts,
        "accuracy": accuracy,
        "num_samples": len(evaluated_items),
        "evaluation_config": normalized_config,
    }

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return evaluation_result


def evaluate_prediction_records(
    prediction_records,
    ground_truth_labels,
    evaluation_config=None,
    ):
    """
    Evaluate structured prediction records against ground truth labels.

    Inputs:
        prediction_records ... list of prediction-record dictionaries expected
                               to contain at least:
                               - predicted_label
                               and typically also metadata such as:
                               - source_image_name
                               - space_index
                               - file_name
                               - class_name
                               - label (for training-side evaluation)

        ground_truth_labels .. list or array-like of true labels
        evaluation_config ... optional label-convention dictionary

    Return:
        image_evaluation ..... dictionary containing:
                               - evaluated_records
                               - confusion_counts
                               - accuracy
                               - num_samples
                               - evaluation_config

    Important note:
    The function assumes that prediction_records are already aligned with the
    order of the ground-truth labels. In the current project, this is naturally
    true because:
    - test ROIs are extracted in parking-map order
    - feature records preserve that order
    - prediction records preserve that order
    """

    # Read configuration values and normalize them up front so the rest of the function
    # can rely on one stable internal convention.
    normalized_config = validate_evaluation_config(evaluation_config)

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not isinstance(prediction_records, list):
        raise TypeError("prediction_records must be a list.")

    ground_truth_labels_list = list(ground_truth_labels)

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if len(prediction_records) != len(ground_truth_labels_list):
        raise ValueError(
            "Number of prediction records does not match number of ground-truth labels. "
            f"prediction_records={len(prediction_records)}, "
            f"ground_truth_labels={len(ground_truth_labels_list)}"
        )

    confusion_counts = initialize_confusion_counts()
    # Start an accumulation structure that will be filled gradually as the function
    # walks through samples, records, rows, or files.
    evaluated_records = []

    # Process items in a deterministic order so the produced outputs stay aligned with
    # the corresponding inputs, labels, or metadata.
    for prediction_record, ground_truth_label in zip(
        prediction_records,
        ground_truth_labels_list,
    ):
        if not isinstance(prediction_record, dict):
            raise TypeError("Each prediction_record must be a dictionary.")

        if "predicted_label" not in prediction_record:
            raise KeyError(
                "Each prediction_record must contain 'predicted_label'."
            )

        predicted_label = validate_predicted_label(
            prediction_record["predicted_label"],
            normalized_config,
        )
        ground_truth_label = validate_ground_truth_label(
            ground_truth_label,
            normalized_config,
        )

        evaluation_outcome = _determine_evaluation_outcome(
            predicted_label=predicted_label,
            ground_truth_label=ground_truth_label,
            evaluation_config=normalized_config,
        )

        confusion_counts[evaluation_outcome] += 1

        evaluated_record = {
            **prediction_record,
            "ground_truth_label": ground_truth_label,
            "evaluation_outcome": evaluation_outcome,
        }
        evaluated_records.append(evaluated_record)

    accuracy = compute_accuracy(confusion_counts)

    # Assemble the standard output dictionary here so downstream modules receive both
    # the computed values and the metadata needed for traceability.
    image_evaluation = {
        "evaluated_records": evaluated_records,
        "confusion_counts": confusion_counts,
        "accuracy": accuracy,
        "num_samples": len(evaluated_records),
        "evaluation_config": normalized_config,
    }

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return image_evaluation


def evaluate_one_image(
    prediction_records,
    ground_truth_labels,
    evaluation_config=None,
    ):
    """
    Evaluate predictions for one image.

    Inputs:
        prediction_records ... list of structured prediction records
        ground_truth_labels .. list of ground-truth labels already loaded
        evaluation_config ... optional label-convention dictionary

    Return:
        image_evaluation ..... same structure as evaluate_prediction_records(...)

    Why this function exists:
    The old evaluation module already exposed evaluate_one_image(...), so
    keeping that name makes the transition to the classifier-based version
    easier while preserving a natural per-image evaluation unit.
    """

    # This wrapper keeps the surrounding API simple and delegates the actual work to the
    # shared helper that already implements the full logic.
    return evaluate_prediction_records(
        prediction_records=prediction_records,
        ground_truth_labels=ground_truth_labels,
        evaluation_config=evaluation_config,
    )


def evaluate_one_test_case(
    prediction_records,
    txt_path,
    evaluation_config=None,
    ):
    """
    Evaluate one test case directly from a ground-truth txt path.

    Inputs:
        prediction_records ... list of structured prediction records
        txt_path ............ path to the corresponding testX.txt file
        evaluation_config ... optional label-convention dictionary

    Return:
        image_evaluation ..... dictionary containing:
                               - evaluated_records
                               - confusion_counts
                               - accuracy
                               - num_samples
                               - evaluation_config
                               - txt_path

    Why this function exists:
    It is often convenient to evaluate a test image in one call, without having
    to manually load the ground-truth labels outside the evaluation module.
    """

    # Convert incoming path-like inputs to Path objects at the start so all later
    # filesystem work uses one consistent path representation.
    txt_path = Path(txt_path)
    ground_truth_labels = load_ground_truth_labels(txt_path)

    # Assemble the standard output dictionary here so downstream modules receive both
    # the computed values and the metadata needed for traceability.
    image_evaluation = evaluate_one_image(
        prediction_records=prediction_records,
        ground_truth_labels=ground_truth_labels,
        evaluation_config=evaluation_config,
    )

    # Assemble the standard output dictionary here so downstream modules receive both
    # the computed values and the metadata needed for traceability.
    image_evaluation = {
        **image_evaluation,
        "txt_path": txt_path,
    }

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return image_evaluation