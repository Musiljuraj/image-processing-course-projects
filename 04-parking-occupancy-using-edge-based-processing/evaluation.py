"""
evaluation.py

Purpose of this module:
- classify parking-space occupancy from normalized edge statistics
- compare predictions with ground truth
- compute confusion-matrix counts and accuracy

Why this module exists:
The project pipeline already knows how to:
- extract one ROI patch per parking space
- preprocess each ROI patch
- run edge detection on each processed ROI
- count the number of edge pixels in the edge map

The next logical stage is to convert those edge statistics into an actual
occupied / empty prediction and then compare those predictions with the
ground-truth labels stored in the corresponding .txt files.

This module collects that classification + evaluation logic in one place.

Design decision used here:
Instead of using a raw edge-pixel count threshold, this module uses a
normalized feature:

    edge_ratio = edge_count / roi_pixel_count

This is important because parking spaces closer to the camera and farther from
the camera can produce ROI patches of different sizes. A raw count would be
biased by ROI area, while a ratio is much more comparable across spaces.

Current responsibilities:
- validate and normalize classification/evaluation configuration
- classify one edge ratio into empty / occupied
- classify all edge-detection records from one image
- validate ground-truth labels against the chosen label convention
- compare predictions with ground truth
- accumulate TP / TN / FP / FN
- compute accuracy
- merge per-image confusion counts into dataset-level totals

Expected label convention:
By default:
- occupied_label = 1
- empty_label = 0

If the dataset uses the opposite convention, this can be changed through the
configuration dictionary without changing the rest of the code.
"""


def validate_ratio_threshold(value):
    """
    Validate the occupancy threshold used for edge-ratio classification.

    Input:
        value ... numeric threshold expected to be in the interval [0, 1]

    Return:
        value ... same numeric value as float if valid

    Why this helper exists:
    edge_ratio is defined as:

        edge_count / roi_pixel_count

    so it should lie between 0 and 1. Therefore, the classification threshold
    used against it should also lie in a sensible range.
    """

    if not isinstance(value, (int, float)):
        raise TypeError("occupancy_threshold_ratio must be an int or float.")

    value = float(value)

    if value < 0.0 or value > 1.0:
        raise ValueError("occupancy_threshold_ratio must be between 0 and 1.")

    return value


def validate_classification_evaluation_config(classification_evaluation_config):
    """
    Validate and normalize the classification/evaluation configuration.

    Input:
        classification_evaluation_config ... dictionary, for example:
            {
                "occupancy_threshold_ratio": 0.08,
                "occupied_label": 1,
                "empty_label": 0,
            }

    Return:
        normalized_config ... validated normalized dictionary

    Why this helper exists:
    Later functions should not repeatedly re-check the same configuration.
    Instead, the configuration is normalized once and then reused consistently.
    """

    if classification_evaluation_config is None:
        classification_evaluation_config = {}

    occupancy_threshold_ratio = validate_ratio_threshold(
        classification_evaluation_config.get("occupancy_threshold_ratio", 0.08)
    )

    occupied_label = classification_evaluation_config.get("occupied_label", 1)
    empty_label = classification_evaluation_config.get("empty_label", 0)

    if occupied_label == empty_label:
        raise ValueError("occupied_label and empty_label must be different values.")

    normalized_config = {
        "occupancy_threshold_ratio": occupancy_threshold_ratio,
        "occupied_label": occupied_label,
        "empty_label": empty_label,
    }

    return normalized_config


def validate_ground_truth_label(label, classification_evaluation_config):
    """
    Validate one ground-truth label.

    Inputs:
        label ........................ one label from a ground-truth .txt file
        classification_evaluation_config ... validated label convention

    Return:
        label ... same value if valid

    Why this helper exists:
    The dataset evaluation should fail clearly if the .txt file contains labels
    outside the configured label convention.
    """

    occupied_label = classification_evaluation_config["occupied_label"]
    empty_label = classification_evaluation_config["empty_label"]

    if label not in (occupied_label, empty_label):
        raise ValueError(
            "Ground-truth label has unsupported value. "
            f"Expected one of ({empty_label}, {occupied_label}), got: {label}"
        )

    return label


def classify_one_edge_ratio(edge_ratio, classification_evaluation_config):
    """
    Classify one parking-space record from its edge ratio.

    Inputs:
        edge_ratio .................. normalized edge density in the ROI:
                                      edge_count / roi_pixel_count
        classification_evaluation_config ... validated config dictionary

    Return:
        predicted_label ............. occupied_label or empty_label

    Decision rule:
        if edge_ratio > occupancy_threshold_ratio -> occupied
        else -> empty

    Why use ">" and not ">=":
    This keeps the threshold interpretation simple:
    the threshold itself belongs to the "empty" side, and only clearly larger
    values are considered occupied. This is easy to reason about and tune.
    """

    if not isinstance(edge_ratio, (int, float)):
        raise TypeError("edge_ratio must be an int or float.")

    edge_ratio = float(edge_ratio)

    if edge_ratio < 0.0 or edge_ratio > 1.0:
        raise ValueError(f"edge_ratio must be between 0 and 1. Got: {edge_ratio}")

    if edge_ratio > classification_evaluation_config["occupancy_threshold_ratio"]:
        return classification_evaluation_config["occupied_label"]

    return classification_evaluation_config["empty_label"]


def classify_all_edge_records(edge_records, classification_evaluation_config):
    """
    Classify all edge-detection records from one image.

    Inputs:
        edge_records ................ list of records returned by edge_detection.py
        classification_evaluation_config ... classification settings

    Return:
        classified_records .......... list of richer records that contain:
                                      - all original edge-record data
                                      - predicted_label
                                      - classification_evaluation_config

    Overall idea:
    This function turns:
        one list of edge-detection records
    into:
        one list of classified records

    This keeps the pipeline symmetric:
    - ROI extraction stage produces ROI records
    - preprocessing stage produces preprocessed records
    - edge-detection stage produces edge records
    - classification stage produces classified records
    """

    normalized_config = validate_classification_evaluation_config(
        classification_evaluation_config
    )

    classified_records = []

    for edge_record in edge_records:
        predicted_label = classify_one_edge_ratio(
            edge_ratio=edge_record["edge_ratio"],
            classification_evaluation_config=normalized_config,
        )

        classified_record = {
            **edge_record,
            "predicted_label": predicted_label,
            "classification_evaluation_config": normalized_config,
        }

        classified_records.append(classified_record)

    return classified_records


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

    merged_counts = {
        "tp": base_counts["tp"] + additional_counts["tp"],
        "tn": base_counts["tn"] + additional_counts["tn"],
        "fp": base_counts["fp"] + additional_counts["fp"],
        "fn": base_counts["fn"] + additional_counts["fn"],
    }

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

    total = (
        confusion_counts["tp"]
        + confusion_counts["tn"]
        + confusion_counts["fp"]
        + confusion_counts["fn"]
    )

    if total == 0:
        return 0.0

    return (confusion_counts["tp"] + confusion_counts["tn"]) / total


def evaluate_one_image(
    classified_records,
    ground_truth_labels,
    classification_evaluation_config,
):
    """
    Evaluate predictions for one image.

    Inputs:
        classified_records .......... output of classify_all_edge_records(...)
        ground_truth_labels ......... list of true labels loaded from testX.txt
        classification_evaluation_config ... classification settings

    Return:
        image_evaluation ............ dictionary containing:
                                      - evaluated_records
                                      - confusion_counts
                                      - accuracy
                                      - num_samples

    Evaluation logic:
    - compare each predicted label to the matching ground-truth label
    - update TP / TN / FP / FN
    - add ground_truth_label and evaluation_outcome to each record

    Why this function exists:
    It gives you one clean evaluation unit per image, which can then be merged
    into dataset-level totals in main.py.
    """

    normalized_config = validate_classification_evaluation_config(
        classification_evaluation_config
    )

    if len(classified_records) != len(ground_truth_labels):
        raise ValueError(
            "Number of classified parking spaces does not match number of "
            "ground-truth labels. "
            f"classified_records={len(classified_records)}, "
            f"ground_truth_labels={len(ground_truth_labels)}" 
        )

    occupied_label = normalized_config["occupied_label"]
    empty_label = normalized_config["empty_label"]

    confusion_counts = initialize_confusion_counts()
    evaluated_records = []

    for classified_record, ground_truth_label in zip(
        classified_records,
        ground_truth_labels,
    ):
        ground_truth_label = validate_ground_truth_label(
            ground_truth_label,
            normalized_config,
        )

        predicted_label = classified_record["predicted_label"]

        if predicted_label == occupied_label and ground_truth_label == occupied_label:
            evaluation_outcome = "tp"
            confusion_counts["tp"] += 1

        elif predicted_label == empty_label and ground_truth_label == empty_label:
            evaluation_outcome = "tn"
            confusion_counts["tn"] += 1

        elif predicted_label == occupied_label and ground_truth_label == empty_label:
            evaluation_outcome = "fp"
            confusion_counts["fp"] += 1

        elif predicted_label == empty_label and ground_truth_label == occupied_label:
            evaluation_outcome = "fn"
            confusion_counts["fn"] += 1

        else:
            raise RuntimeError(
                "Unexpected label combination during evaluation. "
                f"predicted_label={predicted_label}, "
                f"ground_truth_label={ground_truth_label}"
            )

        evaluated_record = {
            **classified_record,
            "ground_truth_label": ground_truth_label,
            "evaluation_outcome": evaluation_outcome,
        }

        evaluated_records.append(evaluated_record)

    accuracy = compute_accuracy(confusion_counts)

    image_evaluation = {
        "evaluated_records": evaluated_records,
        "confusion_counts": confusion_counts,
        "accuracy": accuracy,
        "num_samples": len(evaluated_records),
    }

    return image_evaluation