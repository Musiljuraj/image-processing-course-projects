"""
evaluation.py

This module contains the complete evaluation pipeline used after video
processing is finished.

Its responsibilities are:
- reading ground-truth labels from a text file,
- extracting predicted labels from stored frame results,
- aligning both label sequences,
- computing accuracy and simple confusion-style counts,
- computing localization timing statistics,
- formatting and saving the final evaluation summary.
"""

from statistics import mean


# ---------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------

def normalize_eye_state_label(label):
    """
    Normalize textual eye-state labels into the internal evaluation format.

    The evaluation stage operates with two canonical label names:
    - "open"
    - "close"

    Several textual variants are accepted so that the evaluation remains
    tolerant to minor naming differences in predictions or reference data.

    Any unrecognized label is returned unchanged so it can still be counted
    explicitly in the "other" category.
    """

    if label is None:
        return "unknown"

    value = str(label).strip().lower()

    if value in ("open", "opened"):
        return "open"

    if value in ("close", "closed", "shut"):
        return "close"

    return value


# ---------------------------------------------------------------------
# Input loading and extraction
# ---------------------------------------------------------------------

def load_ground_truth(ground_truth_path):
    """
    Load ground-truth eye-state labels from a text file.

    Expected file structure:
    - one label per line,
    - empty lines ignored,
    - labels normalized into the internal evaluation vocabulary.

    The function returns a list of normalized labels in the same order as
    they appear in the file.
    """

    ground_truth_labels = []

    with open(ground_truth_path, "r", encoding="utf-8") as file:
        for line in file:
            raw_label = line.strip()

            if not raw_label:
                continue

            ground_truth_labels.append(normalize_eye_state_label(raw_label))

    return ground_truth_labels


def extract_predicted_labels(frame_results):
    """
    Extract predicted eye-state labels from the frame-result list.

    Each frame result is expected to contain the key:
        "eye_state"

    The values are normalized before being returned so that prediction labels
    and ground-truth labels follow the same naming convention.
    """

    predicted_labels = []

    for frame_result in frame_results:
        predicted_labels.append(
            normalize_eye_state_label(frame_result.get("eye_state"))
        )

    return predicted_labels


def extract_localization_times(frame_results):
    """
    Extract measured localization times from the stored frame results.

    Missing timing values are ignored. The resulting list contains floating-
    point timing values in milliseconds.
    """

    localization_times = []

    for frame_result in frame_results:
        localization_time = frame_result.get("localization_time_ms")

        if localization_time is not None:
            localization_times.append(float(localization_time))

    return localization_times


# ---------------------------------------------------------------------
# Sequence alignment
# ---------------------------------------------------------------------

def align_label_sequences(predicted_labels, ground_truth_labels):
    """
    Align predicted and reference label sequences by truncating them to the
    same length.

    This simple alignment strategy is sufficient for the current assignment.
    It ensures evaluation can still be performed when:
    - the processing run is interrupted early,
    - the prediction sequence is shorter than the reference sequence,
    - the input data contains a minor length mismatch.

    The function returns both original sequence lengths and the aligned data.
    """

    aligned_count = min(len(predicted_labels), len(ground_truth_labels))

    aligned_predicted = predicted_labels[:aligned_count]
    aligned_ground_truth = ground_truth_labels[:aligned_count]

    return {
        "predicted_count": len(predicted_labels),
        "ground_truth_count": len(ground_truth_labels),
        "aligned_count": aligned_count,
        "predicted_labels": aligned_predicted,
        "ground_truth_labels": aligned_ground_truth,
    }


# ---------------------------------------------------------------------
# Accuracy and confusion-style statistics
# ---------------------------------------------------------------------

def compute_confusion_counts(predicted_labels, ground_truth_labels):
    """
    Compute confusion-style counts for the binary open/close task.

    The count names follow the pattern:
        actual_as_predicted

    Example:
        "open_as_close" means that the ground truth was "open"
        but the prediction was "close".

    Any label pair outside the expected binary scheme is accumulated into
    the "other" category.
    """

    counts = {
        "open_as_open": 0,
        "open_as_close": 0,
        "close_as_open": 0,
        "close_as_close": 0,
        "other": 0,
    }

    for predicted_label, ground_truth_label in zip(predicted_labels, ground_truth_labels):
        if ground_truth_label == "open" and predicted_label == "open":
            counts["open_as_open"] += 1
        elif ground_truth_label == "open" and predicted_label == "close":
            counts["open_as_close"] += 1
        elif ground_truth_label == "close" and predicted_label == "open":
            counts["close_as_open"] += 1
        elif ground_truth_label == "close" and predicted_label == "close":
            counts["close_as_close"] += 1
        else:
            counts["other"] += 1

    return counts


def compute_accuracy(predicted_labels, ground_truth_labels):
    """
    Compute simple frame-level classification accuracy.

    Accuracy is defined as:
        correct_predictions / compared_predictions

    The function returns the number of correct matches, the number of
    compared labels, and the accuracy percentage.
    """

    compared_count = min(len(predicted_labels), len(ground_truth_labels))

    if compared_count == 0:
        return {
            "correct_count": 0,
            "compared_count": 0,
            "accuracy_percent": 0.0,
        }

    correct_count = 0

    for predicted_label, ground_truth_label in zip(predicted_labels, ground_truth_labels):
        if predicted_label == ground_truth_label:
            correct_count += 1

    accuracy_percent = 100.0 * correct_count / compared_count

    return {
        "correct_count": correct_count,
        "compared_count": compared_count,
        "accuracy_percent": accuracy_percent,
    }


# ---------------------------------------------------------------------
# Timing statistics
# ---------------------------------------------------------------------

def compute_timing_stats(localization_times_ms):
    """
    Compute basic descriptive statistics for localization times.

    The returned values are expressed in milliseconds and include:
    - number of measured frames,
    - arithmetic mean,
    - minimum,
    - maximum.
    """

    if not localization_times_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "count": len(localization_times_ms),
        "mean_ms": mean(localization_times_ms),
        "min_ms": min(localization_times_ms),
        "max_ms": max(localization_times_ms),
    }


# ---------------------------------------------------------------------
# Complete evaluation pipeline
# ---------------------------------------------------------------------

def evaluate_results(frame_results, ground_truth_path):
    """
    Run the complete evaluation procedure and return a single summary object.

    The summary dictionary contains:
    - original sequence lengths,
    - aligned comparison length,
    - accuracy information,
    - confusion-style counts,
    - timing statistics.

    This function is the main evaluation entry point used by the main program.
    """

    ground_truth_labels = load_ground_truth(ground_truth_path)
    predicted_labels = extract_predicted_labels(frame_results)
    localization_times_ms = extract_localization_times(frame_results)

    alignment = align_label_sequences(predicted_labels, ground_truth_labels)

    aligned_predicted = alignment["predicted_labels"]
    aligned_ground_truth = alignment["ground_truth_labels"]

    accuracy = compute_accuracy(aligned_predicted, aligned_ground_truth)
    confusion = compute_confusion_counts(aligned_predicted, aligned_ground_truth)
    timing = compute_timing_stats(localization_times_ms)

    summary = {
        "predicted_count": alignment["predicted_count"],
        "ground_truth_count": alignment["ground_truth_count"],
        "aligned_count": alignment["aligned_count"],
        "accuracy": accuracy,
        "confusion": confusion,
        "timing": timing,
    }

    return summary


# ---------------------------------------------------------------------
# Summary formatting and output
# ---------------------------------------------------------------------

def format_evaluation_summary(summary):
    """
    Convert the evaluation summary into a readable multiline text block.

    This text representation is used both for console output and for saving
    the final report file.
    """

    accuracy = summary["accuracy"]
    confusion = summary["confusion"]
    timing = summary["timing"]

    lines = [
        "=== Evaluation summary ===",
        f"Predicted labels:      {summary['predicted_count']}",
        f"Ground-truth labels:   {summary['ground_truth_count']}",
        f"Compared labels:       {summary['aligned_count']}",
        "",
        f"Correct predictions:   {accuracy['correct_count']}",
        f"Accuracy [%]:          {accuracy['accuracy_percent']:.2f}",
        "",
        "Confusion-style counts:",
        f"  open  -> open:       {confusion['open_as_open']}",
        f"  open  -> close:      {confusion['open_as_close']}",
        f"  close -> open:       {confusion['close_as_open']}",
        f"  close -> close:      {confusion['close_as_close']}",
        f"  other:               {confusion['other']}",
        "",
        "Localization timing [ms]:",
        f"  measured frames:     {timing['count']}",
        f"  mean:                {timing['mean_ms']:.3f}",
        f"  min:                 {timing['min_ms']:.3f}",
        f"  max:                 {timing['max_ms']:.3f}",
    ]

    return "\n".join(lines)


def print_evaluation_summary(summary):
    """
    Print the formatted evaluation summary to standard output.

    This function is a convenience wrapper around the formatter so that
    presentation logic stays centralized.
    """

    print(format_evaluation_summary(summary))


def save_evaluation_report(summary, report_path, extra_lines=None):
    """
    Save the formatted evaluation summary to a text report.

    Optional extra lines can be placed before the main summary block. This is
    useful for run configuration details such as file paths or runtime options.
    """

    report_text = format_evaluation_summary(summary)

    if extra_lines:
        report_text = "\n".join(extra_lines) + "\n\n" + report_text

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text + "\n")