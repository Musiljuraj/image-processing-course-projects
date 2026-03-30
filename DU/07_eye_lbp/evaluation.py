"""
evaluation.py

This module contains the complete evaluation pipeline used after video
processing is finished.

Its responsibilities are:
- reading ground-truth labels from a text file,
- extracting predicted labels from stored frame results,
- aligning both label sequences,
- computing accuracy and simple confusion-style counts,
- computing localization, classification, and total frame timing statistics,
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
    """

    predicted_labels = []

    for frame_result in frame_results:
        predicted_labels.append(
            normalize_eye_state_label(frame_result.get("eye_state"))
        )

    return predicted_labels


def extract_timing_values(frame_results, timing_key):
    """
    Extract one timing series from the stored frame results.

    Missing timing values are ignored. The resulting list contains floating-
    point timing values in milliseconds.
    """

    timing_values = []

    for frame_result in frame_results:
        timing_value = frame_result.get(timing_key)

        if timing_value is not None:
            timing_values.append(float(timing_value))

    return timing_values


def extract_localization_times(frame_results):
    """
    Extract measured localization times from the stored frame results.
    """

    return extract_timing_values(frame_results, "localization_time_ms")


def extract_classification_times(frame_results):
    """
    Extract measured classification times from the stored frame results.
    """

    return extract_timing_values(frame_results, "classification_time_ms")


def extract_total_frame_times(frame_results):
    """
    Extract measured total frame-processing times from the stored frame results.
    """

    return extract_timing_values(frame_results, "total_frame_time_ms")


# ---------------------------------------------------------------------
# Sequence alignment
# ---------------------------------------------------------------------

def align_label_sequences(predicted_labels, ground_truth_labels):
    """
    Align predicted and reference label sequences by truncating them to the
    same length.
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

def compute_timing_stats(timing_values_ms):
    """
    Compute basic descriptive statistics for one timing series.

    The returned values are expressed in milliseconds and include:
    - number of measured frames,
    - arithmetic mean,
    - minimum,
    - maximum.
    """

    if not timing_values_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "count": len(timing_values_ms),
        "mean_ms": mean(timing_values_ms),
        "min_ms": min(timing_values_ms),
        "max_ms": max(timing_values_ms),
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
    - timing statistics for localization, classification, and total frame
      processing.
    """

    ground_truth_labels = load_ground_truth(ground_truth_path)
    predicted_labels = extract_predicted_labels(frame_results)

    localization_times_ms = extract_localization_times(frame_results)
    classification_times_ms = extract_classification_times(frame_results)
    total_frame_times_ms = extract_total_frame_times(frame_results)

    alignment = align_label_sequences(predicted_labels, ground_truth_labels)

    aligned_predicted = alignment["predicted_labels"]
    aligned_ground_truth = alignment["ground_truth_labels"]

    accuracy = compute_accuracy(aligned_predicted, aligned_ground_truth)
    confusion = compute_confusion_counts(aligned_predicted, aligned_ground_truth)

    localization_timing = compute_timing_stats(localization_times_ms)
    classification_timing = compute_timing_stats(classification_times_ms)
    total_frame_timing = compute_timing_stats(total_frame_times_ms)

    summary = {
        "predicted_count": alignment["predicted_count"],
        "ground_truth_count": alignment["ground_truth_count"],
        "aligned_count": alignment["aligned_count"],
        "accuracy": accuracy,
        "confusion": confusion,
        "timing": {
            "localization": localization_timing,
            "classification": classification_timing,
            "total_frame": total_frame_timing,
        },
    }

    return summary


# ---------------------------------------------------------------------
# Summary formatting and output
# ---------------------------------------------------------------------

def format_evaluation_summary(summary):
    """
    Convert the evaluation summary into a readable multiline text block.
    """

    accuracy = summary["accuracy"]
    confusion = summary["confusion"]
    localization_timing = summary["timing"]["localization"]
    classification_timing = summary["timing"]["classification"]
    total_frame_timing = summary["timing"]["total_frame"]

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
        f"  measured frames:     {localization_timing['count']}",
        f"  mean:                {localization_timing['mean_ms']:.3f}",
        f"  min:                 {localization_timing['min_ms']:.3f}",
        f"  max:                 {localization_timing['max_ms']:.3f}",
        "",
        "Classification timing [ms]:",
        f"  measured frames:     {classification_timing['count']}",
        f"  mean:                {classification_timing['mean_ms']:.3f}",
        f"  min:                 {classification_timing['min_ms']:.3f}",
        f"  max:                 {classification_timing['max_ms']:.3f}",
        "",
        "Total frame-processing timing [ms]:",
        f"  measured frames:     {total_frame_timing['count']}",
        f"  mean:                {total_frame_timing['mean_ms']:.3f}",
        f"  min:                 {total_frame_timing['min_ms']:.3f}",
        f"  max:                 {total_frame_timing['max_ms']:.3f}",
    ]

    return "\n".join(lines)


def print_evaluation_summary(summary):
    """
    Print the formatted evaluation summary to standard output.
    """

    print(format_evaluation_summary(summary))


def save_evaluation_report(summary, report_path, extra_lines=None):
    """
    Save the formatted evaluation summary to a text report.
    """

    report_text = format_evaluation_summary(summary)

    if extra_lines:
        report_text = "\n".join(extra_lines) + "\n\n" + report_text

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text + "\n")