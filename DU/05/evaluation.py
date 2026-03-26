"""
evaluation.py

Stage 6:
- ground-truth loading
- prediction extraction
- frame-by-frame evaluation
- accuracy computation
- simple timing statistics
"""

from statistics import mean


# ---------------------------------------------------------------------
# Label handling
# ---------------------------------------------------------------------

def normalize_eye_state_label(label):
    """
    Normalize different textual forms of the same eye-state label.

    Internal evaluation labels:
    - "open"
    - "close"

    Unknown values are returned unchanged so they can still be counted.
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
# Ground-truth and prediction loading
# ---------------------------------------------------------------------

def load_ground_truth(ground_truth_path):
    """
    Load ground-truth eye-state labels from a text file.

    Expected simple format:
    - one label per line
    - empty lines are ignored
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
    Extract predicted eye-state labels from stored frame results.
    """

    predicted_labels = []

    for frame_result in frame_results:
        predicted_labels.append(
            normalize_eye_state_label(frame_result.get("eye_state"))
        )

    return predicted_labels


def extract_localization_times(frame_results):
    """
    Extract localization times from stored frame results.

    Missing values are ignored.
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
    Align predicted and ground-truth sequences by truncating both
    to the shorter length.

    This keeps the evaluation simple and robust even if the run is stopped
    early or if lengths differ for another reason.
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
# Accuracy and confusion-style counting
# ---------------------------------------------------------------------

def compute_confusion_counts(predicted_labels, ground_truth_labels):
    """
    Compute simple binary eye-state counts.

    Count names use the form:
    actual_as_predicted
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
    Compute simple frame-level accuracy.
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
    Compute simple localization-time statistics in milliseconds.
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
# Full evaluation wrapper
# ---------------------------------------------------------------------

def evaluate_results(frame_results, ground_truth_path):
    """
    Run the full evaluation pipeline and return one summary dictionary.
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


def print_evaluation_summary(summary):
    """
    Print the most important evaluation results in a readable way.
    """

    accuracy = summary["accuracy"]
    confusion = summary["confusion"]
    timing = summary["timing"]

    print("=== Evaluation summary ===")
    print(f"Predicted labels:      {summary['predicted_count']}")
    print(f"Ground-truth labels:   {summary['ground_truth_count']}")
    print(f"Compared labels:       {summary['aligned_count']}")
    print()

    print(f"Correct predictions:   {accuracy['correct_count']}")
    print(f"Accuracy [%]:          {accuracy['accuracy_percent']:.2f}")
    print()

    print("Confusion-style counts:")
    print(f"  open  -> open:       {confusion['open_as_open']}")
    print(f"  open  -> close:      {confusion['open_as_close']}")
    print(f"  close -> open:       {confusion['close_as_open']}")
    print(f"  close -> close:      {confusion['close_as_close']}")
    print(f"  other:               {confusion['other']}")
    print()

    print("Localization timing [ms]:")
    print(f"  measured frames:     {timing['count']}")
    print(f"  mean:                {timing['mean_ms']:.3f}")
    print(f"  min:                 {timing['min_ms']:.3f}")
    print(f"  max:                 {timing['max_ms']:.3f}")