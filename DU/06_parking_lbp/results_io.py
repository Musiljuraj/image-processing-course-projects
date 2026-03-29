"""
results_io.py

Purpose of this module:
- save experiment results from the current parking LBP pipeline
- keep output formatting and file-writing separate from experiment execution

Why this module exists:
At this stage of the project, experiment code can produce results that combine:
- preprocessing configuration
- LBP configuration
- classifier configuration
- evaluation metrics
- timing information

This module converts such experiment results into:
1. a flat CSV table suitable for sorting and later inspection
2. a readable text summary with ranked top results

This module currently provides:
- ensure_output_directory(...)
- flatten_nested_dict(...)
- flatten_experiment_result(...)
- flatten_experiment_results(...)
- rank_experiment_results(...)
- save_experiment_results_csv(...)
- save_experiment_summary_text(...)
- save_experiment_outputs(...)
"""

from pathlib import Path
import csv
import json


def ensure_output_directory(output_dir):
    """
    Ensure that the given output directory exists.

    Input:
        output_dir ... directory path

    Return:
        output_dir ... pathlib.Path object

    Why this helper exists:
    Saving helpers should not have to repeat directory-creation logic.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def flatten_nested_dict(data, prefix=""):
    """
    Flatten one nested dictionary using prefixed keys.

    Inputs:
        data ..... dictionary to flatten
        prefix ... prefix string added to produced keys

    Return:
        flat_dict ... flat dictionary

    Example:
        input:
            prefix = "lbp"
            data = {
                "neighbors": 8,
                "grid_shape": (4, 4),
            }

        output:
            {
                "lbp_neighbors": 8,
                "lbp_grid_shape": "(4, 4)",
            }

    Why this helper exists:
    CSV output is easier to work with when nested structures such as
    preprocessing_config / lbp_config / classifier_config are flattened.
    """

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError("data passed to flatten_nested_dict(...) must be a dict or None.")

    flat_dict = {}

    for key, value in data.items():
        full_key = f"{prefix}_{key}" if prefix else str(key)

        if isinstance(value, dict):
            nested_flat = flatten_nested_dict(value, prefix=full_key)
            flat_dict.update(nested_flat)
        elif isinstance(value, (list, tuple)):
            flat_dict[full_key] = str(tuple(value)) if isinstance(value, tuple) else str(value)
        else:
            flat_dict[full_key] = value

    return flat_dict


def flatten_experiment_result(experiment_result):
    """
    Flatten one experiment-result dictionary into one CSV-friendly row.

    Input:
        experiment_result ... dictionary that may contain:
                              - preprocessing_config
                              - lbp_config
                              - classifier_config
                              - evaluation_config
                              - confusion_counts
                              - accuracy
                              - num_samples
                              - timing fields
                              - optional extra metadata

    Return:
        flat_result ........ one flat dictionary suitable for CSV output

    Why this helper exists:
    One experiment result is naturally structured, but CSV requires one flat row.
    """

    if not isinstance(experiment_result, dict):
        raise TypeError("experiment_result must be a dictionary.")

    flat_result = {}

    # -------------------------------------------------------------------------
    # 1. flatten known config blocks
    # -------------------------------------------------------------------------
    flat_result.update(
        flatten_nested_dict(
            experiment_result.get("preprocessing_config"),
            prefix="preprocessing",
        )
    )

    flat_result.update(
        flatten_nested_dict(
            experiment_result.get("lbp_config"),
            prefix="lbp",
        )
    )

    flat_result.update(
        flatten_nested_dict(
            experiment_result.get("classifier_config"),
            prefix="classifier",
        )
    )

    flat_result.update(
        flatten_nested_dict(
            experiment_result.get("evaluation_config"),
            prefix="evaluation",
        )
    )

    flat_result.update(
        flatten_nested_dict(
            experiment_result.get("confusion_counts"),
            prefix="confusion",
        )
    )

    # -------------------------------------------------------------------------
    # 2. copy common scalar result fields if present
    # -------------------------------------------------------------------------
    common_scalar_keys = [
        "accuracy",
        "num_samples",
        "processing_time_total",
        "processing_time_training",
        "processing_time_test_feature_preparation",
        "processing_time_prediction",
        "processing_time_per_image",
        "processing_time_per_roi",
        "experiment_index",
        "source_image_name",
        "notes",
    ]

    for key in common_scalar_keys:
        if key in experiment_result:
            flat_result[key] = experiment_result[key]

    # -------------------------------------------------------------------------
    # 3. copy any other non-config scalar-like fields not already handled
    # -------------------------------------------------------------------------
    reserved_keys = {
        "preprocessing_config",
        "lbp_config",
        "classifier_config",
        "evaluation_config",
        "confusion_counts",
        *common_scalar_keys,
    }

    for key, value in experiment_result.items():
        if key in reserved_keys:
            continue

        if isinstance(value, dict):
            # keep any unexpected nested dict, but flatten it under its own name
            flat_result.update(flatten_nested_dict(value, prefix=key))
        elif isinstance(value, (list, tuple)):
            flat_result[key] = str(tuple(value)) if isinstance(value, tuple) else str(value)
        else:
            flat_result[key] = value

    return flat_result


def flatten_experiment_results(experiment_results):
    """
    Flatten a list of experiment results.

    Input:
        experiment_results ... list of experiment-result dictionaries

    Return:
        flat_results ........ list of flat dictionaries

    Why this helper exists:
    Saving multiple results to CSV is simpler after a batch flattening step.
    """

    if not isinstance(experiment_results, list):
        raise TypeError("experiment_results must be a list.")

    flat_results = []

    for experiment_result in experiment_results:
        flat_result = flatten_experiment_result(experiment_result)
        flat_results.append(flat_result)

    return flat_results


def rank_experiment_results(experiment_results):
    """
    Rank experiment results by quality.

    Input:
        experiment_results ... list of experiment-result dictionaries

    Return:
        ranked_results ...... new list sorted by:
                              1. accuracy descending
                              2. processing_time_total ascending
                              3. num_samples descending

    Why this helper exists:
    In this assignment, higher accuracy is the main criterion. When accuracies
    tie, faster total processing is preferred.
    """

    if not isinstance(experiment_results, list):
        raise TypeError("experiment_results must be a list.")

    ranked_results = sorted(
        experiment_results,
        key=lambda result: (
            -(result.get("accuracy", 0.0)),
            result.get("processing_time_total", float("inf")),
            -(result.get("num_samples", 0)),
        ),
    )

    return ranked_results


def save_experiment_results_csv(experiment_results, csv_path):
    """
    Save experiment results as a flat CSV file.

    Inputs:
        experiment_results ... list of experiment-result dictionaries
        csv_path ............ output CSV path

    Return:
        csv_path ............ pathlib.Path to the saved CSV file

    Why this function exists:
    CSV is the most convenient format for:
    - quick inspection
    - spreadsheet sorting/filtering
    - later reporting
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    flat_results = flatten_experiment_results(experiment_results)

    if not flat_results:
        # still create an empty file with no rows
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return csv_path

    # collect union of all keys across all rows
    fieldnames = sorted(
        {
            key
            for flat_result in flat_results
            for key in flat_result.keys()
        }
    )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for flat_result in flat_results:
            writer.writerow(flat_result)

    return csv_path


def _format_config_for_summary(config):
    """
    Format one config dictionary into a short JSON-like one-line string.

    Input:
        config ... dict or None

    Return:
        formatted_config ... string
    """

    if config is None:
        return "{}"

    if not isinstance(config, dict):
        return str(config)

    return json.dumps(config, ensure_ascii=False, sort_keys=True)


def save_experiment_summary_text(
    experiment_results,
    summary_path,
    top_k=10,
):
    """
    Save a readable ranked text summary of experiment results.

    Inputs:
        experiment_results ... list of experiment-result dictionaries
        summary_path ....... output text-file path
        top_k .............. number of top-ranked results to include in detail

    Return:
        summary_path ....... pathlib.Path to the saved summary file

    Why this function exists:
    A text summary is easier to read quickly than a full CSV file and is useful
    for reports or quick inspection of the best configurations.
    """

    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer.")

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ranked_results = rank_experiment_results(experiment_results)

    lines = []
    lines.append("Parking LBP Experiment Summary")
    lines.append("=" * 80)
    lines.append(f"Total experiment count: {len(ranked_results)}")
    lines.append("Ranking rule: accuracy descending, then total processing time ascending.")
    lines.append("")

    if not ranked_results:
        lines.append("No experiment results available.")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return summary_path

    best_result = ranked_results[0]
    best_confusion = best_result.get("confusion_counts", {})

    lines.append("Best result:")
    lines.append(f"  accuracy: {best_result.get('accuracy', '<missing>')}")
    lines.append(f"  num_samples: {best_result.get('num_samples', '<missing>')}")
    lines.append(
        f"  confusion_counts: tp={best_confusion.get('tp', '<missing>')}, "
        f"tn={best_confusion.get('tn', '<missing>')}, "
        f"fp={best_confusion.get('fp', '<missing>')}, "
        f"fn={best_confusion.get('fn', '<missing>')}"
    )
    lines.append(
        f"  processing_time_total: "
        f"{best_result.get('processing_time_total', '<missing>')}"
    )
    lines.append(
        f"  preprocessing_config: "
        f"{_format_config_for_summary(best_result.get('preprocessing_config'))}"
    )
    lines.append(
        f"  lbp_config: "
        f"{_format_config_for_summary(best_result.get('lbp_config'))}"
    )
    lines.append(
        f"  classifier_config: "
        f"{_format_config_for_summary(best_result.get('classifier_config'))}"
    )
    lines.append("")

    lines.append(f"Top {min(top_k, len(ranked_results))} ranked results:")
    lines.append("-" * 80)

    for rank_index, result in enumerate(ranked_results[:top_k], start=1):
        confusion = result.get("confusion_counts", {})

        lines.append(f"Rank #{rank_index}")
        lines.append(f"  accuracy: {result.get('accuracy', '<missing>')}")
        lines.append(f"  num_samples: {result.get('num_samples', '<missing>')}")
        lines.append(
            f"  confusion_counts: tp={confusion.get('tp', '<missing>')}, "
            f"tn={confusion.get('tn', '<missing>')}, "
            f"fp={confusion.get('fp', '<missing>')}, "
            f"fn={confusion.get('fn', '<missing>')}"
        )
        lines.append(
            f"  processing_time_total: "
            f"{result.get('processing_time_total', '<missing>')}"
        )
        lines.append(
            f"  preprocessing_config: "
            f"{_format_config_for_summary(result.get('preprocessing_config'))}"
        )
        lines.append(
            f"  lbp_config: "
            f"{_format_config_for_summary(result.get('lbp_config'))}"
        )
        lines.append(
            f"  classifier_config: "
            f"{_format_config_for_summary(result.get('classifier_config'))}"
        )
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return summary_path


def save_experiment_outputs(
    experiment_results,
    output_dir,
    csv_filename="experiment_results.csv",
    summary_filename="experiment_summary.txt",
    top_k=10,
):
    """
    Save both CSV and text-summary outputs for a list of experiment results.

    Inputs:
        experiment_results ... list of experiment-result dictionaries
        output_dir ........... output directory
        csv_filename ......... CSV filename inside output_dir
        summary_filename ..... text-summary filename inside output_dir
        top_k ................ number of detailed top results in text summary

    Return:
        saved_outputs ........ dictionary containing:
                              - output_dir
                              - csv_path
                              - summary_path

    Why this function exists:
    Main experiment scripts usually want one simple call that saves all
    standard result artifacts together.
    """

    output_dir = ensure_output_directory(output_dir)

    csv_path = output_dir / csv_filename
    summary_path = output_dir / summary_filename

    save_experiment_results_csv(
        experiment_results=experiment_results,
        csv_path=csv_path,
    )

    save_experiment_summary_text(
        experiment_results=experiment_results,
        summary_path=summary_path,
        top_k=top_k,
    )

    saved_outputs = {
        "output_dir": output_dir,
        "csv_path": csv_path,
        "summary_path": summary_path,
    }

    return saved_outputs