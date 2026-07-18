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
# ---------------------------------------------------------------------------
# Module orientation:
# This module is the output formatting and persistence layer for experiment
# results. Upstream code produces nested experiment-result dictionaries; this
# module converts those structured results into a flat CSV table and a readable
# text summary. The key design idea is that experiment execution and result
# saving remain separate concerns, so the orchestration code can focus on
# running experiments while this module focuses on making their outputs easy to
# inspect and report.
# ---------------------------------------------------------------------------
#
# This file is the final reporting boundary of the main experiment pipeline.
# Earlier modules are responsible for producing experiment_result dictionaries
# that contain:
# - nested configuration blocks
# - evaluation metrics
# - timing information
# - optional extra metadata
#
# This module answers the next practical question:
# "How should those results be turned into human-usable output files?"
#
# The module therefore has three main jobs:
# 1. flatten rich nested experiment dictionaries into CSV-friendly row structures
# 2. define the project's official ranking rule for "best result"
# 3. save the results in two complementary forms:
#    - a machine-friendly flat CSV
#    - a human-friendly text summary
#
# The overall information flow here is:
# experiment_result dictionaries
#   -> flattened row dictionaries
#   -> ranked result ordering
#   -> output files on disk
#
# This separation is important because experiment execution and result reporting
# are different responsibilities. The search/orchestration layer should focus on
# producing correct results, while this module focuses on making them easy to
# inspect, compare, and archive.

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

    # This small helper normalizes the output-directory handling used throughout
    # the module.
    #
    # All file-saving functions eventually need a valid directory to write into.
    # Centralizing the conversion to Path plus directory creation here keeps the
    # later saving functions focused on formatting and writing content instead of
    # repeatedly handling the same filesystem preparation work.

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

    # The CSV writer expects one flat dictionary per row. This helper performs the
    # recursive conversion from nested dict structure to a single-level key-value
    # structure, preserving the original hierarchy by prefixing the keys.
    #
    # This matters because experiment results are naturally nested:
    # - preprocessing_config is a dictionary
    # - lbp_config is a dictionary
    # - classifier_config is a dictionary
    # - confusion_counts is a dictionary
    #
    # A CSV row, however, needs simple column/value pairs. Prefixing preserves the
    # meaning of nested fields after flattening, so values such as "neighbors" or
    # "C" do not lose the context of which config block they came from.
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError("data passed to flatten_nested_dict(...) must be a dict or None.")

    flat_dict = {}

    # Walk through the dictionary one item at a time.
    # For nested dictionaries, recurse deeper while extending the prefix.
    # For lists and tuples, convert them to string form because CSV cells store text.
    # For scalar values, store them directly under the fully prefixed key.
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

    # This function defines how one full experiment_result dictionary is converted
    # into one reportable CSV row.
    #
    # It preserves the most important structure in three stages:
    # 1. flatten the known nested config / metric blocks under explicit prefixes
    # 2. copy the standard scalar fields that are especially useful for sorting and
    #    comparison
    # 3. keep any other extra fields in a reasonable CSV-friendly form instead of
    #    silently discarding them
    #
    # In other words, this is the module's main "rich result -> flat row" rule.

    if not isinstance(experiment_result, dict):
        raise TypeError("experiment_result must be a dictionary.")

    flat_result = {}

    # -------------------------------------------------------------------------
    # 1. flatten known config blocks
    # -------------------------------------------------------------------------
    # These are the standard nested structures that the rest of the project is
    # expected to produce. Flattening them first under well-chosen prefixes creates
    # readable and traceable CSV columns such as:
    # - preprocessing_target_size
    # - lbp_neighbors
    # - classifier_C
    # - evaluation_occupied_label
    # - confusion_tp
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
    # These are the scalar fields that are expected to be useful for direct sorting,
    # filtering, and reporting in the output CSV.
    #
    # They are copied explicitly because they are especially important summary values
    # that callers and readers are likely to care about directly.
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
    # This last pass keeps the flattener flexible. If the experiment result contains
    # extra fields beyond the expected standard ones, they are still preserved in a
    # reasonable CSV-friendly form instead of being silently dropped.
    #
    # This is useful because experiment_result dictionaries may evolve over time.
    # The flattener should be permissive enough to preserve extra information when it
    # can be represented sensibly.
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

    # This is the batch version of flatten_experiment_result(...).
    # It converts a whole result collection into a list of CSV-ready row
    # dictionaries while preserving experiment order.
    #
    # The returned flat_results list is what later CSV-writing logic expects.

    if not isinstance(experiment_results, list):
        raise TypeError("experiment_results must be a list.")

    flat_results = []

    # Each structured experiment result is flattened independently, producing one CSV
    # row dictionary per experiment.
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

    # This function defines the project's official meaning of "best result".
    # That definition is reused across:
    # - experiment_search.py when producing ranked_results
    # - save_experiment_summary_text(...) when presenting top-ranked experiments
    # - main.py when printing the best overall result
    #
    # The ranking rule is:
    # 1. higher accuracy is always better
    # 2. if accuracy ties, lower total runtime is better
    # 3. if both still tie, larger evaluated sample count is preferred
    #
    # Encoding this in one place guarantees that ranking is consistent everywhere.

    if not isinstance(experiment_results, list):
        raise TypeError("experiment_results must be a list.")

    # The ranking rule is encoded explicitly here so the whole project uses one
    # consistent definition of "best result". Accuracy dominates, total runtime breaks
    # ties, and sample count is used as a final tie-breaker.
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

    # This is the machine-friendly persistence path of the module.
    # It takes structured experiment results, flattens them into row dictionaries,
    # computes the union of all available columns, and writes everything into a CSV
    # file that can later be inspected, sorted, filtered, or reloaded.
    #
    # The CSV output is especially useful because:
    # - experiment rows can be compared directly
    # - configurations and metrics are easy to scan
    # - inspect_best_config.py can later reconstruct the best configuration from it

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    flat_results = flatten_experiment_results(experiment_results)

    if not flat_results:
        # still create an empty file with no rows
        # This keeps the output behavior predictable even if the caller provides an
        # empty result list.
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return csv_path

    # collect union of all keys across all rows
    # Because different experiment results may contain slightly different fields, the
    # CSV header is built from the union of all keys appearing in all flattened rows.
    #
    # Sorting the field names gives the CSV a stable deterministic column order.
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

        # Write one flat experiment row at a time.
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

    # The text summary is meant for quick reading, so config dictionaries are rendered
    # as compact one-line JSON-like strings rather than as multi-line structures.
    #
    # This helper keeps that formatting rule centralized so the summary-writing code
    # remains readable and consistent.
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

    # This is the human-friendly reporting path of the module.
    # Unlike the CSV writer, which aims for structured completeness, this function
    # aims for quick readability. It produces a ranked plain-text report that:
    # - states the ranking rule
    # - highlights the best overall result
    # - lists the top-k ranked experiment summaries
    #
    # The text summary is useful when someone wants a concise report without opening
    # a spreadsheet or reconstructing meaning from raw CSV columns.

    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer.")

    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Always rank the provided results before summarizing them so the text summary
    # consistently reflects the project's official ranking rule.
    ranked_results = rank_experiment_results(experiment_results)

    lines = []
    lines.append("Parking LBP Experiment Summary")
    lines.append("=" * 80)
    lines.append(f"Total experiment count: {len(ranked_results)}")
    lines.append("Ranking rule: accuracy descending, then total processing time ascending.")
    lines.append("")

    if not ranked_results:
        # Even in the empty-results case, still produce a small informative summary
        # file instead of failing or producing nothing.
        lines.append("No experiment results available.")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return summary_path

    # The first ranked item is the best overall result, so it gets its own highlighted
    # section before the more general top-k ranking list.
    best_result = ranked_results[0]
    best_confusion = best_result.get("confusion_counts", {})

    # Start with a dedicated best-result block so the most important outcome is
    # immediately visible without scanning the whole ranking list.
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

    # The loop below produces one compact human-readable block per ranked result.
    # Each block contains the main metrics plus the three configuration groups that
    # define that experiment.
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
        output_dir ........... destination directory
        csv_filename ......... output CSV filename
        summary_filename ..... output summary filename
        top_k ................ number of top-ranked results to include in summary

    Return:
        saved_outputs ........ dictionary containing:
                               - output_dir
                               - csv_path
                               - summary_path

    Why this function exists:
    Most callers want both standard output formats, so this convenience wrapper
    saves them together and returns the produced paths in one package.
    """

    # This wrapper is the main external entry point of the module.
    # It coordinates the two standard reporting outputs expected by the rest of the
    # project:
    # - CSV file for structured inspection and later reloading
    # - text summary file for quick human-readable reporting
    #
    # Returning all produced paths in one dictionary keeps the caller-side API
    # compact and makes main.py simpler.

    output_dir = ensure_output_directory(output_dir)

    csv_path = output_dir / csv_filename
    summary_path = output_dir / summary_filename

    # Save the machine-friendly flat table first.
    save_experiment_results_csv(
        experiment_results=experiment_results,
        csv_path=csv_path,
    )

    # Save the human-friendly ranked summary next.
    save_experiment_summary_text(
        experiment_results=experiment_results,
        summary_path=summary_path,
        top_k=top_k,
    )

    # Package the produced output paths in one standard return structure.
    saved_outputs = {
        "output_dir": output_dir,
        "csv_path": csv_path,
        "summary_path": summary_path,
    }

    return saved_outputs