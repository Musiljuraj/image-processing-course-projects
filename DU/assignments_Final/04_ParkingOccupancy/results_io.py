"""
results_io.py

Purpose of this module:
- create and manage the results output directory
- save the exhaustive-search results table to a machine-friendly text file
- save a human-readable summary of the best-ranked results

Why this module exists:
The experiment runner should focus on:
- generating valid configurations
- evaluating those configurations
- ranking the results

Saving results to files is a separate responsibility, so it is cleaner to keep
that logic in its own module.

Current responsibilities:
- prepare outputs/results/ if needed
- flatten ranked experiment results into CSV rows
- write the full ranked table to results.csv
- write a compact best-results summary to best_results.txt

Important note:
CSV is used for the main results table because it is still a text file,
but it is much easier to inspect, sort, or import into spreadsheet tools later.
"""

import csv
from pathlib import Path


def ensure_results_directory(results_dir):
    """
    Ensure that the results directory exists.

    Input:
        results_dir ... path-like object pointing to outputs/results/

    Return:
        results_dir ... normalized Path object

    Why this helper exists:
    File-writing functions should not need to repeat directory-creation logic.
    """

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def flatten_experiment_result(experiment_result):
    """
    Flatten one ranked experiment result into a dictionary suitable for CSV.

    Input:
        experiment_result ... one result dictionary returned by
                              experiment_search.run_exhaustive_search(...)

    Return:
        flat_result ........ flat dictionary with scalar fields only

    Why this helper exists:
    The ranked result contains nested configuration dictionaries.
    A CSV row needs one flat set of columns.
    """

    preprocessing_config = experiment_result["preprocessing_config"]
    edge_detection_config = experiment_result["edge_detection_config"]
    classification_evaluation_config = experiment_result[
        "classification_evaluation_config"
    ]

    detector_name = edge_detection_config["detector_name"]

    sobel_ksize = ""
    sobel_threshold = ""
    canny_threshold1 = ""
    canny_threshold2 = ""
    canny_aperture_size = ""
    canny_l2gradient = ""

    if detector_name == "sobel":
        sobel_ksize = edge_detection_config["sobel"]["ksize"]
        sobel_threshold = edge_detection_config["sobel"]["threshold"]

    elif detector_name == "canny":
        canny_threshold1 = edge_detection_config["canny"]["threshold1"]
        canny_threshold2 = edge_detection_config["canny"]["threshold2"]
        canny_aperture_size = edge_detection_config["canny"]["aperture_size"]
        canny_l2gradient = edge_detection_config["canny"]["l2gradient"]

    flat_result = {
        "rank": experiment_result["rank"],
        "experiment_index": experiment_result["experiment_index"],
        "accuracy": experiment_result["accuracy"],
        "tp": experiment_result["tp"],
        "tn": experiment_result["tn"],
        "fp": experiment_result["fp"],
        "fn": experiment_result["fn"],
        "num_samples": experiment_result["num_samples"],
        "num_images": experiment_result["num_images"],
        "filter_name": preprocessing_config["filter_name"],
        "kernel_size": preprocessing_config["kernel_size"],
        "detector_name": detector_name,
        "sobel_ksize": sobel_ksize,
        "sobel_threshold": sobel_threshold,
        "canny_threshold1": canny_threshold1,
        "canny_threshold2": canny_threshold2,
        "canny_aperture_size": canny_aperture_size,
        "canny_l2gradient": canny_l2gradient,
        "occupancy_threshold_ratio": classification_evaluation_config[
            "occupancy_threshold_ratio"
        ],
        "occupied_label": classification_evaluation_config["occupied_label"],
        "empty_label": classification_evaluation_config["empty_label"],
    }

    return flat_result


def save_ranked_results_csv(search_summary, output_path):
    """
    Save the full ranked experiment table to a CSV file.

    Inputs:
        search_summary ... dictionary returned by experiment_search.run_exhaustive_search(...)
        output_path .... full destination path, typically outputs/results/results.csv

    Why this function exists:
    The full exhaustive-search output is best stored in a machine-friendly form
    so it can be inspected later, sorted externally, or loaded into a spreadsheet.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked_results = search_summary["ranked_results"]

    if not ranked_results:
        raise ValueError("Cannot save CSV because ranked_results is empty.")

    flat_results = [flatten_experiment_result(result) for result in ranked_results]
    fieldnames = list(flat_results[0].keys())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_results)


def save_best_results_summary(search_summary, output_path, top_n=10):
    """
    Save a human-readable text summary of the best-ranked configurations.

    Inputs:
        search_summary ... dictionary returned by experiment_search.run_exhaustive_search(...)
        output_path .... full destination path, typically outputs/results/best_results.txt
        top_n .......... how many top-ranked results to include in the summary

    Why this function exists:
    A full CSV is excellent for machine processing, but a concise text summary is
    much easier to read quickly when you want to know:
    - how many configurations were tested
    - which one was best
    - what the top few candidates were
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked_results = search_summary["ranked_results"]
    best_result = search_summary["best_result"]
    total_configurations = search_summary["total_configurations"]

    if not ranked_results:
        raise ValueError("Cannot save summary because ranked_results is empty.")

    top_results = ranked_results[: min(top_n, len(ranked_results))]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("EXHAUSTIVE SEARCH SUMMARY\n")
        f.write("=" * 72 + "\n")
        f.write(f"Total tested configurations: {total_configurations}\n")
        f.write(f"Total ranked results: {len(ranked_results)}\n")
        f.write("\n")

        f.write("BEST CONFIGURATION\n")
        f.write("-" * 72 + "\n")
        f.write(f"Rank: {best_result['rank']}\n")
        f.write(f"Experiment index: {best_result['experiment_index']}\n")
        f.write(f"Accuracy: {best_result['accuracy']:.6f}\n")
        f.write(f"TP: {best_result['tp']}\n")
        f.write(f"TN: {best_result['tn']}\n")
        f.write(f"FP: {best_result['fp']}\n")
        f.write(f"FN: {best_result['fn']}\n")
        f.write(f"Num samples: {best_result['num_samples']}\n")
        f.write(f"Num images: {best_result['num_images']}\n")
        f.write("\n")

        f.write("Best configuration details:\n")
        f.write(f"  preprocessing_config = {best_result['preprocessing_config']}\n")
        f.write(f"  edge_detection_config = {best_result['edge_detection_config']}\n")
        f.write(
            "  classification_evaluation_config = "
            f"{best_result['classification_evaluation_config']}\n"
        )
        f.write("\n")

        f.write(f"TOP {len(top_results)} CONFIGURATIONS\n")
        f.write("-" * 72 + "\n")

        for result in top_results:
            f.write(
                f"Rank {result['rank']:>3} | "
                f"Exp {result['experiment_index']:>4} | "
                f"accuracy={result['accuracy']:.6f} | "
                f"TP={result['tp']} TN={result['tn']} "
                f"FP={result['fp']} FN={result['fn']}\n"
            )
            f.write(f"  preprocessing: {result['preprocessing_config']}\n")
            f.write(f"  edge_detection: {result['edge_detection_config']}\n")
            f.write(
                "  classification_evaluation: "
                f"{result['classification_evaluation_config']}\n"
            )
            f.write("\n")