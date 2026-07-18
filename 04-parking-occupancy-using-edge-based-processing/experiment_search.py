"""
experiment_search.py

Purpose of this module:
- accept a search space defined elsewhere (in main.py)
- generate all valid experiment configurations from that search space
- run the full dataset evaluation for every configuration
- rank/sort all experiment results
- identify the best-performing configuration

Important design choice:
The search space itself is NOT defined here.
It is passed in from main.py, exactly as requested.

So the division of responsibilities is:
- main.py: defines the allowed ranges / candidate values
- experiment_search.py: builds valid combinations and runs them

Current responsibilities:
- build preprocessing configurations from the search space
- build edge-detection configurations from the search space
- build classification/evaluation configurations from the search space
- generate all valid full experiment configurations
- evaluate one configuration on the full dataset
- run exhaustive search over all valid configurations
- rank results and identify the best one
"""

from itertools import product

from parking_io import load_ground_truth_labels
from roi_extraction import extract_all_rois_from_image
from preprocessing import preprocess_all_rois
from edge_detection import detect_edges_all_records
from evaluation import (
    classify_all_edge_records,
    compute_accuracy,
    evaluate_one_image,
    initialize_confusion_counts,
    merge_confusion_counts,
)


def build_ground_truth_cache(test_cases):
    """
    Preload ground-truth labels for all test images.

    Input:
        test_cases ... list returned by parking_io.load_test_images(...)

    Return:
        ground_truth_cache ... dictionary mapping txt_path to parsed label list

    Why this helper exists:
    During exhaustive search, the same ground-truth files would otherwise be
    read again for every configuration. That is unnecessary repeated I/O.

    By caching labels once at the beginning:
    - the code becomes faster
    - the evaluation loop becomes simpler
    """

    ground_truth_cache = {}

    for test_case in test_cases:
        txt_path = test_case["txt_path"]
        ground_truth_labels = load_ground_truth_labels(txt_path)
        ground_truth_cache[txt_path] = ground_truth_labels

    return ground_truth_cache


def build_preprocessing_configurations(search_space):
    """
    Build all valid preprocessing configurations from the search space.

    Input:
        search_space ... full search-space dictionary passed from main.py

    Return:
        preprocessing_configurations ... list of dictionaries, for example:
            [
                {"filter_name": "none", "kernel_size": None},
                {"filter_name": "gaussian", "kernel_size": 3},
                {"filter_name": "gaussian", "kernel_size": 5},
                ...
            ]

    Validity rule used here:
    If filter_name == "none", kernel_size is irrelevant.
    Therefore, only one "none" configuration is produced instead of creating
    many redundant copies with different kernel sizes.
    """

    preprocessing_space = search_space["preprocessing"]
    filter_names = preprocessing_space["filter_names"]
    kernel_sizes = preprocessing_space["kernel_sizes"]

    preprocessing_configurations = []

    for filter_name in filter_names:
        normalized_filter_name = filter_name.strip().lower()

        if normalized_filter_name == "none":
            preprocessing_configurations.append(
                {
                    "filter_name": "none",
                    "kernel_size": None,
                }
            )
        else:
            for kernel_size in kernel_sizes:
                preprocessing_configurations.append(
                    {
                        "filter_name": normalized_filter_name,
                        "kernel_size": kernel_size,
                    }
                )

    return preprocessing_configurations


def build_edge_detection_configurations(search_space):
    """
    Build all valid edge-detection configurations from the search space.

    Input:
        search_space ... full search-space dictionary passed from main.py

    Return:
        edge_detection_configurations ... list of dictionaries

    Supported detectors:
    - Sobel
    - Canny

    Validity rules used here:
    - Sobel configurations are formed only from Sobel parameter ranges
    - Canny configurations are formed only from Canny parameter ranges
    - for Canny, only threshold pairs with threshold1 <= threshold2 are kept
    """

    edge_space = search_space["edge_detection"]
    detector_names = edge_space["detector_names"]

    edge_detection_configurations = []

    for detector_name in detector_names:
        normalized_detector_name = detector_name.strip().lower()

        if normalized_detector_name == "sobel":
            sobel_ksizes = edge_space["sobel_ksizes"]
            sobel_thresholds = edge_space["sobel_thresholds"]

            for sobel_ksize, sobel_threshold in product(
                sobel_ksizes,
                sobel_thresholds,
            ):
                edge_detection_configurations.append(
                    {
                        "detector_name": "sobel",
                        "sobel": {
                            "ksize": sobel_ksize,
                            "threshold": sobel_threshold,
                        },
                    }
                )

        elif normalized_detector_name == "canny":
            threshold1_values = edge_space["canny_threshold1_values"]
            threshold2_values = edge_space["canny_threshold2_values"]
            aperture_sizes = edge_space["canny_aperture_sizes"]
            l2gradient_values = edge_space["canny_l2gradient_values"]

            for threshold1, threshold2, aperture_size, l2gradient in product(
                threshold1_values,
                threshold2_values,
                aperture_sizes,
                l2gradient_values,
            ):
                if threshold1 > threshold2:
                    continue

                edge_detection_configurations.append(
                    {
                        "detector_name": "canny",
                        "canny": {
                            "threshold1": threshold1,
                            "threshold2": threshold2,
                            "aperture_size": aperture_size,
                            "l2gradient": l2gradient,
                        },
                    }
                )

        else:
            raise ValueError(
                f"Unsupported detector_name in search space: {detector_name}"
            )

    return edge_detection_configurations


def build_classification_evaluation_configurations(search_space):
    """
    Build all classification/evaluation configurations from the search space.

    Input:
        search_space ... full search-space dictionary passed from main.py

    Return:
        classification_evaluation_configurations ... list of dictionaries

    Why this helper exists:
    The search space can vary the occupancy threshold ratio while keeping the
    label convention fixed.
    """

    classification_space = search_space["classification_evaluation"]

    occupancy_threshold_ratios = classification_space["occupancy_threshold_ratios"]
    occupied_label = classification_space["occupied_label"]
    empty_label = classification_space["empty_label"]

    classification_evaluation_configurations = []

    for occupancy_threshold_ratio in occupancy_threshold_ratios:
        classification_evaluation_configurations.append(
            {
                "occupancy_threshold_ratio": occupancy_threshold_ratio,
                "occupied_label": occupied_label,
                "empty_label": empty_label,
            }
        )

    return classification_evaluation_configurations


def build_valid_experiment_configurations(search_space):
    """
    Build all valid full experiment configurations from the search space.

    Input:
        search_space ... full search-space dictionary passed from main.py

    Return:
        experiment_configurations ... list of dictionaries where each dictionary
                                     contains:
                                     - preprocessing_config
                                     - edge_detection_config
                                     - classification_evaluation_config

    Overall idea:
    This function forms the Cartesian product of:
    - preprocessing configs
    - edge-detection configs
    - classification/evaluation configs

    Because each sub-configuration list already contains only valid entries,
    the resulting full configurations are valid too.
    """

    preprocessing_configurations = build_preprocessing_configurations(search_space)
    edge_detection_configurations = build_edge_detection_configurations(search_space)
    classification_evaluation_configurations = (
        build_classification_evaluation_configurations(search_space)
    )

    experiment_configurations = []

    for (
        preprocessing_config,
        edge_detection_config,
        classification_evaluation_config,
    ) in product(
        preprocessing_configurations,
        edge_detection_configurations,
        classification_evaluation_configurations,
    ):
        experiment_configurations.append(
            {
                "preprocessing_config": preprocessing_config,
                "edge_detection_config": edge_detection_config,
                "classification_evaluation_config": classification_evaluation_config,
            }
        )

    return experiment_configurations


def evaluate_configuration_on_dataset(
    parking_map,
    test_cases,
    experiment_configuration,
    ground_truth_cache,
):
    """
    Evaluate one full experiment configuration on the entire dataset.

    Inputs:
        parking_map ................. list of parking-space polygons
        test_cases .................. list of dataset image records
        experiment_configuration .... dictionary containing:
                                      - preprocessing_config
                                      - edge_detection_config
                                      - classification_evaluation_config
        ground_truth_cache .......... dictionary mapping txt_path to label list

    Return:
        experiment_result ........... dictionary containing:
                                      - preprocessing_config
                                      - edge_detection_config
                                      - classification_evaluation_config
                                      - tp
                                      - tn
                                      - fp
                                      - fn
                                      - accuracy
                                      - num_samples
                                      - num_images
                                      - per_image_summaries

    This function is the core of the final search stage.
    It reuses the already working pipeline for:
    - ROI extraction
    - preprocessing
    - edge detection
    - classification
    - evaluation

    but applies it to exactly one chosen full configuration.
    """

    preprocessing_config = experiment_configuration["preprocessing_config"]
    edge_detection_config = experiment_configuration["edge_detection_config"]
    classification_evaluation_config = experiment_configuration[
        "classification_evaluation_config"
    ]

    dataset_confusion_counts = initialize_confusion_counts()
    total_num_samples = 0
    per_image_summaries = []

    for test_case in test_cases:
        rois = extract_all_rois_from_image(
            image=test_case["image"],
            parking_map=parking_map,
            image_name=test_case["name"],
        )

        preprocessed_rois = preprocess_all_rois(
            rois=rois,
            preprocessing_config=preprocessing_config,
        )

        edge_records = detect_edges_all_records(
            preprocessed_records=preprocessed_rois,
            edge_detection_config=edge_detection_config,
        )

        classified_records = classify_all_edge_records(
            edge_records=edge_records,
            classification_evaluation_config=classification_evaluation_config,
        )

        ground_truth_labels = ground_truth_cache[test_case["txt_path"]]

        image_evaluation = evaluate_one_image(
            classified_records=classified_records,
            ground_truth_labels=ground_truth_labels,
            classification_evaluation_config=classification_evaluation_config,
        )

        image_confusion_counts = image_evaluation["confusion_counts"]
        image_accuracy = image_evaluation["accuracy"]
        num_samples = image_evaluation["num_samples"]

        total_num_samples += num_samples
        dataset_confusion_counts = merge_confusion_counts(
            dataset_confusion_counts,
            image_confusion_counts,
        )

        per_image_summary = {
            "image_name": test_case["name"],
            "num_samples": num_samples,
            "tp": image_confusion_counts["tp"],
            "tn": image_confusion_counts["tn"],
            "fp": image_confusion_counts["fp"],
            "fn": image_confusion_counts["fn"],
            "accuracy": image_accuracy,
        }
        per_image_summaries.append(per_image_summary)

    accuracy = compute_accuracy(dataset_confusion_counts)

    experiment_result = {
        "preprocessing_config": preprocessing_config,
        "edge_detection_config": edge_detection_config,
        "classification_evaluation_config": classification_evaluation_config,
        "tp": dataset_confusion_counts["tp"],
        "tn": dataset_confusion_counts["tn"],
        "fp": dataset_confusion_counts["fp"],
        "fn": dataset_confusion_counts["fn"],
        "accuracy": accuracy,
        "num_samples": total_num_samples,
        "num_images": len(test_cases),
        "per_image_summaries": per_image_summaries,
    }

    return experiment_result


def rank_experiment_results(experiment_results):
    """
    Rank experiment results from best to worst.

    Input:
        experiment_results ... list of experiment-result dictionaries

    Return:
        ranked_results ...... new list sorted from best to worst

    Ranking strategy:
    1. higher accuracy first
    2. lower FP first
    3. lower FN first
    4. higher TP first

    Why this ranking:
    Accuracy is the primary metric required by the assignment.
    The secondary ordering simply makes ties deterministic and somewhat
    meaningful.
    """

    ranked_results = sorted(
        experiment_results,
        key=lambda result: (
            result["accuracy"],
            -result["fp"],
            -result["fn"],
            result["tp"],
        ),
        reverse=True,
    )

    for rank, result in enumerate(ranked_results, start=1):
        result["rank"] = rank

    return ranked_results


def run_exhaustive_search(
    parking_map,
    test_cases,
    search_space,
    verbose=True,
):
    """
    Run exhaustive search over all valid configurations.

    Inputs:
        parking_map ... list of parking-space polygons
        test_cases .... list of dataset image records
        search_space .. search space defined in main.py
        verbose ....... whether to print progress information

    Return:
        search_summary . dictionary containing:
                         - search_space
                         - total_configurations
                         - ranked_results
                         - best_result

    Processing logic:
    1. preload all ground-truth labels
    2. build all valid experiment configurations
    3. evaluate every configuration on the full dataset
    4. rank all results
    5. identify the best one

    This is the top-level search function that main.py should call.
    """

    ground_truth_cache = build_ground_truth_cache(test_cases)
    experiment_configurations = build_valid_experiment_configurations(search_space)

    experiment_results = []

    total_configurations = len(experiment_configurations)

    if verbose:
        print(f"Total valid configurations to test: {total_configurations}")

    for experiment_index, experiment_configuration in enumerate(
        experiment_configurations,
        start=1,
    ):
        if verbose:
            print(
                f"[{experiment_index}/{total_configurations}] "
                "Evaluating configuration..."
            )

        experiment_result = evaluate_configuration_on_dataset(
            parking_map=parking_map,
            test_cases=test_cases,
            experiment_configuration=experiment_configuration,
            ground_truth_cache=ground_truth_cache,
        )

        experiment_result["experiment_index"] = experiment_index
        experiment_results.append(experiment_result)

        if verbose:
            print(
                "  -> "
                f"accuracy={experiment_result['accuracy']:.6f}, "
                f"TP={experiment_result['tp']}, "
                f"TN={experiment_result['tn']}, "
                f"FP={experiment_result['fp']}, "
                f"FN={experiment_result['fn']}"
            )

    ranked_results = rank_experiment_results(experiment_results)
    best_result = ranked_results[0] if ranked_results else None

    search_summary = {
        "search_space": search_space,
        "total_configurations": total_configurations,
        "ranked_results": ranked_results,
        "best_result": best_result,
    }

    return search_summary