"""
experiment_search.py

Purpose of this module:
- generate multi-configuration experiment combinations
- run end-to-end parking LBP experiments for each configuration
- aggregate evaluation across the test dataset
- return experiment-result dictionaries ready for results_io.py

Why this module exists:
At this stage of the project, the individual building blocks already exist:
- parking map and test images can be loaded
- training data can be loaded
- parking-space ROIs can be extracted
- ROIs can be preprocessed
- LBP features can be computed
- classifiers can be trained and applied
- prediction records can be evaluated against ground truth
- result dictionaries can be saved by results_io.py

The next logical stage is to connect these pieces into a systematic
multi-configuration experiment runner.

This module currently provides:
- ensure_config_list(...)
- build_experiment_configurations(...)
- run_one_experiment(...)
- run_experiment_search(...)
"""
# ---------------------------------------------------------------------------
# Module orientation:
# This module is the experiment orchestration layer of the parking LBP pipeline.
# It does not implement feature extraction or classification itself. Instead, it
# combines configurations, loads shared inputs once, runs complete end-to-end
# experiments for each configuration, measures processing times, aggregates
# evaluation results across the test dataset, and returns standardized
# experiment-result dictionaries that can later be ranked and saved.
# ---------------------------------------------------------------------------

from itertools import product
from pathlib import Path
import time

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
from evaluation import (
    evaluate_one_test_case,
    merge_confusion_counts,
    compute_accuracy,
)
from results_io import rank_experiment_results


def ensure_config_list(configs, config_name):
    """
    Ensure the given config collection is a list of dictionaries.

    Inputs:
        configs ...... expected list of dictionaries
        config_name .. name used in error messages

    Return:
        configs ...... validated list

    Why this helper exists:
    Multi-configuration search is built from Cartesian products of config lists.
    Validating those inputs early makes orchestration safer and error messages
    clearer.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(configs, list):
        raise TypeError(f"{config_name} must be a list.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not configs:
        raise ValueError(f"{config_name} must not be empty.")

    # Process items in a deterministic order so the produced outputs stay aligned with
    # the corresponding inputs, labels, or metadata.
    for index, config in enumerate(configs, start=1):
        if not isinstance(config, dict):
            raise TypeError(
                f"Each item in {config_name} must be a dictionary. "
                f"Item #{index} has type: {type(config).__name__}"
            )

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return configs


def build_experiment_configurations(
    preprocessing_configurations,
    lbp_configurations,
    classifier_configurations,
    evaluation_config=None,
):
    """
    Build the full Cartesian product of experiment configurations.

    Inputs:
        preprocessing_configurations ... list of preprocessing config dicts
        lbp_configurations ........... list of LBP config dicts
        classifier_configurations .... list of classifier config dicts
        evaluation_config ............ optional shared evaluation config dict

    Return:
        experiment_configurations .... list of dictionaries, each containing:
                                       - experiment_index
                                       - preprocessing_config
                                       - lbp_config
                                       - classifier_config
                                       - evaluation_config

    Why this function exists:
    The assignment requires experimenting with multiple LBP configurations and
    reporting accuracy and processing time. This helper creates the set of
    concrete experiment combinations that will later be run end-to-end.
    """

    preprocessing_configurations = ensure_config_list(
        preprocessing_configurations,
        "preprocessing_configurations",
    )
    lbp_configurations = ensure_config_list(
        lbp_configurations,
        "lbp_configurations",
    )
    classifier_configurations = ensure_config_list(
        classifier_configurations,
        "classifier_configurations",
    )

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if evaluation_config is None:
        evaluation_config = {
            "occupied_label": 1,
            "empty_label": 0,
        }

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not isinstance(evaluation_config, dict):
        raise TypeError("evaluation_config must be a dictionary or None.")

    experiment_configurations = []

    # The Cartesian product below explicitly constructs every experiment candidate:
    # every preprocessing setup is paired with every LBP setup and every classifier
    # setup, so the later search step can evaluate the entire requested grid.
    for experiment_index, (
        preprocessing_config,
        lbp_config,
        classifier_config,
    ) in enumerate(
        product(
            preprocessing_configurations,
            lbp_configurations,
            classifier_configurations,
        ),
        start=1,
    ):
        experiment_configuration = {
            "experiment_index": experiment_index,
            "preprocessing_config": preprocessing_config,
            "lbp_config": lbp_config,
            "classifier_config": classifier_config,
            "evaluation_config": evaluation_config,
        }
        experiment_configurations.append(experiment_configuration)

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return experiment_configurations


def run_one_experiment(
    training_records,
    test_cases,
    parking_map,
    preprocessing_config,
    lbp_config,
    classifier_config,
    evaluation_config=None,
):
    """
    Run one full parking-LBP experiment across the selected test dataset.

    Inputs:
        training_records ....... list of training-record dictionaries
        test_cases ............. list of test-case dictionaries from parking_io.py
        parking_map ............ list of parking-space polygons
        preprocessing_config ... one preprocessing config dict
        lbp_config ............. one LBP config dict
        classifier_config ...... one classifier config dict
        evaluation_config ...... optional label-convention dict

    Return:
        experiment_result ...... dictionary containing:
                                 - configs
                                 - confusion_counts
                                 - accuracy
                                 - num_samples
                                 - timing fields
                                 - per_image_results

    Overall idea:
    1. prepare training features
    2. train classifier once
    3. for each test image:
       - extract ROIs
       - prepare test features
       - predict
       - evaluate
    4. aggregate confusion counts and timing
    """

    # Validate the main external inputs first so the whole experiment either runs on a
    # consistent dataset or fails immediately with a clear message.
    if not isinstance(training_records, list):
        raise TypeError("training_records must be a list.")

    if not isinstance(test_cases, list):
        raise TypeError("test_cases must be a list.")

    if not test_cases:
        raise ValueError("test_cases must not be empty.")

    if not isinstance(parking_map, list):
        raise TypeError("parking_map must be a list.")

    if evaluation_config is None:
        evaluation_config = {
            "occupied_label": 1,
            "empty_label": 0,
        }

    if not isinstance(preprocessing_config, dict):
        raise TypeError("preprocessing_config must be a dictionary.")

    if not isinstance(lbp_config, dict):
        raise TypeError("lbp_config must be a dictionary.")

    if not isinstance(classifier_config, dict):
        raise TypeError("classifier_config must be a dictionary.")

    if not isinstance(evaluation_config, dict):
        raise TypeError("evaluation_config must be a dictionary.")

    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # 1. training feature preparation
    # -------------------------------------------------------------------------
    # This phase transforms the raw training image records into LBP feature vectors.
    # The produced X_train / y_train pair is the reusable classifier input that will
    # be fitted once and then applied to all test images in this experiment.
    training_feature_prep_start = time.perf_counter()

    training_feature_records = prepare_training_feature_records(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    X_train, y_train, training_metadata = build_training_matrix_and_labels(
        training_feature_records=training_feature_records
    )

    training_feature_prep_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 2. model training
    # -------------------------------------------------------------------------
    # The classifier is trained once per experiment configuration. This makes the
    # later per-image testing stage reflect a realistic "train once, predict many"
    # workflow rather than retraining separately for each test image.
    training_start = time.perf_counter()

    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=classifier_config,
    )

    training_end = time.perf_counter()

    # -------------------------------------------------------------------------
    # 3. evaluate across all test images
    # -------------------------------------------------------------------------
    # Dataset-level aggregation starts here. Each individual test image is processed
    # in parking-map order, evaluated against its own ground-truth file, and then
    # merged into the running experiment totals.
    dataset_confusion_counts = initialize_confusion_counts_for_search()
    dataset_num_samples = 0

    total_test_feature_preparation_time = 0.0
    total_prediction_time = 0.0

    per_image_results = []

    # Each test case follows the same end-to-end path:
    # full scene image -> ROI records -> preprocessed test features -> predictions
    # -> image-level evaluation -> dataset-level accumulation.
    for test_case in test_cases:
        image_name = test_case["name"]
        image = test_case["image"]
        txt_path = test_case["txt_path"]

        # 3a. ROI extraction
        roi_records = extract_all_rois_from_image(
            image=image,
            parking_map=parking_map,
            image_name=image_name,
        )

        # 3b. test feature preparation
        # This timing block measures how long it takes to turn one test image into
        # classifier-ready LBP feature vectors. That includes preprocessing and LBP
        # extraction but not model training.
        test_feature_prep_start = time.perf_counter()

        test_feature_records = prepare_test_feature_records(
            test_roi_records=roi_records,
            preprocessing_config=preprocessing_config,
            lbp_config=lbp_config,
        )

        X_test, test_metadata = build_test_matrix(test_feature_records)

        test_feature_prep_end = time.perf_counter()
        test_feature_prep_time = test_feature_prep_end - test_feature_prep_start
        total_test_feature_preparation_time += test_feature_prep_time

        # 3c. prediction
        # This timing block isolates classifier inference itself. Score prediction is
        # optional at model level but is attempted here because later inspection and
        # debugging tools can use the score values if they are available.
        prediction_start = time.perf_counter()

        predicted_labels = predict_labels(model=model, X_test=X_test)
        predicted_scores = predict_scores(model=model, X_test=X_test)

        prediction_records = build_prediction_records(
            feature_records=test_feature_records,
            predicted_labels=predicted_labels,
            predicted_scores=predicted_scores,
        )

        prediction_end = time.perf_counter()
        prediction_time = prediction_end - prediction_start
        total_prediction_time += prediction_time

        # 3d. evaluation
        # Image-level evaluation compares the predicted labels for this one test image
        # against the labels loaded from the matching testX.txt file. The result is
        # then merged into the running dataset-level totals.
        image_evaluation = evaluate_one_test_case(
            prediction_records=prediction_records,
            txt_path=txt_path,
            evaluation_config=evaluation_config,
        )

        dataset_confusion_counts = merge_confusion_counts(
            dataset_confusion_counts,
            image_evaluation["confusion_counts"],
        )
        dataset_num_samples += image_evaluation["num_samples"]

        per_image_result = {
            "source_image_name": image_name,
            "txt_path": txt_path,
            "num_samples": image_evaluation["num_samples"],
            "accuracy": image_evaluation["accuracy"],
            "confusion_counts": image_evaluation["confusion_counts"],
            "processing_time_test_feature_preparation": test_feature_prep_time,
            "processing_time_prediction": prediction_time,
            "roi_count": len(roi_records),
            "test_feature_count": X_test.shape[1],
            "test_metadata_count": len(test_metadata),
        }
        per_image_results.append(per_image_result)

    total_end = time.perf_counter()

    processing_time_total = total_end - total_start
    processing_time_training_feature_preparation = (
        training_feature_prep_end - training_feature_prep_start
    )
    processing_time_training = training_end - training_start

    dataset_accuracy = compute_accuracy(dataset_confusion_counts)

    # The final experiment_result dictionary is the standard package that later modules
    # know how to rank, save, summarize, and inspect. It contains both the chosen
    # configurations and the measured outcomes for that exact experiment run.
    experiment_result = {
        "preprocessing_config": preprocessing_config,
        "lbp_config": lbp_config,
        "classifier_config": classifier_config,
        "evaluation_config": evaluation_config,
        "confusion_counts": dataset_confusion_counts,
        "accuracy": dataset_accuracy,
        "num_samples": dataset_num_samples,
        "processing_time_total": processing_time_total,
        "processing_time_training_feature_preparation": (
            processing_time_training_feature_preparation
        ),
        "processing_time_training": processing_time_training,
        "processing_time_test_feature_preparation": (
            total_test_feature_preparation_time
        ),
        "processing_time_prediction": total_prediction_time,
        "processing_time_per_image": (
            processing_time_total / len(test_cases)
            if len(test_cases) > 0
            else 0.0
        ),
        "processing_time_per_roi": (
            processing_time_total / dataset_num_samples
            if dataset_num_samples > 0
            else 0.0
        ),
        "training_sample_count": len(training_records),
        "training_feature_count": X_train.shape[1],
        "training_metadata_count": len(training_metadata),
        "test_image_count": len(test_cases),
        "parking_space_count": len(parking_map),
        "per_image_results": per_image_results,
    }

    # Return the finalized experiment package for ranking and saving.
    return experiment_result


def initialize_confusion_counts_for_search():
    """
    Create an empty confusion-count dictionary for experiment aggregation.

    Return:
        confusion_counts ... dictionary with tp, tn, fp, fn

    Why this helper exists:
    experiment_search.py aggregates per-image evaluation into dataset-level
    totals, so it needs the same confusion-count structure as evaluation.py.
    """

    # This local helper mirrors evaluation.initialize_confusion_counts() but keeps the
    # search module self-contained when it needs a fresh accumulator structure.
    return {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }


def run_experiment_search(
    training_root,
    map_path,
    test_images_dir,
    preprocessing_configurations,
    lbp_configurations,
    classifier_configurations,
    evaluation_config=None,
    max_test_cases=None,
):
    """
    Run the full multi-configuration experiment search.

    Inputs:
        training_root ................. path to data/training
        map_path ...................... path to data/parking_map_python.txt
        test_images_dir ............... path to data/test_images_zao
        preprocessing_configurations .. list of preprocessing config dicts
        lbp_configurations ............ list of LBP config dicts
        classifier_configurations ..... list of classifier config dicts
        evaluation_config ............. optional label-convention dict
        max_test_cases ................ optional limit on number of test images

    Return:
        search_result ................. dictionary containing:
                                        - experiment_configurations
                                        - experiment_results
                                        - ranked_results

    Why this function exists:
    This is the main orchestration entry point for systematic parking-LBP
    experiments. It combines:
    - shared input loading
    - configuration combination generation
    - per-configuration experiment execution
    - ranking of the final results

    The assignment explicitly requires experimenting with multiple LBP
    configurations and reporting accuracy and processing time for each.
    """

    training_root = Path(training_root)
    map_path = Path(map_path)
    test_images_dir = Path(test_images_dir)

    experiment_configurations = build_experiment_configurations(
        preprocessing_configurations=preprocessing_configurations,
        lbp_configurations=lbp_configurations,
        classifier_configurations=classifier_configurations,
        evaluation_config=evaluation_config,
    )

    # -------------------------------------------------------------------------
    # load shared inputs once
    # -------------------------------------------------------------------------
    # Shared data is loaded only once before the search loop begins. This keeps the
    # experiment comparison fair and avoids repeating dataset I/O for each candidate
    # configuration.
    training_records = load_all_training_records(training_root)
    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(test_images_dir)

    if max_test_cases is not None:
        if not isinstance(max_test_cases, int):
            raise TypeError("max_test_cases must be an integer or None.")

        if max_test_cases <= 0:
            raise ValueError("max_test_cases must be positive if provided.")

        test_cases = test_cases[:max_test_cases]

    if not test_cases:
        raise ValueError("No test cases available for experiment search.")

    experiment_results = []

    # The loop below is the actual search. Each experiment configuration is executed
    # independently on the same shared dataset inputs, then its result dictionary is
    # appended to the master results list for later ranking.
    for experiment_configuration in experiment_configurations:
        experiment_result = run_one_experiment(
            training_records=training_records,
            test_cases=test_cases,
            parking_map=parking_map,
            preprocessing_config=experiment_configuration["preprocessing_config"],
            lbp_config=experiment_configuration["lbp_config"],
            classifier_config=experiment_configuration["classifier_config"],
            evaluation_config=experiment_configuration["evaluation_config"],
        )

        experiment_result = {
            "experiment_index": experiment_configuration["experiment_index"],
            **experiment_result,
        }

        experiment_results.append(experiment_result)

    ranked_results = rank_experiment_results(experiment_results)

    search_result = {
        "experiment_configurations": experiment_configurations,
        "experiment_results": experiment_results,
        "ranked_results": ranked_results,
    }

    # Return the full search package so the caller can both inspect the raw runs and
    # use the already-ranked ordering.
    return search_result