"""
step_08_test_experiment_search.py

Simple smoke test for experiment_search.py.

This test verifies:
- the recommended configuration lists can be built,
- the experiment configuration Cartesian product is constructed correctly,
- the quick experiment search can run successfully,
- the returned search result has the expected structure,
- exactly 3 default experiments are produced,
- each experiment result contains accuracy and timing values,
- ranking is produced,
- the experiment-search report can be saved.

The test intentionally uses:
- a reduced frame limit,
- a reduced balanced training subset limit,
so that it remains practical and stable for iterative development.
"""

from pathlib import Path

from experiment_search import (
    get_project_paths,
    ensure_output_directories,
    get_experiment_output_paths,
    get_recommended_experiment_configuration_lists,
    build_experiment_configurations,
    run_experiment_search,
    format_experiment_search_summary,
    save_experiment_search_report,
)


REQUIRED_SEARCH_RESULT_KEYS = {
    "paths",
    "experiment_configurations",
    "experiment_results",
    "ranked_results",
    "max_training_records_per_class",
}

REQUIRED_EXPERIMENT_RESULT_KEYS = {
    "experiment_index",
    "preprocessing_config",
    "lbp_config",
    "classifier_config",
    "training_sample_count",
    "feature_count",
    "class_counts",
    "timing_model_build_total_ms",
    "timing_training_feature_preparation_ms",
    "timing_training_ms",
    "timing_experiment_total_ms",
    "frame_count_processed",
    "stopped_early",
    "max_frames",
    "evaluation_summary",
    "accuracy_percent",
    "compared_count",
    "correct_count",
    "localization_mean_ms",
    "classification_mean_ms",
    "total_frame_mean_ms",
}


def assert_true(condition, message):
    """
    Raise AssertionError with a readable message when the condition is false.
    """
    if not condition:
        raise AssertionError(message)


def main():
    print("=== STEP 08 SMOKE TEST: experiment_search.py ===")
    print()

    # -------------------------------------------------------------
    # 1. Basic path/output preparation
    # -------------------------------------------------------------
    print("[1/5] Checking project paths and experiment output directories...")
    paths = get_project_paths()
    ensure_output_directories(paths)
    output_paths = get_experiment_output_paths(paths)

    assert_true(paths["project_root"].exists(), "project_root does not exist.")
    assert_true(paths["video_path"].exists(), "video_path does not exist.")
    assert_true(paths["ground_truth_path"].exists(), "ground_truth_path does not exist.")
    assert_true(paths["mrl_eyes_dataset_dir"].exists(), "mrl_eyes_dataset_dir does not exist.")
    assert_true(output_paths["experiment_search_report_path"].parent.exists(), "experiment results output directory does not exist.")

    print("[OK] Project paths and output directories are valid.")
    print(f"  project_root: {paths['project_root']}")
    print(f"  video_path:   {paths['video_path']}")
    print(f"  dataset_root: {paths['mrl_eyes_dataset_dir']}")
    print()

    # -------------------------------------------------------------
    # 2. Recommended configuration lists and experiment combinations
    # -------------------------------------------------------------
    print("[2/5] Checking recommended configuration lists...")
    recommended = get_recommended_experiment_configuration_lists()

    assert_true(isinstance(recommended, dict), "Recommended config result must be a dictionary.")
    assert_true("preprocessing_configurations" in recommended, "Missing preprocessing_configurations.")
    assert_true("lbp_configurations" in recommended, "Missing lbp_configurations.")
    assert_true("classifier_configurations" in recommended, "Missing classifier_configurations.")

    preprocessing_configurations = recommended["preprocessing_configurations"]
    lbp_configurations = recommended["lbp_configurations"]
    classifier_configurations = recommended["classifier_configurations"]

    assert_true(len(preprocessing_configurations) == 1, "Expected exactly 1 default preprocessing configuration.")
    assert_true(len(lbp_configurations) == 3, "Expected exactly 3 default LBP configurations.")
    assert_true(len(classifier_configurations) == 1, "Expected exactly 1 default classifier configuration.")

    experiment_configurations = build_experiment_configurations(
        preprocessing_configurations=preprocessing_configurations,
        lbp_configurations=lbp_configurations,
        classifier_configurations=classifier_configurations,
    )

    assert_true(len(experiment_configurations) == 3, "Expected exactly 3 experiment configurations.")

    print("[OK] Recommended configurations are correct.")
    print(f"  preprocessing configs: {len(preprocessing_configurations)}")
    print(f"  lbp configs:           {len(lbp_configurations)}")
    print(f"  classifier configs:    {len(classifier_configurations)}")
    print(f"  total experiments:     {len(experiment_configurations)}")
    print()

    # -------------------------------------------------------------
    # 3. Run quick experiment search
    # -------------------------------------------------------------
    print("[3/5] Running quick experiment search...")
    quick_test_max_frames = 60
    quick_test_max_training_records_per_class = 100

    search_result = run_experiment_search(
        preprocessing_configurations=None,
        lbp_configurations=None,
        classifier_configurations=None,
        max_frames=quick_test_max_frames,
        show_preview=False,
        max_training_records_per_class=quick_test_max_training_records_per_class,
    )

    assert_true(isinstance(search_result, dict), "search_result must be a dictionary.")

    missing_search_keys = REQUIRED_SEARCH_RESULT_KEYS - set(search_result.keys())
    assert_true(not missing_search_keys, f"search_result is missing keys: {sorted(missing_search_keys)}")

    experiment_results = search_result["experiment_results"]
    ranked_results = search_result["ranked_results"]

    assert_true(isinstance(experiment_results, list), "experiment_results must be a list.")
    assert_true(isinstance(ranked_results, list), "ranked_results must be a list.")
    assert_true(len(experiment_results) == 3, "Quick default search should produce exactly 3 experiment results.")
    assert_true(len(ranked_results) == 3, "Ranked results should contain exactly 3 experiment results.")
    assert_true(
        search_result["max_training_records_per_class"] == quick_test_max_training_records_per_class,
        "search_result max_training_records_per_class mismatch."
    )

    print("[OK] Quick experiment search executed successfully.")
    print(f"  experiment result count:             {len(experiment_results)}")
    print(f"  ranked result count:                 {len(ranked_results)}")
    print(f"  max_training_records_per_class:      {search_result['max_training_records_per_class']}")
    print()

    # -------------------------------------------------------------
    # 4. Validate experiment-result structure and ranking
    # -------------------------------------------------------------
    print("[4/5] Validating experiment results and ranking...")
    previous_accuracy = None
    previous_total_frame_mean = None

    for index, experiment_result in enumerate(ranked_results, start=1):
        assert_true(isinstance(experiment_result, dict), f"Experiment result #{index} must be a dictionary.")

        missing_experiment_keys = REQUIRED_EXPERIMENT_RESULT_KEYS - set(experiment_result.keys())
        assert_true(
            not missing_experiment_keys,
            f"Experiment result #{index} is missing keys: {sorted(missing_experiment_keys)}"
        )

        assert_true(experiment_result["training_sample_count"] > 0, f"Experiment #{index} training_sample_count must be positive.")
        assert_true(experiment_result["feature_count"] > 0, f"Experiment #{index} feature_count must be positive.")
        assert_true(experiment_result["frame_count_processed"] > 0, f"Experiment #{index} frame_count_processed must be positive.")
        assert_true(experiment_result["compared_count"] > 0, f"Experiment #{index} compared_count must be positive.")
        assert_true(0.0 <= experiment_result["accuracy_percent"] <= 100.0, f"Experiment #{index} accuracy_percent out of range.")
        assert_true(experiment_result["timing_model_build_total_ms"] > 0.0, f"Experiment #{index} model-build timing must be positive.")
        assert_true(experiment_result["timing_training_feature_preparation_ms"] > 0.0, f"Experiment #{index} feature-preparation timing must be positive.")
        assert_true(experiment_result["timing_training_ms"] > 0.0, f"Experiment #{index} training timing must be positive.")
        assert_true(experiment_result["timing_experiment_total_ms"] > 0.0, f"Experiment #{index} total experiment timing must be positive.")
        assert_true(experiment_result["localization_mean_ms"] >= 0.0, f"Experiment #{index} localization mean must be non-negative.")
        assert_true(experiment_result["classification_mean_ms"] >= 0.0, f"Experiment #{index} classification mean must be non-negative.")
        assert_true(experiment_result["total_frame_mean_ms"] >= 0.0, f"Experiment #{index} total frame mean must be non-negative.")
        assert_true(isinstance(experiment_result["evaluation_summary"], dict), f"Experiment #{index} evaluation_summary must be a dictionary.")
        assert_true(
            experiment_result["training_sample_count"] <= 2 * quick_test_max_training_records_per_class,
            f"Experiment #{index} training_sample_count is larger than expected quick-test subset."
        )

        current_accuracy = float(experiment_result["accuracy_percent"])
        current_total_frame_mean = float(experiment_result["total_frame_mean_ms"])

        if previous_accuracy is not None:
            assert_true(
                current_accuracy <= previous_accuracy + 1e-9 or abs(current_accuracy - previous_accuracy) <= 1e-9,
                "Ranked results are not sorted by non-increasing accuracy."
            )

            if abs(current_accuracy - previous_accuracy) <= 1e-9 and previous_total_frame_mean is not None:
                assert_true(
                    current_total_frame_mean >= previous_total_frame_mean - 1e-9 or abs(current_total_frame_mean - previous_total_frame_mean) <= 1e-9,
                    "Tie-breaking by total_frame_mean_ms does not appear sorted correctly."
                )

        previous_accuracy = current_accuracy
        previous_total_frame_mean = current_total_frame_mean

    print("[OK] Experiment results and ranking are structurally valid.")
    print(f"  best accuracy [%]:      {ranked_results[0]['accuracy_percent']:.2f}")
    print(f"  best total frame [ms]:  {ranked_results[0]['total_frame_mean_ms']:.3f}")
    print()

    # -------------------------------------------------------------
    # 5. Format and save report
    # -------------------------------------------------------------
    print("[5/5] Formatting and saving experiment-search report...")
    summary_text = format_experiment_search_summary(search_result)

    assert_true(isinstance(summary_text, str), "Formatted summary must be a string.")
    assert_true("=== Eye LBP experiment search summary ===" in summary_text, "Formatted summary header missing.")
    assert_true("Accuracy [%]:" in summary_text, "Formatted summary should contain accuracy lines.")

    report_path = output_paths["experiment_search_report_path"]
    save_experiment_search_report(search_result, report_path)

    assert_true(report_path.exists(), f"Experiment search report was not created: {report_path}")

    print("[OK] Experiment-search report saved successfully.")
    print(f"  report_path: {report_path}")
    print()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()