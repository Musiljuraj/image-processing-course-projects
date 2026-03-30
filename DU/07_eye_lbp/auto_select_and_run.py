"""
auto_select_and_run.py

Final orchestration entry point for the eye-LBP assignment.

Purpose:
- run the full multi-configuration experiment search,
- automatically select the best-ranked configuration,
- execute one full final video pipeline run with that configuration,
- save both the experiment-search outputs and the final-run outputs.

This module turns the project into one complete final pipeline:
    search -> rank -> choose best -> final full run
"""

from pathlib import Path

from experiment_search import (
    run_experiment_search,
    get_experiment_output_paths,
    save_experiment_search_report,
    get_best_experiment_result,
    extract_runtime_configs_from_experiment_result,
    save_best_experiment_report,
    format_best_experiment_summary,
)
from main import run_final_pipeline


# ---------------------------------------------------------------------
# Automatic final-run configuration
# ---------------------------------------------------------------------

# Final submission mode: search across the full training set and full video.
FINAL_SEARCH_MAX_FRAMES = 120
FINAL_SEARCH_MAX_TRAINING_RECORDS_PER_CLASS = 150
FINAL_SEARCH_SHOW_PREVIEW = False

FINAL_RUN_SHOW_PREVIEW = False
FINAL_RUN_NAME = "auto_selected_best_run"
FINAL_OUTPUT_FILE_PREFIX = "auto_selected_best"


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def _get_auto_selection_output_paths(paths):
    """
    Build the extra output paths used only by the automatic orchestration layer.
    """

    experiment_output_paths = get_experiment_output_paths(paths)

    return {
        **experiment_output_paths,
        "auto_selection_summary_path": (
            paths["experiment_results_dir"] / "auto_selected_best_summary.txt"
        ),
    }


def _save_auto_selection_summary(best_experiment_result, final_run_result, summary_path):
    """
    Save one readable summary that connects:
    - the best experiment result,
    - the final full rerun executed with that chosen configuration.
    """

    evaluation_summary = final_run_result["evaluation_summary"]
    accuracy = evaluation_summary["accuracy"]
    timing = evaluation_summary["timing"]

    lines = [
        "=== Automatic best-configuration selection summary ===",
        "",
        "--- Best experiment selected from search ---",
        format_best_experiment_summary(best_experiment_result),
        "",
        "--- Final full rerun outputs ---",
        f"Run name:                               {final_run_result['run_name']}",
        f"Run log path:                           {final_run_result['output_files']['run_log_path']}",
        f"Evaluation report path:                 {final_run_result['output_files']['evaluation_report_path']}",
        f"Model build time [ms]:                  {final_run_result['model_build_time_ms']:.3f}",
        f"Frame count processed:                  {final_run_result['frame_count']}",
        f"Frame results stored:                   {final_run_result['frame_results_count']}",
        "",
        "--- Final full rerun evaluation ---",
        f"Compared labels:                        {accuracy['compared_count']}",
        f"Correct predictions:                    {accuracy['correct_count']}",
        f"Accuracy [%]:                           {accuracy['accuracy_percent']:.2f}",
        f"Localization mean [ms]:                 {timing['localization']['mean_ms']:.3f}",
        f"Classification mean [ms]:               {timing['classification']['mean_ms']:.3f}",
        f"Total frame mean [ms]:                  {timing['total_frame']['mean_ms']:.3f}",
    ]

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# Full automatic orchestration
# ---------------------------------------------------------------------

def auto_select_best_and_run_final():
    """
    Execute the final automatic workflow:
    1. run the full experiment search,
    2. choose the best-ranked result,
    3. extract its runtime configs,
    4. run the full final pipeline with those configs,
    5. save all important summary outputs.

    Return:
    - orchestration_result dictionary containing:
        search_result
        best_experiment_result
        selected_configs
        final_run_result
        output_paths
    """

    search_result = run_experiment_search(
        preprocessing_configurations=None,
        lbp_configurations=None,
        classifier_configurations=None,
        max_frames=FINAL_SEARCH_MAX_FRAMES,
        show_preview=FINAL_SEARCH_SHOW_PREVIEW,
        max_training_records_per_class=FINAL_SEARCH_MAX_TRAINING_RECORDS_PER_CLASS,
    )

    output_paths = _get_auto_selection_output_paths(search_result["paths"])

    save_experiment_search_report(
        search_result,
        output_paths["experiment_search_report_path"],
    )

    best_experiment_result = get_best_experiment_result(search_result)
    save_best_experiment_report(
        best_experiment_result,
        output_paths["best_experiment_report_path"],
    )

    selected_configs = extract_runtime_configs_from_experiment_result(
        best_experiment_result
    )

    final_run_result = run_final_pipeline(
        preprocessing_config=selected_configs["preprocessing_config"],
        lbp_config=selected_configs["lbp_config"],
        classifier_config=selected_configs["classifier_config"],
        run_name=FINAL_RUN_NAME,
        show_preview=FINAL_RUN_SHOW_PREVIEW,
        output_file_prefix=FINAL_OUTPUT_FILE_PREFIX,
        print_summary=True,
    )

    _save_auto_selection_summary(
        best_experiment_result=best_experiment_result,
        final_run_result=final_run_result,
        summary_path=output_paths["auto_selection_summary_path"],
    )

    orchestration_result = {
        "search_result": search_result,
        "best_experiment_result": best_experiment_result,
        "selected_configs": selected_configs,
        "final_run_result": final_run_result,
        "output_paths": output_paths,
    }

    return orchestration_result


def main():
    """
    Default entry point for the final automatic solution.
    """

    orchestration_result = auto_select_best_and_run_final()

    best_experiment_result = orchestration_result["best_experiment_result"]
    final_run_result = orchestration_result["final_run_result"]
    output_paths = orchestration_result["output_paths"]

    print()
    print("=== AUTOMATIC BEST-CONFIGURATION FINAL RUN ===")
    print(f"selected experiment index:     {best_experiment_result['experiment_index']}")
    print(f"selected accuracy [%]:         {best_experiment_result['accuracy_percent']:.2f}")
    print(f"selected preprocessing config: {best_experiment_result['preprocessing_config']}")
    print(f"selected lbp config:           {best_experiment_result['lbp_config']}")
    print(f"selected classifier config:    {best_experiment_result['classifier_config']}")
    print()
    print(f"experiment report:             {output_paths['experiment_search_report_path']}")
    print(f"best experiment report:        {output_paths['best_experiment_report_path']}")
    print(f"auto selection summary:        {output_paths['auto_selection_summary_path']}")
    print(f"final run log:                 {final_run_result['output_files']['run_log_path']}")
    print(f"final evaluation report:       {final_run_result['output_files']['evaluation_report_path']}")


if __name__ == "__main__":
    main()