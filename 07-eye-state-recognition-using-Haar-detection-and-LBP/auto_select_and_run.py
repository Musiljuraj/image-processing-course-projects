# This module is the thin top-level orchestrator for the whole final solution.
# It does not implement detection, training, classification, or evaluation by itself.
# Instead, it coordinates the already-built lower layers into one complete automatic run:
#
#     experiment search
#         -> best-result selection
#         -> extraction of the winning runtime configs
#         -> one final full rerun with those configs
#         -> saving summary files that connect the search phase and the final run
#
# In other words, this file is the final "decision and launch" layer of the project.

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

# Path is only needed here for safe path normalization when saving the final
# automatic-selection summary text file.
from pathlib import Path

# These imports are the "search-phase" tools.
# Together they provide everything needed to:
# - run the configuration search,
# - build standard experiment output paths,
# - save the search report,
# - select the best-ranked experiment,
# - convert that best experiment result back into runtime configs,
# - save and format summaries of the winning experiment.
from experiment_search import (
    run_experiment_search,
    get_experiment_output_paths,
    save_experiment_search_report,
    get_best_experiment_result,
    extract_runtime_configs_from_experiment_result,
    save_best_experiment_report,
    format_best_experiment_summary,
)

# This import is the "execution-phase" tool.
# After the best experiment configuration is selected, the actual full final
# rerun is delegated to the main reusable pipeline runner from main.py.
from main import run_final_pipeline


# ---------------------------------------------------------------------
# Automatic final-run configuration
# ---------------------------------------------------------------------

# These constants define how the automatic orchestration layer should behave.
# They intentionally keep the top-level workflow configurable without having to
# change the body of the orchestration functions below.
#
# The general idea is:
# - run the experiment search in a realistic final-submission mode,
# - avoid live preview during automated runs,
# - store the final rerun outputs under a stable, descriptive name.
#
# The experiment search itself is still handled in experiment_search.py.
# This file only decides which values should be passed into that search.

# Final submission mode: search across the full training set and full video.
# max_frames=120 means the search evaluates configurations on the full target
# video length used by the project.
FINAL_SEARCH_MAX_FRAMES = 120

# This limits the balanced training subset used during experiment search.
# The search phase uses this cap so multiple configurations can be compared
# consistently without loading an unnecessarily large training subset.
FINAL_SEARCH_MAX_TRAINING_RECORDS_PER_CLASS = 150

# Preview is disabled during search because this layer is intended to run as an
# automatic batch workflow, not as an interactive inspection session.
FINAL_SEARCH_SHOW_PREVIEW = False

# Preview is also disabled during the final full rerun for the same reason:
# this file is primarily concerned with reproducible final outputs.
FINAL_RUN_SHOW_PREVIEW = False

# This descriptive run name is passed into the reusable final pipeline so the
# generated logs and reports clearly indicate that they came from the
# automatically selected best configuration.
FINAL_RUN_NAME = "auto_selected_best_run"

# This prefix keeps the final rerun outputs separate from ordinary manual runs.
# main.py uses this prefix to generate distinct filenames for the log and report.
FINAL_OUTPUT_FILE_PREFIX = "auto_selected_best"


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def _get_auto_selection_output_paths(paths):
    """
    Build the extra output paths used only by the automatic orchestration layer.
    """

    # experiment_search.py already knows how to build the standard report paths
    # used by the experiment-search stage.
    experiment_output_paths = get_experiment_output_paths(paths)

    # This function extends those standard paths with one extra file that exists
    # only at the top orchestration level:
    #
    # - auto_selection_summary_path
    #
    # That file is meant to connect both phases of the final solution:
    # - which experiment won during search
    # - what happened when that winning configuration was rerun on the final pipeline
    #
    # The returned dictionary keeps the original experiment output paths intact
    # and simply adds the extra automatic-selection summary path.
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

    # The final rerun already contains a structured evaluation summary produced
    # by the evaluation layer. This helper does not recompute anything.
    # It only pulls out the key numbers that are most useful at the top-level
    # orchestration view.
    evaluation_summary = final_run_result["evaluation_summary"]
    accuracy = evaluation_summary["accuracy"]
    timing = evaluation_summary["timing"]

    # The summary is intentionally built as a plain text report so it is easy to
    # inspect outside Python as a final submission artifact.
    #
    # The structure is:
    # 1. heading for the automatic best-configuration selection
    # 2. compact summary of the winning experiment from the search phase
    # 3. paths and basic runtime outputs from the final full rerun
    # 4. final rerun evaluation metrics
    #
    # This makes the file a bridge between:
    # - experiment ranking
    # - final deployment-style execution
    lines = [
        "=== Automatic best-configuration selection summary ===",
        "",
        "--- Best experiment selected from search ---",

        # Reuse the lower-layer formatter so the winning experiment is described
        # in exactly the same terminology as the experiment-search report.
        format_best_experiment_summary(best_experiment_result),

        "",
        "--- Final full rerun outputs ---",

        # These lines document where the final rerun outputs were written and
        # what scale of run was executed.
        f"Run name:                               {final_run_result['run_name']}",
        f"Run log path:                           {final_run_result['output_files']['run_log_path']}",
        f"Evaluation report path:                 {final_run_result['output_files']['evaluation_report_path']}",
        f"Model build time [ms]:                  {final_run_result['model_build_time_ms']:.3f}",
        f"Frame count processed:                  {final_run_result['frame_count']}",
        f"Frame results stored:                   {final_run_result['frame_results_count']}",

        "",
        "--- Final full rerun evaluation ---",

        # These are the key outcome metrics from the final rerun.
        # The goal here is not to dump the whole run_result dictionary, but to
        # surface the most important final comparison and timing numbers in one
        # compact human-readable summary file.
        f"Compared labels:                        {accuracy['compared_count']}",
        f"Correct predictions:                    {accuracy['correct_count']}",
        f"Accuracy [%]:                           {accuracy['accuracy_percent']:.2f}",
        f"Localization mean [ms]:                 {timing['localization']['mean_ms']:.3f}",
        f"Classification mean [ms]:               {timing['classification']['mean_ms']:.3f}",
        f"Total frame mean [ms]:                  {timing['total_frame']['mean_ms']:.3f}",
    ]

    # Convert the target path to a Path object so path handling is consistent
    # even if the caller passed a string-like value.
    summary_path = Path(summary_path)

    # Make sure the parent output directory exists before writing the file.
    # This keeps the helper self-contained and safe even if the caller has not
    # created the directory yet.
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the summary as one plain text file with a trailing newline.
    # Using "\n".join(lines) keeps the report layout explicit and easy to edit.
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

    # Step 1:
    # Run the complete experiment search using the orchestration-level defaults.
    #
    # Passing None for the configuration lists tells experiment_search.py to use
    # its own recommended/default experiment configuration sets.
    #
    # This keeps the orchestration layer simple:
    # it decides "run the full search now",
    # while experiment_search.py decides "which configurations make up that search".
    search_result = run_experiment_search(
        preprocessing_configurations=None,
        lbp_configurations=None,
        classifier_configurations=None,
        max_frames=FINAL_SEARCH_MAX_FRAMES,
        show_preview=FINAL_SEARCH_SHOW_PREVIEW,
        max_training_records_per_class=FINAL_SEARCH_MAX_TRAINING_RECORDS_PER_CLASS,
    )

    # Step 2:
    # Build all output paths relevant to the automatic-selection layer.
    # The search result already includes the shared project paths, so those are
    # reused here to derive the final report destinations.
    output_paths = _get_auto_selection_output_paths(search_result["paths"])

    # Step 3:
    # Save the full experiment-search report.
    # This file documents the whole ranking process, not just the winner.
    save_experiment_search_report(
        search_result,
        output_paths["experiment_search_report_path"],
    )

    # Step 4:
    # Extract the best-ranked experiment result from the search output.
    # This is the core "selection" decision made by the automatic workflow.
    best_experiment_result = get_best_experiment_result(search_result)

    # Save a dedicated summary of the winning experiment so it is easy to inspect
    # without opening the full experiment-search report.
    save_best_experiment_report(
        best_experiment_result,
        output_paths["best_experiment_report_path"],
    )

    # Step 5:
    # Convert the winning experiment result back into the exact runtime configs
    # required by the final reusable pipeline in main.py.
    #
    # This is the handoff point between:
    # - search/ranking logic
    # - actual final execution logic
    selected_configs = extract_runtime_configs_from_experiment_result(
        best_experiment_result
    )

    # Step 6:
    # Execute one full final rerun with the chosen winning configs.
    #
    # This call hands control over to main.py, which performs:
    # - model building
    # - video processing
    # - frame-level classification
    # - evaluation
    # - final output saving
    #
    # The orchestration layer only provides the selected configs and a stable
    # naming/output policy for the automatic final run.
    final_run_result = run_final_pipeline(
        preprocessing_config=selected_configs["preprocessing_config"],
        lbp_config=selected_configs["lbp_config"],
        classifier_config=selected_configs["classifier_config"],
        run_name=FINAL_RUN_NAME,
        show_preview=FINAL_RUN_SHOW_PREVIEW,
        output_file_prefix=FINAL_OUTPUT_FILE_PREFIX,
        print_summary=True,
    )

    # Step 7:
    # Save one top-level summary that explicitly connects the winning search
    # result with the outputs and metrics from the final full rerun.
    #
    # This is the file that best explains the overall automatic workflow in one place.
    _save_auto_selection_summary(
        best_experiment_result=best_experiment_result,
        final_run_result=final_run_result,
        summary_path=output_paths["auto_selection_summary_path"],
    )

    # Step 8:
    # Collect the most important artifacts of the orchestration into one return
    # dictionary so the caller can inspect every stage if needed.
    #
    # The structure intentionally preserves all major layers:
    # - raw search result
    # - chosen winner
    # - extracted configs
    # - final rerun result
    # - output paths
    orchestration_result = {
        "search_result": search_result,
        "best_experiment_result": best_experiment_result,
        "selected_configs": selected_configs,
        "final_run_result": final_run_result,
        "output_paths": output_paths,
    }

    # Return the full structured orchestration output instead of only printing.
    # That makes this function reusable from tests, scripts, or future wrappers.
    return orchestration_result


def main():
    """
    Default entry point for the final automatic solution.
    """

    # Run the complete automatic workflow from search all the way to final rerun.
    orchestration_result = auto_select_best_and_run_final()

    # Pull out the most important top-level pieces so the final console output
    # stays readable and focused on the chosen configuration and generated files.
    best_experiment_result = orchestration_result["best_experiment_result"]
    final_run_result = orchestration_result["final_run_result"]
    output_paths = orchestration_result["output_paths"]

    # Print a compact console summary of:
    # - which experiment was selected
    # - which configs were selected
    # - where the generated reports and logs were saved
    #
    # This is intentionally a human-facing final checkpoint after the fully
    # automatic workflow finishes.
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


# Standard Python script entry point.
# When this file is run directly, the full automatic workflow starts here.
if __name__ == "__main__":
    main()