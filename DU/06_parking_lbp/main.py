# ---------------------------------------------------------------------------
# Module orientation:
# This is the top-level runner for the parking LBP experiment. Its job is to
# define the experiment grid, point the pipeline to the correct dataset paths,
# launch the full experiment search, save ranked outputs, and print a readable
# summary of the best result. All detailed work is delegated to other modules;
# this file mainly defines what should be tried and where the outputs should go.
# ---------------------------------------------------------------------------
from pathlib import Path

from experiment_search import run_experiment_search
from results_io import save_experiment_outputs


def print_best_result(best_result):
    """
    Print a readable summary of the best-ranked experiment result.
    """

    # Read configuration values and normalize them up front so the rest of the function
    # can rely on one stable internal convention.
    confusion_counts = best_result.get("confusion_counts", {})

    print("\n=== BEST RESULT ===")
    print(f"experiment_index            : {best_result.get('experiment_index')}")
    print(f"accuracy                    : {best_result.get('accuracy'):.6f}")
    print(f"num_samples                 : {best_result.get('num_samples')}")
    print(f"processing_time_total       : {best_result.get('processing_time_total'):.6f}s")
    print(
        f"processing_time_per_image   : "
        f"{best_result.get('processing_time_per_image'):.6f}s"
    )
    print(
        f"processing_time_per_roi     : "
        f"{best_result.get('processing_time_per_roi'):.6f}s"
    )

    print("\nconfusion_counts:")
    print(f"  tp: {confusion_counts.get('tp')}")
    print(f"  tn: {confusion_counts.get('tn')}")
    print(f"  fp: {confusion_counts.get('fp')}")
    print(f"  fn: {confusion_counts.get('fn')}")

    print("\npreprocessing_config:")
    print(f"  {best_result.get('preprocessing_config')}")

    print("\nlbp_config:")
    print(f"  {best_result.get('lbp_config')}")

    print("\nclassifier_config:")
    print(f"  {best_result.get('classifier_config')}")


def main():
    """
    Final top-level entry point for the parking LBP experiment pipeline.

    What this script does:
    1. define input/output paths
    2. define experiment configuration grids
    3. run the full experiment search
    4. save CSV + text summary outputs
    5. print the best-ranked result
    """

    # Convert incoming path-like inputs to Path objects at the start so all later
    # filesystem work uses one consistent path representation.
    project_root = Path(__file__).resolve().parent

    # -------------------------------------------------------------------------
    # input paths
    # -------------------------------------------------------------------------
    training_root = project_root / "data" / "training"
    map_path = project_root / "data" / "parking_map_python.txt"
    test_images_dir = project_root / "data" / "test_images_zao"

    # -------------------------------------------------------------------------
    # output paths
    # -------------------------------------------------------------------------
    output_dir = project_root / "outputs" / "results" / "final_run"

    # -------------------------------------------------------------------------
    # experiment search space
    # -------------------------------------------------------------------------
    preprocessing_configurations = [
        {
            "target_size": (80, 80),
            "contrast_method": "none",
            "filter_name": "none",
            "kernel_size": 3,
        },
        {
            "target_size": (80, 80),
            "contrast_method": "clahe",
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": (8, 8),
            "filter_name": "gaussian",
            "kernel_size": 3,
        },
    ]

    # at least 3 LBP configurations for assignment-style comparison
    lbp_configurations = [
        {
            "neighbors": 8,
            "radius": 1,
            "method": "uniform",
            "grid_shape": (2, 2),
            "normalize_histogram": True,
        },
        {
            "neighbors": 8,
            "radius": 1,
            "method": "uniform",
            "grid_shape": (4, 4),
            "normalize_histogram": True,
        },
        {
            "neighbors": 16,
            "radius": 2,
            "method": "uniform",
            "grid_shape": (4, 4),
            "normalize_histogram": True,
        },
    ]

    classifier_configurations = [
        {
            "classifier_name": "knn",
            "n_neighbors": 3,
        },
        {
            "classifier_name": "linear_svm",
            "C": 1.0,
        },
    ]

    evaluation_config = {
        "occupied_label": 1,
        "empty_label": 0,
    }

    print("=== PARKING LBP EXPERIMENT RUN ===")
    print(f"training_root    : {training_root}")
    print(f"map_path         : {map_path}")
    print(f"test_images_dir  : {test_images_dir}")
    print(f"output_dir       : {output_dir}")

    total_experiment_count = (
        len(preprocessing_configurations)
        * len(lbp_configurations)
        * len(classifier_configurations)
    )

    print("\nExperiment grid:")
    print(f"  preprocessing configs : {len(preprocessing_configurations)}")
    print(f"  lbp configs           : {len(lbp_configurations)}")
    print(f"  classifier configs    : {len(classifier_configurations)}")
    print(f"  total experiments     : {total_experiment_count}")

    # -------------------------------------------------------------------------
    # run experiment search
    # -------------------------------------------------------------------------
    print("\nRunning experiment search...")
    search_result = run_experiment_search(
        training_root=training_root,
        map_path=map_path,
        test_images_dir=test_images_dir,
        preprocessing_configurations=preprocessing_configurations,
        lbp_configurations=lbp_configurations,
        classifier_configurations=classifier_configurations,
        evaluation_config=evaluation_config,
        max_test_cases=None,   # use full test set
    )

    experiment_results = search_result["experiment_results"]
    ranked_results = search_result["ranked_results"]

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not ranked_results:
        raise ValueError("No ranked results were produced by experiment search.")

    print(f"Experiment search finished. Result count: {len(experiment_results)}")

    # -------------------------------------------------------------------------
    # save outputs
    # -------------------------------------------------------------------------
    print("\nSaving outputs...")
    saved_outputs = save_experiment_outputs(
        experiment_results=ranked_results,
        output_dir=output_dir,
        csv_filename="parking_lbp_results.csv",
        summary_filename="parking_lbp_summary.txt",
        top_k=10,
    )

    print(f"CSV saved to     : {saved_outputs['csv_path']}")
    print(f"Summary saved to : {saved_outputs['summary_path']}")

    # -------------------------------------------------------------------------
    # print best result
    # -------------------------------------------------------------------------
    best_result = ranked_results[0]
    print_best_result(best_result)

    print("\nMain run finished successfully.")


if __name__ == "__main__":
    main()