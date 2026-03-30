"""
experiment_search.py

Purpose of this module:
- generate multi-configuration experiment combinations for the eye LBP project,
- run end-to-end experiments on the real video for each configuration,
- evaluate each experiment against eye-state ground truth,
- rank the final results,
- save one readable experiment-search report,
- expose explicit helpers for retrieving the best-ranked configuration.

Why this module exists:
At this stage of the eye project, the main building blocks already exist:
- eye training images can be loaded,
- eye images can be preprocessed,
- LBP features can be computed,
- classifiers can be trained and applied,
- the video pipeline can be executed frame by frame,
- predictions can be evaluated against the reference eye-state labels.

The next logical stage is to connect these pieces into a systematic
multi-configuration experiment runner and then reuse the best-ranked
configuration for one final full run.
"""

from copy import deepcopy
from itertools import product
from pathlib import Path
from time import perf_counter
import gc
import cv2
import numpy as np

from detectors import (
    load_cascades,
    detect_faces,
    select_main_face,
    detect_face_parts,
)

from eye_training_io import (
    collect_eye_image_paths,
    parse_eye_filename,
    build_one_eye_training_record,
    load_all_eye_training_records,
)
from eye_preprocessing import validate_preprocessing_config
from lbp_features import validate_lbp_config
from eye_lbp_dataset import (
    prepare_training_feature_records,
    build_training_matrix_and_labels,
)
from eye_lbp_classifier import (
    validate_classifier_config,
    train_classifier,
)
from eye_state_lbp import classify_eye_state_lbp
from evaluation import (
    evaluate_results,
    format_evaluation_summary,
)


# ---------------------------------------------------------------------
# Runtime configuration reused from the main eye-LBP pipeline
# ---------------------------------------------------------------------

ENABLE_MOUTH_DETECTION = True
FACE_DETECTION_DOWNSCALE_FACTOR = 0.75

LBP_FALLBACK_TO_HEURISTIC = True
LBP_FALLBACK_FRAME_LABEL = "close"

DEFAULT_SHOW_PREVIEW = False
PREVIEW_WINDOW_NAME = "ZAO Assignment 06 - Experiment Preview"

DEFAULT_QUICK_TEST_MAX_FRAMES = 120
DEFAULT_QUICK_TEST_MAX_TRAINING_RECORDS_PER_CLASS = 150


# ---------------------------------------------------------------------
# Project-path helpers
# ---------------------------------------------------------------------

def get_project_paths():
    """
    Build and return all relevant project paths.

    This mirrors the path logic of the main program so that the experiment
    runner can be executed as a standalone module from the project root.
    """

    project_root = Path(__file__).resolve().parent

    input_dir = project_root / "input"
    output_dir = project_root / "output"
    reference_dir = project_root / "reference"

    input_archives_dir = input_dir / "archives"
    input_cascades_dir = input_dir / "cascades"
    input_ground_truth_dir = input_dir / "ground_truth"
    input_video_dir = input_dir / "video"
    input_training_dir = input_dir / "training"

    face_cascades_dir = input_cascades_dir / "face"
    eye_cascades_dir = input_cascades_dir / "eye"
    mouth_cascades_dir = input_cascades_dir / "mouth"

    output_annotated_video_dir = output_dir / "annotated_video"
    output_frames_dir = output_dir / "frames"
    output_logs_dir = output_dir / "logs"
    output_reports_dir = output_dir / "reports"
    experiment_output_dir = output_dir / "experiments"
    experiment_results_dir = experiment_output_dir / "results"
    experiment_inspection_dir = experiment_output_dir / "inspection"

    mrl_eyes_dataset_dir = input_training_dir / "mrlEyes_2018_01"
    reference_parking_lbp_dir = reference_dir / "06_ParkingOccupancy_LBP"

    return {
        "project_root": project_root,

        "input_dir": input_dir,
        "output_dir": output_dir,
        "reference_dir": reference_dir,

        "input_archives_dir": input_archives_dir,
        "input_cascades_dir": input_cascades_dir,
        "input_ground_truth_dir": input_ground_truth_dir,
        "input_video_dir": input_video_dir,
        "input_training_dir": input_training_dir,

        "face_cascades_dir": face_cascades_dir,
        "eye_cascades_dir": eye_cascades_dir,
        "mouth_cascades_dir": mouth_cascades_dir,

        "video_path": input_video_dir / "fusek_face_car_01.avi",

        "face_cascade_frontal_path": face_cascades_dir / "haarcascade_frontalface_default.xml",
        "face_cascade_profile_path": face_cascades_dir / "haarcascade_profileface.xml",

        "eye_cascade_path": eye_cascades_dir / "eye_cascade_fusek.xml",
        "mouth_cascade_path": mouth_cascades_dir / "haarcascade_smile.xml",

        "ground_truth_path": input_ground_truth_dir / "eye-state.txt",

        "output_annotated_video_dir": output_annotated_video_dir,
        "output_frames_dir": output_frames_dir,
        "output_logs_dir": output_logs_dir,
        "output_reports_dir": output_reports_dir,
        "experiment_output_dir": experiment_output_dir,
        "experiment_results_dir": experiment_results_dir,
        "experiment_inspection_dir": experiment_inspection_dir,

        "mrl_eyes_dataset_dir": mrl_eyes_dataset_dir,
        "reference_parking_lbp_dir": reference_parking_lbp_dir,
    }


def ensure_output_directories(paths):
    """
    Ensure that all output directories required by the experiment runner exist.
    """

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_annotated_video_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_frames_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["experiment_output_dir"].mkdir(parents=True, exist_ok=True)
    paths["experiment_results_dir"].mkdir(parents=True, exist_ok=True)
    paths["experiment_inspection_dir"].mkdir(parents=True, exist_ok=True)


def get_experiment_output_paths(paths):
    """
    Build the standard output paths used by the experiment runner.
    """

    return {
        "experiment_search_report_path": (
            paths["experiment_results_dir"] / "eye_experiment_search_report.txt"
        ),
        "best_experiment_report_path": (
            paths["experiment_results_dir"] / "best_experiment_summary.txt"
        ),
    }


def open_video(video_path):
    """
    Open the input video and return an OpenCV VideoCapture object.
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    return capture


def convert_to_grayscale(frame):
    """
    Convert a BGR video frame to grayscale.
    """

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def store_frame_result(
    results,
    frame_index,
    face_boxes,
    main_face,
    face_parts,
    eye_state,
    localization_time_ms,
    classification_time_ms,
    total_frame_time_ms,
):
    """
    Store one structured result record for the current frame.

    This mirrors the structure used by main.py so that evaluation.py can be
    reused without modification.
    """

    frame_result = {
        "frame_index": frame_index,
        "face_boxes": face_boxes,
        "main_face": main_face,
        "face_parts": face_parts,
        "eye_state": eye_state,
        "localization_time_ms": localization_time_ms,
        "classification_time_ms": classification_time_ms,
        "total_frame_time_ms": total_frame_time_ms,
    }

    results.append(frame_result)


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

def ensure_config_list(configs, config_name):
    """
    Ensure the given config collection is a non-empty list of dictionaries.
    """

    if not isinstance(configs, list):
        raise TypeError(f"{config_name} must be a list.")

    if not configs:
        raise ValueError(f"{config_name} must not be empty.")

    for index, config in enumerate(configs, start=1):
        if not isinstance(config, dict):
            raise TypeError(
                f"Each item in {config_name} must be a dictionary. "
                f"Item #{index} has type: {type(config).__name__}"
            )

    return configs


def get_recommended_experiment_configuration_lists():
    """
    Return one simple recommended experiment set.

    Design goal:
    keep preprocessing and classifier fixed,
    vary at least three LBP configurations as required by the assignment.

    The returned configuration lists intentionally produce exactly 3
    experiments:
    - 1 preprocessing configuration
    - 3 LBP configurations
    - 1 classifier configuration
    """

    preprocessing_configurations = [
        {
            "grayscale": True,
            "target_size": (80, 40),
            "crop_analysis_band": True,
            "analysis_top_ratio": 0.15,
            "analysis_bottom_ratio": 0.85,
            "contrast_method": "equalize",
            "filter_name": "gaussian",
            "gaussian_kernel_size": (5, 5),
        }
    ]

    lbp_configurations = [
        {
            "neighbors": 8,
            "radius": 1.0,
            "method": "uniform",
            "grid_shape": (1, 1),
            "normalize_histogram": True,
        },
        {
            "neighbors": 8,
            "radius": 1.0,
            "method": "uniform",
            "grid_shape": (2, 1),
            "normalize_histogram": True,
        },
        {
            "neighbors": 8,
            "radius": 2.0,
            "method": "uniform",
            "grid_shape": (2, 2),
            "normalize_histogram": True,
        },
    ]

    classifier_configurations = [
        {
            "classifier_name": "knn",
            "n_neighbors": 3,
        }
    ]

    return {
        "preprocessing_configurations": preprocessing_configurations,
        "lbp_configurations": lbp_configurations,
        "classifier_configurations": classifier_configurations,
    }


def build_experiment_configurations(
    preprocessing_configurations,
    lbp_configurations,
    classifier_configurations,
):
    """
    Build the full Cartesian product of experiment configurations.

    Each returned configuration dictionary contains:
    - experiment_index
    - preprocessing_config
    - lbp_config
    - classifier_config
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

    experiment_configurations = []

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
            "preprocessing_config": deepcopy(preprocessing_config),
            "lbp_config": deepcopy(lbp_config),
            "classifier_config": deepcopy(classifier_config),
        }
        experiment_configurations.append(experiment_configuration)

    return experiment_configurations


# ---------------------------------------------------------------------
# Training-record loading helpers
# ---------------------------------------------------------------------

def _validate_max_training_records_per_class(max_training_records_per_class):
    """
    Validate the optional balanced-training-subset limit.
    """

    if max_training_records_per_class is None:
        return None

    if not isinstance(max_training_records_per_class, int):
        raise TypeError("max_training_records_per_class must be an integer or None.")

    if max_training_records_per_class <= 0:
        raise ValueError("max_training_records_per_class must be positive if provided.")

    return max_training_records_per_class


def load_training_records_for_experiments(
    dataset_root,
    max_training_records_per_class=None,
    grayscale=True,
    recursive=True,
    ignore_invalid_files=False,
):
    """
    Load training records for experiment search.

    Behavior:
    - if max_training_records_per_class is None:
        load the full dataset with images
    - otherwise:
        load a balanced subset directly from disk without first loading the
        entire dataset into memory

    The balanced subset path is important for quick experiments and smoke
    tests because it keeps memory use much lower.
    """

    max_training_records_per_class = _validate_max_training_records_per_class(
        max_training_records_per_class
    )

    if max_training_records_per_class is None:
        return load_all_eye_training_records(
            dataset_root=dataset_root,
            load_images=True,
            grayscale=grayscale,
            recursive=recursive,
            ignore_invalid_files=ignore_invalid_files,
        )

    dataset_root = Path(dataset_root)
    image_paths = collect_eye_image_paths(dataset_root, recursive=recursive)

    selected_records = []
    class_counts = {0: 0, 1: 0}

    for image_path in image_paths:
        try:
            parsed = parse_eye_filename(image_path.name)
        except Exception:
            if ignore_invalid_files:
                continue
            raise

        label = parsed["label"]

        if label not in class_counts:
            continue

        if class_counts[label] >= max_training_records_per_class:
            continue

        record = build_one_eye_training_record(
            image_path=image_path,
            dataset_root=dataset_root,
            load_image=True,
            grayscale=grayscale,
        )

        selected_records.append(record)
        class_counts[label] += 1

        if all(
            class_counts[class_label] >= max_training_records_per_class
            for class_label in class_counts
        ):
            break

    if not selected_records:
        raise ValueError("No training records were loaded for experiment search.")

    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError(
            "Balanced experiment subset must contain both classes. "
            f"Loaded counts: {class_counts}"
        )

    return selected_records


# ---------------------------------------------------------------------
# Model-building helper for one experiment
# ---------------------------------------------------------------------

def build_eye_lbp_model_from_training_records(
    training_records,
    preprocessing_config,
    lbp_config,
    classifier_config,
    image_key="image",
):
    """
    Build one trained model bundle from already-loaded training records.

    This is the experiment-runner equivalent of build_eye_lbp_model(...),
    but it avoids reloading the whole dataset for every experiment.

    Important difference from the main startup bundle:
    this function intentionally returns a lean model bundle and does not keep
    X_train, y_train, or per-sample feature records alive after training.
    That helps keep experiment memory usage much lower.
    """

    if not isinstance(training_records, list):
        raise TypeError("training_records must be a list.")

    normalized_preprocessing_config = validate_preprocessing_config(preprocessing_config)
    normalized_lbp_config = validate_lbp_config(lbp_config)
    normalized_classifier_config = validate_classifier_config(classifier_config)

    model_build_start = perf_counter()

    training_feature_prep_start = perf_counter()

    training_feature_records = prepare_training_feature_records(
        training_records=training_records,
        preprocessing_config=normalized_preprocessing_config,
        lbp_config=normalized_lbp_config,
        image_key=image_key,
    )

    X_train, y_train, training_metadata = build_training_matrix_and_labels(
        training_feature_records=training_feature_records
    )

    training_feature_prep_end = perf_counter()

    training_start = perf_counter()

    trained_model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=normalized_classifier_config,
    )

    training_end = perf_counter()
    model_build_end = perf_counter()

    training_sample_count = int(X_train.shape[0])
    feature_count = int(X_train.shape[1])

    class_counts = {
        "close": int(np.sum(y_train == 0)),
        "open": int(np.sum(y_train == 1)),
    }

    model_bundle = {
        "model": trained_model,
        "preprocessing_config": deepcopy(normalized_preprocessing_config),
        "lbp_config": deepcopy(normalized_lbp_config),
        "classifier_config": deepcopy(normalized_classifier_config),
        "training_sample_count": training_sample_count,
        "feature_count": feature_count,
        "class_counts": class_counts,
        "training_metadata_count": len(training_metadata),
        "timing_model_build_total_ms": (model_build_end - model_build_start) * 1000.0,
        "timing_training_feature_preparation_ms": (
            training_feature_prep_end - training_feature_prep_start
        ) * 1000.0,
        "timing_training_ms": (training_end - training_start) * 1000.0,
    }

    del training_feature_records
    del X_train
    del y_train
    del training_metadata
    gc.collect()

    return model_bundle


# ---------------------------------------------------------------------
# One-experiment runner
# ---------------------------------------------------------------------

def run_one_experiment(
    training_records,
    paths,
    cascades,
    preprocessing_config,
    lbp_config,
    classifier_config,
    max_frames=None,
    show_preview=False,
):
    """
    Run one full eye-LBP experiment on the real video.

    High-level flow:
    1. build one trained model from shared training records
    2. open the real video
    3. process frames exactly like the main LBP pipeline
    4. evaluate predictions against eye-state ground truth
    5. return one experiment-result dictionary
    """

    if not isinstance(training_records, list):
        raise TypeError("training_records must be a list.")

    if not isinstance(paths, dict):
        raise TypeError("paths must be a dictionary.")

    if not isinstance(cascades, dict):
        raise TypeError("cascades must be a dictionary.")

    if max_frames is not None:
        if not isinstance(max_frames, int):
            raise TypeError("max_frames must be an integer or None.")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive if provided.")

    experiment_start = perf_counter()

    model_bundle = build_eye_lbp_model_from_training_records(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        classifier_config=classifier_config,
        image_key="image",
    )

    capture = open_video(paths["video_path"])

    frame_count = 0
    frame_results = []
    previous_face = None
    previous_eye_state = None
    stopped_early = False

    if show_preview:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            ret, frame = capture.read()

            if not ret:
                break

            frame_count += 1
            frame_processing_start = perf_counter()

            gray_frame = convert_to_grayscale(frame)

            localization_start = perf_counter()

            face_boxes = detect_faces(
                gray_frame,
                cascades,
                downscale_factor=FACE_DETECTION_DOWNSCALE_FACTOR
            )

            main_face = select_main_face(face_boxes, previous_face)

            face_parts = detect_face_parts(
                gray_frame,
                main_face,
                cascades,
                enable_mouth_detection=ENABLE_MOUTH_DETECTION
            )

            localization_end = perf_counter()
            localization_time_ms = (localization_end - localization_start) * 1000.0

            classification_start = perf_counter()

            eye_state = classify_eye_state_lbp(
                gray_frame=gray_frame,
                face_parts=face_parts,
                model_bundle=model_bundle,
                previous_eye_state=previous_eye_state,
                fallback_to_heuristic=LBP_FALLBACK_TO_HEURISTIC,
                fallback_frame_label=LBP_FALLBACK_FRAME_LABEL,
                frame_index=frame_count,
                return_details=False,
            )

            classification_end = perf_counter()
            classification_time_ms = (
                classification_end - classification_start
            ) * 1000.0

            frame_processing_end = perf_counter()
            total_frame_time_ms = (
                frame_processing_end - frame_processing_start
            ) * 1000.0

            store_frame_result(
                frame_results,
                frame_count,
                face_boxes,
                main_face,
                face_parts,
                eye_state,
                localization_time_ms,
                classification_time_ms,
                total_frame_time_ms,
            )

            previous_face = main_face
            previous_eye_state = eye_state

            if show_preview:
                preview_frame = frame.copy()

                for (x, y, w, h) in face_boxes:
                    cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if main_face is not None:
                    x, y, w, h = main_face
                    cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

                for (x, y, w, h) in face_parts["eyes"]:
                    cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if ENABLE_MOUTH_DETECTION:
                    for (x, y, w, h) in face_parts["mouth"]:
                        cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                cv2.putText(
                    preview_frame,
                    f"eye state: {eye_state}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    stopped_early = True
                    break

    finally:
        capture.release()
        if show_preview:
            cv2.destroyAllWindows()

    experiment_end = perf_counter()

    evaluation_summary = evaluate_results(
        frame_results=frame_results,
        ground_truth_path=paths["ground_truth_path"],
    )

    experiment_result = {
        "preprocessing_config": deepcopy(model_bundle["preprocessing_config"]),
        "lbp_config": deepcopy(model_bundle["lbp_config"]),
        "classifier_config": deepcopy(model_bundle["classifier_config"]),

        "training_sample_count": model_bundle["training_sample_count"],
        "feature_count": model_bundle["feature_count"],
        "class_counts": deepcopy(model_bundle["class_counts"]),

        "timing_model_build_total_ms": model_bundle["timing_model_build_total_ms"],
        "timing_training_feature_preparation_ms": (
            model_bundle["timing_training_feature_preparation_ms"]
        ),
        "timing_training_ms": model_bundle["timing_training_ms"],
        "timing_experiment_total_ms": (experiment_end - experiment_start) * 1000.0,

        "frame_count_processed": len(frame_results),
        "stopped_early": stopped_early,
        "max_frames": max_frames,

        "evaluation_summary": evaluation_summary,
        "accuracy_percent": evaluation_summary["accuracy"]["accuracy_percent"],
        "compared_count": evaluation_summary["accuracy"]["compared_count"],
        "correct_count": evaluation_summary["accuracy"]["correct_count"],

        "localization_mean_ms": (
            evaluation_summary["timing"]["localization"]["mean_ms"]
        ),
        "classification_mean_ms": (
            evaluation_summary["timing"]["classification"]["mean_ms"]
        ),
        "total_frame_mean_ms": (
            evaluation_summary["timing"]["total_frame"]["mean_ms"]
        ),
    }

    del frame_results
    del model_bundle
    gc.collect()

    return experiment_result


# ---------------------------------------------------------------------
# Ranking and formatting
# ---------------------------------------------------------------------

def rank_experiment_results(experiment_results):
    """
    Rank experiment results.

    Ranking policy:
    1. higher accuracy is better
    2. lower total frame mean time is better
    3. lower classification mean time is better
    4. lower model-build time is better
    """

    if not isinstance(experiment_results, list):
        raise TypeError("experiment_results must be a list.")

    ranked_results = sorted(
        experiment_results,
        key=lambda result: (
            -float(result["accuracy_percent"]),
            float(result["total_frame_mean_ms"]),
            float(result["classification_mean_ms"]),
            float(result["timing_model_build_total_ms"]),
        )
    )

    return ranked_results


def get_best_experiment_result(search_result):
    """
    Return the best-ranked experiment result from a completed search.

    This is a small but important final-solution helper. It makes the
    automatic-selection step explicit instead of requiring callers to reach
    into ranked_results[0] directly.
    """

    if not isinstance(search_result, dict):
        raise TypeError("search_result must be a dictionary.")

    ranked_results = search_result.get("ranked_results")

    if not isinstance(ranked_results, list) or not ranked_results:
        raise ValueError("search_result does not contain any ranked_results.")

    return ranked_results[0]


def extract_runtime_configs_from_experiment_result(experiment_result):
    """
    Extract the three runtime configuration dictionaries from one experiment
    result.

    Return:
    - {
          "preprocessing_config": ...,
          "lbp_config": ...,
          "classifier_config": ...,
      }
    """

    if not isinstance(experiment_result, dict):
        raise TypeError("experiment_result must be a dictionary.")

    required_keys = {
        "preprocessing_config",
        "lbp_config",
        "classifier_config",
    }

    missing_keys = required_keys - set(experiment_result.keys())
    if missing_keys:
        raise KeyError(
            f"experiment_result is missing keys: {sorted(missing_keys)}"
        )

    return {
        "preprocessing_config": deepcopy(experiment_result["preprocessing_config"]),
        "lbp_config": deepcopy(experiment_result["lbp_config"]),
        "classifier_config": deepcopy(experiment_result["classifier_config"]),
    }


def format_one_experiment_result(experiment_result, rank_index=None):
    """
    Format one experiment result as readable multiline text.
    """

    header = (
        f"=== Experiment #{experiment_result['experiment_index']} ==="
        if rank_index is None
        else f"=== Rank {rank_index} / Experiment #{experiment_result['experiment_index']} ==="
    )

    lines = [
        header,
        f"Accuracy [%]:                         {experiment_result['accuracy_percent']:.2f}",
        f"Compared frames:                      {experiment_result['compared_count']}",
        f"Correct predictions:                  {experiment_result['correct_count']}",
        f"Processed frames:                     {experiment_result['frame_count_processed']}",
        f"Stopped early:                        {experiment_result['stopped_early']}",
        "",
        "Timing [ms]:",
        f"  model build total:                  {experiment_result['timing_model_build_total_ms']:.3f}",
        f"  training feature preparation:       {experiment_result['timing_training_feature_preparation_ms']:.3f}",
        f"  training only:                      {experiment_result['timing_training_ms']:.3f}",
        f"  whole experiment total:             {experiment_result['timing_experiment_total_ms']:.3f}",
        f"  localization mean per frame:        {experiment_result['localization_mean_ms']:.3f}",
        f"  classification mean per frame:      {experiment_result['classification_mean_ms']:.3f}",
        f"  total frame mean per frame:         {experiment_result['total_frame_mean_ms']:.3f}",
        "",
        f"Training sample count:                {experiment_result['training_sample_count']}",
        f"Feature count:                        {experiment_result['feature_count']}",
        f"Class counts:                         {experiment_result['class_counts']}",
        "",
        f"Preprocessing config:                 {experiment_result['preprocessing_config']}",
        f"LBP config:                           {experiment_result['lbp_config']}",
        f"Classifier config:                    {experiment_result['classifier_config']}",
        "",
        format_evaluation_summary(experiment_result["evaluation_summary"]),
    ]

    return "\n".join(lines)


def format_experiment_search_summary(search_result):
    """
    Format the full experiment-search output as readable multiline text.
    """

    ranked_results = search_result["ranked_results"]

    lines = [
        "=== Eye LBP experiment search summary ===",
        f"Experiment count: {len(search_result['experiment_results'])}",
        f"Training subset limit per class: {search_result['max_training_records_per_class']}",
        "",
    ]

    for rank_index, experiment_result in enumerate(ranked_results, start=1):
        lines.append(format_one_experiment_result(experiment_result, rank_index=rank_index))
        lines.append("")

    return "\n".join(lines).rstrip()


def format_best_experiment_summary(experiment_result):
    """
    Format one compact summary of the best-ranked experiment.
    """

    lines = [
        "=== Best experiment summary ===",
        f"Experiment index:                     {experiment_result['experiment_index']}",
        f"Accuracy [%]:                         {experiment_result['accuracy_percent']:.2f}",
        f"Compared frames:                      {experiment_result['compared_count']}",
        f"Correct predictions:                  {experiment_result['correct_count']}",
        f"Total frame mean [ms]:                {experiment_result['total_frame_mean_ms']:.3f}",
        f"Classification mean [ms]:             {experiment_result['classification_mean_ms']:.3f}",
        "",
        f"Preprocessing config:                 {experiment_result['preprocessing_config']}",
        f"LBP config:                           {experiment_result['lbp_config']}",
        f"Classifier config:                    {experiment_result['classifier_config']}",
    ]

    return "\n".join(lines)


def save_experiment_search_report(search_result, report_path):
    """
    Save the formatted experiment-search summary to a text file.
    """

    report_text = format_experiment_search_summary(search_result)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text + "\n")


def save_best_experiment_report(experiment_result, report_path):
    """
    Save one compact best-experiment summary to a text file.
    """

    report_text = format_best_experiment_summary(experiment_result)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text + "\n")


# ---------------------------------------------------------------------
# Full experiment-search orchestration
# ---------------------------------------------------------------------

def run_experiment_search(
    preprocessing_configurations=None,
    lbp_configurations=None,
    classifier_configurations=None,
    max_frames=None,
    show_preview=DEFAULT_SHOW_PREVIEW,
    max_training_records_per_class=None,
):
    """
    Run the full multi-configuration experiment search for the eye LBP project.

    The training dataset and cascade files are loaded once and then reused for
    all experiment configurations.

    Inputs:
    - preprocessing_configurations ... list of preprocessing config dicts
    - lbp_configurations ............ list of LBP config dicts
    - classifier_configurations ..... list of classifier config dicts
    - max_frames .................... optional frame limit for quick testing
    - show_preview .................. whether to show live video preview
    - max_training_records_per_class  optional balanced training-subset limit

    Return:
    - search_result ................. dictionary containing:
                                      - paths
                                      - experiment_configurations
                                      - experiment_results
                                      - ranked_results
                                      - max_training_records_per_class
    """

    paths = get_project_paths()
    ensure_output_directories(paths)

    max_training_records_per_class = _validate_max_training_records_per_class(
        max_training_records_per_class
    )

    if (
        preprocessing_configurations is None
        or lbp_configurations is None
        or classifier_configurations is None
    ):
        recommended = get_recommended_experiment_configuration_lists()

        if preprocessing_configurations is None:
            preprocessing_configurations = recommended["preprocessing_configurations"]

        if lbp_configurations is None:
            lbp_configurations = recommended["lbp_configurations"]

        if classifier_configurations is None:
            classifier_configurations = recommended["classifier_configurations"]

    experiment_configurations = build_experiment_configurations(
        preprocessing_configurations=preprocessing_configurations,
        lbp_configurations=lbp_configurations,
        classifier_configurations=classifier_configurations,
    )

    training_records = load_training_records_for_experiments(
        dataset_root=paths["mrl_eyes_dataset_dir"],
        max_training_records_per_class=max_training_records_per_class,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False,
    )

    cascades = load_cascades(paths)

    experiment_results = []

    for experiment_configuration in experiment_configurations:
        experiment_result = run_one_experiment(
            training_records=training_records,
            paths=paths,
            cascades=cascades,
            preprocessing_config=experiment_configuration["preprocessing_config"],
            lbp_config=experiment_configuration["lbp_config"],
            classifier_config=experiment_configuration["classifier_config"],
            max_frames=max_frames,
            show_preview=show_preview,
        )

        experiment_result = {
            "experiment_index": experiment_configuration["experiment_index"],
            **experiment_result,
        }

        experiment_results.append(experiment_result)
        gc.collect()

    ranked_results = rank_experiment_results(experiment_results)

    search_result = {
        "paths": paths,
        "experiment_configurations": experiment_configurations,
        "experiment_results": experiment_results,
        "ranked_results": ranked_results,
        "max_training_records_per_class": max_training_records_per_class,
    }

    return search_result


# ---------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------

def main():
    """
    Run a default quick experiment search.

    This standalone mode is intentionally configured as a quick test:
    - recommended 3 configurations
    - no preview
    - limited number of video frames
    - limited balanced training subset

    For a full search, call run_experiment_search(...) manually with:
        max_frames=None
        max_training_records_per_class=None
    """

    search_result = run_experiment_search(
        preprocessing_configurations=None,
        lbp_configurations=None,
        classifier_configurations=None,
        max_frames=DEFAULT_QUICK_TEST_MAX_FRAMES,
        show_preview=False,
        max_training_records_per_class=DEFAULT_QUICK_TEST_MAX_TRAINING_RECORDS_PER_CLASS,
    )

    report_paths = get_experiment_output_paths(search_result["paths"])
    save_experiment_search_report(
        search_result,
        report_paths["experiment_search_report_path"],
    )
    save_best_experiment_report(
        get_best_experiment_result(search_result),
        report_paths["best_experiment_report_path"],
    )

    print(format_experiment_search_summary(search_result))
    print()
    print(f"Experiment search report saved to: {report_paths['experiment_search_report_path']}")
    print(f"Best experiment report saved to:   {report_paths['best_experiment_report_path']}")


if __name__ == "__main__":
    main()