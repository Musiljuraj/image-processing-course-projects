# This module is the "systematic search" layer of the project.
# The layer above it, auto_select_and_run.py, only decides that an automatic
# search should be executed and that the best result should later be reused for
# one final full rerun.
#
# This file is the layer that actually makes that search possible.
#
# Its role is to connect almost all lower project components into one repeatable
# experiment workflow:
# - build candidate preprocessing / LBP / classifier combinations,
# - reuse one shared training-record pool,
# - train one model for each configuration,
# - run that model on the real video frame by frame,
# - evaluate the predictions against the ground-truth file,
# - rank the finished experiments,
# - expose the best result in a form that the top orchestration layer can reuse.
#
# A useful way to think about this file is:
#
#     lower modules = building blocks
#     experiment_search.py = controlled comparison layer
#     auto_select_and_run.py = final selection / launch layer

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

# deepcopy is used throughout the project whenever configuration dictionaries
# or result dictionaries should be copied safely without allowing later stages
# to accidentally mutate shared structures.
from copy import deepcopy

# product is used to generate the Cartesian product of preprocessing, LBP, and
# classifier configuration lists. That is how the experiment runner turns
# several independent config lists into explicit experiment definitions.
from itertools import product

# Path is used for project-root-relative path construction, exactly like in
# main.py, so the experiment runner can be executed standalone and still find
# the same input/output directories.
from pathlib import Path

# perf_counter is used for all timing measurements in this project because the
# experiment layer needs consistent timing for:
# - model build time,
# - training feature preparation time,
# - training time,
# - per-frame localization time,
# - per-frame classification time,
# - total experiment time.
from time import perf_counter

# gc is used deliberately in this module because experiment search creates
# repeated model bundles and frame-result lists across multiple experiments.
# Explicit garbage collection helps keep memory usage more stable.
import gc

# OpenCV is needed here for:
# - opening the video,
# - converting frames to grayscale,
# - optional preview display,
# - drawing preview annotations.
import cv2

# NumPy is needed mainly for class-count computation from y_train and for
# keeping timing/model statistics in a simple numeric form.
import numpy as np

# These detector functions implement the localization stage reused from the
# normal runtime pipeline:
# - load the Haar cascades once,
# - detect all face candidates in a frame,
# - choose one main face,
# - localize eyes and mouth inside that chosen face.
from detectors import (
    load_cascades,
    detect_faces,
    select_main_face,
    detect_face_parts,
)

# These training-I/O functions provide the dataset side of the experiment layer:
# - gather training image paths,
# - parse the dataset filename format,
# - build one structured training record from disk,
# - optionally load the full dataset into structured records.
from eye_training_io import (
    collect_eye_image_paths,
    parse_eye_filename,
    build_one_eye_training_record,
    load_all_eye_training_records,
)

# These validators normalize configuration dictionaries before they are used by
# downstream preprocessing / feature extraction / classifier stages.
from eye_preprocessing import validate_preprocessing_config
from lbp_features import validate_lbp_config

# This dataset-bridge layer converts structured training records into
# classifier-ready matrices and aligned metadata.
from eye_lbp_dataset import (
    prepare_training_feature_records,
    build_training_matrix_and_labels,
)

# This classifier layer validates the classifier configuration and trains one
# model from X_train and y_train.
from eye_lbp_classifier import (
    validate_classifier_config,
    train_classifier,
)

# This is the runtime LBP frame-level eye-state classifier used during the
# frame-processing loop of each experiment.
from eye_state_lbp import classify_eye_state_lbp

# The evaluation layer is reused unchanged after each experiment finishes.
# The experiment runner only supplies frame_results and the ground-truth path.
from evaluation import (
    evaluate_results,
    format_evaluation_summary,
)


# ---------------------------------------------------------------------
# Runtime configuration reused from the main eye-LBP pipeline
# ---------------------------------------------------------------------

# These constants intentionally mirror the runtime behavior of main.py so that
# experiment runs are comparable to the final pipeline run.
#
# The experiment layer should not evaluate one kind of runtime behavior and
# then hand a completely different runtime behavior to the final pipeline.
# Reusing these settings keeps the search phase aligned with the final run.

# Enable mouth detection inside face-part localization.
# Mouth localization is not the final classification target here, but it is part
# of the shared face-parts structure produced by the localization pipeline.
ENABLE_MOUTH_DETECTION = True

# Face detection can be run on a downscaled grayscale frame for speed.
# The detector module will later rescale detections back to original-frame
# coordinates, so the public outputs remain in the same coordinate system.
FACE_DETECTION_DOWNSCALE_FACTOR = 0.75

# The LBP frame classifier can fall back to the older heuristic eye-state
# classifier when needed. This keeps runtime classification more robust in cases
# where LBP inference cannot produce a clean decision.
LBP_FALLBACK_TO_HEURISTIC = True

# If neither valid LBP evidence nor a reusable previous frame state can be used,
# this label is the final frame-level fallback.
LBP_FALLBACK_FRAME_LABEL = "close"

# The experiment runner is designed primarily for batch runs, so preview is off
# by default unless explicitly requested.
DEFAULT_SHOW_PREVIEW = False

# Separate preview-window name from main.py so experiment runs can be visually
# distinguished from ordinary final-pipeline runs.
PREVIEW_WINDOW_NAME = "ZAO Assignment 06 - Experiment Preview"

# These are the quick-test defaults used by the standalone main() at the bottom
# of this file. They intentionally keep the standalone experiment mode fast and
# manageable for smoke tests.
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

    # The project root is taken as the directory containing this module.
    # From there, every other input/output/reference path is built relative to
    # a stable project-root structure.
    project_root = Path(__file__).resolve().parent

    # Top-level directory grouping:
    # - input ..... raw project inputs
    # - output .... generated run outputs
    # - reference . reference materials
    input_dir = project_root / "input"
    output_dir = project_root / "output"
    reference_dir = project_root / "reference"

    # Input subdirectories:
    # - archives ........ archived source data
    # - cascades ........ Haar cascades
    # - ground_truth .... expected eye-state labels for the video
    # - video ........... target video
    # - training ........ extracted training dataset
    input_archives_dir = input_dir / "archives"
    input_cascades_dir = input_dir / "cascades"
    input_ground_truth_dir = input_dir / "ground_truth"
    input_video_dir = input_dir / "video"
    input_training_dir = input_dir / "training"

    # Detector-specific cascade locations are separated by object type.
    face_cascades_dir = input_cascades_dir / "face"
    eye_cascades_dir = input_cascades_dir / "eye"
    mouth_cascades_dir = input_cascades_dir / "mouth"

    # Output subdirectories:
    # - annotated_video ..... reserved video outputs
    # - frames .............. reserved frame outputs
    # - logs ................ text logs
    # - reports ............. evaluation reports
    # - experiments ......... experiment-search artifacts
    #     - results ......... summary text outputs
    #     - inspection ...... optional inspection artifacts
    output_annotated_video_dir = output_dir / "annotated_video"
    output_frames_dir = output_dir / "frames"
    output_logs_dir = output_dir / "logs"
    output_reports_dir = output_dir / "reports"
    experiment_output_dir = output_dir / "experiments"
    experiment_results_dir = experiment_output_dir / "results"
    experiment_inspection_dir = experiment_output_dir / "inspection"

    # Key dataset/reference locations that downstream stages expect.
    mrl_eyes_dataset_dir = input_training_dir / "mrlEyes_2018_01"
    reference_parking_lbp_dir = reference_dir / "06_ParkingOccupancy_LBP"

    # Return one shared dictionary so every later function can receive a single
    # path bundle instead of many separate path arguments.
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

        # The experiment runner uses the same canonical project inputs as the
        # final pipeline in main.py.
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

    # The experiment runner writes several output artifacts even when it is used
    # only for quick tests, so all relevant output directories are created up
    # front. mkdir(..., exist_ok=True) makes this safe to call repeatedly.
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

    # This helper centralizes the canonical text-output filenames used by the
    # experiment-search layer. Higher-level code can reuse these exact paths
    # instead of recreating them manually.
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

    # OpenCV works most reliably here when given a string path.
    capture = cv2.VideoCapture(str(video_path))

    # A missing or unreadable video file is treated as a hard failure because
    # the whole point of this module is to run real video experiments.
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    return capture


def convert_to_grayscale(frame):
    """
    Convert a BGR video frame to grayscale.
    """

    # The whole localization/classification pipeline operates on grayscale
    # frames, so this is the standard first conversion step for each frame.
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

    # This experiment-side record format intentionally matches the runtime
    # record format from main.py. That shared structure is what allows
    # evaluation.py to remain a generic post-processing layer instead of having
    # one format for final runs and another for experiment runs.
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

    # The caller owns the accumulating results list; this helper only appends
    # one fully structured frame record to it.
    results.append(frame_result)


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

def ensure_config_list(configs, config_name):
    """
    Ensure the given config collection is a non-empty list of dictionaries.
    """

    # At this layer, experiment generation expects explicit lists of config
    # dictionaries. This helper makes sure the structure is valid before the
    # Cartesian-product builder is allowed to combine them.
    if not isinstance(configs, list):
        raise TypeError(f"{config_name} must be a list.")

    if not configs:
        raise ValueError(f"{config_name} must not be empty.")

    # Every configuration item must itself be a dictionary, because lower
    # modules validate named configuration keys, not positional values.
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

    # This is the built-in "recommended search space" used whenever the caller
    # does not provide explicit config lists.
    #
    # The design is intentionally conservative:
    # - preprocessing is fixed to one reasonable baseline,
    # - classifier is fixed to one simple KNN setup,
    # - LBP settings are the main comparison axis.
    #
    # That design keeps the experiment count small and aligned with the stated
    # assignment goal of comparing at least three LBP configurations.
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

    # The three LBP variants mainly explore:
    # - different spatial grid layouts,
    # - a larger radius in the third variant.
    #
    # This gives three non-identical descriptors while keeping the rest of the
    # pipeline stable enough to make the comparison meaningful.
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

    # One small KNN model acts as the fixed classifier baseline.
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

    # First validate the three incoming config collections as experiment-ready
    # lists of dictionaries.
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

    # This list will hold explicit per-experiment configuration bundles,
    # already numbered and safe to pass to run_one_experiment(...).
    experiment_configurations = []

    # product(...) generates every possible combination of:
    # - one preprocessing config
    # - one LBP config
    # - one classifier config
    #
    # enumerate(..., start=1) gives each combination a stable human-readable
    # experiment index.
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
        # deepcopy is used so later stages cannot accidentally mutate the
        # original shared config templates when an experiment is being run.
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

    # None means "no subset cap", so the full dataset can be used.
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

    # Validate the optional subset cap first so the rest of the logic can assume
    # either:
    # - None
    # - positive integer
    max_training_records_per_class = _validate_max_training_records_per_class(
        max_training_records_per_class
    )

    # Full-data mode:
    # reuse the training-I/O module's standard loader and keep the logic simple.
    if max_training_records_per_class is None:
        return load_all_eye_training_records(
            dataset_root=dataset_root,
            load_images=True,
            grayscale=grayscale,
            recursive=recursive,
            ignore_invalid_files=ignore_invalid_files,
        )

    # Subset mode:
    # instead of loading everything and then trimming it, the code walks the
    # image paths directly from disk and stops as soon as both classes have
    # reached the requested cap.
    dataset_root = Path(dataset_root)
    image_paths = collect_eye_image_paths(dataset_root, recursive=recursive)

    selected_records = []

    # The binary class convention is preserved here:
    # - 0 = close
    # - 1 = open
    class_counts = {0: 0, 1: 0}

    # Iterate deterministically through the dataset and build records only until
    # the requested balanced subset is complete.
    for image_path in image_paths:
        try:
            parsed = parse_eye_filename(image_path.name)
        except Exception:
            # Invalid filenames can either be ignored or treated as hard errors,
            # depending on the caller's policy.
            if ignore_invalid_files:
                continue
            raise

        label = parsed["label"]

        # Skip anything outside the expected binary classes, even though the
        # current dataset loader is designed specifically for 0/1 labels.
        if label not in class_counts:
            continue

        # Skip extra samples once this class has already reached its cap.
        if class_counts[label] >= max_training_records_per_class:
            continue

        # Build and load a structured training record for the selected image.
        record = build_one_eye_training_record(
            image_path=image_path,
            dataset_root=dataset_root,
            load_image=True,
            grayscale=grayscale,
        )

        selected_records.append(record)
        class_counts[label] += 1

        # Stop as soon as both classes have enough examples.
        if all(
            class_counts[class_label] >= max_training_records_per_class
            for class_label in class_counts
        ):
            break

    # A subset request that loads nothing is always an error.
    if not selected_records:
        raise ValueError("No training records were loaded for experiment search.")

    # The experiment subset must contain both classes, otherwise the classifier
    # training stage would either fail or produce a meaningless binary model.
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

    # The experiment runner expects the shared training pool to already be a
    # structured list of training records.
    if not isinstance(training_records, list):
        raise TypeError("training_records must be a list.")

    # Validate and normalize all three configuration dictionaries before any
    # expensive preprocessing / feature extraction / training work begins.
    normalized_preprocessing_config = validate_preprocessing_config(preprocessing_config)
    normalized_lbp_config = validate_lbp_config(lbp_config)
    normalized_classifier_config = validate_classifier_config(classifier_config)

    # Start total model-build timing.
    model_build_start = perf_counter()

    # The first measured substage is feature preparation from the training
    # records:
    # raw eye image -> preprocessing -> LBP descriptor -> feature records
    training_feature_prep_start = perf_counter()

    training_feature_records = prepare_training_feature_records(
        training_records=training_records,
        preprocessing_config=normalized_preprocessing_config,
        lbp_config=normalized_lbp_config,
        image_key=image_key,
    )

    # Convert the enriched feature records into the actual classifier training
    # representation:
    # - X_train ........ feature matrix
    # - y_train ........ class labels
    # - training_metadata ... aligned metadata for later inspection if needed
    X_train, y_train, training_metadata = build_training_matrix_and_labels(
        training_feature_records=training_feature_records
    )

    training_feature_prep_end = perf_counter()

    # The second measured substage is actual classifier fitting.
    training_start = perf_counter()

    trained_model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=normalized_classifier_config,
    )

    training_end = perf_counter()
    model_build_end = perf_counter()

    # Store useful structural training statistics in the returned model bundle
    # so experiment reports can describe not just accuracy, but also the scale
    # of the trained model.
    training_sample_count = int(X_train.shape[0])
    feature_count = int(X_train.shape[1])

    # Class counts are computed from y_train so the experiment result captures
    # the actual class balance used during training.
    class_counts = {
        "close": int(np.sum(y_train == 0)),
        "open": int(np.sum(y_train == 1)),
    }

    # The returned model bundle is deliberately lean and runtime-oriented.
    # It keeps:
    # - the trained model,
    # - the normalized configs used to build it,
    # - a few training statistics,
    # - timing measurements.
    #
    # It does not keep the full training matrices or full feature-record list.
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

    # Explicitly release bulky intermediate objects before the next experiment
    # starts. This matters because experiment search may train multiple models
    # in the same Python process.
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

    # Validate the shared inputs first, because this function is the heavy,
    # expensive part of the experiment layer and should fail fast on bad input.
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

    # Start total experiment timing. This includes:
    # - model build
    # - video processing
    # - evaluation
    experiment_start = perf_counter()

    # Build one experiment-specific model bundle from the shared training-record
    # pool and the current experiment configuration.
    model_bundle = build_eye_lbp_model_from_training_records(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        classifier_config=classifier_config,
        image_key="image",
    )

    # Open the real evaluation video.
    capture = open_video(paths["video_path"])

    # Per-experiment runtime state:
    # - frame_count ........... number of frames actually processed
    # - frame_results ......... structured per-frame records
    # - previous_face ......... continuity cue for stable face selection
    # - previous_eye_state .... temporal fallback cue for eye-state aggregation
    # - stopped_early ......... whether the user aborted through preview window
    frame_count = 0
    frame_results = []
    previous_face = None
    previous_eye_state = None
    stopped_early = False

    # Create the preview window only when interactive preview was requested.
    if show_preview:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        # Main frame-processing loop for this one experiment.
        while True:
            # Optional early stop for quick experiments.
            if max_frames is not None and frame_count >= max_frames:
                break

            ret, frame = capture.read()

            # End of video or read failure.
            if not ret:
                break

            frame_count += 1
            frame_processing_start = perf_counter()

            # Convert the raw BGR frame to the grayscale representation expected
            # by the localization and eye-state modules.
            gray_frame = convert_to_grayscale(frame)

            # -------------------------
            # Localization stage
            # -------------------------
            #
            # This stage performs:
            # - face detection
            # - main-face selection
            # - eye/mouth localization within that selected face
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

            # -------------------------
            # Classification stage
            # -------------------------
            #
            # This stage hands the localized face parts plus the trained LBP
            # model bundle into the runtime eye-state classifier.
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

            # Total frame time measures the combined cost of localization and
            # classification for the current frame in this experiment run.
            frame_processing_end = perf_counter()
            total_frame_time_ms = (
                frame_processing_end - frame_processing_start
            ) * 1000.0

            # Persist one structured frame record for later evaluation and timing
            # analysis.
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

            # Update temporal state so the next frame can use continuity cues.
            previous_face = main_face
            previous_eye_state = eye_state

            # Optional preview branch:
            # show the current frame with face / eyes / mouth / frame-level label
            # annotations, mainly for debugging and manual inspection.
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

                # Allow the user to abort the current experiment interactively.
                if key == ord("q") or key == 27:
                    stopped_early = True
                    break

    finally:
        # Always release video resources and close preview windows, even if the
        # loop exits through error or manual interruption.
        capture.release()
        if show_preview:
            cv2.destroyAllWindows()

    experiment_end = perf_counter()

    # After the frame loop finishes, the stored frame_results are compared
    # against the ground-truth file to produce one evaluation summary.
    evaluation_summary = evaluate_results(
        frame_results=frame_results,
        ground_truth_path=paths["ground_truth_path"],
    )

    # This dictionary is the central output of one experiment.
    # It combines:
    # - the exact configs used
    # - model/training statistics
    # - timing measurements
    # - frame-count information
    # - the full evaluation summary
    # - a few top-level fields copied out for easier ranking/reporting
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

    # Explicit cleanup again matters here because each experiment may have
    # accumulated a long frame-results list and a trained model bundle.
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

    # The ranking function is intentionally explicit and deterministic.
    # It accepts only a list because ranking is defined over a sequence of
    # completed experiment-result dictionaries.
    if not isinstance(experiment_results, list):
        raise TypeError("experiment_results must be a list.")

    # Sorting policy:
    # - maximize accuracy first,
    # - then prefer lower runtime cost,
    # - then prefer lower classification cost,
    # - then prefer lower model-build cost.
    #
    # Accuracy is negated because sorted(...) is ascending by default.
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

    # The caller is expected to pass the full search_result bundle returned by
    # run_experiment_search(...), not just a raw ranked list.
    if not isinstance(search_result, dict):
        raise TypeError("search_result must be a dictionary.")

    ranked_results = search_result.get("ranked_results")

    if not isinstance(ranked_results, list) or not ranked_results:
        raise ValueError("search_result does not contain any ranked_results.")

    # Because rank_experiment_results(...) already sorts from best to worst,
    # the winning experiment is always the first item.
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

    # This helper is the handoff point from "experiment result" to "runtime
    # configuration bundle". auto_select_and_run.py uses exactly this function
    # to turn the winning experiment back into the configs that main.py needs.
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

    # Return deep copies so later callers can safely reuse or modify them
    # without mutating the original experiment_result structure.
    return {
        "preprocessing_config": deepcopy(experiment_result["preprocessing_config"]),
        "lbp_config": deepcopy(experiment_result["lbp_config"]),
        "classifier_config": deepcopy(experiment_result["classifier_config"]),
    }


def format_one_experiment_result(experiment_result, rank_index=None):
    """
    Format one experiment result as readable multiline text.
    """

    # The heading changes slightly depending on whether this result is being
    # shown as a raw experiment entry or as part of a ranked list.
    header = (
        f"=== Experiment #{experiment_result['experiment_index']} ==="
        if rank_index is None
        else f"=== Rank {rank_index} / Experiment #{experiment_result['experiment_index']} ==="
    )

    # This formatter deliberately combines:
    # - top-level outcome quality,
    # - timing breakdown,
    # - training/model statistics,
    # - the exact configs used,
    # - the full lower-level evaluation summary.
    #
    # The goal is that a reader can understand not only which experiment won,
    # but also why it might have won and what computational cost it had.
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

    # The search summary is built from already-ranked results, so the final text
    # appears in best-to-worst order rather than execution order.
    ranked_results = search_result["ranked_results"]

    lines = [
        "=== Eye LBP experiment search summary ===",
        f"Experiment count: {len(search_result['experiment_results'])}",
        f"Training subset limit per class: {search_result['max_training_records_per_class']}",
        "",
    ]

    # Append one formatted block per ranked experiment.
    for rank_index, experiment_result in enumerate(ranked_results, start=1):
        lines.append(format_one_experiment_result(experiment_result, rank_index=rank_index))
        lines.append("")

    # rstrip() removes the trailing blank line caused by the final append("").
    return "\n".join(lines).rstrip()


def format_best_experiment_summary(experiment_result):
    """
    Format one compact summary of the best-ranked experiment.
    """

    # This is intentionally shorter than format_one_experiment_result(...).
    # It is meant for the dedicated "best experiment" report and for the
    # top-level automatic-selection summary file.
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

    # First format the whole ranked search output into one text block.
    report_text = format_experiment_search_summary(search_result)

    # Then persist it as a plain UTF-8 text file.
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_text + "\n")


def save_best_experiment_report(experiment_result, report_path):
    """
    Save one compact best-experiment summary to a text file.
    """

    # This is the shorter report that contains only the winner, not the full
    # ranked list.
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

    # Build standard project paths and ensure the experiment output directories
    # exist before any experiment work begins.
    paths = get_project_paths()
    ensure_output_directories(paths)

    # Normalize the optional subset limit once at the entry point.
    max_training_records_per_class = _validate_max_training_records_per_class(
        max_training_records_per_class
    )

    # If any of the three config lists is missing, fill only the missing ones
    # from the recommended default search-space definition.
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

    # Turn the config lists into explicit per-experiment bundles.
    experiment_configurations = build_experiment_configurations(
        preprocessing_configurations=preprocessing_configurations,
        lbp_configurations=lbp_configurations,
        classifier_configurations=classifier_configurations,
    )

    # Load the training records once and reuse them across all experiments.
    # This avoids repeated dataset I/O and keeps the experiment loop focused on
    # model building and runtime evaluation.
    training_records = load_training_records_for_experiments(
        dataset_root=paths["mrl_eyes_dataset_dir"],
        max_training_records_per_class=max_training_records_per_class,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False,
    )

    # Load all Haar cascades once and reuse them across all experiments for the
    # same reason: the experiment loop should not repeat static setup work.
    cascades = load_cascades(paths)

    experiment_results = []

    # Execute one full end-to-end experiment for every configuration bundle.
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

        # Attach the experiment index from the configuration bundle to the
        # returned experiment result so later ranking/reporting can reference it.
        experiment_result = {
            "experiment_index": experiment_configuration["experiment_index"],
            **experiment_result,
        }

        experiment_results.append(experiment_result)

        # Encourage timely cleanup between experiments.
        gc.collect()

    # Rank the completed experiments from best to worst.
    ranked_results = rank_experiment_results(experiment_results)

    # Return one top-level search bundle containing both raw execution order and
    # ranked order, along with the shared paths and the subset-cap metadata.
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

    # Standalone mode is a smoke-test / inspection mode, not the final full
    # search used by auto_select_and_run.py.
    search_result = run_experiment_search(
        preprocessing_configurations=None,
        lbp_configurations=None,
        classifier_configurations=None,
        max_frames=DEFAULT_QUICK_TEST_MAX_FRAMES,
        show_preview=False,
        max_training_records_per_class=DEFAULT_QUICK_TEST_MAX_TRAINING_RECORDS_PER_CLASS,
    )

    # Save both the full ranked report and the short winner-only report.
    report_paths = get_experiment_output_paths(search_result["paths"])
    save_experiment_search_report(
        search_result,
        report_paths["experiment_search_report_path"],
    )
    save_best_experiment_report(
        get_best_experiment_result(search_result),
        report_paths["best_experiment_report_path"],
    )

    # Also print the ranked summary to the console for direct inspection.
    print(format_experiment_search_summary(search_result))
    print()
    print(f"Experiment search report saved to: {report_paths['experiment_search_report_path']}")
    print(f"Best experiment report saved to:   {report_paths['best_experiment_report_path']}")


# Standard standalone entry point for quick local experiment-search execution.
if __name__ == "__main__":
    main()