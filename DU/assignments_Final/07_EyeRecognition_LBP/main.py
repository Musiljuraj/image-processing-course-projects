# This module is the main reusable runtime pipeline of the project.
# It sits directly under the top orchestration layer from auto_select_and_run.py
# and acts as the "execute one full real run" branch of the final solution.
#
# In the overall project flow, this file is responsible for:
#
#     chosen runtime configs
#         -> build startup-trained LBP model bundle
#         -> load detection cascades
#         -> open real input video
#         -> process frames one by one
#             -> detect faces
#             -> select one main face
#             -> localize face parts
#             -> classify frame-level eye state
#             -> store structured frame result
#         -> evaluate all stored frame results against ground truth
#         -> save run log and evaluation report
#
# A key design idea here is reusability.
# This file is not only the preserved manual entry point of the project, but
# also the reusable final-run function called by auto_select_and_run.py after
# experiment search selects the best configuration.
#
# That means this file is the project's concrete execution layer:
# - experiment_search.py decides which configs are worth comparing,
# - auto_select_and_run.py decides which config wins,
# - main.py actually runs the final end-to-end pipeline for one chosen config.

"""
main.py

This module is the main entry point of the project.

Its responsibilities are:
- building the project path configuration,
- opening the input video,
- training the LBP eye-state model once at startup,
- running the frame-by-frame processing loop,
- calling localization and classification modules,
- measuring localization, classification, and total frame-processing time,
- optionally displaying a live annotated preview,
- storing per-frame results,
- invoking the final evaluation step,
- saving the run log and the evaluation report.

Additional final-solution responsibility:
- expose one reusable run_final_pipeline(...) helper so another module can
  automatically select the best configuration from experiment search and then
  execute one full final run with that exact configuration.
"""

# Path is used to build all project-relative directories and file paths from a
# stable root location.
from pathlib import Path

# perf_counter is used for precise runtime timing of:
# - full model build time,
# - localization time per frame,
# - classification time per frame,
# - total frame-processing time.
from time import perf_counter

# datetime is used only for timestamped run-log entries.
from datetime import datetime

# re is used to sanitize optional output filename prefixes into safe,
# filesystem-friendly names.
import re

# OpenCV is used here for:
# - video reading,
# - grayscale conversion,
# - preview-window display,
# - drawing annotations on the live preview.
import cv2

# These are the shared localization functions reused from the detector layer.
# They provide all geometry/localization outputs needed by the runtime loop:
# - load cascade XML classifiers,
# - detect faces,
# - select one stable main face,
# - detect eyes and mouth inside that face.
from detectors import (
    load_cascades,
    detect_faces,
    select_main_face,
    detect_face_parts,
)

# This helper builds one startup-trained LBP model bundle from the training
# dataset before frame-by-frame runtime inference begins.
from eye_lbp_classifier import build_eye_lbp_model

# This helper performs the frame-level LBP eye-state decision during runtime.
from eye_state_lbp import classify_eye_state_lbp

# The evaluation layer is reused after runtime processing ends:
# - compute the final summary,
# - optionally print it,
# - save it as the final evaluation report.
from evaluation import (
    evaluate_results,
    print_evaluation_summary,
    save_evaluation_report,
)


# ---------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------

# Mouth detection is part of the shared face-parts localization structure.
# Even though mouth is not the final classification target, it remains part of
# the runtime outputs and preview visualization.
ENABLE_MOUTH_DETECTION = True

# Face detection is the most global localization step, so it can be run on a
# slightly downscaled grayscale frame for speed. The detector layer later maps
# all boxes back to original-frame coordinates.
FACE_DETECTION_DOWNSCALE_FACTOR = 0.75

# The runtime LBP eye-state classifier can fall back to the older heuristic
# classifier when needed.
LBP_FALLBACK_TO_HEURISTIC = True

# If no valid temporal or classifier-based result is available, this is the
# final frame-level fallback label.
LBP_FALLBACK_FRAME_LABEL = "close"


# ---------------------------------------------------------------------
# Live-preview configuration
# ---------------------------------------------------------------------

# The preserved manual entry point defaults to no live preview, but the reusable
# final pipeline can still enable preview when explicitly requested.
SHOW_TEST_VIDEO = False

# These switches control which overlays appear on the preview frame.
SHOW_FACE_BOXES = True
SHOW_EYE_BOXES = True
SHOW_MOUTH_BOXES = True
SHOW_EYE_STATE_TEXT = True

# Preview does not have to be shown for every processed frame, but the current
# default is every frame.
PREVIEW_EVERY_N_FRAMES = 1

# Window title used for the runtime preview display.
PREVIEW_WINDOW_NAME = "ZAO Assignment 06 - Preview"


# ---------------------------------------------------------------------
# Project path configuration
# ---------------------------------------------------------------------

def get_project_paths():
    """
    Build and return all relevant project paths.
    """

    # The directory containing this file is treated as the project root.
    # From there, all input, output, and reference directories are derived.
    project_root = Path(__file__).resolve().parent

    input_dir = project_root / "input"
    output_dir = project_root / "output"
    reference_dir = project_root / "reference"

    # Group input subdirectories by data role.
    input_archives_dir = input_dir / "archives"
    input_cascades_dir = input_dir / "cascades"
    input_ground_truth_dir = input_dir / "ground_truth"
    input_video_dir = input_dir / "video"
    input_training_dir = input_dir / "training"

    # Group cascade subdirectories by detector type.
    face_cascades_dir = input_cascades_dir / "face"
    eye_cascades_dir = input_cascades_dir / "eye"
    mouth_cascades_dir = input_cascades_dir / "mouth"

    # Group output directories by artifact type.
    output_annotated_video_dir = output_dir / "annotated_video"
    output_frames_dir = output_dir / "frames"
    output_logs_dir = output_dir / "logs"
    output_reports_dir = output_dir / "reports"
    experiment_output_dir = output_dir / "experiments"
    experiment_results_dir = experiment_output_dir / "results"
    experiment_inspection_dir = experiment_output_dir / "inspection"

    # Canonical training/reference directories reused elsewhere in the project.
    mrl_eyes_dataset_dir = input_training_dir / "mrlEyes_2018_01"
    reference_parking_lbp_dir = reference_dir / "06_ParkingOccupancy_LBP"

    # Return one shared path bundle so later code can work with a single paths
    # dictionary instead of many separate path arguments.
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

        # Canonical project inputs used by the final runtime pipeline.
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


# ---------------------------------------------------------------------
# Output-directory and text-output helpers
# ---------------------------------------------------------------------

def ensure_output_directories(paths):
    """
    Ensure that all output directories required by the program exist.
    """

    # Create every runtime and experiment output directory up front so the rest
    # of the pipeline can write logs and reports without repeated directory
    # existence checks.
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_annotated_video_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_frames_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["experiment_output_dir"].mkdir(parents=True, exist_ok=True)
    paths["experiment_results_dir"].mkdir(parents=True, exist_ok=True)
    paths["experiment_inspection_dir"].mkdir(parents=True, exist_ok=True)


def _normalize_output_file_prefix(output_file_prefix):
    """
    Normalize one optional output filename prefix.

    Examples:
    - None              -> None
    - "auto selected"   -> "auto_selected"
    - "best-config#1"   -> "best_config_1"
    """

    # No prefix requested means the default filenames should remain unchanged.
    if output_file_prefix is None:
        return None

    # Normalize user-provided text into a lowercase safe token using only
    # alphanumeric characters and underscores.
    normalized = str(output_file_prefix).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")

    # If sanitization removes everything, treat the prefix as absent.
    if not normalized:
        return None

    return normalized


def get_output_file_paths(paths, output_file_prefix=None):
    """
    Build the output file paths for the run log and evaluation report.

    Behavior:
    - if output_file_prefix is None:
        keep the original filenames
            run_log.txt
            evaluation_report.txt
    - otherwise:
        generate prefixed filenames such as
            run_log_auto_selected_best.txt
            evaluation_report_auto_selected_best.txt

    This allows automatic best-configuration runs to keep their outputs
    separate from quick manual runs.
    """

    # Normalize the optional prefix first so filename construction remains safe
    # and consistent.
    normalized_prefix = _normalize_output_file_prefix(output_file_prefix)

    # Without a prefix, preserve the original fixed filenames.
    if normalized_prefix is None:
        return {
            "run_log_path": paths["output_logs_dir"] / "run_log.txt",
            "evaluation_report_path": paths["output_reports_dir"] / "evaluation_report.txt",
        }

    # With a prefix, generate separate files so different run modes do not
    # overwrite one another.
    return {
        "run_log_path": paths["output_logs_dir"] / f"run_log_{normalized_prefix}.txt",
        "evaluation_report_path": paths["output_reports_dir"] / f"evaluation_report_{normalized_prefix}.txt",
    }


def reset_text_file(file_path):
    """
    Create or clear a text file.
    """

    # Opening in write mode with no body is enough to create or truncate the
    # target log/report file.
    with open(file_path, "w", encoding="utf-8"):
        pass


def log_message(message, log_path=None, print_to_console=True):
    """
    Print one message and optionally append it to the run log.
    """

    # Print to console first when requested, so the runtime can still be watched
    # interactively even when a log file is also being written.
    if print_to_console:
        print(message)

    # When a log path is provided, append the same message to the run log with a
    # timestamp.
    if log_path is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")


# ---------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------

def open_video(video_path):
    """
    Open the input video and return an OpenCV VideoCapture object.
    """

    # OpenCV expects the path in string form.
    capture = cv2.VideoCapture(str(video_path))

    # If the video cannot be opened, the runtime pipeline cannot proceed.
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    return capture


def convert_to_grayscale(frame):
    """
    Convert a BGR video frame to grayscale.
    """

    # The whole localization/classification path in this project operates on
    # grayscale frames.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------
# Result-storage helper
# ---------------------------------------------------------------------

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

    Stored timing values:
    - localization_time_ms ..... face + face-parts localization
    - classification_time_ms ... LBP eye-state inference
    - total_frame_time_ms ...... localization + classification processing time

    The collected records are later used by the evaluation module to compute
    accuracy and timing statistics.
    """

    # Keep one fully structured frame-level record so the later evaluation stage
    # can compare predicted eye_state labels against ground truth and compute
    # timing statistics without rerunning the pipeline.
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
# Live-preview helpers
# ---------------------------------------------------------------------

def build_preview_frame(frame, face_boxes, main_face, face_parts, eye_state):
    """
    Build an annotated preview frame from the current processing results.
    """

    # Work on a copy so preview drawing never mutates the original frame used
    # for processing.
    preview_frame = frame.copy()

    # Draw all detected face candidates and highlight the selected main face.
    if SHOW_FACE_BOXES:
        for (x, y, w, h) in face_boxes:
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if main_face is not None:
            x, y, w, h = main_face
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Draw detected eye boxes.
    if SHOW_EYE_BOXES:
        for (x, y, w, h) in face_parts["eyes"]:
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw detected mouth boxes only if mouth detection is enabled globally.
    if SHOW_MOUTH_BOXES and ENABLE_MOUTH_DETECTION:
        for (x, y, w, h) in face_parts["mouth"]:
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Overlay the current frame-level eye-state decision.
    if SHOW_EYE_STATE_TEXT:
        cv2.putText(
            preview_frame,
            f"eye state: {eye_state}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    return preview_frame


def show_preview_frame(preview_frame):
    """
    Display one preview frame and return the pressed key code.
    """

    # Show the current annotated frame and return the pressed key so the caller
    # can react to quit requests.
    cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)
    key = cv2.waitKey(1) & 0xFF
    return key


# ---------------------------------------------------------------------
# Reusable final-pipeline runner
# ---------------------------------------------------------------------

def run_final_pipeline(
    preprocessing_config=None,
    lbp_config=None,
    classifier_config=None,
    run_name="final_run",
    show_preview=SHOW_TEST_VIDEO,
    output_file_prefix=None,
    print_summary=True,
):
    """
    Execute one full final video pipeline run with explicit configs.

    This is the reusable final-run helper that allows:
    - normal standalone execution from main.py,
    - automatic best-configuration execution from auto_select_and_run.py.

    Inputs:
    - preprocessing_config ..... explicit preprocessing configuration or None
    - lbp_config ............... explicit LBP configuration or None
    - classifier_config ........ explicit classifier configuration or None
    - run_name ................. descriptive run label stored in logs/reports
    - show_preview ............. whether to show live preview
    - output_file_prefix ....... optional filename prefix for log/report files
    - print_summary ............ whether to print the final evaluation summary

    Return:
    - run_result dictionary containing:
        paths
        output_files
        run_name
        model_bundle
        model_build_time_ms
        frame_count
        frame_results_count
        evaluation_summary
    """

    # Step 1:
    # build shared project paths and make sure all output directories exist.
    paths = get_project_paths()
    ensure_output_directories(paths)

    # Step 2:
    # resolve the output log/report paths for this specific run and clear the run
    # log before writing anything new into it.
    output_files = get_output_file_paths(
        paths,
        output_file_prefix=output_file_prefix,
    )
    reset_text_file(output_files["run_log_path"])

    # Step 3:
    # write the initial run-header information to the log.
    log_message("=== ZAO Assignment 06 ===", output_files["run_log_path"])
    log_message(f"Run name: {run_name}", output_files["run_log_path"])
    log_message("Project started.", output_files["run_log_path"])
    log_message("", output_files["run_log_path"])

    # Step 4:
    # build the startup-trained LBP model bundle once before processing the
    # video. This includes dataset loading, preprocessing, LBP extraction, and
    # classifier training inside the lower layers.
    model_build_start = perf_counter()

    log_message("Building startup-trained LBP eye-state model...", output_files["run_log_path"])
    model_bundle = build_eye_lbp_model(
        dataset_root=paths["mrl_eyes_dataset_dir"],
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        classifier_config=classifier_config,
        load_images=True,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False,
        image_key="image",
    )

    model_build_end = perf_counter()
    model_build_time_ms = (model_build_end - model_build_start) * 1000.0

    # Log the key model-build outputs so the run log captures the exact trained
    # setup used for this execution.
    log_message("LBP eye-state model built successfully.", output_files["run_log_path"])
    log_message(
        f"Training samples: {model_bundle['training_sample_count']}, "
        f"feature count: {model_bundle['feature_count']}",
        output_files["run_log_path"]
    )
    log_message(
        f"Class counts: {model_bundle['class_counts']}",
        output_files["run_log_path"]
    )
    log_message(
        f"Model build time [ms]: {model_build_time_ms:.3f}",
        output_files["run_log_path"]
    )
    log_message("", output_files["run_log_path"])

    # Step 5:
    # load the localization cascades and log the initialized runtime components.
    cascades = load_cascades(paths)

    log_message(f"Project root: {paths['project_root']}", output_files["run_log_path"])
    log_message(f"Video path:   {paths['video_path']}", output_files["run_log_path"])
    log_message("Cascades loaded successfully.", output_files["run_log_path"])
    log_message("Frontal face detector initialized.", output_files["run_log_path"])
    log_message("Profile face detector initialized (fallback mode).", output_files["run_log_path"])
    log_message(f"Face detection downscale factor: {FACE_DETECTION_DOWNSCALE_FACTOR}", output_files["run_log_path"])
    log_message("Eye detector initialized.", output_files["run_log_path"])
    log_message("Mouth/smile detector initialized.", output_files["run_log_path"])
    log_message("LBP eye-state classifier initialized.", output_files["run_log_path"])
    log_message("Evaluation module initialized.", output_files["run_log_path"])
    log_message("", output_files["run_log_path"])

    # Step 6:
    # open the real input video and initialize all runtime state used by the
    # frame loop.
    capture = open_video(paths["video_path"])

    frame_count = 0
    frame_results = []
    previous_face = None
    previous_eye_state = None

    log_message("Video opened successfully.", output_files["run_log_path"])
    log_message("Reading frames...", output_files["run_log_path"])

    # Optional live-preview initialization.
    if show_preview:
        log_message(
            "Live preview enabled. Press 'q' or ESC to stop preview/run.",
            output_files["run_log_path"]
        )
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

    log_message("", output_files["run_log_path"])

    try:
        # Step 7:
        # process the input video frame by frame until the video ends or the
        # preview loop is interrupted by the user.
        while True:
            ret, frame = capture.read()

            if not ret:
                break

            frame_count += 1

            # Keep one total-processing timer around the whole frame so both the
            # detailed sub-timings and the end-to-end frame time can be stored.
            frame_processing_start = perf_counter()
            gray_frame = convert_to_grayscale(frame)

            # -------------------------
            # Localization stage
            # -------------------------
            #
            # Find candidate faces, select one stable main face, and localize
            # eyes/mouth inside that face.
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
            # Use the trained LBP runtime classifier to produce one frame-level
            # eye-state label from the current localized face parts.
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
            classification_time_ms = (classification_end - classification_start) * 1000.0

            # End-to-end frame-processing time.
            frame_processing_end = perf_counter()
            total_frame_time_ms = (frame_processing_end - frame_processing_start) * 1000.0

            # Store one structured runtime record for later evaluation.
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

            # Optional preview branch.
            if show_preview and (frame_count % PREVIEW_EVERY_N_FRAMES == 0):
                preview_frame = build_preview_frame(
                    frame,
                    face_boxes,
                    main_face,
                    face_parts,
                    eye_state
                )

                key = show_preview_frame(preview_frame)

                # Allow the preview loop to stop the run early.
                if key == ord("q") or key == 27:
                    log_message("Preview interrupted by user.", output_files["run_log_path"])
                    break

            # Periodic runtime log snapshot every 30 processed frames.
            if frame_count % 30 == 0:
                log_message(
                    (
                        f"Processed frame: {frame_count}, "
                        f"detected faces: {len(face_boxes)}, "
                        f"detected eyes: {len(face_parts['eyes'])}, "
                        f"detected mouth: {len(face_parts['mouth'])}, "
                        f"eye_state: {eye_state}, "
                        f"localization_ms: {localization_time_ms:.3f}, "
                        f"classification_ms: {classification_time_ms:.3f}, "
                        f"total_frame_ms: {total_frame_time_ms:.3f}"
                    ),
                    output_files["run_log_path"]
                )

    finally:
        # Always release video/GUI resources even if the run exits early.
        capture.release()
        cv2.destroyAllWindows()

    # Step 8:
    # finalize the runtime log with summary frame counts.
    log_message("", output_files["run_log_path"])
    log_message("Video reading finished.", output_files["run_log_path"])
    log_message(f"Total frames processed: {frame_count}", output_files["run_log_path"])
    log_message(f"Stored frame results:   {len(frame_results)}", output_files["run_log_path"])
    log_message("", output_files["run_log_path"])

    # Step 9:
    # compare the stored frame-level predictions against the ground-truth label
    # file.
    evaluation_summary = evaluate_results(
        frame_results,
        paths["ground_truth_path"]
    )

    # Optional console summary printout for interactive/manual runs.
    if print_summary:
        print_evaluation_summary(evaluation_summary)

    # Step 10:
    # save the final evaluation report, including both the evaluation summary and
    # extra lines documenting the exact runtime configuration used.
    save_evaluation_report(
        evaluation_summary,
        output_files["evaluation_report_path"],
        extra_lines=[
            "=== Run configuration ===",
            f"Run name: {run_name}",
            f"Project root: {paths['project_root']}",
            f"Video path: {paths['video_path']}",
            f"Ground truth path: {paths['ground_truth_path']}",
            f"Face detection downscale factor: {FACE_DETECTION_DOWNSCALE_FACTOR}",
            f"Mouth detection enabled: {ENABLE_MOUTH_DETECTION}",
            f"Live preview enabled: {show_preview}",
            f"Preview every N frames: {PREVIEW_EVERY_N_FRAMES}",
            f"LBP fallback to heuristic: {LBP_FALLBACK_TO_HEURISTIC}",
            f"LBP fallback frame label: {LBP_FALLBACK_FRAME_LABEL}",
            f"LBP preprocessing config: {model_bundle['preprocessing_config']}",
            f"LBP feature config: {model_bundle['lbp_config']}",
            f"LBP classifier config: {model_bundle['classifier_config']}",
            f"LBP training sample count: {model_bundle['training_sample_count']}",
            f"LBP feature count: {model_bundle['feature_count']}",
            f"LBP training class counts: {model_bundle['class_counts']}",
            f"LBP model build time [ms]: {model_build_time_ms:.3f}",
        ]
    )

    # Log the final output file locations.
    log_message(
        f"Evaluation report saved to: {output_files['evaluation_report_path']}",
        output_files["run_log_path"]
    )
    log_message(
        f"Run log saved to: {output_files['run_log_path']}",
        output_files["run_log_path"]
    )

    # Return one structured run bundle so this function can be reused both by
    # the preserved manual entry point and by the automatic best-configuration
    # orchestration layer.
    run_result = {
        "paths": paths,
        "output_files": output_files,
        "run_name": run_name,
        "model_bundle": model_bundle,
        "model_build_time_ms": model_build_time_ms,
        "frame_count": frame_count,
        "frame_results_count": len(frame_results),
        "evaluation_summary": evaluation_summary,
    }

    return run_result


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main():
    """
    Execute the complete project pipeline using the default runtime configs.

    This preserved entry point remains useful for manual runs. The automatic
    final solution will call run_final_pipeline(...) from auto_select_and_run.py
    with the best-ranked configs found during experiment search.
    """

    # The preserved standalone/manual entry point simply delegates to the
    # reusable final pipeline with default configs and default output names.
    run_final_pipeline(
        preprocessing_config=None,
        lbp_config=None,
        classifier_config=None,
        run_name="manual_default_run",
        show_preview=SHOW_TEST_VIDEO,
        output_file_prefix=None,
        print_summary=True,
    )


# Standard script entry point.
if __name__ == "__main__":
    main()