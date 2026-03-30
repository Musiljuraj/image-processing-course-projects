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

from pathlib import Path
from time import perf_counter
from datetime import datetime
import re
import cv2

from detectors import (
    load_cascades,
    detect_faces,
    select_main_face,
    detect_face_parts,
)

from eye_lbp_classifier import build_eye_lbp_model
from eye_state_lbp import classify_eye_state_lbp

from evaluation import (
    evaluate_results,
    print_evaluation_summary,
    save_evaluation_report,
)


# ---------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------

ENABLE_MOUTH_DETECTION = True
FACE_DETECTION_DOWNSCALE_FACTOR = 0.75

LBP_FALLBACK_TO_HEURISTIC = True
LBP_FALLBACK_FRAME_LABEL = "close"


# ---------------------------------------------------------------------
# Live-preview configuration
# ---------------------------------------------------------------------

SHOW_TEST_VIDEO = False
SHOW_FACE_BOXES = True
SHOW_EYE_BOXES = True
SHOW_MOUTH_BOXES = True
SHOW_EYE_STATE_TEXT = True

PREVIEW_EVERY_N_FRAMES = 1
PREVIEW_WINDOW_NAME = "ZAO Assignment 06 - Preview"


# ---------------------------------------------------------------------
# Project path configuration
# ---------------------------------------------------------------------

def get_project_paths():
    """
    Build and return all relevant project paths.
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


# ---------------------------------------------------------------------
# Output-directory and text-output helpers
# ---------------------------------------------------------------------

def ensure_output_directories(paths):
    """
    Ensure that all output directories required by the program exist.
    """

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

    if output_file_prefix is None:
        return None

    normalized = str(output_file_prefix).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")

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

    normalized_prefix = _normalize_output_file_prefix(output_file_prefix)

    if normalized_prefix is None:
        return {
            "run_log_path": paths["output_logs_dir"] / "run_log.txt",
            "evaluation_report_path": paths["output_reports_dir"] / "evaluation_report.txt",
        }

    return {
        "run_log_path": paths["output_logs_dir"] / f"run_log_{normalized_prefix}.txt",
        "evaluation_report_path": paths["output_reports_dir"] / f"evaluation_report_{normalized_prefix}.txt",
    }


def reset_text_file(file_path):
    """
    Create or clear a text file.
    """

    with open(file_path, "w", encoding="utf-8"):
        pass


def log_message(message, log_path=None, print_to_console=True):
    """
    Print one message and optionally append it to the run log.
    """

    if print_to_console:
        print(message)

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

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    return capture


def convert_to_grayscale(frame):
    """
    Convert a BGR video frame to grayscale.
    """

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

    preview_frame = frame.copy()

    if SHOW_FACE_BOXES:
        for (x, y, w, h) in face_boxes:
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if main_face is not None:
            x, y, w, h = main_face
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    if SHOW_EYE_BOXES:
        for (x, y, w, h) in face_parts["eyes"]:
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if SHOW_MOUTH_BOXES and ENABLE_MOUTH_DETECTION:
        for (x, y, w, h) in face_parts["mouth"]:
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

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

    paths = get_project_paths()
    ensure_output_directories(paths)

    output_files = get_output_file_paths(
        paths,
        output_file_prefix=output_file_prefix,
    )
    reset_text_file(output_files["run_log_path"])

    log_message("=== ZAO Assignment 06 ===", output_files["run_log_path"])
    log_message(f"Run name: {run_name}", output_files["run_log_path"])
    log_message("Project started.", output_files["run_log_path"])
    log_message("", output_files["run_log_path"])

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

    capture = open_video(paths["video_path"])

    frame_count = 0
    frame_results = []
    previous_face = None
    previous_eye_state = None

    log_message("Video opened successfully.", output_files["run_log_path"])
    log_message("Reading frames...", output_files["run_log_path"])

    if show_preview:
        log_message(
            "Live preview enabled. Press 'q' or ESC to stop preview/run.",
            output_files["run_log_path"]
        )
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

    log_message("", output_files["run_log_path"])

    try:
        while True:
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
            classification_time_ms = (classification_end - classification_start) * 1000.0

            frame_processing_end = perf_counter()
            total_frame_time_ms = (frame_processing_end - frame_processing_start) * 1000.0

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

            if show_preview and (frame_count % PREVIEW_EVERY_N_FRAMES == 0):
                preview_frame = build_preview_frame(
                    frame,
                    face_boxes,
                    main_face,
                    face_parts,
                    eye_state
                )

                key = show_preview_frame(preview_frame)

                if key == ord("q") or key == 27:
                    log_message("Preview interrupted by user.", output_files["run_log_path"])
                    break

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
        capture.release()
        cv2.destroyAllWindows()

    log_message("", output_files["run_log_path"])
    log_message("Video reading finished.", output_files["run_log_path"])
    log_message(f"Total frames processed: {frame_count}", output_files["run_log_path"])
    log_message(f"Stored frame results:   {len(frame_results)}", output_files["run_log_path"])
    log_message("", output_files["run_log_path"])

    evaluation_summary = evaluate_results(
        frame_results,
        paths["ground_truth_path"]
    )

    if print_summary:
        print_evaluation_summary(evaluation_summary)

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

    log_message(
        f"Evaluation report saved to: {output_files['evaluation_report_path']}",
        output_files["run_log_path"]
    )
    log_message(
        f"Run log saved to: {output_files['run_log_path']}",
        output_files["run_log_path"]
    )

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

    run_final_pipeline(
        preprocessing_config=None,
        lbp_config=None,
        classifier_config=None,
        run_name="manual_default_run",
        show_preview=SHOW_TEST_VIDEO,
        output_file_prefix=None,
        print_summary=True,
    )


if __name__ == "__main__":
    main()