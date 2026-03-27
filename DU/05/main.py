"""
main.py

This module is the main entry point of the project.

Its responsibilities are:
- building the project path configuration,
- opening the input video,
- running the frame-by-frame processing loop,
- calling localization and classification modules,
- optionally displaying a live annotated preview,
- storing per-frame results,
- invoking the final evaluation step,
- saving the run log and the evaluation report.
"""

from pathlib import Path
from time import perf_counter
from datetime import datetime
import cv2

from detectors import (
    load_cascades,
    detect_faces,
    select_main_face,
    detect_face_parts,
)

from eye_state import classify_eye_state

from evaluation import (
    evaluate_results,
    print_evaluation_summary,
    save_evaluation_report,
)


# ---------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------
#
# These options define the main operational behavior of the program.
# They are intentionally placed near the top of the file so that the program
# configuration is immediately visible and easy to adjust.
# ---------------------------------------------------------------------

ENABLE_MOUTH_DETECTION = True
FACE_DETECTION_DOWNSCALE_FACTOR = 0.75


# ---------------------------------------------------------------------
# Live-preview configuration
# ---------------------------------------------------------------------
#
# The preview is optional and intended mainly for visual inspection of the
# detector behavior during development or demonstration.
#
# The drawing options are separated so that the displayed overlay can be
# tailored without changing the processing logic.
# ---------------------------------------------------------------------

SHOW_TEST_VIDEO = True
SHOW_FACE_BOXES = True
SHOW_EYE_BOXES = True
SHOW_MOUTH_BOXES = True
SHOW_EYE_STATE_TEXT = True

PREVIEW_EVERY_N_FRAMES = 1
PREVIEW_WINDOW_NAME = "ZAO Assignment 05 - Preview"


# ---------------------------------------------------------------------
# Project path configuration
# ---------------------------------------------------------------------

def get_project_paths():
    """
    Build and return all relevant project paths.

    Centralizing path construction in one function keeps file-system knowledge
    out of the processing logic and makes the rest of the program independent
    of the physical directory layout.
    """

    project_root = Path(__file__).resolve().parent

    input_dir = project_root / "input"
    output_dir = project_root / "output"

    input_archives_dir = input_dir / "archives"
    input_cascades_dir = input_dir / "cascades"
    input_ground_truth_dir = input_dir / "ground_truth"
    input_video_dir = input_dir / "video"

    face_cascades_dir = input_cascades_dir / "face"
    eye_cascades_dir = input_cascades_dir / "eye"
    mouth_cascades_dir = input_cascades_dir / "mouth"

    return {
        "project_root": project_root,

        "input_dir": input_dir,
        "output_dir": output_dir,

        "input_archives_dir": input_archives_dir,
        "input_cascades_dir": input_cascades_dir,
        "input_ground_truth_dir": input_ground_truth_dir,
        "input_video_dir": input_video_dir,

        "face_cascades_dir": face_cascades_dir,
        "eye_cascades_dir": eye_cascades_dir,
        "mouth_cascades_dir": mouth_cascades_dir,

        "video_path": input_video_dir / "fusek_face_car_01.avi",

        "face_cascade_frontal_path": face_cascades_dir / "haarcascade_frontalface_default.xml",
        "face_cascade_profile_path": face_cascades_dir / "haarcascade_profileface.xml",

        "eye_cascade_path": eye_cascades_dir / "eye_cascade_fusek.xml",
        "mouth_cascade_path": mouth_cascades_dir / "haarcascade_smile.xml",

        "ground_truth_path": input_ground_truth_dir / "eye-state.txt",

        "output_annotated_video_dir": output_dir / "annotated_video",
        "output_frames_dir": output_dir / "frames",
        "output_logs_dir": output_dir / "logs",
        "output_reports_dir": output_dir / "reports",
    }


# ---------------------------------------------------------------------
# Output-directory and text-output helpers
# ---------------------------------------------------------------------

def ensure_output_directories(paths):
    """
    Ensure that all output directories required by the program exist.

    The current project saves:
    - logs,
    - reports,
    - and potentially other output artifacts in predefined folders.

    Existing directories are left unchanged.
    """

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_annotated_video_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_frames_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_reports_dir"].mkdir(parents=True, exist_ok=True)


def get_output_file_paths(paths):
    """
    Build the output file paths for the run log and evaluation report.

    Fixed file names are used so that each new run overwrites the previous
    final output in a predictable location.
    """

    return {
        "run_log_path": paths["output_logs_dir"] / "run_log.txt",
        "evaluation_report_path": paths["output_reports_dir"] / "evaluation_report.txt",
    }


def reset_text_file(file_path):
    """
    Create or clear a text file.

    This is used at the start of the run so that log and report files always
    correspond to the current execution only.
    """

    with open(file_path, "w", encoding="utf-8"):
        pass


def log_message(message, log_path=None, print_to_console=True):
    """
    Print one message and optionally append it to the run log.

    When a log path is supplied, the message is stored together with a simple
    timestamp so the execution flow can be inspected later.
    """

    if print_to_console:
        print(message)

    if log_path is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")


# ---------------------------------------------------------------------
# Video and preprocessing helpers
# ---------------------------------------------------------------------

def open_video(video_path):
    """
    Open the input video and return an OpenCV VideoCapture object.

    Failure to open the video is treated as a hard error because the full
    processing pipeline depends on sequential frame access.
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    return capture


def convert_to_grayscale(frame):
    """
    Convert a BGR video frame to grayscale.

    Grayscale form is used by the detector pipeline because Haar cascades and
    the subsequent eye-state analysis do not require color information.
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
    localization_time_ms
):
    """
    Store one structured result record for the current frame.

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
    }

    results.append(frame_result)


# ---------------------------------------------------------------------
# Live-preview helpers
# ---------------------------------------------------------------------

def build_preview_frame(frame, face_boxes, main_face, face_parts, eye_state):
    """
    Build an annotated preview frame from the current processing results.

    Drawing convention:
    - green  rectangles mark all detected face candidates,
    - red    rectangle marks the selected main face,
    - blue   rectangles mark detected eye boxes,
    - yellow rectangles mark detected mouth boxes,
    - white  text shows the current frame-level eye-state decision.
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

    This helper isolates the display operation so that the frame loop remains
    readable and the preview behavior is easy to change in one place.
    """

    cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)
    key = cv2.waitKey(1) & 0xFF
    return key


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main():
    """
    Execute the complete project pipeline.

    High-level execution sequence:
    - prepare paths and output directories,
    - reset the run log,
    - load cascade detectors,
    - open the input video,
    - process frames one by one,
    - optionally display a live preview,
    - evaluate the collected frame results,
    - save the report and log.
    """

    paths = get_project_paths()
    ensure_output_directories(paths)

    output_files = get_output_file_paths(paths)
    reset_text_file(output_files["run_log_path"])

    log_message("=== ZAO Assignment 05 ===", output_files["run_log_path"])
    log_message("Project started.", output_files["run_log_path"])
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
    log_message("Eye-state classifier initialized.", output_files["run_log_path"])
    log_message("Evaluation module initialized.", output_files["run_log_path"])
    log_message("", output_files["run_log_path"])

    capture = open_video(paths["video_path"])

    frame_count = 0
    frame_results = []
    previous_face = None

    log_message("Video opened successfully.", output_files["run_log_path"])
    log_message("Reading frames...", output_files["run_log_path"])

    if SHOW_TEST_VIDEO:
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

            # Convert the original BGR frame to grayscale because all current
            # detector and classifier stages operate on grayscale data.
            gray_frame = convert_to_grayscale(frame)

            # Measure only the localization stage here:
            # face detection and face-part detection.
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

            # Classify the eye state using the localized eye regions.
            eye_state = classify_eye_state(gray_frame, face_parts)

            # Store one structured result record for later evaluation.
            store_frame_result(
                frame_results,
                frame_count,
                face_boxes,
                main_face,
                face_parts,
                eye_state,
                localization_time_ms
            )

            # Preserve the selected face so the next frame can prefer temporal
            # continuity when choosing the main face candidate.
            previous_face = main_face

            # Optionally display an annotated live preview of the current frame.
            if SHOW_TEST_VIDEO and (frame_count % PREVIEW_EVERY_N_FRAMES == 0):
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

            # Periodically print and log a compact progress summary so long runs
            # remain observable without producing one line per frame.
            if frame_count % 30 == 0:
                log_message(
                    (
                        f"Processed frame: {frame_count}, "
                        f"detected faces: {len(face_boxes)}, "
                        f"detected eyes: {len(face_parts['eyes'])}, "
                        f"detected mouth: {len(face_parts['mouth'])}, "
                        f"eye_state: {eye_state}, "
                        f"localization_ms: {localization_time_ms:.3f}"
                    ),
                    output_files["run_log_path"]
                )

    finally:
        # Ensure the video resource and any OpenCV windows are released even if
        # the loop exits early or an exception occurs.
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

    print_evaluation_summary(evaluation_summary)

    save_evaluation_report(
        evaluation_summary,
        output_files["evaluation_report_path"],
        extra_lines=[
            "=== Run configuration ===",
            f"Project root: {paths['project_root']}",
            f"Video path: {paths['video_path']}",
            f"Ground truth path: {paths['ground_truth_path']}",
            f"Face detection downscale factor: {FACE_DETECTION_DOWNSCALE_FACTOR}",
            f"Mouth detection enabled: {ENABLE_MOUTH_DETECTION}",
            f"Live preview enabled: {SHOW_TEST_VIDEO}",
            f"Preview every N frames: {PREVIEW_EVERY_N_FRAMES}",
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


if __name__ == "__main__":
    main()