"""
main.py

Current stage:
- project entry point
- path definitions via get_project_paths()
- video opening
- frame reading loop
- grayscale conversion
- detector module connected
- real cascade loading
- frontal + profile face detection
- face-box merging/filtering
- main-face selection
- eye detection inside face ROI
- optional mouth/smile detection inside lower face ROI
- face detection on an optional downscaled grayscale frame
- eye-state classification connected
- evaluation connected
- temporary visual inspection of detected face / eye / mouth boxes
- clean shutdown
"""

from pathlib import Path
from time import perf_counter
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
)


# ---------------------------------------------------------------------
# Runtime settings
# ---------------------------------------------------------------------

ENABLE_MOUTH_DETECTION = False
FACE_DETECTION_DOWNSCALE_FACTOR = 0.75


# ---------------------------------------------------------------------
# Temporary debug settings
# ---------------------------------------------------------------------

TEMP_SHOW_FACE_BOXES = True
TEMP_SHOW_EYE_BOXES = True
TEMP_SHOW_MOUTH_BOXES = False
TEMP_SHOW_EYE_STATE_TEXT = True
TEMP_PREVIEW_EVERY_N_FRAMES = 5


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

def get_project_paths():
    """
    Build and return all important project paths.
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
# Basic helper functions
# ---------------------------------------------------------------------

def open_video(video_path):
    """Open input video and return cv2.VideoCapture object."""

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    return capture


def convert_to_grayscale(frame):
    """Convert one BGR frame to grayscale."""

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


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
    Store one simple per-frame record.
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
# Main program
# ---------------------------------------------------------------------

def main():
    """
    Main project entry point.
    """

    paths = get_project_paths()
    cascades = load_cascades(paths)

    print("=== ZAO Assignment 05 ===")
    print("Project started.")
    print()
    print(f"Project root: {paths['project_root']}")
    print(f"Video path:   {paths['video_path']}")
    print("Cascades loaded successfully.")
    print("Frontal face detector initialized.")
    print("Profile face detector initialized (fallback mode).")
    print(f"Face detection downscale factor: {FACE_DETECTION_DOWNSCALE_FACTOR}")
    print("Eye detector initialized.")

    if ENABLE_MOUTH_DETECTION:
        print("Mouth/smile detector initialized.")
    else:
        print("Mouth/smile detector disabled.")

    print("Eye-state classifier initialized.")
    print("Evaluation module initialized.")
    print()

    capture = open_video(paths["video_path"])

    frame_count = 0
    frame_results = []
    previous_face = None

    print("Video opened successfully.")
    print("Reading frames...")

    if (
        TEMP_SHOW_FACE_BOXES or
        TEMP_SHOW_EYE_BOXES or
        TEMP_SHOW_MOUTH_BOXES or
        TEMP_SHOW_EYE_STATE_TEXT
    ):
        print("Press 'q' or ESC in preview window to stop.")

    print()

    try:
        while True:
            ret, frame = capture.read()

            if not ret:
                break

            frame_count += 1

            # Basic preprocessing
            gray_frame = convert_to_grayscale(frame)

            # Localization timing: detection + face-part localization
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

            # Eye-state classification
            eye_state = classify_eye_state(gray_frame, face_parts)

            store_frame_result(
                frame_results,
                frame_count,
                face_boxes,
                main_face,
                face_parts,
                eye_state,
                localization_time_ms
            )

            previous_face = main_face

            # ---------------------------------------------------------
            # TEMP DEBUG: visual inspection of detected boxes
            #
            # Green rectangles  = all merged face boxes
            # Red rectangle     = selected main face
            # Blue rectangles   = detected eye boxes
            # Yellow rectangles = detected mouth boxes
            # White text        = current eye-state label
            # ---------------------------------------------------------
            if (
                TEMP_SHOW_FACE_BOXES or
                TEMP_SHOW_EYE_BOXES or
                TEMP_SHOW_MOUTH_BOXES or
                TEMP_SHOW_EYE_STATE_TEXT
            ) and (frame_count % TEMP_PREVIEW_EVERY_N_FRAMES == 0):

                debug_frame = frame.copy()

                if TEMP_SHOW_FACE_BOXES:
                    for (x, y, w, h) in face_boxes:
                        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if main_face is not None:
                        x, y, w, h = main_face
                        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

                if TEMP_SHOW_EYE_BOXES:
                    for (x, y, w, h) in face_parts["eyes"]:
                        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if TEMP_SHOW_MOUTH_BOXES and ENABLE_MOUTH_DETECTION:
                    for (x, y, w, h) in face_parts["mouth"]:
                        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                if TEMP_SHOW_EYE_STATE_TEXT:
                    cv2.putText(
                        debug_frame,
                        f"eye state: {eye_state}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                cv2.imshow("Temporary detection preview", debug_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            if frame_count % 30 == 0:
                print(
                    f"Processed frame: {frame_count}, "
                    f"detected faces: {len(face_boxes)}, "
                    f"detected eyes: {len(face_parts['eyes'])}, "
                    f"detected mouth: {len(face_parts['mouth'])}, "
                    f"eye_state: {eye_state}, "
                    f"localization_ms: {localization_time_ms:.3f}"
                )

    finally:
        capture.release()
        cv2.destroyAllWindows()

    print()
    print("Video reading finished.")
    print(f"Total frames processed: {frame_count}")
    print(f"Stored frame results:   {len(frame_results)}")
    print()

    evaluation_summary = evaluate_results(
        frame_results,
        paths["ground_truth_path"]
    )

    print_evaluation_summary(evaluation_summary)


if __name__ == "__main__":
    main()