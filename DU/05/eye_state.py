"""
eye_state.py

Stage 5:
- eye-state classification module
- simple classical open/closed decision
- works on eye boxes already localized by detectors.py
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Basic eye-state parameters
# ---------------------------------------------------------------------

EYE_ROI_PADDING_RATIO = 0.08
EYE_GAUSSIAN_KERNEL_SIZE = (5, 5)

DARK_RATIO_REFERENCE = 0.22
CONTRAST_REFERENCE = 0.20
ASPECT_RATIO_REFERENCE = 0.28

OPEN_SCORE_THRESHOLD = 0.55


# ---------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------

def _clamp_box_to_frame(gray_frame, box):
    """
    Clamp one box to valid frame coordinates.

    Returned format:
    (x1, y1, x2, y2)
    """

    frame_height, frame_width = gray_frame.shape[:2]

    x, y, w, h = box

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + w)
    y2 = min(frame_height, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def extract_eye_roi(gray_frame, eye_box, padding_ratio=EYE_ROI_PADDING_RATIO):
    """
    Extract one eye ROI from the grayscale frame.

    A small padding is added around the detected eye box so the ROI keeps
    a bit more useful context.
    """

    x, y, w, h = eye_box

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    padded_box = (
        x - pad_x,
        y - pad_y,
        w + 2 * pad_x,
        h + 2 * pad_y
    )

    clamped = _clamp_box_to_frame(gray_frame, padded_box)

    if clamped is None:
        return None

    x1, y1, x2, y2 = clamped

    return gray_frame[y1:y2, x1:x2]


def preprocess_eye_roi(eye_roi):
    """
    Preprocess one eye ROI for simple open/closed analysis.

    Current steps:
    - histogram equalization
    - light Gaussian blur
    """

    equalized = cv2.equalizeHist(eye_roi)
    blurred = cv2.GaussianBlur(equalized, EYE_GAUSSIAN_KERNEL_SIZE, 0)

    return blurred


def compute_eye_features(preprocessed_eye_roi, eye_box):
    """
    Compute a few simple features for one eye ROI.

    Current features:
    - dark pixel ratio after Otsu thresholding
    - normalized contrast estimate
    - eye-box aspect ratio (height / width)

    These are simple heuristic features suitable for a school-lab solution.
    """

    if preprocessed_eye_roi is None or preprocessed_eye_roi.size == 0:
        return None

    _, binary_inv = cv2.threshold(
        preprocessed_eye_roi,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    dark_ratio = cv2.countNonZero(binary_inv) / binary_inv.size

    contrast = float(np.std(preprocessed_eye_roi)) / 255.0

    _, _, w, h = eye_box
    aspect_ratio = h / w if w > 0 else 0.0

    return {
        "dark_ratio": dark_ratio,
        "contrast": contrast,
        "aspect_ratio": aspect_ratio
    }


def compute_eye_open_score(features):
    """
    Convert simple eye features into one normalized openness score.

    Each feature is normalized against a reference value and clipped to 1.0.
    The final score is a weighted average.
    """

    if features is None:
        return 0.0

    dark_score = min(features["dark_ratio"] / DARK_RATIO_REFERENCE, 1.0)
    contrast_score = min(features["contrast"] / CONTRAST_REFERENCE, 1.0)
    aspect_score = min(features["aspect_ratio"] / ASPECT_RATIO_REFERENCE, 1.0)

    open_score = (
        0.50 * dark_score +
        0.30 * contrast_score +
        0.20 * aspect_score
    )

    return open_score


def classify_single_eye(gray_frame, eye_box):
    """
    Classify one detected eye box as open or closed.

    Returned value:
    - label: "open" or "closed"
    - score: normalized openness score
    """

    eye_roi = extract_eye_roi(gray_frame, eye_box)

    if eye_roi is None or eye_roi.size == 0:
        return "closed", 0.0

    preprocessed_eye_roi = preprocess_eye_roi(eye_roi)
    features = compute_eye_features(preprocessed_eye_roi, eye_box)
    open_score = compute_eye_open_score(features)

    label = "open" if open_score >= OPEN_SCORE_THRESHOLD else "closed"

    return label, open_score


# ---------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------

def classify_eye_state(gray_frame, face_parts):
    """
    Classify the overall eye state for one frame.

    Current behavior:
    - if no eyes are detected, return "closed"
    - otherwise classify each detected eye separately
    - return "open" if the average openness score is high enough
      or if most detected eyes are classified as open
    """

    eye_boxes = face_parts.get("eyes", [])

    if not eye_boxes:
        return "closed"

    eye_labels = []
    eye_scores = []

    for eye_box in eye_boxes:
        label, score = classify_single_eye(gray_frame, eye_box)
        eye_labels.append(label)
        eye_scores.append(score)

    mean_score = sum(eye_scores) / len(eye_scores)
    open_count = eye_labels.count("open")

    if mean_score >= OPEN_SCORE_THRESHOLD or open_count >= 1:
        return "open"

    return "closed"