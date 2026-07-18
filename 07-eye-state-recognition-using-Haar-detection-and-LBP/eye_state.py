# This module is the original heuristic eye-state decision core of the project.
# It predates the later LBP-based classifier, but it is still an important part
# of the final structure because:
# - it defines the project's original eye-ROI extraction logic,
# - it provides a fully self-contained eye-state classifier,
# - the newer LBP runtime module still reuses it as a fallback path.
#
# In the overall runtime flow, this module performs:
#
#     detected eye box
#         -> padded eye ROI extraction
#         -> fixed-size normalization
#         -> central analysis-band crop
#         -> threshold-based dark-structure analysis
#         -> circle-based iris/pupil evidence search
#         -> eye-level score combination
#         -> frame-level aggregation across detected eyes
#
# The core idea is to classify eyes heuristically without any trained model.
# It looks for visual cues expected from an open eye:
# - a plausible dark central blob,
# - good blob geometry,
# - and ideally circular iris/pupil evidence.
#
# Even though the project later moved to LBP-based learning, this module
# remains deeply connected to the rest of the codebase because it captures the
# original runtime assumptions and still supplies reusable helper logic.

"""
eye_state.py

This module implements frame-level eye-state classification.

Its role is to decide whether the detected eye region in a frame corresponds
to an open or closed eye state. The current implementation combines:
- threshold-based dark-region analysis,
- blob-shape analysis,
- circular iris or pupil evidence detected by Hough transform.

The module is intentionally self-contained so that the rest of the project
can interact with it through a single public function:
    classify_eye_state(gray_frame, face_parts)
"""

# OpenCV provides all image-processing operations in this module:
# - resizing,
# - histogram equalization,
# - blur filtering,
# - thresholding,
# - connected components,
# - Hough circle detection.
import cv2

# NumPy is used mainly for:
# - binary morphology kernels,
# - array checks,
# - circle output normalization from OpenCV.
import numpy as np


# ---------------------------------------------------------------------
# Eye-state parameter configuration
# ---------------------------------------------------------------------
#
# These constants are the entire tuning surface of the heuristic classifier.
# They control:
# - how the eye ROI is extracted and normalized,
# - what dark-ratio / blob geometry is considered plausible,
# - how Hough circle search is configured,
# - how eye-level evidence is turned into open/closed decisions,
# - and how eye-level results are aggregated into one frame-level result.
#
# Keeping all thresholds and ratios together here makes the rest of the module
# easier to read because the later functions can focus on structure and logic
# instead of embedding literal constants inside their bodies.
# ---------------------------------------------------------------------

# A small amount of padding is added around the raw detector eye box so the
# analysis uses slightly more context than the exact detector rectangle.
EYE_ROI_PADDING_RATIO = 0.08

# Every eye ROI is normalized into the same fixed geometry before analysis.
# This makes all later ratio-based thresholds operate in one stable coordinate
# system.
EYE_NORMALIZED_WIDTH = 80
EYE_NORMALIZED_HEIGHT = 40
EYE_ANALYSIS_TOP_RATIO = 0.15
EYE_ANALYSIS_BOTTOM_RATIO = 0.85

# Blur settings used during preprocessing and circle detection.
EYE_GAUSSIAN_KERNEL_SIZE = (5, 5)
EYE_MEDIAN_BLUR_SIZE = 5

# Expected amount of dark structure in the thresholded eye ROI.
# The triangular scoring later rewards values near the target and penalizes
# values that are too small or too large.
DARK_RATIO_TARGET = 0.20
DARK_RATIO_TOLERANCE = 0.22

# Expected blob size characteristics for a plausible pupil/iris-like dark blob.
BLOB_AREA_RATIO_TARGET = 0.07
BLOB_AREA_RATIO_TOLERANCE = 0.10
BLOB_AREA_RATIO_MIN = 0.01
BLOB_AREA_RATIO_MAX = 0.30

# Geometry constraints for a plausible central dark blob.
BLOB_CENTER_DISTANCE_MAX = 0.55
BLOB_ASPECT_RATIO_MAX = 6.00
BLOB_FILL_RATIO_MIN = 0.08

# Hough circle detector settings used to search for iris/pupil-like evidence.
HOUGH_DP = 1.2
HOUGH_MIN_DIST_RATIO = 0.35
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 7
HOUGH_MIN_RADIUS_RATIO = 0.08
HOUGH_MAX_RADIUS_RATIO = 0.30
HOUGH_CENTER_DISTANCE_MAX = 0.40

# Eye-level decision thresholds for the three positive routes:
# - direct circle-based open,
# - threshold-only open,
# - hybrid circle + threshold open.
CIRCLE_OPEN_SCORE_THRESHOLD = 0.37
THRESHOLD_ONLY_OPEN_THRESHOLD = 0.33
THRESHOLD_ONLY_OPEN_SCORE_MIN = 0.32

HYBRID_OPEN_SCORE_THRESHOLD = 0.30
HYBRID_THRESHOLD_SCORE_MIN = 0.24

# Frame-level aggregation thresholds used when combining one or more eye-level
# results into one final frame-level label.
SINGLE_EYE_FRAME_OPEN_THRESHOLD = 0.42
STRONG_EYE_OPEN_THRESHOLD = 0.46
FRAME_MEAN_OPEN_THRESHOLD = 0.40
FRAME_MIN_SUPPORT_THRESHOLD = 0.33

# Debug mode can be enabled to collect and periodically print running
# statistics about the classifier behavior.
DEBUG_EYE_STATE = False
DEBUG_PRINT_EVERY_EYES = 100


# ---------------------------------------------------------------------
# Debug bookkeeping
# ---------------------------------------------------------------------
#
# These counters are only used when DEBUG_EYE_STATE is enabled.
# They make it possible to inspect:
# - how often circle evidence is found,
# - how often a good blob is found,
# - how many eyes/frames are classified as open or closed,
# - which decision path is most often responsible for open-eye decisions,
# - and what the running averages of the main evidence scores look like.
#
# In the final submission setup, debug mode is disabled, so these counters
# remain present but inactive.
# ---------------------------------------------------------------------

DEBUG_COUNTERS = {
    "processed_eyes": 0,
    "processed_frames": 0,
    "frames_without_eyes": 0,
    "circle_found": 0,
    "good_blob_found": 0,
    "predicted_open_eyes": 0,
    "predicted_closed_eyes": 0,
    "open_from_circle": 0,
    "open_from_threshold_only": 0,
    "open_from_hybrid": 0,
    "predicted_open_frames": 0,
    "predicted_closed_frames": 0,
    "sum_open_score": 0.0,
    "sum_threshold_score": 0.0,
    "sum_dark_ratio": 0.0,
    "sum_blob_area_ratio": 0.0,
    "sum_circle_score": 0.0,
}


# ---------------------------------------------------------------------
# Internal geometric and utility helpers
# ---------------------------------------------------------------------

def _clamp_box_to_frame(gray_frame, box):
    """
    Clamp a bounding box to valid image coordinates.

    This prevents invalid array slicing when a detector box partially lies
    outside the image borders.

    The returned coordinates use corner form:
        (x1, y1, x2, y2)
    """

    # Read the frame size so the box can be clamped against valid image bounds.
    frame_height, frame_width = gray_frame.shape[:2]

    x, y, w, h = box

    # Convert the incoming (x, y, w, h) box into clamped corner coordinates.
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + w)
    y2 = min(frame_height, y + h)

    # If the clamped box collapses, it is not usable for slicing.
    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def _triangular_score(value, center, tolerance):
    """
    Compute a simple triangular score centered around an expected value.

    The score is:
    - highest at the specified center,
    - decreases linearly with distance from the center,
    - reaches zero outside the tolerance interval.

    This form is useful for properties such as dark ratio or blob size where
    neither very small nor very large values are desirable.
    """

    # Defensive guard: a non-positive tolerance cannot define a meaningful
    # triangular scoring interval.
    if tolerance <= 0:
        return 0.0

    distance = abs(value - center)

    # Outside the allowed tolerance band, the score drops to zero.
    if distance >= tolerance:
        return 0.0

    # Inside the band, the score decreases linearly with distance from center.
    return 1.0 - (distance / tolerance)


# ---------------------------------------------------------------------
# Optional debug-counter helpers
# ---------------------------------------------------------------------

def _update_eye_debug_counters(
    threshold_evidence,
    circle_evidence,
    threshold_score,
    open_score,
    label,
    decision_source
):
    """
    Update debug counters for one processed eye candidate.

    This function records:
    - how often each evidence type is observed,
    - how often each decision source is responsible for an open-eye result,
    - basic running averages of the relevant scores.

    The function performs no work when debug mode is disabled.
    """

    # In normal final execution this function exits immediately, so all debug
    # bookkeeping is effectively free when debug mode is off.
    if not DEBUG_EYE_STATE:
        return

    # Accumulate raw counts and running sums for later periodic reporting.
    DEBUG_COUNTERS["processed_eyes"] += 1
    DEBUG_COUNTERS["sum_open_score"] += open_score
    DEBUG_COUNTERS["sum_threshold_score"] += threshold_score
    DEBUG_COUNTERS["sum_dark_ratio"] += threshold_evidence["dark_ratio"]
    DEBUG_COUNTERS["sum_blob_area_ratio"] += threshold_evidence["blob_area_ratio"]
    DEBUG_COUNTERS["sum_circle_score"] += circle_evidence["circle_score"]

    if threshold_evidence["has_good_pupil_blob"]:
        DEBUG_COUNTERS["good_blob_found"] += 1

    if circle_evidence["circle_found"]:
        DEBUG_COUNTERS["circle_found"] += 1

    if label == "open":
        DEBUG_COUNTERS["predicted_open_eyes"] += 1
    else:
        DEBUG_COUNTERS["predicted_closed_eyes"] += 1

    # Keep track of which decision route produced an open-eye decision.
    if decision_source == "circle":
        DEBUG_COUNTERS["open_from_circle"] += 1
    elif decision_source == "threshold":
        DEBUG_COUNTERS["open_from_threshold_only"] += 1
    elif decision_source == "hybrid":
        DEBUG_COUNTERS["open_from_hybrid"] += 1

    # Print a periodic diagnostic snapshot every configured number of processed
    # eyes so the scoring behavior can be inspected during tuning.
    if DEBUG_COUNTERS["processed_eyes"] % DEBUG_PRINT_EVERY_EYES == 0:
        processed_eyes = DEBUG_COUNTERS["processed_eyes"]

        mean_open_score = DEBUG_COUNTERS["sum_open_score"] / processed_eyes
        mean_threshold_score = DEBUG_COUNTERS["sum_threshold_score"] / processed_eyes
        mean_dark_ratio = DEBUG_COUNTERS["sum_dark_ratio"] / processed_eyes
        mean_blob_area_ratio = DEBUG_COUNTERS["sum_blob_area_ratio"] / processed_eyes
        mean_circle_score = DEBUG_COUNTERS["sum_circle_score"] / processed_eyes

        print("[eye_state debug]")
        print(f"  processed eyes:           {processed_eyes}")
        print(f"  circle found count:       {DEBUG_COUNTERS['circle_found']}")
        print(f"  good blob count:          {DEBUG_COUNTERS['good_blob_found']}")
        print(f"  predicted open eyes:      {DEBUG_COUNTERS['predicted_open_eyes']}")
        print(f"  predicted closed eyes:    {DEBUG_COUNTERS['predicted_closed_eyes']}")
        print(f"  open from circle:         {DEBUG_COUNTERS['open_from_circle']}")
        print(f"  open from threshold only: {DEBUG_COUNTERS['open_from_threshold_only']}")
        print(f"  open from hybrid:         {DEBUG_COUNTERS['open_from_hybrid']}")
        print(f"  mean open score:          {mean_open_score:.3f}")
        print(f"  mean threshold score:     {mean_threshold_score:.3f}")
        print(f"  mean dark ratio:          {mean_dark_ratio:.3f}")
        print(f"  mean blob area ratio:     {mean_blob_area_ratio:.3f}")
        print(f"  mean circle score:        {mean_circle_score:.3f}")


def _update_frame_debug_counters(frame_label, eye_count):
    """
    Update debug counters at frame level.

    This function tracks how often frames are classified as open or closed and
    how often no eye boxes are available at all. It remains inactive when
    debug mode is disabled.
    """

    # Like the eye-level debug helper, this does nothing unless debug mode is
    # explicitly enabled.
    if not DEBUG_EYE_STATE:
        return

    DEBUG_COUNTERS["processed_frames"] += 1

    if eye_count == 0:
        DEBUG_COUNTERS["frames_without_eyes"] += 1

    if frame_label == "open":
        DEBUG_COUNTERS["predicted_open_frames"] += 1
    else:
        DEBUG_COUNTERS["predicted_closed_frames"] += 1

    # Periodically print frame-level aggregate statistics.
    if DEBUG_COUNTERS["processed_frames"] % DEBUG_PRINT_EVERY_EYES == 0:
        print("[eye_state frame debug]")
        print(f"  processed frames:         {DEBUG_COUNTERS['processed_frames']}")
        print(f"  frames without eyes:      {DEBUG_COUNTERS['frames_without_eyes']}")
        print(f"  predicted open frames:    {DEBUG_COUNTERS['predicted_open_frames']}")
        print(f"  predicted closed frames:  {DEBUG_COUNTERS['predicted_closed_frames']}")


# ---------------------------------------------------------------------
# ROI extraction and preprocessing
# ---------------------------------------------------------------------

def extract_eye_roi(gray_frame, eye_box, padding_ratio=EYE_ROI_PADDING_RATIO):
    """
    Extract the region of interest for one detected eye.

    A small padding is added around the raw eye box so the analysis includes
    a small amount of surrounding context instead of using an overly tight
    crop.

    The function returns a grayscale eye ROI or None if the box becomes
    invalid after clamping.
    """

    x, y, w, h = eye_box

    # Expand the detected box slightly so the ROI includes a little surrounding
    # context instead of a perfectly tight crop.
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    padded_box = (
        x - pad_x,
        y - pad_y,
        w + 2 * pad_x,
        h + 2 * pad_y
    )

    # Clamp the padded box to valid frame coordinates before slicing.
    clamped = _clamp_box_to_frame(gray_frame, padded_box)

    if clamped is None:
        return None

    x1, y1, x2, y2 = clamped

    return gray_frame[y1:y2, x1:x2]


def normalize_eye_roi(eye_roi):
    """
    Normalize the eye ROI to a fixed size.

    A fixed-size representation makes the later analysis stages more stable
    because all geometry- and ratio-based thresholds operate on a consistent
    coordinate system.
    """

    # Missing or empty eye ROIs are treated as invalid and propagate as None.
    if eye_roi is None or eye_roi.size == 0:
        return None

    # Normalize every eye ROI to the project's standard heuristic-analysis size.
    normalized = cv2.resize(
        eye_roi,
        (EYE_NORMALIZED_WIDTH, EYE_NORMALIZED_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    return normalized


def crop_eye_analysis_band(normalized_eye_roi):
    """
    Keep the central vertical band of the normalized eye ROI.

    This reduces the influence of:
    - eyebrows,
    - upper facial texture,
    - lower cheek texture.

    The retained band is intended to focus analysis on the region where
    pupil or iris evidence is most likely to appear.
    """

    # The central analysis band is used to suppress less relevant upper/lower
    # context around the eye itself.
    if normalized_eye_roi is None or normalized_eye_roi.size == 0:
        return None

    roi_height = normalized_eye_roi.shape[0]

    y1 = int(roi_height * EYE_ANALYSIS_TOP_RATIO)
    y2 = int(roi_height * EYE_ANALYSIS_BOTTOM_RATIO)

    if y2 <= y1:
        return None

    return normalized_eye_roi[y1:y2, :]


def preprocess_eye_roi(eye_roi):
    """
    Prepare an eye ROI for classification.

    The preprocessing sequence is:
    - normalize to fixed size,
    - crop the central analysis band,
    - equalize histogram,
    - apply Gaussian blur.

    Histogram equalization improves contrast robustness, while light smoothing
    reduces small local fluctuations that could disturb thresholding and
    circle detection.
    """

    # Step 1: normalize the raw ROI into the fixed heuristic-analysis size.
    normalized_eye_roi = normalize_eye_roi(eye_roi)

    if normalized_eye_roi is None:
        return None

    # Step 2: crop away upper/lower context and keep the central analysis band.
    analysis_band = crop_eye_analysis_band(normalized_eye_roi)

    if analysis_band is None or analysis_band.size == 0:
        return None

    # Step 3: improve contrast so dark structures become easier to separate.
    equalized = cv2.equalizeHist(analysis_band)

    # Step 4: lightly smooth the image so thresholding and Hough detection are
    # less sensitive to very small local fluctuations.
    blurred = cv2.GaussianBlur(equalized, EYE_GAUSSIAN_KERNEL_SIZE, 0)

    return blurred


# ---------------------------------------------------------------------
# Threshold-based evidence extraction
# ---------------------------------------------------------------------

def threshold_eye_roi(preprocessed_eye_roi):
    """
    Threshold the preprocessed eye ROI to isolate dark structures.

    The current implementation uses inverted Otsu thresholding so that dark
    regions become foreground. A small morphological opening then removes
    isolated noise fragments.

    The returned image is a binary mask suitable for connected-component
    analysis.
    """

    # The thresholding stage expects one valid preprocessed grayscale eye image.
    if preprocessed_eye_roi is None or preprocessed_eye_roi.size == 0:
        return None

    # Inverted Otsu thresholding makes dark eye structures become white
    # foreground in the binary mask.
    _, binary_inv = cv2.threshold(
        preprocessed_eye_roi,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # A small opening removes tiny isolated foreground fragments so connected
    # component analysis focuses on more meaningful blobs.
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)

    return cleaned


def compute_threshold_evidence(binary_eye_roi):
    """
    Compute threshold-based evidence from the binary eye mask.

    The function evaluates the strongest dark blob candidate using:
    - overall dark-pixel ratio,
    - blob area ratio,
    - blob distance from ROI center,
    - blob aspect ratio,
    - blob fill ratio.

    These measures are intended to distinguish plausible pupil or iris
    candidates from broad eyelid shadows or accidental dark texture.
    """

    # If thresholding produced nothing usable, return one explicit
    # "no positive evidence" structure so later scoring remains simple.
    if binary_eye_roi is None or binary_eye_roi.size == 0:
        return {
            "dark_ratio": 0.0,
            "blob_area_ratio": 0.0,
            "blob_distance": 1.0,
            "blob_aspect_ratio": 999.0,
            "blob_fill_ratio": 0.0,
            "has_good_pupil_blob": False,
        }

    roi_height, roi_width = binary_eye_roi.shape[:2]
    roi_area = binary_eye_roi.size

    # Overall dark-ratio gives a coarse measure of how much dark structure the
    # eye ROI contains after thresholding.
    dark_ratio = cv2.countNonZero(binary_eye_roi) / roi_area

    # Connected components are used to inspect individual dark blob candidates.
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary_eye_roi,
        connectivity=8
    )

    roi_center_x = roi_width / 2.0
    roi_center_y = roi_height / 2.0
    max_center_distance = max(1.0, (roi_width ** 2 + roi_height ** 2) ** 0.5 / 2.0)

    # These variables track the single best blob candidate encountered so far.
    best_blob_area_ratio = 0.0
    best_blob_distance = 1.0
    best_blob_aspect_ratio = 999.0
    best_blob_fill_ratio = 0.0
    best_blob_score = -1.0

    # Skip label 0 because it is the background component.
    for label_index in range(1, num_labels):
        blob_area = stats[label_index, cv2.CC_STAT_AREA]
        blob_area_ratio = blob_area / roi_area

        blob_width = max(1, stats[label_index, cv2.CC_STAT_WIDTH])
        blob_height = max(1, stats[label_index, cv2.CC_STAT_HEIGHT])

        # Aspect ratio and fill ratio help distinguish compact central blobs from
        # broad, thin, or sparse dark regions.
        blob_aspect_ratio = max(blob_width, blob_height) / max(1, min(blob_width, blob_height))
        blob_fill_ratio = blob_area / max(1, blob_width * blob_height)

        # Distance from ROI center matters because a plausible pupil/iris-like
        # blob should usually not be far from the center of the normalized eye ROI.
        center_x, center_y = centroids[label_index]
        blob_distance = (
            ((center_x - roi_center_x) ** 2 + (center_y - roi_center_y) ** 2) ** 0.5
        ) / max_center_distance

        # Convert raw geometric properties into soft scores.
        area_score = _triangular_score(
            blob_area_ratio,
            BLOB_AREA_RATIO_TARGET,
            BLOB_AREA_RATIO_TOLERANCE
        )
        center_score = max(0.0, 1.0 - blob_distance / max(BLOB_CENTER_DISTANCE_MAX, 1e-6))
        aspect_score = max(
            0.0,
            1.0 - max(0.0, blob_aspect_ratio - 1.0) / max(BLOB_ASPECT_RATIO_MAX - 1.0, 1e-6)
        )
        fill_score = min(blob_fill_ratio / max(BLOB_FILL_RATIO_MIN, 1e-6), 1.0)

        # Combine the component-level cues into one internal blob score used only
        # to decide which blob is the best candidate.
        blob_score = (
            0.30 * area_score +
            0.35 * center_score +
            0.20 * aspect_score +
            0.15 * fill_score
        )

        # Keep only the strongest blob candidate for later threshold scoring.
        if blob_score > best_blob_score:
            best_blob_score = blob_score
            best_blob_area_ratio = blob_area_ratio
            best_blob_distance = blob_distance
            best_blob_aspect_ratio = blob_aspect_ratio
            best_blob_fill_ratio = blob_fill_ratio

    # This boolean summarizes whether the best blob satisfies the main hard
    # plausibility checks expected from a pupil/iris-like dark structure.
    has_good_pupil_blob = (
        BLOB_AREA_RATIO_MIN <= best_blob_area_ratio <= BLOB_AREA_RATIO_MAX and
        best_blob_distance <= BLOB_CENTER_DISTANCE_MAX and
        best_blob_aspect_ratio <= BLOB_ASPECT_RATIO_MAX and
        best_blob_fill_ratio >= BLOB_FILL_RATIO_MIN
    )

    return {
        "dark_ratio": dark_ratio,
        "blob_area_ratio": best_blob_area_ratio,
        "blob_distance": best_blob_distance,
        "blob_aspect_ratio": best_blob_aspect_ratio,
        "blob_fill_ratio": best_blob_fill_ratio,
        "has_good_pupil_blob": has_good_pupil_blob,
    }


# ---------------------------------------------------------------------
# Circle-based evidence extraction
# ---------------------------------------------------------------------

def detect_iris_circle(preprocessed_eye_roi):
    """
    Detect iris or pupil-like circular evidence using HoughCircles.

    The function searches for circular structures in the preprocessed eye ROI
    and scores them according to:
    - proximity to the ROI center,
    - plausible radius.

    The result is used as the strongest single positive cue for an open eye.
    """

    # If there is no valid preprocessed eye ROI, there can be no circle-based
    # positive evidence.
    if preprocessed_eye_roi is None or preprocessed_eye_roi.size == 0:
        return {
            "circle_found": False,
            "circle_score": 0.0,
        }

    roi_height, roi_width = preprocessed_eye_roi.shape[:2]

    # Median blur is used specifically before HoughCircles because it can help
    # suppress small fluctuations while preserving larger circular structures.
    median_blurred = cv2.medianBlur(preprocessed_eye_roi, EYE_MEDIAN_BLUR_SIZE)

    # Convert ratio-based Hough settings into ROI-size-dependent pixel values.
    min_dist = max(1, int(roi_width * HOUGH_MIN_DIST_RATIO))
    min_radius = max(1, int(min(roi_width, roi_height) * HOUGH_MIN_RADIUS_RATIO))
    max_radius = max(min_radius + 1, int(min(roi_width, roi_height) * HOUGH_MAX_RADIUS_RATIO))

    circles = cv2.HoughCircles(
        median_blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=min_dist,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # No detected circles means no circle-based evidence.
    if circles is None:
        return {
            "circle_found": False,
            "circle_score": 0.0,
        }

    circles = np.round(circles[0, :]).astype(int)

    roi_center_x = roi_width / 2.0
    roi_center_y = roi_height / 2.0
    max_center_distance = max(1.0, (roi_width ** 2 + roi_height ** 2) ** 0.5 / 2.0)

    best_score = 0.0
    circle_found = False

    # Evaluate all detected circle candidates and keep the best score.
    for circle_x, circle_y, circle_r in circles:
        center_distance = (
            ((circle_x - roi_center_x) ** 2 + (circle_y - roi_center_y) ** 2) ** 0.5
        ) / max_center_distance

        radius_ratio = circle_r / max(1.0, min(roi_width, roi_height))
        radius_score = min(radius_ratio / max(HOUGH_MAX_RADIUS_RATIO, 1e-6), 1.0)
        center_score = max(0.0, 1.0 - center_distance / max(HOUGH_CENTER_DISTANCE_MAX, 1e-6))

        # The candidate circle score prefers circles that are reasonably central
        # and of plausible size.
        circle_score = 0.70 * center_score + 0.30 * radius_score
        best_score = max(best_score, circle_score)

        # A circle counts as a plausible open-eye cue only if it is not too far
        # from the center of the eye ROI.
        if center_distance <= HOUGH_CENTER_DISTANCE_MAX:
            circle_found = True

    return {
        "circle_found": circle_found,
        "circle_score": best_score,
    }


# ---------------------------------------------------------------------
# Score combination
# ---------------------------------------------------------------------

def compute_threshold_score(threshold_evidence):
    """
    Convert threshold-based evidence into one normalized threshold score.

    The score combines:
    - dark-ratio suitability,
    - blob-size suitability,
    - blob centrality,
    - blob compactness,
    - blob fill quality.

    A penalty is applied when the candidate blob does not satisfy the
    basic "good pupil blob" condition.
    """

    # Convert the raw threshold-evidence measurements into soft component scores.
    dark_ratio_score = _triangular_score(
        threshold_evidence["dark_ratio"],
        DARK_RATIO_TARGET,
        DARK_RATIO_TOLERANCE
    )

    blob_area_score = _triangular_score(
        threshold_evidence["blob_area_ratio"],
        BLOB_AREA_RATIO_TARGET,
        BLOB_AREA_RATIO_TOLERANCE
    )

    blob_center_score = max(
        0.0,
        1.0 - threshold_evidence["blob_distance"] / max(BLOB_CENTER_DISTANCE_MAX, 1e-6)
    )

    blob_aspect_score = max(
        0.0,
        1.0 - max(0.0, threshold_evidence["blob_aspect_ratio"] - 1.0)
        / max(BLOB_ASPECT_RATIO_MAX - 1.0, 1e-6)
    )

    blob_fill_score = min(
        threshold_evidence["blob_fill_ratio"] / max(BLOB_FILL_RATIO_MIN, 1e-6),
        1.0
    )

    # Combine the threshold-side cues into one normalized threshold score.
    threshold_score = (
        0.20 * dark_ratio_score +
        0.25 * blob_area_score +
        0.25 * blob_center_score +
        0.15 * blob_aspect_score +
        0.15 * blob_fill_score
    )

    # If the best blob still fails the hard plausibility checks, apply a penalty
    # so threshold-only evidence becomes less convincing.
    if not threshold_evidence["has_good_pupil_blob"]:
        threshold_score *= 0.78

    return threshold_score


def compute_eye_open_score(threshold_score, circle_evidence):
    """
    Combine threshold-based and circle-based evidence into one eye-open score.

    When a plausible circle is found, circle evidence is weighted more strongly.
    Otherwise the decision relies more heavily on threshold-based evidence.
    """

    circle_score = circle_evidence["circle_score"]

    # Circle evidence is treated as the strongest positive cue for openness.
    # When a plausible circle exists, it gets the larger weight.
    if circle_evidence["circle_found"]:
        open_score = 0.52 * circle_score + 0.48 * threshold_score
    else:
        # Without a plausible circle, threshold evidence dominates the final
        # open-score.
        open_score = 0.18 * circle_score + 0.82 * threshold_score

    return open_score


# ---------------------------------------------------------------------
# Eye-level classification
# ---------------------------------------------------------------------

def classify_single_eye(gray_frame, eye_box):
    """
    Classify one detected eye region as open or closed.

    The decision is made by evaluating three possible positive routes:
    - circle-based open,
    - hybrid circle + threshold open,
    - threshold-only open.

    The function returns:
    - the predicted label,
    - the final open score.
    """

    # Step 1: extract the padded eye ROI from the frame.
    eye_roi = extract_eye_roi(gray_frame, eye_box)

    # If ROI extraction fails, return a closed-eye default with empty evidence.
    if eye_roi is None or eye_roi.size == 0:
        threshold_evidence = {
            "dark_ratio": 0.0,
            "blob_area_ratio": 0.0,
            "blob_distance": 1.0,
            "blob_aspect_ratio": 999.0,
            "blob_fill_ratio": 0.0,
            "has_good_pupil_blob": False,
        }
        circle_evidence = {
            "circle_found": False,
            "circle_score": 0.0,
        }
        _update_eye_debug_counters(
            threshold_evidence,
            circle_evidence,
            0.0,
            0.0,
            "closed",
            "none"
        )
        return "closed", 0.0

    # Step 2: preprocess the eye ROI into the normalized heuristic-analysis
    # representation.
    preprocessed_eye_roi = preprocess_eye_roi(eye_roi)

    # If preprocessing fails, again return a closed-eye default with empty
    # evidence.
    if preprocessed_eye_roi is None or preprocessed_eye_roi.size == 0:
        threshold_evidence = {
            "dark_ratio": 0.0,
            "blob_area_ratio": 0.0,
            "blob_distance": 1.0,
            "blob_aspect_ratio": 999.0,
            "blob_fill_ratio": 0.0,
            "has_good_pupil_blob": False,
        }
        circle_evidence = {
            "circle_found": False,
            "circle_score": 0.0,
        }
        _update_eye_debug_counters(
            threshold_evidence,
            circle_evidence,
            0.0,
            0.0,
            "closed",
            "none"
        )
        return "closed", 0.0

    # Step 3: derive the two main evidence sources:
    # - threshold-side blob evidence
    # - circle-side iris/pupil evidence
    binary_eye_roi = threshold_eye_roi(preprocessed_eye_roi)
    threshold_evidence = compute_threshold_evidence(binary_eye_roi)
    circle_evidence = detect_iris_circle(preprocessed_eye_roi)

    # Step 4: convert those evidence structures into numeric scores.
    threshold_score = compute_threshold_score(threshold_evidence)
    open_score = compute_eye_open_score(threshold_score, circle_evidence)

    # Default assumption is "closed" unless one of the positive decision routes
    # is satisfied.
    decision_source = "none"
    label = "closed"

    # Positive route 1:
    # if a plausible circle is found and the final open-score is strong enough,
    # classify as open directly from the circle cue.
    if circle_evidence["circle_found"] and open_score >= CIRCLE_OPEN_SCORE_THRESHOLD:
        label = "open"
        decision_source = "circle"

    # Positive route 2:
    # require both circle evidence and a good threshold-side blob with enough
    # support, then accept the combined score.
    elif (
        circle_evidence["circle_found"] and
        threshold_evidence["has_good_pupil_blob"] and
        threshold_score >= HYBRID_THRESHOLD_SCORE_MIN and
        open_score >= HYBRID_OPEN_SCORE_THRESHOLD
    ):
        label = "open"
        decision_source = "hybrid"

    # Positive route 3:
    # when circle evidence is not convincing enough, allow threshold-only
    # openness if the blob is plausible and the threshold/open scores are strong.
    elif (
        threshold_evidence["has_good_pupil_blob"] and
        threshold_score >= THRESHOLD_ONLY_OPEN_THRESHOLD and
        open_score >= THRESHOLD_ONLY_OPEN_SCORE_MIN
    ):
        label = "open"
        decision_source = "threshold"

    # Record the outcome for optional debug statistics.
    _update_eye_debug_counters(
        threshold_evidence,
        circle_evidence,
        threshold_score,
        open_score,
        label,
        decision_source
    )

    return label, open_score


# ---------------------------------------------------------------------
# Frame-level public interface
# ---------------------------------------------------------------------

def classify_eye_state(gray_frame, face_parts):
    """
    Classify the overall eye state for one video frame.

    The function uses the currently detected eye boxes from the face-parts
    dictionary and combines the single-eye results into one frame-level label.

    The current aggregation favors an open result when:
    - both eyes support openness sufficiently,
    - or one eye is strongly open and the average support is acceptable,
    - or the average score is globally strong.

    If no eyes are available, the frame is classified as closed.
    """

    # Read the currently localized eye boxes from the shared face-parts
    # structure.
    eye_boxes = face_parts.get("eyes", [])

    # No detected eyes means no positive evidence, so the frame defaults to
    # closed.
    if not eye_boxes:
        _update_frame_debug_counters("closed", 0)
        return "closed"

    eye_labels = []
    eye_scores = []

    # Classify each detected eye independently and collect both the hard label
    # and the soft open-score.
    for eye_box in eye_boxes:
        label, score = classify_single_eye(gray_frame, eye_box)
        eye_labels.append(label)
        eye_scores.append(score)

    eye_count = len(eye_labels)
    open_count = eye_labels.count("open")
    max_score = max(eye_scores)
    mean_score = sum(eye_scores) / eye_count

    # Special handling for a single detected eye:
    # a direct open label is enough, but a sufficiently high score can also
    # promote the frame to open even if the hard label itself was not open.
    if eye_count == 1:
        frame_label = "open" if (eye_labels[0] == "open" or eye_scores[0] >= SINGLE_EYE_FRAME_OPEN_THRESHOLD) else "closed"
        _update_frame_debug_counters(frame_label, eye_count)
        return frame_label

    # Multi-eye aggregation:
    # 1. if two or more eyes vote open and mean support is good -> open
    if open_count >= 2 and mean_score >= FRAME_MIN_SUPPORT_THRESHOLD:
        frame_label = "open"

    # 2. if at least one eye is strongly open and mean support is acceptable -> open
    elif open_count >= 1 and max_score >= STRONG_EYE_OPEN_THRESHOLD and mean_score >= FRAME_MIN_SUPPORT_THRESHOLD:
        frame_label = "open"

    # 3. if the mean support across eyes is globally strong -> open
    elif mean_score >= FRAME_MEAN_OPEN_THRESHOLD:
        frame_label = "open"

    # Otherwise the frame is treated as closed.
    else:
        frame_label = "closed"

    _update_frame_debug_counters(frame_label, eye_count)

    return frame_label