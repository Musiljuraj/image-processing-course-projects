# This module is the localization layer of the whole project.
# It sits before both the heuristic and LBP-based eye-state classifiers and is
# responsible for finding the image regions those later classifiers will work on.
#
# In the runtime flow, this module performs:
#
#     grayscale frame
#         -> face detection
#         -> selection of one main face
#         -> face ROI extraction
#         -> constrained eye detection inside upper face region
#         -> constrained mouth detection inside lower face region
#         -> stable face_parts structure for the rest of the pipeline
#
# A key idea in this module is separation of concerns:
# - low-level geometry helpers stay private,
# - public detector functions expose a small, stable interface,
# - the rest of the project only needs to ask for:
#       load_cascades(...)
#       detect_faces(...)
#       select_main_face(...)
#       detect_face_parts(...)
#
# That keeps the higher-level pipeline modules focused on control flow,
# training, runtime classification, and evaluation rather than raw detector
# mechanics.

"""
detectors.py

This module contains all image-localization routines used by the project.

Its responsibilities are:
- loading Haar cascade classifiers from XML files,
- detecting faces in a grayscale frame,
- selecting one main face for subsequent processing,
- detecting eyes and mouth inside the selected face region.

The module separates low-level geometric helper functions from the public
detector interface so that the main program can stay simple and focused on
pipeline control.
"""

# math is only needed for geometric distance computation between box centers.
import math

# OpenCV provides all detector functionality in this module:
# - Haar cascade loading,
# - cascade-based face / eye / mouth detection,
# - frame flipping,
# - optional frame downscaling.
import cv2


# ---------------------------------------------------------------------
# Detector parameter configuration
# ---------------------------------------------------------------------
#
# These constants define the tuning surface of the localization layer.
# They control:
# - face detector sensitivity,
# - overlap filtering,
# - temporal continuity behavior for selecting the main face,
# - where eyes are searched inside the face ROI,
# - where the mouth is searched inside the face ROI,
# - and how many final eye/mouth boxes are preserved.
#
# Grouping these values here keeps later function bodies readable and makes
# detector tuning possible without modifying the overall detector logic.
# ---------------------------------------------------------------------

# Face cascade settings used for both frontal and profile detection.
FACE_SCALE_FACTOR = 1.15
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (30, 30)

# Post-processing / temporal-selection thresholds for face boxes.
FACE_MERGE_IOU_THRESHOLD = 0.30
FACE_CONTINUITY_IOU_THRESHOLD = 0.10
FACE_CONTINUITY_DISTANCE_FACTOR = 0.60

# Eye cascade settings and post-filtering thresholds.
EYE_SCALE_FACTOR = 1.15
EYE_MIN_NEIGHBORS = 5
EYE_MERGE_IOU_THRESHOLD = 0.20

# Search-area ratios for eyes inside the selected face ROI.
# The eye search is intentionally restricted to the upper face and slightly
# reduced horizontally to suppress obvious false positives near the edges.
EYE_SEARCH_LEFT_MARGIN_RATIO = 0.05
EYE_SEARCH_RIGHT_MARGIN_RATIO = 0.05
EYE_SEARCH_TOP_RATIO = 0.55

# Plausible size ranges for detected eye boxes relative to the face size.
EYE_MIN_WIDTH_RATIO = 0.12
EYE_MIN_HEIGHT_RATIO = 0.08
EYE_MAX_WIDTH_RATIO = 0.45
EYE_MAX_HEIGHT_RATIO = 0.30

# Final number of eye boxes preserved after filtering and selection.
MAX_EYE_COUNT = 2

# Mouth cascade settings and post-filtering thresholds.
MOUTH_SCALE_FACTOR = 1.15
MOUTH_MIN_NEIGHBORS = 15
MOUTH_MERGE_IOU_THRESHOLD = 0.25

# Search-area ratios for mouth detection inside the lower face region.
MOUTH_SEARCH_LEFT_MARGIN_RATIO = 0.10
MOUTH_SEARCH_RIGHT_MARGIN_RATIO = 0.10
MOUTH_SEARCH_TOP_RATIO = 0.45
MOUTH_SEARCH_BOTTOM_RATIO = 1.00

# Plausible size ranges for detected mouth boxes relative to the face size.
MOUTH_MIN_WIDTH_RATIO = 0.25
MOUTH_MIN_HEIGHT_RATIO = 0.10
MOUTH_MAX_WIDTH_RATIO = 0.75
MOUTH_MAX_HEIGHT_RATIO = 0.35

# Final number of mouth boxes preserved after filtering and selection.
MAX_MOUTH_COUNT = 1


# ---------------------------------------------------------------------
# Internal cascade-loading helpers
# ---------------------------------------------------------------------

def _load_single_cascade(cascade_path):
    """
    Load one Haar cascade classifier from an XML file.

    The function converts the provided path to string form so it can be used
    directly by OpenCV. A load failure is treated as a hard error because the
    detector pipeline cannot operate correctly without the cascade file.
    """

    # OpenCV cascade loaders work with string paths, so the incoming path is
    # normalized to string form here.
    cascade = cv2.CascadeClassifier(str(cascade_path))

    # An empty cascade means the XML file could not be loaded correctly.
    # Since the project cannot run the localization stage without the cascade,
    # this is treated as a hard failure instead of a silent fallback.
    if cascade.empty():
        raise FileNotFoundError(f"Cannot load cascade file: {cascade_path}")

    return cascade


# ---------------------------------------------------------------------
# Internal box-conversion and geometry helpers
# ---------------------------------------------------------------------

def _to_box_list(detected_boxes):
    """
    Convert OpenCV detection output into a regular Python list of tuples.

    OpenCV returns an array-like structure of rectangles. The project uses
    a uniform internal representation of bounding boxes:
        (x, y, w, h)

    Converting immediately to tuples keeps subsequent code simple and explicit.
    """

    # Standardize detector output as plain Python tuples so every later helper
    # works with the same bounding-box representation.
    return [tuple(map(int, box)) for box in detected_boxes]


def _box_area(box):
    """
    Compute the area of one bounding box.

    The box is assumed to use the standard project format:
        (x, y, w, h)
    """

    # The area is used repeatedly for:
    # - sorting detections by size,
    # - selecting the largest face,
    # - preferring larger eye/mouth detections.
    return box[2] * box[3]


def _box_center(box):
    """
    Compute the center point of one bounding box.

    The returned center is represented as a floating-point pair:
        (center_x, center_y)
    """

    # Center coordinates are used mainly in temporal continuity logic.
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _center_distance(box_a, box_b):
    """
    Compute Euclidean distance between the centers of two boxes.

    This measure is used in the main-face selection logic as one of the
    continuity cues between the previous frame and the current frame.
    """

    # Convert both boxes to center points and then compute straight-line
    # distance between those centers.
    ax, ay = _box_center(box_a)
    bx, by = _box_center(box_b)

    return math.hypot(ax - bx, ay - by)


def _compute_iou(box_a, box_b):
    """
    Compute Intersection over Union for two bounding boxes.

    IoU is a standard overlap measure. In this project it is used for:
    - removing strongly overlapping duplicate detections,
    - preferring stable face detections across adjacent frames.
    """

    # Convert both boxes from (x, y, w, h) into corner form.
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    # Compute the overlap rectangle.
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # No overlap means IoU is zero.
    if inter_area == 0:
        return 0.0

    union_area = _box_area(box_a) + _box_area(box_b) - inter_area

    # Defensive guard for degenerate box combinations.
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def _unflip_box_horizontally(box, frame_width):
    """
    Convert a box detected on a horizontally flipped image back into the
    coordinate system of the original image.

    This is used for right-profile face detection. The profile cascade is run
    once on the original frame and once on a flipped frame so that both left
    and right profiles can be handled using one profile classifier.
    """

    # Detection on the flipped frame returns coordinates in the flipped image
    # system, so this helper maps them back to the original-frame x position.
    x, y, w, h = box
    original_x = frame_width - x - w

    return (original_x, y, w, h)


# ---------------------------------------------------------------------
# Internal filtering and candidate-selection helpers
# ---------------------------------------------------------------------

def _merge_boxes(boxes, iou_threshold):
    """
    Filter overlapping boxes using a simple size-priority strategy.

    The procedure is:
    - sort candidate boxes from largest to smallest,
    - keep a box only if it does not overlap too strongly with a box that
      has already been accepted.

    This is not a full clustering algorithm. It is a lightweight post-processing
    step that is sufficient for a small school assignment.
    """

    # Empty input stays empty.
    if not boxes:
        return []

    # Larger boxes are considered first, so they get priority when overlap is
    # resolved.
    sorted_boxes = sorted(boxes, key=_box_area, reverse=True)
    merged_boxes = []

    # Keep a candidate only if it does not overlap too strongly with any box
    # that has already been accepted.
    for candidate_box in sorted_boxes:
        keep_box = True

        for kept_box in merged_boxes:
            if _compute_iou(candidate_box, kept_box) >= iou_threshold:
                keep_box = False
                break

        if keep_box:
            merged_boxes.append(candidate_box)

    return merged_boxes


def _select_best_eye_boxes(eye_boxes, max_eye_count=MAX_EYE_COUNT):
    """
    Keep only the most relevant eye candidates.

    The current strategy is intentionally simple:
    - keep the largest candidates,
    - then sort the retained boxes from left to right.

    This produces a stable and easy-to-use output for later stages of the
    pipeline, especially for frame-level eye-state classification.
    """

    # No eye candidates means nothing to select.
    if not eye_boxes:
        return []

    # First keep the largest candidates because they are usually the strongest
    # detections.
    best_boxes = sorted(eye_boxes, key=_box_area, reverse=True)[:max_eye_count]

    # Then reorder the selected eyes left-to-right so later runtime logic sees a
    # stable eye ordering.
    best_boxes = sorted(best_boxes, key=lambda box: box[0])

    return best_boxes


def _select_best_mouth_boxes(mouth_boxes, max_mouth_count=MAX_MOUTH_COUNT):
    """
    Keep only the most relevant mouth or smile candidates.

    The current selection favors:
    - larger detections,
    - detections lower in the face region if sizes are similar.

    This bias is appropriate because the mouth is expected in the lower part
    of the face and should generally occupy a visible area.
    """

    # No mouth candidates means nothing to select.
    if not mouth_boxes:
        return []

    # Sort primarily by area, and secondarily by lower vertical position, then
    # keep only the configured maximum count.
    best_boxes = sorted(
        mouth_boxes,
        key=lambda box: (_box_area(box), box[1] + box[3]),
        reverse=True
    )[:max_mouth_count]

    return best_boxes


# ---------------------------------------------------------------------
# Internal region-preparation helpers
# ---------------------------------------------------------------------

def _extract_face_roi(gray_frame, face_box):
    """
    Extract the face region of interest from the grayscale frame.

    The incoming face box is clamped to valid image coordinates so that the
    function remains safe even if a box lies partially outside the frame.

    The function returns:
    - the cropped face ROI,
    - the clamped coordinates in the form (x1, y1, x2, y2).
    """

    # No selected face means there is no face ROI to extract.
    if face_box is None:
        return None, None

    frame_height, frame_width = gray_frame.shape[:2]

    face_x, face_y, face_w, face_h = face_box

    # Clamp face coordinates to valid image bounds.
    x1 = max(0, face_x)
    y1 = max(0, face_y)
    x2 = min(frame_width, face_x + face_w)
    y2 = min(frame_height, face_y + face_h)

    # Reject degenerate or fully invalid regions.
    if x2 <= x1 or y2 <= y1:
        return None, None

    # Slice the actual grayscale face ROI.
    face_roi = gray_frame[y1:y2, x1:x2]

    return face_roi, (x1, y1, x2, y2)


def _prepare_downscaled_gray_frame(gray_frame, downscale_factor):
    """
    Prepare a grayscale frame for optional faster face detection.

    When the downscale factor is smaller than 1.0, the frame is reduced before
    face detection is performed. This speeds up the most expensive global
    localization stage. All detected boxes are later scaled back to original
    image coordinates.
    """

    # A factor of 1.0 or more means "do not downscale".
    if downscale_factor >= 1.0:
        return gray_frame, 1.0

    # Non-positive factors do not make sense geometrically.
    if downscale_factor <= 0.0:
        raise ValueError("downscale_factor must be greater than 0")

    # Build the reduced frame using area interpolation, which is a sensible
    # choice for image shrinking.
    downscaled_gray = cv2.resize(
        gray_frame,
        None,
        fx=downscale_factor,
        fy=downscale_factor,
        interpolation=cv2.INTER_AREA
    )

    return downscaled_gray, downscale_factor


def _scale_size_for_downscaled_frame(size, downscale_factor):
    """
    Scale a size tuple from original-frame coordinates to coordinates valid
    on the downscaled frame.

    This is needed so that minimum detector sizes remain consistent when
    face detection is executed on a reduced image.
    """

    width, height = size

    # Scale both dimensions and keep at least one pixel in each direction.
    scaled_width = max(1, int(round(width * downscale_factor)))
    scaled_height = max(1, int(round(height * downscale_factor)))

    return (scaled_width, scaled_height)


def _rescale_box_to_original_frame(box, downscale_factor):
    """
    Convert a box detected on a downscaled frame back to original-frame
    coordinates.

    This keeps the public detector interface consistent: all returned boxes
    use the original input-frame coordinate system.
    """

    # If no downscaling was used, the box is already in original coordinates.
    if downscale_factor == 1.0:
        return box

    x, y, w, h = box

    # Undo the coordinate scaling performed during downscaled detection.
    original_x = int(round(x / downscale_factor))
    original_y = int(round(y / downscale_factor))
    original_w = int(round(w / downscale_factor))
    original_h = int(round(h / downscale_factor))

    return (original_x, original_y, original_w, original_h)


# ---------------------------------------------------------------------
# Internal part-localization helpers
# ---------------------------------------------------------------------

def _detect_eyes_in_face_roi(face_roi, face_coords, face_box, cascades):
    """
    Detect eye candidates inside the upper part of the selected face region.

    The search is intentionally limited to the upper facial area and to
    a narrower horizontal band. This reduces false positives and speeds up
    the search compared with scanning the whole face ROI.

    Returned eye boxes are converted back into full-frame coordinates.
    """

    # face_coords give the face ROI location in full-frame coordinates, while
    # face_box gives the original selected face box dimensions.
    x1, y1, x2, y2 = face_coords
    _, _, face_w, face_h = face_box

    roi_height, roi_width = face_roi.shape[:2]

    # Restrict the search to the upper-middle part of the face where eyes are
    # expected.
    search_x1 = int(roi_width * EYE_SEARCH_LEFT_MARGIN_RATIO)
    search_x2 = roi_width - int(roi_width * EYE_SEARCH_RIGHT_MARGIN_RATIO)
    search_y1 = 0
    search_y2 = int(roi_height * EYE_SEARCH_TOP_RATIO)

    # If the search region collapses, no eye search is possible.
    if search_x2 <= search_x1 or search_y2 <= search_y1:
        return []

    eye_roi = face_roi[search_y1:search_y2, search_x1:search_x2]
    eye_cascade = cascades["eye"]

    # Derive plausible absolute eye-size limits from the selected face size.
    min_eye_size = (
        max(12, int(face_w * EYE_MIN_WIDTH_RATIO)),
        max(8, int(face_h * EYE_MIN_HEIGHT_RATIO))
    )

    max_eye_size = (
        max(min_eye_size[0], int(face_w * EYE_MAX_WIDTH_RATIO)),
        max(min_eye_size[1], int(face_h * EYE_MAX_HEIGHT_RATIO))
    )

    # Run the eye cascade only on the constrained search region, not on the full
    # face ROI.
    detected_eyes = eye_cascade.detectMultiScale(
        eye_roi,
        scaleFactor=EYE_SCALE_FACTOR,
        minNeighbors=EYE_MIN_NEIGHBORS,
        minSize=min_eye_size,
        maxSize=max_eye_size
    )

    eye_boxes_full_frame = []

    # Convert detected eye boxes from search-region coordinates back into
    # full-frame coordinates.
    for eye_x, eye_y, eye_w, eye_h in _to_box_list(detected_eyes):
        full_x = x1 + search_x1 + eye_x
        full_y = y1 + search_y1 + eye_y

        eye_boxes_full_frame.append((full_x, full_y, eye_w, eye_h))

    # Filter overlapping detections and keep only the most relevant final eye
    # boxes.
    merged_eye_boxes = _merge_boxes(eye_boxes_full_frame, EYE_MERGE_IOU_THRESHOLD)
    selected_eye_boxes = _select_best_eye_boxes(merged_eye_boxes)

    return selected_eye_boxes


def _detect_mouth_in_face_roi(face_roi, face_coords, face_box, cascades):
    """
    Detect mouth or smile candidates inside the lower part of the selected
    face region.

    The search is limited to the lower facial area because that is where the
    mouth is expected. The result is returned in full-frame coordinates so it
    remains compatible with the rest of the project.
    """

    x1, y1, x2, y2 = face_coords
    _, _, face_w, face_h = face_box

    roi_height, roi_width = face_roi.shape[:2]

    # Restrict the search to the lower part of the face where the mouth is
    # expected.
    search_x1 = int(roi_width * MOUTH_SEARCH_LEFT_MARGIN_RATIO)
    search_x2 = roi_width - int(roi_width * MOUTH_SEARCH_RIGHT_MARGIN_RATIO)
    search_y1 = int(roi_height * MOUTH_SEARCH_TOP_RATIO)
    search_y2 = int(roi_height * MOUTH_SEARCH_BOTTOM_RATIO)

    if search_x2 <= search_x1 or search_y2 <= search_y1:
        return []

    mouth_roi = face_roi[search_y1:search_y2, search_x1:search_x2]
    mouth_cascade = cascades["mouth"]

    # Derive plausible mouth-size limits from the selected face size.
    min_mouth_size = (
        max(20, int(face_w * MOUTH_MIN_WIDTH_RATIO)),
        max(12, int(face_h * MOUTH_MIN_HEIGHT_RATIO))
    )

    max_mouth_size = (
        max(min_mouth_size[0], int(face_w * MOUTH_MAX_WIDTH_RATIO)),
        max(min_mouth_size[1], int(face_h * MOUTH_MAX_HEIGHT_RATIO))
    )

    # Run the mouth/smile cascade only on the constrained lower-face search
    # region.
    detected_mouth = mouth_cascade.detectMultiScale(
        mouth_roi,
        scaleFactor=MOUTH_SCALE_FACTOR,
        minNeighbors=MOUTH_MIN_NEIGHBORS,
        minSize=min_mouth_size,
        maxSize=max_mouth_size
    )

    mouth_boxes_full_frame = []

    # Convert detected mouth boxes from search-region coordinates back into
    # full-frame coordinates.
    for mouth_x, mouth_y, mouth_w, mouth_h in _to_box_list(detected_mouth):
        full_x = x1 + search_x1 + mouth_x
        full_y = y1 + search_y1 + mouth_y

        mouth_boxes_full_frame.append((full_x, full_y, mouth_w, mouth_h))

    # Filter overlaps and keep only the best final mouth candidates.
    merged_mouth_boxes = _merge_boxes(mouth_boxes_full_frame, MOUTH_MERGE_IOU_THRESHOLD)
    selected_mouth_boxes = _select_best_mouth_boxes(merged_mouth_boxes)

    return selected_mouth_boxes


# ---------------------------------------------------------------------
# Public detector interface
# ---------------------------------------------------------------------

def load_cascades(paths):
    """
    Load all cascade classifiers required by the project.

    The returned dictionary uses stable logical names so that the rest of the
    project never needs to know the exact file names or directory layout.
    """

    # Centralize cascade-file loading here so the rest of the project only works
    # with logical detector names and not with file-system details.
    cascades = {
        "face_frontal": _load_single_cascade(paths["face_cascade_frontal_path"]),
        "face_profile": _load_single_cascade(paths["face_cascade_profile_path"]),
        "eye": _load_single_cascade(paths["eye_cascade_path"]),
        "mouth": _load_single_cascade(paths["mouth_cascade_path"]),
    }

    return cascades


def merge_face_boxes(face_boxes, iou_threshold=FACE_MERGE_IOU_THRESHOLD):
    """
    Merge or filter overlapping face boxes.

    This function exists as a public wrapper around the generic internal
    merging helper so that face-specific code can remain explicit and readable.
    """

    # Public wrapper mainly for readability at higher call sites.
    return _merge_boxes(face_boxes, iou_threshold)


def detect_faces(gray_frame, cascades, downscale_factor=1.0):
    """
    Detect faces in one grayscale frame.

    The current strategy is:
    - optionally downscale the frame for faster global detection,
    - run frontal-face detection first,
    - if frontal detection succeeds, use those results directly,
    - otherwise try profile detection on both normal and flipped views,
    - rescale every accepted detection back to original-frame coordinates,
    - merge strongly overlapping detections.

    The returned face boxes always use the original input-frame coordinate
    system, regardless of whether downscaling was used internally.
    """

    # Prepare the working frame for detection. Face detection is the most global
    # and expensive localization step, so optional downscaling is supported here.
    detection_gray, used_downscale_factor = _prepare_downscaled_gray_frame(
        gray_frame,
        downscale_factor
    )

    # Scale the minimum allowed face size so detector behavior remains roughly
    # consistent even when the frame is downscaled.
    detection_min_size = _scale_size_for_downscaled_frame(
        FACE_MIN_SIZE,
        used_downscale_factor
    )

    frontal_cascade = cascades["face_frontal"]
    profile_cascade = cascades["face_profile"]

    # Primary strategy:
    # try frontal-face detection first, because that is the most common and
    # preferred face orientation in the target video.
    frontal_faces = frontal_cascade.detectMultiScale(
        detection_gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=detection_min_size
    )

    frontal_boxes = [
        _rescale_box_to_original_frame(box, used_downscale_factor)
        for box in _to_box_list(frontal_faces)
    ]

    # If frontal detection succeeds at all, use those results directly and do
    # not fall back to profile detection.
    if frontal_boxes:
        return merge_face_boxes(frontal_boxes)

    # Fallback strategy:
    # if no frontal face is found, try profile detection in both directions:
    # - left profile directly,
    # - right profile by flipping the frame horizontally and then unflipping the
    #   resulting detections.
    profile_faces_left = profile_cascade.detectMultiScale(
        detection_gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=detection_min_size
    )

    flipped_gray = cv2.flip(detection_gray, 1)
    profile_faces_right_flipped = profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=detection_min_size
    )

    profile_boxes_left = _to_box_list(profile_faces_left)

    detection_frame_width = detection_gray.shape[1]
    profile_boxes_right = [
        _unflip_box_horizontally(box, detection_frame_width)
        for box in _to_box_list(profile_faces_right_flipped)
    ]

    # Convert both left-profile and right-profile detections back into original
    # full-frame coordinates.
    profile_boxes_left = [
        _rescale_box_to_original_frame(box, used_downscale_factor)
        for box in profile_boxes_left
    ]

    profile_boxes_right = [
        _rescale_box_to_original_frame(box, used_downscale_factor)
        for box in profile_boxes_right
    ]

    all_profile_boxes = profile_boxes_left + profile_boxes_right
    merged_profile_boxes = merge_face_boxes(all_profile_boxes)

    return merged_profile_boxes


def select_main_face(face_boxes, previous_face=None):
    """
    Select one main face from the set of current face detections.

    The current selection logic aims at temporal stability:
    - if there is no previous face, the largest current face is used,
    - if there is a previous face, boxes that overlap with it or remain close
      to it are preferred,
    - if no continuity candidate exists, the function falls back to the
      largest current detection.

    This design keeps the downstream eye and mouth localization focused on
    one stable target face.
    """

    # No face detections means there is no main face.
    if not face_boxes:
        return None

    # The fallback/default choice is always the largest visible face.
    largest_face = max(face_boxes, key=_box_area)

    # Without temporal history, simply use the largest current face.
    if previous_face is None:
        return largest_face

    continuity_candidates = []

    # Compare every current face box against the previous selected face and keep
    # candidates that either overlap enough or remain spatially close enough.
    for candidate_box in face_boxes:
        iou = _compute_iou(candidate_box, previous_face)
        distance = _center_distance(candidate_box, previous_face)

        max_previous_size = max(previous_face[2], previous_face[3])
        distance_limit = max_previous_size * FACE_CONTINUITY_DISTANCE_FACTOR

        if iou >= FACE_CONTINUITY_IOU_THRESHOLD or distance <= distance_limit:
            continuity_candidates.append(candidate_box)

    # If temporally consistent candidates exist, choose the largest among them.
    if continuity_candidates:
        return max(continuity_candidates, key=_box_area)

    # Otherwise fall back to the largest current face.
    return largest_face


def detect_face_parts(gray_frame, face_box, cascades, enable_mouth_detection=True):
    """
    Detect relevant facial parts inside the selected face region.

    The returned dictionary always contains the same keys so that the rest of
    the project can work with a stable interface:
    - "eyes"
    - "mouth"

    If no face is available, both lists are empty.
    """

    # First isolate the face ROI corresponding to the selected main face.
    face_roi, face_coords = _extract_face_roi(gray_frame, face_box)

    # If no valid face ROI exists, return the standard empty structure expected
    # elsewhere in the project.
    if face_roi is None:
        return {
            "eyes": [],
            "mouth": []
        }

    # Detect eyes in the constrained upper-face search area.
    eyes = _detect_eyes_in_face_roi(face_roi, face_coords, face_box, cascades)

    # Detect mouth only when explicitly enabled; otherwise keep a stable empty
    # list so the output structure remains consistent.
    if enable_mouth_detection:
        mouth = _detect_mouth_in_face_roi(face_roi, face_coords, face_box, cascades)
    else:
        mouth = []

    return {
        "eyes": eyes,
        "mouth": mouth
    }