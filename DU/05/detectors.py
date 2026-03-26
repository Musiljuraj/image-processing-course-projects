"""
detectors.py

Current stage:
- detector module created
- real cascade loading implemented
- multi-cascade face detection implemented
- face-box merging/filtering implemented
- main-face selection implemented
- eye detection inside face ROI implemented
- mouth/smile detection inside lower face ROI implemented
- combined face-parts detection implemented

Not implemented yet:
- eye-state classification support logic
"""

import math
import cv2


# ---------------------------------------------------------------------
# Basic detector parameters
# ---------------------------------------------------------------------

FACE_SCALE_FACTOR = 1.15 #1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (30, 30)

FACE_MERGE_IOU_THRESHOLD = 0.30
FACE_CONTINUITY_IOU_THRESHOLD = 0.10
FACE_CONTINUITY_DISTANCE_FACTOR = 0.60

EYE_SCALE_FACTOR = 1.15 #1.05
EYE_MIN_NEIGHBORS = 5
EYE_MERGE_IOU_THRESHOLD = 0.20

EYE_SEARCH_LEFT_MARGIN_RATIO = 0.05
EYE_SEARCH_RIGHT_MARGIN_RATIO = 0.05
EYE_SEARCH_TOP_RATIO = 0.55

EYE_MIN_WIDTH_RATIO = 0.12
EYE_MIN_HEIGHT_RATIO = 0.08
EYE_MAX_WIDTH_RATIO = 0.45
EYE_MAX_HEIGHT_RATIO = 0.30

MAX_EYE_COUNT = 2

MOUTH_SCALE_FACTOR = 1.15 #1.10
MOUTH_MIN_NEIGHBORS = 15
MOUTH_MERGE_IOU_THRESHOLD = 0.25

MOUTH_SEARCH_LEFT_MARGIN_RATIO = 0.10
MOUTH_SEARCH_RIGHT_MARGIN_RATIO = 0.10
MOUTH_SEARCH_TOP_RATIO = 0.45
MOUTH_SEARCH_BOTTOM_RATIO = 1.00

MOUTH_MIN_WIDTH_RATIO = 0.25
MOUTH_MIN_HEIGHT_RATIO = 0.10
MOUTH_MAX_WIDTH_RATIO = 0.75
MOUTH_MAX_HEIGHT_RATIO = 0.35

MAX_MOUTH_COUNT = 1


# ---------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------

def _load_single_cascade(cascade_path):
    """Load one Haar cascade XML file."""

    cascade = cv2.CascadeClassifier(str(cascade_path))

    if cascade.empty():
        raise FileNotFoundError(f"Cannot load cascade file: {cascade_path}")

    return cascade


def _to_box_list(detected_boxes):
    """Convert OpenCV detection output to a plain Python list of tuples."""

    return [tuple(map(int, box)) for box in detected_boxes]


def _box_area(box):
    """Return area of one (x, y, w, h) box."""

    return box[2] * box[3]


def _box_center(box):
    """Return center point of one (x, y, w, h) box."""

    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _center_distance(box_a, box_b):
    """Return Euclidean distance between centers of two boxes."""

    ax, ay = _box_center(box_a)
    bx, by = _box_center(box_b)

    return math.hypot(ax - bx, ay - by)


def _compute_iou(box_a, box_b):
    """
    Compute Intersection over Union for two boxes.

    IoU is used here only as a simple overlap measure for filtering
    and for continuity between frames.
    """

    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    union_area = _box_area(box_a) + _box_area(box_b) - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def _unflip_box_horizontally(box, frame_width):
    """
    Convert a box detected on a horizontally flipped frame back to the
    original frame coordinates.
    """

    x, y, w, h = box
    original_x = frame_width - x - w

    return (original_x, y, w, h)


def _merge_boxes(boxes, iou_threshold):
    """
    Merge/filter overlapping boxes.

    Simple strategy:
    - sort boxes by area from largest to smallest
    - keep a box only if it does not strongly overlap with a box already kept
    """

    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=_box_area, reverse=True)
    merged_boxes = []

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
    Keep only a small number of best eye candidates.

    Current strategy:
    - keep the largest candidates
    - then sort them from left to right
    """

    if not eye_boxes:
        return []

    best_boxes = sorted(eye_boxes, key=_box_area, reverse=True)[:max_eye_count]
    best_boxes = sorted(best_boxes, key=lambda box: box[0])

    return best_boxes


def _select_best_mouth_boxes(mouth_boxes, max_mouth_count=MAX_MOUTH_COUNT):
    """
    Keep only a small number of best mouth/smile candidates.

    Current strategy:
    - prefer larger boxes
    - if sizes are similar, prefer lower boxes
    """

    if not mouth_boxes:
        return []

    best_boxes = sorted(
        mouth_boxes,
        key=lambda box: (_box_area(box), box[1] + box[3]),
        reverse=True
    )[:max_mouth_count]

    return best_boxes


def _extract_face_roi(gray_frame, face_box):
    """
    Clamp a face box to the frame and return:
    - face ROI
    - clamped face coordinates
    """

    if face_box is None:
        return None, None

    frame_height, frame_width = gray_frame.shape[:2]

    face_x, face_y, face_w, face_h = face_box

    x1 = max(0, face_x)
    y1 = max(0, face_y)
    x2 = min(frame_width, face_x + face_w)
    y2 = min(frame_height, face_y + face_h)

    if x2 <= x1 or y2 <= y1:
        return None, None

    face_roi = gray_frame[y1:y2, x1:x2]

    return face_roi, (x1, y1, x2, y2)


def _detect_eyes_in_face_roi(face_roi, face_coords, face_box, cascades):
    """
    Detect eye candidates inside the upper part of the face ROI.

    Returned eye boxes are in full-frame coordinates.
    """

    x1, y1, x2, y2 = face_coords
    _, _, face_w, face_h = face_box

    roi_height, roi_width = face_roi.shape[:2]

    search_x1 = int(roi_width * EYE_SEARCH_LEFT_MARGIN_RATIO)
    search_x2 = roi_width - int(roi_width * EYE_SEARCH_RIGHT_MARGIN_RATIO)
    search_y1 = 0
    search_y2 = int(roi_height * EYE_SEARCH_TOP_RATIO)

    if search_x2 <= search_x1 or search_y2 <= search_y1:
        return []

    eye_roi = face_roi[search_y1:search_y2, search_x1:search_x2]
    eye_cascade = cascades["eye"]

    min_eye_size = (
        max(12, int(face_w * EYE_MIN_WIDTH_RATIO)),
        max(8, int(face_h * EYE_MIN_HEIGHT_RATIO))
    )

    max_eye_size = (
        max(min_eye_size[0], int(face_w * EYE_MAX_WIDTH_RATIO)),
        max(min_eye_size[1], int(face_h * EYE_MAX_HEIGHT_RATIO))
    )

    detected_eyes = eye_cascade.detectMultiScale(
        eye_roi,
        scaleFactor=EYE_SCALE_FACTOR,
        minNeighbors=EYE_MIN_NEIGHBORS,
        minSize=min_eye_size,
        maxSize=max_eye_size
    )

    eye_boxes_full_frame = []

    for eye_x, eye_y, eye_w, eye_h in _to_box_list(detected_eyes):
        full_x = x1 + search_x1 + eye_x
        full_y = y1 + search_y1 + eye_y

        eye_boxes_full_frame.append((full_x, full_y, eye_w, eye_h))

    merged_eye_boxes = _merge_boxes(eye_boxes_full_frame, EYE_MERGE_IOU_THRESHOLD)
    selected_eye_boxes = _select_best_eye_boxes(merged_eye_boxes)

    return selected_eye_boxes


def _detect_mouth_in_face_roi(face_roi, face_coords, face_box, cascades):
    """
    Detect mouth/smile candidates inside the lower part of the face ROI.

    Returned mouth boxes are in full-frame coordinates.
    """

    x1, y1, x2, y2 = face_coords
    _, _, face_w, face_h = face_box

    roi_height, roi_width = face_roi.shape[:2]

    search_x1 = int(roi_width * MOUTH_SEARCH_LEFT_MARGIN_RATIO)
    search_x2 = roi_width - int(roi_width * MOUTH_SEARCH_RIGHT_MARGIN_RATIO)
    search_y1 = int(roi_height * MOUTH_SEARCH_TOP_RATIO)
    search_y2 = int(roi_height * MOUTH_SEARCH_BOTTOM_RATIO)

    if search_x2 <= search_x1 or search_y2 <= search_y1:
        return []

    mouth_roi = face_roi[search_y1:search_y2, search_x1:search_x2]
    mouth_cascade = cascades["mouth"]

    min_mouth_size = (
        max(20, int(face_w * MOUTH_MIN_WIDTH_RATIO)),
        max(12, int(face_h * MOUTH_MIN_HEIGHT_RATIO))
    )

    max_mouth_size = (
        max(min_mouth_size[0], int(face_w * MOUTH_MAX_WIDTH_RATIO)),
        max(min_mouth_size[1], int(face_h * MOUTH_MAX_HEIGHT_RATIO))
    )

    detected_mouth = mouth_cascade.detectMultiScale(
        mouth_roi,
        scaleFactor=MOUTH_SCALE_FACTOR,
        minNeighbors=MOUTH_MIN_NEIGHBORS,
        minSize=min_mouth_size,
        maxSize=max_mouth_size
    )

    mouth_boxes_full_frame = []

    for mouth_x, mouth_y, mouth_w, mouth_h in _to_box_list(detected_mouth):
        full_x = x1 + search_x1 + mouth_x
        full_y = y1 + search_y1 + mouth_y

        mouth_boxes_full_frame.append((full_x, full_y, mouth_w, mouth_h))

    merged_mouth_boxes = _merge_boxes(mouth_boxes_full_frame, MOUTH_MERGE_IOU_THRESHOLD)
    selected_mouth_boxes = _select_best_mouth_boxes(merged_mouth_boxes)

    return selected_mouth_boxes


# ---------------------------------------------------------------------
# Detector module public interface
# ---------------------------------------------------------------------

def load_cascades(paths):
    """
    Load all cascade classifiers needed by the project.
    """

    cascades = {
        "face_frontal": _load_single_cascade(paths["face_cascade_frontal_path"]),
        "face_profile": _load_single_cascade(paths["face_cascade_profile_path"]),
        "eye": _load_single_cascade(paths["eye_cascade_path"]),
        "mouth": _load_single_cascade(paths["mouth_cascade_path"]),
    }

    return cascades


def merge_face_boxes(face_boxes, iou_threshold=FACE_MERGE_IOU_THRESHOLD):
    """
    Merge/filter overlapping face boxes.
    """

    return _merge_boxes(face_boxes, iou_threshold)


def detect_faces(gray_frame, cascades):
    """
    Detect faces in one grayscale frame using:
    - frontal face cascade
    - profile face cascade on original frame
    - profile face cascade on horizontally flipped frame

    Returned result is already merged/filtered.
    """

    frontal_cascade = cascades["face_frontal"]
    profile_cascade = cascades["face_profile"]

    frontal_faces = frontal_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )

    profile_faces_left = profile_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )

    flipped_gray = cv2.flip(gray_frame, 1)
    profile_faces_right_flipped = profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )

    frontal_boxes = _to_box_list(frontal_faces)
    profile_boxes_left = _to_box_list(profile_faces_left)

    frame_width = gray_frame.shape[1]
    profile_boxes_right = [
        _unflip_box_horizontally(box, frame_width)
        for box in _to_box_list(profile_faces_right_flipped)
    ]

    all_face_boxes = frontal_boxes + profile_boxes_left + profile_boxes_right
    merged_face_boxes = merge_face_boxes(all_face_boxes)

    return merged_face_boxes


def select_main_face(face_boxes, previous_face=None):
    """
    Select one main face from detected face boxes.

    Current strategy:
    - if no faces exist, return None
    - if there is no previous face, choose the largest box
    - if previous face exists, prefer a box that overlaps with it
      or stays close to it
    - if no continuity candidate exists, fall back to the largest box
    """

    if not face_boxes:
        return None

    largest_face = max(face_boxes, key=_box_area)

    if previous_face is None:
        return largest_face

    continuity_candidates = []

    for candidate_box in face_boxes:
        iou = _compute_iou(candidate_box, previous_face)
        distance = _center_distance(candidate_box, previous_face)

        max_previous_size = max(previous_face[2], previous_face[3])
        distance_limit = max_previous_size * FACE_CONTINUITY_DISTANCE_FACTOR

        if iou >= FACE_CONTINUITY_IOU_THRESHOLD or distance <= distance_limit:
            continuity_candidates.append(candidate_box)

    if continuity_candidates:
        return max(continuity_candidates, key=_box_area)

    return largest_face


def detect_face_parts(gray_frame, face_box, cascades):
    """
    Detect face parts inside the selected face ROI.

    Current behavior:
    - if no face is selected, return empty result
    - detect eyes in the upper face region
    - detect mouth/smile in the lower face region

    Returned boxes are in full-frame coordinates.
    """

    face_roi, face_coords = _extract_face_roi(gray_frame, face_box)

    if face_roi is None:
        return {
            "eyes": [],
            "mouth": []
        }

    eyes = _detect_eyes_in_face_roi(face_roi, face_coords, face_box, cascades)
    mouth = _detect_mouth_in_face_roi(face_roi, face_coords, face_box, cascades)

    return {
        "eyes": eyes,
        "mouth": mouth
    }