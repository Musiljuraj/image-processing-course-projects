"""
eye_state_lbp.py

This module implements frame-level eye-state classification using the
trained LBP eye-state model.

Its role is to replace the heuristic decision core from eye_state.py while
reusing as much of the existing project structure as possible.

High-level flow:
- reuse the existing eye-ROI extraction logic,
- preprocess each detected eye ROI using the shared preprocessing module,
- convert each preprocessed eye ROI into LBP features,
- classify each eye as open/close using the trained model bundle,
- aggregate eye-level predictions into one frame-level eye-state label.

The module is intentionally designed so that it can later replace the current
heuristic call in main.py with minimal change.
"""

import math

from eye_state import (
    extract_eye_roi,
    classify_eye_state as classify_eye_state_heuristic,
)

from eye_lbp_dataset import prepare_runtime_feature_record
from eye_lbp_classifier import predict_from_runtime_feature_record


# ---------------------------------------------------------------------
# LBP runtime configuration
# ---------------------------------------------------------------------

LBP_FRAME_OPEN_THRESHOLD = 0.50
LBP_STRONG_OPEN_SCORE_THRESHOLD = 0.70

DEFAULT_FALLBACK_FRAME_LABEL = "close"


# ---------------------------------------------------------------------
# Internal normalization helpers
# ---------------------------------------------------------------------

def _normalize_eye_state_label(label):
    """
    Normalize textual eye-state labels into the LBP runtime vocabulary.

    Canonical labels used here:
    - "open"
    - "close"
    """

    if label is None:
        return "unknown"

    value = str(label).strip().lower()

    if value in ("open", "opened"):
        return "open"

    if value in ("close", "closed", "shut"):
        return "close"

    return value


def _get_classifier_name_from_model_bundle(model_bundle):
    """
    Read the classifier name from the model bundle when available.
    """

    if not isinstance(model_bundle, dict):
        return None

    classifier_config = model_bundle.get("classifier_config", {})
    if not isinstance(classifier_config, dict):
        return None

    classifier_name = classifier_config.get("classifier_name")
    if classifier_name is None:
        return None

    return str(classifier_name).strip().lower()


def _validate_model_bundle(model_bundle):
    """
    Validate that the runtime model bundle contains the minimum fields
    required for LBP inference.
    """

    if not isinstance(model_bundle, dict):
        raise TypeError("model_bundle must be a dictionary.")

    required_keys = {
        "model",
        "preprocessing_config",
        "lbp_config",
        "classifier_config",
    }

    missing_keys = required_keys - set(model_bundle.keys())

    if missing_keys:
        raise KeyError(
            f"model_bundle is missing required keys: {sorted(missing_keys)}"
        )

    return model_bundle


def _convert_prediction_to_open_score(predicted_label, predicted_score, classifier_name):
    """
    Convert prediction output into one open-score in the interval [0, 1].

    Rules:
    - if score is None:
        use hard-label fallback (1.0 for open, 0.0 for close)
    - knn:
        predicted_score is already probability-like for class 1
    - linear_svm:
        predicted_score is decision-function-like, convert with sigmoid
    - otherwise:
        fallback to hard-label score
    """

    predicted_label = int(predicted_label)

    if predicted_score is None:
        return 1.0 if predicted_label == 1 else 0.0

    predicted_score = float(predicted_score)
    classifier_name = (classifier_name or "").strip().lower()

    if classifier_name == "knn":
        return max(0.0, min(1.0, predicted_score))

    if classifier_name == "linear_svm":
        return 1.0 / (1.0 + math.exp(-predicted_score))

    return 1.0 if predicted_label == 1 else 0.0


def _build_empty_eye_prediction(eye_box, eye_index, reason):
    """
    Build a standard failed-eye-prediction record.
    """

    return {
        "eye_box": eye_box,
        "eye_index": eye_index,
        "success": False,
        "reason": reason,
        "predicted_label": None,
        "predicted_class_name": "unknown",
        "predicted_score": None,
        "predicted_open_score": None,
        "preprocessed_image_shape": None,
    }


# ---------------------------------------------------------------------
# Eye-level LBP classification
# ---------------------------------------------------------------------

def classify_single_eye_lbp(
    gray_frame,
    eye_box,
    model_bundle,
    eye_index=None,
    frame_index=None,
):
    """
    Classify one detected eye using the trained LBP model bundle.

    Return:
    - prediction_record:
        dictionary containing:
            success
            predicted_label
            predicted_class_name
            predicted_score
            predicted_open_score
            eye_box
            eye_index
            frame_index
            and the runtime feature fields
    """

    validated_model_bundle = _validate_model_bundle(model_bundle)

    eye_roi = extract_eye_roi(gray_frame, eye_box)

    if eye_roi is None or eye_roi.size == 0:
        return _build_empty_eye_prediction(
            eye_box=eye_box,
            eye_index=eye_index,
            reason="invalid_eye_roi",
        )

    runtime_feature_record = prepare_runtime_feature_record(
        runtime_eye_image=eye_roi,
        preprocessing_config=validated_model_bundle["preprocessing_config"],
        lbp_config=validated_model_bundle["lbp_config"],
        metadata={
            "frame_index": frame_index,
            "eye_index": eye_index,
            "eye_box": eye_box,
        },
    )

    prediction_record = predict_from_runtime_feature_record(
        model=validated_model_bundle["model"],
        runtime_feature_record=runtime_feature_record,
    )

    predicted_class_name = _normalize_eye_state_label(
        prediction_record.get("predicted_class_name")
    )

    predicted_label = prediction_record.get("predicted_label")
    predicted_score = prediction_record.get("predicted_score")

    classifier_name = _get_classifier_name_from_model_bundle(validated_model_bundle)
    predicted_open_score = _convert_prediction_to_open_score(
        predicted_label=predicted_label,
        predicted_score=predicted_score,
        classifier_name=classifier_name,
    )

    prediction_record["success"] = True
    prediction_record["predicted_class_name"] = predicted_class_name
    prediction_record["predicted_open_score"] = float(predicted_open_score)

    return prediction_record


# ---------------------------------------------------------------------
# Frame-level aggregation
# ---------------------------------------------------------------------

def aggregate_eye_predictions(
    eye_prediction_records,
    previous_eye_state=None,
    fallback_frame_label=DEFAULT_FALLBACK_FRAME_LABEL,
):
    """
    Aggregate eye-level predictions into one frame-level label.

    Strategy:
    - no valid eye predictions:
        previous_eye_state if available, otherwise fallback_frame_label
    - one valid eye:
        use that eye
    - two or more valid eyes:
        use majority of hard labels,
        break ties with mean predicted_open_score,
        break exact ties with previous_eye_state, then fallback_frame_label
    """

    normalized_previous_eye_state = _normalize_eye_state_label(previous_eye_state)
    normalized_fallback_frame_label = _normalize_eye_state_label(fallback_frame_label)

    valid_eye_predictions = []

    for record in eye_prediction_records:
        if not isinstance(record, dict):
            continue

        if not record.get("success", False):
            continue

        predicted_class_name = _normalize_eye_state_label(
            record.get("predicted_class_name")
        )

        if predicted_class_name not in ("open", "close"):
            continue

        valid_eye_predictions.append(record)

    if not valid_eye_predictions:
        if normalized_previous_eye_state in ("open", "close"):
            frame_label = normalized_previous_eye_state
        else:
            frame_label = normalized_fallback_frame_label

        return frame_label, {
            "valid_eye_count": 0,
            "mean_open_score": None,
            "open_eye_count": 0,
            "close_eye_count": 0,
            "used_previous_eye_state": normalized_previous_eye_state in ("open", "close"),
            "used_fallback_label": normalized_previous_eye_state not in ("open", "close"),
        }

    if len(valid_eye_predictions) == 1:
        single_record = valid_eye_predictions[0]
        frame_label = _normalize_eye_state_label(
            single_record["predicted_class_name"]
        )

        return frame_label, {
            "valid_eye_count": 1,
            "mean_open_score": float(single_record["predicted_open_score"]),
            "open_eye_count": 1 if frame_label == "open" else 0,
            "close_eye_count": 1 if frame_label == "close" else 0,
            "used_previous_eye_state": False,
            "used_fallback_label": False,
        }

    open_eye_count = 0
    close_eye_count = 0
    open_scores = []

    for record in valid_eye_predictions:
        predicted_class_name = _normalize_eye_state_label(
            record["predicted_class_name"]
        )

        if predicted_class_name == "open":
            open_eye_count += 1
        elif predicted_class_name == "close":
            close_eye_count += 1

        open_scores.append(float(record["predicted_open_score"]))

    mean_open_score = sum(open_scores) / max(1, len(open_scores))

    if open_eye_count > close_eye_count:
        frame_label = "open"
    elif close_eye_count > open_eye_count:
        frame_label = "close"
    else:
        if mean_open_score > LBP_FRAME_OPEN_THRESHOLD:
            frame_label = "open"
        elif mean_open_score < LBP_FRAME_OPEN_THRESHOLD:
            frame_label = "close"
        elif normalized_previous_eye_state in ("open", "close"):
            frame_label = normalized_previous_eye_state
        else:
            frame_label = normalized_fallback_frame_label

    return frame_label, {
        "valid_eye_count": len(valid_eye_predictions),
        "mean_open_score": float(mean_open_score),
        "open_eye_count": open_eye_count,
        "close_eye_count": close_eye_count,
        "used_previous_eye_state": (
            open_eye_count == close_eye_count
            and mean_open_score == LBP_FRAME_OPEN_THRESHOLD
            and normalized_previous_eye_state in ("open", "close")
        ),
        "used_fallback_label": (
            open_eye_count == close_eye_count
            and mean_open_score == LBP_FRAME_OPEN_THRESHOLD
            and normalized_previous_eye_state not in ("open", "close")
        ),
    }


# ---------------------------------------------------------------------
# Public frame-level interface
# ---------------------------------------------------------------------

def classify_eye_state_lbp(
    gray_frame,
    face_parts,
    model_bundle,
    previous_eye_state=None,
    fallback_to_heuristic=True,
    fallback_frame_label=DEFAULT_FALLBACK_FRAME_LABEL,
    frame_index=None,
    return_details=False,
):
    """
    Classify frame-level eye state using the trained LBP model bundle.

    Inputs:
    - gray_frame
    - face_parts ........ expected to contain face_parts["eyes"]
    - model_bundle ...... output of build_eye_lbp_model(...)
    - previous_eye_state  optional temporal fallback
    - fallback_to_heuristic
    - fallback_frame_label
    - frame_index
    - return_details ..... when True, also return a details dictionary

    Return:
    - by default:
        frame_label
    - when return_details=True:
        (frame_label, details)
    """

    if face_parts is None:
        face_parts = {"eyes": [], "mouth": []}

    eye_boxes = face_parts.get("eyes", [])

    if model_bundle is None:
        if fallback_to_heuristic:
            heuristic_label = _normalize_eye_state_label(
                classify_eye_state_heuristic(gray_frame, face_parts)
            )

            if return_details:
                return heuristic_label, {
                    "used_model_bundle": False,
                    "used_heuristic_fallback": True,
                    "eye_prediction_records": [],
                    "aggregation": None,
                }

            return heuristic_label

        frame_label = _normalize_eye_state_label(
            previous_eye_state if previous_eye_state is not None else fallback_frame_label
        )

        if return_details:
            return frame_label, {
                "used_model_bundle": False,
                "used_heuristic_fallback": False,
                "eye_prediction_records": [],
                "aggregation": None,
            }

        return frame_label

    eye_prediction_records = []

    for eye_index, eye_box in enumerate(eye_boxes, start=1):
        eye_prediction_record = classify_single_eye_lbp(
            gray_frame=gray_frame,
            eye_box=eye_box,
            model_bundle=model_bundle,
            eye_index=eye_index,
            frame_index=frame_index,
        )
        eye_prediction_records.append(eye_prediction_record)

    frame_label, aggregation_details = aggregate_eye_predictions(
        eye_prediction_records=eye_prediction_records,
        previous_eye_state=previous_eye_state,
        fallback_frame_label=fallback_frame_label,
    )

    used_heuristic_fallback = False

    if fallback_to_heuristic and aggregation_details["valid_eye_count"] == 0:
        heuristic_label = _normalize_eye_state_label(
            classify_eye_state_heuristic(gray_frame, face_parts)
        )

        if heuristic_label in ("open", "close"):
            frame_label = heuristic_label
            used_heuristic_fallback = True

    if return_details:
        details = {
            "used_model_bundle": True,
            "used_heuristic_fallback": used_heuristic_fallback,
            "eye_prediction_records": eye_prediction_records,
            "aggregation": aggregation_details,
        }
        return frame_label, details

    return frame_label