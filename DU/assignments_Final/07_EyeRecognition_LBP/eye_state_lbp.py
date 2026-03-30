# This module is the runtime decision layer for the LBP-based eye-state system.
# The earlier modules in the pipeline already know how to:
# - load training data,
# - preprocess eye images,
# - compute LBP features,
# - train a classifier,
# - produce a reusable model bundle.
#
# This file is the layer that uses those pieces during video processing.
# In the overall runtime flow, it performs:
#
#     detected eye box in a frame
#         -> extract eye ROI
#         -> preprocess runtime eye ROI
#         -> compute LBP features
#         -> classify each eye with the trained model
#         -> convert classifier output into one comparable open-score
#         -> aggregate eye-level decisions into one frame-level eye-state label
#
# A key design choice here is backward compatibility with the earlier project
# structure:
# - it reuses eye ROI extraction from eye_state.py,
# - it can optionally fall back to the older heuristic classifier,
# - it preserves the same frame-level "open"/"close" vocabulary used elsewhere.
#
# That makes this module the practical runtime replacement for the older
# heuristic eye-state core, while still fitting neatly into the same pipeline.

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

# math is needed for converting linear-SVM decision values into a bounded
# probability-like score through the sigmoid function.
import math

# Reuse two important pieces from the earlier heuristic eye-state module:
# - extract_eye_roi ............ keeps runtime eye-crop logic consistent
# - classify_eye_state_heuristic
#     ......................... provides an optional fallback path when the LBP
#                               route cannot produce any valid eye prediction
from eye_state import (
    extract_eye_roi,
    classify_eye_state as classify_eye_state_heuristic,
)

# This helper converts one runtime eye ROI into the same structured LBP feature
# record format used elsewhere in the project.
from eye_lbp_dataset import prepare_runtime_feature_record

# This helper performs classifier inference for one runtime feature record and
# returns a structured prediction record.
from eye_lbp_classifier import predict_from_runtime_feature_record


# ---------------------------------------------------------------------
# LBP runtime configuration
# ---------------------------------------------------------------------

# This threshold is used when two or more eye predictions must be aggregated
# and their hard labels tie. In that situation, the mean open-score becomes the
# tie-break signal.
LBP_FRAME_OPEN_THRESHOLD = 0.50

# This constant is currently part of the runtime configuration vocabulary even
# though the present implementation does not actively use it in the final logic.
# It documents the notion of a "strong open" confidence region for future or
# alternative aggregation strategies.
LBP_STRONG_OPEN_SCORE_THRESHOLD = 0.70

# Final frame-level fallback label used when no better temporal or heuristic
# decision is available.
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

    # Missing labels are normalized to "unknown" so later logic can explicitly
    # detect that no canonical open/close decision exists.
    if label is None:
        return "unknown"

    # Normalize casing and surrounding whitespace once so the rest of the
    # runtime logic only needs to compare against canonical label names.
    value = str(label).strip().lower()

    # Accept a few common textual variants and collapse them into the project's
    # standard "open" label.
    if value in ("open", "opened"):
        return "open"

    # Accept a few common textual variants and collapse them into the project's
    # standard "close" label.
    if value in ("close", "closed", "shut"):
        return "close"

    # Preserve any nonstandard value as-is so it can still be inspected by
    # callers or filtered out later.
    return value


def _get_classifier_name_from_model_bundle(model_bundle):
    """
    Read the classifier name from the model bundle when available.
    """

    # If the incoming object is not even a dictionary, there is no valid
    # classifier config to inspect.
    if not isinstance(model_bundle, dict):
        return None

    # Read the classifier config sub-dictionary in a tolerant way so this helper
    # can be used safely even on partially malformed bundles.
    classifier_config = model_bundle.get("classifier_config", {})
    if not isinstance(classifier_config, dict):
        return None

    classifier_name = classifier_config.get("classifier_name")
    if classifier_name is None:
        return None

    # Return the normalized textual classifier name because later logic uses it
    # to interpret score semantics.
    return str(classifier_name).strip().lower()


def _validate_model_bundle(model_bundle):
    """
    Validate that the runtime model bundle contains the minimum fields
    required for LBP inference.
    """

    # Runtime inference expects the standard model-bundle structure built by the
    # training/classifier pipeline, so the top-level object must be a dictionary.
    if not isinstance(model_bundle, dict):
        raise TypeError("model_bundle must be a dictionary.")

    # These are the minimum pieces required for runtime LBP inference:
    # - the trained model itself,
    # - the preprocessing config needed to preprocess runtime eye ROIs,
    # - the LBP config needed to compute runtime descriptors,
    # - the classifier config needed to interpret scores consistently.
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

    # Normalize the hard label first because even when no usable numeric score
    # exists, the label still provides a binary fallback.
    predicted_label = int(predicted_label)

    # If the classifier did not return any score, reduce the decision to a hard
    # binary open-score:
    # - open  -> 1.0
    # - close -> 0.0
    if predicted_score is None:
        return 1.0 if predicted_label == 1 else 0.0

    predicted_score = float(predicted_score)
    classifier_name = (classifier_name or "").strip().lower()

    # For KNN, the score returned by the classifier layer is already intended to
    # represent the probability-like confidence for class 1 ("open"), so only
    # clamping is needed.
    if classifier_name == "knn":
        return max(0.0, min(1.0, predicted_score))

    # For linear SVM, the classifier layer returns a decision-function value.
    # Apply a sigmoid so aggregation can work with a bounded [0, 1] open-score.
    if classifier_name == "linear_svm":
        return 1.0 / (1.0 + math.exp(-predicted_score))

    # Any other classifier type falls back to hard-label interpretation so the
    # runtime code still has one consistent open-score scale.
    return 1.0 if predicted_label == 1 else 0.0


def _build_empty_eye_prediction(eye_box, eye_index, reason):
    """
    Build a standard failed-eye-prediction record.
    """

    # This standardized record shape allows later aggregation to handle failed
    # eye predictions in exactly the same list structure as successful ones.
    # The success flag and reason field make the failure explicit, while the
    # remaining prediction fields are kept as None.
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

    # Validate that the incoming model bundle contains everything needed for
    # runtime LBP inference before doing any image work.
    validated_model_bundle = _validate_model_bundle(model_bundle)

    # Reuse the existing ROI extraction logic from the earlier heuristic module.
    # This keeps the eye-crop definition identical across both heuristic and
    # LBP-based runtime paths.
    eye_roi = extract_eye_roi(gray_frame, eye_box)

    # If the eye ROI cannot be extracted or is empty, return one standard failed
    # prediction record instead of throwing away the eye silently.
    if eye_roi is None or eye_roi.size == 0:
        return _build_empty_eye_prediction(
            eye_box=eye_box,
            eye_index=eye_index,
            reason="invalid_eye_roi",
        )

    # Convert the raw runtime eye ROI into one structured runtime feature record.
    # This runs the shared runtime-side pipeline:
    # eye ROI -> preprocessing -> LBP features -> runtime feature record
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

    # Run the trained classifier on that runtime feature record and get back a
    # structured prediction record.
    prediction_record = predict_from_runtime_feature_record(
        model=validated_model_bundle["model"],
        runtime_feature_record=runtime_feature_record,
    )

    # Normalize the predicted class name into the runtime open/close vocabulary
    # used by the rest of this module.
    predicted_class_name = _normalize_eye_state_label(
        prediction_record.get("predicted_class_name")
    )

    predicted_label = prediction_record.get("predicted_label")
    predicted_score = prediction_record.get("predicted_score")

    # Read the classifier family so the numeric score can be interpreted
    # correctly and converted into one unified open-score.
    classifier_name = _get_classifier_name_from_model_bundle(validated_model_bundle)
    predicted_open_score = _convert_prediction_to_open_score(
        predicted_label=predicted_label,
        predicted_score=predicted_score,
        classifier_name=classifier_name,
    )

    # Mark the runtime prediction as successful and attach the normalized label
    # plus the unified open-score used later during frame-level aggregation.
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

    # Normalize both temporal and static fallback labels into the canonical
    # runtime vocabulary before the aggregation logic begins.
    normalized_previous_eye_state = _normalize_eye_state_label(previous_eye_state)
    normalized_fallback_frame_label = _normalize_eye_state_label(fallback_frame_label)

    # Only successful records with canonical "open" or "close" labels should
    # participate in the final frame-level decision.
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

    # -------------------------------------------------------------
    # Case 1: no valid eye predictions
    # -------------------------------------------------------------
    #
    # If there is no usable eye-level evidence at all, prefer temporal
    # continuity through previous_eye_state if it is valid.
    # Otherwise fall back to the configured frame-level default label.
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

    # -------------------------------------------------------------
    # Case 2: exactly one valid eye prediction
    # -------------------------------------------------------------
    #
    # When only one valid eye is available, its decision becomes the frame-level
    # decision directly.
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

    # -------------------------------------------------------------
    # Case 3: two or more valid eye predictions
    # -------------------------------------------------------------
    #
    # First count hard open/close votes and collect open-scores for tie-breaking.
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

    # Mean open-score is the secondary soft-evidence signal used when hard votes
    # tie.
    mean_open_score = sum(open_scores) / max(1, len(open_scores))

    # Primary rule: majority of hard labels wins.
    if open_eye_count > close_eye_count:
        frame_label = "open"
    elif close_eye_count > open_eye_count:
        frame_label = "close"
    else:
        # Secondary rule: use mean open-score to resolve hard-label ties.
        if mean_open_score > LBP_FRAME_OPEN_THRESHOLD:
            frame_label = "open"
        elif mean_open_score < LBP_FRAME_OPEN_THRESHOLD:
            frame_label = "close"
        elif normalized_previous_eye_state in ("open", "close"):
            # Exact tie: prefer temporal continuity if available.
            frame_label = normalized_previous_eye_state
        else:
            # Final tie fallback: use the configured frame-level default label.
            frame_label = normalized_fallback_frame_label

    # Return both the aggregated frame label and a compact explanation of how
    # the aggregation resolved.
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

    # If face_parts is missing completely, replace it with the minimal expected
    # structure so the rest of the function can operate safely.
    if face_parts is None:
        face_parts = {"eyes": [], "mouth": []}

    # Read the detected eye boxes from the shared face-parts structure.
    eye_boxes = face_parts.get("eyes", [])

    # -------------------------------------------------------------
    # Branch 1: no model bundle available
    # -------------------------------------------------------------
    #
    # If the trained model bundle is missing, the module can still:
    # - fall back to the older heuristic frame-level classifier, or
    # - fall back to temporal/default label logic if heuristic fallback is disabled.
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

    # -------------------------------------------------------------
    # Branch 2: model bundle available
    # -------------------------------------------------------------
    #
    # Build one eye-level prediction record for each detected eye box.
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

    # Aggregate the eye-level predictions into one frame-level label.
    frame_label, aggregation_details = aggregate_eye_predictions(
        eye_prediction_records=eye_prediction_records,
        previous_eye_state=previous_eye_state,
        fallback_frame_label=fallback_frame_label,
    )

    used_heuristic_fallback = False

    # Optional secondary fallback:
    # if the LBP path produced zero valid eye predictions, try the older
    # heuristic classifier and use it only when it returns a canonical label.
    if fallback_to_heuristic and aggregation_details["valid_eye_count"] == 0:
        heuristic_label = _normalize_eye_state_label(
            classify_eye_state_heuristic(gray_frame, face_parts)
        )

        if heuristic_label in ("open", "close"):
            frame_label = heuristic_label
            used_heuristic_fallback = True

    # Optionally return both the final label and a detailed explanation of how
    # the runtime decision was produced.
    if return_details:
        details = {
            "used_model_bundle": True,
            "used_heuristic_fallback": used_heuristic_fallback,
            "eye_prediction_records": eye_prediction_records,
            "aggregation": aggregation_details,
        }
        return frame_label, details

    return frame_label