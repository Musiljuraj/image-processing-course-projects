# This module is the supervised-learning layer of the LBP pipeline.
# Everything before this point in the project is about turning raw eye images
# into a stable numeric representation:
#
#     training image / runtime eye ROI
#         -> preprocessing
#         -> LBP feature vector
#
# This file is the layer that starts using those feature vectors as classifier
# inputs:
#
#     LBP feature vectors
#         -> validated training matrix / runtime matrix
#         -> trained classifier model
#         -> predicted labels
#         -> optional confidence-like scores
#         -> structured prediction records
#
# The module is intentionally kept separate from:
# - dataset loading details,
# - preprocessing internals,
# - LBP extraction internals,
# - frame-level aggregation logic.
#
# That separation keeps the classifier part of the project testable in
# isolation and makes it easier to swap or compare classifiers later.

"""
eye_lbp_classifier.py

This module trains and applies supervised classifiers for LBP-based
eye-state recognition.

Its responsibilities are:
- validating classifier configuration,
- building the selected classifier model,
- training the model on X_train, y_train,
- predicting labels for feature matrices,
- optionally returning confidence-like scores,
- optionally attaching predictions back to structured feature records,
- providing lightweight helpers for single runtime samples,
- building one startup-trained model bundle for later runtime use.

Binary class convention used in this module:
- 0 = close
- 1 = open

The module is intentionally separate from:
- dataset loading,
- preprocessing,
- LBP feature extraction,
- runtime frame aggregation,
- final evaluation.

This keeps the classifier logic isolated and easy to test.
"""

# deepcopy is used whenever normalized configuration dictionaries are stored
# inside output structures such as the final model bundle. That way, the saved
# bundle keeps its own stable copy of the exact configs that produced it.
from copy import deepcopy

# NumPy is used here as the standard numeric interface for:
# - validated feature matrices,
# - validated label vectors,
# - prediction arrays,
# - summary statistics.
import numpy as np

# The project currently supports a small controlled classifier set:
# - KNN for a simple neighborhood-based baseline
# - linear SVM for a margin-based linear classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# The startup model-building helper reuses the lower training-data pipeline:
# - load structured training records from disk,
# - validate preprocessing config,
# - validate LBP config,
# - prepare training matrix and labels from records.
from eye_training_io import load_all_eye_training_records
from eye_preprocessing import validate_preprocessing_config
from lbp_features import validate_lbp_config
from eye_lbp_dataset import prepare_training_matrix_and_labels


# Supported classifier names are centralized here so config validation and
# user-facing error messages use one consistent vocabulary.
SUPPORTED_CLASSIFIER_NAMES = {"knn", "linear_svm"}

# This is the canonical mapping from the project's binary numeric labels to the
# textual class names used in reports and runtime predictions.
EYE_LABEL_TO_CLASS_NAME = {
    0: "close",
    1: "open",
}


# ---------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------

def get_default_classifier_config():
    """
    Return a fresh default classifier configuration dictionary.

    The default is intentionally simple and suitable for the first working
    version of the assignment.
    """

    # The project defaults to a small KNN model because it is simple, easy to
    # reason about, and works as a clean first baseline for the LBP features.
    return {
        "classifier_name": "knn",
        "n_neighbors": 3,
    }


def normalize_classifier_name(classifier_name):
    """
    Normalize the textual classifier name.

    Example accepted values:
    - "knn"
    - "linear_svm"
    """

    # Classifier selection is string-based, so normalize once here and keep the
    # later validation/dispatch logic simple and predictable.
    if not isinstance(classifier_name, str):
        raise TypeError("classifier_name must be a string.")

    return classifier_name.strip().lower()


def validate_classifier_config(classifier_config=None):
    """
    Validate and normalize classifier configuration.

    Supported configurations:
    - knn:
        {
            "classifier_name": "knn",
            "n_neighbors": 3
        }

    - linear_svm:
        {
            "classifier_name": "linear_svm",
            "C": 1.0
        }

    Missing values are filled from defaults where appropriate.
    """

    # Start from project defaults so omitted keys still get stable values.
    config = get_default_classifier_config()

    # Then merge any caller-provided config values on top of the defaults.
    if classifier_config is not None:
        if not isinstance(classifier_config, dict):
            raise TypeError("classifier_config must be a dictionary.")
        config.update(classifier_config)

    # Read and normalize the requested classifier family first.
    classifier_name = config.get("classifier_name", "knn")
    normalized_classifier_name = normalize_classifier_name(classifier_name)

    # Reject unsupported classifier names early so later code never has to deal
    # with partially valid config structures.
    if normalized_classifier_name not in SUPPORTED_CLASSIFIER_NAMES:
        raise ValueError(
            "Unsupported classifier_name. Expected one of: "
            f"{sorted(SUPPORTED_CLASSIFIER_NAMES)}. "
            f"Got: {classifier_name}"
        )

    # -------------------------------------------------------------
    # KNN configuration branch
    # -------------------------------------------------------------
    #
    # KNN needs only the neighbor count in the current project setup.
    if normalized_classifier_name == "knn":
        n_neighbors = config.get("n_neighbors", 3)

        if not isinstance(n_neighbors, int):
            raise TypeError("n_neighbors must be an integer.")

        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")

        return {
            "classifier_name": normalized_classifier_name,
            "n_neighbors": n_neighbors,
        }

    # -------------------------------------------------------------
    # Linear SVM configuration branch
    # -------------------------------------------------------------
    #
    # The current SVM setup uses a linear kernel and exposes only C, the
    # regularization strength, as the main tunable parameter.
    if normalized_classifier_name == "linear_svm":
        C = config.get("C", 1.0)

        if not isinstance(C, (int, float)):
            raise TypeError("C must be a number.")

        if C <= 0:
            raise ValueError("C must be positive.")

        return {
            "classifier_name": normalized_classifier_name,
            "C": float(C),
        }

    raise ValueError(
        f"Unsupported classifier_name after validation: {normalized_classifier_name}"
    )


def build_classifier_model(classifier_config=None):
    """
    Build an untrained classifier model from configuration.

    Supported models:
    - knn ........ KNeighborsClassifier
    - linear_svm . SVC with linear kernel
    """

    # First normalize the configuration into one known-good classifier spec.
    validated_config = validate_classifier_config(classifier_config)
    classifier_name = validated_config["classifier_name"]

    # Build the selected sklearn model object but do not train it yet.
    if classifier_name == "knn":
        model = KNeighborsClassifier(
            n_neighbors=validated_config["n_neighbors"]
        )
        return model

    if classifier_name == "linear_svm":
        model = SVC(
            kernel="linear",
            C=validated_config["C"],
            probability=False,
        )
        return model

    raise ValueError(
        f"Unsupported classifier_name after validation: {classifier_name}"
    )


# ---------------------------------------------------------------------
# Validation helpers for matrices and labels
# ---------------------------------------------------------------------

def _validate_feature_matrix(X, matrix_name="X"):
    """
    Validate a classifier feature matrix.

    Returns:
    - X_validated ... 2D NumPy array of dtype float32
    """

    # Convert the input into the project's standard numeric matrix format.
    X_validated = np.asarray(X, dtype=np.float32)

    # The classifier layer always expects a 2D matrix:
    # (number_of_samples, number_of_features)
    if X_validated.ndim != 2:
        raise ValueError(f"{matrix_name} must be a 2D array.")

    # A training/test matrix without rows is invalid.
    if X_validated.shape[0] == 0:
        raise ValueError(f"{matrix_name} must contain at least one sample.")

    # A matrix without columns would mean samples have no features at all.
    if X_validated.shape[1] == 0:
        raise ValueError(f"{matrix_name} must contain at least one feature.")

    # Non-finite values would break model training or prediction semantics.
    if not np.isfinite(X_validated).all():
        raise ValueError(f"{matrix_name} contains non-finite values.")

    return X_validated


def _validate_feature_vector(feature_vector, vector_name="feature_vector"):
    """
    Validate one single feature vector and convert it to 2D matrix form.

    Returns:
    - feature_vector_validated ... 1D float32 vector
    - X_single ................... 2D array of shape (1, number_of_features)
    """

    # Convert the input to the standard 1D float32 representation first.
    feature_vector_validated = np.asarray(feature_vector, dtype=np.float32)

    # Single-sample helpers require a one-dimensional feature vector, not a
    # matrix and not a nested structure.
    if feature_vector_validated.ndim != 1:
        raise ValueError(f"{vector_name} must be a 1D array.")

    if feature_vector_validated.shape[0] == 0:
        raise ValueError(f"{vector_name} must contain at least one feature.")

    if not np.isfinite(feature_vector_validated).all():
        raise ValueError(f"{vector_name} contains non-finite values.")

    # Many sklearn prediction methods still expect a 2D matrix, even for one
    # sample, so reshape the validated vector into (1, feature_count).
    X_single = feature_vector_validated.reshape(1, -1).astype(np.float32)

    return feature_vector_validated, X_single


def _validate_label_vector(
    y,
    expected_length=None,
    vector_name="y",
    require_two_classes=False,
):
    """
    Validate a classifier label vector.

    Inputs:
    - y
    - expected_length
    - vector_name
    - require_two_classes:
        if True, at least two distinct classes must be present

    Returns:
    - y_validated ... 1D NumPy array of dtype int64
    """

    # Convert the incoming labels to the project's standard integer vector form.
    y_validated = np.asarray(y, dtype=np.int64)

    # Labels must form a simple 1D vector aligned with matrix rows.
    if y_validated.ndim != 1:
        raise ValueError(f"{vector_name} must be a 1D array.")

    if y_validated.shape[0] == 0:
        raise ValueError(f"{vector_name} must contain at least one label.")

    # When expected_length is provided, enforce row-wise alignment with the
    # associated feature matrix.
    if expected_length is not None and y_validated.shape[0] != expected_length:
        raise ValueError(
            f"{vector_name} length ({y_validated.shape[0]}) does not match "
            f"expected length ({expected_length})."
        )

    # This project is strictly binary, so only labels 0 and 1 are valid.
    unique_labels = np.unique(y_validated)

    if not set(unique_labels.tolist()).issubset({0, 1}):
        raise ValueError(
            f"{vector_name} must contain only binary labels 0 and 1. "
            f"Got: {unique_labels.tolist()}"
        )

    # Training requires at least two classes to be present, otherwise the model
    # would not learn a meaningful binary decision boundary.
    if require_two_classes and unique_labels.shape[0] < 2:
        raise ValueError(
            f"{vector_name} must contain at least two classes for training. "
            f"Got: {unique_labels.tolist()}"
        )

    return y_validated


# ---------------------------------------------------------------------
# Training and prediction
# ---------------------------------------------------------------------

def train_classifier(X_train, y_train, classifier_config=None):
    """
    Train a classifier on the provided training data.

    Inputs:
    - X_train ... 2D feature matrix
    - y_train ... 1D label vector

    Return:
    - trained_model
    """

    # Validate both the feature matrix and label vector first so model fitting
    # always starts from well-formed numeric data.
    X_train_validated = _validate_feature_matrix(X_train, matrix_name="X_train")
    y_train_validated = _validate_label_vector(
        y_train,
        expected_length=X_train_validated.shape[0],
        vector_name="y_train",
        require_two_classes=True,
    )

    # Build the selected classifier family and then fit it on the validated
    # training data.
    model = build_classifier_model(classifier_config)
    model.fit(X_train_validated, y_train_validated)

    return model


def predict_labels(model, X_test):
    """
    Predict class labels for a feature matrix.

    Return:
    - predicted_labels ... 1D NumPy array containing only 0 and 1
    """

    # Prediction requires a real trained model object.
    if model is None:
        raise ValueError("model must not be None.")

    # The model must support the standard sklearn-style predict(...) interface.
    if not hasattr(model, "predict"):
        raise TypeError("model does not provide a predict(...) method.")

    # Validate the test matrix before calling the model.
    X_test_validated = _validate_feature_matrix(X_test, matrix_name="X_test")

    predicted_labels = model.predict(X_test_validated)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)

    # The classifier output must also be a valid binary label vector.
    if predicted_labels.ndim != 1:
        raise ValueError("predicted_labels must be a 1D array.")

    if predicted_labels.shape[0] != X_test_validated.shape[0]:
        raise ValueError(
            "Prediction count does not match the number of test samples."
        )

    if not set(np.unique(predicted_labels).tolist()).issubset({0, 1}):
        raise ValueError(
            "Predicted labels must contain only 0 and 1. "
            f"Got: {np.unique(predicted_labels).tolist()}"
        )

    return predicted_labels


def predict_single_label(model, feature_vector):
    """
    Predict one label for one single feature vector.

    Return:
    - predicted_label ... integer 0 or 1
    """

    # Reuse the single-vector validator so one-sample prediction remains
    # consistent with the batch prediction interface.
    _, X_single = _validate_feature_vector(
        feature_vector,
        vector_name="feature_vector",
    )

    predicted_label = predict_labels(model, X_single)[0]
    return int(predicted_label)


def predict_scores(model, X_test):
    """
    Predict confidence-like scores for a feature matrix if supported.

    Score meaning:
    - if predict_proba(...) is available, score = probability of class 1
    - otherwise, if decision_function(...) is available, score = decision value
      for class 1
    - otherwise None is returned
    """

    # Scores are optional because not every classifier family exposes them in
    # the same way.
    if model is None:
        raise ValueError("model must not be None.")

    X_test_validated = _validate_feature_matrix(X_test, matrix_name="X_test")

    # -------------------------------------------------------------
    # Probability-based score path
    # -------------------------------------------------------------
    #
    # If the model exposes predict_proba(...), use the probability of class 1
    # ("open") as the confidence-like score.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test_validated)
        probabilities = np.asarray(probabilities, dtype=np.float32)

        if probabilities.ndim != 2 or probabilities.shape[0] != X_test_validated.shape[0]:
            raise ValueError("predict_proba(...) returned an unexpected shape.")

        # Prefer the column that explicitly corresponds to class 1 when the
        # model exposes class ordering; otherwise fall back to the last column.
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                positive_class_index = classes.index(1)
            else:
                positive_class_index = probabilities.shape[1] - 1
        else:
            positive_class_index = probabilities.shape[1] - 1

        predicted_scores = probabilities[:, positive_class_index]
        return predicted_scores.astype(np.float32)

    # -------------------------------------------------------------
    # Decision-function path
    # -------------------------------------------------------------
    #
    # If probabilities are unavailable but a decision function exists, use the
    # decision value corresponding to class 1 as the confidence-like score.
    if hasattr(model, "decision_function"):
        decision_values = model.decision_function(X_test_validated)
        decision_values = np.asarray(decision_values, dtype=np.float32)

        # Binary models often return a 1D decision-value vector.
        if decision_values.ndim == 1:
            if decision_values.shape[0] != X_test_validated.shape[0]:
                raise ValueError(
                    "decision_function(...) returned an unexpected shape."
                )
            return decision_values.astype(np.float32)

        # Multi-column decision outputs are also handled by selecting the column
        # corresponding to class 1 when available.
        if decision_values.ndim == 2:
            if decision_values.shape[0] != X_test_validated.shape[0]:
                raise ValueError(
                    "decision_function(...) returned an unexpected shape."
                )

            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    positive_class_index = classes.index(1)
                else:
                    positive_class_index = decision_values.shape[1] - 1
            else:
                positive_class_index = decision_values.shape[1] - 1

            return decision_values[:, positive_class_index].astype(np.float32)

        raise ValueError(
            "decision_function(...) returned an unsupported number of dimensions."
        )

    return None


def predict_single_score(model, feature_vector):
    """
    Predict one confidence-like score for one single feature vector.

    Return:
    - predicted_score ... float or None
    """

    # Validate and reshape the single feature vector into matrix form, then
    # reuse the batch score path.
    _, X_single = _validate_feature_vector(
        feature_vector,
        vector_name="feature_vector",
    )

    predicted_scores = predict_scores(model, X_single)

    if predicted_scores is None:
        return None

    return float(predicted_scores[0])


# ---------------------------------------------------------------------
# Structured prediction helpers
# ---------------------------------------------------------------------

def label_to_class_name(label):
    """
    Convert numeric binary label into class name.

    Mapping:
    - 0 -> close
    - 1 -> open
    """

    # This helper centralizes the project's binary label vocabulary so all
    # downstream prediction records use the same textual class names.
    if label not in EYE_LABEL_TO_CLASS_NAME:
        raise ValueError(
            f"Unsupported binary label: {label}. "
            f"Expected one of: {sorted(EYE_LABEL_TO_CLASS_NAME.keys())}"
        )

    return EYE_LABEL_TO_CLASS_NAME[int(label)]


def build_prediction_records(
    feature_records,
    predicted_labels,
    predicted_scores=None,
):
    """
    Attach predictions back to structured feature records.

    Returned records contain:
    - original feature-record data
    - predicted_label
    - predicted_class_name
    - optionally predicted_score
    """

    # This helper exists so classifier outputs can be reattached to the
    # structured feature-record format used in the rest of the pipeline.
    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    predicted_labels_array = np.asarray(predicted_labels, dtype=np.int64)

    if predicted_labels_array.ndim != 1:
        raise ValueError("predicted_labels must be a 1D array.")

    if len(feature_records) != predicted_labels_array.shape[0]:
        raise ValueError(
            "feature_records length does not match predicted_labels length."
        )

    predicted_scores_array = None

    # When scores are supplied, they must align one-to-one with the labels and
    # feature records.
    if predicted_scores is not None:
        predicted_scores_array = np.asarray(predicted_scores, dtype=np.float32)

        if predicted_scores_array.ndim != 1:
            raise ValueError("predicted_scores must be a 1D array if provided.")

        if predicted_scores_array.shape[0] != predicted_labels_array.shape[0]:
            raise ValueError(
                "predicted_scores length does not match predicted_labels length."
            )

    prediction_records = []

    # Build one enriched prediction record per input feature record while
    # preserving all original metadata and feature fields.
    for index, feature_record in enumerate(feature_records):
        if not isinstance(feature_record, dict):
            raise TypeError("Each feature_record must be a dictionary.")

        predicted_label = int(predicted_labels_array[index])

        prediction_record = {
            **feature_record,
            "predicted_label": predicted_label,
            "predicted_class_name": label_to_class_name(predicted_label),
        }

        if predicted_scores_array is not None:
            prediction_record["predicted_score"] = float(predicted_scores_array[index])

        prediction_records.append(prediction_record)

    return prediction_records


def predict_from_runtime_feature_record(model, runtime_feature_record):
    """
    Predict label and optional score for one runtime feature record.

    The input record is expected to contain:
    - lbp_feature_vector

    The returned record contains:
    - original runtime feature record data
    - predicted_label
    - predicted_class_name
    - optionally predicted_score
    """

    # Runtime inference often starts from one already-prepared runtime feature
    # record containing exactly one LBP feature vector.
    if not isinstance(runtime_feature_record, dict):
        raise TypeError("runtime_feature_record must be a dictionary.")

    if "lbp_feature_vector" not in runtime_feature_record:
        raise KeyError(
            "runtime_feature_record must contain 'lbp_feature_vector'."
        )

    # Predict the hard class label first.
    predicted_label = predict_single_label(
        model=model,
        feature_vector=runtime_feature_record["lbp_feature_vector"],
    )

    # Then try to obtain an optional score from the same single feature vector.
    predicted_score = predict_single_score(
        model=model,
        feature_vector=runtime_feature_record["lbp_feature_vector"],
    )

    # Build one enriched runtime prediction record that preserves the original
    # runtime metadata and feature information.
    prediction_record = {
        **runtime_feature_record,
        "predicted_label": predicted_label,
        "predicted_class_name": label_to_class_name(predicted_label),
    }

    if predicted_score is not None:
        prediction_record["predicted_score"] = predicted_score

    return prediction_record


# ---------------------------------------------------------------------
# Startup training helper
# ---------------------------------------------------------------------

def build_eye_lbp_model(
    dataset_root,
    preprocessing_config=None,
    lbp_config=None,
    classifier_config=None,
    load_images=True,
    grayscale=True,
    recursive=True,
    ignore_invalid_files=False,
    image_key="image",
):
    """
    Build one startup-trained LBP eye-state model bundle.

    High-level pipeline:
    - load all eye training records from dataset_root
    - preprocess them
    - extract LBP features
    - build X_train, y_train, metadata
    - train the classifier
    - return a reusable model bundle

    Inputs:
    - dataset_root ............. path to mrlEyes_2018_01 root
    - preprocessing_config ..... optional preprocessing configuration
    - lbp_config ............... optional LBP configuration
    - classifier_config ........ optional classifier configuration
    - load_images .............. passed to dataset loader
    - grayscale ................ passed to dataset loader
    - recursive ................ passed to dataset loader
    - ignore_invalid_files ..... passed to dataset loader
    - image_key ................ training image key, default: "image"

    Return:
    - model_bundle dictionary containing:
        model
        preprocessing_config
        lbp_config
        classifier_config
        dataset_root
        training_sample_count
        feature_count
        X_train_shape
        y_train_shape
        class_counts
        training_metadata
        training_feature_records
        X_train
        y_train
    """

    # Normalize all three configuration layers first so the final model bundle
    # stores one clean, explicit version of the configs actually used.
    normalized_preprocessing_config = validate_preprocessing_config(preprocessing_config)
    normalized_lbp_config = validate_lbp_config(lbp_config)
    normalized_classifier_config = validate_classifier_config(classifier_config)

    # -------------------------------------------------------------
    # Step 1: load structured training records from disk
    # -------------------------------------------------------------
    #
    # This reuses the training-I/O module, which understands the dataset
    # directory structure and filename-encoded metadata.
    training_records = load_all_eye_training_records(
        dataset_root=dataset_root,
        load_images=load_images,
        grayscale=grayscale,
        recursive=recursive,
        ignore_invalid_files=ignore_invalid_files,
    )

    # -------------------------------------------------------------
    # Step 2: build the full training representation
    # -------------------------------------------------------------
    #
    # This runs the lower pipeline:
    # training records
    #     -> preprocessing
    #     -> LBP feature records
    #     -> X_train, y_train, metadata
    X_train, y_train, training_metadata, training_feature_records = prepare_training_matrix_and_labels(
        training_records=training_records,
        preprocessing_config=normalized_preprocessing_config,
        lbp_config=normalized_lbp_config,
        image_key=image_key,
    )

    # -------------------------------------------------------------
    # Step 3: train the classifier on the prepared training matrix
    # -------------------------------------------------------------
    trained_model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=normalized_classifier_config,
    )

    # Compute simple class-count statistics so the resulting model bundle also
    # documents the class balance it was trained on.
    class_counts = {
        "close": int(np.sum(y_train == 0)),
        "open": int(np.sum(y_train == 1)),
    }

    # -------------------------------------------------------------
    # Step 4: assemble the reusable model bundle
    # -------------------------------------------------------------
    #
    # The model bundle is the standard project object used later by runtime
    # modules such as eye_state_lbp.py.
    #
    # It keeps:
    # - the trained model itself,
    # - the exact configs used,
    # - the dataset root,
    # - dimensions and class counts,
    # - aligned metadata and feature records,
    # - and the full training matrices for later inspection/debugging.
    model_bundle = {
        "model": trained_model,
        "preprocessing_config": deepcopy(normalized_preprocessing_config),
        "lbp_config": deepcopy(normalized_lbp_config),
        "classifier_config": deepcopy(normalized_classifier_config),
        "dataset_root": str(dataset_root),
        "training_sample_count": int(X_train.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "X_train_shape": tuple(int(value) for value in X_train.shape),
        "y_train_shape": tuple(int(value) for value in y_train.shape),
        "class_counts": class_counts,
        "training_metadata": training_metadata,
        "training_feature_records": training_feature_records,
        "X_train": X_train,
        "y_train": y_train,
    }

    return model_bundle


# ---------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------

def train_and_predict(X_train, y_train, X_test, classifier_config=None):
    """
    Train a classifier and predict on test data in one convenience call.

    Return:
    - trained_model
    - predicted_labels
    - predicted_scores
    """

    # This helper is mainly for quick experiments or compact test code where
    # training and immediate prediction should happen in one call.
    trained_model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=classifier_config,
    )

    predicted_labels = predict_labels(
        model=trained_model,
        X_test=X_test,
    )

    predicted_scores = predict_scores(
        model=trained_model,
        X_test=X_test,
    )

    return trained_model, predicted_labels, predicted_scores


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------

def summarize_predictions(predicted_labels, ground_truth_labels=None):
    """
    Summarize predicted labels and optionally compare them with ground truth.

    Inputs:
    - predicted_labels
    - ground_truth_labels ... optional binary labels aligned with predictions

    Return:
    - summary dictionary containing:
        predicted_count
        predicted_close_count
        predicted_open_count
        has_ground_truth
        and, when ground_truth_labels are provided:
            compared_count
            correct_count
            accuracy_percent
            close_as_close
            close_as_open
            open_as_close
            open_as_open

    Binary convention:
    - 0 = close
    - 1 = open
    """

    # Validate the predicted labels first, even in the no-ground-truth case,
    # so the summary always reflects a proper binary prediction vector.
    predicted_labels_validated = _validate_label_vector(
        predicted_labels,
        vector_name="predicted_labels",
        require_two_classes=False,
    )

    summary = {
        "predicted_count": int(predicted_labels_validated.shape[0]),
        "predicted_close_count": int(np.sum(predicted_labels_validated == 0)),
        "predicted_open_count": int(np.sum(predicted_labels_validated == 1)),
        "has_ground_truth": ground_truth_labels is not None,
    }

    # If no reference labels were provided, return only the prediction-side
    # counts.
    if ground_truth_labels is None:
        return summary

    # Validate the reference labels and enforce row-wise alignment with the
    # predicted label vector.
    ground_truth_validated = _validate_label_vector(
        ground_truth_labels,
        expected_length=predicted_labels_validated.shape[0],
        vector_name="ground_truth_labels",
        require_two_classes=False,
    )

    # Compute overall correctness first.
    correct_mask = (predicted_labels_validated == ground_truth_validated)
    correct_count = int(np.sum(correct_mask))
    compared_count = int(predicted_labels_validated.shape[0])

    if compared_count == 0:
        accuracy_percent = 0.0
    else:
        accuracy_percent = 100.0 * correct_count / compared_count

    # Compute confusion-style binary counts under the project's convention:
    # - 0 = close
    # - 1 = open
    close_as_close = int(np.sum(
        (ground_truth_validated == 0) & (predicted_labels_validated == 0)
    ))
    close_as_open = int(np.sum(
        (ground_truth_validated == 0) & (predicted_labels_validated == 1)
    ))
    open_as_close = int(np.sum(
        (ground_truth_validated == 1) & (predicted_labels_validated == 0)
    ))
    open_as_open = int(np.sum(
        (ground_truth_validated == 1) & (predicted_labels_validated == 1)
    ))

    # Extend the summary with comparison results only when ground truth exists.
    summary.update({
        "compared_count": compared_count,
        "correct_count": correct_count,
        "accuracy_percent": float(accuracy_percent),
        "close_as_close": close_as_close,
        "close_as_open": close_as_open,
        "open_as_close": open_as_close,
        "open_as_open": open_as_open,
    })

    return summary