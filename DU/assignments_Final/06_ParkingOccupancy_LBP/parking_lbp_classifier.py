"""
parking_lbp_classifier.py

Purpose of this module:
- train and apply supervised classifiers for parking occupancy recognition
- keep classifier logic separate from dataset loading, preprocessing,
  LBP feature extraction, evaluation, and experiment orchestration

Why this module exists:
At this stage of the project, earlier modules can already produce:
- training feature matrix X_train
- training label vector y_train
- test feature matrix X_test
- metadata aligned with feature rows

The next logical stage is to:
1. validate classifier configuration
2. build the selected model
3. train the model on X_train, y_train
4. predict labels for X_test
5. optionally provide confidence-like scores
6. optionally attach predictions back to structured records

Supported classifier types:
- knn
- linear_svm

This module currently provides:
- normalize_classifier_name(...)
- validate_classifier_config(...)
- build_classifier_model(...)
- train_classifier(...)
- predict_labels(...)
- predict_scores(...)
- build_prediction_records(...)
- train_and_predict(...)
- summarize_predictions(...)
"""
# ---------------------------------------------------------------------------
# Module orientation:
# This module is the supervised learning stage of the parking LBP pipeline.
# Upstream code prepares feature matrices and aligned metadata; this module
# validates classifier settings, builds the model, fits it on training data,
# predicts labels for test samples, and optionally attaches those predictions
# back to structured records. The same binary class convention is used across
# the project: free = 0 and full / occupied = 1.
# ---------------------------------------------------------------------------

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


SUPPORTED_CLASSIFIER_NAMES = {"knn", "linear_svm"}


def normalize_classifier_name(classifier_name):
    """
    Normalize the textual classifier name.

    Input:
        classifier_name ... string such as:
                            "knn", "linear_svm"

    Return:
        normalized_classifier_name ... lowercase stripped string
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(classifier_name, str):
        raise TypeError("classifier_name must be a string.")

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return classifier_name.strip().lower()


def validate_classifier_config(classifier_config):
    """
    Validate and normalize classifier configuration.

    Input:
        classifier_config ... dictionary, for example:
                              {
                                  "classifier_name": "knn",
                                  "n_neighbors": 3
                              }
                              or
                              {
                                  "classifier_name": "linear_svm",
                                  "C": 1.0
                              }

    Return:
        validated_config ... normalized configuration dictionary

    Supported configurations:
    - knn:
        - classifier_name ... "knn"
        - n_neighbors ..... positive integer (default: 3)
    - linear_svm:
        - classifier_name ... "linear_svm"
        - C ............... positive number (default: 1.0)

    Why this function exists:
    Centralized validation keeps model-building and training code simpler and
    ensures that configuration errors are caught early and clearly.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(classifier_config, dict):
        raise TypeError("classifier_config must be a dictionary.")

    # Resolve configuration-dependent values into local variables here so the later
    # logic can use short, consistent names.
    classifier_name = classifier_config.get("classifier_name", "knn")
    # Resolve configuration-dependent values into local variables here so the later
    # logic can use short, consistent names.
    normalized_classifier_name = normalize_classifier_name(classifier_name)

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if normalized_classifier_name not in SUPPORTED_CLASSIFIER_NAMES:
        raise ValueError(
            "Unsupported classifier_name. Expected one of: "
            f"{sorted(SUPPORTED_CLASSIFIER_NAMES)}. "
            f"Got: {classifier_name}"
        )

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if normalized_classifier_name == "knn":
        n_neighbors = classifier_config.get("n_neighbors", 3)

        if not isinstance(n_neighbors, int):
            raise TypeError("n_neighbors must be an integer.")

        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")

        validated_config = {
            "classifier_name": normalized_classifier_name,
            "n_neighbors": n_neighbors,
        }
        return validated_config

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if normalized_classifier_name == "linear_svm":
        C = classifier_config.get("C", 1.0)

        if not isinstance(C, (int, float)):
            raise TypeError("C must be a number.")

        if C <= 0:
            raise ValueError("C must be positive.")

        validated_config = {
            "classifier_name": normalized_classifier_name,
            "C": float(C),
        }
        return validated_config

    raise ValueError(
        f"Unsupported classifier_name after validation: {normalized_classifier_name}"
    )


def build_classifier_model(classifier_config):
    """
    Build an untrained classifier model from configuration.

    Input:
        classifier_config ... classifier configuration dictionary

    Return:
        model ............... untrained classifier instance

    Supported models:
    - knn ........ KNeighborsClassifier
    - linear_svm . SVC with linear kernel
    """

    # Read configuration values and normalize them up front so the rest of the function
    # can rely on one stable internal convention.
    validated_config = validate_classifier_config(classifier_config)
    classifier_name = validated_config["classifier_name"]

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


def _validate_feature_matrix(X, matrix_name="X"):
    """
    Validate a classifier feature matrix.

    Input:
        X ............. expected to be a 2D NumPy-compatible array
        matrix_name ... name used in error messages

    Return:
        X_validated ... 2D NumPy array of dtype float32
    """

    # Convert incoming data into the internal NumPy representation first so the later
    # numerical operations are predictable and shape checks remain simple.
    X_validated = np.asarray(X, dtype=np.float32)

    if X_validated.ndim != 2:
        raise ValueError(f"{matrix_name} must be a 2D array.")

    if X_validated.shape[0] == 0:
        raise ValueError(f"{matrix_name} must contain at least one sample.")

    if X_validated.shape[1] == 0:
        raise ValueError(f"{matrix_name} must contain at least one feature.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not np.isfinite(X_validated).all():
        raise ValueError(f"{matrix_name} contains non-finite values.")

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return X_validated


def _validate_label_vector(y, expected_length=None, vector_name="y"):
    """
    Validate a classifier label vector.

    Inputs:
        y ................. expected to be a 1D NumPy-compatible array
        expected_length ... optional expected number of rows
        vector_name ....... name used in error messages

    Return:
        y_validated ....... 1D NumPy array of dtype int64
    """

    # Convert incoming data into the internal NumPy representation first so the later
    # numerical operations are predictable and shape checks remain simple.
    y_validated = np.asarray(y, dtype=np.int64)

    if y_validated.ndim != 1:
        raise ValueError(f"{vector_name} must be a 1D array.")

    if y_validated.shape[0] == 0:
        raise ValueError(f"{vector_name} must contain at least one label.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if expected_length is not None and y_validated.shape[0] != expected_length:
        raise ValueError(
            f"{vector_name} length ({y_validated.shape[0]}) does not match "
            f"expected length ({expected_length})."
        )

    unique_labels = np.unique(y_validated)

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not set(unique_labels.tolist()).issubset({0, 1}):
        raise ValueError(
            f"{vector_name} must contain only binary labels 0 and 1. "
            f"Got: {unique_labels.tolist()}"
        )

    if unique_labels.shape[0] < 2:
        raise ValueError(
            f"{vector_name} must contain at least two classes for training. "
            f"Got: {unique_labels.tolist()}"
        )

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return y_validated


def train_classifier(X_train, y_train, classifier_config):
    """
    Train a classifier on the provided training data.

    Inputs:
        X_train ............ 2D feature matrix of shape:
                             (number_of_samples, number_of_features)
        y_train ............ 1D label vector of shape:
                             (number_of_samples,)
        classifier_config .. classifier configuration dictionary

    Return:
        trained_model ...... fitted classifier instance

    Why this function exists:
    It provides one standard entry point for model training, keeping the rest
    of the project independent of classifier-specific fitting details.
    """

    # Read configuration values and normalize them up front so the rest of the function
    # can rely on one stable internal convention.
    X_train_validated = _validate_feature_matrix(X_train, matrix_name="X_train")
    # Resolve configuration-dependent values into local variables here so the later
    # logic can use short, consistent names.
    y_train_validated = _validate_label_vector(
        y_train,
        expected_length=X_train_validated.shape[0],
        vector_name="y_train",
    )

    model = build_classifier_model(classifier_config)
    model.fit(X_train_validated, y_train_validated)

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return model


def predict_labels(model, X_test):
    """
    Predict class labels for a feature matrix.

    Inputs:
        model ..... trained classifier instance
        X_test .... 2D feature matrix of shape:
                     (number_of_samples, number_of_features)

    Return:
        predicted_labels ... 1D NumPy array of predicted labels

    Label convention:
        0 = free
        1 = full
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if model is None:
        raise ValueError("model must not be None.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not hasattr(model, "predict"):
        raise TypeError("model does not provide a predict(...) method.")

    # Resolve configuration-dependent values into local variables here so the later
    # logic can use short, consistent names.
    X_test_validated = _validate_feature_matrix(X_test, matrix_name="X_test")

    predicted_labels = model.predict(X_test_validated)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)

    if predicted_labels.ndim != 1:
        raise ValueError("Predicted labels must be a 1D array.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if predicted_labels.shape[0] != X_test_validated.shape[0]:
        raise ValueError(
            "Prediction count does not match the number of test samples."
        )

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not set(np.unique(predicted_labels).tolist()).issubset({0, 1}):
        raise ValueError(
            "Predicted labels must contain only 0 and 1. "
            f"Got: {np.unique(predicted_labels).tolist()}"
        )

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return predicted_labels


def predict_scores(model, X_test):
    """
    Predict confidence-like scores for a feature matrix if supported.

    Inputs:
        model ..... trained classifier instance
        X_test .... 2D feature matrix

    Return:
        predicted_scores ... 1D NumPy array of scores aligned with X_test
                             or None if the model does not support score output

    Score meaning:
    - if predict_proba(...) is available, the score is the probability of class 1
    - otherwise, if decision_function(...) is available, the score is the
      decision value for class 1
    - otherwise None is returned
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if model is None:
        raise ValueError("model must not be None.")

    # Resolve configuration-dependent values into local variables here so the later
    # logic can use short, consistent names.
    X_test_validated = _validate_feature_matrix(X_test, matrix_name="X_test")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test_validated)

        probabilities = np.asarray(probabilities, dtype=np.float32)

        if probabilities.ndim != 2 or probabilities.shape[0] != X_test_validated.shape[0]:
            raise ValueError("predict_proba(...) returned an unexpected shape.")

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

    if hasattr(model, "decision_function"):
        decision_values = model.decision_function(X_test_validated)
        decision_values = np.asarray(decision_values, dtype=np.float32)

        if decision_values.ndim == 1:
            if decision_values.shape[0] != X_test_validated.shape[0]:
                raise ValueError(
                    "decision_function(...) returned an unexpected shape."
                )
            return decision_values.astype(np.float32)

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

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return None


def build_prediction_records(
    feature_records,
    predicted_labels,
    predicted_scores=None,
):
    """
    Attach predictions back to structured feature records.

    Inputs:
        feature_records ..... list of feature-record dictionaries
        predicted_labels ... 1D array-like of predicted labels
        predicted_scores ... optional 1D array-like of scores

    Return:
        prediction_records . list of enriched dictionaries containing:
                             - original feature-record data
                             - predicted_label
                             - optionally predicted_score

    Why this function exists:
    Structured prediction records are much easier to debug and evaluate than
    raw label arrays alone because they preserve metadata alongside outputs.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    predicted_labels_array = np.asarray(predicted_labels, dtype=np.int64)

    if predicted_labels_array.ndim != 1:
        raise ValueError("predicted_labels must be a 1D array.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if len(feature_records) != predicted_labels_array.shape[0]:
        raise ValueError(
            "feature_records length does not match predicted_labels length."
        )

    predicted_scores_array = None
    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if predicted_scores is not None:
        predicted_scores_array = np.asarray(predicted_scores, dtype=np.float32)

        if predicted_scores_array.ndim != 1:
            raise ValueError("predicted_scores must be a 1D array if provided.")

        if predicted_scores_array.shape[0] != predicted_labels_array.shape[0]:
            raise ValueError(
                "predicted_scores length does not match predicted_labels length."
            )

    # Start an accumulation structure that will be filled gradually as the function
    # walks through samples, records, rows, or files.
    prediction_records = []

    # Process items in a deterministic order so the produced outputs stay aligned with
    # the corresponding inputs, labels, or metadata.
    for index, feature_record in enumerate(feature_records):
        if not isinstance(feature_record, dict):
            raise TypeError("Each feature_record must be a dictionary.")

        prediction_record = {
            **feature_record,
            "predicted_label": int(predicted_labels_array[index]),
        }

        if predicted_scores_array is not None:
            prediction_record["predicted_score"] = float(
                predicted_scores_array[index]
            )

        prediction_records.append(prediction_record)

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return prediction_records


def train_and_predict(X_train, y_train, X_test, classifier_config):
    """
    Train a classifier and predict on test data in one convenience call.

    Inputs:
        X_train ............ 2D training feature matrix
        y_train ............ 1D training label vector
        X_test ............. 2D test feature matrix
        classifier_config .. classifier configuration dictionary

    Return:
        trained_model ...... fitted classifier instance
        predicted_labels ... 1D NumPy array of predicted labels
        predicted_scores ... 1D NumPy array of scores or None

    Why this function exists:
    It is a convenient wrapper for experiments and smoke tests where the full
    train->predict workflow is needed repeatedly.
    """

    # Set up the local working state first so the later processing steps can operate on
    # explicit, well-named intermediate values.
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

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return trained_model, predicted_labels, predicted_scores


def summarize_predictions(predicted_labels):
    """
    Summarize the distribution of predicted labels.

    Input:
        predicted_labels ... 1D array-like of predicted labels

    Return:
        summary ............ dictionary containing:
                             - total_count
                             - free_count
                             - full_count
                             - labels_present

    Why this function exists:
    Quick summaries are useful in smoke tests and debugging because they reveal
    whether the model predicts both classes or collapses into only one class.
    """

    # Convert incoming data into the internal NumPy representation first so the later
    # numerical operations are predictable and shape checks remain simple.
    predicted_labels_array = np.asarray(predicted_labels, dtype=np.int64)

    if predicted_labels_array.ndim != 1:
        raise ValueError("predicted_labels must be a 1D array.")

    unique_labels = np.unique(predicted_labels_array)

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not set(unique_labels.tolist()).issubset({0, 1}):
        raise ValueError(
            "predicted_labels must contain only 0 and 1. "
            f"Got: {unique_labels.tolist()}"
        )

    free_count = int(np.sum(predicted_labels_array == 0))
    full_count = int(np.sum(predicted_labels_array == 1))

    # Assemble the standard output dictionary here so downstream modules receive both
    # the computed values and the metadata needed for traceability.
    summary = {
        "total_count": int(predicted_labels_array.shape[0]),
        "free_count": free_count,
        "full_count": full_count,
        "labels_present": unique_labels.tolist(),
    }

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return summary