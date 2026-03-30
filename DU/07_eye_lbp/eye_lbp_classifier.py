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

from copy import deepcopy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from eye_training_io import load_all_eye_training_records
from eye_preprocessing import validate_preprocessing_config
from lbp_features import validate_lbp_config
from eye_lbp_dataset import prepare_training_matrix_and_labels


SUPPORTED_CLASSIFIER_NAMES = {"knn", "linear_svm"}

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

    config = get_default_classifier_config()

    if classifier_config is not None:
        if not isinstance(classifier_config, dict):
            raise TypeError("classifier_config must be a dictionary.")
        config.update(classifier_config)

    classifier_name = config.get("classifier_name", "knn")
    normalized_classifier_name = normalize_classifier_name(classifier_name)

    if normalized_classifier_name not in SUPPORTED_CLASSIFIER_NAMES:
        raise ValueError(
            "Unsupported classifier_name. Expected one of: "
            f"{sorted(SUPPORTED_CLASSIFIER_NAMES)}. "
            f"Got: {classifier_name}"
        )

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


# ---------------------------------------------------------------------
# Validation helpers for matrices and labels
# ---------------------------------------------------------------------

def _validate_feature_matrix(X, matrix_name="X"):
    """
    Validate a classifier feature matrix.

    Returns:
    - X_validated ... 2D NumPy array of dtype float32
    """

    X_validated = np.asarray(X, dtype=np.float32)

    if X_validated.ndim != 2:
        raise ValueError(f"{matrix_name} must be a 2D array.")

    if X_validated.shape[0] == 0:
        raise ValueError(f"{matrix_name} must contain at least one sample.")

    if X_validated.shape[1] == 0:
        raise ValueError(f"{matrix_name} must contain at least one feature.")

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

    feature_vector_validated = np.asarray(feature_vector, dtype=np.float32)

    if feature_vector_validated.ndim != 1:
        raise ValueError(f"{vector_name} must be a 1D array.")

    if feature_vector_validated.shape[0] == 0:
        raise ValueError(f"{vector_name} must contain at least one feature.")

    if not np.isfinite(feature_vector_validated).all():
        raise ValueError(f"{vector_name} contains non-finite values.")

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

    y_validated = np.asarray(y, dtype=np.int64)

    if y_validated.ndim != 1:
        raise ValueError(f"{vector_name} must be a 1D array.")

    if y_validated.shape[0] == 0:
        raise ValueError(f"{vector_name} must contain at least one label.")

    if expected_length is not None and y_validated.shape[0] != expected_length:
        raise ValueError(
            f"{vector_name} length ({y_validated.shape[0]}) does not match "
            f"expected length ({expected_length})."
        )

    unique_labels = np.unique(y_validated)

    if not set(unique_labels.tolist()).issubset({0, 1}):
        raise ValueError(
            f"{vector_name} must contain only binary labels 0 and 1. "
            f"Got: {unique_labels.tolist()}"
        )

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

    X_train_validated = _validate_feature_matrix(X_train, matrix_name="X_train")
    y_train_validated = _validate_label_vector(
        y_train,
        expected_length=X_train_validated.shape[0],
        vector_name="y_train",
        require_two_classes=True,
    )

    model = build_classifier_model(classifier_config)
    model.fit(X_train_validated, y_train_validated)

    return model


def predict_labels(model, X_test):
    """
    Predict class labels for a feature matrix.

    Return:
    - predicted_labels ... 1D NumPy array containing only 0 and 1
    """

    if model is None:
        raise ValueError("model must not be None.")

    if not hasattr(model, "predict"):
        raise TypeError("model does not provide a predict(...) method.")

    X_test_validated = _validate_feature_matrix(X_test, matrix_name="X_test")

    predicted_labels = model.predict(X_test_validated)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)

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

    if model is None:
        raise ValueError("model must not be None.")

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

    return None


def predict_single_score(model, feature_vector):
    """
    Predict one confidence-like score for one single feature vector.

    Return:
    - predicted_score ... float or None
    """

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

    if predicted_scores is not None:
        predicted_scores_array = np.asarray(predicted_scores, dtype=np.float32)

        if predicted_scores_array.ndim != 1:
            raise ValueError("predicted_scores must be a 1D array if provided.")

        if predicted_scores_array.shape[0] != predicted_labels_array.shape[0]:
            raise ValueError(
                "predicted_scores length does not match predicted_labels length."
            )

    prediction_records = []

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

    if not isinstance(runtime_feature_record, dict):
        raise TypeError("runtime_feature_record must be a dictionary.")

    if "lbp_feature_vector" not in runtime_feature_record:
        raise KeyError(
            "runtime_feature_record must contain 'lbp_feature_vector'."
        )

    predicted_label = predict_single_label(
        model=model,
        feature_vector=runtime_feature_record["lbp_feature_vector"],
    )

    predicted_score = predict_single_score(
        model=model,
        feature_vector=runtime_feature_record["lbp_feature_vector"],
    )

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

    normalized_preprocessing_config = validate_preprocessing_config(preprocessing_config)
    normalized_lbp_config = validate_lbp_config(lbp_config)
    normalized_classifier_config = validate_classifier_config(classifier_config)

    training_records = load_all_eye_training_records(
        dataset_root=dataset_root,
        load_images=load_images,
        grayscale=grayscale,
        recursive=recursive,
        ignore_invalid_files=ignore_invalid_files,
    )

    X_train, y_train, training_metadata, training_feature_records = prepare_training_matrix_and_labels(
        training_records=training_records,
        preprocessing_config=normalized_preprocessing_config,
        lbp_config=normalized_lbp_config,
        image_key=image_key,
    )

    trained_model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        classifier_config=normalized_classifier_config,
    )

    class_counts = {
        "close": int(np.sum(y_train == 0)),
        "open": int(np.sum(y_train == 1)),
    }

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

    if ground_truth_labels is None:
        return summary

    ground_truth_validated = _validate_label_vector(
        ground_truth_labels,
        expected_length=predicted_labels_validated.shape[0],
        vector_name="ground_truth_labels",
        require_two_classes=False,
    )

    correct_mask = (predicted_labels_validated == ground_truth_validated)
    correct_count = int(np.sum(correct_mask))
    compared_count = int(predicted_labels_validated.shape[0])

    if compared_count == 0:
        accuracy_percent = 0.0
    else:
        accuracy_percent = 100.0 * correct_count / compared_count

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