"""
eye_lbp_dataset.py

This module connects eye-image records to classifier-ready LBP feature
matrices.

Its responsibilities are:
- sending structured eye records through preprocessing and LBP extraction,
- converting enriched feature records into:
    - feature matrices X
    - label vectors y
    - aligned metadata lists
- providing lightweight runtime helpers for single eye ROIs or lists of
  runtime eye images,
- providing small summary helpers useful for smoke tests and debugging.

The module is intentionally separate from:
- raw dataset loading,
- preprocessing implementation,
- LBP implementation,
- classifier training,
- runtime frame aggregation.

This keeps the project modular and makes each stage easier to test.
"""

from copy import deepcopy
import numpy as np

from eye_preprocessing import (
    validate_preprocessing_config,
    preprocess_all_eye_records,
    preprocess_runtime_eye_roi,
)

from lbp_features import (
    validate_lbp_config,
    extract_lbp_features_from_records,
    extract_lbp_features_from_runtime_eye_image,
)


# ---------------------------------------------------------------------
# Record-to-feature pipeline helpers
# ---------------------------------------------------------------------

def prepare_feature_records_from_eye_records(
    eye_records,
    preprocessing_config=None,
    lbp_config=None,
    image_key="image",
):
    """
    Prepare LBP feature records from structured eye records.

    Inputs:
    - eye_records:
        list of dictionaries expected to contain at least:
            image_key -> raw eye image
    - preprocessing_config:
        configuration for eye_preprocessing.py
    - lbp_config:
        configuration for lbp_features.py
    - image_key:
        record key containing the raw eye image

    Return:
    - feature_records:
        list of enriched records containing at least:
            preprocessed_image
            lbp_image
            lbp_feature_vector
            lbp_feature_length
            lbp_config

    Why this helper exists:
    It provides one clean place that connects:
        structured eye records
            -> preprocessing
            -> LBP feature extraction
    """

    if not isinstance(eye_records, list):
        raise TypeError("eye_records must be a list.")

    normalized_preprocessing_config = validate_preprocessing_config(preprocessing_config)
    normalized_lbp_config = validate_lbp_config(lbp_config)

    preprocessed_records = preprocess_all_eye_records(
        records=eye_records,
        preprocessing_config=normalized_preprocessing_config,
        image_key=image_key,
    )

    feature_records = extract_lbp_features_from_records(
        preprocessed_records=preprocessed_records,
        lbp_config=normalized_lbp_config,
        image_key="preprocessed_image",
    )

    return feature_records


def prepare_training_feature_records(
    training_records,
    preprocessing_config=None,
    lbp_config=None,
    image_key="image",
):
    """
    Prepare LBP feature records for the training dataset.

    Inputs:
    - training_records:
        list of training records produced by eye_training_io.py
    - preprocessing_config:
        configuration for eye_preprocessing.py
    - lbp_config:
        configuration for lbp_features.py
    - image_key:
        record key containing the raw eye image

    Return:
    - training_feature_records:
        list of enriched LBP feature records

    Why this helper exists:
    It is the standard training-side entry point for:
        training records
            -> preprocessing
            -> LBP feature records
    """

    training_feature_records = prepare_feature_records_from_eye_records(
        eye_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        image_key=image_key,
    )

    return training_feature_records


# ---------------------------------------------------------------------
# Runtime-oriented helpers
# ---------------------------------------------------------------------

def prepare_runtime_feature_record(
    runtime_eye_image,
    preprocessing_config=None,
    lbp_config=None,
    metadata=None,
):
    """
    Prepare one LBP feature record from one runtime eye image or eye ROI.

    Inputs:
    - runtime_eye_image:
        raw grayscale or color eye image / ROI
    - preprocessing_config:
        configuration for eye_preprocessing.py
    - lbp_config:
        configuration for lbp_features.py
    - metadata:
        optional dictionary with runtime metadata such as:
            runtime_sample_index
            frame_index
            eye_index

    Return:
    - runtime_feature_record:
        dictionary containing:
            metadata
            preprocessed_image
            preprocessed_image_shape
            preprocessing_config
            lbp_image
            lbp_feature_vector
            lbp_feature_length
            lbp_histogram_bin_count
            lbp_config

    Why this helper exists:
    The training pipeline works with record lists, but runtime eye-state
    prediction will often need a direct path from one detected eye ROI to one
    feature vector.
    """

    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary or None.")

    normalized_preprocessing_config = validate_preprocessing_config(preprocessing_config)
    normalized_lbp_config = validate_lbp_config(lbp_config)

    preprocessed_image = preprocess_runtime_eye_roi(
        eye_roi=runtime_eye_image,
        preprocessing_config=normalized_preprocessing_config,
    )

    runtime_lbp_result = extract_lbp_features_from_runtime_eye_image(
        preprocessed_eye_image=preprocessed_image,
        lbp_config=normalized_lbp_config,
    )

    runtime_feature_record = {
        **metadata,
        "preprocessed_image": preprocessed_image,
        "preprocessed_image_shape": tuple(int(value) for value in preprocessed_image.shape),
        "preprocessing_config": deepcopy(normalized_preprocessing_config),
        "lbp_image": runtime_lbp_result["lbp_image"],
        "lbp_feature_vector": runtime_lbp_result["lbp_feature_vector"],
        "lbp_histogram_bin_count": runtime_lbp_result["lbp_histogram_bin_count"],
        "lbp_feature_length": int(runtime_lbp_result["lbp_feature_vector"].shape[0]),
        "lbp_config": deepcopy(runtime_lbp_result["lbp_config"]),
    }

    return runtime_feature_record


def prepare_runtime_feature_records(
    runtime_eye_images,
    preprocessing_config=None,
    lbp_config=None,
    metadata_list=None,
):
    """
    Prepare LBP feature records from a list of runtime eye images.

    Inputs:
    - runtime_eye_images:
        list of raw runtime eye images / ROIs
    - preprocessing_config:
        configuration for eye_preprocessing.py
    - lbp_config:
        configuration for lbp_features.py
    - metadata_list:
        optional list of dictionaries aligned with runtime_eye_images

    Return:
    - runtime_feature_records:
        list of enriched runtime LBP feature records
    """

    if not isinstance(runtime_eye_images, list):
        raise TypeError("runtime_eye_images must be a list.")

    if metadata_list is not None:
        if not isinstance(metadata_list, list):
            raise TypeError("metadata_list must be a list or None.")
        if len(metadata_list) != len(runtime_eye_images):
            raise ValueError(
                "metadata_list length must match runtime_eye_images length."
            )

    runtime_feature_records = []

    for sample_index, runtime_eye_image in enumerate(runtime_eye_images, start=1):
        if metadata_list is None:
            metadata = {"runtime_sample_index": sample_index}
        else:
            metadata = dict(metadata_list[sample_index - 1])
            metadata.setdefault("runtime_sample_index", sample_index)

        runtime_feature_record = prepare_runtime_feature_record(
            runtime_eye_image=runtime_eye_image,
            preprocessing_config=preprocessing_config,
            lbp_config=lbp_config,
            metadata=metadata,
        )
        runtime_feature_records.append(runtime_feature_record)

    return runtime_feature_records


# ---------------------------------------------------------------------
# Matrix and label construction
# ---------------------------------------------------------------------

def build_feature_matrix(feature_records):
    """
    Convert feature records into a 2D feature matrix X.

    Input:
    - feature_records:
        list of dictionaries expected to contain:
            lbp_feature_vector

    Return:
    - X:
        NumPy array of shape:
            (number_of_samples, number_of_features)

    Why this helper exists:
    Classifier code should work with a standard matrix representation rather
    than directly with lists of dictionaries.
    """

    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    if not feature_records:
        raise ValueError("feature_records must not be empty.")

    feature_vectors = []
    expected_length = None

    for record_index, record in enumerate(feature_records, start=1):
        if not isinstance(record, dict):
            raise TypeError("Each feature record must be a dictionary.")

        if "lbp_feature_vector" not in record:
            raise KeyError(
                "Each feature record must contain 'lbp_feature_vector'."
            )

        feature_vector = np.asarray(record["lbp_feature_vector"], dtype=np.float32)

        if feature_vector.ndim != 1:
            raise ValueError(
                f"Feature vector at record {record_index} must be 1D."
            )

        current_length = len(feature_vector)

        if expected_length is None:
            expected_length = current_length
        elif current_length != expected_length:
            raise ValueError(
                "All feature vectors must have the same length. "
                f"Expected {expected_length}, got {current_length} "
                f"at record {record_index}."
            )

        feature_vectors.append(feature_vector)

    X = np.vstack(feature_vectors).astype(np.float32)

    return X


def build_label_vector(feature_records):
    """
    Extract labels into a 1D target vector y.

    Input:
    - feature_records:
        list of dictionaries expected to contain:
            label

    Return:
    - y:
        NumPy array of shape:
            (number_of_samples,)

    Why this helper exists:
    Supervised classifier training requires a clean target vector aligned with
    the rows of the feature matrix.
    """

    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    if not feature_records:
        raise ValueError("feature_records must not be empty.")

    labels = []

    for record_index, record in enumerate(feature_records, start=1):
        if not isinstance(record, dict):
            raise TypeError("Each feature record must be a dictionary.")

        if "label" not in record:
            raise KeyError(
                f"Feature record {record_index} does not contain 'label'."
            )

        label = record["label"]

        if not isinstance(label, (int, np.integer)):
            raise TypeError(
                f"Label at record {record_index} must be an integer."
            )

        labels.append(int(label))

    y = np.asarray(labels, dtype=np.int64)

    return y


def build_metadata_list(feature_records):
    """
    Extract lightweight metadata aligned with the feature-record order.

    Input:
    - feature_records:
        list of feature-record dictionaries

    Return:
    - metadata_list:
        list of dictionaries containing selected metadata keys

    Metadata keys copied if present:
    - file_path
    - relative_path
    - file_name
    - subject_dir
    - subject_id
    - image_id
    - gender
    - glasses
    - eye_state
    - reflections
    - lighting
    - sensor_id
    - class_name
    - label
    - runtime_sample_index
    - frame_index
    - eye_index
    - preprocessed_image_shape
    - lbp_feature_length

    Why this helper exists:
    Later stages often need to map classifier rows back to original samples
    for debugging, inspection, or evaluation.
    """

    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    metadata_keys = [
        "file_path",
        "relative_path",
        "file_name",
        "subject_dir",
        "subject_id",
        "image_id",
        "gender",
        "glasses",
        "eye_state",
        "reflections",
        "lighting",
        "sensor_id",
        "class_name",
        "label",
        "runtime_sample_index",
        "frame_index",
        "eye_index",
        "preprocessed_image_shape",
        "lbp_feature_length",
    ]

    metadata_list = []

    for record in feature_records:
        if not isinstance(record, dict):
            raise TypeError("Each feature record must be a dictionary.")

        metadata = {}

        for key in metadata_keys:
            if key in record:
                metadata[key] = record[key]

        metadata_list.append(metadata)

    return metadata_list


def build_training_matrix_and_labels(training_feature_records):
    """
    Build the standard training outputs:
    - X_train
    - y_train
    - metadata_list

    Input:
    - training_feature_records:
        list of training feature records

    Return:
    - X_train
    - y_train
    - metadata_list

    Why this helper exists:
    It is the standard supervised-learning wrapper for the training pipeline.
    """

    X_train = build_feature_matrix(training_feature_records)
    y_train = build_label_vector(training_feature_records)
    metadata_list = build_metadata_list(training_feature_records)

    return X_train, y_train, metadata_list


def build_runtime_matrix(runtime_feature_records):
    """
    Build the standard runtime outputs:
    - X_runtime
    - metadata_list

    Input:
    - runtime_feature_records:
        list of runtime feature records

    Return:
    - X_runtime
    - metadata_list

    Why this helper exists:
    Runtime prediction usually needs the feature matrix and aligned metadata
    for traceability.
    """

    X_runtime = build_feature_matrix(runtime_feature_records)
    metadata_list = build_metadata_list(runtime_feature_records)

    return X_runtime, metadata_list


# ---------------------------------------------------------------------
# Optional convenience wrappers
# ---------------------------------------------------------------------

def prepare_training_matrix_and_labels(
    training_records,
    preprocessing_config=None,
    lbp_config=None,
    image_key="image",
):
    """
    Full convenience wrapper for the training pipeline.

    Inputs:
    - training_records
    - preprocessing_config
    - lbp_config
    - image_key

    Return:
    - X_train
    - y_train
    - metadata_list
    - training_feature_records

    Why this helper exists:
    It can be convenient during experiments to perform the whole
    training-data preparation in one call.
    """

    training_feature_records = prepare_training_feature_records(
        training_records=training_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        image_key=image_key,
    )

    X_train, y_train, metadata_list = build_training_matrix_and_labels(
        training_feature_records
    )

    return X_train, y_train, metadata_list, training_feature_records


def prepare_runtime_matrix(
    runtime_eye_images,
    preprocessing_config=None,
    lbp_config=None,
    metadata_list=None,
):
    """
    Full convenience wrapper for runtime eye images.

    Inputs:
    - runtime_eye_images
    - preprocessing_config
    - lbp_config
    - metadata_list

    Return:
    - X_runtime
    - runtime_metadata_list
    - runtime_feature_records

    Why this helper exists:
    It keeps later runtime integration simple when many runtime eye images
    need to be converted to matrix form quickly.
    """

    runtime_feature_records = prepare_runtime_feature_records(
        runtime_eye_images=runtime_eye_images,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
        metadata_list=metadata_list,
    )

    X_runtime, runtime_metadata_list = build_runtime_matrix(
        runtime_feature_records
    )

    return X_runtime, runtime_metadata_list, runtime_feature_records


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------

def summarize_feature_records(feature_records):
    """
    Summarize prepared feature records.

    Input:
    - feature_records:
        list of feature-record dictionaries

    Return:
    - summary:
        dictionary containing:
            total_count
            feature_lengths_present
            class_names_present
            labels_present
            subject_ids_present
            has_labels

    Why this helper exists:
    Small summaries are useful for smoke tests and quick verification before
    classifier training and runtime prediction.
    """

    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    feature_lengths_present = set()
    class_names_present = set()
    labels_present = set()
    subject_ids_present = set()
    has_labels = False

    for record_index, record in enumerate(feature_records, start=1):
        if not isinstance(record, dict):
            raise TypeError("Each feature record must be a dictionary.")

        if "lbp_feature_vector" not in record:
            raise KeyError(
                f"Feature record {record_index} does not contain "
                "'lbp_feature_vector'."
            )

        feature_vector = np.asarray(record["lbp_feature_vector"])
        if feature_vector.ndim != 1:
            raise ValueError(
                f"Feature vector at record {record_index} must be 1D."
            )

        feature_lengths_present.add(len(feature_vector))

        if "class_name" in record:
            class_names_present.add(record["class_name"])

        if "label" in record:
            labels_present.add(int(record["label"]))
            has_labels = True

        if "subject_id" in record:
            subject_ids_present.add(record["subject_id"])

    summary = {
        "total_count": len(feature_records),
        "feature_lengths_present": sorted(feature_lengths_present),
        "class_names_present": sorted(class_names_present),
        "labels_present": sorted(labels_present),
        "subject_ids_present": sorted(subject_ids_present),
        "has_labels": has_labels,
    }

    return summary