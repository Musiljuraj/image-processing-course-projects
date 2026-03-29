"""
parking_lbp_dataset.py

Purpose of this module:
- connect image-based records to classifier-ready LBP feature matrices
- keep dataset preparation separate from raw data loading, preprocessing,
  LBP implementation, classifier training, and evaluation

Why this module exists:
At this stage of the project, the individual building blocks already exist:
- training data can be loaded as structured training records
- test parking spaces can be extracted as ROI records
- ROI-like records can be preprocessed into normalized grayscale images
- preprocessed records can be converted into LBP feature records

The next logical stage is to connect these steps cleanly so that later
classifier logic can receive:
- feature matrices X
- label vectors y
- metadata that maps rows back to original samples

This module currently provides:
- build_roi_like_record_from_training_record(...)
- build_roi_like_records_from_training_records(...)
- prepare_feature_records_from_roi_records(...)
- prepare_training_feature_records(...)
- prepare_test_feature_records(...)
- build_feature_matrix(...)
- build_label_vector(...)
- build_metadata_list(...)
- build_training_matrix_and_labels(...)
- build_test_matrix(...)
- summarize_feature_records(...)
"""
# ---------------------------------------------------------------------------
# Module orientation:
# This module is the bridge between image-oriented records and matrix-oriented
# machine-learning inputs. It takes training records or ROI records, sends them
# through preprocessing and LBP extraction, and then converts the resulting
# feature records into X / y arrays plus aligned metadata. In other words, this
# is the place where rich record dictionaries are translated into classifier-
# ready numeric tensors without losing traceability back to the original sample.
# ---------------------------------------------------------------------------

import numpy as np

from preprocessing import preprocess_all_rois
from lbp_features import extract_lbp_features_from_records


def build_roi_like_record_from_training_record(training_record, space_index):
    """
    Convert one training record into an ROI-like record compatible with
    preprocessing.py.

    Inputs:
        training_record ... dictionary expected to contain:
                            - image
                            - file_name
                            - file_path
                            - class_name
                            - label

        space_index ..... positive integer used as a synthetic ROI index

    Return:
        roi_like_record . dictionary containing:
                          - source_image_name
                          - space_index
                          - polygon
                          - roi_image
                          - file_path
                          - file_name
                          - class_name
                          - label

    Why this function exists:
    Training samples are already cropped parking patches, but preprocessing.py
    expects ROI-style records containing "roi_image" and related metadata.
    This helper adapts the training-record structure to that expected format.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(training_record, dict):
        raise TypeError("training_record must be a dictionary.")

    required_keys = {"image", "file_name", "file_path", "class_name", "label"}
    # Assemble the standard output dictionary here so downstream modules receive both
    # the computed values and the metadata needed for traceability.
    missing_keys = required_keys - set(training_record.keys())

    if missing_keys:
        raise KeyError(
            "training_record is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not isinstance(space_index, int):
        raise TypeError("space_index must be an integer.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if space_index <= 0:
        raise ValueError("space_index must be a positive integer.")

    # Assemble the standard output dictionary here so downstream modules receive both
    # the computed values and the metadata needed for traceability.
    roi_like_record = {
        "source_image_name": training_record["file_name"],
        "space_index": space_index,
        "polygon": None,
        "roi_image": training_record["image"],
        "file_path": training_record["file_path"],
        "file_name": training_record["file_name"],
        "class_name": training_record["class_name"],
        "label": training_record["label"],
    }

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return roi_like_record


def build_roi_like_records_from_training_records(training_records):
    """
    Convert a list of training records into ROI-like records.

    Input:
        training_records ... list of training-record dictionaries

    Return:
        roi_like_records . list of ROI-like dictionaries compatible with
                           preprocessing.py

    Why this function exists:
    It is often useful to reuse the same preprocessing pipeline for:
    - extracted parking ROIs from test images
    - already cropped parking patches from the training set
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(training_records, list):
        raise TypeError("training_records must be a list.")

    # Start an accumulation structure that will be filled gradually as the function
    # walks through samples, records, rows, or files.
    roi_like_records = []

    # Process items in a deterministic order so the produced outputs stay aligned with
    # the corresponding inputs, labels, or metadata.
    for space_index, training_record in enumerate(training_records, start=1):
        roi_like_record = build_roi_like_record_from_training_record(
            training_record=training_record,
            space_index=space_index,
        )
        roi_like_records.append(roi_like_record)

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return roi_like_records


def prepare_feature_records_from_roi_records(
    roi_records,
    preprocessing_config,
    lbp_config,
):
    """
    Prepare LBP feature records from ROI-like records.

    Inputs:
        roi_records ............ list of ROI-style records
        preprocessing_config ... dictionary for preprocessing.py
        lbp_config ............. dictionary for lbp_features.py

    Return:
        feature_records ........ list of LBP feature records containing:
                                 - metadata from earlier stages
                                 - processed_image
                                 - lbp_image
                                 - lbp_feature_vector
                                 - lbp_config

    Why this function exists:
    This helper connects:
    ROI records
        -> preprocessing
        -> LBP feature extraction
    in one clean place.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(roi_records, list):
        raise TypeError("roi_records must be a list.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not isinstance(preprocessing_config, dict):
        raise TypeError("preprocessing_config must be a dictionary.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not isinstance(lbp_config, dict):
        raise TypeError("lbp_config must be a dictionary.")

    preprocessed_records = preprocess_all_rois(
        rois=roi_records,
        preprocessing_config=preprocessing_config,
    )

    feature_records = extract_lbp_features_from_records(
        preprocessed_records=preprocessed_records,
        lbp_config=lbp_config,
    )

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return feature_records


def prepare_training_feature_records(
    training_records,
    preprocessing_config,
    lbp_config,
):
    """
    Prepare LBP feature records for the training dataset.

    Inputs:
        training_records ....... list of training-record dictionaries
        preprocessing_config ... dictionary for preprocessing.py
        lbp_config ............. dictionary for lbp_features.py

    Return:
        training_feature_records ... list of enriched LBP feature records

    Why this function exists:
    Training records start with the structure produced by parking_training_io.py,
    so they must first be adapted into ROI-like records before the common
    preprocessing + LBP pipeline can be applied.
    """

    # Set up the local working state first so the later processing steps can operate on
    # explicit, well-named intermediate values.
    roi_like_records = build_roi_like_records_from_training_records(
        training_records=training_records,
    )

    training_feature_records = prepare_feature_records_from_roi_records(
        roi_records=roi_like_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return training_feature_records


def prepare_test_feature_records(
    test_roi_records,
    preprocessing_config,
    lbp_config,
):
    """
    Prepare LBP feature records for extracted test ROIs.

    Inputs:
        test_roi_records ....... list of ROI records from roi_extraction.py
        preprocessing_config ... dictionary for preprocessing.py
        lbp_config ............. dictionary for lbp_features.py

    Return:
        test_feature_records ... list of enriched LBP feature records

    Why this function exists:
    Test parking slots already arrive in ROI-style structure, so they can be
    sent directly through the common preprocessing + LBP feature pipeline.
    """

    # Set up the local working state first so the later processing steps can operate on
    # explicit, well-named intermediate values.
    test_feature_records = prepare_feature_records_from_roi_records(
        roi_records=test_roi_records,
        preprocessing_config=preprocessing_config,
        lbp_config=lbp_config,
    )

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return test_feature_records


def build_feature_matrix(feature_records):
    """
    Convert feature records into a 2D feature matrix X.

    Input:
        feature_records ... list of LBP feature-record dictionaries

    Return:
        X ............... NumPy array of shape:
                          (number_of_samples, number_of_features)

    Why this function exists:
    Classifier code should not work directly with lists of dictionaries. It
    should receive a standard feature matrix with one row per sample.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not feature_records:
        raise ValueError("feature_records must not be empty.")

    # Start an accumulation structure that will be filled gradually as the function
    # walks through samples, records, rows, or files.
    feature_vectors = []
    expected_length = None

    # Process items in a deterministic order so the produced outputs stay aligned with
    # the corresponding inputs, labels, or metadata.
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

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return X


def build_label_vector(feature_records):
    """
    Extract labels into a 1D target vector y.

    Input:
        feature_records ... list of feature-record dictionaries expected to
                            contain the key 'label'

    Return:
        y ............... NumPy array of shape:
                          (number_of_samples,)

    Why this function exists:
    Supervised classifier training requires a clean target vector aligned with
    the rows of the feature matrix.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not feature_records:
        raise ValueError("feature_records must not be empty.")

    # Start an accumulation structure that will be filled gradually as the function
    # walks through samples, records, rows, or files.
    labels = []

    # Process items in a deterministic order so the produced outputs stay aligned with
    # the corresponding inputs, labels, or metadata.
    for record_index, record in enumerate(feature_records, start=1):
        if not isinstance(record, dict):
            raise TypeError("Each feature record must be a dictionary.")

        if "label" not in record:
            raise KeyError(
                f"Feature record {record_index} does not contain 'label'."
            )

        label = record["label"]

        if not isinstance(label, int):
            raise TypeError(
                f"Label at record {record_index} must be an integer."
            )

        labels.append(label)

    y = np.asarray(labels, dtype=np.int64)

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return y


def build_metadata_list(feature_records):
    """
    Extract lightweight metadata aligned with the feature-record order.

    Input:
        feature_records ... list of feature-record dictionaries

    Return:
        metadata_list ... list of dictionaries containing selected metadata keys

    Metadata keys copied if present:
        - source_image_name
        - space_index
        - polygon
        - file_path
        - file_name
        - class_name
        - label

    Why this function exists:
    Later stages often need to map classifier rows back to original samples for
    debugging, inspection, or evaluation.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    metadata_keys = [
        "source_image_name",
        "space_index",
        "polygon",
        "file_path",
        "file_name",
        "class_name",
        "label",
    ]

    # Start an accumulation structure that will be filled gradually as the function
    # walks through samples, records, rows, or files.
    metadata_list = []

    # Process the collection item by item, updating the running result structure as each
    # sample contributes its part of the final output.
    for record in feature_records:
        if not isinstance(record, dict):
            raise TypeError("Each feature record must be a dictionary.")

        metadata = {}

        for key in metadata_keys:
            if key in record:
                metadata[key] = record[key]

        metadata_list.append(metadata)

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return metadata_list


def build_training_matrix_and_labels(training_feature_records):
    """
    Build the standard training outputs:
    - X_train
    - y_train
    - metadata_list

    Input:
        training_feature_records ... list of training feature records

    Return:
        X_train ................. 2D NumPy feature matrix
        y_train ................. 1D NumPy label vector
        metadata_list ........... list of aligned metadata dictionaries

    Why this function exists:
    It is a convenience wrapper for the common supervised-learning case.
    """

    # Set up the local working state first so the later processing steps can operate on
    # explicit, well-named intermediate values.
    X_train = build_feature_matrix(training_feature_records)
    y_train = build_label_vector(training_feature_records)
    metadata_list = build_metadata_list(training_feature_records)

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return X_train, y_train, metadata_list


def build_test_matrix(test_feature_records):
    """
    Build the standard test outputs:
    - X_test
    - metadata_list

    Input:
        test_feature_records ... list of test feature records

    Return:
        X_test .............. 2D NumPy feature matrix
        metadata_list ....... list of aligned metadata dictionaries

    Why this function exists:
    Prediction usually needs the test feature matrix and metadata for tracing
    predictions back to parking-space indices and source images.
    """

    # Set up the local working state first so the later processing steps can operate on
    # explicit, well-named intermediate values.
    X_test = build_feature_matrix(test_feature_records)
    metadata_list = build_metadata_list(test_feature_records)

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return X_test, metadata_list


def summarize_feature_records(feature_records):
    """
    Summarize prepared feature records.

    Input:
        feature_records ... list of feature-record dictionaries

    Return:
        summary ............ dictionary containing:
                             - total_count
                             - feature_lengths_present
                             - class_names_present
                             - labels_present
                             - source_image_names_present
                             - has_labels

    Why this function exists:
    Small summaries are very useful for smoke tests and quick verification
    before moving on to classifier training and prediction.
    """

    # Reject unsupported inputs immediately so the main body of the function can assume
    # the expected data structure and fail with clear errors when needed.
    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    feature_lengths_present = set()
    class_names_present = set()
    labels_present = set()
    source_image_names_present = set()
    has_labels = False

    # Process items in a deterministic order so the produced outputs stay aligned with
    # the corresponding inputs, labels, or metadata.
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
            labels_present.add(record["label"])
            has_labels = True

        if "source_image_name" in record:
            source_image_names_present.add(record["source_image_name"])

    # Assemble the standard output dictionary here so downstream modules receive both
    # the computed values and the metadata needed for traceability.
    summary = {
        "total_count": len(feature_records),
        "feature_lengths_present": sorted(feature_lengths_present),
        "class_names_present": sorted(class_names_present),
        "labels_present": sorted(labels_present),
        "source_image_names_present": sorted(source_image_names_present),
        "has_labels": has_labels,
    }

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return summary