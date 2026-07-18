# This module is the "record-to-features" bridge of the project.
# It sits between:
# - the earlier data/preprocessing layers, which still work with structured
#   records and images,
# - and the later classifier layer, which expects clean numeric matrices.
#
# In other words, this file is where the project crosses from
# "structured eye samples with metadata"
# into
# "classifier-ready feature vectors and matrices".
#
# It handles both main usage directions:
#
# 1. training side
#    training records
#        -> preprocessing
#        -> LBP extraction
#        -> feature records
#        -> X_train, y_train, metadata
#
# 2. runtime side
#    detected eye ROI(s)
#        -> preprocessing
#        -> LBP extraction
#        -> runtime feature record(s)
#        -> X_runtime, metadata
#
# A key design decision here is that the module does not implement
# preprocessing itself and does not implement LBP itself.
# It only coordinates those two lower-level stages and reshapes their outputs
# into forms that the classifier and runtime code can consume directly.

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

# deepcopy is used when validated configs are attached to runtime feature
# records. This keeps stored metadata stable even if the original config
# objects are later changed elsewhere.
from copy import deepcopy

# NumPy is the final numeric representation layer here.
# This module uses it mainly for:
# - feature-vector normalization,
# - matrix construction,
# - label-vector construction,
# - basic validation of dimensionality and dtype consistency.
import numpy as np

# These imports provide the shared preprocessing stage used by both:
# - training records,
# - runtime eye ROIs.
#
# This module does not re-implement preprocessing logic. It calls into the
# dedicated preprocessing layer and then continues with LBP extraction.
from eye_preprocessing import (
    validate_preprocessing_config,
    preprocess_all_eye_records,
    preprocess_runtime_eye_roi,
)

# These imports provide the shared LBP feature-extraction stage.
# Again, this module does not implement LBP itself. It only connects its
# outputs to the matrix/metadata structures needed by classifier code and
# runtime inference code.
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

    # This helper is the most general "record list -> LBP feature records"
    # entry point in the module, so it first validates that the incoming
    # container has the expected high-level type.
    if not isinstance(eye_records, list):
        raise TypeError("eye_records must be a list.")

    # Normalize both configs once at the start.
    # That keeps later processing deterministic and ensures the same validated
    # config objects are reused throughout the whole record list.
    normalized_preprocessing_config = validate_preprocessing_config(preprocessing_config)
    normalized_lbp_config = validate_lbp_config(lbp_config)

    # Stage 1:
    # pass the structured records through the shared preprocessing module.
    #
    # Each returned record still preserves the original metadata, but is
    # enriched with:
    # - preprocessed_image
    # - preprocessed_image_shape
    # - preprocessing_config
    preprocessed_records = preprocess_all_eye_records(
        records=eye_records,
        preprocessing_config=normalized_preprocessing_config,
        image_key=image_key,
    )

    # Stage 2:
    # pass those preprocessed records into the shared LBP extraction module.
    #
    # The raw input image for this stage is no longer the original image under
    # image_key, but the already standardized image stored as
    # "preprocessed_image".
    feature_records = extract_lbp_features_from_records(
        preprocessed_records=preprocessed_records,
        lbp_config=normalized_lbp_config,
        image_key="preprocessed_image",
    )

    # The returned records are now fully feature-enriched and can later be
    # converted into matrices, labels, and metadata lists.
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

    # This helper is mostly semantic.
    # The generic record-to-feature pipeline already knows how to transform
    # structured eye records, but this wrapper gives the training path a clear,
    # self-documenting entry point.
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

    # Runtime prediction often starts from a single detected eye ROI, so this
    # helper builds one complete runtime feature record around that one image.
    #
    # Optional metadata is accepted because runtime code often wants to preserve
    # context such as:
    # - which frame this eye came from,
    # - which eye index it was,
    # - what sample order it had in a batch.
    if metadata is None:
        metadata = {}
    elif not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary or None.")

    # Validate the preprocessing and LBP configs once so the single runtime
    # sample follows the exact same normalization rules as the training-side
    # samples.
    normalized_preprocessing_config = validate_preprocessing_config(preprocessing_config)
    normalized_lbp_config = validate_lbp_config(lbp_config)

    # Step 1:
    # preprocess the runtime eye ROI into the same normalized format used in
    # the training-side pipeline.
    preprocessed_image = preprocess_runtime_eye_roi(
        eye_roi=runtime_eye_image,
        preprocessing_config=normalized_preprocessing_config,
    )

    # Step 2:
    # compute the LBP representation and feature vector from that preprocessed
    # runtime eye image.
    runtime_lbp_result = extract_lbp_features_from_runtime_eye_image(
        preprocessed_eye_image=preprocessed_image,
        lbp_config=normalized_lbp_config,
    )

    # Step 3:
    # assemble one structured runtime feature record.
    #
    # The record combines:
    # - caller-provided metadata,
    # - the normalized preprocessed image,
    # - the computed LBP image,
    # - the final feature vector and related dimensional metadata,
    # - the exact configs that produced those outputs.
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

    # This is the batch version of the single-runtime-sample helper.
    # It exists so runtime code can process multiple detected eye ROIs in one
    # consistent pass.
    if not isinstance(runtime_eye_images, list):
        raise TypeError("runtime_eye_images must be a list.")

    # When metadata is supplied, it must align one-to-one with the list of eye
    # images so each produced record receives the correct contextual metadata.
    if metadata_list is not None:
        if not isinstance(metadata_list, list):
            raise TypeError("metadata_list must be a list or None.")
        if len(metadata_list) != len(runtime_eye_images):
            raise ValueError(
                "metadata_list length must match runtime_eye_images length."
            )

    runtime_feature_records = []

    # Walk through each runtime image and convert it into one full runtime
    # feature record.
    for sample_index, runtime_eye_image in enumerate(runtime_eye_images, start=1):
        # If no metadata list was supplied, attach a default runtime sample
        # index so the records still remain traceable in order.
        if metadata_list is None:
            metadata = {"runtime_sample_index": sample_index}
        else:
            # Copy the user-provided metadata dict so modifications here do not
            # affect the original list supplied by the caller.
            metadata = dict(metadata_list[sample_index - 1])

            # Ensure each record has a stable runtime_sample_index even when the
            # caller did not provide one explicitly.
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

    # The classifier layer should not need to know how feature vectors are
    # stored inside dictionaries. This helper extracts them and stacks them into
    # the standard 2D matrix representation expected by training and inference
    # code.
    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    if not feature_records:
        raise ValueError("feature_records must not be empty.")

    feature_vectors = []
    expected_length = None

    # Validate every record and every feature vector before stacking them.
    # This keeps matrix construction strict and prevents silent shape mismatch.
    for record_index, record in enumerate(feature_records, start=1):
        if not isinstance(record, dict):
            raise TypeError("Each feature record must be a dictionary.")

        if "lbp_feature_vector" not in record:
            raise KeyError(
                "Each feature record must contain 'lbp_feature_vector'."
            )

        feature_vector = np.asarray(record["lbp_feature_vector"], dtype=np.float32)

        # Every feature vector must be one-dimensional.
        if feature_vector.ndim != 1:
            raise ValueError(
                f"Feature vector at record {record_index} must be 1D."
            )

        current_length = len(feature_vector)

        # The first vector defines the expected feature length.
        # All later vectors must match that length exactly, otherwise the matrix
        # would not be well-formed.
        if expected_length is None:
            expected_length = current_length
        elif current_length != expected_length:
            raise ValueError(
                "All feature vectors must have the same length. "
                f"Expected {expected_length}, got {current_length} "
                f"at record {record_index}."
            )

        feature_vectors.append(feature_vector)

    # Stack the validated feature vectors row-wise into one classifier-ready
    # feature matrix.
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

    # This helper extracts the training labels from the same feature records
    # that build_feature_matrix(...) turns into X. The goal is to preserve
    # row-wise alignment:
    # row i in X corresponds to element i in y.
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

        # The project uses integer binary labels for supervised learning, so
        # every label must be an integer-like value.
        if not isinstance(label, (int, np.integer)):
            raise TypeError(
                f"Label at record {record_index} must be an integer."
            )

        labels.append(int(label))

    # Convert the collected labels to the standard 1D integer vector form.
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

    # The point of this helper is alignment, not full duplication.
    # It extracts a lightweight subset of fields that are most useful for
    # tracing rows of X back to the original source samples.
    if not isinstance(feature_records, list):
        raise TypeError("feature_records must be a list.")

    # These keys intentionally cover both:
    # - training-side metadata,
    # - runtime-side metadata.
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

        # Copy only the selected keys that are actually present.
        # This keeps the metadata list compact and stable.
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

    # This wrapper is the standard "training feature records -> classifier
    # training inputs" conversion step used by the higher training pipeline.
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

    # Runtime inference does not need labels, only:
    # - the feature matrix for prediction,
    # - the aligned metadata for later traceability and result attachment.
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

    # This is the most compact training-side wrapper in the module.
    # It performs the whole chain:
    #
    #   training records
    #       -> training feature records
    #       -> X_train, y_train, metadata_list
    #
    # and returns both the final matrix outputs and the intermediate feature
    # records for inspection/debugging.
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

    # This is the runtime-side equivalent of the full training wrapper.
    # It performs the whole runtime conversion chain in one call and returns:
    # - X_runtime ........ numeric input for prediction
    # - runtime_metadata_list ... row-aligned traceability info
    # - runtime_feature_records .. richer intermediate records for debugging
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

    # This summary helper is intentionally lightweight.
    # It does not try to inspect the actual classifier values in depth.
    # Instead, it checks the structural consistency of the feature records:
    # - how many there are,
    # - what feature lengths are present,
    # - what class names / numeric labels appear,
    # - which subject IDs are represented,
    # - whether label information exists at all.
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

        # Record the feature length so quick summary checks can immediately show
        # whether all feature records share a consistent descriptor dimension.
        feature_lengths_present.add(len(feature_vector))

        # Keep track of the categorical metadata present in the record set.
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