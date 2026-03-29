"""
parking_training_io.py

Purpose of this module:
- load and organize the parking-occupancy training dataset
- keep training-data loading separate from test-data loading, preprocessing,
  feature extraction, classifier training, and evaluation

Why this module exists:
The parking task now uses a supervised training set stored in two folders:
- free ... empty parking-space samples
- full ... occupied parking-space samples

This module converts that folder-based dataset into structured training records
that later stages of the project can preprocess, transform into LBP features,
and use for classifier training.

This module currently provides:
- validate_training_dataset_structure(...)
- collect_image_paths_from_class_dir(...)
- load_one_training_image(...)
- build_one_training_record(...)
- load_training_records_from_class_dir(...)
- load_all_training_records(...)
- summarize_training_records(...)
"""

from pathlib import Path

import cv2


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

LABEL_BY_CLASS_NAME = {
    "free": 0,
    "full": 1,
}


def validate_training_dataset_structure(training_root):
    """
    Validate the expected folder structure of the parking training dataset.

    Input:
        training_root ... full path to data/training

    Return:
        class_dirs ..... dictionary:
                         {
                             "free": Path(.../free),
                             "full": Path(.../full),
                         }

    Expected structure:
        training_root/
            free/
            full/

    Why this helper exists:
    It is better to detect missing dataset folders early and raise one clear
    error than to fail later in image collection or image loading.
    """

    training_root = Path(training_root)

    if not training_root.exists():
        raise FileNotFoundError(
            f"Training root directory does not exist: {training_root}"
        )

    if not training_root.is_dir():
        raise NotADirectoryError(
            f"Training root path is not a directory: {training_root}"
        )

    free_dir = training_root / "free"
    full_dir = training_root / "full"

    if not free_dir.exists():
        raise FileNotFoundError(f"Missing training class directory: {free_dir}")

    if not free_dir.is_dir():
        raise NotADirectoryError(
            f"Training class path is not a directory: {free_dir}"
        )

    if not full_dir.exists():
        raise FileNotFoundError(f"Missing training class directory: {full_dir}")

    if not full_dir.is_dir():
        raise NotADirectoryError(
            f"Training class path is not a directory: {full_dir}"
        )

    class_dirs = {
        "free": free_dir,
        "full": full_dir,
    }

    return class_dirs


def collect_image_paths_from_class_dir(
    class_dir,
    supported_extensions=SUPPORTED_IMAGE_EXTENSIONS,
):
    """
    Collect all supported image paths from one class directory.

    Inputs:
        class_dir .............. full path to one class directory
                                 for example:
                                 data/training/free
                                 data/training/full

        supported_extensions ... set of supported lowercase suffixes such as:
                                 {".png", ".jpg", ".jpeg", ".bmp"}

    Return:
        image_paths ............ sorted list of pathlib.Path objects

    Why this helper exists:
    The free/ and full/ folders should be handled the same way:
    - scan directory content
    - keep only supported image files
    - sort them deterministically
    """

    class_dir = Path(class_dir)

    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory does not exist: {class_dir}")

    if not class_dir.is_dir():
        raise NotADirectoryError(f"Class path is not a directory: {class_dir}")

    normalized_extensions = {ext.lower() for ext in supported_extensions}

    image_paths = []

    for path in class_dir.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() in normalized_extensions:
            image_paths.append(path)

    image_paths = sorted(image_paths, key=lambda path: path.name.lower())

    return image_paths


def load_one_training_image(image_path):
    """
    Load one training image from disk.

    Input:
        image_path ... full path to one training image file

    Return:
        image ...... image array loaded by cv2.imread(...)

    Why this helper exists:
    Image loading should be centralized so that any read failure produces one
    clear and consistent error message.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Training image file does not exist: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Training image path is not a file: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read training image: {image_path}")

    return image


def build_one_training_record(image_path, image, class_name, label):
    """
    Build one structured training record.

    Inputs:
        image_path ... full path to the image file
        image ...... loaded image array
        class_name . "free" or "full"
        label ...... numeric class label
                      free = 0
                      full = 1

    Return:
        training_record ... dictionary containing:
                            - file_path
                            - file_name
                            - class_name
                            - label
                            - image

    Why this function exists:
    Later stages of the project should work with consistent record dictionaries,
    not with loose tuples or mixed ad-hoc structures.
    """

    image_path = Path(image_path)

    training_record = {
        "file_path": image_path,
        "file_name": image_path.name,
        "class_name": class_name,
        "label": label,
        "image": image,
    }

    return training_record


def load_training_records_from_class_dir(
    class_dir,
    class_name,
    label,
    supported_extensions=SUPPORTED_IMAGE_EXTENSIONS,
):
    """
    Load all training records from one class directory.

    Inputs:
        class_dir .............. full path to one class directory
        class_name ............. expected class name, for example:
                                 "free" or "full"
        label .................. numeric class label
        supported_extensions ... allowed image suffixes

    Return:
        training_records ....... list of training-record dictionaries

    Why this function exists:
    Both free/ and full/ follow the same loading pattern, so this logic should
    be implemented once and reused for both classes.
    """

    if not isinstance(class_name, str):
        raise TypeError("class_name must be a string.")

    normalized_class_name = class_name.strip().lower()

    if normalized_class_name not in LABEL_BY_CLASS_NAME:
        raise ValueError(
            "Unsupported class_name. Expected one of: "
            f"{list(LABEL_BY_CLASS_NAME.keys())}. Got: {class_name}"
        )

    if not isinstance(label, int):
        raise TypeError("label must be an integer.")

    image_paths = collect_image_paths_from_class_dir(
        class_dir=class_dir,
        supported_extensions=supported_extensions,
    )

    if not image_paths:
        raise ValueError(
            f"No supported training images were found in directory: {class_dir}"
        )

    training_records = []

    for image_path in image_paths:
        image = load_one_training_image(image_path)

        training_record = build_one_training_record(
            image_path=image_path,
            image=image,
            class_name=normalized_class_name,
            label=label,
        )

        training_records.append(training_record)

    return training_records


def load_all_training_records(
    training_root,
    supported_extensions=SUPPORTED_IMAGE_EXTENSIONS,
):
    """
    Load the full parking training dataset.

    Input:
        training_root .......... full path to data/training
        supported_extensions ... allowed image suffixes

    Return:
        training_records ....... combined list of all training records from:
                                 - free (label 0)
                                 - full (label 1)

    Label convention used by this project:
        free = 0
        full = 1

    Why this function exists:
    Most later modules should not care about directory traversal details.
    They should be able to call one function and receive ready-to-use records.
    """

    class_dirs = validate_training_dataset_structure(training_root)

    free_records = load_training_records_from_class_dir(
        class_dir=class_dirs["free"],
        class_name="free",
        label=LABEL_BY_CLASS_NAME["free"],
        supported_extensions=supported_extensions,
    )

    full_records = load_training_records_from_class_dir(
        class_dir=class_dirs["full"],
        class_name="full",
        label=LABEL_BY_CLASS_NAME["full"],
        supported_extensions=supported_extensions,
    )

    training_records = free_records + full_records

    return training_records


def summarize_training_records(training_records):
    """
    Summarize the loaded training dataset.

    Input:
        training_records ... list of training-record dictionaries

    Return:
        summary ............ dictionary containing:
                             - total_count
                             - free_count
                             - full_count
                             - class_names_present
                             - labels_present

    Why this function exists:
    Small summaries are very useful for smoke tests and quick verification of
    dataset loading before moving on to preprocessing and classifier training.
    """

    if not isinstance(training_records, list):
        raise TypeError("training_records must be a list.")

    free_count = 0
    full_count = 0
    class_names_present = set()
    labels_present = set()

    for record in training_records:
        if not isinstance(record, dict):
            raise TypeError("Each training record must be a dictionary.")

        if "class_name" not in record:
            raise KeyError("Each training record must contain 'class_name'.")

        if "label" not in record:
            raise KeyError("Each training record must contain 'label'.")

        class_name = record["class_name"]
        label = record["label"]

        class_names_present.add(class_name)
        labels_present.add(label)

        if class_name == "free":
            free_count += 1
        elif class_name == "full":
            full_count += 1
        else:
            raise ValueError(
                f"Unsupported class_name in training record: {class_name}"
            )

    summary = {
        "total_count": len(training_records),
        "free_count": free_count,
        "full_count": full_count,
        "class_names_present": sorted(class_names_present),
        "labels_present": sorted(labels_present),
    }

    return summary