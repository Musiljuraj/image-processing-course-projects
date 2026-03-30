"""
eye_training_io.py

This module loads and structures the training dataset used for the
LBP-based eye-state classifier.

Its responsibilities are:
- locating eye-image files inside the extracted mrlEyes_2018_01 dataset,
- parsing metadata encoded in file names,
- loading eye images from disk,
- converting each sample into one structured training record,
- providing simple dataset-summary helpers for inspection.

The module is intentionally independent from the runtime video pipeline so it
can be developed and tested separately before the LBP feature-extraction and
classifier stages are implemented.
"""

from pathlib import Path
from collections import Counter
import re
import cv2


# ---------------------------------------------------------------------
# Dataset format configuration
# ---------------------------------------------------------------------
#
# The mrlEyes_2018_01 dataset uses file names in the following pattern:
#
#     s0001_00001_0_0_0_0_0_01.png
#
# The encoded fields are interpreted as:
# - subject identifier
# - image identifier
# - gender
# - glasses
# - eye state
# - reflections
# - lighting condition
# - sensor identifier
#
# For the current assignment, only the eye-state label is essential for
# training:
# - 0 -> closed
# - 1 -> open
# ---------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

EYE_STATE_LABELS = {
    0: "close",
    1: "open",
}


# ---------------------------------------------------------------------
# Internal path and parsing helpers
# ---------------------------------------------------------------------

def _ensure_dataset_root(dataset_root):
    """
    Convert the dataset root to Path form and validate that it exists.

    A missing dataset directory is treated as a hard error because the later
    training stages depend on deterministic access to the extracted image set.
    """

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_root}")

    return dataset_root


def _is_supported_image_file(file_path):
    """
    Return True if the file has a supported image extension.

    The check is case-insensitive so that the loader remains tolerant to
    different archive or operating-system behaviors.
    """

    return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _natural_sort_key(value):
    """
    Build a natural-sort key for stable file ordering.

    Natural sorting keeps numeric parts in human order, for example:
        image2 < image10

    This is useful for reproducible dataset traversal and debugging output.
    """

    parts = re.split(r"(\d+)", str(value).lower())
    key = []

    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)

    return key


def _parse_int_field(raw_value, field_name, file_name):
    """
    Parse one integer field from a dataset file name.

    A descriptive ValueError is raised when parsing fails so that malformed
    file names are easy to diagnose.
    """

    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot parse field '{field_name}' from file name '{file_name}'. "
            f"Expected integer, got '{raw_value}'."
        ) from exc


def _normalize_eye_state_label(label_value):
    """
    Convert the numeric eye-state label into the internal class name.

    Supported label mapping:
    - 0 -> close
    - 1 -> open

    Any other value is treated as invalid because this loader is intended
    specifically for the binary open/close assignment task.
    """

    if label_value not in EYE_STATE_LABELS:
        raise ValueError(
            f"Unsupported eye-state label value: {label_value}. "
            f"Expected one of: {sorted(EYE_STATE_LABELS.keys())}"
        )

    return EYE_STATE_LABELS[label_value]


# ---------------------------------------------------------------------
# Public dataset-discovery helpers
# ---------------------------------------------------------------------

def collect_eye_image_paths(dataset_root, recursive=True):
    """
    Collect all supported eye-image file paths from the dataset directory.

    Parameters:
    - dataset_root:
        Root directory of the extracted mrlEyes_2018_01 dataset.
    - recursive:
        When True, subdirectories are scanned recursively.

    The returned paths are naturally sorted for deterministic traversal.
    """

    dataset_root = _ensure_dataset_root(dataset_root)

    iterator = dataset_root.rglob("*") if recursive else dataset_root.glob("*")

    image_paths = [
        path for path in iterator
        if _is_supported_image_file(path)
    ]

    image_paths.sort(key=_natural_sort_key)

    return image_paths


# ---------------------------------------------------------------------
# Public file-name parsing helpers
# ---------------------------------------------------------------------

def parse_eye_filename(file_name):
    """
    Parse one mrlEyes file name into structured metadata.

    Expected stem pattern:
        subject_id_image_id_gender_glasses_eye_state_reflections_lighting_sensor

    Example:
        s0001_00001_0_0_0_0_0_01.png

    The function returns a dictionary containing the parsed metadata and the
    normalized class label used by the future classifier pipeline.
    """

    path_obj = Path(file_name)
    stem = path_obj.stem
    parts = stem.split("_")

    if len(parts) < 8:
        raise ValueError(
            f"Unexpected mrlEyes file-name format: '{file_name}'. "
            f"Expected at least 8 underscore-separated fields, got {len(parts)}."
        )

    subject_id = parts[0]
    image_id = _parse_int_field(parts[1], "image_id", file_name)
    gender = _parse_int_field(parts[2], "gender", file_name)
    glasses = _parse_int_field(parts[3], "glasses", file_name)
    eye_state = _parse_int_field(parts[4], "eye_state", file_name)
    reflections = _parse_int_field(parts[5], "reflections", file_name)
    lighting = _parse_int_field(parts[6], "lighting", file_name)
    sensor_id = _parse_int_field(parts[7], "sensor_id", file_name)

    class_name = _normalize_eye_state_label(eye_state)

    parsed = {
        "subject_id": subject_id,
        "image_id": image_id,
        "gender": gender,
        "glasses": glasses,
        "eye_state": eye_state,
        "reflections": reflections,
        "lighting": lighting,
        "sensor_id": sensor_id,
        "label": eye_state,
        "class_name": class_name,
        "raw_parts": parts,
    }

    return parsed


# ---------------------------------------------------------------------
# Public image-loading helpers
# ---------------------------------------------------------------------

def load_one_eye_training_image(image_path, grayscale=True):
    """
    Load one eye-training image from disk.

    Parameters:
    - image_path:
        Path to one dataset image.
    - grayscale:
        When True, the image is loaded directly in grayscale form.

    Grayscale is the default because the current project pipeline and the
    planned LBP-based representation operate on grayscale data.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Training image does not exist: {image_path}")

    read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), read_flag)

    if image is None:
        raise ValueError(f"Cannot load training image: {image_path}")

    return image


# ---------------------------------------------------------------------
# Public record-construction helpers
# ---------------------------------------------------------------------

def build_one_eye_training_record(image_path, dataset_root=None, load_image=True, grayscale=True):
    """
    Build one structured training record from a dataset image path.

    The returned record contains:
    - file and path information,
    - parsed metadata from the file name,
    - the normalized binary label,
    - the loaded image array (optional).

    When dataset_root is provided, a relative path is also stored so later
    dataset reports remain readable and stable across machines.
    """

    image_path = Path(image_path)

    parsed = parse_eye_filename(image_path.name)

    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        try:
            relative_path = image_path.relative_to(dataset_root)
        except ValueError:
            relative_path = image_path.name
    else:
        relative_path = image_path.name

    image = None
    image_shape = None

    if load_image:
        image = load_one_eye_training_image(image_path, grayscale=grayscale)
        image_shape = tuple(int(value) for value in image.shape)

    subject_dir = image_path.parent.name

    record = {
        "file_path": image_path,
        "relative_path": str(relative_path),
        "file_name": image_path.name,
        "subject_dir": subject_dir,

        "subject_id": parsed["subject_id"],
        "image_id": parsed["image_id"],
        "gender": parsed["gender"],
        "glasses": parsed["glasses"],
        "eye_state": parsed["eye_state"],
        "reflections": parsed["reflections"],
        "lighting": parsed["lighting"],
        "sensor_id": parsed["sensor_id"],

        "label": parsed["label"],
        "class_name": parsed["class_name"],

        "image": image,
        "image_shape": image_shape,
    }

    return record


def load_all_eye_training_records(
    dataset_root,
    load_images=True,
    grayscale=True,
    recursive=True,
    ignore_invalid_files=False
):
    """
    Load all training records from the extracted mrlEyes dataset.

    Parameters:
    - dataset_root:
        Root directory of the extracted dataset.
    - load_images:
        When True, each image is loaded into memory and stored in its record.
    - grayscale:
        When True, images are loaded in grayscale form.
    - recursive:
        When True, subdirectories are scanned recursively.
    - ignore_invalid_files:
        When True, malformed files are skipped instead of stopping the load.

    The function returns a list of structured record dictionaries.
    """

    dataset_root = _ensure_dataset_root(dataset_root)
    image_paths = collect_eye_image_paths(dataset_root, recursive=recursive)

    records = []

    for image_path in image_paths:
        try:
            record = build_one_eye_training_record(
                image_path,
                dataset_root=dataset_root,
                load_image=load_images,
                grayscale=grayscale
            )
            records.append(record)

        except Exception:
            if ignore_invalid_files:
                continue
            raise

    return records


# ---------------------------------------------------------------------
# Public dataset-summary helpers
# ---------------------------------------------------------------------

def summarize_eye_training_records(records):
    """
    Compute a compact summary of the loaded eye-training records.

    The summary is intended for quick inspection before feature extraction and
    classifier training. It reports:
    - total record count,
    - close/open class counts,
    - per-subject sample counts,
    - optional image-shape statistics.
    """

    total_count = len(records)

    label_counter = Counter()
    subject_counter = Counter()
    image_shape_counter = Counter()

    for record in records:
        label_counter[record.get("class_name", "unknown")] += 1
        subject_counter[record.get("subject_id", "unknown")] += 1

        image_shape = record.get("image_shape")
        if image_shape is not None:
            image_shape_counter[image_shape] += 1

    summary = {
        "total_count": total_count,
        "class_counts": {
            "close": label_counter.get("close", 0),
            "open": label_counter.get("open", 0),
        },
        "subject_count": len(subject_counter),
        "samples_per_subject": dict(sorted(subject_counter.items(), key=lambda item: _natural_sort_key(item[0]))),
        "image_shape_counts": dict(sorted(image_shape_counter.items(), key=lambda item: item[0])),
    }

    return summary


def format_eye_training_summary(summary):
    """
    Convert a dataset summary dictionary into a readable multiline text block.

    This helper is useful for quick console inspection after the loader is
    tested for the first time.
    """

    lines = [
        "=== Eye training dataset summary ===",
        f"Total records:         {summary['total_count']}",
        f"Close-eye samples:     {summary['class_counts']['close']}",
        f"Open-eye samples:      {summary['class_counts']['open']}",
        f"Subject count:         {summary['subject_count']}",
    ]

    image_shape_counts = summary.get("image_shape_counts", {})
    if image_shape_counts:
        lines.append("")
        lines.append("Image shapes:")
        for image_shape, count in image_shape_counts.items():
            lines.append(f"  {image_shape}: {count}")

    return "\n".join(lines)


def print_eye_training_summary(summary):
    """
    Print the formatted eye-training dataset summary to standard output.
    """

    print(format_eye_training_summary(summary))


# ---------------------------------------------------------------------
# Optional standalone smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    default_dataset_root = Path("input/training/mrlEyes_2018_01")

    records = load_all_eye_training_records(
        default_dataset_root,
        load_images=False,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False
    )

    summary = summarize_eye_training_records(records)
    print_eye_training_summary(summary)

    if records:
        print("")
        print("First record example:")
        for key, value in records[0].items():
            if key == "image":
                print(f"  {key}: <not loaded>")
            else:
                print(f"  {key}: {value}")