# This module is the training-data entry layer of the LBP pipeline.
# The experiment-search layer and the startup model-building layer both depend
# on it whenever the project needs to move from raw files on disk to structured
# in-memory training records.
#
# In the overall project flow, this module is responsible for the earliest
# dataset-specific transformation:
#
#     dataset directory on disk
#         -> file paths
#         -> parsed filename metadata
#         -> optional image arrays
#         -> structured training records
#
# Later modules then reuse those structured records for:
# - preprocessing,
# - LBP feature extraction,
# - classifier training,
# - debugging and dataset inspection.
#
# A key design choice here is that the module does not do any preprocessing,
# feature extraction, or learning. It only makes the training dataset readable,
# explicit, and consistent for the rest of the pipeline.

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

# Path is used throughout this module so all filesystem operations are handled
# through a consistent object-based interface instead of raw strings.
from pathlib import Path

# Counter is used for compact dataset summaries, especially for:
# - class counts,
# - subject counts,
# - image-shape counts.
from collections import Counter

# re is only needed for the natural-sort helper, which keeps file ordering
# stable and human-readable when file names contain numbers.
import re

# OpenCV is used here only for loading image files from disk.
# This module deliberately stops at image loading and does not perform any
# preprocessing or analysis beyond that.
import cv2


# ---------------------------------------------------------------------
# Dataset format configuration
# ---------------------------------------------------------------------
#
# These constants document the assumptions this module makes about the training
# dataset.
#
# The mrlEyes_2018_01 dataset encodes a lot of metadata directly in the file
# names. The most important field for the current project is eye_state, because
# that becomes the binary target label used later by the classifier:
# - 0 -> close
# - 1 -> open
#
# The loader keeps the other filename-derived fields too, because they remain
# useful for inspection, debugging, and summary reports even if they are not
# currently used as model inputs.
# ---------------------------------------------------------------------

# Only files with these extensions are treated as loadable eye-image files.
# The set is intentionally a bit broader than just .png so the loader is not
# unnecessarily fragile if the dataset format changes or is re-exported.
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

# This is the project's canonical mapping from numeric dataset labels to the
# textual class names used consistently in later modules and reports.
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

    # Normalize the incoming value to Path form first so the rest of the module
    # can use Path methods regardless of whether the caller passed a string or a
    # Path object.
    dataset_root = Path(dataset_root)

    # The loader fails early and explicitly if the dataset directory is missing.
    # That makes setup problems obvious before any later pipeline stage starts.
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    # Even if the path exists, it must still be a directory, not a file or
    # another unsupported filesystem object.
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_root}")

    return dataset_root


def _is_supported_image_file(file_path):
    """
    Return True if the file has a supported image extension.

    The check is case-insensitive so that the loader remains tolerant to
    different archive or operating-system behaviors.
    """

    # This helper keeps dataset scanning logic simple by centralizing the rules
    # for "is this path a real image sample we want to consider?"
    #
    # Both conditions matter:
    # - it must be an actual file,
    # - it must have one of the supported image suffixes.
    return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def _natural_sort_key(value):
    """
    Build a natural-sort key for stable file ordering.

    Natural sorting keeps numeric parts in human order, for example:
        image2 < image10

    This is useful for reproducible dataset traversal and debugging output.
    """

    # Convert the input to lower-case string form and split it into alternating
    # non-numeric and numeric pieces.
    #
    # Example idea:
    #     "s10_img2" -> ["s", "10", "_img", "2", ""]
    #
    # Numeric pieces will later be compared as integers instead of strings so
    # that 2 comes before 10.
    parts = re.split(r"(\d+)", str(value).lower())
    key = []

    # Build the final mixed sort key:
    # - numeric fragments become integers,
    # - everything else remains text.
    #
    # This produces a stable ordering that is much easier to read and debug than
    # plain lexicographic sorting.
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

    # The dataset filename convention encodes multiple metadata fields as
    # integers. This helper makes the parsing logic reusable and keeps the error
    # messages explicit about:
    # - which field failed,
    # - in which file name,
    # - what raw value caused the problem.
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

    # The training pipeline currently assumes a strict binary task.
    # Rejecting unsupported values here prevents later modules from silently
    # training on labels that do not fit the project's binary convention.
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

    # First validate and normalize the dataset root so the scan always starts
    # from a known-good directory.
    dataset_root = _ensure_dataset_root(dataset_root)

    # The caller can choose between:
    # - recursive scan through all subdirectories,
    # - shallow scan of only the top-level directory.
    #
    # Recursive mode is the default because the dataset is commonly organized in
    # per-subject subdirectories.
    iterator = dataset_root.rglob("*") if recursive else dataset_root.glob("*")

    # Keep only supported image files and ignore everything else.
    image_paths = [
        path for path in iterator
        if _is_supported_image_file(path)
    ]

    # Natural sorting is important here so record order is deterministic across
    # runs and easier to inspect manually.
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

    # Convert the input into a Path object so stem extraction works cleanly
    # whether the caller passed a bare file name or a longer path-like value.
    path_obj = Path(file_name)
    stem = path_obj.stem

    # The dataset stores metadata as underscore-separated fields in the filename.
    parts = stem.split("_")

    # This loader expects at least the standard eight-field mrlEyes structure.
    # If fewer fields are present, the file is treated as malformed.
    if len(parts) < 8:
        raise ValueError(
            f"Unexpected mrlEyes file-name format: '{file_name}'. "
            f"Expected at least 8 underscore-separated fields, got {len(parts)}."
        )

    # Parse each encoded metadata field explicitly so later records expose the
    # original dataset semantics in structured form.
    subject_id = parts[0]
    image_id = _parse_int_field(parts[1], "image_id", file_name)
    gender = _parse_int_field(parts[2], "gender", file_name)
    glasses = _parse_int_field(parts[3], "glasses", file_name)
    eye_state = _parse_int_field(parts[4], "eye_state", file_name)
    reflections = _parse_int_field(parts[5], "reflections", file_name)
    lighting = _parse_int_field(parts[6], "lighting", file_name)
    sensor_id = _parse_int_field(parts[7], "sensor_id", file_name)

    # Convert the numeric eye-state label to the project's canonical textual
    # class name so later modules can use consistent terminology.
    class_name = _normalize_eye_state_label(eye_state)

    # Return all parsed fields in one structured dictionary.
    #
    # Both representations of the label are kept:
    # - label ...... numeric value used later by the classifier
    # - class_name . human-readable textual form used in summaries and reports
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

    # Normalize to Path so path validation and messaging stay consistent.
    image_path = Path(image_path)

    # Missing files are treated as explicit load failures.
    if not image_path.exists():
        raise FileNotFoundError(f"Training image does not exist: {image_path}")

    # The caller decides whether images should be loaded directly in grayscale or
    # preserved in color.
    #
    # For this project, grayscale is the natural default because later
    # preprocessing and LBP extraction operate on grayscale eye data.
    read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), read_flag)

    # cv2.imread returns None on load failure, so that condition is converted
    # into a clearer Python-level exception here.
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

    # Normalize the incoming file path first.
    image_path = Path(image_path)

    # Parse all metadata encoded in the file name. This is the main link between
    # the raw dataset naming convention and the project's structured record
    # representation.
    parsed = parse_eye_filename(image_path.name)

    # If the dataset root is known, store a relative path to keep records and
    # later summaries cleaner and more portable across machines.
    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        try:
            relative_path = image_path.relative_to(dataset_root)
        except ValueError:
            # If the image path is not actually inside dataset_root, fall back to
            # the file name rather than failing here.
            relative_path = image_path.name
    else:
        relative_path = image_path.name

    # Image loading is optional because some workflows only need metadata and
    # labels, while others need the actual image arrays in memory.
    image = None
    image_shape = None

    if load_image:
        image = load_one_eye_training_image(image_path, grayscale=grayscale)
        image_shape = tuple(int(value) for value in image.shape)

    # Keep the parent directory name as subject_dir because it can be useful for
    # debugging or subject-level inspection in dataset summaries.
    subject_dir = image_path.parent.name

    # Build the final structured record.
    #
    # The record intentionally combines:
    # - filesystem information,
    # - parsed metadata,
    # - binary classification label,
    # - optional loaded image data.
    #
    # This is the central data structure later modules work with before the data
    # is converted into preprocessed images or LBP feature vectors.
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

    # Validate the dataset root and then collect the candidate image paths.
    dataset_root = _ensure_dataset_root(dataset_root)
    image_paths = collect_eye_image_paths(dataset_root, recursive=recursive)

    # The final output is a list of structured training records in stable order.
    records = []

    # Each discovered image path is converted into one record.
    #
    # The caller can decide whether malformed files should:
    # - stop the whole load immediately,
    # - be skipped and ignored.
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

    # The summary compresses the main structural properties of the loaded
    # dataset into a small inspection dictionary.
    total_count = len(records)

    label_counter = Counter()
    subject_counter = Counter()
    image_shape_counter = Counter()

    # Walk through the records once and gather the most useful counts for quick
    # sanity checks before the project moves on to preprocessing and training.
    for record in records:
        label_counter[record.get("class_name", "unknown")] += 1
        subject_counter[record.get("subject_id", "unknown")] += 1

        image_shape = record.get("image_shape")
        if image_shape is not None:
            image_shape_counter[image_shape] += 1

    # The summary keeps the class counts explicitly under the canonical
    # "close"/"open" names used throughout the project.
    #
    # samples_per_subject is sorted naturally so subject ordering stays easy to
    # read when the summary is printed.
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

    # Build a compact text report from the summary dictionary.
    # The formatting is intentionally plain and stable so it can be printed
    # during smoke tests or quick debugging sessions.
    lines = [
        "=== Eye training dataset summary ===",
        f"Total records:         {summary['total_count']}",
        f"Close-eye samples:     {summary['class_counts']['close']}",
        f"Open-eye samples:      {summary['class_counts']['open']}",
        f"Subject count:         {summary['subject_count']}",
    ]

    # Image-shape counts are optional because they exist only when images were
    # actually loaded and their shapes were captured.
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

    # This helper exists only as a thin convenience wrapper around the formatter
    # so callers do not have to remember both steps.
    print(format_eye_training_summary(summary))


# ---------------------------------------------------------------------
# Optional standalone smoke test
# ---------------------------------------------------------------------

# The standalone block gives this module a minimal self-test mode.
# It allows the loader to be run directly so the dataset structure can be
# verified before the later preprocessing, LBP, and classifier layers are used.
if __name__ == "__main__":
    # Default dataset location expected by the project.
    default_dataset_root = Path("input/training/mrlEyes_2018_01")

    # Load only metadata records by default in this smoke test to keep the test
    # lighter and focused on dataset structure rather than image memory usage.
    records = load_all_eye_training_records(
        default_dataset_root,
        load_images=False,
        grayscale=True,
        recursive=True,
        ignore_invalid_files=False
    )

    # Print a compact summary of the loaded dataset.
    summary = summarize_eye_training_records(records)
    print_eye_training_summary(summary)

    # Also print one example record so the structure of the loader output is
    # easy to inspect manually.
    if records:
        print("")
        print("First record example:")
        for key, value in records[0].items():
            if key == "image":
                print(f"  {key}: <not loaded>")
            else:
                print(f"  {key}: {value}")