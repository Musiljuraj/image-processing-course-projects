"""
eye_preprocessing.py

This module implements shared eye-image preprocessing for both:
- training images loaded from the mrlEyes_2018_01 dataset,
- runtime eye ROIs extracted from detected video frames.

Its responsibilities are:
- converting input eye images to a stable grayscale representation,
- resizing them to a fixed size,
- optionally cropping the central eye-analysis band,
- optionally applying contrast normalization,
- optionally applying light filtering,
- attaching the preprocessed image back to structured training records.

The module is intentionally independent from the later LBP feature-extraction
and classifier logic so it can be developed and tested as a standalone step.
"""

from copy import deepcopy
import cv2


# ---------------------------------------------------------------------
# Default preprocessing configuration
# ---------------------------------------------------------------------
#
# The default values are intentionally close to the preprocessing logic used
# in the current eye_state.py implementation:
# - fixed normalized size,
# - central analysis-band crop,
# - histogram equalization,
# - light Gaussian blur.
#
# This makes the transition from the current heuristic eye-state pipeline to
# the future LBP-based pipeline smoother and easier to debug.
# ---------------------------------------------------------------------

DEFAULT_TARGET_SIZE = (80, 40)

DEFAULT_ANALYSIS_TOP_RATIO = 0.15
DEFAULT_ANALYSIS_BOTTOM_RATIO = 0.85

DEFAULT_CONTRAST_METHOD = "equalize"
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_GRID_SIZE = (8, 8)

DEFAULT_FILTER_NAME = "gaussian"
DEFAULT_GAUSSIAN_KERNEL_SIZE = (5, 5)
DEFAULT_MEDIAN_BLUR_SIZE = 5


# ---------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------

def _is_grayscale_image(image):
    """
    Return True when the image is already in a single-channel grayscale form.
    """

    return image is not None and len(image.shape) == 2


def _validate_target_size(target_size):
    """
    Validate and normalize the target size.

    The accepted form is:
        (width, height)

    Both values must be positive integers.
    """

    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
        raise ValueError(
            f"target_size must be a tuple/list of length 2, got: {target_size}"
        )

    width = int(target_size[0])
    height = int(target_size[1])

    if width <= 0 or height <= 0:
        raise ValueError(
            f"target_size values must be positive, got: {(width, height)}"
        )

    return (width, height)


def _validate_ratio(value, name):
    """
    Validate one ratio expected to lie in the interval [0.0, 1.0].
    """

    value = float(value)

    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in the interval [0.0, 1.0], got: {value}")

    return value


def _validate_analysis_band(top_ratio, bottom_ratio):
    """
    Validate the vertical crop interval used for the eye-analysis band.
    """

    top_ratio = _validate_ratio(top_ratio, "analysis_top_ratio")
    bottom_ratio = _validate_ratio(bottom_ratio, "analysis_bottom_ratio")

    if bottom_ratio <= top_ratio:
        raise ValueError(
            f"analysis_bottom_ratio must be greater than analysis_top_ratio, "
            f"got top={top_ratio}, bottom={bottom_ratio}"
        )

    return top_ratio, bottom_ratio


def _validate_odd_kernel_size(kernel_size, name):
    """
    Validate a 2D kernel size.

    The accepted form is:
        (width, height)

    Both values must be positive odd integers.
    """

    if not isinstance(kernel_size, (tuple, list)) or len(kernel_size) != 2:
        raise ValueError(
            f"{name} must be a tuple/list of length 2, got: {kernel_size}"
        )

    width = int(kernel_size[0])
    height = int(kernel_size[1])

    if width <= 0 or height <= 0:
        raise ValueError(f"{name} values must be positive, got: {(width, height)}")

    if width % 2 == 0 or height % 2 == 0:
        raise ValueError(f"{name} values must be odd, got: {(width, height)}")

    return (width, height)


def _validate_odd_integer(value, name):
    """
    Validate one positive odd integer parameter.
    """

    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be positive, got: {value}")

    if value % 2 == 0:
        raise ValueError(f"{name} must be odd, got: {value}")

    return value


# ---------------------------------------------------------------------
# Public configuration helpers
# ---------------------------------------------------------------------

def get_default_preprocessing_config():
    """
    Return a fresh default preprocessing configuration dictionary.
    """

    return {
        "grayscale": True,
        "target_size": DEFAULT_TARGET_SIZE,
        "crop_analysis_band": True,
        "analysis_top_ratio": DEFAULT_ANALYSIS_TOP_RATIO,
        "analysis_bottom_ratio": DEFAULT_ANALYSIS_BOTTOM_RATIO,
        "contrast_method": DEFAULT_CONTRAST_METHOD,
        "clahe_clip_limit": DEFAULT_CLAHE_CLIP_LIMIT,
        "clahe_tile_grid_size": DEFAULT_CLAHE_TILE_GRID_SIZE,
        "filter_name": DEFAULT_FILTER_NAME,
        "gaussian_kernel_size": DEFAULT_GAUSSIAN_KERNEL_SIZE,
        "median_blur_size": DEFAULT_MEDIAN_BLUR_SIZE,
    }


def validate_preprocessing_config(preprocessing_config=None):
    """
    Validate and normalize a preprocessing configuration.

    Missing keys are filled from defaults so later code can rely on a stable
    configuration structure.
    """

    config = get_default_preprocessing_config()

    if preprocessing_config is not None:
        config.update(preprocessing_config)

    config["grayscale"] = bool(config["grayscale"])
    config["target_size"] = _validate_target_size(config["target_size"])

    config["crop_analysis_band"] = bool(config["crop_analysis_band"])
    config["analysis_top_ratio"], config["analysis_bottom_ratio"] = _validate_analysis_band(
        config["analysis_top_ratio"],
        config["analysis_bottom_ratio"]
    )

    config["contrast_method"] = str(config["contrast_method"]).strip().lower()
    if config["contrast_method"] not in ("none", "equalize", "clahe"):
        raise ValueError(
            f"Unsupported contrast_method: {config['contrast_method']}. "
            f"Expected one of: none, equalize, clahe."
        )

    config["clahe_clip_limit"] = float(config["clahe_clip_limit"])
    if config["clahe_clip_limit"] <= 0:
        raise ValueError(
            f"clahe_clip_limit must be greater than 0, got: {config['clahe_clip_limit']}"
        )

    config["clahe_tile_grid_size"] = _validate_target_size(config["clahe_tile_grid_size"])

    config["filter_name"] = str(config["filter_name"]).strip().lower()
    if config["filter_name"] not in ("none", "gaussian", "median"):
        raise ValueError(
            f"Unsupported filter_name: {config['filter_name']}. "
            f"Expected one of: none, gaussian, median."
        )

    config["gaussian_kernel_size"] = _validate_odd_kernel_size(
        config["gaussian_kernel_size"],
        "gaussian_kernel_size"
    )

    config["median_blur_size"] = _validate_odd_integer(
        config["median_blur_size"],
        "median_blur_size"
    )

    return config


# ---------------------------------------------------------------------
# Public image-conversion helpers
# ---------------------------------------------------------------------

def convert_eye_to_grayscale(image):
    """
    Convert one eye image to grayscale.

    Accepted inputs:
    - already grayscale image,
    - BGR image,
    - BGRA image.

    The function raises an error for unsupported image shapes.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if _is_grayscale_image(image):
        return image

    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(
        f"Unsupported image shape for grayscale conversion: {image.shape}"
    )


def resize_eye(gray_image, target_size):
    """
    Resize one grayscale eye image to a fixed target size.

    target_size uses the form:
        (width, height)
    """

    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot resize an empty eye image.")

    target_size = _validate_target_size(target_size)

    resized = cv2.resize(
        gray_image,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    return resized


def crop_eye_analysis_band(gray_image, analysis_top_ratio, analysis_bottom_ratio):
    """
    Keep only the central vertical band of the eye image.

    This mirrors the idea already used in the current eye_state.py:
    remove parts of the crop that are more likely to contain eyebrows,
    cheek texture, or other less relevant context.
    """

    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot crop analysis band from an empty eye image.")

    analysis_top_ratio, analysis_bottom_ratio = _validate_analysis_band(
        analysis_top_ratio,
        analysis_bottom_ratio
    )

    image_height = gray_image.shape[0]

    y1 = int(round(image_height * analysis_top_ratio))
    y2 = int(round(image_height * analysis_bottom_ratio))

    y1 = max(0, min(image_height, y1))
    y2 = max(0, min(image_height, y2))

    if y2 <= y1:
        raise ValueError(
            f"Invalid cropped analysis band coordinates: y1={y1}, y2={y2}"
        )

    cropped = gray_image[y1:y2, :]

    if cropped.size == 0:
        raise ValueError("Cropped analysis band is empty.")

    return cropped


def normalize_eye_contrast(
    gray_image,
    contrast_method="equalize",
    clahe_clip_limit=DEFAULT_CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size=DEFAULT_CLAHE_TILE_GRID_SIZE
):
    """
    Apply optional contrast normalization to one grayscale eye image.

    Supported methods:
    - "none"
    - "equalize"
    - "clahe"
    """

    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot normalize contrast of an empty eye image.")

    contrast_method = str(contrast_method).strip().lower()

    if contrast_method == "none":
        return gray_image

    if contrast_method == "equalize":
        return cv2.equalizeHist(gray_image)

    if contrast_method == "clahe":
        clahe_tile_grid_size = _validate_target_size(clahe_tile_grid_size)
        clahe_clip_limit = float(clahe_clip_limit)

        if clahe_clip_limit <= 0:
            raise ValueError(
                f"clahe_clip_limit must be greater than 0, got: {clahe_clip_limit}"
            )

        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )
        return clahe.apply(gray_image)

    raise ValueError(
        f"Unsupported contrast_method: {contrast_method}. "
        f"Expected one of: none, equalize, clahe."
    )


def filter_eye_image(
    gray_image,
    filter_name="gaussian",
    gaussian_kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
    median_blur_size=DEFAULT_MEDIAN_BLUR_SIZE
):
    """
    Apply optional light filtering to one grayscale eye image.

    Supported filters:
    - "none"
    - "gaussian"
    - "median"
    """

    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot filter an empty eye image.")

    filter_name = str(filter_name).strip().lower()

    if filter_name == "none":
        return gray_image

    if filter_name == "gaussian":
        gaussian_kernel_size = _validate_odd_kernel_size(
            gaussian_kernel_size,
            "gaussian_kernel_size"
        )
        return cv2.GaussianBlur(gray_image, gaussian_kernel_size, 0)

    if filter_name == "median":
        median_blur_size = _validate_odd_integer(
            median_blur_size,
            "median_blur_size"
        )
        return cv2.medianBlur(gray_image, median_blur_size)

    raise ValueError(
        f"Unsupported filter_name: {filter_name}. "
        f"Expected one of: none, gaussian, median."
    )


# ---------------------------------------------------------------------
# Public end-to-end image preprocessing
# ---------------------------------------------------------------------

def preprocess_one_eye_image(image, preprocessing_config=None):
    """
    Preprocess one eye image using the shared preprocessing pipeline.

    Processing order:
    - grayscale conversion,
    - fixed-size resizing,
    - optional analysis-band crop,
    - optional contrast normalization,
    - optional filtering.

    The result is a single grayscale image ready for later feature extraction.
    """

    config = validate_preprocessing_config(preprocessing_config)

    if image is None:
        raise ValueError("Input image is None.")

    if config["grayscale"]:
        processed = convert_eye_to_grayscale(image)
    else:
        processed = image

    processed = resize_eye(processed, config["target_size"])

    if config["crop_analysis_band"]:
        processed = crop_eye_analysis_band(
            processed,
            config["analysis_top_ratio"],
            config["analysis_bottom_ratio"]
        )

    processed = normalize_eye_contrast(
        processed,
        contrast_method=config["contrast_method"],
        clahe_clip_limit=config["clahe_clip_limit"],
        clahe_tile_grid_size=config["clahe_tile_grid_size"]
    )

    processed = filter_eye_image(
        processed,
        filter_name=config["filter_name"],
        gaussian_kernel_size=config["gaussian_kernel_size"],
        median_blur_size=config["median_blur_size"]
    )

    return processed


def preprocess_runtime_eye_roi(eye_roi, preprocessing_config=None):
    """
    Preprocess one runtime eye ROI.

    This is a thin semantic wrapper around preprocess_one_eye_image so that
    later runtime code can use a clearly named helper.
    """

    return preprocess_one_eye_image(eye_roi, preprocessing_config=preprocessing_config)


# ---------------------------------------------------------------------
# Public record-oriented preprocessing
# ---------------------------------------------------------------------

def preprocess_one_eye_record(record, preprocessing_config=None, image_key="image"):
    """
    Preprocess one structured training record.

    The returned record is a shallow copy of the input record extended with:
    - "preprocessed_image"
    - "preprocessed_image_shape"
    - "preprocessing_config"

    The original record is left unchanged.
    """

    if record is None:
        raise ValueError("Input record is None.")

    if image_key not in record:
        raise KeyError(
            f"Record does not contain the expected image key: '{image_key}'"
        )

    image = record[image_key]

    if image is None:
        raise ValueError(
            f"Record field '{image_key}' is None. "
            f"Load images first before preprocessing records."
        )

    normalized_config = validate_preprocessing_config(preprocessing_config)
    preprocessed_image = preprocess_one_eye_image(
        image,
        preprocessing_config=normalized_config
    )

    processed_record = dict(record)
    processed_record["preprocessed_image"] = preprocessed_image
    processed_record["preprocessed_image_shape"] = tuple(
        int(value) for value in preprocessed_image.shape
    )
    processed_record["preprocessing_config"] = deepcopy(normalized_config)

    return processed_record


def preprocess_all_eye_records(records, preprocessing_config=None, image_key="image"):
    """
    Preprocess a whole list of structured eye-training records.

    Every returned record keeps the original metadata and adds the same
    preprocessing fields as preprocess_one_eye_record().
    """

    if records is None:
        raise ValueError("Input record list is None.")

    normalized_config = validate_preprocessing_config(preprocessing_config)

    processed_records = []

    for record in records:
        processed_record = preprocess_one_eye_record(
            record,
            preprocessing_config=normalized_config,
            image_key=image_key
        )
        processed_records.append(processed_record)

    return processed_records


# ---------------------------------------------------------------------
# Public summary helpers
# ---------------------------------------------------------------------

def summarize_preprocessed_eye_records(records):
    """
    Compute a compact summary of preprocessed eye records.

    The summary is useful for quick inspection before LBP extraction.
    """

    total_count = len(records)
    shape_counts = {}

    for record in records:
        image_shape = record.get("preprocessed_image_shape")
        if image_shape is None:
            continue

        shape_counts[image_shape] = shape_counts.get(image_shape, 0) + 1

    summary = {
        "total_count": total_count,
        "preprocessed_shape_counts": dict(sorted(shape_counts.items(), key=lambda item: item[0])),
    }

    return summary


def format_preprocessed_eye_summary(summary):
    """
    Convert a preprocessed-record summary into a readable multiline text block.
    """

    lines = [
        "=== Preprocessed eye dataset summary ===",
        f"Total records:         {summary['total_count']}",
    ]

    shape_counts = summary.get("preprocessed_shape_counts", {})
    if shape_counts:
        lines.append("")
        lines.append("Preprocessed image shapes:")
        for image_shape, count in shape_counts.items():
            lines.append(f"  {image_shape}: {count}")

    return "\n".join(lines)


def print_preprocessed_eye_summary(summary):
    """
    Print the formatted preprocessed-record summary to standard output.
    """

    print(format_preprocessed_eye_summary(summary))