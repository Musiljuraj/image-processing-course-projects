# This module is the shared image-normalization layer of the project.
# It sits between raw eye images and the later LBP feature-extraction stage.
#
# In the overall pipeline, its job is to make all eye inputs look structurally
# comparable before feature extraction happens:
#
#     raw eye image / runtime eye ROI
#         -> grayscale normalization
#         -> fixed-size resizing
#         -> optional central-band crop
#         -> optional contrast normalization
#         -> optional light filtering
#         -> preprocessed eye image
#
# The key design idea is consistency:
# the exact same preprocessing vocabulary is used for both
# - training dataset images, and
# - runtime eye ROIs extracted from the video.
#
# That keeps the training-time representation and runtime representation aligned,
# which is important because the classifier should see the same kind of inputs
# during training and during final inference.

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

# deepcopy is used when preprocessing configuration dictionaries are attached
# back to processed records. This keeps record-level metadata stable and avoids
# accidental mutation of shared config objects elsewhere in the pipeline.
from copy import deepcopy

# OpenCV provides all actual image operations in this module:
# - color conversion,
# - resizing,
# - histogram equalization / CLAHE,
# - Gaussian blur,
# - median blur.
import cv2


# ---------------------------------------------------------------------
# Default preprocessing configuration
# ---------------------------------------------------------------------
#
# These constants define the project's default preprocessing behavior.
#
# The defaults are intentionally chosen to stay close to the already existing
# heuristic eye-state pipeline from eye_state.py, so the transition to the
# LBP-based pipeline remains conceptually smooth:
# - the eye is normalized to a fixed size,
# - only the central analysis band is kept,
# - contrast is improved,
# - light smoothing is applied.
#
# This means the later LBP feature stage starts from an eye image that is
# already stabilized in size, cropped to the most relevant vertical region,
# and cleaned up just enough to reduce sensitivity to noise.
# ---------------------------------------------------------------------

# Standard target size used across the whole project for normalized eye images.
DEFAULT_TARGET_SIZE = (80, 40)

# Vertical crop ratios defining the central analysis band kept from the eye.
# The goal is to reduce influence from eyebrows and lower-face texture while
# preserving the visually relevant eye region.
DEFAULT_ANALYSIS_TOP_RATIO = 0.15
DEFAULT_ANALYSIS_BOTTOM_RATIO = 0.85

# Contrast-normalization defaults.
# Histogram equalization is the default baseline because it is simple and works
# well as a generic grayscale contrast enhancement step.
DEFAULT_CONTRAST_METHOD = "equalize"
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_GRID_SIZE = (8, 8)

# Light denoising / smoothing defaults.
# Gaussian blur is the default because it reduces small local fluctuations
# without changing overall image structure too aggressively.
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

    # In this project, a grayscale image is represented as a 2D array.
    # This helper exists so later conversion logic can quickly detect when
    # grayscale conversion is unnecessary.
    return image is not None and len(image.shape) == 2


def _validate_target_size(target_size):
    """
    Validate and normalize the target size.

    The accepted form is:
        (width, height)

    Both values must be positive integers.
    """

    # The whole preprocessing pipeline assumes image sizes are provided in the
    # standard OpenCV-friendly form: (width, height).
    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
        raise ValueError(
            f"target_size must be a tuple/list of length 2, got: {target_size}"
        )

    # Normalize both values to plain integers so later operations have a stable
    # representation even if the caller passed other numeric types.
    width = int(target_size[0])
    height = int(target_size[1])

    # Zero or negative image sizes are always invalid.
    if width <= 0 or height <= 0:
        raise ValueError(
            f"target_size values must be positive, got: {(width, height)}"
        )

    return (width, height)


def _validate_ratio(value, name):
    """
    Validate one ratio expected to lie in the interval [0.0, 1.0].
    """

    # Many preprocessing parameters are ratios describing how much of an image
    # should be kept or where a crop boundary should be placed.
    value = float(value)

    # Ratios outside [0, 1] would not make geometric sense in this module.
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in the interval [0.0, 1.0], got: {value}")

    return value


def _validate_analysis_band(top_ratio, bottom_ratio):
    """
    Validate the vertical crop interval used for the eye-analysis band.
    """

    # First validate both endpoints as proper image-relative ratios.
    top_ratio = _validate_ratio(top_ratio, "analysis_top_ratio")
    bottom_ratio = _validate_ratio(bottom_ratio, "analysis_bottom_ratio")

    # The bottom boundary must lie below the top boundary, otherwise the crop
    # interval would be empty or inverted.
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

    # Gaussian filters expect a 2D kernel size in (width, height) form.
    if not isinstance(kernel_size, (tuple, list)) or len(kernel_size) != 2:
        raise ValueError(
            f"{name} must be a tuple/list of length 2, got: {kernel_size}"
        )

    width = int(kernel_size[0])
    height = int(kernel_size[1])

    # Both dimensions must be strictly positive.
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} values must be positive, got: {(width, height)}")

    # Odd dimensions are required by the intended OpenCV blur operations.
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError(f"{name} values must be odd, got: {(width, height)}")

    return (width, height)


def _validate_odd_integer(value, name):
    """
    Validate one positive odd integer parameter.
    """

    # Median blur uses one odd integer kernel size rather than a 2D tuple.
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

    # A new dictionary is returned every time so callers can safely modify the
    # resulting config without affecting any shared global state.
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

    # Start from the default config so missing caller-provided fields inherit
    # stable project defaults.
    config = get_default_preprocessing_config()

    # Then override defaults with any explicitly supplied settings.
    if preprocessing_config is not None:
        config.update(preprocessing_config)

    # Normalize simple boolean / tuple / ratio fields first.
    config["grayscale"] = bool(config["grayscale"])
    config["target_size"] = _validate_target_size(config["target_size"])

    config["crop_analysis_band"] = bool(config["crop_analysis_band"])
    config["analysis_top_ratio"], config["analysis_bottom_ratio"] = _validate_analysis_band(
        config["analysis_top_ratio"],
        config["analysis_bottom_ratio"]
    )

    # Normalize and validate the contrast-normalization branch.
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

    # Normalize and validate the filtering branch.
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

    # The returned config is now complete and normalized, so later functions can
    # rely on it without repeating structural validation.
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

    # None is always invalid because later OpenCV operations need a real array.
    if image is None:
        raise ValueError("Input image is None.")

    # If the image is already grayscale, keep it unchanged.
    if _is_grayscale_image(image):
        return image

    # Standard 3-channel BGR image -> grayscale.
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Standard 4-channel BGRA image -> grayscale.
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    # Anything else is outside the supported image formats for this pipeline.
    raise ValueError(
        f"Unsupported image shape for grayscale conversion: {image.shape}"
    )


def resize_eye(gray_image, target_size):
    """
    Resize one grayscale eye image to a fixed target size.

    target_size uses the form:
        (width, height)
    """

    # Resizing an empty or missing image would be meaningless and would likely
    # fail inside OpenCV anyway, so it is rejected explicitly here.
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot resize an empty eye image.")

    # Normalize the requested target size before calling OpenCV.
    target_size = _validate_target_size(target_size)

    # INTER_AREA is a reasonable generic choice here for stable resizing of eye
    # images into the project's standard normalized shape.
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

    # The crop only makes sense on a non-empty grayscale image.
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot crop analysis band from an empty eye image.")

    # Normalize and validate the crop interval first.
    analysis_top_ratio, analysis_bottom_ratio = _validate_analysis_band(
        analysis_top_ratio,
        analysis_bottom_ratio
    )

    # Compute integer crop boundaries relative to image height.
    image_height = gray_image.shape[0]

    y1 = int(round(image_height * analysis_top_ratio))
    y2 = int(round(image_height * analysis_bottom_ratio))

    # Clamp the crop coordinates to valid image bounds.
    y1 = max(0, min(image_height, y1))
    y2 = max(0, min(image_height, y2))

    # If the interval collapses, the crop is invalid.
    if y2 <= y1:
        raise ValueError(
            f"Invalid cropped analysis band coordinates: y1={y1}, y2={y2}"
        )

    # Keep all columns and only the selected vertical band.
    cropped = gray_image[y1:y2, :]

    # Defensive check to reject a still-empty crop.
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

    # Contrast normalization only makes sense on a valid grayscale image.
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot normalize contrast of an empty eye image.")

    # Normalize the textual method name once so dispatch stays predictable.
    contrast_method = str(contrast_method).strip().lower()

    # "none" means keep the grayscale image as it is.
    if contrast_method == "none":
        return gray_image

    # Global histogram equalization is the project's default baseline.
    if contrast_method == "equalize":
        return cv2.equalizeHist(gray_image)

    # CLAHE is the more locally adaptive contrast-improvement option.
    if contrast_method == "clahe":
        clahe_tile_grid_size = _validate_target_size(clahe_tile_grid_size)
        clahe_clip_limit = float(clahe_clip_limit)

        if clahe_clip_limit <= 0:
            raise ValueError(
                f"clahe_clip_limit must be greater than 0, got: {clahe_clip_limit}"
            )

        # Build the CLAHE operator with validated settings and apply it.
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size
        )
        return clahe.apply(gray_image)

    # Anything else is unsupported configuration.
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

    # Filtering only makes sense on a real non-empty image.
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Cannot filter an empty eye image.")

    # Normalize the filter selector for predictable branching.
    filter_name = str(filter_name).strip().lower()

    # "none" means skip filtering completely.
    if filter_name == "none":
        return gray_image

    # Gaussian blur is used as a light smoothing step for reducing small local
    # fluctuations while preserving overall structure.
    if filter_name == "gaussian":
        gaussian_kernel_size = _validate_odd_kernel_size(
            gaussian_kernel_size,
            "gaussian_kernel_size"
        )
        return cv2.GaussianBlur(gray_image, gaussian_kernel_size, 0)

    # Median blur is an alternative light denoising option that can be useful
    # for impulse-like noise patterns.
    if filter_name == "median":
        median_blur_size = _validate_odd_integer(
            median_blur_size,
            "median_blur_size"
        )
        return cv2.medianBlur(gray_image, median_blur_size)

    # Any unsupported filter name is rejected explicitly.
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

    # Validate and normalize the preprocessing configuration first so every
    # later decision in this function works from one complete config dictionary.
    config = validate_preprocessing_config(preprocessing_config)

    # Reject a missing image before any image-processing operations begin.
    if image is None:
        raise ValueError("Input image is None.")

    # -------------------------------------------------------------
    # Step 1: grayscale normalization
    # -------------------------------------------------------------
    #
    # Most later stages in the project assume grayscale eye images.
    # If grayscale processing is enabled, convert supported input formats into a
    # single grayscale representation. Otherwise, preserve the incoming image.
    if config["grayscale"]:
        processed = convert_eye_to_grayscale(image)
    else:
        processed = image

    # -------------------------------------------------------------
    # Step 2: fixed-size normalization
    # -------------------------------------------------------------
    #
    # This ensures every eye sample reaches later stages in a consistent size,
    # which is especially important for stable LBP feature extraction.
    processed = resize_eye(processed, config["target_size"])

    # -------------------------------------------------------------
    # Step 3: optional analysis-band crop
    # -------------------------------------------------------------
    #
    # This keeps only the central vertical eye band when enabled, reducing the
    # influence of less relevant surrounding structures.
    if config["crop_analysis_band"]:
        processed = crop_eye_analysis_band(
            processed,
            config["analysis_top_ratio"],
            config["analysis_bottom_ratio"]
        )

    # -------------------------------------------------------------
    # Step 4: optional contrast normalization
    # -------------------------------------------------------------
    #
    # This improves consistency of intensity distribution between samples before
    # later feature extraction.
    processed = normalize_eye_contrast(
        processed,
        contrast_method=config["contrast_method"],
        clahe_clip_limit=config["clahe_clip_limit"],
        clahe_tile_grid_size=config["clahe_tile_grid_size"]
    )

    # -------------------------------------------------------------
    # Step 5: optional light filtering
    # -------------------------------------------------------------
    #
    # This acts as the final cleanup/smoothing step before the image is handed
    # over to the LBP feature-extraction layer.
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

    # The logic is identical to general single-image preprocessing.
    # This wrapper exists purely to make runtime call sites read more clearly:
    # they are preprocessing a detected eye ROI, not an arbitrary image.
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

    # A missing record is always invalid.
    if record is None:
        raise ValueError("Input record is None.")

    # The caller chooses which record field contains the raw image, but that key
    # must actually be present.
    if image_key not in record:
        raise KeyError(
            f"Record does not contain the expected image key: '{image_key}'"
        )

    image = record[image_key]

    # Records used for preprocessing must already contain loaded image data.
    if image is None:
        raise ValueError(
            f"Record field '{image_key}' is None. "
            f"Load images first before preprocessing records."
        )

    # Normalize the config once and preprocess the record's image using the same
    # shared single-image pipeline as everywhere else in the project.
    normalized_config = validate_preprocessing_config(preprocessing_config)
    preprocessed_image = preprocess_one_eye_image(
        image,
        preprocessing_config=normalized_config
    )

    # Build a shallow copy of the original record so metadata is preserved while
    # the original input record remains untouched.
    processed_record = dict(record)

    # Attach the main preprocessing outputs:
    # - the processed image,
    # - its resulting shape,
    # - the exact normalized config that produced it.
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

    # The whole-record-list wrapper exists so the training-side pipeline can
    # preprocess complete datasets in one simple call.
    if records is None:
        raise ValueError("Input record list is None.")

    # Normalize the config once and reuse it for every record to keep behavior
    # consistent across the whole collection.
    normalized_config = validate_preprocessing_config(preprocessing_config)

    processed_records = []

    # Process each record independently and accumulate the enriched results.
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

    # This summary focuses mainly on how many records exist and what processed
    # image shapes they ended up with.
    total_count = len(records)
    shape_counts = {}

    # Count each observed preprocessed image shape so preprocessing consistency
    # can be checked quickly.
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

    # Convert the compact summary dictionary into plain text that is easy to
    # print during smoke tests or quick debugging runs.
    lines = [
        "=== Preprocessed eye dataset summary ===",
        f"Total records:         {summary['total_count']}",
    ]

    # If processed image-shape counts are available, list them explicitly so it
    # is easy to verify that preprocessing produced the expected geometry.
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

    # Thin convenience wrapper so callers can print in one step.
    print(format_preprocessed_eye_summary(summary))