"""
lbp_features.py

This module computes LBP-based feature representations from preprocessed
eye images.

Its responsibilities are:
- validating and normalizing LBP configuration,
- computing one LBP-coded image,
- converting LBP codes into histogram-based descriptors,
- supporting optional spatial-grid descriptors,
- extending preprocessed eye records with LBP outputs,
- providing lightweight summary helpers for inspection.

The module is intentionally separated from:
- dataset loading,
- image preprocessing,
- classifier training,
- runtime frame processing,
so that the LBP feature stage can be tested independently.
"""

from copy import deepcopy
import numpy as np


# ---------------------------------------------------------------------
# Supported configuration
# ---------------------------------------------------------------------

SUPPORTED_LBP_METHODS = {"default", "uniform"}

DEFAULT_NEIGHBORS = 8
DEFAULT_RADIUS = 1.0
DEFAULT_METHOD = "uniform"
DEFAULT_GRID_SHAPE = (1, 1)
DEFAULT_NORMALIZE_HISTOGRAM = True


# ---------------------------------------------------------------------
# Basic configuration helpers
# ---------------------------------------------------------------------

def get_default_lbp_config():
    """
    Return a fresh default LBP configuration dictionary.

    The chosen defaults are intentionally simple and suitable for the first
    smoke tests and first working training pipeline.
    """

    return {
        "neighbors": DEFAULT_NEIGHBORS,
        "radius": DEFAULT_RADIUS,
        "method": DEFAULT_METHOD,
        "grid_shape": DEFAULT_GRID_SHAPE,
        "normalize_histogram": DEFAULT_NORMALIZE_HISTOGRAM,
    }


def normalize_lbp_method_name(method):
    """
    Normalize the textual name of the selected LBP method.

    Example accepted values:
    - "default"
    - "uniform"
    """

    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    return method.strip().lower()


def validate_lbp_parameters(neighbors, radius, method):
    """
    Validate the basic LBP configuration.

    Returns:
    - validated_neighbors
    - validated_radius
    - normalized_method
    """

    if not isinstance(neighbors, int):
        raise TypeError("neighbors must be an integer.")

    if neighbors <= 0:
        raise ValueError("neighbors must be a positive integer.")

    if not isinstance(radius, (int, float)):
        raise TypeError("radius must be a number.")

    if radius <= 0:
        raise ValueError("radius must be positive.")

    normalized_method = normalize_lbp_method_name(method)

    if normalized_method not in SUPPORTED_LBP_METHODS:
        raise ValueError(
            "Unsupported LBP method. Expected one of: "
            f"{sorted(SUPPORTED_LBP_METHODS)}. Got: {method}"
        )

    return neighbors, float(radius), normalized_method


def validate_grid_shape(grid_shape):
    """
    Validate the spatial grid shape used for spatial LBP descriptors.

    Expected form:
        (grid_rows, grid_cols)
    """

    if not isinstance(grid_shape, (tuple, list)):
        raise TypeError("grid_shape must be a tuple or list of two integers.")

    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain exactly two values.")

    grid_rows, grid_cols = grid_shape

    if not isinstance(grid_rows, int) or not isinstance(grid_cols, int):
        raise TypeError("Both grid_shape values must be integers.")

    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("Both grid_shape values must be positive integers.")

    return (grid_rows, grid_cols)


def validate_lbp_config(lbp_config=None):
    """
    Validate and normalize the full LBP configuration.

    Missing keys are filled from defaults so later code can rely on one
    stable config structure.
    """

    config = get_default_lbp_config()

    if lbp_config is not None:
        if not isinstance(lbp_config, dict):
            raise TypeError("lbp_config must be a dictionary.")
        config.update(lbp_config)

    validated_neighbors, validated_radius, normalized_method = validate_lbp_parameters(
        neighbors=config["neighbors"],
        radius=config["radius"],
        method=config["method"],
    )

    validated_grid_shape = validate_grid_shape(config["grid_shape"])

    normalize_histogram = config["normalize_histogram"]
    if not isinstance(normalize_histogram, bool):
        raise TypeError("normalize_histogram must be a boolean.")

    validated_config = {
        "neighbors": validated_neighbors,
        "radius": validated_radius,
        "method": normalized_method,
        "grid_shape": validated_grid_shape,
        "normalize_histogram": normalize_histogram,
    }

    return validated_config


def get_lbp_histogram_bin_count(neighbors, method):
    """
    Determine the number of histogram bins required for the selected LBP setup.

    Practical meaning:
    - "default" -> 2 ** neighbors bins
    - "uniform" -> neighbors + 2 bins
    """

    validated_neighbors, _, normalized_method = validate_lbp_parameters(
        neighbors=neighbors,
        radius=1,
        method=method,
    )

    if normalized_method == "default":
        return 2 ** validated_neighbors

    if normalized_method == "uniform":
        return validated_neighbors + 2

    raise ValueError(
        f"Unsupported method after validation: {normalized_method}"
    )


# ---------------------------------------------------------------------
# Low-level image sampling helpers
# ---------------------------------------------------------------------

def _bilinear_sample_gray_image(gray_image, sample_y, sample_x):
    """
    Sample a grayscale image at floating-point coordinates using bilinear
    interpolation with border clipping.

    This is needed because circular LBP neighborhoods often land on
    non-integer pixel coordinates.
    """

    image_float = gray_image.astype(np.float32, copy=False)

    height, width = image_float.shape

    clipped_y = np.clip(sample_y, 0, height - 1)
    clipped_x = np.clip(sample_x, 0, width - 1)

    y0 = np.floor(clipped_y).astype(np.int32)
    x0 = np.floor(clipped_x).astype(np.int32)

    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)

    dy = clipped_y - y0
    dx = clipped_x - x0

    wa = (1.0 - dy) * (1.0 - dx)
    wb = (1.0 - dy) * dx
    wc = dy * (1.0 - dx)
    wd = dy * dx

    Ia = image_float[y0, x0]
    Ib = image_float[y0, x1]
    Ic = image_float[y1, x0]
    Id = image_float[y1, x1]

    sampled_values = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)

    return sampled_values


def _compute_binary_neighbor_patterns(gray_image, neighbors, radius):
    """
    Compute binary comparison patterns for all LBP neighbors.

    Returns:
        binary_patterns of shape:
            (neighbors, height, width)
    """

    if gray_image.ndim != 2:
        raise ValueError(
            "_compute_binary_neighbor_patterns expects a single-channel image."
        )

    image_float = gray_image.astype(np.float32, copy=False)
    height, width = image_float.shape

    y_coords, x_coords = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    center_values = image_float
    binary_patterns = np.zeros((neighbors, height, width), dtype=np.uint8)

    for neighbor_index in range(neighbors):
        angle = (2.0 * np.pi * neighbor_index) / neighbors

        offset_x = radius * np.cos(angle)
        offset_y = -radius * np.sin(angle)

        sample_x = x_coords + offset_x
        sample_y = y_coords + offset_y

        neighbor_values = _bilinear_sample_gray_image(
            gray_image=image_float,
            sample_y=sample_y,
            sample_x=sample_x,
        )

        binary_patterns[neighbor_index] = (
            neighbor_values >= center_values
        ).astype(np.uint8)

    return binary_patterns


# ---------------------------------------------------------------------
# Core LBP image computation
# ---------------------------------------------------------------------

def compute_lbp_image(gray_image, neighbors=8, radius=1, method="uniform"):
    """
    Compute one LBP-coded image from a preprocessed grayscale eye image.
    """

    if not isinstance(gray_image, np.ndarray):
        raise TypeError("gray_image must be a NumPy array.")

    if gray_image.ndim != 2:
        raise ValueError("compute_lbp_image expects a single-channel image.")

    validated_neighbors, validated_radius, normalized_method = validate_lbp_parameters(
        neighbors=neighbors,
        radius=radius,
        method=method,
    )

    binary_patterns = _compute_binary_neighbor_patterns(
        gray_image=gray_image,
        neighbors=validated_neighbors,
        radius=validated_radius,
    )

    if normalized_method == "default":
        lbp_image = np.zeros(gray_image.shape, dtype=np.uint32)

        for neighbor_index in range(validated_neighbors):
            lbp_image |= (
                binary_patterns[neighbor_index].astype(np.uint32)
                << neighbor_index
            )

        return lbp_image

    if normalized_method == "uniform":
        transitions = np.zeros(gray_image.shape, dtype=np.uint8)

        for neighbor_index in range(validated_neighbors):
            current_pattern = binary_patterns[neighbor_index]
            next_pattern = binary_patterns[
                (neighbor_index + 1) % validated_neighbors
            ]
            transitions += (current_pattern != next_pattern).astype(np.uint8)

        one_count = binary_patterns.sum(axis=0, dtype=np.uint16)

        lbp_image = np.where(
            transitions <= 2,
            one_count,
            validated_neighbors + 1,
        ).astype(np.uint16)

        return lbp_image

    raise ValueError(
        f"Unsupported LBP method after validation: {normalized_method}"
    )


# ---------------------------------------------------------------------
# Histogram and spatial-descriptor helpers
# ---------------------------------------------------------------------

def compute_lbp_histogram(lbp_image, bin_count, normalize_histogram=True):
    """
    Convert one LBP image into one histogram descriptor.
    """

    if not isinstance(lbp_image, np.ndarray):
        raise TypeError("lbp_image must be a NumPy array.")

    if lbp_image.ndim != 2:
        raise ValueError("compute_lbp_histogram expects a 2D LBP image.")

    if not isinstance(bin_count, int):
        raise TypeError("bin_count must be an integer.")

    if bin_count <= 0:
        raise ValueError("bin_count must be a positive integer.")

    if not isinstance(normalize_histogram, bool):
        raise TypeError("normalize_histogram must be a boolean.")

    lbp_values = lbp_image.astype(np.int64, copy=False)

    if np.any(lbp_values < 0):
        raise ValueError("lbp_image contains negative codes.")

    if np.any(lbp_values >= bin_count):
        raise ValueError(
            "lbp_image contains codes outside the requested histogram range."
        )

    histogram = np.bincount(
        lbp_values.ravel(),
        minlength=bin_count,
    ).astype(np.float32)

    if normalize_histogram:
        histogram_sum = histogram.sum()
        if histogram_sum > 0:
            histogram = histogram / histogram_sum

    return histogram


def split_image_into_grid_cells(image, grid_shape):
    """
    Split a 2D image into a regular spatial grid.

    Returns:
        list of 2D cells in row-major order
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("split_image_into_grid_cells expects a 2D image.")

    grid_rows, grid_cols = validate_grid_shape(grid_shape)

    image_height, image_width = image.shape

    if grid_rows > image_height:
        raise ValueError("grid_rows cannot be larger than image height.")

    if grid_cols > image_width:
        raise ValueError("grid_cols cannot be larger than image width.")

    row_boundaries = np.linspace(0, image_height, grid_rows + 1, dtype=int)
    col_boundaries = np.linspace(0, image_width, grid_cols + 1, dtype=int)

    cells = []

    for row_index in range(grid_rows):
        row_start = row_boundaries[row_index]
        row_end = row_boundaries[row_index + 1]

        for col_index in range(grid_cols):
            col_start = col_boundaries[col_index]
            col_end = col_boundaries[col_index + 1]

            cell = image[row_start:row_end, col_start:col_end]

            if cell.size == 0:
                raise ValueError(
                    "Grid split produced an empty cell. "
                    "Check grid_shape relative to image size."
                )

            cells.append(cell)

    return cells


def compute_spatial_lbp_descriptor(
    gray_image,
    neighbors=8,
    radius=1,
    method="uniform",
    grid_shape=(1, 1),
    normalize_histogram=True,
):
    """
    Compute a spatial LBP descriptor for one grayscale eye image.

    Returns:
    - lbp_image
    - descriptor_vector
    """

    validated_neighbors, validated_radius, normalized_method = validate_lbp_parameters(
        neighbors=neighbors,
        radius=radius,
        method=method,
    )

    validated_grid_shape = validate_grid_shape(grid_shape)
    bin_count = get_lbp_histogram_bin_count(
        neighbors=validated_neighbors,
        method=normalized_method,
    )

    lbp_image = compute_lbp_image(
        gray_image=gray_image,
        neighbors=validated_neighbors,
        radius=validated_radius,
        method=normalized_method,
    )

    lbp_cells = split_image_into_grid_cells(
        image=lbp_image,
        grid_shape=validated_grid_shape,
    )

    cell_histograms = []

    for cell in lbp_cells:
        cell_histogram = compute_lbp_histogram(
            lbp_image=cell,
            bin_count=bin_count,
            normalize_histogram=normalize_histogram,
        )
        cell_histograms.append(cell_histogram)

    descriptor_vector = np.concatenate(cell_histograms).astype(np.float32)

    return lbp_image, descriptor_vector


# ---------------------------------------------------------------------
# Runtime-oriented helper
# ---------------------------------------------------------------------

def extract_lbp_features_from_runtime_eye_image(preprocessed_eye_image, lbp_config=None):
    """
    Extract LBP features from one already-preprocessed runtime eye image.

    This helper is intended for later use in the video pipeline where the
    image is already preprocessed and only the LBP descriptor is needed.
    """

    if not isinstance(preprocessed_eye_image, np.ndarray):
        raise TypeError("preprocessed_eye_image must be a NumPy array.")

    if preprocessed_eye_image.ndim != 2:
        raise ValueError(
            "extract_lbp_features_from_runtime_eye_image expects a 2D grayscale image."
        )

    normalized_config = validate_lbp_config(lbp_config)
    bin_count = get_lbp_histogram_bin_count(
        neighbors=normalized_config["neighbors"],
        method=normalized_config["method"],
    )

    lbp_image, lbp_feature_vector = compute_spatial_lbp_descriptor(
        gray_image=preprocessed_eye_image,
        neighbors=normalized_config["neighbors"],
        radius=normalized_config["radius"],
        method=normalized_config["method"],
        grid_shape=normalized_config["grid_shape"],
        normalize_histogram=normalized_config["normalize_histogram"],
    )

    return {
        "lbp_image": lbp_image,
        "lbp_feature_vector": lbp_feature_vector,
        "lbp_histogram_bin_count": bin_count,
        "lbp_config": deepcopy(normalized_config),
    }


# ---------------------------------------------------------------------
# Record-oriented wrappers
# ---------------------------------------------------------------------

def extract_lbp_features_from_record(
    preprocessed_record,
    lbp_config=None,
    image_key="preprocessed_image"
):
    """
    Extract LBP features from one preprocessed eye record.

    Expected input:
    - dictionary containing at least:
        image_key -> 2D preprocessed grayscale image

    Returned record contains:
    - original record data
    - lbp_image
    - lbp_feature_vector
    - lbp_histogram_bin_count
    - lbp_feature_length
    - lbp_config
    """

    if not isinstance(preprocessed_record, dict):
        raise TypeError("preprocessed_record must be a dictionary.")

    if image_key not in preprocessed_record:
        raise KeyError(
            f"preprocessed_record must contain '{image_key}'."
        )

    preprocessed_image = preprocessed_record[image_key]

    if preprocessed_image is None:
        raise ValueError(
            f"Record field '{image_key}' is None."
        )

    normalized_config = validate_lbp_config(lbp_config)
    runtime_result = extract_lbp_features_from_runtime_eye_image(
        preprocessed_eye_image=preprocessed_image,
        lbp_config=normalized_config,
    )

    lbp_feature_record = {
        **preprocessed_record,
        "lbp_image": runtime_result["lbp_image"],
        "lbp_feature_vector": runtime_result["lbp_feature_vector"],
        "lbp_histogram_bin_count": runtime_result["lbp_histogram_bin_count"],
        "lbp_feature_length": int(runtime_result["lbp_feature_vector"].shape[0]),
        "lbp_config": deepcopy(runtime_result["lbp_config"]),
    }

    return lbp_feature_record


def extract_lbp_features_from_records(
    preprocessed_records,
    lbp_config=None,
    image_key="preprocessed_image"
):
    """
    Extract LBP features from a list of preprocessed eye records.
    """

    if not isinstance(preprocessed_records, list):
        raise TypeError("preprocessed_records must be a list.")

    normalized_config = validate_lbp_config(lbp_config)
    lbp_feature_records = []

    for preprocessed_record in preprocessed_records:
        lbp_feature_record = extract_lbp_features_from_record(
            preprocessed_record=preprocessed_record,
            lbp_config=normalized_config,
            image_key=image_key,
        )
        lbp_feature_records.append(lbp_feature_record)

    return lbp_feature_records


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------

def summarize_lbp_feature_records(lbp_feature_records):
    """
    Compute a compact summary of extracted LBP feature records.
    """

    if not isinstance(lbp_feature_records, list):
        raise TypeError("lbp_feature_records must be a list.")

    total_count = len(lbp_feature_records)
    feature_length_counts = {}
    histogram_bin_count_counts = {}

    for record in lbp_feature_records:
        feature_length = record.get("lbp_feature_length")
        bin_count = record.get("lbp_histogram_bin_count")

        if feature_length is not None:
            feature_length_counts[feature_length] = (
                feature_length_counts.get(feature_length, 0) + 1
            )

        if bin_count is not None:
            histogram_bin_count_counts[bin_count] = (
                histogram_bin_count_counts.get(bin_count, 0) + 1
            )

    summary = {
        "total_count": total_count,
        "feature_length_counts": dict(sorted(feature_length_counts.items(), key=lambda item: item[0])),
        "histogram_bin_count_counts": dict(sorted(histogram_bin_count_counts.items(), key=lambda item: item[0])),
    }

    return summary


def format_lbp_feature_summary(summary):
    """
    Convert an LBP-feature summary into a readable multiline text block.
    """

    lines = [
        "=== LBP feature summary ===",
        f"Total records:         {summary['total_count']}",
    ]

    feature_length_counts = summary.get("feature_length_counts", {})
    if feature_length_counts:
        lines.append("")
        lines.append("Feature lengths:")
        for feature_length, count in feature_length_counts.items():
            lines.append(f"  {feature_length}: {count}")

    histogram_bin_count_counts = summary.get("histogram_bin_count_counts", {})
    if histogram_bin_count_counts:
        lines.append("")
        lines.append("Histogram bin counts:")
        for bin_count, count in histogram_bin_count_counts.items():
            lines.append(f"  {bin_count}: {count}")

    return "\n".join(lines)


def print_lbp_feature_summary(summary):
    """
    Print the formatted LBP-feature summary to standard output.
    """

    print(format_lbp_feature_summary(summary))