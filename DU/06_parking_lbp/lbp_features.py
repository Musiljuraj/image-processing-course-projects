"""
lbp_features.py

Purpose of this module:
- compute LBP-based feature representations from preprocessed grayscale images
- keep LBP descriptor logic separate from preprocessing, dataset loading,
  classifier training, evaluation, and debug-output generation

Why this module exists:
After preprocessing, the project already knows how to produce one normalized
grayscale image for each parking ROI or training sample. The next logical stage
is to convert that normalized image into a texture descriptor that later
classifier logic can use.

In practice, this module provides:
1. validation and normalization of LBP configuration
2. computation of one LBP-coded image
3. computation of histogram-based LBP descriptors
4. optional spatial-grid descriptor construction
5. record-based wrappers that extend preprocessed records with LBP outputs

This module currently provides:
- normalize_lbp_method_name(...)
- validate_lbp_parameters(...)
- validate_grid_shape(...)
- get_lbp_histogram_bin_count(...)
- compute_lbp_image(...)
- compute_lbp_histogram(...)
- split_image_into_grid_cells(...)
- compute_spatial_lbp_descriptor(...)
- extract_lbp_features_from_record(...)
- extract_lbp_features_from_records(...)
"""

import numpy as np


SUPPORTED_LBP_METHODS = {"default", "uniform"}


def normalize_lbp_method_name(method):
    """
    Normalize the textual name of the selected LBP method.

    Input:
        method ... string such as:
                   "default", "uniform"

    Return:
        normalized_method ... lowercase stripped string
    """

    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    return method.strip().lower()


def validate_lbp_parameters(neighbors, radius, method):
    """
    Validate the basic LBP configuration.

    Inputs:
        neighbors ... expected to be a positive integer
        radius .... expected to be a positive number
        method .... expected to be one of the supported LBP methods

    Return:
        validated_neighbors ... positive integer
        validated_radius .... positive float
        normalized_method ... normalized method string

    Why this helper exists:
    LBP descriptor construction depends on a small number of core parameters.
    Keeping their validation in one place makes later functions clearer and
    keeps error handling consistent.
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

    Input:
        grid_shape ... expected to be a tuple or list of:
                       (grid_rows, grid_cols)

    Return:
        validated_grid_shape ... normalized tuple:
                                 (grid_rows, grid_cols)

    Why this helper exists:
    Spatial LBP descriptors split the image into multiple cells. Keeping the
    validation here makes the later grid-based logic safer and easier to read.
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


def get_lbp_histogram_bin_count(neighbors, method):
    """
    Determine the number of histogram bins required for the selected LBP setup.

    Inputs:
        neighbors ... positive integer
        method .... normalized or raw method string

    Return:
        bin_count ... number of histogram bins

    Practical meaning:
    - for "default" LBP, the number of possible codes is 2 ** neighbors
    - for "uniform" LBP, the common compact representation uses:
      neighbors + 2 bins
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


def _bilinear_sample_gray_image(gray_image, sample_y, sample_x):
    """
    Sample a grayscale image at floating-point coordinates using bilinear
    interpolation with border clipping.

    Inputs:
        gray_image ... single-channel grayscale image
        sample_y ... array of y coordinates
        sample_x ... array of x coordinates

    Return:
        sampled_values ... array of sampled grayscale intensities

    Why this helper exists:
    Circular LBP neighborhoods usually do not land exactly on integer pixel
    coordinates, especially for larger radii. Bilinear interpolation therefore
    gives a smoother and more faithful estimate of neighbor values.
    """

    image_float = gray_image.astype(np.float32, copy=False)

    height, width = image_float.shape

    clipped_y = np.clip(sample_y, 0, height - 1)
    clipped_x = np.clip(sample_x, 0, width - 1)

    y0 = np.floor(clipped_y).astype(np.int32)
    x0 = np.floor(clipped_x).astype(np.int32)

    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)

    wa = (y1 - clipped_y) * (x1 - clipped_x)
    wb = (y1 - clipped_y) * (clipped_x - x0)
    wc = (clipped_y - y0) * (x1 - clipped_x)
    wd = (clipped_y - y0) * (clipped_x - x0)

    Ia = image_float[y0, x0]
    Ib = image_float[y0, x1]
    Ic = image_float[y1, x0]
    Id = image_float[y1, x1]

    sampled_values = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)

    return sampled_values


def _compute_binary_neighbor_patterns(gray_image, neighbors, radius):
    """
    Compute binary comparison patterns for all LBP neighbors.

    Inputs:
        gray_image ... single-channel grayscale image
        neighbors ... positive integer
        radius .... positive float

    Return:
        binary_patterns ... NumPy array of shape:
                            (neighbors, height, width)
                            with values 0 or 1

    Why this helper exists:
    Both "default" and "uniform" LBP are derived from the same set of binary
    neighbor comparisons, so it is useful to compute them once and reuse them.
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


def compute_lbp_image(gray_image, neighbors=8, radius=1, method="uniform"):
    """
    Compute one LBP-coded image from a preprocessed grayscale image.

    Inputs:
        gray_image ... single-channel grayscale image
        neighbors ... positive integer
        radius .... positive number
        method .... one of:
                     "default", "uniform"

    Return:
        lbp_image ... 2D NumPy array of LBP codes

    Why this function exists:
    This is the core low-level LBP computation. Later functions can then convert
    the LBP image into histogram-based descriptors.
    """

    if not isinstance(gray_image, np.ndarray):
        raise TypeError("gray_image must be a NumPy array.")

    if gray_image.ndim != 2:
        raise ValueError("compute_lbp_image expects a single-channel image.")

    validated_neighbors, validated_radius, normalized_method = (
        validate_lbp_parameters(
            neighbors=neighbors,
            radius=radius,
            method=method,
        )
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


def compute_lbp_histogram(
    lbp_image,
    bin_count,
    normalize_histogram=True,
):
    """
    Convert one LBP image into one histogram descriptor.

    Inputs:
        lbp_image ............... 2D NumPy array of LBP codes
        bin_count ............... positive integer number of bins
        normalize_histogram ..... if True, normalize histogram to sum to 1

    Return:
        histogram ............... 1D NumPy array of length bin_count

    Why this function exists:
    Classifiers usually work with compact feature vectors rather than directly
    with full LBP images. Histogram descriptors are the standard next step.
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
    Split an image into a regular spatial grid.

    Inputs:
        image ...... 2D image array
        grid_shape . tuple:
                     (grid_rows, grid_cols)

    Return:
        cells ...... list of 2D image-cell arrays in row-major order

    Why this function exists:
    A single global histogram loses most spatial layout information. Splitting
    the image into cells helps preserve rough spatial structure.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")

    if image.ndim != 2:
        raise ValueError("split_image_into_grid_cells expects a 2D image.")

    grid_rows, grid_cols = validate_grid_shape(grid_shape)

    image_height, image_width = image.shape

    if grid_rows > image_height:
        raise ValueError(
            "grid_rows cannot be larger than image height."
        )

    if grid_cols > image_width:
        raise ValueError(
            "grid_cols cannot be larger than image width."
        )

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
    Compute a spatial LBP descriptor for one grayscale image.

    Inputs:
        gray_image .............. single-channel grayscale image
        neighbors ............... positive integer
        radius .................. positive number
        method .................. "default" or "uniform"
        grid_shape .............. tuple:
                                  (grid_rows, grid_cols)
        normalize_histogram ..... if True, normalize each cell histogram

    Return:
        lbp_image ............... computed LBP-coded image
        descriptor_vector ....... 1D NumPy array formed by concatenating
                                  cell histograms

    Why this function exists:
    This is the main practical descriptor builder for the project:
    - compute one LBP image
    - split it into cells
    - compute one histogram per cell
    - concatenate histograms into one feature vector
    """

    validated_neighbors, validated_radius, normalized_method = (
        validate_lbp_parameters(
            neighbors=neighbors,
            radius=radius,
            method=method,
        )
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


def extract_lbp_features_from_record(preprocessed_record, lbp_config):
    """
    Extract LBP features from one preprocessed record.

    Inputs:
        preprocessed_record ... dictionary expected to contain:
                                - processed_image
                                and typically also metadata from earlier stages

        lbp_config ............ dictionary describing the LBP settings,
                                for example:
                                {
                                    "neighbors": 8,
                                    "radius": 1,
                                    "method": "uniform",
                                    "grid_shape": (4, 4),
                                    "normalize_histogram": True
                                }

    Return:
        lbp_feature_record .... dictionary containing:
                                - original record data
                                - lbp_image
                                - lbp_feature_vector
                                - lbp_config
                                - lbp_histogram_bin_count

    Why this function exists:
    It matches the record-based style already used in the rest of the project:
    one structured input record comes in, one richer structured record comes
    out.
    """

    if not isinstance(preprocessed_record, dict):
        raise TypeError("preprocessed_record must be a dictionary.")

    if "processed_image" not in preprocessed_record:
        raise KeyError(
            "preprocessed_record must contain 'processed_image'."
        )

    if not isinstance(lbp_config, dict):
        raise TypeError("lbp_config must be a dictionary.")

    neighbors = lbp_config.get("neighbors", 8)
    radius = lbp_config.get("radius", 1)
    method = lbp_config.get("method", "uniform")
    grid_shape = lbp_config.get("grid_shape", (1, 1))
    normalize_histogram = lbp_config.get("normalize_histogram", True)

    if not isinstance(normalize_histogram, bool):
        raise TypeError("normalize_histogram must be a boolean.")

    validated_neighbors, validated_radius, normalized_method = (
        validate_lbp_parameters(
            neighbors=neighbors,
            radius=radius,
            method=method,
        )
    )

    validated_grid_shape = validate_grid_shape(grid_shape)
    bin_count = get_lbp_histogram_bin_count(
        neighbors=validated_neighbors,
        method=normalized_method,
    )

    lbp_image, lbp_feature_vector = compute_spatial_lbp_descriptor(
        gray_image=preprocessed_record["processed_image"],
        neighbors=validated_neighbors,
        radius=validated_radius,
        method=normalized_method,
        grid_shape=validated_grid_shape,
        normalize_histogram=normalize_histogram,
    )

    lbp_feature_record = {
        **preprocessed_record,
        "lbp_image": lbp_image,
        "lbp_feature_vector": lbp_feature_vector,
        "lbp_histogram_bin_count": bin_count,
        "lbp_config": {
            "neighbors": validated_neighbors,
            "radius": validated_radius,
            "method": normalized_method,
            "grid_shape": validated_grid_shape,
            "normalize_histogram": normalize_histogram,
        },
    }

    return lbp_feature_record


def extract_lbp_features_from_records(preprocessed_records, lbp_config):
    """
    Extract LBP features from a list of preprocessed records.

    Inputs:
        preprocessed_records ... list of preprocessed record dictionaries
        lbp_config ............ dictionary describing LBP settings

    Return:
        lbp_feature_records ... list of enriched LBP feature records

    Overall idea:
    This function mirrors the structure used in other project modules:
    - one image / record
    - one feature record
    or
    - many input records
    - many output feature records
    """

    if not isinstance(preprocessed_records, list):
        raise TypeError("preprocessed_records must be a list.")

    lbp_feature_records = []

    for preprocessed_record in preprocessed_records:
        lbp_feature_record = extract_lbp_features_from_record(
            preprocessed_record=preprocessed_record,
            lbp_config=lbp_config,
        )
        lbp_feature_records.append(lbp_feature_record)

    return lbp_feature_records