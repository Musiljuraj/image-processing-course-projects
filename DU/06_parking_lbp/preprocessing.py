"""
preprocessing.py

Purpose of this module:
- perform preprocessing on ROI patches before LBP descriptor extraction
- keep preprocessing logic separate from data loading, geometry, ROI extraction,
  feature extraction, classifier training, and debug-output generation

Why this module exists:
After ROI extraction, the project already knows how to:
- load the parking-space map
- load test images
- extract each parking space as its own ROI patch

The next logical stage is to normalize those ROI patches so that LBP features
are computed from a stable and comparable image representation.

In practice, that means:
1. convert each ROI patch to grayscale
2. resize each ROI patch to one fixed target size
3. optionally normalize contrast / illumination
4. optionally apply a light smoothing filter

Current responsibilities:
- convert one ROI image to grayscale
- resize one grayscale ROI image to a target size
- optionally normalize contrast
- optionally apply one selected filter
- preprocess one ROI record
- preprocess all ROI records from one source image
"""

import cv2


def normalize_filter_name(filter_name):
    """
    Normalize the textual name of the selected smoothing filter.

    Input:
        filter_name ... string such as:
                        "none", "box", "gaussian", "median"

    Return:
        normalized_filter_name ... lowercase stripped string
    """

    if not isinstance(filter_name, str):
        raise TypeError("filter_name must be a string.")

    return filter_name.strip().lower()


def normalize_contrast_method_name(contrast_method):
    """
    Normalize the textual name of the selected contrast normalization method.

    Input:
        contrast_method ... string such as:
                            "none", "equalize_hist", "clahe"

    Return:
        normalized_contrast_method ... lowercase stripped string
    """

    if not isinstance(contrast_method, str):
        raise TypeError("contrast_method must be a string.")

    return contrast_method.strip().lower()


def validate_kernel_size(kernel_size):
    """
    Validate the kernel size used by smoothing filters.

    Input:
        kernel_size ... expected to be a positive odd integer

    Return:
        kernel_size ... same value if valid

    Why this helper exists:
    For Gaussian and median filtering, OpenCV expects an odd kernel size.
    Using one validation function keeps error handling in one place.
    """

    if not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer.")

    if kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer.")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    return kernel_size


def validate_target_size(target_size):
    """
    Validate the target size used for ROI resizing.

    Input:
        target_size ... expected to be a tuple or list of:
                        (target_width, target_height)

    Return:
        validated_target_size ... normalized tuple:
                                  (target_width, target_height)

    Why this helper exists:
    LBP descriptors should be computed from ROI images of a consistent size.
    Keeping the validation here makes resizing configuration safer and clearer.
    """

    if not isinstance(target_size, (tuple, list)):
        raise TypeError("target_size must be a tuple or list of two integers.")

    if len(target_size) != 2:
        raise ValueError("target_size must contain exactly two values.")

    target_width, target_height = target_size

    if not isinstance(target_width, int) or not isinstance(target_height, int):
        raise TypeError("Both target_size values must be integers.")

    if target_width <= 0 or target_height <= 0:
        raise ValueError("Both target_size values must be positive integers.")

    return (target_width, target_height)


def validate_clahe_tile_grid_size(tile_grid_size):
    """
    Validate CLAHE tile-grid size.

    Input:
        tile_grid_size ... expected to be a tuple or list of:
                           (grid_width, grid_height)

    Return:
        validated_tile_grid_size ... normalized tuple:
                                     (grid_width, grid_height)
    """

    if not isinstance(tile_grid_size, (tuple, list)):
        raise TypeError(
            "clahe_tile_grid_size must be a tuple or list of two integers."
        )

    if len(tile_grid_size) != 2:
        raise ValueError(
            "clahe_tile_grid_size must contain exactly two values."
        )

    grid_width, grid_height = tile_grid_size

    if not isinstance(grid_width, int) or not isinstance(grid_height, int):
        raise TypeError(
            "Both clahe_tile_grid_size values must be integers."
        )

    if grid_width <= 0 or grid_height <= 0:
        raise ValueError(
            "Both clahe_tile_grid_size values must be positive integers."
        )

    return (grid_width, grid_height)


def convert_roi_to_grayscale(roi_image):
    """
    Convert one ROI patch to grayscale.

    Input:
        roi_image ... ROI image array
                      usually a BGR image loaded/produced by OpenCV

    Return:
        grayscale_image ... single-channel grayscale image

    Why this function exists:
    LBP descriptors are typically computed from intensity patterns rather than
    from full color information. Grayscale conversion therefore simplifies the
    image while preserving the brightness structure that will later determine
    local texture patterns.

    Additional behavior:
    - if the input image is already single-channel, a copy is returned
    - unsupported image shapes raise a clear error
    """

    if roi_image.ndim == 2:
        return roi_image.copy()

    if roi_image.ndim == 3 and roi_image.shape[2] == 3:
        grayscale_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        return grayscale_image

    raise ValueError(
        "ROI image has unsupported shape for grayscale conversion: "
        f"{roi_image.shape}"
    )


def resize_grayscale_roi(gray_image, target_size=(80, 80)):
    """
    Resize one grayscale ROI image to a fixed target size.

    Inputs:
        gray_image ... single-channel grayscale image
        target_size ... tuple:
                        (target_width, target_height)

    Return:
        resized_image ... resized grayscale image

    Why this function exists:
    LBP histograms and spatial LBP descriptors are easier to compare when all
    ROI patches have a consistent size. This also aligns extracted parking
    patches with the fixed-size training samples.
    """

    if gray_image.ndim != 2:
        raise ValueError(
            "resize_grayscale_roi expects a single-channel grayscale image."
        )

    validated_target_size = validate_target_size(target_size)
    target_width, target_height = validated_target_size

    resized_image = cv2.resize(
        gray_image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized_image


def apply_contrast_normalization(
    gray_image,
    contrast_method="none",
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=(8, 8),
):
    """
    Apply the selected contrast normalization method to one grayscale image.

    Inputs:
        gray_image ................. single-channel grayscale image
        contrast_method ............ one of:
                                     "none", "equalize_hist", "clahe"
        clahe_clip_limit ........... positive number used by CLAHE
        clahe_tile_grid_size ....... tuple:
                                     (grid_width, grid_height)

    Return:
        normalized_image ... grayscale image after contrast normalization

    Supported methods:
    - "none"          : no contrast normalization, return a copy
    - "equalize_hist" : global histogram equalization
    - "clahe"         : adaptive histogram equalization

    Why this function exists:
    Parking-space images may differ in brightness, shadows, or exposure.
    Moderate contrast normalization can make later LBP descriptors more stable.
    """

    if gray_image.ndim != 2:
        raise ValueError(
            "apply_contrast_normalization expects a single-channel image."
        )

    normalized_contrast_method = normalize_contrast_method_name(
        contrast_method
    )

    if normalized_contrast_method == "none":
        return gray_image.copy()

    if normalized_contrast_method == "equalize_hist":
        normalized_image = cv2.equalizeHist(gray_image)
        return normalized_image

    if normalized_contrast_method == "clahe":
        if not isinstance(clahe_clip_limit, (int, float)):
            raise TypeError("clahe_clip_limit must be a number.")

        if clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive.")

        validated_tile_grid_size = validate_clahe_tile_grid_size(
            clahe_tile_grid_size
        )

        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=validated_tile_grid_size,
        )
        normalized_image = clahe.apply(gray_image)
        return normalized_image

    raise ValueError(
        "Unsupported contrast_method. Expected one of: "
        "'none', 'equalize_hist', 'clahe'. "
        f"Got: {contrast_method}"
    )


def apply_filter(gray_image, filter_name="none", kernel_size=5):
    """
    Apply the selected smoothing filter to one grayscale image.

    Inputs:
        gray_image ... single-channel grayscale image
        filter_name ... one of:
                        "none", "box", "gaussian", "median"
        kernel_size ... positive odd integer for filters that use a kernel

    Return:
        processed_image ... filtered grayscale image

    Supported filters:
    - "none"     : no filtering, return a copy of the input
    - "box"      : simple average blur
    - "gaussian" : Gaussian blur
    - "median"   : median filter

    Why this function exists:
    For the LBP pipeline, filtering is optional and should usually be light.
    Small amounts of smoothing may reduce noise, but too much smoothing can
    weaken local texture patterns that LBP depends on.
    """

    if gray_image.ndim != 2:
        raise ValueError("apply_filter expects a single-channel image.")

    normalized_filter_name = normalize_filter_name(filter_name)

    if normalized_filter_name == "none":
        return gray_image.copy()

    validated_kernel_size = validate_kernel_size(kernel_size)

    if normalized_filter_name == "box":
        processed_image = cv2.blur(
            gray_image,
            (validated_kernel_size, validated_kernel_size),
        )
        return processed_image

    if normalized_filter_name == "gaussian":
        processed_image = cv2.GaussianBlur(
            gray_image,
            (validated_kernel_size, validated_kernel_size),
            0,
        )
        return processed_image

    if normalized_filter_name == "median":
        processed_image = cv2.medianBlur(gray_image, validated_kernel_size)
        return processed_image

    raise ValueError(
        "Unsupported filter_name. Expected one of: "
        "'none', 'box', 'gaussian', 'median'. "
        f"Got: {filter_name}"
    )


def preprocess_one_roi(roi_record, preprocessing_config):
    """
    Preprocess one ROI record.

    Inputs:
        roi_record ........ dictionary produced by roi_extraction.py
                            expected keys:
                            - source_image_name
                            - space_index
                            - polygon
                            - roi_image

        preprocessing_config ... dictionary describing preprocessing settings,
                                 for example:
                                 {
                                     "target_size": (80, 80),
                                     "contrast_method": "clahe",
                                     "clahe_clip_limit": 2.0,
                                     "clahe_tile_grid_size": (8, 8),
                                     "filter_name": "gaussian",
                                     "kernel_size": 3
                                 }

    Return:
        preprocessed_record ... dictionary that contains:
                                - original metadata and raw ROI image
                                - grayscale image
                                - resized image
                                - contrast-normalized image
                                - processed image
                                - normalized preprocessing config used

    Why this function exists:
    It is the preprocessing-stage equivalent of extract_one_roi(...):
    one structured input record comes in, one richer structured record comes
    out. The final key "processed_image" is the image that should be used for
    later LBP descriptor extraction.
    """

    target_size = preprocessing_config.get("target_size", (80, 80))
    contrast_method = preprocessing_config.get("contrast_method", "none")
    clahe_clip_limit = preprocessing_config.get("clahe_clip_limit", 2.0)
    clahe_tile_grid_size = preprocessing_config.get(
        "clahe_tile_grid_size",
        (8, 8),
    )
    filter_name = preprocessing_config.get("filter_name", "none")
    kernel_size = preprocessing_config.get("kernel_size", 5)

    validated_target_size = validate_target_size(target_size)
    normalized_contrast_method = normalize_contrast_method_name(
        contrast_method
    )
    normalized_filter_name = normalize_filter_name(filter_name)

    grayscale_image = convert_roi_to_grayscale(roi_record["roi_image"])

    resized_image = resize_grayscale_roi(
        gray_image=grayscale_image,
        target_size=validated_target_size,
    )

    contrast_normalized_image = apply_contrast_normalization(
        gray_image=resized_image,
        contrast_method=normalized_contrast_method,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=clahe_tile_grid_size,
    )

    processed_image = apply_filter(
        gray_image=contrast_normalized_image,
        filter_name=normalized_filter_name,
        kernel_size=kernel_size,
    )

    preprocessed_record = {
        **roi_record,
        "grayscale_image": grayscale_image,
        "resized_image": resized_image,
        "contrast_normalized_image": contrast_normalized_image,
        "processed_image": processed_image,
        "preprocessing_config": {
            "target_size": validated_target_size,
            "contrast_method": normalized_contrast_method,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_grid_size": validate_clahe_tile_grid_size(
                clahe_tile_grid_size
            ),
            "filter_name": normalized_filter_name,
            "kernel_size": kernel_size,
        },
    }

    return preprocessed_record


def preprocess_all_rois(rois, preprocessing_config):
    """
    Preprocess all ROI records from one source image.

    Inputs:
        rois ................. list of ROI records returned by roi_extraction.py
        preprocessing_config . dictionary describing preprocessing settings

    Return:
        preprocessed_rois .... list of preprocessed ROI records

    Overall idea:
    This function turns:
        one list of raw ROI records
    into:
        one list of preprocessed ROI records

    This mirrors the structure used in roi_extraction.py and keeps the pipeline
    easy to follow:
    - one image
    - many ROIs
    - many preprocessed ROIs
    """

    preprocessed_rois = []

    for roi_record in rois:
        preprocessed_record = preprocess_one_roi(
            roi_record=roi_record,
            preprocessing_config=preprocessing_config,
        )
        preprocessed_rois.append(preprocessed_record)

    return preprocessed_rois