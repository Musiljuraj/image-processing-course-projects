"""
preprocessing.py

Purpose of this module:
- perform preprocessing on ROI patches before edge detection
- keep preprocessing logic separate from data loading, geometry, ROI extraction,
  and debug-output generation

Why this module exists:
After Step 2, the project already knows how to:
- load the parking-space map
- load test images
- extract each parking space as its own ROI patch

The next logical stage is to prepare those ROI patches for later edge detection.
In practice, that means:
1. convert each ROI patch to grayscale
2. optionally apply a smoothing filter

This module collects that logic in one dedicated place.

Current responsibilities:
- convert one ROI image to grayscale
- apply one selected filter to one grayscale ROI image
- preprocess one ROI record
- preprocess all ROI records from one source image

"""

import cv2


def normalize_filter_name(filter_name):
    """
    Normalize the textual name of the selected filter.

    Input:
        filter_name ... string such as:
                        "none", "box", "gaussian", "median"

    Return:
        normalized_filter_name ... lowercase stripped string
    """

    if not isinstance(filter_name, str):
        raise TypeError("filter_name must be a string.")

    return filter_name.strip().lower()


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

    Note:
    The "none" filter does not actually use the kernel size, but we still keep
    the validation logic available for filters that do.
    """

    if not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer.")

    if kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer.")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    return kernel_size


def convert_roi_to_grayscale(roi_image):
    """
    Convert one ROI patch to grayscale.

    Input:
        roi_image ... ROI image array
                      usually a BGR image loaded/produced by OpenCV

    Return:
        grayscale_image ... single-channel grayscale image

    Why this function exists:
    Edge detectors operate on intensity structure, not on full color
    information. Grayscale conversion therefore simplifies the image while
    preserving the brightness transitions that will later matter for edge
    detection.

    Additional behavior:
    - if the input image is already single-channel, a copy is returned
    - unsupported image shapes raise a clear error
    """

    # if the ROI image is already grayscale (single-channel),
    # return a copy so later code can safely modify it if needed
    if roi_image.ndim == 2:
        return roi_image.copy()

    # if the ROI image is a normal 3-channel BGR image,
    # convert it to grayscale
    if roi_image.ndim == 3 and roi_image.shape[2] == 3:
        grayscale_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        return grayscale_image

    raise ValueError(
        "ROI image has unsupported shape for grayscale conversion: "
        f"{roi_image.shape}"
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
    This is the place where preprocessing becomes configurable.
    Later experiments can compare the effect of different filters and kernel
    sizes without changing the rest of the project structure.
    """

    normalized_filter_name = normalize_filter_name(filter_name)

    # "none" means: do not smooth the image,
    # but return a copy so the caller always receives a standalone array
    if normalized_filter_name == "none":
        return gray_image.copy()

    validated_kernel_size = validate_kernel_size(kernel_size)

    if normalized_filter_name == "box":
        # simple averaging filter
        processed_image = cv2.blur(
            gray_image,
            (validated_kernel_size, validated_kernel_size),
        )
        return processed_image

    if normalized_filter_name == "gaussian":
        # Gaussian smoothing with automatically derived sigma
        processed_image = cv2.GaussianBlur(
            gray_image,
            (validated_kernel_size, validated_kernel_size),
            0,
        )
        return processed_image

    if normalized_filter_name == "median":
        # median filter using one odd scalar kernel size
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
                                     "filter_name": "gaussian",
                                     "kernel_size": 5
                                 }

    Return:
        preprocessed_record ... dictionary that contains:
                                - original metadata and raw ROI image
                                - grayscale image
                                - processed image
                                - normalized preprocessing config used

    Why this function exists:
    It is the preprocessing-stage equivalent of extract_one_roi(...):
    one structured input record comes in, one richer structured record comes out.
    """

    # read configuration with sensible defaults
    filter_name = preprocessing_config.get("filter_name", "none")
    kernel_size = preprocessing_config.get("kernel_size", 5)

    # normalize filter name once
    normalized_filter_name = normalize_filter_name(filter_name)

    # convert the raw ROI to grayscale
    grayscale_image = convert_roi_to_grayscale(roi_record["roi_image"])

    # apply the selected filter to the grayscale image
    processed_image = apply_filter(
        gray_image=grayscale_image,
        filter_name=normalized_filter_name,
        kernel_size=kernel_size,
    )

    # keep the original ROI record information and extend it
    # with preprocessing outputs and the actual config used
    preprocessed_record = {
        **roi_record,
        "grayscale_image": grayscale_image,
        "processed_image": processed_image,
        "preprocessing_config": {
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