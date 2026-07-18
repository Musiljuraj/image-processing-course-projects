"""
edge_detection.py

Purpose of this module:
- perform edge detection on preprocessed ROI patches
- keep detector-specific logic separate from data loading, geometry,
  ROI extraction, preprocessing, debug-output generation, and evaluation

Why this module exists:
After preprocessing, each parking-space ROI has already been:
- extracted from the full parking-lot image
- converted to grayscale
- optionally smoothed by a selected filter

The next logical stage is to run an edge detector on that processed ROI image.

This module collects that logic in one dedicated place.

Current responsibilities:
- normalize and validate the selected edge-detector name
- validate detector-specific configuration parameters
- run Sobel-based edge detection
- run Canny-based edge detection
- count nonzero edge pixels in the resulting edge image
- compute total ROI pixel count
- compute normalized edge ratio
- process one preprocessed ROI record
- process all preprocessed ROI records from one source image

Important design idea:
This module does not only return edge images. It returns structured records
that still carry metadata such as:
- source image name
- parking-space index
- original polygon
- raw ROI image
- grayscale image
- processed image
- edge image
- edge pixel count
- ROI pixel count
- edge ratio
- edge-detection configuration used

That is useful because later stages (classification and evaluation) will still
need to know where each edge image came from and how it was produced.
"""

import cv2


def normalize_detector_name(detector_name):
    """
    Normalize the textual name of the selected edge detector.

    Input:
        detector_name ... string such as:
                          "sobel", "canny"

    Return:
        normalized_detector_name ... lowercase stripped string

    Why this helper exists:
    It is useful to normalize user- or config-provided detector names once,
    so the rest of the code can work with one consistent representation.
    """

    if not isinstance(detector_name, str):
        raise TypeError("detector_name must be a string.")

    return detector_name.strip().lower()


def validate_sobel_kernel_size(ksize):
    """
    Validate the Sobel kernel size.

    Input:
        ksize ... expected to be a positive odd integer

    Return:
        ksize ... same value if valid

    Why this helper exists:
    Sobel uses a derivative kernel whose size must be odd and positive.
    Using one validation function keeps error handling in one place.
    """

    if not isinstance(ksize, int):
        raise TypeError("Sobel kernel size must be an integer.")

    if ksize <= 0:
        raise ValueError("Sobel kernel size must be positive.")

    if ksize % 2 == 0:
        raise ValueError("Sobel kernel size must be odd.")

    return ksize


def validate_nonnegative_number(value, name):
    """
    Validate that the given value is a non-negative number.

    Inputs:
        value ... numeric value to validate
        name .... textual name used in error messages

    Return:
        value ... same numeric value if valid
    """

    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float.")

    if value < 0:
        raise ValueError(f"{name} must be non-negative.")

    return value


def validate_canny_aperture_size(aperture_size):
    """
    Validate the aperture size for Canny.

    Input:
        aperture_size ... expected to be one of 3, 5, 7

    Return:
        aperture_size ... same value if valid

    Why this helper exists:
    OpenCV Canny uses Sobel internally, and apertureSize must be one of the
    supported odd values. This helper keeps that validation explicit.
    """

    if not isinstance(aperture_size, int):
        raise TypeError("Canny aperture_size must be an integer.")

    if aperture_size not in (3, 5, 7):
        raise ValueError("Canny aperture_size must be one of: 3, 5, 7.")

    return aperture_size


def validate_processed_image(processed_image):
    """
    Validate that the processed ROI image is a single-channel grayscale image.

    Input:
        processed_image ... image produced by preprocessing.py

    Return:
        processed_image ... same image if valid

    Why this helper exists:
    Both Sobel and Canny in this project are expected to operate on a
    single-channel grayscale / filtered ROI image.
    """

    if processed_image is None:
        raise ValueError("processed_image must not be None.")

    if processed_image.ndim != 2:
        raise ValueError(
            "processed_image must be a single-channel image. "
            f"Got shape: {processed_image.shape}"
        )

    return processed_image


def validate_sobel_config(sobel_config):
    """
    Validate and normalize the Sobel configuration dictionary.

    Input:
        sobel_config ... dictionary, for example:
                         {
                             "ksize": 3,
                             "threshold": 100
                         }

    Return:
        normalized_sobel_config ... normalized validated dictionary
    """

    if sobel_config is None:
        sobel_config = {}

    ksize = validate_sobel_kernel_size(sobel_config.get("ksize", 3))
    threshold = validate_nonnegative_number(
        sobel_config.get("threshold", 100),
        "Sobel threshold",
    )

    normalized_sobel_config = {
        "ksize": ksize,
        "threshold": threshold,
    }

    return normalized_sobel_config


def validate_canny_config(canny_config):
    """
    Validate and normalize the Canny configuration dictionary.

    Input:
        canny_config ... dictionary, for example:
                         {
                             "threshold1": 50,
                             "threshold2": 150,
                             "aperture_size": 3,
                             "l2gradient": False
                         }

    Return:
        normalized_canny_config ... normalized validated dictionary
    """

    if canny_config is None:
        canny_config = {}

    threshold1 = validate_nonnegative_number(
        canny_config.get("threshold1", 50),
        "Canny threshold1",
    )
    threshold2 = validate_nonnegative_number(
        canny_config.get("threshold2", 150),
        "Canny threshold2",
    )

    if threshold1 > threshold2:
        raise ValueError("Canny threshold1 should be less than or equal to threshold2.")

    aperture_size = validate_canny_aperture_size(
        canny_config.get("aperture_size", 3)
    )

    l2gradient = bool(canny_config.get("l2gradient", False))

    normalized_canny_config = {
        "threshold1": threshold1,
        "threshold2": threshold2,
        "aperture_size": aperture_size,
        "l2gradient": l2gradient,
    }

    return normalized_canny_config


def detect_edges_sobel(processed_image, sobel_config=None):
    """
    Run Sobel-based edge detection on one processed ROI image.

    Inputs:
        processed_image ... single-channel processed ROI image
        sobel_config ..... dictionary with Sobel parameters

    Return:
        edge_image ....... binary Sobel edge map
        used_config ...... validated normalized Sobel config

    Processing logic:
    1. compute Sobel derivative in x direction
    2. compute Sobel derivative in y direction
    3. convert both to absolute 8-bit images
    4. combine x and y responses
    5. threshold the combined gradient image into a binary edge map

    Why thresholding is needed here:
    Sobel produces gradient-strength information, not a final binary edge map.
    Since the later parking-occupancy logic will use nonzero edge-pixel counts,
    we convert the gradient image into a clean binary representation.
    """

    processed_image = validate_processed_image(processed_image)
    used_config = validate_sobel_config(sobel_config)

    # compute derivatives in horizontal and vertical directions
    sobel_x = cv2.Sobel(processed_image, cv2.CV_64F, 1, 0, ksize=used_config["ksize"])
    sobel_y = cv2.Sobel(processed_image, cv2.CV_64F, 0, 1, ksize=used_config["ksize"])

    # convert gradient images to absolute 8-bit representation
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    # combine the two directional responses into one gradient image
    combined_gradient = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    # threshold the combined gradient image to obtain a binary edge map
    _, edge_image = cv2.threshold(
        combined_gradient,
        used_config["threshold"],
        255,
        cv2.THRESH_BINARY,
    )

    return edge_image, used_config


def detect_edges_canny(processed_image, canny_config=None):
    """
    Run Canny edge detection on one processed ROI image.

    Inputs:
        processed_image ... single-channel processed ROI image
        canny_config ..... dictionary with Canny parameters

    Return:
        edge_image ....... binary Canny edge map
        used_config ...... validated normalized Canny config

    Processing logic:
    1. use the already preprocessed grayscale ROI image
    2. apply OpenCV Canny with the selected thresholds and aperture
    3. return the resulting binary edge image
    """

    processed_image = validate_processed_image(processed_image)
    used_config = validate_canny_config(canny_config)

    edge_image = cv2.Canny(
        processed_image,
        threshold1=used_config["threshold1"],
        threshold2=used_config["threshold2"],
        apertureSize=used_config["aperture_size"],
        L2gradient=used_config["l2gradient"],
    )

    return edge_image, used_config


def compute_edge_statistics(edge_image):
    """
    Compute edge statistics from one binary edge image.

    Input:
        edge_image ... single-channel binary or binary-like edge map

    Return:
        edge_statistics ... dictionary containing:
                            - edge_count
                            - roi_pixel_count
                            - edge_ratio

    Definitions:
        edge_count ...... number of nonzero pixels in the edge image
        roi_pixel_count . total number of pixels in the ROI patch
        edge_ratio ...... edge_count / roi_pixel_count

    Why this helper exists:
    The project now uses a normalized feature for classification.
    A raw edge count is still useful for debugging, but the ratio is more
    comparable across parking spaces with different ROI sizes.
    """

    if edge_image is None:
        raise ValueError("edge_image must not be None.")

    if edge_image.ndim != 2:
        raise ValueError(
            "edge_image must be a single-channel image. "
            f"Got shape: {edge_image.shape}"
        )

    edge_count = int(cv2.countNonZero(edge_image))
    roi_pixel_count = int(edge_image.shape[0] * edge_image.shape[1])

    if roi_pixel_count <= 0:
        raise ValueError("roi_pixel_count must be positive.")

    edge_ratio = float(edge_count) / float(roi_pixel_count)

    edge_statistics = {
        "edge_count": edge_count,
        "roi_pixel_count": roi_pixel_count,
        "edge_ratio": edge_ratio,
    }

    return edge_statistics


def detect_edges_one_record(preprocessed_record, edge_detection_config):
    """
    Run edge detection for one preprocessed ROI record.

    Inputs:
        preprocessed_record ... dictionary produced by preprocessing.py
                                expected to contain:
                                - source_image_name
                                - space_index
                                - polygon
                                - roi_image
                                - grayscale_image
                                - processed_image

        edge_detection_config ... dictionary describing detector selection
                                  and detector-specific parameters, e.g.:
                                  {
                                      "detector_name": "canny",
                                      "canny": {
                                          "threshold1": 50,
                                          "threshold2": 150,
                                          "aperture_size": 3,
                                          "l2gradient": False
                                      },
                                      "sobel": {
                                          "ksize": 3,
                                          "threshold": 100
                                      }
                                  }

    Return:
        edge_record ............ dictionary containing:
                                 - all existing data from the input record
                                 - edge_image
                                 - edge_count
                                 - roi_pixel_count
                                 - edge_ratio
                                 - edge_detection_config

    Why this function exists:
    It is the edge-detection-stage equivalent of preprocess_one_roi(...):
    one structured input record comes in, one richer structured record comes out.
    """

    detector_name = normalize_detector_name(
        edge_detection_config.get("detector_name", "canny")
    )

    if detector_name == "sobel":
        edge_image, used_detector_config = detect_edges_sobel(
            processed_image=preprocessed_record["processed_image"],
            sobel_config=edge_detection_config.get("sobel", {}),
        )

    elif detector_name == "canny":
        edge_image, used_detector_config = detect_edges_canny(
            processed_image=preprocessed_record["processed_image"],
            canny_config=edge_detection_config.get("canny", {}),
        )

    else:
        raise ValueError(
            "Unsupported detector_name. Expected one of: "
            "'sobel', 'canny'. "
            f"Got: {edge_detection_config.get('detector_name')}"
        )

    edge_statistics = compute_edge_statistics(edge_image)

    edge_record = {
        **preprocessed_record,
        "edge_image": edge_image,
        "edge_count": edge_statistics["edge_count"],
        "roi_pixel_count": edge_statistics["roi_pixel_count"],
        "edge_ratio": edge_statistics["edge_ratio"],
        "edge_detection_config": {
            "detector_name": detector_name,
            detector_name: used_detector_config,
        },
    }

    return edge_record


def detect_edges_all_records(preprocessed_records, edge_detection_config):
    """
    Run edge detection for all preprocessed ROI records from one source image.

    Inputs:
        preprocessed_records ... list of records returned by preprocessing.py
        edge_detection_config . dictionary describing edge-detector selection
                                and parameters

    Return:
        edge_records .......... list of edge-detection records

    Overall idea:
    This function turns:
        one list of preprocessed ROI records
    into:
        one list of edge-detection records

    This mirrors the structure used in roi_extraction.py and preprocessing.py
    and keeps the pipeline easy to follow:
    - one image
    - many ROIs
    - many preprocessed ROIs
    - many edge-detection records
    """

    edge_records = []

    for preprocessed_record in preprocessed_records:
        edge_record = detect_edges_one_record(
            preprocessed_record=preprocessed_record,
            edge_detection_config=edge_detection_config,
        )
        edge_records.append(edge_record)

    return edge_records