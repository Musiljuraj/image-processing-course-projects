import cv2
import numpy as np

"""
Frame-preparation module.

This module performs the small but important conversion step between raw frame acquisition
and template matching.

In the overall runtime flow, the frame arrives from capture.py as a normal BGR OpenCV image.
Before matcher.py compares that frame against the loaded templates, the frame has to be put
into the same representation that the templates were loaded in. That is the purpose of this file.

Project flow around this module:

capture.py
    -> downloads and decodes the current ROI frame as a BGR image
preprocessing.py
    -> optionally converts that frame to grayscale
    -> otherwise returns a safe copy of the original frame
templates.py
    -> loads template images using the same grayscale-vs-color decision
matcher.py
    -> compares prepared frame and loaded templates in matching-compatible format

So preprocessing.py is the live-frame normalization layer of the project.
It does not decide anything about matching thresholds, click logic, prediction, or bridge
communication. Its only job is to make sure the current frame is in the correct visual format
for the matching stage.
"""


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert one BGR OpenCV frame into a grayscale image.

    This is the direct image-conversion helper used by prepare_frame() when grayscale mode
    is enabled. The conversion reduces the frame from a 3-channel color image to a single-channel
    intensity image, which must stay consistent with how templates were loaded in templates.py.
    """
    # Use OpenCV's standard BGR-to-grayscale conversion so the live frame matches the project's
    # grayscale template-loading mode when use_gray is enabled.
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def prepare_frame(frame_bgr: np.ndarray, use_gray: bool = True) -> np.ndarray:
    """
    Prepare the live frame for matching.

    The logic here is intentionally minimal and controlled by one configuration choice:
    whether matching should run in grayscale mode or color mode.

    Behavior:
    - if use_gray is True:
      convert the incoming BGR frame to grayscale
    - if use_gray is False:
      return a copy of the original BGR frame

    Returning a copy in the color path keeps the function behavior explicit and safe:
    the caller receives a dedicated working frame for matching rather than another reference
    to the original input array.
    """
    # In grayscale mode, convert the live BGR frame into the same single-channel format that
    # grayscale templates were loaded in, so matcher.py compares like with like.
    if use_gray:
        return to_gray(frame_bgr)

    # In color mode, keep the original visual information but return a copy rather than the
    # original array reference. This makes the prepared frame a separate working image.
    return frame_bgr.copy()