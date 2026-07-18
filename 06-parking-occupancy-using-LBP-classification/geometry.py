"""
geometry.py

Purpose of this module:
- provide geometric helper functions related to parking-space shape
- keep perspective-transform logic separate from data loading

This module currently provides:
- order_points(...)
- four_point_transform(...)

Why this module exists:
The geometry of one parking place is a separate concern from:
- reading files
- saving debug images
- later filtering and classification

This separation keeps the project easier to understand and maintain.
"""
# ---------------------------------------------------------------------------
# Module orientation:
# This module contains the geometric normalization step that converts one
# parking-space quadrilateral from the original camera view into one rectangular
# ROI patch. Later modules assume that every parking-space sample can be treated
# as an ordinary image patch; this module is the reason that assumption holds.
# ---------------------------------------------------------------------------
#
# This file is the mathematical core behind ROI extraction.
# roi_extraction.py is responsible for the record structure of one ROI sample,
# but this module is responsible for the geometric fact that a parking-space
# polygon from the full scene can be turned into a normalized rectangular image.
#
# In the full project flow, this module performs the transformation:
# parking-space quadrilateral in the original perspective view
#   -> consistently ordered corner points
#   -> perspective transform matrix
#   -> rectangular top-down ROI patch
#
# Without this geometric normalization step, later modules would be forced to
# work directly on skewed, perspective-distorted parking-space regions. The
# whole preprocessing and LBP pipeline is simpler and more comparable precisely
# because this module converts each parking space into a clean rectangular patch.

import cv2
import numpy as np


def order_points(pts):
    """
    Order 4 points of a quadrilateral into a consistent order:
    top-left, top-right, bottom-right, bottom-left.

    Input:
        pts ... NumPy array of shape (4, 2)
                each row is one point [x, y]

    Output:
        rect ... NumPy array of shape (4, 2)
                 ordered as:
                 rect[0] = top-left
                 rect[1] = top-right
                 rect[2] = bottom-right
                 rect[3] = bottom-left

    Why this is needed:
    The 4 corner points of one parking space may not be stored in the exact
    order needed by perspective transformation. Before we can warp a parking
    space into a rectangular ROI patch, we must impose a consistent order.

    Logic used:
    - top-left     has the smallest x + y
    - bottom-right has the largest x + y
    - top-right    has the smallest y - x
    - bottom-left  has the largest y - x
    """

    # This helper solves the first geometric problem: even if the four corners of a
    # parking-space quadrilateral are correct, they may arrive in an arbitrary order.
    # A perspective transform, however, needs a stable semantic ordering of corners.
    #
    # So this function takes one unordered 4-point set and converts it into the
    # canonical order used by the rest of the module:
    # - top-left
    # - top-right
    # - bottom-right
    # - bottom-left
    #
    # That ordering convention becomes the bridge between the source quadrilateral
    # and the destination rectangle used later in the perspective transform.

    # Set up the local working state first so the later processing steps can operate on
    # explicit, well-named intermediate values.
    # Start with an empty 4x2 array that will receive the points in canonical order.
    rect = np.zeros((4, 2), dtype="float32")

    # compute x + y for every point
    # In image coordinates, the top-left point tends to have the smallest x+y sum,
    # while the bottom-right point tends to have the largest x+y sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # compute y - x for every point
    # Using y-x helps separate the remaining two corners:
    # - the smallest difference tends to be top-right
    # - the largest difference tends to be bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    # Return the corners in the exact order expected by the perspective-transform
    # logic below.
    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return rect


def four_point_transform(image, one_c):
    """
    Apply a perspective transform to one parking-space quadrilateral.

    Parameters:
        image ... original parking-lot image
        one_c ... one parking-space definition; may be:
                  - flat list of 8 numbers:
                    [x1, y1, x2, y2, x3, y3, x4, y4]
                  - or array of shape (4, 2)

    Returns:
        warped ... rectangular top-down view of that parking place

    Why this function matters:
    Step 2 is not just a simple crop. The parking places in the original image
    are viewed under perspective and therefore appear as skewed quadrilaterals.
    This function converts such a quadrilateral into a cleaner rectangular ROI
    patch that can later be processed more consistently.

    Practical effect:
    - input  -> one skewed parking-space region in the source image
    - output -> one normalized rectangular parking-space patch
    """

    # This is the main geometric normalization routine of the module.
    #
    # It takes one parking-space definition from the parking map and turns it into
    # one rectangular ROI patch cut out of the original full-scene image.
    #
    # The logic is:
    # 1. interpret the input coordinates as four 2D points
    # 2. impose a stable corner ordering
    # 3. estimate the target rectangle size from the quadrilateral edge lengths
    # 4. define the destination rectangle corners
    # 5. compute the perspective transform matrix
    # 6. warp the image region into the destination rectangle
    #
    # The returned warped image is exactly the roi_image that later becomes part of
    # the ROI record in roi_extraction.py.

    # convert input into a NumPy array of shape (4, 2)
    # This accepts either a flat 8-value definition or an already shaped 4x2 array
    # and normalizes both cases into one standard internal representation.
    pts = np.array(one_c, dtype="float32").reshape(4, 2)

    # order the source points consistently
    # Perspective warping only makes sense when the source corners have a stable
    # semantic ordering that matches the destination rectangle ordering.
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute width of the output patch
    # widthA ... bottom edge length
    # widthB ... top edge length
    #
    # The quadrilateral may not be perfectly rectangular in the source image, so the
    # top and bottom edge lengths can differ slightly. Taking the maximum makes the
    # destination rectangle large enough to contain the transformed parking space.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute height of the output patch
    # heightA ... right edge length
    # heightB ... left edge length
    #
    # The same logic is applied vertically: the destination rectangle height is based
    # on the larger of the two side lengths so the warped patch has a valid full
    # extent.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # guard against degenerate cases where width or height becomes zero
    # A zero or negative destination size would mean the parking-space coordinates
    # are geometrically invalid for perspective warping.
    if maxWidth <= 0 or maxHeight <= 0:
        raise ValueError(
            "Perspective transform produced a non-positive ROI size. "
            "Check parking-space coordinates."
        )

    # destination points describing a straight rectangle
    # These are the canonical rectangle corners that the source quadrilateral will be
    # mapped onto. Their order matches the ordered source corner convention:
    # - top-left
    # - top-right
    # - bottom-right
    # - bottom-left
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )

    # compute perspective transform matrix and apply it
    # OpenCV first estimates the homography that maps the source quadrilateral to the
    # destination rectangle, then uses that mapping to resample the corresponding
    # image region into a new rectangular patch.
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the final normalized parking-space ROI patch.
    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return warped