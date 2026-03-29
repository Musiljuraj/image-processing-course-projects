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

    rect = np.zeros((4, 2), dtype="float32")

    # compute x + y for every point
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # compute y - x for every point
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

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

    # convert input into a NumPy array of shape (4, 2)
    pts = np.array(one_c, dtype="float32").reshape(4, 2)

    # order the source points consistently
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute width of the output patch
    # widthA ... bottom edge length
    # widthB ... top edge length
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute height of the output patch
    # heightA ... right edge length
    # heightB ... left edge length
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # guard against degenerate cases where width or height becomes zero
    if maxWidth <= 0 or maxHeight <= 0:
        raise ValueError(
            "Perspective transform produced a non-positive ROI size. "
            "Check parking-space coordinates."
        )

    # destination points describing a straight rectangle
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
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped