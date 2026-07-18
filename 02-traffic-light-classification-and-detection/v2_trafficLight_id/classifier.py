# This module is the color-classification layer of the whole traffic-light
# pipeline.
#
# The module sits after localization performed by detect.py.
# detect.py is responsible for finding candidate traffic-light regions,
# while this file is responsible for deciding which signal color dominates
# inside one already extracted crop.
#
# In the runtime flow, this module performs:
#
#     detected traffic-light crop
#         -> mild blur for noise suppression
#         -> conversion from BGR to HSV color space
#         -> construction of red and green binary masks
#         -> small morphological cleanup of both masks
#         -> counting mask pixels
#         -> final red-vs-green decision
#
# A key design idea in this module is simplicity:
# - build_color_masks(...) creates cleaned binary evidence maps for both colors,
# - predict_color(...) turns those evidence maps into one final label.
#
# That keeps the pipeline easy to understand:
# detect.py answers "where is the traffic light?"
# classifier.py answers "is it red or green?"

"""
classifier.py

This module contains the complete color-decision logic for one detected
traffic-light crop.

Its responsibilities are:
- slightly smoothing the crop to reduce small image noise,
- converting the crop from BGR to HSV representation,
- building binary masks for red and green candidate pixels,
- cleaning those masks with simple morphology,
- comparing the amount of red and green evidence,
- returning the final predicted traffic-light color.

The module intentionally works only on already extracted image crops.
It does not try to localize traffic lights in the full image. That task belongs
to detect.py, which calls the functions from this module after YOLO has already
provided one bounding box.
"""

# OpenCV provides all low-level image-processing operations used here:
# - Gaussian blur,
# - color-space conversion,
# - range thresholding in HSV,
# - binary mask combination,
# - morphology,
# - nonzero-pixel counting.
import cv2 as cv

# NumPy is available mainly because binary-mask and morphology workflows often
# use NumPy-compatible image arrays. In the current version, the active kernel
# is created through OpenCV rather than np.ones(...), but NumPy remains part of
# the module imports.
import numpy as np


def build_color_masks(img):
    """
    Build cleaned binary masks for red and green color evidence inside one
    detected traffic-light crop.

    High-level flow:
    - slightly blur the crop,
    - convert it to HSV,
    - threshold red and green ranges,
    - clean both masks with small morphology,
    - return both binary masks.

    The returned masks are later consumed by predict_color(...), which compares
    how much red and green evidence is present.
    """

    # Apply a small Gaussian blur before any color thresholding.
    #
    # The goal is not strong smoothing, but only mild suppression of isolated
    # pixel noise and tiny local intensity fluctuations that could otherwise
    # create unstable mask fragments.
    # small blur to reduce noise
    blur = cv.GaussianBlur(img, (5, 5), 0)

    # Convert the blurred crop from OpenCV's default BGR representation into
    # HSV.
    #
    # HSV is used because color segmentation is easier and more intuitive there:
    # hue represents the actual color family,
    # saturation filters weak/grayish pixels,
    # value captures brightness.
    # convert to HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    # Build the binary mask for red pixels.
    #
    # Red is special in HSV because its hue range wraps around the circular hue
    # axis. That is why red is represented by two separate intervals:
    # - one near the low end of the hue scale,
    # - one near the high end of the hue scale.
    #
    # Both thresholded masks are later combined into one final red mask.
    # red mask (2 ranges because red wraps around HSV)
    red1 = cv.inRange(hsv, (0, 70, 70), (4, 255, 255))
    red2 = cv.inRange(hsv, (160, 70, 70), (179, 255, 255))
    red_mask = cv.bitwise_or(red1, red2)

    # Build the binary mask for green pixels.
    #
    # Unlike red, green does not wrap around the hue boundary in the same way
    # here, so one continuous HSV interval is enough.
    # green mask
    green_mask = cv.inRange(hsv, (35, 50, 50), (95, 255, 255))

    # Clean both masks with small morphological operations.
    #
    # The kernel is intentionally small and elliptical so the cleanup remains
    # gentle and better matches small rounded traffic-light signal regions.
    #
    # The sequence used here is:
    # - MORPH_OPEN  ... removes tiny isolated noise blobs,
    # - MORPH_CLOSE ... fills tiny holes and reconnects small gaps.
    #
    # This is done separately for the red mask and the green mask so both color
    # evidence maps become cleaner before pixel counting.
    # simple morphology to remove tiny noise
    #kernel = np.ones((3, 3), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
    green_mask = cv.morphologyEx(green_mask, cv.MORPH_OPEN, kernel)
    green_mask = cv.morphologyEx(green_mask, cv.MORPH_CLOSE, kernel)

    # Return both cleaned binary masks so the decision stage can compare them.
    return red_mask, green_mask 

def predict_color(red_mask, green_mask): 
    """
    Predict the final traffic-light color from already prepared red and green
    binary masks.

    The decision logic is intentionally simple:
    - count how many pixels belong to the red mask,
    - count how many pixels belong to the green mask,
    - whichever mask has more supporting pixels wins.

    The function returns:
    - the final textual prediction,
    - the red pixel count,
    - the green pixel count.

    Returning the raw counts keeps the function useful not only for final
    prediction, but also for debugging or later report extensions.
    """

    # Count the number of nonzero pixels in each binary mask.
    #
    # These counts represent the total amount of red and green evidence found in
    # the crop after thresholding and morphological cleanup.
    red_pixels = cv.countNonZero(red_mask)
    green_pixels = cv.countNonZero(green_mask) 

    # Compare the two evidence totals and choose the final label.
    #
    # The current rule is deliberately minimal:
    # - if red evidence is stronger, predict "red",
    # - otherwise predict "green".
    #
    # This means equal counts are resolved in favor of green in the current
    # implementation.
    if red_pixels > green_pixels:
        prediction = "red"
    else:
        prediction = "green"

    # Return both the final decision and the underlying pixel counts.
    return prediction, red_pixels, green_pixels