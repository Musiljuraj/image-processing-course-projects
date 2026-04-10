# This module is the color-evidence extraction and decision layer of the project.
# It sits below main.py and provides the actual logic that turns one loaded image
# into one simple color prediction.
#
# In the overall runtime flow, this module performs:
#
#     input BGR image
#         -> light blur for noise suppression
#         -> conversion into HSV color space
#         -> construction of red and green binary masks
#         -> light morphological cleanup of both masks
#         -> counting of red-like and green-like pixels
#         -> final red-vs-green comparison
#         -> predicted class returned to main.py
#
# A key design idea here is to separate:
# - color-region extraction
# - final class decision
#
# That is why this file exposes two connected functions:
# - build_color_masks(...) prepares the binary evidence maps
# - predict_color(...) turns those masks into the final label
#
# main.py then uses those outputs for batch evaluation, confusion-matrix
# bookkeeping, and saving images into output folders.

import cv2 as cv
import numpy as np


def build_color_masks(img):
    # This function extracts the two main evidence masks used in the whole
    # project:
    # - red_mask ..... pixels that look red in HSV space
    # - green_mask ... pixels that look green in HSV space
    #
    # The goal is not to classify the image directly here, but to create two
    # cleaned binary maps that describe where red-like and green-like regions
    # appear in the image.

    # small blur to reduce noise
    #
    # A light Gaussian blur is applied first so that isolated pixel-level noise
    # and tiny local fluctuations do not create unstable color-mask fragments.
    # This makes the later HSV thresholding a little more robust.
    blur = cv.GaussianBlur(img, (5, 5), 0)

    # convert to HSV
    #
    # The image is converted from OpenCV's default BGR representation into HSV.
    # HSV is more suitable here because hue separates the actual color identity
    # from brightness and saturation much more naturally than raw BGR values.
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    # red mask (2 ranges because red wraps around HSV)
    #
    # Red is special in HSV because it lies near both ends of the hue scale.
    # That means one continuous red interval in perception becomes two numeric
    # intervals in HSV representation:
    # - one near hue 0
    # - another near hue 179
    #
    # Both ranges are therefore thresholded separately and then combined into
    # one final red mask.
    red1 = cv.inRange(hsv, (0, 70, 70), (4, 255, 255))
    red2 = cv.inRange(hsv, (160, 70, 70), (179, 255, 255))
    red_mask = cv.bitwise_or(red1, red2)

    # green mask
    #
    # Green occupies one more compact region in HSV, so one threshold interval
    # is sufficient here. The result is a binary mask of green-like pixels.
    green_mask = cv.inRange(hsv, (35, 50, 50), (95, 255, 255))

    # simple morphology to remove tiny noise
    #
    # After thresholding, both masks can still contain small isolated blobs,
    # holes, or rough edges caused by local image noise or thresholding
    # artifacts.
    #
    # A small elliptical kernel is used for basic morphology:
    # - opening removes tiny foreground noise
    # - closing fills small gaps inside otherwise valid color regions
    #
    # The same cleanup is applied to both red and green masks so that the final
    # comparison is based on more stable connected color evidence.
    #kernel = np.ones((3, 3), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
    green_mask = cv.morphologyEx(green_mask, cv.MORPH_OPEN, kernel)
    green_mask = cv.morphologyEx(green_mask, cv.MORPH_CLOSE, kernel)

    # masked outputs
    #
    # These masked color-only images were useful during development for manual
    # visual inspection of what each mask actually selected from the original
    # image.
    #
    # They are not used by the final batch pipeline in main.py, but they remain
    # here as convenient optional debugging artifacts that can still help with
    # manual inspection or future threshold tuning.
    red_only = cv.bitwise_and(img, img, mask=red_mask) #used during developing, keeping it here for potential manual inspection 
    green_only = cv.bitwise_and(img, img, mask=green_mask) #used during developing, keeping it here for potential manual inspection 

    # Return the two cleaned binary evidence masks to the caller.
    # main.py will pass them into predict_color(...) to obtain the final class.
    return red_mask, green_mask 

def predict_color(red_mask, green_mask): 
    # This function is the final decision layer of the module.
    # It does not look at the original image anymore. Instead, it receives the
    # already prepared red and green masks and decides which color has stronger
    # support.

    # Count how many non-zero pixels remain in each cleaned mask.
    # These counts act as the final numeric evidence values:
    # - more red pixels   -> stronger red evidence
    # - more green pixels -> stronger green evidence
    red_pixels = cv.countNonZero(red_mask)
    green_pixels = cv.countNonZero(green_mask) 

    # The decision rule is intentionally simple and fully transparent:
    # - if red evidence is stronger than green evidence, predict "red"
    # - otherwise predict "green"
    #
    # In the tie case, the code also falls into the "green" branch because only
    # a strict red advantage is treated as sufficient for a red prediction.
    if red_pixels > green_pixels:
        prediction = "red"
    else:
        prediction = "green"

    # Return:
    # - the final predicted label
    # - the raw red-pixel count
    # - the raw green-pixel count
    #
    # The counts are not needed for the decision itself anymore, but they are
    # useful for debugging, reporting, and understanding why a particular image
    # was classified the way it was.
    return prediction, red_pixels, green_pixels