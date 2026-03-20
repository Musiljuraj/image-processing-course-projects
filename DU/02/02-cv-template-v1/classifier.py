import cv2 as cv
import numpy as np


def build_color_masks(img):
    # small blur to reduce noise
    blur = cv.GaussianBlur(img, (5, 5), 0)

    # convert to HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    # red mask (2 ranges because red wraps around HSV)
    red1 = cv.inRange(hsv, (0, 70, 70), (4, 255, 255))
    red2 = cv.inRange(hsv, (160, 70, 70), (179, 255, 255))
    red_mask = cv.bitwise_or(red1, red2)

    # green mask
    green_mask = cv.inRange(hsv, (35, 50, 50), (95, 255, 255))

    # simple morphology to remove tiny noise
    #kernel = np.ones((3, 3), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
    green_mask = cv.morphologyEx(green_mask, cv.MORPH_OPEN, kernel)
    green_mask = cv.morphologyEx(green_mask, cv.MORPH_CLOSE, kernel)

    # masked outputs
    red_only = cv.bitwise_and(img, img, mask=red_mask) #used during developing, keeping it here for potential manual inspection 
    green_only = cv.bitwise_and(img, img, mask=green_mask) #used during developing, keeping it here for potential manual inspection 

    return red_mask, green_mask 

def predict_color(red_mask, green_mask): 
    red_pixels = cv.countNonZero(red_mask)
    green_pixels = cv.countNonZero(green_mask) 

    if red_pixels > green_pixels:
        prediction = "red"
    else:
        prediction = "green"

    return prediction, red_pixels, green_pixels