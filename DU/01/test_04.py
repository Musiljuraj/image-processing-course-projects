import cv2 as cv
import numpy as np

img = cv.imread("img.png", cv.IMREAD_COLOR)

#cv.imshow("img", img)
#cv.waitKey(0)
#cv.destroyAllWindows()

if img is None:
    print("Error.")
else:
    cv.imshow("img", img)
    cv.waitKey(0)
    
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray_img)
    cv.waitKey(0)

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow('HSV', hsv_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

