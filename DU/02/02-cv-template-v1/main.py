import cv2 as cv
import numpy as np

#### 1 
image = cv.imread("bluel.png")

image = cv.resize(image, (300, 600))

b, g, r = cv.split(image)

# Display each channel
cv.imshow("Blue Channel", b)
cv.imshow("Green Channel", g)
cv.imshow("Red Channel", r)
cv.imshow("image", image)
cv.waitKey()
