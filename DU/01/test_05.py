import cv2 as cv
import numpy as np

WINDOW = "Kamera"
DEFAULT_DEVICE = "http://127.0.0.1:8080/video"

cap = cv.VideoCapture(DEFAULT_DEVICE)

if not cap.isOpened():
    print("Error")
    exit()
else:
    print("Success")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("Kamera - original", frame)
    cv.imshow("Kamera - gray", gray_frame)

    if (cv.waitKey(1) & 0xFF) == ord("q"):
        break 

cap.release()
cv.destroyAllWindows()