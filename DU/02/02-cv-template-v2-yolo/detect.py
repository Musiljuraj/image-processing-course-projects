# TODO pip install ultralytics

import cv2 as cv
from ultralytics import YOLO
import numpy as np

# Load the YOLOv11 model
model = YOLO('tlight-v11.pt')

cv.namedWindow('frame', 0)

frame = cv.imread('test-big-zao/0a8eedc2-00000000.jpg')
frame_paint = frame.copy()

# Run YOLOv11 inference on the frame
results = model.predict(frame, imgsz=480, conf=0.2)
print(results)
        
cv.imshow('frame', frame)        
cv.waitKey()        
