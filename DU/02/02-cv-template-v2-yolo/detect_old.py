import os 
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from classifier import build_color_masks, predict_color

# Load the YOLOv11 model
model = YOLO('tlight-v11.pt')

cv.namedWindow('frame', 0)
cv.namedWindow('crop', 0) 
cv.namedWindow('red mask', 0) 
cv.namedWindow('green mask', 0)

folder = "test-big-zao" 
files = sorted(os.listdir(folder))

out_crops = "out-crops"
os.makedirs(out_crops, exist_ok=True)


for filename in files:
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image_path = folder + "/" + filename
        frame = cv.imread(image_path)
        
        frame_paint = frame.copy()
        first_crop = None 
        first_red_mask = None 
        first_green_mask = None 

        # Run YOLOv11 inference on the frame
        results = model.predict(frame, imgsz=480, conf=0.2)

        # take detections from the image result
        result = results[0]
        boxes = result.boxes
        names = result.names

        print("Image:", filename) 
        print("Number of detections:", len(boxes)) 

        # go through all detected boxes
        for i, box in enumerate(boxes):
            # bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            #make sure the crop does not go outside the image
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = frame[y1:y2, x1:x2]
            crop_path = out_crops + "/" + filename.rsplit(".", 1)[0] + "_crop_" + str(i) + ".png"  #CHANGED
            cv.imwrite(crop_path, crop)

            red_mask, green_mask = build_color_masks(crop) 
            prediction, red_pixels, green_pixels = predict_color(red_mask, green_mask)

            print("Detection", i, "prediction:", prediction.upper()) 
            print(" crop saved to:", crop_path)
            print("--------------------------")

            # draw rectangle
            cv.rectangle(frame_paint, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw only color prediction above rectangle
            color_label = prediction.upper()
            cv.putText(frame_paint, color_label, (x1, y1 - 10),  
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

            if i == 0: 
                first_crop = crop.copy()
                first_red_mask = red_mask.copy() 
                first_green_mask = green_mask.copy() 

        # show annotated image
        cv.imshow("frame", frame_paint)

        if first_crop is not None:
            cv.imshow("crop", first_crop)

        if first_red_mask is not None: 
            cv.imshow("red mask", first_red_mask)

        if first_green_mask is not None:
            cv.imshow("green mask", first_green_mask) 

        cv.waitKey(0)


cv.destroyAllWindows()       
