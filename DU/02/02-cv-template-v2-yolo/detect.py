import os
import cv2 as cv
from ultralytics import YOLO
from classifier import build_color_masks, predict_color

# Load the YOLOv11 model
model = YOLO('tlight-v11.pt')

folder = "test-big-zao"
files = sorted(os.listdir(folder))

out_crops = "out-crops"
os.makedirs(out_crops, exist_ok=True)

out_detect = "out-detect" 
os.makedirs(out_detect, exist_ok=True) 

for filename in files:
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image_path = folder + "/" + filename
        frame = cv.imread(image_path)

        frame_paint = frame.copy()

        # Run YOLOv11 inference on the frame
        results = model.predict(frame, imgsz=480, conf=0.2, verbose=False)

        # take detections from the image result
        result = results[0]

        boxes = result.boxes  #CHANGED

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
            crop_path = out_crops + "/" + filename.rsplit(".", 1)[0] + "_crop_" + str(i) + ".png"
            cv.imwrite(crop_path, crop)

            red_mask, green_mask = build_color_masks(crop)
            prediction, red_pixels, green_pixels = predict_color(red_mask, green_mask)

            # draw rectangle
            cv.rectangle(frame_paint, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw only color prediction above rectangle
            color_label = prediction.upper()
            cv.putText(frame_paint, color_label, (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        detect_path = out_detect + "/" + filename 
        cv.imwrite(detect_path, frame_paint)  