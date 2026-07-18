# This module is the top-level runtime script of the whole traffic-light
# detection pipeline.
#
# It does not implement the low-level color logic itself. Instead, it acts as
# the orchestration layer that connects:
# - image loading from one input folder,
# - YOLO-based traffic-light localization,
# - crop extraction for each detected traffic light,
# - color-mask construction from classifier.py,
# - simple red-vs-green decision making from classifier.py,
# - output-image generation with drawn predictions.
#
# In the runtime flow, this module performs:
#
#     input image
#         -> YOLO detection
#         -> one crop per detected traffic light
#         -> red/green mask generation on the crop
#         -> pixel-count comparison
#         -> final color label
#         -> annotated output image
#
# A useful way to think about this file is:
# - detect.py decides where possible traffic lights are,
# - classifier.py decides which color is dominant inside each detected crop.
#
# That separation keeps the full pipeline simple:
# localization is handled here,
# while color classification is delegated to the helper module.

"""
detect.py

This module contains the complete top-level runtime loop for the traffic-light
project.

Its responsibilities are:
- loading the trained YOLO model,
- scanning one input folder for images,
- running detection on each image,
- extracting one crop for every detected bounding box,
- passing each crop to the color-classification helper functions,
- drawing the final prediction back onto the original image,
- saving both raw crops and annotated final outputs.

The module is intentionally written as a direct script rather than as a set of
helper functions. That makes the execution order very explicit:
load model -> iterate through files -> detect -> crop -> classify -> save.
"""

# os is used for simple file-system work in this script:
# - reading file names from the input folder,
# - creating output folders if they do not exist yet.
import os

# OpenCV is used here for all image I/O and visualization work:
# - reading images from disk,
# - copying frames before drawing,
# - saving crops,
# - drawing rectangles and labels,
# - writing the final annotated outputs.
import cv2 as cv

# Ultralytics provides the YOLO detector used as the localization stage of the
# pipeline. Its role in this file is to return bounding boxes for candidate
# traffic lights inside each image.
from ultralytics import YOLO

# The helper functions imported from classifier.py perform the actual
# color-classification stage on one detected crop:
# - build_color_masks(...) prepares red and green binary masks,
# - predict_color(...) compares red and green evidence and returns the label.
from classifier import build_color_masks, predict_color

# Load the trained YOLO model once at startup so the same detector instance can
# be reused for all images in the folder.
#
# This is the localization model of the pipeline. Every later crop and every
# later color decision depends on the bounding boxes produced here.
model = YOLO('tlight-v11.pt')

# This is the input folder containing the images that will be processed by the
# runtime loop.
folder = "test-big-zao"

# The file list is sorted so the script processes the images in a stable and
# predictable order. That makes debugging and output inspection easier because
# repeated runs follow the same sequence.
files = sorted(os.listdir(folder))

# This output folder stores the raw image crops extracted from detected traffic
# lights. Keeping these crops is useful for later inspection of what exactly the
# classifier saw as input.
out_crops = "out-crops"
os.makedirs(out_crops, exist_ok=True)

# This output folder stores the final annotated images with bounding boxes and
# predicted traffic-light color labels drawn on top of the original image.
out_detect = "out-detect" 
os.makedirs(out_detect, exist_ok=True) 

# Main runtime loop over all files found in the input folder.
#
# The script processes only common image formats. Every accepted image goes
# through the full detection-and-classification pipeline independently.
for filename in files:
    # Restrict processing to supported image extensions so non-image files in
    # the folder do not break the pipeline.
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Build the full path to the current input image and load it from disk.
        image_path = folder + "/" + filename
        frame = cv.imread(image_path)

        # Create a writable copy of the original image. This separate copy is
        # used only for visualization so the raw loaded image remains untouched
        # for crop extraction.
        frame_paint = frame.copy()

        # Run YOLO inference on the current image.
        #
        # imgsz=480 fixes the detector input size,
        # conf=0.2 keeps relatively permissive detections,
        # verbose=False suppresses extra console output during batch processing.
        #
        # The output of this stage is a list of detection results for the image.
        results = model.predict(frame, imgsz=480, conf=0.2, verbose=False)

        # This script processes one image at a time, so only the first result
        # item is needed here.
        result = results[0]

        # Extract the set of bounding boxes returned by YOLO for this image.
        # Each box represents one candidate traffic-light region that will later
        # be cropped and classified by color.
        boxes = result.boxes  #CHANGED

        # Iterate through all detected boxes in the current image.
        #
        # Each detected box becomes:
        # - one crop saved to disk,
        # - one color prediction,
        # - one annotation drawn on the output image.
        for i, box in enumerate(boxes):
            # Read the bounding-box coordinates in corner form:
            # (x1, y1) = top-left corner
            # (x2, y2) = bottom-right corner
            #
            # YOLO returns these values as tensors/numeric objects, so they are
            # explicitly converted to Python integers before being used for
            # slicing and drawing.
            x1, y1, x2, y2 = box.xyxy[0]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Clamp the box coordinates to valid image boundaries.
            #
            # This protects the crop operation from going outside the image when
            # a detection lies very close to the border or when rounding pushes
            # coordinates slightly beyond the valid range.
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Extract the detected traffic-light region from the original image.
            # This crop is the exact input passed into the color-classification
            # helper functions.
            crop = frame[y1:y2, x1:x2]

            # Build a stable output filename for the crop.
            #
            # The original image name is preserved, and an additional crop index
            # is attached so multiple detections from the same image can be saved
            # separately without overwriting each other.
            crop_path = out_crops + "/" + filename.rsplit(".", 1)[0] + "_crop_" + str(i) + ".png"
            cv.imwrite(crop_path, crop)

            # Run the color-classification stage on the extracted crop.
            #
            # Step 1:
            # build binary masks for red and green candidate pixels.
            #
            # Step 2:
            # compare the amount of red and green evidence and return the final
            # traffic-light color prediction together with the raw pixel counts.
            #
            # The pixel counts are not drawn in the final output here, but they
            # are still returned because they are part of the classifier helper
            # interface and are useful for debugging or later extensions.
            red_mask, green_mask = build_color_masks(crop)
            prediction, red_pixels, green_pixels = predict_color(red_mask, green_mask)

            # Draw the detected bounding box on the visualization image.
            #
            # The rectangle marks the region that YOLO localized and that was
            # subsequently passed to the color-classification helper layer.
            cv.rectangle(frame_paint, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Convert the predicted color label to uppercase so the visualization
            # looks more like a final output annotation than an internal variable
            # value.
            color_label = prediction.upper()

            # Draw the final color prediction just above the detected box.
            #
            # This is the end product of the whole pipeline for one detection:
            # - YOLO provided the location,
            # - classifier.py provided the color decision,
            # - this script writes the result back onto the image.
            cv.putText(frame_paint, color_label, (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # After all detections in the current image have been processed, save
        # the fully annotated visualization image to the final output folder.
        detect_path = out_detect + "/" + filename 
        cv.imwrite(detect_path, frame_paint)  