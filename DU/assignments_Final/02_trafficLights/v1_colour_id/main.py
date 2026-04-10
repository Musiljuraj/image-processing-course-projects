# This module is the top-level evaluation runner of the small color-classification project.
# It does not implement the actual color decision logic itself. Instead, it coordinates
# the whole end-to-end flow around the lower-level classifier module.
#
# In the overall runtime flow, this file performs:
#
#     input folder with test images
#         -> iterate through every supported image file
#         -> load one image from disk
#         -> ask classifier.py to build red/green masks
#         -> ask classifier.py to compare red-vs-green evidence
#         -> infer the expected class from the filename prefix
#         -> compare expected class with predicted class
#         -> update global accuracy and confusion-matrix counters
#         -> save the image into the output folder of the predicted class
#         -> print one final summary after all files are processed
#
# A key idea here is separation of concerns:
# - this file controls dataset traversal, evaluation, and result bookkeeping,
# - classifier.py contains the actual color-evidence extraction and decision logic.
#
# That keeps this module focused on orchestration:
# it is the place where the project turns
#     "many individual image classifications"
# into
#     "one complete batch evaluation with summary statistics".

import os
import cv2 as cv
from classifier import build_color_masks, predict_color

# This is the input directory containing all test images that should be processed
# by the batch evaluation run.
folder = "test-images"

# The file list is sorted so the run order is deterministic.
# That makes console output, saved outputs, and debugging easier to reproduce.
files = sorted(os.listdir(folder))

# Create output dirs if not already created
#
# The project stores processed images into two separate output folders based on
# the predicted class. This makes the final results easy to inspect manually:
# - out-red contains images predicted as red
# - out-green contains images predicted as green
out_red = "out-red" 
out_green = "out-green"  
os.makedirs(out_red, exist_ok=True)  
os.makedirs(out_green, exist_ok=True)  

# These counters track the overall batch-evaluation result:
# - total ........ number of images that were actually evaluated
# - correct ...... number of correct predictions
# - wrong ........ number of incorrect predictions
#
# Files that do not encode a known expected class in their filename are skipped
# and therefore do not contribute to these counters.
total = 0 
correct = 0
wrong = 0

# These four counters implement the confusion matrix for the binary problem:
# - tp ........ truth red, predicted red
# - fn ........ truth red, predicted green
# - fp ........ truth green, predicted red
# - tn ........ truth green, predicted green
#
# In this project, "red" is treated as the positive class.
tp = 0
fn = 0
fp = 0
tn = 0

# Process all images
#
# The main batch loop walks through every file found in the input folder.
# Only image files with supported extensions are processed.
for filename in files:
    # The project accepts common raster-image formats used for the test set.
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Build the full file path and load the image from disk.
        image_path = folder + "/" + filename
        img = cv.imread(image_path)

        # This is the actual classification stage delegated to classifier.py:
        # 1. build color masks that isolate red-like and green-like pixels
        # 2. compare the amount of red-vs-green evidence
        #
        # The returned pixel counts are useful for debugging and for understanding
        # why the final color label was chosen.
        red_mask, green_mask = build_color_masks(img)
        prediction, red_pixels, green_pixels = predict_color(red_mask, green_mask)

        # The expected ground-truth class is encoded directly in the filename.
        # This file therefore acts as both:
        # - the batch runner
        # - the evaluation layer
        #
        # If the filename does not start with a recognized class prefix, the file
        # is skipped because there is no trustworthy expected label to compare
        # against.
        if filename.startswith("red"):
            expected = "red"
        elif filename.startswith("green"):
            expected = "green"
        else:
            print(filename, " skipped - unknown expected class")
            print("-----------------------------------")
            continue

        # Count only images that actually reached the evaluation stage.
        total = total + 1 

        # Compare the predicted class with the expected class and update the
        # main success/failure counters.
        if prediction == expected:
            result = "OK"
            correct = correct + 1 
        else:
            result = "FAIL"
            wrong = wrong + 1
        
        # Update the confusion-matrix counters.
        #
        # The matrix is built explicitly from the pair:
        #     expected class + predicted class
        #
        # This keeps the final summary easy to read and easy to verify.
        if expected == "red" and prediction == "red":
            tp = tp + 1 
        elif expected == "red" and prediction == "green":
            fn = fn + 1 
        elif expected == "green" and prediction == "red":
            fp = fp + 1 
        elif expected == "green" and prediction == "green":
            tn = tn + 1 
        
        # Choose the output folder according to the predicted class, not the
        # expected class.
        #
        # This is important because the saved folder structure represents how the
        # classifier itself grouped the images, which is useful for manual review
        # of both correct and incorrect predictions.
        if prediction == "red": 
            output_path = out_red + "/" + filename 
        elif prediction == "green": 
            output_path = out_green + "/" + filename 

        # Save the original image into the folder of its predicted class.
        # The image itself is not modified here; only its destination is decided
        # by the classifier result.
        cv.imwrite(output_path, img) 

        # These lines were used during development for detailed per-image
        # inspection. They remain commented out so the script can stay quiet by
        # default while still making quick manual debugging easy when needed.
        #print(filename, " expected:", expected, " predicted:", prediction, " result:", result)
        #print("red pixels:", red_pixels, " green pixels:", green_pixels)
        #print("saved to:", output_path) 
        #print("-----------------------------------")

        # These preview windows were also used during development to inspect:
        # - the original image
        # - the red mask
        # - the green mask
        #
        # They are disabled for normal batch execution because the intended use
        # of this script is automatic processing of the full test folder.
        # cv.imshow("original", img)
        # cv.imshow("red mask", red_mask)
        # cv.imshow("green mask", green_mask)
        # cv.waitKey(0)

# After all images have been processed, compute final accuracy from the
# confusion-matrix totals.
#
# The formula uses:
#     (true positives + true negatives) / all evaluated samples
accuracy = (tp + tn) / (tp + tn + fp + fn) 

# Convert the raw accuracy ratio into percentage form for human-readable output.
accuracy_percent = accuracy * 100 

# Print the final batch summary.
#
# The output is intentionally split into:
# - general totals
# - confusion matrix
# - final accuracy
#
# This gives both a quick overview and a more detailed breakdown of classifier
# behavior across the two classes.
print("SUMMARY")
print("total:", total)  
print("correct:", correct) 
print("wrong:", wrong)
print("-----------------------------------")
print("CONFUSION MATRIX") 
print("                 Prediction-red   Prediction-green") 
print("Truth-red            ", tp, "                 ", fn)
print("Truth-green          ", fp, "                 ", tn)
print("-----------------------------------")
print(f"Accuracy: {accuracy_percent:.1f} %")  

# This window-cleanup call was relevant only for the older interactive-preview
# workflow. It remains disabled because the normal batch run does not open any
# windows.
# cv.destroyAllWindows()