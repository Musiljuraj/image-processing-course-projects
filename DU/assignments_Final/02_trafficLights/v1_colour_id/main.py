import os
import cv2 as cv
from classifier import build_color_masks, predict_color

folder = "test-images"
files = sorted(os.listdir(folder))

#Create output dirs if not already created
out_red = "out-red" 
out_green = "out-green"  
os.makedirs(out_red, exist_ok=True)  
os.makedirs(out_green, exist_ok=True)  

total = 0 
correct = 0
wrong = 0

tp = 0
fn = 0
fp = 0
tn = 0

#Process all images
for filename in files:
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        image_path = folder + "/" + filename
        img = cv.imread(image_path)

        red_mask, green_mask = build_color_masks(img)
        prediction, red_pixels, green_pixels = predict_color(red_mask, green_mask)

        if filename.startswith("red"):
            expected = "red"
        elif filename.startswith("green"):
            expected = "green"
        else:
            print(filename, " skipped - unknown expected class")
            print("-----------------------------------")
            continue

        total = total + 1 

        if prediction == expected:
            result = "OK"
            correct = correct + 1 
        else:
            result = "FAIL"
            wrong = wrong + 1
        
        if expected == "red" and prediction == "red":
            tp = tp + 1 
        elif expected == "red" and prediction == "green":
            fn = fn + 1 
        elif expected == "green" and prediction == "red":
            fp = fp + 1 
        elif expected == "green" and prediction == "green":
            tn = tn + 1 
        
        if prediction == "red": 
            output_path = out_red + "/" + filename 
        elif prediction == "green": 
            output_path = out_green + "/" + filename 

        cv.imwrite(output_path, img) 

        #print(filename, " expected:", expected, " predicted:", prediction, " result:", result)
        #print("red pixels:", red_pixels, " green pixels:", green_pixels)
        #print("saved to:", output_path) 
        #print("-----------------------------------")

        # cv.imshow("original", img)
        # cv.imshow("red mask", red_mask)
        # cv.imshow("green mask", green_mask)
        # cv.waitKey(0)

accuracy = (tp + tn) / (tp + tn + fp + fn) 
accuracy_percent = accuracy * 100 
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
# cv.destroyAllWindows()

