PROJECT SUMMARY



Purpose: This project detects traffic lights in road images and classifies the visible signal as RED or GREEN. Its aim is to take ordinary input images, localize traffic-light regions using a trained YOLO model, analyze the color content inside each detected crop, and save visual results.



**Mechanism**: The pipeline starts in detect.py. First, a trained YOLO model (tlight-v11.pt) is loaded. Then all images from the folder test-big-zao are processed one by one. 

For each image, YOLO predicts bounding boxes of candidate traffic lights. Each box is cropped from the original image and passed to classifier.py. There, the crop is slightly blurred, converted from BGR to HSV, thresholded into red and green masks, both masks are cleaned by small morphological opening and closing, and the number of red and green pixels is counted. 

If red pixels > green pixels, the crop is classified as red; otherwise it is classified as green (so ties also end as green).



**Call stack** / used modules and functions:

1\. detect.py = entry script and orchestration layer. It starts immediately at top level (there is no main() function).

2\. detect.py -> YOLO('tlight-v11.pt'): loads the trained detector.

3\. detect.py -> os.listdir(folder): reads files from test-big-zao.

4\. detect.py -> cv.imread(image\_path): loads each input image.

5\. detect.py -> model.predict(frame, imgsz=480, conf=0.2, verbose=False): detects traffic-light boxes.

6\. detect.py -> result.boxes / box.xyxy: reads bounding-box coordinates.

7\. detect.py -> crop = frame\[y1:y2, x1:x2]: extracts detected traffic-light crop.

8\. detect.py -> cv.imwrite(crop\_path, crop): saves each raw crop into out-crops.

9\. detect.py -> classifier.build\_color\_masks(crop): creates cleaned binary red/green masks from the crop.

10\. classifier.py -> build\_color\_masks(img): Gaussian blur -> HSV conversion -> red mask (2 HSV ranges because red wraps in HSV) + green mask (1 range) -> morphological cleanup -> returns red\_mask, green\_mask.

11\. detect.py -> classifier.predict\_color(red\_mask, green\_mask): final color decision.

12\. classifier.py -> predict\_color(...): counts nonzero pixels in both masks and returns prediction, red\_pixels, green\_pixels.

13\. detect.py -> cv.rectangle(...) and cv.putText(...): draws box and predicted label on a copy of the original image.

14\. detect.py -> cv.imwrite(detect\_path, frame\_paint): saves the final annotated image into out-detect.



**Inputs / required conditions:** The script needs:

\- detect.py and classifier.py in the same working directory,

\- trained YOLO weights file named exactly tlight-v11.pt,

\- input folder named exactly test-big-zao,

\- inside that folder: image files in .png, .jpg, or .jpeg format.

Important practical condition: the project only classifies RED vs GREEN after detection; it does not explicitly handle yellow or “off” states, and any non-red result is effectively labeled green.



**Outputs**: The project produces two output forms:

1\. out-crops/ = one saved PNG crop for each detected traffic light, named like originalname\_crop\_i.png.

2\. out-detect/ = one final annotated image per processed input image, with detected boxes and text label RED or GREEN, saved under the original filename.

If an image contains no detections, the annotated image is still saved, but without traffic-light boxes; no crop is produced for that image.



**Run procedure:** Before running, check that tlight-v11.pt exists, test-big-zao exists, and it contains readable .png/.jpg/.jpeg images. Then run the project from the directory containing both Python files and the model file with: **python detect.py**. The script automatically creates out-crops and out-detect if they do not exist, processes all supported images from test-big-zao, extracts every detected traffic-light crop, classifies each crop as red or green, and saves both the raw crops and the final annotated images.

