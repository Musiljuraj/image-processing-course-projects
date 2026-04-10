PROJECT SUMMARY – RED/GREEN IMAGE CLASSIFICATION



Purpose:

This project batch-classifies images as red or green and simultaneously evaluates how accurate the classification is over a test set.



**How it works:**

The program starts in **main.py.** It **loads all supported images** from the folder test-images, then **for each image calls classifier.py**. In classifier.py, build\_color\_masks(img) slightly blurs the image, converts it from BGR to HSV, creates a red mask and a green mask using HSV thresholds, and cleans both masks with simple morphology. Then predict\_color(red\_mask, green\_mask) counts red and green pixels and predicts red if red pixels > green pixels, otherwise green.



**Call stack:**

main.py -> loads files and images -> build\_color\_masks(img) in classifier.py -> predict\_color(red\_mask, green\_mask) in classifier.py -> main.py compares prediction with expected class from filename -> updates correct/wrong counts and confusion matrix -> saves image into predicted output folder -> prints final summary.



**Modules / functions:**

\- main.py: top-level runner; handles dataset traversal, evaluation, saving outputs, and summary printing.

\- classifier.py: image-analysis logic.

\- build\_color\_masks(img): builds cleaned binary red/green masks in HSV space.

\- predict\_color(red\_mask, green\_mask): compares red vs green pixel counts and returns final class.



**Input:**

The project needs a folder named test-images in the same directory as main.py. It should contain readable .png, .jpg, or .jpeg images. To be evaluated, filenames must start with red or green, because the expected class is taken from the filename prefix.



**Output:**

The project creates two folders: out-red and out-green. Each processed original image is saved into one of them according to the predicted class. It also prints to console: total images, correct, wrong, confusion matrix, and final accuracy percentage.



**How to run:**

1\. Make sure main.py and classifier.py are in the same folder.

2\. Make sure Python, OpenCV (cv2), and NumPy are installed.

3\. Create test-images and place valid red/green test images inside.

4\. Check that filenames begin with red or green and use supported extensions.

5\. Run: python main.py



Important conditions:

\- test-images must exist,

\- images must be readable,

\- at least one valid evaluable image should be present,

\- the method is simple color-threshold classification, so results depend on visible red/green content, lighting, and image quality.

