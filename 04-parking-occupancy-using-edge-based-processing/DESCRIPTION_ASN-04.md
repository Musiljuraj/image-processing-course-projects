PROJECT SUMMARY – PARKING OCCUPANCY DETECTION BY CLASSICAL IMAGE ANALYSIS



**Purpose:**

This project detects whether each predefined parking space in a fixed-camera parking-lot dataset is OCCUPIED or EMPTY, and automatically searches for the best preprocessing + edge-detection + classification settings on the whole dataset.



**Mechanism**:

The project uses a parking-space map (4-point polygon for each space). For every test image, it rectifies each parking place into its own rectangular ROI by perspective transform, converts ROI to grayscale, optionally smooths it, runs edge detection, computes edge\_count, roi\_pixel\_count and edge\_ratio = edge\_count / roi\_pixel\_count, then classifies the space as occupied if edge\_ratio > occupancy\_threshold\_ratio, otherwise empty. Predictions are compared with ground-truth labels from matching .txt files, TP/TN/FP/FN and accuracy are computed, and the whole pipeline is repeated for all valid parameter combinations defined in main.py. Results are ranked and the best configuration is saved.



**Call stack** / used modules:

1\. **main.py** = entry point. main() resolves project paths, checks input folders, loads parking map and test images, defines search\_space, calls experiment\_search.run\_exhaustive\_search(), then saves results by results\_io.ensure\_results\_directory(), save\_ranked\_results\_csv(), save\_best\_results\_summary().

2\. **experiment\_search.py** = exhaustive-search layer. run\_exhaustive\_search() builds ground-truth cache, generates valid configurations, evaluates every configuration on the full dataset, ranks all results, returns best\_result.

3\. Per tested configuration, **evaluate\_configuration\_on\_dataset**() runs:

&#x20;  roi\_extraction.extract\_all\_rois\_from\_image() -> extract\_one\_roi() -> geometry.four\_point\_transform() -> geometry.order\_points()

&#x20;  -> preprocessing.preprocess\_all\_rois() -> preprocess\_one\_roi() -> convert\_roi\_to\_grayscale() -> apply\_filter()

&#x20;  -> edge\_detection.detect\_edges\_all\_records() -> detect\_edges\_one\_record() -> detect\_edges\_canny() or detect\_edges\_sobel() -> compute\_edge\_statistics()

&#x20;  -> evaluation.classify\_all\_edge\_records() -> classify\_one\_edge\_ratio()

&#x20;  -> evaluation.evaluate\_one\_image() -> validate\_ground\_truth\_label(), initialize\_confusion\_counts(), compute\_accuracy()

&#x20;  -> experiment\_search.merge\_confusion\_counts() for dataset totals.

4\. After all configurations are tested, **experiment\_search.rank\_experiment\_results() sorts them and results\_io.py writes outputs**.

5\. inspect\_best\_config.py = optional inspection script. It loads the best configuration from outputs/results/results.csv, reruns the same pipeline for one selected image, and saves overlay / processed ROI / edge-map / summary outputs through debug\_utils.py.



**Brief module roles:**

\- main.py: top-level orchestration and search-space definition.

\- parking\_io.py: loads parking\_map\_python.txt, test images, and ground-truth labels.

\- geometry.py: point ordering + perspective warp of one parking-space polygon.

\- roi\_extraction.py: extracts one/all parking-space ROI patches.

\- preprocessing.py: grayscale conversion and optional smoothing filter.

\- edge\_detection.py: Sobel/Canny selection, execution, edge statistics.

\- evaluation.py: occupied/empty decision from edge\_ratio and evaluation metrics.

\- experiment\_search.py: valid-config generation, exhaustive evaluation, ranking.

\- results\_io.py: saves ranked CSV and text summary.

\- debug\_utils.py: saves debug overlay and image patches.

\- inspect\_best\_config.py: visual inspection of the best-ranked configuration.



**Input** / required data:

The project expects this structure relative to the script location:

data/parking\_map\_python.txt

data/test\_images\_zao/test1.jpg, test1.txt, test2.jpg, test2.txt, ...

parking\_map\_python.txt must contain one parking-space quadrilateral per non-empty line as 8 numbers: x1 y1 x2 y2 x3 y3 x4 y4. Each .jpg must be readable and must have a matching .txt with the same basename in the same folder. Each .txt must contain integer labels separated by whitespace or commas; the number of labels must match the number of parking spaces. Current label convention in main.py is occupied=1, empty=0. Practical condition: all images must come from the same fixed camera view so the parking map aligns with them.



**Output:**

Running main.py produces:

\- outputs/results/results.csv = full ranked table of all tested configurations and their metrics

\- outputs/results/best\_results.txt = human-readable summary of the best and top-ranked configurations

It also prints progress and the best configuration to console.

Running inspect\_best\_config.py additionally produces:

\- outputs/inspection/best\_config/<image\_name>/overlay/...

\- outputs/inspection/best\_config/<image\_name>/processed/...

\- outputs/inspection/best\_config/<image\_name>/edges/...

\- outputs/inspection/best\_config/<image\_name>/inspection\_summary.txt

Optional raw/grayscale ROI folders are saved only if enabled inside inspect\_best\_config.py.



**Functional run commands:**

python3 main.py

python3 inspect\_best\_config.py



**Important note about arguments:**

This version has **no CLI arguments.** Paths, the exhaustive search space, selected inspection image (debug\_image\_index), and inspection save options are edited directly inside main.py and inspect\_best\_config.py.



**Minimal run procedure / initial checks:**

1\. Keep all modules in one project folder and keep the exact data/ layout.

2\. Check that data/parking\_map\_python.txt exists and matches the camera geometry of the dataset.

3\. Check that data/test\_images\_zao/ contains readable .jpg images and each image has matching .txt labels.

4\. Check that every .txt has the correct number of labels and uses the expected label convention.

5\. Ensure Python 3 with OpenCV (cv2) and NumPy is installed.

6\. Run: python3 main.py

7\. Read outputs/results/results.csv and outputs/results/best\_results.txt.

8\. For visual inspection of the best found setup, run: python3 inspect\_best\_config.py

