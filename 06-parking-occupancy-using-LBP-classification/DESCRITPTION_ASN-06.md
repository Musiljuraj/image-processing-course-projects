PROJECT SUMMARY — PARKING LBP OCCUPANCY CLASSIFICATION



**Purpose:** 

This project detects parking-space occupancy in full parking-lot images. Its aim is to classify each mapped parking slot as empty/free (0) or occupied/full (1) using LBP texture features and a supervised classifier, while also comparing multiple preprocessing/LBP/classifier configurations and ranking them by accuracy and processing time.



**Mechanism:** 

The project starts in **main.py**. It defines dataset paths, experiment grids, and runs a full experiment search. Shared inputs are loaded once: training patches from data/training/{free,full}, parking-space polygons from data/parking\_map\_python.txt, and test images + matching ground-truth files from data/test\_images\_zao. 

For each configuration, the pipeline prepares training features, trains one classifier, then processes every test image as: full image -> ordered parking-space ROIs -> preprocessing (grayscale, resize, optional contrast normalization, optional filtering) -> LBP image + spatial histogram descriptor -> feature matrix -> classifier prediction -> comparison with ground-truth labels from testN.txt. 

Results are aggregated into TP/TN/FP/FN, accuracy, and timing; all experiments are then ranked.



**Call stack** / used modules and functions: 



**main.py:main()** -> experiment\_search.run\_experiment\_search() -> parking\_training\_io.load\_all\_training\_records() \[loads training images from free/full folders, labels free=0 and full=1] + parking\_io.load\_parking\_map() \[loads slot quadrilaterals] + parking\_io.load\_test\_images() \[loads sorted testN.jpg and matching testN.txt]. 



**For each experiment:** experiment\_search.run\_one\_experiment() -> parking\_lbp\_dataset.prepare\_training\_feature\_records() \[converts training patches to ROI-like records and sends them through the common pipeline] -> preprocessing.preprocess\_all\_rois() / preprocess\_one\_roi() \[grayscale, resize, contrast normalization, filter] -> lbp\_features.extract\_lbp\_features\_from\_records() / compute\_spatial\_lbp\_descriptor() \[computes LBP-coded image, splits it into a grid, builds per-cell histograms, concatenates them into one feature vector] -> parking\_lbp\_dataset.build\_training\_matrix\_and\_labels() -> parking\_lbp\_classifier.train\_classifier() \[KNN or linear SVM]. 

**Test branch per image**: roi\_extraction.extract\_all\_rois\_from\_image() -> extract\_one\_roi() -> geometry.four\_point\_transform() \[rectifies one skewed parking quadrilateral into one rectangular ROI patch] -> parking\_lbp\_dataset.prepare\_test\_feature\_records() -> build\_test\_matrix() -> parking\_lbp\_classifier.predict\_labels() / predict\_scores() / build\_prediction\_records() -> evaluation.evaluate\_one\_test\_case() \[loads testN.txt labels and computes confusion counts + accuracy]. 

**After all experiments**: results\_io.rank\_experiment\_results() -> results\_io.save\_experiment\_outputs().





**Input data** / required structure: 

In the project root, you need: data/training/free/ and data/training/full/ containing readable training patch images (.png/.jpg/.jpeg/.bmp); data/parking\_map\_python.txt containing one parking space per line in format x1 y1 x2 y2 x3 y3 x4 y4; data/test\_images\_zao/ containing test images named testN.jpg and matching testN.txt files in the same folder. Each testN.txt must contain whitespace-separated integer labels in the same order as parking\_map\_python.txt; convention is 0 = empty/free, 1 = occupied/full. Required Python packages: cv2, numpy, scikit-learn.



**Output**: 

Main run produces console summary plus **outputs/results/final\_run/parking\_lbp\_results.csv** (flat ranked table of all experiment results) and **outputs/results/final\_run/parking\_lbp\_summary.txt** (human-readable best/top results). The auxiliary inspection workflow in **inspect\_best\_config.py** reruns the best-ranked configuration on one selected test image and saves **outputs/inspection/best\_config/overlay/<test>\_overlay.jpg**, **outputs/inspection/best\_config/roi/space\_XX.jpg**, **outputs/inspection/best\_config/processed/space\_XX.jpg, outputs/inspection/best\_config/lbp/space\_XX.jpg, and outputs/inspection/best\_config/report/<test>\_inspection\_report.txt**.



**Run procedure / checks**: 

First verify that free/ and full/ both exist and are non-empty, every testN.jpg has matching testN.txt, each txt label count matches the number of parking slots in parking\_map\_python.txt, and required libraries are installed. Then run the complete experiment search:

**python main.py**

After that, to inspect the winning configuration on one test image and save visual/debug outputs, run:

**python inspect\_best\_config.py**

Important note: the provided scripts have no CLI argument parser; input paths, output paths, and experiment grids are hardcoded in main.py and inspect\_best\_config.py, so different paths or custom configurations require editing those files or calling the underlying functions manually from Python.

