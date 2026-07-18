PROJECT SUMMARY – VIDEO-BASED EYE-STATE DETECTION (OPEN/CLOSED)



**Purpose**: 

This project processes a face video frame by frame, localizes one main face, detects its eyes (and optionally mouth), classifies the frame eye state as OPEN or CLOSED, and then evaluates the predicted frame labels against ground truth while also reporting localization-time statistics. 



**Mechanism**: 

The pipeline starts by loading Haar cascade XML detectors for frontal face, profile face, eyes, and mouth. 

For each video frame, **main.py** converts the image to grayscale, 

**detectors.py** runs **face detection** (frontal first; if nothing is found, profile detection is tried on the original and horizontally flipped frame), overlapping detections are merged, and one main face is selected with preference for temporal continuity from the previous frame. 

**Inside that face ROI, eyes are searched** in the upper face region and **mouth in the lower region**. 

Then **eye\_state.py** classifies detected eye regions using a hybrid of dark-region threshold analysis, connected-component/blob analysis, and Hough-circle evidence for iris/pupil; frame-level aggregation returns OPEN or CLOSED. 

After the video ends, **evaluation.py** loads eye-state ground truth, aligns prediction and reference sequences, computes accuracy, confusion-style counts, and localization timing statistics, prints them, and saves a report.



**Call stack** / used modules and functions:

1\. **main.py** is the entry point; execution starts at `if \_\_name\_\_ == "\_\_main\_\_": main()`. main() calls `get\_project\_paths()` (builds fixed input/output paths), `ensure\_output\_directories()`, `get\_output\_file\_paths()`, `reset\_text\_file()`, `load\_cascades()`, and **`open\_video()**`. In the frame loop it calls `convert\_to\_grayscale()`, `**detect\_faces()`, `select\_main\_face()`, `detect\_face\_parts()`, `classify\_eye\_state()`, and `store\_frame\_result()**`. After processing it calls `**evaluate\_results()**`, `print\_evaluation\_summary()`, and `save\_evaluation\_report()`. 



2\. **detectors.py** = localization layer. `load\_cascades()` loads all XML classifiers. `detect\_faces()` optionally downscales the grayscale frame, runs frontal-face detection, otherwise profile detection on normal + flipped views, rescales boxes back, and merges overlaps; internally it uses helpers such as `\_prepare\_downscaled\_gray\_frame()`, `\_unflip\_box\_horizontally()`, `\_rescale\_box\_to\_original\_frame()`, and `merge\_face\_boxes()`. `select\_main\_face()` chooses the largest or most temporally consistent face. `detect\_face\_parts()` extracts the face ROI and calls `\_detect\_eyes\_in\_face\_roi()` and optionally `\_detect\_mouth\_in\_face\_roi()`. 



3\. **eye\_state.py** = eye-state classification layer. `classify\_eye\_state()` is the public frame-level classifier. For each detected eye, it calls `classify\_single\_eye()`, which uses `extract\_eye\_roi()`, `preprocess\_eye\_roi()` (normalization, band crop, equalization, blur), `threshold\_eye\_roi()`, `compute\_threshold\_evidence()`, `detect\_iris\_circle()`, `compute\_threshold\_score()`, and `compute\_eye\_open\_score()`, then decides OPEN/CLOSED; frame-level logic combines one or two eye results into the final frame label. If no eyes are detected, the frame is classified as CLOSED. 



4\. **evaluation.py** = post-processing and evaluation. `evaluate\_results()` calls `load\_ground\_truth()`, `extract\_predicted\_labels()`, `extract\_localization\_times()`, `align\_label\_sequences()`, `compute\_accuracy()`, `compute\_confusion\_counts()`, and `compute\_timing\_stats()`. Output text is created by `format\_evaluation\_summary()`, printed by `print\_evaluation\_summary()`, and saved by `save\_evaluation\_report()`. :contentReference\[oaicite:10]{index=10}



**Input / required data**: 

The project uses fixed paths relative to main.py. 

Required files are: `input/video/fusek\_face\_car\_01.avi`, `input/ground\_truth/eye-state.txt`, `input/cascades/face/haarcascade\_frontalface\_default.xml`, `input/cascades/face/haarcascade\_profileface.xml`, `input/cascades/eye/eye\_cascade\_fusek.xml`, and `input/cascades/mouth/haarcascade\_smile.xml`. 

The video must be readable by OpenCV. Ground truth must be a text file with one non-empty eye-state label per line; accepted forms include `open/opened` and `close/closed/shut`. 

Python needs OpenCV (`cv2`) and NumPy installed. Because `SHOW\_TEST\_VIDEO = True` in the current code, a GUI/display is needed unless you manually switch preview off in main.py. 





**Output**: 

The program stores one structured record per processed frame in memory and finally produces **two actual text outputs**: `**output/logs/run\_log.txt**` and `**output/reports/evaluation\_report.txt**`. It also prints the evaluation summary to console. Directories `output/annotated\_video`, `output/frames`, `output/logs`, and `output/reports` are created automatically, but in the current submitted code only the log and report are written; annotated video/individual frames are displayed live only in a preview window and are not saved. The report contains compared-sequence lengths, correct count, accuracy percentage, confusion-style counts, and localization timing statistics (mean/min/max in ms). 



**Run** command(s): 

From the project root (folder containing main.py), use `python main.py` or `**python3 main.py**`. This version has **no CLI arguments; file names, paths, preview mode, mouth detection, and downscale factor are configured directly in main.py.** The effective hardcoded input video is `input/video/fusek\_face\_car\_01.avi` and the hardcoded ground truth is `input/ground\_truth/eye-state.txt`. :contentReference\[oaicite:16]{index=16}



**Minimal run procedure** / initial checks:

1\. Keep `main.py`, `detectors.py`, `eye\_state.py`, and `evaluation.py` in one project folder. :contentReference\[oaicite:17]{index=17}

2\. Verify the exact required folder/file layout under `input/` and that all four cascade XML files exist. :contentReference\[oaicite:18]{index=18} :contentReference\[oaicite:19]{index=19}

3\. Check that `fusek\_face\_car\_01.avi` opens correctly and that `eye-state.txt` contains one label per line for the processed video frames. Minor length mismatch is tolerated because evaluation truncates both sequences to the shorter length. :contentReference\[oaicite:20]{index=20} :contentReference\[oaicite:21]{index=21}

4\. Ensure Python 3 + OpenCV + NumPy are installed. :contentReference\[oaicite:22]{index=22} :contentReference\[oaicite:23]{index=23}

5\. Run `python3 main.py` (or `python main.py`). If preview is enabled, press `q` or `Esc` to stop early; otherwise let the video finish. :contentReference\[oaicite:24]{index=24}

6\. Read `output/logs/run\_log.txt` and `output/reports/evaluation\_report.txt` for the final result. :contentReference\[oaicite:25]{index=25} :contentReference\[oaicite:26]{index=26}

