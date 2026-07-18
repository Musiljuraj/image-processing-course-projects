PROJECT SUMMARY — VIDEO EYE-STATE RECOGNITION USING HAAR LOCALIZATION + LBP CLASSIFICATION



**Purpose:**

This project detects a face in a video, localizes the eyes (and optionally mouth), and classifies the eye state frame-by-frame as OPEN or CLOSE. Its final goal is not only runtime eye-state recognition on the target video, but also automatic comparison of several LBP configurations and final rerun with the best one.



**Mechanism:**

The project has two layers. First, it **builds a trained eye-state model** from the mrlEyes training dataset: dataset image -> preprocessing -> LBP descriptor -> classifier training. 

Second, **it runs the real video pipeline**: frame -> grayscale -> face detection -> stable main-face selection -> eye localization -> eye ROI extraction -> same preprocessing -> LBP feature extraction -> classifier prediction for each eye -> aggregation into one frame-level OPEN/CLOSE label -> storage of timings and predictions -> comparison with ground truth labels from eye-state.txt.



Preprocessing normalizes eye images to a fixed representation (grayscale, resize to 80x40, optional central analysis-band crop, contrast normalization, light filtering). LBP then converts the normalized eye image into histogram-based texture features. The classifier layer supports KNN and linear SVM; the recommended experiment search uses KNN (k=3) and compares 3 LBP variants. Runtime classification is done by the LBP model; if needed, the old heuristic eye-state module is used as fallback.



**Whole call stack:**

**1. Final automatic version:**

&#x20;  **auto\_select\_and\_run.py** -> main() -> auto\_select\_best\_and\_run\_final()

&#x20;  -> experiment\_search.run\_experiment\_search()

&#x20;  -> build experiment configs + load shared training records

&#x20;  -> for each experiment: build\_eye\_lbp\_model\_from\_training\_records() -> process video exactly like runtime pipeline -> evaluate\_results()

&#x20;  -> rank results -> get\_best\_experiment\_result() -> extract\_runtime\_configs\_from\_experiment\_result()

&#x20;  -> main.run\_final\_pipeline() with winning configs

&#x20;  -> save search report, winner summary, final rerun log/report, and one top-level auto-selection summary.



**2. Manual/default runtime version**:

&#x20;  main.py -> main() -> run\_final\_pipeline()

&#x20;  -> get\_project\_paths(), ensure\_output\_directories(), get\_output\_file\_paths()

&#x20;  -> eye\_lbp\_classifier.build\_eye\_lbp\_model()

&#x20;  -> eye\_training\_io.load\_all\_eye\_training\_records()

&#x20;  -> eye\_preprocessing.preprocess\_\*()

&#x20;  -> lbp\_features.extract\_\* / compute\_\*()

&#x20;  -> eye\_lbp\_dataset.build\_training\_matrix\_and\_labels()

&#x20;  -> eye\_lbp\_classifier.train\_classifier()

&#x20;  -> detectors.load\_cascades()

&#x20;  -> open video and for each frame:

&#x20;     detect\_faces() -> select\_main\_face() -> detect\_face\_parts()

&#x20;     -> eye\_state\_lbp.classify\_eye\_state\_lbp()

&#x20;        -> classify\_single\_eye\_lbp()

&#x20;        -> eye\_state.extract\_eye\_roi()

&#x20;        -> eye\_lbp\_dataset.prepare\_runtime\_feature\_record()

&#x20;        -> eye\_preprocessing.preprocess\_runtime\_eye\_roi()

&#x20;        -> lbp\_features.extract\_lbp\_features\_from\_runtime\_eye\_image()

&#x20;        -> eye\_lbp\_classifier.predict\_from\_runtime\_feature\_record()

&#x20;        -> aggregate\_eye\_predictions()

&#x20;     -> store\_frame\_result()

&#x20;  -> evaluation.evaluate\_results()

&#x20;  -> evaluation.save\_evaluation\_report()



**Brief module/function roles:**

\- **auto\_select\_and\_run.py**: highest orchestration layer; runs experiment search, selects winner, executes final rerun, saves combined summary.

\- **experiment\_search.py**: systematic comparison layer; generates configuration combinations, trains/evaluates each experiment, ranks them, saves search reports.

\- **main.py**: reusable end-to-end runtime pipeline for one chosen configuration.

\- **detectors.py**: Haar cascade localization; loads cascades, detects faces, selects temporally stable main face, detects eyes/mouth inside face ROI.

\- **eye\_training\_io.py**: training-data loader; reads mrlEyes images from disk, parses filename metadata, builds structured training records.

\- **eye\_preprocessing.py**: shared normalization of eye images/ROIs before LBP.

\- **lbp\_features.py**: computes LBP image and histogram descriptor, including grid-based spatial LBP.

\- **eye\_lbp\_dataset.py**: bridge from structured eye records to X\_train / y\_train and runtime feature records.

\- **eye\_lbp\_classifier.py**: validates classifier config, trains model, predicts labels/scores, builds reusable model bundle.

\- **eye\_state\_lbp.py**: runtime eye-state decision layer; classifies each detected eye ROI and aggregates to one frame label.

\- **eye\_state.py**: older heuristic eye-state classifier; reused mainly for ROI extraction and optional fallback.

\- **evaluation.py**: compares predicted OPEN/CLOSE sequence with ground truth, computes confusion-style counts, accuracy, and timing statistics, formats/saves report.



**Required input data and where they must be stored:**

Project is path-driven; there are no required CLI arguments. Inputs must exist in fixed project-relative locations:

\- input/video/fusek\_face\_car\_01.avi                      = target video

\- input/ground\_truth/eye-state.txt                       = one eye-state label per line

\- input/training/mrlEyes\_2018\_01/                        = extracted training eye dataset

\- input/cascades/face/haarcascade\_frontalface\_default.xml

\- input/cascades/face/haarcascade\_profileface.xml

\- input/cascades/eye/eye\_cascade\_fusek.xml

\- input/cascades/mouth/haarcascade\_smile.xml



**Training dataset condition**:

The training dataset images must follow the mrlEyes filename format:

subject\_id\_image\_id\_gender\_glasses\_eye\_state\_reflections\_lighting\_sensor

example: s0001\_00001\_0\_0\_0\_0\_0\_01.png

Binary labels are 0 = close, 1 = open.



**What the project produces as output:**

Manual/default run (main.py):

\- output/logs/run\_log.txt

\- output/reports/evaluation\_report.txt

\- console summary of accuracy and timings



**Automatic best-configuration run (auto\_select\_and\_run.py):**

\- output/experiments/results/eye\_experiment\_search\_report.txt

\- output/experiments/results/best\_experiment\_summary.txt

\- output/experiments/results/auto\_selected\_best\_summary.txt

\- output/logs/run\_log\_auto\_selected\_best.txt

\- output/reports/evaluation\_report\_auto\_selected\_best.txt

\- console summary of selected experiment and final output paths



What these outputs mean:

The final run log contains the chronological runtime record of the whole selected run: model build, detector initialization, frame-processing progress, and final save locations. The final evaluation report contains the real end result of the final full rerun: number of processed frames, compared labels, correct predictions, accuracy, confusion-style counts, and timing statistics for localization, classification, and total frame processing. The experiment-search report summarizes all tested configurations and their ranking. The best-experiment summary describes only the winning configuration selected during the search phase. The automatic selection summary connects both phases together: which experiment won and what happened when that winning configuration was rerun in the final pipeline.



**Difference in accuracies:**

Two different accuracies appear in the output because they come from two different phases:

\- 88.33% = accuracy of the selected experiment during the experiment-search phase

\- 74.67% = accuracy of the final full rerun after that configuration was selected



**Reason for the difference:**

The experiment search is not evaluated under exactly the same conditions as the final rerun. During search, the project uses a reduced/controlled evaluation setup so configurations can be compared quickly and consistently (in this project specifically, the search is limited to max\_frames = 120 and max\_training\_records\_per\_class = 150). After the best configuration is chosen, the project runs the final full pipeline under larger real conditions: the full target video is processed and a much larger training set is used to build the final model. Because the data volume, runtime conditions, and evaluation scope are different, the search-phase accuracy and final-run accuracy do not have to match. Therefore, 88.33% is the ranking/result used to choose the best configuration, while 74.67% is the true final performance of the complete final rerun.



**Important note:**

The code creates output/annotated\_video and output/frames directories, but in the current uploaded version it mainly saves text reports/logs; it does not actually export annotated video or per-frame image files as a primary output.



**Functional run commands:**

1\. Manual default full pipeline:

&#x20;  python main.py



2\. Automatic search -> best configuration selection -> final rerun:

&#x20;  **python auto\_select\_and\_run.py**



3\. Experiment search only (report generation without final rerun):

&#x20;  python experiment\_search.py



4\. Optional explicit Python call with preview enabled:

&#x20;  python -c "from main import run\_final\_pipeline; run\_final\_pipeline(run\_name='preview\_run', show\_preview=True, output\_file\_prefix='preview\_run', print\_summary=True)"

x

Run procedure / initial checks:

1\. Make sure Python environment contains at least OpenCV (cv2), NumPy, and scikit-learn.

2\. Place the project files in one project root and create the exact input/ and output/ structure above.

3\. Check that all 4 cascade XML files exist and are readable.

4\. Check that input/video/fusek\_face\_car\_01.avi exists and can be opened.

5\. Check that input/ground\_truth/eye-state.txt exists, is non-empty, and contains valid open/close labels line-by-line.

6\. Check that input/training/mrlEyes\_2018\_01 exists and contains readable eye images with valid mrlEyes filenames from both classes (open and close must both be present).

7\. Run either python main.py for one default final run, or python auto\_select\_and\_run.py for the full automatic solution.

8\. After completion, inspect output/logs/, output/reports/, and output/experiments/results/ for the generated text reports.

