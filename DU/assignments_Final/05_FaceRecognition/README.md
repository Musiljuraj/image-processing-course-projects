# ZAO Assignment 05

## Overview

This project implements a complete video-processing pipeline for detection and analysis of facial features in a driver video sequence. The program reads an input video frame by frame, detects the driver's face, localizes facial parts inside the selected face region, classifies the eye state as **open** or **close**, and evaluates the result against the provided ground-truth labels.

The implementation is organized as a small modular Python project intended for a school laboratory assignment. The main focus is:
- practical use of Haar cascade detectors,
- structured frame-by-frame processing,
- implementation of a classical eye-state classifier based on image-processing heuristics,
- final quantitative evaluation of the solution.

---

## Assignment Goal

The main purpose of the assignment is to build a working pipeline that is able to:

1. read and process a video sequence,
2. detect the main face in each frame,
3. detect important facial parts inside the selected face,
4. classify the eye state for each processed frame,
5. compare predictions with ground truth,
6. report the achieved accuracy and timing statistics.

The project is intentionally implemented as a classical computer-vision solution using OpenCV and Haar cascades, without machine learning training.

---

## Implemented Functionality

The final implementation includes:

- loading of all required Haar cascades,
- frontal face detection,
- profile face detection as fallback,
- face-box merging and filtering,
- selection of one stable main face,
- eye detection inside the upper face region,
- mouth/smile detection inside the lower face region,
- eye-state classification using:
  - threshold-based dark-region evidence,
  - blob-shape analysis,
  - circular iris/pupil evidence from `HoughCircles()`,
- optional live preview of processed frames,
- per-frame result storage,
- evaluation against ground truth,
- timing measurement of localization,
- saving of a final evaluation report,
- saving of a run log.

---

## Project Structure

```text
.
├── input
│   ├── archives
│   ├── cascades
│   │   ├── eye
│   │   │   └── eye_cascade_fusek.xml
│   │   ├── face
│   │   │   ├── haarcascade_frontalface_default.xml
│   │   │   └── haarcascade_profileface.xml
│   │   └── mouth
│   │       └── haarcascade_smile.xml
│   ├── ground_truth
│   │   └── eye-state.txt
│   └── video
│       └── fusek_face_car_01.avi
├── output
│   ├── annotated_video
│   ├── frames
│   ├── logs
│   └── reports
├── detectors.py
├── eye_state.py
├── evaluation.py
└── main.py

# Final Selected Solution

## The final selected solution uses the HoughCircles-based version of eye_state.py with the following characteristics:

normalized eye ROI processing,
threshold-based dark-region analysis,
blob filtering by area, center distance, aspect ratio, and fill ratio,
HoughCircles() as the strongest single open-eye cue,
combination of:
circle-only open route,
threshold-only open route,
hybrid circle + threshold route,
frame-level aggregation of per-eye results.

This final version was selected because it achieved the best balance among the tested Hough-based configurations.


# Tested HoughCircles-Based Parameter / Configuration Variants

## Approx. 72% accuracy

Key constants:

DARK_RATIO_TARGET = 0.18
DARK_RATIO_TOLERANCE = 0.18
BLOB_AREA_RATIO_TARGET = 0.06
BLOB_AREA_RATIO_TOLERANCE = 0.08
BLOB_AREA_RATIO_MAX = 0.28
BLOB_CENTER_DISTANCE_MAX = 0.50
BLOB_ASPECT_RATIO_MAX = 5.50
BLOB_FILL_RATIO_MIN = 0.10
HOUGH_CENTER_DISTANCE_MAX = 0.38
CIRCLE_OPEN_SCORE_THRESHOLD = 0.42
THRESHOLD_ONLY_OPEN_THRESHOLD = 0.42
THRESHOLD_ONLY_OPEN_SCORE_MIN = 0.40
SINGLE_EYE_FRAME_OPEN_THRESHOLD = 0.48
STRONG_EYE_OPEN_THRESHOLD = 0.52
FRAME_MEAN_OPEN_THRESHOLD = 0.46
FRAME_MIN_SUPPORT_THRESHOLD = 0.38

Interpretation:
too conservative,
open-eye recall still too low,
many real open frames were classified as closed


## Approx. 81% accuracy

Key constants:

DARK_RATIO_TARGET = 0.20
DARK_RATIO_TOLERANCE = 0.22
BLOB_AREA_RATIO_TARGET = 0.07
BLOB_AREA_RATIO_TOLERANCE = 0.10
BLOB_AREA_RATIO_MAX = 0.32
BLOB_CENTER_DISTANCE_MAX = 0.58
BLOB_ASPECT_RATIO_MAX = 6.50
BLOB_FILL_RATIO_MIN = 0.07
HOUGH_CENTER_DISTANCE_MAX = 0.42
CIRCLE_OPEN_SCORE_THRESHOLD = 0.35
THRESHOLD_ONLY_OPEN_THRESHOLD = 0.30
THRESHOLD_ONLY_OPEN_SCORE_MIN = 0.30
HYBRID_OPEN_SCORE_THRESHOLD = 0.27
HYBRID_THRESHOLD_SCORE_MIN = 0.22
SINGLE_EYE_FRAME_OPEN_THRESHOLD = 0.40
STRONG_EYE_OPEN_THRESHOLD = 0.44
FRAME_MEAN_OPEN_THRESHOLD = 0.39
FRAME_MIN_SUPPORT_THRESHOLD = 0.31

Interpretation:
much better open-eye recall,
but still somewhat unstable because the decision logic was already quite permissive.


## Approx. 82% accuracy

Key constants:

DARK_RATIO_TARGET = 0.20
DARK_RATIO_TOLERANCE = 0.22
BLOB_AREA_RATIO_TARGET = 0.07
BLOB_AREA_RATIO_TOLERANCE = 0.10
BLOB_AREA_RATIO_MAX = 0.30
BLOB_CENTER_DISTANCE_MAX = 0.55
BLOB_ASPECT_RATIO_MAX = 6.00
BLOB_FILL_RATIO_MIN = 0.08
HOUGH_CENTER_DISTANCE_MAX = 0.40
CIRCLE_OPEN_SCORE_THRESHOLD = 0.38
THRESHOLD_ONLY_OPEN_THRESHOLD = 0.34
THRESHOLD_ONLY_OPEN_SCORE_MIN = 0.34
HYBRID_OPEN_SCORE_THRESHOLD = 0.32
HYBRID_THRESHOLD_SCORE_MIN = 0.28
SINGLE_EYE_FRAME_OPEN_THRESHOLD = 0.44
STRONG_EYE_OPEN_THRESHOLD = 0.48
FRAME_MEAN_OPEN_THRESHOLD = 0.42
FRAME_MIN_SUPPORT_THRESHOLD = 0.34

Interpretation:
better balance between false-open and false-closed decisions,
more stable than the earlier permissive variant



## Approx. 83% accuracy — final selected tuning

Key constants:

DARK_RATIO_TARGET = 0.20
DARK_RATIO_TOLERANCE = 0.22
BLOB_AREA_RATIO_TARGET = 0.07
BLOB_AREA_RATIO_TOLERANCE = 0.10
BLOB_AREA_RATIO_MIN = 0.01
BLOB_AREA_RATIO_MAX = 0.30
BLOB_CENTER_DISTANCE_MAX = 0.55
BLOB_ASPECT_RATIO_MAX = 6.00
BLOB_FILL_RATIO_MIN = 0.08
HOUGH_CENTER_DISTANCE_MAX = 0.40
CIRCLE_OPEN_SCORE_THRESHOLD = 0.37
THRESHOLD_ONLY_OPEN_THRESHOLD = 0.33
THRESHOLD_ONLY_OPEN_SCORE_MIN = 0.32
HYBRID_OPEN_SCORE_THRESHOLD = 0.30
HYBRID_THRESHOLD_SCORE_MIN = 0.24
SINGLE_EYE_FRAME_OPEN_THRESHOLD = 0.42
STRONG_EYE_OPEN_THRESHOLD = 0.46
FRAME_MEAN_OPEN_THRESHOLD = 0.40
FRAME_MIN_SUPPORT_THRESHOLD = 0.33

Interpretation:

best compromise between recall and precision,
threshold-only route remained active but controlled,
hybrid route remained useful,
no large accuracy gain was achieved by loosening the classifier further.
