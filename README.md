# Image Processing and Computer Vision Course Projects

This repository contains selected image-processing and computer-vision assignments completed during my studies in Computational and Applied Mathematics at VŠB – Technical University of Ostrava.

The projects cover classical image-processing methods, object and feature detection, visual automation, handcrafted feature extraction, machine learning, and neural-network classification. Each top-level folder represents an independent assignment with its own code, input data, outputs, and documentation.

## Skills Demonstrated

* Image and video preprocessing
* Colour-based classification and object detection
* Edge detection and region-of-interest processing
* Template matching and automated visual interaction
* Face, eye, and facial-feature detection
* Local Binary Pattern feature extraction
* KNN and SVM classification
* MLP and convolutional neural-network training
* Quantitative evaluation and comparison of alternative methods

## Project Overview

| Project                                                                                                       | Problem                                                                       | Main Approach                                                                                 |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [Traffic-light classification and detection](./02-traffic-light-classification-and-detection)                 | Identify the active traffic-light colour and locate traffic lights in images. | Colour analysis, image preprocessing, and object detection                                    |
| [Dart Master game bot](./03-game-bot-dart-master)                                                             | Detect visual targets in a game and automatically interact with them.         | Template matching, target localization, and automated clicking                                |
| [Duck Hunt game bot](./03-game-bot-duck-hunt)                                                                 | Detect moving game objects represented by different visual templates.         | Multi-template and multi-scale matching with automated interaction                            |
| [Parking occupancy using edge-based processing](./04-parking-occupancy-using-edge-based-processing)           | Determine whether individual parking spaces are occupied.                     | Parking-space ROI extraction, Sobel/Canny edge detection, and parameter evaluation            |
| [Face, eye, and mouth detection](./05-face-eye-and-mouth-detection)                                           | Detect faces and facial features in video frames.                             | Haar-cascade detection, image preprocessing, and heuristic feature analysis                   |
| [Parking occupancy using LBP classification](./06-parking-occupancy-using-LBP-classification)                 | Classify parking spaces using texture information.                            | Local Binary Patterns, feature extraction, and supervised classification                      |
| [Eye-state recognition using Haar detection and LBP](./07-eye-state-recognition-using-Haar-detection-and-LBP) | Recognize whether detected eyes are open or closed in video.                  | Face and eye localization, LBP features, KNN/SVM classification, and configuration comparison |
| [Traffic-sign classification using MLP and CNN](./08-traffic-sign-classification-using-MLP-and-CNN)           | Classify traffic signs from the GTSRB dataset.                                | Comparison of multilayer perceptron and convolutional neural-network models                   |

## Selected Projects

### Eye-State Recognition Using Haar Detection and LBP

This project combines video processing, face and eye localization, handcrafted texture features, and supervised classification. Multiple preprocessing and classification configurations are evaluated against labelled reference data.

[Open the project](./07-eye-state-recognition-using-Haar-detection-and-LBP)

### Parking Occupancy Classification

Two different solutions to the same practical problem are included:

* an approach based on edge detection and edge statistics;
* an approach based on LBP texture descriptors and machine-learning classifiers.

Together, the projects demonstrate the comparison of manually designed image-processing rules with a feature-based classification pipeline.

[Open the edge-based project](./04-parking-occupancy-using-edge-based-processing)

[Open the LBP project](./06-parking-occupancy-using-LBP-classification)

### Visual Game Bots

The Dart Master and Duck Hunt assignments connect image analysis with automated interaction. They use image templates to locate targets in a live visual stream and send the detected coordinates to an external control application.

[Open the Dart Master project](./03-game-bot-dart-master)

[Open the Duck Hunt project](./03-game-bot-duck-hunt)

## Technologies and Methods

### Languages and Libraries

* Python
* OpenCV
* NumPy
* scikit-learn
* PyTorch
* torchvision

### Image-Processing and Machine-Learning Methods

* Colour-space processing and thresholding
* Sobel and Canny edge detection
* Template matching
* Haar cascades
* Local Binary Patterns
* K-nearest neighbours
* Support-vector machines
* Multilayer perceptrons
* Convolutional neural networks

## Repository Organization

Each top-level folder is an independent academic assignment. The projects do not share one common execution environment, so the repository intentionally does not contain a single global `requirements.txt`.

Depending on the assignment, execution may require:

* project-specific Python packages;
* included images or videos;
* external datasets;
* Haar cascade files;
* trained model files;
* a local helper application for automated game interaction.

Detailed descriptions, source files, input data, and available results are stored inside the corresponding project folders.

## Academic Context

The assignments were completed for the course **Fundamentals of Image Processing** as part of the **Computational and Applied Mathematics** study programme at **VŠB – Technical University of Ostrava**.

The projects are presented as academic demonstrations of practical image-processing and computer-vision methods. They are not intended to be production-ready systems.
