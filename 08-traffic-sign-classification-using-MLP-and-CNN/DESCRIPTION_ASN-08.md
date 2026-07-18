PROJECT SUMMARY – GTSRB TRAFFIC-SIGN CLASSIFICATION (MLP vs CNN)



**Purpose:**

This project is a complete PyTorch experiment for German Traffic Sign Recognition Benchmark (GTSRB) classification. Its goal is not only to classify 43 traffic-sign classes, but mainly to compare two baseline neural-network architectures under the same conditions: a simple MLP (fully connected network) and a CNN (convolutional network). The project measures and compares final training loss, test accuracy, and training time for both models.



**Mechanism:**

The experiment starts in **main.py**. First, it selects device automatically (CUDA GPU if available, otherwise CPU). Then it creates one shared data pipeline from **data\_pipeline.py** so both models use exactly the same input source and preprocessing. The GTSRB train and test splits are loaded through torchvision; each image is resized to 32x32 and converted to a tensor. These datasets are wrapped into DataLoaders (train shuffled, test not shuffled, batch size 32).

After that, **main.py creates one shared loss function: CrossEntropyLoss.** Then it **trains the MLP model**, measures its training time, **evaluates** it on the test set, and prints its final metrics. Next, it repeats **the same procedure for the CNN model** with the same data, same optimizer type (Adam), same learning rate (0.001), same epoch count (10), and same evaluation procedure. 

Finally, it prints a direct comparison of both models.



**Call stack / project flow:**

python main.py -> if \_\_name\_\_ == "\_\_main\_\_": main() in main.py

\-> main() selects device and calls get\_data\_loaders(batch\_size=32) from data\_pipeline.py

\-> get\_data\_loaders() calls get\_datasets()

\-> get\_datasets() calls get\_transforms()

\-> get\_transforms() returns Resize((32,32)) + ToTensor preprocessing

\-> get\_datasets() creates GTSRB train/test datasets (downloaded automatically to ./data if missing)

\-> get\_data\_loaders() returns train\_loader and test\_loader

\-> main() creates CrossEntropyLoss

\-> main() creates MLPClassifier() from models.py and Adam optimizer

\-> main() calls train\_model(...) from train\_eval.py

\-> train\_model() moves model to device, loops over epochs and batches, runs forward pass, computes loss, backpropagates, updates weights, and returns list of average epoch losses

\-> model forward for MLPClassifier: Flatten -> Linear(3072,256) -> ReLU -> Linear(256,43)

\-> main() calls evaluate\_model(...)

\-> evaluate\_model() runs test forward passes without gradients, uses argmax over logits, counts correct predictions, and returns accuracy in %

\-> main() repeats the same for CNNClassifier()

\-> model forward for CNNClassifier: Conv(3->16, 3x3) -> ReLU -> MaxPool -> Conv(16->32, 3x3) -> ReLU -> MaxPool -> Flatten -> Linear(2048,128) -> ReLU -> Linear(128,43)

\-> main() prints final side-by-side comparison and brief interpretation.

Auxiliary functions in main.py: synchronize\_if\_needed(device) ensures correct CUDA timing; format\_seconds(seconds) formats printed time values.



**Modules / functions:**

\- **main.py**: top-level orchestration; controls the whole experiment, timing, evaluation, and final comparison.

\- main(): complete run procedure for both models.

\- synchronize\_if\_needed(): CUDA timing helper.

\- format\_seconds(): formatting helper for printed training times.

\- **data\_pipeline.py**: shared input layer for both compared models.

\- get\_transforms(): defines preprocessing (resize to 32x32, convert to tensor).

\- get\_datasets(): creates GTSRB train/test datasets from ./data, with automatic download if missing.

\- get\_data\_loaders(): wraps datasets into DataLoaders.

\- **models.py**: architecture definitions only.

\- MLPClassifier: dense baseline for flattened images.

\- CNNClassifier: convolutional baseline preserving local spatial structure.

\- **train\_eval.py**: shared optimization/evaluation logic reused by both models.

\- train\_model(): training loop, returns average loss for each epoch.

\- evaluate\_model(): test-set evaluation loop, returns classification accuracy.



**Input:**

The required input is the GTSRB dataset: RGB traffic-sign images belonging to 43 classes. The project expects the dataset under ./data, but in the current version the code uses download=True, so if the dataset is not present it is downloaded automatically there. Therefore you need either:

1\) internet access on first run, or

2\) an already prepared GTSRB dataset stored under ./data in the standard torchvision structure.

No manual image-by-image input folder is used in this project.



**Output:**

The main output is console text only. The project prints:

\- selected device,

\- dataset sizes,

\- batch size / epochs / learning rate,

\- per-epoch average training loss,

\- final MLP loss, accuracy, and training time,

\- final CNN loss, accuracy, and training time,

\- final comparison summary (better accuracy, lower loss, faster training, metric differences),

\- short interpretation.

The project does NOT save trained models, plots, prediction files, or annotated images. The only filesystem side effect besides console output is possible automatic download/storage of the GTSRB dataset into ./data.



**How to run:**

Current code has no CLI arguments; the run command is simply:

python main.py

or:

python3 main.py



Initial check / run procedure:

1\. Make sure these files are in the same working directory: main.py, data\_pipeline.py, models.py, train\_eval.py.

2\. Make sure Python environment contains PyTorch and torchvision.

3\. Check that ./data is writable.

4\. If GTSRB is not already in ./data, ensure internet access for the first run; otherwise place the standard GTSRB dataset there beforehand.

5\. Run: python main.py

6\. The script automatically loads/downloads GTSRB, preprocesses all images to 32x32 tensors, trains MLP for 10 epochs, evaluates it, then trains CNN for 10 epochs, evaluates it, and finally prints a direct comparison of both models. If you want different batch size / epoch count / learning rate / data path, they must be changed in source constants (main.py and data\_pipeline.py), because the current version does not expose them as command-line arguments.

