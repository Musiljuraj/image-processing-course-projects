# This module is the offline image-processing pipeline of the whole CV5 task.
# It sits at the top of the batch-processing variant of the project and is
# responsible for turning one folder of input images into two output products:
# - extracted crops of one selected object class,
# - annotated full images with bounding boxes, labels, and per-image counts.
#
# In the runtime flow, this module performs:
#
#     command-line configuration
#         -> YOLO model selection and initialization
#         -> input-folder validation
#         -> output-folder preparation
#         -> image-file discovery
#         -> per-image inference with class filtering
#         -> safe box clamping
#         -> crop extraction and saving
#         -> OpenCV annotation of the full image
#         -> export of the annotated image
#
# A key design idea in this module is that it is an "offline batch pipeline":
# - it processes already saved image files,
# - it does not work with a live camera or stream,
# - it runs the same detection logic image by image in a deterministic folder
#   traversal order,
# - it produces persistent output artifacts on disk.
#
# This makes it the static-image counterpart of the live-stream module
# cv5_videostream.py. Both modules use the same general terminology:
# - selected class
# - YOLO inference
# - box clamping
# - crop extraction
# - OpenCV annotation
# but this file applies that logic to a folder of images instead of a frame
# stream.

"""
PŘÍPRAVA PROSTŘEDÍ:
* Instalace balíku Ultralytics: 'pip install ultralytics' 
* https://github.com/ultralytics/ultralytics
* https://docs.ultralytics.com/modes/predict/#inference-arguments

Zadání na cvičení:

1. Inicializace detekčního modelu YOLO.
2. Načtení všech obrazových souborů z adresáře "bmw_100" a spuštění modelu YOLO na těchto souborech.
3. Implementace parametrů příkazové řádky: 
   - pro definici ID detekované třídy (např. 2 pro automobily).
   - pro velikost modelu
   - pro nazev vstupniho adresare
   - pro nazev vystupniho adresare
4. Uložení extrahovaných výřezů objektů do určené složky (např. 'car').
5. Vykreslení ohraničujících rámečků kolem objektu do původních obrazů (využití OpenCV). 
   Pokuste se o vykreslení ohraničujících rámečků (bounding boxes), které vrátí YOLO model pomocí funkce cv2.rectangle() případně informací o objektech + cv2.putText()
   - možnost nastavit barvu jako parametr příkazové řádky 
6. Vložení textové informace o celkovém počtu detekcí dané třídy do obrazu (např. levý dolní roh).

* ukázka spuštění s definicí barvy a specifické třídy:
python cv5_zadani.py --class_id 0 --color 0 0 255 --output_dir persons

0	osoba
1	jízdní kolo
2	osobní automobil
3	motocykl
5	autobus

Purpose of this module:
- run YOLO on every supported image in one input folder,
- keep only detections of one selected class,
- save extracted object crops into a dedicated output subfolder,
- save full annotated images into a separate output subfolder,
- make the whole workflow configurable from command-line arguments.

Why this module exists:
The exercise is not only about calling a pretrained detector once. It is about
building one complete processing pipeline around YOLO:
- configuration comes from CLI arguments,
- images are loaded from disk in bulk,
- detections are filtered to one task-relevant class,
- detections are converted into reusable outputs,
- OpenCV is used for human-readable visualization.

That makes this file the main orchestration layer of the offline solution.
"""

# OpenCV is used here for all image I/O and visualization work:
# - loading input images,
# - drawing rectangles,
# - drawing text labels,
# - saving crops,
# - saving annotated output images.
import cv2 as cv

# YOLO from Ultralytics is the detector backend used by the whole pipeline.
# The module builds one model instance and then reuses it for all images.
from ultralytics import YOLO

# argparse provides the command-line interface through which the whole pipeline
# is configured.
import argparse

# os is currently imported as part of the script environment, even though the
# core path handling in this file is done through pathlib.Path.
import os 

# Path is the main path-handling abstraction used in the pipeline:
# - validating the input folder,
# - iterating through image files,
# - creating output directories,
# - constructing crop/output filenames.
from pathlib import Path


# ---------------------------------------------------------------------
# Supported input-image extensions
# ---------------------------------------------------------------------
#
# This set defines which files in the input directory will be treated as images.
# Filtering by extension keeps folder iteration simple and prevents the pipeline
# from trying to process unrelated files.
# ---------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------
# Command-line configuration
# ---------------------------------------------------------------------
#
# This helper collects all runtime options for the batch pipeline.
# The argument parser is the main interface through which the user controls:
# - which class should be extracted,
# - which YOLO model should be used,
# - where input images are located,
# - where outputs should be stored,
# - how boxes should be drawn,
# - and how strict inference should be.
# ---------------------------------------------------------------------

# parse command-line parameters
def parse_args():
    """
    Parse and return all command-line arguments used by the offline pipeline.

    The parser defines:
    - assignment-required arguments,
    - a few optional quality-of-life overrides,
    - annotation controls,
    - inference-threshold controls,
    - and a verbose mode for easier inspection.

    The returned namespace becomes the single runtime-configuration object used
    by main().
    """

    # Build the top-level parser describing the purpose of this script as one
    # complete folder-processing YOLO pipeline.
    p = argparse.ArgumentParser(
        description="CV5: Run YOLO on images in a folder, save class-specific crops and OpenCV-annotated images."
    )

    # -----------------------------------------------------------------
    # Required assignment-facing arguments
    # -----------------------------------------------------------------
    #
    # These options directly reflect the exercise specification:
    # - choose one target class,
    # - choose one model size,
    # - choose input and output directories.
    p.add_argument("--class_id", type=int, required=True, help="Class ID to extract (e.g. 2 for car in COCO).")
    p.add_argument("--model_size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                   help="Model size: n/s/m/l/x -> yolov8{size}.pt")
    p.add_argument("--input_dir", type=str, default="bmw_100", help="Input directory with images (default: bmw_100).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Base output directory (e.g. 'car' or 'persons').")

    # -----------------------------------------------------------------
    # Optional overrides and extra controls
    # -----------------------------------------------------------------
    #
    # These options make the script more flexible without changing the core
    # assignment logic.
    #
    # --model:
    #     allows direct explicit model-path selection and overrides model_size
    # --color:
    #     controls OpenCV annotation color in BGR order
    # --conf / --iou:
    #     control YOLO inference strictness
    # --line_thickness:
    #     controls box thickness in annotated outputs
    # --verbose:
    #     prints a compact summary of the active runtime configuration
    p.add_argument("--model", type=str, default=None,
                   help="Optional model file/path/name (overrides --model_size), e.g. yolov8s.pt or yolov10n.pt.")
    p.add_argument("--color", type=int, nargs=3, default=[0, 255, 0], metavar=("B", "G", "R"),
                   help="BGR color for boxes, e.g. --color 0 0 255 for red.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    p.add_argument("--line_thickness", type=int, default=2, help="Rectangle thickness.") 
    p.add_argument("--verbose", action="store_true", help="If set, print extra info.")

    # Parse once and return the ready-to-use namespace.
    return p.parse_args()


# ---------------------------------------------------------------------
# Bounding-box safety helper
# ---------------------------------------------------------------------
#
# YOLO detections are typically valid, but detection coordinates can still end
# up at or slightly beyond image boundaries. This helper keeps every later crop
# operation safe by forcing the box back into the valid image area.
# ---------------------------------------------------------------------

# ensure that bounding boxes will stay inside the image
# YOLO output coordinates may be slightly outside the image, this clamps them into valid pixel coordinates so cropping never crashes.
def clamp_box(x1, y1, x2, y2, w, h):
    """
    Clamp one bounding box into valid image coordinates.

    Inputs:
    - x1, y1, x2, y2:
        raw YOLO box coordinates
    - w, h:
        image width and height

    Behavior:
    - convert the coordinates to integers,
    - clamp them into the valid image extent,
    - ensure the resulting box is never empty.

    Why this helper exists:
    crop extraction later assumes that the final box can be safely used for
    NumPy slicing. This helper guarantees that assumption.
    """

    # Clamp the top-left corner into valid pixel coordinates.
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))

    # Clamp the bottom-right corner into valid image-end coordinates.
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    # If clamping collapsed the box horizontally, widen it minimally so later
    # slicing still produces a non-empty crop.
    if x2 <= x1:
        x2 = min(w, x1 + 1)

    # If clamping collapsed the box vertically, raise its height minimally for
    # the same reason.
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


# ---------------------------------------------------------------------
# Input-image discovery
# ---------------------------------------------------------------------
#
# This helper scans the input directory and keeps only supported image files.
# Sorting is used so the offline batch run processes files in a stable order.
# ---------------------------------------------------------------------

# load all the images from input_dir defined as CL arg
def list_images(input_dir: Path):
    """
    Return all supported image files from the input directory.

    Inputs:
    - input_dir:
        directory expected to contain the source images

    Return:
    - list of Path objects for supported image files only

    Why sorting matters:
    The pipeline is an offline batch process, so stable ordering makes the run
    easier to inspect and reproduce.
    """

    # Collect all supported image files into one explicit list rather than
    # processing directory entries on the fly.
    files = []

    # Iterate through the directory in sorted order and keep only real files
    # whose suffix matches one of the supported image extensions.
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)

    return files


def main():
    """
    Run the full offline YOLO image-folder pipeline.

    High-level flow:
    1. parse runtime configuration from CLI arguments,
    2. validate the input directory,
    3. create output directories,
    4. choose and initialize the YOLO model,
    5. load the list of input images,
    6. process each image independently,
    7. save crops and annotated outputs,
    8. print the final output locations.

    This function is the orchestration layer of the module:
    it does not implement the detector itself, but it connects all helper steps
    into one complete batch-processing workflow.
    """

    # -------------------------------------------------------------
    # Step 1: read runtime configuration
    # -------------------------------------------------------------
    #
    # All later behavior in the pipeline depends on the parsed CLI arguments,
    # so argument parsing is the first step in the run.
    args = parse_args()

    # -------------------------------------------------------------
    # Step 2: validate the input directory
    # -------------------------------------------------------------
    #
    # The batch pipeline works only on a real directory of input images.
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")
    
    # -------------------------------------------------------------
    # Step 3: prepare the output directory structure
    # -------------------------------------------------------------
    #
    # The pipeline intentionally separates its outputs into two subfolders:
    # - crops:
    #     extracted object-only cutouts
    # - annotated:
    #     full original images with visualized detections
    #
    # This keeps the two output products logically distinct and avoids writing
    # directly into the input directory.
    base_out = Path(args.output_dir)
    crops_dir = base_out / "crops"
    ann_dir = base_out / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 4: choose and initialize the YOLO model
    # -------------------------------------------------------------
    #
    # Model selection follows a two-level policy:
    # - if --model is given, it overrides everything and is used directly,
    # - otherwise the script builds a standard YOLOv8 weight name from the
    #   requested model-size suffix.
    #
    # The model is created once and then reused for all images in the batch.
     # Model selection:
    # - default uses YOLOv8 by filename template yolov8{size}.pt
    # - --model overrides and can point to any supported pretrained weights
    model_name = args.model if args.model else f"yolov8{args.model_size}.pt"
    model = YOLO(model_name)

    # Store the class-name mapping from the model if available.
    # This later allows annotations to display readable class names instead of
    # only numeric class IDs.
    # Class names mapping (for labels)
    names = getattr(model, "names", None)

    # -------------------------------------------------------------
    # Step 5: discover input images
    # -------------------------------------------------------------
    #
    # The whole folder is scanned once before processing starts.
    # An empty folder is treated as a hard stop because there is nothing to run
    # inference on.
    # get list of all images 
    files = list_images(input_dir)
    if not files:
        raise SystemExit(f"No image files found in {input_dir} (supported: {sorted(IMAGE_EXTS)})")

    # -------------------------------------------------------------
    # Step 6: extract a few frequently reused runtime values
    # -------------------------------------------------------------
    #
    # These values are read once from the CLI configuration and then reused
    # throughout the per-image processing loop.
    # colour of the boxes and choosen class for identification
    box_color = tuple(args.color)  # OpenCV uses BGR
    class_id = int(args.class_id)

    # Verbose mode prints the active batch-configuration summary before the
    # heavy processing loop starts.
    if args.verbose:
        print(f"Model: {model_name}")
        print(f"Input: {input_dir} ({len(files)} images)")
        print(f"Output: {base_out}")
        print(f"Class ID: {class_id}")
        print(f"Box color (BGR): {box_color}")
        print(f"conf={args.conf}, iou={args.iou}")
    
    # -------------------------------------------------------------
    # Local helper: readable class-name lookup
    # -------------------------------------------------------------
    #
    # YOLO models may store their class mapping either as:
    # - a dictionary,
    # - a list/tuple,
    # - or not at all.
    #
    # This helper hides that variation and always returns a readable string.
    # Helper to get a readable class name for the chosen class_id
    def class_name_for(cid: int) -> str:
        if isinstance(names, dict) and cid in names:
            return str(names[cid])
        if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
            return str(names[cid])
        return str(cid)

    # Resolve the selected class name once so it can be reused for every
    # detection label in the processing loop.
    cls_name = class_name_for(class_id)

    # -------------------------------------------------------------
    # Step 7: process every input image independently
    # -------------------------------------------------------------
    #
    # The offline batch pipeline now walks through the discovered image list and
    # runs the full detection/export/annotation workflow on each image.
    # process each image
    for img_path in files:
        # Load the input image in color mode.
        # A failed read is handled non-fatally so one broken file does not stop
        # the whole batch run.
        img = cv.imread(str(img_path), cv.IMREAD_COLOR)
        if img is None:
            print(f"WARNING: Could not read image: {img_path}")
            continue

        # Read the image dimensions once because they are needed later for:
        # - safe box clamping,
        # - placement of the bottom-left count overlay.
        h, w = img.shape[:2]

        # Work on a copy for annotation so the originally loaded image remains
        # unchanged for crop extraction.
        annotated = img.copy()

        # ---------------------------------------------------------
        # Step 7A: run YOLO inference with direct class filtering
        # ---------------------------------------------------------
        #
        # The selected class is passed directly into YOLO inference using
        # classes=[class_id]. That means the returned detections are already
        # restricted to the class relevant for this run.
        # Run YOLO inference,  class filtering is realized directly in inference "classes = [class_id]"
        results = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            classes=[class_id],
            verbose=False,
        )

        # The current pipeline expects one result object per processed image, so
        # it takes the first result entry.
        r0 = results[0]
        boxes = r0.boxes
        det_count = 0

        # ---------------------------------------------------------
        # Step 7B: process all detections in the current image
        # ---------------------------------------------------------
        #
        # For every kept detection, the pipeline performs three connected tasks:
        # - clamp coordinates safely,
        # - extract and save the crop,
        # - draw annotation into the full-image copy.
        # processing all detections (boxes) in one image
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                det_count += 1

                # Read the raw YOLO box and clamp it so every later slice stays
                # valid even near image borders.
                # clamp (safety guard, to stay inside the image)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

                # -------------------------------------------------
                # Crop extraction and saving
                # -------------------------------------------------
                #
                # The crop is cut from the original full image, not from the
                # annotated copy. The filename stores:
                # - original image stem,
                # - detection index in the image,
                # - selected class ID,
                # - confidence encoded as an integer.
                # crop and save (with important info's in img's name)
                crop = img[y1:y2, x1:x2].copy()
                conf = float(boxes.conf[i].item())
                stem = img_path.stem
                crop_name = f"{stem}_det{det_count:03d}_cls{class_id}_c{int(conf * 1000)}.png"
                cv.imwrite(str(crops_dir / crop_name), crop)

                # -------------------------------------------------
                # Full-image annotation
                # -------------------------------------------------
                #
                # The annotation copy gets:
                # - one rectangle around the detected object,
                # - one per-box label with class name and confidence.
                # Draw bounding box into "annotated" copy of an image
                cv.rectangle(annotated, (x1, y1), (x2, y2), box_color, args.line_thickness)

                # Draw per-box label (class name + confidence)
                label = f"{cls_name} {conf:.2f}"
                y_text = y1 - 8 if y1 - 8 > 10 else y1 + 20
                cv.putText(
                    annotated,
                    label,
                    (x1, y_text),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                    lineType=cv.LINE_AA,
                    )
        
        # ---------------------------------------------------------
        # Step 7C: write the per-image total-count overlay
        # ---------------------------------------------------------
        #
        # This summary text is drawn once per processed image and reports how
        # many detections of the selected class were found in that image.
        # Total count - insert information about total count of detections of objects of chosen class (bottom-left)
        cv.putText(
            annotated,
            f"Detections of class {class_id}: {det_count}",
            (10, h - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv.LINE_AA,
        )

        # ---------------------------------------------------------
        # Step 7D: save the annotated output image
        # ---------------------------------------------------------
        #
        # The full annotated image is written into the dedicated annotated
        # output directory with a predictable suffix-based filename.
        #
        # A compact console line is printed for each input image so the batch
        # progress remains visible during the run.
        # save annotated copy for each image, print info about ach image
        out_name = f"{img_path.stem}_annotated.jpg"
        cv.imwrite(str(ann_dir / out_name), annotated)
        print(f"{img_path.name}: {det_count} detections (class {class_id})")

    # -------------------------------------------------------------
    # Step 8: final run summary
    # -------------------------------------------------------------
    #
    # Print the resolved output paths after the batch finishes so the caller can
    # immediately see where both output products were stored.
    print("\nDone.")
    print("Crops:", crops_dir.resolve())
    print("Annotated:", ann_dir.resolve())


# Standard Python entry-point guard.
# This ensures the pipeline runs only when the file is executed directly.
if __name__ == "__main__":
    main()