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
"""

import cv2 as cv
from ultralytics import YOLO

import argparse #parsing command-line arguments
import os 
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

#parse command-line parameters
def parse_args():
    p = argparse.ArgumentParser(
        description="CV5: Run YOLO on images in a folder, save class-specific crops and OpenCV-annotated images."
    )

    # Required by assignment
    p.add_argument("--class_id", type=int, required=True, help="Class ID to extract (e.g. 2 for car in COCO).")
    p.add_argument("--model_size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                   help="Model size: n/s/m/l/x -> yolov8{size}.pt")
    p.add_argument("--input_dir", type=str, default="bmw_100", help="Input directory with images (default: bmw_100).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Base output directory (e.g. 'car' or 'persons').")

    # Optional overrides / extra controls
    p.add_argument("--model", type=str, default=None,
                   help="Optional model file/path/name (overrides --model_size), e.g. yolov8s.pt or yolov10n.pt.")
    p.add_argument("--color", type=int, nargs=3, default=[0, 255, 0], metavar=("B", "G", "R"),
                   help="BGR color for boxes, e.g. --color 0 0 255 for red.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    p.add_argument("--line_thickness", type=int, default=2, help="Rectangle thickness.") 
    p.add_argument("--verbose", action="store_true", help="If set, print extra info.")
    return p.parse_args()

#ensure that bounding boxes will stay inside the image
#YOLO output coordinates may be slightly outside the image, this clamps them into valid pixel coordinates so cropping never crashes.
def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


#load all the images from input_dir defined as CL arg
def list_images(input_dir: Path):
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def main():
    #read command-line args (parameters)
    args = parse_args()

    #extract input_dir, (folder with input images)
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")
    
    # Create output folders for "crops" and "annotated" images (I choose not to draw into original images in input_dir)
    base_out = Path(args.output_dir)
    crops_dir = base_out / "crops"
    ann_dir = base_out / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

     # Model selection:
    # - default uses YOLOv8 by filename template yolov8{size}.pt
    # - --model overrides and can point to any supported pretrained weights
    model_name = args.model if args.model else f"yolov8{args.model_size}.pt"
    model = YOLO(model_name)

    # Class names mapping (for labels)
    names = getattr(model, "names", None)

    #get list of all images 
    files = list_images(input_dir)
    if not files:
        raise SystemExit(f"No image files found in {input_dir} (supported: {sorted(IMAGE_EXTS)})")

    #colour of the boxes and choosen class for identification
    box_color = tuple(args.color)  # OpenCV uses BGR
    class_id = int(args.class_id)
    if args.verbose:
        print(f"Model: {model_name}")
        print(f"Input: {input_dir} ({len(files)} images)")
        print(f"Output: {base_out}")
        print(f"Class ID: {class_id}")
        print(f"Box color (BGR): {box_color}")
        print(f"conf={args.conf}, iou={args.iou}")
    
    # Helper to get a readable class name for the chosen class_id
    def class_name_for(cid: int) -> str:
        if isinstance(names, dict) and cid in names:
            return str(names[cid])
        if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
            return str(names[cid])
        return str(cid)

    cls_name = class_name_for(class_id)


    #process each image
    for img_path in files:
        img = cv.imread(str(img_path), cv.IMREAD_COLOR)
        if img is None:
            print(f"WARNING: Could not read image: {img_path}")
            continue

        h, w = img.shape[:2]
        annotated = img.copy()

        # Run YOLO inference,  class filtering is realized directly in inference "classes = [class_id]"
        results = model.predict(
            source=img,
            conf=args.conf,
            iou=args.iou,
            classes=[class_id],
            verbose=False,
        )

        r0 = results[0]
        boxes = r0.boxes
        det_count = 0

        #processing all detections (boxes) in one image
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                det_count += 1

                #clamp (safety guard, to stay inside the image)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

                #crop and save (with important info's in img's name)
                crop = img[y1:y2, x1:x2].copy()
                conf = float(boxes.conf[i].item())
                stem = img_path.stem
                crop_name = f"{stem}_det{det_count:03d}_cls{class_id}_c{int(conf * 1000)}.png"
                cv.imwrite(str(crops_dir / crop_name), crop)

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
        
        # Total count - insert information bout total count of detections of objects of chosen class (bottom-left)
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

        #save annotated copy for each image, print info about ach image
        out_name = f"{img_path.stem}_annotated.jpg"
        cv.imwrite(str(ann_dir / out_name), annotated)
        print(f"{img_path.name}: {det_count} detections (class {class_id})")


    print("\nDone.")
    print("Crops:", crops_dir.resolve())
    print("Annotated:", ann_dir.resolve())


if __name__ == "__main__":
    main()