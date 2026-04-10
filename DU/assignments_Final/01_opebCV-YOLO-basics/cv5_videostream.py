# This module is the live-stream processing pipeline of the whole CV5 task.
# It sits at the top of the streaming variant of the project and is responsible
# for taking one continuous video source and turning it into a live annotated
# detection stream.
#
# In the runtime flow, this module performs:
#
#     command-line configuration
#         -> YOLO model selection and initialization
#         -> live source opening
#         -> optional output-writer preparation
#         -> per-frame acquisition
#         -> class-filtered YOLO inference
#         -> safe box clamping
#         -> per-frame annotation
#         -> optional sampled crop export
#         -> optional FPS overlay
#         -> optional live display
#         -> optional annotated-video saving
#
# A key design idea in this module is that it is the "live-stream counterpart"
# of cv5_solution.py:
# - both modules use the same selected-class concept,
# - both modules run class-filtered YOLO inference,
# - both modules use box clamping before crop extraction,
# - both modules draw OpenCV annotations,
# - both modules report the total number of detections of the selected class.
#
# The difference is in the processing mode:
# - cv5_solution.py works on a fixed folder of input images and writes two
#   offline output folders,
# - this file works on a continuous frame source and keeps producing annotated
#   frames until the stream ends or the user stops the run.
#
# That makes this module the streaming orchestration layer of the project.

"""
# cv5_stream_wsl.py
# Live stream YOLO in WSL (Ubuntu under WSL): read frames from Windows MJPEG server URL,
# run YOLO, draw boxes + labels + per-frame count, show in an OpenCV window.

Purpose of this module:
- open one live video source,
- run YOLO on every incoming frame,
- keep only detections of one selected class,
- annotate the live frame with boxes, labels, and a per-frame detection count,
- optionally show the result in an OpenCV window,
- optionally save the annotated stream to a video file,
- optionally save sampled crops of detections.

Why this module exists:
The offline batch pipeline is useful for saved images, but many practical
computer-vision tasks need the same detection logic to run on a continuous
stream. This file reuses the same main concepts as the offline solution while
adapting them to live processing:
- frame acquisition replaces folder iteration,
- per-frame annotation replaces per-image annotation,
- continuous display replaces static output inspection,
- optional video writing replaces only image export.

The script is written with WSL usage in mind, where the video source is usually
a Windows-hosted MJPEG stream URL rather than a directly opened Linux camera
device.
"""

# argparse provides the command-line interface through which the whole live
# pipeline is configured.
import argparse

# time is used here mainly for FPS measurement and overlay.
import time

# Path is used only for optional crop-output directory handling.
from pathlib import Path

# OpenCV is responsible for:
# - opening the live video source,
# - drawing rectangles and text,
# - creating the optional display window,
# - reading keyboard input from the window,
# - writing optional video output,
# - saving optional crops.
import cv2 as cv

# YOLO from Ultralytics is the detector backend reused for every incoming frame.
from ultralytics import YOLO


# ---------------------------------------------------------------------
# Command-line configuration
# ---------------------------------------------------------------------
#
# This helper defines all runtime options of the streaming pipeline.
# The parser controls:
# - which object class should be tracked,
# - which model should be loaded,
# - which live source should be opened,
# - how annotations should look,
# - whether FPS should be shown,
# - whether a GUI window should be opened,
# - whether annotated video and sampled crops should be saved.
# ---------------------------------------------------------------------

def parse_args():
    """
    Parse and return all command-line arguments used by the live-stream
    pipeline.

    The returned namespace becomes the single runtime-configuration object
    consumed by main().
    """

    # Build the top-level parser describing the script as one complete
    # live-stream detection pipeline.
    p = argparse.ArgumentParser(
        description="CV5 (WSL): Live MJPEG stream -> YOLO -> draw boxes/labels/count -> display (and optional save)."
    )

    # -----------------------------------------------------------------
    # Core detector/model arguments
    # -----------------------------------------------------------------
    #
    # These options define the essential detection setup:
    # - selected class ID,
    # - default YOLOv8 model-size suffix,
    # - optional direct model override.
    p.add_argument("--class_id", type=int, required=True, help="Class ID to detect (e.g. 2 for car in COCO).")
    p.add_argument("--model_size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                   help="n/s/m/l/x -> yolov8{size}.pt (used if --model not given).")
    p.add_argument("--model", type=str, default=None,
                   help="Optional model file/path/name (overrides --model_size), e.g. yolov8s.pt.")

    # -----------------------------------------------------------------
    # Annotation and inference controls
    # -----------------------------------------------------------------
    #
    # These options mirror the offline solution so both modules stay aligned in
    # terminology and behavior.
    p.add_argument("--color", type=int, nargs=3, default=[0, 255, 0], metavar=("B", "G", "R"),
                   help="BGR color for boxes/text, e.g. --color 0 0 255 for red.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    p.add_argument("--line_thickness", type=int, default=2, help="Rectangle thickness.")
    p.add_argument("--verbose", action="store_true", help="Print extra info.")

    # -----------------------------------------------------------------
    # Live-source selection
    # -----------------------------------------------------------------
    #
    # In the streaming pipeline, folder traversal is replaced by one live source.
    # The default source is a Windows-hosted MJPEG stream URL suitable for WSL.
    p.add_argument("--source", type=str, default="http://127.0.0.1:8080/video",
                   help="WSL stream URL or video path. Default: http://127.0.0.1:8080/video")

    # -----------------------------------------------------------------
    # Display controls
    # -----------------------------------------------------------------
    #
    # These options control whether the script opens an OpenCV window, what the
    # window should be called, and whether FPS should be drawn into the frame.
    p.add_argument("--window_name", type=str, default="YOLO Live (WSL)", help="OpenCV window title.")
    p.add_argument("--show_fps", action="store_true", help="Overlay FPS in the top-left corner.")
    p.add_argument("--headless", action="store_true",
                   help="If set, do not open an OpenCV window (useful if GUI is unavailable).")

    # -----------------------------------------------------------------
    # Optional persistent outputs
    # -----------------------------------------------------------------
    #
    # The live pipeline can optionally persist:
    # - the full annotated stream as a video file,
    # - sampled crops of detected objects.
    #
    # These outputs are optional because live mode can easily generate a large
    # amount of data very quickly.
    p.add_argument("--save_video", type=str, default=None,
                   help="Optional output video file path (e.g. out.mp4). If set, writes annotated frames to a file.")

    p.add_argument("--save_crops_dir", type=str, default=None,
                   help="Optional directory to save crops (WARNING: can create many files).")
    p.add_argument("--save_every_n_frames", type=int, default=0,
                   help="Save crops every N frames (0 = never). Use e.g. 30 to save once per ~1s at 30 FPS.")
    p.add_argument("--max_crops_per_frame", type=int, default=50,
                   help="Safety limit: maximum crops saved per frame when saving is enabled.")

    # This note is intentionally left in the code as a project-structure marker:
    # the module can consume a stream and annotate it, but it does not yet
    # implement active restreaming to another endpoint.
    # MISSING/TODO: "redirect to another live stream" (RTSP/RTMP/WebRTC).
    # OpenCV does not natively publish RTSP/RTMP reliably in all builds.
    # If you need real restreaming, typically you pipe frames to FFmpeg or use GStreamer.

    return p.parse_args()


# ---------------------------------------------------------------------
# Bounding-box safety helper
# ---------------------------------------------------------------------
#
# The live pipeline uses the same box-clamping idea as the offline pipeline:
# YOLO box coordinates are forced back into the valid frame area before they
# are used for drawing or crop extraction.
# ---------------------------------------------------------------------

def clamp_box(x1, y1, x2, y2, w, h):
    """
    Clamp one bounding box into valid frame coordinates and ensure that the
    final box is non-empty.

    This helper keeps live per-frame processing safe near frame borders and
    makes optional crop extraction reliable.
    """

    # Clamp the top-left corner into valid pixel coordinates.
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))

    # Clamp the bottom-right corner into valid frame-end coordinates.
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))

    # If horizontal clamping collapsed the box, widen it minimally.
    if x2 <= x1:
        x2 = min(w, x1 + 1)

    # If vertical clamping collapsed the box, heighten it minimally.
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def class_name_for(names, cid: int) -> str:
    """
    Return a readable class name for one class ID.

    The helper supports both common Ultralytics class-name layouts:
    - dictionary mapping
    - list/tuple mapping

    If no mapping is available, the numeric class ID itself is returned as text.
    """

    # Handle dictionary-style class-name storage.
    if isinstance(names, dict) and cid in names:
        return str(names[cid])

    # Handle sequence-style class-name storage.
    if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
        return str(names[cid])

    # Fallback for unknown or unavailable mappings.
    return str(cid)


def main():
    """
    Run the full live-stream YOLO pipeline.

    High-level flow:
    1. parse runtime configuration,
    2. select and initialize the YOLO model,
    3. open the live source,
    4. prepare optional output objects,
    5. enter the continuous frame-processing loop,
    6. annotate each frame,
    7. optionally display and/or save the result,
    8. release all resources cleanly at the end.

    This function is the orchestration layer of the streaming solution. It
    keeps the same selected-class and annotation logic as cv5_solution.py, but
    adapts it to continuous live processing.
    """

    # -------------------------------------------------------------
    # Step 1: read runtime configuration
    # -------------------------------------------------------------
    args = parse_args()

    # -------------------------------------------------------------
    # Step 2: choose and initialize the YOLO model
    # -------------------------------------------------------------
    #
    # Model selection follows the same logic as the offline pipeline:
    # - --model has highest priority,
    # - otherwise a standard yolov8{size}.pt name is built.
    model_name = args.model if args.model else f"yolov8{args.model_size}.pt"
    model = YOLO(model_name)

    # Store the model's class-name mapping if available so annotation labels can
    # use readable names instead of only numeric IDs.
    names = getattr(model, "names", None)

    # Extract a few frequently reused runtime values once before the frame loop.
    class_id = int(args.class_id)
    cls_name = class_name_for(names, class_id)
    box_color = tuple(args.color)  # OpenCV uses BGR

    # -------------------------------------------------------------
    # Step 3: open the live source
    # -------------------------------------------------------------
    #
    # In this module, the input is one continuous live source rather than a
    # folder of files.
    source = args.source
    cap = cv.VideoCapture(source)

    # A failed capture open is treated as a hard stop because the whole
    # streaming pipeline depends on an active frame source.
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open video source: {source}\n\n"
            "WSL hint: Make sure the Windows MJPEG server is running and reachable.\n"
            "- Windows browser check: http://127.0.0.1:8080/\n"
            "- WSL quick check: curl -s http://127.0.0.1:8080/ | head\n"
            "- In WSL: use VideoCapture('http://127.0.0.1:8080/video') (NOT VideoCapture(0))."
        )

    # Verbose mode prints the active streaming configuration before the live
    # loop begins.
    if args.verbose:
        print(f"Model: {model_name}")
        print(f"Source: {source}")
        print(f"Class ID: {class_id} ({cls_name})")
        print(f"conf={args.conf}, iou={args.iou}, color(BGR)={box_color}")
        if args.save_video:
            print(f"Saving video to: {args.save_video}")
        if args.save_crops_dir:
            print(f"Saving crops to: {args.save_crops_dir} every {args.save_every_n_frames} frames")
        print("Quit: press 'q' (or ESC if window is shown).")

    # -------------------------------------------------------------
    # Step 4: prepare optional persistent outputs
    # -------------------------------------------------------------
    #
    # The optional video writer is not created immediately because the frame
    # size is not known until the first frame is read.
    writer = None
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    # Optional crop-output directory is created only when crop saving is
    # explicitly enabled.
    crops_dir = None
    if args.save_crops_dir:
        crops_dir = Path(args.save_crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 5: initialize runtime bookkeeping
    # -------------------------------------------------------------
    #
    # These values are reused during the live loop:
    # - FPS state for optional overlay,
    # - frame index for crop sampling,
    # - optional GUI window.
    last_t = time.time()
    fps = 0.0
    frame_idx = 0

    if not args.headless:
        cv.namedWindow(args.window_name, cv.WINDOW_NORMAL)

    # -------------------------------------------------------------
    # Step 6: continuous frame-processing loop
    # -------------------------------------------------------------
    #
    # The loop continues until:
    # - the stream ends,
    # - frame acquisition fails,
    # - or the user stops the run from the GUI window.
    while True:
        # Read the next frame from the live source.
        ok, frame = cap.read()
        if not ok or frame is None:
            if args.verbose:
                print("Frame grab failed or end of stream.")
            break

        # Update frame-level bookkeeping for this iteration.
        frame_idx += 1
        h, w = frame.shape[:2]

        # Work on a copy so the original frame remains untouched for optional
        # crop extraction.
        annotated = frame.copy()

        # ---------------------------------------------------------
        # Step 6A: run class-filtered YOLO inference
        # ---------------------------------------------------------
        #
        # The selected class is passed directly into YOLO inference so only the
        # detections relevant to the current run are returned.
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            classes=[class_id],
            verbose=False,
        )

        # The pipeline expects one result object for the current frame.
        r0 = results[0]
        boxes = r0.boxes
        det_count = 0

        # ---------------------------------------------------------
        # Step 6B: process all detections in the current frame
        # ---------------------------------------------------------
        #
        # For each kept detection, the live pipeline performs:
        # - safe box clamping,
        # - full-frame annotation,
        # - optional sampled crop saving.
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                det_count += 1

                # Read the raw YOLO box and clamp it into valid frame
                # coordinates before drawing or slicing.
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

                # Read the confidence score once because it is needed for both
                # text annotation and optional crop naming.
                conf = float(boxes.conf[i].item())

                # Draw the bounding rectangle into the annotation frame.
                cv.rectangle(annotated, (x1, y1), (x2, y2), box_color, args.line_thickness)

                # Draw a per-box label containing the readable class name and
                # the detection confidence.
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

                # -------------------------------------------------
                # Optional sampled crop export
                # -------------------------------------------------
                #
                # Live mode can generate a very large number of detections, so
                # crop saving is guarded by:
                # - explicit crop-saving enablement,
                # - frame sampling via save_every_n_frames,
                # - a per-frame safety cap.
                if crops_dir and args.save_every_n_frames > 0:
                    if (frame_idx % args.save_every_n_frames) == 0 and det_count <= args.max_crops_per_frame:
                        crop = frame[y1:y2, x1:x2].copy()
                        crop_name = f"frame{frame_idx:06d}_det{det_count:03d}_cls{class_id}_c{int(conf*1000)}.png"
                        cv.imwrite(str(crops_dir / crop_name), crop)

        # ---------------------------------------------------------
        # Step 6C: draw per-frame total-count overlay
        # ---------------------------------------------------------
        #
        # This summary text reports how many detections of the selected class
        # were found in the current frame.
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
        # Step 6D: optionally draw FPS
        # ---------------------------------------------------------
        #
        # FPS is measured from the time difference between consecutive processed
        # frames and is shown only when requested.
        if args.show_fps:
            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps = 1.0 / dt
            last_t = now
            cv.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                lineType=cv.LINE_AA,
            )

        # ---------------------------------------------------------
        # Step 6E: lazily initialize the optional video writer
        # ---------------------------------------------------------
        #
        # The writer is created only after the first frame is known, because the
        # output resolution must match the actual stream resolution.
        #
        # The source FPS may not always be reported reliably, so a fallback is
        # used when necessary.
        if args.save_video and writer is None:
            src_fps = cap.get(cv.CAP_PROP_FPS)
            if not src_fps or src_fps <= 1e-3:
                src_fps = 30.0  # fallback
            writer = cv.VideoWriter(args.save_video, fourcc, float(src_fps), (w, h))

        # If video saving is enabled and the writer exists, persist the current
        # annotated frame.
        if writer is not None:
            writer.write(annotated)

        # ---------------------------------------------------------
        # Step 6F: display or headless execution
        # ---------------------------------------------------------
        #
        # In GUI mode, the frame is shown and the user can stop the run with
        # ESC or q.
        #
        # In headless mode, the pipeline keeps running without opening a window.
        if not args.headless:
            cv.imshow(args.window_name, annotated)
            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC or q
                break
        else:
            # Headless mode does not poll keyboard input because no OpenCV
            # window exists in that mode.
            pass

        # This TODO stays as a clear boundary marker of the current module:
        # it can annotate and optionally save a live stream, but it does not yet
        # publish the annotated stream outward as another live endpoint.
        # MISSING/TODO: Redirect annotated frames into another live stream endpoint (RTSP/RTMP).
        # Typical solutions:
        # - Pipe annotated frames to FFmpeg subprocess (stdin) and publish RTSP/RTMP
        # - Use GStreamer pipeline if your OpenCV build supports it

    # -------------------------------------------------------------
    # Step 7: release resources cleanly
    # -------------------------------------------------------------
    #
    # Live pipelines must always clean up capture, writer, and window resources
    # when the run ends.
    cap.release()
    if writer is not None:
        writer.release()
    if not args.headless:
        cv.destroyAllWindows()

    # Final verbose-mode completion message.
    if args.verbose:
        print("Done.")


# Standard Python entry-point guard.
# This ensures the live pipeline starts only when the file is executed
# directly.
if __name__ == "__main__":
    main()