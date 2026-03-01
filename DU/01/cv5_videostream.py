# cv5_stream_wsl.py
# Live stream YOLO in WSL (Ubuntu under WSL): read frames from Windows MJPEG server URL,
# run YOLO, draw boxes + labels + per-frame count, show in an OpenCV window.

import argparse
import time
from pathlib import Path

import cv2 as cv
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(
        description="CV5 (WSL): Live MJPEG stream -> YOLO -> draw boxes/labels/count -> display (and optional save)."
    )

    # SAME AS YOUR CURRENT SCRIPT (kept) :contentReference[oaicite:3]{index=3} (shown outside code)
    p.add_argument("--class_id", type=int, required=True, help="Class ID to detect (e.g. 2 for car in COCO).")
    p.add_argument("--model_size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                   help="n/s/m/l/x -> yolov8{size}.pt (used if --model not given).")
    p.add_argument("--model", type=str, default=None,
                   help="Optional model file/path/name (overrides --model_size), e.g. yolov8s.pt.")

    p.add_argument("--color", type=int, nargs=3, default=[0, 255, 0], metavar=("B", "G", "R"),
                   help="BGR color for boxes/text, e.g. --color 0 0 255 for red.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    p.add_argument("--line_thickness", type=int, default=2, help="Rectangle thickness.")
    p.add_argument("--verbose", action="store_true", help="Print extra info.")

    # CHANGE (WSL): replace folder iteration with a live stream source.
    # Default is your Windows MJPEG server URL (so you DON'T accidentally use VideoCapture(0) in WSL).
    p.add_argument("--source", type=str, default="http://127.0.0.1:8080/video",
                   help="WSL stream URL or video path. Default: http://127.0.0.1:8080/video")

    # CHANGE: display controls (live)
    p.add_argument("--window_name", type=str, default="YOLO Live (WSL)", help="OpenCV window title.")
    p.add_argument("--show_fps", action="store_true", help="Overlay FPS in the top-left corner.")
    p.add_argument("--headless", action="store_true",
                   help="If set, do not open an OpenCV window (useful if GUI is unavailable).")

    # OPTIONAL: save annotated output video (useful if WSL window display is not working)
    p.add_argument("--save_video", type=str, default=None,
                   help="Optional output video file path (e.g. out.mp4). If set, writes annotated frames to a file.")

    # OPTIONAL: save crops occasionally (NOT required for live mode; can generate many files quickly)
    p.add_argument("--save_crops_dir", type=str, default=None,
                   help="Optional directory to save crops (WARNING: can create many files).")
    p.add_argument("--save_every_n_frames", type=int, default=0,
                   help="Save crops every N frames (0 = never). Use e.g. 30 to save once per ~1s at 30 FPS.")
    p.add_argument("--max_crops_per_frame", type=int, default=50,
                   help="Safety limit: maximum crops saved per frame when saving is enabled.")

    # MISSING/TODO: "redirect to another live stream" (RTSP/RTMP/WebRTC).
    # OpenCV does not natively publish RTSP/RTMP reliably in all builds.
    # If you need real restreaming, typically you pipe frames to FFmpeg or use GStreamer.

    return p.parse_args()


def clamp_box(x1, y1, x2, y2, w, h):
    """Clamp box coords into image bounds and ensure non-empty box."""
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def class_name_for(names, cid: int) -> str:
    """Get readable class name for a class id, fallback to numeric."""
    if isinstance(names, dict) and cid in names:
        return str(names[cid])
    if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
        return str(names[cid])
    return str(cid)


def main():
    args = parse_args()

    # SAME AS YOUR CURRENT SCRIPT: model selection (YOLOv8 by default) :contentReference[oaicite:4]{index=4} (shown outside code)
    model_name = args.model if args.model else f"yolov8{args.model_size}.pt"
    model = YOLO(model_name)
    names = getattr(model, "names", None)

    class_id = int(args.class_id)
    cls_name = class_name_for(names, class_id)
    box_color = tuple(args.color)  # OpenCV uses BGR

    # CHANGE: OpenCV capture uses MJPEG URL by default (WSL procedure).
    source = args.source

    cap = cv.VideoCapture(source)

    if not cap.isOpened():
        # CHANGE: provide WSL-specific troubleshooting hints (based on your procedure).
        raise SystemExit(
            f"Could not open video source: {source}\n\n"
            "WSL hint: Make sure the Windows MJPEG server is running and reachable.\n"
            "- Windows browser check: http://127.0.0.1:8080/\n"
            "- WSL quick check: curl -s http://127.0.0.1:8080/ | head\n"
            "- In WSL: use VideoCapture('http://127.0.0.1:8080/video') (NOT VideoCapture(0))."
        )

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

    # CHANGE: Optional VideoWriter setup (created after first frame so we know width/height).
    writer = None
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    # CHANGE: Optional crops output folder
    crops_dir = None
    if args.save_crops_dir:
        crops_dir = Path(args.save_crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)

    # CHANGE: FPS tracking (optional overlay)
    last_t = time.time()
    fps = 0.0

    # CHANGE: Frame counter (useful for crop sampling)
    frame_idx = 0

    if not args.headless:
        cv.namedWindow(args.window_name, cv.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if args.verbose:
                print("Frame grab failed or end of stream.")
            break

        frame_idx += 1
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # SAME CORE IDEA AS YOUR CURRENT SCRIPT: filter by class during inference :contentReference[oaicite:5]{index=5} (shown outside code)
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            classes=[class_id],
            verbose=False,
        )

        r0 = results[0]
        boxes = r0.boxes
        det_count = 0

        # Process detections for this frame
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                det_count += 1

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

                conf = float(boxes.conf[i].item())

                # Draw bounding box (OpenCV)
                cv.rectangle(annotated, (x1, y1), (x2, y2), box_color, args.line_thickness)

                # Middle-ground: always draw per-box label (class name + confidence)
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

                # OPTIONAL: Save crops (sampled; can explode in file count)
                if crops_dir and args.save_every_n_frames > 0:
                    if (frame_idx % args.save_every_n_frames) == 0 and det_count <= args.max_crops_per_frame:
                        crop = frame[y1:y2, x1:x2].copy()
                        crop_name = f"frame{frame_idx:06d}_det{det_count:03d}_cls{class_id}_c{int(conf*1000)}.png"
                        cv.imwrite(str(crops_dir / crop_name), crop)

        # Required-style overlay: total detections of selected class in the frame (bottom-left)
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

        # Optional FPS overlay
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

        # CHANGE: Initialize writer after we know the frame size
        if args.save_video and writer is None:
            # Try to read FPS from stream; fallback if unavailable
            src_fps = cap.get(cv.CAP_PROP_FPS)
            if not src_fps or src_fps <= 1e-3:
                src_fps = 30.0  # fallback
            writer = cv.VideoWriter(args.save_video, fourcc, float(src_fps), (w, h))

        # OPTIONAL: write video output
        if writer is not None:
            writer.write(annotated)

        # CHANGE: Display OR headless mode
        if not args.headless:
            cv.imshow(args.window_name, annotated)
            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC or q
                break
        else:
            # Headless quit option (Ctrl+C). No window, so no key polling here.
            pass

        # MISSING/TODO: Redirect annotated frames into another live stream endpoint (RTSP/RTMP).
        # Typical solutions:
        # - Pipe annotated frames to FFmpeg subprocess (stdin) and publish RTSP/RTMP
        # - Use GStreamer pipeline if your OpenCV build supports it

    cap.release()
    if writer is not None:
        writer.release()
    if not args.headless:
        cv.destroyAllWindows()

    if args.verbose:
        print("Done.")


if __name__ == "__main__":
    main()