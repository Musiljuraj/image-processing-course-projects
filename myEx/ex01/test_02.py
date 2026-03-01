import cv2 as cv

WINDOW = "Kamera"
DEFAULT_DEVICE = "http://127.0.0.1:8080/video"


def process_frame(frame):
    # TODO: add your processing here (draw, detect, etc.)
    return frame

def run_camera(device=DEFAULT_DEVICE):
    cap = cv.VideoCapture(device)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device: {device}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame)
            cv.imshow(WINDOW, frame)

            if (cv.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

def main():
    run_camera(DEFAULT_DEVICE)

if __name__ == "__main__":
    main()