from pathlib import Path

import cv2
import numpy as np
import requests

"""
Low-level bridge communication test. Used during developing.
"""

# Local Host.
BRIDGE_BASE = "http://127.0.0.1:8080"


def main() -> None:
    out_path = Path("assets/debug_captures/bridge_capture_test.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config_resp = requests.get(f"{BRIDGE_BASE}/config", timeout=2)
    config_resp.raise_for_status()
    config_data = config_resp.json()
    print("Bridge config:", config_data)

    frame_resp = requests.get(f"{BRIDGE_BASE}/frame.jpg", timeout=2)
    frame_resp.raise_for_status()

    frame_bytes = np.frombuffer(frame_resp.content, dtype=np.uint8)
    frame_bgr = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        raise RuntimeError("Could not decode JPEG frame from Windows bridge.")

    ok = cv2.imwrite(str(out_path), frame_bgr)
    if not ok:
        raise RuntimeError(f"Could not save image to {out_path}")

    print(f"Saved: {out_path}")
    print(f"Captured frame size: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}")


if __name__ == "__main__":
    main()