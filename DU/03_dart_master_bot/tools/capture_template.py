from pathlib import Path
import cv2

from bot.config import BotConfig
from bot.capture import BridgeFrameSource


def main() -> None:
    """
    Utility script for saving one ROI image fetched from the Windows bridge into assets/debug_captures/bridge_roi_capture.png.
    Used during developing.
    """
    cfg = BotConfig()
    out_path = Path("assets/debug_captures/bridge_roi_capture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_source = BridgeFrameSource(cfg.bridge)

    health = frame_source.check_health()
    print("Bridge health:", health)

    bridge_config = frame_source.get_bridge_config()
    print("Bridge config:", bridge_config)

    frame_bgr = frame_source.grab_frame()

    ok = cv2.imwrite(str(out_path), frame_bgr)
    if not ok:
        raise RuntimeError(f"Could not save image to {out_path}")

    print(f"Saved: {out_path}")
    print("This image was captured from the Windows bridge, not from local WSL screen grab.")


if __name__ == "__main__":
    main()