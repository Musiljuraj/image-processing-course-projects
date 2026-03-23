import cv2                     # OpenCV; used here to decode JPEG bytes into an image matrix
import numpy as np            # NumPy; used here to wrap raw response bytes into a uint8 array
import requests               # HTTP client library for talking to the Windows bridge
from bot.config import BridgeConfig  # Bridge-related settings (base URL, endpoints, timeout)


class BridgeFrameSource:
    """
    Frame fetcher: class BridgeFrameSource talks to the Windows bridge over HTTP.
        check bridge health,
        read bridge config,
        download the current ROI image from /frame.jpg,
        decode JPEG bytes into an OpenCV BGR image
    """

    def __init__(self, bridge_cfg: BridgeConfig):
        # Store the bridge configuration object inside this instance
        # so all methods can use the same base URL, endpoints, and timeout.
        self.bridge_cfg = bridge_cfg

    def _url(self, endpoint: str) -> str:
        # Internal helper method:
        # combine the configured bridge base URL with a specific endpoint path.
        # Example:
        #   base_url = "http://127.0.0.1:8080"
        #   endpoint = "/frame.jpg"
        #   result = "http://127.0.0.1:8080/frame.jpg"
        return f"{self.bridge_cfg.base_url}{endpoint}"

    def check_health(self) -> dict:
        # Send a GET request to the bridge health endpoint.
        response = requests.get(
            self._url(self.bridge_cfg.health_endpoint),
            timeout=self.bridge_cfg.timeout_seconds,
        )

        # Raise an exception if the HTTP status code indicates failure.
        response.raise_for_status()

        # Parse and return the JSON body as a Python dictionary-like object.
        return response.json()

    def get_bridge_config(self) -> dict:
        # Send a GET request to the bridge config endpoint.
        response = requests.get(
            self._url(self.bridge_cfg.config_endpoint),
            timeout=self.bridge_cfg.timeout_seconds,
        )

        # Raise an exception if the HTTP request failed.
        response.raise_for_status()

        # Parse and return the bridge configuration JSON.
        return response.json()

    def grab_frame(self) -> np.ndarray:
        # Send a GET request to the frame endpoint, which should return a JPEG image.
        response = requests.get(
            self._url(self.bridge_cfg.frame_endpoint),
            timeout=self.bridge_cfg.timeout_seconds,
        )

        # Raise an exception if the HTTP request failed.
        response.raise_for_status()

        # Convert the raw response bytes into a 1D NumPy array of uint8 values.
        # At this point the data is still compressed JPEG data, not yet a decoded image.
        frame_bytes = np.frombuffer(response.content, dtype=np.uint8)

        # Decode the JPEG byte array into a normal OpenCV color image.
        # OpenCV color images use BGR channel order by default.
        frame_bgr = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

        # If decoding failed, stop immediately with a clear error.
        if frame_bgr is None:
            raise RuntimeError("Could not decode frame from Windows bridge.")

        # Return the decoded BGR frame for further processing.
        return frame_bgr

















