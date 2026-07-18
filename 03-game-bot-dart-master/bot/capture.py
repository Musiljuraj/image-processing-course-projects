"""
Bridge-side frame acquisition module.

This module is responsible for obtaining the current ROI screenshot from the Windows bridge
and converting the returned JPEG bytes into a normal OpenCV image matrix.

In the overall runtime flow, this module appears very early in the pipeline:

main.py
    -> builds config and starts the controller
controller.py
    -> asks BridgeFrameSource for the newest frame each iteration
capture.py
    -> talks to the Windows bridge over HTTP
    -> downloads the current ROI image
    -> decodes it into an OpenCV BGR frame
preprocessing.py / matcher.py
    -> consume that frame for visual detection

So this file is the acquisition layer of the bot. It does not do matching, prediction,
or clicking. It only retrieves the live image input and exposes a small amount of
bridge-side status/config inspection.
"""

import cv2                     # OpenCV; used here to decode JPEG bytes into an image matrix
import numpy as np            # NumPy; used here to wrap raw response bytes into a uint8 array
import requests               # HTTP client library for talking to the Windows bridge
from bot.config import BridgeConfig  # Bridge-related settings (base URL, endpoints, timeout)


class BridgeFrameSource:
    """
    Frame fetcher for the Windows bridge.

    This class encapsulates all bridge requests related to obtaining image data or inspecting
    bridge-side status. It keeps bridge communication details in one place so controller.py
    can simply ask for a frame instead of constructing raw HTTP requests itself.

    Main responsibilities:
    - build full request URLs from BridgeConfig
    - check whether the bridge is alive
    - read the bridge's exposed configuration
    - download the current ROI image from /frame.jpg
    - decode the returned JPEG bytes into a BGR OpenCV frame
    """

    def __init__(self, bridge_cfg: BridgeConfig):
        """
        Store bridge communication settings inside the frame-source object.

        The controller passes in the BridgeConfig section from the shared BotConfig so this
        instance can use the same base URL, endpoints, and timeout values for every request.
        """
        # Store the bridge configuration object inside this instance so all request methods
        # use one shared set of bridge-related runtime settings.
        self.bridge_cfg = bridge_cfg

    def _url(self, endpoint: str) -> str:
        """
        Build a full request URL from the configured bridge base URL and one endpoint path.

        This small helper prevents URL-concatenation logic from being repeated in each request
        method and keeps the request-building pattern consistent across health/config/frame calls.
        """
        # Combine the configured base URL with the requested endpoint.
        # Example:
        #   base_url = "http://127.0.0.1:8080"
        #   endpoint = "/frame.jpg"
        #   result   = "http://127.0.0.1:8080/frame.jpg"
        return f"{self.bridge_cfg.base_url}{endpoint}"

    def check_health(self) -> dict:
        """
        Query the bridge health endpoint and return the parsed JSON response.

        This is mainly used by controller.py at startup to verify that the external bridge
        process is reachable before entering the continuous loop.
        """
        # Send a GET request to the bridge health endpoint.
        # The timeout comes from config so bridge communication stays centrally configurable.
        response = requests.get(
            self._url(self.bridge_cfg.health_endpoint),
            timeout=self.bridge_cfg.timeout_seconds,
        )

        # Raise an exception if the HTTP status code indicates failure.
        # This makes bridge connectivity problems surface immediately instead of being ignored.
        response.raise_for_status()

        # Parse and return the JSON body as a Python dictionary-like object.
        return response.json()

    def get_bridge_config(self) -> dict:
        """
        Query the bridge config endpoint and return the parsed JSON response.

        This provides a way to inspect the configuration that the Windows bridge itself exposes.
        It is separate from the local BotConfig and reflects what the bridge reports from its side.
        """
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
        """
        Download the current ROI frame from the bridge and decode it into an OpenCV image.

        The bridge returns compressed JPEG bytes. This method converts those bytes into the
        standard BGR image format used by the rest of the project before preprocessing.
        """
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

        # If decoding failed, stop immediately with a clear error because the rest of the
        # pipeline requires a valid image matrix.
        if frame_bgr is None:
            raise RuntimeError("Could not decode frame from Windows bridge.")

        # Return the decoded BGR frame for further processing by preprocessing.py and matcher.py.
        return frame_bgr