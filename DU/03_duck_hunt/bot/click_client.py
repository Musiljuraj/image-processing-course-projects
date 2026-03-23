import requests  # HTTP client library used to send click requests to the Windows bridge
from bot.config import BridgeConfig  # Bridge-related settings: base URL, click endpoint, timeout


class BridgeClickClient:
    """
    Sends ROI-local coordinates to the Windows bridge, which performs the real click on Windows.
    """

    def __init__(self, bridge_cfg: BridgeConfig):
        # Store the bridge configuration inside this object
        # so all methods can use the same base URL, endpoints, and timeout.
        self.bridge_cfg = bridge_cfg

    def _url(self, endpoint: str) -> str:
        # Internal helper method:
        # combine the configured base URL with a specific endpoint path.
        # Example:
        #   base_url = "http://127.0.0.1:8080"
        #   endpoint = "/click_local"
        #   result = "http://127.0.0.1:8080/click_local"
        return f"{self.bridge_cfg.base_url}{endpoint}"

    def click_local(self, x_local: int, y_local: int) -> dict:
        """
        Send one ROI-local click request to the Windows bridge.

        x_local, y_local are coordinates inside the ROI image,
        not absolute screen coordinates.
        """
        # Build the JSON payload that will be sent to the bridge.
        # The coordinates are explicitly converted to integers so the bridge
        # receives normal pixel coordinates.
        payload = {
            "x": int(x_local),
            "y": int(y_local),
        }

        # Send a POST request to the bridge click endpoint.
        # The payload is sent as JSON in the request body.
        response = requests.post(
            self._url(self.bridge_cfg.click_endpoint),
            json=payload,
            timeout=self.bridge_cfg.timeout_seconds,
        )

        # Raise an exception if the HTTP status indicates failure.
        response.raise_for_status()

        # Parse and return the bridge response as JSON.
        return response.json()

    def click_center(self, center: tuple[int, int]) -> dict:
        """
        Convenience wrapper for click_local() when we already
        have the center as a tuple (x, y).
        """
        # Extract x and y from the tuple and reuse the main click_local() method.
        return self.click_local(center[0], center[1])
