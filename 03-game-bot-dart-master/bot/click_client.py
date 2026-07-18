import requests  # HTTP client library used to send click requests to the Windows bridge
from bot.config import BridgeConfig  # Bridge-related settings: base URL, click endpoint, timeout


class BridgeClickClient:
    """
    Bridge-side click sending module.

    This class is the counterpart to capture.py:
    - capture.py pulls ROI frames from the Windows bridge
    - click_client.py pushes ROI-local click commands back to the Windows bridge

    The class does not decide *when* a click should happen and it does not compute *where*
    the click should happen. Those decisions belong to controller.py.

    Its only responsibility is to take an already chosen point, package it into the bridge's
    expected JSON format, send it to the configured click endpoint, and return the bridge's
    response.

    This keeps actuation separate from decision-making:
    controller.py owns the decision,
    click_client.py owns the HTTP delivery of that decision.
    """

    def __init__(self, bridge_cfg: BridgeConfig):
        """
        Store bridge communication settings for all later click requests.

        The shared BridgeConfig comes from the top-level BotConfig created in main.py.
        Keeping it on the instance means every click request uses the same base URL,
        endpoint names, and timeout settings as the rest of the bridge-facing modules.
        """
        # Store the bridge configuration inside this object
        # so all methods can use the same base URL, endpoints, and timeout.
        self.bridge_cfg = bridge_cfg

    def _url(self, endpoint: str) -> str:
        """
        Internal helper that builds one full request URL from the configured base URL
        and a specific endpoint path.

        This keeps URL construction consistent and avoids repeating the same string
        concatenation logic inside each request method.
        """
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

        x_local and y_local are coordinates inside the ROI image, not absolute screen coordinates.
        That detail is important for the whole project structure:
        matcher.py and controller.py operate entirely in ROI-local image coordinates,
        and the Windows bridge is the layer responsible for translating those local coordinates
        into the real click on the Windows side.

        The method returns the bridge response as parsed JSON so controller.py can log it
        together with the rest of the iteration outcome.
        """
        # Build the JSON payload exactly in the format expected by the bridge.
        # The explicit int(...) conversion makes sure the transmitted coordinates are normal
        # integer pixel positions even if the caller passed values that originated from a
        # computation step.
        payload = {
            "x": int(x_local),
            "y": int(y_local),
        }

        # Send the click as an HTTP POST request to the configured click endpoint.
        # The coordinates are sent in the JSON body, and the same bridge timeout used by the
        # rest of the bridge communication is applied here as well.
        response = requests.post(
            self._url(self.bridge_cfg.click_endpoint),
            json=payload,
            timeout=self.bridge_cfg.timeout_seconds,
        )

        # Raise an exception immediately if the HTTP status indicates failure.
        # This keeps click-delivery problems visible instead of silently hiding them.
        response.raise_for_status()

        # Parse and return the bridge response.
        # controller.py uses this returned data mainly for iteration logging after a click.
        return response.json()

    def click_center(self, center: tuple[int, int]) -> dict:
        """
        Convenience wrapper for click_local() when the chosen point already exists as one
        center tuple in the common project format (x, y).

        This matches the way controller.py typically stores and passes points:
        matcher.py produces centers as tuples,
        controller.py chooses one final click point as a tuple,
        and this helper forwards that tuple into the main click_local() method.

        So this method exists mainly to keep the controller code cleaner and more readable.
        """
        # Extract x and y from the tuple and delegate the real sending work to click_local().
        # This keeps all payload-building and POST-request logic centralized in one method.
        return self.click_local(center[0], center[1])