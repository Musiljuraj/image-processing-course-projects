import math                      # math.hypot(...) is used for motion-vector magnitude calculations
import time                      # time.time() and time.sleep(...) are used for click timing and loop pacing
from collections import deque    # deque is used as a short fixed-length history of recent detected centers
from pathlib import Path         # Path is used for debug image output paths

import cv2                       # OpenCV is used here for debug text drawing and image saving

from bot.capture import BridgeFrameSource          # Frame fetcher: gets ROI screenshots from the Windows bridge
from bot.click_client import BridgeClickClient     # Click sender: sends ROI-local clicks to the Windows bridge
from bot.config import BotConfig                   # Top-level bot configuration object
from bot.debug_view import draw_match              # Draws debug annotations on frames
from bot.matcher import MatchResult, TemplateMatcher  # MatchResult = match data container, TemplateMatcher = matching engine
from bot.preprocessing import prepare_frame        # Prepares frame for matching (e.g. grayscale conversion)
from bot.templates import Template, load_templates # Template object and template-loading function


class DartMasterController:
    """
    Asks the Windows bridge for a fresh screenshot of the game ROI. That is done through BridgeFrameSource in capture.py
    Decides where to click: either the detected center directly, or a slightly predicted future point
    if prediction is enabled and judged safe enough.

    Central coordinator:

    DartMasterController creates and connects the main components - frame source, click client, matcher,
        loaded templates, state variables like click timing and motion history.

    Main jobs: repeatedly fetch a frame, preprocess it, find the best match, update motion history,
        choose the click point, decide whether clicking is allowed, send the click,
        optionally save a debug image, wait and continue.

    Contains the optional conservative prediction logic: short history of centers, weighted average motion,
        minimum motion threshold, direction consistency check, max forward jump clamp
    """
    def __init__(self, cfg: BotConfig):
        # Store the whole configuration object so all controller methods can use it.
        self.cfg = cfg

        # Create the main worker objects used by the controller.
        self.frame_source = BridgeFrameSource(cfg.bridge)      # Gets frames from the Windows bridge
        self.click_client = BridgeClickClient(cfg.bridge)      # Sends clicks to the Windows bridge
        self.matcher = TemplateMatcher(cfg.match.method)       # Performs template matching

        # Load all template files once during startup.
        # This avoids reloading them every iteration.
        self.templates: list[Template] = load_templates(
            cfg.templates.paths,
            use_gray=cfg.templates.use_gray,
        )

        # Runtime state:
        # - when the last click happened
        # - how many loop iterations have been run
        self.last_click_time: float = 0.0
        self.iteration_index: int = 0

        # Keep the loop responsive when no click occurred.
        self.idle_sleep_seconds: float = 0.001

        # Short history of recent detected centers, used only by optional prediction logic.
        # maxlen automatically drops the oldest point when the history becomes too long.
        self.center_history: deque[tuple[int, int]] = deque(
            maxlen=self.cfg.prediction.history_size
        )
        self.missed_prediction_frames: int = 0  # Counts weak/missing frames for prediction-history reset

        # Debug metadata for saved frames.
        self.last_click_point: tuple[int, int] | None = None
        self.last_prediction_used: bool = False

    def run(self) -> None:
        """
        Main continuous loop.

        Stop the bot with Ctrl+C in the terminal.
        """
        # Check whether the bridge is alive before entering the main loop.
        health = self.frame_source.check_health()
        print("Bridge health:", health)

        # Print loaded template names for startup verification.
        print("Loaded templates:", [template.name for template in self.templates])
        print("Controller started. Stop with Ctrl+C.")

        try:
            while True:
                # Count iterations for logging and debug filenames.
                self.iteration_index += 1

                # Perform one complete bot cycle.
                self._run_one_iteration()
        except KeyboardInterrupt:
            # Clean manual shutdown if the user presses Ctrl+C.
            print("\nController stopped by user.")

    def _run_one_iteration(self) -> None:
        # Fetch the newest frame from the Windows bridge.
        frame_bgr = self.frame_source.grab_frame()

        # Prepare the frame for matching (for example, convert to grayscale if configured).
        work_frame = prepare_frame(
            frame_bgr,
            use_gray=self.cfg.templates.use_gray,
        )

        # Stable baseline step:
        # use full-frame template matching exactly as before.
        best_match = self.matcher.match_best(work_frame, self.templates)

        # Update motion history for optional prediction logic.
        self._update_motion_history(best_match)

        # Choose the actual click point:
        # - detected center directly, or
        # - a conservative predicted point if prediction is enabled and safe.
        click_point, prediction_used = self._choose_click_point(best_match, frame_bgr.shape)
        self.last_click_point = click_point
        self.last_prediction_used = prediction_used

        # Track whether a click happened during this iteration.
        clicked = False
        click_response = None

        # Decide whether clicking is allowed right now.
        if self._should_click(best_match, click_point):
            # Send the click to the Windows bridge.
            click_response = self.click_client.click_center(click_point)

            # Update cooldown timing state.
            self.last_click_time = time.time()
            clicked = True

            # Print a detailed click log for this iteration.
            print(
                f"[{self.iteration_index}] CLICK "
                f"template={best_match.template_name} "
                f"score={best_match.score:.4f} "
                f"detected_center={best_match.center} "
                f"click_point={click_point} "
                f"prediction_used={prediction_used} "
                f"response={click_response}"
            )

        else:
            # If no click happened, distinguish between:
            # - no match at all
            # - match exists, but clicking was suppressed
            if best_match is None:
                print(f"[{self.iteration_index}] no match")
            else:
                print(
                    f"[{self.iteration_index}] no click "
                    f"template={best_match.template_name} "
                    f"score={best_match.score:.4f} "
                    f"detected_center={best_match.center} "
                    f"click_point={click_point} "
                    f"prediction_used={prediction_used} "
                    f"threshold={self.cfg.match.threshold:.4f}"
                )

        # Save an annotated debug frame only if debug saving is enabled.
        if self.cfg.debug.save_debug_frames:
            self._save_debug_frame(frame_bgr, best_match, clicked, click_point, prediction_used)

        # After a click, wait for the configured cooldown.
        # Otherwise sleep only a tiny amount to keep the loop responsive.
        if clicked:
            time.sleep(self.cfg.click.cooldown_seconds)
        else:
            time.sleep(self.idle_sleep_seconds)

    def _update_motion_history(self, best_match: MatchResult | None) -> None:
        """
        Update short center history only when the match is good enough.

        If too many weak/missing frames happen in a row, forget the motion history.
        """
        # If prediction is disabled, do nothing at all.
        if not self.cfg.prediction.enabled:
            return

        # Update history only when a match exists and is strong enough.
        if best_match is not None and best_match.score >= self.cfg.prediction.update_threshold:
            self.center_history.append(best_match.center)
            self.missed_prediction_frames = 0
        else:
            # Weak or missing frame for prediction purposes.
            self.missed_prediction_frames += 1

            # If too many weak/missing frames occur in a row, clear history
            # so prediction does not rely on stale motion.
            if self.missed_prediction_frames > self.cfg.prediction.max_missed_frames:
                self.center_history.clear()

    def _choose_click_point(
        self,
        best_match: MatchResult | None,
        frame_shape,
    ) -> tuple[tuple[int, int] | None, bool]:
        """
        Return:
        - click point
        - whether prediction was actually used
        """
        # If there is no match, there is no click point.
        if best_match is None:
            return None, False

        # Stable baseline behavior:
        # if prediction is disabled, click the detected center directly.
        if not self.cfg.prediction.enabled:
            return best_match.center, False

        # Try to compute a conservative predicted point.
        predicted = self._predict_click_point()

        # If prediction could not produce a safe/usable point,
        # fall back to the detected center.
        if predicted is None:
            return best_match.center, False

        # Clamp the predicted point into valid frame bounds and report
        # that prediction was actually used.
        return self._clamp_point(predicted, frame_shape), True

    def _predict_click_point(self) -> tuple[int, int] | None:
        """
        Use only a short history of recent centers.
        No local search, no track box, no search restriction.

        Prediction is used only if:
        - enough history exists
        - recent motion is large enough
        - recent direction is consistent enough
        """
        # Convert the deque history to a normal list for easier processing.
        points = list(self.center_history)

        # Require enough history points before prediction is allowed.
        if len(points) < self.cfg.prediction.min_history:
            return None

        # Convert consecutive center points into step vectors (dx, dy).
        steps: list[tuple[float, float]] = []
        for i in range(1, len(points)):
            dx = float(points[i][0] - points[i - 1][0])
            dy = float(points[i][1] - points[i - 1][1])
            steps.append((dx, dy))

        # If no steps exist, prediction cannot continue.
        if not steps:
            return None

        # Weighted average: newer motion gets more influence.
        weights = list(range(1, len(steps) + 1))
        weight_sum = float(sum(weights))

        # Compute weighted average motion in x and y.
        avg_dx = sum(w * dx for w, (dx, dy) in zip(weights, steps)) / weight_sum
        avg_dy = sum(w * dy for w, (dx, dy) in zip(weights, steps)) / weight_sum

        # Compute overall motion magnitude.
        avg_motion = math.hypot(avg_dx, avg_dy)

        # Ignore tiny motion as jitter/noise.
        if avg_motion < self.cfg.prediction.min_motion_px:
            return None

        # Check whether recent step directions agree with the average direction.
        direction_x = avg_dx / avg_motion
        direction_y = avg_dy / avg_motion

        direction_scores = []
        for dx, dy in steps:
            mag = math.hypot(dx, dy)

            # Skip almost-zero steps to avoid unstable direction calculations.
            if mag < 1e-6:
                continue

            # Direction agreement score:
            # near 1  = same direction
            # near 0  = unrelated/orthogonal
            # below 0 = opposite direction
            score = (dx * direction_x + dy * direction_y) / mag
            direction_scores.append(score)

        # If no usable direction scores exist, prediction cannot continue.
        if not direction_scores:
            return None

        # Average direction consistency must be high enough.
        direction_consistency = sum(direction_scores) / len(direction_scores)
        if direction_consistency < self.cfg.prediction.min_direction_consistency:
            return None

        # Compute a conservative forward offset based on recent average motion.
        pred_dx = avg_dx * self.cfg.prediction.prediction_scale
        pred_dy = avg_dy * self.cfg.prediction.prediction_scale

        # Apply a safety cap so the forward jump is never too large.
        pred_mag = math.hypot(pred_dx, pred_dy)
        if pred_mag > self.cfg.prediction.max_prediction_px and pred_mag > 0:
            shrink = self.cfg.prediction.max_prediction_px / pred_mag
            pred_dx *= shrink
            pred_dy *= shrink

        # Predict the next click point by shifting the newest center point
        # forward by the capped/scaled prediction offset.
        last_x, last_y = points[-1]
        predicted = (
            int(round(last_x + pred_dx)),
            int(round(last_y + pred_dy)),
        )
        return predicted

    def _should_click(
        self,
        best_match: MatchResult | None,
        click_point: tuple[int, int] | None,
    ) -> bool:
        # No match means no click.
        if best_match is None:
            return False

        # No click point means no click.
        if click_point is None:
            return False

        # Match must pass the configured detection threshold.
        if best_match.score < self.cfg.match.threshold:
            return False

        # Enforce click cooldown.
        now = time.time()
        if now - self.last_click_time < self.cfg.click.cooldown_seconds:
            return False

        # All checks passed.
        return True

    def _clamp_point(self, point: tuple[int, int], frame_shape) -> tuple[int, int]:
        # Extract frame height and width.
        frame_h, frame_w = frame_shape[:2]

        # Clamp x and y so they stay inside valid pixel boundaries.
        x = min(max(0, int(point[0])), frame_w - 1)
        y = min(max(0, int(point[1])), frame_h - 1)
        return (x, y)

    def _save_debug_frame(
        self,
        frame_bgr,
        best_match: MatchResult | None,
        clicked: bool,
        click_point: tuple[int, int] | None,
        prediction_used: bool,
    ) -> None:
        # Read the configured debug output directory and make sure it exists.
        debug_dir: Path = self.cfg.debug.debug_dir
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Create the main annotated debug image.
        annotated = draw_match(
            frame_bgr=frame_bgr,
            match=best_match,
            threshold=self.cfg.match.threshold,
            click_point=click_point,
            history_points=list(self.center_history),
            prediction_used=prediction_used,
        )

        # Add one extra status line showing whether a click happened.
        status_text = f"clicked={clicked}"
        cv2.putText(
            annotated,
            status_text,
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # Build a per-iteration filename like controller_debug_00001.png
        out_path = debug_dir / f"controller_debug_{self.iteration_index:05d}.png"

        # Save the annotated image to disk.
        ok = cv2.imwrite(str(out_path), annotated)

        # Raise a clear error if saving failed.
        if not ok:
            raise RuntimeError(f"Could not save debug image to {out_path}")
