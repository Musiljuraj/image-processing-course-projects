"""
Main runtime coordinator of the bot.

This module contains the continuous execution loop that turns the static startup state
prepared in main.py into live behavior.

The controller sits exactly in the middle of the whole project architecture:

- main.py creates BotConfig and then hands control to DartMasterController
- capture.py provides fresh ROI frames from the Windows bridge
- preprocessing.py converts those frames into the format expected by matching
- templates.py provides preloaded template objects
- matcher.py produces the strongest visual match in the current frame
- click_client.py sends the chosen ROI-local click point back to the Windows bridge
- debug_view.py generates annotated debug images when debug saving is enabled

Because of that position, controller.py is the "decision layer" of the project.
It does not define low-level bridge communication, image loading, or template matching rules.
Instead, it connects those modules into one runtime pipeline and decides, frame by frame:

1. fetch the newest frame
2. prepare it for matching
3. find the best template match
4. update short motion history when prediction is enabled
5. choose the actual click point
6. decide whether a click is allowed right now
7. send the click if allowed
8. optionally save a debug frame
9. sleep briefly and continue

The optional prediction logic is deliberately conservative.
The baseline behavior of the project is still direct-center clicking based on full-frame
template matching, and prediction only adjusts that center when enough recent motion evidence
exists and passes multiple safety checks.
"""

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
    Central runtime controller of the bot.

    This class owns the long-running loop and the runtime state that must survive across
    iterations, such as click cooldown timing, iteration counters, and the short recent-center
    history used by optional prediction.

    At startup, the controller creates and stores the main worker components it needs:
    frame source, click client, matcher, and the list of loaded templates.

    During each loop iteration, it performs one full end-to-end decision cycle:
    acquire frame -> prepare frame -> match template -> optionally update motion history ->
    choose click point -> check click rules -> possibly click -> possibly save debug frame.

    The controller therefore acts as the project's operational "brain":
    lower-level modules each do one specialized task, while this class decides how they are
    used together and in what order.
    """
    def __init__(self, cfg: BotConfig):
        # Store the whole configuration object so all controller methods can use it.
        # This keeps the controller aligned with the same shared configuration tree built in main.py.
        self.cfg = cfg

        # Create the main worker objects used by the controller.
        # These wrap the two bridge-facing responsibilities:
        # - fetching the latest ROI frame
        # - sending ROI-local click commands
        #
        # The matcher is also created here so the chosen OpenCV matching method is fixed
        # once at startup rather than passed around repeatedly.
        self.frame_source = BridgeFrameSource(cfg.bridge)      # Gets frames from the Windows bridge
        self.click_client = BridgeClickClient(cfg.bridge)      # Sends clicks to the Windows bridge
        self.matcher = TemplateMatcher(cfg.match.method)       # Performs template matching

        # Load all template files once during startup.
        # This is important because templates are static reference images, so repeatedly loading
        # them inside the loop would be unnecessary overhead.
        #
        # The grayscale choice is shared with frame preprocessing so both the live frame and the
        # template images stay in the same representation.
        self.templates: list[Template] = load_templates(
            cfg.templates.paths,
            use_gray=cfg.templates.use_gray,
        )

        # Runtime state that persists across loop iterations.
        #
        # last_click_time:
        #   remembers when the previous click happened so cooldown logic can suppress rapid repeats.
        #
        # iteration_index:
        #   increments once per loop and is used both for human-readable logs and debug filenames.
        self.last_click_time: float = 0.0
        self.iteration_index: int = 0

        # Very short sleep used when no click occurred.
        # This keeps the loop highly responsive without becoming a completely tight busy-spin loop.
        self.idle_sleep_seconds: float = 0.001

        # Short history of recent detected centers, used only by optional prediction logic.
        # deque(maxlen=...) automatically discards the oldest point when the history becomes full,
        # which naturally enforces the "short recent history" design described in config.py.
        self.center_history: deque[tuple[int, int]] = deque(
            maxlen=self.cfg.prediction.history_size
        )

        # Counts how many weak or missing frames have happened in a row from the perspective of
        # motion prediction. Once this exceeds the configured limit, old history is discarded.
        self.missed_prediction_frames: int = 0

        # Debug-related state from the most recent iteration.
        # These values are stored so the controller can keep track of what point was actually used
        # and whether prediction influenced that point.
        self.last_click_point: tuple[int, int] | None = None
        self.last_prediction_used: bool = False

    def run(self) -> None:
        """
        Main continuous loop.

        The controller first verifies that the bridge is reachable, then prints a small startup
        summary for runtime visibility, and finally enters an infinite loop that repeatedly executes
        one full iteration of the bot.

        Normal shutdown is manual through Ctrl+C.
        """
        # Before entering the loop, verify that the Windows bridge is alive and responding.
        # This gives immediate feedback if the bot cannot communicate with its external bridge layer.
        health = self.frame_source.check_health()
        print("Bridge health:", health)

        # Print the names of all loaded templates.
        # This confirms that template loading succeeded and shows exactly which references are active.
        print("Loaded templates:", [template.name for template in self.templates])
        print("Controller started. Stop with Ctrl+C.")

        try:
            while True:
                # Count iterations for logging and debug filenames.
                # Incrementing at the start makes iteration numbering human-friendly and 1-based.
                self.iteration_index += 1

                # Perform one complete bot cycle.
                # The actual per-frame logic is intentionally factored into a separate method so
                # the outer run loop stays simple and readable.
                self._run_one_iteration()
        except KeyboardInterrupt:
            # Clean manual shutdown if the user presses Ctrl+C.
            # This mirrors the project-wide treatment of manual stop as normal behavior, not failure.
            print("\nController stopped by user.")

    def _run_one_iteration(self) -> None:
        """
        Execute one full live decision cycle for the current frame.

        This is the core operational pipeline of the bot. One iteration means:
        - get the newest frame from the bridge
        - prepare it for matching
        - find the strongest visual match
        - update short motion history for optional prediction
        - choose the final click point
        - decide whether clicking is currently allowed
        - click or skip
        - optionally save a debug frame
        - sleep according to click/no-click outcome
        """
        # Fetch the newest frame from the Windows bridge.
        # The returned image is the current ROI snapshot in BGR format.
        frame_bgr = self.frame_source.grab_frame()

        # Prepare the frame for matching.
        # In the current project, this mainly means keeping preprocessing consistent with the
        # template-loading mode, especially the grayscale vs. color choice.
        work_frame = prepare_frame(
            frame_bgr,
            use_gray=self.cfg.templates.use_gray,
        )

        # Stable baseline detection step:
        # use full-frame template matching to locate the strongest candidate in the current frame.
        #
        # The matcher always returns the strongest result it can find among the loaded templates,
        # but that does not automatically mean a click will happen. Click permission is checked later.
        best_match = self.matcher.match_best(work_frame, self.templates)

        # Update short motion history for optional prediction logic.
        # Only sufficiently trustworthy matches are allowed to influence the recent-center history.
        self._update_motion_history(best_match)

        # Choose the actual click point:
        # - detected center directly for baseline behavior, or
        # - a conservative predicted point if prediction is enabled and the prediction logic
        #   decides that recent motion evidence is strong and stable enough.
        click_point, prediction_used = self._choose_click_point(best_match, frame_bgr.shape)
        self.last_click_point = click_point
        self.last_prediction_used = prediction_used

        # Track what happened in this iteration so later logic can react accordingly.
        clicked = False
        click_response = None

        # Decide whether clicking is allowed right now.
        # This combines detection quality, point availability, and cooldown timing.
        if self._should_click(best_match, click_point):
            # Send the click to the Windows bridge.
            # The bridge is responsible for turning ROI-local coordinates into the real action.
            click_response = self.click_client.click_center(click_point)

            # Record the time of this click so cooldown suppression can be enforced in future iterations.
            self.last_click_time = time.time()
            clicked = True

            # Print a detailed click log for this iteration.
            # This log ties together the chosen template, raw detected center, final click point,
            # prediction usage, and bridge response so one line captures the whole decision outcome.
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
            # - a match exists, but clicking was suppressed by threshold/cooldown/point checks
            #
            # This distinction is useful because "no visual candidate" and "candidate found but not
            # allowed to click" are operationally very different situations.
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
        # Debug images visualize the detection and click decision path without changing bot behavior.
        if self.cfg.debug.save_debug_frames:
            self._save_debug_frame(frame_bgr, best_match, clicked, click_point, prediction_used)

        # End-of-iteration pacing:
        # - after a click, wait for the configured cooldown so repeated clicks cannot occur immediately
        # - otherwise, sleep only a tiny amount to keep the loop responsive
        if clicked:
            time.sleep(self.cfg.click.cooldown_seconds)
        else:
            time.sleep(self.idle_sleep_seconds)

    def _update_motion_history(self, best_match: MatchResult | None) -> None:
        """
        Update short recent-center history for optional prediction.

        The controller never lets weak or missing detections influence motion prediction as if they
        were trustworthy movement. Instead, recent-center history is updated only when prediction is
        enabled and the current match passes the dedicated prediction update threshold.

        If too many weak or missing frames happen in a row, the history is cleared so future
        prediction cannot rely on stale motion from an older tracking phase.
        """
        # If prediction is disabled, the controller should behave like the stable baseline and
        # completely ignore motion-history maintenance.
        if not self.cfg.prediction.enabled:
            return

        # Update history only when a match exists and is strong enough to be trusted for motion.
        # This threshold is intentionally separate from the click threshold because the system may
        # want to learn motion from somewhat weaker frames without necessarily clicking on them.
        if best_match is not None and best_match.score >= self.cfg.prediction.update_threshold:
            self.center_history.append(best_match.center)
            self.missed_prediction_frames = 0
        else:
            # Weak or missing frame from the perspective of prediction.
            # The controller remembers that recent evidence has become unreliable.
            self.missed_prediction_frames += 1

            # If too many weak/missing frames occur in a row, clear history so prediction does not
            # continue from stale old movement that may no longer represent the current target.
            if self.missed_prediction_frames > self.cfg.prediction.max_missed_frames:
                self.center_history.clear()

    def _choose_click_point(
        self,
        best_match: MatchResult | None,
        frame_shape,
    ) -> tuple[tuple[int, int] | None, bool]:
        """
        Decide which point should actually be used for clicking.

        Return value:
        - click point (or None if no match exists)
        - whether prediction was actually used

        This method is the bridge between visual detection and actuation:
        it starts from the match center provided by matcher.py and only replaces that center with
        a predicted point when prediction is enabled and produces a safe usable result.
        """
        # If there is no match, there is no meaningful click point to choose.
        if best_match is None:
            return None, False

        # Stable baseline behavior:
        # when prediction is disabled, always click the detected center directly.
        # This is the project's default runtime mode.
        if not self.cfg.prediction.enabled:
            return best_match.center, False

        # Try to compute a conservative predicted point from recent motion history.
        predicted = self._predict_click_point()

        # If prediction could not produce a safe or usable point, fall back to the detected center.
        # This fallback preserves normal behavior whenever the motion evidence is insufficient.
        if predicted is None:
            return best_match.center, False

        # Clamp the predicted point into valid frame bounds.
        # Even a reasonable prediction should never be allowed to leave the image area.
        return self._clamp_point(predicted, frame_shape), True

    def _predict_click_point(self) -> tuple[int, int] | None:
        """
        Compute a conservative predicted click point from short recent-center history.

        The logic intentionally stays simple and safety-oriented:
        it does not open a local search region, track a box, or restrict later matching.
        It only looks at a short sequence of recent detected centers and asks whether they form
        a strong enough pattern to justify a small forward shift.

        Prediction is allowed only if:
        - enough recent history exists
        - average motion is large enough to count as real motion rather than jitter
        - recent movement direction is consistent enough
        - the final forward shift stays inside the configured safety cap

        If any of those conditions fail, the method returns None and the controller falls back
        to direct-center clicking.
        """
        # Convert the deque history to a normal list for easier indexing and repeated processing.
        points = list(self.center_history)

        # Require enough history points before prediction is allowed.
        # This prevents the controller from extrapolating from too little evidence.
        if len(points) < self.cfg.prediction.min_history:
            return None

        # Convert consecutive center points into motion step vectors (dx, dy).
        # Each step represents how the detected center moved from one frame to the next.
        steps: list[tuple[float, float]] = []
        for i in range(1, len(points)):
            dx = float(points[i][0] - points[i - 1][0])
            dy = float(points[i][1] - points[i - 1][1])
            steps.append((dx, dy))

        # If no steps exist, there is nothing to extrapolate from.
        if not steps:
            return None

        # Build linearly increasing weights so newer motion has more influence than older motion.
        # This makes prediction responsive to recent changes without fully ignoring older recent steps.
        weights = list(range(1, len(steps) + 1))
        weight_sum = float(sum(weights))

        # Compute weighted average motion in x and y.
        # This gives one compact "recent average movement" vector.
        avg_dx = sum(w * dx for w, (dx, dy) in zip(weights, steps)) / weight_sum
        avg_dy = sum(w * dy for w, (dx, dy) in zip(weights, steps)) / weight_sum

        # Compute overall average motion magnitude.
        # This is used to separate real movement from tiny jitter-like fluctuations.
        avg_motion = math.hypot(avg_dx, avg_dy)

        # Ignore tiny motion as jitter/noise.
        # If movement is too small, prediction is considered unsafe and unnecessary.
        if avg_motion < self.cfg.prediction.min_motion_px:
            return None

        # Normalize the average motion vector into a unit direction vector.
        # This gives the dominant recent movement direction.
        direction_x = avg_dx / avg_motion
        direction_y = avg_dy / avg_motion

        # Score how well each recent step agrees with that dominant average direction.
        # This measures whether motion is coherent or erratic.
        direction_scores = []
        for dx, dy in steps:
            mag = math.hypot(dx, dy)

            # Skip almost-zero steps because their direction is unstable and would inject noise.
            if mag < 1e-6:
                continue

            # Direction agreement score based on normalized dot product:
            # near  1 -> same direction
            # near  0 -> unrelated / approximately orthogonal
            # below 0 -> opposite direction
            score = (dx * direction_x + dy * direction_y) / mag
            direction_scores.append(score)

        # If no usable direction scores exist, direction stability cannot be established.
        if not direction_scores:
            return None

        # Require the recent direction pattern to be consistent enough.
        # This is one of the main safeguards against predicting during unstable motion.
        direction_consistency = sum(direction_scores) / len(direction_scores)
        if direction_consistency < self.cfg.prediction.min_direction_consistency:
            return None

        # Compute a conservative forward offset based on recent average motion.
        # The configured prediction_scale keeps this as a partial forward step, not a full leap.
        pred_dx = avg_dx * self.cfg.prediction.prediction_scale
        pred_dy = avg_dy * self.cfg.prediction.prediction_scale

        # Apply a safety cap so the forward jump is never too large.
        # Even if the estimated motion is large, the final prediction must remain bounded.
        pred_mag = math.hypot(pred_dx, pred_dy)
        if pred_mag > self.cfg.prediction.max_prediction_px and pred_mag > 0:
            shrink = self.cfg.prediction.max_prediction_px / pred_mag
            pred_dx *= shrink
            pred_dy *= shrink

        # Predict the next click point by shifting the newest detected center forward
        # along the capped/scaled motion estimate.
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
        """
        Decide whether the controller is allowed to click in the current iteration.

        A click is allowed only when all of the following are true:
        - a match exists
        - a click point exists
        - the match score passes the configured main threshold
        - click cooldown has expired

        This method is intentionally strict and compact because it is the final gate between
        visual detection and real actuation.
        """
        # No match means no click.
        if best_match is None:
            return False

        # No click point means no click.
        # This covers cases where there is no usable center or prediction output.
        if click_point is None:
            return False

        # Match must pass the configured detection threshold.
        # The matcher's "best result" is not enough on its own; it still has to be strong enough.
        if best_match.score < self.cfg.match.threshold:
            return False

        # Enforce click cooldown so repeated clicks cannot happen too close together.
        now = time.time()
        if now - self.last_click_time < self.cfg.click.cooldown_seconds:
            return False

        # All checks passed, so clicking is allowed.
        return True

    def _clamp_point(self, point: tuple[int, int], frame_shape) -> tuple[int, int]:
        """
        Clamp a point so it stays inside valid frame bounds.

        Prediction can shift the click point forward, so this safety step guarantees that the final
        point remains inside the image dimensions before it is ever sent to the bridge.
        """
        # Extract frame height and width from the image shape.
        frame_h, frame_w = frame_shape[:2]

        # Clamp x and y into valid inclusive pixel ranges:
        # x in [0, frame_w - 1]
        # y in [0, frame_h - 1]
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
        """
        Save one annotated debug image for the current iteration.

        The saved debug image summarizes the controller's visual understanding of the frame:
        match rectangle, center, click point, motion history, prediction usage, and click outcome.

        This method does not affect the bot's decisions.
        It only turns the already-made decision path into a persistent visual artifact.
        """
        # Read the configured debug output directory and ensure it exists before writing files.
        # parents=True allows nested directory creation, and exist_ok=True makes repeated calls safe.
        debug_dir: Path = self.cfg.debug.debug_dir
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Build the main annotated debug image using the dedicated drawing helper.
        # draw_match() handles the visual overlays for match geometry, click point, and motion history.
        annotated = draw_match(
            frame_bgr=frame_bgr,
            match=best_match,
            threshold=self.cfg.match.threshold,
            click_point=click_point,
            history_points=list(self.center_history),
            prediction_used=prediction_used,
        )

        # Add one extra controller-level status line showing whether a click actually happened.
        # This is added here instead of inside debug_view.py because "clicked" is a controller outcome.
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

        # Build a deterministic per-iteration filename like controller_debug_00001.png.
        # Zero-padding keeps saved files naturally ordered when listed in the filesystem.
        out_path = debug_dir / f"controller_debug_{self.iteration_index:05d}.png"

        # Save the annotated image to disk.
        ok = cv2.imwrite(str(out_path), annotated)

        # Raise a clear error if saving failed.
        # Debug saving is optional, but once enabled it should fail loudly if output cannot be written.
        if not ok:
            raise RuntimeError(f"Could not save debug image to {out_path}")