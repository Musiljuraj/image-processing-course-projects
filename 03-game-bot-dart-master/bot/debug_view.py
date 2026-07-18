import cv2
import numpy as np

from bot.matcher import MatchResult


def draw_match(
    frame_bgr: np.ndarray,
    match: MatchResult | None,
    threshold: float,
    click_point: tuple[int, int] | None = None,
    history_points: list[tuple[int, int]] | None = None,
    prediction_used: bool = False,
) -> np.ndarray:
    """
    Debug-annotation renderer for one frame.

    This module does not participate in the actual bot decision logic.
    By the time this function is called, controller.py has already done the important work:
    it has acquired the frame, found the best match, chosen the click point, and decided
    whether prediction influenced that click point.

    The purpose of this function is purely explanatory and diagnostic:
    it turns the controller's internal understanding of the current frame into a visible,
    annotated image that can be saved for inspection.

    In the project flow, this function is used only from controller.py when debug-frame saving
    is enabled. It receives the raw frame together with the controller's current interpretation
    of that frame, and draws visual overlays that explain:

    - recent motion history, if prediction is enabled and history exists
    - the match rectangle
    - the detected center point
    - the final chosen click point
    - text showing score, threshold pass, and whether prediction was used

    So this module is the visual explanation layer of the project.
    It does not change decisions; it only visualizes them.
    """
    # Start from a copy of the original frame so debug drawing never modifies the source image
    # that came from capture.py. All annotations are placed onto this separate canvas.
    canvas = frame_bgr.copy()

    # If motion-history points were provided, draw them as a small connected path.
    # This gives a visual explanation of the short recent-center history maintained by controller.py
    # for the optional conservative prediction logic.
    if history_points:
        for idx, pt in enumerate(history_points):
            # Draw the individual recorded center point as a small filled circle.
            cv2.circle(canvas, pt, 3, (255, 200, 0), -1)

            # Starting from the second point, connect it to the previous one so the history
            # becomes a visible trajectory rather than just isolated dots.
            if idx > 0:
                cv2.line(canvas, history_points[idx - 1], pt, (255, 200, 0), 1)

    # If there is no match at all, the function cannot draw match geometry.
    # In that case it produces a simpler debug frame that explicitly says no match result exists,
    # then returns immediately.
    if match is None:
        cv2.putText(
            canvas,
            "No match result",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return canvas

    # Decide whether the current match passes the main controller threshold.
    # This mirrors the acceptance concept used later by controller.py when deciding if a click
    # is allowed, so the debug image visually shows whether the match is above or below that bar.
    passed = match.score >= threshold

    # Choose rectangle/text color based on threshold result:
    # - green for passed match
    # - orange for below-threshold match
    #
    # This makes the acceptance state visible immediately without having to read the text overlay.
    box_color = (0, 255, 0) if passed else (0, 165, 255)

    # Draw the matched template rectangle.
    # The corners come from matcher.py, which computed them from the chosen template size and
    # the best match location inside the frame.
    cv2.rectangle(canvas, match.top_left, match.bottom_right, box_color, 2)

    # Draw the detected center point of the matched rectangle.
    # This is the baseline click point in non-predictive mode and the reference point used by
    # controller.py before any optional forward prediction adjustment.
    cv2.circle(canvas, match.center, 4, (0, 0, 255), -1)

    # If the controller provided a final click point, draw it as a marker.
    # This point may be identical to match.center in baseline mode, or slightly shifted forward
    # when conservative prediction was actually used.
    if click_point is not None:
        cv2.drawMarker(
            canvas,
            click_point,
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )

    # Build the text overlay lines.
    #
    # text_1:
    #   identifies which template won and how strong the match score was
    #
    # text_2:
    #   shows the detected center and whether the match passed the threshold
    #
    # text_3:
    #   shows the final click point and whether prediction influenced that point
    text_1 = f"{match.template_name} | score={match.score:.3f}"
    text_2 = f"center={match.center} | threshold_pass={passed}"
    text_3 = f"click_point={click_point} | prediction_used={prediction_used}"

    # Draw the first text line near the top of the frame.
    # Its color matches the pass/fail rectangle color so template identity and match quality
    # stay visually connected.
    cv2.putText(
        canvas,
        text_1,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        box_color,
        2,
    )

    # Draw the second text line below the first one.
    # This continues the match-status explanation with the detected center and threshold result.
    cv2.putText(
        canvas,
        text_2,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        box_color,
        2,
    )

    # Draw the third text line below the match-status lines.
    # Magenta is used here to visually associate this line with the click marker color,
    # because both describe the final actuation point and prediction usage.
    cv2.putText(
        canvas,
        text_3,
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2,
    )

    # Return the fully annotated debug frame.
    # controller.py may then save this image to disk as a persistent explanation of one iteration.
    return canvas