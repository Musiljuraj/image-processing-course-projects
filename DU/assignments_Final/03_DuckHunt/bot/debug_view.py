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
    Visual explanation: 
    If debugging is enabled, debug_view.py draws rectangles, centers, click markers,
    optional motion history, and match metadata into saved debug images. 

    Draws:
    - motion history points and lines
    - match rectangle
    - detected center
    - click point marker
    - text with template name, score, threshold pass, prediction usage  
    - winning scale and scaled matched size 

    Used mainly for saved debug images.  
    """
    canvas = frame_bgr.copy()

    if history_points:
        for idx, pt in enumerate(history_points):
            cv2.circle(canvas, pt, 3, (255, 200, 0), -1)
            if idx > 0:
                cv2.line(canvas, history_points[idx - 1], pt, (255, 200, 0), 1)

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

    passed = match.score >= threshold
    box_color = (0, 255, 0) if passed else (0, 165, 255)

    cv2.rectangle(canvas, match.top_left, match.bottom_right, box_color, 2)
    cv2.circle(canvas, match.center, 4, (0, 0, 255), -1)

    if click_point is not None:
        cv2.drawMarker(
            canvas,
            click_point,
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )

    text_1 = f"{match.template_name} | score={match.score:.3f}"
    text_2 = (
        f"scale={match.scale:.2f} | size={match.matched_width}x{match.matched_height}"
    )  
    text_3 = f"center={match.center} | threshold_pass={passed}"  
    text_4 = f"click_point={click_point} | prediction_used={prediction_used}"  

    cv2.putText(
        canvas,
        text_1,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        box_color,
        2,
    )
    cv2.putText(
        canvas,
        text_2,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        box_color,
        2,
    )
    cv2.putText(
        canvas,
        text_3,
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        box_color,
        2,
    )
    cv2.putText(
        canvas,
        text_4,
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2,
    )

    return canvas