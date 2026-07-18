"""
Matching engine of the project.

This module is responsible for turning a prepared live frame and a set of already loaded
template images into one structured "best match" result.

In the overall runtime flow, matcher.py sits after frame acquisition and preprocessing,
and before click-point selection:

capture.py
    -> provides the current ROI frame from the Windows bridge
preprocessing.py
    -> converts that frame into the representation expected by matching
templates.py
    -> provides loaded template objects with image data and dimensions
matcher.py
    -> compares every template against the prepared frame
    -> finds the strongest candidate
    -> converts raw OpenCV output into one normalized MatchResult object
controller.py
    -> decides whether that result is strong enough to click
    -> optionally uses the center point for conservative prediction logic

So this module is the visual decision input layer.
It does not decide whether to click, and it does not communicate with the bridge.
Its job is narrower and very important: locate the best visual candidate in the current frame
and describe it in a structured, controller-friendly way.
"""

from dataclasses import dataclass  # dataclass = lightweight class mainly for holding related data

import cv2                         # OpenCV; provides template matching and min/max-location tools
import numpy as np                # NumPy; used here for the frame type hint (np.ndarray)
from bot.templates import Template  # Structured template object loaded by templates.py


@dataclass
class MatchResult:
    """
    Structured representation of one chosen template match.

    Instead of returning several separate values from the matching process, the module wraps the
    outcome into one small data object. This makes the rest of the project easier to read because
    controller.py can work with named fields such as score, center, and template_name.

    The fields describe both identification and geometry:
    - which template won
    - how strong the match was
    - where the matched rectangle is
    - where the rectangle center is

    That center is especially important because it becomes the baseline click point in controller.py.
    """
    template_name: str                 # Name of the template that produced this match
    score: float                       # Match score; higher is treated as better
    top_left: tuple[int, int]          # Top-left corner of the matched rectangle
    bottom_right: tuple[int, int]      # Bottom-right corner of the matched rectangle
    center: tuple[int, int]            # Center point of the matched rectangle


def _is_sqdiff_method(method: int) -> bool:
    """
    Tell whether the chosen OpenCV matching method belongs to the SQDIFF family.

    This matters because OpenCV's template-matching methods do not all interpret scores the same way:
    - SQDIFF methods treat lower values as better matches
    - other common methods treat higher values as better matches

    The rest of this module wants one consistent interpretation, so this helper is used by
    match_best() to decide how raw OpenCV output should be normalized.
    """
    # Return True if the matching method is one of the SQDIFF variants.
    # These methods are special because the best location is the minimum, not the maximum.
    return method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)


class TemplateMatcher:
    """
    Matching engine that compares loaded templates against the current prepared frame.

    The matcher itself is intentionally simple:
    it stores the chosen OpenCV matching method once during construction,
    then for each frame it loops through all templates, computes one best result per template,
    and finally keeps only the strongest overall result.

    This matches the project's baseline design:
    full-frame template matching, multiple possible templates, one final best candidate.
    """

    def __init__(self, method: int):
        """
        Store the OpenCV matching method that will be reused for all later matching calls.

        Keeping the method on the matcher instance avoids passing it around repeatedly and makes
        the runtime configuration from config.py part of the matcher's fixed operating mode.
        """
        # Store the selected OpenCV template-matching method inside the matcher object.
        self.method = method

    def match_best(self, frame: np.ndarray, templates: list[Template]) -> MatchResult | None:
        """
        Compare all templates against the prepared frame and return the strongest overall match.

        High-level logic:
        1. Start with no chosen result.
        2. For each template:
           - run cv2.matchTemplate(...)
           - extract the best location from the result map
           - normalize score handling so "higher is better" regardless of method family
           - compute rectangle corners and center
           - wrap everything into MatchResult
        3. Keep only the strongest MatchResult across all templates.
        4. Return that result, or None if the template list is empty.

        This method does not apply the controller's click threshold.
        It simply returns the strongest candidate that exists.
        Threshold-based acceptance happens later in controller.py.
        """
        # At the start of matching, nothing has been selected yet.
        # This variable will eventually hold the strongest match found across all templates.
        best_result = None

        # Try every loaded template and keep only the strongest result.
        for template in templates:
            # Compute the OpenCV match score map for the current frame/template pair.
            #
            # result_map is not a single score. It is a 2D map of scores over all possible template
            # positions in the frame. Each location answers the question:
            # "How well would this template fit if placed here?"
            result_map = cv2.matchTemplate(frame, template.image, self.method)

            # Extract both ends of the score spectrum and their locations.
            #
            # OpenCV always gives:
            # - min_val / min_loc
            # - max_val / max_loc
            #
            # Which one is the real "best" depends on the matching method family.
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_map)

            # Normalize method-specific behavior into one common project-wide interpretation:
            # "higher score is better".
            #
            # For SQDIFF methods:
            # - lower raw values are better
            # - min_loc is the correct best location
            # - score is inverted here into 1.0 - min_val so the rest of the system can still
            #   think in terms of larger-is-better scores
            #
            # For the other methods:
            # - higher raw values are already better
            # - max_loc is the correct best location
            if _is_sqdiff_method(self.method):
                score = 1.0 - float(min_val)
                top_left = min_loc
            else:
                score = float(max_val)
                top_left = max_loc

            # Compute the bottom-right corner of the matched rectangle by adding template size.
            # The top-left point comes from OpenCV; the template dimensions come from templates.py.
            bottom_right = (
                top_left[0] + template.width,
                top_left[1] + template.height,
            )

            # Compute the rectangle center.
            # This is the most operationally important geometric point because controller.py uses it
            # as the direct click target in baseline mode and as the recent-center input for optional
            # prediction mode.
            center = (
                top_left[0] + template.width // 2,
                top_left[1] + template.height // 2,
            )

            # Package the current template's best match into one structured result object.
            # This makes later comparison and downstream controller logic much clearer.
            current = MatchResult(
                template_name=template.name,
                score=score,
                top_left=top_left,
                bottom_right=bottom_right,
                center=center,
            )

            # Keep this result if:
            # - it is the first candidate encountered, or
            # - it is stronger than the current best candidate
            #
            # Because scores were normalized above, this comparison can always use the same
            # "greater than" rule regardless of the raw OpenCV method family.
            if best_result is None or current.score > best_result.score:
                best_result = current

        # After all templates have been evaluated, return the strongest one found.
        # If the template list was empty, no candidate exists and the result stays None.
        return best_result