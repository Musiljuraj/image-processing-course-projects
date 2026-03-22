"""
Matching engine.
Defines MatchResult - dataclass containing the chosen template name, score, bounding box, and center.
Defines class TemplateMatcher with method match_best(...), which loops through all templates, 
runs cv2.matchTemplate, extracts the best location, computes rectangle and center, 
creates a MatchResult, and returns the best overall match.
"""

from dataclasses import dataclass  # dataclass = lightweight class mainly for holding related data

import cv2                         # OpenCV; provides template matching and min/max-location tools
import numpy as np                # NumPy; used here for the frame type hint (np.ndarray)
from bot.templates import Template  # Structured template object loaded by templates.py


@dataclass
class MatchResult:
    """
    Dataclass containing the chosen template name, score, bounding box, and center.
    """
    template_name: str                 # Name of the template that produced this match
    score: float                       # Match score; higher is treated as better
    top_left: tuple[int, int]          # Top-left corner of the matched rectangle
    bottom_right: tuple[int, int]      # Bottom-right corner of the matched rectangle
    center: tuple[int, int]            # Center point of the matched rectangle


def _is_sqdiff_method(method: int) -> bool:
    # Helper function:
    # return True if the matching method belongs to the SQDIFF family.
    # These methods are handled differently because the code takes min_loc
    # instead of max_loc and converts score to 1.0 - min_val.
    return method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)


class TemplateMatcher:
    """
    Compares dart-board templates (optionally multiple of them) against the current frame using OpenCV template matching.
    """

    def __init__(self, method: int):
        # Store the OpenCV matching method inside the matcher object.
        # This makes the matcher configurable while keeping match_best() simpler.
        self.method = method

    def match_best(self, frame: np.ndarray, templates: list[Template]) -> MatchResult | None:
        """
        Loops through all templates,
        runs cv2.matchTemplate, extracts the best location, computes rectangle and center,
        creates a MatchResult, and returns the best overall match.
        """
        # At the start, we have not found any candidate yet.
        best_result = None

        # Try every template and keep only the strongest match.
        for template in templates:
            # Compute the OpenCV template-match score map for this frame/template pair.
            result_map = cv2.matchTemplate(frame, template.image, self.method)

            # Find minimum and maximum values in the score map and where they occur.
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_map)

            # Some OpenCV methods treat the minimum as the best location (SQDIFF family).
            # Others treat the maximum as the best location.
            # This branch converts both cases into a unified "higher score is better" form.
            if _is_sqdiff_method(self.method):
                score = 1.0 - float(min_val)
                top_left = min_loc
            else:
                score = float(max_val)
                top_left = max_loc

            # Compute the bottom-right corner of the matched rectangle
            # by adding the template size to the top-left point.
            bottom_right = (
                top_left[0] + template.width,
                top_left[1] + template.height,
            )

            # Compute the rectangle center.
            # Integer division (// 2) keeps the coordinates as integer pixel values.
            center = (
                top_left[0] + template.width // 2,
                top_left[1] + template.height // 2,
            )

            # Package the current template's best match into one structured result object.
            current = MatchResult(
                template_name=template.name,
                score=score,
                top_left=top_left,
                bottom_right=bottom_right,
                center=center,
            )

            # If this is the first result, or if it is stronger than the current best,
            # replace best_result with this one.
            if best_result is None or current.score > best_result.score:
                best_result = current

        # After all templates are checked, return the strongest match found.
        # If templates was empty, this remains None.
        return best_result