"""
Matching engine.  
Defines MatchResult - dataclass containing the chosen template name, score, bounding box, center, and winning scale.
Defines class TemplateMatcher with method match_best(...), which loops through all templates and configured scales,
runs cv2.matchTemplate, extracts the best location, computes rectangle and center from the scaled template size,
creates a MatchResult, and returns the best overall match.
"""

from dataclasses import dataclass  # dataclass = lightweight class mainly for holding related data

import cv2                         # OpenCV; provides template matching and min/max-location tools
import numpy as np                # NumPy; used here for the frame type hint (np.ndarray)
from bot.templates import Template  # Structured template object loaded by templates.py


@dataclass
class MatchResult:
    """
    Dataclass containing the chosen template name, score, bounding box, center, and winning scale.  # CHANGED
    """
    template_name: str                 # Name of the template that produced this match
    score: float                       # Match score; higher is treated as better
    top_left: tuple[int, int]          # Top-left corner of the matched rectangle
    bottom_right: tuple[int, int]      # Bottom-right corner of the matched rectangle
    center: tuple[int, int]            # Center point of the matched rectangle
    scale: float                       # Scale at which this template won
    matched_width: int                 # Final scaled template width used for this match
    matched_height: int                # Final scaled template height used for this match


def _is_sqdiff_method(method: int) -> bool:
    # Helper function:
    # return True if the matching method belongs to the SQDIFF family.
    # These methods are handled differently because the code takes min_loc
    # instead of max_loc and converts score to 1.0 - min_val.
    return method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)


def _resize_template_for_scale( 
    template: Template,         
    scale: float,               
) -> tuple[np.ndarray, int, int] | None:
    """
    Return a resized template image and its scaled width/height for one scale.
    Return None if the scale produces an invalid size.
    """
    if scale <= 0:
        return None

    scaled_width = int(round(template.width * scale))
    scaled_height = int(round(template.height * scale))

    if scaled_width < 1 or scaled_height < 1:
        return None 

    # Reuse the original template image when the scale is exactly 1.0-sized after rounding.
    if scaled_width == template.width and scaled_height == template.height:
        return template.image, template.width, template.height 

    # Use INTER_AREA when shrinking and INTER_LINEAR when enlarging.
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR 
    scaled_image = cv2.resize(
        template.image,      
        (scaled_width, scaled_height),
        interpolation=interpolation,
    )

    return scaled_image, scaled_width, scaled_height 


class TemplateMatcher:
    """
    Compares templates against the current frame using OpenCV template matching
    across multiple configured scales.
    """

    def __init__(self, method: int, scales: tuple[float, ...] | list[float]): 
        # Store the OpenCV matching method inside the matcher object.
        # This makes the matcher configurable while keeping match_best() simpler.
        self.method = method

        # Store positive scales only. If nothing valid is provided, fall back to 1.0. 
        cleaned_scales = tuple(float(scale) for scale in scales if float(scale) > 0.0)
        self.scales = cleaned_scales if cleaned_scales else (1.0,)

    def match_best(self, frame: np.ndarray, templates: list[Template]) -> MatchResult | None:
        """
        Loops through all templates and configured scales,
        runs cv2.matchTemplate, extracts the best location, computes rectangle and center,
        creates a MatchResult, and returns the best overall match.
        """
        # At the start, we have not found any candidate yet.
        best_result = None

        # Frame dimensions are needed so we can skip scaled templates
        # that are larger than the frame.
        frame_height, frame_width = frame.shape[:2] 

        # Try every template and every configured scale,
        # then keep only the strongest overall match.
        for template in templates:
            for scale in self.scales: 
                resized = _resize_template_for_scale(template, scale)
                if resized is None:
                    continue

                scaled_image, scaled_width, scaled_height = resized

                # Skip impossible cases where the scaled template is larger than the frame.
                if scaled_width > frame_width or scaled_height > frame_height:
                    continue

                # Compute the OpenCV template-match score map for this frame/template-scale pair.
                result_map = cv2.matchTemplate(frame, scaled_image, self.method)

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
                # by adding the SCALED template size to the top-left point. 
                bottom_right = (
                    top_left[0] + scaled_width,  
                    top_left[1] + scaled_height,  
                )

                # Compute the rectangle center from the SCALED template size. 
                center = (
                    top_left[0] + scaled_width // 2,   
                    top_left[1] + scaled_height // 2, 
                )

                # Package the current template-scale match into one structured result object. 
                current = MatchResult(
                    template_name=template.name,
                    score=score,
                    top_left=top_left,
                    bottom_right=bottom_right,
                    center=center,
                    scale=float(scale),          
                    matched_width=scaled_width,  
                    matched_height=scaled_height,  
                )

                # If this is the first result, or if it is stronger than the current best,
                # replace best_result with this one.
                if best_result is None or current.score > best_result.score:
                    best_result = current

        # After all template-scale pairs are checked, return the strongest match found.
        # If templates was empty, or no valid scaled template fit inside the frame, this remains None. 
        return best_result