import cv2
import numpy as np

"""
Image preparation step. Converts BGR to grayscale. 
Either returns grayscale or a copy of the original frame, depending on configuration.
"""

def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def prepare_frame(frame_bgr: np.ndarray, use_gray: bool = True) -> np.ndarray:
    if use_gray:
        return to_gray(frame_bgr)
    return frame_bgr.copy()