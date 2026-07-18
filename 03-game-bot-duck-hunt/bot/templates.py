"""
Template loader. Transform image files on disk into structured in-memory objects ready for matching.
"""

from dataclasses import dataclass  # dataclass = simple class mainly for holding related data
from pathlib import Path           # Path = modern object for filesystem paths

import cv2                         # OpenCV; used here to read template image files
import numpy as np                # NumPy; used here in the image type hint (np.ndarray)


@dataclass
class Template:
    """
    Dataclass storing template name, image, width, and height.
    """
    name: str           # Human-readable template name, usually derived from filename without extension
    image: np.ndarray   # Loaded template image as a NumPy/OpenCV image array
    width: int          # Template width in pixels
    height: int         # Template height in pixels


def load_template(path: Path, use_gray: bool = True) -> Template:
    # Choose OpenCV image-loading mode:
    # - grayscale if use_gray is True
    # - color otherwise
    flag = cv2.IMREAD_GRAYSCALE if use_gray else cv2.IMREAD_COLOR

    # Load the image file from disk using OpenCV.
    # Path is converted to string because cv2.imread expects a path string.
    image = cv2.imread(str(path), flag)

    # If OpenCV could not load the file, stop immediately with a clear error.
    if image is None:
        raise FileNotFoundError(f"Could not load template: {path}")

    # Extract image dimensions.
    # image.shape is usually:
    # - (height, width) for grayscale
    # - (height, width, channels) for color
    # Taking [:2] gives height and width only.
    height, width = image.shape[:2]

    # Build and return a structured Template object.
    return Template(
        name=path.stem,   # Filename without extension, e.g. "dartboard_main"
        image=image,      # Loaded image matrix
        width=width,      # Width in pixels
        height=height,    # Height in pixels
    )


def load_templates(paths: list[Path], use_gray: bool = True) -> list[Template]:
    # Load all template paths using the single-template loader
    # and return them as a list of Template objects.
    return [load_template(path, use_gray=use_gray) for path in paths]




