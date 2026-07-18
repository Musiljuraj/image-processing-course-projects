"""
Template-loading module.

This module is responsible for turning template image files stored on disk into structured
in-memory objects that the rest of the project can use during runtime.

In the overall flow of the project, templates.py sits on the static-reference side of the
matching pipeline:

config.py
    -> defines which template file paths should be used
templates.py
    -> loads those files from disk
    -> converts them into Template objects containing image data and geometry
matcher.py
    -> compares those Template objects against the prepared live frame
controller.py
    -> loads templates once at startup and reuses them during the whole loop

So this module does not deal with live frames, bridge communication, or clicking.
Its job is to prepare the fixed reference images that represent what the bot is trying to find.

A key structural idea here is that template loading is separated from matching:
the files are read once during startup, converted into a clean structured form,
and then matcher.py works only with those already prepared Template objects.
"""

from dataclasses import dataclass  # dataclass = simple class mainly for holding related data
from pathlib import Path           # Path = modern object for filesystem paths

import cv2                         # OpenCV; used here to read template image files
import numpy as np                # NumPy; used here in the image type hint (np.ndarray)


@dataclass
class Template:
    """
    Structured in-memory representation of one loaded template image.

    Instead of passing around raw image arrays alone, the project stores each template as one
    small object containing both the image and the metadata that matcher.py needs to interpret it.

    This keeps all template-related information together:
    - name   -> human-readable identity of the template
    - image  -> actual pixel matrix used for matching
    - width  -> horizontal size of the template
    - height -> vertical size of the template

    The width and height are precomputed during loading so matcher.py can directly build
    bounding rectangles and center points without having to re-read image dimensions later.
    """
    name: str           # Human-readable template name, usually derived from filename without extension
    image: np.ndarray   # Loaded template image as a NumPy/OpenCV image array
    width: int          # Template width in pixels
    height: int         # Template height in pixels


def load_template(path: Path, use_gray: bool = True) -> Template:
    """
    Load one template image file from disk and convert it into a Template object.

    This is the single-template loader used by the bulk loader below.
    It performs three main steps:

    1. Choose the correct OpenCV image-reading mode based on grayscale configuration.
    2. Read the image file from disk.
    3. Extract image geometry and package everything into one Template object.

    The grayscale choice is important because the project keeps template loading and live-frame
    preparation aligned. If templates are loaded in grayscale, preprocessing.py should also
    convert live frames to grayscale before matcher.py compares them.
    """
    # Choose the OpenCV image-loading mode so the template format matches the runtime matching mode.
    #
    # If grayscale mode is enabled, the template is loaded as a single-channel image.
    # Otherwise it is loaded as a normal color image.
    flag = cv2.IMREAD_GRAYSCALE if use_gray else cv2.IMREAD_COLOR

    # Load the image file from disk using OpenCV.
    # Path is converted to string because cv2.imread expects a filesystem path in string form.
    image = cv2.imread(str(path), flag)

    # Stop immediately if the template could not be loaded.
    # Matching cannot continue with a missing or unreadable reference image, so this is treated
    # as a startup-time failure rather than something to silently ignore.
    if image is None:
        raise FileNotFoundError(f"Could not load template: {path}")

    # Extract image dimensions.
    #
    # image.shape is typically:
    # - (height, width) for grayscale images
    # - (height, width, channels) for color images
    #
    # Taking [:2] works for both cases and gives only the geometric dimensions needed later.
    height, width = image.shape[:2]

    # Build and return one structured Template object.
    #
    # name:
    #   taken from the filename stem so logs and match results can refer to a readable template name
    #
    # image:
    #   the actual loaded image matrix that matcher.py will compare against live frames
    #
    # width/height:
    #   stored explicitly so matcher.py can compute match rectangles and center points directly
    return Template(
        name=path.stem,   # Filename without extension, e.g. "dartboard_main"
        image=image,      # Loaded image matrix
        width=width,      # Width in pixels
        height=height,    # Height in pixels
    )


def load_templates(paths: list[Path], use_gray: bool = True) -> list[Template]:
    """
    Load multiple template files and return them as a list of Template objects.

    This is the startup-oriented bulk loader used by controller.py.
    The controller reads the configured template paths from config.py, calls this function once,
    and then reuses the returned Template objects for the whole runtime loop.

    So this function is the bridge between:
    - static template path configuration
    - usable in-memory template objects ready for matcher.py
    """
    # Load every configured template path through the single-template loader.
    #
    # Keeping the actual loading logic inside load_template(...) avoids duplication and ensures
    # that every template goes through exactly the same validation and packaging steps.
    return [load_template(path, use_gray=use_gray) for path in paths]