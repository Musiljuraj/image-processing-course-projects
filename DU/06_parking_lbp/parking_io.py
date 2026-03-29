"""
parking_io.py

Purpose of this module:
- load and parse the parking-space map file
- load and sort the test images
- load and parse the ground-truth label files
- keep all file-reading responsibilities in one place

Why this module exists:
When the project grows, it is useful to separate input/output and file parsing
from geometry, ROI extraction, preprocessing, edge detection, and evaluation.

This module currently provides:
- parse_parking_line(...)
- load_parking_map(...)
- image_number_key(...)
- load_test_images(...)
- parse_ground_truth_labels(...)
- load_ground_truth_labels(...)
"""

from pathlib import Path
import re

import cv2
import numpy as np


def parse_parking_line(line, line_no):
    """
    Parse one line from parking_map_python.txt.

    Expected line structure:
        x1 y1 x2 y2 x3 y3 x4 y4

    Meaning:
        (x1, y1) = first corner of one parking place
        (x2, y2) = second corner
        (x3, y3) = third corner
        (x4, y4) = fourth corner

    So each valid line contains:
    - 8 numbers total
    - 4 image points
    - 1 parking-space quadrilateral

    Example line:
        149 699 228 705 141 879 61 864

    Parsed result:
        [[149., 699.],
         [228., 705.],
         [141., 879.],
         [ 61., 864.]]

    Return value:
        NumPy array of shape (4, 2)

    Notes:
    - empty lines are allowed and skipped
    - malformed lines raise a clear error
    """

    # remove leading and trailing whitespace
    stripped = line.strip()

    # allow empty lines to be skipped safely
    if not stripped:
        return None

    # split by whitespace and convert all items to integers
    values = list(map(int, stripped.split()))

    # each valid parking-space definition must have exactly 8 numbers
    if len(values) != 8:
        raise ValueError(
            f"Line {line_no} in parking_map_python.txt does not contain 8 numbers."
        )

    # convert the flat list:
    # [x1, y1, x2, y2, x3, y3, x4, y4]
    # into a (4, 2) array:
    # [[x1, y1],
    #  [x2, y2],
    #  [x3, y3],
    #  [x4, y4]]
    points = np.array(values, dtype="float32").reshape(4, 2)

    return points


def load_parking_map(map_path):
    """
    Load all parking-space definitions from parking_map_python.txt.

    Input:
        map_path ... full path to data/parking_map_python.txt

    Return:
        parking_map ... list of NumPy arrays
                        each element has shape (4, 2)
                        each element describes one parking space

    Important idea:
    The map file contains one line per parking place.
    Therefore:
    - number of non-empty lines in the file
    - number of parking-space polygons returned here
    should match.
    """

    parking_map = []

    with open(map_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            points = parse_parking_line(line, line_no)

            if points is None:
                continue

            parking_map.append(points)

    return parking_map


def image_number_key(path):
    """
    Extract the numeric part from filenames such as:
        test1.jpg, test2.jpg, test10.jpg

    Why this helper exists:
    Normal string sorting would produce:
        test1.jpg, test10.jpg, test11.jpg, test2.jpg, ...
    which is not the desired natural order.

    With this helper, sorting becomes:
        test1.jpg, test2.jpg, test3.jpg, ..., test10.jpg, ...

    Input:
        path ... pathlib.Path object

    Return:
        integer extracted from filename if possible,
        otherwise fallback to raw filename
    """

    match = re.search(r"test(\d+)\.jpg$", path.name)

    if match:
        return int(match.group(1))

    return path.name


def load_test_images(images_dir):
    """
    Load all test images from data/test_images_zao.

    For every .jpg file, this function also checks that a matching .txt file
    exists in the same directory. The .txt file will be needed later as
    ground truth for evaluation.

    Input:
        images_dir ... full path to data/test_images_zao

    Return:
        test_cases ... list of dictionaries
                       each dictionary contains:
                       - name       ... image stem, e.g. "test1"
                       - image_path ... full path to .jpg file
                       - txt_path   ... full path to matching .txt file
                       - image      ... image loaded by cv2.imread(...)

    Why this structure is useful:
    Later parts of the project need both:
    - the image pixels
    - the metadata about where the image and ground truth came from
    """

    # find all JPG images and sort them in natural numeric order
    image_paths = sorted(images_dir.glob("*.jpg"), key=image_number_key)

    test_cases = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # expected ground-truth text file
        # example:
        # test1.jpg -> test1.txt
        txt_path = image_path.with_suffix(".txt")

        if not txt_path.exists():
            raise FileNotFoundError(
                f"Missing ground-truth file for image {image_path.name}: {txt_path.name}"
            )

        test_cases.append(
            {
                "name": image_path.stem,
                "image_path": image_path,
                "txt_path": txt_path,
                "image": image,
            }
        )

    return test_cases


def parse_ground_truth_labels(text, txt_path="<unknown>"):
    """
    Parse the content of one ground-truth .txt file.

    Inputs:
        text ..... raw file content as one string
        txt_path . optional path used only for clearer error messages

    Return:
        labels ... list of integers

    Expected format:
    The parser is intentionally flexible with respect to line layout:
    - labels may appear one per line
    - or multiple labels may appear on the same line
    - any whitespace separation is accepted

    Example accepted formats:
        0 1 0 1 1
    or
        0
        1
        0
        1
        1

    Why this helper exists:
    It keeps raw text parsing separate from file opening. This makes the logic
    easier to test and easier to reuse if needed.
    """

    if not isinstance(text, str):
        raise TypeError("Ground-truth file content must be a string.")

    # normalize commas to spaces just in case the file uses them as separators
    normalized_text = text.replace(",", " ").strip()

    if not normalized_text:
        raise ValueError(f"Ground-truth file is empty: {txt_path}")

    tokens = normalized_text.split()

    labels = []
    for token in tokens:
        try:
            labels.append(int(token))
        except ValueError as exc:
            raise ValueError(
                f"Ground-truth file contains a non-integer token '{token}': {txt_path}"
            ) from exc

    return labels


def load_ground_truth_labels(txt_path):
    """
    Load the ground-truth labels from one testX.txt file.

    Input:
        txt_path ... full path to the corresponding .txt file

    Return:
        labels ... list of integer labels, one per parking space

    Why this function exists:
    The later evaluation stage should work with ready-to-use label lists, not
    raw file content. Since parking_io.py already owns dataset loading, this is
    the correct place to add ground-truth parsing.
    """

    txt_path = Path(txt_path)

    if not txt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    labels = parse_ground_truth_labels(text, txt_path=txt_path)

    return labels
