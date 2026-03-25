"""
roi_extraction.py

Purpose of this module:
- extract parking-space ROI patches from one parking-lot image
- keep ROI logic separate from geometry, file loading, and debug output

This module currently provides:
- extract_one_roi(...)
- extract_all_rois_from_image(...)

Why this module exists:
Step 2 introduces a new central concept in the project:
one parking place should become one independent ROI patch.

That is a distinct processing phase, so it deserves its own module.
"""

from geometry import four_point_transform


def extract_one_roi(image, parking_polygon, space_index, image_name):
    """
    Extract one parking-space ROI from the given image.

    Inputs:
        image ............ original parking-lot image
        parking_polygon .. one parking-space polygon of shape (4, 2)
        space_index ...... 1-based parking-space index
        image_name ....... source image name, e.g. "test1"

    Return:
        roi_record ....... dictionary containing:
                           - source_image_name
                           - space_index
                           - polygon
                           - roi_image

    Why return a dictionary instead of only the patch image:
    Later stages of the project will likely need metadata together with the ROI.
    For example:
    - which parking place it came from
    - which source image it belongs to
    - which original polygon produced it
    """

    # apply perspective transform to obtain a rectangular ROI patch
    roi_image = four_point_transform(image, parking_polygon)

    roi_record = {
        "source_image_name": image_name,
        "space_index": space_index,
        "polygon": parking_polygon,
        "roi_image": roi_image,
    }

    return roi_record


def extract_all_rois_from_image(image, parking_map, image_name):
    """
    Extract all parking-space ROIs from one source image.

    Inputs:
        image ......... original parking-lot image
        parking_map ... list of parking-space polygons
        image_name .... source image name, e.g. "test1"

    Return:
        rois .......... list of ROI records
                        one record per parking space

    Overall idea:
    One large parking-lot image is transformed into many small parking-space
    patches. This is the key output of Step 2.

    Logical effect:
    - before: one image containing the whole scene
    - after: one list of independent ROI patches, one per parking space
    """

    rois = []

    # enumerate parking spaces starting from 1,
    # because parking-space numbering is usually more natural in 1-based form
    for space_index, parking_polygon in enumerate(parking_map, start=1):
        roi_record = extract_one_roi(
            image=image,
            parking_polygon=parking_polygon,
            space_index=space_index,
            image_name=image_name,
        )
        rois.append(roi_record)

    return rois