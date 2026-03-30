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
# ---------------------------------------------------------------------------
# Module orientation:
# This module turns the global parking-lot scene into independent parking-space
# samples. It relies on the geometric transform defined in geometry.py and
# packages each extracted patch as an ROI record, which is the standard input
# form expected by preprocessing.py and later pipeline stages.
# ---------------------------------------------------------------------------
#
# This file is the test-side transition from:
# "one full image of the whole parking lot"
# to
# "many individual parking-space samples"
#
# That transition is one of the most important conceptual steps in the whole
# project. Up to this point, the test side still works with scene-level inputs:
# - one full parking-lot image
# - one ordered parking map describing all parking spaces
#
# After this module, the pipeline changes granularity completely:
# - one parking space becomes one ROI record
# - many ROI records become the sample collection for preprocessing
# - later stages no longer operate on the full scene directly
#
# The record vocabulary established here is reused across the rest of the
# project:
# - source_image_name ... which full-scene test image the patch came from
# - space_index ....... which parking-space position it corresponds to
# - polygon ........... which original quadrilateral defined that space
# - roi_image ......... the extracted normalized parking-space patch itself

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

    # This function is the smallest scene-to-sample conversion unit in the test
    # pipeline.
    #
    # It takes:
    # - one full parking-lot image
    # - one parking-space quadrilateral from the parking map
    # - one parking-space index
    # - one image name
    #
    # and produces:
    # - one structured ROI record representing exactly one parking space
    #
    # The actual geometric normalization is delegated to geometry.py, because this
    # module is responsible for the ROI-record concept, not for perspective math.

    # apply perspective transform to obtain a rectangular ROI patch
    # The source parking space appears as a skewed quadrilateral in the full scene.
    # four_point_transform(...) converts that quadrilateral into a normalized
    # rectangular patch that can later be treated as an ordinary image sample.
    roi_image = four_point_transform(image, parking_polygon)

    # Package the extracted patch together with the metadata needed by all later
    # stages. This is the canonical ROI-record structure used by preprocessing.py
    # and by the rest of the test-side feature pipeline.
    # Assemble the standard output dictionary here so downstream modules receive both
    # the computed values and the metadata needed for traceability.
    roi_record = {
        "source_image_name": image_name,
        "space_index": space_index,
        "polygon": parking_polygon,
        "roi_image": roi_image,
    }

    # Return the finalized single-ROI record.
    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
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

    # This function lifts the one-space extraction logic to the whole parking map.
    #
    # It preserves the order of parking_map exactly, which is crucial because that
    # ordering becomes the shared alignment used later for:
    # - feature-record order
    # - predicted-label order
    # - matching against ground-truth labels from the corresponding testX.txt file
    #
    # In other words, the ordering of ROI records created here is part of the
    # correctness contract of the whole evaluation pipeline.

    # Set up the local working state first so the later processing steps can operate on
    # explicit, well-named intermediate values.
    rois = []

    # enumerate parking spaces starting from 1,
    # because parking-space numbering is usually more natural in 1-based form
    #
    # Each parking polygon is extracted independently into its own ROI record.
    # The resulting list stays in parking-map order, which later lets evaluation
    # compare predicted parking-space labels against ground truth in the same order.
    for space_index, parking_polygon in enumerate(parking_map, start=1):
        roi_record = extract_one_roi(
            image=image,
            parking_polygon=parking_polygon,
            space_index=space_index,
            image_name=image_name,
        )
        rois.append(roi_record)

    # Return the full ordered ROI-record list for this source image.
    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return rois