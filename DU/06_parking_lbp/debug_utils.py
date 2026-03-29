"""
debug_utils.py

Purpose of this module:
- provide helper utilities for visual debugging and saving intermediate outputs

Why this module exists:
Debug visualization is very useful during development, but it should not clutter
the main processing logic. By keeping these helpers here, main.py remains easier
to read and the project stays cleaner.

Current debug outputs:
1. full parking-lot image with parking-space polygons drawn
2. raw ROI patches extracted from one chosen test image
3. grayscale ROI patches derived from those ROIs
4. filtered ROI patches derived from those grayscale ROIs

This module currently provides:
- draw_parking_map(...)
- save_overlay_image(...)
- save_processed_patches(...)
"""
# ---------------------------------------------------------------------------
# Module orientation:
# This module sits beside the main pipeline and exists only for inspection and
# debugging outputs. It does not change predictions or evaluation results.
# Instead, it turns internal data structures such as parking-space polygons,
# ROI records, and processed image patches into files that can be looked at by
# a human. Throughout the project, the same record terminology is preserved:
# raw extraction produces ROI records, preprocessing extends them, and later
# stages may save one chosen image representation from those records.
# ---------------------------------------------------------------------------

from pathlib import Path

import cv2


def draw_parking_map(image, parking_map):
    """
    Draw all parking-space quadrilaterals on a copy of the given image.

    Input:
        image ......... one original parking-lot image
        parking_map ... list of parking-space polygons

    Return:
        vis ........... image copy with parking-space outlines and indices drawn

    Purpose:
    This function provides a visual reference linking the original full image
    with the parking-space map. It remains useful even after Step 2, because
    it helps verify:
    - map alignment
    - parking-space numbering
    - relationship between the full scene and later ROI patches
    """

    # Set up the local working state first so the later processing steps can operate on
    # explicit, well-named intermediate values.
    vis = image.copy()

    # Process items in a deterministic order so the produced outputs stay aligned with
    # the corresponding inputs, labels, or metadata.
    for i, pts in enumerate(parking_map, start=1):
        pts_int = pts.astype("int32")

        # draw polygon boundary
        cv2.polylines(
            vis,
            [pts_int],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

        # use the first point as a simple label position
        label_pos = tuple(pts_int[0])

        cv2.putText(
            vis,
            str(i),
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return vis


def save_overlay_image(image, output_path):
    """
    Save one debug overlay image to disk.

    Inputs:
        image ......... image to save
        output_path ... full destination path

    Purpose:
    Wrap cv2.imwrite(...) in a small helper that:
    - ensures the parent directory exists
    - raises a clear error if saving fails
    """

    # Convert incoming path-like inputs to Path objects at the start so all later
    # filesystem work uses one consistent path representation.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), image)
    # Guard the function boundary with explicit checks so invalid inputs are rejected
    # before they can silently corrupt later stages.
    if not ok:
        raise IOError(f"Could not save debug overlay image: {output_path}")


# =============================================================================
# generic saver for raw ROI patches, grayscale patches, filtered patches,
# and later any other image representation stored inside a record dictionary
# =============================================================================
def save_processed_patches(records, output_dir, image_key):
    """
    Save one selected image representation from each record to disk.

    Inputs:
        records ..... list of dictionaries
                      each record is expected to contain:
                      - "space_index"
                      - one image entry selected by image_key

        output_dir .. destination directory for saved images

        image_key ... dictionary key that selects which image array should be
                      written to disk, for example:
                      - "roi_image"
                      - "grayscale_image"
                      - "processed_image"

    Return:
        saved_paths .. list of full output paths written to disk

    Naming convention:
        space_01.jpg
        space_02.jpg
        ...
        space_56.jpg

    Why this function exists:
    Instead of having multiple nearly identical saving helpers such as:
    - save_roi_patches(...)
    - save_grayscale_patches(...)
    - save_filtered_patches(...)

    it is cleaner to use one generic function that can save any chosen image
    representation stored in the record dictionaries.

    Examples of use:
        save_processed_patches(rois, out_dir, "roi_image")
        save_processed_patches(preprocessed_rois, out_dir, "grayscale_image")
        save_processed_patches(preprocessed_rois, out_dir, "processed_image")
    """

    # Convert incoming path-like inputs to Path objects at the start so all later
    # filesystem work uses one consistent path representation.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # if there is nothing to save, return an empty list
    if not records:
        return []

    # determine how many digits to use in filenames
    # example:
    # 56 spaces  -> 2 digits
    # 120 spaces -> 3 digits
    digits = max(2, len(str(len(records))))

    # Start an accumulation structure that will be filled gradually as the function
    # walks through samples, records, rows, or files.
    saved_paths = []

    # Process the collection item by item, updating the running result structure as each
    # sample contributes its part of the final output.
    for record in records:
        if "space_index" not in record:
            raise KeyError(
                "Each record passed to save_processed_patches(...) must contain "
                "'space_index'."
            )

        if image_key not in record:
            raise KeyError(
                f"Record for space {record['space_index']} does not contain "
                f"requested image key: {image_key}"
            )

        image = record[image_key]

        if image is None:
            raise ValueError(
                f"Record for space {record['space_index']} contains None under "
                f"key '{image_key}'."
            )

        filename = f"space_{record['space_index']:0{digits}d}.jpg"
        output_path = output_dir / filename

        ok = cv2.imwrite(str(output_path), image)
        if not ok:
            raise IOError(f"Could not save image patch: {output_path}")

        saved_paths.append(output_path)

    # Return the finalized value only after all normalization, accumulation, and
    # packaging steps have established the expected public output form.
    return saved_paths