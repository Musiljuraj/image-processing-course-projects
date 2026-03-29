from pathlib import Path

from parking_io import load_parking_map, load_test_images
from roi_extraction import extract_all_rois_from_image
from debug_utils import save_processed_patches, draw_parking_map, save_overlay_image


def main():
    # project root = folder where this script is located
    project_root = Path(__file__).resolve().parent

    # input paths
    map_path = project_root / "data" / "parking_map_python.txt"
    test_images_dir = project_root / "data" / "test_images_zao"

    # output paths
    overlay_output_path = project_root / "outputs" / "debug" / "test1_overlay.jpg"
    roi_output_dir = project_root / "outputs" / "roi" / "test1"

    # load map and test cases
    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(test_images_dir)

    # take only the first test image for the first milestone
    test_case = test_cases[0]

    image_name = test_case["name"]
    image = test_case["image"]

    print(f"Loaded image: {image_name}")
    print(f"Number of parking spaces in map: {len(parking_map)}")

    # save overlay for visual check
    overlay = draw_parking_map(image, parking_map)
    save_overlay_image(overlay, overlay_output_path)
    print(f"Saved overlay image to: {overlay_output_path}")

    # extract ROIs
    rois = extract_all_rois_from_image(
        image=image,
        parking_map=parking_map,
        image_name=image_name,
    )

    print(f"Extracted ROI count: {len(rois)}")

    # save raw ROI patches
    saved_paths = save_processed_patches(
        records=rois,
        output_dir=roi_output_dir,
        image_key="roi_image",
    )

    print(f"Saved {len(saved_paths)} ROI patches to: {roi_output_dir}")


if __name__ == "__main__":
    main()