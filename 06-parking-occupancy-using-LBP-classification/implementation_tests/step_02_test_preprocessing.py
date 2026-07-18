from pathlib import Path

from parking_io import load_parking_map, load_test_images
from roi_extraction import extract_all_rois_from_image
from preprocessing import preprocess_all_rois
from debug_utils import draw_parking_map, save_overlay_image, save_processed_patches


def main():
    project_root = Path(__file__).resolve().parent

    # Input paths
    map_path = project_root / "data" / "parking_map_python.txt"
    test_images_dir = project_root / "data" / "test_images_zao"

    # Output paths
    debug_dir = project_root / "outputs" / "debug"
    roi_dir = project_root / "outputs" / "roi" / "test1"
    preprocessed_dir = project_root / "outputs" / "preprocessed" / "test1"

    # Choose one preprocessing configuration for the smoke test
    preprocessing_config = {
        "target_size": (80, 80),
        "contrast_method": "clahe",
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": (8, 8),
        "filter_name": "gaussian",
        "kernel_size": 3,
    }

    # Load map and test images
    parking_map = load_parking_map(map_path)
    test_cases = load_test_images(test_images_dir)

    if not test_cases:
        raise ValueError("No test images were found.")

    # Use only the first test image for this smoke test
    test_case = test_cases[0]
    image_name = test_case["name"]
    image = test_case["image"]

    print(f"Loaded image: {image_name}")
    print(f"Parking spaces in map: {len(parking_map)}")

    # Save parking map overlay for visual confirmation
    overlay_image = draw_parking_map(image, parking_map)
    overlay_path = debug_dir / f"{image_name}_overlay.jpg"
    save_overlay_image(overlay_image, overlay_path)
    print(f"Saved overlay image: {overlay_path}")

    # Extract ROIs
    rois = extract_all_rois_from_image(
        image=image,
        parking_map=parking_map,
        image_name=image_name,
    )
    print(f"Extracted ROI count: {len(rois)}")

    # Save raw ROI images (optional but useful)
    save_processed_patches(
        records=rois,
        output_dir=roi_dir,
        image_key="roi_image",
    )
    print(f"Saved raw ROI images to: {roi_dir}")

    # Preprocess all ROIs
    preprocessed_rois = preprocess_all_rois(
        rois=rois,
        preprocessing_config=preprocessing_config,
    )
    print(f"Preprocessed ROI count: {len(preprocessed_rois)}")

    # Save a few examples from each stage
    sample_records = preprocessed_rois[:10]

    save_processed_patches(
        records=sample_records,
        output_dir=preprocessed_dir / "grayscale",
        image_key="grayscale_image",
    )

    save_processed_patches(
        records=sample_records,
        output_dir=preprocessed_dir / "resized",
        image_key="resized_image",
    )

    save_processed_patches(
        records=sample_records,
        output_dir=preprocessed_dir / "contrast_normalized",
        image_key="contrast_normalized_image",
    )

    save_processed_patches(
        records=sample_records,
        output_dir=preprocessed_dir / "processed",
        image_key="processed_image",
    )

    print(f"Saved preprocessing stage outputs to: {preprocessed_dir}")
    print("Smoke test finished successfully.")


if __name__ == "__main__":
    main()