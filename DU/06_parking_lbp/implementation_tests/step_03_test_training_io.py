from pathlib import Path

import cv2

from parking_training_io import (
    load_all_training_records,
    summarize_training_records,
)


def save_debug_image(image, output_path):
    """
    Save one image to disk and raise a clear error if saving fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise IOError(f"Could not save debug image: {output_path}")


def find_first_record_by_class(training_records, class_name):
    """
    Return the first record with the requested class_name.
    """
    for record in training_records:
        if record["class_name"] == class_name:
            return record
    return None


def main():
    project_root = Path(__file__).resolve().parent

    training_root = project_root / "data" / "training"
    debug_output_dir = project_root / "outputs" / "debug" / "training_io"

    print("=== STEP 3 SMOKE TEST: parking_training_io ===")
    print(f"Training root: {training_root}")

    # Load all training records
    training_records = load_all_training_records(training_root)

    # Summarize dataset
    summary = summarize_training_records(training_records)

    print("\nDataset summary:")
    print(f"  Total samples : {summary['total_count']}")
    print(f"  Free samples  : {summary['free_count']}")
    print(f"  Full samples  : {summary['full_count']}")
    print(f"  Classes found : {summary['class_names_present']}")
    print(f"  Labels found  : {summary['labels_present']}")

    if not training_records:
        raise ValueError("No training records were loaded.")

    # Print a few example records
    print("\nFirst 5 loaded records:")
    for index, record in enumerate(training_records[:5], start=1):
        print(
            f"  {index}. "
            f"class_name={record['class_name']}, "
            f"label={record['label']}, "
            f"file_name={record['file_name']}, "
            f"image_shape={record['image'].shape}"
        )

    # Save one example image from each class
    free_record = find_first_record_by_class(training_records, "free")
    full_record = find_first_record_by_class(training_records, "full")

    if free_record is None:
        raise ValueError("Could not find any 'free' training record.")

    if full_record is None:
        raise ValueError("Could not find any 'full' training record.")

    free_output_path = debug_output_dir / "free_example.jpg"
    full_output_path = debug_output_dir / "full_example.jpg"

    save_debug_image(free_record["image"], free_output_path)
    save_debug_image(full_record["image"], full_output_path)

    print("\nSaved example images:")
    print(f"  Free example : {free_output_path}")
    print(f"  Full example : {full_output_path}")

    print("\nSmoke test finished successfully.")


if __name__ == "__main__":
    main()