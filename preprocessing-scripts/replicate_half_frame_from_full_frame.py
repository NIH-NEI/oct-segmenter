import sys

import h5py
from pathlib import Path
import shutil


if __name__ == "__main__":
    """
    This script is used to copy into a directory the same "left" and "right"
    images from a dataset that contains full-frame images. This script was
    used to populate the "train", "val" and "test" directories which then
    were used in the "oct-segmenter generate" command.
    This script takes as inputs:
        - Path to the full-frame dataset (hdf5 file),
        - String that can be "train", "val" or "test". This tells which dataset
        within the HDF5 file we are building.
        - Directory containing the half-frame datasets.
        - Output directory to copy the half-frame images to.

    Example:
    python preprocessing-scripts/replicate_half_frame_from_full_frame.py \
        data/experiment-12/training_dataset.hdf5 \
        train \
        data/half-frame-dataset \
        data/half-frame-training/
    """
    dataset_path = Path(sys.argv[1])
    partition = sys.argv[2]
    half_frame_input_dir = Path(sys.argv[3])
    output_dir = Path(sys.argv[4])

    dataset = h5py.File(dataset_path, "r")
    image_paths = [
        Path(str(x, "ascii")) for x in dataset.get(partition + "_images_source")
    ]

    for image in image_paths:
        left_tiff_filename = Path(half_frame_input_dir) / Path(
            image.stem + "_left.tiff"
        )
        left_csv_filename = Path(half_frame_input_dir) / Path(image.stem + "_left.csv")
        right_tiff_filename = Path(half_frame_input_dir) / Path(
            image.stem + "_right.tiff"
        )
        right_csv_filename = Path(half_frame_input_dir) / Path(
            image.stem + "_right.csv"
        )

        if left_tiff_filename.is_file():
            shutil.copyfile(
                left_tiff_filename,
                Path(output_dir) / Path(image.stem + "_left.tiff"),
            )
        else:
            print(f"ERROR: File: {left_tiff_filename} not found")
            exit(1)

        if left_csv_filename.is_file():
            shutil.copyfile(
                left_csv_filename,
                Path(output_dir) / Path(image.stem + "_left.csv"),
            )
        else:
            print(f"ERROR: File: {left_csv_filename} not found")
            exit(1)

        if right_tiff_filename.is_file():
            shutil.copyfile(
                right_tiff_filename,
                Path(output_dir) / Path(image.stem + "_right.tiff"),
            )
        else:
            print(f"ERROR: File: {right_tiff_filename} not found")
            exit(1)

        if right_csv_filename.is_file():
            shutil.copyfile(
                right_csv_filename,
                Path(output_dir) / Path(image.stem + "_right.csv"),
            )
        else:
            print(f"ERROR: File: {right_csv_filename} not found")
            exit(1)
