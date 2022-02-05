import os
import sys

import h5py
from pathlib import Path

from oct_segmenter.preprocessing import generic_dataset as generator


def generate_test_dataset(
    test_input_dir: Path,
    output_file: Path,
    input_format: str="none"
) -> h5py.File:
    test_hdf5_file = generator.generate_generic_dataset(
        test_input_dir,
        output_file,
        input_format,
    )

    test_hdf5_file["test_images"] = test_hdf5_file["xhat"]
    test_hdf5_file["test_labels"] = test_hdf5_file["yhat"]
    test_hdf5_file["test_segs"] = test_hdf5_file["segs"]
    test_hdf5_file["test_images_source"] = test_hdf5_file["image_source"]

    del test_hdf5_file["xhat"]
    del test_hdf5_file["yhat"]
    del test_hdf5_file["segs"]
    del test_hdf5_file["image_source"]

    return test_hdf5_file
