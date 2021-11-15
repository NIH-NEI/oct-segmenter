import os
import sys

import h5py
from pathlib import Path

from oct_segmenter.preprocessing import generic_dataset as generator

test_hdf5_file = "/tmp/testing_dataset.hdf5"


def generate_test_dataset(test_input_dir, output_file, wayne_format=False):
    if wayne_format:
        generator.generate_hdf5_file_wayne(test_input_dir, test_hdf5_file)
    else:
        generator.generate_hdf5_file(test_input_dir, test_hdf5_file)

    output_file_path = Path(output_file)
    output_dir = output_file_path.parent
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_hf = h5py.File(output_file_path, "w")
    input_hf = h5py.File(test_hdf5_file, "r")

    output_hf.create_dataset("test_images", data=input_hf["xhat"])
    output_hf.create_dataset("test_labels", data=input_hf["yhat"])
    output_hf.create_dataset("test_images_source", data=input_hf["image_source"])
    output_hf.create_dataset("test_segs", data=input_hf["segs"])
    output_hf.close()

    os.remove(test_hdf5_file)
