import generate_dataset as generator
import h5py
import os
import sys

from pathlib import Path

test_hdf5_file = "/tmp/testing_dataset.hdf5"
dataset_name_1 = "test_images"
dataset_name_2 = "test_labels"

def generate_test_dataset(test_input_dir, output_file):
    generator.generate_hdf5_file(test_input_dir, test_hdf5_file)

    output_file_path = Path(output_file)
    output_dir = output_file_path.parent
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_hf = h5py.File(output_file_path, "w")
    input_hf = h5py.File(test_hdf5_file, "r")

    output_hf.create_dataset(dataset_name_1, data=input_hf["xhat"])
    output_hf.create_dataset(dataset_name_2, data=input_hf["yhat"])
    output_hf.create_dataset(dataset_name_1 + "_source", data=input_hf["image_source"])
    output_hf.close()

    os.remove(test_hdf5_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_test_dataset.py <path/to/test/input/dir> <output_file_name>")

    input_test_dir = sys.argv[1]
    output_file_path = sys.argv[2]
    generate_test_dataset(input_test_dir, output_file_path)

