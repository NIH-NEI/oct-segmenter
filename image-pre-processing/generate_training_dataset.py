import generate_dataset as generator
import h5py
import os
import sys

from pathlib import Path

training_hdf5_file = "/tmp/training_dataset.hdf5"
validation_hdf5_file = "/tmp/validation_dataset.hdf5"

dataset_name_1 = "train_images"
dataset_name_2 = "train_labels"
dataset_name_3 = "val_images"
dataset_name_4 = "val_labels"

def generate_datasets(train_input_dir, validation_input_dir, output_file):
    generator.generate_hdf5_file(train_input_dir, training_hdf5_file)
    generator.generate_hdf5_file(validation_input_dir, validation_hdf5_file)

    output_file_path = Path(output_file)
    output_dir = output_file_path.parent
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_hf = h5py.File(output_file_path, "w")

    input_hf_1 = h5py.File(training_hdf5_file, "r")
    input_hf_2 = h5py.File(validation_hdf5_file, "r")

    output_hf.create_dataset(dataset_name_1, data=input_hf_1["xhat"])
    output_hf.create_dataset(dataset_name_2, data=input_hf_1["yhat"])
    output_hf.create_dataset(dataset_name_1 + "_source", data=input_hf_1["image_source"])
    
    output_hf.create_dataset(dataset_name_3, data=input_hf_2["xhat"])
    output_hf.create_dataset(dataset_name_4, data=input_hf_2["yhat"])
    output_hf.create_dataset(dataset_name_3 + "_source", data=input_hf_2["image_source"])
    output_hf.close()

    os.remove(training_hdf5_file)
    os.remove(validation_hdf5_file)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_training_dataset.py <path/to/training/input/dir> <path/to/validation/input/dir> <output_file_name>")

    train_input_dir = sys.argv[1]
    validation_input_dir = sys.argv[2]
    output_file = sys.argv[3]

    generate_datasets(train_input_dir, validation_input_dir, output_file)