import h5py
import os
import sys

from pathlib import Path


def merge_datasets(first_hdf5, second_hdf5, dataset_name_1, dataset_name_2, dataset_name_3, dataset_name_4, output_file_name, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_file_path = Path(output_dir + "/" + output_file_name + ".hdf5")
    output_hf = h5py.File(output_file_path, "w")

    input_hf_1 = h5py.File(first_hdf5, "r")
    input_hf_2 = h5py.File(second_hdf5, "r")


    output_hf.create_dataset(dataset_name_1, data=input_hf_1["xhat"])
    output_hf.create_dataset(dataset_name_2, data=input_hf_1["yhat"])
    output_hf.create_dataset(dataset_name_1 + "_source", data=input_hf_1["image_source"])
    
    output_hf.create_dataset(dataset_name_3, data=input_hf_2["xhat"])
    output_hf.create_dataset(dataset_name_4, data=input_hf_2["yhat"])
    output_hf.create_dataset(dataset_name_3 + "_source", data=input_hf_2["image_source"])
    output_hf.close()


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python merge_datasets.py <path/to/first/hdf5> <path/to/second/hdf5> <dataset name 1> <dataset name 2> \
            <dataset name 3> <dataset name 4> <output_file_name> <output file dir>")
    first_hdf5 = sys.argv[1]
    second_hdf5 = sys.argv[2]
    dataset_name_1 = sys.argv[3]
    dataset_name_2 = sys.argv[4]
    dataset_name_3 = sys.argv[5]
    dataset_name_4 = sys.argv[6]
    output_file = sys.argv[7]
    output_dir = sys.argv[8]

    merge_datasets(first_hdf5, second_hdf5, dataset_name_1, dataset_name_2, dataset_name_3, dataset_name_4, output_file, output_dir)