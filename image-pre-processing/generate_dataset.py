import h5py
import os
import preprocess
import sys

from pathlib import Path

def process_directory(input_dir, output_dir, save_file=False):
    img_file_names = []
    img_file_data = [] # Original image (xhat)
    labeled_file_data = [] # Segmenation map (yhat)

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".tiff") or file.endswith(".TIFF"):
                image_file = Path(os.path.join(subdir, file))
                print(f"Processing file: {image_file}")
                img_name_left, img_array_left, seg_map_left, img_name_right, img_array_right, seg_map_right = preprocess.process_image(image_file, output_dir, save_file)
                img_file_names.extend([[img_name_left, "left".encode("ascii")], [img_name_right, "right".encode("ascii")]])
                img_file_data.extend([img_array_left, img_array_right])
                labeled_file_data.extend([seg_map_left, seg_map_right])
    return img_file_names, img_file_data, labeled_file_data

def generate_hdf5_file(input_dir, file_name, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    file_path = Path(output_dir + "/" + file_name + ".hdf5")
    hf = h5py.File(file_path, "w")
    img_file_names, img_file_data, labeled_file_data = process_directory(input_dir, output_dir, save_file=False)
    hf.create_dataset("xhat", data=img_file_data)
    hf.create_dataset("yhat", data=labeled_file_data)
    hf.create_dataset("image_source", data=img_file_names)
    hf.close()

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python generate_dataset.py </path/to/input/dir> <HDF5 file name> </path/to/output/dir>")
        exit(1)

    input_dir = sys.argv[1]
    hdf5_file_name = sys.argv[2]
    file_ouptput_dir = sys.argv[3]
    generate_hdf5_file(input_dir, hdf5_file_name, file_ouptput_dir)