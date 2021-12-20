import os
import sys

import h5py
from pathlib import Path

from oct_segmenter.preprocessing.image_labeling import generate_image_label, generate_image_label_wayne


def process_directory(input_dir, output_dir, save_file=False):
    img_file_names = []
    img_file_data = [] # Original image (xhat)
    labeled_file_data = [] # Segmenation map (yhat)
    segments_data = [] # Contains the boundaries

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".tiff") or file.endswith(".TIFF") and not file.startswith("."):
                image_file = Path(os.path.join(subdir, file))
                print(f"Processing file: {image_file}")
                img_name_left, img_array_left, seg_map_left, segs_left, img_name_right, img_array_right, seg_map_right, segs_right = generate_image_label(image_file, output_dir, save_file)
                if img_file_data:
                    if img_file_data[0].shape == img_array.shape:
                        img_file_names.extend([[img_name_left, "left".encode("ascii")], [img_name_right, "right".encode("ascii")]])
                        img_file_data.extend([img_array_left, img_array_right])
                        labeled_file_data.extend([seg_map_left, seg_map_right])
                        segments_data.extend([segs_left, segs_right])
                    else:
                        print(f"WARNING: Image {img_name} has size {img_array.shape}. Different from other dataset samples of size: {img_file_data[0].shape}.")
                else:
                    img_file_names.extend([[img_name_left, "left".encode("ascii")], [img_name_right, "right".encode("ascii")]])
                    img_file_data.extend([img_array_left, img_array_right])
                    labeled_file_data.extend([seg_map_left, seg_map_right])
                    segments_data.extend([segs_left, segs_right])

    return img_file_names, img_file_data, segments_data, labeled_file_data


def process_directory_wayne(input_dir, output_dir, save_file=False):
    img_file_names = []
    img_file_data = [] # Original image (xhat)
    labeled_file_data = [] # Segmenation map (yhat)
    segments_data = [] # Contains the boundaries

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if (file.endswith(".tiff") or file.endswith(".TIFF")) and not file.startswith("."):
                image_file = Path(os.path.join(subdir, file))
                print(f"Processing file from wayne: {image_file}")
                img_name, img_array, seg_map, segs = generate_image_label_wayne(image_file, output_dir, save_file)
                if img_file_data:
                    if img_file_data[0].shape == img_array.shape:
                        img_file_names.extend([img_name])
                        img_file_data.extend([img_array])
                        labeled_file_data.extend([seg_map])
                        segments_data.extend([segs])
                    else:
                        print(f"WARNING: Image {img_name} has size {img_array.shape}. Different from other dataset samples of size: {img_file_data[0].shape}.")
                else:
                    img_file_names.extend([img_name])
                    img_file_data.extend([img_array])
                    labeled_file_data.extend([seg_map])
                    segments_data.extend([segs])

    return img_file_names, img_file_data, segments_data, labeled_file_data


def generate_generic_dataset(
    input_dir: Path,
    file_name: Path,
    wayne_format: bool,
    backing_store: bool=True,
) -> h5py.File:
    if not os.path.isdir(file_name.parent):
        os.makedirs(file_name.parent)

    hf = h5py.File(file_name, "w", driver="core", backing_store=backing_store)

    if wayne_format:
        img_file_names, img_file_data, segments_data, labeled_file_data = process_directory_wayne(input_dir, str(file_name.parent), save_file=False)
    else:
        img_file_names, img_file_data, segments_data, labeled_file_data = process_directory(input_dir, str(file_name.parent), save_file=False)

    hf.create_dataset("xhat", data=img_file_data)
    hf.create_dataset("yhat", data=labeled_file_data)
    hf.create_dataset("segs", data=segments_data)
    hf.create_dataset("image_source", data=img_file_names)

    return hf
