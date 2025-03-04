from __future__ import annotations

import os

import h5py
import logging as log
import math
from pathlib import Path
from typing import List, Optional
from typeguard import typechecked

from oct_segmenter.preprocessing.image_labeling_labelme import (
    generate_image_label_labelme,
)
from oct_segmenter.preprocessing.image_labeling_mask import (
    generate_image_label_mask,
)
from oct_segmenter.preprocessing.image_labeling_visual_core import (
    generate_image_label_visual_core,
)
from oct_segmenter.preprocessing.image_labeling_wayne import (
    generate_image_label_wayne,
)


def process_directory(input_dir, output_dir, save_file=False):
    img_file_names = []
    img_file_data = []  # Original image (xhat)
    labeled_file_data = []  # Segmenation map (yhat)
    segments_data = []  # Contains the boundaries

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if (
                file.endswith(".tiff")
                or file.endswith(".TIFF")
                and not file.startswith(".")
            ):
                image_file = Path(os.path.join(subdir, file))
                print(f"Processing file from Visual Core: {image_file}")
                (
                    img_name_left,
                    img_array_left,
                    seg_map_left,
                    segs_left,
                    img_name_right,
                    img_array_right,
                    seg_map_right,
                    segs_right,
                ) = generate_image_label_visual_core(image_file, output_dir, save_file)
                if img_name_left:
                    img_file_names.extend(
                        [
                            [img_name_left, "left".encode("ascii")],
                            [img_name_right, "right".encode("ascii")],
                        ]
                    )
                    img_file_data.extend([img_array_left, img_array_right])
                    labeled_file_data.extend([seg_map_left, seg_map_right])
                    segments_data.extend([segs_left, segs_right])

    if len(img_file_data) == 0:
        log.info("No images were processed successfully. Exiting...")
        exit(1)

    # Images a transposed so dim 0 is image width
    if not all(x.shape[0] == img_file_data[0].shape[0] for x in img_file_data):
        log.error(
            "Images contain different widths. All images should have same "
            "width. Exiting..."
        )
        exit(1)

    if not all(x.shape[1] == img_file_data[0].shape[1] for x in img_file_data):
        log.error(
            "Images contain different heights. All images should have same "
            "height. Exiting..."
        )
        exit(1)

    return img_file_names, img_file_data, segments_data, labeled_file_data


@typechecked
def process_directory_wayne(input_dir: Path, output_dir: Path, save_file: bool = False):
    img_file_names = []
    img_file_data = []  # Original image (xhat)
    labeled_file_data = []  # Segmenation map (yhat)
    segments_data = []  # Contains the boundaries

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if (
                file.endswith(".tiff") or file.endswith(".TIFF")
            ) and not file.startswith("."):
                image_file = Path(os.path.join(subdir, file))
                print(
                    "Processing file from Wayne State University format: "
                    f"{image_file}"
                )
                (
                    img_name,
                    img_array,
                    seg_map,
                    segs,
                ) = generate_image_label_wayne(image_file, output_dir, save_file)
                if img_name:
                    img_file_names.extend([img_name])
                    img_file_data.extend([img_array])
                    labeled_file_data.extend([seg_map])
                    segments_data.extend([segs])

    if len(img_file_data) == 0:
        log.info("No images were processed successfully. Exiting...")
        exit(1)

    # Images a transposed so dim 0 is image width
    if not all(x.shape[0] == img_file_data[0].shape[0] for x in img_file_data):
        log.error(
            "Images contain different widths. All images should have same "
            "width. Exiting..."
        )
        exit(1)

    if not all(x.shape[1] == img_file_data[0].shape[1] for x in img_file_data):
        log.error(
            "Images contain different heights. All images should have same "
            "height. Exiting..."
        )
        exit(1)

    return img_file_names, img_file_data, segments_data, labeled_file_data


@typechecked
def process_directory_mask(
    input_dir: Path,
    output_dir: Path,
    rgb_format: bool,
    save_file: bool = False,
):
    img_file_names = []
    img_file_data = []  # Original image (xhat)
    labeled_file_data = []  # Segmenation map (yhat)
    segments_data = []  # Contains the boundaries

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if (
                file.endswith(".tiff") or file.endswith(".TIFF")
            ) and not file.startswith("."):
                image_file = Path(os.path.join(subdir, file))
                print(f"Processing file in mask format: {image_file}")
                img_name, img_array, seg_map, segs = generate_image_label_mask(
                    image_file, output_dir, rgb_format, save_file
                )
                if img_name:
                    img_file_names.extend([img_name])
                    img_file_data.extend([img_array])
                    labeled_file_data.extend([seg_map])
                    segments_data.extend([segs])

    if len(img_file_data) == 0:
        log.info("No images were processed successfully. Exiting...")
        exit(1)

    if not all(x.shape[0] == img_file_data[0].shape[0] for x in img_file_data):
        log.error(
            "Images contain different heights. All images should have same "
            "height. Exiting..."
        )
        exit(1)

    if not all(x.shape[1] == img_file_data[0].shape[1] for x in img_file_data):
        log.error(
            "Images contain different widths. All images should have same "
            "width. Exiting..."
        )
        exit(1)

    return img_file_names, img_file_data, segments_data, labeled_file_data


@typechecked
def process_directory_labelme(
    input_dir: Path,
    output_dir: Path,
    layer_names: List[str],
    save_file: bool = False,
):
    img_file_names = []
    img_file_data = []  # Original image (xhat)
    labeled_file_data = []  # Segmenation map (yhat)
    segments_data = []  # Contains the boundaries

    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json") and not file.startswith("."):
                image_file = Path(os.path.join(subdir, file))
                print(f"Processing file: {image_file}")
                (img_name, img_array, seg_map, segs,) = generate_image_label_labelme(
                    image_file,
                    output_dir,
                    layer_names,
                    save_file,
                )

                if img_name:
                    img_file_names.extend([img_name])
                    img_file_data.extend([img_array])
                    labeled_file_data.extend([seg_map])
                    segments_data.extend([segs])

    if len(img_file_data) == 0:
        log.info("No images were processed successfully. Exiting...")
        exit(1)

    if not all(x.shape[0] == img_file_data[0].shape[0] for x in img_file_data):
        log.error(
            "Images contain different widths. All images should have same "
            "width. Exiting..."
        )
        exit(1)

    if not all(x.shape[1] == img_file_data[0].shape[1] for x in img_file_data):
        log.error(
            "Images contain different heights. All images should have same "
            "height. Exiting..."
        )
        exit(1)

    return img_file_names, img_file_data, segments_data, labeled_file_data


def crop_images_to_same_size(img_file_data, segments_data, labeled_file_data):
    """
    The images can have different size because the layers might not go from
    left to right (layers might hit the top of the image). Thus, we take the
    smallest of them and crop the rest so that all end up with the same size.
    This is necessary for:
    - easy generation of matrices in the HDF5 file
    - generating batches for training
    """
    min_rows = float("inf")
    min_columns = float("inf")

    for img in img_file_data:
        min_rows = min_rows if min_rows < img.shape[0] else img.shape[0]
        min_columns = min_columns if min_columns < img.shape[1] else img.shape[1]

    """
    Recall that the function "generate_image_label()" transposes the original
    images to make it compatible with what the u-net model expects. Thus,
    height of original image is "num_columns"
    and width is "num_rows"
    """
    log.info(f"Smallest image width: {min_rows}")
    log.info(f"Smallest image height: {min_columns}")

    for i in range(len(img_file_data)):
        img = img_file_data[i]
        segs = segments_data[i]
        label_img = labeled_file_data[i]

        row_pixels_to_remove = img.shape[0] - min_rows
        upper_rows_to_remove = int(row_pixels_to_remove // 2)
        lower_rows_to_remove = math.ceil(row_pixels_to_remove / 2)

        img_file_data[i] = img[
            upper_rows_to_remove : img.shape[0] - lower_rows_to_remove,
            :min_columns,
            :,
        ]
        segments_data[i] = segs[
            :, upper_rows_to_remove : segs.shape[1] - lower_rows_to_remove
        ]
        labeled_file_data[i] = label_img[
            upper_rows_to_remove : label_img.shape[0] - lower_rows_to_remove,
            :min_columns,
            :,
        ]


@typechecked
def generate_generic_dataset(
    input_dir: Path,
    file_name: Path,
    input_format: str,
    rgb_format: bool,
    layer_names: Optional[List[str]],
    backing_store: bool = True,
) -> h5py.File:
    if not os.path.isdir(file_name.parent):
        os.makedirs(file_name.parent)

    hf = h5py.File(file_name, "w", driver="core", backing_store=backing_store)

    if input_format == "wayne":
        (
            img_file_names,
            img_file_data,
            _,
            labeled_file_data,
        ) = process_directory_wayne(input_dir, file_name.parent, save_file=False)
    elif input_format == "labelme":
        (
            img_file_names,
            img_file_data,
            _,
            labeled_file_data,
        ) = process_directory_labelme(
            input_dir, file_name.parent, layer_names, save_file=False
        )
    elif input_format == "mask":
        (img_file_names, img_file_data, _, labeled_file_data,) = process_directory_mask(
            input_dir, file_name.parent, rgb_format, save_file=False
        )
    elif input_format == "visual":
        (
            img_file_names,
            img_file_data,
            _,
            labeled_file_data,
        ) = process_directory(input_dir, file_name.parent, save_file=False)
    else:
        log.error(f"Unrecognized input format: {input_format}. Exiting...")
        exit(1)

    hf.create_dataset("xhat", data=img_file_data)
    hf.create_dataset("yhat", data=labeled_file_data)
    hf.create_dataset("image_source", data=img_file_names)

    return hf
