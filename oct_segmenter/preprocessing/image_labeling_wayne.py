import os
import sys

import json
import logging as log
import numpy as np
from pathlib import Path
import PIL.Image

from oct_segmenter.preprocessing import UNET_IMAGE_DIMENSION_MULTIPLICITY
from oct_segmenter.preprocessing import utils
from oct_segmenter.preprocessing.image_labeling_common import create_label_image, generate_boundary


def create_polygon_wayne(boundary, extra_points, label, image_height):
    shape = {}
    shape["label"] = label

    points = []

    for x, y in zip(range(0, len(boundary)), boundary):
        point = [x, y]
        assert(y > 0)
        points.append(point)

    points.extend(extra_points)

    shape["points"] = points

    shape["group_id"] = None
    shape["shape_type"] = "polygon"
    shape["flags"] = {}

    return shape


def create_labelme_file_wayne(img, annotations, in_file_name, out_file_name, save_file):
    file = {}
    img_data = utils.pil_to_data(img)
    file['imageData'] = str(utils.img_data_to_img_b64(img_data), "utf-8")
    file["imagePath"] = str(in_file_name)
    file["version"] = "4.5.9"
    file["flags"] = {}

    shapes = []

    # Bottom polygom
    extra_points = [[img.width-1, img.height-1], [0, img.height-1]]
    shapes.append(create_polygon_wayne(annotations[0], extra_points, "polygon_0", img.height))

    for i in range(1, len(annotations)):
        y_coordinates = annotations[i-1]
        extra_points = [(x, y) for x, y in enumerate(y_coordinates)]
        extra_points.reverse()
        shapes.append(create_polygon_wayne(annotations[i], extra_points, f"polygon_{i}", img.height))

    extra_points = [[img.width - 1, 0], [0, 0]]
    shapes.append(create_polygon_wayne(annotations[len(annotations) - 1], extra_points, "polygon_6", img.height))

    file["shapes"] = shapes
    file["imageHeight"] = img.height
    file["imageWidth"] = img.width

    if save_file:
        with open(out_file_name, 'w') as outfile:
            json.dump(file, outfile)

    return file


def process_annotations(annotations, left_margin, right_margin, top_margin):
    """
    CSVs from Wayne State University might contain 0s in some columns. This function:
    - Outer loop loops through layers.
    - First inner loop replaces < 0s from the left side with the first non-zero value from the left
    - Second inner loop replaces < 0s from the right side with the first non-zero value from the right
    """

    """
    Since the convention is that the boundary is the first line of the 'next' layer when traversing
    the image from top to bottom and we will be building the labelme file from botton to top we need
    to first substract 1 from all annotations.
    """
    for annot in annotations:
        for i in range(len(annot)):
            annot[i] -= 1

    for annot in annotations:
        for i in range(len(annot)):
            if annot[i] <= 0:
                annot[i] = next(x for x in annot[i+1:] if x > 0)
            else:
                break

        for i in range(len(annot) -1, 0, -1):
            if annot[i] <= 0:
                annot[i] = next(x for x in annot[::-1] if x > 0)
            else:
                break

        for i in range(1, len(annot) - 1):
            if annot[i] <= 0:
                log.error("Found inner column less or equal to 0. Exiting...")
                exit(1)

    width = len(annotations[0])
    return [[x - top_margin  for x in annot[left_margin:width-right_margin]] for annot in annotations]


def generate_image_label_wayne(image_path, output_dir, save_file=True):
    if save_file and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    csv_path = image_path.parent / Path(image_path.stem + ".csv")

    with open(csv_path, "r") as f:
        annotations = []
        for line in f.readlines():
            try:
                annotations.append([int(x) for x in line.replace(" ", "").rstrip("\n").split(",")])
            except ValueError:
                err_msg = " ".join(("Failed to parse CSV line. Make sure to pass the '-w' if you",
                    "are using the Wayne State Format"))
                log.error(err_msg)
                log.error(f"Conflicting line in {csv_path}: {line}")
                exit(1)

    img = PIL.Image.open(image_path, "r")

    if img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0 \
        or img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0:
        warn_msg = " ".join((f"Image dimensions need to be a multiple of 16",
            f"Image: {image_path} is {img.width} by {img.height}. Skipping..."))
        log.warn(warn_msg)
        return None, None, None, None

    for x_coords in annotations:
        if len(x_coords) != img.width:
            err_msg = " ".join((
                "The number of data points has to be equal to the image width:",
                f"{img.width}. Found line in CSV file: {csv_path} with length {len(x_coords)}.",
                "Please check CSV"
            ))
            log.error(err_msg)
            exit(1)

    img = utils.convert_to_grayscale(img)

    img_path = Path(output_dir + "/" + image_path.stem + ".json")

    # U-net architecture requires images with dimensions that are multiple of 16
    new_width = (int) (img.width // 16) * 16
    left_margin = (int) ((img.width - new_width)/2)
    right_margin = (int) (left_margin + new_width)
    right_margin_width = (int) (img.width - right_margin)

    new_height = (int) (img.height // 16) * 16
    top_margin = (int) ((img.height - new_height)/2)
    bottom_margin = (int) (top_margin + new_height)

    assert(new_width == img.width)
    assert(new_height == img.height)

    img = img.crop((left_margin, top_margin, right_margin, bottom_margin))
    annotations = process_annotations(annotations, left_margin, right_margin_width, top_margin)
    labelme_img_json = create_labelme_file_wayne(img, annotations, image_path, img_path, save_file)
    label_img = create_label_image(labelme_img_json, output_dir + "/" + image_path.stem + "_label.png", save_file)
    segs = generate_boundary(label_img)

    if save_file:
        np.savetxt(output_dir + "/" + image_path.stem + "_matrix.txt", label_img, fmt="%d")
        np.savetxt(output_dir + "/" + image_path.stem + "_segs.csv", segs, fmt="%d", delimiter=",")

    img = np.transpose(utils.pil_to_array(img))
    img = img[..., np.newaxis]
    label_img = np.transpose(label_img)
    label_img = label_img[..., np.newaxis]

    return str(image_path).encode("ascii"), img, label_img, segs
