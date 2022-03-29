import os

import logging as log
import numpy as np
from pathlib import Path
import PIL.Image

from oct_segmenter.preprocessing import UNET_IMAGE_DIMENSION_MULTIPLICITY
from oct_segmenter.preprocessing import utils
from oct_segmenter.preprocessing.image_labeling_common import generate_boundary


def generate_image_label_mask(image_path, output_dir, save_file=True):
    if save_file and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    csv_path = image_path.parent / Path(image_path.stem + ".csv")

    with open(csv_path, "r") as f:
        mask = []
        for line in f.readlines():
            try:
                mask.append([int(x) for x in line.replace(" ", "").rstrip("\n").split(",")])
            except ValueError:
                err_msg = "Failed to parse CSV line."
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

    if len(mask) != img.height:
        err_msg = " ".join((
            "The number of lines in CSV file has to be equal to the image height:",
            f"{img.height}. Found in CSV file: {len(mask)} lines.",
            "Please check CSV"
        ))
        log.error(err_msg)
        exit(1)

    for x_coords in mask:
        if len(x_coords) != img.width:
            err_msg = " ".join((
                "The number of data points has to be equal to the image width:",
                f"{img.width}. Found line in CSV file: {csv_path} with length {len(x_coords)}.",
                "Please check CSV"
            ))
            log.error(err_msg)
            exit(1)

    img = utils.convert_to_grayscale(img)
    mask = np.array(mask)
    segs = generate_boundary(mask)

    if save_file:
        utils.lblsave(output_dir + "/" + image_path.stem + "_label.png", mask)
        np.savetxt(output_dir + "/" + image_path.stem + "_matrix.txt", mask, fmt="%d")
        np.savetxt(output_dir + "/" + image_path.stem + "_segs.csv", segs, fmt="%d", delimiter=",")

    img = np.transpose(utils.pil_to_array(img))
    img = img[..., np.newaxis]
    mask = np.transpose(mask)
    mask = mask[..., np.newaxis]

    return str(image_path).encode("ascii"), img, mask, segs
