import os

import logging as log
import numpy as np
from pathlib import Path
import PIL.Image
from typeguard import typechecked
from typing import Tuple

from oct_segmenter.common import utils
from oct_segmenter.preprocessing import UNET_IMAGE_DIMENSION_MULTIPLICITY
from oct_segmenter.preprocessing.image_labeling_common import generate_boundary


@typechecked
def generate_image_label_mask(
    image_path: Path,
    output_dir: Path,
    rgb_format: bool,
    save_file: bool = True,
) -> Tuple[bytes, np.ndarray, np.ndarray, np.ndarray]:
    if save_file and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    csv_path = image_path.parent / Path(image_path.stem + ".csv")

    with open(csv_path, "r") as f:
        mask = []
        for line in f.readlines():
            try:
                mask.append(
                    [int(x) for x in line.replace(" ", "").rstrip("\n").split(",")]
                )
            except ValueError:
                log.error(
                    "Failed to parse CSV line. Conflicting line in ",
                    f"{csv_path}: {line}",
                )
                exit(1)

    img = PIL.Image.open(image_path, "r")

    if (
        img.height % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0
        or img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0
    ):
        log.warn(
            "Image dimensions need to be a multiple of ",
            f"{UNET_IMAGE_DIMENSION_MULTIPLICITY}. "
            f"Image: {image_path} is {img.width} by {img.height}.",
            "Skipping...",
        )
        return None, None, None, None

    if len(mask) != img.height:
        log.error(
            "The number of lines in CSV file has to be equal to the image",
            f"height: {img.height}. Found in CSV file: {len(mask)} lines.",
            "Please check CSV",
        )
        exit(1)

    for x_coords in mask:
        if len(x_coords) != img.width:
            log.error(
                "The number of data points has to be equal to the image ",
                f"width: {img.width}. Found line in CSV file: {csv_path} ",
                f"with length {len(x_coords)}. Please check CSV",
            )
            exit(1)

    if rgb_format:
        img = img.convert("RGB")
    else:
        img = utils.convert_to_grayscale(img)

    mask = np.array(mask)
    segs = generate_boundary(mask)

    if save_file:
        utils.lblsave(output_dir + "/" + image_path.stem + "_label.png", mask)
        np.savetxt(output_dir + "/" + image_path.stem + "_matrix.txt", mask, fmt="%d")
        np.savetxt(
            output_dir + "/" + image_path.stem + "_segs.csv",
            segs,
            fmt="%d",
            delimiter=",",
        )

    img = utils.pil_to_array(img)
    ndim = 3  # Make sure images images have dim: (height, width, num_channels)
    # Adds one (i.e. num_channel) dimension when img is 2D.
    padded_shape = (img.shape + (1,) * ndim)[:ndim]
    img = img.reshape(padded_shape)
    mask = mask[..., np.newaxis]

    return str(image_path).encode("ascii"), img, mask, segs
