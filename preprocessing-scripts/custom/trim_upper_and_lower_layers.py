import os
import sys

import logging as log
import numpy as np
from pathlib import Path
from PIL import Image

MARGIN = 10
UNET_IMAGE_DIMENSION_MULTIPLICITY = 32


if __name__ == "__main__":
    """
    This script takes as input a directory containing TIFF files and their \
    corresponding CSVs. It trims the upper and lower layers of the images so \
    that the model can focus on all layers. It looks at the whole dataset \
    first to find the image with the shortest top layer and uses that as the \
    constraint to trim all the dataset. The same procedure is used for the \
    bottom layer.

    Example:
    python preprocessing-scripts/custom/trim_upper_and_lower_layers.py \
        data/experiment-10/images/
    """
    log.basicConfig()
    log.getLogger().setLevel(log.INFO)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    overall_layer_1_highest_pixel = float("inf")
    overall_layer_6_lowest_pixel = float("-inf")
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".csv") and not filename.startswith("."):
                arr = np.loadtxt(Path(root) / Path(filename), delimiter=",")
                layer_1 = np.argmax(arr == 1, axis=0)  # Layer 1 boundary
                layer_1_highest_pixel = np.min(layer_1)
                if overall_layer_1_highest_pixel > layer_1_highest_pixel:
                    overall_layer_1_highest_pixel = layer_1_highest_pixel

                layer_6 = np.argmax(arr == 6, axis=0)  # Layer 1 boundary
                layer_6_lowest_pixel = np.max(layer_6)
                if overall_layer_6_lowest_pixel < layer_6_lowest_pixel:
                    overall_layer_6_lowest_pixel = layer_6_lowest_pixel

        log.info(f"Layer 1 Highest Pixel: {overall_layer_1_highest_pixel}")
        log.info(f"Layer 6 Lowest Pixel: {overall_layer_6_lowest_pixel}")

        top_margin = overall_layer_1_highest_pixel - MARGIN
        bottom_margin = overall_layer_6_lowest_pixel + MARGIN

        new_height = bottom_margin - top_margin
        reminder = new_height % UNET_IMAGE_DIMENSION_MULTIPLICITY
        complement = UNET_IMAGE_DIMENSION_MULTIPLICITY - reminder
        new_height += complement

        log.info(f"New image height: {new_height}")

        top_margin -= int(complement / 2 + complement % 2)
        bottom_margin += int(complement / 2)

        log.info(f"Top Margin: {top_margin}")
        log.info(f"Bottom Margin: {bottom_margin}")

        for root, _, files in os.walk(input_dir):
            for filename in files:
                if filename.lower().endswith(
                    ".tiff"
                ) and not filename.startswith("."):
                    img = Image.open(Path(root) / Path(filename), "r")
                    cropped_img = img.crop(
                        (0, top_margin, img.width, bottom_margin)
                    )
                    assert (
                        cropped_img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY
                        == 0
                    )
                    cropped_img.save(Path(output_dir) / Path(filename))
                if filename.lower().endswith(
                    ".csv"
                ) and not filename.startswith("."):
                    arr = np.loadtxt(
                        Path(root) / Path(filename), delimiter=","
                    )
                    arr = arr[top_margin:bottom_margin, :]
                    np.savetxt(
                        Path(output_dir) / Path(filename),
                        arr,
                        delimiter=",",
                        fmt="%d",
                    )
