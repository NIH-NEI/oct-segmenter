import os
import sys

import json
import numpy as np
from pathlib import Path
import PIL.Image

from oct_segmenter.preprocessing import utils


def generate_side_region_input_image(image_path):
    """Generates the numpy matrices that can be fed to the Unet model
    for prediction. It crops the input image (left and right sections), 
    performing dimension expansion and transpose.

    Parameters
    ----------
    image_path: str
        Path to the input image (i.e. .tiff file)
    
    Returns
    -------
    img_left, img_right: np.array, np.array
        The numpy matrices that can be fed to Unet for prediction.
    """
    img = PIL.Image.open(image_path, "r")

    if img.mode == "RGBA" or img.mode == "RGB":
        img = img.convert("L")
    elif img.mode == "I;16":
        img = img.point(lambda i : i*(1./256)).convert("L")
    elif img.mode == "L":
        pass
    else:
        print(f"Unexpected mode: {img.mode}")
        exit(1)

    img_left = img.crop((x_left_start, 0, x_left_end, img.height))
    img_left = np.transpose(utils.pil_to_array(img_left))
    img_left = img_left[..., np.newaxis]
    img_right = img.crop((x_right_start, 0, x_right_end, img.height))
    img_right = np.transpose(utils.pil_to_array(img_right))
    img_right = img_right[..., np.newaxis]

    return img_left, img_right


def generate_input_image(image_path):
    """
    Generates the numpy matrix that can be fed to the Unet model
    for prediction. It performs dimension expansion and transpose.

    Parameters
    ----------
    image_path: str
        Path to the input image (i.e. .tiff file)

    Returns
    -------
    img: np.array, np.array
        The numpy matrices that can be fed to Unet for prediction.
    """
    img = PIL.Image.open(image_path, "r")
    if img.mode == "RGBA" or img.mode == "RGB":
        img = img.convert("L")
    elif img.mode == "I;16":
        img = img.point(lambda i : i*(1./256)).convert("L")
    elif img.mode == "L":
        pass
    else:
        print(f"Unexpected mode: {img.mode}")
        exit(1)

    # U-net architecture requires images with dimensions that are multiple of 16
    new_width = (int) (img.width // 16) * 16
    left_margin = (int) ((img.width - new_width)/2)
    right_margin = (int) (left_margin + new_width)

    img = img.crop((left_margin, 0, right_margin, img.height))
    img = np.transpose(utils.pil_to_array(img))
    img = img[..., np.newaxis]
    return img
