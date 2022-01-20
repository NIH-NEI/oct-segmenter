from __future__ import annotations

import os
import sys

import json
import numpy as np
from pathlib import Path
import PIL.Image

from oct_segmenter.preprocessing import VISUAL_CORE_BOUND_X_LEFT_START,\
    VISUAL_CORE_BOUND_X_LEFT_END, VISUAL_CORE_BOUND_X_RIGHT_START, VISUAL_CORE_BOUND_X_RIGHT_END
from oct_segmenter.preprocessing import utils


def generate_side_region_input_image(image_path: Path) -> tuple[np.array]:
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
    img = utils.convert_to_grayscale(PIL.Image.open(image_path, "r"))


    img_left = img.crop((VISUAL_CORE_BOUND_X_LEFT_START, 0, VISUAL_CORE_BOUND_X_LEFT_END, img.height))
    img_left = np.transpose(utils.pil_to_array(img_left))
    img_left = img_left[..., np.newaxis]
    img_right = img.crop((VISUAL_CORE_BOUND_X_RIGHT_START, 0, VISUAL_CORE_BOUND_X_RIGHT_END, img.height))
    img_right = np.transpose(utils.pil_to_array(img_right))
    img_right = img_right[..., np.newaxis]

    return img_left, img_right


def generate_input_image(image_path: Path) -> np.array:
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
    cropped_img, _, _ = utils.make_img_size_multiple(img, 16)
    cropped_img = np.transpose(utils.pil_to_array(cropped_img))
    cropped_img = cropped_img[..., np.newaxis]
    return cropped_img
