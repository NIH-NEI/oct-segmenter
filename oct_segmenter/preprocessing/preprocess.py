from __future__ import annotations

import logging as log
import numpy as np
from pathlib import Path
import PIL.Image

from oct_segmenter.common import utils
from oct_segmenter.preprocessing import VISUAL_CORE_BOUND_X_LEFT_START,\
    VISUAL_CORE_BOUND_X_LEFT_END, VISUAL_CORE_BOUND_X_RIGHT_START, VISUAL_CORE_BOUND_X_RIGHT_END,\
    UNET_IMAGE_DIMENSION_MULTIPLICITY


def generate_side_region_input_image(image_path: Path, flip_top_bottom: bool) -> tuple[np.array]:
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

    if img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0 \
        or img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0:
        warn_msg = " ".join((f"Image dimensions need to be a multiple of 16",
            f"Image: {image_path} is {img.width} by {img.height}. Skipping..."))
        log.warn(warn_msg)
        return None, None

    if flip_top_bottom:
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    img_left = img.crop((VISUAL_CORE_BOUND_X_LEFT_START, 0, VISUAL_CORE_BOUND_X_LEFT_END, img.height))
    img_left = utils.make_height_multiple(img_left, cut_bottom=True)
    img_left = np.transpose(utils.pil_to_array(img_left))
    img_left = img_left[..., np.newaxis]
    img_right = img.crop((VISUAL_CORE_BOUND_X_RIGHT_START, 0, VISUAL_CORE_BOUND_X_RIGHT_END, img.height))
    img_right = utils.make_height_multiple(img_right, cut_bottom=True)
    img_right = np.transpose(utils.pil_to_array(img_right))
    img_right = img_right[..., np.newaxis]

    return img_left, img_right


def generate_input_image(image_path: Path, flip_top_bottom: bool=False) -> np.array:
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

    if img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0 \
        or img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0:
        warn_msg = " ".join((f"Image dimensions need to be a multiple of 16",
            f"Image: {image_path} is {img.width} by {img.height}. Skipping..."))
        log.warn(warn_msg)
        return None

    if flip_top_bottom:
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    img = utils.convert_to_grayscale(img)

    # U-net architecture requires images with dimensions that are multiple of 16
    cropped_img, _, _ = utils.make_img_size_multiple(img, 16)
    cropped_img = np.transpose(utils.pil_to_array(cropped_img))
    cropped_img = cropped_img[..., np.newaxis]
    return cropped_img
