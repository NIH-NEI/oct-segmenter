from __future__ import annotations

import logging as log
import numpy as np
from pathlib import Path
import PIL.Image
from typeguard import typechecked
from typing import Tuple

from oct_segmenter.common import utils
from oct_segmenter.preprocessing import (
    VISUAL_CORE_BOUND_X_LEFT_START,
    VISUAL_CORE_BOUND_X_LEFT_END,
    VISUAL_CORE_BOUND_X_RIGHT_START,
    VISUAL_CORE_BOUND_X_RIGHT_END,
    UNET_IMAGE_DIMENSION_MULTIPLICITY,
)


@typechecked
def generate_side_region_input_image(
    image_path: Path, flip_top_bottom: bool
) -> Tuple[np.ndarray]:
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

    if (
        img.height % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0
        or img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0
    ):
        log.warn(
            "Image dimensions need to be a multiple of ",
            f"{UNET_IMAGE_DIMENSION_MULTIPLICITY}",
            f"Image: {image_path} is {img.width} by {img.height}. ",
            "Skipping...",
        )
        return None, None

    if flip_top_bottom:
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    img_left = img.crop(
        (
            VISUAL_CORE_BOUND_X_LEFT_START,
            0,
            VISUAL_CORE_BOUND_X_LEFT_END,
            img.height,
        )
    )
    img_left = utils.make_height_multiple(img_left, cut_bottom=True)
    img_left = utils.pil_to_array(img_left)
    img_left = img_left[..., np.newaxis]
    img_right = img.crop(
        (
            VISUAL_CORE_BOUND_X_RIGHT_START,
            0,
            VISUAL_CORE_BOUND_X_RIGHT_END,
            img.height,
        )
    )
    img_right = utils.make_height_multiple(img_right, cut_bottom=True)
    img_right = utils.pil_to_array(img_right)
    img_right = img_right[..., np.newaxis]

    return img_left, img_right


@typechecked
def generate_input_image(image_path: Path) -> np.ndarray:
    """
    Generates the numpy matrix that can be fed to the model
    for prediction.

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
    img = utils.pil_to_array(img)
    ndim = 3  # Make sure images images have dim: (height, width, num_channels)
    # Adds one (i.e. num_channel) dimension when img is 2D.
    padded_shape = (img.shape + (1,) * ndim)[:ndim]
    img = img.reshape(padded_shape)

    return img
