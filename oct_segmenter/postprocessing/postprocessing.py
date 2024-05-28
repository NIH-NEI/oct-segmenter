from __future__ import annotations

import logging as log
import numpy as np
from pathlib import Path
from typeguard import typechecked
from typing import Dict, Optional

from oct_segmenter import (
    VISUAL_FUNCTION_CORE_LAYER_NAMES,
    WAYNE_STATE_LAYER_NAMES,
)
from oct_segmenter.common import utils


@typechecked
def create_labelme_file_from_boundaries(
    img_arr: np.ndarray,
    image_name: Path,
    boundaries: np.ndarray,
    spacing: int = 20,
) -> Optional[Dict]:

    image_width = img_arr.shape[1]
    boundaries_width = boundaries.shape[1]
    if image_width != boundaries_width:
        log.warn(
            f"Image width: {image_width} is not equal to boundaries width: ",
            f"{boundaries_width}. Skipping...",
        )
        return None

    num_boundaries = boundaries.shape[0]
    if num_boundaries == 3:
        layer_names = VISUAL_FUNCTION_CORE_LAYER_NAMES
    elif num_boundaries == 6:
        layer_names = WAYNE_STATE_LAYER_NAMES
    else:
        log.error(f"Unrecognized number of layers: {num_boundaries}")
        exit(1)

    image_height, image_width = img_arr.shape

    file = {}
    file["imageData"] = str(utils.img_arr_to_b64(img_arr), "utf-8")
    file["imagePath"] = str(image_name)
    file["version"] = "4.5.9"
    file["flags"] = {}

    shapes = []

    for layer_name, boundary in zip(layer_names, boundaries):
        shape = {}
        shape["label"] = layer_name
        shape["points"] = []
        shape["shape_type"] = "linestrip"
        shape["flags"] = {}
        shape["group_id"] = None

        for x in range(0, len(boundary), spacing):
            shape["points"].append([x, int(boundary[x])])

        if image_width % spacing != 0:  # Add right-most point
            shape["points"].append([image_width - 1, int(boundary[image_width - 1])])

        shapes.append(shape)

    file["shapes"] = shapes
    file["imageHeight"] = image_height
    file["imageWidth"] = image_width

    return file
