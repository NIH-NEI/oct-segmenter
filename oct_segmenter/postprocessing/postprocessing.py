import numpy as np
from pathlib import Path
from typeguard import typechecked

from oct_segmenter.common import utils


@typechecked
def create_labelme_file_from_boundaries(
    img_arr: np.array,
    image_name: Path,
    boundaries: np.array
):
    LAYER_NAMES = ["RNFL-vitreous", "GCL-RNFL", "INL-IPL", "ONL-OPL", "ELM", "RPE"]
    SPACING = 20

    image_height, image_width = img_arr.shape

    file = {}
    file['imageData'] = str(utils.img_arr_to_b64(img_arr), "utf-8")
    file["imagePath"] = str(image_name)
    file["version"] = "4.5.9"
    file["flags"] = {}

    shapes = []

    for layer_name, boundary in zip(LAYER_NAMES, boundaries):
        shape = {}
        shape["label"] = layer_name
        shape["points"] = []
        shape["shape_type"] = "linestrip"
        shape["flags"] = {}
        shape["group_id"] = None

        for x in range(0, len(boundary), SPACING):
            shape["points"].append([x, int(boundary[x])])

        if image_width % SPACING != 0: # Add right-most point
            shape["points"].append([image_width - 1, int(boundary[image_width - 1])])

        shapes.append(shape)

    file["shapes"] = shapes
    file["imageHeight"] = image_height
    file["imageWidth"] = image_width

    return file