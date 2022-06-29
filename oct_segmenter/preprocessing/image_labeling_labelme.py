import os

import json
import logging as log
import math
import numpy as np
from pathlib import Path
from PIL import Image
from typeguard import typechecked


MIN_WIDTH_THRESHOLD = 780

from oct_segmenter.common import utils
from oct_segmenter.preprocessing import UNET_IMAGE_DIMENSION_MULTIPLICITY
from oct_segmenter.preprocessing.image_labeling_common import create_label_image, generate_boundary


def order_label_lines_from_left_to_right(shapes):
    """
    This function handles the case where the labeler started labeling the layer from right to left
    instead. The rest of the processing assumes that label points go from left to right.
    """
    for shape in shapes:
        if shape["points"][0][0] > shape["points"][-1][0]:
            shape["points"].reverse()


@typechecked
def order_layers_from_top_to_bottom(shapes: list[dict]) -> list[str]:
    layers_height = {}
    for shape in shapes:
        layers_height[shape["label"]] = shape["points"][0][1]

    bottom_to_top_layers = [key for key, _ in sorted(layers_height.items(), key=lambda item: item[1])]
    return bottom_to_top_layers


def get_vertical_margins(shapes) -> tuple[int, int]:
    left_margin = float("-inf")
    for shape in shapes:
        left_margin = left_margin if left_margin > shape["points"][0][0] else shape["points"][0][0]

    right_margin = float("inf")
    for shape in shapes:
        right_margin = right_margin if right_margin < shape["points"][-1][0] else shape["points"][-1][0]

    # The margins here can be floats since they can come from using shapes
    left_margin = math.ceil(left_margin)
    right_margin = int(right_margin)
    """
    The current U-net is being used requires the dimensions of the images to be a multiple of 16
    because it has 4 pooling layers that reduce the size of the image by half each time

    For example: In this case the left most point is at 0 and right-most at 15; total width is 16.
    We shouldn't remove any pixel. Thus:
    right_margin == 15
    left_margin == 0
    pixels_to_remove = (15 - 0 + 1) % 16 = 0
    """
    pixels_to_remove = (right_margin - left_margin + 1) % UNET_IMAGE_DIMENSION_MULTIPLICITY
    pixels_to_remove_left = pixels_to_remove // 2
    pixels_to_remove_right = math.ceil(pixels_to_remove / 2)

    return left_margin + pixels_to_remove_left, right_margin - pixels_to_remove_right


def get_multiplicity_height(img_height):
    return img_height - img_height % 16


def interpolate(x1, y1, x2, y2, xhat):
    a = (y2 - y1)/(x2 - x1)
    b = y1 - a * x1

    yhat = a * xhat + b

    return yhat


def adjust_and_shift_layer(shape, shift, img_width):
    new_points = []

    # These points are the closest points that fall outside the width of the image
    outside_left_point = None
    outside_right_point = None

    # Shift all the points to the left
    for point in shape["points"]:
        if point[0] - shift < 0:
            outside_left_point = [point[0] - shift, point[1]]
            continue

        if point[0] - shift >= img_width:
            outside_right_point = [point[0] - shift, point[1]]
            break

        new_point = [point[0] - shift, point[1]]
        new_points.append(new_point)

    # Checks
    """
    Checks:
    This function is for a single layer. Since the shift is calculated by taking the layer that
    starts most to the right it is guaranteed that for all the other layers after shifting them by
    "shift" amount, the x-coordinate will become negative. For the layer that starts most to the
    right, the x-coordinate of the first point can be exactly 0 or between -1 and 0 because
    in the "get_vertical_margins()" function, the left_margin is math.ceiled (shifted right) and
    also cropped to be multiple of 16.
    Idem for the right side.
    """
    if outside_left_point != None:
        # Add left side-edge point
        x1 = outside_left_point[0]
        y1 = outside_left_point[1]
        x2 = new_points[0][0]
        y2 = new_points[0][1]
        xhat = 0
        yhat = interpolate(x1, y1, x2, y2, xhat)
        new_points.insert(0, [0, yhat])
    elif new_points[0][0] == 0:
        pass
    else:
        print("ERROR: The leftmost point is not negative or equal to 0; does not reach the left side of the image")
        exit(1)

    if outside_right_point != None:
        # Add right side-edge point
        x1 = new_points[-1][0]
        y1 = new_points[-1][1]
        x2 = outside_right_point[0]
        y2 = outside_right_point[1]
        xhat = img_width - 1
        yhat = interpolate(x1, y1, x2, y2, xhat)
        new_points.append([img_width - 1, yhat])
    elif new_points[-1][0] == img_width-1:
        pass
    else:
        print("ERROR: The rightmost point is lower than width-1; it does not reach right side of the image")
        exit(1)

    return new_points


@typechecked
def create_labelme_file(
    img: Image,
    shapes: list[dict],
    shift: int,
    layer_names: list[str],
    original_file_path: str,
    out_file_name: Path,
    save_file: bool,
):
    file = {}
    layer_names.insert(0, "background")
    img_data = utils.pil_to_data(img)
    file['imageData'] = str(utils.img_data_to_img_b64(img_data), "utf-8")
    file["imagePath"] = original_file_path
    file["version"] = "4.5.9"
    file["flags"] = {}

    shapes_dict = {}

    for shape in shapes:
        shapes_dict[shape["label"]] = shape

    layer_points = [[0, 0], [img.width - 1, 0]]
    for i in range(1, len(layer_names)):
        shape = shapes_dict[layer_names[i]]
        extra_points = layer_points
        extra_points.reverse()
        layer_points = adjust_and_shift_layer(shape, shift, img.width)
        polygon = layer_points.copy()
        polygon.extend(extra_points)
        shape["shape_type"] = "polygon"
        shape["points"] = polygon
        shape["label"] = layer_names[i - 1]

    # Lower polygon
    extra_points = layer_points
    extra_points.reverse()
    polygon = [[0, img.height-1], [img.width-1, img.height-1]]
    polygon.extend(extra_points)
    shape = {}
    shape["points"] = polygon
    shape["label"] = layer_names[i]
    shape["group_id"] = None
    shape["shape_type"] = "polygon"
    shape["flags"] = {}

    shapes.append(shape)

    file["shapes"] = shapes
    file["imageHeight"] = img.height
    file["imageWidth"] = img.width

    if save_file:
        with open(out_file_name, 'w') as outfile:
            json.dump(file, outfile)

    return file


@typechecked
def generate_image_label_labelme(
    img_path: Path,
    output_dir: Path,
    layer_names: list[str],
    save_file: bool=True,
):
    if save_file and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(img_path) as f:
        data = json.load(f)

    img_layers = len(data["shapes"])
    if img_layers != len(layer_names):
        log.warn(f"Labelme file {img_path} has unexpected {img_layers} layers. Skipping...")
        return None, None, None, None

    layer_set = set()
    for layer in data["shapes"]:
        layer_set.add(layer["label"])

    if len(layer_set) != len(layer_names):
        log.warn(f"Labelme file {img_path} has missing layers. Skipping...")
        return None, None, None, None

    order_label_lines_from_left_to_right(data["shapes"])

    top_to_bottom_layers = order_layers_from_top_to_bottom(data["shapes"])

    left_margin, right_margin = get_vertical_margins(data["shapes"]) # (int, int)
    labeled_region_width = right_margin - left_margin
    if labeled_region_width < MIN_WIDTH_THRESHOLD:
        warn_msg = " ".join((f"Labeled region of file {img_path} is {labeled_region_width}.",
            f"Below minimum threshold: {MIN_WIDTH_THRESHOLD}. Skipping..."))
        log.warn(warn_msg)
        return None, None, None, None

    multiplicty_height = get_multiplicity_height(data["imageHeight"])
    img = utils.img_b64_to_pil(data["imageData"])

    if img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0 \
        or img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0:
        warn_msg = " ".join((f"Image dimensions need to be a multiple of 16",
            f"Image: {img_path} is {img.width} by {img.height}. Skipping..."))
        log.warn(warn_msg)
        return None, None, None, None

    img = utils.convert_to_grayscale(img)

    # Since margins are 0-indexed we need to add 1 to the margin to get right width
    img = img.crop((left_margin, 0, right_margin + 1, multiplicty_height))

    assert(img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY == 0)
    assert(img.height % UNET_IMAGE_DIMENSION_MULTIPLICITY == 0)

    # Create cropped/shifted labelme file
    output_img_path = output_dir / Path(img_path.stem + "_cropped.json")

    labelme_img_json = create_labelme_file(
        img,
        data["shapes"],
        left_margin,
        top_to_bottom_layers,
        data["imagePath"],
        output_img_path,
        save_file,
    )

    # Generate image segmentation map
    segmentation_map_img = create_label_image(labelme_img_json, output_dir / Path(img_path.stem + "_label.json"), save_file)

    if save_file:
        np.savetxt(output_dir / Path(img_path.stem + "_matrix.txt"), utils.pil_to_array(segmentation_map_img), fmt="%d", delimiter=",")

    # Generate boundaries out of the segmentation map image
    boundaries = generate_boundary(segmentation_map_img)

    """
    The images need to be transposed because the `model` expects
    and array of shape (image_width, image_height) and recall that
    in an array (rows x columns) the rows are the height and columns
    are the width.
    """
    img = np.transpose(utils.pil_to_array(img))
    img = img[..., np.newaxis]
    segmentation_map_img = np.transpose(segmentation_map_img)
    segmentation_map_img = segmentation_map_img[..., np.newaxis]

    return str(img_path).encode("ascii"), img, segmentation_map_img, boundaries
