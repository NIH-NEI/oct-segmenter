import os

import json
import math
import numpy as np
from pathlib import Path
from PIL import Image

# Layers from bottom to top
LAYER_NAMES = ["ILM", "ELM", "RPE"]

MIN_WIDTH_THRESHOLD = 780

from oct_segmenter.preprocessing import utils
from oct_segmenter.preprocessing.image_labeling_common import create_label_image, generate_boundary


def order_label_lines_from_left_to_right(shapes):
    """
    This function handles the case where the labeler started labeling the layer from right to left
    instead. The rest of the processing assumes that label points go from left to right.
    """
    for shape in shapes:
        if shape["points"][0][0] > shape["points"][-1][0]:
            shape["points"].reverse()


def get_vertical_margins(shapes) -> (int, int):
    left_margin = float("-inf")
    for shape in shapes:
        left_margin = left_margin if left_margin > shape["points"][0][0] else shape["points"][0][0]

    right_margin = float("inf")
    for shape in shapes:
        right_margin = right_margin if right_margin < shape["points"][-1][0] else shape["points"][-1][0]

    # The margins here can be floats since they can come from using shapes
    left_margin = math.ceil(left_margin)
    right_margin = int(right_margin)

    # The current U-net is being used requires the dimensions of the images to be a multiple of 16
    # because it has 4 pooling layers that reduce the size of the image by half each time
    pixels_to_remove = (right_margin - left_margin) % 16
    pixels_to_remove_left = pixels_to_remove // 2
    pixels_to_remove_right = math.ceil(pixels_to_remove / 2)

    return left_margin + pixels_to_remove_left, right_margin - pixels_to_remove_right


def get_bottom_margin(img_height):
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


def create_labelme_file(img: Image, shapes, shift, original_file_path, out_file_name, save_file):
    file = {}
    img_data = utils.pil_to_data(img)
    file['imageData'] = str(utils.img_data_to_img_b64(img_data), "utf-8")
    file["imagePath"] = str(original_file_path)
    file["version"] = "4.5.9"
    file["flags"] = {}

    shapes_dict = {}

    for shape in shapes:
        shapes_dict[shape["label"]] = shape

    layer_points = [[0, img.height-1], [img.width-1, img.height-1]]
    for i in range(len(LAYER_NAMES)):
        shape = shapes_dict[LAYER_NAMES[i]]
        extra_points = layer_points
        extra_points.reverse()
        layer_points = adjust_and_shift_layer(shape, shift, img.width)
        polygon = layer_points.copy()
        polygon.extend(extra_points)
        shape["shape_type"] = "polygon"
        shape["points"] = polygon

    # Upper polygon
    extra_points = layer_points
    extra_points.reverse()
    polygon = [[0, 0], [img.width - 1, 0]]
    polygon.extend(extra_points)
    shape = {}
    shape["points"] = polygon
    shape["label"] = "background"
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


def generate_image_label_labelme(img_path: Path, output_dir, save_file=True):
    if save_file and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(img_path) as f:
        data = json.load(f)

    img_layers = len(data["shapes"])
    if img_layers != len(LAYER_NAMES):
        log.warn(f"Labelme file {img_path} has unexpected {img_layers} layers. Skipping...")
        return None, None, None, None

    layer_set = set()
    for layer in data["shapes"]:
        layer_set.add(layer["label"])

    if len(layer_set) != len(LAYER_NAMES):
        log.warn(f"Labelme file {img_path} has missing layers. Skipping...")
        return None, None, None, None

    order_label_lines_from_left_to_right(data["shapes"])

    left_margin, right_margin = get_vertical_margins(data["shapes"]) # (int, int)
    labeled_region_width = right_margin - left_margin
    if labeled_region_width < MIN_WIDTH_THRESHOLD:
        warn_msg = " ".join((f"Labeled region of file {img_path} is {labeled_region_width}.",
            f"Below minimum threshold: {MIN_WIDTH_THRESHOLD}. Skipping..."))
        log.warn(warn_msg)
        return None, None, None, None

    bottom_margin = get_bottom_margin(data["imageHeight"])
    img = utils.img_b64_to_pil(data["imageData"])
    img = utils.convert_to_grayscale(img)

    img = img.crop((left_margin, 0, right_margin, bottom_margin))

    # Create cropped/shifted labelme file
    output_img_path = Path(output_dir + "/" + img_path.stem + "_cropped.json")
    labelme_img_json = create_labelme_file(
        img,
        data["shapes"],
        left_margin,
        data["imagePath"],
        output_img_path,
        save_file
    )

    # Generate image segmentation map
    segmentation_map_img = create_label_image(labelme_img_json, output_dir + img_path.stem + "_label.json" , save_file)

    if save_file:
        np.savetxt(output_dir + "/" + img_path.stem + "_matrix.txt", utils.pil_to_array(segmentation_map_img), fmt="%d")

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
