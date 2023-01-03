import os

import json
import logging as log
import numpy as np
from pathlib import Path
import PIL.Image

from oct_segmenter.common import utils
from oct_segmenter.preprocessing import VISUAL_CORE_BOUND_X_LEFT_START,\
    VISUAL_CORE_BOUND_X_LEFT_END, VISUAL_CORE_BOUND_X_RIGHT_START, VISUAL_CORE_BOUND_X_RIGHT_END,\
    UNET_IMAGE_DIMENSION_MULTIPLICITY
from oct_segmenter.preprocessing.image_labeling_common import create_label_image, generate_boundary

VISUAL_CORE_LAYER_DATA_POINTS = 20

"""
The images and labels recieved from the Visual Core group are labeled in the following manner:
- Each image has a correspoing CSV file. The CSV file contain six rows where:
    - The CSV values are 1-index based since they are generated from MATLAB
    - Each row represents a boundary; they contain 20 columns.
    - The first three rows represent the y-coordinate of the boundaries for x-intervals = [50-59, 60-69, ..., 230-239, 240-249] (0-index), [1-10, 51-60, ..., 241-250] (1-index)
    - The last three rows represent the y-coordinate of the boundaries for x-intervals = [750-759, 760-769, ..., 930-939, 940-949] (0-index), [751-760, 761-770, ..., 941-950]
    - The script uses the ceiling of the middle point for the x, y coordinates of the labels. i.e [54, 64, ..., 244] (0-index), [55, 65, ..., 245] (1-index)

- Given the U-net architecture of the Kugelman paper which has X pooling layers, we need the dimensions of the image to be multiple
  of 16. The closest 16-multiple for the width is 192.

- The following script crops the original image and generates two images (left and right): (the resulting width is 192)
    - The left image goes from x = [53, 245) (0-index), [54, 246) (1-index)
    - The right image goes from x = [753, 945)

- Then the boundaries are added as polygons. Notes:
   - The labeled points coming from the CSV are added at the center of the x-interval (i.e. 54, 64, ..., 754, 764, ...) which in the
   cropped images translate to (1, 11, 21, ..., 191)
   - An additional point is added at coordinate (x, y) = (0, boundary[0]) to make the side of the polygon parallel to the vertical
   side of the image.

- For the y-coordinate, the Ys given on the CSV file are starting from bottom:
    y_for_array = img.height - 1 - (y_csv - 1)   # The `-1` is because the CSV is 1-index.
    y_for_array = img.height - y_csv

- Finally the image is converted into a segmentation map: A 2D-matrix where each element represents the class
the pixel belongs to.
"""


def create_polygon_visual_core(boundary, extra_points, label, image_height):
    shape = {}
    shape["label"] = label

    points = []

    # Add left side-edge point
    points.append([0, image_height - boundary[0]])

    for x, y in zip(range(1, 192, 10), boundary):
        point = [x, image_height - y]
        points.append(point)

    points.extend(extra_points)

    shape["points"] = points

    shape["group_id"] = None
    shape["shape_type"] = "polygon"
    shape["flags"] = {}

    return shape


def create_labelme_file_visual_core(img, annotations, in_file_name, out_file_name, save_file):
    file = {}
    img_data = utils.pil_to_data(img)
    file['imageData'] = str(utils.img_data_to_img_b64(img_data), "utf-8")
    file["imagePath"] = str(in_file_name)
    file["version"] = "4.5.9"
    file["flags"] = {}

    shapes = []

    # Bottom polygom
    extra_points = [[img.width-1, img.height-1], [0, img.height-1]]
    shapes.append(create_polygon_visual_core(annotations[0], extra_points, "polygon_0", img.height))

    # Second polygon
    y_coordinates = [img.height - x for x in annotations[0]]
    extra_points = [(0, y_coordinates[0])]
    extra_points.extend([(x, y) for x, y in zip(range(1, 192, 10), y_coordinates)])
    extra_points.reverse()
    shapes.append(create_polygon_visual_core(annotations[1], extra_points, "polygon_1", img.height))

    # Third polygon
    y_coordinates = [img.height - x for x in annotations[1]]
    extra_points = [(0, y_coordinates[0])]
    extra_points.extend([(x, y) for x, y in zip(range(1, 192, 10), y_coordinates)])
    extra_points.reverse()
    shapes.append(create_polygon_visual_core(annotations[2], extra_points, "polygon_2", img.height))

    # Upper polygon
    extra_points = [[img.width - 1, 0], [0, 0]]
    shapes.append(create_polygon_visual_core(annotations[2], extra_points, "polygon_3", img.height))

    file["shapes"] = shapes
    file["imageHeight"] = img.height
    file["imageWidth"] = img.width

    if save_file:
        with open(out_file_name, 'w') as outfile:
            json.dump(file, outfile)

    return file


def generate_image_label_visual_core(image_path: Path, output_dir, save_file=True):
    if save_file and not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    csv_path = image_path.parent / Path(image_path.stem + ".csv")

    with open(csv_path, "r") as f:
        annotations = []
        for line in f.readlines():
            try:
                layer_x_coords = [int(x) for x in line.replace(" ", "").rstrip(',\n').split(",")]
                if len(layer_x_coords) != VISUAL_CORE_LAYER_DATA_POINTS:
                    err_msg = " ".join((
                        f"Found {len(layer_x_coords)} points for a given layer in file: {csv_path}.",
                        f"Expected: {VISUAL_CORE_LAYER_DATA_POINTS}. Make sure to pass the '-w'",
                        f"if you are using the Wayne State Format"))
                    log.error(err_msg)
                    exit(1)
                annotations.append(layer_x_coords)
            except ValueError:
                err_msg = " ".join(("Failed to parse CSV line. Make sure to pass the '-w' if you",
                    "are using the Wayne State Format"))
                log.error(err_msg)
                log.error(f"Conflicting line in {csv_path}: {line}")
                exit(1)

    '''
    The original image provided by NIH is a TIFF file with a pixel depth of 16-bit.
    The PIL mode is I;16 (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)
    If we try to generate the segmentation map directly from this image we get the error:

    File "/home/balvisio/.local/lib/python3.8/site-packages/imgviz/color.py", line 46, in gray2rgb
    assert gray.dtype == np.uint8, "gray dtype must be np.uint8"
    AssertionError: gray dtype must be np.uint8

    The `imgviz` package expects the gray image to be uint8.

    Thus, we need to convert the PIL image to uint8 before we call `labelme_json_to_dataset`
    However, there is a bug in PIL reported here:

    https://github.com/python-pillow/Pillow/issues/3011
    https://github.com/python-pillow/Pillow/pull/3838

    Thus we need to do a workaround posted here:

    https://stackoverflow.com/questions/43978819/convert-tiff-i16-to-jpg-with-pil-pillow

    '''
    img = utils.convert_to_grayscale(PIL.Image.open(image_path, "r"))

    if img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0 \
        or img.width % UNET_IMAGE_DIMENSION_MULTIPLICITY != 0:
        warn_msg = " ".join((f"Image dimensions need to be a multiple of 16",
            f"Image: {image_path} is {img.width} by {img.height}. Skipping..."))
        log.warn(warn_msg)
        return None, None, None, None, None, None, None, None

    img_left_path = Path(output_dir + "/" + image_path.stem + "_left.json")
    img_right_path = Path(output_dir + "/" + image_path.stem + "_right.json")


    img_left = img.crop((VISUAL_CORE_BOUND_X_LEFT_START, 0, VISUAL_CORE_BOUND_X_LEFT_END, img.height))
    labelme_img_left_json = create_labelme_file_visual_core(img_left, annotations[:3], image_path, img_left_path, save_file)
    label_img_left = create_label_image(labelme_img_left_json, output_dir + "/" + image_path.stem + "_left_label.png", save_file)
    segs_left = generate_boundary(label_img_left)

    img_right = img.crop((VISUAL_CORE_BOUND_X_RIGHT_START, 0, VISUAL_CORE_BOUND_X_RIGHT_END, img.height))
    lebelme_img_right_json = create_labelme_file_visual_core(img_right, annotations[3:], image_path, img_right_path, save_file)
    label_img_right = create_label_image(lebelme_img_right_json, output_dir + "/" + image_path.stem + "_right_label.png", save_file)
    segs_right = generate_boundary(label_img_right)

    """
    The images need to be transposed because the `model` expects
    and array of shape (image_width, image_height) and recall that
    in an array (rows x columns) the rows are the height and columns
    are the width.
    """
    if save_file:
        np.savetxt(output_dir + "/" + image_path.stem + "_left_matrix.txt", utils.pil_to_array(label_img_left), fmt="%d")
        np.savetxt(output_dir + "/" + image_path.stem + "_right_matrix.txt", utils.pil_to_array(label_img_right), fmt="%d")

    ndim = 3  # Make sure images images have dim: (height, width, num_channels)
    # Adds one (i.e. num_channel) dimension when img is 2D.
    img_left = utils.pil_to_array(img_left)
    padded_shape = (img_left.shape + (1,)*ndim)[:ndim]
    img_left = img_left.reshape(padded_shape)

    img_right = utils.pil_to_array(img_right)
    padded_shape = (img_right.shape + (1,)*ndim)[:ndim]
    img_right = img_right.reshape(padded_shape)

    label_img_left = label_img_left[..., np.newaxis]
    label_img_right = label_img_right[..., np.newaxis]

    return str(image_path).encode("ascii"), img_left, label_img_left, segs_left, str(image_path).encode("ascii"), img_right, label_img_right, segs_right
