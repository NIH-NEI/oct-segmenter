import json
import numpy as np
import os
import PIL.Image
import utils
import subprocess
import shutil
import sys

from pathlib import Path

'''
The images and labels recieved from the NEI are labeled in the following manner:
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
'''

x_left_start = 53
x_left_end = 245

x_right_start = 753
x_right_end = 945


def create_polygon(boundary, extra_points, label, image_height):
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

def create_labelme_file(img, annotations, in_file_name, out_file_name):
    file = {}
    img_data = utils.pil_to_data(img)
    file['imageData'] = str(utils.img_data_to_img_b64(img_data), "utf-8")
    file["imagePath"] = str(in_file_name)
    file["version"] = "4.5.9"
    file["flags"] = {}

    shapes = []

    # Bottom polygom
    extra_points = [[img.width-1, img.height-1], [0, img.height-1]]
    shapes.append(create_polygon(annotations[0], extra_points, "polygon_0", img.height))

    # Second polygon
    y_coordinates = [img.height - x for x in annotations[0]]
    extra_points = [(0, y_coordinates[0])]
    extra_points.extend([(x, y) for x, y in zip(range(1, 192, 10), y_coordinates)])
    extra_points.reverse()
    shapes.append(create_polygon(annotations[1], extra_points, "polygon_1", img.height))

    # Third polygon
    y_coordinates = [img.height - x for x in annotations[1]]
    extra_points = [(0, y_coordinates[0])]
    extra_points.extend([(x, y) for x, y in zip(range(1, 192, 10), y_coordinates)])
    extra_points.reverse()
    shapes.append(create_polygon(annotations[2], extra_points, "polygon_2", img.height))

    # Upper polygon
    extra_points = [[img.width - 1, 0], [0, 0]]
    shapes.append(create_polygon(annotations[2], extra_points, "polygon_3", img.height))

    file["shapes"] = shapes
    file["imageHeight"] = img.height
    file["imageWidth"] = img.width

    with open(out_file_name, 'w') as outfile:
        json.dump(file, outfile)


def create_label_image(img_path, output_dir, save_file=True):
    tmp_output_dir = output_dir + "/tmp/"
    if not os.path.isdir(tmp_output_dir):
        os.mkdir(tmp_output_dir)

    subprocess.run(["labelme_json_to_dataset", img_path, "-o", tmp_output_dir])

    label_img = PIL.Image.open(tmp_output_dir + "/label.png")
    # labelme creates the segmentation map (label.png) using [1, 2, 3, 4, ...] and we want [0, 1, 2, 3, ...]
    label_img = label_img.point(lambda i : i-1)
    if save_file:
        label_img.save(output_dir + "/" + img_path.stem + "_label.png")

    shutil.rmtree(tmp_output_dir)

    return np.asarray(label_img)

def process_image(image_path, output_dir, save_file=True):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    csv_path = image_path.parent / Path(image_path.stem + ".csv")

    with open(csv_path, "r") as f:
        annotations = []
        for line in f.readlines():
            annotations.append([int(x) for x in line.rstrip(' ,\n').split(" ,")])

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
    img = PIL.Image.open(image_path, "r")

    if img.mode == "RGBA":
        img = img.convert("L")
    elif img.mode == "I;16":
        img = img.point(lambda i : i*(1./256)).convert("L")
    else:
        print(f"Unexpected mode: {img.mode}")
        exit(1)

    if save_file:
        write_dir = output_dir
    else: # If false we still save in a tmp location for use of `labelme_json_to_dataset` process
        write_dir = output_dir + "/tmp/"
        if not os.path.isdir(write_dir):
            os.mkdir(write_dir)

    img_left_path = Path(write_dir + "/" + image_path.stem + "_left.json")
    img_right_path = Path(write_dir + "/" + image_path.stem + "_right.json")

    img_left = img.crop((x_left_start, 0, x_left_end, img.height))
    create_labelme_file(img_left, annotations[:3], image_path, img_left_path)
    label_img_left = create_label_image(img_left_path, write_dir, save_file)

    img_right = img.crop((x_right_start, 0, x_right_end, img.height))
    create_labelme_file(img_right, annotations[3:], image_path, img_right_path)
    label_img_right = create_label_image(img_right_path, write_dir, save_file)

    if not save_file:
        shutil.rmtree(write_dir)

    img_left = utils.pil_to_array(img_left)
    img_left = img_left[..., np.newaxis]
    img_right = utils.pil_to_array(img_right)
    img_right = img_right[..., np.newaxis]
    label_img_left = label_img_left[..., np.newaxis]
    label_img_right = label_img_right[..., np.newaxis]

    return str(image_path).encode("ascii"), img_left, label_img_left, str(image_path).encode("ascii"), img_right, label_img_right

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <path/to/tiff> <path/to/ouptut/dir>")
        exit(1)

    image_path = Path(sys.argv[1])
    process_image(image_path, sys.argv[2])