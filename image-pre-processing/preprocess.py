import json
import utils

from pathlib import Path

'''
The images and labels recieved from the NEI are labeled in the following manner:
- Each image has a correspoing CSV file. The CSV file contain six rows where:
    - Each row represents a boundary; they contain 20 columns.
    - The first three rows represent the y-coordinate of the boundaries for x = [50, 60, ..., 230, 240]
    - The last three rows represent the y-coordinate of the boundaries for x = [750, 760, ..., 930, 940]

- The following script crops the original image and generates two images (left and right):
    - The left image goes from x = [50, 240]
    - The right image goes from x = [750, 940]

- Then the boundaries are added as polygons.

- Finally the image is converted into a segmentation map: A 2D-matrix where each element represents the class
the pixel belongs to.
'''

x_left_start = 50
x_left_end = 240

x_right_start = 750
x_right_end = 940


def create_linestrip(y_annotation, label, image_height):
    shape = {}
    shape["label"] = label

    points = []

    for x, y in zip(range(0, 200, 10), y_annotation):
        point = [x, image_height - y]
        points.append(point)

    shape["points"] = points

    shape["group_id"] = None
    shape["shape_type"] = "linestrip"
    shape["flags"] = {}

    return shape

def create_polygon(boundary, extra_points, label, image_height):
    shape = {}
    shape["label"] = label

    points = []

    for x, y in zip(range(0, 200, 10), boundary):
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
    file['imageData'] = str(utils.img_data_to_img_b64(utils.pil_to_data(img)), "utf-8")
    file["imagePath"] = str(in_file_name)
    file["version"] = "4.5.9"
    file["flags"] = {}

    shapes = []

    # Bottom polygom
    extra_points = [[img.width, img.height], [0, img.height]]
    shapes.append(create_polygon(annotations[0], extra_points, "polygon_0", img.height))

    # Second polygon
    y_coordinates = [img.height - x for x in annotations[0]]
    extra_points = [(x, y) for x, y in zip(range(0, 200, 10), y_coordinates)]
    extra_points.reverse()
    shapes.append(create_polygon(annotations[1], extra_points, "polygon_1", img.height))

    # Third polygon
    y_coordinates = [img.height - x for x in annotations[1]]
    extra_points = [(x, y) for x, y in zip(range(0, 200, 10), y_coordinates)]
    extra_points.reverse()
    shapes.append(create_polygon(annotations[2], extra_points, "polygon_2", img.height))

    # Upper polygon
    extra_points = [[img.width, 0], [0, 0]]
    shapes.append(create_polygon(annotations[2], extra_points, "polygon_3", img.height))

    file["shapes"] = shapes
    file["imageHeight"] = img.height
    file["imageWidth"] = img.width

    with open(out_file_name, 'w') as outfile:
        json.dump(file, outfile)

def process_image(image_path):
    im = open(image_path, "rb")

    csv_path = image_path.parent / Path(image_path.stem + ".csv")

    with open(csv_path, "r") as f:
        annotations = []
        for line in f.readlines():
            annotations.append([int(x) for x in line.rstrip(' ,\n').split(" ,")])

    img, _ = utils.img_data_to_png_data(im.read())

    img_left_path = image_path.parent / Path(image_path.stem + "_left.json")
    img_left = img.crop((x_left_start, 0, x_left_end, img.height))
    create_labelme_file(img_left, annotations[:3], image_path, img_left_path)
    
    img_right_path = image_path.parent / Path(image_path.stem + "_right.json")
    img_right = img.crop((x_right_start, 0, x_right_end, img.height))
    create_labelme_file(img_right, annotations[3:], image_path, img_right_path)


if __name__ == "__main__":
    image_path = Path("example-input/001.tiff")
    process_image(image_path)