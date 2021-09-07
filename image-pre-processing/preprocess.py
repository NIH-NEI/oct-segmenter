import json
import utils

x_left = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
x_right = [750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940]

def create_linestrip(x_annotation, y_annotation, label, image_height):
    shape = {}
    shape["label"] = label

    points = []

    for x, y in zip(x_annotation, y_annotation):
        point = [x, image_height - y]
        points.append(point)

    shape["points"] = points

    shape["group_id"] = None
    shape["shape_type"] = "linestrip"
    shape["flags"] = {}

    return shape


im = open("example-input/001.tiff", "rb")
with open("example-input/001.csv", "r") as f:
    annotations = []
    for line in f.readlines():
        annotations.append([int(x) for x in line.rstrip(' ,\n').split(" ,")])

img, data = utils.img_data_to_png_data(im.read())
image_height = img.height
file = {}
file['imageData'] = str(utils.img_data_to_img_b64(data), "utf-8")
file["imagePath"] = "001.tiff"
file["version"] = "4.5.9"
file["flags"] = {}

shapes = []

for i, annotation in enumerate(annotations[:3]):
    shapes.append(create_linestrip(x_left, annotation, f"left_{i}", image_height))

for i, annotation in enumerate(annotations[3:]):
    shapes.append(create_linestrip(x_right, annotation, f"right_{i}", image_height))


file["shapes"] = shapes
file["imageHeight"] = img.height
file["imageWidth"] = img.width

with open('001.json', 'w') as outfile:
    json.dump(file, outfile)