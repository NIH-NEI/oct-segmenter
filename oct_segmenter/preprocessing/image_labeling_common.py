import numpy as np

from oct_segmenter.preprocessing import utils


def generate_boundary(img_array):
    """
    Convention used by code in model: Considering the image in a
    top to bottom fashion, the areas do not include the pixel of the
    boundary of which it ends; in other words, boundaries belong to
    the first pixel of the "next region". For example: If the first
    boundary is an array of 2's the top region will be height 2 (0th
    and 1st row)
    """
    boundaries = []
    num_classes = np.amax(img_array)

    for i in range(1, num_classes + 1):
        boundaries.append([x for x in np.argmax(img_array == i, axis=0)])
    return np.array(boundaries)


def image_to_label(labelme_img_json):
    label_name_to_value = {"_background_": 0}
    for shape in sorted(labelme_img_json["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    shape = (labelme_img_json["imageHeight"], labelme_img_json["imageWidth"])

    lbl, _ = utils.shapes_to_label(
        shape, labelme_img_json["shapes"], label_name_to_value
    )

    return lbl


def create_label_image(labelme_img_json, output_name, save_file=True):
    label_arr = image_to_label(labelme_img_json)

    # labelme creates the segmentation map (label.png) using [1, 2, 3, 4, ...] and we want [0, 1, 2, 3, ...]
    # We apply the convention that the top layer is labeled as 0 and increase downwards.
    # This is done for consistency across images, so that get_boundaries() works, and because
    # it is the convention used by the unet repo
    num_classes = np.max(label_arr) # Get max value
    _, idx = np.unique(label_arr[:,0], return_index=True) # Gets the row indices of the first occurences of each class in the first column
    index_list = label_arr[np.sort(idx),0] # index_list lists the classes as they appear in the array from top to bottom

    # Mapping dictionary
    index_dict = {}
    for i in range(num_classes):
        index_dict[index_list[i]] = i

    def convert(x):
        return index_dict[x]

    vfunc = np.vectorize(convert)
    label_arr = vfunc(label_arr)

    if save_file:
        utils.lblsave(output_name, label_arr)

    return label_arr
