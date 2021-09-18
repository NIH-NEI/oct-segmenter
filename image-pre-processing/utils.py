import base64
import io

import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps

def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil


def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr

def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr

def img_data_to_img_b64(img_data):
    imarray = np.array(img_data)
    img_b64 = base64.b64encode(imarray)
    return img_b64

def img_data_to_png_data(img_data):
    with io.BytesIO() as f:
        f.write(img_data)
        img = PIL.Image.open(f)

        with io.BytesIO() as f:
            img.save(f, "PNG")
            f.seek(0)
            return img, f.read()

def pil_to_data(img):
    with io.BytesIO() as f:
        img.save(f, "PNG")
        f.seek(0)
        return f.read()

def pil_to_array(img_pil):
    img_arr = np.array(img_pil)
    return img_arr