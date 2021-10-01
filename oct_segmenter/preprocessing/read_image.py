import numpy as np
import PIL.Image

label_png = 'example-input/001.tiff'
lbl = np.asarray(PIL.Image.open(label_png))
print(lbl.dtype)

print(np.unique(lbl))
print(lbl.shape)

print(lbl[50:100])